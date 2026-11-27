"""Graph construction.

Setup graph (build_setup_graph): the configure graph, then
  persist -> route_configured -> env -> data -> smoke -> route_smoke
  route_smoke (routing.decide_smoke, table-driven):
    passed -> END (ready); infra -> END (infra); unfixable / any cap -> END (capped)
    known  -> apply_known_fix -> route_fix: add_package -> env (rebuild) | else -> smoke
    critic -> critic -> route_critic: patched -> smoke | else -> END (capped)
  env -> route_env: data (first time) | smoke (rebuild after add_package) | END (failed)
Counters (known_fixes_used <= MAX_KNOWN_FIXES, critic_attempts <= MAX_CRITIC_ATTEMPTS) bound
every loop; all IO is injected through SetupDeps.

Full pipeline (--until plan / run / vault; graph/nodes/pipeline.py), orchestrator-workers:
  configure -> resume -> {subtask_a, b, c} | persist (stored A/B/C output reused)
  persist -> route_configured: env | plan (resumed: a passed smoke with the current smoke inputs)
  smoke ready -> plan -> route_plan: END (plan_failed / --until plan)
                       | Send(claim_worker, claim) per current plan (structural fan-out)
  claim_worker -> join (fan-in; worker_results reducer) -> route_join: END (--until run)
                                                                     | evaluate
  evaluate -> route_reflect: reflect (Reflexion loop on and a claim decides "reflect":
              routing.decide_reflect) | vault | END (--no-vault)
  reflect -> route_after_reflect: apply_lesson | evaluate (no lesson: recorded failure)
  apply_lesson -> route_apply: Send(claim_worker, claim) per claim to re-run | evaluate
  (reeval / stop); the loop is bounded by the breaker (trials per claim, GPU hours, tokens)
  and by "one lesson per trial": a trial is never reflected on twice
  vault -> END

Configure graph: START -> ingest_spec -> configure -> {subtask_a, subtask_b, subtask_c}
(parallel) -> persist (runs once, after all three) -> END. ingest_spec is a no-op without
spec_path; if it fails (status configure_failed) the graph ends there, before any clone.
Only persist writes evidence/events; the subtask branches write LLM telemetry through
MeteredLLM.on_record (lock-safe appends).

Default fallback routes (both graphs): every node is wrapped by guard_node (an unhandled
exception -> fallback_reason "fallback: <node>: <error type>", nothing else written; once set,
later nodes are no-ops) and every router by guard_router (fallback set by a node -> the
fallback node; a raise or a target outside its path map -> the fallback node). Each router's
own default branch sends unexpected state to the fallback node (routing.FALLBACK_NODE -> END);
the plain edges to END (configure graph persist, setup graph vault) are guarded the same way.
So every fallback passes the fallback node, which writes one `fallback` evidence per
invocation (fallback_evidence). The CLI turns GraphRecursionError into
"fallback: graph: GraphRecursionError" (not a graph route: no evidence). Claims stay open.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.graph.nodes.configure import configure
from reproducer.graph.nodes.critic import critic_node
from reproducer.graph.nodes.data import data_node
from reproducer.graph.nodes.env import env_node
from reproducer.graph.nodes.fixes import apply_known_fix_node
from reproducer.graph.nodes.ingest_spec import ingest_spec
from reproducer.graph.nodes.persist import persist
from reproducer.graph.nodes.pipeline import (
    apply_node,
    claim_worker,
    evaluate_node,
    join_node,
    plan_node,
    reflect_node,
    resume_node,
    vault_node,
)
from reproducer.graph.nodes.smoke import smoke_node
from reproducer.graph.nodes.subtasks import subtask_a, subtask_b, subtask_c
from reproducer.graph.routing import (
    APPLY_NODE,
    FALLBACK_NODE,
    REFLECT_NODE_NAME,
    WORKER_NODE,
    fallback_reason,
    route_after_reflect,
    route_apply,
    route_configured,
    route_critic,
    route_data,
    route_env,
    route_fix,
    route_ingest,
    route_join,
    route_plan,
    route_reflect,
    route_resume,
    route_smoke,
)
from reproducer.graph.state import ConfigureDeps, ConfigureState, SetupDeps, SetupState

SUBTASK_NODES = {"subtask_a": subtask_a, "subtask_b": subtask_b, "subtask_c": subtask_c}
NodeFn = Callable[[Mapping[str, Any]], Any]


def guard_node(name: str, fn: NodeFn) -> NodeFn:
    """Default fallback for a node: once fallback_reason is set the node is a no-op ({}); an
    unhandled exception becomes fallback_reason "fallback: <name>: <error type>" (+ the
    message in fallback_detail) instead of crashing the graph. LangGraph control flow
    (GraphBubbleUp: interrupts) is re-raised."""

    def run(state: Mapping[str, Any]) -> Any:
        if state.get("fallback_reason"):
            return {}
        try:
            return fn(state)
        except GraphBubbleUp:
            raise
        except Exception as e:
            return {
                "fallback_reason": fallback_reason(name, type(e).__name__),
                "fallback_detail": str(e)[:500],
            }

    run.__name__ = name
    return run


def guard_router(name: str, fn: Callable[[Mapping[str, Any]], Any], targets: Iterable[str]) -> Any:
    """Default fallback for a router: fallback set -> END; a raise or a target outside its
    path map -> Send(fallback node, reason). The router itself never raises."""
    allowed = {*targets, FALLBACK_NODE}

    def route(state: Mapping[str, Any]) -> Any:
        if state.get("fallback_reason"):  # set by a node: record it (fallback node -> END)
            return Send(FALLBACK_NODE, {
                "fallback_reason": state["fallback_reason"],
                "fallback_detail": state.get("fallback_detail") or "",
            })  # fmt: skip
        try:
            out = fn(state)
        except Exception as e:
            return Send(FALLBACK_NODE, {"fallback_reason": fallback_reason(name, type(e).__name__)})
        items = out if isinstance(out, list) else [out]
        names = [i.node if isinstance(i, Send) else i for i in items]
        if not items or any(n not in allowed for n in names):
            return Send(
                FALLBACK_NODE, {"fallback_reason": fallback_reason(name, f"target={out!r}")}
            )
        return out

    route.__name__ = name
    return route


def fallback_where(reason: str) -> str:
    """The router or node a reason "fallback: <where>: <value>" names ("unknown" if none)."""
    parts = reason.split(": ", 2)
    return parts[1] if len(parts) >= 2 and parts[0] == "fallback" and parts[1] else "unknown"


def fallback_evidence(reason: str, detail: str, stamp: str) -> Evidence:
    """One graph invocation's fallback (authority rule, derived). stamp (time + random) keeps
    the same fallback in two invocations two records."""
    return make_evidence(
        kind="fallback",
        source="graph",
        locator=f"fallback:{stamp}",
        content={"router_or_node": fallback_where(reason), "reason": reason, "detail": detail},
        confidence="derived",
        authority="rule",
    )


def fallback_stamp() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def fallback_node(
    state: Mapping[str, Any], deps: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The fallback route's node: records the reason (Send payload) in the state and, with a
    store in deps, as `fallback` evidence (a store error does not hide the fallback); then
    END. Every fallback route passes here once, so one record per invocation."""
    reason = state.get("fallback_reason") or "fallback: unknown"
    store = (deps or {}).get("store")
    if store is not None:
        try:
            ev = fallback_evidence(
                reason, str(state.get("fallback_detail") or ""), fallback_stamp()
            )
            store.append_evidence(ev)
        except (OSError, ValueError):
            pass
    return {"fallback_reason": reason}


def route_end(state: Mapping[str, Any]) -> str:
    """A plain edge to END, as a router so guard_router sends a fallback to the fallback node."""
    return END


def _node(graph: StateGraph, name: str, fn: Callable[..., Any], deps: Any) -> None:
    graph.add_node(name, guard_node(name, partial(fn, deps=deps)))


def _route(graph: StateGraph, source: str, router: Callable[..., Any], targets: list[str]) -> None:
    graph.add_conditional_edges(
        source,
        guard_router(router.__name__, router, targets),
        [*targets, FALLBACK_NODE],
    )


def _add_fallback(graph: StateGraph, deps: Any) -> None:
    graph.add_node(FALLBACK_NODE, partial(fallback_node, deps=deps))
    graph.add_edge(FALLBACK_NODE, END)


def _add_configure(graph: StateGraph, deps: ConfigureDeps, *, resume: bool = False) -> None:
    _add_fallback(graph, deps)
    _node(graph, "ingest_spec", ingest_spec, deps)
    _node(graph, "configure", configure, deps)
    for name, fn in SUBTASK_NODES.items():
        _node(graph, name, fn, deps)
    _node(graph, "persist", persist, deps)
    graph.add_edge(START, "ingest_spec")
    _route(graph, "ingest_spec", route_ingest, ["configure", END])
    if resume:  # setup graph: configure -> resume -> subtasks | persist
        _node(graph, "resume", resume_node, deps)
        graph.add_edge("configure", "resume")
        _route(graph, "resume", route_resume, [*SUBTASK_NODES, "persist"])
    else:
        for name in SUBTASK_NODES:
            graph.add_edge("configure", name)
    graph.add_edge(list(SUBTASK_NODES), "persist")  # fan-in: waits for all three


def build_configure_graph(deps: ConfigureDeps) -> Any:
    """Compiled graph; call .invoke(initial_state) with the ConfigureState inputs."""
    graph = StateGraph(ConfigureState)
    _add_configure(graph, deps)
    _route(graph, "persist", route_end, [END])
    return graph.compile()


# Each smoke round is at most smoke -> fix -> env (3 steps); the caps allow at most
# MAX_KNOWN_FIXES + MAX_CRITIC_ATTEMPTS + 1 smoke runs; each Reflexion round is evaluate ->
# reflect -> apply_lesson -> claim_worker -> join (5 steps), at most MAX_TRIALS_PER_CLAIM
# rounds. So this is never the binding limit.
SETUP_RECURSION_LIMIT = 200


def build_setup_graph(deps: SetupDeps) -> Any:
    """Compiled setup graph; .invoke(inputs, config={"recursion_limit":
    SETUP_RECURSION_LIMIT}) with the ConfigureState inputs (+ until, timeouts)."""
    graph = StateGraph(SetupState)
    _add_configure(graph, deps, resume=True)
    for name, fn in {
        "env": env_node,
        "data": data_node,
        "smoke": smoke_node,
        "apply_known_fix": apply_known_fix_node,
        "critic": critic_node,
        "plan": plan_node,
        WORKER_NODE: claim_worker,
        "join": join_node,
        "evaluate": evaluate_node,
        REFLECT_NODE_NAME: reflect_node,
        APPLY_NODE: apply_node,
        "vault": vault_node,
    }.items():
        _node(graph, name, fn, deps)
    _route(graph, "persist", route_configured, ["env", "plan", END])
    _route(graph, "env", route_env, ["data", "smoke", END])
    _route(graph, "data", route_data, ["smoke", END])
    _route(graph, "smoke", route_smoke, ["apply_known_fix", "critic", "plan", END])
    _route(graph, "apply_known_fix", route_fix, ["env", "smoke"])
    _route(graph, "critic", route_critic, ["smoke", END])
    _route(graph, "plan", route_plan, [WORKER_NODE, END])
    graph.add_edge(WORKER_NODE, "join")  # fan-in: join runs once after all workers
    _route(graph, "join", route_join, ["evaluate", END])
    _route(graph, "evaluate", route_reflect, [REFLECT_NODE_NAME, "vault", END])
    _route(graph, REFLECT_NODE_NAME, route_after_reflect, [APPLY_NODE, "evaluate"])
    _route(graph, APPLY_NODE, route_apply, [WORKER_NODE, "evaluate"])
    _route(graph, "vault", route_end, [END])
    return graph.compile()
