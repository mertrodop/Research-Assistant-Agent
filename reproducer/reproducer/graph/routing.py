"""Pure, table-driven routing for the setup graph (no IO, no LLM).

decide_smoke() is THE decision after a smoke run: smoke_loop (CLI `smoke --auto-fix`) and the
setup graph's route_smoke both use it, so the two cannot drift apart. Order of the checks
(same as the original smoke_loop):
  passed                                   -> ready   (stop "passed")
  no auto-fix                              -> stop    (stop "no_auto_fix")
  class infra                              -> infra   (stop "infra"; the critic is never called)
  class critic without critic / traceback  -> stop    (stop "critic")
  class critic, critic_attempts >= cap     -> capped  (stop "max_critic")
  class critic                             -> critic
  class known, known_fixes_used >= cap     -> capped  (stop "max_fixes")
  class known                              -> known   (apply the zero-token fix)
  class unfixable (incl. a known fix that would be a no-op: the loop guard) -> capped
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from langgraph.graph import END
from langgraph.types import Send

from reproducer.graph.nodes.critic import MAX_CRITIC_ATTEMPTS, CriticStatus
from reproducer.graph.state import ConfigureStatus, SetupStatus
from reproducer.rules.known_fixes import FixKind

StopReason = Literal[
    "passed",
    "infra",
    "critic",  # critic class but no critic configured
    "unfixable",
    "max_fixes",
    "no_auto_fix",
    "max_critic",  # max_critic LLM attempts used
    "no_new_evidence",  # same error signature as the last critic input
    "critic_failed",  # the critic gave no valid patch
    "awaiting_approval",  # a validated critic patch waits for a human (approval_request)
]
SmokeRoute = Literal["ready", "infra", "known", "critic", "capped", "stop"]
ClassName = Literal["infra", "known", "critic", "unfixable"]

MAX_KNOWN_FIXES = 5  # known (zero-token) fixes per setup run


@dataclass(frozen=True)
class SmokeDecision:
    route: SmokeRoute
    stop: StopReason | None  # None: the loop continues (known / critic)
    reason: str
    capped: bool  # the smoke_result evidence of this run is written with capped=True


@dataclass(frozen=True)
class LastSmoke:
    """What routing needs from the last smoke run (graph state key `last_smoke`)."""

    passed: bool
    signal: str
    cls: ClassName | None  # None when passed
    reason: str  # classification reason ("" when passed)
    has_traceback: bool


# class -> (route, stop, capped) when no cap / special case applies
_BY_CLASS: dict[str, tuple[SmokeRoute, StopReason | None, bool]] = {
    "infra": ("infra", "infra", False),
    "critic": ("critic", None, False),
    "known": ("known", None, False),
    "unfixable": ("capped", "unfixable", True),
}


def decide_smoke(
    smoke: LastSmoke,
    *,
    known_fixes_used: int,
    critic_attempts: int,
    max_fixes: int = MAX_KNOWN_FIXES,
    max_critic: int = MAX_CRITIC_ATTEMPTS,
    auto_fix: bool = True,
    has_critic: bool = True,
) -> SmokeDecision:
    if smoke.passed:
        return SmokeDecision("ready", "passed", f"smoke passed ({smoke.signal})", False)
    if smoke.cls is None:
        raise ValueError("a failed smoke run needs a classification")
    if not auto_fix:
        return SmokeDecision("stop", "no_auto_fix", smoke.reason, False)
    if smoke.cls == "critic" and (not has_critic or not smoke.has_traceback):
        return SmokeDecision("stop", "critic", smoke.reason, False)
    if smoke.cls == "critic" and critic_attempts >= max_critic:
        return SmokeDecision(
            "capped", "max_critic", f"{max_critic} critic attempts used; {smoke.reason}", True
        )
    if smoke.cls == "known" and known_fixes_used >= max_fixes:
        return SmokeDecision(
            "capped", "max_fixes", f"{max_fixes} known fixes applied; {smoke.reason}", True
        )
    route, stop, capped = _BY_CLASS[smoke.cls]
    return SmokeDecision(route, stop, smoke.reason, capped)


# --- default fallback route -----------------------------------------------------------------
# Every router has an explicit default branch: a state value it does not expect goes to
# FALLBACK_NODE (Send with the reason), which records fallback_reason
# "fallback: <router>: <key>=<value>" and ends the graph. build.guard_router adds the same for
# a router that raises or returns a target outside its path map; build.guard_node turns a
# node exception into fallback_reason "fallback: <node>: <error type>".

FALLBACK_NODE = "fallback"
_MAX_VALUE_CHARS = 120


def fallback_reason(where: str, value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    return f"fallback: {where}: {text[:_MAX_VALUE_CHARS]}"


def fallback(router: str, key: str, value: Any) -> Send:
    """The default branch of `router`: unexpected state[key] == value."""
    return Send(FALLBACK_NODE, {"fallback_reason": fallback_reason(router, f"{key}={value!r}")})


def is_fallback(target: Any) -> bool:
    return isinstance(target, Send) and target.node == FALLBACK_NODE


CONFIGURE_STATUSES = frozenset(get_args(ConfigureStatus))
SETUP_STATUSES = frozenset(get_args(SetupStatus))
FIX_KINDS = frozenset(get_args(FixKind))
CRITIC_STATUSES = frozenset(get_args(CriticStatus)) | {"error"}  # error: critic_node failed


# --- graph routers (pure functions of the state) ---------------------------------------------

UNTIL_STAGES = ("configure", "env", "data", "smoke", "plan", "run", "vault")
PIPELINE_STAGES = frozenset({"plan", "run", "vault"})  # past smoke: plan + runs (+ vault)
SUBTASK_NODE_NAMES = ("subtask_a", "subtask_b", "subtask_c")


def _until(state: Mapping[str, Any], stage: str) -> bool:
    return state.get("until", "smoke") == stage


def _unexpected(
    router: str, state: Mapping[str, Any], *, flags: Iterable[str] = (), status: bool = False,
    setup: bool = False,
) -> Send | None:  # fmt: skip
    """The shared default checks: until a known stage, `flags` bool (or absent), status a
    ConfigureStatus, setup_status a SetupStatus (each absent is fine)."""
    until = state.get("until", "smoke")
    if until not in UNTIL_STAGES:
        return fallback(router, "until", until)
    for key in flags:
        if key in state and not isinstance(state[key], bool):
            return fallback(router, key, state[key])
    if status and "status" in state and state["status"] not in CONFIGURE_STATUSES:
        return fallback(router, "status", state["status"])
    if setup and "setup_status" in state and state["setup_status"] not in SETUP_STATUSES:
        return fallback(router, "setup_status", state["setup_status"])
    return None


def route_ingest(state: Mapping[str, Any]) -> str | Send:
    """ingest_spec failed (configure_failed) -> END; else configure."""
    if (bad := _unexpected("route_ingest", state, status=True)) is not None:
        return bad
    return END if state.get("status") == "configure_failed" else "configure"


def full_pipeline(state: Mapping[str, Any]) -> bool:
    """--until plan / run / vault: the graph continues past a ready smoke."""
    return state.get("until", "smoke") in PIPELINE_STAGES


def route_resume(state: Mapping[str, Any]) -> str | list[str] | Send:
    """Stored A/B/C output for this source (resume node) -> persist; else the three subtasks."""
    resumed = state.get("resumed", ())
    if not isinstance(resumed, (list, tuple, set, frozenset)):
        return fallback("route_resume", "resumed", resumed)
    if "subtasks" in resumed:
        return "persist"
    return list(SUBTASK_NODE_NAMES)


def route_configured(state: Mapping[str, Any]) -> str | Send:
    """configured -> env, or plan when the setup was resumed (a passed smoke with the
    current smoke input digest); configure_failed / running / absent -> END; --until
    configure -> END; an unknown status or until -> fallback."""
    if (bad := _unexpected("route_configured", state, status=True)) is not None:
        return bad
    resumed = state.get("resumed", ())
    if not isinstance(resumed, (list, tuple, set, frozenset)):
        return fallback("route_configured", "resumed", resumed)
    if state.get("status") != "configured" or _until(state, "configure"):
        return END
    return "plan" if "smoke" in state.get("resumed", ()) else "env"


def route_env(state: Mapping[str, Any]) -> str | Send:
    """env built -> data (first time) or smoke (data already provisioned: a rebuild after an
    add_package fix); failed or --until env -> END."""
    if (bad := _unexpected("route_env", state, flags=("env_ok", "data_ok"))) is not None:
        return bad
    if not state.get("env_ok") or _until(state, "env"):
        return END
    return "smoke" if state.get("data_ok") else "data"


def route_data(state: Mapping[str, Any]) -> str | Send:
    if (bad := _unexpected("route_data", state, flags=("data_ok",))) is not None:
        return bad
    if not state.get("data_ok") or _until(state, "data"):
        return END
    return "smoke"


SMOKE_TARGETS: dict[SmokeRoute, str] = {
    "ready": END,
    "infra": END,
    "capped": END,
    "stop": END,
    "known": "apply_known_fix",
    "critic": "critic",
}


def route_smoke(state: Mapping[str, Any]) -> str | Send:
    """Table-driven: decide_smoke(last smoke, counters) -> next node. A smoke-node failure
    (setup_status already final) -> END. Full pipeline: ready -> plan. A last_smoke that is
    not a LastSmoke, or one decide_smoke rejects -> fallback."""
    if (bad := _unexpected("route_smoke", state, setup=True)) is not None:
        return bad
    if state.get("setup_status") == "ready" and full_pipeline(state):
        return "plan"
    if state.get("setup_status", "running") != "running" or "last_smoke" not in state:
        return END
    last = state["last_smoke"]
    if not isinstance(last, LastSmoke):
        return fallback("route_smoke", "last_smoke", type(last).__name__)
    try:
        d = decide_smoke(
            last,
            known_fixes_used=state.get("known_fixes_used", 0),
            critic_attempts=state.get("critic_attempts", 0),
        )
    except ValueError:
        return fallback("route_smoke", "last_smoke.cls", last.cls)
    return SMOKE_TARGETS.get(d.route) or fallback("route_smoke", "route", d.route)


def route_fix(state: Mapping[str, Any]) -> str | Send:
    """add_package changes the env plan -> env (rebuild); every other fix kind -> smoke; an
    unknown (or missing) fix kind -> fallback."""
    kind = state.get("last_fix_kind")
    if kind not in FIX_KINDS:
        return fallback("route_fix", "last_fix_kind", kind)
    return "env" if kind == "add_package" else "smoke"


def route_critic(state: Mapping[str, Any]) -> str | Send:
    """patched -> smoke; a known other status (skipped / no target / invalid / error / ...)
    or none -> END; an unknown status -> fallback."""
    status = state.get("last_critic_status")
    if status is not None and status not in CRITIC_STATUSES:
        return fallback("route_critic", "last_critic_status", status)
    return "smoke" if status == "patched" else END


WORKER_NODE = "claim_worker"
WORKER_KEYS = ("out_dir", "smoke_timeout_s", "build_timeout_s", "no_checkpoints")


def selected_claims(state: Mapping[str, Any], claim_ids: Iterable[str]) -> list[str]:
    """claim_ids in order, only the --claim one when state["only_claim"] is set."""
    only = state.get("only_claim")
    return [c for c in claim_ids if only is None or c == only]


def route_plan(state: Mapping[str, Any]) -> str | list[Send] | Send:
    """plan_failed -> END; --until plan -> END; else one claim_worker per current plan
    (structural fan-out: a Send per claim id, in plan order; payload = that id, the plan
    order (the workers take the GPU in it) + the run inputs). --claim: only that claim."""
    if (bad := _unexpected("route_plan", state, setup=True)) is not None:
        return bad
    if state.get("setup_status") == "plan_failed" or _until(state, "plan"):
        return END
    raw = state.get("plans") or []
    if not isinstance(raw, (list, tuple)) or not all(isinstance(c, str) for c in raw):
        return fallback("route_plan", "plans", raw)
    plans = selected_claims(state, raw)
    if not plans:
        return END
    base = {k: state[k] for k in WORKER_KEYS if k in state}
    order = list(plans)
    return [
        Send(WORKER_NODE, {**base, "worker_claim_id": cid, "worker_order": order}) for cid in order
    ]


def route_join(state: Mapping[str, Any]) -> str | Send:
    """After the fan-in: --until run -> END; else evaluate (always, skips included)."""
    if (bad := _unexpected("route_join", state)) is not None:
        return bad
    return END if _until(state, "run") else "evaluate"


def route_evaluate(state: Mapping[str, Any]) -> str:
    """vault unless --no-vault or the evaluation failed."""
    if state.get("no_vault") or state.get("evaluate_error"):
        return END
    return "vault"


def final_setup_status(state: Mapping[str, Any]) -> str:
    """The END label of a finished setup graph run: the status a node set, else
    configure_failed (ingest/configure did not configure) or stopped (--until reached);
    fallback whenever a default fallback route was taken."""
    if state.get("fallback_reason"):
        return "fallback"
    status = state.get("setup_status", "running")
    if status != "running":
        return str(status)
    return "configure_failed" if state.get("status") != "configured" else "stopped"


# --- Reflexion loop (M4): evaluate -> reflect -> apply -> (rerun | reeval) -> evaluate ---------

ReflectDecision = Literal["done", "reflect", "open"]
REFLECT_CLASSES = frozenset({"out_of_tolerance", "diverged"})
APPLY_NODE = "apply_lesson"
# applied changes per trial whose re-built plan left the rendered command unchanged (no run,
# no trial consumed): after this many, the claim stays open instead of another fresh lesson
MAX_CHANGES_PER_TRIAL = 2
REFLECT_NODE_NAME = "reflect"


def decide_reflect(
    info: Mapping[str, Any], failed: Mapping[str, Any] | None = None
) -> tuple[ReflectDecision, str]:
    """Per claim, from its newest run (pure; info is built from evidence by the evaluate node:
    trial, outcome, can_try, tripped, applied = outcome of the newest lesson_applied for that
    trial, unchanged = the current plan renders the same command as that trial's run, changes
    = lesson_applied records of that trial). In order:
      no run                                   -> open  "no completed trial to reflect on"
      verified                                 -> done
      reflection of this trial failed here     -> open  "reflection failed: ..."
      lesson of this trial was none            -> open  "reflexion stopped: no grounded change"
      applied, command unchanged, changes >= MAX_CHANGES_PER_TRIAL -> open
      applied, command changed                 -> open  waiting for its run (e.g. approval)
      applied, command unchanged               -> (no trial used) the checks below: a fresh
                                                  lesson from the changed inputs
      not out_of_tolerance / diverged          -> open  "<class>: not reflected"
      breaker tripped                          -> open  "breaker: <which cap>"
      else                                     -> reflect
    Never reflect twice on one trial: a new lesson needs a new run (one change per trial)."""
    outcome, trial = info.get("outcome"), info.get("trial")
    if outcome is None:
        return "open", "no completed trial to reflect on"
    if outcome == "verified":
        return "done", "verified"
    if failed is not None and failed.get("trial") == trial:
        return "open", f"reflection failed: {failed.get('reason')}"
    applied = info.get("applied")
    if applied == "stopped_no_change":
        return "open", "reflexion stopped: no grounded change"
    if applied is not None and info.get("unchanged"):
        if int(info.get("changes") or 1) >= MAX_CHANGES_PER_TRIAL:
            return "open", (
                f"{info.get('changes')} applied change(s) of trial {trial} left the command "
                "unchanged"
            )
    elif applied is not None:
        return "open", f"lesson of trial {trial} applied ({applied}); no new run of it yet"
    if outcome not in REFLECT_CLASSES:
        return "open", f"{outcome}: not reflected"
    if not info.get("can_try"):
        return "open", f"breaker: {info.get('tripped') or 'budget exhausted'}"
    return "reflect", ""


def reflect_decisions(state: Mapping[str, Any]) -> dict[str, tuple[ReflectDecision, str]]:
    failed = state.get("reflect_failures") or {}
    return {
        cid: decide_reflect(info, failed.get(cid))
        for cid, info in (state.get("reflect_info") or {}).items()
    }


def route_reflect(state: Mapping[str, Any]) -> str | Send:
    """After evaluate: reflect when the loop is on and some claim decides "reflect"; else the
    usual route_evaluate (vault / END)."""
    info = state.get("reflect_info") or {}
    if not isinstance(info, Mapping) or not all(isinstance(v, Mapping) for v in info.values()):
        return fallback("route_reflect", "reflect_info", type(info).__name__)
    if state.get("reflect") and not state.get("evaluate_error"):
        if any(d == "reflect" for d, _ in reflect_decisions(state).values()):
            return REFLECT_NODE_NAME
    return route_evaluate(state)


def route_after_reflect(state: Mapping[str, Any]) -> str | Send:
    """A lesson to apply -> apply; else back to evaluate (the failures are recorded, so the
    decision is open there and the loop ends)."""
    pending = state.get("reflect_pending")
    if pending is not None and not isinstance(pending, (list, tuple)):
        return fallback("route_after_reflect", "reflect_pending", type(pending).__name__)
    return APPLY_NODE if state.get("reflect_pending") else "evaluate"


def route_apply(state: Mapping[str, Any]) -> str | list[Send] | Send:
    """Claims whose change needs a new run -> one claim_worker each (their plans already
    re-built); else evaluate (a re-evaluation or a stop is in the store)."""
    rerun = state.get("rerun_claims") or []
    if not isinstance(rerun, (list, tuple)) or not all(isinstance(c, str) for c in rerun):
        return fallback("route_apply", "rerun_claims", rerun)
    if not rerun:
        return "evaluate"
    base = {k: state[k] for k in WORKER_KEYS if k in state}
    order = list(rerun)
    return [
        Send(WORKER_NODE, {**base, "worker_claim_id": cid, "worker_order": order}) for cid in order
    ]
