"""Default fallback routes: every router has an explicit default branch (unexpected state ->
the fallback node -> END with "fallback: <router>: <value>"), no guarded router raises, and a
node exception becomes a clean "fallback: <node>: <error type>" stop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.types import Send

from reproducer import cli
from reproducer.domain.evidence import make_evidence
from reproducer.graph.build import (
    fallback_evidence,
    fallback_node,
    fallback_where,
    guard_node,
    guard_router,
    route_end,
)
from reproducer.graph.routing import (
    FALLBACK_NODE,
    LastSmoke,
    final_setup_status,
    is_fallback,
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
from reproducer.providers.store import RunStore

Router = Callable[[Any], Any]
NO_CLS = LastSmoke(passed=False, signal="traceback", cls=None, reason="", has_traceback=True)

# (router, unexpected state, the "<key>=<value>" the reason names)
UNEXPECTED: list[tuple[Router, dict[str, Any], str]] = [
    (route_ingest, {"status": "banana"}, "status='banana'"),
    (route_resume, {"resumed": 7}, "resumed=7"),
    (route_configured, {"status": "configured", "until": "moon"}, "until='moon'"),
    (route_configured, {"status": "half-configured"}, "status='half-configured'"),
    (route_configured, {"status": "configured", "resumed": "smoke"}, "resumed='smoke'"),
    (route_env, {"env_ok": "yes"}, "env_ok='yes'"),
    (route_env, {"env_ok": True, "data_ok": 1}, "data_ok=1"),
    (route_data, {"data_ok": None}, "data_ok=None"),
    (route_smoke, {"setup_status": "exploded"}, "setup_status='exploded'"),
    (route_smoke, {"setup_status": "running", "last_smoke": {"passed": True}}, "last_smoke='dict'"),
    (route_smoke, {"setup_status": "running", "last_smoke": NO_CLS}, "last_smoke.cls=None"),
    (route_fix, {"last_fix_kind": "rm_rf"}, "last_fix_kind='rm_rf'"),
    (route_fix, {}, "last_fix_kind=None"),
    (route_critic, {"last_critic_status": "maybe"}, "last_critic_status='maybe'"),
    (route_plan, {"setup_status": "??"}, "setup_status='??'"),
    (route_plan, {"setup_status": "ready", "plans": "c1"}, "plans='c1'"),
    (route_join, {"until": "forever"}, "until='forever'"),
    (route_reflect, {"reflect": True, "reflect_info": ["c1"]}, "reflect_info='list'"),
    (route_after_reflect, {"reflect_pending": "lesson"}, "reflect_pending='str'"),
    (route_apply, {"rerun_claims": {"c1": 1}}, "rerun_claims={'c1': 1}"),
]


@pytest.mark.parametrize(("router", "state", "value"), UNEXPECTED)
def test_router_default_branch_is_the_fallback(router: Router, state: dict, value: str) -> None:
    out = router(state)
    assert is_fallback(out), out
    assert out.arg == {"fallback_reason": f"fallback: {router.__name__}: {value}"}
    # guarded the same way the graph wires it: still the fallback, never a raise
    assert is_fallback(guard_router(router.__name__, router, ["x"])(state))


@pytest.mark.parametrize(("router", "state", "value"), UNEXPECTED)
def test_routers_send_a_node_fallback_to_the_fallback_node(
    router: Router, state: dict, value: str
) -> None:
    """A fallback a node set is recorded by the fallback node (-> END), whatever the router."""
    guarded = guard_router(router.__name__, router, ["x"])
    out = guarded({**state, "fallback_reason": "fallback: env: RuntimeError",
                   "fallback_detail": "disk on fire"})  # fmt: skip
    assert is_fallback(out)
    assert out.arg == {"fallback_reason": "fallback: env: RuntimeError",
                       "fallback_detail": "disk on fire"}  # fmt: skip


def test_guard_router_catches_raises_and_unknown_targets() -> None:
    def boom(state: Any) -> str:
        raise KeyError("k")

    out = guard_router("route_x", boom, ["a"])({})
    assert isinstance(out, Send) and out.node == FALLBACK_NODE
    assert out.arg == {"fallback_reason": "fallback: route_x: KeyError"}
    out = guard_router("route_y", lambda s: "nowhere", ["a", END])({})
    assert out.arg == {"fallback_reason": "fallback: route_y: target='nowhere'"}
    out = guard_router("route_z", lambda s: [], ["a"])({})  # an empty fan-out
    assert is_fallback(out)
    assert guard_router("route_ok", lambda s: "a", ["a"])({}) == "a"
    sends = [Send("w", {"c": 1})]
    assert guard_router("route_ok", lambda s: sends, ["w"])({}) == sends


def test_guard_node_turns_an_exception_into_a_clean_stop() -> None:
    def node(state: Any) -> dict[str, Any]:
        raise RuntimeError("disk on fire")

    guarded = guard_node("env", node)
    assert guarded({}) == {
        "fallback_reason": "fallback: env: RuntimeError",
        "fallback_detail": "disk on fire",
    }
    calls: list[Any] = []
    later = guard_node("data", lambda s: calls.append(s) or {"data_ok": True})
    assert later({"fallback_reason": "fallback: env: RuntimeError"}) == {}  # no-op after it
    assert calls == []
    assert later({}) == {"data_ok": True}


def test_fallback_node_and_final_status(tmp_path: Path) -> None:
    reason = "fallback: route_fix: last_fix_kind='rm_rf'"
    assert fallback_node({"fallback_reason": reason}) == {"fallback_reason": reason}
    store = RunStore(tmp_path)
    for _ in range(2):  # two invocations, the same fallback -> two records
        fallback_node({"fallback_reason": reason}, deps={"store": store})
    evs = list(store.load_evidence().values())
    assert [ev.kind for ev in evs] == ["fallback", "fallback"]
    assert evs[0].locator != evs[1].locator
    assert evs[0].content == {"router_or_node": "route_fix", "reason": reason, "detail": ""}
    assert (evs[0].authority, evs[0].confidence) == ("rule", "derived")
    state = {"status": "configured", "setup_status": "ready", "fallback_reason": reason}
    assert final_setup_status(state) == "fallback"


def test_fallback_where_and_evidence_shape() -> None:
    assert fallback_where("fallback: env: RuntimeError") == "env"
    assert fallback_where("fallback: route_smoke: setup_status='x: y'") == "route_smoke"
    assert fallback_where("weird") == "unknown"
    ev = fallback_evidence("fallback: env: RuntimeError", "boom", "s1")
    assert ev.locator == "fallback:s1" and ev.content["detail"] == "boom"
    with pytest.raises(ValueError, match="fallback evidence requires"):
        make_evidence(kind="fallback", source="graph", locator="x", content={"reason": "r"},
                      confidence="derived", authority="rule")  # fmt: skip


def test_route_end_is_guarded() -> None:
    guarded = guard_router("route_end", route_end, [END])
    assert guarded({}) == END
    out = guarded({"fallback_reason": "fallback: vault: OSError"})
    assert is_fallback(out) and out.arg["fallback_reason"] == "fallback: vault: OSError"


def test_graph_recursion_error_is_a_fallback(capsys: Any) -> None:
    fb = cli.graph_fallback(GraphRecursionError("Recursion limit of 200 reached"))
    assert fb["fallback_reason"] == "fallback: graph: GraphRecursionError"
    cli.print_fallback(fb)
    out = capsys.readouterr().out
    assert "stop:           fallback (fallback: graph: GraphRecursionError: Recursion" in out
    assert "stay open" in out
