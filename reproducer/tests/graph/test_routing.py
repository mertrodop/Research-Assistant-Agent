"""Table-driven tests for the pure setup-graph routers (graph/routing.py)."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from reproducer.graph.routing import (
    MAX_CRITIC_ATTEMPTS,
    MAX_KNOWN_FIXES,
    SMOKE_TARGETS,
    LastSmoke,
    decide_smoke,
    final_setup_status,
    route_configured,
    route_critic,
    route_data,
    route_env,
    route_fix,
    route_smoke,
)

SIGNALS = ("traceback", "import_error", "oom", "timeout_no_progress", "infra")
PASSING = ("passed_exit0", "passed_progress")


def _failed(cls: str, *, tb: bool = True, signal: str = "traceback") -> LastSmoke:
    return LastSmoke(passed=False, signal=signal, cls=cls, reason="why", has_traceback=tb)  # type: ignore[arg-type]


# (cls, has_traceback, known_fixes_used, critic_attempts) -> (route, stop, capped)
BELOW = (0, 0)
TABLE: list[tuple[str, bool, tuple[int, int], tuple[str, str | None, bool]]] = [
    ("infra", True, BELOW, ("infra", "infra", False)),
    ("infra", False, BELOW, ("infra", "infra", False)),
    ("infra", True, (MAX_KNOWN_FIXES, MAX_CRITIC_ATTEMPTS), ("infra", "infra", False)),
    ("known", True, BELOW, ("known", None, False)),
    ("known", False, BELOW, ("known", None, False)),
    ("known", True, (MAX_KNOWN_FIXES - 1, 0), ("known", None, False)),
    ("known", True, (MAX_KNOWN_FIXES, 0), ("capped", "max_fixes", True)),
    ("known", True, (MAX_KNOWN_FIXES + 1, 0), ("capped", "max_fixes", True)),
    ("known", True, (0, MAX_CRITIC_ATTEMPTS), ("known", None, False)),  # critic cap irrelevant
    ("critic", True, BELOW, ("critic", None, False)),
    ("critic", True, (0, MAX_CRITIC_ATTEMPTS - 1), ("critic", None, False)),
    ("critic", True, (0, MAX_CRITIC_ATTEMPTS), ("capped", "max_critic", True)),
    ("critic", True, (MAX_KNOWN_FIXES, 0), ("critic", None, False)),  # fix cap irrelevant
    ("critic", False, BELOW, ("stop", "critic", False)),  # no traceback to show the critic
    ("unfixable", True, BELOW, ("capped", "unfixable", True)),
    ("unfixable", False, (MAX_KNOWN_FIXES, MAX_CRITIC_ATTEMPTS), ("capped", "unfixable", True)),
]


@pytest.mark.parametrize(("cls", "tb", "counters", "expected"), TABLE)
@pytest.mark.parametrize("signal", SIGNALS)
def test_decide_smoke_table(
    cls: str, tb: bool, counters: tuple[int, int], expected: tuple[str, Any, bool], signal: str
) -> None:
    d = decide_smoke(
        _failed(cls, tb=tb, signal=signal),
        known_fixes_used=counters[0],
        critic_attempts=counters[1],
    )
    assert (d.route, d.stop, d.capped) == expected
    assert "why" in d.reason


@pytest.mark.parametrize("signal", PASSING)
@pytest.mark.parametrize("counters", [BELOW, (MAX_KNOWN_FIXES, MAX_CRITIC_ATTEMPTS)])
def test_passed_is_ready_whatever_the_counters(signal: str, counters: tuple[int, int]) -> None:
    last = LastSmoke(passed=True, signal=signal, cls=None, reason="", has_traceback=False)
    d = decide_smoke(last, known_fixes_used=counters[0], critic_attempts=counters[1])
    assert (d.route, d.stop, d.capped) == ("ready", "passed", False)
    assert d.reason == f"smoke passed ({signal})"


def test_cap_reasons_name_the_cap() -> None:
    assert decide_smoke(
        _failed("known"), known_fixes_used=MAX_KNOWN_FIXES, critic_attempts=0
    ).reason.startswith(f"{MAX_KNOWN_FIXES} known fixes applied; ")
    assert decide_smoke(
        _failed("critic"), known_fixes_used=0, critic_attempts=MAX_CRITIC_ATTEMPTS
    ).reason.startswith(f"{MAX_CRITIC_ATTEMPTS} critic attempts used; ")


@pytest.mark.parametrize("cls", ["infra", "known", "critic", "unfixable"])
def test_no_auto_fix_stops_on_every_failure(cls: str) -> None:
    d = decide_smoke(_failed(cls), known_fixes_used=0, critic_attempts=0, auto_fix=False)
    assert (d.route, d.stop, d.capped) == ("stop", "no_auto_fix", False)


def test_critic_without_a_critic_stops() -> None:
    d = decide_smoke(_failed("critic"), known_fixes_used=0, critic_attempts=0, has_critic=False)
    assert (d.route, d.stop, d.capped) == ("stop", "critic", False)


def test_failed_without_classification_is_an_error() -> None:
    last = LastSmoke(passed=False, signal="traceback", cls=None, reason="", has_traceback=True)
    with pytest.raises(ValueError):
        decide_smoke(last, known_fixes_used=0, critic_attempts=0)


# --- route_smoke: decide_smoke -> node --------------------------------------------------------


@pytest.mark.parametrize(("cls", "tb", "counters", "expected"), TABLE)
def test_route_smoke_maps_decisions_to_nodes(
    cls: str, tb: bool, counters: tuple[int, int], expected: tuple[str, Any, bool]
) -> None:
    state = {
        "setup_status": "running",
        "last_smoke": _failed(cls, tb=tb),
        "known_fixes_used": counters[0],
        "critic_attempts": counters[1],
    }
    assert route_smoke(state) == SMOKE_TARGETS[expected[0]]  # type: ignore[index]


def test_route_smoke_targets() -> None:
    assert SMOKE_TARGETS == {
        "ready": END, "infra": END, "capped": END, "stop": END,
        "known": "apply_known_fix", "critic": "critic",
    }  # fmt: skip


@pytest.mark.parametrize("status", ["ready", "infra", "capped", "smoke_failed"])
def test_route_smoke_ends_when_the_node_finished(status: str) -> None:
    state = {"setup_status": status, "last_smoke": _failed("known"), "known_fixes_used": 0}
    assert route_smoke(state) == END
    assert route_smoke({"setup_status": "running"}) == END  # smoke node failed before a run


# --- the other routers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"status": "configured"}, "env"),
        ({"status": "configured", "until": "smoke"}, "env"),
        ({"status": "configured", "until": "env"}, "env"),
        ({"status": "configured", "until": "configure"}, END),
        ({"status": "configure_failed"}, END),
        ({"status": "running"}, END),
        ({}, END),
    ],
)
def test_route_configured(state: dict[str, Any], expected: str) -> None:
    assert route_configured(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"env_ok": True}, "data"),
        ({"env_ok": True, "data_ok": True}, "smoke"),  # rebuild after add_package
        ({"env_ok": True, "until": "env"}, END),
        ({"env_ok": False}, END),
        ({"env_ok": False, "data_ok": True}, END),
        ({}, END),
    ],
)
def test_route_env(state: dict[str, Any], expected: str) -> None:
    assert route_env(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"data_ok": True}, "smoke"),
        ({"data_ok": True, "until": "data"}, END),
        ({"data_ok": False}, END),
        ({}, END),
    ],
)
def test_route_data(state: dict[str, Any], expected: str) -> None:
    assert route_data(state) == expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("add_package", "env"), ("prep_dir", "smoke"), ("override", "smoke"), ("add_mount", "smoke")],
)
def test_route_fix(kind: str, expected: str) -> None:
    assert route_fix({"last_fix_kind": kind}) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [("patched", "smoke"), ("invalid", END), ("skipped", END), ("no_target", END),
     ("error", END), (None, END)],
)  # fmt: skip
def test_route_critic(status: str | None, expected: str) -> None:
    assert route_critic({} if status is None else {"last_critic_status": status}) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"setup_status": "ready", "status": "configured"}, "ready"),
        ({"setup_status": "capped", "status": "configured"}, "capped"),
        ({"setup_status": "running", "status": "configure_failed"}, "configure_failed"),
        ({"status": "configure_failed"}, "configure_failed"),
        ({"setup_status": "running", "status": "configured", "until": "env"}, "stopped"),
    ],
)
def test_final_setup_status(state: dict[str, Any], expected: str) -> None:
    assert final_setup_status(state) == expected
