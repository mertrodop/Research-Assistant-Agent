"""Persist node: the configure graph's fan-in and its single writer of evidence/events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reproducer.domain.unknowns import UnknownOpened
from reproducer.graph.nodes.subtasks import SUBTASK_KINDS
from reproducer.graph.state import event_key


def persist(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Fan-in, single writer: append evidence (store dedupes by id) and events not already
    stored (an UnknownOpened whose unknown is already open is skipped: first wins).
    Status: configured iff every subtask reported ok."""
    store = deps["store"]
    ev_written = sum(store.append_evidence(ev) for ev in state.get("evidence", ()))
    stored = store.load_events()
    seen = {event_key(e) for e in stored}
    opened = {e.unknown_id for e in stored if isinstance(e, UnknownOpened)}
    n_events = 0
    for event in state.get("events", ()):
        key = event_key(event)
        if key in seen or (isinstance(event, UnknownOpened) and event.unknown_id in opened):
            continue
        store.append_event(event)
        seen.add(key)
        if isinstance(event, UnknownOpened):
            opened.add(event.unknown_id)
        n_events += 1
    outcomes = state.get("outcomes", ())
    kinds = {o.kind for o in outcomes}
    ok = kinds == set(SUBTASK_KINDS) and all(o.ok for o in outcomes)
    return {
        "status": "configured" if ok else "configure_failed",
        "evidence_written": ev_written,
        "events_written": n_events,
    }
