"""Graph state for the configure graph, with its pure reducers.

Parallel branches (subtask_a/b/c) return partial updates; LangGraph merges each key with
its reducer. Reducers are pure and give the same result whatever order the branches
finish in (events: same set; their relative order is kept within one update).
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

from reproducer.domain.evidence import Evidence
from reproducer.domain.unknowns import Assumption, UnknownOpened, UnknownResolved

if TYPE_CHECKING:
    from reproducer.providers.llm import Budget, MeteredLLM
    from reproducer.providers.store import RunStore

UnknownEventT = UnknownOpened | UnknownResolved | Assumption
ConfigureStatus = Literal["running", "configured", "configure_failed"]
MAX_CONFIGURE_ATTEMPTS = 1  # M1: one pass; the M2 critic loop raises this


def merge_evidence(
    left: Iterable[Evidence] | None, right: Iterable[Evidence] | None
) -> list[Evidence]:
    """Union by evidence_id (ids are content hashes, so equal id = equal record), sorted by
    (kind, locator, evidence_id): the result does not depend on merge order."""
    by_id: dict[str, Evidence] = {}
    for ev in (*(left or ()), *(right or ())):
        by_id.setdefault(ev.evidence_id, ev)
    return sorted(by_id.values(), key=lambda e: (e.kind, e.locator, e.evidence_id))


def event_key(event: UnknownEventT) -> str:
    """Identity for exact-duplicate detection: event type + all fields."""
    return f"{type(event).__name__}:{event.model_dump_json()}"


def merge_events(
    left: Iterable[UnknownEventT] | None, right: Iterable[UnknownEventT] | None
) -> list[UnknownEventT]:
    """Append right to left, dropping events equal to one already present (exact duplicate:
    same type and same fields). Order is otherwise kept (opened before resolved)."""
    out: list[UnknownEventT] = []
    seen: set[str] = set()
    for event in (*(left or ()), *(right or ())):
        key = event_key(event)
        if key not in seen:
            seen.add(key)
            out.append(event)
    return out


class SubtaskSummary(BaseModel):
    """What one subtask branch reports to the fan-in (no payload: that is in evidence)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["A", "B", "C"]
    ok: bool
    attempts: int
    tokens: int
    latency_s: float
    errors: tuple[str, ...] = ()


class ConfigureState(TypedDict, total=False):
    # inputs
    out_dir: str
    repo_url: str
    commit: str | None
    max_chars: int
    max_tokens: int
    # produced by the graph
    source: str
    evidence: Annotated[list[Evidence], merge_evidence]
    events: Annotated[list[UnknownEventT], merge_events]
    outcomes: Annotated[list[SubtaskSummary], operator.add]
    status: ConfigureStatus
    configure_attempts: int  # bounded by MAX_CONFIGURE_ATTEMPTS
    evidence_written: int  # set by persist
    events_written: int  # set by persist


class ConfigureDeps(TypedDict):
    """Injected IO. store: the single writer target (persist + telemetry); llm_for(node) ->
    MeteredLLM sharing `budget`."""

    store: RunStore
    llm_for: Callable[[str], MeteredLLM]
    budget: Budget
