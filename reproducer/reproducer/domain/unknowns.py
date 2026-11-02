"""Unknowns: setup details not stated by paper or repo.

Event-sourced and append-only. An Unknown is never guessed: it is resolved only by
observed/derived evidence from a trusted authority. Assumptions let a run proceed but do not
resolve the Unknown (the ledger later marks such claims fidelity=inferred).
Pure domain: no file, network or subprocess IO.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.evidence import Evidence, make_evidence

UNKNOWN_PREFIX = "unk:"
RESOLVING_CONFIDENCE = frozenset({"observed", "derived"})
RESOLVING_AUTHORITY = frozenset({"paper", "repo", "execution", "human"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def unknown_id_for(name: str) -> str:
    return UNKNOWN_PREFIX + name


# --- Events --------------------------------------------------------------------------------


class UnknownOpened(_Frozen):
    event: Literal["opened"] = "opened"
    unknown_id: str
    name: str
    claim_ids: tuple[str, ...] = ()  # empty = global
    evidence_id: str  # the paper_setup evidence with value None

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        if self.unknown_id != unknown_id_for(self.name):
            raise ValueError(f"unknown_id {self.unknown_id!r} != {unknown_id_for(self.name)!r}")
        return self


class UnknownResolved(_Frozen):
    """Build via resolve(...) so the evidence policy is enforced."""

    event: Literal["resolved"] = "resolved"
    unknown_id: str
    value: str
    evidence_id: str


class Assumption(_Frozen):
    """Inferred / LLM-proposed value. Does NOT resolve the Unknown."""

    event: Literal["assumption"] = "assumption"
    unknown_id: str
    value: str
    evidence_id: str


UnknownEvent = Annotated[
    Union[UnknownOpened, UnknownResolved, Assumption],  # noqa: UP007
    Field(discriminator="event"),
]
_EVENT_ADAPTER: TypeAdapter[UnknownEvent] = TypeAdapter(UnknownEvent)


def event_from_jsonl_line(line: str) -> UnknownOpened | UnknownResolved | Assumption:
    """Parse one JSONL line into the event type named by its `event` field."""
    return _EVENT_ADAPTER.validate_json(line.strip())


# --- Resolution policy ---------------------------------------------------------------------


def _check_resolving_evidence(evidence: Evidence) -> None:
    if evidence.confidence not in RESOLVING_CONFIDENCE:
        raise ValueError(
            f"evidence {evidence.evidence_id} has confidence {evidence.confidence!r}; "
            f"resolution requires one of {sorted(RESOLVING_CONFIDENCE)}"
        )
    if evidence.authority not in RESOLVING_AUTHORITY:
        raise ValueError(
            f"evidence {evidence.evidence_id} has authority {evidence.authority!r}; "
            f"resolution requires one of {sorted(RESOLVING_AUTHORITY)}"
        )


def resolve(opened: UnknownOpened, evidence: Evidence, value: str) -> UnknownResolved:
    """Resolve an Unknown; raises ValueError unless the evidence is observed/derived and
    comes from paper/repo/execution/human."""
    _check_resolving_evidence(evidence)
    return UnknownResolved(
        unknown_id=opened.unknown_id, value=value, evidence_id=evidence.evidence_id
    )


# --- Fold ----------------------------------------------------------------------------------


class UnknownState(_Frozen):
    name: str
    claim_ids: tuple[str, ...]
    status: Literal["open", "resolved"]
    value: str | None = None
    resolved_by: str | None = None
    assumptions: tuple[Assumption, ...] = ()


def fold_unknowns(
    events: Iterable[UnknownOpened | UnknownResolved | Assumption],
    evidence: Mapping[str, Evidence],
) -> dict[str, UnknownState]:
    """Fold events (in order) into per-unknown state.

    Raises ValueError on: evidence_id missing from the index; Resolved/Assumption for an
    unopened unknown; a Resolved whose evidence fails the resolve() policy; a second Resolved
    with a different value (conflict -> needs human); an Assumption for an already-resolved
    unknown; a re-Open differing from the first.
    """
    opened: dict[str, UnknownOpened] = {}
    states: dict[str, UnknownState] = {}
    for ev in events:
        if ev.evidence_id not in evidence:
            raise ValueError(
                f"{ev.event} event for {ev.unknown_id!r} references unknown "
                f"evidence_id {ev.evidence_id!r}"
            )
        if isinstance(ev, UnknownOpened):
            prior = opened.get(ev.unknown_id)
            if prior is None:
                opened[ev.unknown_id] = ev
                states[ev.unknown_id] = UnknownState(
                    name=ev.name, claim_ids=ev.claim_ids, status="open"
                )
            elif prior != ev:
                raise ValueError(f"conflicting re-open of {ev.unknown_id!r}")
            continue

        state = states.get(ev.unknown_id)
        if state is None:
            raise ValueError(f"{ev.event} event for unopened unknown {ev.unknown_id!r}")

        if isinstance(ev, UnknownResolved):
            _check_resolving_evidence(evidence[ev.evidence_id])
            if state.status == "resolved":
                if state.value != ev.value:
                    raise ValueError(
                        f"conflicting resolution for {ev.unknown_id!r}: "
                        f"{state.value!r} vs {ev.value!r} (needs human)"
                    )
                continue  # idempotent; keep first resolved_by
            states[ev.unknown_id] = state.model_copy(
                update={"status": "resolved", "value": ev.value, "resolved_by": ev.evidence_id}
            )
        else:  # Assumption
            if state.status == "resolved":
                raise ValueError(
                    f"unknown already resolved; assumption not allowed ({ev.unknown_id!r})"
                )
            states[ev.unknown_id] = state.model_copy(
                update={"assumptions": (*state.assumptions, ev)}
            )
    return states


# --- Spec -> evidence ----------------------------------------------------------------------


def spec_to_evidence(
    spec: ReproSpec, source: str
) -> tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]]:
    """One paper_claim per ResultClaim, one paper_setup per SetupField, and one
    UnknownOpened (global, claim_ids=()) per SetupField with value None."""
    records: list[Evidence] = []
    unknowns: list[UnknownOpened] = []
    for claim in spec.claims:
        records.append(
            make_evidence(
                kind="paper_claim",
                source=source,
                locator=claim.location,
                content=claim.model_dump(mode="json", exclude={"location"}),
                confidence="observed",
                authority="paper",
            )
        )
    for field in spec.setup:
        ev = make_evidence(
            kind="paper_setup",
            source=source,
            locator=field.location,
            content={"name": field.name, "value": field.value, "quote": field.quote},
            confidence="observed",
            authority="paper",
        )
        records.append(ev)
        if field.value is None:
            unknowns.append(
                UnknownOpened(
                    unknown_id=unknown_id_for(field.name),
                    name=field.name,
                    claim_ids=(),
                    evidence_id=ev.evidence_id,
                )
            )
    return tuple(records), tuple(unknowns)
