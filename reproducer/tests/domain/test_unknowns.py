"""Tests for reproducer.domain.unknowns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.evidence import Evidence, make_evidence, to_jsonl_line
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    event_from_jsonl_line,
    fold_unknowns,
    resolve,
    spec_to_evidence,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "repro_spec_example.json"
SOURCE = "paper:resnet18-cifar10"


@pytest.fixture
def spec() -> ReproSpec:
    return ReproSpec.model_validate_json(FIXTURE.read_text())


@pytest.fixture
def converted(spec: ReproSpec) -> tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]]:
    return spec_to_evidence(spec, SOURCE)


@pytest.fixture
def opened(converted: tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]]) -> UnknownOpened:
    return converted[1][0]


def _repo_ev(value: str = "5e-4", **overrides: Any) -> Evidence:
    kwargs: dict[str, Any] = {
        "kind": "repo_file",
        "source": "example-lab/resnet18@abc123",
        "locator": "configs/resnet18.yaml:L7",
        "content": {"weight_decay": value},
        "confidence": "observed",
        "authority": "repo",
    }
    kwargs.update(overrides)
    return make_evidence(**kwargs)


def _llm_ev() -> Evidence:
    return make_evidence(
        kind="critic_patch",
        source="run-1",
        locator="llm",
        content={"weight_decay": "1e-4"},
        confidence="inferred",
        authority="llm",
    )


@pytest.fixture
def index(converted: tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]]) -> dict[str, Evidence]:
    evs = [*converted[0], _repo_ev(), _repo_ev("1e-4", locator="other"), _llm_ev()]
    return {e.evidence_id: e for e in evs}


# --- spec_to_evidence ----------------------------------------------------------------------


def test_spec_to_evidence_counts(
    converted: tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]],
) -> None:
    records, unknowns = converted
    kinds = [r.kind for r in records]
    assert kinds.count("paper_claim") == 2
    assert kinds.count("paper_setup") == 3
    assert all(r.confidence == "observed" and r.authority == "paper" for r in records)
    assert all(r.source == SOURCE for r in records)
    assert len(unknowns) == 1


def test_spec_to_evidence_content(
    converted: tuple[tuple[Evidence, ...], tuple[UnknownOpened, ...]],
) -> None:
    records, (unk,) = converted
    claim = records[0]
    assert claim.locator == "Table 1"
    assert "location" not in claim.content
    assert claim.content["claim_id"] == "c1"
    assert claim.content["unit"] == "%"
    wd = next(r for r in records if r.kind == "paper_setup" and r.content["name"] == "weight_decay")
    assert wd.content == {
        "name": "weight_decay",
        "value": None,
        "quote": "We use standard weight decay.",
    }
    assert unk.unknown_id == "unk:weight_decay"
    assert unk.name == "weight_decay"
    assert unk.claim_ids == ()
    assert unk.evidence_id == wd.evidence_id


def test_opened_rejects_bad_id() -> None:
    with pytest.raises(ValidationError):
        UnknownOpened(unknown_id="weight_decay", name="weight_decay", evidence_id="e")


# --- resolve -------------------------------------------------------------------------------


def test_resolve_accepts_repo_observed(opened: UnknownOpened) -> None:
    ev = _repo_ev()
    r = resolve(opened, ev, "5e-4")
    assert r == UnknownResolved(
        unknown_id="unk:weight_decay", value="5e-4", evidence_id=ev.evidence_id
    )


def test_resolve_rejects_llm_authority(opened: UnknownOpened) -> None:
    with pytest.raises(ValueError, match="authority"):
        resolve(opened, _repo_ev(authority="llm"), "5e-4")


def test_resolve_rejects_inferred_confidence(opened: UnknownOpened) -> None:
    with pytest.raises(ValueError, match="confidence"):
        resolve(opened, _repo_ev(confidence="inferred"), "5e-4")


# --- fold ----------------------------------------------------------------------------------


def test_fold_open_then_resolved(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    ev = _repo_ev()
    states = fold_unknowns([opened], index)
    assert states["unk:weight_decay"].status == "open"
    states = fold_unknowns([opened, resolve(opened, ev, "5e-4")], index)
    s = states["unk:weight_decay"]
    assert (s.status, s.value, s.resolved_by) == ("resolved", "5e-4", ev.evidence_id)


def test_fold_assumption_keeps_open(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    a = Assumption(unknown_id=opened.unknown_id, value="1e-4", evidence_id=_llm_ev().evidence_id)
    s = fold_unknowns([opened, a], index)["unk:weight_decay"]
    assert s.status == "open"
    assert s.value is None
    assert s.assumptions == (a,)


def test_fold_assumption_before_resolution_kept(
    opened: UnknownOpened, index: dict[str, Evidence]
) -> None:
    a = Assumption(unknown_id=opened.unknown_id, value="1e-4", evidence_id=_llm_ev().evidence_id)
    s = fold_unknowns([opened, a, resolve(opened, _repo_ev(), "5e-4")], index)["unk:weight_decay"]
    assert (s.status, s.value) == ("resolved", "5e-4")
    assert s.assumptions == (a,)


def test_fold_assumption_after_resolution_raises(
    opened: UnknownOpened, index: dict[str, Evidence]
) -> None:
    a = Assumption(unknown_id=opened.unknown_id, value="1e-4", evidence_id=_llm_ev().evidence_id)
    with pytest.raises(ValueError, match="already resolved; assumption not allowed"):
        fold_unknowns([opened, resolve(opened, _repo_ev(), "5e-4"), a], index)


def test_fold_assumption_requires_stored_evidence(
    opened: UnknownOpened, index: dict[str, Evidence]
) -> None:
    a = Assumption(unknown_id=opened.unknown_id, value="1e-4", evidence_id="not-stored")
    with pytest.raises(ValueError, match="evidence_id"):
        fold_unknowns([opened, a], index)


def test_fold_same_value_idempotent(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    first = _repo_ev()
    second = _repo_ev(locator="other", content={"weight_decay": "5e-4", "dup": True})
    index = {**index, second.evidence_id: second}
    events = [opened, resolve(opened, first, "5e-4"), resolve(opened, second, "5e-4")]
    s = fold_unknowns(events, index)["unk:weight_decay"]
    assert s.resolved_by == first.evidence_id


def test_fold_different_value_raises(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    events = [
        opened,
        resolve(opened, _repo_ev(), "5e-4"),
        resolve(opened, _repo_ev("1e-4", locator="other"), "1e-4"),
    ]
    with pytest.raises(ValueError, match="conflicting resolution"):
        fold_unknowns(events, index)


def test_fold_unopened_raises(index: dict[str, Evidence]) -> None:
    eid = _repo_ev().evidence_id
    with pytest.raises(ValueError, match="unopened"):
        fold_unknowns([UnknownResolved(unknown_id="unk:x", value="1", evidence_id=eid)], index)
    with pytest.raises(ValueError, match="unopened"):
        fold_unknowns([Assumption(unknown_id="unk:x", value="1", evidence_id=eid)], index)


def test_fold_missing_evidence_raises(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    with pytest.raises(ValueError, match="evidence_id"):
        fold_unknowns([opened], {})
    bad = UnknownResolved(unknown_id=opened.unknown_id, value="1", evidence_id="missing")
    with pytest.raises(ValueError, match="evidence_id"):
        fold_unknowns([opened, bad], index)


def test_fold_rechecks_resolution_policy(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    # Hand-built Resolved bypassing resolve(): fold must still reject llm/inferred evidence.
    forged = UnknownResolved(
        unknown_id=opened.unknown_id, value="1e-4", evidence_id=_llm_ev().evidence_id
    )
    with pytest.raises(ValueError, match="confidence|authority"):
        fold_unknowns([opened, forged], index)


def test_fold_reopen(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    assert fold_unknowns([opened, opened], index)["unk:weight_decay"].status == "open"
    differing = opened.model_copy(update={"claim_ids": ("c1",)})
    with pytest.raises(ValueError, match="re-open"):
        fold_unknowns([opened, differing], index)


# --- event JSONL ---------------------------------------------------------------------------


def test_mixed_event_jsonl_round_trip(opened: UnknownOpened, index: dict[str, Evidence]) -> None:
    events = [
        opened,
        Assumption(unknown_id=opened.unknown_id, value="1e-4", evidence_id=_llm_ev().evidence_id),
        resolve(opened, _repo_ev(), "5e-4"),
    ]
    lines = [to_jsonl_line(e) for e in events]
    assert [line.count('"event":') for line in lines] == [1, 1, 1]
    reloaded = [event_from_jsonl_line(line) for line in lines]
    assert reloaded == events
    assert [type(e) for e in reloaded] == [UnknownOpened, Assumption, UnknownResolved]
    assert fold_unknowns(reloaded, index) == fold_unknowns(events, index)


def test_event_from_jsonl_rejects_unknown_tag() -> None:
    with pytest.raises(ValidationError):
        event_from_jsonl_line(
            '{"event":"bogus","unknown_id":"unk:x","value":"1","evidence_id":"e"}'
        )
