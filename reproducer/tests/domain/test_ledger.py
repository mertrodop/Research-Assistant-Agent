"""Tests for reproducer.domain.ledger."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from reproducer.domain.contracts import ExperimentResult, ReproSpec, ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import (
    LedgerEntry,
    Tolerance,
    ToleranceConfig,
    compute_ledger,
    make_run_evidence,
    make_smoke_evidence,
    within_tolerance,
)
from reproducer.domain.unknowns import UnknownState, spec_to_evidence

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "repro_spec_example.json"
SOURCE = "paper:resnet18-cifar10"
METRIC = "top1_accuracy"
TOL = ToleranceConfig(default=Tolerance(abs=0.5))  # c1=95.3 passes in [94.8, 95.8]
ASSUMPTION_ID = "a" * 64


@pytest.fixture
def spec() -> ReproSpec:
    return ReproSpec.model_validate_json(FIXTURE.read_text())


@pytest.fixture
def claims(spec: ReproSpec) -> tuple[ResultClaim, ...]:
    return spec.claims


@pytest.fixture
def claim_evidence(spec: ReproSpec) -> dict[str, Evidence]:
    records, _ = spec_to_evidence(spec, SOURCE)
    return {ev.evidence_id: ev for ev in records}


def _run(
    run_id: str,
    value: float | None = 95.3,
    *,
    claim_id: str = "c1",
    exit_code: int = 0,
    faithful: bool | None = None,
    assumed: bool = False,
) -> Evidence:
    if faithful is None:
        faithful = not assumed
    metrics = {} if value is None else {METRIC: value}
    result = ExperimentResult(
        claim_id=claim_id,
        run_id=run_id,
        exit_code=exit_code,
        metrics=metrics,
        log_ref=f"logs/{run_id}.txt",
        config_digest="sha256:cfg",
        duration=10.0,
        gpu_hours=0.1,
    )
    return make_run_evidence(
        result,
        spec_faithful=faithful,
        assumption_evidence_ids=(ASSUMPTION_ID,) if assumed else (),
        source="run",
    )


def _open_unknown(claim_ids: tuple[str, ...] = ()) -> dict[str, UnknownState]:
    return {
        "unk:weight_decay": UnknownState(name="weight_decay", claim_ids=claim_ids, status="open")
    }


def _ledger(
    claims: tuple[ResultClaim, ...],
    claim_evidence: dict[str, Evidence],
    extra: list[Evidence],
    unknowns: dict[str, UnknownState] | None = None,
) -> dict[str, LedgerEntry]:
    ev = dict(claim_evidence)
    ev.update({e.evidence_id: e for e in extra})
    return compute_ledger(claims, ev, unknowns or {}, TOL)


# --- Tolerance -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ours", "paper", "tol", "expected"),
    [
        (95.0, 95.3, Tolerance(abs=0.5), True),  # abs only
        (94.7, 95.3, Tolerance(abs=0.5), False),
        (99.0, 100.0, Tolerance(rel=0.01), True),  # rel only
        (98.9, 100.0, Tolerance(rel=0.01), False),
        (98.5, 100.0, Tolerance(abs=2.0, rel=0.01), True),  # both: abs wins
        (98.5, 100.0, Tolerance(abs=0.1, rel=0.02), True),  # both: rel wins
        (97.0, 100.0, Tolerance(abs=2.0, rel=0.02), False),
        (1.5, 1.0, Tolerance(abs=0.5), True),  # boundary equality (exact in binary)
        (75.0, 100.0, Tolerance(rel=0.25), True),  # boundary equality, rel
        (0.0, 0.0, Tolerance(rel=0.1), True),  # paper=0 with rel only: exact match passes
        (math.nan, 1.0, Tolerance(abs=10.0), False),
        (math.inf, 1.0, Tolerance(abs=10.0), False),
    ],
)
def test_within_tolerance(ours: float, paper: float, tol: Tolerance, expected: bool) -> None:
    assert within_tolerance(ours, paper, tol) is expected


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"abs": -0.1}, {"rel": -1.0}, {"abs": math.inf}, {"rel": math.nan}],
)
def test_tolerance_invalid(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Tolerance(**kwargs)


@pytest.mark.parametrize(
    ("per_metric", "per_claim", "expected"),
    [
        ({}, {}, Tolerance(abs=0.5)),
        ({METRIC: Tolerance(abs=1.0)}, {}, Tolerance(abs=1.0)),
        ({METRIC: Tolerance(abs=1.0)}, {"c1": Tolerance(rel=0.1)}, Tolerance(rel=0.1)),
        ({"other": Tolerance(abs=1.0)}, {"c2": Tolerance(rel=0.1)}, Tolerance(abs=0.5)),
    ],
)
def test_tolerance_lookup_precedence(
    claims: tuple[ResultClaim, ...],
    per_metric: dict[str, Tolerance],
    per_claim: dict[str, Tolerance],
    expected: Tolerance,
) -> None:
    cfg = ToleranceConfig(default=Tolerance(abs=0.5), per_metric=per_metric, per_claim=per_claim)
    assert cfg.tolerance_for(claims[0]) == expected


# --- Run / smoke evidence ------------------------------------------------------------------


def test_run_evidence_fields() -> None:
    ev = _run("r1", 95.0, assumed=True)
    assert ev.kind == "run_result"
    assert ev.authority == "execution"
    assert ev.confidence == "observed"
    assert ev.locator == "r1"
    assert ev.links == (ASSUMPTION_ID,)
    assert ev.content["assumptions_used"] is True
    assert ev.content["spec_faithful"] is False
    assert ev.content["diverged"] is False
    for key in ExperimentResult.model_fields:
        assert key in ev.content


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_run_evidence_non_finite_marks_diverged(bad: float) -> None:
    ev = _run("r1", bad)
    assert ev.content["metrics"][METRIC] is None
    assert ev.content["diverged"] is True


def test_smoke_evidence_fields() -> None:
    ev = make_smoke_evidence(passed=False, attempts=3, capped=True, source="s", locator="smoke")
    assert ev.kind == "smoke_result"
    assert ev.authority == "execution"
    assert ev.confidence == "observed"
    assert ev.content == {"passed": False, "attempts": 3, "capped": True}


# --- Ledger status -------------------------------------------------------------------------


def test_verified_spec_faithful(claims, claim_evidence) -> None:
    run = _run("r1", 95.1)
    entry = _ledger(claims, claim_evidence, [run])["c1"]
    assert entry.status == "verified"
    assert entry.fidelity == "spec_faithful"
    assert entry.attempts == 1


@pytest.mark.parametrize(
    ("faithful", "assumed"),
    [(False, False), (False, True)],
)
def test_verified_inferred(claims, claim_evidence, faithful: bool, assumed: bool) -> None:
    run = _run("r1", 95.3, faithful=faithful, assumed=assumed)
    entry = _ledger(claims, claim_evidence, [run])["c1"]
    assert entry.status == "verified"
    assert entry.fidelity == "inferred"


def test_verified_prefers_spec_faithful_run(claims, claim_evidence) -> None:
    inferred_exact = _run("r1", 95.3, assumed=True)
    faithful_close = _run("r2", 95.6)
    entry = _ledger(claims, claim_evidence, [inferred_exact, faithful_close])["c1"]
    assert entry.status == "verified"
    assert entry.fidelity == "spec_faithful"
    # best_run is still the smallest abs error overall
    assert entry.best_run_evidence_id == inferred_exact.evidence_id


@pytest.mark.parametrize(
    ("runs", "unknowns"),
    [
        pytest.param([], {}, id="no-runs"),
        pytest.param([_run("r1", 90.0), _run("r2", 90.0)], {}, id="out-of-tol-below-max"),
        pytest.param(
            [_run("r1", 90.0), _run("r2", 90.0), _run("r3", 90.0, assumed=True)],
            {},
            id="max-tries-with-assumption",
        ),
        pytest.param(
            [_run("r1", 90.0), _run("r2", 90.0), _run("r3", 90.0, faithful=False)],
            {},
            id="max-tries-not-spec-faithful",
        ),
        pytest.param(
            [_run("r1", 90.0), _run("r2", 90.0), _run("r3", 90.0)],
            _open_unknown(),
            id="max-tries-global-unknown-open",
        ),
        pytest.param(
            [_run("r1", 90.0), _run("r2", 90.0), _run("r3", 90.0)],
            _open_unknown(("c1",)),
            id="max-tries-claim-unknown-open",
        ),
        pytest.param(
            [_run("r1", exit_code=1), _run("r2", exit_code=1), _run("r3", exit_code=1)],
            {},
            id="max-tries-no-completed-run",
        ),
    ],
)
def test_open(claims, claim_evidence, runs: list[Evidence], unknowns) -> None:
    entry = _ledger(claims, claim_evidence, runs, unknowns)["c1"]
    assert entry.status == "open"
    assert entry.fidelity is None
    assert entry.attempts == len(runs)


def test_open_unknowns_listed(claims, claim_evidence) -> None:
    unknowns = {
        **_open_unknown(),
        "unk:lr": UnknownState(name="lr", claim_ids=("c2",), status="open"),
        "unk:bs": UnknownState(name="bs", claim_ids=(), status="resolved", value="128"),
    }
    ledger = _ledger(claims, claim_evidence, [], unknowns)
    assert ledger["c1"].open_unknowns == ("unk:weight_decay",)
    assert ledger["c2"].open_unknowns == ("unk:lr", "unk:weight_decay")


@pytest.mark.parametrize(
    "unknowns",
    [
        pytest.param({}, id="no-unknowns"),
        pytest.param(_open_unknown(("c2",)), id="unknown-scoped-to-other-claim"),
        pytest.param(
            {
                "unk:weight_decay": UnknownState(
                    name="weight_decay", claim_ids=(), status="resolved", value="5e-4"
                )
            },
            id="unknown-resolved",
        ),
    ],
)
def test_contradicted(claims, claim_evidence, unknowns) -> None:
    runs = [_run("r1", 90.0), _run("r2", 91.0), _run("r3", 92.0)]
    entry = _ledger(claims, claim_evidence, runs, unknowns)["c1"]
    assert entry.status == "contradicted"
    assert entry.fidelity is None
    assert entry.attempts == 3


def test_blocked(claims, claim_evidence) -> None:
    smoke = make_smoke_evidence(passed=False, attempts=3, capped=True, source="s", locator="sm")
    ledger = _ledger(claims, claim_evidence, [smoke])
    assert ledger["c1"].status == "blocked"
    assert ledger["c2"].status == "blocked"


@pytest.mark.parametrize(
    ("passed", "capped", "with_run"),
    [(False, True, True), (False, False, False), (True, True, False)],
)
def test_not_blocked(claims, claim_evidence, passed: bool, capped: bool, with_run: bool) -> None:
    smoke = make_smoke_evidence(passed=passed, attempts=3, capped=capped, source="s", locator="x")
    extra = [smoke, _run("r1", exit_code=1)] if with_run else [smoke]
    assert _ledger(claims, claim_evidence, extra)["c1"].status == "open"


def test_diverged_run_counts_as_attempt_not_completed(claims, claim_evidence) -> None:
    runs = [_run("r1", math.nan), _run("r2", math.nan), _run("r3", math.nan)]
    entry = _ledger(claims, claim_evidence, runs)["c1"]
    assert entry.attempts == 3
    assert entry.status == "open"
    assert entry.best_run_evidence_id is None
    assert entry.best_abs_error is None


def test_other_metric_nan_still_verified(claims, claim_evidence) -> None:
    result = ExperimentResult(
        claim_id="c1",
        run_id="r1",
        exit_code=0,
        metrics={METRIC: 95.3, "loss": math.nan},
        log_ref="l",
        config_digest="d",
        duration=1.0,
        gpu_hours=0.0,
    )
    ev = make_run_evidence(result, spec_faithful=True, assumption_evidence_ids=(), source="run")
    assert ev.content["diverged"] is True
    assert ev.content["metrics"] == {METRIC: 95.3, "loss": None}
    entry = _ledger(claims, claim_evidence, [ev])["c1"]
    assert entry.status == "verified"
    assert entry.fidelity == "spec_faithful"
    assert entry.best_run_evidence_id == ev.evidence_id


def test_spec_faithful_with_assumptions_rejected() -> None:
    with pytest.raises(ValueError, match="spec_faithful=True is incompatible"):
        _run("r1", 95.3, faithful=True, assumed=True)


def test_missing_metric_not_completed(claims, claim_evidence) -> None:
    run = _run("r1", None)
    entry = _ledger(claims, claim_evidence, [run])["c1"]
    assert entry.status == "open"
    assert entry.attempts == 1
    assert entry.best_run_evidence_id is None


def test_best_run_and_evidence_ids(claims, claim_evidence) -> None:
    r1 = _run("r1", 90.0)
    r2 = _run("r2", 93.3)
    r3 = _run("r3", exit_code=1)
    other = _run("x1", 93.1, claim_id="c2")
    ledger = _ledger(claims, claim_evidence, [r1, r2, r3, other])
    c1 = ledger["c1"]
    assert c1.best_run_evidence_id == r2.evidence_id
    assert c1.best_abs_error == pytest.approx(2.0)
    claim_ids = [
        eid
        for eid, ev in claim_evidence.items()
        if ev.kind == "paper_claim" and ev.content["claim_id"] == "c1"
    ]
    assert c1.evidence_ids == (*claim_ids, r1.evidence_id, r2.evidence_id, r3.evidence_id)
    assert ledger["c2"].status == "verified"
    assert ledger["c2"].evidence_ids[-1] == other.evidence_id


def test_malformed_run_content_raises(claims, claim_evidence) -> None:
    bad = make_evidence(
        kind="run_result",
        source="run",
        locator="r1",
        content={"claim_id": "c1"},
        confidence="observed",
        authority="execution",
    )
    with pytest.raises(ValueError, match="malformed run_result"):
        _ledger(claims, claim_evidence, [bad])


def test_ledger_does_not_mutate_inputs(claims, claim_evidence) -> None:
    before = (claims, dict(claim_evidence), TOL.model_dump())
    _ledger(claims, claim_evidence, [_run("r1", 90.0)])
    assert before == (claims, dict(claim_evidence), TOL.model_dump())
