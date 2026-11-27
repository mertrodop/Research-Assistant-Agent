"""Tests for reproducer.domain.evaluate (evaluator + inferred reasons) and the ledger's use of
pre-registered tolerance evidence."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import C10, C100

from reproducer.domain.contracts import ExperimentResult, ResultClaim
from reproducer.domain.evaluate import evaluate_claims
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import (
    Tolerance,
    ToleranceConfig,
    compute_ledger,
    make_aborted_run_evidence,
    make_run_evidence,
    paper_num_runs,
    tolerance_evidence,
)
from reproducer.domain.unknowns import Assumption, UnknownState

REPO = "repo:uoguelph-mlrg/Cutout@abc"
FALLBACK = ToleranceConfig(default=Tolerance(abs=0.05))


def _paper_claim_id(evidence: dict[str, Evidence], cid: str) -> str:
    return next(
        i for i, ev in evidence.items()
        if ev.kind == "paper_claim" and ev.content["claim_id"] == cid
    )  # fmt: skip


def _with_tolerances(
    evidence: dict[str, Evidence], claims: dict[str, ResultClaim]
) -> dict[str, Evidence]:
    out = dict(evidence)
    for cid, c in claims.items():
        t = tolerance_evidence(c, claim_evidence_id=_paper_claim_id(evidence, cid), source="p")
        out[t.evidence_id] = t
    return out


def _env_plan(deviations: list[dict[str, Any]] | None = None) -> Evidence:
    plan = {
        "base_image": "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
        "deviations": deviations if deviations is not None else [{
            "field": "torch", "stated": "0.4.0", "chosen": "1.12.1",
            "reason": "no CUDA build for sm_89", "rule_source": ["https://x"],
        }],
        "assumptions": [
            {"name": "python_version", "value": "3.7", "reason": "not stated"},
            {"name": "cuda_version", "value": "11.3", "reason": "not stated"},
            {"name": "unpinned:tqdm", "value": None, "reason": "unpinned"},
        ],
    }  # fmt: skip
    return make_evidence(kind="env_spec", source=REPO, locator="env_plan",
                         content={"plan": plan}, confidence="derived", authority="rule",
                         links=("e" * 64,))  # fmt: skip


def _run_plan(cid: str, *, deviation_note: bool = False) -> Evidence:
    notes = [
        {"field": "seed", "status": "assumption", "detail": "repo default --seed 0"},
        {"field": "eval_checkpoint", "status": "assumption", "detail": "which=last"},
        {"field": "batch_size", "status": "consistent", "detail": "128 = repo default 128"},
    ]
    if deviation_note:
        notes.append({"field": "momentum", "status": "deviation", "detail": "repo uses 0.8"})
    content = {
        "claim_id": cid,
        "plan": {"depends_on_unknowns": ["unk:eval_checkpoint", "unk:seed"]},
        "assumptions": [
            {"event": "assumption", "unknown_id": "unk:seed", "value": "0", "evidence_id": "m"},
            {"event": "assumption", "unknown_id": "unk:eval_checkpoint", "value": "last",
             "evidence_id": "m"},
        ],
        "notes": notes,
        "deviations": [{"field": "lr_schedule", "stated": "x5", "chosen": "x10",
                        "reason": "hard-coded", "rule_source": ["train.py"]}]
        if deviation_note else [],
    }  # fmt: skip
    return make_evidence(kind="run_plan", source=REPO, locator=f"run_plan:{cid}",
                         content=content, confidence="derived", authority="rule",
                         links=("m" * 64,))  # fmt: skip


def _run(
    cid: str,
    value: float | None,
    *,
    run_id: str = "r1",
    faithful: bool = False,
    links: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = ("a" * 64,),
    digest: str = "D" * 64,
    exit_code: int = 0,
) -> Evidence:
    result = ExperimentResult(
        claim_id=cid, run_id=run_id, exit_code=exit_code,
        metrics={} if value is None else {"raw:test_acc": 1 - value / 100, "test_error": value},
        log_ref="runs/x/run.log", config_digest=digest, duration=3600.0, gpu_hours=1.0,
    )  # fmt: skip
    return make_run_evidence(
        result,
        spec_faithful=faithful,
        assumption_evidence_ids=() if faithful else assumptions,
        source=REPO,
        links=links,
    )


def _states(plan_id: str, *, open_: bool = True) -> dict[str, UnknownState]:
    status = "open" if open_ else "resolved"
    return {
        "unk:seed": UnknownState(name="seed", claim_ids=(), status=status, assumptions=(
            Assumption(unknown_id="unk:seed", value="0", evidence_id=plan_id),)),
        "unk:eval_checkpoint": UnknownState(name="eval_checkpoint", claim_ids=(), status=status,
                                            assumptions=(Assumption(
                                                unknown_id="unk:eval_checkpoint", value="last",
                                                evidence_id=plan_id),)),
        "unk:python_version": UnknownState(name="python_version", claim_ids=(), status=status),
        "unk:cuda_version": UnknownState(name="cuda_version", claim_ids=(), status=status),
    }  # fmt: skip


def _store(*records: Evidence, base: dict[str, Evidence]) -> dict[str, Evidence]:
    out = dict(base)
    for r in records:
        out[r.evidence_id] = r
    return out


@pytest.fixture
def base(
    cutout_evidence: dict[str, Evidence], cutout_claims: dict[str, ResultClaim]
) -> dict[str, Evidence]:
    return _with_tolerances(cutout_evidence, cutout_claims)


@pytest.fixture
def claims(cutout_claims: dict[str, ResultClaim]) -> tuple[ResultClaim, ...]:
    return (cutout_claims[C10], cutout_claims[C100])


# --- verdicts --------------------------------------------------------------------------------


def test_within_tolerance_inferred_lists_every_reason(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    env = _env_plan()
    plan = _run_plan(C10, deviation_note=True)
    fix = make_evidence(kind="known_fix", source=REPO, locator="known_fix:1", content={
        "action": {"kind": "batch", "params": {"batch_size": 64}, "deviation": "batch 128 -> 64"},
    }, confidence="derived", authority="rule")  # fmt: skip
    benign = make_evidence(kind="known_fix", source=REPO, locator="known_fix:2", content={
        "action": {"kind": "prep_dir", "params": {"dir": "logs"}, "deviation": None},
    }, confidence="derived", authority="rule")  # fmt: skip

    def patch(pid: str) -> Evidence:
        return make_evidence(kind="critic_patch", source=REPO, locator=f"critic_patch:{pid}",
                             content={"ok": True, "patch_id": pid * 12, "file": "train.py",
                                      "rationale": f"fix {pid}"},
                             confidence="inferred", authority="llm")  # fmt: skip

    kept, dropped = patch("a"), patch("b")
    revert = make_evidence(kind="critic_revert", source=REPO, locator="critic_revert:1",
                           content={"patch_id": "b" * 12}, confidence="derived",
                           authority="rule")  # fmt: skip
    run = _run(C10, 4.10, links=(plan.evidence_id,))
    ev = _store(env, plan, fix, benign, kept, dropped, revert, run, base=base)
    out = evaluate_claims(claims, ev, _states(plan.evidence_id))[C10]

    assert (out.status, out.fidelity, out.verdict) == ("verified", "inferred", "within_tolerance")
    assert out.tolerance.abs == pytest.approx(0.26) and out.paper_value == 3.99
    assert out.our_value == 4.10 and out.abs_diff == pytest.approx(0.11)
    assert out.our_raw == {"raw:test_acc": pytest.approx(0.959)}
    assert out.best_run_evidence_id == run.evidence_id and out.gpu_hours == 1.0
    texts = [(r.kind, r.text) for r in out.inferred_reasons]
    assert texts == [
        ("env_deviation", "torch 0.4.0 -> 1.12.1 (no CUDA build for sm_89)"),
        ("plan_deviation", "lr_schedule x5 -> x10 (hard-coded)"),
        ("plan_deviation", "momentum: repo uses 0.8"),
        ("known_fix_deviation", "batch {'batch_size': 64}: batch 128 -> 64"),
        ("critic_patch", "patch aaaaaaaaaaaa on train.py: fix a"),  # the reverted one is gone
        ("open_unknown", "eval_checkpoint not stated by the paper; assumed last (which=last)"),
        ("open_unknown", "seed not stated by the paper; assumed 0 (repo default --seed 0)"),
        ("env_assumption", "python_version = 3.7 (not stated)"),
        ("env_assumption", "cuda_version = 11.3 (not stated)"),
        ("protocol", "single run vs paper mean of 5"),
    ]
    assert out.inferred_reasons[0].evidence_ids == (env.evidence_id,)
    assert [r.text for r in out.info] == [
        "tqdm unpinned (no version stated; pip resolves at build time)"
    ]
    # C100 has no run: open, no reasons
    c100 = evaluate_claims(claims, ev, _states(plan.evidence_id))[C100]
    assert (c100.status, c100.verdict, c100.our_value, c100.inferred_reasons) == (
        "open", "no_completed_run", None, ()
    )  # fmt: skip


def test_outside_tolerance_is_open(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    plan = _run_plan(C100)
    run = _run(C100, 22.80, links=(plan.evidence_id,))
    out = evaluate_claims(claims, _store(_env_plan(), plan, run, base=base),
                          _states(plan.evidence_id))[C100]  # fmt: skip
    assert (out.status, out.fidelity, out.verdict) == ("open", None, "outside_tolerance")
    assert out.within_tolerance is False and out.abs_diff == pytest.approx(0.84)
    assert out.run_fidelity == "inferred" and out.inferred_reasons  # reasons still shown


def test_contradicted_only_per_existing_rules(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    resolved = _states("x", open_=False)
    no_num_runs = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    runs = [_run(C10, 5.0, run_id=f"r{i}", faithful=True) for i in range(3)]
    ev = _store(*runs, base=no_num_runs)
    assert evaluate_claims(claims, ev, resolved)[C10].status == "contradicted"
    # an open unknown blocks contradiction
    assert evaluate_claims(claims, ev, _states("x"))[C10].status == "open"
    # fewer than max_tries runs: still open
    two = _store(*runs[:2], base=no_num_runs)
    assert evaluate_claims(claims, two, resolved)[C10].status == "open"
    # paper averages 5 runs: 3 faithful runs are not protocol-faithful -> open, not contradicted
    assert evaluate_claims(claims, _store(*runs, base=base), resolved)[C10].status == "open"
    five = [_run(C10, 5.0, run_id=f"r{i}", faithful=True) for i in range(5)]
    assert evaluate_claims(claims, _store(*five, base=base), resolved)[C10].status == (
        "contradicted"
    )


def test_missing_tolerance_evidence_is_an_error(
    cutout_evidence: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    with pytest.raises(ValueError, match="no pre-registered tolerance evidence"):
        evaluate_claims(claims, cutout_evidence, {})


def test_ledger_prefers_tolerance_evidence_over_fallback(
    base: dict[str, Evidence],
    cutout_evidence: dict[str, Evidence],
    claims: tuple[ResultClaim, ...],
) -> None:
    run = _run(C10, 4.10)  # |diff| 0.11: inside 0.26 (evidence), outside 0.05 (fallback)
    with_tol = compute_ledger(claims, _store(run, base=base), {}, FALLBACK)[C10]
    assert with_tol.status == "verified" and with_tol.tolerance_evidence_id is not None
    assert with_tol.tolerance == Tolerance(abs=0.26)
    without = compute_ledger(claims, _store(run, base=cutout_evidence), {}, FALLBACK)[C10]
    assert without.status == "open" and without.tolerance_evidence_id is None
    assert without.tolerance == Tolerance(abs=0.05)


# --- protocol (num_runs) ---------------------------------------------------------------------


def test_num_runs_reason(base: dict[str, Evidence], claims: tuple[ResultClaim, ...]) -> None:
    assert paper_num_runs(base) is not None and paper_num_runs(base)[0] == 5  # type: ignore[index]
    one = evaluate_claims(claims, _store(_run(C10, 4.0), base=base), {})[C10]
    assert [r.text for r in one.inferred_reasons if r.kind == "protocol"] == [
        "single run vs paper mean of 5"
    ]
    two = _store(_run(C10, 4.0), _run(C10, 4.2, run_id="r2"), base=base)
    got = evaluate_claims(claims, two, {})[C10]
    assert got.completed_runs == 2
    assert [r.text for r in got.inferred_reasons if r.kind == "protocol"] == [
        "2 runs vs paper mean of 5"
    ]
    five = _store(*(_run(C10, 4.0, run_id=f"r{i}") for i in range(5)), base=base)
    assert not [r for r in evaluate_claims(claims, five, {})[C10].inferred_reasons
                if r.kind == "protocol"]  # fmt: skip


def test_protocol_alone_makes_fidelity_inferred(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    one = evaluate_claims(claims, _store(_run(C10, 4.0, faithful=True), base=base), {})[C10]
    assert (one.status, one.fidelity, one.run_fidelity) == ("verified", "inferred", "inferred")
    assert [r.kind for r in one.inferred_reasons] == ["protocol"]
    # the stored run evidence still says spec_faithful: the protocol rule is applied at read time
    assert _run(C10, 4.0, faithful=True).content["spec_faithful"] is True
    five = _store(*(_run(C10, 4.0, run_id=f"r{i}", faithful=True) for i in range(5)), base=base)
    got = evaluate_claims(claims, five, {})[C10]
    assert (got.fidelity, got.run_fidelity, got.inferred_reasons) == (
        "spec_faithful", "spec_faithful", ()
    )  # fmt: skip
    # no num_runs in the paper: one faithful run is spec_faithful
    no_num_runs = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    solo = _store(_run(C10, 4.0, faithful=True), base=no_num_runs)
    assert evaluate_claims(claims, solo, {})[C10].fidelity == "spec_faithful"


def test_ledger_protocol_rule(base: dict[str, Evidence], claims: tuple[ResultClaim, ...]) -> None:
    entry = compute_ledger(claims, _store(_run(C10, 4.0, faithful=True), base=base), {}, FALLBACK)
    assert (entry[C10].status, entry[C10].fidelity) == ("verified", "inferred")


def test_unpinned_packages_are_info_only(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    env = _env_plan(deviations=[])
    no_num_runs = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    run = _run(C10, 4.0, faithful=True)
    out = evaluate_claims(claims, _store(env, run, base=no_num_runs), {})[C10]
    assert out.fidelity == "spec_faithful" and out.inferred_reasons == ()
    assert [(r.kind, r.text) for r in out.info] == [
        ("unpinned_packages", "tqdm unpinned (no version stated; pip resolves at build time)")
    ]
    assert out.info[0].evidence_ids == (env.evidence_id,)


def test_unrecorded_cause_is_reported_not_hidden(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    no_protocol = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    out = evaluate_claims(claims, _store(_run(C10, 4.0), base=no_protocol), {})[C10]
    assert out.fidelity == "inferred"
    assert [r.kind for r in out.inferred_reasons] == ["unexplained"]


# --- run -> plan resolution, aborted runs ----------------------------------------------------


def test_plan_found_via_preflight_for_runs_without_plan_link(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    plan = _run_plan(C10)
    pf = make_evidence(kind="preflight", source=REPO, locator=f"preflight:{C10}",
                       content={"run_digest": "D" * 64, "plan_evidence_id": plan.evidence_id},
                       confidence="observed", authority="execution")  # fmt: skip
    run = _run(C10, 4.1)  # legacy: links only the assumption evidence
    out = evaluate_claims(claims, _store(plan, pf, run, base=base),
                          _states(plan.evidence_id))[C10]  # fmt: skip
    assert [r.text for r in out.inferred_reasons if r.kind == "open_unknown"] == [
        "eval_checkpoint not stated by the paper; assumed last (which=last)",
        "seed not stated by the paper; assumed 0 (repo default --seed 0)",
    ]


def test_aborted_run_counts_hours_and_attempts_but_never_completes(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    aborted = make_aborted_run_evidence(
        claim_id=C10, run_id="r-ab", log_ref="runs/r-ab/run.log", config_digest="D" * 64,
        duration_s=1800.0, reason="KeyboardInterrupt", spec_faithful=False,
        assumption_evidence_ids=("a" * 64,), source=REPO, links=("p" * 64, "q" * 64),
    )  # fmt: skip
    c = aborted.content
    assert (c["status"], c["exit_code"], c["metrics"], c["gpu_hours"]) == ("aborted", None, {}, 0.5)
    assert aborted.links == ("a" * 64, "p" * 64, "q" * 64)
    out = evaluate_claims(claims, _store(aborted, base=base), {})[C10]
    assert (out.status, out.attempts, out.completed_runs, out.aborted_runs) == ("open", 1, 0, 1)
    assert out.gpu_hours == 0.5 and out.verdict == "no_completed_run"


def test_run_evidence_links_are_additive() -> None:
    old = _run(C10, 4.0)
    new = _run(C10, 4.0, links=("p" * 64, "a" * 64))
    assert old.links == ("a" * 64,)
    assert new.links == ("a" * 64, "p" * 64)  # assumptions first, deduped
    assert new.evidence_id == old.evidence_id  # links are not part of the id


# --- unknowns answered after the run -----------------------------------------------------------


def _answered(plan_id: str, human_id: str, seed: str, ckpt: str) -> dict[str, UnknownState]:
    def st(name: str, assumed: str, value: str) -> UnknownState:
        return UnknownState(name=name, claim_ids=(), status="resolved", value=value,
                            resolved_by=human_id, assumptions=(Assumption(
                                unknown_id=f"unk:{name}", value=assumed,
                                evidence_id=plan_id),))  # fmt: skip

    return {
        "unk:seed": st("seed", "0", seed),
        "unk:eval_checkpoint": st("eval_checkpoint", "last", ckpt),
    }


def test_answer_after_run_differing_is_a_deviation_equal_is_confirmed(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    human = make_evidence(kind="human_input", source="human:cli", locator="answer:seed",
                          content={"unknown": "seed", "value": "1", "note": ""},
                          confidence="observed", authority="human")  # fmt: skip
    no_protocol = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    plan = _run_plan(C10)
    run = _run(C10, 4.0, links=(plan.evidence_id,), assumptions=(plan.evidence_id,))
    ev = _store(human, plan, run, base=no_protocol)

    # seed: ran with 0, human answered 1 -> deviation; eval_checkpoint last = last -> confirmed
    out = evaluate_claims(claims, ev, _answered(plan.evidence_id, human.evidence_id, "1", "last"))
    assert out[C10].fidelity == "inferred"
    assert [(r.kind, r.text) for r in out[C10].inferred_reasons] == [
        ("plan_deviation", "ran with seed=0, human answered seed=1")
    ]
    assert out[C10].inferred_reasons[0].evidence_ids == (plan.evidence_id, human.evidence_id)

    # both answers equal the assumed values: confirmed, no assumption reason, spec-faithful
    same = evaluate_claims(claims, ev, _answered(plan.evidence_id, human.evidence_id, "0.0",
                                                 "last"))[C10]  # fmt: skip
    assert (same.status, same.fidelity, same.run_fidelity) == (
        "verified", "spec_faithful", "spec_faithful"
    )  # fmt: skip
    assert same.inferred_reasons == ()
    # the stored run evidence is untouched (append-only): confirmation is applied at read time
    assert run.content["assumptions_used"] is True


def test_confirmed_assumptions_do_not_hide_other_causes(
    base: dict[str, Evidence], claims: tuple[ResultClaim, ...]
) -> None:
    human = make_evidence(kind="human_input", source="human:cli", locator="answer:seed",
                          content={"unknown": "seed", "value": "0", "note": ""},
                          confidence="observed", authority="human")  # fmt: skip
    no_protocol = {k: v for k, v in base.items() if v.content.get("name") != "num_runs"}
    plan = _run_plan(C10, deviation_note=True)
    run = _run(C10, 4.0, links=(plan.evidence_id,), assumptions=(plan.evidence_id,))
    ev = _store(human, plan, run, base=no_protocol)
    out = evaluate_claims(claims, ev, _answered(plan.evidence_id, human.evidence_id, "0",
                                                "last"))[C10]  # fmt: skip
    assert out.fidelity == "inferred"
    assert {r.kind for r in out.inferred_reasons} == {"plan_deviation"}
    assert not any("ran with" in r.text for r in out.inferred_reasons)
