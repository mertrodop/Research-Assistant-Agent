"""Reflect node (M4-S2): trigger, gated lesson, lesson evidence, idempotent reuse (FakeLLM)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from reproducer.cli import main
from reproducer.domain.breaker import breaker_state
from reproducer.domain.contracts import ArgOption, ExperimentResult, ResultClaim
from reproducer.domain.evaluate import ClaimEvaluation
from reproducer.domain.evidence import make_evidence
from reproducer.domain.ledger import Tolerance, make_run_evidence
from reproducer.domain.run_plan import parse_metric_line
from reproducer.domain.unknowns import UnknownOpened
from reproducer.graph.nodes.diagnose import ClaimDiagnosis, backfill_curves, diagnose
from reproducer.graph.nodes.reflect import (
    SYSTEM,
    all_lessons,
    current_lesson,
    lessons_for,
    reflect_claim,
    stale_lessons,
)
from reproducer.providers.llm import FakeLLM, LLMError
from reproducer.providers.store import RunStore

PATTERN = parse_metric_line("tqdm.write('test_acc: %.3f' % (test_acc))").pattern
CONV = "accuracy_fraction_to_error_percent"
SPEC = (
    ArgOption(option_strings=("--dataset",), dest="dataset", default="cifar10",
              choices=("cifar10", "cifar100")),
    ArgOption(option_strings=("--epochs",), dest="epochs", type="int", default=200),
    ArgOption(option_strings=("--seed",), dest="seed", type="int", default=0),
)  # fmt: skip
EPOCHS_QUOTE = "we train for 200 epochs"


def _claim(value: float) -> ResultClaim:
    return ResultClaim(claim_id="c10", metric="test_error", value=value, unit="%",
                       dataset="CIFAR-10", split="test", setting="ResNet18", location="T1",
                       quote="ResNet18 4.0")  # fmt: skip


def _setup(store: RunStore) -> str:
    """paper setup evidence (epochs stated, seed unknown) + the unk:seed open event; returns
    the run_plan evidence id."""
    epochs = make_evidence(kind="paper_setup", source="paper", locator="S3",
                           content={"name": "epochs", "value": "200", "quote": EPOCHS_QUOTE},
                           confidence="observed", authority="paper")  # fmt: skip
    seed = make_evidence(kind="paper_setup", source="paper", locator="-",
                         content={"name": "seed", "value": None, "quote": ""},
                         confidence="observed", authority="paper")  # fmt: skip
    body = {
        "claim_id": "c10", "entrypoint": "train.py", "metric": "test_error",
        "args": [{"flag": "--dataset", "value": "cifar10", "source": "claim",
                  "evidence_id": "e" * 64}],
        "depends_on_unknowns": ["unk:seed"],
        "metric_source": {"metric_line_evidence_id": "m" * 64, "pattern": PATTERN,
                          "which": "last", "conversion": CONV},
    }  # fmt: skip
    plan = make_evidence(
        kind="run_plan", source="s", locator="run_plan:c10",
        content={"claim_id": "c10", "plan": body, "command": "python train.py --dataset cifar10",
                 "assumptions": [{"unknown_id": "unk:seed", "value": "0"}]},
        confidence="derived", authority="rule", links=("p" * 64,),
    )  # fmt: skip
    for ev in (epochs, seed, plan):
        store.append_evidence(ev)
    store.append_event(UnknownOpened(unknown_id="unk:seed", name="seed", claim_ids=(),
                                     evidence_id=seed.evidence_id))  # fmt: skip
    return plan.evidence_id


def _add_run(out: Path, store: RunStore, plan_id: str, n: int, acc: float) -> None:
    run_dir = out / "runs" / f"run-{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text(
        "".join(f"test_acc: {a:.3f}\n" for a in (0.5, 0.8, acc)) if math.isfinite(acc) else ""
    )
    err = (1.0 - acc) * 100.0 if math.isfinite(acc) else float("nan")
    res = ExperimentResult(
        claim_id="c10", run_id=f"run-{n}", exit_code=0, metrics={"test_error": err},
        log_ref=f"runs/run-{n}/run.log", config_digest="d" * 64, duration=60.0, gpu_hours=0.01,
    )  # fmt: skip
    store.append_evidence(make_run_evidence(res, spec_faithful=False, assumption_evidence_ids=(),
                                            source="s", links=(plan_id,)))  # fmt: skip
    backfill_curves(out, store)


def _diagnose(out: Path, store: RunStore, paper: float = 4.0) -> tuple[ClaimDiagnosis, Any]:
    ev = ClaimEvaluation(
        claim_id="c10", metric="test_error", unit="%", paper_value=paper,
        tolerance=Tolerance(abs=0.5), tolerance_evidence_id="t" * 64, tolerance_reason="",
        best_run_evidence_id=None, our_value=None, our_raw={}, abs_diff=None,
        within_tolerance=None, verdict="no_completed_run", status="open", fidelity=None,
        run_fidelity=None, attempts=1, completed_runs=1, aborted_runs=0, gpu_hours=0.01,
        inferred_reasons=(),
    )  # fmt: skip
    (d,), breaker = diagnose(out, store, [_claim(paper)], {"c10": ev}, gpu_h_cap=10.0,
                             token_budget=100_000)  # fmt: skip
    return d, breaker


def _lesson_json(trial: int = 1, step: str = "eval_protocol", change: Any = "which") -> str:
    if change == "which":
        change = {
            "kind": "eval_protocol",
            "target": "metric_source.which",
            "new_value": "best",
            "support": None,
            "rationale": "the curve's best epoch is well below the final",
        }
    return json.dumps({"claim_id": "c10", "trial": trial, "step": step,
                       "evidence_ids": ["RUN", "CURVE"],
                       "text": "Final error is 1 point above the paper; the curve plateaued.",
                       "proposed_change": change})  # fmt: skip


@pytest.fixture
def oot(tmp_path: Path) -> tuple[Path, RunStore, str]:
    """A store whose newest run is out_of_tolerance (5.0 % vs paper 4.0 ± 0.5)."""
    store = RunStore(tmp_path)
    plan_id = _setup(store)
    _add_run(tmp_path, store, plan_id, 1, 0.95)
    return tmp_path, store, plan_id


def test_out_of_tolerance_writes_one_lesson_and_reuses_it(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    assert d.runs[-1].outcome.cls == "out_of_tolerance"
    llm = FakeLLM([_lesson_json()])
    r = reflect_claim(store, d, breaker, llm=llm, argparse_spec=SPEC)
    assert r.status == "lesson" and r.stored and not r.reused and r.attempts == 1
    assert r.lesson is not None and r.lesson.proposed_change is not None
    (ev,) = lessons_for(store.load_evidence(), "c10")
    assert ev.kind == "lesson" and ev.authority == "llm" and ev.confidence == "derived"
    assert ev.evidence_id == r.evidence.evidence_id  # type: ignore[union-attr]
    run_id = d.runs[-1].run.evidence_id
    assert run_id in ev.links and d.runs[-1].curve_evidence_id in ev.links
    assert ev.content["trial"] == 1 and ev.content["run_evidence_id"] == run_id

    # the prompt: compact observations, never raw logs
    system, prompt = llm.prompts[0]
    assert system == SYSTEM
    assert "test_acc: 0.950" not in prompt and "PARTIAL" not in prompt
    for part in ("out_of_tolerance", "plateau", "unk:seed", "--dataset cifar10  [claim]",
                 EPOCHS_QUOTE, "metric_source.which: one of ['best']"):  # fmt: skip
        assert part in prompt

    # same inputs -> reuse, no LLM call (FakeLLM has no responses left)
    before = (out / "evidence.jsonl").read_text()
    again = reflect_claim(store, d, breaker, llm=FakeLLM([]), argparse_spec=SPEC)
    assert again.reused and again.stale_ids == () and again.lesson == r.lesson
    assert (out / "evidence.jsonl").read_text() == before


def _change_inputs(store: RunStore) -> None:
    """New paper evidence: a new citable text, so the prompt and input digest change."""
    store.append_evidence(make_evidence(
        kind="paper_setup", source="paper", locator="S4",
        content={"name": "batch_size", "value": "128", "quote": "batches of 128 images"},
        confidence="observed", authority="paper",
    ))  # fmt: skip


def test_changed_inputs_write_a_fresh_lesson_and_stale_is_never_applied(
    oot: tuple[Path, RunStore, str],
) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    old = reflect_claim(store, d, breaker, llm=FakeLLM([_lesson_json()]), argparse_spec=SPEC)
    assert old.stored and old.evidence is not None
    _change_inputs(store)
    ev = store.load_evidence()

    # dry run with changed inputs: a fresh lesson, nothing stored, the old one reads stale
    dry = reflect_claim(store, d, breaker, llm=FakeLLM([_lesson_json(change=None)]),
                        dry_run=True, argparse_spec=SPEC)  # fmt: skip
    assert not dry.reused and not dry.stored and dry.input_digest != old.input_digest
    assert dry.stale_ids == (old.evidence.evidence_id,)
    assert current_lesson(ev, "c10", 1, dry.input_digest) is None  # stale never applied
    assert len(all_lessons(store.load_evidence(), "c10")) == 1

    fresh = reflect_claim(store, d, breaker, llm=FakeLLM([_lesson_json(change=None)]),
                          argparse_spec=SPEC)  # fmt: skip
    assert fresh.stored and not fresh.reused and fresh.stale_ids == (old.evidence.evidence_id,)
    ev = store.load_evidence()
    assert [e.evidence_id for e in all_lessons(ev, "c10")] == [
        old.evidence.evidence_id,
        fresh.evidence.evidence_id,  # type: ignore[union-attr]
    ]  # the stale one stays in the store
    cur = current_lesson(ev, "c10", 1, fresh.input_digest)
    assert cur is not None and cur.evidence_id == fresh.evidence.evidence_id  # type: ignore[union-attr]
    assert [e.evidence_id for e in stale_lessons(ev, "c10", 1, fresh.input_digest)] == [
        old.evidence.evidence_id
    ]
    assert current_lesson(ev, "c10", 1, old.input_digest) is not None  # only under old inputs

    # same changed inputs again -> the fresh lesson is reused, no LLM call
    again = reflect_claim(store, d, breaker, llm=FakeLLM([]), argparse_spec=SPEC)
    assert again.reused and again.evidence == fresh.evidence


def test_dry_run_stores_nothing(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    before = (out / "evidence.jsonl").read_text()
    r = reflect_claim(store, d, breaker, llm=FakeLLM([_lesson_json()]), dry_run=True,
                      argparse_spec=SPEC)  # fmt: skip
    assert r.status == "lesson" and not r.stored and r.evidence is not None
    assert (out / "evidence.jsonl").read_text() == before


def test_arg_change_grounded_in_paper_is_accepted(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    ch = {
        "kind": "arg",
        "target": "--epochs",
        "new_value": "200",
        "support": {"evidence_id": "P1", "quote": EPOCHS_QUOTE},
        "rationale": "paper epochs",
    }
    seed = {**ch, "target": "--seed", "new_value": "1"}  # no paper seed: rejected
    llm = FakeLLM([_lesson_json(step="config_vs_paper", change=seed),
                   _lesson_json(step="config_vs_paper", change=ch)])  # fmt: skip
    r = reflect_claim(store, d, breaker, llm=llm, argparse_spec=SPEC)
    assert r.status == "lesson" and r.attempts == 2 and "seed fishing" in r.errors[0]
    assert r.evidence is not None and r.evidence.content["gate_attempts"] == 2
    assert r.lesson is not None and r.lesson.proposed_change is not None
    assert r.lesson.proposed_change.support is not None


def test_gate_failure_stores_nothing(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    bad = _lesson_json(step="training")  # eval_protocol change at step training
    r = reflect_claim(store, d, breaker, llm=FakeLLM([bad] * 3), argparse_spec=SPEC)
    assert r.status == "failed" and r.attempts == 3 and len(r.errors) == 3
    assert lessons_for(store.load_evidence(), "c10") == []


def test_previous_lesson_is_memory_and_step_is_monotonic(
    oot: tuple[Path, RunStore, str],
) -> None:
    out, store, plan_id = oot
    d, breaker = _diagnose(out, store)
    first = FakeLLM([_lesson_json(step="config_vs_paper", change=None)])
    reflect_claim(store, d, breaker, llm=first, argparse_spec=SPEC)
    _add_run(out, store, plan_id, 2, 0.94)  # trial 2, still out of tolerance
    d, breaker = _diagnose(out, store)
    llm = FakeLLM([_lesson_json(trial=2), _lesson_json(trial=2, step="training", change=None)])
    r = reflect_claim(store, d, breaker, llm=llm, argparse_spec=SPEC)
    assert r.status == "lesson" and r.trial == 2 and r.attempts == 2
    assert "goes back" in r.errors[0]
    prompt = llm.prompts[0][1]
    assert "trial 1 (config_vs_paper): Final error is 1 point above" in prompt
    assert [ev.content["trial"] for ev in lessons_for(store.load_evidence(), "c10")] == [1, 2]


def test_diverged_triggers(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    plan_id = _setup(store)
    _add_run(tmp_path, store, plan_id, 1, float("nan"))
    d, breaker = _diagnose(tmp_path, store)
    assert d.runs[-1].outcome.cls == "diverged"
    js = json.loads(_lesson_json(change=None))
    js["evidence_ids"] = ["RUN"]  # no curve points
    r = reflect_claim(store, d, breaker, llm=FakeLLM([json.dumps(js)]), argparse_spec=SPEC)
    assert r.status == "lesson" and r.stored


def test_verified_skips_and_force_needs_dry_run(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store, paper=5.0)  # 5.0 vs 5.0: verified
    assert d.runs[-1].outcome.cls == "verified"
    r = reflect_claim(store, d, breaker, llm=FakeLLM([]))
    assert r.status == "skipped" and "verified" in r.reason
    with pytest.raises(ValueError, match="never stored"):
        reflect_claim(store, d, breaker, llm=FakeLLM([]), force=True)
    les = _lesson_json(step="training", change={"kind": "none", "target": "none",
                                                "new_value": None, "support": None,
                                                "rationale": "within tolerance"})  # fmt: skip
    f = reflect_claim(store, d, breaker, llm=FakeLLM([les]), force=True, dry_run=True,
                      argparse_spec=SPEC)  # fmt: skip
    assert f.status == "lesson" and f.forced and not f.stored
    assert lessons_for(store.load_evidence(), "c10") == []


def test_no_budget_skips(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, _ = _diagnose(out, store)
    spent = breaker_state(["c10"], store.load_evidence(), gpu_h_cap=10.0, token_budget=1000,
                          tokens_used=0, max_per_claim=1)  # fmt: skip
    r = reflect_claim(store, d, spent, llm=FakeLLM([]))
    assert r.status == "skipped" and "no trial budget" in r.reason
    tokens = breaker_state(["c10"], store.load_evidence(), gpu_h_cap=10.0, token_budget=10,
                           tokens_used=10)  # fmt: skip
    assert reflect_claim(store, d, tokens, llm=FakeLLM([])).status == "skipped"


def test_llm_error_propagates(oot: tuple[Path, RunStore, str]) -> None:
    out, store, _ = oot
    d, breaker = _diagnose(out, store)
    with pytest.raises(LLMError):
        reflect_claim(store, d, breaker, llm=FakeLLM([]), argparse_spec=SPEC)


def test_cli_force_requires_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["reflect", "--out", str(tmp_path), "--force"]) == 2
    assert "--force requires --dry-run" in capsys.readouterr().err


# --- live --------------------------------------------------------------------------------------

CUTOUT_E2E = Path(__file__).resolve().parents[2] / "runs" / "cutout_e2e"


@pytest.mark.llm_live
def test_live_reflect_dry_run_force_on_cutout(capsys: pytest.CaptureFixture[str]) -> None:
    """reflect --dry-run --force on the Cutout store (backend as configured): a validated
    lesson, printed with tokens / latency, and nothing written to the store."""
    if not (CUTOUT_E2E / "evidence.jsonl").is_file():
        pytest.skip(f"no store at {CUTOUT_E2E}")
    if not (os.environ.get("REPRODUCER_LLM_REFLECT") or os.environ.get("REPRODUCER_LLM")):
        pytest.skip("no LLM configured (REPRODUCER_LLM[_REFLECT])")
    files = ("evidence.jsonl", "llm_calls.jsonl", "unknown_events.jsonl")
    before = {f: (CUTOUT_E2E / f).read_bytes() for f in files if (CUTOUT_E2E / f).exists()}
    code = main(["reflect", "--out", str(CUTOUT_E2E), "--claim", "c10plus_resnet18_cutout",
                 "--dry-run", "--force"])  # fmt: skip
    printed = capsys.readouterr().out
    with capsys.disabled():
        print("\n" + printed)
    assert code == 0, printed
    assert "dry run, not stored" in printed and "tokens=" in printed
    after = {f: (CUTOUT_E2E / f).read_bytes() for f in files if (CUTOUT_E2E / f).exists()}
    assert after == before
