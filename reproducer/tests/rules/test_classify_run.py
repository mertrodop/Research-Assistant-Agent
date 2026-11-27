"""Tests for reproducer.rules.classify_run (one full run -> RunOutcome)."""

from __future__ import annotations

from typing import Any

import pytest

from reproducer.domain.contracts import ExperimentResult
from reproducer.domain.curve import summarize
from reproducer.domain.evaluate import ClaimEvaluation
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import Tolerance, make_aborted_run_evidence, make_run_evidence
from reproducer.domain.run_plan import parse_metric_line
from reproducer.rules.classify import ClassifyContext
from reproducer.rules.classify_run import classify_run, direction_vs_paper, loss_explosion

CID = "c10"
PATTERN = parse_metric_line("tqdm.write('test_acc: %.3f' % (test_acc))").pattern


def _eval(paper: float = 3.99, tol: float = 0.26) -> ClaimEvaluation:
    return ClaimEvaluation(
        claim_id=CID, metric="test_error", unit="%", paper_value=paper,
        tolerance=Tolerance(abs=tol), tolerance_evidence_id="t" * 64, tolerance_reason="2 x std",
        best_run_evidence_id=None, our_value=None, our_raw={}, abs_diff=None,
        within_tolerance=None, verdict="no_completed_run", status="open", fidelity=None,
        run_fidelity=None, attempts=0, completed_runs=0, aborted_runs=0, gpu_hours=0.0,
        inferred_reasons=(),
    )  # fmt: skip


def _run(exit_code: int = 0, metrics: dict[str, float] | None = None, **kw: Any) -> Evidence:
    res = ExperimentResult(
        claim_id=kw.get("claim_id", CID), run_id="run-1", exit_code=exit_code,
        metrics={} if metrics is None else metrics, log_ref="runs/run-1/run.log",
        config_digest="d" * 64, duration=10.0, gpu_hours=0.01,
    )  # fmt: skip
    return make_run_evidence(res, spec_faithful=True, assumption_evidence_ids=(), source="s")


def test_verified_within_tolerance_with_direction() -> None:
    o = classify_run(_eval(), _run(metrics={"test_error": 4.10}), "")
    assert o.cls == "verified" and o.direction == "worse"
    assert o.diff == pytest.approx(0.11) and o.tol == 0.26


def test_out_of_tolerance_worse() -> None:
    o = classify_run(_eval(), _run(metrics={"test_error": 5.0}), "")
    assert o.cls == "out_of_tolerance" and o.direction == "worse"
    assert "worse than the paper" in o.reason


def test_out_of_tolerance_better_than_paper() -> None:
    o = classify_run(_eval(), _run(metrics={"test_error": 3.0}), "")
    assert o.cls == "out_of_tolerance" and o.direction == "better"
    assert o.diff == pytest.approx(-0.99)


def test_direction_for_max_metric_and_unknown() -> None:
    assert direction_vs_paper("test_acc", 1.0) == "better"
    assert direction_vs_paper("test_acc", -1.0) == "worse"
    assert direction_vs_paper("test_error", 0.0) is None
    assert direction_vs_paper("bleu_like_thing", 1.0) is None


def test_diverged_nonfinite_metric() -> None:
    run = _run(metrics={"test_error": float("nan"), "raw:test_acc": float("nan")})
    assert run.content["diverged"] is True
    o = classify_run(_eval(), run, "test_acc: nan\n")
    assert o.cls == "diverged" and o.value is None and o.direction is None


def test_diverged_nonfinite_curve() -> None:
    curve = summarize([(0, 0.5), (1, float("nan"))], metric="test_error", column="test_acc",
                      conversion="accuracy_fraction_to_error_percent", direction="min",
                      source="csv", source_ref="x.csv")  # fmt: skip
    o = classify_run(_eval(), _run(exit_code=0, metrics={}), "", curve=curve)
    assert o.cls == "diverged" and "curve" in o.reason


@pytest.mark.parametrize(
    "tail",
    ["Epoch 3: loss=nan acc=0.1", "xentropy: inf", "Loss is NaN, stopping", "loss: 3.2e7"],
)
def test_diverged_loss_explosion(tail: str) -> None:
    assert loss_explosion(tail) is not None
    o = classify_run(_eval(), _run(exit_code=1), tail)
    assert o.cls == "diverged"


def test_no_loss_explosion_on_normal_loss() -> None:
    assert loss_explosion("Epoch 3: loss=0.42 xentropy=0.028") is None


def test_crashed_after_progress_goes_through_classify() -> None:
    tail = (
        "test_acc: 0.500\n"
        "Traceback (most recent call last):\n"
        '  File "train.py", line 10, in <module>\n'
        "    x = y[3]\n"
        "IndexError: list index out of range\n"
    )
    o = classify_run(_eval(), _run(exit_code=1), tail, metric_pattern=PATTERN,
                     ctx=ClassifyContext(repo_files=frozenset({"train.py"})))  # fmt: skip
    assert o.cls == "crashed" and o.classification is not None
    assert o.classification.cls == "critic" and "IndexError" in o.reason


def test_crashed_oom_is_known_fix_like_smoke() -> None:
    from reproducer.domain.smoke_plan import BudgetArg

    tail = (
        "test_acc: 0.500\nTraceback (most recent call last):\n"
        '  File "train.py", line 5, in <module>\n    loss.backward()\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n"
    )
    batch = BudgetArg(option_strings=("--batch_size",), dest="batch_size", default=128)
    ctx = ClassifyContext(repo_files=frozenset({"train.py"}), arguments=(batch,))
    o = classify_run(_eval(), _run(exit_code=1), tail, metric_pattern=PATTERN, ctx=ctx,
                     command="python train.py --batch_size 128")  # fmt: skip
    assert o.cls == "crashed" and o.classification is not None
    assert o.classification.cls == "known" and o.classification.pattern == "oom"


def test_nonzero_exit_without_progress_is_not_completed() -> None:
    o = classify_run(_eval(), _run(exit_code=2), "usage: train.py ...\n", metric_pattern=PATTERN)
    assert o.cls == "not_completed" and "before any progress" in o.reason


def test_not_completed_cases() -> None:
    assert classify_run(_eval(), _run(exit_code=0), "").cls == "not_completed"  # no metric
    assert classify_run(_eval(), _run(exit_code=-1), "test_acc: 0.5").cls == "not_completed"
    aborted = make_aborted_run_evidence(
        claim_id=CID, run_id="run-2", log_ref="x", config_digest="d" * 64, duration_s=60,
        reason="KeyboardInterrupt", spec_faithful=True, assumption_evidence_ids=(), source="s",
    )  # fmt: skip
    o = classify_run(_eval(), aborted, "")
    assert o.cls == "not_completed" and "KeyboardInterrupt" in o.reason


def test_rejects_other_kinds_and_claims() -> None:
    other = make_evidence(kind="workflow", source="s", locator="l", content={},
                          confidence="derived", authority="rule")  # fmt: skip
    with pytest.raises(ValueError):
        classify_run(_eval(), other, "")
    with pytest.raises(ValueError):
        classify_run(_eval(), _run(claim_id="c100", metrics={"test_error": 4.0}), "")
