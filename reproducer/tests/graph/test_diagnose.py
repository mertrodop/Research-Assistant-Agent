"""Tests for reproducer.graph.nodes.diagnose (run_curve evidence, backfill, diagnosis) and the
run_curve hook in execute_job."""

from __future__ import annotations

from pathlib import Path

import pytest

from reproducer.domain.contracts import ExperimentResult, ResultClaim
from reproducer.domain.evaluate import ClaimEvaluation
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import Tolerance, make_aborted_run_evidence, make_run_evidence
from reproducer.domain.run_plan import parse_metric_line
from reproducer.graph.nodes.diagnose import (
    CurveSpec,
    backfill_curves,
    compute_curve,
    curve_spec_for,
    diagnose,
    partial_run,
    record_run_curve,
    run_curve_evidence,
    wants_curve,
)
from reproducer.graph.nodes.execute import MetricSpec, RunJob, execute_job
from reproducer.providers.docker_runner import RunResult
from reproducer.providers.store import RunStore

PATTERN = parse_metric_line("tqdm.write('test_acc: %.3f' % (test_acc))").pattern
CONV = "accuracy_fraction_to_error_percent"
SPEC = CurveSpec(claim_metric="test_error", pattern=PATTERN, conversion=CONV, raw_name="test_acc")
CSV = 'dataset,cifar10\n""\nepoch,train_acc,test_acc\n0,0.5,0.60\n1,0.8,0.90\n2,0.9,0.95\n'
LOG = "test_acc: 0.600\ntest_acc: 0.900\ntest_acc: 0.950\n"


def _plan() -> Evidence:
    body = {"claim_id": "c10", "metric": "test_error",
            "metric_source": {"metric_line_evidence_id": "m" * 64, "pattern": PATTERN,
                              "which": "last", "conversion": CONV}}  # fmt: skip
    return make_evidence(kind="run_plan", source="s", locator="run_plan:c10",
                         content={"claim_id": "c10", "plan": body, "command": "python train.py"},
                         confidence="derived", authority="rule", links=("p" * 64,))  # fmt: skip


def _store_with_run(
    out: Path, *, csv: str | None = CSV, refs: bool = False, log: str = LOG, exit_code: int = 0
) -> tuple[RunStore, Evidence]:
    store = RunStore(out)
    plan = _plan()
    store.append_evidence(plan)
    run_dir = out / "runs" / "run-c10-1"
    (run_dir / "artifacts" / "logs").mkdir(parents=True)
    (run_dir / "run.log").write_text(log)
    if csv is not None:
        (run_dir / "artifacts" / "logs" / "c10.csv").write_text(csv)
    res = ExperimentResult(
        claim_id="c10", run_id="run-c10-1", exit_code=exit_code,
        metrics={"test_error": 5.0, "raw:test_acc": 0.95} if exit_code == 0 else {},
        log_ref="runs/run-c10-1/run.log",
        config_digest="d" * 64, duration=60.0, gpu_hours=0.0167,
    )  # fmt: skip
    run = make_run_evidence(
        res, spec_faithful=True, assumption_evidence_ids=(), source="s",
        links=(plan.evidence_id,),
        artifacts=["runs/run-c10-1/artifacts/logs/c10.csv"] if refs and csv else None,
    )  # fmt: skip
    store.append_evidence(run)
    return store, run


@pytest.mark.parametrize("refs", [True, False])  # artifacts refs vs older run dir scan
def test_curve_from_csv_artifact(tmp_path: Path, refs: bool) -> None:
    store, run = _store_with_run(tmp_path, refs=refs)
    c = compute_curve(tmp_path, run, SPEC)
    assert c is not None and c.source == "csv" and c.n_points == 3
    assert c.source_ref == "runs/run-c10-1/artifacts/logs/c10.csv"
    assert c.best == pytest.approx(5.0) and c.best_epoch == 2


def test_curve_falls_back_to_log_lines(tmp_path: Path) -> None:
    _, run = _store_with_run(tmp_path, csv=None)
    c = compute_curve(tmp_path, run, SPEC)
    assert c is not None and c.source == "log" and c.n_points == 3
    _, run2 = _store_with_run(tmp_path / "b", csv="epoch,loss\n0,1.0\n")  # no metric column
    c2 = compute_curve(tmp_path / "b", run2, SPEC)
    assert c2 is not None and c2.source == "log"
    _, run3 = _store_with_run(tmp_path / "c", csv=None, log="nothing\n")
    assert compute_curve(tmp_path / "c", run3, SPEC) is None


def test_curve_spec_from_the_runs_plan(tmp_path: Path) -> None:
    store, run = _store_with_run(tmp_path)
    assert curve_spec_for(run, store.load_evidence()) == SPEC


def test_run_curve_evidence_kind_rules(tmp_path: Path) -> None:
    store, run = _store_with_run(tmp_path)
    c = compute_curve(tmp_path, run, SPEC)
    assert c is not None
    ev = run_curve_evidence(run, c)
    assert ev.kind == "run_curve" and ev.authority == "rule" and ev.confidence == "derived"
    assert ev.links == (run.evidence_id,)
    with pytest.raises(ValueError):
        make_evidence(kind="run_curve", source="s", locator="x",
                      content={"run_evidence_id": run.evidence_id}, confidence="observed",
                      authority="rule", links=(run.evidence_id,))  # fmt: skip
    with pytest.raises(ValueError):
        make_evidence(kind="run_curve", source="s", locator="x",
                      content={"run_evidence_id": run.evidence_id}, confidence="derived",
                      authority="rule", links=())  # fmt: skip


def test_backfill_is_append_only_and_idempotent(tmp_path: Path) -> None:
    store, run = _store_with_run(tmp_path)
    before = (tmp_path / "evidence.jsonl").read_text()
    new = backfill_curves(tmp_path, store)
    assert len(new) == 1 and new[0].content["run_evidence_id"] == run.evidence_id
    after = (tmp_path / "evidence.jsonl").read_text()
    assert after.startswith(before)  # nothing edited, only appended
    assert backfill_curves(tmp_path, store) == []
    assert (tmp_path / "evidence.jsonl").read_text() == after
    again = record_run_curve(tmp_path, store, run, SPEC, store.load_evidence())
    assert again is not None and again.evidence_id == new[0].evidence_id


def test_diagnose_reports_outcome_curve_and_budget(tmp_path: Path) -> None:
    store, run = _store_with_run(tmp_path)
    backfill_curves(tmp_path, store)
    claim = ResultClaim(claim_id="c10", metric="test_error", value=4.0, unit="%",
                        dataset="CIFAR-10", split="test", setting="s", location="T1",
                        quote="4.0")  # fmt: skip
    ev = ClaimEvaluation(
        claim_id="c10", metric="test_error", unit="%", paper_value=4.0,
        tolerance=Tolerance(abs=0.5), tolerance_evidence_id="t" * 64, tolerance_reason="",
        best_run_evidence_id=run.evidence_id, our_value=5.0, our_raw={}, abs_diff=1.0,
        within_tolerance=False, verdict="outside_tolerance", status="open", fidelity=None,
        run_fidelity="spec_faithful", attempts=1, completed_runs=1, aborted_runs=0,
        gpu_hours=0.0167, inferred_reasons=(),
    )  # fmt: skip
    diags, breaker = diagnose(tmp_path, store, [claim], {"c10": ev}, gpu_h_cap=10.0,
                              token_budget=1000)  # fmt: skip
    (d,) = diags
    (r,) = d.runs
    assert r.outcome.cls == "out_of_tolerance" and r.outcome.direction == "worse"
    assert r.curve is not None and r.curve.source == "csv" and r.curve_evidence_id
    assert breaker.per_claim["c10"].remaining == 2 and breaker.tripped == ()


class _Runner:
    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        (Path(workdir) / "logs" / "c10.csv").write_text(CSV)
        Path(log_path).write_text(LOG)
        return RunResult(exit_code=0, timed_out=False, oom_killed=False, duration_s=36.0,
                         log_path=Path(log_path), name=name)  # fmt: skip


def test_execute_job_records_run_curve(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('hi')\n")
    job = RunJob(
        claim_id="c10", source="s", image="img:tag", clone_dir=repo, argv=("python", "train.py"),
        patches=(), prep_dirs=("logs",), data_mounts=(),
        metric=MetricSpec(claim_metric="test_error", pattern=PATTERN, which="last",
                          conversion=CONV, raw_name="test_acc"),
        run_digest="D" * 64, timeout_s=60.0, planned_epochs=3, spec_faithful=True,
        assumption_evidence_ids=(), gpu=False,
    )  # fmt: skip
    out = tmp_path / "out"
    store = RunStore(out)
    rep = execute_job(job, out, store, runner=_Runner(), progress_every_s=0)
    curves = [e for e in store.load_evidence().values() if e.kind == "run_curve"]
    assert len(curves) == 1 and curves[0].links == (rep.evidence.evidence_id,)
    assert curves[0].content["summary"]["source"] == "csv"
    assert backfill_curves(out, store) == []  # already recorded


# --- partial curves of crashed / diverged runs --------------------------------------------------


def test_crashed_run_gets_a_partial_curve(tmp_path: Path) -> None:
    log = LOG + "Traceback (most recent call last):\nRuntimeError: CUDA error\n"
    store, run = _store_with_run(tmp_path, csv=None, log=log, exit_code=1)
    assert partial_run(run) and wants_curve(run)
    (ev,) = backfill_curves(tmp_path, store)
    assert ev.content["partial"] is True and ev.content["summary"]["n_points"] == 3
    assert ev.links == (run.evidence_id,)
    assert backfill_curves(tmp_path, store) == []  # idempotent


def test_completed_run_curve_has_no_partial_flag(tmp_path: Path) -> None:
    store, run = _store_with_run(tmp_path)
    assert not partial_run(run)
    (ev,) = backfill_curves(tmp_path, store)
    assert "partial" not in ev.content


def test_diverged_run_curve_is_partial(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_evidence(_plan())
    run_dir = tmp_path / "runs" / "run-c10-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("test_acc: 0.500\ntest_acc: 0.400\n")
    res = ExperimentResult(
        claim_id="c10", run_id="run-c10-1", exit_code=0, metrics={"test_error": float("nan")},
        log_ref="runs/run-c10-1/run.log", config_digest="d" * 64, duration=1.0, gpu_hours=0.0,
    )  # fmt: skip
    run = make_run_evidence(res, spec_faithful=True, assumption_evidence_ids=(), source="s",
                            links=(_plan().evidence_id,))  # fmt: skip
    store.append_evidence(run)
    assert run.content["diverged"] and partial_run(run)
    (ev,) = backfill_curves(tmp_path, store)
    assert ev.content["partial"] is True


@pytest.mark.parametrize("exit_code", [-1, 1])
def test_no_curve_without_points_or_exit_code(tmp_path: Path, exit_code: int) -> None:
    log = "nothing\n" if exit_code == 1 else LOG  # crashed before any point / timed out
    store, run = _store_with_run(tmp_path, csv=None, log=log, exit_code=exit_code)
    assert backfill_curves(tmp_path, store) == []
    assert partial_run(run) is (exit_code == 1)


def test_aborted_run_gets_no_curve(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_evidence(_plan())
    run_dir = tmp_path / "runs" / "run-c10-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text(LOG)
    run = make_aborted_run_evidence(
        claim_id="c10", run_id="run-c10-1", log_ref="runs/run-c10-1/run.log",
        config_digest="d" * 64, duration_s=5.0, reason="interrupted", spec_faithful=True,
        assumption_evidence_ids=(), source="s", links=(_plan().evidence_id,),
    )  # fmt: skip
    store.append_evidence(run)
    assert not partial_run(run) and not wants_curve(run)
    assert backfill_curves(tmp_path, store) == []


class _CrashRunner:
    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        Path(log_path).write_text(LOG + "RuntimeError: boom\n")
        return RunResult(exit_code=1, timed_out=False, oom_killed=False, duration_s=36.0,
                         log_path=Path(log_path), name=name)  # fmt: skip


def test_execute_job_records_partial_curve_for_a_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('hi')\n")
    job = RunJob(
        claim_id="c10", source="s", image="img:tag", clone_dir=repo, argv=("python", "train.py"),
        patches=(), prep_dirs=("logs",), data_mounts=(),
        metric=MetricSpec(claim_metric="test_error", pattern=PATTERN, which="last",
                          conversion=CONV, raw_name="test_acc"),
        run_digest="D" * 64, timeout_s=60.0, planned_epochs=3, spec_faithful=True,
        assumption_evidence_ids=(), gpu=False,
    )  # fmt: skip
    out = tmp_path / "out"
    store = RunStore(out)
    rep = execute_job(job, out, store, runner=_CrashRunner(), progress_every_s=0)
    curves = [e for e in store.load_evidence().values() if e.kind == "run_curve"]
    assert len(curves) == 1 and curves[0].links == (rep.evidence.evidence_id,)
    assert curves[0].content["partial"] is True
    assert curves[0].content["summary"]["source"] == "log"
