"""Tests for reproducer.graph.nodes.execute (pre-flight estimate / caps, run digest
idempotency, metric -> ExperimentResult, spec_faithful, artifacts).

Unit tests use a fake runner. The `docker` test runs a fake 3-epoch repo through the real
sandbox (python:3.10-slim test image, CPU):
    .venv/bin/pytest -q -m docker -rs
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from reproducer.cli import print_claim_run
from reproducer.domain.contracts import ArgOption, ExperimentResult
from reproducer.domain.evidence import make_evidence
from reproducer.domain.ledger import make_run_evidence
from reproducer.domain.run_plan import parse_metric_line
from reproducer.domain.smoke_plan import Override
from reproducer.graph.nodes.execute import (
    Caps,
    ClaimRun,
    MetricSpec,
    RunJob,
    RunSetupError,
    caps_from_env,
    completed_run,
    copy_artifacts,
    decide,
    estimate_gpu_hours,
    eval_overhead_s,
    execute_job,
    experiment_result,
    full_argv,
    is_spec_faithful,
    planned_epochs,
    progress_line,
    raw_metric_name,
    run_digest,
    run_timeout_s,
    spent_gpu_hours,
)
from reproducer.providers.docker_runner import RunResult, build_image, run_container
from reproducer.providers.store import RunStore, resolve_ref
from reproducer.rules.known_fixes import FixState

METRIC_LINE = "tqdm.write('test_acc: %.3f' % (test_acc))"
PATTERN = parse_metric_line(METRIC_LINE).pattern
MS = MetricSpec(
    claim_metric="test_error",
    pattern=PATTERN,
    which="last",
    conversion="accuracy_fraction_to_error_percent",
    raw_name="test_acc",
)
SPEC = (
    ArgOption(option_strings=("--epochs",), dest="epochs", type="int", default=200),
    ArgOption(option_strings=("--dataset", "-d"), dest="dataset", default="cifar10"),
)
LOG3 = "Epoch 0\ntest_acc: 0.500\nEpoch 1\ntest_acc: 0.600\nEpoch 2\ntest_acc: 0.700\n"


# --- estimate + caps -------------------------------------------------------------------------


def test_planned_epochs_from_arg_default_or_none() -> None:
    assert planned_epochs([("--dataset", "cifar100")], SPEC) == 200  # argparse default
    assert planned_epochs([("--epochs", "30")], SPEC) == 30  # the plan's argument
    assert planned_epochs([], SPEC[1:]) is None  # no epoch argument
    bad = (ArgOption(option_strings=("--epochs",), dest="epochs", type="int", default=None),)
    assert planned_epochs([], bad) is None


def test_estimate_math_cutout_numbers() -> None:
    # Cutout smoke: tqdm 28 s/epoch, 33.3 s wall for the 1-epoch run, 200 epochs
    overhead = eval_overhead_s(
        sec_per_epoch=28.0, smoke_duration_s=33.3, timing_method="tqdm", epochs=200
    )
    assert overhead == pytest.approx(5.3 * 200)
    est = estimate_gpu_hours(28.0, 200, overhead)
    assert est == pytest.approx((28.0 * 200 + 5.3 * 200) / 3600)
    assert 1.7 <= est <= 2.0
    # a wall-clock timing already includes evaluation and start-up
    assert eval_overhead_s(sec_per_epoch=33.0, smoke_duration_s=33.0, timing_method="wall",
                           epochs=200) == 0.0  # fmt: skip
    assert run_timeout_s(est, 4.0) == pytest.approx(1.5 * est * 3600 + 600)
    assert run_timeout_s(3.5, 4.0) == 4.0 * 3600  # capped by the per-run cap


def _run_ev(gpu_hours: float, exit_code: int = 0, metrics: dict[str, float] | None = None,
            digest: str = "d", run_id: str = "r1"):  # type: ignore[no-untyped-def]  # fmt: skip
    res = ExperimentResult(
        claim_id="c", run_id=run_id, exit_code=exit_code,
        metrics={"test_error": 3.9} if metrics is None else metrics, log_ref="runs/x/run.log",
        config_digest=digest, duration=gpu_hours * 3600, gpu_hours=gpu_hours,
    )  # fmt: skip
    return make_run_evidence(res, spec_faithful=False, assumption_evidence_ids=(), source="s")


def test_cap_decisions() -> None:
    caps = caps_from_env({}, 0.0)
    assert (caps.run_h, caps.paper_h) == (4.0, 10.0)  # defaults
    assert decide(True, 1.85, caps)[0] == "run"
    d, reason = decide(True, 4.5, caps)
    assert d == "skip_cap" and "per-run cap" in reason
    assert decide(False, 1.85, caps)[0] == "skip_smoke_failed"
    assert decide(True, None, caps)[0] == "skip_cap"  # no estimate: fail closed

    # paper cap minus the GPU hours already spent (from run evidence, completed or not)
    spent = spent_gpu_hours([_run_ev(3.0), _run_ev(3.5, exit_code=1, run_id="r2"),
                             _run_ev(1.0, run_id="r3")])  # fmt: skip
    assert spent == pytest.approx(7.5)
    caps = caps_from_env({"REPRODUCER_PAPER_GPU_HOURS": "10"}, spent)
    assert caps.paper_left_h == pytest.approx(2.5)
    assert decide(True, 1.85, caps)[0] == "run"
    d, reason = decide(True, 2.6, caps)
    assert d == "skip_cap" and "paper budget left" in reason
    # env overrides
    caps = caps_from_env({"REPRODUCER_RUN_GPU_HOURS": "1.5"}, 0.0)
    assert decide(True, 1.85, caps)[0] == "skip_cap"
    with pytest.raises(ValueError, match="REPRODUCER_RUN_GPU_HOURS"):
        caps_from_env({"REPRODUCER_RUN_GPU_HOURS": "lots"}, 0.0)
    assert Caps(4, 10, 12).paper_left_h == 0.0


# --- run digest / idempotency ----------------------------------------------------------------


def test_run_digest_covers_fix_state_patches_and_command() -> None:
    cmd = "python train.py --dataset cifar10 --batch_size 128"
    s0 = FixState(prep_dirs=("logs",))
    halved = FixState(
        prep_dirs=("logs",),
        overrides=(
            Override(dest="batch_size", option="--batch_size", value="64", previous="128"),
        ),  # fmt: skip
    )
    assert full_argv(cmd, halved)[-2:] == ("--batch_size", "64")
    assert full_argv(cmd, s0) == tuple(cmd.split())  # no budget override
    base = run_digest(image="img:1", state=s0, patches=(), argv=full_argv(cmd, s0))
    assert base == run_digest(image="img:1", state=s0, patches=(), argv=full_argv(cmd, s0))
    assert base != run_digest(image="img:2", state=s0, patches=(), argv=full_argv(cmd, s0))
    assert base != run_digest(image="img:1", state=s0, patches=("diff",), argv=full_argv(cmd, s0))
    assert base != run_digest(image="img:1", state=halved, patches=(),
                              argv=full_argv(cmd, halved))  # fmt: skip


def test_completed_run_skips_but_crashed_reruns() -> None:
    done = _run_ev(1.8, digest="D")
    crashed = _run_ev(0.4, exit_code=137, metrics={"raw:test_acc": 0.5}, digest="D", run_id="r2")
    no_metric = _run_ev(1.8, metrics={"raw:test_acc": 0.5}, digest="D", run_id="r3")
    assert completed_run([done], "D", "test_error") == done  # completed -> skip
    assert completed_run([crashed, no_metric], "D", "test_error") is None  # -> re-run
    assert completed_run([done], "other", "test_error") is None  # other inputs
    nan = _run_ev(1.8, metrics={"test_error": float("nan")}, digest="D", run_id="r4")
    assert completed_run([nan], "D", "test_error") is None  # diverged: not completed


# --- metric -> ExperimentResult --------------------------------------------------------------


def _result(log: str, exit_code: int | None = 0, timed_out: bool = False):  # type: ignore[no-untyped-def]
    return experiment_result(
        claim_id="c10", run_id="run-1", exit_code=exit_code, timed_out=timed_out, log_text=log,
        ms=MS, log_ref="runs/run-1/run.log", config_digest="D", duration_s=7200.0,
    )  # fmt: skip


def test_metric_last_value_converted() -> None:
    res, reading = _result(LOG3)
    assert res.metrics["raw:test_acc"] == 0.7
    assert res.metrics["test_error"] == pytest.approx(30.0)  # (1 - 0.7) * 100, the LAST line
    assert (res.exit_code, res.gpu_hours, res.config_digest) == (0, 2.0, "D")
    assert reading.note == "ok"
    ev = make_run_evidence(res, spec_faithful=False, assumption_evidence_ids=(), source="s")
    assert completed_run([ev], "D", "test_error") == ev


def test_best_uses_the_printed_metric_direction() -> None:
    best = MetricSpec(**{**MS.__dict__, "which": "best"})
    res, _ = experiment_result(
        claim_id="c", run_id="r", exit_code=0, timed_out=False,
        log_text="test_acc: 0.8\ntest_acc: 0.7\n", ms=best, log_ref="l", config_digest="D",
        duration_s=1.0,
    )  # fmt: skip
    assert res.metrics["raw:test_acc"] == 0.8  # max accuracy, not min


@pytest.mark.parametrize(
    "log, exit_code, timed_out, note",
    [
        ("Epoch 0\nloss 2.3\n", 0, False, "no metric line"),  # missing metric
        (LOG3, 1, False, "did not finish"),  # crashed after printing
        (LOG3, 137, True, "did not finish"),  # killed at the timeout
        (LOG3, None, False, "did not finish"),  # no exit code reported
        ("test_acc: 70.5\n", 0, False, "conversion failed"),  # not a fraction
    ],
)
def test_not_completed_runs_have_no_claim_metric(
    log: str, exit_code: int | None, timed_out: bool, note: str
) -> None:
    res, reading = _result(log, exit_code, timed_out)
    assert "test_error" not in res.metrics and note in reading.note
    ev = make_run_evidence(res, spec_faithful=False, assumption_evidence_ids=(), source="s")
    assert completed_run([ev], "D", "test_error") is None


def test_raw_metric_name_from_the_metric_line() -> None:
    assert raw_metric_name(["print('loss', x)", METRIC_LINE], PATTERN) == "test_acc"
    assert raw_metric_name([], PATTERN) == "test_acc"  # fallback: the pattern's literal


def test_progress_line() -> None:
    tail = "Epoch 0: 100%|##| 391/391 [00:28<00:00]\ntest_acc: 0.4\r\rEpoch 1:  12%| 47/391 ["
    assert progress_line(tail, PATTERN, 200) == "epoch 1/200 done, current 'Epoch 1', bar 47/391"
    assert progress_line("loading...\n", PATTERN, 200) is None


# --- spec_faithful ---------------------------------------------------------------------------

CLEAN = {
    "plan_faithful": True,
    "env_deviations": 0,
    "plan_deviations": 0,
    "known_fix_deviations": 0,
    "critic_patches": 0,
    "assumption_evidence_ids": (),
}


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"plan_faithful": False}, False),  # plan depends on open unknowns / assumed flags
        ({"env_deviations": 1}, False),  # e.g. torch 0.4.0 -> 1.12.1
        ({"plan_deviations": 1}, False),  # inconsistent setup check
        ({"known_fix_deviations": 1}, False),  # OOM batch halving without grad accumulation
        ({"critic_patches": 1}, False),  # a critic patch on the stack
        ({"assumption_evidence_ids": ("e",)}, False),  # env assumption on an open unknown
    ],
)
def test_spec_faithful_truth_table(change: dict[str, object], expected: bool) -> None:
    assert is_spec_faithful(**{**CLEAN, **change}) is expected  # type: ignore[arg-type]


def test_preflight_evidence_kind_is_execution_observed() -> None:
    ok = make_evidence(kind="preflight", source="s", locator="preflight:c", content={"d": 1},
                       confidence="observed", authority="execution")  # fmt: skip
    assert ok.kind == "preflight"
    with pytest.raises(ValueError, match="preflight evidence requires"):
        make_evidence(kind="preflight", source="s", locator="preflight:c", content={"d": 1},
                      confidence="derived", authority="rule")  # fmt: skip


# --- artifacts + execute_job (fake runner) ---------------------------------------------------


def _repo(tmp_path: Path, train: str = "print('hi')\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(train)
    return repo


def _job(repo: Path, **kw: object) -> RunJob:
    base: dict[str, object] = {
        "claim_id": "c10",
        "source": "repo:x@0",
        "image": "img:tag",
        "clone_dir": repo,
        "argv": ("python", "train.py", "--epochs", "3"),
        "patches": (),
        "prep_dirs": ("logs", "checkpoints"),
        "data_mounts": (),
        "metric": MS,
        "run_digest": "D" * 64,
        "timeout_s": 120.0,
        "planned_epochs": 3,
        "spec_faithful": False,
        "assumption_evidence_ids": ("a" * 64,),
        "gpu": False,
        "mem": "512m",
    }
    base.update(kw)
    return RunJob(**base)  # type: ignore[arg-type]


class FakeRunner:
    """Writes outputs into the per-run copy like train.py would, then a log."""

    def __init__(self, exit_code: int = 0, log: str = LOG3) -> None:
        self.exit_code, self.log = exit_code, log
        self.seen: list[str] = []

    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        work = Path(workdir)
        self.seen = sorted(p.name for p in work.iterdir())
        (work / "logs" / "c10.csv").write_text("epoch,test_acc\n2,0.7\n")
        (work / "checkpoints" / "c10.pt").write_bytes(b"weights")
        Path(log_path).write_text(self.log)
        return RunResult(exit_code=self.exit_code, timed_out=False, oom_killed=False,
                         duration_s=36.0, log_path=Path(log_path), name=name)  # fmt: skip


def test_copy_artifacts_skips_checkpoints_and_symlink_dirs(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "logs").mkdir(parents=True)
    (work / "logs" / "a.csv").write_text("x")
    (work / "checkpoints").mkdir()
    (work / "checkpoints" / "m.pt").write_text("w")
    (work / "checkpoints" / "m.pth").write_text("w")
    (work / "checkpoints" / "notes.txt").write_text("n")
    (work / "evil").symlink_to("/etc")
    got = copy_artifacts(work, tmp_path / "art", ["logs", "checkpoints", "evil", "missing"],
                         no_checkpoints=True)  # fmt: skip
    assert got == ["checkpoints/notes.txt", "logs/a.csv"]
    assert not (tmp_path / "art" / "evil").exists()


@pytest.mark.parametrize("no_checkpoints", [False, True])
def test_execute_job_keeps_artifacts_before_deleting_workdir(
    tmp_path: Path, no_checkpoints: bool
) -> None:
    store = RunStore(tmp_path / "out")
    runner = FakeRunner()
    rep = execute_job(_job(_repo(tmp_path)), tmp_path / "out", store, runner=runner,
                      no_checkpoints=no_checkpoints, progress_every_s=0)  # fmt: skip
    assert {"train.py", "logs", "checkpoints"} <= set(runner.seen)  # prep dirs in the copy
    assert not (rep.run_dir / "work").exists()  # working copy deleted
    art = rep.run_dir / "artifacts"
    assert (art / "logs" / "c10.csv").read_text().endswith("0.7\n")
    assert (art / "checkpoints" / "c10.pt").exists() is (not no_checkpoints)
    assert rep.completed and rep.result.metrics["test_error"] == pytest.approx(30.0)
    ev = store.load_evidence()[rep.evidence.evidence_id]
    assert (ev.kind, ev.authority, ev.confidence) == ("run_result", "execution", "observed")
    assert ev.content["config_digest"] == "D" * 64
    assert ev.content["spec_faithful"] is False and ev.content["assumptions_used"] is True
    assert ev.links == ("a" * 64,)
    assert ev.content["log_ref"].startswith("runs/run-c10-")
    run = rep.run_dir.name
    expected = [f"runs/{run}/artifacts/logs/c10.csv"]
    if not no_checkpoints:
        expected.insert(0, f"runs/{run}/artifacts/checkpoints/c10.pt")
    assert ev.content["artifacts"] == expected  # store-relative, sorted
    assert all(resolve_ref(tmp_path / "out", a).is_file() for a in ev.content["artifacts"])


def test_execute_job_log_ref_is_out_dir_relative_with_relative_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path("runs/paper")  # relative, as the CLI passes it
    store = RunStore(out)
    rep = execute_job(_job(_repo(tmp_path)), out, store, runner=FakeRunner(log="test_acc: 0.959\n"),
                      progress_every_s=0)  # fmt: skip
    ref = rep.result.log_ref
    assert ref.startswith("runs/run-c10-") and ref.endswith("/run.log")  # not runs/paper/...
    assert resolve_ref(out, ref).read_text() == "test_acc: 0.959\n"
    # printout: claim precision (stored value keeps full precision)
    assert rep.result.metrics["test_error"] == 100 * (1 - 0.959) != 4.1
    run = ClaimRun("c10", "python train.py", "img:tag", "D" * 64, "ran", report=rep, decimals=2)
    print_claim_run(run, out)
    printed = capsys.readouterr().out
    assert "raw raw:test_acc=0.959 -> test_error=4.10  (ok)" in printed
    assert "4.100000000000003" not in printed
    assert f"log:          {out / ref}" in printed


def test_print_already_completed_uses_claim_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ev = _run_ev(1.0, metrics={"test_error": 100 * (1 - 0.959), "raw:test_acc": 0.959})
    run = ClaimRun("c10", "python train.py", "img:tag", "d", "already_completed", existing=ev,
                   decimals=2)  # fmt: skip
    print_claim_run(run, tmp_path)
    printed = capsys.readouterr().out
    assert "metrics raw:test_acc=0.959, test_error=4.10); skipped" in printed
    assert ev.content["metrics"]["test_error"] == 100 * (1 - 0.959)  # stored value unchanged


def test_execute_job_crash_still_keeps_artifacts_and_is_not_completed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "out")
    rep = execute_job(_job(_repo(tmp_path)), tmp_path / "out", store,
                      runner=FakeRunner(exit_code=1, log="test_acc: 0.5\nTraceback ...\n"),
                      progress_every_s=0)  # fmt: skip
    assert not rep.completed and "test_error" not in rep.result.metrics
    assert rep.result.metrics == {"raw:test_acc": 0.5}
    assert (rep.run_dir / "artifacts" / "logs" / "c10.csv").exists()
    assert completed_run(store.load_evidence().values(), "D" * 64, "test_error") is None


def test_execute_job_links_plan_and_preflight(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "out")
    job = _job(_repo(tmp_path), links=("p" * 64, "q" * 64))
    rep = execute_job(job, tmp_path / "out", store, runner=FakeRunner(), progress_every_s=0)
    stored = store.load_evidence()[rep.evidence.evidence_id]
    assert stored.links == ("a" * 64, "p" * 64, "q" * 64)  # assumptions, run_plan, preflight


class RaisingRunner:
    """Starts the 'container' (writes a partial log), then is interrupted."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        Path(log_path).write_text("test_acc: 0.4\n")
        raise self.exc


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("docker died")])
def test_execute_job_abort_records_aborted_run_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    import types

    import reproducer.graph.nodes.execute as ex

    ticks = iter((100.0, 1900.0))  # start, abort: 1800 s = 0.5 GPU h
    monkeypatch.setattr(ex, "time", types.SimpleNamespace(monotonic=lambda: next(ticks)))
    out = tmp_path / "out"
    store = RunStore(out)
    job = _job(_repo(tmp_path), links=("p" * 64, "q" * 64))
    seen: list[str] = []
    with pytest.raises(type(exc)):
        execute_job(job, out, store, runner=RaisingRunner(exc), progress=seen.append,
                    progress_every_s=0)  # fmt: skip
    runs = [e for e in store.load_evidence().values() if e.kind == "run_result"]
    assert len(runs) == 1
    c = runs[0].content
    assert (c["status"], c["exit_code"], c["metrics"]) == ("aborted", None, {})
    assert (c["duration"], c["gpu_hours"], c["config_digest"]) == (1800.0, 0.5, "D" * 64)
    assert type(exc).__name__ in c["abort_reason"]
    assert runs[0].links == ("a" * 64, "p" * 64, "q" * 64)
    assert spent_gpu_hours(store.load_evidence().values()) == 0.5  # counts toward the cap
    assert completed_run(store.load_evidence().values(), "D" * 64, "test_error") is None
    assert not list((out / "runs").glob("*/work"))  # working copy still deleted
    assert seen and "aborted" in seen[-1]


@pytest.mark.parametrize(
    ("stage", "kw"),
    [
        ("prepare_workdir", {"clone_dir": Path("/nonexistent/clone")}),
        ("image_check", {}),
    ],
)
def test_execute_job_setup_failure_is_a_note_not_a_run(
    tmp_path: Path, stage: str, kw: dict[str, object]
) -> None:
    out = tmp_path / "out"
    store = RunStore(out)
    job = _job(_repo(tmp_path), links=("p" * 64, "q" * 64), **kw)
    runner = FakeRunner()
    with pytest.raises((FileNotFoundError, RunSetupError)):
        execute_job(job, out, store, runner=runner, progress_every_s=0,
                    image_exists=lambda image: False)  # fmt: skip
    assert runner.seen == []  # the container never started
    evidence = store.load_evidence().values()
    assert not [e for e in evidence if e.kind == "run_result"]  # no attempt
    notes = [e for e in evidence if e.kind == "setup_failed"]
    assert len(notes) == 1 and notes[0].content["stage"] == stage
    assert notes[0].links == ("p" * 64, "q" * 64)
    assert spent_gpu_hours(evidence) == 0.0


def test_run_claims_links_new_run_result_to_plan_and_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types
    from dataclasses import dataclass, field

    import reproducer.graph.nodes.execute as ex
    from reproducer.domain.contracts import ResultClaim
    from reproducer.graph.nodes.critic import CriticState

    out = tmp_path / "out"
    store = RunStore(out)
    repo = _repo(tmp_path)
    metric_ev = make_evidence(kind="repo_file", source="repo:x@0", locator="train.py#metric_lines",
                              content={"result": [{"line": METRIC_LINE}]}, confidence="derived",
                              authority="repo")  # fmt: skip
    plan_ev = make_evidence(kind="run_plan", source="repo:x@0", locator="run_plan:c10",
                            content={"claim_id": "c10"}, confidence="derived", authority="rule",
                            links=("m" * 64,))  # fmt: skip
    smoke_ev = make_evidence(kind="smoke_result", source="repo:x@0", locator="smoke:1",
                             content={"passed": True, "attempts": 1, "capped": False,
                                      "sec_per_epoch": 1.0, "timing": {"method": "wall"},
                                      "duration_s": 1.0}, confidence="observed",
                             authority="execution")  # fmt: skip
    for ev in (metric_ev, plan_ev, smoke_ev):
        store.append_evidence(ev)

    claim = ResultClaim(claim_id="c10", metric="test_error", value=4.0, unit="%",
                        dataset="CIFAR-10", split="test", setting=None, location="T1",
                        quote="4.0")  # fmt: skip
    plan = types.SimpleNamespace(
        entrypoint="train.py", args=(), depends_on_unknowns=(),
        metric_source=types.SimpleNamespace(metric_line_evidence_id=metric_ev.evidence_id,
                                            pattern=PATTERN, which="last",
                                            conversion="accuracy_fraction_to_error_percent"),
        is_spec_faithful=lambda states: False,
    )  # fmt: skip
    planned = types.SimpleNamespace(claim=claim, plan=plan, evidence=plan_ev, notes=(),
                                    deviations=(), assumptions=(),
                                    command="python train.py --epochs 3")  # fmt: skip
    current = types.SimpleNamespace(target=types.SimpleNamespace(argparse_spec=SPEC),
                                    input_digest="I" * 64, plans=(planned,))  # fmt: skip

    @dataclass(frozen=True)
    class FakeSetup:
        example: str = ""
        source: str = "repo:x@0"
        repo_dir: Path = repo
        fix_state0: FixState = FixState(prep_dirs=("logs", "checkpoints"))
        critic_state0: CriticState = field(default_factory=CriticState)
        known_fixes0: int = 0
        cache: str = str(tmp_path / "cache")
        base_mounts: tuple[tuple[str, str], ...] = ()

    env_ev = make_evidence(kind="env_spec", source="repo:x@0", locator="env_plan",
                           content={"plan": {}}, confidence="derived", authority="rule",
                           links=("e" * 64,))  # fmt: skip
    monkeypatch.setattr(ex, "load_current_plans", lambda store, repo_dir, partial=False: current)
    monkeypatch.setattr(ex, "prepare_smoke", lambda out, store, entrypoint: FakeSetup())
    monkeypatch.setattr(ex, "make_build", lambda setup, out, store, **kw: (
        lambda state: types.SimpleNamespace(image="img:tag", arguments=())))  # fmt: skip
    monkeypatch.setattr(ex, "_stored_smoke", lambda evidence, source, digest: smoke_ev)
    monkeypatch.setattr(ex, "env_plan_to_evidence", lambda *a, **kw: env_ev)
    plan_env = types.SimpleNamespace(
        plan=types.SimpleNamespace(deviations=()), gpu=None,
        spec_evidence=types.SimpleNamespace(source="repo:x@0", evidence_id="s" * 64),
    )  # fmt: skip
    deps = {
        "runner": FakeRunner(), "plan_env": lambda out, cpu: plan_env,
        "image_exists": lambda image: True, "build_image": None, "unload": lambda: None,
        "make_critic": lambda setup: None, "env": {},
    }  # fmt: skip
    runs = ex.run_claims(out, store, deps)
    assert [r.status for r in runs] == ["ran"]
    rep, pf = runs[0].report, runs[0].preflight
    assert rep is not None and pf is not None and rep.completed
    stored = store.load_evidence()[rep.evidence.evidence_id]
    assert plan_ev.evidence_id in stored.links and pf.evidence.evidence_id in stored.links
    assert stored.links[-2:] == (plan_ev.evidence_id, pf.evidence.evidence_id)


# --- docker: a fake 3-epoch repo through the real sandbox ------------------------------------

TINY_DOCKERFILE = "FROM python:3.10-slim\nENV PYTHONDONTWRITEBYTECODE=1\n"
FAKE_TRAIN = """\
import argparse, os, sys
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=3)
a = p.parse_args()
for epoch, acc in zip(range(a.epochs), (0.5, 0.6, 0.7)):
    sys.stderr.write(f'\\rEpoch {epoch}: 100%|##| 10/10 [00:01<00:00, 9.0it/s]\\n')
    print('test_acc: %.3f' % acc, flush=True)
    with open('logs/fake.csv', 'a') as f:
        f.write(f'{epoch},{acc}\\n')
with open('checkpoints/fake.pt', 'wb') as f:
    f.write(b'weights')
"""


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    tag = "reproducer/test-tiny:latest"
    log = tmp_path_factory.mktemp("build") / "build.log"
    assert build_image(TINY_DOCKERFILE, tag, timeout_s=600, log_path=log).ok
    return tag


@pytest.mark.docker
def test_docker_fake_repo_run_evidence_and_artifacts(tiny_image: str, tmp_path: Path) -> None:
    out = tmp_path / "out"
    store = RunStore(out)
    job = _job(_repo(tmp_path, FAKE_TRAIN), image=tiny_image)
    rep = execute_job(job, out, store, runner=run_container, progress_every_s=0)
    assert rep.result.exit_code == 0, (rep.run_dir / "run.log").read_text()
    assert rep.completed
    ev = store.load_evidence()[rep.evidence.evidence_id]
    assert ev.content["metrics"]["test_error"] == pytest.approx(30.0)  # last of 0.5/0.6/0.7
    assert ev.content["metrics"]["raw:test_acc"] == 0.7
    art = rep.run_dir / "artifacts"
    assert (art / "logs" / "fake.csv").read_text() == "0,0.5\n1,0.6\n2,0.7\n"
    assert (art / "checkpoints" / "fake.pt").read_bytes() == b"weights"
    assert ev.content["artifacts"] == [
        f"runs/{rep.run_dir.name}/artifacts/checkpoints/fake.pt",
        f"runs/{rep.run_dir.name}/artifacts/logs/fake.csv",
    ]
    assert not (rep.run_dir / "work").exists()
