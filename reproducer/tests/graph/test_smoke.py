"""Tests for reproducer.graph.nodes.smoke.

Unit tests use a fake runner (no Docker). Tests marked `docker` run tiny fake repos through
the real smoke path (run_container, python:3.10-slim test image, CPU, our trivial probe):
    .venv/bin/pytest -q -m docker -rs
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from reproducer.domain.smoke_plan import BudgetArg
from reproducer.graph.nodes.smoke import PROBE_OK, SmokeInputs, run_smoke
from reproducer.providers.docker_runner import RunResult, build_image

EPOCHS = (BudgetArg(option_strings=("--epochs",), dest="epochs"),)
OK_PROBE = ("python", "-c", f"print({PROBE_OK!r})")


def _repo(tmp_path: Path, train: str, helper: str = "def boom():\n    return 1\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(train)
    (repo / "helper.py").write_text(helper)
    return repo


def _inputs(tmp_path: Path, repo: Path, **kw: object) -> SmokeInputs:
    base: dict[str, object] = {
        "image": "img:tag",
        "clone_dir": repo,
        "example": "python train.py --lr 0.1",
        "arguments": EPOCHS,
        "third_party": (),
        "local_modules": ("helper",),
        "run_root": tmp_path / "smoke",
        "gpu": False,
        "timeout_s": 30.0,
        "probe_cmd": OK_PROBE,
        "mem": "512m",
    }
    base.update(kw)
    return SmokeInputs(**base)  # type: ignore[arg-type]


# --- unit (fake runner) ------------------------------------------------------------------


class FakeRunner:
    def __init__(self, outcomes: dict[str, tuple[int | None, str, bool]]) -> None:
        self.outcomes = outcomes  # stage -> (exit, log, timed_out)
        self.calls: list[tuple[str, list[str], list[str]]] = []

    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        stage = name.rsplit("-", 1)[1]
        self.calls.append((stage, cmd, sorted(p.name for p in Path(workdir).iterdir())))
        code, log, timed_out = self.outcomes[stage]
        Path(log_path).write_text(log)
        return RunResult(
            exit_code=code, timed_out=timed_out, oom_killed=False, duration_s=3.0,
            log_path=Path(log_path), name=name,
        )  # fmt: skip


def test_infra_failure_stops_before_repo_code(tmp_path: Path) -> None:
    runner = FakeRunner({"probe": (1, "RuntimeError: CUDA driver initialization failed\n", False)})
    res = run_smoke(_inputs(tmp_path, _repo(tmp_path, "x = 1\n")), runner=runner)
    assert (res.signal, res.stage, res.passed) == ("infra", "probe", False)
    assert [c[0] for c in runner.calls] == ["probe"]
    assert runner.calls[0][2] == []  # probe workdir is empty: no repo code
    assert not any(p.name.endswith("-work") for p in res.run_dir.iterdir())  # workdirs removed


def test_import_error_stops_before_run(tmp_path: Path) -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "/work/helper.py", line 1, in <module>\n'
        "    import nope\n"
        "ModuleNotFoundError: No module named 'nope'\n"
    )
    runner = FakeRunner({"probe": (0, PROBE_OK, False), "imports": (1, log, False)})
    res = run_smoke(_inputs(tmp_path, _repo(tmp_path, "x = 1\n")), runner=runner)
    assert (res.signal, res.stage) == ("import_error", "imports")
    assert res.traceback is not None and res.traceback.exc_type == "ModuleNotFoundError"
    assert (
        res.traceback.repo_frame is not None and res.traceback.repo_frame.file == "/work/helper.py"
    )
    stage, cmd, files = runner.calls[1]
    assert cmd[:2] == ["python", "-c"] and cmd[-1] == '["helper"]'  # not the entrypoint
    assert "train.py" in files  # fresh copy of the clone


def test_run_command_and_details(tmp_path: Path) -> None:
    runner = FakeRunner(
        {
            "probe": (0, PROBE_OK, False),
            "imports": (0, "REPRODUCER_IMPORTS {}", False),
            "run": (0, "100%|##| 10/10 [00:07<00:00, 1.4it/s]\n", False),
        }
    )
    res = run_smoke(_inputs(tmp_path, _repo(tmp_path, "x = 1\n")), runner=runner)
    assert runner.calls[2][1] == ["python", "train.py", "--lr", "0.1", "--epochs", "1"]
    assert (res.signal, res.passed, res.timing.sec_per_epoch) == ("passed_exit0", True, 7.0)
    d = res.details(attempt=1, log_ref="smoke/x/run.log", image="img:tag")
    assert d["command"] == "python train.py --lr 0.1 --epochs 1"
    assert d["overrides"] == [
        {"dest": "epochs", "option": "--epochs", "value": "1", "previous": None}
    ]
    assert d["traceback"] is None and d["progress_seen"] is True and d["attempt"] == 1


# --- docker: the real smoke path -----------------------------------------------------------

TINY_DOCKERFILE = "FROM python:3.10-slim\nENV PYTHONDONTWRITEBYTECODE=1\n"
PROGRESS = (
    "import sys\n"
    "for i in range(1, 6):\n"
    "    sys.stderr.write(f'\\r{i*20}%|##| {i}/5 [00:0{i}<00:0{5-i}, 1.00it/s]')\n"
    "    sys.stderr.flush()\n"
)
ARGS = "import argparse\np = argparse.ArgumentParser(); p.add_argument('--epochs', type=int)\n"


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    tag = "reproducer/test-tiny:latest"
    log = tmp_path_factory.mktemp("build") / "build.log"
    assert build_image(TINY_DOCKERFILE, tag, timeout_s=600, log_path=log).ok
    return tag


@pytest.mark.docker
def test_docker_smoke_progress_then_exit0(tiny_image: str, tmp_path: Path) -> None:
    train = ARGS + "a, _ = p.parse_known_args(); print('EPOCHS', a.epochs, flush=True)\n" + PROGRESS
    res = run_smoke(_inputs(tmp_path, _repo(tmp_path, train), image=tiny_image))
    log = res.log_path.read_text()
    assert (res.signal, res.passed, res.exit_code) == ("passed_exit0", True, 0), log
    assert "EPOCHS 1" in log  # the budget override reached the script
    assert res.progress_seen and res.timing.method == "tqdm" and res.timing.sec_per_epoch == 5.0
    assert not (res.run_dir / "run-work").exists() and res.log_path.exists()


@pytest.mark.docker
def test_docker_smoke_traceback_with_repo_frame(tiny_image: str, tmp_path: Path) -> None:
    helper = "def boom():\n    raise ValueError('bad shape')\n"
    repo = _repo(tmp_path, "import helper\nprint('start', flush=True)\nhelper.boom()\n", helper)
    res = run_smoke(_inputs(tmp_path, repo, image=tiny_image))
    assert (res.signal, res.passed, res.exit_code) == ("traceback", False, 1)
    tb = res.traceback
    assert tb is not None and (tb.exc_type, tb.message) == ("ValueError", "bad shape")
    assert tb.repo_frame is not None and tb.repo_frame.file == "/work/helper.py"
    assert tb.repo_frame.func == "boom"


@pytest.mark.docker
def test_docker_smoke_sleep_without_progress_times_out(tiny_image: str, tmp_path: Path) -> None:
    repo = _repo(tmp_path, "import time\nprint('loading', flush=True)\ntime.sleep(120)\n")
    res = run_smoke(_inputs(tmp_path, repo, image=tiny_image, timeout_s=6.0))
    assert (res.signal, res.passed, res.timed_out) == ("timeout_no_progress", False, True)
    assert res.traceback is None and res.timing.sec_per_epoch is None


@pytest.mark.docker
def test_docker_smoke_progress_then_timeout_passes(tiny_image: str, tmp_path: Path) -> None:
    repo = _repo(tmp_path, PROGRESS + "import time\ntime.sleep(120)\n")
    res = run_smoke(_inputs(tmp_path, repo, image=tiny_image, timeout_s=6.0))
    assert (res.signal, res.passed, res.timed_out) == ("passed_progress", True, True)


@pytest.mark.docker
def test_docker_smoke_import_error_and_infra(tiny_image: str, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    repo = _repo(tmp_path / "a", "import helper\n", "import not_a_real_module\n")
    res = run_smoke(_inputs(tmp_path / "a", repo, image=tiny_image))
    assert (res.signal, res.stage) == ("import_error", "imports")
    assert res.traceback is not None and res.traceback.exc_type == "ModuleNotFoundError"
    assert (
        res.traceback.repo_frame is not None and res.traceback.repo_frame.file == "/work/helper.py"
    )
    assert "run" not in res.stage_logs  # the entrypoint never ran

    bad_probe = ("python", "-c", "import sys; sys.exit('no GPU')")
    res = run_smoke(_inputs(tmp_path / "a", repo, image=tiny_image, probe_cmd=bad_probe))
    assert (res.signal, res.stage) == ("infra", "probe")
    assert set(res.stage_logs) == {"probe"}


# --- auto-fix loop -----------------------------------------------------------------------


def test_loop_stops_on_critic_without_fixes(tmp_path: Path) -> None:
    from reproducer.graph.nodes.smoke import smoke_loop
    from reproducer.rules.classify import ClassifyContext

    log = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 3, in <module>\n'
        "    m.fc\n"
        "AttributeError: no fc\n"
    )
    runner = FakeRunner(
        {"probe": (0, PROBE_OK, False), "imports": (0, "", False), "run": (1, log, False)}
    )
    repo = _repo(tmp_path, "x = 1\n")
    recorded = []
    out = smoke_loop(
        lambda s: _inputs(tmp_path, repo),
        ClassifyContext(repo_files=frozenset({"train.py", "helper.py"})),
        record=recorded.append,
        runner=runner,
    )
    assert out.stop == "critic" and len(out.iterations) == 1 and recorded == list(out.iterations)
    assert out.iterations[0].capped is False and out.iterations[0].action is None


AUTOFIX_TRAIN = (
    ARGS
    + "p.parse_known_args()\n"
    + "open('logs/x.csv', 'w').write('epoch,acc\\n')\n"
    + PROGRESS
    + "open('ckpt/model.pt', 'wb').write(b'0')\n"
)


@pytest.mark.docker
def test_docker_auto_fix_creates_output_dirs_and_passes(tiny_image: str, tmp_path: Path) -> None:
    from dataclasses import replace

    from reproducer.graph.nodes.smoke import smoke_loop
    from reproducer.rules.classify import ClassifyContext

    repo = _repo(tmp_path, AUTOFIX_TRAIN)
    base = _inputs(tmp_path, repo, image=tiny_image)
    recorded = []
    out = smoke_loop(
        lambda s: replace(base, prep_dirs=s.prep_dirs, fix_overrides=s.overrides),
        ClassifyContext(repo_files=frozenset({"train.py", "helper.py"}), arguments=EPOCHS),
        record=recorded.append,
    )
    steps = [
        (it.result.signal, it.classification.pattern if it.classification else None,
         it.action.params.get("dir") if it.action else None)
        for it in out.iterations
    ]  # fmt: skip
    assert steps == [
        ("traceback", "missing_output_dir", "logs"),
        ("traceback", "missing_output_dir", "ckpt"),
        ("passed_exit0", None, None),
    ], [it.result.log_path.read_text() for it in out.iterations]
    assert out.stop == "passed" and out.state.prep_dirs == ("logs", "ckpt")
    assert len({it.input_digest for it in out.iterations}) == 3  # each fix = new inputs
    assert all(not it.capped for it in out.iterations) and recorded == list(out.iterations)
    assert not (repo / "logs").exists()  # only the per-run copies got the dirs
    assert out.iterations[-1].result.timing.sec_per_epoch == 5.0
