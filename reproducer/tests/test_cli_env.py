"""CLI helpers for env / gpu-check that need no Docker or GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from reproducer import cli
from reproducer.domain.env_rules import EnvAssumption, EnvPlan, env_plan_to_evidence
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    fold_unknowns,
    resolve,
    unknown_id_for,
)
from reproducer.providers.docker_runner import RunResult
from reproducer.providers.llm import OllamaLLM


def test_unload_ollama_only_ollama_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_unload(self: OllamaLLM) -> bool:
        calls.append(self.model)
        return True

    monkeypatch.setattr(OllamaLLM, "unload", fake_unload)
    env = {
        "REPRODUCER_LLM": "ollama:qwen2.5-coder:7b",
        "REPRODUCER_LLM_CRITIC": "openrouter:meta-llama/llama-3.1-8b-instruct:free",
        "REPRODUCER_LLM_SUBTASK_A": "ollama:qwen2.5-coder:7b",  # same spec -> once
        "OTHER": "ollama:ignored",
    }
    assert cli.unload_ollama(env) == ["ollama:qwen2.5-coder:7b"]
    assert calls == ["qwen2.5-coder:7b"]


def test_unload_ollama_is_best_effort(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def boom(self: OllamaLLM) -> bool:
        raise ConnectionError("ollama down")

    monkeypatch.setattr(OllamaLLM, "unload", boom)
    assert cli.unload_ollama({"REPRODUCER_LLM": "ollama:m"}) == []
    assert "ollama down" in capsys.readouterr().err


def test_parser_has_env_and_gpu_check() -> None:
    p = cli.build_parser()
    assert p.parse_args(["env", "--out", "runs/x"]).func is cli._cmd_env
    gc: argparse.Namespace = p.parse_args(["gpu-check", "--out", "runs/x"])
    assert gc.func is cli._cmd_gpu_check and gc.keep_workdir is False
    assert p.parse_args(["gpu-check", "--out", "x", "--keep-workdir"]).keep_workdir is True


# --- env: plan assumptions -> Assumption events ----------------------------------------------


def _plan(*assumptions: tuple[str, str | None]) -> EnvPlan:
    return EnvPlan(
        base_image="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
        python="3.7",
        torch="1.12.1",
        torchvision="0.13.1",
        cuda="11.3",
        assumptions=tuple(EnvAssumption(name=n, value=v, reason="r") for n, v in assumptions),
    )


def _setup_ev(name: str) -> Evidence:
    return make_evidence(
        kind="paper_setup",
        source="paper",
        locator="sec 4",
        content={"name": name, "value": None, "quote": "q"},
        confidence="observed",
        authority="paper",
    )


def _opened(name: str) -> tuple[Evidence, UnknownOpened]:
    ev = _setup_ev(name)
    return ev, UnknownOpened(unknown_id=unknown_id_for(name), name=name, evidence_id=ev.evidence_id)


PLAN = _plan(
    ("python_version", "3.7"),
    ("cuda_version", "11.3"),
    ("torch_version", "1.12.1"),  # never opened -> skipped
    ("unpinned:tqdm", None),  # value-less -> skipped
)


def test_env_assumptions_only_for_open_unknowns() -> None:
    plan_ev = env_plan_to_evidence(PLAN, "repo@abc")
    py_ev, py_open = _opened("python_version")
    cu_ev, cu_open = _opened("cuda_version")
    evidence = {e.evidence_id: e for e in (plan_ev, py_ev, cu_ev)}
    events = [py_open, cu_open]
    out = cli.env_assumption_events(
        PLAN, plan_ev.evidence_id, events, fold_unknowns(events, evidence)
    )
    assert out == [
        Assumption(unknown_id="unk:python_version", value="3.7", evidence_id=plan_ev.evidence_id),
        Assumption(unknown_id="unk:cuda_version", value="11.3", evidence_id=plan_ev.evidence_id),
    ]
    folded = fold_unknowns([*events, *out], evidence)
    for uid, value in (("unk:python_version", "3.7"), ("unk:cuda_version", "11.3")):
        assert folded[uid].status == "open" and folded[uid].value is None  # never resolved
        assert [a.value for a in folded[uid].assumptions] == [value]


def test_env_assumptions_skip_resolved_and_repeats() -> None:
    plan_ev = env_plan_to_evidence(PLAN, "repo@abc")
    py_ev, py_open = _opened("python_version")
    cu_ev, cu_open = _opened("cuda_version")
    repo_ev = make_evidence(
        kind="repo_file",
        source="repo@abc",
        locator="README.md",
        content={"text": "python 3.7"},
        confidence="observed",
        authority="repo",
    )
    evidence = {e.evidence_id: e for e in (plan_ev, py_ev, cu_ev, repo_ev)}
    events: list[Any] = [py_open, cu_open, resolve(py_open, repo_ev, "3.7")]
    first = cli.env_assumption_events(
        PLAN, plan_ev.evidence_id, events, fold_unknowns(events, evidence)
    )
    assert [a.unknown_id for a in first] == ["unk:cuda_version"]  # python resolved -> skipped
    events += first
    again = cli.env_assumption_events(
        PLAN, plan_ev.evidence_id, events, fold_unknowns(events, evidence)
    )
    assert again == []  # rerunning `env` does not duplicate


def test_env_assumptions_none_when_nothing_opened() -> None:
    plan_ev = env_plan_to_evidence(PLAN, "repo@abc")
    assert cli.env_assumption_events(PLAN, plan_ev.evidence_id, [], {}) == []


# --- gpu-check: per-run workdir cleanup ------------------------------------------------------


@pytest.mark.parametrize("keep", [False, True])
def test_gpu_check_workdir_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keep: bool, capsys: Any
) -> None:
    out = tmp_path / "run"
    (out / "repo").mkdir(parents=True)
    (out / "repo" / "train.py").write_text("print(1)\n")

    class _PE:
        tag = "reproducer/x:abc"

    seen: dict[str, Any] = {}

    def fake_run(image: str, cmd: list[str], **kw: Any) -> RunResult:
        seen.update(kw)
        assert (kw["workdir"] / "train.py").exists()  # the copy exists during the run
        kw["out_dir"].mkdir(parents=True)
        (kw["out_dir"] / "result.txt").write_text("ok")
        kw["log_path"].write_text("2.0 cuda ok\n")
        return RunResult(0, False, False, 1.0, kw["log_path"], kw["name"])

    monkeypatch.setattr(cli, "_plan_env", lambda out, cpu_only: _PE())
    monkeypatch.setattr(cli, "image_exists", lambda tag: True)
    monkeypatch.setattr(cli, "unload_ollama", lambda: [])
    monkeypatch.setattr(cli, "run_container", fake_run)
    argv = ["gpu-check", "--out", str(out)] + (["--keep-workdir"] if keep else [])
    assert cli.main(argv) == 0
    assert seen["workdir"].exists() is keep
    assert (seen["out_dir"] / "result.txt").read_text() == "ok"  # /out always kept
    assert seen["log_path"].read_text() == "2.0 cuda ok\n"  # log always kept
    assert (out / "repo" / "train.py").exists()  # the clone is never touched
    assert ("kept" if keep else "removed") in capsys.readouterr().out


def test_gpu_check_removes_workdir_when_run_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    (out / "repo").mkdir(parents=True)
    seen: dict[str, Any] = {}

    class _PE:
        tag = "reproducer/x:abc"

    def boom(image: str, cmd: list[str], **kw: Any) -> RunResult:
        seen.update(kw)
        raise RuntimeError("docker died")

    monkeypatch.setattr(cli, "_plan_env", lambda out, cpu_only: _PE())
    monkeypatch.setattr(cli, "image_exists", lambda tag: True)
    monkeypatch.setattr(cli, "unload_ollama", lambda: [])
    monkeypatch.setattr(cli, "run_container", boom)
    with pytest.raises(RuntimeError):
        cli.main(["gpu-check", "--out", str(out)])
    assert not seen["workdir"].exists()


def test_cleanup_workdir_keep(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    cli.cleanup_workdir(tmp_path / "w", keep=True)
    assert (tmp_path / "w").exists()
    cli.cleanup_workdir(tmp_path / "w", keep=False)
    assert not (tmp_path / "w").exists()
    cli.cleanup_workdir(tmp_path / "w", keep=False)  # already gone: no error
