"""Tests for the configure graph (graph/build.py) with FakeLLM on a local fixture repo."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.unknowns import UnknownOpened
from reproducer.graph.build import build_configure_graph
from reproducer.providers.git import clone_repo
from reproducer.providers.llm import Budget, FakeLLM, LLMError, MeteredLLM
from reproducer.providers.store import EVIDENCE_FILE, RunStore

URL = "https://github.com/org/proj"  # never contacted: <out>/repo already exists (reuse)
_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, env=_GIT_ENV, check=True)


def make_fixture_repo(tmp_path: Path) -> Path:
    """<tmp>/runs/proj with <out>/repo cloned from a tiny local repo. Returns out."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "README.md").write_text("# demo\nTrains a small model.\n")
    (src / "requirements.txt").write_text("numpy\n")
    (src / "train.py").write_text(
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--epochs')\n"
        "if __name__ == '__main__':\n    print('test_acc', p.parse_args())\n"
    )
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "init")
    out = tmp_path / "runs" / "proj"
    clone_repo(src.as_uri(), out / "repo")
    return out


ANSWERS: dict[str, Callable[[dict[str, str]], Any]] = {
    "subtask_a": lambda ids: {
        "packages": [
            {
                "name": "numpy",
                "version": None,
                "citation": {"evidence_id": ids["requirements.txt"], "quote": "numpy"},
            }
        ],
        "unknowns": [{"name": "python_version", "why": "not stated"}],
    },
    "subtask_b": lambda ids: {"entrypoints": [{"path": "train.py", "kind": "train"}]},
    "subtask_c": lambda ids: {"unknowns": [{"name": "dataset_source", "why": "not given"}]},
}


def fake_llm_for(
    answers: dict[str, Callable[[dict[str, str]], Any]],
    budget: Budget,
    barrier: threading.Barrier | None = None,
) -> Callable[[str], MeteredLLM]:
    """deps['llm_for']: one FakeLLM per node; an answer that raises propagates as LLMError."""

    def factory(node: str) -> MeteredLLM:
        def respond(system: str, prompt: str) -> str:
            if barrier is not None:
                barrier.wait()  # all three branches must be in flight at once
            ids = dict(re.findall(r"## (\S+) \[(E\d+)\]", prompt))  # path -> alias
            return json.dumps(answers[node](ids))

        return MeteredLLM(FakeLLM(respond), budget, node)

    return factory


def run_graph(out: Path, answers: Any = ANSWERS, barrier: Any = None) -> dict[str, Any]:
    budget = Budget()
    store = RunStore(out)
    graph = build_configure_graph(
        {"store": store, "llm_for": fake_llm_for(answers, budget, barrier), "budget": budget}
    )
    return graph.invoke(
        {
            "out_dir": str(out),
            "repo_url": URL,
            "commit": None,
            "max_chars": 12_000,
            "max_tokens": 1_500,
            "status": "running",
            "configure_attempts": 0,
        }
    )


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_all_ok_configured_and_persisted_once(tmp_path: Path) -> None:
    out = make_fixture_repo(tmp_path)
    final = run_graph(out, barrier=threading.Barrier(3, timeout=10))
    assert final["status"] == "configured"
    assert final["configure_attempts"] == 1
    assert sorted((o.kind, o.ok, o.attempts) for o in final["outcomes"]) == [
        ("A", True, 1),
        ("B", True, 1),
        ("C", True, 1),
    ]
    assert all(o.tokens > 0 for o in final["outcomes"])

    lines = _lines(out / EVIDENCE_FILE)
    ids = [json.loads(line)["evidence_id"] for line in lines]
    assert len(ids) == len(set(ids)) == len(final["evidence"]) == final["evidence_written"]
    kinds = [json.loads(line)["kind"] for line in lines]
    assert {"env_spec", "entrypoints", "workflow", "readme"} <= set(kinds)

    store = RunStore(out)
    opened = sorted(e.unknown_id for e in store.load_events() if isinstance(e, UnknownOpened))
    assert opened == ["unk:dataset_source", "unk:python_version"]
    assert final["events_written"] == 2
    nodes = sorted(r.node for r in store.load_call_records())
    assert nodes == ["subtask_a", "subtask_b", "subtask_c"]  # all three branches ran


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    out = make_fixture_repo(tmp_path)
    run_graph(out)
    n_ev, n_events = len(_lines(out / EVIDENCE_FILE)), len(RunStore(out).load_events())
    final = run_graph(out)
    assert final["status"] == "configured"
    assert final["evidence_written"] == 0 and final["events_written"] == 0
    assert len(_lines(out / EVIDENCE_FILE)) == n_ev
    assert len(RunStore(out).load_events()) == n_events


def test_failed_subtask_c_keeps_a_and_b_evidence(tmp_path: Path) -> None:
    out = make_fixture_repo(tmp_path)
    answers = {**ANSWERS, "subtask_c": lambda ids: {"unknowns": [{"name": "bogus"}]}}
    final = run_graph(out, answers)
    assert final["status"] == "configure_failed"
    c = next(o for o in final["outcomes"] if o.kind == "C")
    assert not c.ok and c.attempts == 3 and len(c.errors) == 3
    kinds = {ev.kind for ev in RunStore(out).load_evidence().values()}
    assert {"env_spec", "entrypoints"} <= kinds and "workflow" not in kinds
    opened = [e.unknown_id for e in RunStore(out).load_events() if isinstance(e, UnknownOpened)]
    assert opened == ["unk:python_version"]


def test_llm_error_in_c_is_a_failed_outcome_not_a_crash(tmp_path: Path) -> None:
    out = make_fixture_repo(tmp_path)

    def boom(ids: dict[str, str]) -> Any:
        raise LLMError("backend down")

    final = run_graph(out, {**ANSWERS, "subtask_c": boom})
    assert final["status"] == "configure_failed"
    c = next(o for o in final["outcomes"] if o.kind == "C")
    assert not c.ok and c.errors == ("LLMError: backend down",)
    kinds = {ev.kind for ev in RunStore(out).load_evidence().values()}
    assert {"env_spec", "entrypoints"} <= kinds and "workflow" not in kinds


def test_commit_mismatch_on_reused_clone_raises_and_writes_nothing(tmp_path: Path) -> None:
    from reproducer.providers.git import CloneError

    out = make_fixture_repo(tmp_path)
    budget = Budget()
    graph = build_configure_graph(
        {"store": RunStore(out), "llm_for": fake_llm_for(ANSWERS, budget), "budget": budget}
    )
    with pytest.raises(CloneError, match="not the requested commit"):
        graph.invoke({"out_dir": str(out), "repo_url": URL, "commit": "b" * 40})
    assert (out / EVIDENCE_FILE).read_text() == ""
