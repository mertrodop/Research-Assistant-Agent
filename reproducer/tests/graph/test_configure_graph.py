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


def test_commit_mismatch_on_reused_clone_falls_back_and_writes_only_the_fallback(
    tmp_path: Path,
) -> None:
    out = make_fixture_repo(tmp_path)
    budget = Budget()
    graph = build_configure_graph(
        {"store": RunStore(out), "llm_for": fake_llm_for(ANSWERS, budget), "budget": budget}
    )
    final = graph.invoke({"out_dir": str(out), "repo_url": URL, "commit": "b" * 40})
    assert final["fallback_reason"] == "fallback: configure: CloneError"
    assert "not the requested commit" in final["fallback_detail"]
    assert not final.get("outcomes")  # the subtasks were no-ops: no LLM call
    # nothing but the one fallback record of this invocation (persist -> fallback node)
    stored = list(RunStore(out).load_evidence().values())
    assert [ev.kind for ev in stored] == ["fallback"]
    assert stored[0].content["reason"] == "fallback: configure: CloneError"
    assert stored[0].content["router_or_node"] == "configure"
    assert "not the requested commit" in stored[0].content["detail"]


# --------------------------------------------------------------------------- spec input

FIXTURE_SPEC = Path(__file__).resolve().parents[1] / "fixtures" / "repro_spec_example.json"


def write_spec(tmp_path: Path, name: str = "demo", **changes: Any) -> Path:
    """Copy of the fixture spec at <tmp>/specs/<name>/spec.json (paper source paper:<name>)."""
    data = {**json.loads(FIXTURE_SPEC.read_text()), **changes}
    path = tmp_path / "specs" / name / "spec.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    return path


def run_spec_graph(out: Path, spec_path: Path, answers: Any = ANSWERS) -> dict[str, Any]:
    budget = Budget()
    graph = build_configure_graph(
        {"store": RunStore(out), "llm_for": fake_llm_for(answers, budget), "budget": budget}
    )
    return graph.invoke(
        {"out_dir": str(out), "spec_path": str(spec_path), "status": "running", "commit": None}
    )


def _ledger(out: Path, spec: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    from reproducer.cli import unknowns_by_origin
    from reproducer.domain.ledger import Tolerance, ToleranceConfig, compute_ledger
    from reproducer.domain.unknowns import fold_unknowns

    store = RunStore(out)
    evidence, events = store.load_evidence(), store.load_events()
    unknowns = fold_unknowns(events, evidence)  # raises on a conflicting re-open
    tol = ToleranceConfig(default=Tolerance(abs=0.5))
    return (
        compute_ledger(spec.claims, evidence, unknowns, tol),
        unknowns_by_origin(events, evidence, unknowns),
    )


def test_spec_drives_configure_and_ledger_is_open(tmp_path: Path) -> None:
    out = make_fixture_repo(tmp_path)
    spec_path = write_spec(tmp_path)
    final = run_spec_graph(out, spec_path)
    assert final["status"] == "configured"
    assert final["paper_source"] == "paper:demo"
    assert final["repo_url"] == "https://github.com/example-lab/resnet18-cifar10"
    assert final["source"].startswith("repo:example-lab/resnet18-cifar10@")

    stored = RunStore(out).load_evidence().values()
    paper = [ev for ev in stored if ev.source == "paper:demo"]
    assert sorted(ev.kind for ev in paper) == ["paper_claim"] * 2 + ["paper_setup"] * 3
    assert {"env_spec", "entrypoints", "workflow"} <= {ev.kind for ev in stored}

    ledger, origins = _ledger(out, final["spec"])
    assert set(ledger) == {"c1", "c2"}
    expected = ("unk:dataset_source", "unk:python_version", "unk:weight_decay")
    for entry in ledger.values():
        assert entry.status == "open" and entry.fidelity is None and entry.attempts == 0
        assert entry.open_unknowns == expected
        assert len(entry.evidence_ids) == 1  # its paper_claim
    assert origins == {
        "paper setup": ["weight_decay"],
        "repo subtasks": ["dataset_source", "python_version"],
    }

    # idempotent re-run: nothing new written, same ledger
    n_ev, n_events = len(_lines(out / EVIDENCE_FILE)), len(RunStore(out).load_events())
    again = run_spec_graph(out, spec_path)
    assert again["status"] == "configured"
    assert again["evidence_written"] == 0 and again["events_written"] == 0
    assert len(_lines(out / EVIDENCE_FILE)) == n_ev
    assert len(RunStore(out).load_events()) == n_events
    assert _ledger(out, again["spec"]) == (ledger, origins)


def _opened(out: Path) -> list[UnknownOpened]:
    return [e for e in RunStore(out).load_events() if isinstance(e, UnknownOpened)]


def test_paper_null_colliding_with_subtask_proposal_opens_once(tmp_path: Path) -> None:
    """Paper null setup 'python_version' == subtask A's proposal: opened once, by the paper
    (ingest_spec runs first; A skips it as already open in the state)."""
    out = make_fixture_repo(tmp_path)
    setup = [{"name": "python_version", "value": None, "location": "", "quote": ""}]
    final = run_spec_graph(out, write_spec(tmp_path, setup=setup))
    assert final["status"] == "configured"
    opened = _opened(out)
    assert sorted(e.unknown_id for e in opened) == ["unk:dataset_source", "unk:python_version"]
    py = next(e for e in opened if e.name == "python_version")
    assert RunStore(out).load_evidence()[py.evidence_id].kind == "paper_setup"
    _, origins = _ledger(out, final["spec"])
    assert origins == {"paper setup": ["python_version"], "repo subtasks": ["dataset_source"]}


def test_subtask_unknown_opened_first_is_not_reopened_by_paper(tmp_path: Path) -> None:
    """Reverse order: a URL-only run opened 'python_version' from subtask A; a later spec run
    whose paper has the same null name does not re-open it (first wins, fold stays valid)."""
    out = make_fixture_repo(tmp_path)
    run_graph(out)
    first = {e.unknown_id: e for e in _opened(out)}
    setup = [{"name": "python_version", "value": None, "location": "", "quote": ""}]
    final = run_spec_graph(out, write_spec(tmp_path, setup=setup))
    assert final["status"] == "configured" and final["events_written"] == 0
    assert {e.unknown_id: e for e in _opened(out)} == first
    _, origins = _ledger(out, final["spec"])
    assert origins == {"paper setup": [], "repo subtasks": ["dataset_source", "python_version"]}
