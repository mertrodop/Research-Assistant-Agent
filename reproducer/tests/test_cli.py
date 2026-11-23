"""Tests for `python -m reproducer.cli inventory`, `subtasks` and `configure` (offline by
default; one llm_live test)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reproducer.cli import main
from reproducer.providers.git import clone_repo
from reproducer.providers.store import EVIDENCE_FILE, RunStore

URL = "https://github.com/org/proj"  # never contacted: tests only use --reuse-clone paths
_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def cloned_out(tmp_path: Path) -> tuple[Path, str, str]:
    """<out>/repo cloned from a local two-commit repo at HEAD. Returns (out, first, second)."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# v1\n")
    (src / "requirements.txt").write_text("numpy\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    first = _git(src, "rev-parse", "HEAD")
    (src / "README.md").write_text("# v2\n")
    _git(src, "commit", "-q", "-am", "second")
    second = _git(src, "rev-parse", "HEAD")
    out = tmp_path / "runs" / "proj"
    clone_repo(src.as_uri(), out / "repo")
    return out, first, second


def _evidence_lines(out: Path) -> int:
    return len((out / EVIDENCE_FILE).read_text().splitlines())


def test_reuse_clone_rerun_writes_no_new_evidence(
    cloned_out: tuple[Path, str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    out, _, second = cloned_out
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    first_run = capsys.readouterr().out
    assert f"commit:        {second}" in first_run
    assert "evidence:      3 written (3 built)" in first_run  # inventory + readme + reqs
    n = _evidence_lines(out)

    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    assert "evidence:      0 written (3 built)" in capsys.readouterr().out
    assert _evidence_lines(out) == n
    sources = {ev.source for ev in RunStore(out).load_evidence().values()}
    assert sources == {f"repo:org/proj@{second}"}


def test_inventory_stores_derived_facts_without_executing_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = tmp_path / "SENTINEL"
    src = tmp_path / "src"
    (src / "configs").mkdir(parents=True)
    (src / "README.md").write_text("# demo\n")
    (src / "requirements.txt").write_text("torch>=1.7\ngit+https://github.com/x/y.git\n")
    (src / "setup.py").write_text(
        f"import os\nos.system('touch {sentinel}')\n"
        "from setuptools import setup\n"
        "setup(install_requires=['ftfy'], python_requires='>=3.7')\n"
    )
    (src / "train.py").write_text(
        f"import argparse, os\nos.system('touch {sentinel}')\n"
        "p = argparse.ArgumentParser()\np.add_argument('--lr')\n"
        "if __name__ == '__main__':\n    p.parse_args()\n"
    )
    (src / "configs" / "base.yaml").write_text("model:\n  lr: 0.1\n")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "init")
    out = tmp_path / "runs" / "demo"
    clone_repo(src.as_uri(), out / "repo")

    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    stdout = capsys.readouterr().out
    assert "extracted:     config_keys=1 entrypoint=1 requirements_txt=1 setup_py=1" in stdout
    # requirements.txt/setup.py already recorded by the inventory step (same ids, deduped)
    assert "derived:       4 written (4 built), +2 source-file records" in stdout
    assert not sentinel.exists()

    derived = {
        ev.locator: ev
        for ev in RunStore(out).load_evidence().values()
        if ev.confidence == "derived"
    }
    assert set(derived) == {
        "requirements.txt#parsed",
        "setup.py#parsed",
        "train.py#parsed",
        "configs/base.yaml#parsed",
    }
    setup = derived["setup.py#parsed"].content["result"]
    assert setup["install_requires"] == ["ftfy"] and setup["python_requires"] == ">=3.7"
    reqs = derived["requirements.txt#parsed"].content["result"]
    assert [u["reason"] for u in reqs["unsupported"]] == ["url"]
    assert derived["train.py#parsed"].content["result"]["arg_names"] == ["--lr"]
    stored = RunStore(out).load_evidence()
    assert all(link in stored for ev in derived.values() for link in ev.links)

    n = _evidence_lines(out)
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    assert "derived:       0 written (4 built), +0 source-file records" in capsys.readouterr().out
    assert _evidence_lines(out) == n
    assert not sentinel.exists()


def test_reuse_clone_same_commit_ok(cloned_out: tuple[Path, str, str]) -> None:
    out, _, second = cloned_out
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone", "--commit", second]) == 0


def test_reuse_clone_commit_mismatch_errors(
    cloned_out: tuple[Path, str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    out, first, second = cloned_out
    rc = main(["inventory", URL, "--out", str(out), "--reuse-clone", "--commit", first])
    assert rc == 1
    err = capsys.readouterr().err
    assert second in err and first in err and "--reuse-clone" in err
    assert not (out / EVIDENCE_FILE).exists()


def test_reuse_clone_not_a_repo_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs" / "x"
    (out / "repo").mkdir(parents=True)
    (out / "repo" / "README.md").write_text("plain folder")
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 1
    assert "not a usable git checkout" in capsys.readouterr().err
    assert not (out / EVIDENCE_FILE).exists()


def test_without_reuse_flag_existing_repo_still_refused(
    cloned_out: tuple[Path, str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    out, _, _ = cloned_out
    assert main(["inventory", URL, "--out", str(out)]) == 1
    assert "exists and is not an empty directory" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["abc1234", "A" * 40, "main"])
def test_bad_commit_rejected_before_git(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    out = tmp_path / "o"
    assert main(["inventory", URL, "--out", str(out), "--commit", bad]) == 2
    assert "40-character" in capsys.readouterr().err
    assert not out.exists()


def test_non_github_url_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inventory", "https://gitlab.com/a/b", "--out", str(tmp_path / "o")]) == 2
    assert "GitHub" in capsys.readouterr().err


# --------------------------------------------------------------------------- subtasks


def _fake_llm_for_node(answers: dict[str, Any]) -> Any:
    """llm_for_node replacement: FakeLLM answering per node, metered on the shared budget."""
    import json
    import re

    from reproducer.providers.llm import FakeLLM, MeteredLLM

    def factory(node: str, budget: Any) -> MeteredLLM:
        def respond(system: str, prompt: str) -> str:
            ids = dict(re.findall(r"## (\S+) \[(E\d+)\]", prompt))  # path -> alias
            return json.dumps(answers[node](ids))

        return MeteredLLM(FakeLLM(respond), budget, node)

    return factory


def test_subtasks_appends_evidence_events_and_call_records(
    cloned_out: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import reproducer.cli as cli
    from reproducer.domain.unknowns import UnknownOpened

    out, _, _ = cloned_out
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    answers = {
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
        "subtask_b": lambda ids: {"entrypoints": []},  # repo has no candidates -> fails
        "subtask_c": lambda ids: {"unknowns": [{"name": "dataset_source", "why": "not given"}]},
    }
    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(answers))
    capsys.readouterr()
    assert main(["subtasks", "--out", str(out)]) == 1  # B failed
    text = capsys.readouterr().out
    assert "A env           ok=True attempts=1" in text
    assert "B entrypoints   ok=False attempts=3" in text
    assert "C workflow      ok=True attempts=1" in text

    store = RunStore(out)
    stored = list(store.load_evidence().values())
    kinds = [ev.kind for ev in stored]
    assert kinds.count("env_spec") == 1 and kinds.count("workflow") == 1
    assert "entrypoints" not in kinds
    env = next(ev for ev in stored if ev.kind == "env_spec")
    req_id = next(ev.evidence_id for ev in stored if ev.locator == "requirements.txt")
    assert env.content["packages"][0]["source"] == "repo_file"
    assert env.content["packages"][0]["citation"]["evidence_id"] == req_id  # alias resolved
    assert env.links == (req_id,)
    assert {ev.locator for ev in stored} >= {"README.md#versions", "requirements.txt#versions"}
    events = store.load_events()
    opened = [e.unknown_id for e in events if isinstance(e, UnknownOpened)]
    assert opened == ["unk:python_version", "unk:dataset_source"]
    nodes = [r.node for r in store.load_call_records()]
    assert nodes == ["subtask_a", "subtask_b", "subtask_b", "subtask_b", "subtask_c"]

    # rerun: same answers -> same evidence ids (deduped), no re-opened unknowns
    n_lines = _evidence_lines(out)
    main(["subtasks", "--out", str(out)])
    assert _evidence_lines(out) == n_lines
    assert len(store.load_events()) == 2


def test_subtasks_without_store_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["subtasks", "--out", str(tmp_path / "none")]) == 2
    assert "run `inventory` first" in capsys.readouterr().err


def test_inventory_stores_shell_scripts_and_subtasks_finds_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reproducer.cli as cli

    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    _git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# p\n")
    (src / "train.py").write_text(
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--epochs')\n"
    )
    (src / "scripts" / "run.sh").write_text("python train.py \\\n  --epochs 3\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "c")
    out = tmp_path / "runs" / "p"
    clone_repo(src.as_uri(), out / "repo")
    assert main(["inventory", URL, "--out", str(out), "--reuse-clone"]) == 0
    assert "shell scripts: 1 found, 1 written" in capsys.readouterr().out

    answers = {
        "subtask_a": lambda ids: {},
        "subtask_b": lambda ids: {
            "entrypoints": [
                {
                    "path": "train.py",
                    "kind": "train",
                    "example": {"evidence_id": ids["Verified"], "quote": "--epochs 3"},
                }
            ]
        },
        "subtask_c": lambda ids: {},
    }

    def ids_with_commands(factory: Any) -> Any:
        import re

        def wrapped(node: str, budget: Any) -> Any:
            m = factory(node, budget)
            inner = m.inner._fn

            def fn(system: str, prompt: str) -> str:
                alias = re.search(r"- \[(E\d+)\] \"python train.py --epochs 3\"", prompt)
                return inner(
                    system, prompt + (f"\n## Verified [{alias.group(1)}]" if alias else "")
                )

            m.inner._fn = fn
            return m

        return wrapped

    monkeypatch.setattr(cli, "llm_for_node", ids_with_commands(_fake_llm_for_node(answers)))
    assert main(["subtasks", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "commands:       1 found by code" in text
    assert "train.py kind=train args=['--epochs'] example=True" in text
    stored = RunStore(out).load_evidence().values()
    cmd_ev = next(ev for ev in stored if ev.locator == "scripts/run.sh#commands")
    assert cmd_ev.content["result"][0]["line"] == "python train.py --epochs 3"
    b = next(ev for ev in stored if ev.kind == "entrypoints")
    assert b.links == (cmd_ev.evidence_id,)


# --------------------------------------------------------------------------- configure


def _configure_fixture(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# demo\n")
    (src / "requirements.txt").write_text("numpy\n")
    (src / "train.py").write_text(
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--epochs')\n"
        "if __name__ == '__main__':\n    p.parse_args()\n"
    )
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "c")
    out = tmp_path / "runs" / "demo"
    clone_repo(src.as_uri(), out / "repo")
    return out


_CONFIGURE_ANSWERS: dict[str, Any] = {
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


def test_configure_runs_graph_and_reruns_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reproducer.cli as cli

    out = _configure_fixture(tmp_path)
    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(_CONFIGURE_ANSWERS))
    assert main(["configure", URL, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "status:         configured" in text
    for line in ("A env           ok=True attempts=1", "B entrypoints   ok=True attempts=1"):
        assert line in text
    assert "C workflow      ok=True attempts=1" in text
    n = _evidence_lines(out)
    assert f"evidence:       {n} written ({n} in state)" in text
    assert "events:         2 written" in text and "wall time:" in text
    kinds = [ev.kind for ev in RunStore(out).load_evidence().values()]
    assert kinds.count("env_spec") == kinds.count("entrypoints") == kinds.count("workflow") == 1

    assert main(["configure", URL, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"evidence:       0 written ({n} in state)" in text
    assert "events:         0 written" in text
    assert _evidence_lines(out) == n


def test_configure_failed_subtask_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reproducer.cli as cli

    out = _configure_fixture(tmp_path)
    answers = {**_CONFIGURE_ANSWERS, "subtask_b": lambda ids: {"entrypoints": []}}
    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(answers))
    assert main(["configure", URL, "--out", str(out)]) == 1
    text = capsys.readouterr().out
    assert "status:         configure_failed" in text
    assert "B entrypoints   ok=False attempts=3" in text
    assert "  error attempt 1:" in text
    kinds = {ev.kind for ev in RunStore(out).load_evidence().values()}
    assert {"env_spec", "workflow"} <= kinds and "entrypoints" not in kinds


def test_configure_without_llm_config_errors_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in [v for v in os.environ if v.startswith("REPRODUCER_LLM")]:
        monkeypatch.delenv(var)
    out = tmp_path / "o"
    assert main(["configure", URL, "--out", str(out)]) == 2
    assert "no LLM configured" in capsys.readouterr().err
    assert not out.exists()


CUTOUT_URL = "https://github.com/uoguelph-mlrg/Cutout"
CUTOUT_CLONE = Path(__file__).resolve().parents[1] / "runs" / "cutout" / "repo"


@pytest.mark.llm_live
def test_live_configure_cutout_reuses_clone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Configure graph on Cutout with the configured backend (REPRODUCER_LLM*), reusing a
    copy of runs/cutout/repo (no network clone)."""
    import shutil

    if not (CUTOUT_CLONE / ".git").is_dir():
        pytest.skip(f"no Cutout clone at {CUTOUT_CLONE}")
    if not os.environ.get("REPRODUCER_LLM"):
        pytest.skip("REPRODUCER_LLM not set")
    out = tmp_path / "cutout"
    shutil.copytree(CUTOUT_CLONE, out / "repo", symlinks=True)
    rc = main(["configure", CUTOUT_URL, "--out", str(out)])
    text = capsys.readouterr().out
    print(text)
    assert rc == 0, text
    assert "status:         configured" in text
    kinds = {ev.kind for ev in RunStore(out).load_evidence().values()}
    assert {"env_spec", "entrypoints", "workflow"} <= kinds
    b = next(ev for ev in RunStore(out).load_evidence().values() if ev.kind == "entrypoints")
    assert [e["path"] for e in b.content["entrypoints"]] == ["train.py"]


def test_subtasks_prompts_independent_of_store_order_and_match_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    import reproducer.cli as cli

    src = tmp_path / "src"
    (src / "configs").mkdir(parents=True)
    _git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# p\nRun `python train.py --epochs 3`\n")
    (src / "requirements.txt").write_text("numpy\n")
    (src / "requirements-dev.txt").write_text("pytest\n")
    (src / "run.sh").write_text("python train.py --epochs 3\n")
    (src / "configs" / "a.yaml").write_text("lr: 0.1\n")
    (src / "configs" / "b.yaml").write_text("bs: 8\n")
    (src / "train.py").write_text(
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--epochs')\n"
        "if __name__ == '__main__':\n    p.parse_args()\n"
    )
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "c")
    outs = [tmp_path / "runs" / n for n in ("fwd", "rev", "graph")]
    clone_repo(src.as_uri(), outs[0] / "repo")
    for out in outs[1:]:
        shutil.copytree(outs[0] / "repo", out / "repo", symlinks=True)

    assert main(["inventory", URL, "--out", str(outs[0]), "--reuse-clone"]) == 0
    lines = (outs[0] / EVIDENCE_FILE).read_text().splitlines()
    (outs[1] / EVIDENCE_FILE).write_text("\n".join(reversed(lines)) + "\n")  # same set

    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(_CONFIGURE_ANSWERS))
    main(["subtasks", "--out", str(outs[0])])
    main(["subtasks", "--out", str(outs[1])])
    main(["configure", URL, "--out", str(outs[2])])
    capsys.readouterr()

    def prompts(out: Path) -> list[tuple[str, str]]:
        return sorted((r.node, r.prompt_sha256) for r in RunStore(out).load_call_records())

    assert prompts(outs[0]) == prompts(outs[1]) == prompts(outs[2])
    assert {n for n, _ in prompts(outs[0])} == {"subtask_a", "subtask_b", "subtask_c"}


# --------------------------------------------------------------------------- configure --spec

FIXTURE_SPEC = Path(__file__).resolve().parent / "fixtures" / "repro_spec_example.json"


def _spec_copy(tmp_path: Path, **changes: Any) -> Path:
    import json

    path = tmp_path / "evals" / "demo" / "spec.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**json.loads(FIXTURE_SPEC.read_text()), **changes}))
    return path


def test_configure_from_spec_prints_ledger_and_unknowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reproducer.cli as cli

    out = _configure_fixture(tmp_path)
    spec = _spec_copy(tmp_path)
    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(_CONFIGURE_ANSWERS))
    assert main(["configure", "--spec", str(spec), "--out", str(out), "--tol-abs", "0.3"]) == 0
    text = capsys.readouterr().out
    assert "status:         configured" in text
    assert "paper:          paper:demo" in text
    assert "repo:           https://github.com/example-lab/resnet18-cifar10" in text
    assert "events:         3 written" in text  # weight_decay (paper) + 2 subtask proposals
    assert "ledger:         tolerance abs=0.3 rel=None" in text
    for cid, value in (("c1", "95.3"), ("c2", "93.1")):
        assert (
            f"  {cid}  metric=top1_accuracy  paper={value} %  status=open  fidelity=-  "
            "attempts=0" in text
        )
    assert text.count("    open unknowns: dataset_source, python_version, weight_decay") == 2
    assert "  paper setup:    weight_decay" in text
    assert "  repo subtasks:  dataset_source, python_version" in text

    assert main(["configure", "--spec", str(spec), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "events:         0 written" in text and "status=open" in text


def test_configure_spec_and_url_are_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o"
    assert main(["configure", "--out", str(out)]) == 2
    assert main(["configure", URL, "--spec", "x.json", "--out", str(out)]) == 2
    assert "either a GitHub URL or --spec" in capsys.readouterr().err
    assert not out.exists()


def test_configure_bad_spec_fails_without_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reproducer.cli as cli

    monkeypatch.setattr(cli, "llm_for_node", _fake_llm_for_node(_CONFIGURE_ANSWERS))
    spec = _spec_copy(tmp_path, code_links=["https://example.org/code.zip"])
    out = tmp_path / "runs" / "demo"
    assert main(["configure", "--spec", str(spec), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert "status:         configure_failed" in captured.out
    assert "no GitHub repository in code_links" in captured.err
    assert not (out / "repo").exists()
