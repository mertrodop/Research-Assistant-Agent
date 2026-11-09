"""Tests for `python -m reproducer.cli inventory` (offline: clones come from a local repo)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
