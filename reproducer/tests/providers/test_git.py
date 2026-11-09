"""Tests for reproducer.providers.git (offline; git runs only inside tmp_path)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from reproducer.domain.inventory import FileInfo
from reproducer.graph.nodes.configure import classify_files
from reproducer.providers.git import (
    CloneError,
    UnsafePathError,
    clone_repo,
    head_sha,
    list_files,
    read_text,
    validate_commit,
)

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
def source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Local repo with two commits. Returns (path, first_sha, second_sha)."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "README.md").write_text("# Demo v1\n")
    (src / "requirements.txt").write_text("torch==2.0\n")
    (src / "configs").mkdir()
    (src / "configs" / "a.yaml").write_text("lr: 0.1\n")
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET\n")
    os.symlink(outside, src / "escape_link")
    os.symlink("README.md", src / "inner_link")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    first = _git(src, "rev-parse", "HEAD")
    (src / "README.md").write_text("# Demo v2\n")
    _git(src, "commit", "-q", "-am", "second")
    second = _git(src, "rev-parse", "HEAD")
    return src, first, second


def test_clone_head(source_repo: tuple[Path, str, str], tmp_path: Path) -> None:
    src, _, second = source_repo
    dest = tmp_path / "out" / "repo"
    sha = clone_repo(src.as_uri(), dest)
    assert sha == second
    assert len(sha) == 40
    assert (dest / "README.md").read_text() == "# Demo v2\n"


def test_clone_specific_commit(source_repo: tuple[Path, str, str], tmp_path: Path) -> None:
    src, first, _ = source_repo
    dest = tmp_path / "pinned"
    sha = clone_repo(src.as_uri(), dest, commit=first)
    assert sha == first
    assert (dest / "README.md").read_text() == "# Demo v1\n"


def test_clone_into_existing_empty_dir_ok(
    source_repo: tuple[Path, str, str], tmp_path: Path
) -> None:
    src, _, second = source_repo
    dest = tmp_path / "empty"
    dest.mkdir()
    assert clone_repo(src.as_uri(), dest) == second


def test_clone_refuses_non_empty_dest(source_repo: tuple[Path, str, str], tmp_path: Path) -> None:
    src, _, _ = source_repo
    dest = tmp_path / "busy"
    dest.mkdir()
    (dest / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        clone_repo(src.as_uri(), dest)
    assert (dest / "keep.txt").read_text() == "mine"
    assert sorted(p.name for p in dest.iterdir()) == ["keep.txt"]


def test_clone_bad_url_raises_and_cleans_up(tmp_path: Path) -> None:
    dest = tmp_path / "nope"
    with pytest.raises(CloneError) as ei:
        clone_repo((tmp_path / "does-not-exist").as_uri(), dest)
    assert "failed" in str(ei.value)
    assert not dest.exists()


def test_clone_bad_commit_raises(source_repo: tuple[Path, str, str], tmp_path: Path) -> None:
    src, _, _ = source_repo
    dest = tmp_path / "badcommit"
    with pytest.raises(CloneError):
        clone_repo(src.as_uri(), dest, commit="0" * 40)
    assert not dest.exists()


def test_clone_redacts_credentials() -> None:
    from reproducer.providers.git import _redact

    msg = _redact("fatal: could not read https://user:ghp_TOKEN@github.com/a/b.git")
    assert "ghp_TOKEN" not in msg
    assert "https://***@github.com/a/b.git" in msg


def test_list_files_skips_git_and_flags_symlinks(
    source_repo: tuple[Path, str, str], tmp_path: Path
) -> None:
    src, _, _ = source_repo
    dest = tmp_path / "c"
    clone_repo(src.as_uri(), dest)
    files = list_files(dest)
    by_path = {f.path: f for f in files}
    assert set(by_path) == {
        "README.md",
        "requirements.txt",
        "configs/a.yaml",
        "escape_link",
        "inner_link",
    }
    assert not any(p.startswith(".git/") for p in by_path)
    assert by_path["escape_link"] == FileInfo(path="escape_link", size=0, is_symlink=True)
    assert by_path["README.md"].size == len("# Demo v2\n")
    assert [f.path for f in files] == sorted(by_path)


def test_list_files_symlinked_dir_not_followed(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "real").mkdir(parents=True)
    (repo / "real" / "x.py").write_text("x")
    os.symlink(tmp_path, repo / "loop")  # would recurse forever / escape if followed
    files = {f.path: f for f in list_files(repo)}
    assert files == {
        "real/x.py": FileInfo(path="real/x.py", size=1, is_symlink=False),
        "loop": FileInfo(path="loop", size=0, is_symlink=True),
    }


def test_read_text_ok_and_inner_symlink(source_repo: tuple[Path, str, str]) -> None:
    src, _, _ = source_repo
    assert read_text(src, "README.md") == ("# Demo v2\n", False)
    assert read_text(src, "inner_link") == ("# Demo v2\n", False)


@pytest.mark.parametrize("bad", ["escape_link", "../secret.txt", "configs/../../secret.txt"])
def test_read_text_refuses_escape(source_repo: tuple[Path, str, str], bad: str) -> None:
    src, _, _ = source_repo
    with pytest.raises(UnsafePathError):
        read_text(src, bad)


def test_read_text_refuses_absolute_and_git_dir(source_repo: tuple[Path, str, str]) -> None:
    src, _, _ = source_repo
    with pytest.raises(UnsafePathError):
        read_text(src, str(src / "README.md"))
    with pytest.raises(UnsafePathError):
        read_text(src, ".git/config")


def test_read_text_truncation_and_decode(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("a" * 100)
    assert read_text(tmp_path, "big.txt", max_bytes=10) == ("a" * 10, True)
    assert read_text(tmp_path, "big.txt", max_bytes=100) == ("a" * 100, False)
    (tmp_path / "bin.dat").write_bytes(b"ok\xff\xfe")
    text, truncated = read_text(tmp_path, "bin.dat")
    assert text == "ok��" and truncated is False


def test_read_text_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text(tmp_path, "nope.txt")


@pytest.mark.parametrize(
    "bad", ["abc123", "A" * 40, "g" * 40, "a" * 39, "a" * 41, "main", " " + "a" * 40, ""]
)
def test_validate_commit_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        validate_commit(bad)


def test_validate_commit_accepts_full_sha() -> None:
    assert (
        validate_commit("0123456789abcdef" * 2 + "01234567") == "0123456789abcdef" * 2 + "01234567"
    )


def test_clone_short_commit_rejected_before_git(
    source_repo: tuple[Path, str, str], tmp_path: Path
) -> None:
    src, first, _ = source_repo
    dest = tmp_path / "short"
    with pytest.raises(ValueError, match="40-character"):
        clone_repo(src.as_uri(), dest, commit=first[:7])
    assert not dest.exists()


def test_head_sha_of_clone(source_repo: tuple[Path, str, str], tmp_path: Path) -> None:
    src, first, _ = source_repo
    dest = tmp_path / "h"
    clone_repo(src.as_uri(), dest, commit=first)
    assert head_sha(dest) == first


def test_head_sha_plain_dir_inside_outer_repo_rejected(
    source_repo: tuple[Path, str, str],
) -> None:
    src, _, _ = source_repo
    inner = src / "runs" / "x" / "repo"  # plain folder nested in a real repo
    inner.mkdir(parents=True)
    (inner / "file.txt").write_text("not a repo")
    with pytest.raises(CloneError, match="not a usable git checkout"):
        head_sha(inner)


def test_head_sha_subdir_of_repo_rejected(source_repo: tuple[Path, str, str]) -> None:
    src, _, _ = source_repo
    with pytest.raises(CloneError):
        head_sha(src / "configs")


def test_head_sha_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(CloneError):
        head_sha(tmp_path / "nope")


@pytest.mark.network
def test_network_clone_clip(tmp_path: Path) -> None:
    dest = tmp_path / "clip"
    sha = clone_repo("https://github.com/openai/CLIP", dest)
    assert len(sha) == 40
    inv = classify_files(list_files(dest), sha)
    assert inv.readme is not None and inv.readme.lower().startswith("readme")
    assert inv.requirement_files
    text, _ = read_text(dest, inv.readme)
    assert "CLIP" in text
