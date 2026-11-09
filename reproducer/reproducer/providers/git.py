"""Git provider: shallow-clone a paper's repo and read it without executing anything.

Safety:
  - git runs only via argument lists (no shell), with GIT_TERMINAL_PROMPT=0,
    GIT_LFS_SKIP_SMUDGE=1, hooks disabled and submodule recursion off.
  - Nothing from the clone is ever executed (no setup.py, scripts, hooks).
  - The destination must be absent or empty; it is never overwritten.
  - read_text refuses any path that resolves outside the repo (../ or symlink escape).
  - Credentials embedded in URLs are redacted from error messages.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

from reproducer.domain.inventory import FileInfo

# Applied to every git invocation: never run repo hooks, never touch submodules.
_SAFE_CONFIG = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "submodule.recurse=false",
    "-c",
    "advice.detachedHead=false",
)
_STDERR_CHARS = 2000
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class CloneError(RuntimeError):
    """git clone/fetch/checkout failed (message carries git's stderr, credentials redacted)."""


class UnsafePathError(ValueError):
    """Requested path resolves outside the repository."""


def validate_commit(commit: str) -> str:
    """Require a full 40-char lowercase hex sha (checked before any git runs)."""
    if not isinstance(commit, str) or not COMMIT_RE.match(commit):
        raise ValueError(f"commit must be a full 40-character lowercase hex sha, got {commit!r}")
    return commit


def _redact(text: str) -> str:
    """Drop userinfo (user:token@) from any URL in text."""
    return re.sub(r"(\w+://)[^/@\s]+@", r"\1***@", text)


def _git_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env.update(extra or {})
    return env


def _run_git(
    args: list[str],
    *,
    cwd: Path | None,
    deadline: float,
    extra_env: dict[str, str] | None = None,
) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CloneError(f"git {args[0]} timed out (overall clone timeout exceeded)")
    cmd = ["git", *_SAFE_CONFIG, *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=_git_env(extra_env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise CloneError(f"git {args[0]} timed out") from None
    if proc.returncode != 0:
        stderr = _redact(proc.stderr.strip())[-_STDERR_CHARS:]
        raise CloneError(f"git {args[0]} failed (exit {proc.returncode}): {stderr}")
    return proc.stdout.strip()


def clone_repo(url: str, dest: Path, commit: str | None = None, timeout_s: int = 300) -> str:
    """Shallow-clone url into dest (optionally at commit). Returns the full HEAD sha."""
    if commit is not None:
        validate_commit(commit)
    dest = Path(dest)
    existed = dest.exists()
    if existed and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"clone destination {dest} exists and is not an empty directory")
    deadline = time.monotonic() + timeout_s
    try:
        if commit is None:
            _run_git(
                ["clone", "--depth", "1", "--no-recurse-submodules", "--", url, str(dest)],
                cwd=None,
                deadline=deadline,
            )
        else:
            _run_git(["init", "-q", str(dest)], cwd=None, deadline=deadline)
            _run_git(["remote", "add", "origin", url], cwd=dest, deadline=deadline)
            _run_git(
                ["fetch", "--depth", "1", "--no-recurse-submodules", "origin", commit],
                cwd=dest,
                deadline=deadline,
            )
            _run_git(["checkout", "-q", "--detach", "FETCH_HEAD"], cwd=dest, deadline=deadline)
        return _run_git(["rev-parse", "HEAD"], cwd=dest, deadline=deadline)
    except BaseException:
        # dest was absent or empty before we started, so removing what we wrote is safe.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            if existed:
                dest.mkdir(exist_ok=True)
        raise


def head_sha(repo_dir: Path, timeout_s: int = 30) -> str:
    """Full HEAD sha of an existing checkout whose top level is exactly repo_dir.

    Repo discovery is fenced at repo_dir's parent (GIT_CEILING_DIRECTORIES), so a plain
    folder nested inside another repo (e.g. runs/x/repo inside the project) is rejected
    instead of reporting the outer repo's HEAD. Raises CloneError if not a git checkout.
    """
    root = Path(repo_dir).resolve()
    if not root.is_dir():
        raise CloneError(f"{repo_dir} is not a directory")
    fence = {"GIT_CEILING_DIRECTORIES": str(root.parent)}
    deadline = time.monotonic() + timeout_s
    try:
        top = _run_git(
            ["rev-parse", "--show-toplevel"], cwd=root, deadline=deadline, extra_env=fence
        )
        if Path(top).resolve() != root:
            raise CloneError(f"{repo_dir} is not the top level of a git checkout")
        sha = _run_git(["rev-parse", "HEAD"], cwd=root, deadline=deadline, extra_env=fence)
    except CloneError as e:
        raise CloneError(f"{repo_dir} is not a usable git checkout: {e}") from None
    if not COMMIT_RE.match(sha):
        raise CloneError(f"{repo_dir}: unexpected HEAD {sha!r}")
    return sha


def list_files(repo_dir: Path) -> tuple[FileInfo, ...]:
    """All regular files and symlinks under repo_dir, sorted, skipping .git/. No link follows."""
    root = Path(repo_dir)
    out: list[FileInfo] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        keep: list[str] = []
        for d in dirnames:
            if d == ".git":
                continue
            p = here / d
            if p.is_symlink():  # symlinked dir: list it, never descend
                out.append(FileInfo(path=p.relative_to(root).as_posix(), size=0, is_symlink=True))
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in filenames:
            p = here / name
            st = p.lstat()
            rel = p.relative_to(root).as_posix()
            if stat.S_ISLNK(st.st_mode):
                out.append(FileInfo(path=rel, size=0, is_symlink=True))
            elif stat.S_ISREG(st.st_mode):
                out.append(FileInfo(path=rel, size=st.st_size, is_symlink=False))
    return tuple(sorted(out, key=lambda f: f.path))


def read_text(repo_dir: Path, rel_path: str, max_bytes: int = 50_000) -> tuple[str, bool]:
    """Read up to max_bytes of a repo file as UTF-8 (errors replaced). Returns (text, truncated).

    Raises UnsafePathError if the path (after resolving symlinks) leaves repo_dir.
    """
    root = Path(repo_dir).resolve(strict=True)
    rel = Path(rel_path)
    if rel.is_absolute():
        raise UnsafePathError(f"absolute path not allowed: {rel_path!r}")
    target = (root / rel).resolve(strict=False)
    if target != root and root not in target.parents:
        raise UnsafePathError(f"path {rel_path!r} resolves outside the repository")
    if ".git" in target.relative_to(root).parts:
        raise UnsafePathError(f"path {rel_path!r} is inside .git/")
    if not target.is_file():
        raise FileNotFoundError(f"not a regular file in repo: {rel_path!r}")
    with open(target, "rb") as f:
        raw = f.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    return raw[:max_bytes].decode("utf-8", errors="replace"), truncated
