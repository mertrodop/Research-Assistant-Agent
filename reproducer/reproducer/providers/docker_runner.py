"""Docker runner provider: build environment images and run UNTRUSTED repo code in containers.

Safety (every container started here):
  - docker runs via argument lists only (no shell); arguments are built by the pure functions
    build_args / run_args so the flags can be tested without Docker;
  - no network (--network none), all capabilities dropped, no-new-privileges, read-only root
    filesystem with a size-capped /tmp tmpfs, memory / CPU / pids limits, host uid:gid (never
    root), HOME=/tmp;
  - only three kinds of mounts: the per-run working copy at /work (rw), the output dir at /out
    (rw) and data dirs (ro), which may be nested under /work (e.g. /work/data) but never at
    /work, /out, /tmp, / or under /out;
  - images are built with NO build context (`docker build -` with the Dockerfile on stdin), so
    nothing from the host or the repo is copied into an image;
  - the repo clone is never mounted or modified: prepare_workdir makes a fresh copy (no .git)
    per run and applies patches there;
  - on timeout the container is killed; it is always removed at the end (docker rm -f);
  - a build timeout kills the docker CLI, which cancels the BuildKit build (see build_image).
Logs are streamed to a file and capped (head + tail kept).
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from reproducer.domain.env_rules import EnvPlan, render_dockerfile

DEFAULT_LOG_CAP_BYTES = 20 * 1024 * 1024
WORK_MOUNT = "/work"
OUT_MOUNT = "/out"
TMPFS_SIZE = "2g"
SHM_SIZE = "2g"  # PyTorch DataLoader workers share tensors via /dev/shm (64 MB default)
DOCKER_CMD_TIMEOUT_S = 60.0

_PAPER_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class DockerError(RuntimeError):
    """A docker CLI call (inspect/kill/rm) failed unexpectedly."""


class PatchError(RuntimeError):
    """A patch did not apply to the working copy."""


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    tag: str
    cached: bool
    duration_s: float
    log_path: Path | None
    timed_out: bool = False


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None  # None if the container never reported one
    timed_out: bool
    oom_killed: bool
    duration_s: float
    log_path: Path
    name: str


# --- pure helpers ----------------------------------------------------------------------------


def image_tag(plan: EnvPlan, paper: str) -> str:
    """reproducer/<paper>:<sha256(rendered Dockerfile)[:12]>. Same plan -> same tag."""
    if not _PAPER_RE.match(paper):
        raise ValueError(f"paper name must be a lowercase docker repo component, got {paper!r}")
    digest = hashlib.sha256(render_dockerfile(plan).encode("utf-8")).hexdigest()[:12]
    return f"reproducer/{paper}:{digest}"


def build_args(tag: str) -> list[str]:
    """`docker build` with the Dockerfile on stdin and no build context."""
    return ["docker", "build", "--progress=plain", "-t", tag, "-"]


DataMounts = Mapping[str, str] | Sequence[tuple[str, str]]


def mount_pairs(data_mounts: DataMounts) -> list[tuple[str, str]]:
    """(host, container) pairs: a host -> container mapping, or pairs (one host dir may then
    be mounted at several targets)."""
    if isinstance(data_mounts, Mapping):
        return [(str(h), str(c)) for h, c in data_mounts.items()]
    return [(str(h), str(c)) for h, c in data_mounts]


def _check_mount_path(path: str, what: str) -> None:
    if not os.path.isabs(path):
        raise ValueError(f"{what} must be an absolute path, got {path!r}")
    if ":" in path or "," in path:
        raise ValueError(f"{what} may not contain ':' or ',', got {path!r}")


def run_args(
    image: str,
    cmd: Sequence[str],
    *,
    name: str,
    workdir: str,
    out_dir: str,
    data_mounts: DataMounts = {},  # noqa: B006 - read-only mapping
    gpu: bool,
    uid: int,
    gid: int,
    mem: str = "12g",
    cpus: float = 4.0,
    pids: int = 512,
) -> list[str]:
    """The full `docker run` argument list (see module docstring for the policy)."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid container name {name!r}")
    if not cmd:
        raise ValueError("cmd must not be empty")
    _check_mount_path(workdir, "workdir")
    _check_mount_path(out_dir, "out_dir")
    reserved = {WORK_MOUNT, OUT_MOUNT, "/tmp", "/"}
    pairs = mount_pairs(data_mounts)
    targets = [c for _, c in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError(f"duplicate data mount targets: {targets}")
    for host, container in pairs:
        _check_mount_path(host, "data mount source")
        _check_mount_path(container, "data mount target")
        if posixpath.normpath(container) != container or container.startswith("//"):
            raise ValueError(f"data mount target {container!r} is not a normalized path")
        # read-only data may sit INSIDE the rw working copy (e.g. /work/data, where the repo
        # expects it); never at /work or /out themselves, nor anywhere under /out
        if container in reserved or container.startswith(OUT_MOUNT + "/"):
            raise ValueError(f"data mount target {container!r} collides with a reserved path")
    args = [
        "docker",
        "run",
        "--rm=false",
        "--name",
        name,
        "--network",
        "none",
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONUNBUFFERED=1",  # progress printed to stdout reaches the log before a timeout
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,size={TMPFS_SIZE}",
        "--shm-size",
        SHM_SIZE,
        "--memory",
        mem,
        "--memory-swap",
        mem,  # no swap beyond the memory limit
        "--cpus",
        f"{cpus:g}",
        "--pids-limit",
        str(pids),
    ]
    if gpu:
        args += ["--gpus", "all"]
    args += ["-v", f"{workdir}:{WORK_MOUNT}:rw", "-v", f"{out_dir}:{OUT_MOUNT}:rw"]
    for host, container in sorted(pairs, key=lambda hc: (hc[1], hc[0])):
        args += ["-v", f"{host}:{container}:ro"]
    args += ["-w", WORK_MOUNT, image, *cmd]
    return args


class CappedLog:
    """Write-through log capped at max_bytes: the first half is written as it arrives, the
    last half is kept in memory and appended on close() behind an omission marker."""

    def __init__(self, path: Path, max_bytes: int = DEFAULT_LOG_CAP_BYTES) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO[bytes] = open(self.path, "wb")  # noqa: SIM115 - closed in close()
        self._head_left = max_bytes // 2
        self._tail_cap = max_bytes - self._head_left
        self._tail: deque[bytes] = deque()
        self._tail_len = 0
        self._omitted = 0
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._head_left > 0:
                head, data = data[: self._head_left], data[self._head_left :]
                self._f.write(head)
                self._f.flush()
                self._head_left -= len(head)
            if not data:
                return
            self._tail.append(data)
            self._tail_len += len(data)
            while self._tail_len > self._tail_cap:
                first = self._tail[0]
                drop = min(len(first), self._tail_len - self._tail_cap)
                self._omitted += drop
                self._tail_len -= drop
                if drop == len(first):
                    self._tail.popleft()
                else:
                    self._tail[0] = first[drop:]

    def close(self) -> None:
        with self._lock:
            if self._f.closed:
                return
            if self._omitted:
                self._f.write(f"\n[... {self._omitted} bytes omitted ...]\n".encode())
            self._f.write(b"".join(self._tail))
            self._f.close()

    def __enter__(self) -> CappedLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- IO --------------------------------------------------------------------------------------


def new_run_dir(parent: Path, prefix: str) -> Path:
    """Create and return a fresh per-run dir <parent>/<prefix>-<YYYYmmdd-HHMMSS>-<random>.
    Atomic (tempfile.mkdtemp): unique even for concurrent runs in the same second, created
    exactly once, by us. Its name is also a valid container-name suffix."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-{stamp}-", dir=parent))


def _docker(
    args: Sequence[str], timeout_s: float = DOCKER_CMD_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), capture_output=True, text=True, timeout=timeout_s, check=False
    )


def image_exists(tag: str) -> bool:
    return _docker(["docker", "image", "inspect", tag]).returncode == 0


def image_size_bytes(tag: str) -> int | None:
    r = _docker(["docker", "image", "inspect", "-f", "{{.Size}}", tag])
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None


def _stream(
    args: Sequence[str], log: CappedLog, timeout_s: float, stdin_data: bytes | None = None
) -> tuple[int | None, bool, subprocess.Popen[bytes], threading.Thread]:
    """Run args, stream stdout+stderr into log. Returns (returncode, timed_out, proc, reader);
    on timeout the process is still running: the caller stops it, then joins reader."""
    proc = subprocess.Popen(
        list(args),
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def pump() -> None:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):  # type: ignore[union-attr]
            log.write(chunk)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    if stdin_data is not None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(stdin_data)
        finally:
            proc.stdin.close()
    try:
        rc: int | None = proc.wait(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True
    if not timed_out:
        reader.join()
    return rc, timed_out, proc, reader


def build_image(dockerfile_text: str, tag: str, *, timeout_s: float, log_path: Path) -> BuildResult:
    """Build tag from the Dockerfile text (stdin, no context). Skipped if the tag exists.

    Timeout: the `docker build` client is killed (SIGKILL). With BuildKit this cancels the
    build: the daemon ends the build when the client session closes, so the RUN step's
    processes stop and no image is tagged. Verified on Docker 29.4.3 / BuildKit v0.29.0
    (test_docker_build_timeout_cancels_buildkit), so no explicit cancel call is made."""
    start = time.monotonic()
    if image_exists(tag):
        return BuildResult(ok=True, tag=tag, cached=True, duration_s=0.0, log_path=None)
    with CappedLog(log_path) as log:
        rc, timed_out, proc, reader = _stream(
            build_args(tag), log, timeout_s, stdin_data=dockerfile_text.encode("utf-8")
        )
        if timed_out:
            proc.kill()
            proc.wait()
            reader.join(timeout=5)
            log.write(f"\n[reproducer] build timed out after {timeout_s:.0f}s\n".encode())
    return BuildResult(
        ok=rc == 0 and not timed_out,
        tag=tag,
        cached=False,
        duration_s=time.monotonic() - start,
        log_path=Path(log_path),
        timed_out=timed_out,
    )


def prepare_workdir(
    clone_dir: Path, patches: Sequence[str], dest: Path, prep_dirs: Sequence[str] = ()
) -> Path:
    """Fresh copy of clone_dir (without .git) at dest, then apply each patch in order with
    `git apply` inside dest, then create prep_dirs (known fixes) inside dest. dest must not
    exist. The clone is never modified. Symlinks are copied as links (never followed). Raises
    PatchError naming the failing patch."""
    clone_dir, dest = Path(clone_dir), Path(dest)
    if not clone_dir.is_dir():
        raise FileNotFoundError(f"clone dir {clone_dir} does not exist")
    if dest.exists():
        raise FileExistsError(f"workdir {dest} already exists; use a fresh path per run")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(clone_dir, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    env = dict(os.environ)
    # dest may sit inside another git repo (e.g. this project): stop git from discovering it,
    # otherwise paths would be resolved against that repo's root.
    env["GIT_CEILING_DIRECTORIES"] = str(dest.parent.resolve())
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    for i, patch in enumerate(patches, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
            f.write(patch if patch.endswith("\n") else patch + "\n")
            patch_file = f.name
        try:
            r = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", patch_file],
                cwd=dest,
                env=env,
                capture_output=True,
                text=True,
                timeout=DOCKER_CMD_TIMEOUT_S,
                check=False,
            )
        finally:
            os.unlink(patch_file)
        if r.returncode != 0:
            raise PatchError(f"patch {i}/{len(patches)} did not apply: {r.stderr.strip()[:2000]}")
    for d in prep_dirs:
        make_prep_dir(dest, d)
    return dest


def make_prep_dir(dest: Path, rel: str) -> Path:
    """Create repo-relative dir `rel` inside the per-run copy dest (known fix prep_dir).
    Refuses absolute paths, '..', non-normalized paths and symlinks on the way."""
    if (
        not rel
        or rel.startswith("/")
        or posixpath.normpath(rel) != rel
        or rel == "."
        or rel.split("/")[0] == ".."
    ):
        raise ValueError(f"prep dir must be a normalized repo-relative path, got {rel!r}")
    p = Path(dest)
    for part in rel.split("/"):
        p = p / part
        if p.is_symlink():
            raise ValueError(f"prep dir {rel!r}: {p} is a symlink")
        if p.exists() and not p.is_dir():
            raise ValueError(f"prep dir {rel!r}: {p} is not a directory")
        p.mkdir(exist_ok=True)
    return p


def _container_state(name: str) -> tuple[int | None, bool]:
    r = _docker(["docker", "inspect", "-f", "{{.State.ExitCode}} {{.State.OOMKilled}}", name])
    if r.returncode != 0:
        return None, False
    code, oom = r.stdout.split()
    return int(code), oom == "true"


def _make_mountpoints(workdir: Path, targets: Iterable[str]) -> None:
    """Create mountpoints of data mounts nested under /work in the working copy, as the host
    user (otherwise dockerd creates them root-owned). Refuses symlinks on the way (a repo
    symlink must not redirect a mount) and non-directories at the target."""
    for target in targets:
        if not target.startswith(WORK_MOUNT + "/"):
            continue
        p = workdir
        for part in target[len(WORK_MOUNT) + 1 :].split("/"):
            p = p / part
            if p.is_symlink():
                raise ValueError(f"data mount target {target!r}: {p} is a symlink")
            if p.exists() and not p.is_dir():
                raise ValueError(f"data mount target {target!r}: {p} is not a directory")
            p.mkdir(exist_ok=True)


def run_container(
    image: str,
    cmd: list[str],
    *,
    workdir: Path,
    out_dir: Path,
    data_mounts: DataMounts = {},  # noqa: B006 - read-only mapping
    gpu: bool,
    timeout_s: float,
    mem: str = "12g",
    cpus: float | None = None,
    pids: int = 512,
    log_path: Path,
    name: str,
) -> RunResult:
    """Run cmd in a locked-down container (run_args). On timeout: docker kill. OOM state is
    read with docker inspect before the container is removed; it is always removed."""
    workdir, out_dir = Path(workdir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if cpus is None:
        cpus = float(max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = run_args(
        image,
        cmd,
        name=name,
        workdir=str(workdir),
        out_dir=str(out_dir),
        data_mounts=[(str(Path(h).resolve()), c) for h, c in mount_pairs(data_mounts)],
        gpu=gpu,
        uid=os.getuid(),
        gid=os.getgid(),
        mem=mem,
        cpus=cpus,
        pids=pids,
    )
    _make_mountpoints(workdir, [c for _, c in mount_pairs(data_mounts)])
    start = time.monotonic()
    timed_out = False
    try:
        with CappedLog(log_path) as log:
            _, timed_out, proc, reader = _stream(args, log, timeout_s)
            if timed_out:
                _docker(["docker", "kill", name])
                try:
                    proc.wait(timeout=DOCKER_CMD_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                reader.join(timeout=5)
                log.write(f"\n[reproducer] killed after {timeout_s:.0f}s timeout\n".encode())
        duration = time.monotonic() - start
        exit_code, oom = _container_state(name)
    finally:
        _docker(["docker", "rm", "-f", name])
    return RunResult(
        exit_code=exit_code,
        timed_out=timed_out,
        oom_killed=oom,
        duration_s=duration,
        log_path=Path(log_path),
        name=name,
    )
