"""Dataset provider: provision registry datasets into a shared host cache with a TRUSTED container.

Repo code runs with --network none, so datasets are fetched beforehand, here, and later
mounted read-only where the repo expects them. Trust boundary:
  - only names in domain.data_plan.KNOWN_DATASETS; the container runs OUR fixed script
    (TRUSTED_SCRIPT, `python -I -c`), whose only input is a JSON payload built from the
    registry (class names + split kwargs) — no repo code, no repo strings;
  - same image as the runs (so the torchvision that reads the data is the one that wrote it);
    no repo mount at all, only the cache dir at /data;
  - check: --network none, /data read-only; torchvision's constructors with download=False
    verify the files (CIFAR/SVHN/STL10: md5 of every file; MNIST: raw files present);
  - download: --network bridge (the only networked container we start), /data rw, then a
    second network-none check; torchvision checks the archive md5 after downloading;
  - otherwise locked down like docker_runner.run_args: host uid:gid, cap-drop ALL,
    no-new-privileges, read-only root, tmpfs /tmp, memory/cpu/pids limits, always removed.
Container output is logged to a capped file per step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from reproducer.domain.data_plan import KNOWN_DATASETS, DatasetEntry
from reproducer.providers.docker_runner import (
    DOCKER_CMD_TIMEOUT_S,
    TMPFS_SIZE,
    CappedLog,
    _container_state,
    _docker,
    _stream,
    new_run_dir,
)

DATA_CACHE_ENV = "REPRODUCER_DATA_CACHE"
CONTAINER_ROOT = "/data"
RESULT_MARKER = "REPRODUCER_RESULT "
_RESULT_RE = re.compile(r"^REPRODUCER_RESULT (\{.*\})\s*$", re.MULTILINE)

# Runs inside the image's python (may be 3.7): keep it 3.7-compatible.
TRUSTED_SCRIPT = """\
import json, sys
import torchvision
import torchvision.datasets as D
req = json.loads(sys.argv[1])
res = {}
for item in req["items"]:
    try:
        cls = getattr(D, item["cls"])
        sizes = [len(cls(req["root"], download=req["download"], **kw)) for kw in item["splits"]]
        res[item["name"]] = {"ok": True, "sizes": sizes}
    except Exception as e:
        res[item["name"]] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
out = {"torchvision": torchvision.__version__, "results": res}
sys.stdout.flush()
print("\\nREPRODUCER_RESULT " + json.dumps(out))
"""

Mode = Literal["check", "download"]


def data_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """$REPRODUCER_DATA_CACHE, else ~/.cache/reproducer/data/torchvision (one shared root)."""
    env = os.environ if env is None else env
    override = env.get(DATA_CACHE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "reproducer" / "data" / "torchvision"


def trusted_payload(names: Sequence[str], *, download: bool) -> str:
    """JSON argument for TRUSTED_SCRIPT, built from the registry only."""
    items = []
    for n in names:
        entry = KNOWN_DATASETS[n]  # KeyError for anything outside the registry
        items.append({"name": entry.name, "cls": entry.cls, "splits": list(entry.splits)})
    return json.dumps(
        {"root": CONTAINER_ROOT, "download": download, "items": items}, sort_keys=True
    )


def trusted_args(
    image: str,
    *,
    mode: Mode,
    cache_dir: str,
    payload: str,
    name: str,
    uid: int,
    gid: int,
) -> list[str]:
    """`docker run` for the trusted script. check: no network, cache ro; download: bridge
    network, cache rw. No other mounts."""
    if not os.path.isabs(cache_dir) or ":" in cache_dir or "," in cache_dir:
        raise ValueError(f"cache dir must be absolute without ':' or ',', got {cache_dir!r}")
    network, access = ("bridge", "rw") if mode == "download" else ("none", "ro")
    return [
        "docker",
        "run",
        "--rm=false",
        "--name",
        name,
        "--network",
        network,
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,size={TMPFS_SIZE}",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "-v",
        f"{cache_dir}:{CONTAINER_ROOT}:{access}",
        "-w",
        "/tmp",
        image,
        "python",
        "-I",
        "-c",
        TRUSTED_SCRIPT,
        payload,
    ]


def parse_result(log_text: str) -> dict[str, Any] | None:
    """The last REPRODUCER_RESULT line of a trusted run, or None."""
    found = _RESULT_RE.findall(log_text)
    if not found:
        return None
    try:
        out = json.loads(found[-1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


@dataclass(frozen=True)
class TrustedRun:
    mode: Mode
    exit_code: int | None
    timed_out: bool
    log_path: Path
    results: dict[str, Any]  # name -> {"ok", "sizes"|"error"}
    torchvision: str | None


def run_trusted(
    image: str,
    names: Sequence[str],
    *,
    mode: Mode,
    cache_dir: Path,
    log_path: Path,
    timeout_s: float,
) -> TrustedRun:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)  # host-owned, never created by dockerd
    name = f"reproducer-data-{mode}-{uuid.uuid4().hex[:12]}"
    args = trusted_args(
        image,
        mode=mode,
        cache_dir=str(cache_dir),
        payload=trusted_payload(names, download=mode == "download"),
        name=name,
        uid=os.getuid(),
        gid=os.getgid(),
    )
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
        exit_code, _ = _container_state(name)
    finally:
        _docker(["docker", "rm", "-f", name])
    parsed = parse_result(Path(log_path).read_text(errors="replace")) or {}
    results = parsed.get("results") if isinstance(parsed.get("results"), dict) else {}
    return TrustedRun(
        mode=mode,
        exit_code=exit_code,
        timed_out=timed_out,
        log_path=Path(log_path),
        results=dict(results),
        torchvision=parsed.get("torchvision"),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_files(cache_dir: Path, entry: DatasetEntry) -> list[dict[str, Any]]:
    """[{path (relative to cache_dir), size, sha256}] for the dataset's own files, sorted.
    Symlinks are skipped (never followed)."""
    out: list[dict[str, Any]] = []
    for rel in entry.paths:
        p = cache_dir / rel
        if p.is_symlink() or not p.exists():
            continue
        files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        for f in files:
            if f.is_symlink():
                continue
            out.append(
                {
                    "path": f.relative_to(cache_dir).as_posix(),
                    "size": f.stat().st_size,
                    "sha256": _sha256(f),
                }
            )
    return sorted(out, key=lambda d: d["path"])


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    state: Literal["present", "downloaded", "failed"]
    verified: bool
    split_sizes: tuple[int, ...] = ()
    files: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    logs: tuple[Path, ...] = field(default_factory=tuple)


def ensure_datasets(
    names: Iterable[str],
    image: str,
    *,
    log_dir: Path,
    cache_dir: Path | None = None,
    timeout_s: float = 3600.0,
) -> dict[str, DatasetStatus]:
    """Per registry name: already present (network-none integrity check passes), else
    downloaded by the trusted container and checked again. Raises ValueError for names
    outside the registry (nothing is run then)."""
    wanted = list(dict.fromkeys(names))
    unknown = [n for n in wanted if n not in KNOWN_DATASETS]
    if unknown:
        raise ValueError(f"not in the dataset registry (never downloaded): {unknown}")
    if not wanted:
        return {}
    cache = (cache_dir or data_cache_dir()).resolve()
    log_dir = new_run_dir(Path(log_dir), "ensure")  # unique per call: logs never overwritten

    def check(ns: Sequence[str], tag: str) -> TrustedRun:
        return run_trusted(
            image, ns, mode="check", cache_dir=cache,
            log_path=log_dir / f"{tag}.log", timeout_s=min(timeout_s, 900.0),
        )  # fmt: skip

    first = check(wanted, "check")
    out: dict[str, DatasetStatus] = {}
    missing: list[str] = []
    for n in wanted:
        r = first.results.get(n, {})
        if r.get("ok"):
            out[n] = DatasetStatus(
                name=n, state="present", verified=True,
                split_sizes=tuple(r.get("sizes", ())), logs=(first.log_path,),
            )  # fmt: skip
        else:
            missing.append(n)
    if missing:
        dl = run_trusted(
            image, missing, mode="download", cache_dir=cache,
            log_path=log_dir / "download.log", timeout_s=timeout_s,
        )  # fmt: skip
        again = check(missing, "recheck")
        for n in missing:
            r = again.results.get(n, {})
            logs = (first.log_path, dl.log_path, again.log_path)
            if r.get("ok"):
                out[n] = DatasetStatus(
                    name=n, state="downloaded", verified=True,
                    split_sizes=tuple(r.get("sizes", ())), logs=logs,
                )  # fmt: skip
            else:
                err = dl.results.get(n, {}).get("error") or r.get("error")
                if dl.timed_out:
                    err = f"download timed out after {timeout_s:.0f}s"
                elif err is None:
                    err = f"trusted container exit code {dl.exit_code} (see log)"
                out[n] = DatasetStatus(
                    name=n, state="failed", verified=False, error=str(err), logs=logs
                )
    return {n: replace(out[n], files=tuple(cached_files(cache, KNOWN_DATASETS[n]))) for n in wanted}
