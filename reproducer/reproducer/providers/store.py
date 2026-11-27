"""Append-only JSONL run store (the only file IO for evidence and unknown events).

One folder per paper holding:
  evidence.jsonl        one canonical Evidence per line, deduplicated by evidence_id on write
  unknown_events.jsonl  one UnknownOpened / UnknownResolved / Assumption per line, in order
  llm_calls.jsonl       one LLM CallRecord per line (telemetry, not evidence; no dedupe)

Files are never rewritten or truncated; there is no update/delete API. The ledger is not
stored: it is always recomputed from these two files.

Concurrency: every append holds an exclusive fcntl.flock on the target file, and is
flushed + fsynced before the lock is released. Readers take a shared lock. If another
writer (another RunStore instance or process) grew evidence.jsonl since we last looked, the
in-memory id set is rebuilt from disk under the lock before the dedupe check.

Paths stored in evidence (log_ref, ...) are relative to the store root (the paper out dir),
so a store stays valid when moved: write them with make_ref, read them with resolve_ref.
"""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, TypeVar

from reproducer.domain.evidence import (
    Evidence,
    canonical_json,
    evidence_from_jsonl_line,
    to_jsonl_line,
)
from reproducer.domain.telemetry import CallRecord
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    event_from_jsonl_line,
)

EVIDENCE_FILE = "evidence.jsonl"
EVENTS_FILE = "unknown_events.jsonl"
LLM_CALLS_FILE = "llm_calls.jsonl"  # telemetry (CallRecord), not evidence; no dedupe
RUN_LOCK_FILE = "run.lock"  # held by `run` for its whole duration (one run per store)

UnknownEventT = UnknownOpened | UnknownResolved | Assumption
T = TypeVar("T")


def make_ref(out_dir: Path, path: Path) -> str:
    """`path` as a ref relative to the store root `out_dir` (posix). Both sides are resolved
    first, so a relative out dir and an absolute path still match. A path outside the store
    falls back to its absolute form."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(out_dir).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_ref(out_dir: Path, ref: str, *, project_root: Path | None = None) -> Path:
    """The file a stored ref points to. New refs are relative to the store root `out_dir`;
    older ones may be relative to the project root (the cwd by default) or absolute. Tries
    the store root first, then the project root; if neither exists, the store-root form."""
    p = Path(ref)
    if p.is_absolute():
        return p
    in_store = Path(out_dir) / p
    if in_store.exists():
        return in_store
    in_project = (Path.cwd() if project_root is None else Path(project_root)) / p
    return in_project if in_project.exists() else in_store


@contextmanager
def _flocked(f: BinaryIO, mode: int) -> Iterator[None]:
    fcntl.flock(f.fileno(), mode)
    try:
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _parse_lines(path: Path, raw: bytes, parse: Callable[[str], T]) -> list[T]:
    """Parse every non-blank line; trailing blank lines are allowed, interior ones are not.

    Any failure raises ValueError naming the file and 1-based line number.
    """
    lines = raw.split(b"\n")
    while lines and not lines[-1].strip():
        lines.pop()
    out: list[T] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}:{lineno}: blank line inside JSONL file")
        try:
            out.append(parse(line.decode("utf-8")))
        except ValueError as e:  # includes pydantic ValidationError and UnicodeDecodeError
            raise ValueError(f"{path}:{lineno}: invalid record: {e}") from e
    return out


def _read_locked(path: Path) -> bytes:
    with open(path, "rb") as f, _flocked(f, fcntl.LOCK_SH):
        return f.read()


def _check_tail(f: BinaryIO, path: Path, size: int) -> None:
    """Refuse to append after a torn (newline-less) last line; never repair it ourselves."""
    if size == 0:
        return
    f.seek(size - 1)
    if f.read(1) != b"\n":
        raise ValueError(f"{path}: last line is not newline-terminated (torn write?)")


def _write_durable(f: BinaryIO, data: bytes) -> None:
    f.write(data)
    f.flush()
    os.fsync(f.fileno())


class RunLocked(RuntimeError):
    """Another process holds <out>/run.lock."""


@contextmanager
def run_lock(root: Path) -> Iterator[Path]:
    """Exclusive, non-blocking fcntl.flock on <root>/run.lock for the whole block. Raises
    RunLocked at once if another process holds it. The holder's pid is written into the file
    (informational); the file is never deleted (the lock, not the file, is the signal) and the
    kernel drops the lock if the holder dies."""
    path = Path(root) / RUN_LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            holder = f.read().decode("utf-8", errors="replace").strip() or "?"
            raise RunLocked(
                f"another `run` is in progress on {root} (pid {holder}, {path})"
            ) from None
        f.truncate(0)
        _write_durable(f, f"{os.getpid()}\n".encode())
        try:
            yield path
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def lock_holder(root: Path) -> str | None:
    """Read-only probe of <root>/run.lock: the holder's pid text ("?" if unreadable) while a
    process holds the lock, else None. Never creates or writes the file; a shared lock is
    taken for an instant, which conflicts only with a holder's exclusive one."""
    path = Path(root) / RUN_LOCK_FILE
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return None
    with f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return f.read().decode("utf-8", errors="replace").strip() or "?"
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return None


class NotARunStore(FileNotFoundError):
    """RunStore(root, create=False) on a folder without evidence.jsonl."""


class RunStore:
    def __init__(self, root: Path, *, create: bool = True) -> None:
        """create=False (read-only commands): nothing is created; a folder without
        evidence.jsonl raises NotARunStore, and missing events / call files read as empty."""
        self.root = Path(root)
        self.evidence_path = self.root / EVIDENCE_FILE
        self.events_path = self.root / EVENTS_FILE
        self.llm_calls_path = self.root / LLM_CALLS_FILE
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            for p in (self.evidence_path, self.events_path, self.llm_calls_path):
                open(p, "ab").close()  # create if missing; never truncates
        elif not self.evidence_path.is_file():
            raise NotARunStore(f"{self.root}: not a run store (no {EVIDENCE_FILE})")

        self._lock = threading.Lock()
        self._ids: set[str] = set()
        self._evidence_size = 0
        with self._lock:
            self._reload_ids_unlocked(_read_locked(self.evidence_path))

    # --- writes --------------------------------------------------------------------------

    def append_evidence(self, ev: Evidence) -> bool:
        """Append ev unless its evidence_id is already stored. Returns True if written."""
        data = (to_jsonl_line(ev) + "\n").encode("utf-8")
        with self._lock:
            if ev.evidence_id in self._ids:
                return False
            with open(self.evidence_path, "a+b") as f, _flocked(f, fcntl.LOCK_EX):
                size = os.fstat(f.fileno()).st_size
                if size != self._evidence_size:  # someone else appended: resync ids
                    f.seek(0)
                    self._reload_ids_unlocked(f.read())
                    if ev.evidence_id in self._ids:
                        return False
                _check_tail(f, self.evidence_path, size)
                _write_durable(f, data)
                self._evidence_size = size + len(data)
                self._ids.add(ev.evidence_id)
                return True

    def append_event(self, event: UnknownEventT) -> None:
        """Append one unknown event (no dedupe; fold_unknowns handles repeated Opened)."""
        data = (to_jsonl_line(event) + "\n").encode("utf-8")
        with open(self.events_path, "a+b") as f, _flocked(f, fcntl.LOCK_EX):
            _check_tail(f, self.events_path, os.fstat(f.fileno()).st_size)
            _write_durable(f, data)

    def append_call_record(self, record: CallRecord) -> None:
        """Append one LLM CallRecord (telemetry; prompt is stored only as its sha256)."""
        line = (
            to_jsonl_line(record)
            if record.evicted
            else canonical_json(record.model_dump(mode="json", exclude={"evicted"}))
        )
        data = (line + "\n").encode("utf-8")
        with open(self.llm_calls_path, "a+b") as f, _flocked(f, fcntl.LOCK_EX):
            _check_tail(f, self.llm_calls_path, os.fstat(f.fileno()).st_size)
            _write_durable(f, data)

    # --- reads ---------------------------------------------------------------------------

    def load_evidence(self) -> dict[str, Evidence]:
        """All stored evidence by id, in file order (first occurrence wins)."""
        records = _parse_lines(
            self.evidence_path, _read_locked(self.evidence_path), evidence_from_jsonl_line
        )
        out: dict[str, Evidence] = {}
        for ev in records:
            out.setdefault(ev.evidence_id, ev)
        return out

    def load_events(self) -> tuple[UnknownEventT, ...]:
        if not self.events_path.exists():  # only possible with create=False
            return ()
        return tuple(
            _parse_lines(self.events_path, _read_locked(self.events_path), event_from_jsonl_line)
        )

    def load_call_records(self) -> tuple[CallRecord, ...]:
        if not self.llm_calls_path.exists():  # only possible with create=False
            return ()
        return tuple(
            _parse_lines(
                self.llm_calls_path,
                _read_locked(self.llm_calls_path),
                CallRecord.model_validate_json,
            )
        )

    # --- internals -----------------------------------------------------------------------

    def _reload_ids_unlocked(self, raw: bytes) -> None:
        records = _parse_lines(self.evidence_path, raw, evidence_from_jsonl_line)
        self._ids = {ev.evidence_id for ev in records}
        self._evidence_size = len(raw)
