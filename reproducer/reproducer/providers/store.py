"""Append-only JSONL run store (the only file IO for evidence and unknown events).

One folder per paper holding:
  evidence.jsonl        one canonical Evidence per line, deduplicated by evidence_id on write
  unknown_events.jsonl  one UnknownOpened / UnknownResolved / Assumption per line, in order

Files are never rewritten or truncated; there is no update/delete API. The ledger is not
stored: it is always recomputed from these two files.

Concurrency: every append holds an exclusive fcntl.flock on the target file, and is
flushed + fsynced before the lock is released. Readers take a shared lock. If another
writer (another RunStore instance or process) grew evidence.jsonl since we last looked, the
in-memory id set is rebuilt from disk under the lock before the dedupe check.
"""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, TypeVar

from reproducer.domain.evidence import Evidence, evidence_from_jsonl_line, to_jsonl_line
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    event_from_jsonl_line,
)

EVIDENCE_FILE = "evidence.jsonl"
EVENTS_FILE = "unknown_events.jsonl"

UnknownEventT = UnknownOpened | UnknownResolved | Assumption
T = TypeVar("T")


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


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.root / EVIDENCE_FILE
        self.events_path = self.root / EVENTS_FILE
        for p in (self.evidence_path, self.events_path):
            open(p, "ab").close()  # create if missing; never truncates

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
        return tuple(
            _parse_lines(self.events_path, _read_locked(self.events_path), event_from_jsonl_line)
        )

    # --- internals -----------------------------------------------------------------------

    def _reload_ids_unlocked(self, raw: bytes) -> None:
        records = _parse_lines(self.evidence_path, raw, evidence_from_jsonl_line)
        self._ids = {ev.evidence_id for ev in records}
        self._evidence_size = len(raw)
