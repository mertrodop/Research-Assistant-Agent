"""Vault provider: writes rendered notes into an Obsidian vault (the only vault IO).

Notes are derived views of the store, so writing is conservative:
  - every note is validated (safe path, intact managed block) before any is written; one
    bad note means nothing is written;
  - a missing file is created;
  - an existing file keeps all text outside the managed block (merge_note); a file without
    markers gets a managed block appended, user text is never overwritten;
  - a file whose content would not change is not written (mtime untouched);
  - every write is atomic (temp file in the same directory + fsync + os.replace);
  - notes are never deleted, even when the store no longer renders them.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from reproducer.domain.evidence import Evidence
from reproducer.domain.vault_render import merge_note
from reproducer.providers.store import make_ref, resolve_ref

DEFAULT_MODE = 0o644
MAX_RUN_FILES = 50  # artifact refs listed per run note (fallback listing)


@dataclass(frozen=True)
class WriteCounts:
    created: tuple[str, ...] = field(default_factory=tuple)
    updated: tuple[str, ...] = field(default_factory=tuple)
    unchanged: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> int:
        return len(self.created) + len(self.updated)

    @property
    def total(self) -> int:
        return self.changed + len(self.unchanged)


def note_target(vault_root: Path, rel: str) -> Path:
    """vault_root / rel; rel must be a relative posix path that stays inside the vault."""
    p = PurePosixPath(rel)
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError(f"unsafe note path {rel!r}")
    return Path(vault_root).joinpath(*p.parts)


def atomic_write(path: Path, data: str) -> None:
    """Write `data` (UTF-8) to `path` via a temp file in the same directory + os.replace, so
    readers (Obsidian, sync tools) see either the old or the new file, never a partial one.
    Keeps the mode of an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else DEFAULT_MODE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_notes(vault_root: Path, notes: Mapping[str, str]) -> WriteCounts:
    """Write rendered notes {relative_path: markdown} under vault_root (see module doc).
    Every note is validated first (safe path, intact managed block) and merged in memory;
    on any error a ValueError lists every bad note and nothing is written."""
    root = Path(vault_root)
    plan: list[tuple[str, Path, str | None, str]] = []
    errors: list[str] = []
    for rel in sorted(notes):
        try:
            path = note_target(root, rel)
            existing = path.read_text(encoding="utf-8") if path.exists() else None
            plan.append((rel, path, existing, merge_note(existing, notes[rel])))
        except ValueError as e:
            errors.append(f"{rel}: {e}")
    if errors:
        raise ValueError("nothing written; invalid notes:\n  " + "\n  ".join(errors))

    created: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    for rel, path, existing, merged in plan:
        if existing is None:
            atomic_write(path, merged)
            created.append(rel)
        elif merged != existing:
            atomic_write(path, merged)
            updated.append(rel)
        else:
            unchanged.append(rel)
    return WriteCounts(tuple(created), tuple(updated), tuple(unchanged))


def run_files(out_dir: Path, evidence: Mapping[str, Evidence]) -> dict[str, list[str]]:
    """Fallback for older run_results without an "artifacts" content key: run_id ->
    store-relative refs of the files in its run dir (the folder of its log), log excluded,
    sorted, at most MAX_RUN_FILES. Read-only."""
    out: dict[str, list[str]] = {}
    for ev in evidence.values():
        log_ref = ev.content.get("log_ref") if ev.kind == "run_result" else None
        if not log_ref or "artifacts" in ev.content:
            continue
        log = resolve_ref(out_dir, str(log_ref))
        run_dir = log.parent
        if not run_dir.is_dir():
            continue
        files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p != log)
        out[str(ev.content.get("run_id", ev.locator))] = [
            make_ref(out_dir, p) for p in files[:MAX_RUN_FILES]
        ]
    return out
