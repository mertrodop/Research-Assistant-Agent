"""Tests for reproducer.providers.vault: create / managed-block update / unchanged skip,
atomic writes, no deletes, unsafe paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reproducer.domain.vault_render import MANAGED_BEGIN, MANAGED_END
from reproducer.providers import vault
from reproducer.providers.vault import atomic_write, note_target, write_notes


def _note(body: str, status: str = "open") -> str:
    return f"---\ntype: claim\nstatus: {status}\n---\n\n{MANAGED_BEGIN}\n{body}\n{MANAGED_END}\n"


def test_create_then_unchanged(tmp_path: Path) -> None:
    notes = {"Claims/a.md": _note("A"), "Papers/p.md": _note("P")}
    c1 = write_notes(tmp_path, notes)
    assert (c1.created, c1.updated, c1.unchanged) == (("Claims/a.md", "Papers/p.md"), (), ())
    assert (tmp_path / "Claims/a.md").read_text() == notes["Claims/a.md"]
    mtime = (tmp_path / "Claims/a.md").stat().st_mtime_ns
    c2 = write_notes(tmp_path, notes)
    assert (c2.created, c2.updated, c2.changed) == ((), (), 0)
    assert c2.unchanged == ("Claims/a.md", "Papers/p.md")
    assert (tmp_path / "Claims/a.md").stat().st_mtime_ns == mtime  # not rewritten


def test_update_replaces_only_managed_block(tmp_path: Path) -> None:
    path = tmp_path / "Claims/a.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\ntype: claim\nstatus: open\nmine: 1\n---\nbefore\n{MANAGED_BEGIN}\nold\n"
        f"{MANAGED_END}\nafter\n"
    )
    counts = write_notes(tmp_path, {"Claims/a.md": _note("new", status="verified")})
    assert counts.updated == ("Claims/a.md",)
    assert path.read_text() == (
        f"---\ntype: claim\nstatus: verified\nmine: 1\n---\nbefore\n{MANAGED_BEGIN}\nnew\n"
        f"{MANAGED_END}\nafter\n"
    )


def test_missing_markers_append_block(tmp_path: Path) -> None:
    path = tmp_path / "Claims/a.md"
    path.parent.mkdir(parents=True)
    path.write_text("my own note\n")
    write_notes(tmp_path, {"Claims/a.md": _note("A")})
    out = path.read_text()
    assert "my own note\n" in out and out.endswith(f"{MANAGED_BEGIN}\nA\n{MANAGED_END}\n")


def test_damaged_block_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "Claims/a.md"
    path.parent.mkdir(parents=True)
    damaged = f"{MANAGED_BEGIN}\nuser kept typing\n"
    path.write_text(damaged)
    with pytest.raises(ValueError, match="damaged managed block"):
        write_notes(tmp_path, {"Claims/a.md": _note("A")})
    assert path.read_text() == damaged


def test_never_deletes(tmp_path: Path) -> None:
    write_notes(tmp_path, {"Claims/a.md": _note("A"), "Claims/b.md": _note("B")})
    (tmp_path / "Claims/user.md").write_text("mine")
    write_notes(tmp_path, {"Claims/a.md": _note("A2")})
    assert sorted(p.name for p in (tmp_path / "Claims").iterdir()) == ["a.md", "b.md", "user.md"]


def test_atomic_write_keeps_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "n.md"
    atomic_write(path, "old")
    os.chmod(path, 0o600)

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(vault.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write(path, "new")
    assert path.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["n.md"]  # temp file cleaned up
    monkeypatch.undo()
    atomic_write(path, "new")
    assert path.read_text() == "new" and (path.stat().st_mode & 0o777) == 0o600  # mode kept


@pytest.mark.parametrize("rel", ["../x.md", "/abs.md", "a/../../x.md", "", "."])
def test_unsafe_paths_rejected(tmp_path: Path, rel: str) -> None:
    with pytest.raises(ValueError, match="unsafe note path"):
        note_target(tmp_path, rel)


def test_one_damaged_note_means_nothing_is_written(tmp_path: Path) -> None:
    write_notes(tmp_path, {"Claims/a.md": _note("A")})
    bad = tmp_path / "Claims/z.md"
    bad.write_text(f"{MANAGED_BEGIN}\nno end\n")
    notes = {"Claims/a.md": _note("A2"), "Papers/p.md": _note("P"), "Claims/z.md": _note("Z")}
    with pytest.raises(ValueError, match="nothing written") as exc:
        write_notes(tmp_path, notes)
    assert "Claims/z.md" in str(exc.value)
    assert (tmp_path / "Claims/a.md").read_text() == _note("A")  # not updated
    assert not (tmp_path / "Papers").exists()  # not created


def test_unsafe_path_means_nothing_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe note path"):
        write_notes(tmp_path, {"Claims/a.md": _note("A"), "../escape.md": _note("X")})
    assert list(tmp_path.iterdir()) == []
