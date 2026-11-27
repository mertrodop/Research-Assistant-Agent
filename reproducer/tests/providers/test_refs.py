"""Tests for store refs: make_ref (new refs, relative to the paper out dir) and resolve_ref
(reads both the out-dir-relative and the older project-root-relative form)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reproducer.providers.store import make_ref, resolve_ref


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("log\n")
    return p


def test_make_ref_is_out_dir_relative_for_relative_and_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path("runs/paper")  # relative out dir (as the CLI gets it)
    log = _touch(out / "runs" / "run-1" / "run.log")
    assert make_ref(out, log) == "runs/run-1/run.log"
    assert make_ref(out, log.resolve()) == "runs/run-1/run.log"  # absolute (mkdtemp, 3.12+)
    assert make_ref(out.resolve(), log) == "runs/run-1/run.log"
    assert make_ref(out, out / "smoke" / "s-1" / "run.log") == "smoke/s-1/run.log"


def test_make_ref_outside_the_store_is_absolute(tmp_path: Path) -> None:
    other = _touch(tmp_path / "elsewhere" / "run.log")
    assert make_ref(tmp_path / "out", other) == str(other.resolve())


def test_resolve_ref_reads_the_out_dir_relative_form(tmp_path: Path) -> None:
    out = tmp_path / "out"
    log = _touch(out / "runs" / "run-1" / "run.log")
    assert resolve_ref(out, "runs/run-1/run.log", project_root=tmp_path) == log


def test_resolve_ref_reads_the_project_root_relative_form(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "paper"
    log = _touch(out / "smoke" / "s-1" / "run.log")
    ref = "runs/paper/smoke/s-1/run.log"  # older evidence: relative to the project root
    assert resolve_ref(out, ref, project_root=tmp_path) == tmp_path / ref
    assert resolve_ref(out, ref, project_root=tmp_path).resolve() == log.resolve()


def test_resolve_ref_project_root_defaults_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "runs" / "paper" / "smoke" / "s-1" / "run.log")
    got = resolve_ref(Path("runs/paper"), "runs/paper/smoke/s-1/run.log")
    assert got.is_file()


def test_resolve_ref_prefers_out_dir_and_falls_back_to_it(tmp_path: Path) -> None:
    out = tmp_path / "out"
    both = "x/run.log"
    in_out = _touch(out / both)
    _touch(tmp_path / both)
    assert resolve_ref(out, both, project_root=tmp_path) == in_out  # out dir first
    # neither exists: the (new) out-dir form
    assert resolve_ref(out, "missing/run.log", project_root=tmp_path) == out / "missing/run.log"


def test_resolve_ref_absolute_ref_is_returned_as_is(tmp_path: Path) -> None:
    log = _touch(tmp_path / "abs" / "run.log")
    assert resolve_ref(tmp_path / "out", str(log)) == log
