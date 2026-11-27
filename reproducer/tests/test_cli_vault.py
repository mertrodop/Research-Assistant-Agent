"""Tests for `python -m reproducer.cli vault` (read-only on the store)."""

from __future__ import annotations

from pathlib import Path

import pytest
from domain.vault_fixture import C10, cutout_store, run_id

from reproducer.cli import main, run_files, vault_root
from reproducer.providers.store import EVENTS_FILE, EVIDENCE_FILE, LLM_CALLS_FILE, RunStore


@pytest.fixture
def out(tmp_path: Path) -> Path:
    evidence, events, calls = cutout_store()
    out = tmp_path / "runs" / "cutout"
    store = RunStore(out)
    for ev in evidence.values():
        store.append_evidence(ev)
    for e in events:
        store.append_event(e)
    for c in calls:
        store.append_call_record(c)
    run_dir = out / "runs" / run_id(C10)
    (run_dir / "artifacts" / "logs").mkdir(parents=True)
    (run_dir / "run.log").write_text("log\n")
    (run_dir / "artifacts" / "logs" / "cifar10_resnet18.csv").write_text("epoch,acc\n")
    return out


def _store_bytes(out: Path) -> dict[str, bytes]:
    return {n: (out / n).read_bytes() for n in (EVIDENCE_FILE, EVENTS_FILE, LLM_CALLS_FILE)}


def test_vault_writes_then_is_idempotent(
    out: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REPRODUCER_VAULT", raising=False)
    before = _store_bytes(out)
    assert main(["vault", "--out", str(out)]) == 0
    first = capsys.readouterr().out
    root = out / "vault_out"
    assert "created:        10" in first and "updated:        0" in first
    assert f"paper note:     {root / 'Papers/cutout.md'}" in first
    run_note = (root / "Runs" / f"{run_id(C10)}.md").read_text()
    assert f"`runs/{run_id(C10)}/artifacts/logs/cifar10_resnet18.csv`" in run_note
    assert "run.log`" in run_note and run_note.count("run.log") == 1  # log not an artifact

    assert main(["vault", "--out", str(out)]) == 0
    second = capsys.readouterr().out
    assert "created:        0" in second and "updated:        0" in second
    assert "unchanged:      10" in second
    assert _store_bytes(out) == before  # store untouched


def test_vault_root_resolution(tmp_path: Path) -> None:
    out = tmp_path / "o"
    assert vault_root(out, None, {}) == out / "vault_out"
    assert vault_root(out, None, {"REPRODUCER_VAULT": "/v"}) == Path("/v")
    assert vault_root(out, "/x", {"REPRODUCER_VAULT": "/v"}) == Path("/x")


def test_vault_env_and_flag(out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPRODUCER_VAULT", str(tmp_path / "shared"))
    assert main(["vault", "--out", str(out)]) == 0
    assert (tmp_path / "shared" / "Papers" / "cutout.md").is_file()
    assert main(["vault", "--out", str(out), "--vault", str(tmp_path / "flag")]) == 0
    assert (tmp_path / "flag" / "Papers" / "cutout.md").is_file()


def test_run_files_only_lists_runs_without_stored_artifacts(out: Path) -> None:
    evidence = RunStore(out, create=False).load_evidence()
    (out / "runs" / "run-c100plus_resnet18_cutout-20261124-1").mkdir(parents=True)
    (out / "runs" / "run-c100plus_resnet18_cutout-20261124-1" / "x.csv").write_text("x")
    listed = run_files(out, evidence)
    assert list(listed) == [run_id(C10)]  # the C100 record carries "artifacts"


@pytest.mark.parametrize("cmd", ["vault", "ledger"])
def test_read_only_commands_without_store(
    tmp_path: Path, cmd: str, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing"
    assert main([cmd, "--out", str(missing)]) == 2
    assert "not a run store" in capsys.readouterr().err
    assert not missing.exists()
    missing.mkdir()
    assert main([cmd, "--out", str(missing)]) == 2
    assert list(missing.iterdir()) == []  # nothing created


def test_ledger_does_not_create_missing_store_files(out: Path) -> None:
    (out / LLM_CALLS_FILE).unlink()
    assert main(["ledger", "--out", str(out)]) == 0
    assert not (out / LLM_CALLS_FILE).exists()
