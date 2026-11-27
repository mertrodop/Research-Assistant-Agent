"""Tests for `python -m reproducer.cli eval-report` (read-only on every store)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from domain.eval_fixture import full_store

from reproducer.cli import load_manifest, main
from reproducer.domain.contracts import ReproSpec
from reproducer.domain.evidence import make_evidence
from reproducer.providers.store import RunStore


def _write_store(out: Path) -> None:
    evidence, events, calls = full_store()
    store = RunStore(out)
    for ev in evidence.values():
        store.append_evidence(ev)
    for e in events:
        store.append_event(e)
    for c in calls:
        store.append_call_record(c)


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    _write_store(runs / "paper")
    noclaims = RunStore(runs / "noclaims")
    noclaims.append_evidence(make_evidence(
        kind="readme", source="repo:x@1", locator="README.md", content={"text": "hi"},
        confidence="observed", authority="repo",
    ))  # fmt: skip
    cases = [
        {"name": "paper", "spec": "evals/paper/spec.json", "out": str(runs / "paper"),
         "note": "fake"},
        {"name": "mixup", "spec": "evals/mixup/spec.json", "out": str(runs / "mixup"),
         "note": "placeholder"},
        {"name": "noclaims", "spec": "", "out": str(runs / "noclaims"), "note": ""},
    ]  # fmt: skip
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"cases": cases}))
    return path


def test_eval_report_is_read_only_and_deterministic(
    manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    before = _tree(runs)
    out = tmp_path / "report"
    assert main(["eval-report", "--manifest", str(manifest), "--out", str(out)]) == 0
    first_md = (out / "report.md").read_text()
    first_json = (out / "report.json").read_text()
    assert main(["eval-report", "--manifest", str(manifest), "--out", str(out)]) == 0
    assert _tree(runs) == before  # checksums of every store file unchanged, nothing added
    assert not (runs / "mixup").exists()
    assert (out / "report.md").read_text() == first_md
    assert (out / "report.json").read_text() == first_json
    assert first_md in capsys.readouterr().out

    data = json.loads(first_json)
    by = {p["name"]: p for p in data["papers"]}
    assert by["paper"]["status"] == "ok"
    assert by["mixup"]["status"] == "not run"
    assert "no store" in by["mixup"]["reason"]
    assert (by["noclaims"]["status"], by["noclaims"]["reason"]) == (
        "not run", "no paper_claim evidence")  # fmt: skip
    assert data["totals"]["papers_run"] == 1 and data["totals"]["papers_not_run"] == 2
    assert by["paper"]["trajectory"]["known_fixes"] == 3
    for section in ("## Totals", "## paper", "### Outcome", "### Trajectory", "### Efficiency"):
        assert section in first_md


def test_eval_report_bad_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "m.json"
    bad.write_text("{}")
    assert main(["eval-report", "--manifest", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def test_manifest_cases_and_synthetic_toy_critic_spec() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = {c["name"]: c for c in load_manifest(root / "evals" / "manifest.json")}
    assert cases["toy_critic"]["out"] == "runs/toy_critic"
    spec_path = root / cases["toy_critic"]["spec"]
    spec = ReproSpec.model_validate_json(spec_path.read_text())
    assert [c.claim_id for c in spec.claims] == ["toyC_err"]
    assert all("SYNTHETIC" in c.setting and "SYNTHETIC" in c.location for c in spec.claims)
    assert "SYNTHETIC" in (spec_path.parent / "README.md").read_text()
    train = (spec_path.parent / "toyrepo" / "train.py").read_text()
    assert 'stats["correct"]' in train or "stats['correct']" in train  # the built-in bug
    assert "num_correct" in (spec_path.parent / "toyrepo" / "utils" / "metrics.py").read_text()
