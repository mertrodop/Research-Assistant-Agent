"""Tests for the ingest_spec node (graph/nodes/ingest_spec.py) and its failure routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.evidence import Evidence
from reproducer.domain.unknowns import UnknownOpened
from reproducer.graph.build import build_configure_graph
from reproducer.graph.nodes.ingest_spec import first_github_link, ingest_spec, paper_source_for
from reproducer.providers.llm import Budget
from reproducer.providers.store import EVENTS_FILE, EVIDENCE_FILE, RunStore

FIXTURE_SPEC = Path(__file__).resolve().parents[1] / "fixtures" / "repro_spec_example.json"


def _spec_file(tmp_path: Path, data: Any, name: str = "demo") -> Path:
    path = tmp_path / name / "spec.json"
    path.parent.mkdir(parents=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def _fixture(**changes: Any) -> dict[str, Any]:
    return {**json.loads(FIXTURE_SPEC.read_text()), **changes}


def test_no_spec_path_is_a_noop() -> None:
    assert ingest_spec({"repo_url": "https://github.com/a/b"}, {}) == {}


def test_valid_spec_gives_paper_evidence_and_unknowns(tmp_path: Path) -> None:
    path = _spec_file(tmp_path, _fixture(), name="resnet")
    out = ingest_spec({"spec_path": str(path)}, {})
    assert "status" not in out  # status stays running
    assert out["paper_source"] == "paper:resnet"
    assert out["repo_url"] == "https://github.com/example-lab/resnet18-cifar10"
    assert [c.claim_id for c in out["spec"].claims] == ["c1", "c2"]
    evidence: list[Evidence] = out["evidence"]
    assert all(ev.source == "paper:resnet" and ev.authority == "paper" for ev in evidence)
    assert sorted(ev.kind for ev in evidence) == ["paper_claim"] * 2 + ["paper_setup"] * 3
    (opened,) = out["events"]
    assert isinstance(opened, UnknownOpened)
    assert (opened.unknown_id, opened.claim_ids) == ("unk:weight_decay", ())
    setup = next(ev for ev in evidence if ev.evidence_id == opened.evidence_id)
    assert setup.kind == "paper_setup" and setup.content["value"] is None
    # deterministic: same spec -> same evidence ids
    again = ingest_spec({"spec_path": str(path)}, {})
    assert [ev.evidence_id for ev in again["evidence"]] == [ev.evidence_id for ev in evidence]


def test_first_github_link_skips_non_github() -> None:
    links = ("https://gitlab.com/a/b", "https://github.com/c/d", "https://github.com/e/f")
    assert first_github_link(links) == "https://github.com/c/d"
    assert first_github_link(("https://example.org/code.zip",)) is None


def test_paper_source_is_spec_folder_name() -> None:
    assert paper_source_for("evals/cutout/spec.json") == "paper:cutout"


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ("{not json", "invalid ReproSpec"),
        (_fixture(claims=[]), "invalid ReproSpec"),
        (_fixture(extra_field=1), "invalid ReproSpec"),
        (_fixture(code_links=["https://gitlab.com/a/b"]), "no GitHub repository"),
    ],
)
def test_bad_spec_fails_before_clone(tmp_path: Path, data: Any, reason: str) -> None:
    out = tmp_path / "runs" / "x"
    path = _spec_file(tmp_path, data)
    assert reason in ingest_spec({"spec_path": str(path)}, {})["failure_reason"]

    def no_llm(node: str) -> Any:
        raise AssertionError("no LLM call expected")

    graph = build_configure_graph({"store": RunStore(out), "llm_for": no_llm, "budget": Budget()})
    final = graph.invoke({"out_dir": str(out), "spec_path": str(path), "status": "running"})
    assert final["status"] == "configure_failed"
    assert reason in final["failure_reason"]
    assert "source" not in final and "repo_url" not in final  # configure never ran
    assert not (out / "repo").exists()  # nothing cloned
    assert (out / EVIDENCE_FILE).read_text() == (out / EVENTS_FILE).read_text() == ""


def test_missing_spec_file_fails(tmp_path: Path) -> None:
    out = ingest_spec({"spec_path": str(tmp_path / "nope" / "spec.json")}, {})
    assert out["status"] == "configure_failed"
    assert "cannot read spec" in out["failure_reason"]
