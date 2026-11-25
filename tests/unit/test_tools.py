import json

import pytest

from conftest import make_pdf
from ra_agent.config import Settings
from ra_agent.llm.client import NullVisionClient
from ra_agent.ocr.engine import NativeTextEngine
from ra_agent.schema import Route
from ra_agent.tools import TOOL_KINDS, Workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "paper.pdf")
    settings = Settings()
    settings.paths.raw_dir = str(raw)
    settings.paths.vault_dir = str(tmp_path / "vault")
    settings.paths.staging_dir = str(tmp_path / "staging")
    settings.paths.memory_dir = str(tmp_path / "memory")
    settings.pipeline.cache_dir = str(tmp_path / "cache")
    settings.crossref.cache_dir = str(tmp_path / "crossref")
    # no network: no external record, no references
    monkeypatch.setattr("ra_agent.ocr.synth.lookup", lambda meta, crossref: None)
    monkeypatch.setattr("ra_agent.ocr.pipeline.resolve_references", lambda *a, **k: [])
    monkeypatch.setattr("ra_agent.tools.resolve_references", lambda *a, **k: [])
    engines = {route: NativeTextEngine() for route in Route}
    return Workspace(settings, engines=engines, vision=NullVisionClient())


def small(result):
    assert all(len(json.dumps(v)) <= 2000 for v in result.values()), result
    assert len(result["summary"]) <= 400


def test_ingest_ok_unverified_metadata(ws):
    result = ws.ingest_paper("paper.pdf")
    assert result["ok"] and result["slug"] and result["problems"] == []
    assert result["metadata"] == "unverified"
    small(result)
    assert (ws.staging / "Papers" / f"{result['slug']}.md").is_file()


def test_ingest_missing_and_outside(ws, tmp_path):
    bad = ws.ingest_paper("missing.pdf")
    assert not bad["ok"] and "missing.pdf" in bad["error"]
    other = make_pdf(tmp_path / "other.pdf")
    assert not ws.ingest_paper(other)["ok"]
    assert not ws.ingest_paper("../other.pdf")["ok"]


def test_paper_must_be_slug(ws):
    for call in (ws.ingest_paper, ws.stage_note, ws.commit_note, ws.vault_read, ws.check_references):
        result = call("../../etc/passwd")
        assert not result["ok"] and "passwd" in result["error"]


def test_reprocess_stage_commit(ws, tmp_path):
    slug = ws.ingest_paper("paper.pdf")["slug"]
    page = ws.reprocess_page(slug, 1, "text_layer")
    assert page["ok"] and page["status"] in {"verified", "needs-review"}
    assert not ws.reprocess_page(slug, 9, "text_layer")["ok"]
    staged = ws.stage_note(slug)
    assert staged["ok"] and staged["diff"].startswith("<<<DATA")
    small(staged)
    # a second workspace restores the document from staging/<slug>.json
    again = Workspace(ws.settings, engines=ws.pipeline.engines, vision=NullVisionClient())
    assert again.stage_note(slug)["ok"]
    assert ws.commit_note(slug)["ok"]
    assert (tmp_path / "vault" / "Papers" / f"{slug}.md").is_file()
    assert ws.vault_search("pipeline")["ok"]
    read = ws.vault_read(slug)
    assert read["ok"] and "<<<DATA" in read["text"]
    small(read)


def test_describe_figure_without_vision(ws):
    slug = ws.ingest_paper("paper.pdf")["slug"]
    assert not ws.describe_figure(slug, 1)["ok"]  # fixture has no figures


def test_check_references_and_critique(ws):
    slug = ws.ingest_paper("paper.pdf")["slug"]
    assert ws.check_references(slug)["ok"]
    assert ws.critique_page(slug, 1)["ok"]


def test_save_procedure(ws, tmp_path):
    result = ws.save_procedure("Retry Docling twice!", "1. rerun\n2. compare")
    assert result["ok"]
    assert (tmp_path / "memory" / "procedures" / "retry-docling-twice.md").read_text().startswith("1. rerun")


def test_list_papers(ws):
    result = ws.list_papers()
    assert result["ok"] and result["papers"][0]["pdf"] == "paper.pdf"


def test_tool_kinds():
    assert TOOL_KINDS["commit_note"] == TOOL_KINDS["save_procedure"] == "write"
    assert set(TOOL_KINDS.values()) == {"read", "compute", "write"}
