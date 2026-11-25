from __future__ import annotations

import fitz
import pytest

from ra_agent.config import Settings
from ra_agent.llm.client import NullVisionClient
from ra_agent.ocr.engine import NativeTextEngine
from ra_agent.schema import Route
from ra_agent.tools import Workspace

BODY = """Introduction

Retrieval augmented generation improves factual grounding in scientific writing.
We evaluate three baselines on four datasets and report accuracy in Table 1.

Methods

We combine a page router with a deterministic gate and a critic that only sees
failing pages. Every retry is capped by a circuit breaker of three attempts.

Results

The pipeline verified most pages on the first attempt and flagged the rest.

References

Tosun, E. (2024). Grounded writing assistants. Journal of Agents, 12(3), 1-20.
Tuygan, E. (2023). Citation validation for LLMs. In Proceedings of ACL, 100-110.
"""


def make_pdf(path, pages: int = 2, text: str = BODY, title: str = "Test Paper") -> str:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=10)
    doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return str(path)


def make_blank_pdf(path) -> str:
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(60, 60, 500, 700), color=(0, 0, 0), fill=(0.2, 0.2, 0.2))
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "paper.pdf")


@pytest.fixture
def blank_pdf(tmp_path):
    return make_blank_pdf(tmp_path / "scan.pdf")


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
