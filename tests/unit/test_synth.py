from __future__ import annotations

from ra_agent.config import MetaSettings
from ra_agent.ocr import synth
from ra_agent.schema import OcrDocument, PageResult, Reference, ReferenceCheck, Route

PAGE_ONE = """PaperBench: Evaluating AI's Ability to Replicate AI Research
Giulio Starace * Oliver Jaffe * Dane Sherburn
OpenAI
Abstract
We introduce PaperBench.
arXiv:2504.01848v3 [cs.AI] 7 Apr 2025
"""

RECORD = {
    "title": "PaperBench: Evaluating AI's Ability to Replicate AI Research",
    "authors": ["Giulio Starace", "Oliver Jaffe", "Dane Sherburn"],
    "year": "2025",
    "doi": "",
    "venue": "arXiv cs.AI",
    "source": "arxiv",
}


def document(title: str, status: str = "verified", refs: list[str] | None = None) -> OcrDocument:
    page = PageResult(page_no=1, route=Route.TEXT_LAYER, markdown=PAGE_ONE, status=status)
    checks = [
        ReferenceCheck(reference=Reference(raw=f"ref {i}"), status=s)
        for i, s in enumerate(refs or [])
    ]
    return OcrDocument(
        source_pdf="data/raw/x.pdf", sha256="0" * 64, title=title, pages=[page], references=checks
    )


def test_draft_reads_identity_from_page_one():
    meta = synth.draft_meta(document("PaperBench: Evaluating AI's Ability to Replicate AI Research"))
    assert meta.arxiv_id == "2504.01848"
    assert meta.year == "2025"
    assert {"Giulio Starace", "Oliver Jaffe", "Dane Sherburn"} <= set(meta.authors)


def test_optimizer_fixes_ocr_title_from_record(monkeypatch):
    monkeypatch.setattr(synth, "lookup", lambda meta, client: RECORD)
    meta, checks, iterations = synth.optimize(document("PaperBeneh: Evalu ating"), None, MetaSettings())
    assert meta.title == RECORD["title"]
    assert meta.venue == "arXiv cs.AI"
    assert iterations == 2  # iter 1 applies the fix list, iter 2 confirms
    assert all(check.passed for check in checks)


def test_no_record_means_needs_review(monkeypatch):
    monkeypatch.setattr(synth, "lookup", lambda meta, client: None)
    _, checks, _ = synth.optimize(document("Unknown"), None, MetaSettings())
    identity = next(c for c in checks if c.name == "identity")
    assert not identity.passed and identity.severity.value == "blocker"


def test_circuit_breaker_stops_a_fix_that_never_converges(monkeypatch):
    calls = iter(range(100))
    monkeypatch.setattr(
        synth, "lookup", lambda meta, client: {**RECORD, "year": str(2000 + next(calls))}
    )
    _, checks, iterations = synth.optimize(document(RECORD["title"]), None, MetaSettings(max_iter=3))
    assert iterations == 3
    assert not next(c for c in checks if c.name == "identity").passed


def test_review_pages_and_bad_citations_are_reported(monkeypatch):
    monkeypatch.setattr(synth, "lookup", lambda meta, client: RECORD)
    doc = document(RECORD["title"], status="needs-review", refs=["verified", "unresolved", "mismatch", "skipped"])
    _, checks, _ = synth.optimize(doc, None, MetaSettings(max_unresolved_refs=0.3))
    by_name = {c.name: c for c in checks}
    assert not by_name["ocr_pages"].passed
    assert not by_name["citations"].passed and "2/3" in by_name["citations"].detail
