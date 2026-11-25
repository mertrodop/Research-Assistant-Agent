from __future__ import annotations

import yaml

from ra_agent.config import Settings
from ra_agent.ocr.engine import NativeTextEngine
from ra_agent.schema import Route
from ra_agent.ocr.pipeline import OcrPipeline, PageCache, parse_pages
from ra_agent.vault.note import render_note, write_note


class BrokenEngine:
    """Always returns nothing: forces the retry ladder and the circuit breaker."""

    name = "broken"

    def ocr_page(self, pdf_path, page_no, dpi=0):
        return "", None


class CrashingEngine:
    name = "crashing"

    def ocr_page(self, pdf_path, page_no, dpi=0):
        from ra_agent.ocr.engine import OcrEngineError

        raise OcrEngineError("CUDA out of memory")


def fakes(engine) -> dict:
    return {Route.DOCLING: engine, Route.GRANITE: engine, Route.TEXT_LAYER: engine}


def settings(tmp_path) -> Settings:
    config = Settings()
    config.pipeline.cache_dir = str(tmp_path / "cache")
    config.paths.vault_dir = str(tmp_path / "vault")
    config.pipeline.concurrency = 2
    return config


def test_ladder_ends_on_text_layer(pdf, tmp_path):
    engines = {
        Route.DOCLING: BrokenEngine(),
        Route.GRANITE: BrokenEngine(),
        Route.TEXT_LAYER: NativeTextEngine(),
    }
    document = OcrPipeline(settings=settings(tmp_path), engines=engines).run(pdf)
    assert document.status == "verified"
    assert all(
        p.route is Route.TEXT_LAYER and p.attempt == 3 and p.status == "verified"
        for p in document.pages
    )


def test_native_pipeline_verifies_text_pdf(pdf, tmp_path):
    document = OcrPipeline(
        settings=settings(tmp_path), engines=fakes(NativeTextEngine())
    ).run(pdf)
    assert document.status == "verified"
    assert len(document.pages) == 2
    assert all(page.route is Route.DOCLING for page in document.pages)
    assert "Retrieval augmented generation" in document.markdown
    assert document.title == "Test Paper"


def test_page_subset(pdf, tmp_path):
    document = OcrPipeline(settings=settings(tmp_path), engines=fakes(NativeTextEngine())).run(pdf, pages=[2])
    assert [page.page_no for page in document.pages] == [2]


def test_parse_pages():
    assert parse_pages("1-3,7") == [1, 2, 3, 7]
    assert parse_pages(None) is None


def test_circuit_breaker_marks_needs_review(blank_pdf, tmp_path):
    config = settings(tmp_path)
    document = OcrPipeline(settings=config, engines=fakes(BrokenEngine())).run(blank_pdf)
    assert document.status == "needs-review"
    page = document.pages[0]
    assert page.status == "needs-review"
    assert page.attempt == config.pipeline.max_attempts
    assert "non_empty" in page.review_reasons


def test_cache_avoids_second_engine_call(pdf, tmp_path):
    config = settings(tmp_path)
    cache = PageCache(config.pipeline.cache_dir)
    first = OcrPipeline(settings=config, cache=cache, engines=fakes(NativeTextEngine())).run(pdf)
    second = OcrPipeline(settings=config, cache=cache, engines=fakes(BrokenEngine())).run(pdf)
    assert first.status == "verified"
    assert second.status == "verified"
    assert all(page.latency_s == 0.0 for page in second.pages)


def test_note_frontmatter_and_validation(pdf, tmp_path):
    config = settings(tmp_path)
    document = OcrPipeline(settings=config, engines=fakes(NativeTextEngine())).run(pdf)
    path = write_note(document, config.paths.vault_dir, "docling")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---")[1])
    assert front["status"] == "verified"
    assert front["pages"] == 2
    assert "## Validation" in text
    assert "| 1 | docling | 1 | verified |" in text


def test_flat_and_vault_layouts(pdf, tmp_path):
    config = settings(tmp_path)
    document = OcrPipeline(settings=config, engines=fakes(NativeTextEngine())).run(pdf)
    flat = write_note(document, tmp_path / "processed", "text_layer")
    vault = write_note(document, tmp_path / "vault", "text_layer", layout="vault")
    assert flat == tmp_path / "processed" / f"{document.sha256[:12]}.md"
    assert vault == (
        tmp_path / "vault" / "Resources" / "Parsed" / f"{document.sha256[:12]}.md"
    )
    assert flat.read_text(encoding="utf-8") == vault.read_text(encoding="utf-8")


def test_render_note_needs_review_tag(blank_pdf, tmp_path):
    config = settings(tmp_path)
    document = OcrPipeline(settings=config, engines=fakes(BrokenEngine())).run(blank_pdf)
    text = render_note(document, "broken")
    assert "needs-review" in text


def test_engine_crash_is_recorded_not_raised(blank_pdf, tmp_path):
    config = settings(tmp_path)
    document = OcrPipeline(settings=config, engines=fakes(CrashingEngine())).run(blank_pdf)
    page = document.pages[0]
    assert document.status == "needs-review"
    assert page.error == "CUDA out of memory"
    assert "engine_error" in page.review_reasons
    assert page.attempt == config.pipeline.max_attempts


def test_native_engine_separates_hanging_indent_entries(tmp_path):
    import fitz

    from ra_agent.ocr.engine import NativeTextEngine

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for x, text in [(72, "Andrew, R. L. and Vines, T. Assessing the reproducibility"), (82, "of discriminant function analyses. PeerJ, 2015."),
                    (72, "Belz, A. A systematic review of reproducibility research"), (82, "in natural language processing. EACL, 2021.")]:
        page.insert_text((x, y), text, fontsize=9)
        y += 11
    path = tmp_path / "refs.pdf"
    doc.save(path)
    text, _ = NativeTextEngine().ocr_page(path, 1)
    blocks = [b for b in text.split("\n\n") if b.strip()]
    assert len(blocks) == 2 and blocks[1].startswith("Belz")
