from __future__ import annotations

from ra_agent.config import Settings
from ra_agent.ocr.engine import NativeTextEngine
from ra_agent.schema import Route
from ra_agent.ocr.pipeline import OcrPipeline, PageCache


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


def test_vision_down_keeps_plain_figures_and_ingest_succeeds(tmp_path):
    from ra_agent.llm.client import VisionError
    from tests.unit.test_figures import make_pdf_with_figure

    class DownVision:
        model = "down"

        def describe(self, image_path, prompt):
            raise VisionError("endpoint down")

        def complete(self, prompt):
            raise VisionError("endpoint down")

    config = settings(tmp_path)
    config.critic.enabled = False
    config.crossref.enabled = False
    config.vision_extra.cache_dir = str(tmp_path / "vcache")
    pdf = make_pdf_with_figure(tmp_path / "fig.pdf")
    document = OcrPipeline(
        settings=config, engines=fakes(NativeTextEngine()), vision=DownVision()
    ).run(pdf, figures_dir=tmp_path / "figs")
    assert (tmp_path / "figs" / "fig-1.png").is_file()
    assert "![Figure 1: A bar chart of scores.](figures/fig-1.png)" in document.markdown
    assert "> ⚠ description unavailable" in document.markdown


def test_granite_retries_on_cpu_after_cuda_oom():
    from types import SimpleNamespace

    from ra_agent.config import EngineSettings
    from ra_agent.ocr.engine import GraniteDoclingEngine

    devices = []

    class Engine(GraniteDoclingEngine):
        def _converter(self, dpi=0):
            devices.append(self.settings.device)
            if self.settings.device != "cpu":
                raise RuntimeError("CUDA out of memory. Tried to allocate 52.00 MiB")
            doc = SimpleNamespace(export_to_markdown=lambda: "page text", export_to_doctags=lambda: None)
            return SimpleNamespace(convert=lambda **kw: SimpleNamespace(document=doc, status=None))

    engine = Engine(EngineSettings(device="cuda"))
    assert engine.ocr_page("x.pdf", 1) == ("page text", None)
    assert engine.ocr_page("x.pdf", 2)[0] == "page text"
    assert devices == ["cuda", "cpu", "cpu"]  # one retry, then it stays on CPU
    assert EngineSettings().device == "cpu"
