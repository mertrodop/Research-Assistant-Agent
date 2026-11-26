from __future__ import annotations

import fitz

from ra_agent.config import Settings
from ra_agent.llm.client import NullVisionClient
from ra_agent.ocr.pipeline import OcrPipeline
from ra_agent.ocr.router import choose_route
from ra_agent.ocr.render import PageStats
from ra_agent.schema import Route

TEXT = (
    "We summarize which files are shown to the judge for each requirement type in the table below. "
    "Submissions without a reproduce script score zero on all execution nodes of the rubric.\n"
    "Files Code Execution\nSource code 1 2\nreproduce log 3 4\n"
)
BROKEN = "| Files | Code | Execution |\n|---|---|---|\n| Source code reproduce log | ✓ ✓ | ✓ ✗ |"
GOOD = "| Files | Code | Execution |\n|---|---|---|\n| Source code | ✓ | ✓ |\n| reproduce log | ✓ | ✗ |"
INVENTED = "| Files | Code | Execution |\n|---|---|---|\n| Source code | 1 | 2 |\n| reproduce log | 3 | 99 |"
DOCTAGS = "<doctag><otsl><loc_40><loc_100><loc_400><loc_200><fcel></otsl></doctag>"


class Docling:
    name = "docling"

    def ocr_page(self, pdf_path, page_no, dpi=0):
        return f"{TEXT}\n\n{BROKEN}\n", DOCTAGS


class Fake:
    model = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def describe(self, image_path, prompt):
        self.calls += 1
        return self.reply


def make(tmp_path, text=TEXT):
    path = tmp_path / "t.pdf"
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=10)
    doc.save(path)
    return path


def pipeline(tmp_path, vision, engines=None):
    config = Settings()
    config.pipeline.cache_dir = str(tmp_path / "c")
    config.vision_extra.cache_dir = str(tmp_path / "v")
    return OcrPipeline(settings=config, engines=engines or {Route.DOCLING: Docling()}, vision=vision)


def test_good_llm_table_replaces_broken_one(tmp_path):
    page = pipeline(tmp_path, Fake(GOOD)).process_page(make(tmp_path), "s", 1, Route.DOCLING)
    assert "| reproduce log | ✓ | ✗ |" in page.markdown
    assert "Check marks read from the image" in page.markdown
    assert page.llm_tables == [1]
    assert not page.blockers


def test_invented_number_is_rejected_and_blocker_stays(tmp_path):
    page = pipeline(tmp_path, Fake(INVENTED)).process_page(make(tmp_path), "s", 1, Route.DOCLING)
    assert BROKEN in page.markdown and page.llm_tables == []
    assert "table_merged_rows" in page.review_reasons


def test_null_client_never_rereads(tmp_path):
    page = pipeline(tmp_path, NullVisionClient()).process_page(make(tmp_path), "s", 1, Route.DOCLING)
    assert BROKEN in page.markdown and page.llm_tables == []



def test_low_text_page_uses_vlm_first_and_null_skips_it(tmp_path):
    pdf = make(tmp_path, "Short page with some text only, about one hundred characters long, no more than that.")
    page = pipeline(tmp_path, Fake("# Short page\n\nShort page with some text only, about one hundred characters long, no more than that.")).process_page(pdf, "s", 1)
    assert page.route is Route.VLM and page.engine == "vision_llm" and page.dpi == 144
    assert page.status == "verified"
    assert Route.VLM not in pipeline(tmp_path, NullVisionClient()).engines


def test_vlm_ladder():
    stats = PageStats(page_no=1, text_layer_chars=120, ink_ratio=0.1, width_pt=595, height_pt=842)
    r = Settings().router
    assert [choose_route(stats, r, a, vlm=True) for a in (1, 2, 3)] == [Route.VLM, Route.GRANITE, Route.TEXT_LAYER]
