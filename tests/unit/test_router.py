from __future__ import annotations

from ra_agent.config import RenderSettings, RouterSettings
from ra_agent.ocr.render import PageStats
from ra_agent.ocr.router import choose_route, dpi_for
from ra_agent.schema import Route


def stats(chars: int, ink: float = 0.05) -> PageStats:
    return PageStats(page_no=1, text_layer_chars=chars, ink_ratio=ink, width_pt=595, height_pt=842)


def test_text_layer_goes_native():
    assert choose_route(stats(1200), RouterSettings()) is Route.NATIVE_TEXT


def test_scanned_page_goes_vlm():
    assert choose_route(stats(0, ink=0.12), RouterSettings()) is Route.VLM_OCR


def test_retry_escalates_to_high_dpi():
    assert choose_route(stats(1200), RouterSettings(), attempt=2) is Route.VLM_OCR_HIGH_DPI


def test_dpi_ladder():
    render = RenderSettings(dpi=200, high_dpi=300)
    assert dpi_for(Route.NATIVE_TEXT, render) == 0
    assert dpi_for(Route.VLM_OCR, render) == 200
    assert dpi_for(Route.VLM_OCR_HIGH_DPI, render) == 300
