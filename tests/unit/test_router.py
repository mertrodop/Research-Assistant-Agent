from __future__ import annotations

from ra_agent.config import RenderSettings, RouterSettings
from ra_agent.ocr.render import PageStats
from ra_agent.ocr.router import choose_route, dpi_for
from ra_agent.schema import Route


def stats(chars: int, ink: float = 0.05) -> PageStats:
    return PageStats(page_no=1, text_layer_chars=chars, ink_ratio=ink, width_pt=595, height_pt=842)


def test_text_page_ladder():
    r = RouterSettings()
    assert choose_route(stats(1200), r, 1) is Route.DOCLING
    assert choose_route(stats(1200), r, 2) is Route.GRANITE
    assert choose_route(stats(1200), r, 3) is Route.TEXT_LAYER


def test_scanned_page_ladder_is_granite():
    r = RouterSettings()
    assert all(choose_route(stats(0, ink=0.12), r, a) is Route.GRANITE for a in (1, 2, 3))


def test_dpi_ladder():
    render = RenderSettings(dpi=200, high_dpi=300)
    assert dpi_for(Route.TEXT_LAYER, render) == 0
    assert dpi_for(Route.DOCLING, render) == 0
    assert dpi_for(Route.GRANITE, render) == 200
    assert dpi_for(Route.GRANITE, render, attempt=2) == 300


def test_third_attempt_uses_max_dpi():
    # attempt 3 must differ from attempt 2, otherwise the retry repeats the same failure
    render = RenderSettings(dpi=144, high_dpi=216, max_dpi=288)
    assert dpi_for(Route.GRANITE, render, attempt=2) == 216
    assert dpi_for(Route.GRANITE, render, attempt=3) == 288
