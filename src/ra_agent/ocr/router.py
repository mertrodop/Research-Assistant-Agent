from __future__ import annotations

from ..config import RenderSettings, RouterSettings
from ..schema import Route
from .render import PageStats


def choose_route(
    stats: PageStats,
    router: RouterSettings,
    attempt: int = 1,
) -> Route:
    """Deterministic routing. Code decides, never an LLM."""
    if attempt >= 2:
        return Route.VLM_OCR_HIGH_DPI
    if stats.text_layer_chars >= router.native_text_min_chars:
        return Route.NATIVE_TEXT
    return Route.VLM_OCR


def dpi_for(route: Route, render: RenderSettings, attempt: int = 1) -> int:
    """DPI ladder: base -> high -> max. Each retry must change the input,
    otherwise a deterministic engine returns the same failure again."""
    if route is Route.NATIVE_TEXT:
        return 0
    if route is Route.VLM_OCR_HIGH_DPI:
        return render.max_dpi if attempt >= 3 else render.high_dpi
    return render.dpi
