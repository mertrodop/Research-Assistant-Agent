from __future__ import annotations

from ..config import RenderSettings, RouterSettings
from ..schema import Route
from .render import PageStats


def choose_route(
    stats: PageStats,
    router: RouterSettings,
    attempt: int = 1,
    vlm: bool = False,
) -> Route:
    """Deterministic routing ladder. Code decides, never an LLM.

    Text-layer page: docling -> granite @ high dpi -> text layer.
    Low-text page: vision LLM (when available) -> granite -> text layer, or granite at
    base, high and max dpi when there is no text layer to fall back on.
    """
    if stats.text_layer_chars < router.native_text_min_chars:
        if not vlm:
            return Route.GRANITE
        last = Route.TEXT_LAYER if stats.text_layer_chars >= 50 else Route.GRANITE
        return (Route.VLM, Route.GRANITE, last)[min(attempt, 3) - 1]
    return (Route.DOCLING, Route.GRANITE, Route.TEXT_LAYER)[min(attempt, 3) - 1]


def dpi_for(route: Route, render: RenderSettings, attempt: int = 1) -> int:
    """DPI ladder: base -> high -> max. Each retry must change the input,
    otherwise a deterministic engine returns the same failure again. The vision
    model reads a page rendered at the same dpi."""
    if route not in (Route.GRANITE, Route.VLM):
        return 0
    return (render.dpi, render.high_dpi, render.max_dpi)[min(attempt, 3) - 1]
