from __future__ import annotations

from ..config import RenderSettings, RouterSettings
from ..schema import Route
from .render import PageStats


def choose_route(
    stats: PageStats,
    router: RouterSettings,
    attempt: int = 1,
) -> Route:
    """Deterministic routing ladder. Code decides, never an LLM.

    Text-layer page: docling -> granite @ high dpi -> text layer.
    Scanned page: granite at base, high and max dpi.
    """
    if stats.text_layer_chars < router.native_text_min_chars:
        return Route.GRANITE
    return (Route.DOCLING, Route.GRANITE, Route.TEXT_LAYER)[min(attempt, 3) - 1]


def dpi_for(route: Route, render: RenderSettings, attempt: int = 1) -> int:
    """DPI ladder: base -> high -> max. Each retry must change the input,
    otherwise a deterministic engine returns the same failure again."""
    if route is not Route.GRANITE:
        return 0
    return (render.dpi, render.high_dpi, render.max_dpi)[min(attempt, 3) - 1]
