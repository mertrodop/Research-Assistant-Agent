"""OCR pipeline: parallel pages -> routing -> code gate -> evaluator/optimizer."""

from .assemble import assemble, section_tree
from .engine import (
    GraniteDoclingEngine,
    NativeTextEngine,
    OcrEngineError,
    build_engine,
)
from .gate import has_blockers, run_gate
from .pipeline import OcrPipeline, PageCache, parse_pages
from .render import page_count, page_stats, pdf_sha256, render_page
from .router import choose_route, dpi_for

__all__ = [
    "OcrPipeline",
    "PageCache",
    "parse_pages",
    "assemble",
    "section_tree",
    "run_gate",
    "has_blockers",
    "choose_route",
    "dpi_for",
    "build_engine",
    "GraniteDoclingEngine",
    "NativeTextEngine",
    "OcrEngineError",
    "pdf_sha256",
    "page_count",
    "page_stats",
    "render_page",
]
