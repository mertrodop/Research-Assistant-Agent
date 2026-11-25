"""OCR pipeline: parallel pages -> routing -> code gate -> evaluator/optimizer."""

from .assemble import assemble, section_tree
from .engine import (
    GraniteDoclingEngine,
    NativeTextEngine,
    OcrEngineError,
    DoclingStandardEngine,
    build_engines,
)
from .gate import has_blockers, run_gate
from .pipeline import OcrPipeline, PageCache
from .render import page_count, page_stats, pdf_sha256, render_page
from .router import choose_route, dpi_for

__all__ = [
    "OcrPipeline",
    "PageCache",
    "assemble",
    "section_tree",
    "run_gate",
    "has_blockers",
    "choose_route",
    "dpi_for",
    "build_engines",
    "DoclingStandardEngine",
    "GraniteDoclingEngine",
    "NativeTextEngine",
    "OcrEngineError",
    "pdf_sha256",
    "page_count",
    "page_stats",
    "render_page",
]
