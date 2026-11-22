from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

from ..config import EngineSettings


class OcrEngineError(RuntimeError):
    """Raised when an engine cannot produce output for a page."""


class OcrEngine(Protocol):
    name: str

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int
    ) -> tuple[str, str | None]: ...


class NativeTextEngine:
    """PyMuPDF text layer. Free, deterministic, only valid when a text layer exists."""

    name = "native_text"

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        import fitz

        with fitz.open(pdf_path) as doc:
            text = doc.load_page(page_no - 1).get_text("text") or ""
        return text.strip(), None


class GraniteDoclingEngine:
    """Granite-Docling-258M through docling's VlmPipeline, one page at a time.

    docling is imported lazily so the rest of the pipeline runs without it.
    A converter is cached per thread: docling converters are not thread safe.
    """

    name = "granite_docling"

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or EngineSettings()
        self._local = threading.local()

    def _converter(self):
        converter = getattr(self._local, "converter", None)
        if converter is None:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                VlmConvertOptions,
                VlmPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline

            vlm_options = VlmConvertOptions.from_preset("granite_docling")
            pipeline_options = VlmPipelineOptions(vlm_options=vlm_options)
            device = self.settings.device.lower()
            if device in {"auto", "cpu", "cuda", "mps", "xpu"}:
                pipeline_options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice(device)
                )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            self._local.converter = converter
        return converter

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        try:
            result = self._converter().convert(
                source=str(pdf_path),
                page_range=(page_no, page_no),
                raises_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - engine failures are data, not crashes
            raise OcrEngineError(f"{type(exc).__name__}: {exc}") from exc
        document = result.document
        markdown = (document.export_to_markdown() or "").strip()
        if not markdown and result.status is not None and result.status.name != "SUCCESS":
            raise OcrEngineError(f"docling status {result.status.name}")
        doctags: str | None = None
        try:
            doctags = document.export_to_doctags()
        except Exception:
            doctags = None
        return markdown, doctags


def build_engine(settings: EngineSettings) -> OcrEngine:
    if settings.name == "native":
        return NativeTextEngine()
    return GraniteDoclingEngine(settings)
