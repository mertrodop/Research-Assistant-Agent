from __future__ import annotations

import html
import re
import string
import threading
from pathlib import Path
from typing import Protocol

from ..config import EngineSettings
from ..schema import Route


_INDENT = (5.0, 14.0)  # points; hanging indent of a reference entry


class OcrEngineError(RuntimeError):
    """Raised when an engine cannot produce output for a page."""


class OcrEngine(Protocol):
    name: str

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int
    ) -> tuple[str, str | None]: ...


class NativeTextEngine:
    """PyMuPDF text layer. Free, deterministic, only valid when a text layer exists."""

    name = "text_layer"

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        import fitz

        with fitz.open(pdf_path) as doc:
            # same flags and clip as get_text("text"); "dict" defaults drop figure text
            data = doc.load_page(page_no - 1).get_text(
                "dict", flags=fitz.TEXTFLAGS_TEXT, clip=fitz.INFINITE_RECT()
            )
        # Same lines, same order as get_text("text"); blocks are not separated.
        lines = [
            (line["bbox"][0], "".join(span["text"] for span in line["spans"]))
            for block in data["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
        ]
        out: list[str] = []
        for i, (x0, text) in enumerate(lines):
            # Hanging indent (bibliographies): a new entry starts where the next
            # line is indented, or where the text dedents after an indented line.
            indented_next = i + 1 < len(lines) and _INDENT[0] <= lines[i + 1][0] - x0 <= _INDENT[1]
            dedent = i > 0 and _INDENT[0] <= lines[i - 1][0] - x0 <= _INDENT[1]
            if (indented_next or dedent) and out and out[-1] != "":
                out.append("")
            out.append(text)
        return "\n".join(out).strip(), None


def docling_markdown(document) -> str:
    """Markdown of a docling document; footnote items move to the end as `[^N]: text`
    (N is the number the text starts with, else fn-a, fn-b, ...; a digit in the
    label would count as a number in the gate)."""
    from docling_core.types.doc import DocItemLabel

    labels = set(DocItemLabel) - {DocItemLabel.FOOTNOTE}
    markdown = html.unescape(document.export_to_markdown(labels=labels) or "").strip()
    notes = []
    for item, _ in document.iterate_items():
        if item.label is DocItemLabel.FOOTNOTE:
            text = html.unescape(" ".join(item.text.split()))
            number = re.match(r"(\d+)\s+(.*)", text)
            label, text = number.groups() if number else (f"fn-{string.ascii_lowercase[len(notes) % 26]}", text)
            notes.append(f"[^{label}]: {text}")
    return "\n\n".join([markdown, *notes]).strip()


class DoclingStandardEngine:
    """Docling's standard layout pipeline (clean Markdown, ~1 s/page warm on CPU).

    A converter is cached per thread: docling converters are not thread safe.
    """

    name = "docling"

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or EngineSettings()
        self._local = threading.local()

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        try:
            converter = getattr(self._local, "converter", None)
            if converter is None:
                from docling.document_converter import DocumentConverter

                converter = self._local.converter = DocumentConverter()
            result = converter.convert(
                source=str(pdf_path),
                page_range=(page_no, page_no),
                raises_on_error=False,
            )
            document = result.document
            markdown = docling_markdown(document)
            doctags = document.export_to_doctags()
        except Exception as exc:  # noqa: BLE001 - engine failures are data, not crashes
            raise OcrEngineError(f"{type(exc).__name__}: {exc}") from exc
        return markdown, doctags


class GraniteDoclingEngine:
    """Granite-Docling-258M through docling's VlmPipeline, one page at a time.

    docling is imported lazily so the rest of the pipeline runs without it.
    A converter is cached per thread: docling converters are not thread safe.
    """

    name = "granite_docling"

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or EngineSettings()
        self._local = threading.local()

    def _converter(self, dpi: int = 0):
        # One converter per (thread, scale): docling converters are not thread
        # safe, and the render scale is fixed when the converter is built.
        converters = getattr(self._local, "converters", None)
        if converters is None:
            converters = self._local.converters = {}
        converter = converters.get(dpi)
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
            if dpi > 0:
                vlm_options.scale = dpi / 72  # PDF points are 1/72 inch
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
            converters[dpi] = converter
        return converter

    def ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        try:
            return self._ocr_page(pdf_path, page_no, dpi)
        except OcrEngineError as exc:
            # a small GPU runs out of memory mid-paper: retry once on CPU, then stay there
            if "out of memory" not in str(exc).lower() or self.settings.device == "cpu":
                raise
            self.settings = self.settings.model_copy(update={"device": "cpu"})
            self._local = threading.local()
            return self._ocr_page(pdf_path, page_no, dpi)

    def _ocr_page(
        self, pdf_path: str | Path, page_no: int, dpi: int = 0
    ) -> tuple[str, str | None]:
        try:
            result = self._converter(dpi).convert(
                source=str(pdf_path),
                page_range=(page_no, page_no),
                raises_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - engine failures are data, not crashes
            raise OcrEngineError(f"{type(exc).__name__}: {exc}") from exc
        document = result.document
        markdown = html.unescape(document.export_to_markdown() or "").strip()
        if not markdown and result.status is not None and result.status.name != "SUCCESS":
            errors = "; ".join(e.error_message for e in getattr(result, "errors", []) or [])
            raise OcrEngineError(f"docling status {result.status.name}: {errors}"[:500])
        doctags: str | None = None
        try:
            doctags = document.export_to_doctags()
        except Exception:
            doctags = None
        return markdown, doctags


def build_engines(settings: EngineSettings) -> dict[Route, OcrEngine]:
    return {
        Route.DOCLING: DoclingStandardEngine(settings),
        Route.GRANITE: GraniteDoclingEngine(settings),
        Route.TEXT_LAYER: NativeTextEngine(),
    }
