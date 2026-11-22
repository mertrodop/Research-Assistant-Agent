from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageImage:
    page_no: int
    png: bytes
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class PageStats:
    page_no: int
    text_layer_chars: int
    ink_ratio: float
    width_pt: float
    height_pt: float


def pdf_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: str | Path) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


def render_page(path: str | Path, page_no: int, dpi: int) -> PageImage:
    with fitz.open(path) as doc:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        return PageImage(
            page_no=page_no,
            png=pix.tobytes("png"),
            width=pix.width,
            height=pix.height,
            dpi=dpi,
        )


def page_stats(
    path: str | Path,
    page_no: int,
    ink_threshold: int = 200,
    probe_dpi: int = 72,
) -> PageStats:
    with fitz.open(path) as doc:
        page = doc.load_page(page_no - 1)
        text = page.get_text("text") or ""
        pix = page.get_pixmap(dpi=probe_dpi, colorspace=fitz.csGRAY, alpha=False)
        samples = pix.samples
        dark = sum(1 for value in samples if value < ink_threshold)
        ink_ratio = dark / len(samples) if samples else 0.0
        return PageStats(
            page_no=page_no,
            text_layer_chars=len(text.strip()),
            ink_ratio=ink_ratio,
            width_pt=page.rect.width,
            height_pt=page.rect.height,
        )
