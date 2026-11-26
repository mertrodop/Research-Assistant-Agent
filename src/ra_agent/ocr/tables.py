"""Fallback: a table that still fails its checks after Docling and row repair is read
again by the vision model; the text layer decides whether to trust the reply."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz

from ..llm.client import LlmClient, VisionError
from .gate import _MARKS, _cells, check_table_cells, check_table_headers, check_table_merged_rows, check_tables
from .vlm import TABLE_PROMPT, read_image

_LOC = re.compile(r"<otsl>((?:<loc_\d+>){4})")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")  # numbers and words; marks never match
_CHECKS = (check_tables, check_table_headers, check_table_cells, check_table_merged_rows)
_NOTE = (
    "> [!warning]- Check marks read from the image\n"
    "> This table was re-read from the page image; its check marks are not in the text layer."
)


def table_boxes(doctags: str | None) -> list[tuple[float, ...]]:
    """Table boxes as page fractions (x0, y0, x1, y1), in reading order. Docling's
    location grid is 500 x 500."""
    return [tuple(int(v) / 500 for v in re.findall(r"\d+", m)) for m in _LOC.findall(doctags or "")]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(unicodedata.normalize("NFKC", text).lower().replace(",", "")))


def ungrounded(table: str, text_layer: str) -> list[str]:
    """Words and numbers of the table that the text layer does not contain."""
    return sorted(_tokens(table) - _tokens(text_layer))


def _failing(block: str) -> list[str]:
    return [r.name for check in _CHECKS if not (r := check(block)).passed]


def _crop(pdf_path: Path, page_no: int, box: tuple[float, ...], dpi: int) -> bytes:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_no - 1)
        w, h = page.rect.width, page.rect.height
        clip = fitz.Rect(box[0] * w, box[1] * h, box[2] * w, box[3] * h) + (-2, -2, 2, 2)
        return page.get_pixmap(dpi=dpi, clip=clip & page.rect, alpha=False).tobytes("png")


def _blocks(markdown: str) -> list[tuple[bool, str]]:
    """(is_table, text) runs in order; consecutive pipe lines form one table."""
    out: list[tuple[bool, str]] = []
    for line in markdown.splitlines():
        table = line.strip().startswith("|") and line.strip().endswith("|")
        if out and out[-1][0] == table:
            out[-1] = (table, out[-1][1] + "\n" + line)
        else:
            out.append((table, line))
    return out


def reread_tables(
    markdown: str,
    doctags: str | None,
    text_layer: str,
    pdf_path: Path,
    page_no: int,
    dpi: int,
    client: LlmClient,
    cache_dir: str | Path,
) -> tuple[str, list[int], dict[int, str]]:
    """Replace each still-failing table by the vision reading when it is grounded and
    passes the table checks. Returns (markdown, replaced table numbers, reasons for
    the tables kept). Without a text layer nothing can be grounded: unchanged."""
    boxes = table_boxes(doctags)
    blocks, replaced, kept, n = _blocks(markdown), [], {}, 0
    if len(text_layer.strip()) < 50:
        return markdown, replaced, kept
    for i, (is_table, text) in enumerate(blocks):
        if not is_table:
            continue
        n += 1
        failing = _failing(text)
        if not failing or n > len(boxes):
            continue
        try:
            new = read_image(client, _crop(pdf_path, page_no, boxes[n - 1], dpi), TABLE_PROMPT, cache_dir)
        except VisionError as exc:
            kept[n] = f"vision error: {str(exc)[:80]}"
            continue
        new = "\n".join(l for l in new.splitlines() if l.strip().startswith("|"))
        missing, still = ungrounded(new, text_layer), _failing(new)
        if not new or missing or still:
            kept[n] = f"not grounded: {', '.join(missing[:8])}" if missing else (
                f"still fails {', '.join(still)}" if new else "no table in reply")
            continue
        if any(m in c for c in sum((_cells(r) for r in new.splitlines()), []) for m in _MARKS):
            new += "\n\n" + _NOTE
        blocks[i] = (True, new)
        replaced.append(n)
    return "\n".join(t for _, t in blocks), replaced, kept
