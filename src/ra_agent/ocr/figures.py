from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

_CAPTION = re.compile(r"^(?:figure|fig\.?)\s*(\d+)\b", re.I)


@dataclass(frozen=True)
class Figure:
    page_no: int
    index: int
    filename: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]
    caption: str = ""
    label: str = ""
    kind: str = "raster"


def _caption_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    blocks: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        stripped = " ".join(text.split())
        if _CAPTION.match(stripped):
            blocks.append((fitz.Rect(x0, y0, x1, y1), stripped))
    return blocks


def _match_caption(
    bbox: fitz.Rect, captions: list[tuple[fitz.Rect, str]], max_gap: float
) -> str:
    best: tuple[float, str] | None = None
    for rect, text in captions:
        if rect.y0 >= bbox.y1 - 2:
            gap = rect.y0 - bbox.y1
        elif rect.y1 <= bbox.y0 + 2:
            gap = bbox.y0 - rect.y1
        else:
            continue
        if gap <= max_gap and (best is None or gap < best[0]):
            best = (gap, text)
    return best[1] if best else ""


def _label(caption: str) -> str:
    match = _CAPTION.match(caption)
    return f"Figure {match.group(1)}" if match else ""


def extract_figures(
    pdf_path: str | Path,
    page_no: int,
    out_dir: str | Path,
    min_side: int = 100,
    min_area_ratio: float = 0.01,
    max_caption_gap: float = 140.0,
    page_render_fallback: bool = True,
    render_dpi: int = 150,
    start_index: int = 1,
) -> list[Figure]:
    """Save raster figures of one page as PNG and link them to their captions.

    Vector-only figures carry no raster object, so when a page has a figure
    caption but no usable raster, the page is rendered once as a fallback.
    Files are named `fig-N.png` where N is the figure index, counted from
    `start_index` so the caller can number figures across the whole paper.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_no - 1)
        page_area = page.rect.width * page.rect.height
        captions = _caption_blocks(page)
        seen: dict[int, str] = {}

        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref") or 0
            bbox = fitz.Rect(info["bbox"])
            if bbox.width < min_side or bbox.height < min_side:
                continue
            if page_area and (bbox.width * bbox.height) / page_area < min_area_ratio:
                continue
            if xref <= 0:
                continue
            if xref not in seen:
                try:
                    raw = doc.extract_image(xref)
                except Exception:
                    continue
                if not raw or not raw.get("image"):
                    continue
                name = f"fig-{start_index + len(figures)}.png"
                target = out / name
                pix = fitz.Pixmap(raw["image"])
                if pix.colorspace and pix.colorspace.n not in (1, 3):
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(target)
                seen[xref] = name
            caption = _match_caption(bbox, captions, max_caption_gap)
            figures.append(
                Figure(
                    page_no=page_no,
                    index=start_index + len(figures),
                    filename=seen[xref],
                    width=int(bbox.width),
                    height=int(bbox.height),
                    bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    caption=caption,
                    label=_label(caption),
                )
            )

        if not figures and captions and page_render_fallback:
            name = f"fig-{start_index}.png"
            page.get_pixmap(dpi=render_dpi, colorspace=fitz.csRGB, alpha=False).save(
                out / name
            )
            rect, caption = captions[0]
            figures.append(
                Figure(
                    page_no=page_no,
                    index=start_index,
                    filename=name,
                    width=int(page.rect.width),
                    height=int(page.rect.height),
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    caption=caption,
                    label=_label(caption),
                    kind="page_render",
                )
            )

    return figures
