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


def _vector_region(
    page: fitz.Page, caption: fitz.Rect, max_gap: float, taken: list[fitz.Rect]
) -> fitz.Rect | None:
    """The drawing cluster(s) of a vector figure next to its caption, with their labels.

    Figures in the caption's column, above it (below it when nothing is above).
    Clusters chain while they sit within 25 pt of the growing region."""
    clusters = [
        r for r in page.cluster_drawings()
        if (r.width >= 40 or r.height >= 20) and not any(r.intersects(t) for t in taken)
    ]
    column = [r for r in clusters if r.x1 > caption.x0 - 5 and r.x0 < caption.x1 + 5]
    for above in (True, False):
        near = [
            r for r in column
            if (r.y1 <= caption.y0 + 2 and caption.y0 - r.y1 <= max_gap)
            or (not above and r.y0 >= caption.y1 - 2 and r.y0 - caption.y1 <= max_gap)
        ]
        side = [r for r in near if (r.y1 <= caption.y0 + 2) == above]
        if not side:
            continue
        region = min(side, key=lambda r: abs(r.y1 - caption.y0) if above else abs(r.y0 - caption.y1))
        region = fitz.Rect(region)
        grown = True
        while grown:
            grown = False
            for r in column:
                if r.intersects(region + (-25, -25, 25, 25)) and not region.contains(r):
                    region |= r
                    grown = True
        # axis labels and legends are short text blocks around the drawings
        pad = region + (-14, -14, 14, 14)
        for block in page.get_text("blocks"):
            rect = fitz.Rect(block[:4])
            if (len(block[4]) < 160 and rect.intersects(pad) and not rect.intersects(caption)
                    and rect.y1 <= (caption.y0 + 2 if above else 10**6)):
                region |= rect
        return region
    return None


def _gap_region(page: fitz.Page, caption: fitz.Rect) -> fitz.Rect | None:
    """No drawings found: the empty column space between the text above and the caption."""
    top = 0.0
    for block in page.get_text("blocks"):
        rect = fitz.Rect(block[:4])
        if len(block[4]) >= 160 and rect.y1 <= caption.y0 and rect.x1 > caption.x0 and rect.x0 < caption.x1:
            top = max(top, rect.y1)
    region = fitz.Rect(caption.x0, top, caption.x1, caption.y0)
    return region if region.height >= 40 else None


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

        matched = {f.caption for f in figures}
        taken = [fitz.Rect(f.bbox) for f in figures]
        extra: list[tuple[fitz.Rect, fitz.Rect, str, str]] = []
        for rect, caption in captions:
            if caption in matched:
                continue
            region = _vector_region(page, rect, max_caption_gap, taken)
            kind = "vector"
            if region is None:
                region = _gap_region(page, rect)
                if region is None:
                    if not page_render_fallback or figures or extra:
                        continue
                    region, kind = page.rect, "page_render"  # last resort: the whole page
            else:
                taken.append(region)
                region = (region + (-6, -6, 6, 6)) & page.rect
            if region.width >= min_side and region.height >= min_side / 2:
                extra.append((region, rect, caption, kind))
        for region, rect, caption, kind in sorted(extra, key=lambda e: (int(_CAPTION.match(e[2]).group(1)), e[0].y0)):
            name = f"fig-{start_index + len(figures)}.png"
            page.get_pixmap(dpi=render_dpi, clip=region, colorspace=fitz.csRGB, alpha=False).save(out / name)
            figures.append(
                Figure(
                    page_no=page_no,
                    index=start_index + len(figures),
                    filename=name,
                    width=int(region.width),
                    height=int(region.height),
                    bbox=(region.x0, region.y0, region.x1, region.y1),
                    caption=caption,
                    label=_label(caption),
                    kind=kind,
                )
            )

    return figures
