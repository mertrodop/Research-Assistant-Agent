"""Vision-LLM readers: a whole low-text page and a single table region.

The model only proposes. Code accepts a table only after it is grounded in the
PDF text layer (see tables.py); a page goes through the normal gate.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import fitz

from ..llm.client import LlmClient, VisionError
from .engine import OcrEngineError

_FENCE = re.compile(r"^```(?:markdown|md)?\s*|\s*```$", re.M)

_STRICT = """Copy the text exactly as printed. Do not summarize, translate, correct or add anything.
Never invent numbers or words; if something is unreadable, leave it out."""

PAGE_PROMPT = f"""You are transcribing one page of a scientific paper into Markdown.
{_STRICT}
Use # headings, GitHub-Markdown tables, and LaTeX for equations. Answer with the Markdown only."""

TABLE_PROMPT = f"""The image is one table from a scientific paper.
{_STRICT}
Answer with exactly one GitHub-Markdown table and nothing else: a header row, a separator row,
one Markdown row per printed row. Use an empty cell when a cell is unreadable. Do not add rows.
Write a check mark as ✓ and a cross as ✗, never as a letter. Copy the header exactly, column for column."""


def read_image(client: LlmClient, png: bytes, prompt: str, cache_dir: str | Path) -> str:
    """One vision call on PNG bytes, cached by (model, prompt, image). Raises VisionError."""
    key = hashlib.sha256(client.model.encode() + prompt.encode() + png).hexdigest()[:24]
    cached = Path(cache_dir) / f"{key}.md"
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.png"
        image.write_bytes(png)
        reply = _FENCE.sub("", client.describe(image, prompt)).strip()
    if reply:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(reply, encoding="utf-8")
    return reply


class VisionPageEngine:
    """Whole page as Markdown from the vision model (OcrEngine protocol)."""

    name = "vision_llm"

    def __init__(self, client: LlmClient, cache_dir: str | Path) -> None:
        self.client, self.cache_dir = client, cache_dir

    def ocr_page(self, pdf_path: str | Path, page_no: int, dpi: int = 0) -> tuple[str, str | None]:
        with fitz.open(pdf_path) as doc:
            png = doc.load_page(page_no - 1).get_pixmap(dpi=dpi or 144, alpha=False).tobytes("png")
        try:
            return read_image(self.client, png, PAGE_PROMPT, self.cache_dir), None
        except VisionError as exc:
            raise OcrEngineError(f"vision: {exc}") from exc
