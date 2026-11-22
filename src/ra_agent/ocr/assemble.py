from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ..schema import FigureRef, PageResult

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_WORDLIST_CANDIDATES = (
    "/usr/share/dict/words",
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
)


@lru_cache(maxsize=1)
def load_wordlist() -> frozenset[str]:
    """Optional system dictionary. Missing wordlist degrades to a pure heuristic."""
    for candidate in _WORDLIST_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                words = path.read_text(encoding="utf-8", errors="ignore").split()
            except OSError:
                continue
            return frozenset(word.lower() for word in words)
    return frozenset()


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()]


def _margin_key(line: str) -> str:
    """Page numbers change per page, so digits are normalised before counting."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_margins(pages: list[PageResult], min_pages: int) -> tuple[set[str], set[str]]:
    firsts: Counter[str] = Counter()
    lasts: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in _lines(page.markdown) if line]
        if not lines:
            continue
        firsts[_margin_key(lines[0])] += 1
        lasts[_margin_key(lines[-1])] += 1
    headers = {key for key, count in firsts.items() if count >= min_pages}
    footers = {key for key, count in lasts.items() if count >= min_pages}
    return headers, footers


def strip_margins(pages: list[PageResult]) -> list[PageResult]:
    if len(pages) < 3:
        return pages
    min_pages = max(3, len(pages) // 2)
    headers, footers = _repeated_margins(pages, min_pages)
    cleaned: list[PageResult] = []
    for page in pages:
        lines = _lines(page.markdown)
        while lines and _margin_key(lines[0]) in headers:
            lines.pop(0)
        while lines and _margin_key(lines[-1]) in footers:
            lines.pop()
        cleaned.append(page.model_copy(update={"markdown": "\n".join(lines).strip()}))
    return cleaned


def _join(match: re.Match[str], words: frozenset[str]) -> str:
    head, tail = match.group(1), match.group(2)
    joined = f"{head}{tail}"
    if not words:
        return joined
    if joined.lower() in words:
        return joined  # trans-former -> transformer, docu-ment -> document
    if head.lower() in words and tail.lower() in words:
        return f"{head}-{tail}"  # state-of-the-art, well-known
    return joined


def dehyphenate(text: str, words: frozenset[str] | None = None) -> str:
    """Rejoin words broken across a line. Keeps the hyphen when both halves are words."""
    vocabulary = load_wordlist() if words is None else words
    return _HYPHEN_BREAK.sub(lambda match: _join(match, vocabulary), text)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _interpretation_lines(figure: FigureRef) -> list[str]:
    """Render the LLM reading as a blockquote under the image, with its caveats."""
    reading = figure.interpretation
    if reading is None:
        return []
    if reading.error:
        return [f"> reading failed: {reading.error}", ""]
    lines = [f"> **{reading.kind or 'figure'}** — {reading.summary}"]
    for item in reading.key_values:
        lines.append(f"> - {item.get('label', '')}: {item.get('value', '')}")
    for takeaway in reading.takeaways:
        lines.append(f"> - {takeaway}")
    if reading.ungrounded:
        lines.append(
            f"> ⚠ not corroborated by page text: {', '.join(reading.ungrounded)}"
        )
    lines.append("")
    return lines


def _insert_figures(markdown: str, figures: list[FigureRef]) -> str:
    """Place each figure image right above the line that carries its caption."""
    if not figures:
        return markdown
    lines = markdown.splitlines()
    shared: dict[str, int] = {}
    for figure in figures:
        key = figure.label or f"Figure {figure.index}"
        shared[key] = shared.get(key, 0) + 1
    seen: dict[str, int] = {}
    pending: list[FigureRef] = []
    for figure in figures:
        key = figure.label or f"Figure {figure.index}"
        seen[key] = seen.get(key, 0) + 1
        alt = key if shared[key] == 1 else f"{key} ({seen[key]}/{shared[key]})"
        target = _norm(figure.caption)[:60] if figure.caption else ""
        block = [f"![{alt}]({figure.rel_path})", ""] + _interpretation_lines(figure)
        placed = False
        if target:
            for index, line in enumerate(lines):
                if _norm(line).startswith(target):
                    lines[index:index] = block
                    placed = True
                    break
        if not placed:
            pending.append(figure)

    if pending:
        lines.append("")
        for figure in pending:
            key = figure.label or f"Figure {figure.index}"
            alt = key if shared[key] == 1 else f"{key} ({seen[key]}/{shared[key]})"
            lines.append(f"![{alt}]({figure.rel_path})")
            lines.append("")
            lines.extend(_interpretation_lines(figure))
    return "\n".join(lines).strip()


def assemble(pages: list[PageResult], figures: dict[int, list[FigureRef]] | None = None) -> str:
    ordered = sorted(pages, key=lambda page: page.page_no)
    cleaned = strip_margins(ordered)
    by_page = figures or {}
    parts: list[str] = []
    for page in cleaned:
        if not page.markdown.strip() and not by_page.get(page.page_no):
            continue
        body = dehyphenate(page.markdown)
        body = _insert_figures(body, by_page.get(page.page_no, []))
        if body.strip():
            parts.append(body)
    return "\n\n".join(parts).strip()


def section_tree(markdown: str) -> list[tuple[int, str]]:
    tree: list[tuple[int, str]] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            tree.append((len(match.group(1)), match.group(2).strip()))
    return tree
