from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ..schema import FigureRef, PageResult
from .interpret import figure_block

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


_MARKER = "<!-- image -->"


def _place_figures(body: str, figures: list[FigureRef], rel_dir: str) -> str:
    """Figure k goes at the page's k-th image marker. Spare figures follow the
    last paragraph; spare markers are dropped."""
    blocks = [figure_block(figure, rel_dir) for figure in figures]
    head, *tails = body.split(_MARKER)
    parts = [head]
    for position, tail in enumerate(tails):
        parts.append(blocks[position] if position < len(blocks) else "")
        parts.append(tail)
    parts.extend(["\n\n" + block for block in blocks[len(tails) :]])
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


def assemble(
    pages: list[PageResult],
    figures: dict[int, list[FigureRef]] | None = None,
    rel_dir: str = "figures",
) -> str:
    ordered = sorted(pages, key=lambda page: page.page_no)
    cleaned = strip_margins(ordered)
    by_page = figures or {}
    parts: list[str] = []
    for page in cleaned:
        body = _place_figures(
            dehyphenate(page.markdown), by_page.get(page.page_no, []), rel_dir
        )
        if body:
            parts.append(body)
    return "\n\n".join(parts).strip()


def section_tree(markdown: str) -> list[tuple[int, str]]:
    tree: list[tuple[int, str]] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            tree.append((len(match.group(1)), match.group(2).strip()))
    return tree
