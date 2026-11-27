from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ..schema import FigureRef, PageResult
from .crossref import _REFERENCES, split_references
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


_PLAIN_BREAK = re.compile(r"^(?:#|\||>|[-*+]\s|\d{1,2}[.)]\s|!\[|<|```|\$\$)")


def reflow(text: str) -> str:
    """Join the hard line breaks of the PDF text layer inside plain paragraphs."""
    out = []
    for paragraph in text.split("\n\n"):
        lines = paragraph.split("\n")
        plain = not any(_PLAIN_BREAK.match(line.strip()) for line in lines)
        text = " ".join(line.strip() for line in lines if line.strip()) if plain else paragraph
        if plain and out and not out[-1].startswith("#") and text[:1].islower() and not re.search(r"[.!?:;)\"”]$", out[-1]):
            out[-1] += " " + text  # a paragraph cut by the column break continues here
        else:
            out.append(text)
    return "\n\n".join(out)


def _place_figures(body: str, figures: list[FigureRef], rel_dir: str) -> str:
    """Each figure replaces its own caption paragraph ("Figure N: ...") with image,
    verbatim caption and reading. A figure without a caption paragraph takes the
    next image marker, else follows the last paragraph; spare markers are dropped."""
    paragraphs = body.split("\n\n")
    spare: list[FigureRef] = []
    for figure in figures:
        number = re.search(r"\d+", figure.label)
        caption = re.compile(rf"^\W*(?:figure|fig\.?)\s*{number.group()}\s*[:.]", re.I) if number else None
        at = next((i for i, par in enumerate(paragraphs) if caption and caption.match(par)), None)
        if at is None:
            spare.append(figure)
        else:
            paragraphs[at] = figure_block(figure, rel_dir, caption=" ".join(paragraphs[at].split()))
    head, *tails = "\n\n".join(paragraphs).split(_MARKER)
    parts = [head]
    for position, tail in enumerate(tails):
        parts.append(figure_block(spare[position], rel_dir, caption=spare[position].caption) if position < len(spare) else "")
        parts.append(tail)
    parts.extend("\n\n" + figure_block(f, rel_dir, caption=f.caption) for f in spare[len(tails) :])
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


def restore_hyphens(markdown: str, pdf_path: str | Path) -> str:
    """Docling joins "last-" + "token" into "lasttoken" at a line break. The PDF text layer
    still has "last-token" unbroken elsewhere: a joined word that never occurs in the PDF
    but whose hyphenated form does gets its hyphen back."""
    import fitz

    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    plain = text.replace("-\n", "\x00")  # a hyphen at a line end proves nothing
    joined = {
        word.replace("-", ""): word
        for word in re.findall(r"(?<![\w-])[A-Za-z]{2,}(?:-[A-Za-z]{2,})+(?![\w-])", plain)
        if "\x00" not in word
    }
    words = set(re.findall(r"[A-Za-z]+", text))
    pattern = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, joined)) + r")(?![\w-])") if joined else None
    if pattern is None:
        return markdown
    return pattern.sub(lambda m: m.group(1) if m.group(1) in words else joined[m.group(1)], markdown)


def fix_fragment_headings(markdown: str) -> str:
    """A "## and Inoue, 2025)." is the tail of the paragraph above, not a heading."""
    return re.sub(r"\n+##\s+([a-z][^\n]*)", r" \1", markdown)


def tidy_references(markdown: str) -> str:
    """One bullet per bibliography entry, whichever way the pages were read."""
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if _REFERENCES.match(line.strip())), None)
    if start is None:
        return markdown
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^#{1,6}\s", lines[i].strip())), len(lines))
    entries = split_references("\n".join(lines[start:end]))
    if not entries:
        return markdown
    return "\n".join(lines[: start + 1] + [""] + [f"- {entry.raw}" for entry in entries] + [""] + lines[end:])


def assemble(
    pages: list[PageResult],
    figures: dict[int, list[FigureRef]] | None = None,
    rel_dir: str = "figures",
    pdf_path: str | Path | None = None,
) -> str:
    ordered = sorted(pages, key=lambda page: page.page_no)
    cleaned = strip_margins(ordered)
    by_page = figures or {}
    parts: list[str] = []
    for page in cleaned:
        text = dehyphenate(page.markdown)
        body = _place_figures(
            reflow(text) if page.engine == "text_layer" else text, by_page.get(page.page_no, []), rel_dir
        )
        if body:
            parts.append(body)
    markdown = fix_fragment_headings("\n\n".join(parts).strip())
    if pdf_path and Path(pdf_path).is_file():
        markdown = restore_hyphens(markdown, pdf_path)
    return tidy_references(markdown)


def section_tree(markdown: str) -> list[tuple[int, str]]:
    tree: list[tuple[int, str]] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            tree.append((len(match.group(1)), match.group(2).strip()))
    return tree
