"""Deterministic repair of a table row that docling collapsed from N rows."""

from __future__ import annotations


from .gate import MARKS, _cells, _is_separator


def _marks(cell: str) -> list[str]:
    tokens = cell.split()
    return tokens if tokens and all(t in MARKS for t in tokens) else []


def _split_label(label: str, n: int, text_layer: str) -> list[str] | None:
    """N consecutive text-layer lines (mark-only lines skipped) that join to the label."""
    want = " ".join(label.split())
    lines = [l.strip() for l in text_layer.splitlines() if l.strip() and not _marks(l)]
    for i in range(len(lines) - n + 1):
        if " ".join(lines[i : i + n]) == want:
            return lines[i : i + n]
    return None


def repair_merged_rows(markdown: str, text_layer: str) -> str:
    """Split a row whose mark cells all hold the same N >= 2 check marks into N rows,
    labelled from the page's text-layer lines. No exact match: left unchanged."""
    out: list[str] = []
    for line in markdown.splitlines():
        s = line.strip()
        cells = _cells(s) if s.startswith("|") and s.endswith("|") and not _is_separator(s) else []
        runs = [_marks(c) for c in cells[1:]]
        n = len(runs[0]) if runs else 0
        labels = (
            _split_label(cells[0], n, text_layer)
            if n >= 2 and all(len(r) == n for r in runs)
            else None
        )
        if labels is None:
            out.append(line)
            continue
        out += [
            "| " + " | ".join([labels[k], *(r[k] for r in runs)]) + " |" for k in range(n)
        ]
    return "\n".join(out)
