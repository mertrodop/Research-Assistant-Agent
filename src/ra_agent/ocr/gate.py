from __future__ import annotations

import re
import string
from collections import Counter

from ..config import GateSettings
from ..schema import CheckResult, Severity

_PRINTABLE = set(string.printable)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_TAG_OPEN = re.compile(r"<([a-zA-Z][\w-]*)>")
_TAG_CLOSE = re.compile(r"</([a-zA-Z][\w-]*)>")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _check(
    name: str,
    passed: bool,
    severity: Severity = Severity.BLOCKER,
    detail: str = "",
    evidence: str = "",
) -> CheckResult:
    return CheckResult(
        name=name, passed=passed, severity=severity, detail=detail, evidence=evidence
    )


def check_non_empty(
    markdown: str, settings: GateSettings, ink_ratio: float = 1.0
) -> CheckResult:
    size = len(markdown.strip())
    if size >= settings.min_chars:
        return _check("non_empty", True, detail=f"{size} chars")
    sparse = ink_ratio < settings.sparse_ink
    return _check(
        "non_empty",
        False,
        Severity.WARN if sparse else Severity.BLOCKER,
        detail=f"{size} chars (min {settings.min_chars}), ink {ink_ratio:.4f}"
        + (" -> sparse page" if sparse else ""),
        evidence=markdown.strip()[:120],
    )


def check_printable(markdown: str, settings: GateSettings) -> CheckResult:
    if not markdown:
        return _check("printable_ratio", False, detail="empty output")
    bad = sum(1 for char in markdown if char not in _PRINTABLE)
    ratio = 1 - bad / len(markdown)
    return _check(
        "printable_ratio",
        ratio >= settings.min_printable_ratio,
        detail=f"{ratio:.3f} printable (min {settings.min_printable_ratio})",
        evidence=f"{bad} non-printable chars",
    )


def check_repetition(markdown: str, settings: GateSettings) -> CheckResult:
    tokens = re.findall(r"\S+", markdown.lower())
    size = settings.ngram_size
    if len(tokens) < size * 2:
        return _check("repetition_loop", True, detail="too short to loop")
    grams = Counter(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
    gram, count = grams.most_common(1)[0]
    return _check(
        "repetition_loop",
        count <= settings.max_ngram_repeat,
        detail=f"max {size}-gram repeat {count} (max {settings.max_ngram_repeat})",
        evidence=" ".join(gram)[:160],
    )


def check_coverage(
    markdown: str, text_layer_chars: int, settings: GateSettings
) -> CheckResult:
    if text_layer_chars < 50:
        return _check("coverage", True, Severity.INFO, detail="no text layer baseline")
    expected = max(1.0, text_layer_chars / 6)
    words = len(re.findall(r"\S+", markdown))
    ratio = words / expected
    return _check(
        "coverage",
        settings.min_coverage <= ratio <= settings.max_coverage,
        detail=f"{words} words vs ~{expected:.0f} expected (ratio {ratio:.2f})",
    )


def check_headings(markdown: str) -> CheckResult:
    levels = [len(match.group(1)) for match in re.finditer(r"^(#{1,6})\s+\S", markdown, re.M)]
    jumps = [
        (previous, current)
        for previous, current in zip(levels, levels[1:])
        if current - previous > 1
    ]
    return _check(
        "heading_levels",
        not jumps,
        Severity.WARN,
        detail=f"{len(levels)} headings, {len(jumps)} level jumps",
        evidence=str(jumps[:5]),
    )


def _table_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(stripped)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def check_tables(markdown: str) -> CheckResult:
    blocks = _table_blocks(markdown)
    for block in blocks:
        widths = {len(row.strip("|").split("|")) for row in block}
        if len(widths) > 1:
            return _check(
                "table_columns",
                False,
                detail=f"inconsistent column counts {sorted(widths)}",
                evidence=block[0][:160],
            )
    return _check("table_columns", True, detail=f"{len(blocks)} tables consistent")


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def _is_separator(row: str) -> bool:
    cells = _cells(row)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)


def _is_data_row(row: str) -> bool:
    cells = [cell for cell in _cells(row) if cell]
    return bool(cells) and all(_NUMBER.fullmatch(cell) for cell in cells)


def check_table_headers(markdown: str) -> CheckResult:
    """A markdown table needs a text header row above its separator row.

    Two failure modes are caught: the separator moved to the first row (header
    dropped entirely), and the header row survived only as data (all numeric),
    which is what Granite-Docling does when it loses a table header.
    """
    blocks = _table_blocks(markdown)
    for block in blocks:
        if _is_separator(block[0]):
            return _check(
                "table_header",
                False,
                detail="table starts with a separator row: header row lost",
                evidence=block[0][:160],
            )
        if len(block) > 1 and _is_separator(block[1]) and _is_data_row(block[0]):
            return _check(
                "table_header",
                False,
                detail="header row is numeric: header text lost, first data row promoted",
                evidence=block[0][:160],
            )
    return _check("table_header", True, detail=f"{len(blocks)} tables have headers")


def check_equations(markdown: str) -> CheckResult:
    problems: list[str] = []
    if markdown.count("$$") % 2:
        problems.append("unbalanced $$")
    if markdown.count("\\[") != markdown.count("\\]"):
        problems.append("unbalanced \\[ \\]")
    if markdown.count("\\begin{") != markdown.count("\\end{"):
        problems.append("unbalanced \\begin/\\end")
    return _check(
        "equation_delimiters",
        not problems,
        detail=", ".join(problems) or "balanced",
    )


def check_doctags(doctags: str | None) -> CheckResult:
    if not doctags:
        return _check("doctags_balance", True, Severity.INFO, detail="no doctags")
    stack: list[str] = []
    for match in re.finditer(r"</?([a-zA-Z][\w-]*)>", doctags):
        token = match.group(0)
        name = match.group(1)
        if token.startswith("</"):
            if not stack or stack[-1] != name:
                return _check(
                    "doctags_balance",
                    False,
                    Severity.WARN,
                    detail=f"unexpected closing tag </{name}>",
                )
            stack.pop()
        else:
            stack.append(name)
    return _check(
        "doctags_balance",
        not stack,
        Severity.WARN,
        detail=f"{len(stack)} unclosed tags",
        evidence=", ".join(stack[:5]),
    )


def _dedupe_spanning_cells(markdown: str) -> str:
    """docling repeats a cell that spans columns once per column; count it once."""
    lines = []
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            kept = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
            line = "| " + " | ".join(kept) + " |"
        lines.append(line)
    return "\n".join(lines)


def _numbers(text: str) -> Counter:
    """Multiset of numbers. 1,008 -> 1008; any other comma separates numbers
    (footnote markers stick to commas in the text layer: "o1,11")."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", _dedupe_spanning_cells(text))
    return Counter(re.findall(r"\d+(?:\.\d+)?", text))


def check_numbers(markdown: str, text_layer: str) -> CheckResult:
    """Every number in the OCR output must exist in the text layer, as often.

    A multiset, not a set: a misread value (21.0 -> 27.0), a dropped decimal point
    (0.45 -> 0 45) or a duplicated table all leave numbers the page does not have,
    even when each value also appears somewhere else on the page. Those are
    blockers. Missing numbers (margin stamps, page numbers) stay a warning.
    """
    if len(text_layer.strip()) < 50:
        return _check("numeric_agreement", True, Severity.INFO, detail="no baseline")
    expected, found = _numbers(text_layer), _numbers(markdown)
    if not expected and not found:
        return _check("numeric_agreement", True, Severity.INFO, detail="no numbers")
    unsupported = found - expected
    if unsupported:
        return _check(
            "numeric_agreement",
            False,
            Severity.BLOCKER,
            detail=f"{sum(unsupported.values())} numbers not in the text layer",
            evidence=", ".join(sorted(unsupported)[:8]),
        )
    missing = expected - found
    ratio = 1 - sum(missing.values()) / max(1, sum(expected.values()))
    return _check(
        "numeric_agreement",
        ratio >= 0.8,
        Severity.WARN,
        detail=f"{ratio:.2f} of baseline numbers present",
        evidence=", ".join(sorted(missing)[:8]),
    )


def check_references(markdown: str) -> CheckResult:
    lines = markdown.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^#{1,6}\s*references\b", line.strip(), re.I)
        ),
        None,
    )
    if start is None:
        return _check("reference_fields", True, Severity.INFO, detail="no references")
    entries = [line.strip() for line in lines[start + 1 :] if len(line.strip()) > 30]
    if not entries:
        return _check("reference_fields", False, Severity.WARN, detail="empty section")
    with_year = [entry for entry in entries if _YEAR.search(entry)]
    ratio = len(with_year) / len(entries)
    return _check(
        "reference_fields",
        ratio >= 0.8,
        Severity.WARN,
        detail=f"{len(with_year)}/{len(entries)} entries carry a year",
        evidence=next((e for e in entries if e not in with_year), "")[:160],
    )


def check_figure_grounding(figures: list) -> CheckResult:
    """Every number an LLM reported about a figure must exist in the page text."""
    if not figures:
        return _check("figure_grounding", True, Severity.INFO, detail="no figures")
    ungrounded: list[str] = []
    errors: list[str] = []
    for figure in figures:
        reading = getattr(figure, "interpretation", None)
        if reading is None:
            continue
        if reading.error:
            errors.append(f"{figure.label or figure.filename}: {reading.error[:80]}")
        ungrounded.extend(reading.ungrounded)
    if ungrounded:
        return _check(
            "figure_grounding",
            False,
            Severity.WARN,
            detail=f"{len(ungrounded)} numbers not found in page text",
            evidence=", ".join(sorted(set(ungrounded))[:8]),
        )
    if errors:
        return _check(
            "figure_grounding",
            False,
            Severity.WARN,
            detail=f"{len(errors)} figure readings failed",
            evidence="; ".join(errors[:3]),
        )
    return _check("figure_grounding", True, detail=f"{len(figures)} figures grounded")


def run_gate(
    markdown: str,
    settings: GateSettings,
    text_layer_chars: int = 0,
    text_layer: str = "",
    doctags: str | None = None,
    ink_ratio: float = 1.0,
) -> list[CheckResult]:
    return [
        check_non_empty(markdown, settings, ink_ratio),
        check_printable(markdown, settings),
        check_repetition(markdown, settings),
        check_coverage(markdown, text_layer_chars, settings),
        check_headings(markdown),
        check_tables(markdown),
        check_table_headers(markdown),
        check_equations(markdown),
        check_doctags(doctags),
        check_numbers(markdown, text_layer),
        check_references(markdown),
    ]


def has_blockers(checks: list[CheckResult]) -> bool:
    return any(not check.passed and check.severity is Severity.BLOCKER for check in checks)
