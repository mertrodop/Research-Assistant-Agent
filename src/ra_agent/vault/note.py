from __future__ import annotations

import re

import yaml
from rapidfuzz import fuzz

from ..schema import NoteFrontmatter, OcrDocument, Route


def dump_frontmatter(data: dict) -> str:
    """No line folding: a folded long value would break line-level edits."""
    body = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, width=float("inf")
    ).strip()
    return f"---\n{body}\n---\n"


def _frontmatter(document: OcrDocument) -> str:
    verified = sum(1 for page in document.pages if page.status == "verified")
    figures = [figure for page in document.pages for figure in page.figures]
    meta = NoteFrontmatter(
        title=document.title,
        source_pdf=document.source_pdf,
        sha256=document.sha256,
        metadata=(
            "verified"
            if document.meta and document.meta.source in {"arxiv", "crossref"}
            else "unverified"
        ),
        engines={
            route.value: sum(1 for page in document.pages if page.route is route)
            for route in Route
        },
        pages=len(document.pages),
        verified_pages=verified,
        status=document.status,
        created_at=document.created_at.isoformat(timespec="seconds"),
        tags=["ocr", document.status],
        figures=len(figures),
        figure_files=[f"figures/{figure.filename}" for figure in figures],
        figure_readings=sum(
            1
            for figure in figures
            if figure.interpretation is not None and not figure.interpretation.error
        ),
        **(
            document.meta.model_dump(include={"authors", "year", "venue", "doi", "arxiv_id", "code"})
            if document.meta
            else {}
        ),
    )
    return dump_frontmatter(meta.model_dump())


def _validation(document: OcrDocument) -> str:
    lines = ["## Validation", "", "| page | route | attempt | status | failed checks |", "|---|---|---|---|---|"]
    for page in document.pages:
        if page.status == "verified" and not page.error and not page.review_reasons:
            continue
        failed = ", ".join(page.review_reasons) or "-"
        if page.error:
            failed = f"{failed} ({page.error[:80]})"
        lines.append(
            f"| {page.page_no} | {page.route.value} | {page.attempt} | {page.status} | {failed} |"
        )
    return "\n".join(lines)


def _references(document: OcrDocument) -> str:
    if all(check.status == "verified" for check in document.references):
        return ""
    lines = [
        "## References check",
        "",
        "| status | score | strategy | reference | matched |",
        "|---|---|---|---|---|",
    ]
    for check in document.references:
        if check.status == "verified":
            continue
        raw = check.reference.raw.replace("|", "/")[:120]
        matched = (check.matched_title or check.detail).replace("|", "/")[:80]
        lines.append(
            f"| {check.status} | {check.score:.2f} | {check.strategy or '-'} | {raw} | {matched} |"
        )
    return "\n".join(lines)


def _doc_checks(document: OcrDocument) -> str:
    if not document.doc_checks:
        return ""
    lines = [
        "## Document checks",
        "",
        f"Evaluator-optimizer iterations: {document.meta_iterations}",
        "",
        "| check | passed | severity | detail |",
        "|---|---|---|---|",
    ]
    for check in document.doc_checks:
        detail = check.detail.replace("|", "/")[:300]
        lines.append(f"| {check.name} | {check.passed} | {check.severity.value} | {detail} |")
    return "\n".join(lines)


def _drop_title_heading(markdown: str, title: str) -> str:
    """The note starts with `# title`; drop the engine's first heading if it repeats it."""
    def flat(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.lower()))

    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if fuzz.ratio(flat(line), flat(title)) >= 90:
                return "\n".join(lines[:i] + lines[i + 1 :]).lstrip("\n")
            break
    return markdown


def render_note(document: OcrDocument) -> str:
    # The pipeline writes figure links as figures/fig-N.png; the note lives in
    # Papers/<slug>/md/ and the figures in Papers/<slug>/figures/.
    body = _drop_title_heading(document.markdown, document.title).replace("](figures/", "](../figures/")
    parts = [
        _frontmatter(document),
        f"# {document.title}",
        "",
        body,
        "",
        _validation(document),
    ]
    for section in (_doc_checks(document), _references(document)):
        if section:
            parts.extend(["", section])
    return "\n".join(parts).strip() + "\n"
