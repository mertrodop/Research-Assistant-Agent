from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from ..schema import NoteFrontmatter, OcrDocument


def _frontmatter(document: OcrDocument, engine: str) -> str:
    verified = sum(1 for page in document.pages if page.status == "verified")
    figures = [figure for page in document.pages for figure in page.figures]
    meta = NoteFrontmatter(
        title=document.title,
        source_pdf=document.source_pdf,
        sha256=document.sha256,
        ocr_engine=engine,
        pages=len(document.pages),
        verified_pages=verified,
        status=document.status,
        created_at=document.created_at.isoformat(timespec="seconds"),
        tags=["ocr", document.status],
        figures=len(figures),
        figure_files=[figure.rel_path for figure in figures],
        figure_readings=sum(
            1
            for figure in figures
            if figure.interpretation is not None and not figure.interpretation.error
        ),
        **(
            document.meta.model_dump(include={"authors", "year", "venue", "doi", "arxiv_id"})
            if document.meta
            else {}
        ),
    )
    body = yaml.safe_dump(
        meta.model_dump(), sort_keys=False, allow_unicode=True
    ).strip()
    return f"---\n{body}\n---\n"


def _validation(document: OcrDocument) -> str:
    lines = ["## Validation", "", "| page | route | attempt | status | failed checks |", "|---|---|---|---|---|"]
    for page in document.pages:
        failed = ", ".join(page.review_reasons) or "-"
        if page.error:
            failed = f"{failed} ({page.error[:80]})"
        lines.append(
            f"| {page.page_no} | {page.route.value} | {page.attempt} | {page.status} | {failed} |"
        )
    return "\n".join(lines)


def _references(document: OcrDocument) -> str:
    if not document.references:
        return ""
    lines = [
        "## References check",
        "",
        "| status | score | strategy | reference | matched |",
        "|---|---|---|---|---|",
    ]
    for check in document.references:
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


def render_note(document: OcrDocument, engine: str) -> str:
    parts = [
        _frontmatter(document, engine),
        f"# {document.title}",
        "",
        document.markdown,
        "",
        _validation(document),
    ]
    for section in (_doc_checks(document), _references(document)):
        if section:
            parts.extend(["", section])
    return "\n".join(parts).strip() + "\n"


def note_filename(document: OcrDocument) -> str:
    return f"{document.sha256[:12]}.md"


def write_note(
    document: OcrDocument,
    out_dir: str | Path,
    engine: str,
    layout: Literal["flat", "vault"] = "flat",
) -> Path:
    """flat -> <out_dir>/<sha12>.md, vault -> <out_dir>/Resources/Parsed/<sha12>.md."""
    root = Path(out_dir)
    if layout == "vault":
        root = root / "Resources" / "Parsed"
    target = root / note_filename(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_note(document, engine), encoding="utf-8")
    return target
