from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    DOCLING = "docling"
    GRANITE = "granite"
    VLM = "vlm"
    TEXT_LAYER = "text_layer"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCKER = "blocker"


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Severity = Severity.BLOCKER
    detail: str = ""
    evidence: str = ""


class FigureInterpretation(BaseModel):
    """Vision reading of one figure. Numbers must be grounded in the page text."""

    kind: Literal["chart", "table", "diagram", "photo", "equation", "other"] = "other"
    summary: str = ""
    markdown: str = ""
    ungrounded: list[str] = Field(default_factory=list)
    model: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.ungrounded


class FigureRef(BaseModel):
    page_no: int
    index: int
    filename: str
    width: int
    height: int
    caption: str = ""
    label: str = ""
    kind: str = "raster"
    interpretation: FigureInterpretation | None = None


class PageResult(BaseModel):
    page_no: int
    route: Route
    markdown: str
    doctags: str | None = None
    engine: str = ""
    dpi: int = 0
    attempt: int = 1
    latency_s: float = 0.0
    text_layer_chars: int = 0
    ink_ratio: float = 0.0
    figures: list[FigureRef] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    status: Literal["verified", "needs-review"] = "verified"
    review_reasons: list[str] = Field(default_factory=list)
    error: str = ""
    llm_tables: list[int] = Field(default_factory=list)  # tables replaced by the vision model

    @property
    def blockers(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.BLOCKER]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARN]

    @property
    def problems(self) -> list[str]:
        """One sentence per failed check; the agent reads these to pick a re-run."""
        return [_sentence(self, c) for c in self.checks if not c.passed]


def _sentence(page: PageResult, check: CheckResult) -> str:
    evidence = f" ({check.evidence})" if check.evidence else ""
    return (
        f"page {page.page_no}: {check.name}: {check.detail}{evidence}; "
        f"engine {page.engine}, attempt {page.attempt}"
    )


def localize(paper: str, page: PageResult, check: CheckResult) -> str:
    return f"{paper}, {_sentence(page, check)}"


class Reference(BaseModel):
    raw: str
    title: str = ""
    year: str = ""


class ReferenceCheck(BaseModel):
    reference: Reference
    status: str = "skipped"  # verified | mismatch | unresolved | skipped
    matched_title: str = ""
    matched_year: str = ""
    score: float = 0.0
    detail: str = ""
    strategy: str = ""
    attempts: list[str] = Field(default_factory=list)


class PaperMeta(BaseModel):
    """Paper identity: drafted from page 1, corrected from arXiv / Crossref."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source: str = ""
    code: list[str] = Field(default_factory=list)


class OcrDocument(BaseModel):
    source_pdf: str
    sha256: str
    title: str = ""
    pages: list[PageResult] = Field(default_factory=list)
    markdown: str = ""
    status: Literal["verified", "needs-review"] = "verified"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    references: list[ReferenceCheck] = Field(default_factory=list)
    meta: PaperMeta | None = None
    meta_iterations: int = 0
    doc_checks: list[CheckResult] = Field(default_factory=list)

    @property
    def review_pages(self) -> list[int]:
        return [p.page_no for p in self.pages if p.status == "needs-review"]

    @property
    def unresolved_references(self) -> list[ReferenceCheck]:
        return [ref for ref in self.references if ref.status in {"mismatch", "unresolved"}]


class NoteFrontmatter(BaseModel):
    title: str
    source_pdf: str
    sha256: str
    pages: int
    verified_pages: int
    status: Literal["verified", "needs-review"]
    metadata: Literal["verified", "unverified"] = "unverified"
    engines: dict[str, int] = Field(default_factory=dict)
    created_at: str
    tags: list[str] = Field(default_factory=list)
    figures: int = 0
    figure_files: list[str] = Field(default_factory=list)
    figure_readings: int = 0
    code: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
