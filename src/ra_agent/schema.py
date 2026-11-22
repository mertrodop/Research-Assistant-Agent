from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    NATIVE_TEXT = "native_text"
    VLM_OCR = "vlm_ocr"
    VLM_OCR_HIGH_DPI = "vlm_ocr_high_dpi"


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
    """LLM reading of one figure. Every number must be grounded in the page text."""

    summary: str = ""
    kind: str = ""
    key_values: list[dict[str, str]] = Field(default_factory=list)
    takeaways: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    model: str = ""
    ungrounded: list[str] = Field(default_factory=list)
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

    @property
    def rel_path(self) -> str:
        return f"figures/{self.filename}"


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

    @property
    def blockers(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.BLOCKER]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARN]


class OcrDocument(BaseModel):
    source_pdf: str
    sha256: str
    title: str = ""
    pages: list[PageResult] = Field(default_factory=list)
    markdown: str = ""
    status: Literal["verified", "needs-review"] = "verified"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def review_pages(self) -> list[int]:
        return [p.page_no for p in self.pages if p.status == "needs-review"]


class NoteFrontmatter(BaseModel):
    title: str
    source_pdf: str
    sha256: str
    ocr_engine: str
    pages: int
    verified_pages: int
    status: Literal["verified", "needs-review"]
    created_at: str
    tags: list[str] = Field(default_factory=list)
    figures: int = 0
    figure_files: list[str] = Field(default_factory=list)
    figure_readings: int = 0
