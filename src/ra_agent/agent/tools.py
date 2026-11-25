"""Tool registry for the agent loop.

Every tool declares a pydantic argument model, so the model's proposal is
validated by code before anything runs. A tool never raises: failures come back
as observations, which is what lets the loop reflect and retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..ocr.gate import has_blockers, run_gate
from ..ocr.pipeline import OcrPipeline, parse_pages
from ..ocr.render import page_count, pdf_sha256
from ..schema import PageResult


class ToolError(RuntimeError):
    """Raised inside a tool; converted into an observation by the loop."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel], dict]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_model.model_json_schema(),
        }


class ListPapersArgs(BaseModel):
    pass


class OcrArgs(BaseModel):
    pdf: str = Field(description="path to a PDF, relative to the repo or absolute")
    pages: str = Field(default="", description="page spec such as 1-3,7; empty means all")


class GateArgs(BaseModel):
    pdf: str
    page: int = Field(ge=1)


class ReferencesArgs(BaseModel):
    pdf: str
    max_attempts: int = Field(default=4, ge=1, le=8)


class ReadNoteArgs(BaseModel):
    sha: str = Field(min_length=4, description="sha256 prefix of a processed note")


class WriteReportArgs(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1)


def _resolve(pdf: str) -> Path:
    path = Path(pdf)
    if path.is_file():
        return path
    raise ToolError(f"no such PDF: {pdf}")


def build_tools(settings: Settings, pipeline: OcrPipeline | None = None) -> dict[str, Tool]:
    """Bind tools to a settings object so the agent works on the real pipeline."""
    engine = pipeline or OcrPipeline(settings=settings)

    def list_papers(_: ListPapersArgs) -> dict:
        root = Path(settings.paths.raw_dir)
        if not root.is_dir():
            return {"papers": [], "note": f"{root} does not exist"}
        papers = []
        for pdf in sorted(root.glob("*.pdf")):
            papers.append(
                {
                    "path": str(pdf),
                    "sha": pdf_sha256(pdf)[:12],
                    "pages": page_count(pdf),
                    "size_kb": round(pdf.stat().st_size / 1024),
                }
            )
        return {"papers": papers, "count": len(papers)}

    def ocr(args: OcrArgs) -> dict:
        pdf = _resolve(args.pdf)
        document = engine.run(pdf, pages=parse_pages(args.pages or None))
        return {
            "sha": document.sha256[:12],
            "title": document.title,
            "status": document.status,
            "pages": len(document.pages),
            "review_pages": document.review_pages,
            "chars": len(document.markdown),
        }

    def gate(args: GateArgs) -> dict:
        pdf = _resolve(args.pdf)
        document = engine.run(pdf, pages=[args.page])
        page: PageResult = document.pages[0]
        return {
            "page": page.page_no,
            "route": page.route.value,
            "status": page.status,
            "failed": page.review_reasons,
            "blockers": [check.name for check in page.blockers],
            "warnings": [check.name for check in page.warnings],
            "chars": len(page.markdown),
        }

    def references(args: ReferencesArgs) -> dict:
        pdf = _resolve(args.pdf)
        local = settings.model_copy(deep=True)
        local.crossref.enabled = True
        local.references.max_attempts = args.max_attempts
        document = OcrPipeline(settings=local, engines=engine.engines).run(pdf)
        counts: dict[str, int] = {}
        for check in document.references:
            counts[check.status] = counts.get(check.status, 0) + 1
        return {
            "sha": document.sha256[:12],
            "total": len(document.references),
            "counts": counts,
            "unresolved": [
                check.reference.raw[:90]
                for check in document.references
                if check.status in {"mismatch", "unresolved"}
            ][:5],
        }

    def read_note(args: ReadNoteArgs) -> dict:
        root = Path(settings.paths.staging_dir) / "Papers"
        matches = [
            note
            for note in sorted(root.glob("*.md"))
            if f"sha256: {args.sha}" in note.read_text(encoding="utf-8")[:2000]
        ]
        if not matches:
            raise ToolError(f"no processed note for sha prefix {args.sha}")
        text = matches[0].read_text(encoding="utf-8")
        return {
            "path": str(matches[0]),
            "chars": len(text),
            "head": text[:1200],
        }

    def write_report(args: WriteReportArgs) -> dict:
        safe = "".join(char for char in args.name if char.isalnum() or char in "-_ ").strip()
        if not safe:
            raise ToolError("report name has no usable characters")
        target = Path(settings.paths.staging_dir) / "reports" / f"{safe}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.body, encoding="utf-8")
        return {"path": str(target), "chars": len(args.body)}

    tools = [
        Tool(
            "list_papers",
            "List the PDFs available under data/raw with their sha and page count.",
            ListPapersArgs,
            list_papers,
        ),
        Tool(
            "ocr",
            "Run the OCR pipeline on a PDF and return the verification status.",
            OcrArgs,
            ocr,
        ),
        Tool(
            "gate",
            "Run the Tier 1 gate on one page and return the failing checks.",
            GateArgs,
            gate,
        ),
        Tool(
            "references",
            "Resolve the bibliography of a PDF against Crossref.",
            ReferencesArgs,
            references,
        ),
        Tool(
            "read_note",
            "Read a processed note by sha256 prefix.",
            ReadNoteArgs,
            read_note,
        ),
        Tool(
            "write_report",
            "Write a markdown report under data/processed/reports.",
            WriteReportArgs,
            write_report,
        ),
    ]
    return {tool.name: tool for tool in tools}


def call_tool(tool: Tool, raw_args: dict) -> dict:
    """Validate arguments with code, then run. Never raises."""
    try:
        args = tool.args_model.model_validate(raw_args or {})
    except ValidationError as exc:
        return {"ok": False, "error": f"invalid arguments: {exc.errors()[:2]}"}
    try:
        result = tool.run(args)
    except ToolError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a tool failure is an observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"ok": True, "result": result}


def catalog(tools: dict[str, Tool]) -> str:
    return json.dumps([tool.schema() for tool in tools.values()], indent=1)
