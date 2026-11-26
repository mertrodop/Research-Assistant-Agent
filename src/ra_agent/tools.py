"""Workspace: the tools the agent (and the MCP server) can call.

Every method returns a small JSON-able dict and never raises. PDF text inside a
result is data, so it is wrapped as <<<DATA ... DATA>>>; large content stays on
disk and the result carries a `handle` path instead.
"""

import functools
import inspect
import json
from pathlib import Path
from typing import Literal

from .config import Settings
from .llm.client import LlmClient, build_llm
from .ocr.assemble import assemble
from .ocr.critic import critique_page as _critique
from .ocr.crossref import split_references
from .ocr.gate import document_status
from .ocr.interpret import describe_figure as _describe, figure_block
from .ocr.pipeline import OcrPipeline
from .ocr.refagent import resolve_references
from .ocr.synth import ocr_pages_check
from .ocr.render import page_count, pdf_sha256
from .schema import OcrDocument, PageResult, Route, Severity, localize
from .vault import Vault
from .vault.store import _check_slug, slugify

TOOL_KINDS: dict[str, Literal["read", "compute", "write"]] = {
    "list_papers": "read",
    "vault_search": "read",
    "vault_read": "read",
    "ingest_paper": "compute",
    "reprocess_page": "compute",
    "describe_figure": "compute",
    "check_references": "compute",
    "critique_page": "compute",
    "stage_note": "compute",
    "commit_note": "write",
    "save_procedure": "write",
}


class ToolError(RuntimeError):
    """A failure whose message already names the input."""


def _data(text: str, limit: int = 1500) -> str:
    text = text[:limit].replace("DATA>>>", "DATA> >>").replace("<<<DATA", "<< <DATA")  # a PDF cannot close the delimiter
    return f"<<<DATA\n{text}\nDATA>>>"


def _failed(page: PageResult) -> tuple[int, int]:
    """(blockers, other failed checks): the order in which page results are compared."""
    failed = [c for c in page.checks if not c.passed]
    blockers = sum(c.severity is Severity.BLOCKER for c in failed)
    return blockers, len(failed) - blockers


def _findings(slug: str, pages: list[PageResult], limit: int = 8) -> tuple[list[str], list[str]]:
    """Blocker findings (`problems`, worth a reprocess) and warning-only findings."""
    problems, warnings = [], []
    for page in pages:
        for check in page.checks:
            if not check.passed:
                bucket = problems if check.severity is Severity.BLOCKER else warnings
                bucket.append(_data(localize(slug, page, check), 200))
    return problems[:limit], warnings[:limit]


def _tool(fn):
    """Never raise, always return ok + summary (<= 400 chars)."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def run(*args, **kwargs) -> dict:
        try:
            result = fn(*args, **kwargs)
            result.setdefault("ok", True)
            result["summary"] = str(result.get("summary", ""))[:400]
            return result
        except Exception as exc:  # noqa: BLE001 - the agent reads the error
            if isinstance(exc, ToolError):
                error = str(exc)
            else:
                try:
                    given = ", ".join(
                        f"{k}={str(v)[:60]!r}"
                        for k, v in list(sig.bind(*args, **kwargs).arguments.items())[1:]
                    )
                except Exception:  # noqa: BLE001 - bad call shape: the error must still come back
                    given = "arguments that do not match the tool signature"
                error = f"{fn.__name__} failed for {given}: {type(exc).__name__}: {exc}"
            return {"ok": False, "error": error[:400], "summary": error[:400]}

    return run


class Workspace:
    def __init__(
        self,
        settings: Settings,
        engines: dict | None = None,
        vision: LlmClient | None = None,
    ) -> None:
        # ingest always resolves references and runs the document evaluator-optimizer
        self.settings = settings.model_copy(deep=True)
        self.settings.crossref.enabled = True
        self.vision = vision or build_llm(settings.models["vision"])
        self.pipeline = OcrPipeline(self.settings, engines=engines, vision=self.vision)
        paths = self.settings.paths
        self.paths = paths
        self.staging = Path(paths.staging_dir)
        self.vault = Vault(Path(paths.vault_dir), self.staging)
        self._docs: dict[str, OcrDocument] = {}

    # -- state: last OcrDocument per slug, in memory and staging/<slug>.json
    def _doc(self, paper: str) -> OcrDocument:
        try:
            _check_slug(paper)
        except ValueError:
            raise ToolError(f"paper {paper!r} is not a valid slug") from None
        if paper not in self._docs:
            path = self.staging / f"{paper}.json"
            if not path.is_file():
                raise ToolError(f"paper {paper!r} has not been ingested")
            self._docs[paper] = OcrDocument.model_validate_json(path.read_text(encoding="utf-8"))
        return self._docs[paper]

    def _save(self, slug: str, doc: OcrDocument) -> None:
        self._docs[slug] = doc
        self.staging.mkdir(parents=True, exist_ok=True)
        (self.staging / f"{slug}.json").write_text(doc.model_dump_json(), encoding="utf-8")

    def _pdf(self, pdf: str) -> Path:
        path = Path(pdf)
        if len(path.parts) == 1:
            path = self.paths.find_pdf(pdf)
        if path.resolve() not in {p.resolve() for p in self.paths.list_pdfs()}:
            raise ToolError(f"no such PDF {pdf!r} in {self.paths.inbox_dir} or a paper's raw/ folder")
        return path.resolve()

    def _figures_dir(self, doc: OcrDocument) -> Path:
        return self.staging / "figures" / doc.sha256[:12]

    # -- read
    @_tool
    def list_papers(self) -> dict:
        pdfs = self.paths.list_pdfs()
        papers = [{"pdf": p.name, "pages": page_count(p)} for p in pdfs[:20]]
        ingested = sorted(p.stem for p in self.staging.glob("*.json"))[:20]
        return {
            "papers": papers,
            "ingested": ingested,
            "summary": f"{len(pdfs)} PDFs in the Inbox and the vault, {len(ingested)} ingested",
        }

    @_tool
    def vault_search(self, query: str) -> dict:
        hits = self.vault.search(query)
        lines = [f"{h['slug']} | {h['title']} | {h['line']}" for h in hits]
        return {
            "hits": _data("\n".join(lines)),
            "slugs": [h["slug"] for h in hits],
            "summary": f"{len(hits)} notes match {query!r}",
        }

    @_tool
    def vault_read(self, note: str) -> dict:
        try:
            text = self.vault.read(note, max_chars=1500)
        except ValueError:
            raise ToolError(f"note {note!r} is not a valid slug") from None
        except FileNotFoundError:
            raise ToolError(f"note {note!r} is not in the vault") from None
        return {"text": _data(text), "summary": f"first {len(text)} chars of {note}"}

    # -- compute
    @_tool
    def ingest_paper(self, pdf: str) -> dict:
        path = self._pdf(pdf)
        figures = self.staging / "figures" / pdf_sha256(path)[:12]
        doc = self.pipeline.run(path, figures_dir=figures)
        slug = self.vault.stage(doc, figures).slug
        self._save(slug, doc)
        problems, warnings = _findings(slug, doc.pages)
        metadata = "verified" if doc.meta and doc.meta.source in {"arxiv", "crossref"} else "unverified"
        return {
            "slug": slug,
            "status": doc.status,
            "metadata": metadata,
            "pages": len(doc.pages),
            "review_pages": doc.review_pages,
            "problems": problems,
            "warnings": warnings,
            "handle": str(self.vault.note(self.staging, slug)),
            "summary": (
                f"{slug}: {doc.status}, {len(doc.pages)} pages, metadata {metadata}, "
                f"{len(problems)} problems, {len(warnings)} warnings"
            ),
        }

    @_tool
    def reprocess_page(
        self, paper: str, page: int, engine: Literal["docling", "granite", "vlm", "text_layer"]
    ) -> dict:
        doc = self._doc(paper)
        old = next((p for p in doc.pages if p.page_no == page), None)
        if old is None:
            raise ToolError(f"paper {paper!r} has no page {page}")
        new = self.pipeline.process_page(
            Path(doc.source_pdf), doc.sha256, page, Route(engine), old.attempt + 1
        )
        new = new.model_copy(update={"figures": old.figures})
        # keep the better page: fewer blockers, then fewer other failed checks; a tie keeps the current one
        kept = "new" if _failed(new) < _failed(old) else "previous"
        why = (
            f"kept the {kept} result: {engine} has {_failed(new)[0]} blockers and {_failed(new)[1]} warnings, "
            f"the current {old.engine} result has {_failed(old)[0]} blockers and {_failed(old)[1]} warnings"
        )
        problems, warnings = _findings(paper, [new if kept == "new" else old], 5)
        if kept == "previous":
            return {
                "status": old.status, "engine": old.engine, "kept": kept,
                "problems": problems, "warnings": warnings,
                "summary": f"{paper} page {page}: {why}",
            }
        pages = [new if p.page_no == page else p for p in doc.pages]
        doc = doc.model_copy(update={"pages": pages})
        checks = [ocr_pages_check(doc) if c.name == "ocr_pages" else c for c in doc.doc_checks]
        doc = doc.model_copy(
            update={
                "markdown": assemble(pages, {p.page_no: p.figures for p in pages}, "figures"),
                "doc_checks": checks,
                "status": document_status(pages, checks),
            }
        )
        self._save(paper, doc)
        return {
            "status": new.status, "engine": new.engine, "kept": kept,
            "problems": problems, "warnings": warnings,
            "summary": f"{paper} page {page}: {why}",
        }

    @_tool
    def describe_figure(self, paper: str, figure: int) -> dict:
        doc = self._doc(paper)
        found = next(
            ((p, f) for p in doc.pages for f in p.figures if f.index == figure), None
        )
        if found is None:
            raise ToolError(f"paper {paper!r} has no figure {figure}")
        page, old = found
        new = _describe(
            old, self._figures_dir(doc), page.markdown, self.vision,
            self.settings.vision_extra, doc.sha256,
        )
        pages = [
            p.model_copy(update={"figures": [new if f.index == figure else f for f in p.figures]})
            for p in doc.pages
        ]
        self._save(paper, doc.model_copy(update={"pages": pages}))
        reading = new.interpretation
        state = "unavailable" if reading is None or reading.error else "described"
        return {
            "description": _data(figure_block(new, "figures")),
            "summary": f"{paper} figure {figure}: description {state}",
        }

    @_tool
    def check_references(self, paper: str) -> dict:
        doc = self._doc(paper)
        checks = resolve_references(
            split_references(doc.markdown),
            self.pipeline._crossref_client(),
            self.settings.references,
            self.vision if self.settings.references.use_llm else None,
        )
        self._save(paper, doc.model_copy(update={"references": checks}))
        counts: dict[str, int] = {}
        for check in checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        bad = [c.reference.raw[:90] for c in checks if c.status in {"mismatch", "unresolved"}]
        return {
            "counts": counts,
            "unresolved": _data("\n".join(bad[:5])),
            "summary": f"{paper}: {len(checks)} references, {counts}",
        }

    @_tool
    def critique_page(self, paper: str, page: int) -> dict:
        doc = self._doc(paper)
        found = next((p for p in doc.pages if p.page_no == page), None)
        if found is None:
            raise ToolError(f"paper {paper!r} has no page {page}")
        pdf = Path(doc.source_pdf)
        check = _critique(
            found.markdown,
            self.vision,
            self.settings.critic,
            page,
            image_path=self.pipeline._page_png(pdf, doc.sha256, page),
            text_layer=self.pipeline._text_layer(pdf, page),
        )
        return {
            "passed": check.passed,
            "severity": check.severity.value,
            "detail": _data(check.detail),
            "summary": f"{paper} page {page} critic: {'passed' if check.passed else 'failed'}",
        }

    @_tool
    def stage_note(self, paper: str) -> dict:
        doc = self._doc(paper)
        staged = self.vault.stage(doc, self._figures_dir(doc))
        diff = self.vault.diff(staged.slug)
        return {
            "slug": staged.slug,
            "diff": _data(diff),
            "handle": str(staged.note_path),
            "summary": f"staged {staged.slug} ({staged.status}), diff {len(diff)} chars",
        }

    # -- fallback (called by the agent graph, not exposed as a tool)
    @_tool
    def fallback_ingest(self, paper: str) -> dict:
        """Deterministic ingest: no figure descriptions, staged as needs-review, never committed."""
        try:
            doc = self._doc(paper)
            sha = doc.sha256
        except ToolError:  # not ingested yet: treat it as a PDF name
            path = self._pdf(paper if paper.lower().endswith(".pdf") else f"{paper}.pdf")
            doc, sha = None, pdf_sha256(path)
        committed = next(
            (n.stem for n in self.vault.notes(self.vault.root) if self.vault._sha_at(self.vault.root, n.stem) == sha),
            None,
        )
        if committed:
            return {"slug": committed, "summary": f"{committed} is already in the vault; fallback did nothing"}
        if doc is None:
            figures = self.settings.figures
            describe, figures.describe = figures.describe, False
            try:
                doc = self.pipeline.run(path, figures_dir=self.staging / "figures" / pdf_sha256(path)[:12])
            finally:
                figures.describe = describe
        doc = doc.model_copy(update={"status": "needs-review"})
        staged = self.vault.stage(doc, self._figures_dir(doc))
        self._save(staged.slug, doc)
        return {
            "slug": staged.slug,
            "status": doc.status,
            "handle": str(staged.note_path),
            "summary": f"fallback staged {staged.slug} as needs-review; nothing written to the vault",
        }

    # -- write (gated by human approval in the agent)
    @_tool
    def commit_note(self, paper: str) -> dict:
        self._doc(paper)  # validates the slug
        try:
            target = self.vault.commit(paper)
        except FileNotFoundError:
            raise ToolError(f"nothing staged for {paper!r}; call stage_note first") from None
        self._docs.pop(paper, None)
        (self.staging / f"{paper}.json").unlink(missing_ok=True)
        return {"path": str(target), "summary": f"committed {paper} to the vault"}

    @_tool
    def save_procedure(self, name: str, text: str) -> dict:
        if not text.strip():
            raise ToolError(f"procedure {name!r} has no text")
        target = Path(self.settings.paths.memory_dir) / "procedures" / f"{slugify(name, 'procedure')}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "summary": f"saved procedure {target.stem}"}
