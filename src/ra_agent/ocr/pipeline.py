from __future__ import annotations

import importlib.metadata
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import Settings
from ..schema import CheckResult, FigureRef, OcrDocument, PageResult, Route, Severity
from ..llm.client import LlmClient, NullVisionClient
from .assemble import assemble
from .critic import critique_page
from .crossref import CrossrefClient, ReferenceCheck, split_references
from .engine import NativeTextEngine, OcrEngine, OcrEngineError, build_engines
from .figures import extract_figures
from .interpret import describe_figures
from .repair import repair_merged_rows
from .tables import reread_tables
from .vlm import VisionPageEngine
from .refagent import resolve_references
from .synth import optimize
from .gate import check_figure_grounding, document_status, has_blockers, run_gate
from .render import PageStats, page_count, page_stats, pdf_sha256, render_page
from .router import choose_route, dpi_for


PIPELINE_VERSION = "2"  # bump whenever the processing of page output (engines, markdown conversion) changes


def _cache_version() -> str:
    try:
        docling = importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        docling = "none"
    return f"v{PIPELINE_VERSION}-docling{docling}"


class PageCache:
    """Caches OCR output per (version, pdf sha, page, route, dpi). Only clean pages are stored."""

    def __init__(self, root: str | Path, version: str | None = None) -> None:
        self.root = Path(root) / (version or _cache_version())

    def _path(self, sha: str, page_no: int, route: Route, dpi: int) -> Path:
        return self.root / sha / f"p{page_no:04d}_{route.value}_{dpi}.json"

    def get(
        self, sha: str, page_no: int, route: Route, dpi: int
    ) -> tuple[str, str | None] | None:
        path = self._path(sha, page_no, route, dpi)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["markdown"], data.get("doctags")

    def put(
        self,
        sha: str,
        page_no: int,
        route: Route,
        dpi: int,
        markdown: str,
        doctags: str | None,
    ) -> None:
        path = self._path(sha, page_no, route, dpi)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"markdown": markdown, "doctags": doctags}, ensure_ascii=False),
            encoding="utf-8",
        )


class OcrPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        engines: dict[Route, OcrEngine] | None = None,
        cache: PageCache | None = None,
        vision: LlmClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.engines = dict(engines or build_engines(self.settings.engine))
        self.cache = cache or PageCache(self.settings.pipeline.cache_dir)
        self.vision = vision
        self._native = NativeTextEngine()
        self._vision_cache = Path(self.settings.vision_extra.cache_dir) / "ocr"
        if vision is not None and not isinstance(vision, NullVisionClient):
            self.engines.setdefault(Route.VLM, VisionPageEngine(vision, self._vision_cache))

    def _text_layer(self, pdf_path: Path, page_no: int) -> str:
        text, _ = self._native.ocr_page(pdf_path, page_no)
        return text

    def _page_png(self, pdf_path: Path, sha: str, page_no: int) -> Path:
        """Page image the critic compares against, cached next to the OCR cache."""
        path = self.cache.root / sha / f"p{page_no:04d}_critic.png"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_page(pdf_path, page_no, self.settings.critic.dpi).png)
        return path

    def process_page(
        self,
        pdf_path: Path,
        sha: str,
        page_no: int,
        route: Route | None = None,
        attempt: int = 1,
        stats: PageStats | None = None,
    ) -> PageResult:
        """OCR and gate one page. `route=None` lets the router choose for `attempt`."""
        if stats is None:
            stats = page_stats(
                pdf_path, page_no, ink_threshold=self.settings.render.ink_threshold
            )
        route = route or choose_route(stats, self.settings.router, attempt, Route.VLM in self.engines)
        dpi = dpi_for(route, self.settings.render, attempt)
        engine = self.engines[route]

        cached = self.cache.get(sha, page_no=stats.page_no, route=route, dpi=dpi)
        error = ""
        if cached is not None:
            markdown, doctags = cached
            latency = 0.0
        else:
            started = time.perf_counter()
            try:
                markdown, doctags = engine.ocr_page(pdf_path, stats.page_no, dpi)
            except OcrEngineError as exc:
                markdown, doctags, error = "", None, str(exc)
            latency = time.perf_counter() - started

        llm_tables: list[int] = []
        rejects: list[str] = []
        text_layer = (
            self._text_layer(pdf_path, stats.page_no)
            if stats.text_layer_chars >= 50
            else ""
        )
        if route is Route.DOCLING:
            markdown = repair_merged_rows(markdown, text_layer)
            if self.vision is not None and not isinstance(self.vision, NullVisionClient):
                markdown, llm_tables, rejects = reread_tables(
                    markdown, doctags, text_layer, pdf_path, stats.page_no,
                    self.settings.render.dpi, self.vision, self._vision_cache / "tables",
                )
        checks = run_gate(
            markdown,
            self.settings.gate,
            text_layer_chars=stats.text_layer_chars,
            text_layer=text_layer,
            doctags=doctags,
            ink_ratio=stats.ink_ratio,
            route=route,
        )
        if stats.text_layer_chars < self.settings.router.native_text_min_chars:
            checks.append(
                CheckResult(
                    name="no_text_layer",
                    passed=False,
                    severity=Severity.WARN,
                    detail="content cannot be checked against the PDF text; read from the image",
                )
            )
        if error:
            checks.insert(
                0,
                CheckResult(
                    name="engine_error",
                    passed=False,
                    severity=Severity.BLOCKER,
                    detail=error[:400],
                ),
            )
        failed = has_blockers(checks)
        if (
            not failed
            and self.settings.critic.enabled
            and self.vision is not None
            and markdown.strip()
        ):
            checks.append(
                critique_page(
                    markdown,
                    self.vision,
                    self.settings.critic,
                    stats.page_no,
                    image_path=self._page_png(pdf_path, sha, stats.page_no),
                    text_layer=text_layer,
                )
            )
            failed = has_blockers(checks)
        if cached is None and not failed:
            self.cache.put(sha, stats.page_no, route, dpi, markdown, doctags)

        return PageResult(
            page_no=stats.page_no,
            route=route,
            markdown=markdown,
            doctags=doctags,
            engine=engine.name,
            dpi=dpi,
            attempt=attempt,
            latency_s=round(latency, 3),
            text_layer_chars=stats.text_layer_chars,
            ink_ratio=round(stats.ink_ratio, 4),
            checks=checks,
            status="needs-review" if failed else "verified",
            review_reasons=[check.name for check in checks if not check.passed],
            error=error,
            llm_tables=llm_tables,
            llm_table_rejects=rejects,
        )

    def run(
        self,
        pdf_path: str | Path,
        pages: list[int] | None = None,
        figures_dir: str | Path | None = None,
    ) -> OcrDocument:
        pdf_path = Path(pdf_path)
        sha = pdf_sha256(pdf_path)
        total = page_count(pdf_path)
        wanted = pages or list(range(1, total + 1))
        wanted = [page for page in wanted if 1 <= page <= total]

        stats = {
            page: page_stats(
                pdf_path, page, ink_threshold=self.settings.render.ink_threshold
            )
            for page in wanted
        }

        results: dict[int, PageResult] = {}
        with ThreadPoolExecutor(max_workers=self.settings.pipeline.concurrency) as pool:
            futures = {
                page: pool.submit(self.process_page, pdf_path, sha, page, None, 1, stats[page])
                for page in wanted
            }
            for page, future in futures.items():
                results[page] = future.result()

            for attempt in range(2, self.settings.pipeline.max_attempts + 1):
                retry = [p for p in wanted if results[p].status == "needs-review"]
                if not retry:
                    break
                futures = {
                    page: pool.submit(
                        self.process_page, pdf_path, sha, page, None, attempt, stats[page]
                    )
                    for page in retry
                }
                for page, future in futures.items():
                    results[page] = future.result()

        ordered = [results[page] for page in sorted(results)]
        figures: dict[int, list[FigureRef]] = {}
        if figures_dir is not None and self.settings.figures.enabled:
            figures = self._extract_figures(pdf_path, ordered, Path(figures_dir))
            describing = (
                self.settings.figures.describe
                and self.vision is not None
                and not isinstance(self.vision, NullVisionClient)
            )
            if describing:
                figures = self._describe_figures(
                    figures, Path(figures_dir), ordered, sha
                )
            ordered = [
                page.model_copy(update={"figures": figures.get(page.page_no, [])})
                for page in ordered
            ]
            if describing:
                checked: list[PageResult] = []
                for page in ordered:
                    result = check_figure_grounding(page.figures)
                    checked.append(
                        page.model_copy(
                            update={
                                "checks": page.checks + [result],
                                "review_reasons": page.review_reasons
                                + ([] if result.passed else [result.name]),
                            }
                        )
                    )
                ordered = checked
        markdown = assemble(ordered, figures, "figures")
        references: list[ReferenceCheck] = []
        if self.settings.crossref.enabled:
            references = resolve_references(
                split_references(markdown),
                self._crossref_client(),
                self.settings.references,
                self.vision if self.settings.references.use_llm else None,
            )
        status = document_status(ordered, [])
        document = OcrDocument(
            source_pdf=str(pdf_path),
            sha256=sha,
            title=self._title(pdf_path, markdown),
            pages=ordered,
            markdown=markdown,
            status=status,
            references=references,
        )
        if self.settings.crossref.enabled:
            meta, doc_checks, iterations = optimize(
                document, self._crossref_client(), self.settings.meta
            )
            document = document.model_copy(
                update={
                    "meta": meta,
                    "title": meta.title or document.title,
                    "doc_checks": doc_checks,
                    "meta_iterations": iterations,
                    "status": document_status(ordered, doc_checks),
                }
            )
        return document

    def _crossref_client(self) -> CrossrefClient:
        config = self.settings.crossref
        return CrossrefClient(
            cache_dir=config.cache_dir,
            timeout_s=config.timeout_s,
            mailto=config.mailto,
            min_score=config.min_score,
            year_tolerance=config.year_tolerance,
        )

    def _describe_figures(
        self,
        figures: dict[int, list[FigureRef]],
        image_dir: Path,
        pages: list[PageResult],
        sha: str,
    ) -> dict[int, list[FigureRef]]:
        flat = [figure for group in figures.values() for figure in group]
        if not flat:
            return figures
        texts = {page.page_no: page.markdown for page in pages}
        described = describe_figures(
            flat,
            image_dir,
            texts,
            self.vision,
            self.settings.vision_extra,
            sha,
        )
        grouped: dict[int, list[FigureRef]] = {}
        for figure in described:
            grouped.setdefault(figure.page_no, []).append(figure)
        return grouped

    def _extract_figures(
        self, pdf_path: Path, pages: list[PageResult], out_dir: Path
    ) -> dict[int, list[FigureRef]]:
        config = self.settings.figures
        found: dict[int, list[FigureRef]] = {}
        count = 0
        for page in pages:
            try:
                raw = extract_figures(
                    pdf_path,
                    page.page_no,
                    out_dir,
                    min_side=config.min_side,
                    min_area_ratio=config.min_area_ratio,
                    max_caption_gap=config.max_caption_gap,
                    page_render_fallback=config.page_render_fallback,
                    render_dpi=config.render_dpi,
                    start_index=count + 1,
                )
            except Exception:
                continue
            count += len(raw)
            if raw:
                found[page.page_no] = [
                    FigureRef(
                        page_no=figure.page_no,
                        index=figure.index,
                        filename=figure.filename,
                        width=figure.width,
                        height=figure.height,
                        caption=figure.caption,
                        label=figure.label,
                        kind=figure.kind,
                    )
                    for figure in raw
                ]
        return found

    @staticmethod
    def _title(pdf_path: Path, markdown: str) -> str:
        import fitz

        with fitz.open(pdf_path) as doc:
            meta_title = (doc.metadata or {}).get("title", "").strip()
        if meta_title:
            return meta_title
        for line in markdown.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return pdf_path.stem
