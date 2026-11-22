from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, load_env_file
from .llm.client import build_vision_client
from .ocr.engine import build_engine
from .ocr.pipeline import OcrPipeline, parse_pages
from .ocr.render import pdf_sha256
from .vault.note import write_note


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ra-ocr", description="Research Assistant OCR pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="OCR a PDF into a processed note")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--config", type=Path, default=None)
    ingest.add_argument("--pages", default=None, help="e.g. 1-3,7")
    ingest.add_argument("--engine", choices=["granite_docling", "native"], default=None)
    ingest.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps", "xpu"], default=None
    )
    ingest.add_argument(
        "--out", default=None, help="output dir (default: data/processed)"
    )
    ingest.add_argument(
        "--vault", default=None, help="Obsidian vault root (Resources/Parsed layout)"
    )
    ingest.add_argument("--no-write", action="store_true")
    ingest.add_argument("--no-figures", action="store_true")
    ingest.add_argument(
        "--figures-llm", action="store_true", help="interpret figures with a vision model"
    )
    ingest.add_argument("--vision-model", default=None)
    ingest.add_argument("--vision-base-url", default=None)
    ingest.add_argument("--critic", action="store_true", help="Tier 2 rubric critic")
    ingest.add_argument("--crossref", action="store_true", help="verify references")
    ingest.add_argument("--json", action="store_true", dest="as_json")

    batch = sub.add_parser("batch", help="OCR every PDF under a directory")
    batch.add_argument("dir", type=Path, nargs="?", default=None)
    batch.add_argument("--config", type=Path, default=None)
    batch.add_argument("--engine", choices=["granite_docling", "native"], default=None)
    batch.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps", "xpu"], default=None
    )
    batch.add_argument("--out", default=None)
    batch.add_argument("--no-write", action="store_true")
    batch.add_argument("--no-figures", action="store_true")
    batch.add_argument("--figures-llm", action="store_true")
    batch.add_argument("--vision-model", default=None)
    batch.add_argument("--vision-base-url", default=None)
    batch.add_argument("--critic", action="store_true")
    batch.add_argument("--crossref", action="store_true")
    return parser


def _summary(document) -> str:
    lines = [
        f"pdf      : {document.source_pdf}",
        f"sha256   : {document.sha256[:12]}",
        f"title    : {document.title}",
        f"pages    : {len(document.pages)}",
        f"status   : {document.status}",
        "",
        "page  route                 attempt  status        failed",
    ]
    for page in document.pages:
        failed = ",".join(page.review_reasons) or "-"
        lines.append(
            f"{page.page_no:>4}  {page.route.value:<20}  {page.attempt:>7}  {page.status:<12}  {failed}"
        )
    return "\n".join(lines)


def _ingest(args, settings: Settings) -> int:
    pipeline = OcrPipeline(
        settings=settings,
        engine=build_engine(settings.engine),
        vision=build_vision_client(settings.vision),
    )
    out = args.out or settings.data.processed_dir
    figures_dir = None
    if not args.no_figures:
        figures_dir = Path(out) / "figures" / pdf_sha256(args.pdf)[:12]
    document = pipeline.run(
        args.pdf, pages=parse_pages(args.pages), figures_dir=figures_dir
    )

    note_path = None
    if not args.no_write:
        if args.vault:
            note_path = write_note(
                document, args.vault, pipeline.engine.name, layout="vault"
            )
        else:
            note_path = write_note(document, out, pipeline.engine.name)

    if args.as_json:
        payload = document.model_dump(mode="json")
        payload["note_path"] = str(note_path) if note_path else None
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_summary(document))
        if note_path:
            print(f"\nnote     : {note_path}")
    return 0 if document.status == "verified" else 1


def _batch(args, settings: Settings) -> int:
    root = Path(args.dir or settings.data.raw_dir)
    pdfs = sorted(root.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs under {root}", file=sys.stderr)
        return 2
    out = args.out or settings.data.processed_dir
    pipeline = OcrPipeline(
        settings=settings,
        engine=build_engine(settings.engine),
        vision=build_vision_client(settings.vision),
    )
    failures = 0
    for pdf in pdfs:
        figures_dir = None
        if not args.no_figures:
            figures_dir = Path(out) / "figures" / pdf_sha256(pdf)[:12]
        document = pipeline.run(pdf, figures_dir=figures_dir)
        note_path = None if args.no_write else write_note(document, out, pipeline.engine.name)
        print(_summary(document))
        if note_path:
            print(f"note     : {note_path}\n")
        failures += document.status != "verified"
    print(f"{len(pdfs) - failures}/{len(pdfs)} verified")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_env_file()
    settings = Settings.load(args.config)
    if args.engine:
        settings.engine.name = args.engine
    if args.device:
        settings.engine.device = args.device
    if getattr(args, "figures_llm", False):
        settings.vision.enabled = True
    if getattr(args, "vision_model", None):
        settings.vision.model = args.vision_model
    if getattr(args, "vision_base_url", None):
        settings.vision.base_url = args.vision_base_url
    if getattr(args, "critic", False):
        settings.critic.enabled = True
        settings.vision.enabled = True
    if getattr(args, "crossref", False):
        settings.crossref.enabled = True
    if args.command == "batch":
        return _batch(args, settings)
    return _ingest(args, settings)


if __name__ == "__main__":
    sys.exit(main())
