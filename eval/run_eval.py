"""Evaluate the OCR pipeline on the papers under data/raw.

    python eval/run_eval.py gate              # Tier 1 gate: fault injection, offline
    python eval/run_eval.py critic --n 4      # Tier 2 critic, image vs text-only (API key)
    python eval/run_eval.py granite --n 3     # Granite-Docling vs text layer (slow on CPU)
    python eval/run_eval.py document          # references + paper metadata, needs network
    python eval/run_eval.py ingest            # engines vs text layer, real ingest + UNIT_TESTS
    python eval/run_eval.py scanned --n 3     # rasterized pages through Granite (very slow on CPU)
    python eval/run_eval.py figures           # figure models: JSON validity, grounding, latency
    python eval/run_eval.py all

Results land in eval/results/<part>.json. Every part is deterministic and cached,
so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from ra_agent.config import Settings  # noqa: E402
from ra_agent.ocr.engine import GraniteDoclingEngine, NativeTextEngine  # noqa: E402
from ra_agent.schema import Route  # noqa: E402
from ra_agent.ocr.gate import has_blockers, run_gate  # noqa: E402
from ra_agent.ocr.render import page_count, page_stats, pdf_sha256  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "eval" / "results"
CACHE = ROOT / ".cache" / "eval"
SETTINGS = Settings.load(ROOT / "config" / "pipeline.yaml")


def words(text: str) -> list[str]:
    text = re.sub(r"<!--.*?-->", " ", text)
    return re.findall(r"\w+", text.lower())


def text_metrics(hypothesis: str, reference: str) -> dict:
    """WER is order sensitive (two-column layouts hurt it); bag-of-words F1 is not."""
    hyp, ref = words(hypothesis), words(reference)
    common = sum((Counter(hyp) & Counter(ref)).values())
    precision = common / len(hyp) if hyp else 0.0
    recall = common / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return {
        "wer": round(Levenshtein.distance(hyp, ref) / max(1, len(ref)), 3),
        "bow_precision": round(precision, 3),
        "bow_recall": round(recall, 3),
        "bow_f1": round(f1, 3),
        "hyp_words": len(hyp),
        "ref_words": len(ref),
    }


def gate_verdict(markdown: str, pdf: Path, page: int, text_layer: str) -> dict:
    stats = page_stats(pdf, page, ink_threshold=SETTINGS.render.ink_threshold)
    checks = run_gate(
        markdown,
        SETTINGS.gate,
        text_layer_chars=stats.text_layer_chars,
        text_layer=text_layer,
        ink_ratio=stats.ink_ratio,
    )
    return {
        "blocked": has_blockers(checks),
        "failed": [c.name for c in checks if not c.passed],
    }


# --- Part 1: Tier 1 gate, fault injection -------------------------------------
# Each operator imitates a failure mode a VLM OCR engine actually shows.


def _truncate(text: str, rng: random.Random) -> str:
    return text[: len(text) // 5]


def _loop(text: str, rng: random.Random) -> str:
    tokens = text.split()
    start = rng.randrange(max(1, len(tokens) - 12))
    return " ".join(tokens[: start + 12] + tokens[start : start + 12] * 10)


def _garbage(text: str, rng: random.Random) -> str:
    chars = list(text)
    for index in rng.sample(range(len(chars)), k=len(chars) // 10):
        chars[index] = rng.choice("\ufffd\u25a1\u2592\x07")
    return "".join(chars)


def _drop_numbers(text: str, rng: random.Random) -> str:
    return re.sub(r"\d+(?:[.,]\d+)?", "", text)


def _alter_numbers(text: str, rng: random.Random) -> str:
    return re.sub(r"\d", lambda m: str((int(m.group(0)) + 3) % 10), text)


def _alter_one_number(text: str, rng: random.Random) -> str:
    """The realistic case: one result value misread, e.g. 21.0 -> 27.0."""
    numbers = list(re.finditer(r"\b\d+\.\d+\b", text)) or list(re.finditer(r"\b\d{2,}\b", text))
    if not numbers:
        return text
    match = numbers[len(numbers) // 2]
    value = match.group(0)
    changed = value[:-1] + str((int(value[-1]) + 6) % 10)
    return text[: match.start()] + changed + text[match.end() :]


def _empty(text: str, rng: random.Random) -> str:
    return ""


def _header_lost(text: str, rng: random.Random) -> str:
    return text + "\n\n| 1 | 0.91 | 0.88 |\n|---|---|---|\n| 2 | 0.85 | 0.80 |\n"


def _unbalanced_math(text: str, rng: random.Random) -> str:
    return text + "\n\n$$ L = \\sum_i \\ell(x_i)\n"


def _hallucinated_sentence(text: str, rng: random.Random) -> str:
    return text + "\n\nOur method improves accuracy by a large margin on every benchmark."


OPERATORS = {
    "truncate_80pct": _truncate,
    "repetition_loop": _loop,
    "garbage_chars_10pct": _garbage,
    "drop_numbers": _drop_numbers,
    "alter_numbers": _alter_numbers,
    "alter_one_number": _alter_one_number,
    "empty": _empty,
    "table_header_lost": _header_lost,
    "unbalanced_math": _unbalanced_math,
    "hallucinated_sentence": _hallucinated_sentence,
}


def eval_gate() -> dict:
    native = NativeTextEngine()
    rng = random.Random(0)
    clean_blocked, clean_warned, pages = 0, 0, 0
    per_operator = {name: {"blocked": 0, "flagged": 0, "n": 0} for name in OPERATORS}
    for pdf in sorted(RAW.glob("*.pdf")):
        for page in range(1, page_count(pdf) + 1):
            text, _ = native.ocr_page(pdf, page)
            if len(text) < 200:
                continue
            pages += 1
            clean = gate_verdict(text, pdf, page, text)
            clean_blocked += clean["blocked"]
            clean_warned += bool(clean["failed"])
            for name, operator in OPERATORS.items():
                verdict = gate_verdict(operator(text, rng), pdf, page, text)
                row = per_operator[name]
                row["n"] += 1
                row["blocked"] += verdict["blocked"]
                row["flagged"] += bool(verdict["failed"])
    for row in per_operator.values():
        row["block_rate"] = round(row["blocked"] / row["n"], 3)
        row["flag_rate"] = round(row["flagged"] / row["n"], 3)
    return {
        "pages": pages,
        "clean_false_block_rate": round(clean_blocked / pages, 3),
        "clean_any_flag_rate": round(clean_warned / pages, 3),
        "operators": per_operator,
    }


# --- Part 1b: Tier 2 critic, does it catch what the gate cannot? -------------


MODES = ("text_only", "image", "image+text_layer")


def _changed_token(text: str, corrupted: str) -> str:
    index = next((i for i, (a, b) in enumerate(zip(text, corrupted)) if a != b), -1)
    if index < 0:
        return ""
    match = re.search(r"[\d.]+", corrupted[max(0, index - 6) : index + 6])
    return match.group(0) if match else ""


INJECTED = {
    "hallucinated_sentence": lambda text, corrupted: "improves accuracy by a large margin",
    "alter_one_number": _changed_token,
}


class CachingClient:
    """Caches raw model replies by prompt + image, so code-side rules can change freely."""

    def __init__(self, client, root: Path) -> None:
        self.client, self.root, self.model = client, root, client.model

    def _cached(self, key: str, call) -> str:
        import hashlib

        path = self.root / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".txt")
        if path.is_file():
            return path.read_text(encoding="utf-8")
        reply = call()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reply, encoding="utf-8")
        return reply

    def complete(self, prompt: str) -> str:
        return self._cached(self.model + prompt, lambda: self.client.complete(prompt))

    def describe(self, image_path, prompt: str) -> str:
        key = self.model + str(image_path) + prompt
        return self._cached(key, lambda: self.client.describe(image_path, prompt))


def eval_critic(n_pages: int) -> dict:
    from ra_agent.config import load_env_file
    from ra_agent.llm.client import build_llm
    from ra_agent.ocr.critic import critique_page
    from ra_agent.ocr.render import render_page

    load_env_file(ROOT / ".env")
    settings = SETTINGS.model_copy(deep=True)
    client = CachingClient(build_llm(settings.models["vision"]), CACHE / "critic" / "replies")
    native = NativeTextEngine()
    variants = {
        "clean": lambda t, r: t,
        "hallucinated_sentence": _hallucinated_sentence,
        "alter_one_number": _alter_one_number,
    }
    rows = []
    for pdf in sorted(RAW.glob("*.pdf")):
        sha = pdf_sha256(pdf)[:12]
        pages = [p for p in range(2, page_count(pdf) + 1) if len(native.ocr_page(pdf, p)[0]) > 2500]
        for page in pages[:: max(1, len(pages) // n_pages)][:n_pages]:
            text, _ = native.ocr_page(pdf, page)
            image = CACHE / "critic" / f"{sha}_p{page:03d}.png"
            if not image.is_file():
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(render_page(pdf, page, settings.critic.dpi).png)
            for variant, operator in variants.items():
                corrupted = operator(text, random.Random(0))
                for mode in MODES:
                    check = critique_page(
                        corrupted,
                        client,
                        settings.critic,
                        page,
                        image_path=image if mode.startswith("image") else None,
                        text_layer=text if mode == "image+text_layer" else "",
                    )
                    result = check.model_dump(mode="json")
                    gate = gate_verdict(corrupted, pdf, page, text)
                    marker = INJECTED.get(variant, lambda t, c: "")(text, corrupted)
                    rows.append(
                        {
                            "pdf": pdf.name,
                            "page": page,
                            "variant": variant,
                            "mode": mode,
                            "critic_blocked": result["severity"] == "blocker" and not result["passed"],
                            "critic_error": result["severity"] == "warn",
                            # localized: the critic quoted the injected text, not a lucky miss
                            "critic_localized": bool(marker) and marker in result["evidence"],
                            "detail": result["detail"],
                            "gate_blocked": gate["blocked"],
                            "gate_flagged": bool(gate["failed"]),
                        }
                    )
                    print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for variant in variants:
        for mode in MODES:
            sel = [r for r in rows if r["variant"] == variant and r["mode"] == mode]
            summary[f"{variant}/{mode}"] = {
                "n": len(sel),
                "critic_block_rate": round(sum(r["critic_blocked"] for r in sel) / len(sel), 3),
                "critic_localized_rate": round(sum(r["critic_localized"] for r in sel) / len(sel), 3),
                "critic_errors": sum(r["critic_error"] for r in sel),
                "gate_block_rate": round(sum(r["gate_blocked"] for r in sel) / len(sel), 3),
                "gate_flag_rate": round(sum(r["gate_flagged"] for r in sel) / len(sel), 3),
            }
    return {"model": settings.models["vision"].model, "dpi": settings.critic.dpi, "summary": summary, "rows": rows}


# --- Part 2: Granite-Docling against the text layer ---------------------------


def sample_pages(pdf: Path, n: int) -> list[int]:
    """Title page, first page with a table caption, then evenly spaced body pages."""
    native = NativeTextEngine()
    total = page_count(pdf)
    texts = {p: native.ocr_page(pdf, p)[0] for p in range(1, total + 1)}
    chosen = [1]
    table = next((p for p, t in texts.items() if re.search(r"^Table \d", t, re.M)), None)
    if table:
        chosen.append(table)
    step = max(1, total // (n + 1))
    for page in range(step, total + 1, step):
        if len(chosen) >= n:
            break
        if page not in chosen and len(texts[page]) > 500:
            chosen.append(page)
    return sorted(chosen[:n])


def eval_granite(n: int, dpi: int) -> dict:
    engine = GraniteDoclingEngine(SETTINGS.engine.model_copy(update={"device": "cpu"}))
    native = NativeTextEngine()
    rows = []
    for pdf in sorted(RAW.glob("*.pdf")):
        sha = pdf_sha256(pdf)[:12]
        for page in sample_pages(pdf, n):
            cache = CACHE / "granite" / f"{sha}_p{page:03d}_{dpi}.json"
            if cache.is_file():
                data = json.loads(cache.read_text(encoding="utf-8"))
            else:
                started = time.perf_counter()
                try:
                    markdown, _ = engine.ocr_page(pdf, page, dpi)
                    error = ""
                except Exception as exc:  # noqa: BLE001 - a failed page is a data point
                    markdown, error = "", str(exc)[:300]
                data = {
                    "markdown": markdown,
                    "error": error,
                    "latency_s": round(time.perf_counter() - started, 1),
                }
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            text_layer, _ = native.ocr_page(pdf, page)
            row = {"pdf": pdf.name, "page": page, "latency_s": data["latency_s"]}
            row.update(text_metrics(data["markdown"], text_layer))
            row.update(gate_verdict(data["markdown"], pdf, page, text_layer))
            row["error"] = data["error"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"dpi": dpi, "device": "cpu", "pages": rows}


# --- Part 3: document level, references and paper metadata --------------------


def eval_document() -> dict:
    from ra_agent.ocr.pipeline import OcrPipeline

    settings = Settings.load(ROOT / "config" / "pipeline.yaml")
    settings.crossref.enabled = True
    settings.figures.enabled = False
    pipeline = OcrPipeline(
        settings=settings,
        engines={r: NativeTextEngine() for r in Route},  # text layer only, offline
    )
    papers = []
    for pdf in sorted(RAW.glob("*.pdf")):
        document = pipeline.run(pdf)
        statuses = Counter(ref.status for ref in document.references)
        strategies = Counter(
            ref.strategy for ref in document.references if ref.status == "verified"
        )
        papers.append(
            {
                "pdf": pdf.name,
                "status": document.status,
                "references": len(document.references),
                "reference_status": dict(statuses),
                "verified_by_strategy": dict(strategies),
                "meta": document.meta.model_dump() if document.meta else None,
                "doc_checks": [c.model_dump(mode="json") for c in document.doc_checks],
                "optimizer_iterations": document.meta_iterations,
            }
        )
        print(json.dumps(papers[-1], ensure_ascii=False)[:600], flush=True)
    return {"papers": papers}


# --- Part 4: ingest, engines and unit tests on real notes ----------------------

STAGING = OUT / "staging"  # separate from the user's staging/


def _page_row(pdf: Path, page: int, markdown: str, latency: float, error: str = "") -> dict:
    text_layer = NativeTextEngine().ocr_page(pdf, page)[0]
    row = {"pdf": pdf.name, "page": page, "latency_s": latency, "error": error}
    row.update(text_metrics(markdown, text_layer))
    row.update(gate_verdict(markdown, pdf, page, text_layer))
    return row


def eval_ingest(n: int) -> dict:
    from ra_agent.config import load_env_file
    from ra_agent.ocr.engine import DoclingStandardEngine
    from ra_agent.tools import Workspace
    import unit_tests
    from unit_tests import UNIT_TESTS

    load_env_file(ROOT / ".env")
    docling, native = DoclingStandardEngine(SETTINGS.engine), NativeTextEngine()
    granite_cache = CACHE / "granite"
    rows = []
    for pdf in sorted(RAW.glob("*.pdf")):
        sha = pdf_sha256(pdf)[:12]
        for page in sample_pages(pdf, n):
            cache = CACHE / "docling" / f"{sha}_p{page:03d}.json"
            if cache.is_file():
                data = json.loads(cache.read_text(encoding="utf-8"))
            else:
                started = time.perf_counter()
                markdown, _ = docling.ocr_page(pdf, page, SETTINGS.render.dpi)
                data = {"markdown": markdown, "latency_s": round(time.perf_counter() - started, 1)}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            rows.append({"engine": "docling", **_page_row(pdf, page, data["markdown"], data["latency_s"])})
            gcache = granite_cache / f"{sha}_p{page:03d}_{SETTINGS.render.dpi}.json"
            if gcache.is_file():  # written by the granite part; never recomputed here
                g = json.loads(gcache.read_text(encoding="utf-8"))
                rows.append({"engine": "granite", **_page_row(pdf, page, g["markdown"], g["latency_s"], g["error"])})
            started = time.perf_counter()
            text = native.ocr_page(pdf, page)[0]
            rows.append({"engine": "text_layer", **_page_row(pdf, page, text, round(time.perf_counter() - started, 3))})
            print(json.dumps(rows[-3:]), flush=True)
    engines = {}
    for name in ("docling", "granite", "text_layer"):
        sel = [r for r in rows if r["engine"] == name]
        if not sel:
            continue
        engines[name] = {
            "pages": len(sel),
            "mean_wer": round(sum(r["wer"] for r in sel) / len(sel), 3),
            "mean_bow_f1": round(sum(r["bow_f1"] for r in sel) / len(sel), 3),
            "gate_blocked": sum(r["blocked"] for r in sel),
            "mean_s_per_page": round(sum(r["latency_s"] for r in sel) / len(sel), 2),
        }
    # full ingest of both papers with the default routing, into a separate staging dir
    settings = SETTINGS.model_copy(deep=True)
    settings.paths.staging_dir = str(STAGING)
    settings.engine.device = "cpu"  # the 2 GB GPU OOMs on Granite
    workspace = Workspace(settings)
    notes, ingest_times = {}, {}
    for pdf in sorted(RAW.glob("*.pdf")):
        started = time.perf_counter()
        result = workspace.ingest_paper(pdf.name)
        ingest_times[pdf.name] = round(time.perf_counter() - started, 1)
        if not result["ok"]:
            raise SystemExit(f"ingest failed: {result['error']}")
        notes[result["slug"]] = Path(result["handle"])
    unit_tests.NOTE_DIR = STAGING / "Papers"
    passed, failed = 0, []
    for slug, name, check in UNIT_TESTS:
        path = next((p for s, p in notes.items() if s.startswith(slug)), None)
        try:
            ok = path is not None and bool(check(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a crashing check is a failed check
            ok = False
        passed += ok
        if not ok:
            failed.append(f"{slug}:{name}")
    return {
        "engines": engines,
        "unit_tests": {"passed": passed, "total": len(UNIT_TESTS), "failed": failed},
        "ingest_seconds": ingest_times,
        "pages": rows,
    }


# --- Part 5: scanned pages, Granite on rasterized copies ----------------------


def make_scanned(pdf: Path, out: Path, dpi: int = 150) -> Path:
    """Image-only copy of `pdf`: every page is a raster, no text layer."""
    import fitz

    out.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as src, fitz.open() as dst:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            new = dst.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("jpeg"))
        dst.save(out)
    return out


def eval_scanned(n: int, dpi: int = 150) -> dict:
    from ra_agent.config import load_env_file
    from ra_agent.llm.client import build_llm
    from ra_agent.ocr.critic import critique_page
    from ra_agent.ocr.engine import build_engines
    from ra_agent.ocr.pipeline import OcrPipeline
    from ra_agent.ocr.render import render_page

    load_env_file(ROOT / ".env")
    settings = SETTINGS.model_copy(deep=True)
    settings.engine.device = "cpu"
    settings.critic.enabled = False  # the critic is run by hand below, on injected errors
    settings.pipeline.cache_dir = str(CACHE / "scanned_ocr")
    client = CachingClient(build_llm(settings.models["vision"]), CACHE / "scanned" / "replies")
    pipeline = OcrPipeline(settings, engines=build_engines(settings.engine), vision=client)
    rows = []
    for pdf in sorted(RAW.glob("*.pdf")):
        scan = make_scanned(pdf, ROOT / "data" / "eval" / f"{pdf.stem}-scanned.pdf", dpi)
        sha = pdf_sha256(scan)
        for page in sample_pages(pdf, n):
            started = time.perf_counter()
            result = pipeline.process_page(scan, sha, page, Route.GRANITE)
            latency = round(time.perf_counter() - started, 1)  # 0-ish when cached by the pipeline
            reference = NativeTextEngine().ocr_page(pdf, page)[0]
            row = {"pdf": pdf.name, "page": page, "latency_s": latency, "error": result.error}
            row.update(text_metrics(result.markdown, reference))
            row["gate_blocked"] = bool(result.blockers)
            row["gate_failed"] = [c.name for c in result.checks if not c.passed]
            image = CACHE / "scanned" / f"{pdf.stem[:12]}_p{page:03d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(render_page(scan, page, settings.critic.dpi).png)
            critic = {}
            for variant, operator in {
                "clean": lambda t, r: t,
                "hallucinated_sentence": _hallucinated_sentence,
                "alter_one_number": _alter_one_number,
            }.items():
                corrupted = operator(result.markdown, random.Random(0))
                check = critique_page(corrupted, client, settings.critic, page, image_path=image)
                critic[variant] = {
                    "blocked": check.severity.value == "blocker" and not check.passed,
                    "detail": check.detail[:160],
                }
            row["critic_image"] = critic
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"dpi": dpi, "device": "cpu", "pages": rows}


# --- Part 6: figure models ----------------------------------------------------

FIGURE_MODELS = [
    "qwen/qwen3-vl-8b-instruct",
    "google/gemma-4-31b-it:free",
    "google/gemini-2.5-flash",
]


def _figure_cases(per_paper: int = 3) -> list[tuple[str, object, str, Path]]:
    """Six captioned figures read from the real ingests: (paper, FigureRef, page text, dir)."""
    from ra_agent.schema import OcrDocument

    cases = []
    for path in sorted(STAGING.glob("*.json")):
        doc = OcrDocument.model_validate_json(path.read_text(encoding="utf-8"))
        figs = [(p, f) for p in doc.pages for f in p.figures if f.caption and f.kind != "page"]
        for page, fig in figs[:: max(1, len(figs) // per_paper)][:per_paper]:
            cases.append((path.stem, fig, page.markdown, STAGING / "figures" / doc.sha256[:12]))
    return cases


def eval_figures() -> dict:
    from ra_agent.config import VisionExtra, load_env_file
    from ra_agent.llm.client import build_llm
    from ra_agent.ocr.interpret import _numbers, describe_figure

    load_env_file(ROOT / ".env")
    cases = _figure_cases()
    rows = []
    for model in FIGURE_MODELS:
        settings = SETTINGS.models["vision"].model_copy(
            update={"model": model, "base_url": "https://openrouter.ai/api/v1"}
        )
        client = CachingClient(build_llm(settings), CACHE / "figures" / "replies")
        extra = VisionExtra(cache_dir=str(CACHE / "figures" / "readings"))
        for slug, fig, page_text, image_dir in cases:
            cache = CACHE / "figures" / "rows" / f"{re.sub(r'[^a-z0-9]+', '_', model)}__{slug[:12]}_{fig.index}.json"
            if cache.is_file():
                row = json.loads(cache.read_text(encoding="utf-8"))
            else:
                for attempt in range(4):  # free models rate-limit: back off, then record it
                    started = time.perf_counter()
                    reading = describe_figure(fig, image_dir, page_text, client, extra, image_dir.name).interpretation
                    latency = round(time.perf_counter() - started, 1)
                    if "429" not in reading.error and "HTTP 5" not in reading.error:
                        break
                    time.sleep(10 * 2**attempt)
                reported = _numbers(reading.summary) | _numbers(reading.markdown)
                row = {
                    "model": model,
                    "figure": f"{slug[:12]}:fig-{fig.index}",
                    "valid_json": not reading.error,
                    "error": reading.error,
                    "numbers": len(reported),
                    "ungrounded": len(reading.ungrounded),
                    "latency_s": latency,
                }
                if row["valid_json"] or "HTTP" not in row["error"]:  # do not freeze rate-limit failures
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(row), encoding="utf-8")
            rows.append(row)
            print(json.dumps(row), flush=True)
    models = {}
    for model in FIGURE_MODELS:
        sel = [r for r in rows if r["model"] == model]
        numbers = sum(r["numbers"] for r in sel)
        models[model] = {
            "figures": len(sel),
            "valid_json_rate": round(sum(r["valid_json"] for r in sel) / len(sel), 3),
            "http_failures": sum("HTTP" in r["error"] for r in sel),
            "grounded_number_rate": round(1 - sum(r["ungrounded"] for r in sel) / numbers, 3) if numbers else None,
            "numbers_reported": numbers,
            "mean_latency_s": round(sum(r["latency_s"] for r in sel) / len(sel), 1),
        }
    return {"models": models, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["gate", "critic", "granite", "document", "ingest", "scanned", "figures", "all"])
    parser.add_argument("--n", type=int, default=3, help="pages per paper (granite, critic)")
    parser.add_argument("--dpi", type=int, default=SETTINGS.render.dpi)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["gate", "document", "granite", "ingest", "scanned", "figures"] if args.part == "all" else [args.part]
    for part in parts:
        if part == "gate":
            result = eval_gate()
        elif part == "critic":
            result = eval_critic(args.n)
        elif part == "granite":
            result = eval_granite(args.n, args.dpi)
        elif part == "ingest":
            result = eval_ingest(args.n)
        elif part == "scanned":
            result = eval_scanned(args.n)
        elif part == "figures":
            result = eval_figures()
        else:
            result = eval_document()
        path = OUT / f"{part}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
