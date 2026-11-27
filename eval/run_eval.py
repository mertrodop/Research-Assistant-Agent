"""Evaluate the OCR pipeline on the dev papers in vault/Papers/*/raw/.

    python eval/run_eval.py gate              # Tier 1 gate: fault injection, offline
    python eval/run_eval.py ingest            # engines vs text layer, real ingest + UNIT_TESTS
    python eval/run_eval.py all               # gate + ingest

Results land in eval/results/<part>.json. Every part is deterministic and cached,
so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from ra_agent.config import Settings  # noqa: E402
from ra_agent.ocr.engine import NativeTextEngine  # noqa: E402
from ra_agent.ocr.gate import has_blockers, run_gate  # noqa: E402
from ra_agent.ocr.render import page_count, page_stats, pdf_sha256  # noqa: E402

DATA = ROOT / "eval" / "data"  # generated fixtures, git-ignored
RAW = DATA / "raw"  # copies of the dev PDFs: the vault moves its own on commit
for _pdf in (ROOT / "vault" / "Papers").glob("*/raw/*.pdf"):
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / _pdf.name).exists():
        shutil.copy2(_pdf, RAW / _pdf.name)
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


# --- Part 3: document level, references and paper metadata --------------------


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
    from unit_tests import run_unit_tests

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
    settings.paths.inbox_dir = str(RAW)  # the dev PDFs, wherever the vault keeps its own
    workspace = Workspace(settings)
    ingest_times = {}
    for pdf in sorted(RAW.glob("*.pdf")):
        started = time.perf_counter()
        result = workspace.ingest_paper(pdf.name)
        ingest_times[pdf.name] = round(time.perf_counter() - started, 1)
        if not result["ok"]:
            raise SystemExit(f"ingest failed: {result['error']}")
    tests = run_unit_tests(STAGING / "Papers")
    return {
        "engines": engines,
        "unit_tests": {k: tests[k] for k in ("passed", "total", "failed")},
        "ingest_seconds": ingest_times,
        "pages": rows,
    }


# --- Part 5: scanned pages, Granite on rasterized copies ----------------------


# --- Part 6: figure models ----------------------------------------------------

FIGURE_MODELS = [
    "qwen/qwen3-vl-8b-instruct",
    "google/gemma-4-31b-it:free",
    "google/gemini-2.5-flash",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["gate", "ingest", "all"])
    parser.add_argument("--n", type=int, default=3, help="pages per paper (ingest)")
    parser.add_argument("--dpi", type=int, default=SETTINGS.render.dpi)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["gate", "ingest"] if args.part == "all" else [args.part]
    for part in parts:
        if part == "gate":
            result = eval_gate()
        elif part == "ingest":
            result = eval_ingest(args.n)
        path = OUT / f"{part}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
