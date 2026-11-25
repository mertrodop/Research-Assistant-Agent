"""Document level evaluator-optimizer (slide 6, right half).

    Draft Synthesizer -> Evaluator (OCR, Citation, DOI, Author, Venue) <-> arXiv / Crossref
          ^                                   |
          +------- JSON fix list, iter < 3 ---+   -> verified note | #needs-review

The draft is the paper's identity as read from page 1. The evaluator compares it
with an external record and returns a fix list; the optimizer applies the fixes
and evaluates again, because a fixed field can unlock a better lookup (a DOI found
on arXiv leads to the Crossref record that carries the venue). Code owns the loop,
the comparison and the stop; external records are the only source of a fix.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET  # trusted endpoint; expat >= 2.4.1 blocks entity bombs
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from ..config import MetaSettings
from ..schema import CheckResult, OcrDocument, PaperMeta, Severity
from .crossref import CrossrefClient
from .engine import NativeTextEngine

_ARXIV_ID = re.compile(r"(?:arxiv[:\s/]*)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NAME = re.compile(r"^[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-.]*){1,3}$")
_ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
ARXIV_API = "https://export.arxiv.org/api/query"


def _surname(name: str) -> str:
    return name.replace(".", " ").split()[-1].lower() if name.split() else ""


def _guess_authors(page_one: str, title: str) -> list[str]:
    """Author lines sit between the title and the abstract on page 1."""
    lines = [line.strip() for line in page_one.splitlines() if line.strip()]
    start = next((i for i, line in enumerate(lines) if title and line in title), 0)
    names: list[str] = []
    for line in lines[start + 1 : start + 8]:
        if re.match(r"^(abstract|introduction|1\.?\s)", line, re.I):
            break
        for part in re.split(r",|\band\b|\*|†|\d", line):
            part = part.strip()
            if _NAME.match(part) and len(part.split()) <= 4:
                names.append(part)
    return names[:40]


def _text_layer_head(document: OcrDocument) -> str:
    try:
        return NativeTextEngine().ocr_page(Path(document.source_pdf), 1)[0][:4000]
    except Exception:  # missing or unreadable PDF: no extra evidence, not an error
        return ""


def draft_meta(document: OcrDocument) -> PaperMeta:
    """Generator: read the paper identity from page 1, no network."""
    page_one = document.pages[0].markdown if document.pages else ""
    head = page_one[:4000]
    # arXiv stamp on page 1 (Docling drops the margin stamp, the text layer keeps it),
    # else the file name
    stamp = r"arxiv[:\s/]*(\d{4}\.\d{4,5})"
    arxiv = (
        re.search(stamp, head, re.I)
        or re.search(stamp, _text_layer_head(document), re.I)
        or _ARXIV_ID.fullmatch(Path(document.source_pdf).stem)
    )
    doi = _DOI.search(head)
    year = _YEAR.search(head)
    return PaperMeta(
        title=document.title,
        authors=_guess_authors(page_one, document.title),
        year=year.group(0) if year else "",
        doi=doi.group(1).rstrip(".,;)") if doi else "",
        arxiv_id=arxiv.group(1) if arxiv else "",
    )


_last_call = [0.0]


def _arxiv_entries(params: dict, timeout_s: float) -> list[ET.Element] | None:
    """arXiv asks clients to wait 3 s between calls; None means the call failed."""
    wait = 3.0 - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()
    try:
        response = httpx.get(ARXIV_API, params=params, timeout=timeout_s)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError):
        return None
    return [e for e in root.findall("a:entry", _ATOM) if e.find("a:title", _ATOM) is not None]


def _cached(cache: Path, produce) -> object:
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    value = produce()
    if value is not None:  # failures are not cached, the next run retries
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def fetch_arxiv(arxiv_id: str, cache_dir: str | Path, timeout_s: float = 20.0) -> dict | None:
    def produce():
        entries = _arxiv_entries({"id_list": arxiv_id}, timeout_s)
        return _record(entries[0]) if entries else None

    return _cached(Path(cache_dir) / f"arxiv-{arxiv_id}.json", produce)


def search_arxiv(title: str, cache_dir: str | Path, timeout_s: float = 20.0) -> list[dict]:
    """Title search: ML venues (PMLR, OpenReview) have no DOI, but most are on arXiv."""
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", title))[:200]
    if len(phrase) < 12:
        return []
    slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower())[:80].strip("-")

    def produce():
        entries = _arxiv_entries(
            {"search_query": f'ti:"{phrase}"', "max_results": "3"}, timeout_s
        )
        return None if entries is None else [_record(e) for e in entries]

    return _cached(Path(cache_dir) / f"arxiv-ti-{slug}.json", produce) or []


def _record(entry: ET.Element) -> dict:
    def text(path: str) -> str:
        node = entry.find(path, _ATOM)
        return " ".join((node.text or "").split()) if node is not None else ""

    category = entry.find("x:primary_category", _ATOM)
    return {
        "title": text("a:title"),
        "authors": [
            " ".join(author.findtext("a:name", "", _ATOM).split())
            for author in entry.findall("a:author", _ATOM)
        ],
        "year": text("a:published")[:4],
        "doi": text("x:doi"),
        "venue": text("x:journal_ref")
        or (f"arXiv {category.get('term')}" if category is not None else "arXiv"),
        "source": "arxiv",
    }


def _from_crossref(item: dict) -> dict:
    parts = item.get("issued", {}).get("date-parts") or [[None]]
    return {
        "title": (item.get("title") or [""])[0],
        "authors": [
            f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])
        ],
        "year": str(parts[0][0]) if parts and parts[0] and parts[0][0] else "",
        "doi": item.get("DOI", ""),
        "venue": (item.get("container-title") or [""])[0],
        "source": "crossref",
    }


def lookup(meta: PaperMeta, crossref: CrossrefClient) -> dict | None:
    """Extrinsic evidence. arXiv id first, then DOI, then a title search."""
    record = fetch_arxiv(meta.arxiv_id, crossref.cache_dir, crossref.timeout_s) if meta.arxiv_id else None
    if meta.doi and not meta.doi.lower().startswith("10.48550/"):
        item = crossref.lookup_doi(meta.doi)
        if item:
            published = _from_crossref(item)
            if record:  # arXiv knows the preprint, Crossref knows the venue
                record = {**record, "venue": published["venue"] or record["venue"], "doi": meta.doi}
            else:
                record = published
    if record is None and len(meta.title) >= 12:
        hits = crossref.search(meta.title)
        best = max(hits, key=lambda i: crossref.score_title(meta.title, (i.get("title") or [""])[0]), default=None)
        if best and crossref.score_title(meta.title, (best.get("title") or [""])[0]) >= crossref.min_score:
            record = _from_crossref(best)
    if record is None and len(meta.title) >= 12:
        hits = crossref.search_arxiv(meta.title)
        best = max(hits, key=lambda h: crossref.score_title(meta.title, h["title"]), default=None)
        if best and crossref.score_title(meta.title, best["title"]) >= crossref.min_score:
            record = best
    return record


def evaluate(meta: PaperMeta, record: dict, settings: MetaSettings) -> list[dict]:
    """Evaluator: compare draft and record, return a JSON fix list."""
    fixes: list[dict] = []
    score = fuzz.token_set_ratio(meta.title.lower(), record["title"].lower()) / 100
    if record["title"] and score < settings.min_title_score:
        fixes.append({"field": "title", "value": record["title"], "reason": f"title similarity {score:.2f}"})
    wanted = {_surname(name) for name in record["authors"]} - {""}
    have = {_surname(name) for name in meta.authors}
    overlap = len(wanted & have) / len(wanted) if wanted else 1.0
    if overlap < settings.min_author_overlap:
        fixes.append({"field": "authors", "value": record["authors"], "reason": f"author overlap {overlap:.2f}"})
    if record["year"] and meta.year != record["year"]:
        fixes.append({"field": "year", "value": record["year"], "reason": f"year {meta.year or '-'} vs {record['year']}"})
    if record["doi"] and meta.doi.lower() != record["doi"].lower():
        fixes.append({"field": "doi", "value": record["doi"], "reason": "doi missing or different"})
    if record["venue"] and meta.venue != record["venue"]:
        fixes.append({"field": "venue", "value": record["venue"], "reason": "venue from record"})
    return fixes


def ocr_pages_check(document: OcrDocument) -> CheckResult:
    review = document.review_pages
    return CheckResult(
        name="ocr_pages",
        passed=not review,
        severity=Severity.BLOCKER,
        detail=f"{len(review)} pages need review" if review else "all pages verified",
        evidence=", ".join(map(str, review[:20])),
    )


def optimize(
    document: OcrDocument, crossref: CrossrefClient, settings: MetaSettings
) -> tuple[PaperMeta, list[CheckResult], int]:
    """Run draft -> evaluate -> fix until the fix list is empty or iter == max_iter."""
    meta = draft_meta(document)
    trace: list[str] = []
    record = None
    iteration = 0
    converged = False
    for iteration in range(1, settings.max_iter + 1):
        record = lookup(meta, crossref)
        if record is None:
            trace.append(f"iter {iteration}: no external record")
            break
        fixes = evaluate(meta, record, settings)
        if not fixes:
            trace.append(f"iter {iteration}: draft matches {record['source']}")
            converged = True
            break
        trace.append(f"iter {iteration}: " + "; ".join(f"{f['field']} ({f['reason']})" for f in fixes))
        meta = meta.model_copy(update={f["field"]: f["value"] for f in fixes})
    meta = meta.model_copy(update={"source": record["source"] if record else ""})

    checks = [
        CheckResult(
            name="identity",
            passed=converged,  # iter == max_iter without convergence -> #needs-review
            severity=Severity.BLOCKER,
            detail=" | ".join(trace),
            evidence=meta.arxiv_id or meta.doi or meta.title[:80],
        )
    ]
    checks.append(ocr_pages_check(document))
    judged = [r for r in document.references if r.status != "skipped"]
    bad = document.unresolved_references
    ratio = len(bad) / len(judged) if judged else 0.0
    checks.append(
        CheckResult(
            name="citations",
            passed=ratio <= settings.max_unresolved_refs,
            severity=Severity.WARN,
            detail=f"{len(bad)}/{len(judged)} checked references unresolved or mismatched ({ratio:.2f})",
            evidence="; ".join(r.reference.raw[:60] for r in bad[:3]),
        )
    )
    return meta, checks, iteration
