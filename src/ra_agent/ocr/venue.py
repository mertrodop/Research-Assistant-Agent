"""Venue and DOI of the published version of a paper.

arXiv's journal_ref is usually empty, so the draft ends up with "arXiv cs.CL" even for an
ICLR paper. Semantic Scholar, OpenAlex and DBLP know the published venue; the first record
with one wins. Only when none has a venue does a web-searching model look it up, and its
answer counts only with a source URL taken from the search results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from ..llm.client import VisionError
from ..schema import PaperMeta

DBLP_API = "https://dblp.org/search/publ/api"
OPENALEX_API = "https://api.openalex.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,year,venue,externalIds,abstract,citationCount"
_PREPRINT = re.compile(r"^(arxiv|corr)\b", re.I)
_JSON = re.compile(r"\{.*\}", re.S)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _cached(cache: Path, produce):
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    value = produce()
    if value is not None:  # failures are not cached, the next run retries
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def _get(url: str, params: dict, timeout_s: float) -> dict | None:
    """None on any failure: S2 answers 429 without a key (set S2_API_KEY), DBLP may show a bot check."""
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if "semanticscholar" in url and os.environ.get("S2_API_KEY") else {}
    for attempt in range(2):  # one retry after 2 s: Semantic Scholar answers 429 now and then, even with a key
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=timeout_s)
            if response.status_code == 429 and attempt == 0:
                time.sleep(2)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None
    return None


def _s2_record(item: dict) -> dict:
    return {
        "title": item.get("title") or "",
        "venue": item.get("venue") or "",
        "year": str(item.get("year") or ""),
        "doi": (item.get("externalIds") or {}).get("DOI") or "",
        "source": "semanticscholar",
        "abstract": item.get("abstract") or "",
        "citation_count": item.get("citationCount"),
    }


def search_s2(meta: PaperMeta, cache_dir: Path, timeout_s: float) -> list[dict]:
    if meta.arxiv_id:
        url, params, key = f"{S2_API}/ARXIV:{meta.arxiv_id}", {"fields": S2_FIELDS}, f"s2-{meta.arxiv_id}"
    else:
        url, params = f"{S2_API}/search/match", {"query": meta.title, "fields": S2_FIELDS}
        key = "s2-ti-" + hashlib.sha256(meta.title.encode()).hexdigest()[:16]

    def produce():
        data = _get(url, params, timeout_s)
        if data is None:
            return None
        items = data.get("data") if "data" in data else [data]
        return [_s2_record(i) for i in items or [] if i.get("title")]

    return _cached(cache_dir / f"{key}.json", produce) or []


def search_dblp(title: str, cache_dir: Path, timeout_s: float) -> list[dict]:
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", title))[:200]
    if len(phrase) < 12:
        return []

    def produce():
        data = _get(DBLP_API, {"q": phrase, "format": "json", "h": "5"}, timeout_s)
        if data is None:
            return None
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        return [
            {
                "title": (h.get("info", {}).get("title") or "").rstrip("."),
                "venue": h.get("info", {}).get("venue") or "",
                "year": h.get("info", {}).get("year") or "",
                "doi": h.get("info", {}).get("doi") or "",
                "source": "dblp",
            }
            for h in hits
        ]

    key = "dblp-" + hashlib.sha256(phrase.encode()).hexdigest()[:16]
    return _cached(cache_dir / f"{key}.json", produce) or []


def search_openalex(title: str, cache_dir: Path, timeout_s: float) -> list[dict]:
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", title))[:200]
    if len(phrase) < 12:
        return []

    def produce():
        params = {"search": phrase, "per-page": "3", "select": "title,publication_year,doi,primary_location"}
        data = _get(OPENALEX_API, params, timeout_s)
        if data is None:
            return None
        return [
            {
                "title": w.get("title") or "",
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "year": str(w.get("publication_year") or ""),
                "doi": re.sub(r"^https?://doi\.org/", "", w.get("doi") or ""),
                "source": "openalex",
            }
            for w in data.get("results", [])
        ]

    key = "openalex-" + hashlib.sha256(phrase.encode()).hexdigest()[:16]
    return _cached(cache_dir / f"{key}.json", produce) or []


def candidates(meta: PaperMeta, cache_dir: str | Path, timeout_s: float, min_title: float) -> list[dict]:
    """Records whose title matches the paper's title; each may carry a venue and a DOI."""
    cache = Path(cache_dir)
    found = (
        search_s2(meta, cache, timeout_s)
        + search_dblp(meta.title, cache, timeout_s)
        + search_openalex(meta.title, cache, timeout_s)
    )
    return [r for r in found if fuzz.token_set_ratio(meta.title.lower(), r["title"].lower()) / 100 >= min_title]


def _web_venue(verifier, meta: PaperMeta, cache_dir: Path) -> tuple[str, str]:
    """Fallback when no record has a venue: a web-searching model looks it up. (venue, source url) or ('', '').

    The answer counts only with a source URL from the search; no citation, no venue.
    """
    prompt = f"""Find where this exact paper was formally published (conference, workshop or journal; not arXiv or another preprint server).
Title: {meta.title}
Authors: {", ".join(meta.authors[:6])}
Year of the preprint: {meta.year or "unknown"}
Use only a source page about THIS paper (publisher, proceedings, OpenReview, ACL Anthology, DBLP ...). A paper that only mentions or cites it does not count.
Answer with JSON only: {{"venue": "<name and year, e.g. ICLR 2025>", "url": "<the source page>"}}
If it was not published anywhere beyond a preprint, or you cannot find a source, answer {{"venue": "", "url": ""}}."""
    key = hashlib.sha256((verifier.model + prompt).encode()).hexdigest()[:24]

    def produce():
        try:
            text, urls = verifier.search(prompt)
        except VisionError:
            return None
        return {"reply": text, "urls": urls}

    found = _cached(cache_dir / f"venue-web-{key}.json", produce) or {}
    match = _JSON.search(found.get("reply", ""))
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return "", ""
    venue, url = str(answer.get("venue") or "").strip(), str(answer.get("url") or "").strip()
    cited = url in found.get("urls", []) or any(url and url.rstrip("/") == u.rstrip("/") for u in found.get("urls", []))
    return (venue, url) if venue and not _PREPRINT.match(venue) and cited else ("", "")


def refine(meta: PaperMeta, verifier, crossref, min_title: float) -> PaperMeta:
    """Fill the published venue and a DOI. Keeps the current venue when nothing is found."""
    cache = Path(crossref.cache_dir)
    cands = candidates(meta, cache, crossref.timeout_s, min_title)
    published = [c for c in cands if c["venue"] and not _PREPRINT.match(c["venue"].strip())]
    venue = published[0]["venue"] if published else ""
    if not venue and verifier is not None:
        venue, _ = _web_venue(verifier, meta, cache)
    update: dict = {}
    if venue:
        update["venue"] = venue
    if not meta.doi:
        dois = [c["doi"] for c in cands if c["doi"] and not c["doi"].lower().startswith("10.48550/")]
        update["doi"] = dois[0] if dois else (f"10.48550/arXiv.{meta.arxiv_id}" if meta.arxiv_id else "")
    s2 = next((c for c in cands if c["source"] == "semanticscholar"), None)
    if s2:  # Semantic Scholar only: the other sources do not carry these
        update.update({k: s2[k] for k in ("abstract", "citation_count") if s2.get(k) not in (None, "")})
    return meta.model_copy(update=update)
