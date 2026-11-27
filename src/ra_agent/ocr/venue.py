"""Venue and DOI of the published version of a paper.

arXiv's journal_ref is usually empty, so the draft ends up with "arXiv cs.CL" even for an
ICLR paper. OpenAlex knows the published venue, the abstract and the citation count: it is
looked up by arXiv id when there is one, else by title. The first record with a venue wins. Only when none has a venue does a web-searching model look it up, and its
answer counts only with a source URL taken from the search results.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from ..llm.client import VisionError
from ..schema import PaperMeta

OPENALEX_API = "https://api.openalex.org/works"
FIELDS = "title,publication_year,doi,cited_by_count,abstract_inverted_index,primary_location,locations"
_PREPRINT = re.compile(r"^(arxiv|corr)\b", re.I)
_JSON = re.compile(r"\{.*\}", re.S)


def _cached(cache: Path, produce):
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    value = produce()
    if value is not None:  # failures are not cached, the next run retries
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def _get(url: str, params: dict, timeout_s: float) -> dict | None:
    """None on any failure: the lookup then simply finds nothing."""
    try:
        response = httpx.get(url, params=params, timeout=timeout_s)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def _abstract(index: dict | None) -> str:
    """OpenAlex stores the abstract as {word: [positions]}."""
    words = sorted((pos, word) for word, positions in (index or {}).items() for pos in positions)
    return " ".join(word for _, word in words)


def _record(work: dict) -> dict:
    names = [((loc or {}).get("source") or {}).get("display_name") or "" for loc in [work.get("primary_location")] + (work.get("locations") or [])]
    venue = next((n for n in names if n and not _PREPRINT.match(n)), names[0] if names and names[0] else "")
    return {
        "title": work.get("title") or "",
        "venue": venue,
        "year": str(work.get("publication_year") or ""),
        "doi": re.sub(r"^https?://doi\.org/", "", work.get("doi") or ""),
        "abstract": _abstract(work.get("abstract_inverted_index")),
        "citation_count": work.get("cited_by_count"),
        "source": "openalex",
    }


def lookup_arxiv(arxiv_id: str, cache_dir: Path, timeout_s: float) -> list[dict]:
    """The work with this arXiv id (its DataCite DOI), exact, no title matching."""
    def produce():
        work = _get(f"{OPENALEX_API}/doi:10.48550/arXiv.{arxiv_id}", {"select": FIELDS}, timeout_s)
        return None if work is None else [_record(work)]

    return _cached(cache_dir / f"openalex-arxiv-{arxiv_id}.json", produce) or []


def search_openalex(title: str, cache_dir: Path, timeout_s: float) -> list[dict]:
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", title))[:200]
    if len(phrase) < 12:
        return []

    def produce():
        data = _get(OPENALEX_API, {"search": phrase, "per-page": "3", "select": FIELDS}, timeout_s)
        return None if data is None else [_record(w) for w in data.get("results", [])]

    key = "openalex-" + hashlib.sha256(phrase.encode()).hexdigest()[:16]
    return _cached(cache_dir / f"{key}.json", produce) or []


def candidates(meta: PaperMeta, cache_dir: str | Path, timeout_s: float, min_title: float) -> list[dict]:
    """OpenAlex records for the paper: by arXiv id when known, else the title matches."""
    cache = Path(cache_dir)
    if meta.arxiv_id:
        return lookup_arxiv(meta.arxiv_id, cache, timeout_s)
    found = search_openalex(meta.title, cache, timeout_s)
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
    if cands:
        update.update({k: cands[0][k] for k in ("abstract", "citation_count") if cands[0].get(k) not in (None, "")})
    return meta.model_copy(update=update)
