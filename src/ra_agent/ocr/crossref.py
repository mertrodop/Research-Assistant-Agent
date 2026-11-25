from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from ..schema import Reference, ReferenceCheck

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ENTRY_START = re.compile(r"^\s*(?:\[?\d{1,3}\]?\.?|[-*])\s+")
_REFERENCES = re.compile(r"^\s*(?:#{1,6}\s*)?references\s*$", re.I)
_CONTINUATION = re.compile(
    r"^(?:accessed|url|https?:|//|doi|arxiv|in proceedings|in proc\.|pp\.|vol\.)",
    re.I,
)
_NOISE = re.compile(r"^(?:table|figure|fig\.|algorithm|listing)\s*\d", re.I)
_APPENDIX = re.compile(r"^(?:appendix\b|[A-Z]\.\s+[A-Z][a-z]|[A-Z]\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$)")
_NEW_ENTRY = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z'\u2019\-]+,\s+[A-Z]\."  # Cai, C.
    r"|[A-Z][A-Za-z'\u2019\-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-z'\u2019\-]+,\s+[A-Z]\."  # Tom E. Hardwicke, Manuel
    r"|[A-Z][A-Za-z'\u2019\-]+\.\s"  # Anthropic.
    r"|[A-Z][a-z]+ [A-Z][a-z]+ and\s"  # Sheeba Samuel and
    r")"
)
_UNFINISHED = re.compile(r"(?:,|\band)\s*$")  # author list wraps onto the next line
API = "https://api.crossref.org/works"


def split_references(markdown: str) -> list[Reference]:
    """Collect bibliography entries from a References section.

    Handles both numbered entries and the blank-line separated style that OCR
    produces, where a wrapped entry continues on the next block.
    """
    lines = markdown.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _REFERENCES.match(line.strip())), None
    )
    if start is None:
        return []

    # Blank lines mark entries when the engine saw the layout (hanging indents,
    # list items); then name patterns would only split entries that wrap.
    layout = sum(1 for line in lines[start + 1 : start + 200] if not line.strip()) >= 3
    blocks: list[str] = []
    buffer: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s", stripped):  # a heading, not "#!/usr/bin/env"
            break
        unfinished = bool(buffer) and bool(_UNFINISHED.search(buffer[-1]))
        if _APPENDIX.match(stripped) and len(stripped) < 90 and not unfinished:
            break
        if not stripped:
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
            continue
        if _ENTRY_START.match(line) and buffer:
            blocks.append(" ".join(buffer))
            buffer = [stripped]
        elif (
            not layout
            and _NEW_ENTRY.match(stripped)
            and buffer
            and len(" ".join(buffer)) > 40
            and not _UNFINISHED.search(buffer[-1])
        ):
            blocks.append(" ".join(buffer))
            buffer = [stripped]
        else:
            buffer.append(stripped)
    if buffer:
        blocks.append(" ".join(buffer))

    entries: list[str] = []
    for block in blocks:
        if entries and (
            _CONTINUATION.match(block)
            or block[:1].islower()
            or _UNFINISHED.search(entries[-1])
        ):
            entries[-1] = f"{entries[-1]} {block}"
        else:
            entries.append(block)

    references: list[Reference] = []
    for entry in entries:
        cleaned = _ENTRY_START.sub("", entry).strip()
        if len(cleaned) < 25 or _NOISE.match(cleaned):
            continue
        year = _YEAR.search(cleaned)
        references.append(
            Reference(
                raw=cleaned,
                title=_guess_title(cleaned),
                year=year.group(0) if year else "",
            )
        )
    return references


def _guess_title(entry: str) -> str:
    """Pick the segment that reads like a title, not an author list.

    Author lists are comma heavy and full of initials, so commas and one-letter
    tokens are penalised; the longest remaining span usually is the title.
    """
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", entry) if part.strip()]
    if not parts:
        return entry[:200]

    def score(part: str) -> float:
        words = part.split()
        if not words:
            return -1e9
        commas = part.count(",")
        initials = sum(1 for word in words if len(word.strip(".,")) == 1)
        return len(words) - 2.5 * commas - 3.0 * initials

    best = max(parts, key=score)
    return best.strip(" .")[:220]


def _looks_like_authors(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    commas = text.count(",")
    initials = sum(1 for word in words if len(word.strip(".,")) == 1)
    return commas >= 3 or initials >= 2


class CrossrefClient:
    """Crossref lookup with an on-disk cache so reruns stay offline and cheap."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/crossref",
        timeout_s: float = 20.0,
        mailto: str = "",
        min_score: float = 0.82,
        year_tolerance: int = 1,
        rows: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_s = timeout_s
        self.mailto = mailto
        self.min_score = min_score
        self.year_tolerance = year_tolerance
        self.rows = rows

    def _cache_path(self, query: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower())[:80].strip("-")
        return self.cache_dir / f"{slug}.json"

    def _fetch(self, query: str) -> list[dict]:
        cache = self._cache_path(query)
        if cache.is_file():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        params = {"query.bibliographic": query, "rows": str(self.rows)}
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            response = httpx.get(API, params=params, timeout=self.timeout_s)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return []
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return items

    def search(self, query: str) -> list[dict]:
        """Bibliographic search. Kept separate so tests can stub the network."""
        return self._fetch(query)

    def lookup_doi(self, doi: str) -> dict | None:
        """Direct DOI lookup, the most reliable strategy when a DOI is present."""
        slug = re.sub(r"[^a-z0-9]+", "-", doi.lower())[:90].strip("-")
        cache = self.cache_dir / f"doi-{slug}.json"
        if cache.is_file():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        params = {"mailto": self.mailto} if self.mailto else {}
        try:
            response = httpx.get(
                f"{API}/{doi}", params=params, timeout=self.timeout_s
            )
            response.raise_for_status()
            item = response.json().get("message", {})
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return None
        if not item:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        return item

    def search_arxiv(self, title: str) -> list[dict]:
        """arXiv title search, shares this client's cache and timeout."""
        from .synth import search_arxiv  # local import: synth imports this module

        return search_arxiv(title, self.cache_dir, self.timeout_s)

    def score_title(self, reference_title: str, candidate: str) -> float:
        return fuzz.token_set_ratio(reference_title.lower(), candidate.lower()) / 100

    def evaluate(
        self, reference: Reference, item: dict
    ) -> ReferenceCheck:
        """Score one Crossref record against a reference and decide its status."""
        matched_title = (item.get("title") or [""])[0]
        parts = item.get("issued", {}).get("date-parts") or [[None]]
        matched_year = str(parts[0][0]) if parts and parts[0] and parts[0][0] else ""
        score = self.score_title(reference.title, matched_title) if matched_title else 0.0

        if score < self.min_score:
            return ReferenceCheck(
                reference=reference,
                status="mismatch",
                matched_title=matched_title,
                matched_year=matched_year,
                score=round(score, 3),
                detail=f"title similarity {score:.2f} below {self.min_score}",
            )
        if reference.year and matched_year:
            delta = abs(int(reference.year) - int(matched_year))
            if delta > self.year_tolerance:
                return ReferenceCheck(
                    reference=reference,
                    status="mismatch",
                    matched_title=matched_title,
                    matched_year=matched_year,
                    score=round(score, 3),
                    detail=f"year {reference.year} vs {matched_year}",
                )
        return ReferenceCheck(
            reference=reference,
            status="verified",
            matched_title=matched_title,
            matched_year=matched_year,
            score=round(score, 3),
        )

    def check(self, reference: Reference) -> ReferenceCheck:
        if not reference.title or len(reference.title) < 12:
            return ReferenceCheck(reference=reference, status="skipped", detail="title too short")
        if _looks_like_authors(reference.title):
            return ReferenceCheck(
                reference=reference,
                status="skipped",
                detail="could not isolate a title from the author list",
            )
        items = self.search(reference.title[:220])
        if not items:
            return ReferenceCheck(reference=reference, status="unresolved", detail="no crossref hit")

        best: tuple[float, dict] | None = None
        for item in items:
            titles = item.get("title") or []
            candidate = titles[0] if titles else ""
            if not candidate:
                continue
            score = self.score_title(reference.title, candidate)
            if best is None or score > best[0]:
                best = (score, item)
        if best is None:
            return ReferenceCheck(reference=reference, status="unresolved", detail="no title in hits")
        return self.evaluate(reference, best[1])


def check_references(
    markdown: str, client: CrossrefClient | None = None
) -> list[ReferenceCheck]:
    references = split_references(markdown)
    if not references:
        return []
    client = client or CrossrefClient()
    return [client.check(reference) for reference in references]
