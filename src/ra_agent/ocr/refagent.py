"""Agentic reference resolution.

The naive pipeline asked Crossref one question per entry and gave up when the
answer did not match. That produced a high mismatch rate for reasons that had
nothing to do with fabricated citations: OCR merges entries, titles get
confused with author lists, and a single bibliographic query is a weak probe.

This module turns reference checking into the same loop the OCR pipeline uses:
perceive the entry, act with one strategy, evaluate the result, reflect on the
failure, retry with a better strategy, and stop at a circuit breaker. Code owns
the loop and the decision; the LLM only proposes an identity when the regex
heuristics cannot isolate a title.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import ReferenceSettings
from ..llm.client import LlmClient, VisionError
from ..schema import Reference, ReferenceCheck
from .crossref import CrossrefClient

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_ARXIV = re.compile(r"\barxiv[:\s/]*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"\barxiv[:\s/]*([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_TRAILING_PUNCT = ".,;:)]}"

IDENTITY_PROMPT = """Extract the bibliographic identity of one reference entry.

Entry:
{entry}

Rules:
- Copy the title exactly as printed, without the author list.
- If a DOI or arXiv identifier is present, copy it verbatim.
- If a field is absent, use an empty string. Never guess.
- Answer with strict JSON only, no prose and no code fences:
{{"title": "", "authors": "", "year": "", "doi": "", "arxiv": ""}}
"""


@dataclass
class ReferenceIdentity:
    title: str = ""
    authors: str = ""
    year: str = ""
    doi: str = ""
    arxiv: str = ""
    source: str = "heuristic"

    @property
    def usable(self) -> bool:
        return bool(self.doi or self.arxiv or len(self.title) >= 12)


@dataclass
class ResolutionTrace:
    attempts: list[str] = field(default_factory=list)
    network: int = 0
    best_mismatch: ReferenceCheck | None = None

    def add(self, strategy: str, outcome: str, network: bool = False) -> None:
        self.attempts.append(f"{strategy}: {outcome}")
        if network:
            self.network += 1

    def note_mismatch(self, check: ReferenceCheck) -> None:
        """Keep the strongest near miss so the final status stays informative."""
        if check.status != "mismatch":
            return
        if self.best_mismatch is None or check.score > self.best_mismatch.score:
            self.best_mismatch = check


def _clean(value: str) -> str:
    return value.strip().strip(_TRAILING_PUNCT).strip()


def extract_identity(reference: Reference, raw: str = "") -> ReferenceIdentity:
    """Regex-first identity extraction. Deterministic and free."""
    text = raw or reference.raw
    doi = _DOI.search(text)
    arxiv = _ARXIV.search(text) or _ARXIV_OLD.search(text)
    year = reference.year or (_YEAR.search(text).group(0) if _YEAR.search(text) else "")
    return ReferenceIdentity(
        title=reference.title,
        year=year,
        doi=_clean(doi.group(1)) if doi else "",
        arxiv=_clean(arxiv.group(1)) if arxiv else "",
        source="regex",
    )


def _parse_identity(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw or "", flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0:
        raise ValueError("no JSON object in response")
    candidate = cleaned[start : end + 1] if end > start else cleaned[start:]
    return json.loads(candidate)


def llm_identity(
    reference: Reference, client: LlmClient | None
) -> ReferenceIdentity | None:
    """Ask the model for a title when the heuristics only produced an author list."""
    if client is None:
        return None
    try:
        data = _parse_identity(client.complete(IDENTITY_PROMPT.format(entry=reference.raw[:1200])))
    except (VisionError, ValueError, json.JSONDecodeError):
        return None
    identity = ReferenceIdentity(
        title=_clean(str(data.get("title", "")))[:220],
        authors=_clean(str(data.get("authors", "")))[:200],
        year=_clean(str(data.get("year", "")))[:4],
        doi=_clean(str(data.get("doi", "")))[:120],
        arxiv=_clean(str(data.get("arxiv", "")))[:40],
        source="llm",
    )
    return identity if identity.usable else None


class ReferenceAgent:
    """Resolve one reference with a strategy ladder and a circuit breaker."""

    def __init__(
        self,
        crossref: CrossrefClient,
        settings: ReferenceSettings | None = None,
        client: LlmClient | None = None,
    ) -> None:
        self.crossref = crossref
        self.settings = settings or ReferenceSettings()
        self.client = client

    def _accept(self, check: ReferenceCheck, strategy: str, trace: ResolutionTrace) -> ReferenceCheck:
        trace.add(strategy, f"{check.status} score={check.score}", network=True)
        return check.model_copy(update={"strategy": strategy, "attempts": trace.attempts})

    @staticmethod
    def _scored_reference(identity: ReferenceIdentity, reference: Reference) -> Reference:
        """Score against the identity we resolved, not the raw OCR title."""
        return reference.model_copy(
            update={
                "title": identity.title or reference.title,
                "year": identity.year or reference.year,
            }
        )

    def _try_doi(self, identity: ReferenceIdentity, reference: Reference, trace: ResolutionTrace) -> ReferenceCheck | None:
        if not identity.doi:
            trace.add("doi", "no identifier")
            return None
        item = self.crossref.lookup_doi(identity.doi)
        if not item:
            trace.add("doi", "no record", network=True)
            return None
        check = self.crossref.evaluate(self._scored_reference(identity, reference), item)
        if check.status == "verified":
            return self._accept(check, "doi", trace)
        trace.add("doi", f"{check.status} score={check.score}", network=True)
        trace.note_mismatch(check)
        return None

    def _try_arxiv(self, identity: ReferenceIdentity, reference: Reference, trace: ResolutionTrace) -> ReferenceCheck | None:
        if not identity.arxiv:
            trace.add("arxiv", "no identifier")
            return None
        item = self.crossref.lookup_doi(f"10.48550/arXiv.{identity.arxiv}")
        if not item:
            trace.add("arxiv", "no record", network=True)
            return None
        check = self.crossref.evaluate(self._scored_reference(identity, reference), item)
        if check.status == "verified":
            return self._accept(check, "arxiv", trace)
        trace.add("arxiv", f"{check.status} score={check.score}", network=True)
        trace.note_mismatch(check)
        return None

    def _try_query(
        self,
        query: str,
        strategy: str,
        identity: ReferenceIdentity,
        reference: Reference,
        trace: ResolutionTrace,
    ) -> ReferenceCheck | None:
        if len(query) < 12:
            trace.add(strategy, "query too short")
            return None
        items = self.crossref.search(query[:220])
        if not items:
            trace.add(strategy, "no hits", network=True)
            return None
        scored = self._scored_reference(identity, reference)
        best: tuple[float, dict] | None = None
        for item in items:
            titles = item.get("title") or []
            candidate = titles[0] if titles else ""
            if not candidate:
                continue
            score = self.crossref.score_title(scored.title, candidate)
            if best is None or score > best[0]:
                best = (score, item)
        if best is None:
            trace.add(strategy, "no titles in hits", network=True)
            return None
        check = self.crossref.evaluate(scored, best[1])
        if check.status == "verified":
            return self._accept(check, strategy, trace)
        trace.add(strategy, f"{check.status} score={check.score}", network=True)
        trace.note_mismatch(check)
        return None

    def resolve(self, reference: Reference) -> ReferenceCheck:
        trace = ResolutionTrace()
        identity = extract_identity(reference)
        budget = max(1, self.settings.max_attempts)

        strategies: list = []
        if identity.doi:
            strategies.append(lambda: self._try_doi(identity, reference, trace))
        if identity.arxiv:
            strategies.append(lambda: self._try_arxiv(identity, reference, trace))
        strategies.append(
            lambda: self._try_query(identity.title, "title", identity, reference, trace)
        )
        strategies.append(
            lambda: self._try_query(
                f"{identity.title} {identity.year}".strip(),
                "title_year",
                identity,
                reference,
                trace,
            )
        )

        for strategy in strategies[:budget]:
            result = strategy()
            if result is not None:
                return result

        if trace.network < budget and self.client is not None:
            proposed = llm_identity(reference, self.client)
            if proposed is None:
                trace.add("llm_identity", "no usable identity")
            else:
                identity = proposed
                trace.add("llm_identity", f"title={identity.title[:60]!r}")
                for strategy in (
                    lambda: self._try_doi(identity, reference, trace),
                    lambda: self._try_query(
                        identity.title, "llm_title", identity, reference, trace
                    ),
                ):
                    if trace.network >= budget:
                        break
                    result = strategy()
                    if result is not None:
                        return result

        if trace.best_mismatch is not None:
            if trace.best_mismatch.score >= self.settings.mismatch_floor:
                return trace.best_mismatch.model_copy(
                    update={"strategy": "exhausted", "attempts": trace.attempts}
                )
            trace.add(
                "floor",
                f"best score {trace.best_mismatch.score} below {self.settings.mismatch_floor}",
            )

        status = "skipped" if not identity.usable else "unresolved"
        detail = (
            "no usable identity after every strategy"
            if status == "skipped"
            else "no strategy produced a verified match"
        )
        return ReferenceCheck(
            reference=reference,
            status=status,
            detail=detail,
            strategy="exhausted",
            attempts=trace.attempts,
        )


def resolve_references(
    references: list[Reference],
    crossref: CrossrefClient,
    settings: ReferenceSettings | None = None,
    client: LlmClient | None = None,
) -> list[ReferenceCheck]:
    agent = ReferenceAgent(crossref, settings, client)
    return [agent.resolve(reference) for reference in references]
