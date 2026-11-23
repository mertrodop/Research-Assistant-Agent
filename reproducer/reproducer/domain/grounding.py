"""Grounded string types for LLM output, checked against Pydantic validation context.

Pure: no IO. Same pattern as RunPlan: the facts to ground against are passed as
`model_validate*(..., context={...})`. Missing or malformed context fails closed with a
ValueError (surfaced as a ValidationError), never a silent pass.

    RepoPath     normalized posix relative path; must be in context["repo_files"]
    EvidenceRef  must be in context["evidence_ids"]
    Citation     {evidence_id: alias | EvidenceRef, quote}; the quote is snapped onto a verbatim
                 span of context["texts"][evidence_id] (see snap_quote); stored with full id
"""

from __future__ import annotations

import difflib
import math
import posixpath
import re
from collections.abc import Mapping, Sequence, Set
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo, model_validator

_MAX_SUGGESTIONS = 3


def _context_set(info: ValidationInfo, key: str, type_name: str) -> Set[str]:
    ctx: Any = info.context
    if not isinstance(ctx, Mapping) or key not in ctx:
        raise ValueError(f"{type_name} requires validation context with key {key!r}")
    values = ctx[key]
    if not isinstance(values, Set):
        raise ValueError(f"validation context {key!r} must be a set, got {type(values).__name__}")
    return values


def _suggest(value: str, choices: Set[str]) -> str:
    close = difflib.get_close_matches(value, sorted(choices), n=_MAX_SUGGESTIONS, cutoff=0.6)
    return f"; did you mean one of {close}?" if close else ""


def normalize_repo_path(value: str) -> str:
    """'./src/../train.py' -> 'train.py'. Rejects absolute paths and paths leaving the repo."""
    raw = value.strip()
    if not raw:
        raise ValueError("path is empty")
    if raw.startswith("/"):
        raise ValueError(f"path must be relative to the repo root, got {value!r}")
    norm = posixpath.normpath(raw)
    if norm == "." or norm == ".." or norm.startswith("../"):
        raise ValueError(f"path escapes the repo root: {value!r}")
    return norm


def _check_repo_path(value: str, info: ValidationInfo) -> str:
    repo_files = _context_set(info, "repo_files", "RepoPath")
    norm = normalize_repo_path(value)
    if norm not in repo_files:
        raise ValueError(f"file not in repo: {norm!r}{_suggest(norm, repo_files)}")
    return norm


def _check_evidence_ref(value: str, info: ValidationInfo) -> str:
    evidence_ids = _context_set(info, "evidence_ids", "EvidenceRef")
    if value not in evidence_ids:
        raise ValueError(f"unknown evidence id: {value!r}")
    return value


RepoPath = Annotated[str, AfterValidator(_check_repo_path)]
EvidenceRef = Annotated[str, AfterValidator(_check_evidence_ref)]


# --- Citation ------------------------------------------------------------------------------

CITATION_MAX_QUOTE_CHARS = 300
SNAP_MIN_RATIO = 0.92
SNAP_LENGTH_TOLERANCE = 0.15
_N_CANDIDATES = 3
_CANDIDATE_CHARS = 100
_DIGITS_RE = re.compile(r"\d+")
_ANCHOR_RE = re.compile(r"\w{3,}")


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return " ".join(text.split())


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """normalize_ws(text) plus, for each normalized char, its index in the original text."""
    chars: list[str] = []
    pos: list[int] = []
    pending_space = False
    for i, c in enumerate(text):
        if c.isspace():
            pending_space = bool(chars)
            continue
        if pending_space:
            chars.append(" ")
            pos.append(i - 1)
            pending_space = False
        chars.append(c)
        pos.append(i)
    return "".join(chars), pos


def _lower_same_len(text: str) -> str:
    return "".join(lc if len(lc := c.lower()) == 1 else c for c in text)


class QuoteNotFound(ValueError):
    """No acceptable span; carries the closest candidate spans (verbatim, shortened)."""

    def __init__(self, message: str, candidates: Sequence[str]) -> None:
        self.candidates = tuple(candidates)
        shown = "; ".join(repr(c) for c in self.candidates) or "none"
        super().__init__(f"{message}; closest spans in the source: {shown}")


def snap_quote(quote: str, text: str, *, fuzzy: bool = True) -> str:
    """Return the VERBATIM span of `text` that `quote` refers to, or raise QuoteNotFound.

    Match order on whitespace-normalized strings: (a) exact substring, (b) case-insensitive,
    (c) only if fuzzy: best difflib window with ratio >= SNAP_MIN_RATIO and length within
    +-SNAP_LENGTH_TOLERANCE of the quote. For (c) every digit sequence of the quote must also
    occur in the span (no "1.0" snapping onto "0.4.0").
    """
    q = normalize_ws(quote)
    if not q:
        raise QuoteNotFound("quote is blank", ())
    norm, pos = _normalize_with_map(text)

    def span(start: int, end: int) -> str:
        return text[pos[start] : pos[end - 1] + 1]

    at = norm.find(q)
    if at < 0:
        at = _lower_same_len(norm).find(_lower_same_len(q))
    if at >= 0:
        return span(at, at + len(q))
    if not fuzzy:
        raise QuoteNotFound(f"quote not found exactly in the source: {q[:80]!r}", ())

    ql, nl = _lower_same_len(q), _lower_same_len(norm)
    lo = max(1, math.floor(len(q) * (1 - SNAP_LENGTH_TOLERANCE)))
    hi = math.ceil(len(q) * (1 + SNAP_LENGTH_TOLERANCE))
    starts: set[int] = set(range(0, max(1, len(nl) - lo + 1), max(1, len(q) // 2)))
    for m in _ANCHOR_RE.finditer(ql):
        for occ in re.finditer(re.escape(m.group()), nl):
            base = occ.start() - m.start()
            starts.update(range(base - 2, base + 3))
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(ql)
    scored: list[tuple[float, int, int]] = []
    for st in sorted(x for x in starts if 0 <= x < len(nl)):
        best = (0.0, st, min(len(nl), st + len(q)))
        for ln in range(lo, hi + 1):
            if st + ln > len(nl):
                break
            sm.set_seq1(nl[st : st + ln])
            if sm.real_quick_ratio() <= best[0] or sm.quick_ratio() <= best[0]:
                continue
            r = sm.ratio()
            if r > best[0]:
                best = (r, st, st + ln)
        scored.append(best)
    scored.sort(key=lambda t: (-t[0], t[1]))

    q_digits = set(_DIGITS_RE.findall(q))
    for r, st, en in scored:
        if r < SNAP_MIN_RATIO:
            break
        if q_digits <= set(_DIGITS_RE.findall(norm[st:en])):
            return span(st, en)

    picked: list[tuple[int, int]] = []
    for _, st, en in scored:
        if all(abs(st - p) >= max(1, len(q) // 2) for p, _ in picked):
            picked.append((st, en))
        if len(picked) == _N_CANDIDATES:
            break
    cands = [normalize_ws(span(st, en))[:_CANDIDATE_CHARS] for st, en in picked if en > st]
    raise QuoteNotFound(f"quote not found in the source: {q[:80]!r}", cands)


def _citation_context(info: ValidationInfo) -> Mapping[str, Any]:
    ctx: Any = info.context
    if not isinstance(ctx, Mapping) or not isinstance(ctx.get("texts"), Mapping):
        raise ValueError("Citation requires validation context with key 'texts' (mapping)")
    return ctx


def resolve_evidence_id(value: str, ctx: Mapping[str, Any]) -> str:
    """Alias (via ctx["aliases"]: alias -> full id) or full id -> full id in evidence_ids."""
    aliases = ctx.get("aliases") or {}
    if not isinstance(aliases, Mapping):
        raise ValueError("validation context 'aliases' must be a mapping")
    full = aliases.get(value.strip(), value.strip())
    ids = ctx.get("evidence_ids")
    if not isinstance(ids, Set) or full not in ids:
        known = sorted(aliases)[:20]
        raise ValueError(f"unknown evidence id {value!r}; use one of the aliases {known}")
    return full


def _snaps(quote: str, text: str, *, fuzzy: bool) -> str | None:
    try:
        return snap_quote(quote, text, fuzzy=fuzzy)
    except QuoteNotFound:
        return None


def _relocate(
    quote: str, cited: str, miss: QuoteNotFound, ctx: Mapping[str, Any]
) -> tuple[str, str]:
    """The quote is not in the cited text. If it matches exactly (ws-exact or case-insensitive)
    in exactly one OTHER text, return (that evidence_id, verbatim span). Otherwise raise,
    naming where it was found (several exact matches, or fuzzy-only matches)."""
    ids = ctx.get("evidence_ids")
    others = [
        (i, t)
        for i, t in ctx["texts"].items()
        if i != cited and isinstance(t, str) and isinstance(ids, Set) and i in ids
    ]
    alias_of = {v: k for k, v in (ctx.get("aliases") or {}).items()}

    def label(i: str) -> str:
        return alias_of.get(i, i[:12])

    exact = [(i, sp) for i, t in others if (sp := _snaps(quote, t, fuzzy=False)) is not None]
    if len(exact) == 1:
        return exact[0]
    if exact:
        where = sorted(label(i) for i, _ in exact)
        raise ValueError(
            f"quote is not in {label(cited)} but appears in several texts {where}; "
            "cite the one you mean"
        )
    close = sorted(label(i) for i, t in others if _snaps(quote, t, fuzzy=True) is not None)
    if close:
        raise ValueError(
            f"quote is not in {label(cited)}; a close but not exact match is in {close}: "
            "cite that alias and copy the text character by character"
        )
    raise miss


class Citation(BaseModel):
    """A quote from a known evidence text.

    Input: evidence_id is an alias ("E3", via context["aliases"]) or a full id; quote is the
    model's text (1..300 chars). Stored: the full evidence id and the VERBATIM source span the
    quote snapped to (see snap_quote). If the quote is not in the cited text but matches
    exactly in exactly one other context text, the citation is relocated to that text.
    Needs context texts (evidence_id -> text) and evidence_ids; missing context fails closed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: EvidenceRef
    quote: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _resolve_and_snap(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, Mapping):
            return data
        ctx = _citation_context(info)
        eid, quote = data.get("evidence_id"), data.get("quote")
        if not isinstance(eid, str) or not isinstance(quote, str):
            return data  # let field validation report the type errors
        if not 1 <= len(quote) <= CITATION_MAX_QUOTE_CHARS:
            raise ValueError(f"quote must be 1..{CITATION_MAX_QUOTE_CHARS} chars, got {len(quote)}")
        full = resolve_evidence_id(eid, ctx)
        text = ctx["texts"].get(full)
        if not isinstance(text, str):
            raise ValueError(f"no text available for evidence id {eid!r}")
        try:
            return {**data, "evidence_id": full, "quote": snap_quote(quote, text)}
        except QuoteNotFound as miss:
            other, snapped = _relocate(quote, full, miss, ctx)
            return {**data, "evidence_id": other, "quote": snapped}
