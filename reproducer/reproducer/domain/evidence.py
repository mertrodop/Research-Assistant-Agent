"""Evidence model: immutable, append-only, content-hashed.

Pure domain: no file, network or subprocess IO. JSONL helpers only convert to/from strings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

EvidenceKind = Literal[
    "paper_claim",
    "paper_setup",
    "repo_file",
    "readme",
    "env_spec",
    "entrypoints",
    "workflow",
    "smoke_result",
    "run_result",
    "critic_patch",
    "lesson",
    "human_input",
    "llm_proposal",
]
Confidence = Literal["observed", "derived", "inferred"]
Authority = Literal["paper", "repo", "execution", "llm", "human"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact, UTF-8 kept, NaN/inf rejected."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def compute_evidence_id(kind: str, source: str, locator: str, content: dict[str, Any]) -> str:
    """sha256 hex over the identifying fields only (confidence/authority/links excluded)."""
    payload = {"kind": kind, "source": source, "locator": locator, "content": content}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Evidence(_Frozen):
    evidence_id: str
    kind: EvidenceKind
    source: str  # e.g. paper id, repo@commit, run_id
    locator: str  # e.g. "Table 2", "train.py:L42"
    content: dict[str, Any]  # JSON-compatible
    confidence: Confidence
    authority: Authority
    links: tuple[str, ...] = ()  # evidence_ids

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        try:
            expected = compute_evidence_id(self.kind, self.source, self.locator, self.content)
        except (TypeError, ValueError) as e:
            raise ValueError(f"content is not canonical-JSON compatible: {e}") from e
        if self.evidence_id != expected:
            raise ValueError(
                f"evidence_id mismatch: got {self.evidence_id!r}, expected {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_llm_proposal(self) -> Self:
        if self.kind == "llm_proposal" and (
            self.authority != "llm" or self.confidence != "inferred"
        ):
            raise ValueError(
                "llm_proposal evidence requires authority='llm' and confidence='inferred', "
                f"got authority={self.authority!r}, confidence={self.confidence!r}"
            )
        return self


def make_evidence(
    *,
    kind: EvidenceKind,
    source: str,
    locator: str,
    content: dict[str, Any],
    confidence: Confidence,
    authority: Authority,
    links: Iterable[str] = (),
) -> Evidence:
    """Build Evidence with a computed id. Content is stored as a detached, normalized copy."""
    try:
        normalized = json.loads(canonical_json(content))
    except (TypeError, ValueError) as e:
        raise ValueError(f"content is not canonical-JSON compatible: {e}") from e
    return Evidence.model_validate(
        {
            "evidence_id": compute_evidence_id(kind, source, locator, normalized),
            "kind": kind,
            "source": source,
            "locator": locator,
            "content": normalized,
            "confidence": confidence,
            "authority": authority,
            "links": tuple(links),
        }
    )


def to_jsonl_line(model: BaseModel) -> str:
    """One canonical JSON line (no trailing newline)."""
    return canonical_json(model.model_dump(mode="json"))


def evidence_from_jsonl_line(line: str) -> Evidence:
    """Parse one JSONL line; the hash is re-verified (tampered ids are rejected)."""
    return Evidence.model_validate_json(line.strip())


def dedupe(records: Iterable[Evidence]) -> tuple[Evidence, ...]:
    """Keep the first occurrence of each evidence_id, preserving order."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for r in records:
        if r.evidence_id not in seen:
            seen.add(r.evidence_id)
            out.append(r)
    return tuple(out)
