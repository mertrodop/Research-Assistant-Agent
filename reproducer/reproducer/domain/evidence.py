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
    "lesson",  # a validated Reflexion lesson (graph/nodes/reflect.py; authority llm,
    # derived): content {claim_id, trial, input_digest, run_evidence_id, lesson}; links: the
    # evidence it cites. Keyed by (claim, trial, input_digest); others of the trial are stale
    "human_input",
    "llm_proposal",
    "dataset_cache",  # a dataset verified in the shared cache (authority execution, observed)
    "known_fix",  # a zero-token rule fix (authority rule, derived; links: the smoke evidence)
    "critic_revert",  # a critic patch that left the error signature unchanged, dropped from
    # the patch stack (authority rule, derived; links: the critic_patch + the new smoke)
    "tolerance",  # a pre-registered claim tolerance, written before any run (authority rule,
    # derived; links: the paper_claim evidence)
    "claim_mapping",  # the orchestrator's validated claim -> repo mapping (authority llm,
    # derived; links: every evidence it cites)
    "run_plan",  # a RunPlan built by code from the claim mapping (authority rule, derived;
    # content: plan + rendered command + notes; links: claim_mapping, paper_claim, tolerance)
    "preflight",  # the go / no-go before a full run (authority execution, observed): plan
    # digest, smoke timing, GPU-hour estimate, caps, decision; links: run_plan + smoke_result
    "setup_failed",  # a full run that failed before its container started (workdir copy,
    # patch, image check, ...): authority execution, observed; NOT a run attempt, no GPU
    # hours; links: run_plan + preflight
    "approval_request",  # a gated action waiting for a human (authority rule, derived): content
    # {action, subject, cost, detail, input_digest}; links: the triggering evidence
    "approval_decision",  # a human's answer to an approval_request (authority human,
    # observed): content {request_id, decision: approved|rejected, note}; links: the request
    "lesson_applied",  # code applying (or stopping on) one validated, current lesson
    # (graph/nodes/apply_lesson.py; authority rule, derived): content {claim_id, trial,
    # lesson_id, kind, target, new_value, support_evidence_id, outcome: applied | resolved |
    # reeval | stopped_no_change, detail}; links: the lesson (+ its support evidence). Plans
    # fold the applied arg / eval_protocol changes; the lesson ids enter the plan digest
    "critic_trace",  # the ReAct critic's trajectory (graph/nodes/critic_react.py; authority
    # llm, derived): content {file, error_signature, status, mode, steps: [{step, tool, args,
    # result_chars, truncated, ok, note}], outcome, fallback_reason, fallback_status,
    # llm_calls, tokens, thoughts}; links: the smoke evidence. Additive: never folded
    "run_curve",  # a run's training curve projected by code (domain/curve.py; authority
    # rule, derived): content {claim_id, run_evidence_id, run_id, summary[, partial: true for
    # a crashed / diverged run]}; links: the run_result. One per run (idempotent by run id)
    "fallback",  # a graph invocation that took the default fallback route (graph/build.py
    # fallback_node; authority rule, derived): content {router_or_node, reason, detail};
    # locator fallback:<invocation stamp>. One per invocation; claims stay open
]
Confidence = Literal["observed", "derived", "inferred"]
# "rule": a deterministic code decision (env plan, known fix). Like "llm", it never resolves
# an Unknown: a rule picks a value to proceed with, it does not observe the paper or repo.
Authority = Literal["paper", "repo", "execution", "llm", "human", "rule"]


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

    @model_validator(mode="after")
    def _check_tolerance(self) -> Self:
        if self.kind == "tolerance" and (
            self.authority != "rule" or self.confidence != "derived" or not self.links
        ):
            raise ValueError(
                "tolerance evidence requires authority='rule', confidence='derived' and a link "
                f"to the paper_claim, got authority={self.authority!r}, "
                f"confidence={self.confidence!r}, links={list(self.links)}"
            )
        return self

    @model_validator(mode="after")
    def _check_plan_kinds(self) -> Self:
        required = {"claim_mapping": "llm", "run_plan": "rule", "lesson": "llm"}.get(self.kind)
        if required is not None and (
            self.authority != required or self.confidence != "derived" or not self.links
        ):
            raise ValueError(
                f"{self.kind} evidence requires authority={required!r}, confidence='derived' "
                f"and links, got authority={self.authority!r}, confidence={self.confidence!r}, "
                f"links={list(self.links)}"
            )
        return self

    @model_validator(mode="after")
    def _check_lesson_applied(self) -> Self:
        if self.kind == "lesson_applied" and (
            self.authority != "rule"
            or self.confidence != "derived"
            or not self.links
            or self.links[0] != self.content.get("lesson_id")
        ):
            raise ValueError(
                "lesson_applied evidence requires authority='rule', confidence='derived' and "
                f"links starting with the lesson_id, got authority={self.authority!r}, "
                f"confidence={self.confidence!r}, links={list(self.links)}"
            )
        return self

    @model_validator(mode="after")
    def _check_run_curve(self) -> Self:
        if self.kind == "run_curve" and (
            self.authority != "rule"
            or self.confidence != "derived"
            or self.links != (self.content.get("run_evidence_id"),)
        ):
            raise ValueError(
                "run_curve evidence requires authority='rule', confidence='derived' and links "
                f"== (run_evidence_id,), got authority={self.authority!r}, "
                f"confidence={self.confidence!r}, links={list(self.links)}"
            )
        return self

    @model_validator(mode="after")
    def _check_preflight(self) -> Self:
        if self.kind == "preflight" and (
            self.authority != "execution" or self.confidence != "observed"
        ):
            raise ValueError(
                "preflight evidence requires authority='execution' and confidence='observed', "
                f"got authority={self.authority!r}, confidence={self.confidence!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_approval(self) -> Self:
        if self.kind == "approval_request" and (
            self.authority != "rule" or self.confidence != "derived" or not self.links
        ):
            raise ValueError(
                "approval_request evidence requires authority='rule', confidence='derived' and "
                f"a link to the triggering evidence, got authority={self.authority!r}, "
                f"confidence={self.confidence!r}, links={list(self.links)}"
            )
        if self.kind == "approval_decision":
            # only a human approves: never llm / rule / execution authority
            if self.authority != "human" or self.confidence != "observed":
                raise ValueError(
                    "approval_decision evidence requires authority='human' and "
                    f"confidence='observed', got authority={self.authority!r}, "
                    f"confidence={self.confidence!r}"
                )
            rid = self.content.get("request_id")
            if self.content.get("decision") not in ("approved", "rejected") or self.links != (rid,):
                raise ValueError(
                    "approval_decision needs decision approved|rejected and links == "
                    f"(request_id,), got {self.content.get('decision')!r}, {list(self.links)}"
                )
        return self

    @model_validator(mode="after")
    def _check_fallback(self) -> Self:
        if self.kind == "fallback" and (
            self.authority != "rule"
            or self.confidence != "derived"
            or set(self.content) != {"router_or_node", "reason", "detail"}
        ):
            raise ValueError(
                "fallback evidence requires authority='rule', confidence='derived' and content "
                f"{{router_or_node, reason, detail}}, got authority={self.authority!r}, "
                f"confidence={self.confidence!r}, keys={sorted(self.content)}"
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
