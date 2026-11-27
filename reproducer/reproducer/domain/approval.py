"""Human-in-the-loop approvals and the escalation queue (pure: no file, network or subprocess IO).

A gated action (full_run, critic_patch, shared_vault_write) that the policy says needs a
human is stored as `approval_request` evidence (authority rule, confidence derived, links the
evidence that triggered it); the graph stops with awaiting_approval. A human answers with
`approval_decision` evidence (authority human, confidence observed). Both are append-only:
nothing is ever rewritten, and the LLM can never approve (a decision must be human evidence).

Identity: every request carries an input digest over what the action would do (the run
digest, the critic inputs; for a shared vault write only the vault path). The same digest
means the same request, so a re-run never files a duplicate: it finds the stored request and
its decision.

Standing approval (shared_vault_write): one request per vault path. Approving it once allows
every later write to that path (content changes file no new request); a later reject revokes
it (and a later approve restores it): the NEWEST human decision counts. Every other action
is decided once (the first decision counts; another one is refused).

Policy (env REPRODUCER_APPROVAL):
  strict   every gated action
  default  full_run when estimate_h >= REPRODUCER_APPROVE_GPU_HOURS (default 1.0); every
           critic_patch; shared_vault_write only when the vault is REPRODUCER_SHARED_VAULT
  off      none (policy_warning says so)

Human answers to unknowns: `human_input` evidence (authority human, confidence observed)
resolves an OPEN unknown through unknowns.resolve(); a conflicting answer raises.
"""

from __future__ import annotations

import hashlib
import math
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from reproducer.domain.evidence import Evidence, canonical_json, make_evidence
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    UnknownState,
    fold_unknowns,
    resolve,
    unknown_id_for,
)

Action = Literal["full_run", "critic_patch", "shared_vault_write"]
DecisionValue = Literal["approved", "rejected"]
Status = Literal["approved", "rejected", "pending"]
Mode = Literal["strict", "default", "off"]

ACTIONS: tuple[Action, ...] = ("full_run", "critic_patch", "shared_vault_write")
APPROVAL_ENV = "REPRODUCER_APPROVAL"
GPU_HOURS_ENV = "REPRODUCER_APPROVE_GPU_HOURS"
SHARED_VAULT_ENV = "REPRODUCER_SHARED_VAULT"
DEFAULT_GPU_HOURS = 1.0
HUMAN_SOURCE = "human:cli"
REJECTED_PREFIX = "rejected_by_human"


# --- policy ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalPolicy:
    mode: Mode = "default"
    gpu_hours: float = DEFAULT_GPU_HOURS
    shared_vault: str | None = None  # compared as a normalized path string


def policy_from_env(env: Mapping[str, str]) -> ApprovalPolicy:
    """Raises ValueError on an unknown mode or a bad GPU-hour threshold."""
    raw = env.get(APPROVAL_ENV, "").strip().lower() or "default"
    if raw not in ("strict", "default", "off"):
        raise ValueError(f"{APPROVAL_ENV}={raw!r}: expected strict, default or off")
    hours_raw = env.get(GPU_HOURS_ENV, "").strip()
    hours = DEFAULT_GPU_HOURS
    if hours_raw:
        try:
            hours = float(hours_raw)
        except ValueError:
            raise ValueError(
                f"{GPU_HOURS_ENV}={hours_raw!r} is not a number of GPU hours"
            ) from None
        if not math.isfinite(hours) or hours < 0:
            raise ValueError(f"{GPU_HOURS_ENV}={hours_raw!r} must be finite and >= 0")
    shared = env.get(SHARED_VAULT_ENV, "").strip() or None
    return ApprovalPolicy(mode=raw, gpu_hours=hours, shared_vault=shared)  # type: ignore[arg-type]


def policy_warning(policy: ApprovalPolicy) -> str | None:
    if policy.mode == "off":
        return (
            f"warning: {APPROVAL_ENV}=off: full runs, critic patches and shared vault writes "
            "proceed WITHOUT human approval"
        )
    return None


def _same_path(a: str, b: str) -> bool:
    return posixpath.normpath(a) == posixpath.normpath(b)


def needs_approval(
    action: Action, cost: float, policy: ApprovalPolicy, *, subject: str = ""
) -> tuple[bool, str]:
    """(needs a human, why). cost: estimate_h (full_run), changed lines (critic_patch), note
    count (shared_vault_write). subject: the vault path for shared_vault_write."""
    if action not in ACTIONS:
        raise ValueError(f"unknown gated action {action!r}")
    if policy.mode == "off":
        return False, f"{APPROVAL_ENV}=off"
    if policy.mode == "strict":
        return True, f"{APPROVAL_ENV}=strict: every {action} needs approval"
    if action == "full_run":
        if cost >= policy.gpu_hours:
            return True, f"estimate {cost:.2f} h >= {policy.gpu_hours:g} h approval threshold"
        return False, f"estimate {cost:.2f} h < {policy.gpu_hours:g} h approval threshold"
    if action == "critic_patch":
        return True, f"every critic patch needs approval ({int(cost)} changed line(s))"
    if policy.shared_vault and _same_path(subject, policy.shared_vault):
        return True, f"{subject} is the shared vault ({SHARED_VAULT_ENV})"
    return False, "not the shared vault"


# --- requests + decisions --------------------------------------------------------------------


def request_digest(action: Action, parts: Mapping[str, Any]) -> str:
    """sha256 over the action and what it would do (canonical JSON)."""
    payload = {"action": action, **parts}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def approval_request_evidence(
    *,
    action: Action,
    subject: str,
    cost: Mapping[str, Any],
    detail: Mapping[str, Any],
    input_digest: str,
    source: str,
    links: Sequence[str],
    reason: str = "",
) -> Evidence:
    return make_evidence(
        kind="approval_request",
        source=source,
        locator=f"approval:{action}:{subject}",
        content={
            "action": action,
            "subject": subject,
            "cost": dict(cost),
            "detail": dict(detail),
            "input_digest": input_digest,
            "reason": reason,
        },
        confidence="derived",
        authority="rule",
        links=tuple(links),
    )


def standing(request: Evidence) -> bool:
    """A standing approval: the newest decision counts and may be changed later."""
    return request.content.get("action") == "shared_vault_write"


def approval_decision_evidence(
    request: Evidence, decision: DecisionValue, note: str, revision: int = 0
) -> Evidence:
    """revision: the number of earlier decisions on a standing request (keeps a re-approval
    after a revoke a distinct evidence record); 0 is left out of the content."""
    if request.kind != "approval_request":
        raise ValueError(f"{request.evidence_id[:12]} is a {request.kind}, not an approval_request")
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be approved or rejected, got {decision!r}")
    if revision < 0:
        raise ValueError(f"revision must be >= 0, got {revision}")
    content: dict[str, Any] = {
        "request_id": request.evidence_id,
        "decision": decision,
        "note": note,
    }
    if revision:
        content["revision"] = revision
    return make_evidence(
        kind="approval_decision",
        source=HUMAN_SOURCE,
        locator=f"approval_decision:{request.evidence_id[:12]}",
        content=content,
        confidence="observed",
        authority="human",
        links=(request.evidence_id,),
    )


def find_request(evidence: Mapping[str, Evidence], input_digest: str) -> Evidence | None:
    """The first stored request with this input digest (same digest = same request)."""
    return next((ev for ev in evidence.values() if ev.kind == "approval_request"
                 and ev.content.get("input_digest") == input_digest), None)  # fmt: skip


def decisions_for(evidence: Mapping[str, Evidence], request_id: str) -> list[Evidence]:
    """Every human decision on the request, in file order."""
    return [ev for ev in evidence.values() if ev.kind == "approval_decision"
            and ev.authority == "human" and ev.content.get("request_id") == request_id]  # fmt: skip


def decision_for(evidence: Mapping[str, Evidence], request_id: str) -> Evidence | None:
    """The human decision that counts: the newest one for a standing request (shared vault
    write), else the first (file order; later ones are ignored)."""
    decs = decisions_for(evidence, request_id)
    if not decs:
        return None
    req = evidence.get(request_id)
    return decs[-1] if req is not None and standing(req) else decs[0]


def find_decision(evidence: Mapping[str, Evidence], input_digest: str) -> Status:
    """approved | rejected | pending (pending also when no request is stored yet)."""
    req = find_request(evidence, input_digest)
    dec = decision_for(evidence, req.evidence_id) if req is not None else None
    return "pending" if dec is None else dec.content["decision"]


@dataclass(frozen=True)
class GateResult:
    """What a gate decided. status None: no approval needed (proceed)."""

    status: Status | None
    reason: str
    request: Evidence | None = None
    decision: Evidence | None = None
    new_request: bool = False

    @property
    def proceed(self) -> bool:
        return self.status in (None, "approved")

    @property
    def note(self) -> str:
        return str(self.decision.content.get("note", "")) if self.decision is not None else ""

    @property
    def rejected_reason(self) -> str:
        return f"{REJECTED_PREFIX}: {self.note or '(no note)'}"


def gate(
    evidence: Mapping[str, Evidence],
    request: Evidence,
    policy: ApprovalPolicy,
    cost: float,
) -> GateResult:
    """Decide one gated action. `request` is the request that WOULD be filed (it is filed by
    the caller only when new_request is True). A stored request with the same input digest
    is reused, whatever the policy says now (a human decision is never ignored)."""
    c = request.content
    digest = str(c["input_digest"])
    stored = find_request(evidence, digest)
    if stored is None:
        needed, why = needs_approval(c["action"], cost, policy, subject=str(c["subject"]))
        if not needed:
            return GateResult(None, why)
        return GateResult("pending", why, request, None, new_request=True)
    dec = decision_for(evidence, stored.evidence_id)
    if dec is None:
        return GateResult("pending", "awaiting a human decision", stored)
    return GateResult(dec.content["decision"], str(dec.content.get("note", "")), stored, dec)


# --- queue -----------------------------------------------------------------------------------


def pending_requests(evidence: Mapping[str, Evidence]) -> list[Evidence]:
    decided = {str(ev.content.get("request_id")) for ev in evidence.values()
               if ev.kind == "approval_decision" and ev.authority == "human"}  # fmt: skip
    return [ev for ev in evidence.values()
            if ev.kind == "approval_request" and ev.evidence_id not in decided]  # fmt: skip


def resolve_request_id(
    evidence: Mapping[str, Evidence], prefix: str, decision: DecisionValue | None = None
) -> Evidence:
    """The approval_request whose id starts with prefix (>= 6 chars); ValueError if none,
    ambiguous, or already decided (a duplicate decision). A standing request (shared vault
    write) may be decided again with the other decision (revoke / restore); `decision` is
    the one about to be stored."""
    prefix = prefix.strip().lower()
    if len(prefix) < 6:
        raise ValueError(f"request id {prefix!r} is too short (give at least 6 hex chars)")
    hits = [ev for ev in evidence.values()
            if ev.kind == "approval_request" and ev.evidence_id.startswith(prefix)]  # fmt: skip
    if not hits:
        raise ValueError(f"no approval request {prefix!r}")
    if len(hits) > 1:
        raise ValueError(f"request id {prefix!r} is ambiguous ({len(hits)} matches)")
    dec = decision_for(evidence, hits[0].evidence_id)
    if dec is not None and standing(hits[0]) and decision is not None:
        if dec.content.get("decision") != decision:
            return hits[0]
        raise ValueError(
            f"standing request {prefix} is already {decision} "
            f"(decision {dec.evidence_id[:12]}); nothing to change"
        )
    if dec is not None:
        raise ValueError(
            f"request {prefix} was already {dec.content['decision']} "
            f"(decision {dec.evidence_id[:12]}); decisions are append-only"
        )
    return hits[0]


def cost_text(content: Mapping[str, Any]) -> str:
    cost = content.get("cost") or {}
    return ", ".join(f"{k}={v}" for k, v in cost.items()) or "-"


def detail_summary(content: Mapping[str, Any], width: int = 100) -> str:
    """One line: the command, the patched file + first diff line, or the note count."""
    d = content.get("detail") or {}
    action = content.get("action")
    if action == "full_run":
        text = str(d.get("command", ""))
    elif action == "critic_patch":
        text = f"{d.get('file')}: {d.get('rationale', '')}"
    else:
        notes = d.get("notes") or []
        text = f"{len(notes)} note(s): " + ", ".join(str(n) for n in notes[:3])
        text += ", ..." if len(notes) > 3 else ""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


@dataclass(frozen=True)
class EscalationQueue:
    requests: tuple[Evidence, ...]
    unknowns: tuple[UnknownState, ...]

    @property
    def empty(self) -> bool:
        return not self.requests and not self.unknowns


def escalation_queue(
    evidence: Mapping[str, Evidence], unknowns: Mapping[str, UnknownState]
) -> EscalationQueue:
    """Pending approval requests (file order) + open unknowns (by name)."""
    opened = sorted((st for st in unknowns.values() if st.status == "open"), key=lambda s: s.name)
    return EscalationQueue(tuple(pending_requests(evidence)), tuple(opened))


# --- human answers to unknowns ---------------------------------------------------------------


def human_input_evidence(name: str, value: str, note: str, opened_id: str) -> Evidence:
    return make_evidence(
        kind="human_input",
        source=HUMAN_SOURCE,
        locator=f"answer:{name}",
        content={"unknown": name, "value": value, "note": note},
        confidence="observed",
        authority="human",
        links=(opened_id,),
    )


@dataclass(frozen=True)
class Answer:
    evidence: Evidence
    event: UnknownResolved | None  # None: already resolved to the same value (no-op)


def answer_unknown(
    events: Iterable[UnknownOpened | UnknownResolved | Assumption],
    evidence: Mapping[str, Evidence],
    name: str,
    value: str,
    note: str,
) -> Answer:
    """human_input evidence + the UnknownResolved event (via resolve()) for an OPEN unknown.
    Already resolved to the same value -> event None; to a different value -> ValueError
    (fold_unknowns' conflict rule); never opened -> ValueError."""
    events = tuple(events)
    uid = unknown_id_for(name)
    opened = next((e for e in events if isinstance(e, UnknownOpened) and e.unknown_id == uid),
                  None)  # fmt: skip
    if opened is None:
        raise ValueError(f"no unknown named {name!r}")
    if not value.strip():
        raise ValueError("the answer value must not be empty")
    ev = human_input_evidence(name, value, note, opened.evidence_id)
    states = fold_unknowns(events, evidence)
    st = states[uid]
    if st.status == "resolved" and st.value == value:
        return Answer(ev, None)
    event = resolve(opened, ev, value)
    fold_unknowns((*events, event), {**evidence, ev.evidence_id: ev})  # raises on a conflict
    return Answer(ev, event)


def human_resolution(evidence: Mapping[str, Evidence], st: UnknownState) -> Evidence | None:
    """The human_input evidence that resolved this unknown, if a human did."""
    ev = evidence.get(st.resolved_by or "")
    return ev if ev is not None and ev.kind == "human_input" else None
