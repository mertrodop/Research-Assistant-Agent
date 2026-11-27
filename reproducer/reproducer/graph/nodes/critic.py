"""Grounded critic: one minimal patch for a `critic` failure, from the REAL traceback.

Never intrinsic self-critique: the only inputs are the parsed traceback of the last smoke run
(verbatim), the failing file as it is now (clone + patch stack, read from the host clone,
never executed), a numbered window of +-WINDOW lines around the repo frame line, and the
command. The LLM proposes a CriticPatch; gated_call validates it in code (contracts.py:
failing file only, `original` verbatim, <= 20 changed lines, no metric line, no numeric
literal change, patched file parses) and retries with the real validation error.

Signature gating: the critic runs only if the error signature (exc_type + normalized message
+ repo frame) differs from the last critic input for this repo, or if a patch for this very
signature was just found ineffective (failed_patches): otherwise there is no new evidence and
it is skipped (no LLM call).

Ineffective patch: when the re-smoke after a patch shows the SAME signature, the loop records
critic_revert evidence (authority rule, derived; links: the critic_patch and the new smoke),
drops the patch from the stack and calls the critic again with "previous patch <diff> did not
change the error" in the prompt. The failed attempt stays counted (cap MAX_CRITIC_ATTEMPTS).

Patches are rendered to a unified diff in code and stored as critic_patch evidence (authority
llm, confidence inferred). They are applied only to per-run copies (prepare_workdir), never
to the clone. Failed critic attempts are stored too (ok=False), so a later invocation does
not ask again for the same signature and the attempt cap holds across invocations.

Human approval (make_critic_fn with an ApprovalPolicy; domain/approval.py): a patch that
passed code validation is NOT stored or applied when the policy gates it. It goes into an
approval_request (the proposed diff + edits live in the request, not on the stack) and the
outcome is awaiting_approval (the loop stops, nothing is pushed). The request's input digest
is over the critic inputs (signature, command, patch stack, reverted patches, rejections), so
a re-run finds the same request without an LLM call: approved -> the stored patch is used
(then applied and stored as critic_patch as usual); rejected -> a failed critic attempt
(critic_patch ok=False, rejected_by_human, authority rule; links request + decision) and the
next critic call sees "human rejected this patch: <note>" with the diff (same cap).

Mode (REPRODUCER_CRITIC_MODE=react|single, default react): react runs the bounded ReAct loop
(critic_react.py) with read-only tools on a virtual repo view (critic_tools.py: clone +
patch stack, never written) and the same grounded first message; its submit_patch is
validated by the same CriticPatch code. When the loop ends without a valid patch (no tool
support, protocol / context error, step / token / wall-clock cap, loop detection) the
single-shot critic runs once and the reason is recorded. The trajectory is stored as
critic_trace evidence (authority llm, derived; links the smoke evidence). Signature gating,
approval gate, the attempt cap and reverts are the same in both modes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from reproducer.domain.approval import (
    ApprovalPolicy,
    approval_request_evidence,
    decision_for,
    find_request,
    gate,
    policy_from_env,
    request_digest,
)
from reproducer.domain.contracts import CRITIC_MAX_CHANGED_LINES, CriticPatch, LineEdit
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.patching import (
    apply_line_edits,
    diff_changed_lines,
    diff_sha256,
    error_signature,
    repo_relative,
    split_lines,
    unified_diff,
)
from reproducer.domain.telemetry import CallRecord
from reproducer.domain.traceback_parse import ParsedTraceback
from reproducer.graph.eviction import ContextEvictionFailed, call_with_eviction, shrink_prompt
from reproducer.graph.nodes.critic_react import (
    CriticTrace,
    ReactLimits,
    run_react,
    with_fallback,
)
from reproducer.graph.nodes.critic_tools import view_for
from reproducer.graph.nodes.subtasks import ContextSection
from reproducer.providers.git import UnsafePathError, read_text
from reproducer.providers.llm import LLM, BudgetExceeded, LLMError, MeteredLLM
from reproducer.providers.store import RunStore

CRITIC_NODE = "critic"
WINDOW = 40
MAX_CRITIC_ATTEMPTS = 3
MAX_FILE_BYTES = 1_000_000
CRITIC_MAX_TOKENS = 4000  # thinking models count reasoning against the limit

_RULES = (
    "You fix a crash in a research code repository with ONE minimal patch. You see the real "
    "traceback and the failing file. Rules (checked by code; violations are rejected):\n"
    "- patch only the failing file named in the task;\n"
    f"- at most 3 edits and at most {CRITIC_MAX_CHANGED_LINES} changed lines (removed + "
    "added);\n"
    "- each edit replaces lines start_line..end_line; `original` must be EXACTLY those lines "
    "of the file (without the line-number prefix), `replacement` the new lines with the same "
    "indentation;\n"
    "- never touch metric/evaluation-reporting lines (marked [METRIC]);\n"
    "- never change any numeric value (learning rate, epochs, sizes, constants): the "
    "multiset of number literals must stay the same (exempt: subscript indices like x[0] and "
    "integer dim=/axis= values or .view/.reshape/... args);"
    "\n- no tuning toward a target number, no new features, no refactors: fix the crash only;\n"
    "- the patched file must be valid Python.\n"
)
SYSTEM = _RULES + (
    "Answer with JSON only: "
    '{"file": str, "rationale": str (<= 300 chars), "edits": [{"start_line": int, '
    '"end_line": int, "original": str, "replacement": str}]}'
)

REACT_SYSTEM = _RULES + (
    "Work in steps: think briefly, then call a tool. read_file / grep / list_dir inspect a "
    "READ-ONLY view of the repository (pinned commit + patches already applied); use them "
    "only when the window you are shown is not enough (e.g. a function defined elsewhere). "
    "Finish by calling submit_patch with the patch to the failing file; if code rejects it, "
    "the error comes back as the result: fix the patch and call submit_patch again. At most "
    "5 tool calls in total; an identical repeated call is rejected. Line numbers refer to the "
    "failing file as shown to you."
)
REACT_TAIL = "Investigate only if needed, then call submit_patch with that patch."

CriticMode = Literal["react", "single"]
CRITIC_MODES: tuple[CriticMode, ...] = ("react", "single")
CRITIC_MODE_ENV = "REPRODUCER_CRITIC_MODE"


def critic_mode_from_env(env: Mapping[str, str]) -> CriticMode:
    value = (env.get(CRITIC_MODE_ENV) or "react").strip()
    if value not in CRITIC_MODES:
        raise ValueError(f"{CRITIC_MODE_ENV} must be react|single, got {value!r}")
    return value  # type: ignore[return-value]


CriticStatus = Literal[
    "patched", "invalid", "skipped", "no_target",
    "awaiting_approval",  # a validated patch waits in an approval_request (not applied)
    "rejected",  # a human rejected the patch: a failed attempt (stored by the critic fn)
]  # fmt: skip


@dataclass(frozen=True)
class AppliedPatch:
    """A validated critic patch in the stack (applied to per-run copies in order)."""

    file: str
    diff: str
    edits: tuple[LineEdit, ...]
    rationale: str
    error_signature: str

    @property
    def patch_id(self) -> str:
        return diff_sha256(self.diff)


@dataclass(frozen=True)
class CriticOutcome:
    status: CriticStatus
    signature: str
    reason: str
    patch: AppliedPatch | None = None
    gate_attempts: int = 0  # LLM calls made (0 when skipped / no target)
    errors: tuple[str, ...] = ()  # gate validation errors, one per rejected answer
    file: str | None = None
    request_id: str | None = None  # the approval_request (awaiting_approval / rejected)
    evicted: tuple[str, ...] = ()  # prompt sections dropped by context eviction
    trace: CriticTrace | None = None  # the ReAct trajectory (react mode only)

    @property
    def counts_as_attempt(self) -> bool:
        """An LLM critic attempt (towards MAX_CRITIC_ATTEMPTS)."""
        return self.status in ("patched", "invalid")


# --- pure helpers --------------------------------------------------------------------------


def fold_text(clone_text: str, file: str, stack: Iterable[AppliedPatch]) -> str:
    """The file text after the patch stack (edits replayed in order)."""
    text = clone_text
    for p in stack:
        if p.file == file:
            text = apply_line_edits(text, p.edits)
    return text


def protected_lines(text: str, metric_line_texts: Iterable[str]) -> frozenset[int]:
    """1-based line numbers of `text` whose stripped content is a known metric line (by text,
    so line shifts from earlier patches do not matter)."""
    wanted = {t.strip() for t in metric_line_texts if t.strip()}
    return frozenset(i for i, ln in enumerate(split_lines(text), start=1) if ln.strip() in wanted)


def numbered_window(
    text: str, center: int, radius: int = WINDOW, marked: Set[int] = frozenset()
) -> str:
    lines = split_lines(text)
    lo, hi = max(1, center - radius), min(len(lines), center + radius)
    width = len(str(hi))
    out = []
    for n in range(lo, hi + 1):
        tag = ">>" if n == center else ("[METRIC]" if n in marked else "")
        body = lines[n - 1].rstrip("\n").rstrip("\r")
        out.append(f"{tag:>8} {n:>{width}}| {body}")
    return "\n".join(out)


def critic_prompt(
    tb: ParsedTraceback,
    command: str,
    file: str,
    text: str,
    line: int,
    marked: Set[int],
    failed: Sequence[AppliedPatch] = (),
    rejected: Sequence[tuple[str, str]] = (),
) -> str:
    n = len(split_lines(text))
    return (
        f"Command (run from the repo root):\n{command}\n\n"
        f"Traceback (verbatim):\n{tb.raw}\n\n"
        f"Failing file: {file} ({n} lines). The innermost repo frame is line {line} (marked "
        f">>). Lines {max(1, line - WINDOW)}-{min(n, line + WINDOW)} of the CURRENT file, as "
        "'<line number>| <code>':\n"
        f"{numbered_window(text, line, WINDOW, marked)}\n\n"
        f"{_failed_section(failed)}"
        f"{_rejected_section(rejected)}"
        f"Return the minimal patch to {file} that fixes this error."
    )


def critic_prompt_builder(
    tb: ParsedTraceback,
    command: str,
    file: str,
    text: str,
    line: int,
    marked: Set[int],
    failed: Sequence[AppliedPatch] = (),
    rejected: Sequence[tuple[str, str]] = (),
) -> Callable[[int | None], tuple[str, str, tuple[str, ...]]]:
    """Eviction builder: critic_prompt unchanged when it fits the budget (None: always); else
    the parts packed by priority (window > traceback > command > rejected > failed)."""
    full = critic_prompt(tb, command, file, text, line, marked, failed, rejected)
    n = len(split_lines(text))
    sections = [
        ContextSection(title="Command (run from the repo root)", text=command, priority=90),
        ContextSection(title="Traceback (verbatim)", text=tb.raw, priority=95),
        ContextSection(
            title=f"Failing file {file} ({n} lines), innermost repo frame line {line} (>>)",
            text=numbered_window(text, line, WINDOW, marked),
            priority=100,
        ),
        ContextSection(title="Human-rejected patches", text=_rejected_section(rejected),
                       priority=50),
        ContextSection(title="Reverted patches", text=_failed_section(failed), priority=40),
    ]  # fmt: skip
    sections = [s for s in sections if s.text]
    tail = f"Return the minimal patch to {file} that fixes this error."

    def build(budget: int | None) -> tuple[str, str, tuple[str, ...]]:
        prompt, evicted = shrink_prompt(full, sections, budget, tail=tail)
        return SYSTEM, prompt, evicted

    return build


def _failed_section(failed: Sequence[AppliedPatch]) -> str:
    if not failed:
        return ""
    parts = [
        f"Previous patch {i} (reverted; the file above is WITHOUT it):\n{p.diff.rstrip()}\n"
        f"did not change the error: the re-run failed with the same traceback.\n"
        for i, p in enumerate(failed, start=1)
    ]
    return "".join(parts) + "Do not repeat these changes; find the actual cause.\n\n"


def _rejected_section(rejected: Sequence[tuple[str, str]]) -> str:
    """(diff, note) of patches a human rejected; they were never applied."""
    if not rejected:
        return ""
    parts = [
        f"Proposed patch {i} (NOT applied; the file above is WITHOUT it):\n{diff.rstrip()}\n"
        f"human rejected this patch: {note or '(no note)'}\n"
        for i, (diff, note) in enumerate(rejected, start=1)
    ]
    return "".join(parts) + "Do not propose these changes again; respect the human's note.\n\n"


def _failing_file(tb: ParsedTraceback, repo_files: Set[str]) -> tuple[str, int] | None:
    rf = tb.repo_frame
    if rf is None:
        return None
    rel = repo_relative(rf.file)
    if rel is None or rel not in repo_files:
        return None
    return rel, rf.line


# --- node ----------------------------------------------------------------------------------


def run_critic(
    llm: LLM,
    *,
    traceback: ParsedTraceback,
    command: str,
    clone_dir: Path,
    repo_files: Set[str],
    stack: Sequence[AppliedPatch] = (),
    metric_line_texts: Mapping[str, Sequence[str]] | None = None,
    last_signature: str | None = None,
    failed_patches: Sequence[AppliedPatch] = (),
    rejected: Sequence[tuple[str, str]] = (),
    max_retries: int = 2,
    max_tokens: int = CRITIC_MAX_TOKENS,
    mode: CriticMode = "react",
    react_limits: ReactLimits | None = None,
) -> CriticOutcome:
    """One critic attempt for a `critic` classification (see module docstring).
    failed_patches: reverted patches for this signature, passed only right after a revert
    (that revert is the new evidence); their diffs are shown in the prompt. rejected: (diff,
    note) of human-rejected patches for this signature, shown in the prompt.
    mode react: the ReAct loop first; without a valid patch, the single-shot critic once.
    gate_attempts / errors then sum both (ReAct submits + single-shot answers)."""
    if mode not in CRITIC_MODES:
        raise ValueError(f"critic mode must be react|single, got {mode!r}")
    sig = error_signature(traceback.model_dump(mode="json"))
    if last_signature is not None and sig == last_signature and not failed_patches:
        return CriticOutcome(
            status="skipped",
            signature=sig,
            reason="same error signature as the last critic input: no new evidence",
        )
    target = _failing_file(traceback, repo_files)
    if target is None:
        return CriticOutcome(
            status="no_target", signature=sig, reason="no repo frame in a repo file to patch"
        )
    file, line = target
    try:
        clone_text, truncated = read_text(clone_dir, file, max_bytes=MAX_FILE_BYTES)
    except (OSError, UnsafePathError) as e:
        return CriticOutcome(status="no_target", signature=sig, reason=f"cannot read {file}: {e}")
    if truncated:
        return CriticOutcome(
            status="no_target", signature=sig, reason=f"{file} is larger than {MAX_FILE_BYTES} B"
        )
    text = fold_text(clone_text, file, stack)
    texts = (metric_line_texts or {}).get(file, ())
    marked = protected_lines(text, texts)
    context: dict[str, Any] = {
        "repo_files": frozenset(repo_files),
        "failing_file": file,
        "file_text": text,
        "metric_lines": {file: marked},
    }
    trace: CriticTrace | None = None
    pre_attempts, pre_errors = 0, ()
    if mode == "react":
        res = run_react(
            llm,
            system=REACT_SYSTEM,
            prompt=critic_prompt(traceback, command, file, text, line, marked, failed_patches,
                                 rejected) + "\n" + REACT_TAIL,
            view=view_for(clone_dir, repo_files, stack, metric_line_texts),
            context=context,
            max_tokens=max_tokens,
            limits=react_limits,
        )  # fmt: skip
        if res.patch is not None:
            return _patched(res.patch, sig, file, text, res.submits, res.errors, (), res.trace)
        trace, pre_attempts, pre_errors = res.trace, res.submits, res.errors
    out = _single_shot(llm, traceback, command, file, text, line, marked, failed_patches,
                       rejected, sig, context, max_tokens, max_retries)  # fmt: skip
    if trace is None:
        return out
    return replace(out, gate_attempts=pre_attempts + out.gate_attempts,
                   errors=(*pre_errors, *out.errors),
                   trace=with_fallback(trace, out.status))  # fmt: skip


def _patched(
    patch: CriticPatch,
    sig: str,
    file: str,
    text: str,
    attempts: int,
    errors: tuple[str, ...],
    evicted: tuple[str, ...],
    trace: CriticTrace | None = None,
) -> CriticOutcome:
    diff = unified_diff(file, text, apply_line_edits(text, patch.edits))
    applied = AppliedPatch(
        file=file, diff=diff, edits=patch.edits, rationale=patch.rationale, error_signature=sig
    )
    return CriticOutcome(status="patched", signature=sig, reason=patch.rationale, patch=applied,
                         gate_attempts=attempts, errors=errors, file=file, evicted=evicted,
                         trace=trace)  # fmt: skip


def _single_shot(
    llm: LLM,
    traceback: ParsedTraceback,
    command: str,
    file: str,
    text: str,
    line: int,
    marked: frozenset[int],
    failed_patches: Sequence[AppliedPatch],
    rejected: Sequence[tuple[str, str]],
    sig: str,
    context: Mapping[str, Any],
    max_tokens: int,
    max_retries: int,
) -> CriticOutcome:
    """The single-shot gated critic call (with context eviction), as before ReAct."""
    try:
        call = call_with_eviction(
            llm,
            critic_prompt_builder(
                traceback, command, file, text, line, marked, failed_patches, rejected
            ),
            schema=CriticPatch,
            context=context,
            max_tokens=max_tokens,
            max_retries=max_retries,
            examples={
                "edits": '[{"start_line": 12, "end_line": 12, "original": "    print(x.data[0])", '
                '"replacement": "    print(x.item())"}]'
            },
        )
    except ContextEvictionFailed as e:  # clean fallback stop: counts as a failed attempt
        return CriticOutcome(
            status="invalid", signature=sig, reason=str(e), errors=(str(e),), file=file
        )
    gate = call.result
    if not gate.ok or gate.value is None:
        return CriticOutcome(
            status="invalid",
            signature=sig,
            reason=f"no valid patch after {gate.attempts} answer(s)",
            gate_attempts=gate.attempts,
            errors=gate.errors,
            file=file,
            evicted=call.evicted,
        )
    return _patched(gate.value, sig, file, text, gate.attempts, gate.errors, call.evicted)


# --- evidence ------------------------------------------------------------------------------


def critic_patch_evidence(
    outcome: CriticOutcome, *, source: str, smoke_evidence_id: str
) -> Evidence:
    """critic_patch evidence (authority llm, confidence inferred), linked to the smoke
    evidence whose traceback it answers. ok=False records a failed attempt (no diff)."""
    if not outcome.counts_as_attempt:
        raise ValueError(f"no critic_patch evidence for a {outcome.status!r} outcome")
    p = outcome.patch
    content: dict[str, Any] = {
        "ok": p is not None,
        "file": outcome.file,
        "error_signature": outcome.signature,
        "gate_attempts": outcome.gate_attempts,
        "gate_errors": list(outcome.errors),
        "rationale": p.rationale if p else None,
        "diff": p.diff if p else None,
        "patch_id": p.patch_id if p else None,
        "edits": [e.model_dump(mode="json") for e in p.edits] if p else [],
    }
    if outcome.evicted:
        content["evicted"] = list(outcome.evicted)
    return make_evidence(
        kind="critic_patch",
        source=source,
        locator=f"critic_patch:{smoke_evidence_id[:12]}",
        content=content,
        confidence="inferred",
        authority="llm",
        links=(smoke_evidence_id,),
    )


def critic_revert_evidence(
    patch: AppliedPatch, *, source: str, patch_evidence_id: str, smoke_evidence_id: str
) -> Evidence:
    """critic_revert evidence: the re-smoke after `patch` failed with the same error signature,
    so the patch is dropped from the stack. A code decision (authority rule, derived), linked
    to the critic_patch evidence and the new smoke evidence."""
    return make_evidence(
        kind="critic_revert",
        source=source,
        locator=f"critic_revert:{smoke_evidence_id[:12]}",
        content={
            "patch_id": patch.patch_id,
            "file": patch.file,
            "diff": patch.diff,
            "error_signature": patch.error_signature,
            "reason": "re-smoke after the patch failed with the same error signature",
        },
        confidence="derived",
        authority="rule",
        links=(patch_evidence_id, smoke_evidence_id),
    )


def critic_trace_evidence(
    outcome: CriticOutcome, *, source: str, smoke_evidence_id: str
) -> Evidence:
    """critic_trace evidence: the ReAct trajectory of one critic call (steps: tool, args,
    result size / truncated; outcome; fallback reason). Authority llm (the model chose the
    actions), confidence derived; linked to the smoke evidence whose traceback it answers."""
    t = outcome.trace
    if t is None:
        raise ValueError(f"no critic_trace evidence for a {outcome.status!r} outcome without trace")
    content = {"file": outcome.file, "error_signature": outcome.signature,
               "status": outcome.status, **t.as_json()}  # fmt: skip
    return make_evidence(
        kind="critic_trace",
        source=source,
        locator=f"critic_trace:{smoke_evidence_id[:12]}",
        content=content,
        confidence="derived",
        authority="llm",
        links=(smoke_evidence_id,),
    )


# --- human approval of a patch ----------------------------------------------------------------


def critic_request_digest(
    signature: str,
    command: str,
    stack: Sequence[AppliedPatch],
    failed: Sequence[AppliedPatch],
    rejections: Sequence[Evidence],
) -> str:
    """Everything the critic call depends on: the same inputs -> the same request."""
    return request_digest("critic_patch", {
        "signature": signature,
        "command": command,
        "stack": [p.patch_id for p in stack],
        "failed": [p.patch_id for p in failed],
        "rejected": [str(r.content.get("request_id")) for r in rejections],
    })  # fmt: skip


def critic_request_evidence(
    patch: AppliedPatch, *, digest: str, source: str, links: Sequence[str], reason: str
) -> Evidence:
    """The proposed patch, kept in the request (not on the stack) until a human decides."""
    return approval_request_evidence(
        action="critic_patch",
        subject=patch.file,
        cost={"changed_lines": diff_changed_lines(patch.diff)},
        detail={
            "file": patch.file,
            "diff": patch.diff,
            "edits": [e.model_dump(mode="json") for e in patch.edits],
            "rationale": patch.rationale,
            "error_signature": patch.error_signature,
        },
        input_digest=digest,
        source=source,
        links=links,
        reason=reason,
    )


def patch_from_request(request: Evidence) -> AppliedPatch:
    d = request.content["detail"]
    return AppliedPatch(
        file=str(d["file"]),
        diff=str(d["diff"]),
        edits=tuple(LineEdit.model_validate(e) for e in d["edits"]),
        rationale=str(d["rationale"]),
        error_signature=str(d["error_signature"]),
    )


def critic_rejection_evidence(request: Evidence, decision: Evidence) -> Evidence:
    """A human-rejected patch as a FAILED critic attempt (critic_patch ok=False, so the
    attempt cap counts it); a code record of the human decision (authority rule, derived),
    linked to the request and the decision."""
    d = request.content["detail"]
    return make_evidence(
        kind="critic_patch",
        source=request.source,
        locator=f"critic_patch:rejected:{request.evidence_id[:12]}",
        content={
            "ok": False,
            "file": d.get("file"),
            "error_signature": d.get("error_signature"),
            "gate_attempts": 0,
            "gate_errors": [],
            "rationale": d.get("rationale"),
            "diff": None,
            "patch_id": None,
            "edits": [],
            "rejected_by_human": True,
            "request_id": request.evidence_id,
            "rejected_diff": d.get("diff"),
            "note": decision.content.get("note", ""),
        },
        confidence="derived",
        authority="rule",
        links=(request.evidence_id, decision.evidence_id),
    )


def recorded_rejections(evidence: Mapping[str, Evidence], signature: str) -> list[Evidence]:
    return [ev for ev in evidence.values() if ev.kind == "critic_patch"
            and ev.content.get("rejected_by_human")
            and ev.content.get("error_signature") == signature]  # fmt: skip


def rejection_is_newest(evidence: Mapping[str, Evidence], signature: str) -> bool:
    """The newest critic attempt for this signature is a human rejection: new evidence for
    the critic (signature gating must not skip it)."""
    last = next((ev for ev in reversed(list(evidence.values())) if ev.kind == "critic_patch"
                 and ev.content.get("error_signature") == signature), None)  # fmt: skip
    return last is not None and bool(last.content.get("rejected_by_human"))


@dataclass(frozen=True)
class CriticState:
    """Folded from stored critic_patch / critic_revert evidence of one repo (file order).
    pending: the last patch if it is the newest critic event (its effect is not judged yet);
    failed: reverted patches, in order."""

    stack: tuple[AppliedPatch, ...] = ()
    last_signature: str | None = None
    attempts: int = 0
    pending: AppliedPatch | None = None
    failed: tuple[AppliedPatch, ...] = ()

    def push(self, outcome: CriticOutcome) -> CriticState:
        """State after a critic attempt (patched or invalid)."""
        p = outcome.patch
        return replace(
            self,
            stack=self.stack + ((p,) if p is not None else ()),
            last_signature=outcome.signature,
            attempts=self.attempts + 1,
            pending=p,
        )

    def revert(self, patch_id: str) -> tuple[CriticState, AppliedPatch | None]:
        """Drop the newest stack entry with patch_id; (new state, the dropped patch)."""
        idx = next((i for i in range(len(self.stack) - 1, -1, -1)
                    if self.stack[i].patch_id == patch_id), None)  # fmt: skip
        if idx is None:
            return replace(self, pending=None), None
        p = self.stack[idx]
        stack = self.stack[:idx] + self.stack[idx + 1 :]
        return replace(self, stack=stack, pending=None, failed=(*self.failed, p)), p

    def failed_for(self, signature: str) -> tuple[AppliedPatch, ...]:
        return tuple(p for p in self.failed if p.error_signature == signature)


def fold_critic_state(evidence: Iterable[Evidence]) -> CriticState:
    state = CriticState()
    for ev in evidence:
        c = ev.content
        if ev.kind == "critic_revert":
            state, _ = state.revert(str(c["patch_id"]))
        elif ev.kind == "critic_patch":
            sig = str(c["error_signature"])
            patch = None
            if c.get("ok"):
                patch = AppliedPatch(
                    file=str(c["file"]),
                    diff=str(c["diff"]),
                    edits=tuple(LineEdit.model_validate(e) for e in c["edits"]),
                    rationale=str(c["rationale"]),
                    error_signature=sig,
                )
            state = state.push(
                CriticOutcome(
                    status="patched" if patch else "invalid", signature=sig, reason="",
                    patch=patch,
                )
            )  # fmt: skip
    return state


# --- critic function for the smoke loop / graph ----------------------------------------------


def make_critic_fn(
    llm_factory: Callable[[], MeteredLLM],
    store: RunStore,
    *,
    clone_dir: Path,
    repo_files: Set[str],
    metric_texts: Mapping[str, Sequence[str]],
    on_calls: Callable[[Sequence[CallRecord]], None] | None = None,
    approval: ApprovalPolicy | None = None,
    source: str = "critic",
    trigger_ids: Sequence[str] = (),
    mode: CriticMode | None = None,
) -> Callable[..., CriticOutcome]:
    """The smoke loop's critic: the LLM is created only when the critic is actually needed
    (its calls are recorded in the store); on_calls gets the call records of each attempt.
    approval: gate validated patches (module docstring); None = no gate. trigger_ids: what
    an approval_request links to (the smoke's entrypoint evidence). mode: react | single
    (None: REPRODUCER_CRITIC_MODE from the environment, default react)."""
    critic_mode = critic_mode_from_env(os.environ) if mode is None else mode
    box: list[MeteredLLM] = []
    if approval is not None and not trigger_ids:
        raise ValueError("an approval-gated critic needs trigger_ids for its requests")

    def critic_fn(
        tb: ParsedTraceback,
        command: str,
        stack: tuple[AppliedPatch, ...],
        last: str | None,
        failed: tuple[AppliedPatch, ...],
    ) -> CriticOutcome:
        rejected: list[tuple[str, str]] = []
        digest = ""
        evidence: Mapping[str, Evidence] = {}
        if approval is not None:
            sig = error_signature(tb.model_dump(mode="json"))
            evidence = store.load_evidence()
            rejections = recorded_rejections(evidence, sig)
            digest = critic_request_digest(sig, command, stack, failed, rejections)
            stored = find_request(evidence, digest)
            if stored is not None:  # same inputs as an earlier call: no LLM call
                return _stored_outcome(store, evidence, stored, sig)
            if rejection_is_newest(evidence, sig):
                last = None  # the rejection is new evidence
            rejected = [(str(r.content.get("rejected_diff") or ""), str(r.content.get("note", "")))
                        for r in rejections]  # fmt: skip
        if not box:
            box.append(llm_factory())
            box[0].on_record = store.append_call_record
        llm = box[0]
        n0 = len(llm.records())
        out = run_critic(
            llm,
            traceback=tb,
            command=command,
            clone_dir=clone_dir,
            repo_files=repo_files,
            stack=stack,
            metric_line_texts=metric_texts,
            last_signature=last,
            failed_patches=failed,
            rejected=rejected,
            mode=critic_mode,
        )
        calls = llm.records()[n0:]
        if calls and on_calls is not None:
            on_calls(calls)
        if approval is None or out.status != "patched" or out.patch is None:
            return out
        request = critic_request_evidence(
            out.patch, digest=digest, source=source, links=trigger_ids, reason=""
        )
        g = gate(evidence, request, approval, diff_changed_lines(out.patch.diff))
        if g.status is None:  # the policy does not gate it: applied as before
            return out
        request = critic_request_evidence(
            out.patch, digest=digest, source=source, links=trigger_ids, reason=g.reason
        )
        store.append_evidence(request)
        return replace(out, status="awaiting_approval", patch=None, request_id=request.evidence_id,
                       reason=f"patch to {out.patch.file} awaits approval "
                              f"(request {request.evidence_id[:12]}: {g.reason})")  # fmt: skip

    return critic_fn


def _stored_outcome(
    store: RunStore, evidence: Mapping[str, Evidence], request: Evidence, sig: str
) -> CriticOutcome:
    """The outcome for a stored request: pending -> awaiting_approval; approved -> the stored
    patch (applied as today); rejected -> record the failed attempt, outcome rejected."""
    patch = patch_from_request(request)
    rid = request.evidence_id
    dec = decision_for(evidence, rid)
    if dec is None:
        return CriticOutcome(status="awaiting_approval", signature=sig, file=patch.file,
                             request_id=rid,
                             reason=f"patch to {patch.file} awaits approval (request "
                                    f"{rid[:12]})")  # fmt: skip
    if dec.content["decision"] == "approved":
        return CriticOutcome(status="patched", signature=sig, reason=patch.rationale,
                             patch=patch, file=patch.file, request_id=rid)  # fmt: skip
    store.append_evidence(critic_rejection_evidence(request, dec))
    note = str(dec.content.get("note", "")) or "(no note)"
    return CriticOutcome(status="rejected", signature=sig, file=patch.file, request_id=rid,
                         reason=f"human rejected patch {patch.patch_id[:12]} to {patch.file}: "
                                f"{note}")  # fmt: skip


def _critic_stop(
    state: Mapping[str, Any],
    deps: Mapping[str, Any],
    *,
    capped: bool,
    status: str,
    reason: str,
    e: Exception,
) -> dict[str, Any]:
    """The critic could not run: record the pending iteration (capped as given) and stop."""
    from reproducer.graph.nodes.smoke import record_iteration, summarize_iteration

    step, setup = state["pending"], state["smoke_setup"]
    it = replace(step.iteration, capped=capped)
    rec = record_iteration(deps["store"], Path(state["out_dir"]), setup, it)
    return {
        "pending": None,
        "last_critic_status": "error",
        "smoke_log": [summarize_iteration(state["smoke_iterations"], it, rec)],
        "setup_status": status,
        "stop_reason": reason,
        "stop_detail": f"critic error: {type(e).__name__}: {e}",
    }


def critic_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Graph node: one critic attempt for the pending `critic` smoke iteration, then record
    that iteration (smoke_result + critic_revert + critic_patch). patched -> re-smoke; skipped,
    no target, invalid -> capped; BudgetExceeded -> capped (stop_reason budget); LLM backend /
    config error -> llm_unavailable, NOT capped (retryable: the claims stay open).
    deps: store, llm_for."""
    from reproducer.graph.nodes.smoke import (  # smoke imports this module
        critic_step,
        record_iteration,
        summarize_iteration,
    )

    attempts = state.get("critic_attempts", 0)
    if attempts >= MAX_CRITIC_ATTEMPTS:  # route_smoke never sends us here at the cap
        raise RuntimeError(f"critic cap {MAX_CRITIC_ATTEMPTS} reached ({attempts} attempts)")
    step, setup = state["pending"], state["smoke_setup"]
    out, store = Path(state["out_dir"]), deps["store"]
    fn = make_critic_fn(
        lambda: deps["llm_for"](CRITIC_NODE),
        store,
        clone_dir=setup.repo_dir,
        repo_files=setup.repo_paths,
        metric_texts=setup.metric_texts,
        approval=policy_from_env(deps.get("env", os.environ)),
        source=setup.source,
        trigger_ids=(setup.entry_evidence.evidence_id,),
        mode=critic_mode_from_env(deps.get("env", os.environ)),
    )
    try:
        cs = critic_step(step.iteration, state["critic_state"], fn, step.failed)
    except BudgetExceeded as e:  # a cap: the claims are blocked
        return _critic_stop(state, deps, capped=True, status="capped", reason="budget", e=e)
    except (LLMError, ValueError) as e:  # backend down / not configured: retryable, stays open
        return _critic_stop(
            state, deps, capped=False, status="llm_unavailable", reason="llm_unavailable", e=e
        )
    rec = record_iteration(store, out, setup, cs.iteration)
    summary = summarize_iteration(state["smoke_iterations"], cs.iteration, rec)
    outcome = cs.iteration.critic
    update: dict[str, Any] = {
        "pending": None,
        "critic_state": cs.critic_state,
        "critic_attempts": cs.critic_state.attempts,
        "last_critic_status": "patched"
        if cs.stop is None
        else (outcome.status if outcome is not None else "error"),
        "smoke_log": [summary],
    }
    if cs.stop == "awaiting_approval":
        update.update(last_critic_status="awaiting_approval", setup_status="awaiting_approval",
                      stop_reason="awaiting_approval", stop_detail=cs.reason)  # fmt: skip
    elif cs.stop is not None:
        update.update(setup_status="capped", stop_reason=cs.stop, stop_detail=cs.reason)
    return update
