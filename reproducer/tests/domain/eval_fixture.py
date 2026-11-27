"""A fake paper store for the eval report with every counted item: the Cutout-like vault store
(claims, tolerances, plans, one completed run per claim) plus known fixes, critic patches
(first-try ok, retried ok, failed, human-rejected), a revert, two ReAct traces (one with a
redundant step and a fallback), approvals (approved, rejected, pending), lessons (one applied,
one not), and LLM calls over gated and ungated nodes (api + local, one evicted)."""

from __future__ import annotations

from typing import Any

try:  # collected from tests/domain (vault_render style) or from tests/ (cli style)
    from vault_fixture import C10, REPO, cutout_store
except ModuleNotFoundError:
    from domain.vault_fixture import C10, REPO, cutout_store

from reproducer.domain.approval import approval_decision_evidence, approval_request_evidence
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.telemetry import CallRecord
from reproducer.domain.unknowns import Assumption, UnknownOpened, UnknownResolved

UnknownEventT = UnknownOpened | UnknownResolved | Assumption


def _ev(
    kind: Any,
    locator: str,
    content: dict[str, Any],
    authority: Any = "rule",
    links: tuple[str, ...] = (),
) -> Evidence:
    return make_evidence(kind=kind, source=REPO, locator=locator, content=content,
                         confidence="derived", authority=authority, links=links)  # fmt: skip


def _call(
    node: str, backend: str, tin: int, tout: int, evicted: tuple[str, ...] = ()
) -> CallRecord:
    return CallRecord(node=node, backend=backend, model="m", input_tokens=tin,
                      output_tokens=tout, latency_s=1.5, prompt_sha256="0" * 64,
                      evicted=evicted)  # fmt: skip


def full_store() -> tuple[dict[str, Evidence], list[UnknownEventT], list[CallRecord]]:
    evidence, events, _ = cutout_store()
    extra: list[Evidence] = []
    smoke = next(ev.evidence_id for ev in evidence.values() if ev.kind == "preflight")
    extra += [_ev("known_fix", "known_fix:1", {"pattern": "missing_output_dir", "n": 1}),
              _ev("known_fix", "known_fix:2", {"pattern": "missing_output_dir", "n": 2}),
              _ev("known_fix", "known_fix:3", {"pattern": "oom"})]  # fmt: skip
    # critic, single mode: one critic_patch per smoke, gate_attempts = its answers
    for i, (ok, attempts) in enumerate([(True, 1), (True, 2), (False, 3)]):
        content = {"ok": ok, "gate_attempts": attempts, "file": "train.py"}
        extra.append(_ev("critic_patch", f"critic_patch:{i}", content, "llm", (f"single{i}",)))
    extra.append(_ev("critic_patch", "critic_patch:rejected:x",
                     {"ok": False, "gate_attempts": 0, "rejected_by_human": True}))  # fmt: skip
    extra.append(_ev("critic_revert", "critic_revert:1", {"patch_id": "p1"}))

    # critic, ReAct mode: one critic_trace per smoke (+ its critic_patch for the same smoke)
    def step(n: int, tool: str, args: dict[str, Any], note: str = "") -> dict[str, Any]:
        return {"step": n, "tool": tool, "args": args, "note": note}

    def trace(smoke_id: str, steps: list[dict[str, Any]], calls: int, **more: Any) -> Evidence:
        content = {
            "mode": "react",
            "outcome": "patched",
            "fallback_reason": None,
            "fallback_status": None,
            "steps": steps,
            "llm_calls": calls,
            **more,
        }
        return _ev("critic_trace", f"critic_trace:{smoke_id}", content, "llm", (smoke_id,))

    fix = {"file": "train.py", "edits": []}
    # A: investigates (one redundant read), first submit rejected, second accepted
    extra.append(trace("reactA", [
        step(1, "read_file", {"path": "a.py"}), step(2, "grep", {"q": "x"}),
        step(3, "read_file", {"path": "a.py"}, "rejected: repeated call"),
        step(4, "submit_patch", fix, "invalid patch"),
        step(5, "submit_patch", {**fix, "v": 2}, "accepted"),
    ], 5))  # fmt: skip
    # B: the toy_critic shape: read the helper, first submit accepted (its patch: same smoke)
    extra.append(trace("reactB", [
        step(1, "read_file", {"path": "utils/metrics.py", "start_line": 1, "end_line": 60}),
        step(2, "submit_patch", fix, "accepted"),
    ], 2))  # fmt: skip
    extra.append(
        _ev(
            "critic_patch",
            "critic_patch:reactB",
            {"ok": True, "gate_attempts": 1, "file": "train.py"},
            "llm",
            ("reactB",),
        )
    )
    # C: two rejected submits -> fallback to single-shot, which patched on its first answer
    extra.append(trace("reactC", [
        step(1, "read_file", {"path": "b.py"}), step(2, "submit_patch", fix, "invalid patch"),
        step(3, "submit_patch", {**fix, "x": 1}, "invalid patch"),
    ], 3, outcome="fallback", fallback_reason="2 rejected calls (loop detected)",
        fallback_status="patched"))  # fmt: skip
    extra.append(
        _ev(
            "critic_patch",
            "critic_patch:reactC",
            {"ok": True, "gate_attempts": 3, "file": "train.py"},
            "llm",
            ("reactC",),
        )
    )

    reqs = [approval_request_evidence(action=a, subject=C10, cost={}, detail={},
                                      input_digest=a * 4, source=REPO, links=(smoke,))
            for a in ("full_run", "critic_patch", "shared_vault_write")]  # fmt: skip
    extra += reqs
    # full_run: approved, a later reject is ignored (first counts); critic_patch: rejected;
    # shared_vault_write (standing): approved then revoked -> rejected (newest counts)
    for req, decision, rev in ((reqs[0], "approved", 0), (reqs[1], "rejected", 0),
                               (reqs[0], "rejected", 1), (reqs[2], "approved", 0),
                               (reqs[2], "rejected", 1)):  # fmt: skip
        extra.append(approval_decision_evidence(req, decision, "n", revision=rev))  # type: ignore[arg-type]

    def lesson(trial: int, step: str, kind: str) -> Evidence:
        les = {"step": step, "proposed_change": {"kind": kind}}
        content = {"claim_id": C10, "trial": trial, "lesson": les, "gate_attempts": 1}
        return _ev("lesson", f"lesson:{C10}:trial{trial}", content, "llm", (smoke,))

    l1 = lesson(1, "eval_protocol", "eval_protocol")
    l2 = lesson(2, "config_vs_paper", "arg")
    applied = make_evidence(
        kind="lesson_applied",
        source=REPO,
        locator=f"lesson_applied:{C10}:trial1",
        content={"claim_id": C10, "lesson_id": l1.evidence_id, "outcome": "reeval"},
        confidence="derived",
        authority="rule",
        links=(l1.evidence_id,),
    )
    extra += [l1, l2, applied]  # fmt: skip
    for n in (1, 2):
        meta = {"gate_attempts": 2} if n == 1 else {}  # n == 2: an older record without it
        content = {"n": n, **meta}
        extra.append(_ev("claim_mapping", "orchestrator:claim_mapping", content, "llm", (smoke,)))
    for i, (reason, detail) in enumerate([("fallback: env: RuntimeError", "disk on fire"),
                                          ("fallback: env: RuntimeError", "disk on fire"),
                                          ("fallback: route_smoke: x", "")]):  # fmt: skip
        where = reason.split(": ")[1]
        content = {"router_or_node": where, "reason": reason, "detail": detail}
        extra.append(
            make_evidence(
                kind="fallback",
                source="graph",
                locator=f"fallback:{i}",
                content=content,
                confidence="derived",
                authority="rule",
            )
        )
    for ev in extra:
        evidence[ev.evidence_id] = ev

    calls = [
        _call("subtask_a", "ollama", 100, 10),
        _call("orchestrator", "anthropic", 1000, 100),
        _call("orchestrator", "anthropic", 1200, 100, evicted=("repo files",)),
        _call("orchestrator", "anthropic", 1000, 100),
        _call("reflect", "anthropic", 500, 50),
        _call("reflect", "anthropic", 600, 50),
        _call("reflect", "anthropic", 500, 50),
        # ReAct 5 + 2 + 3 calls, single-shot after C's fallback 1, single mode 1 + 2 + 3, + 1
        *[_call("critic", "anthropic", 300, 30) for _ in range(18)],
        _call("misc", "ollama", 10, 1),
    ]
    return evidence, events, calls
