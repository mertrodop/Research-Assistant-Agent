"""Evaluation harness report (Lecture 6: outcome / trajectory / efficiency) over paper stores.

Pure domain: no file, network or subprocess IO. The CLI (`eval-report`) reads each store
read-only (RunStore(create=False)) and passes evidence, folded unknowns and LLM call records
in; everything here is recomputed from them, deterministically (sorted, no clock).

Outcome per claim      exactly evaluate_claims (= compute_ledger): paper vs ours, |diff|, the
                       pre-registered tolerance, status, fidelity; inferred reasons counted by
                       kind (info notes never count).
Trajectory per paper
  gates                per LLM node: calls (llm_calls.jsonl) and accepted gated outputs
                       (evidence the gate produced: subtask_a env_spec subtask:A, subtask_b
                       entrypoints subtask:B, subtask_c workflow subtask:C, orchestrator
                       claim_mapping, reflect lesson). New records carry gate_attempts (the
                       tries of the gate that answered): exact, pass@1 = first-try / accepted.
                       Older records without it are ESTIMATED from the calls the exact ones do
                       not explain: retries = rest - old, first-try = max(0, old - retries)
                       (exact when every retried gate passed on its 2nd try and none failed
                       outright; else a lower bound). "other calls" = calls not explained by
                       stored attempts (gates that failed outright, eviction rebuilds).
  critic gate          one invocation per ReAct critic_trace (mode react): accepted when a
                       submit_patch step was accepted (or, after a fallback, the single-shot
                       critic patched); retries = submits rejected before it (+ the extra
                       single-shot answers after a fallback); pass@1 = accepted on the first
                       submit. Tool steps are reported as steps (each is one LLM call, not a
                       retry). A critic_patch without a ReAct trace for its smoke (single
                       mode) is one invocation with its gate_attempts (first-try ok / gated).
  known fixes          known_fix evidence, by pattern
  critic               LLM calls of node critic; patches applied (critic_patch ok) / failed /
                       rejected by a human; reverted (critic_revert)
  ReAct                critic_trace: traces, steps, redundant calls (a step repeating an
                       earlier step's tool + args in the same trace), fallbacks to single-shot
  approvals            approval_request (by action) and the decision that counts for each
                       (approval.decision_for: approved / rejected / pending)
  lessons              per lesson evidence: claim, trial, step, change kind, and what code did
                       with it (lesson_applied outcome, "-" if not applied)
  trials               run_result evidence per claim
  breaker trips        breaker_state over the store (GPU cap / token budget / max trials given)
  fallbacks            by reason: `fallback` evidence (one per graph invocation that took the
                       default fallback route) + critic ReAct fallbacks to single-shot
                       (critic_trace fallback_reason, prefixed "critic react: ")
  evictions            LLM calls whose prompt had sections evicted
Efficiency per paper   LLM calls + tokens api / local by node, GPU hours and wall time of every
                       run_result.
Totals                 claims verified (and how many spec-faithful), open, blocked,
                       contradicted; GPU hours; tokens.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.approval import decision_for
from reproducer.domain.breaker import breaker_state
from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evaluate import ClaimEvaluation, evaluate_claims
from reproducer.domain.evidence import Evidence
from reproducer.domain.metric_format import format_metric, metric_decimals
from reproducer.domain.telemetry import LOCAL_BACKENDS, CallRecord
from reproducer.domain.unknowns import UnknownState

CaseStatus = Literal["ok", "not run", "error"]
STATUSES = ("verified", "open", "blocked", "contradicted")
CRITIC_NODE = "critic"
# node -> (evidence kind, locator or None for any) of the gate's accepted output
GATED_OUTPUTS: dict[str, tuple[str, str | None]] = {
    "subtask_a": ("env_spec", "subtask:A"),
    "subtask_b": ("entrypoints", "subtask:B"),
    "subtask_c": ("workflow", "subtask:C"),
    "orchestrator": ("claim_mapping", None),
    "reflect": ("lesson", None),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimOutcome(_Frozen):
    claim_id: str
    metric: str
    unit: str | None
    paper: float
    ours: float | None
    abs_diff: float | None
    tolerance: float  # the bound |diff| is compared with
    within_tolerance: bool | None
    status: str
    fidelity: str | None  # set only when verified (ledger)
    run_fidelity: str | None  # the best run's
    inferred_reasons: dict[str, int]  # kind -> count
    decimals: int


class GateStat(_Frozen):
    node: str
    calls: int
    accepted: int | None  # None: not a gated node
    retries: int | None
    pass_at_1: float | None
    exact: bool  # every accepted output (or critic attempt) stored its gate_attempts
    other_calls: int = 0  # calls not explained by stored attempts (failed gates, evictions)
    steps: int | None = None  # critic: ReAct tool steps (one LLM call each, not retries)


class LessonRow(_Frozen):
    claim_id: str
    trial: int
    step: str
    change_kind: str
    outcome: str  # lesson_applied outcome, "-" if none


class NodeUsage(_Frozen):
    node: str
    calls: int
    api_tokens: int
    local_tokens: int
    latency_s: float


class Trajectory(_Frozen):
    gates: tuple[GateStat, ...]
    known_fixes: int
    known_fixes_by_pattern: dict[str, int]
    critic_calls: int
    patches_applied: int
    patches_failed: int
    patches_rejected: int
    patches_reverted: int
    react_traces: int
    react_steps: int
    react_redundant: int
    approvals_requested: int
    approvals_by_action: dict[str, int]
    approvals_approved: int
    approvals_rejected: int
    approvals_pending: int
    lessons: tuple[LessonRow, ...]
    lessons_per_claim: dict[str, int]
    trials_per_claim: dict[str, int]
    breaker_trips: tuple[str, ...]
    fallbacks: int
    fallbacks_by_reason: dict[str, int]
    evictions: int


class Efficiency(_Frozen):
    by_node: tuple[NodeUsage, ...]
    llm_calls: int
    api_tokens: int
    local_tokens: int
    gpu_hours: float
    runs: int
    wall_time_s: float


class PaperReport(_Frozen):
    name: str
    spec: str
    out: str
    note: str
    status: CaseStatus
    reason: str = ""
    claims: tuple[ClaimOutcome, ...] = ()
    trajectory: Trajectory | None = None
    efficiency: Efficiency | None = None


class Totals(_Frozen):
    papers_run: int
    papers_not_run: int
    papers_error: int
    claims: int
    verified: int
    verified_spec_faithful: int
    spec_faithful_pct: float | None  # of verified
    open: int
    blocked: int
    contradicted: int
    gpu_hours: float
    llm_calls: int
    api_tokens: int
    local_tokens: int


class EvalReport(_Frozen):
    papers: tuple[PaperReport, ...]
    totals: Totals


# --- per paper ------------------------------------------------------------------------------


def _bound(e: ClaimEvaluation) -> float:
    return max(e.tolerance.abs or 0.0, (e.tolerance.rel or 0.0) * abs(e.paper_value))


def claim_outcomes(
    claims: Sequence[ResultClaim], evals: Mapping[str, ClaimEvaluation]
) -> tuple[ClaimOutcome, ...]:
    out = []
    for c in claims:
        e = evals[c.claim_id]
        out.append(ClaimOutcome(
            claim_id=c.claim_id, metric=e.metric, unit=e.unit, paper=e.paper_value,
            ours=e.our_value, abs_diff=e.abs_diff, tolerance=_bound(e),
            within_tolerance=e.within_tolerance, status=e.status, fidelity=e.fidelity,
            run_fidelity=e.run_fidelity,
            inferred_reasons=dict(sorted(Counter(r.kind for r in e.inferred_reasons).items())),
            decimals=metric_decimals(c),
        ))  # fmt: skip
    return tuple(out)


def _of(evidence: Iterable[Evidence], kind: str) -> list[Evidence]:
    return [ev for ev in evidence if ev.kind == kind]


def _attempts(ev: Evidence) -> int | None:
    v = ev.content.get("gate_attempts")
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _gate(node: str, n: int, records: Sequence[Evidence]) -> GateStat:
    """One gated node: exact for records with gate_attempts, estimated for the others."""
    exact = [a for a in (_attempts(ev) for ev in records) if a is not None]
    old = len(records) - len(exact)
    tries = sum(exact)
    retries_old = max(0, n - tries - old) if old else 0
    first = sum(1 for a in exact if a == 1) + max(0, old - retries_old)
    accepted = len(records)
    p1 = round(first / accepted, 4) if accepted else (0.0 if n else None)
    return GateStat(
        node=node, calls=n, accepted=accepted, retries=(tries - len(exact)) + retries_old,
        pass_at_1=p1, exact=old == 0 and accepted > 0,
        other_calls=max(0, n - tries - old - retries_old),
    )  # fmt: skip


def _submits(trace: Evidence) -> list[str]:
    """The notes of the trace's submit_patch steps, in order."""
    steps = trace.content.get("steps") or []
    return [str(s.get("note") or "") for s in steps
            if isinstance(s, Mapping) and s.get("tool") == "submit_patch"]  # fmt: skip


def critic_gate(evidence: Sequence[Evidence], n: int) -> GateStat | None:
    """The critic's gate (see the module docstring): ReAct traces + single-mode patches."""
    traces = [t for t in _of(evidence, "critic_trace") if t.content.get("mode") == "react"]
    patches = _of(evidence, "critic_patch")
    patch_by_smoke = {p.links[0]: p for p in patches if p.links and _attempts(p)}
    traced = {t.links[0] for t in traces if t.links}
    invocations = accepted = first = retries = steps = react_calls = single_calls = 0
    for t in traces:
        invocations += 1
        notes = _submits(t)
        steps += len(t.content.get("steps") or [])
        react_calls += int(t.content.get("llm_calls") or 0)
        hit = next((i for i, x in enumerate(notes) if x.endswith("accepted")), None)
        if hit is not None:
            accepted += 1
            first += hit == 0
            retries += hit
            continue
        retries += len(notes)
        single = patch_by_smoke.get(t.links[0]) if t.links else None
        if single is not None:  # fallback to single-shot: its answers beyond the ReAct submits
            answers = max(0, (_attempts(single) or 0) - len(notes))
            single_calls += answers
            retries += max(0, answers - 1)
        if t.content.get("fallback_status") == "patched":
            accepted += 1
    for p in patches:  # single mode: a gated critic_patch without a ReAct trace for its smoke
        a = _attempts(p) or 0
        if a == 0 or (p.links and p.links[0] in traced):
            continue
        invocations += 1
        single_calls += a
        retries += a - 1
        if p.content.get("ok"):
            accepted += 1
            first += a == 1
    if invocations == 0 and n == 0:
        return None
    return GateStat(
        node=CRITIC_NODE, calls=n, accepted=accepted, retries=retries,
        pass_at_1=round(first / invocations, 4) if invocations else None, exact=True,
        other_calls=max(0, n - react_calls - single_calls), steps=steps,
    )  # fmt: skip


def gate_stats(evidence: Sequence[Evidence], calls: Sequence[CallRecord]) -> tuple[GateStat, ...]:
    by_node = Counter(r.node for r in calls)
    nodes = sorted(set(by_node) | set(GATED_OUTPUTS) | {CRITIC_NODE})
    out: list[GateStat] = []
    for node in nodes:
        n = by_node.get(node, 0)
        if node == CRITIC_NODE:
            stat = critic_gate(evidence, n)
            if stat is not None:
                out.append(stat)
            continue
        if node not in GATED_OUTPUTS:
            out.append(GateStat(node=node, calls=n, accepted=None, retries=None,
                                pass_at_1=None, exact=False, other_calls=n))  # fmt: skip
            continue
        kind, locator = GATED_OUTPUTS[node]
        records = [ev for ev in _of(evidence, kind) if locator is None or ev.locator == locator]
        if n == 0 and not records:
            continue
        out.append(_gate(node, n, records))
    return tuple(out)


def _redundant_steps(steps: Sequence[Any]) -> int:
    seen: set[str] = set()
    n = 0
    for s in steps:
        if not isinstance(s, Mapping):
            continue
        key = json.dumps([s.get("tool"), s.get("args")], sort_keys=True, default=str)
        if key in seen:
            n += 1
        seen.add(key)
    return n


def trajectory(
    claims: Sequence[ResultClaim],
    evidence: Mapping[str, Evidence],
    calls: Sequence[CallRecord],
    *,
    gpu_h_cap: float,
    token_budget: int,
    max_trials: int,
) -> Trajectory:
    evs = list(evidence.values())
    claim_ids = [c.claim_id for c in claims]

    fixes = _of(evs, "known_fix")
    patches = _of(evs, "critic_patch")
    traces = _of(evs, "critic_trace")
    steps = [t.content.get("steps") or [] for t in traces]
    fallback_reasons = [str(ev.content.get("reason")) for ev in _of(evs, "fallback")] + [
        f"critic react: {t.content['fallback_reason']}"
        for t in traces
        if t.content.get("fallback_reason")
    ]

    requests = _of(evs, "approval_request")
    counted = [decision_for(evidence, r.evidence_id) for r in requests]
    decided = [None if d is None else d.content.get("decision") for d in counted]

    applied: dict[str, str] = {}
    for ev in _of(evs, "lesson_applied"):
        applied.setdefault(str(ev.content.get("lesson_id")), str(ev.content.get("outcome")))
    lessons: list[LessonRow] = []
    for ev in _of(evs, "lesson"):
        les = ev.content.get("lesson") or {}
        change = les.get("proposed_change") or {}
        lessons.append(LessonRow(
            claim_id=str(ev.content.get("claim_id")), trial=int(ev.content.get("trial") or 0),
            step=str(les.get("step", "-")), change_kind=str(change.get("kind", "none")),
            outcome=applied.get(ev.evidence_id, "-"),
        ))  # fmt: skip
    lessons.sort(key=lambda r: (r.claim_id, r.trial, r.step, r.change_kind, r.outcome))

    runs = _of(evs, "run_result")
    trials = {cid: sum(1 for r in runs if r.content.get("claim_id") == cid) for cid in claim_ids}
    used = sum(r.input_tokens + r.output_tokens for r in calls)
    breaker = breaker_state(claim_ids, evidence, gpu_h_cap=gpu_h_cap, token_budget=token_budget,
                            tokens_used=used, max_per_claim=max_trials)  # fmt: skip

    return Trajectory(
        gates=gate_stats(evs, calls),
        known_fixes=len(fixes),
        known_fixes_by_pattern=dict(
            sorted(Counter(str(f.content.get("pattern")) for f in fixes).items())
        ),
        critic_calls=sum(1 for r in calls if r.node == CRITIC_NODE),
        patches_applied=sum(1 for p in patches if p.content.get("ok")),
        patches_failed=sum(
            1 for p in patches if not p.content.get("ok") and not p.content.get("rejected_by_human")
        ),
        patches_rejected=sum(1 for p in patches if p.content.get("rejected_by_human")),
        patches_reverted=len(_of(evs, "critic_revert")),
        react_traces=len(traces),
        react_steps=sum(len(s) for s in steps),
        react_redundant=sum(_redundant_steps(s) for s in steps),
        approvals_requested=len(requests),
        approvals_by_action=dict(
            sorted(Counter(str(r.content.get("action")) for r in requests).items())
        ),
        approvals_approved=sum(1 for d in decided if d == "approved"),
        approvals_rejected=sum(1 for d in decided if d == "rejected"),
        approvals_pending=sum(1 for d in decided if d is None),
        lessons=tuple(lessons),
        lessons_per_claim={cid: sum(1 for r in lessons if r.claim_id == cid) for cid in claim_ids},
        trials_per_claim=trials,
        breaker_trips=breaker.tripped,
        fallbacks=len(fallback_reasons),
        fallbacks_by_reason=dict(sorted(Counter(fallback_reasons).items())),
        evictions=sum(1 for r in calls if r.evicted),
    )


def efficiency(evidence: Mapping[str, Evidence], calls: Sequence[CallRecord]) -> Efficiency:
    nodes: dict[str, dict[str, Any]] = {}
    for r in calls:
        n = nodes.setdefault(r.node, {"calls": 0, "api": 0, "local": 0, "lat": 0.0})
        n["calls"] += 1
        n["local" if r.backend in LOCAL_BACKENDS else "api"] += r.input_tokens + r.output_tokens
        n["lat"] += r.latency_s
    by_node = tuple(
        NodeUsage(node=k, calls=v["calls"], api_tokens=v["api"], local_tokens=v["local"],
                  latency_s=round(v["lat"], 2))
        for k, v in sorted(nodes.items())
    )  # fmt: skip
    runs = _of(evidence.values(), "run_result")

    def num(ev: Evidence, key: str) -> float:
        v = ev.content.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    return Efficiency(
        by_node=by_node,
        llm_calls=len(calls),
        api_tokens=sum(n.api_tokens for n in by_node),
        local_tokens=sum(n.local_tokens for n in by_node),
        gpu_hours=round(sum(num(r, "gpu_hours") for r in runs), 4),
        runs=len(runs),
        wall_time_s=round(sum(num(r, "duration") for r in runs), 1),
    )


def paper_report(
    case: Mapping[str, str],
    claims: Sequence[ResultClaim],
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    calls: Sequence[CallRecord],
    *,
    gpu_h_cap: float,
    token_budget: int,
    max_trials: int,
) -> PaperReport:
    """One store's report. No claims -> "not run"; no pre-registered tolerance -> "error"."""
    base = {k: str(case.get(k, "")) for k in ("name", "spec", "out", "note")}
    if not claims:
        return PaperReport(**base, status="not run", reason="no paper_claim evidence")
    try:
        evals = evaluate_claims(claims, evidence, unknowns, max_tries=max_trials)
    except ValueError as e:
        return PaperReport(**base, status="error", reason=str(e))
    return PaperReport(
        **base,
        status="ok",
        claims=claim_outcomes(claims, evals),
        trajectory=trajectory(
            claims,
            evidence,
            calls,
            gpu_h_cap=gpu_h_cap,
            token_budget=token_budget,
            max_trials=max_trials,
        ),
        efficiency=efficiency(evidence, calls),
    )


def not_run(case: Mapping[str, str], reason: str) -> PaperReport:
    base = {k: str(case.get(k, "")) for k in ("name", "spec", "out", "note")}
    return PaperReport(**base, status="not run", reason=reason)


def totals(papers: Sequence[PaperReport]) -> Totals:
    claims = [c for p in papers for c in p.claims]
    st = Counter(c.status for c in claims)
    faithful = sum(1 for c in claims if c.status == "verified" and c.fidelity == "spec_faithful")
    effs = [p.efficiency for p in papers if p.efficiency is not None]
    return Totals(
        papers_run=sum(1 for p in papers if p.status == "ok"),
        papers_not_run=sum(1 for p in papers if p.status == "not run"),
        papers_error=sum(1 for p in papers if p.status == "error"),
        claims=len(claims),
        verified=st["verified"],
        verified_spec_faithful=faithful,
        spec_faithful_pct=round(100.0 * faithful / st["verified"], 1) if st["verified"] else None,
        open=st["open"],
        blocked=st["blocked"],
        contradicted=st["contradicted"],
        gpu_hours=round(sum(e.gpu_hours for e in effs), 4),
        llm_calls=sum(e.llm_calls for e in effs),
        api_tokens=sum(e.api_tokens for e in effs),
        local_tokens=sum(e.local_tokens for e in effs),
    )


def build_report(papers: Sequence[PaperReport]) -> EvalReport:
    return EvalReport(papers=tuple(papers), totals=totals(papers))


def report_json(report: EvalReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


# --- markdown -------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return lines


def _fmt(v: float | None, decimals: int) -> str:
    return "-" if v is None else format_metric(v, decimals)


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def _counts(d: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def render_markdown(report: EvalReport) -> str:
    t = report.totals
    md: list[str] = ["# Evaluation report", ""]
    md.append("Recomputed read-only from each paper store (outcome / trajectory / efficiency).")
    md.append("")
    md.append("## Totals")
    md.append("")
    faithful = "-" if t.spec_faithful_pct is None else f"{t.spec_faithful_pct:g}%"
    md += _table(
        ["papers run", "not run", "error", "claims", "verified", "spec-faithful (of verified)",
         "open", "blocked", "contradicted", "GPU h", "LLM calls", "tokens api", "tokens local"],
        [[t.papers_run, t.papers_not_run, t.papers_error, t.claims, t.verified,
          f"{t.verified_spec_faithful} ({faithful})", t.open, t.blocked, t.contradicted,
          f"{t.gpu_hours:.3f}", t.llm_calls, t.api_tokens, t.local_tokens]],
    )  # fmt: skip
    md.append("")
    md.append("## Papers")
    md.append("")
    md += _table(
        ["case", "status", "store", "spec", "note"],
        [[p.name, p.status + (f" ({p.reason})" if p.reason else ""), p.out, p.spec, p.note or "-"]
         for p in report.papers],
    )  # fmt: skip

    for p in report.papers:
        if p.status != "ok" or p.trajectory is None or p.efficiency is None:
            continue
        tr, ef = p.trajectory, p.efficiency
        md += ["", f"## {p.name}", "", "### Outcome", ""]
        md += _table(
            ["claim", "metric", "paper", "ours", "|diff|", "tol", "status", "fidelity",
             "inferred reasons"],
            [[c.claim_id, c.metric + (f" ({c.unit})" if c.unit else ""),
              _fmt(c.paper, c.decimals), _fmt(c.ours, c.decimals), _fmt(c.abs_diff, c.decimals),
              _fmt(c.tolerance, c.decimals), c.status,
              c.fidelity or f"- (best run: {c.run_fidelity or '-'})",
              _counts(c.inferred_reasons)] for c in p.claims],
        )  # fmt: skip
        md += ["", "### Trajectory", "", "Gate attempts per node (exact from stored "
               "gate_attempts; est. = estimated from calls vs accepted outputs):", ""]  # fmt: skip
        md += _table(
            ["node", "LLM calls", "accepted", "retries", "steps", "other calls", "pass@1",
             "exact"],
            [[g.node, g.calls, "-" if g.accepted is None else g.accepted,
              "-" if g.retries is None else g.retries, "-" if g.steps is None else g.steps,
              g.other_calls, _pct(g.pass_at_1), "yes" if g.exact else "est."]
             for g in tr.gates],
        )  # fmt: skip
        md.append("")
        rows = [
            ["known fixes", f"{tr.known_fixes} ({_counts(tr.known_fixes_by_pattern)})"],
            ["critic calls", tr.critic_calls],
            ["critic patches applied / failed / rejected / reverted",
             f"{tr.patches_applied} / {tr.patches_failed} / {tr.patches_rejected} / "
             f"{tr.patches_reverted}"],
            ["ReAct traces / steps / redundant calls",
             f"{tr.react_traces} / {tr.react_steps} / {tr.react_redundant}"],
            ["approvals requested / approved / rejected / pending",
             f"{tr.approvals_requested} ({_counts(tr.approvals_by_action)}) / "
             f"{tr.approvals_approved} / {tr.approvals_rejected} / {tr.approvals_pending}"],
            ["trials per claim", _counts(tr.trials_per_claim)],
            ["lessons per claim", _counts(tr.lessons_per_claim)],
            ["breaker trips", "; ".join(tr.breaker_trips) or "none"],
            ["fallbacks", f"{tr.fallbacks}" + (f" ({_counts(tr.fallbacks_by_reason)})"
                                               if tr.fallbacks_by_reason else "")],
            ["evictions", tr.evictions],
        ]  # fmt: skip
        md += _table(["item", "count"], rows)
        if tr.lessons:
            md += ["", "Lessons:", ""]
            md += _table(["claim", "trial", "step", "change", "applied"],
                         [[r.claim_id, r.trial, r.step, r.change_kind, r.outcome]
                          for r in tr.lessons])  # fmt: skip
        md += ["", "### Efficiency", ""]
        md += _table(
            ["node", "calls", "tokens api", "tokens local", "latency s"],
            [[n.node, n.calls, n.api_tokens, n.local_tokens, f"{n.latency_s:.1f}"]
             for n in ef.by_node]
            + [["**total**", ef.llm_calls, ef.api_tokens, ef.local_tokens, ""]],
        )  # fmt: skip
        md.append("")
        md += _table(
            ["runs", "GPU h", "run wall time"],
            [[ef.runs, f"{ef.gpu_hours:.4f}", f"{ef.wall_time_s:.1f} s "
              f"({ef.wall_time_s / 3600:.2f} h)"]],
        )  # fmt: skip
    return "\n".join(md) + "\n"
