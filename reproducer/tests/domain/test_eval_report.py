"""Tests for reproducer.domain.eval_report (pure aggregation over one fake store)."""

from __future__ import annotations

import json

from eval_fixture import full_store
from vault_fixture import C10, C100

from reproducer.domain.eval_report import (
    build_report,
    critic_gate,
    gate_stats,
    not_run,
    paper_report,
    render_markdown,
    report_json,
)
from reproducer.domain.evaluate import evaluate_claims
from reproducer.domain.evidence import make_evidence
from reproducer.domain.telemetry import CallRecord
from reproducer.domain.unknowns import fold_unknowns
from reproducer.graph.nodes.orchestrator import paper_claims

CASE = {"name": "cutout", "spec": "evals/cutout/spec.json", "out": "runs/x", "note": "n"}


def _report(max_trials: int = 3, token_budget: int = 200_000):
    evidence, events, calls = full_store()
    unknowns = fold_unknowns(events, evidence)
    claims = tuple(c for c, _ in paper_claims(evidence).values())
    rep = paper_report(CASE, claims, evidence, unknowns, calls, gpu_h_cap=10.0,
                       token_budget=token_budget, max_trials=max_trials)  # fmt: skip
    return rep, claims, evidence, unknowns


def test_outcome_matches_evaluator() -> None:
    rep, claims, evidence, unknowns = _report()
    evals = evaluate_claims(claims, evidence, unknowns)
    assert rep.status == "ok"
    assert [c.claim_id for c in rep.claims] == [C10, C100]
    for c in rep.claims:
        e = evals[c.claim_id]
        assert (c.paper, c.ours, c.abs_diff, c.status, c.fidelity) == (
            e.paper_value, e.our_value, e.abs_diff, e.status, e.fidelity)  # fmt: skip
        assert sum(c.inferred_reasons.values()) == len(e.inferred_reasons)
        assert c.inferred_reasons["env_deviation"] == 1  # the torch deviation of the env plan
        assert list(c.inferred_reasons) == sorted(c.inferred_reasons)


def test_gates_estimate_and_critic_exact() -> None:
    rep, *_ = _report()
    assert rep.trajectory is not None
    g = {s.node: s for s in rep.trajectory.gates}
    # mixed: one mapping with gate_attempts 2 (exact, not first-try) + one older record: the 1
    # call the exact one does not explain is that record's first try (estimate)
    o = g["orchestrator"]
    assert (o.calls, o.accepted, o.retries, o.pass_at_1, o.exact, o.other_calls) == (
        3, 2, 1, 0.5, False, 0)  # fmt: skip
    # exact: both lessons first-try; the 3rd call is not explained by any stored lesson
    r = g["reflect"]
    assert (r.calls, r.accepted, r.retries, r.pass_at_1, r.exact, r.other_calls) == (
        3, 2, 0, 1.0, True, 1)  # fmt: skip
    # old records only (no gate_attempts): the estimate, as before
    a = g["subtask_a"]
    assert (a.calls, a.accepted, a.pass_at_1, a.exact) == (1, 1, 1.0, False)
    # critic: 6 invocations = ReAct traces A, B, C + single-mode patches (1 ok, 2 ok, 3 failed);
    # B's patch is not counted again, the human-rejected one is not a gate
    c = g["critic"]
    assert (c.calls, c.accepted, c.pass_at_1, c.exact) == (18, 5, 0.3333, True)  # B + single 1
    # A 1 rejected submit, C 2 (+ 0 extra single-shot answers), single mode 0 + 1 + 2
    assert c.retries == 6
    assert c.steps == 10  # tool steps of A, B, C: LLM calls, not retries
    assert c.other_calls == 1  # 18 - ReAct 10 - single-shot 7
    assert g["misc"].accepted is None and g["misc"].pass_at_1 is None


def test_trajectory_counts() -> None:
    rep, *_ = _report()
    tr = rep.trajectory
    assert tr is not None
    assert tr.known_fixes == 3
    assert tr.known_fixes_by_pattern == {"missing_output_dir": 2, "oom": 1}
    assert tr.critic_calls == 18
    # applied: single mode 2 + ReAct B + C (single-shot after its fallback)
    applied = (tr.patches_applied, tr.patches_failed, tr.patches_rejected, tr.patches_reverted)
    assert applied == (4, 1, 1, 1)
    assert (tr.react_traces, tr.react_steps, tr.react_redundant) == (3, 10, 1)
    assert tr.fallbacks == 4
    assert tr.fallbacks_by_reason == {
        "critic react: 2 rejected calls (loop detected)": 1,
        "fallback: env: RuntimeError": 2,  # two invocations, two records
        "fallback: route_smoke: x": 1,
    }
    assert tr.approvals_requested == 3
    assert tr.approvals_by_action == {"critic_patch": 1, "full_run": 1, "shared_vault_write": 1}
    # full_run: first decision (approved); critic_patch rejected; standing vault write: newest
    # (rejected)
    assert (tr.approvals_approved, tr.approvals_rejected, tr.approvals_pending) == (1, 2, 0)
    assert [(r.trial, r.step, r.change_kind, r.outcome) for r in tr.lessons] == [
        (1, "eval_protocol", "eval_protocol", "reeval"),
        (2, "config_vs_paper", "arg", "-"),
    ]
    assert tr.lessons_per_claim == {C10: 2, C100: 0}
    assert tr.trials_per_claim == {C10: 1, C100: 1}
    assert tr.breaker_trips == ()
    assert tr.evictions == 1


def test_critic_gate_toy_critic_store_shape() -> None:
    """runs/toy_critic: one ReAct trace (read_file utils/metrics.py, submit accepted), its
    approved critic_patch with gate_attempts 0, 2 critic calls -> 1 accepted on the first
    submit, 2 steps, no retries."""
    smoke = "s" * 64
    trace = make_evidence(kind="critic_trace", source="r", locator=f"critic_trace:{smoke[:12]}",
                          content={"mode": "react", "outcome": "patched", "llm_calls": 2,
                                   "fallback_reason": None, "fallback_status": None,
                                   "status": "awaiting_approval",
                                   "steps": [{"step": 1, "tool": "read_file", "note": "",
                                              "args": '{"path":"utils/metrics.py"}'},
                                             {"step": 2, "tool": "submit_patch",
                                              "note": "accepted", "args": "{}"}]},
                          confidence="derived", authority="llm", links=(smoke,))  # fmt: skip
    patch = make_evidence(kind="critic_patch", source="r", locator=f"critic_patch:{smoke[:12]}",
                          content={"ok": True, "gate_attempts": 0, "file": "train.py"},
                          confidence="inferred", authority="llm", links=(smoke,))  # fmt: skip
    calls = [CallRecord(node="critic", backend="openrouter", model="m", input_tokens=1,
                        output_tokens=1, latency_s=1.0, prompt_sha256="0" * 64)] * 2  # fmt: skip
    c = critic_gate([trace, patch], 2)
    assert c is not None
    assert (c.calls, c.accepted, c.retries, c.steps, c.pass_at_1, c.other_calls) == (
        2, 1, 0, 2, 1.0, 0)  # fmt: skip
    assert {g.node: g for g in gate_stats([trace, patch], calls)}["critic"] == c


def test_breaker_trips() -> None:
    rep, *_ = _report(max_trials=1, token_budget=100)
    tr = rep.trajectory
    assert tr is not None
    assert any(t.startswith(f"claim {C10}: 1/1") for t in tr.breaker_trips)
    assert any(t.startswith("global:") for t in tr.breaker_trips)
    assert any(t.startswith("tokens:") for t in tr.breaker_trips)


def test_efficiency_by_node_and_backend() -> None:
    rep, _, evidence, _ = _report()
    ef = rep.efficiency
    assert ef is not None
    by = {n.node: n for n in ef.by_node}
    assert (by["subtask_a"].api_tokens, by["subtask_a"].local_tokens) == (0, 110)
    assert by["orchestrator"].api_tokens == 1100 + 1300 + 1100
    assert (ef.llm_calls, ef.local_tokens) == (26, 110 + 11)
    assert ef.api_tokens == sum(n.api_tokens for n in ef.by_node)
    runs = [ev for ev in evidence.values() if ev.kind == "run_result"]
    assert ef.runs == 2
    assert ef.gpu_hours == round(sum(r.content["gpu_hours"] for r in runs), 4)
    assert ef.wall_time_s == round(5804.9 + 6074.4, 1)


def test_totals_and_not_run() -> None:
    rep, *_ = _report()
    missing = not_run({"name": "mixup", "out": "runs/mixup"}, "no store at runs/mixup")
    report = build_report([rep, missing])
    t = report.totals
    assert (t.papers_run, t.papers_not_run, t.claims) == (1, 1, 2)
    assert t.verified + t.open + t.blocked + t.contradicted == 2
    assert t.gpu_hours == rep.efficiency.gpu_hours  # type: ignore[union-attr]
    md = render_markdown(report)
    assert "| mixup | not run (no store at runs/mixup) |" in md
    assert "## mixup" not in md
    assert "\\|diff\\|" in md


def test_no_claims_is_not_run() -> None:
    rep = paper_report(CASE, (), {}, {}, (), gpu_h_cap=1.0, token_budget=1, max_trials=3)
    assert (rep.status, rep.reason) == ("not run", "no paper_claim evidence")


def test_deterministic() -> None:
    a = build_report([_report()[0]])
    b = build_report([_report()[0]])
    assert report_json(a) == report_json(b)
    assert render_markdown(a) == render_markdown(b)
    assert json.loads(report_json(a))["totals"]["claims"] == 2
