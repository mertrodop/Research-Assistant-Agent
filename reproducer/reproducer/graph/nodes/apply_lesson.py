"""Apply lesson node (M4-S3): code applies the ONE change of a current, validated lesson.

The LLM never edits a plan: it proposed a Change (graph/nodes/reflect.py, contracts.Lesson);
this node records what code does with it as lesson_applied evidence (authority rule, derived;
links: the lesson + its support), then re-plans deterministically (orchestrator: the stored
claim mapping is reused, no LLM call; the applied lesson ids enter the plan digest):

  arg            the plan sets the flag (source paper, evidence = the paper support)
                 -> new run digest -> the usual pre-flight + full-run approval gate -> execute
  assumption     the unknown is resolved with the support evidence (unknowns.resolve: paper /
                 repo evidence whose snapped quote states the value) -> the plan uses the
                 resolved value -> new run (gated as usual)
  eval_protocol  the plan's metric_source changes; the log of the run the lesson is about is
                 RE-EVALUATED without running: a run_result with reeval_of = that run, gpu_hours
                 0, links: lesson + original run (+ the new plan). It counts as a trial.
  none           nothing changes; lesson_applied outcome stopped_no_change: the claim stays
                 open ("reflexion stopped: no grounded change")
An arg / assumption change whose re-built plan renders the SAME command as the run the lesson
is about needs no run: action "evaluate" (ledger / fidelity recomputed, no trial consumed);
a claim still missing then gets a fresh lesson from the changed inputs (routing.decide_reflect,
at most MAX_CHANGES_PER_TRIAL applied changes per trial).
Cross-claim effects (plan_changed_claims): a change can re-build OTHER claims' plans (e.g. a
resolved shared unknown); such a claim without a completed run of its current command runs too
(gated, a trial of its own; its cause is the other claim's lesson_applied, trial_causes).
Only the current lesson for (claim, trial, input digest) is ever applied (a stale one never
is), each lesson at most once (a stored lesson_applied for it is returned as is).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reproducer.domain.contracts import ExperimentResult
from reproducer.domain.evaluate import run_plan_for
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import make_run_evidence
from reproducer.domain.unknowns import UnknownOpened, resolve
from reproducer.graph.nodes.diagnose import (
    curve_spec_for,
    read_log_tail,
    record_run_curve,
)
from reproducer.graph.nodes.execute import MetricSpec, raw_metric_name, read_metric
from reproducer.graph.nodes.orchestrator import run_orchestrator
from reproducer.graph.nodes.reflect import current_lesson
from reproducer.providers.llm import LLM
from reproducer.providers.store import RunStore

ApplyAction = Literal["rerun", "reeval", "evaluate", "stop"]
STOPPED_NO_CHANGE = "stopped_no_change"
REEVAL_LOG_BYTES = 64 * 1024 * 1024  # the whole log of a re-evaluated run (metric lines)


@dataclass(frozen=True)
class ApplyOutcome:
    claim_id: str
    action: ApplyAction  # rerun: execute the new plan; reeval: done, evaluate; stop: open
    detail: str
    record: Evidence | None  # the lesson_applied evidence
    reeval_run: Evidence | None = None
    reused: bool = False  # the lesson was applied before (stored lesson_applied)


def applied_record(evidence: Mapping[str, Evidence], lesson_id: str) -> Evidence | None:
    return next((ev for ev in evidence.values() if ev.kind == "lesson_applied"
                 and ev.content.get("lesson_id") == lesson_id), None)  # fmt: skip


def applied_for_trial(
    evidence: Mapping[str, Evidence], claim_id: str, trial: int
) -> Evidence | None:
    """The newest lesson_applied of (claim, trial): a change already made for that trial."""
    hits = [
        ev
        for ev in evidence.values()
        if ev.kind == "lesson_applied"
        and ev.content.get("claim_id") == claim_id
        and ev.content.get("trial") == trial
    ]
    return hits[-1] if hits else None


def lesson_applied_evidence(lesson_ev: Evidence, outcome: str, detail: str) -> Evidence:
    c = lesson_ev.content
    ch: Mapping[str, Any] = c["lesson"].get("proposed_change") or {}
    sup = ch.get("support") or {}
    support_id = sup.get("evidence_id")
    return make_evidence(
        kind="lesson_applied",
        source=lesson_ev.source,
        locator=f"lesson_applied:{c['claim_id']}:trial{c['trial']}",
        content={
            "claim_id": c["claim_id"],
            "trial": c["trial"],
            "lesson_id": lesson_ev.evidence_id,
            "kind": ch.get("kind", "none"),
            "target": ch.get("target"),
            "new_value": ch.get("new_value"),
            "support_evidence_id": support_id,
            "outcome": outcome,
            "detail": detail,
        },
        confidence="derived",
        authority="rule",
        links=tuple(dict.fromkeys([lesson_ev.evidence_id, *([support_id] if support_id else [])])),
    )


def run_command(run: Evidence, evidence: Mapping[str, Evidence]) -> str | None:
    """The rendered command of the plan a run_result ran (evaluate.run_plan_for)."""
    records = list(evidence.values())
    i = next((k for k, ev in enumerate(records) if ev.evidence_id == run.evidence_id), None)
    plan = run_plan_for(run, evidence, records if i is None else records[:i])
    cmd = plan.content.get("command") if plan is not None else None
    return str(cmd) if cmd else None


def claim_runs(evidence: Mapping[str, Evidence], claim_id: str) -> list[Evidence]:
    return [ev for ev in evidence.values() if ev.kind == "run_result"
            and ev.content.get("claim_id") == claim_id]  # fmt: skip


def plan_changed_claims(evidence: Mapping[str, Evidence], current: Mapping[str, str]) -> list[str]:
    """Claims (in `current` order: claim_id -> current plan command) that ran before but have
    no completed run of their current command: their plan changed since (e.g. a shared
    unknown was resolved by another claim's lesson)."""
    out: list[str] = []
    for cid, command in current.items():
        runs = claim_runs(evidence, cid)
        if not runs:
            continue
        done = {run_command(r, evidence) for r in runs if r.content.get("exit_code") == 0}
        if command not in done:
            out.append(cid)
    return out


def trial_causes(evidence: Mapping[str, Evidence], claim_id: str) -> dict[int, list[Evidence]]:
    """trial -> lesson_applied evidence of OTHER claims that caused it: for a trial k >= 2 of a
    claim without an own applied change at trial k - 1, the other claims' lesson_applied
    stored between its runs k - 1 and k."""
    records = list(evidence.values())
    pos = {ev.evidence_id: i for i, ev in enumerate(records)}
    runs = claim_runs(evidence, claim_id)
    out: dict[int, list[Evidence]] = {}
    for k in range(2, len(runs) + 1):
        own = applied_for_trial(evidence, claim_id, k - 1)
        if own is not None and own.content.get("outcome") != STOPPED_NO_CHANGE:
            continue
        lo, hi = pos[runs[k - 2].evidence_id], pos[runs[k - 1].evidence_id]
        hits = [ev for ev in records[lo + 1 : hi] if ev.kind == "lesson_applied"
                and ev.content.get("claim_id") != claim_id]  # fmt: skip
        if hits:
            out[k] = hits
    return out


def _outcome_of(record: Evidence, reused: bool) -> ApplyOutcome:
    c = record.content
    out = str(c.get("outcome"))
    action: ApplyAction = (
        "stop" if out == STOPPED_NO_CHANGE else "reeval" if out == "reeval" else "rerun"
    )
    return ApplyOutcome(str(c["claim_id"]), action, str(c.get("detail")), record, reused=reused)


def _replan(
    store: RunStore, claim_id: str, llm_factory: Callable[[], LLM] | None, repo_dir: Path | None
) -> Any:
    """run_orchestrator: the stored mapping is reused (no LLM call for the same inputs); the
    new run plans fold the applied lesson. Raises ValueError if the claim does not plan."""
    out = run_orchestrator(store, llm_factory=llm_factory, repo_dir=repo_dir)
    plan = next((p for p in out.plans if p.claim.claim_id == claim_id), None)
    if plan is None:
        why = "; ".join(e for e in out.errors if e.startswith(f"{claim_id}:")) or "not planned"
        raise ValueError(f"re-plan after the lesson failed: {why}")
    return plan


def _after_replan(
    store: RunStore,
    cid: str,
    lesson_ev: Evidence,
    rec: Evidence,
    llm_factory: Callable[[], LLM] | None,
    repo_dir: Path,
) -> ApplyOutcome:
    """Re-plan; rerun when the rendered command changed, else evaluate (no run, no trial)."""
    try:
        planned = _replan(store, cid, llm_factory, repo_dir)
    except ValueError as e:
        return ApplyOutcome(cid, "stop", str(e), rec)
    evidence = store.load_evidence()
    orig = evidence.get(str(lesson_ev.content.get("run_evidence_id")))
    before = run_command(orig, evidence) if orig is not None else None
    if before is not None and planned.command == before:
        return ApplyOutcome(cid, "evaluate", f"{rec.content['detail']}; the rendered command is "
                            "unchanged: no new run (no trial)", rec)  # fmt: skip
    return ApplyOutcome(cid, "rerun", rec.content["detail"], rec)


def reevaluate(out: Path, store: RunStore, lesson_ev: Evidence, planned: Any) -> Evidence:
    """A run_result re-reading the log of the run the lesson is about with the new plan's
    metric_source (nothing executed, 0 GPU hours), linked to the lesson, the original run and
    the new plan; reeval_of = the original run. Its run_curve is recorded too."""
    evidence = store.load_evidence()
    orig = evidence[str(lesson_ev.content["run_evidence_id"])]
    c = orig.content
    ms = planned.plan.metric_source
    line_ev = evidence.get(ms.metric_line_evidence_id)
    lines = [str(r.get("line")) for r in (line_ev.content.get("result", []) if line_ev else [])
             if isinstance(r, Mapping)]  # fmt: skip
    spec = MetricSpec(
        claim_metric=planned.claim.metric,
        pattern=ms.pattern,
        which=ms.which,
        conversion=ms.conversion,
        raw_name=raw_metric_name(lines, ms.pattern),
    )
    log_text = read_log_tail(out, orig, REEVAL_LOG_BYTES)
    code = c.get("exit_code")
    code = code if isinstance(code, int) and not isinstance(code, bool) else -1
    reading = read_metric(log_text, spec, exit_code=code, timed_out=False)
    metrics: dict[str, float] = {}
    if reading.raw is not None:
        metrics[f"raw:{spec.raw_name}"] = reading.raw
    if reading.value is not None:
        metrics[spec.claim_metric] = reading.value
    result = ExperimentResult(
        claim_id=str(c["claim_id"]), run_id=f"{c.get('run_id')}-reeval-{lesson_ev.evidence_id[:8]}",
        exit_code=code, metrics=metrics, log_ref=str(c.get("log_ref", "")),
        config_digest=str(c.get("config_digest", "")), duration=0.0, gpu_hours=0.0,
    )  # fmt: skip
    plan_like = {"run_plan", "preflight"}
    assumed = tuple(i for i in orig.links
                    if i not in evidence or evidence[i].kind not in plan_like)  # fmt: skip
    faithful = bool(c.get("spec_faithful")) and not assumed
    run = make_run_evidence(
        result, spec_faithful=faithful, assumption_evidence_ids=() if faithful else assumed,
        source=orig.source,
        links=(lesson_ev.evidence_id, orig.evidence_id, planned.evidence.evidence_id),
        reeval_of=orig.evidence_id,
    )  # fmt: skip
    store.append_evidence(run)
    evidence = store.load_evidence()
    cspec = curve_spec_for(run, evidence)
    if cspec is not None:
        try:
            record_run_curve(out, store, run, cspec, evidence)
        except (ValueError, OSError):
            pass  # the curve is optional; `diagnose` backfills it
    return run


def apply_change(
    out: Path,
    store: RunStore,
    lesson_ev: Evidence,
    *,
    input_digest: str,
    llm_factory: Callable[[], LLM] | None = None,
) -> ApplyOutcome:
    """Apply the lesson's one change (see the module docstring). Raises ValueError for a lesson
    that is not the current one for (claim, trial, input_digest) (a stale lesson is never
    applied). An earlier application of the same lesson is returned as is (reused)."""
    if lesson_ev.kind != "lesson":
        raise ValueError(f"not a lesson: {lesson_ev.kind}")
    evidence = store.load_evidence()
    c = lesson_ev.content
    cid, trial = str(c["claim_id"]), int(c["trial"])
    cur = current_lesson(evidence, cid, trial, input_digest)
    if cur is None or cur.evidence_id != lesson_ev.evidence_id:
        raise ValueError(
            f"lesson {lesson_ev.evidence_id[:12]} is not the current lesson of {cid} trial "
            f"{trial} for these inputs: a stale lesson is never applied"
        )
    done = applied_record(evidence, lesson_ev.evidence_id)
    if done is not None:
        return _outcome_of(done, reused=True)
    ch: Mapping[str, Any] = c["lesson"].get("proposed_change") or {}
    kind = ch.get("kind", "none")
    repo_dir = out / "repo"

    def record(outcome: str, detail: str) -> Evidence:
        ev = lesson_applied_evidence(lesson_ev, outcome, detail)
        store.append_evidence(ev)
        return ev

    if kind == "none":
        rec = record(STOPPED_NO_CHANGE, "reflexion stopped: no grounded change")
        return ApplyOutcome(cid, "stop", rec.content["detail"], rec)

    target, value = str(ch.get("target")), ch.get("new_value")
    if kind == "assumption":
        sup = evidence.get(str((ch.get("support") or {}).get("evidence_id")))
        opened = next(
            (
                e
                for e in store.load_events()
                if isinstance(e, UnknownOpened) and e.unknown_id == target
            ),
            None,
        )
        if sup is None or opened is None or value is None:
            raise ValueError(f"cannot resolve {target}: no support evidence / opened unknown")
        store.append_event(resolve(opened, sup, str(value)))
        rec = record("resolved", f"{target} resolved to {value} by {sup.kind} "
                                 f"{sup.evidence_id[:12]}")  # fmt: skip
        return _after_replan(store, cid, lesson_ev, rec, llm_factory, repo_dir)

    if kind == "arg":
        rec = record("applied", f"plan sets {target} {value if value is not None else ''}"
                                .rstrip() + " (paper)")  # fmt: skip
        return _after_replan(store, cid, lesson_ev, rec, llm_factory, repo_dir)

    if kind == "eval_protocol":
        rec = record("reeval", f"{target} = {value}; re-evaluated the logged run (0 GPU h)")
        try:
            planned = _replan(store, cid, llm_factory, repo_dir)
        except ValueError as e:
            return ApplyOutcome(cid, "stop", str(e), rec)
        run = reevaluate(out, store, lesson_ev, planned)
        return ApplyOutcome(cid, "reeval", rec.content["detail"], rec, reeval_run=run)

    raise ValueError(f"unknown change kind {kind!r}")
