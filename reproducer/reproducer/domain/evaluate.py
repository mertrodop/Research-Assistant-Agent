"""Evaluator: per claim, the paper value against our best completed run, with the
pre-registered tolerance, the ledger status / fidelity, and every reason the result is
inferred rather than spec-faithful, all built by code from stored evidence.

Pure domain: no file, network or subprocess IO. Status and fidelity are exactly
compute_ledger's; this module only adds the comparison numbers and the reasons.

Fidelity reasons (inferred_reasons) are what makes a result inferred, including the protocol
(fewer completed runs than the paper's num_runs).

Unknowns resolved AFTER a run (e.g. a human answer) are compared with the value the run
assumed: a different value is a deviation ("ran with X=a, human answered X=b"); the same
value confirms the assumption (no reason). A run whose only cause of infidelity was such
confirmed assumptions counts as spec-faithful (confirmed_run_ids -> compute_ledger).

Info notes (info) never affect fidelity: e.g. packages left unpinned because the repo states
no version.

Which evidence explains a run (the run_result's own links first, else by file order):
  run_plan    a linked run_plan, else the plan of the newest preflight before the run with the
              same run digest (preflight.run_digest == run_result.config_digest)
  env plan    a linked env plan (env_spec, locator env_plan), else the newest env plan of the
              run's source stored before the run
  known fixes / critic patches: those of the run's source stored before the run (critic
              patches minus the reverted ones = the patch stack the run used)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evidence import Evidence
from reproducer.domain.ledger import (
    ClaimStatus,
    Fidelity,
    Tolerance,
    ToleranceConfig,
    compute_ledger,
    paper_num_runs,
    protocol_faithful,
    stored_tolerance,
    within_tolerance,
)
from reproducer.domain.unknowns import UnknownState

ReasonKind = Literal[
    "env_deviation",
    "plan_deviation",
    "known_fix_deviation",
    "critic_patch",
    "open_unknown",
    "env_assumption",
    "protocol",
    "unexplained",
    "unpinned_packages",  # info only
]
Verdict = Literal["within_tolerance", "outside_tolerance", "no_completed_run"]
UNPINNED_PREFIX = "unpinned:"
ENV_PLAN_LOCATOR = "env_plan"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InferredReason(_Frozen):
    kind: ReasonKind
    text: str
    evidence_ids: tuple[str, ...] = ()


class ClaimEvaluation(_Frozen):
    claim_id: str
    metric: str
    unit: str | None
    paper_value: float
    tolerance: Tolerance
    tolerance_evidence_id: str
    tolerance_reason: str
    best_run_evidence_id: str | None
    our_value: float | None  # converted to the claim metric
    our_raw: dict[str, float | None]  # raw:<name> values the metric was converted from
    abs_diff: float | None
    within_tolerance: bool | None  # None: no completed run
    verdict: Verdict
    status: ClaimStatus
    fidelity: Fidelity | None  # compute_ledger's: set only when verified
    run_fidelity: Fidelity | None  # the best run's (incl. protocol); None: no completed run
    attempts: int  # run_result evidence for the claim (completed, failed or aborted)
    completed_runs: int
    aborted_runs: int
    gpu_hours: float  # all run_results of the claim
    inferred_reasons: tuple[InferredReason, ...]
    info: tuple[InferredReason, ...] = ()  # informational; never affects fidelity


# --- helpers ---------------------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def run_value(content: Mapping[str, Any], metric: str) -> float | None:
    """The claim metric of a COMPLETED run (exit 0, metric present and finite), else None."""
    if content.get("exit_code") != 0:
        return None
    metrics = content.get("metrics")
    return _finite(metrics.get(metric)) if isinstance(metrics, Mapping) else None


def is_aborted(content: Mapping[str, Any]) -> bool:
    return content.get("status") == "aborted"


def _before(records: Sequence[Evidence], run: Evidence) -> list[Evidence]:
    """Evidence stored before `run` (file order)."""
    for i, ev in enumerate(records):
        if ev.evidence_id == run.evidence_id:
            return list(records[:i])
    return list(records)


def _is_env_plan(ev: Evidence) -> bool:
    return ev.kind == "env_spec" and ev.locator == ENV_PLAN_LOCATOR


def run_plan_for(
    run: Evidence, evidence: Mapping[str, Evidence], before: Sequence[Evidence]
) -> Evidence | None:
    for eid in run.links:
        ev = evidence.get(eid)
        if ev is not None and ev.kind == "run_plan":
            return ev
    digest = run.content.get("config_digest")
    for ev in reversed(before):
        if ev.kind == "preflight" and ev.content.get("run_digest") == digest:
            plan = evidence.get(str(ev.content.get("plan_evidence_id")))
            if plan is not None and plan.kind == "run_plan":
                return plan
    return None


def env_plan_for(
    run: Evidence, evidence: Mapping[str, Evidence], before: Sequence[Evidence]
) -> Evidence | None:
    for eid in run.links:
        ev = evidence.get(eid)
        if ev is not None and _is_env_plan(ev):
            return ev
    return next((ev for ev in reversed(before) if _is_env_plan(ev) and ev.source == run.source),
                None)  # fmt: skip


def applied_critic_patches(before: Iterable[Evidence], source: str) -> list[Evidence]:
    """critic_patch evidence (ok) of `source` that no later critic_revert dropped."""
    stack: list[Evidence] = []
    for ev in before:
        if ev.source != source:
            continue
        if ev.kind == "critic_patch" and ev.content.get("ok") and ev.content.get("patch_id"):
            stack.append(ev)
        elif ev.kind == "critic_revert":
            pid = ev.content.get("patch_id")
            stack = [p for p in stack if p.content.get("patch_id") != pid]
    return stack


def _run_faithful(content: Mapping[str, Any]) -> bool:
    """The run evidence's own spec-faithful flag (protocol not included)."""
    return content.get("spec_faithful") is True and not content.get("assumptions_used")


def info_notes(run: Evidence, evidence: Mapping[str, Evidence]) -> tuple[InferredReason, ...]:
    """Informational notes for the run (never affect fidelity): unpinned packages."""
    env = env_plan_for(run, evidence, _before(list(evidence.values()), run))
    plan = env.content.get("plan", {}) if env is not None else {}
    assumptions = plan.get("assumptions", ()) if isinstance(plan, Mapping) else ()
    names = [
        str(a["name"])[len(UNPINNED_PREFIX) :]
        for a in assumptions
        if isinstance(a, Mapping) and str(a.get("name", "")).startswith(UNPINNED_PREFIX)
    ]
    if env is None or not names:
        return ()
    text = f"{', '.join(names)} unpinned (no version stated; pip resolves at build time)"
    return (InferredReason(kind="unpinned_packages", text=text, evidence_ids=(env.evidence_id,)),)


def _same_value(a: str, b: str) -> bool:
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _resolution_text(
    name: str, assumed: str | None, st: UnknownState, evidence: Mapping[str, Evidence]
) -> str:
    ev = evidence.get(st.resolved_by or "")
    who = "human answered" if ev is not None and ev.kind == "human_input" else "resolved to"
    ran = f"{name}={assumed}" if assumed is not None else f"{name} assumed (value not recorded)"
    return f"ran with {ran}, {who} {name}={st.value}"


def _deviation_text(d: Any) -> str:
    if isinstance(d, Mapping) and {"field", "stated", "chosen"} <= set(d):
        why = f" ({d['reason']})" if d.get("reason") else ""
        return f"{d['field']} {d['stated']} -> {d['chosen']}{why}"
    return str(d)


# --- reasons ---------------------------------------------------------------------------------


def inferred_reasons(
    claim: ResultClaim,
    run: Evidence,
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    completed_runs: int,
) -> tuple[InferredReason, ...]:
    """Every recorded reason this run is not spec-faithful, plus the protocol gap (fewer runs
    than the paper averaged). A run marked not spec-faithful with no recorded cause gets an
    `unexplained` reason, so an inferred result never shows an empty list (a run confirmed
    by later answers is spec-faithful: no reason)."""
    recorded, confirmed = _recorded_reasons(run, evidence, unknowns)
    out = list(recorded)
    is_confirmed = bool(run.content.get("assumptions_used")) and confirmed > 0 and not recorded

    nr = paper_num_runs(evidence)
    if nr is not None and completed_runs < nr[0]:
        ours = "single run" if completed_runs == 1 else f"{completed_runs} runs"
        out.append(InferredReason(kind="protocol", text=f"{ours} vs paper mean of {nr[0]}",
                                  evidence_ids=(nr[1],)))  # fmt: skip

    if not _run_faithful(run.content) and not is_confirmed and not recorded:
        out.append(InferredReason(kind="unexplained",
                                  text="run marked not spec-faithful; no recorded cause found",
                                  evidence_ids=(run.evidence_id,)))  # fmt: skip
    return tuple(out)


def run_confirmed(
    run: Evidence, evidence: Mapping[str, Evidence], unknowns: Mapping[str, UnknownState]
) -> bool:
    """The run is marked not spec-faithful only because it used assumed values, and every
    one of them was later resolved to the same value (no other recorded cause)."""
    if _run_faithful(run.content) or not run.content.get("assumptions_used"):
        return False
    out, confirmed = _recorded_reasons(run, evidence, unknowns)
    return confirmed > 0 and not out


def confirmed_run_ids(
    evidence: Mapping[str, Evidence], unknowns: Mapping[str, UnknownState]
) -> frozenset[str]:
    """run_result ids that count as spec-faithful by confirmation (run_confirmed)."""
    return frozenset(
        ev.evidence_id
        for ev in evidence.values()
        if ev.kind == "run_result" and run_confirmed(ev, evidence, unknowns)
    )


def _recorded_reasons(
    run: Evidence, evidence: Mapping[str, Evidence], unknowns: Mapping[str, UnknownState]
) -> tuple[list[InferredReason], int]:
    """(recorded reasons except protocol / unexplained, number of assumptions confirmed by a
    later resolution to the same value)."""
    records = list(evidence.values())
    before = _before(records, run)
    out: list[InferredReason] = []

    env = env_plan_for(run, evidence, before)
    env_plan = env.content.get("plan", {}) if env is not None else {}
    for d in env_plan.get("deviations", ()) if isinstance(env_plan, Mapping) else ():
        out.append(InferredReason(kind="env_deviation", text=_deviation_text(d),
                                  evidence_ids=(env.evidence_id,) if env else ()))  # fmt: skip

    plan = run_plan_for(run, evidence, before)
    pc = plan.content if plan is not None else {}
    pid = (plan.evidence_id,) if plan is not None else ()
    for d in pc.get("deviations", ()):
        out.append(InferredReason(kind="plan_deviation", text=_deviation_text(d), evidence_ids=pid))
    notes = [n for n in pc.get("notes", ()) if isinstance(n, Mapping)]
    for n in notes:
        if n.get("status") == "deviation":
            out.append(InferredReason(kind="plan_deviation",
                                      text=f"{n.get('field')}: {n.get('detail')}",
                                      evidence_ids=pid))  # fmt: skip

    for ev in before:
        if ev.kind != "known_fix" or ev.source != run.source:
            continue
        action = ev.content.get("action")
        if isinstance(action, Mapping) and action.get("deviation"):
            out.append(InferredReason(
                kind="known_fix_deviation",
                text=f"{action.get('kind')} {action.get('params')}: {action['deviation']}",
                evidence_ids=(ev.evidence_id,),
            ))  # fmt: skip

    for ev in applied_critic_patches(before, run.source):
        c = ev.content
        out.append(InferredReason(
            kind="critic_patch",
            text=f"patch {str(c['patch_id'])[:12]} on {c.get('file')}: {c.get('rationale')}",
            evidence_ids=(ev.evidence_id,),
        ))  # fmt: skip

    # open unknowns the plan depends on, with the value the plan assumed
    plan_body = pc.get("plan", {})
    deps = plan_body.get("depends_on_unknowns", ()) if isinstance(plan_body, Mapping) else ()
    plan_assumed = {
        str(a.get("unknown_id")): str(a.get("value"))
        for a in pc.get("assumptions", ())
        if isinstance(a, Mapping)
    }
    note_detail = {str(n.get("field")): str(n.get("detail")) for n in notes}
    confirmed = 0
    for uid in sorted(deps):
        st = unknowns.get(uid)
        if st is None:
            continue
        value = next((a.value for a in st.assumptions if plan and a.evidence_id == pid[0]),
                     plan_assumed.get(uid))  # fmt: skip
        if st.status == "resolved" and st.value is not None:
            if value is not None and _same_value(value, st.value):
                confirmed += 1  # the assumption was right: no reason
            else:
                out.append(InferredReason(kind="plan_deviation",
                                          text=_resolution_text(st.name, value, st, evidence),
                                          evidence_ids=(*pid, st.resolved_by)
                                          if st.resolved_by else pid))  # fmt: skip
            continue
        if st.status != "open":
            continue
        detail = note_detail.get(st.name)
        text = f"{st.name} not stated by the paper; assumed {value if value is not None else '?'}"
        out.append(InferredReason(kind="open_unknown",
                                  text=text + (f" ({detail})" if detail else ""),
                                  evidence_ids=pid))  # fmt: skip

    # env assumptions for unknowns that are still open (python, cuda, ...); resolved later:
    # a different value is a deviation, the same value confirms the assumption
    if env is not None and isinstance(env_plan, Mapping):
        for a in env_plan.get("assumptions", ()):
            if not isinstance(a, Mapping) or a.get("value") is None:
                continue
            st = unknowns.get(f"unk:{a.get('name')}")
            if st is None:
                continue
            if st.status == "resolved" and st.value is not None:
                if _same_value(str(a.get("value")), st.value):
                    confirmed += 1
                else:
                    out.append(InferredReason(
                        kind="env_deviation",
                        text=_resolution_text(str(a.get("name")), str(a.get("value")), st,
                                              evidence),
                        evidence_ids=(env.evidence_id,),
                    ))  # fmt: skip
                continue
            if st.status != "open":
                continue
            out.append(InferredReason(
                kind="env_assumption",
                text=f"{a.get('name')} = {a.get('value')} ({a.get('reason')})",
                evidence_ids=(env.evidence_id,),
            ))  # fmt: skip

    return out, confirmed


# --- evaluate --------------------------------------------------------------------------------


def evaluate_claims(
    claims: Sequence[ResultClaim],
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    *,
    max_tries: int = 3,
) -> dict[str, ClaimEvaluation]:
    """Per claim (in order). Raises ValueError if a claim has no pre-registered tolerance
    evidence: the evaluator never falls back to a tolerance chosen after the runs."""
    tols: dict[str, tuple[Tolerance, str]] = {}
    missing: list[str] = []
    for c in claims:
        st = stored_tolerance(evidence, c)
        if st is None:
            missing.append(c.claim_id)
        else:
            tols[c.claim_id] = st
    if missing:
        raise ValueError(f"no pre-registered tolerance evidence for {missing}: run `plan` first")
    config = ToleranceConfig(
        default=Tolerance(abs=0.0), per_claim={cid: t for cid, (t, _) in tols.items()}
    )
    confirmed = confirmed_run_ids(evidence, unknowns)
    ledger = compute_ledger(
        tuple(claims), evidence, unknowns, config, max_tries=max_tries, confirmed_runs=confirmed
    )

    out: dict[str, ClaimEvaluation] = {}
    for claim in claims:
        cid = claim.claim_id
        entry = ledger[cid]
        tol, tol_id = tols[cid]
        runs = [ev for ev in evidence.values() if ev.kind == "run_result"
                and ev.content.get("claim_id") == cid]  # fmt: skip
        completed = [ev for ev in runs if run_value(ev.content, claim.metric) is not None]
        gpu = sum(_finite(ev.content.get("gpu_hours")) or 0.0 for ev in runs)
        best = evidence.get(entry.best_run_evidence_id) if entry.best_run_evidence_id else None

        ours = run_value(best.content, claim.metric) if best is not None else None
        raw: dict[str, float | None] = {}
        reasons: tuple[InferredReason, ...] = ()
        info: tuple[InferredReason, ...] = ()
        run_fidelity: Fidelity | None = None
        if best is not None:
            m = best.content.get("metrics") or {}
            raw = {k: _finite(v) for k, v in m.items() if k.startswith("raw:")}
            reasons = inferred_reasons(claim, best, evidence, unknowns, len(completed))
            info = info_notes(best, evidence)
            faithful = (
                _run_faithful(best.content) or best.evidence_id in confirmed
            ) and protocol_faithful(len(completed), paper_num_runs(evidence))
            run_fidelity = "spec_faithful" if faithful else "inferred"
        within = within_tolerance(ours, claim.value, tol) if ours is not None else None
        verdict: Verdict = (
            "no_completed_run" if within is None
            else "within_tolerance" if within else "outside_tolerance"
        )  # fmt: skip
        out[cid] = ClaimEvaluation(
            claim_id=cid,
            metric=claim.metric,
            unit=claim.unit,
            paper_value=claim.value,
            tolerance=tol,
            tolerance_evidence_id=tol_id,
            tolerance_reason=str(evidence[tol_id].content.get("reason", "")),
            best_run_evidence_id=best.evidence_id if best is not None else None,
            our_value=ours,
            our_raw=raw,
            abs_diff=entry.best_abs_error,
            within_tolerance=within,
            verdict=verdict,
            status=entry.status,
            fidelity=entry.fidelity,
            run_fidelity=run_fidelity,
            attempts=entry.attempts,
            completed_runs=len(completed),
            aborted_runs=sum(1 for ev in runs if is_aborted(ev.content)),
            gpu_hours=round(gpu, 4),
            inferred_reasons=reasons,
            info=info,
        )
    return out
