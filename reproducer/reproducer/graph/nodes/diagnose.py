"""Diagnose node (M4, code only, no LLM): per claim, the outcome class of each full run, its
curve summary and the trial budget, all from the store.

  run_curve   per run that exited 0: the curve from its per-epoch CSV artifact (the
              run_result's "artifacts" refs, else the files under <run dir>/artifacts for
              older records), else from the metric lines of its log (domain/curve.py), stored
              as run_curve evidence (authority rule, derived; links the run_result).
              A crashed or diverged run (not aborted; a non-zero exit other than -1, or a
              non-finite metric) that produced points gets a partial one (content
              partial: true): the curve up to where it stopped.
              Idempotent by run evidence id: a run with a stored run_curve is never redone.
              execute.py records it right after a run; `diagnose` backfills older runs.
  outcome     rules/classify_run.classify_run on the run evidence, the log tail and the curve
  breaker     domain/breaker.breaker_state (trials, GPU hours, tokens)
Only run_curve evidence is ever written here (append-only).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reproducer.domain.breaker import (
    BreakerState,
    breaker_state,
    max_trials_from_env,
    token_budget_from_env,
)
from reproducer.domain.contracts import ResultClaim
from reproducer.domain.curve import (
    CurveSummary,
    points_from_csv,
    points_from_log,
    summarize,
)
from reproducer.domain.evaluate import ClaimEvaluation, is_aborted, run_plan_for
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.run_plan import metric_direction, parse_metric_line
from reproducer.domain.telemetry import token_totals
from reproducer.providers.store import RunStore, make_ref, resolve_ref
from reproducer.rules.classify import ClassifyContext
from reproducer.rules.classify_run import RunOutcome, classify_run

LOG_TAIL_BYTES = 64 * 1024
RAW_PREFIX = "raw:"


@dataclass(frozen=True)
class CurveSpec:
    """How to read a run's metric: the claim metric, the plan's regex and conversion, and the
    repo's raw metric name (CSV column)."""

    claim_metric: str
    pattern: str
    conversion: str
    raw_name: str


@dataclass(frozen=True)
class RunDiagnosis:
    run: Evidence
    outcome: RunOutcome
    curve: CurveSummary | None
    curve_evidence_id: str | None


@dataclass(frozen=True)
class ClaimDiagnosis:
    claim: ResultClaim
    evaluation: ClaimEvaluation
    runs: tuple[RunDiagnosis, ...]


# --- curve spec + inputs -----------------------------------------------------------------------


def _plan_of(run: Evidence, evidence: Mapping[str, Evidence]) -> Evidence | None:
    records = list(evidence.values())
    i = next((i for i, ev in enumerate(records) if ev.evidence_id == run.evidence_id), None)
    return run_plan_for(run, evidence, records if i is None else records[:i])


def _raw_name(run: Evidence, pattern: str, metric_lines: Sequence[str]) -> str:
    metrics = run.content.get("metrics")
    if isinstance(metrics, Mapping):
        names = [k[len(RAW_PREFIX) :] for k in metrics if k.startswith(RAW_PREFIX)]
        if names:
            return names[0]
    for line in metric_lines:
        try:
            mp = parse_metric_line(line)
        except ValueError:
            continue
        if mp.pattern == pattern:
            return mp.name
    m = re.search(r"([A-Za-z_]\w*)", pattern.replace(r"(?<!\w)", ""))
    return m.group(1) if m else "metric"


def curve_spec_for(run: Evidence, evidence: Mapping[str, Evidence]) -> CurveSpec | None:
    """The CurveSpec from the run's plan (evaluate.run_plan_for); None without a plan."""
    plan = _plan_of(run, evidence)
    body = plan.content.get("plan") if plan is not None else None
    if not isinstance(body, Mapping) or not isinstance(body.get("metric_source"), Mapping):
        return None
    ms = body["metric_source"]
    line_ev = evidence.get(str(ms.get("metric_line_evidence_id")))
    lines = [
        str(r.get("line"))
        for r in (line_ev.content.get("result", []) if line_ev is not None else [])
        if isinstance(r, Mapping)
    ]
    pattern = str(ms["pattern"])
    return CurveSpec(
        claim_metric=str(body.get("metric")),
        pattern=pattern,
        conversion=str(ms["conversion"]),
        raw_name=_raw_name(run, pattern, lines),
    )


def run_dir_of(out: Path, run: Evidence) -> Path:
    return resolve_ref(out, str(run.content.get("log_ref", ""))).parent


def artifact_csvs(out: Path, run: Evidence) -> list[Path]:
    """The run's kept CSV artifacts: its "artifacts" refs, else <run dir>/artifacts/**.csv."""
    refs = run.content.get("artifacts")
    if isinstance(refs, list):
        paths = [resolve_ref(out, str(r)) for r in refs]
    else:
        root = run_dir_of(out, run) / "artifacts"
        paths = sorted(root.rglob("*")) if root.is_dir() else []
    return sorted(p for p in paths if p.suffix.lower() == ".csv" and p.is_file())


def read_log_tail(out: Path, run: Evidence, max_bytes: int = LOG_TAIL_BYTES) -> str:
    path = resolve_ref(out, str(run.content.get("log_ref", "")))
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _direction(claim_metric: str) -> str:
    try:
        return metric_direction(claim_metric)
    except ValueError:
        return "max"


def compute_curve(out: Path, run: Evidence, spec: CurveSpec) -> CurveSummary | None:
    """The curve from the first CSV artifact with an epoch and the raw metric column, else from
    the log's metric lines; None when neither has a point. ValueError: conversion failed."""
    common = dict(metric=spec.claim_metric, column=spec.raw_name, conversion=spec.conversion,
                  direction=_direction(spec.claim_metric))  # fmt: skip
    for path in artifact_csvs(out, run):
        text = _read_text(path)
        pts = points_from_csv(text, spec.raw_name) if text is not None else None
        if pts:
            return summarize(pts, source="csv", source_ref=make_ref(out, path), **common)  # type: ignore[arg-type]
    log_path = resolve_ref(out, str(run.content.get("log_ref", "")))
    text = _read_text(log_path)
    pts = points_from_log(text, spec.pattern) if text is not None else []
    if not pts:
        return None
    return summarize(pts, source="log", source_ref=make_ref(out, log_path), **common)  # type: ignore[arg-type]


# --- run_curve evidence ------------------------------------------------------------------------


def partial_run(run: Evidence) -> bool:
    """A crashed or diverged run_result: not aborted, and a non-zero exit other than -1 (no
    exit code / timeout) or a non-finite metric. Its curve stops where the run did."""
    if run.kind != "run_result" or is_aborted(run.content):
        return False
    code = run.content.get("exit_code")
    crashed = isinstance(code, int) and not isinstance(code, bool) and code not in (0, -1)
    return crashed or bool(run.content.get("diverged"))


def run_curve_evidence(run: Evidence, summary: CurveSummary) -> Evidence:
    """content partial: true (only) for a crashed / diverged run (partial_run)."""
    content = {
        "claim_id": run.content.get("claim_id"),
        "run_evidence_id": run.evidence_id,
        "run_id": run.content.get("run_id"),
        "summary": summary.model_dump(mode="json"),
    }
    if partial_run(run):
        content["partial"] = True
    return make_evidence(
        kind="run_curve",
        source=run.source,
        locator=f"curve:{run.content.get('run_id', run.locator)}",
        content=content,
        confidence="derived",
        authority="rule",
        links=(run.evidence_id,),
    )


def stored_curve(evidence: Mapping[str, Evidence], run_evidence_id: str) -> Evidence | None:
    return next((ev for ev in evidence.values() if ev.kind == "run_curve"
                 and ev.content.get("run_evidence_id") == run_evidence_id), None)  # fmt: skip


def wants_curve(run: Evidence) -> bool:
    """A run_result that ran to the end (exit 0), or a crashed / diverged one (partial_run)."""
    return run.kind == "run_result" and (run.content.get("exit_code") == 0 or partial_run(run))


def record_run_curve(
    out: Path, store: RunStore, run: Evidence, spec: CurveSpec, evidence: Mapping[str, Evidence]
) -> Evidence | None:
    """The run's stored run_curve, else compute and append it (None: no curve points)."""
    hit = stored_curve(evidence, run.evidence_id)
    if hit is not None:
        return hit
    summary = compute_curve(out, run, spec)
    if summary is None:
        return None
    ev = run_curve_evidence(run, summary)
    store.append_evidence(ev)
    return ev


def backfill_curves(out: Path, store: RunStore) -> list[Evidence]:
    """run_curve evidence for every run that exited 0 or crashed / diverged (wants_curve) and
    has none yet; returns the new ones.
    Runs without a plan, curve points or a valid conversion are skipped."""
    evidence = store.load_evidence()
    new: list[Evidence] = []
    for run in [ev for ev in evidence.values() if wants_curve(ev)]:
        if stored_curve(evidence, run.evidence_id) is not None:
            continue
        spec = curve_spec_for(run, evidence)
        if spec is None:
            continue
        try:
            ev = record_run_curve(out, store, run, spec, evidence)
        except ValueError:
            continue
        if ev is not None:
            new.append(ev)
    return new


# --- diagnose ----------------------------------------------------------------------------------


def _curve_of(ev: Evidence | None) -> CurveSummary | None:
    return CurveSummary.model_validate(ev.content["summary"]) if ev is not None else None


def diagnose(
    out: Path,
    store: RunStore,
    claims: Sequence[ResultClaim],
    evals: Mapping[str, ClaimEvaluation],
    *,
    gpu_h_cap: float,
    token_budget: int | None = None,
    ctx: ClassifyContext | None = None,
    max_trials: int | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[list[ClaimDiagnosis], BreakerState]:
    """Per claim (in order) every run with its outcome and curve; plus the breaker state.
    token_budget / max_trials default to REPRODUCER_TOKEN_BUDGET / REPRODUCER_MAX_TRIALS of
    `env` (default os.environ). Read-only (call backfill_curves first to store missing
    curves)."""
    env = os.environ if env is None else env
    if token_budget is None:
        token_budget = token_budget_from_env(env)
    if max_trials is None:
        max_trials = max_trials_from_env(env)
    evidence = store.load_evidence()
    out_claims: list[ClaimDiagnosis] = []
    for claim in claims:
        ev_c = evals[claim.claim_id]
        runs: list[RunDiagnosis] = []
        for run in evidence.values():
            if run.kind != "run_result" or run.content.get("claim_id") != claim.claim_id:
                continue
            curve_ev = stored_curve(evidence, run.evidence_id)
            curve = _curve_of(curve_ev)
            spec = curve_spec_for(run, evidence)
            plan = _plan_of(run, evidence)
            outcome = classify_run(
                ev_c, run, read_log_tail(out, run),
                metric_pattern=spec.pattern if spec is not None else None, curve=curve, ctx=ctx,
                command=str(plan.content.get("command", "")) if plan is not None else "",
            )  # fmt: skip
            runs.append(RunDiagnosis(run, outcome, curve,
                                     curve_ev.evidence_id if curve_ev else None))  # fmt: skip
        out_claims.append(ClaimDiagnosis(claim, ev_c, tuple(runs)))
    tokens = token_totals(store.load_call_records())
    state = breaker_state(
        [c.claim_id for c in claims], evidence, gpu_h_cap=gpu_h_cap,
        token_budget=token_budget, tokens_used=tokens["api"] + tokens["local"],
        max_per_claim=max_trials,
    )  # fmt: skip
    return out_claims, state
