"""Ledger: per-claim status computed by code from evidence + unknowns.

Tolerance lives here and is never written by an LLM. The ledger only reads claims, evidence,
unknowns and tolerance; it never alters them.
Pure domain: no file, network or subprocess IO.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import Self

from reproducer.domain.contracts import ExperimentResult, ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.unknowns import UnknownState

ClaimStatus = Literal["verified", "open", "contradicted", "blocked"]
Fidelity = Literal["spec_faithful", "inferred"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Tolerance -----------------------------------------------------------------------------


class Tolerance(_Frozen):
    abs: float | None = None
    rel: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.abs is None and self.rel is None:
            raise ValueError("Tolerance needs at least one of abs/rel")
        for name, v in (("abs", self.abs), ("rel", self.rel)):
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError(f"Tolerance.{name} must be finite and >= 0, got {v!r}")
        return self


class ToleranceConfig(_Frozen):
    default: Tolerance
    per_metric: dict[str, Tolerance] = {}
    per_claim: dict[str, Tolerance] = {}

    def tolerance_for(self, claim: ResultClaim) -> Tolerance:
        """Lookup order: per_claim > per_metric > default."""
        if claim.claim_id in self.per_claim:
            return self.per_claim[claim.claim_id]
        if claim.metric in self.per_metric:
            return self.per_metric[claim.metric]
        return self.default


def within_tolerance(ours: float, paper: float, tol: Tolerance) -> bool:
    """|ours - paper| <= max(abs or 0, rel*|paper| or 0). Non-finite `ours` never passes."""
    if not math.isfinite(ours):
        return False
    bound = max(tol.abs or 0.0, (tol.rel or 0.0) * abs(paper))
    return abs(ours - paper) <= bound


# --- Pre-registered tolerance (written before any run) -------------------------------------

SIGMA_K = 2.0  # tolerance = SIGMA_K * the paper's reported std
DEFAULT_TOLERANCE_ABS = 0.5
_NUM = r"\d+(?:\.\d+)?"
_PM_RE = re.compile(rf"(?<![\d.])({_NUM})\s*(?:±|\+/-|\+-|\\pm)\s*({_NUM})(?![\d.])")


def sigma_from_quote(claim: ResultClaim) -> float | None:
    """The "± x" right after the claim value in the claim quote; None if the value never
    carries one, or if several occurrences of the value carry different sigmas."""
    sigmas = {float(m[2]) for m in _PM_RE.finditer(claim.quote) if float(m[1]) == claim.value}
    return sigmas.pop() if len(sigmas) == 1 else None


def pre_registered_tolerance(claim: ResultClaim) -> tuple[Tolerance, float | None, str]:
    """(tolerance, sigma, reason): abs = SIGMA_K * sigma from the quote, else the default."""
    sigma = sigma_from_quote(claim)
    if sigma is None:
        reason = (
            f"no '± sigma' after {claim.value} in the claim quote: "
            f"default abs {DEFAULT_TOLERANCE_ABS}"
        )
        return Tolerance(abs=DEFAULT_TOLERANCE_ABS), None, reason
    reason = f"{SIGMA_K:g} x paper std (± {sigma:g} after {claim.value} in the claim quote)"
    return Tolerance(abs=SIGMA_K * sigma), sigma, reason


def tolerance_evidence(claim: ResultClaim, *, claim_evidence_id: str, source: str) -> Evidence:
    """tolerance evidence for a claim (authority rule, derived), linked to its paper_claim
    evidence. Written before any run; never changed afterwards."""
    tol, sigma, reason = pre_registered_tolerance(claim)
    return make_evidence(
        kind="tolerance",
        source=source,
        locator=f"tolerance:{claim.claim_id}",
        content={
            "claim_id": claim.claim_id,
            "metric": claim.metric,
            "value": claim.value,
            "unit": claim.unit,
            "abs": tol.abs,
            "rel": tol.rel,
            "sigma": sigma,
            "k": SIGMA_K if sigma is not None else None,
            "reason": reason,
        },
        confidence="derived",
        authority="rule",
        links=(claim_evidence_id,),
    )


# --- Run / smoke evidence ------------------------------------------------------------------


def make_run_evidence(
    result: ExperimentResult,
    *,
    spec_faithful: bool,
    assumption_evidence_ids: tuple[str, ...],
    source: str,
    links: tuple[str, ...] = (),
    artifacts: Sequence[str] | None = None,
    reeval_of: str | None = None,
) -> Evidence:
    """run_result evidence. Non-finite metrics become None and mark the run diverged
    (informational only; the ledger decides completion from the claimed metric). Evidence
    links = the assumption evidence ids, then `links` (e.g. run_plan + preflight), deduped.
    `artifacts` (store-relative refs of the kept output files) adds an "artifacts" content key;
    None leaves it out (older records have none). `reeval_of` (a run_result evidence id) marks
    a re-evaluation of that run's log under a changed metric source (nothing executed, 0 GPU
    hours); the key is left out otherwise.

    Raises ValueError if spec_faithful=True while assumptions were used.
    """
    if spec_faithful and assumption_evidence_ids:
        raise ValueError("spec_faithful=True is incompatible with assumption_evidence_ids")
    metrics: dict[str, float | None] = {}
    diverged = False
    for name, value in result.metrics.items():
        if math.isfinite(value):
            metrics[name] = value
        else:
            metrics[name] = None
            diverged = True
    content = result.model_dump(mode="python")
    content["metrics"] = metrics
    content["spec_faithful"] = spec_faithful
    content["assumptions_used"] = bool(assumption_evidence_ids)
    content["diverged"] = diverged
    if artifacts is not None:
        content["artifacts"] = sorted(artifacts)
    if reeval_of is not None:
        content["reeval_of"] = reeval_of
    return make_evidence(
        kind="run_result",
        source=source,
        locator=result.run_id,
        content=content,
        confidence="observed",
        authority="execution",
        links=tuple(dict.fromkeys((*assumption_evidence_ids, *links))),
    )


RUN_ABORTED = "aborted"


def make_aborted_run_evidence(
    *,
    claim_id: str,
    run_id: str,
    log_ref: str,
    config_digest: str,
    duration_s: float,
    reason: str,
    spec_faithful: bool,
    assumption_evidence_ids: tuple[str, ...],
    source: str,
    links: tuple[str, ...] = (),
) -> Evidence:
    """run_result evidence for a full run that was interrupted (Ctrl-C, an exception) before
    it produced a result: status "aborted", exit_code None, no metrics; never completed. The
    GPU time used so far is recorded so it counts toward the paper GPU-hour cap."""
    if spec_faithful and assumption_evidence_ids:
        raise ValueError("spec_faithful=True is incompatible with assumption_evidence_ids")
    duration = max(0.0, duration_s)
    return make_evidence(
        kind="run_result",
        source=source,
        locator=run_id,
        content={
            "claim_id": claim_id,
            "run_id": run_id,
            "exit_code": None,
            "status": RUN_ABORTED,
            "abort_reason": reason,
            "metrics": {},
            "log_ref": log_ref,
            "config_digest": config_digest,
            "duration": round(duration, 1),
            "gpu_hours": round(duration / 3600.0, 4),
            "spec_faithful": spec_faithful,
            "assumptions_used": bool(assumption_evidence_ids),
            "diverged": False,
        },
        confidence="observed",
        authority="execution",
        links=tuple(dict.fromkeys((*assumption_evidence_ids, *links))),
    )


def make_smoke_evidence(
    *,
    passed: bool,
    attempts: int,
    capped: bool,
    source: str,
    locator: str,
    details: Mapping[str, Any] | None = None,
    links: tuple[str, ...] = (),
) -> Evidence:
    """smoke_result evidence. passed/attempts/capped are what the ledger reads; `details`
    (signal, command, overrides, timing, parsed traceback, log_ref, attempt, ...) is added to
    the content as is and may not redefine those three keys. Without details the content (and
    id) is exactly {passed, attempts, capped}."""
    base: dict[str, Any] = {"passed": passed, "attempts": attempts, "capped": capped}
    extra = dict(details or {})
    clash = sorted(set(extra) & set(base))
    if clash:
        raise ValueError(f"smoke details may not redefine {clash}")
    return make_evidence(
        kind="smoke_result",
        source=source,
        locator=locator,
        content={**base, **extra},
        confidence="observed",
        authority="execution",
        links=links,
    )


# --- Ledger --------------------------------------------------------------------------------

NUM_RUNS_SETUP = "num_runs"


def paper_num_runs(evidence: Mapping[str, Evidence]) -> tuple[int, str] | None:
    """(num_runs, evidence_id) from the paper setup, if it states an integer > 0."""
    for ev in evidence.values():
        if ev.kind == "paper_setup" and ev.content.get("name") == NUM_RUNS_SETUP:
            try:
                n = int(str(ev.content.get("value")).strip())
            except ValueError:
                return None
            return (n, ev.evidence_id) if n > 0 else None
    return None


def protocol_faithful(completed_runs: int, num_runs: tuple[int, str] | None) -> bool:
    """Part of the spec-faithful rule, computed at read time: at least as many completed runs
    as the paper averaged over (no stated num_runs -> nothing to match)."""
    return num_runs is None or completed_runs >= num_runs[0]


class LedgerEntry(_Frozen):
    claim_id: str
    status: ClaimStatus
    fidelity: Fidelity | None = None  # only set when verified
    attempts: int
    best_run_evidence_id: str | None = None
    best_abs_error: float | None = None
    open_unknowns: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()  # claim evidence + its runs
    tolerance: Tolerance | None = None  # the tolerance used for this claim
    tolerance_evidence_id: str | None = None  # None: the ToleranceConfig fallback was used

    @model_validator(mode="after")
    def _fidelity_only_when_verified(self) -> Self:
        if (self.status == "verified") != (self.fidelity is not None):
            raise ValueError("fidelity must be set iff status is verified")
        return self


class _RunView(BaseModel):
    """Read-only view of run_result content (other ExperimentResult fields ignored)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    claim_id: str
    exit_code: int | None  # None: aborted run (never completed)
    metrics: dict[str, float | None]
    spec_faithful: bool
    assumptions_used: bool
    diverged: bool


class _Run(_Frozen):
    evidence_id: str
    view: _RunView

    @property
    def faithful(self) -> bool:
        return self.view.spec_faithful and not self.view.assumptions_used


def _parse_run(ev: Evidence) -> _Run:
    try:
        view = _RunView.model_validate(ev.content)
    except ValidationError as e:
        raise ValueError(f"malformed run_result evidence {ev.evidence_id}: {e}") from e
    return _Run(evidence_id=ev.evidence_id, view=view)


def _metric_value(run: _Run, metric: str) -> float | None:
    """The claim metric if the run completed: exit 0 and the claimed metric present and finite.

    `diverged` (any non-finite metric) is informational and does not affect completion.
    """
    v = run.view
    if v.exit_code != 0:
        return None
    value = v.metrics.get(metric)
    if value is None or not math.isfinite(value):
        return None
    return value


def _is_failed_capped_smoke(ev: Evidence) -> bool:
    return ev.content.get("capped") is True and ev.content.get("passed") is False


def stored_tolerance(
    evidence: Mapping[str, Evidence], claim: ResultClaim
) -> tuple[Tolerance, str] | None:
    """The pre-registered tolerance evidence of this claim (same claim_id, metric and value;
    the first one stored wins) as (tolerance, evidence_id); None if there is none."""
    for ev in evidence.values():
        c = ev.content
        if (
            ev.kind == "tolerance"
            and c.get("claim_id") == claim.claim_id
            and c.get("metric") == claim.metric
            and c.get("value") == claim.value
        ):
            return Tolerance(abs=c.get("abs"), rel=c.get("rel")), ev.evidence_id
    return None


def compute_ledger(
    claims: tuple[ResultClaim, ...],
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    tolerance: ToleranceConfig,
    max_tries: int = 3,
    confirmed_runs: AbstractSet[str] = frozenset(),
) -> dict[str, LedgerEntry]:
    """Per-claim status. Precedence: verified > contradicted > blocked > open.

    A run counts as spec-faithful only if its evidence says so (spec_faithful, no assumptions)
    AND the claim has at least the paper's num_runs completed runs (protocol_faithful). This
    holds for the verified fidelity and for the all-faithful condition of contradicted.
    confirmed_runs: run_result ids whose only cause of infidelity was assumed values that were
    later resolved to exactly those values (evaluate.confirmed_run_ids); they count as
    spec-faithful.

    Tolerance: the claim's pre-registered tolerance evidence; `tolerance` (the config) is only
    the fallback for a claim without one. The entry records which was used.

    Smoke blocking is per paper/repo; scope by repo@commit when multiple code_links are
    supported.
    """
    records = list(evidence.values())
    runs_all = [_parse_run(ev) for ev in records if ev.kind == "run_result"]
    smoke_blocked = any(ev.kind == "smoke_result" and _is_failed_capped_smoke(ev) for ev in records)
    num_runs = paper_num_runs(evidence)

    ledger: dict[str, LedgerEntry] = {}
    for claim in claims:
        cid = claim.claim_id
        stored = stored_tolerance(evidence, claim)
        tol = stored[0] if stored else tolerance.tolerance_for(claim)
        claim_ev_ids = [
            ev.evidence_id
            for ev in records
            if ev.kind == "paper_claim" and ev.content.get("claim_id") == cid
        ]
        runs = [r for r in runs_all if r.view.claim_id == cid]
        open_unknowns = tuple(
            sorted(
                uid
                for uid, st in unknowns.items()
                if st.status == "open" and (not st.claim_ids or cid in st.claim_ids)
            )
        )

        # (run, abs_error, within) for completed runs only
        completed: list[tuple[_Run, float, bool]] = []
        for r in runs:
            ours = _metric_value(r, claim.metric)
            if ours is not None:
                completed.append(
                    (r, abs(ours - claim.value), within_tolerance(ours, claim.value, tol))
                )

        def faithful(r: _Run) -> bool:
            return r.faithful or r.evidence_id in confirmed_runs

        best = min(completed, key=lambda c: (c[1], c[0].evidence_id), default=None)
        passing = [c for c in completed if c[2]]
        protocol_ok = protocol_faithful(len(completed), num_runs)

        status: ClaimStatus
        fidelity: Fidelity | None = None
        if passing:
            chosen = min(passing, key=lambda c: (not faithful(c[0]), c[1], c[0].evidence_id))
            status = "verified"
            fidelity = "spec_faithful" if faithful(chosen[0]) and protocol_ok else "inferred"
        elif (
            len(runs) >= max_tries
            and completed
            and all(faithful(c[0]) for c in completed)
            and protocol_ok
            and any(not c[2] for c in completed)
            and not open_unknowns
        ):
            status = "contradicted"
        elif not runs and smoke_blocked:
            status = "blocked"
        else:
            status = "open"

        ledger[cid] = LedgerEntry(
            claim_id=cid,
            status=status,
            fidelity=fidelity,
            attempts=len(runs),
            best_run_evidence_id=best[0].evidence_id if best else None,
            best_abs_error=best[1] if best else None,
            open_unknowns=open_unknowns,
            evidence_ids=(*claim_ev_ids, *(r.evidence_id for r in runs)),
            tolerance=tol,
            tolerance_evidence_id=stored[1] if stored else None,
        )
    return ledger
