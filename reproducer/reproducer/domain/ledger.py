"""Ledger: per-claim status computed by code from evidence + unknowns.

Tolerance lives here and is never written by an LLM. The ledger only reads claims, evidence,
unknowns and tolerance; it never alters them.
Pure domain: no file, network or subprocess IO.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

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


# --- Run / smoke evidence ------------------------------------------------------------------


def make_run_evidence(
    result: ExperimentResult,
    *,
    spec_faithful: bool,
    assumption_evidence_ids: tuple[str, ...],
    source: str,
) -> Evidence:
    """run_result evidence. Non-finite metrics become None and mark the run diverged
    (informational only; the ledger decides completion from the claimed metric).

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
    return make_evidence(
        kind="run_result",
        source=source,
        locator=result.run_id,
        content=content,
        confidence="observed",
        authority="execution",
        links=assumption_evidence_ids,
    )


def make_smoke_evidence(
    *, passed: bool, attempts: int, capped: bool, source: str, locator: str
) -> Evidence:
    return make_evidence(
        kind="smoke_result",
        source=source,
        locator=locator,
        content={"passed": passed, "attempts": attempts, "capped": capped},
        confidence="observed",
        authority="execution",
    )


# --- Ledger --------------------------------------------------------------------------------


class LedgerEntry(_Frozen):
    claim_id: str
    status: ClaimStatus
    fidelity: Fidelity | None = None  # only set when verified
    attempts: int
    best_run_evidence_id: str | None = None
    best_abs_error: float | None = None
    open_unknowns: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()  # claim evidence + its runs

    @model_validator(mode="after")
    def _fidelity_only_when_verified(self) -> Self:
        if (self.status == "verified") != (self.fidelity is not None):
            raise ValueError("fidelity must be set iff status is verified")
        return self


class _RunView(BaseModel):
    """Read-only view of run_result content (other ExperimentResult fields ignored)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    claim_id: str
    exit_code: int
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


def compute_ledger(
    claims: tuple[ResultClaim, ...],
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    tolerance: ToleranceConfig,
    max_tries: int = 3,
) -> dict[str, LedgerEntry]:
    """Per-claim status. Precedence: verified > contradicted > blocked > open.

    Smoke blocking is per paper/repo; scope by repo@commit when multiple code_links are
    supported.
    """
    records = list(evidence.values())
    runs_all = [_parse_run(ev) for ev in records if ev.kind == "run_result"]
    smoke_blocked = any(ev.kind == "smoke_result" and _is_failed_capped_smoke(ev) for ev in records)

    ledger: dict[str, LedgerEntry] = {}
    for claim in claims:
        cid = claim.claim_id
        tol = tolerance.tolerance_for(claim)
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

        best = min(completed, key=lambda c: (c[1], c[0].evidence_id), default=None)
        passing = [c for c in completed if c[2]]

        status: ClaimStatus
        fidelity: Fidelity | None = None
        if passing:
            chosen = min(passing, key=lambda c: (not c[0].faithful, c[1], c[0].evidence_id))
            status = "verified"
            fidelity = "spec_faithful" if chosen[0].faithful else "inferred"
        elif (
            len(runs) >= max_tries
            and completed
            and all(c[0].faithful for c in completed)
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
        )
    return ledger
