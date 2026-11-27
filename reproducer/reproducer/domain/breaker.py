"""Trial budget / circuit breaker (M4): derived from stored evidence only, never counted in
memory.

Pure domain: no file, network or subprocess IO (the env limits are read from a mapping).

  trials per claim   run_result evidence of the claim: runs that finished (exit 0 or not) and
                     aborted runs (status "aborted": written only after the container
                     started). setup_failed notes (the container never started) do not count.
  max per claim      REPRODUCER_MAX_TRIALS (default MAX_TRIALS_PER_CLAIM = 3)
  global             N = n_claims * max per claim; used = trials of those claims
  GPU hours          the paper cap minus the gpu_hours of every stored run_result
  tokens             REPRODUCER_TOKEN_BUDGET (default 200000) minus the tokens used (LLM call
                     telemetry, passed in)
tripped: every exhausted budget, as a reason (a claim at its limit trips only that claim; the
global / GPU / token budgets trip every claim). At the limit a claim stays OPEN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from reproducer.domain.evidence import Evidence

MAX_TRIALS_PER_CLAIM = 3  # default of REPRODUCER_MAX_TRIALS
DEFAULT_TOKEN_BUDGET = 200_000  # default of REPRODUCER_TOKEN_BUDGET
TOKEN_BUDGET_ENV = "REPRODUCER_TOKEN_BUDGET"
MAX_TRIALS_ENV = "REPRODUCER_MAX_TRIALS"


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if v < 1:
        raise ValueError(f"{name}={raw!r} must be >= 1")
    return v


def token_budget_from_env(env: Mapping[str, str]) -> int:
    """REPRODUCER_TOKEN_BUDGET (default 200000): the paper's token budget for the breaker."""
    return _positive_int(env, TOKEN_BUDGET_ENV, DEFAULT_TOKEN_BUDGET)


def max_trials_from_env(env: Mapping[str, str]) -> int:
    """REPRODUCER_MAX_TRIALS (default 3): trials per claim (breaker, ledger, vault, CLI)."""
    return _positive_int(env, MAX_TRIALS_ENV, MAX_TRIALS_PER_CLAIM)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimBudget(_Frozen):
    claim_id: str
    trials: int
    aborted: int
    max_trials: int

    @property
    def remaining(self) -> int:
        return max(0, self.max_trials - self.trials)


class BreakerState(_Frozen):
    per_claim: dict[str, ClaimBudget]
    global_max: int
    global_used: int
    gpu_h_cap: float
    gpu_h_spent: float
    token_budget: int
    tokens_used: int
    tripped: tuple[str, ...]

    @property
    def global_remaining(self) -> int:
        return max(0, self.global_max - self.global_used)

    @property
    def gpu_h_remaining(self) -> float:
        return max(0.0, round(self.gpu_h_cap - self.gpu_h_spent, 4))

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def global_tripped(self) -> bool:
        return (
            self.global_remaining == 0 or self.gpu_h_remaining == 0 or (self.tokens_remaining == 0)
        )

    def can_try(self, claim_id: str) -> bool:
        """Another trial of the claim fits every budget."""
        b = self.per_claim.get(claim_id)
        return b is not None and b.remaining > 0 and not self.global_tripped


def _gpu_hours(ev: Evidence) -> float:
    v = ev.content.get("gpu_hours")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0.0
    return float(v)


def breaker_state(
    claim_ids: Sequence[str],
    evidence: Mapping[str, Evidence],
    *,
    gpu_h_cap: float,
    token_budget: int,
    tokens_used: int,
    max_per_claim: int = MAX_TRIALS_PER_CLAIM,
) -> BreakerState:
    runs = [ev for ev in evidence.values() if ev.kind == "run_result"]
    per: dict[str, ClaimBudget] = {}
    for cid in claim_ids:
        mine = [ev for ev in runs if ev.content.get("claim_id") == cid]
        per[cid] = ClaimBudget(
            claim_id=cid,
            trials=len(mine),
            aborted=sum(1 for ev in mine if ev.content.get("status") == "aborted"),
            max_trials=max_per_claim,
        )
    spent = round(sum(_gpu_hours(ev) for ev in runs), 4)
    state = BreakerState(
        per_claim=per,
        global_max=len(per) * max_per_claim,
        global_used=sum(b.trials for b in per.values()),
        gpu_h_cap=gpu_h_cap,
        gpu_h_spent=spent,
        token_budget=token_budget,
        tokens_used=tokens_used,
        tripped=(),
    )
    tripped = [
        f"claim {b.claim_id}: {b.trials}/{b.max_trials} trials used"
        for b in per.values()
        if b.remaining == 0
    ]
    if state.global_remaining == 0:
        tripped.append(f"global: {state.global_used}/{state.global_max} trials used")
    if state.gpu_h_remaining == 0:
        tripped.append(f"gpu: {spent:.3f} h spent >= paper cap {gpu_h_cap:g} h")
    if state.tokens_remaining == 0:
        tripped.append(f"tokens: {tokens_used} used >= budget {token_budget}")
    return state.model_copy(update={"tripped": tuple(tripped)})
