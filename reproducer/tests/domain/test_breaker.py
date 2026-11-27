"""Tests for reproducer.domain.breaker (trial budget derived from evidence)."""

from __future__ import annotations

from reproducer.domain.breaker import breaker_state
from reproducer.domain.contracts import ExperimentResult
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import make_aborted_run_evidence, make_run_evidence

CLAIMS = ("c10", "c100")


def _run(cid: str, n: int, exit_code: int = 0, gpu_h: float = 1.0) -> Evidence:
    res = ExperimentResult(
        claim_id=cid, run_id=f"run-{cid}-{n}", exit_code=exit_code,
        metrics={"test_error": 4.0} if exit_code == 0 else {}, log_ref="x",
        config_digest="d" * 64, duration=gpu_h * 3600, gpu_hours=gpu_h,
    )  # fmt: skip
    return make_run_evidence(res, spec_faithful=True, assumption_evidence_ids=(), source="s")


def _aborted(cid: str, n: int) -> Evidence:
    return make_aborted_run_evidence(
        claim_id=cid, run_id=f"abort-{cid}-{n}", log_ref="x", config_digest="d" * 64,
        duration_s=1800, reason="KeyboardInterrupt", spec_faithful=True,
        assumption_evidence_ids=(), source="s",
    )  # fmt: skip


def _setup_failed(cid: str) -> Evidence:
    return make_evidence(kind="setup_failed", source="s", locator=f"sf-{cid}",
                         content={"claim_id": cid, "stage": "image_check"},
                         confidence="observed", authority="execution")  # fmt: skip


def _state(evs: list[Evidence], **kw):  # type: ignore[no-untyped-def]
    args = dict(gpu_h_cap=10.0, token_budget=1000, tokens_used=0) | kw
    return breaker_state(CLAIMS, {e.evidence_id: e for e in evs}, **args)


def test_empty_store_full_budget() -> None:
    b = _state([])
    assert b.global_max == 6 and b.global_remaining == 6 and b.tripped == ()
    assert b.per_claim["c10"].remaining == 3 and b.can_try("c10")
    assert b.gpu_h_remaining == 10.0 and b.tokens_remaining == 1000


def test_aborted_counts_only_after_container_start() -> None:
    b = _state([_run("c10", 1), _aborted("c10", 1), _setup_failed("c10")])
    c = b.per_claim["c10"]
    assert c.trials == 2 and c.aborted == 1 and c.remaining == 1  # setup_failed: no trial
    assert b.global_used == 2 and b.gpu_h_spent == 1.5  # aborted GPU time counts
    assert b.per_claim["c100"].trials == 0


def test_crashed_runs_count_as_trials() -> None:
    b = _state([_run("c10", 1, exit_code=1)])
    assert b.per_claim["c10"].trials == 1


def test_per_claim_cap_trips_only_that_claim() -> None:
    b = _state([_run("c10", i, gpu_h=0.1) for i in range(3)])
    assert not b.can_try("c10") and b.can_try("c100")
    assert b.tripped == ("claim c10: 3/3 trials used",)


def test_global_cap_trips() -> None:
    runs = [_run(c, i, gpu_h=0.1) for c in CLAIMS for i in range(3)]
    b = _state(runs)
    assert b.global_remaining == 0 and not b.can_try("c100")
    assert "global: 6/6 trials used" in b.tripped


def test_gpu_cap_trips_every_claim() -> None:
    b = _state([_run("c10", 1, gpu_h=6.0), _run("c100", 1, gpu_h=4.5)])
    assert b.gpu_h_remaining == 0 and not b.can_try("c10") and not b.can_try("c100")
    assert any(t.startswith("gpu: 10.500 h spent") for t in b.tripped)


def test_token_budget_trips() -> None:
    b = _state([], tokens_used=1200)
    assert b.tokens_remaining == 0 and not b.can_try("c10")
    assert b.tripped == ("tokens: 1200 used >= budget 1000",)


def test_runs_of_other_claims_do_not_use_the_trial_budget() -> None:
    b = _state([_run("other", 1)])
    assert b.global_used == 0 and b.gpu_h_spent == 1.0  # GPU is per paper


def test_limits_from_env() -> None:
    import pytest

    from reproducer.domain.breaker import max_trials_from_env, token_budget_from_env

    assert (token_budget_from_env({}), max_trials_from_env({})) == (200_000, 3)
    env = {"REPRODUCER_TOKEN_BUDGET": "5000", "REPRODUCER_MAX_TRIALS": "5"}
    assert (token_budget_from_env(env), max_trials_from_env(env)) == (5000, 5)
    for bad in ("0", "-1", "x", "2.5"):
        with pytest.raises(ValueError, match="REPRODUCER_MAX_TRIALS"):
            max_trials_from_env({"REPRODUCER_MAX_TRIALS": bad})
