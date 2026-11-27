"""Tests for the pre-registered tolerance (reproducer.domain.ledger)."""

from __future__ import annotations

import pytest
from conftest import C10, C100
from pydantic import ValidationError

from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import (
    DEFAULT_TOLERANCE_ABS,
    pre_registered_tolerance,
    sigma_from_quote,
    tolerance_evidence,
)


@pytest.mark.parametrize(("cid", "sigma", "tol"), [(C10, 0.13, 0.26), (C100, 0.24, 0.48)])
def test_sigma_from_cutout_quote(
    cid: str, sigma: float, tol: float, cutout_claims: dict[str, ResultClaim]
) -> None:
    claim = cutout_claims[cid]
    assert sigma_from_quote(claim) == sigma  # not the sigma of the neighbouring column
    t, s, reason = pre_registered_tolerance(claim)
    assert (t.abs, t.rel, s) == (pytest.approx(tol), None, sigma)
    assert "paper std" in reason


def test_missing_sigma_uses_default(cutout_claims: dict[str, ResultClaim]) -> None:
    claim = cutout_claims[C10].model_copy(update={"quote": "ResNet18 + cutout 3.99"})
    t, s, reason = pre_registered_tolerance(claim)
    assert (t.abs, s) == (DEFAULT_TOLERANCE_ABS, None) and DEFAULT_TOLERANCE_ABS == 0.5
    assert "default" in reason


def test_ambiguous_sigma_is_not_guessed(cutout_claims: dict[str, ResultClaim]) -> None:
    claim = cutout_claims[C10].model_copy(update={"quote": "3.99 ± 0.13 and 3.99 ± 0.20"})
    assert sigma_from_quote(claim) is None
    other = cutout_claims[C10].model_copy(update={"quote": "3.99 +/- 0.13"})
    assert sigma_from_quote(other) == 0.13


def test_tolerance_evidence(
    cutout_claims: dict[str, ResultClaim], cutout_evidence: dict[str, Evidence]
) -> None:
    claim = cutout_claims[C10]
    claim_id = next(
        i
        for i, ev in cutout_evidence.items()
        if ev.kind == "paper_claim" and ev.content["claim_id"] == C10
    )
    ev = tolerance_evidence(claim, claim_evidence_id=claim_id, source="arxiv:1708.04552")
    assert (ev.kind, ev.authority, ev.confidence, ev.links) == (
        "tolerance",
        "rule",
        "derived",
        (claim_id,),
    )
    assert ev.content["abs"] == pytest.approx(0.26) and ev.content["sigma"] == 0.13
    assert ev.content["rel"] is None and ev.content["claim_id"] == C10


def test_tolerance_kind_requires_rule_derived_and_link() -> None:
    base = {"kind": "tolerance", "source": "s", "locator": "tolerance:c", "content": {"abs": 1}}
    with pytest.raises(ValidationError, match="tolerance evidence requires"):
        make_evidence(**base, confidence="observed", authority="paper", links=("x",))
    with pytest.raises(ValidationError, match="tolerance evidence requires"):
        make_evidence(**base, confidence="derived", authority="rule")
