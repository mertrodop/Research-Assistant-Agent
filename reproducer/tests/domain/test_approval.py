"""Pure approval policy, request / decision evidence, queue and human answers."""

from __future__ import annotations

import pytest

from reproducer.domain.approval import (
    ApprovalPolicy,
    answer_unknown,
    approval_decision_evidence,
    approval_request_evidence,
    escalation_queue,
    find_decision,
    gate,
    needs_approval,
    policy_from_env,
    policy_warning,
    request_digest,
    resolve_request_id,
)
from reproducer.domain.evidence import make_evidence
from reproducer.domain.unknowns import UnknownOpened, fold_unknowns, unknown_id_for

STRICT, DEFAULT, OFF = (ApprovalPolicy(mode=m) for m in ("strict", "default", "off"))  # type: ignore[arg-type]
SHARED = ApprovalPolicy(mode="default", shared_vault="/vaults/team")


@pytest.mark.parametrize(
    ("action", "cost", "policy", "subject", "expected"),
    [
        ("full_run", 0.1, STRICT, "", True),
        ("critic_patch", 1, STRICT, "", True),
        ("shared_vault_write", 3, STRICT, "/tmp/mine", True),
        ("full_run", 0.99, DEFAULT, "", False),
        ("full_run", 1.0, DEFAULT, "", True),  # threshold is inclusive
        ("full_run", 2.5, DEFAULT, "", True),
        ("critic_patch", 1, DEFAULT, "", True),
        ("shared_vault_write", 3, DEFAULT, "/vaults/team", False),  # no shared vault set
        ("shared_vault_write", 3, SHARED, "/vaults/team", True),
        ("shared_vault_write", 3, SHARED, "/vaults/team/", True),  # normalized
        ("shared_vault_write", 3, SHARED, "/tmp/mine", False),
        ("full_run", 99.0, OFF, "", False),
        ("critic_patch", 1, OFF, "", False),
        ("shared_vault_write", 3, OFF, "/vaults/team", False),
    ],
)
def test_policy_table(
    action: str, cost: float, policy: ApprovalPolicy, subject: str, expected: bool
) -> None:
    needed, reason = needs_approval(action, cost, policy, subject=subject)  # type: ignore[arg-type]
    assert needed is expected and reason


def test_policy_from_env_and_thresholds() -> None:
    assert policy_from_env({}) == ApprovalPolicy("default", 1.0, None)
    p = policy_from_env({"REPRODUCER_APPROVAL": "Strict", "REPRODUCER_APPROVE_GPU_HOURS": "2.5",
                         "REPRODUCER_SHARED_VAULT": "/v"})  # fmt: skip
    assert p == ApprovalPolicy("strict", 2.5, "/v")
    two = policy_from_env({"REPRODUCER_APPROVE_GPU_HOURS": "2"})
    assert needs_approval("full_run", 1.5, two)[0] is False
    assert needs_approval("full_run", 2.0, two)[0] is True
    for bad in ({"REPRODUCER_APPROVAL": "maybe"}, {"REPRODUCER_APPROVE_GPU_HOURS": "x"},
                {"REPRODUCER_APPROVE_GPU_HOURS": "-1"}):  # fmt: skip
        with pytest.raises(ValueError):
            policy_from_env(bad)
    assert policy_warning(OFF) and "WITHOUT human approval" in policy_warning(OFF)  # type: ignore[operator]
    assert policy_warning(DEFAULT) is None


def _request(digest: str = "d" * 64, subject: str = "c1") -> object:
    return approval_request_evidence(
        action="full_run", subject=subject, cost={"gpu_h": 2.0}, detail={"command": "python x"},
        input_digest=digest, source="repo:x@0", links=("p" * 64,),
    )  # fmt: skip


def test_decisions_are_human_only_and_find_decision() -> None:
    req = _request()
    ev = {req.evidence_id: req}  # type: ignore[attr-defined]
    assert find_decision(ev, "d" * 64) == "pending"
    assert find_decision({}, "d" * 64) == "pending"
    dec = approval_decision_evidence(req, "approved", "ok")  # type: ignore[arg-type]
    assert dec.authority == "human" and dec.confidence == "observed"
    assert find_decision({**ev, dec.evidence_id: dec}, "d" * 64) == "approved"
    rej = approval_decision_evidence(req, "rejected", "too costly")  # type: ignore[arg-type]
    assert find_decision({**ev, rej.evidence_id: rej}, "d" * 64) == "rejected"
    with pytest.raises(ValueError):  # the LLM (or code) can never approve
        make_evidence(kind="approval_decision", source="llm", locator="x",
                      content={"request_id": req.evidence_id, "decision": "approved", "note": ""},  # type: ignore[attr-defined]
                      confidence="inferred", authority="llm", links=(req.evidence_id,))  # type: ignore[attr-defined]  # fmt: skip
    with pytest.raises(ValueError):  # a request must link its trigger
        make_evidence(kind="approval_request", source="s", locator="x", content={},
                      confidence="derived", authority="rule")  # fmt: skip


def test_gate_files_once_and_follows_the_decision() -> None:
    req = _request()
    g = gate({}, req, DEFAULT, 2.0)  # type: ignore[arg-type]
    assert g.status == "pending" and g.new_request
    assert gate({}, req, DEFAULT, 0.5).status is None  # type: ignore[arg-type]
    ev = {req.evidence_id: req}  # type: ignore[attr-defined]
    again = gate(ev, req, OFF, 2.0)  # type: ignore[arg-type]
    assert again.status == "pending" and not again.new_request  # stored request is kept
    dec = approval_decision_evidence(req, "rejected", "no")  # type: ignore[arg-type]
    g = gate({**ev, dec.evidence_id: dec}, req, DEFAULT, 2.0)  # type: ignore[arg-type]
    assert g.status == "rejected" and g.rejected_reason == "rejected_by_human: no"


def test_resolve_request_id_errors() -> None:
    req = _request()
    ev = {req.evidence_id: req}  # type: ignore[attr-defined]
    assert resolve_request_id(ev, req.evidence_id[:12]) is req  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="no approval request"):
        resolve_request_id(ev, "ffffffffffff")
    with pytest.raises(ValueError, match="too short"):
        resolve_request_id(ev, "ab")
    dec = approval_decision_evidence(req, "approved", "")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already approved"):
        resolve_request_id({**ev, dec.evidence_id: dec}, req.evidence_id[:12])  # type: ignore[attr-defined]


def test_request_digest_identity() -> None:
    a = request_digest("full_run", {"claim_id": "c1", "run_digest": "r"})
    assert a == request_digest("full_run", {"run_digest": "r", "claim_id": "c1"})
    assert a != request_digest("critic_patch", {"claim_id": "c1", "run_digest": "r"})


# --- human answers to unknowns ---------------------------------------------------------------


def _unknown_store(name: str = "momentum") -> tuple[list[object], dict[str, object]]:
    setup = make_evidence(kind="paper_setup", source="paper:p", locator="Sec 4",
                          content={"name": name, "value": None, "quote": "q"},
                          confidence="observed", authority="paper")  # fmt: skip
    opened = UnknownOpened(
        unknown_id=unknown_id_for(name), name=name, evidence_id=setup.evidence_id
    )
    return [opened], {setup.evidence_id: setup}


def test_answer_resolves_open_unknown_and_conflicts_raise() -> None:
    events, evidence = _unknown_store()
    ans = answer_unknown(events, evidence, "momentum", "0.9", "from the authors' email")  # type: ignore[arg-type]
    assert ans.evidence.kind == "human_input" and ans.evidence.authority == "human"
    assert ans.event is not None and ans.event.value == "0.9"
    events2 = [*events, ans.event]
    evidence2 = {**evidence, ans.evidence.evidence_id: ans.evidence}
    st = fold_unknowns(events2, evidence2)["unk:momentum"]  # type: ignore[arg-type]
    assert st.status == "resolved" and st.resolved_by == ans.evidence.evidence_id
    same = answer_unknown(events2, evidence2, "momentum", "0.9", "again")  # type: ignore[arg-type]
    assert same.event is None  # already resolved to that value: no-op
    with pytest.raises(ValueError, match="conflicting resolution"):
        answer_unknown(events2, evidence2, "momentum", "0.5", "other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no unknown named"):
        answer_unknown(events, evidence, "nesterov", "true", "x")  # type: ignore[arg-type]


def test_queue_lists_pending_requests_and_open_unknowns() -> None:
    events, evidence = _unknown_store()
    req, decided = _request("a" * 64), _request("b" * 64, subject="c2")
    dec = approval_decision_evidence(decided, "approved", "")  # type: ignore[arg-type]
    ev = {**evidence, req.evidence_id: req, decided.evidence_id: decided, dec.evidence_id: dec}  # type: ignore[attr-defined]
    q = escalation_queue(ev, fold_unknowns(events, ev))  # type: ignore[arg-type]
    assert [r.evidence_id for r in q.requests] == [req.evidence_id]  # type: ignore[attr-defined]
    assert [u.name for u in q.unknowns] == ["momentum"] and not q.empty


def test_answer_shrinks_ledger_reasons_and_shows_in_vault() -> None:
    from vault_fixture import C10, cutout_store

    from reproducer.domain.evaluate import evaluate_claims
    from reproducer.domain.vault_render import _claims_from_evidence, render_vault

    evidence, events, calls = cutout_store()
    claims = tuple(_claims_from_evidence(evidence).values())

    def evaluate(ev, evs):  # type: ignore[no-untyped-def]
        return evaluate_claims(claims, ev, fold_unknowns(evs, ev))

    before = [
        r for r in evaluate(evidence, events)[C10].inferred_reasons if r.kind == "open_unknown"
    ]
    assert any(r.text.startswith("seed ") for r in before)
    ans = answer_unknown(events, evidence, "seed", "0", "repo README: seed 0")
    assert ans.event is not None
    evidence2 = {**evidence, ans.evidence.evidence_id: ans.evidence}
    events2 = [*events, ans.event]
    evals = evaluate(evidence2, events2)
    after = [r for r in evals[C10].inferred_reasons if r.kind == "open_unknown"]
    assert len(after) == len(before) - 1 and not any(r.text.startswith("seed ") for r in after)

    notes = render_vault(evidence2, events2, fold_unknowns(events2, evidence2), evals,
                         call_records=calls)  # fmt: skip
    unk = next(t for p, t in notes.items() if p.startswith("Unknowns/") and p.endswith("seed.md"))
    assert "resolved by human: repo README: seed 0" in unk
    claim = next(t for p, t in notes.items() if p.startswith("Claims/") and p.endswith(f"{C10}.md"))
    assert "(human answer: repo README: seed 0)" in claim
