"""End-to-end M0 story: spec -> evidence/unknowns -> assumption run -> resolution ->
faithful run -> contradicted claim -> reload from disk gives the identical ledger."""

from __future__ import annotations

from pathlib import Path

from reproducer.domain.contracts import ExperimentResult, ReproSpec
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import (
    LedgerEntry,
    Tolerance,
    ToleranceConfig,
    compute_ledger,
    make_run_evidence,
)
from reproducer.domain.unknowns import (
    Assumption,
    fold_unknowns,
    resolve,
    spec_to_evidence,
    unknown_id_for,
)
from reproducer.providers.store import RunStore

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "repro_spec_example.json"
PAPER = "paper:resnet18-cifar10"
REPO = "repo:example-lab/resnet18-cifar10@abc123"
METRIC = "top1_accuracy"
TOL = ToleranceConfig(default=Tolerance(abs=0.5))
WD = unknown_id_for("weight_decay")


def _ledger(store: RunStore, spec: ReproSpec) -> dict[str, LedgerEntry]:
    evidence = store.load_evidence()
    unknowns = fold_unknowns(store.load_events(), evidence)
    return compute_ledger(spec.claims, evidence, unknowns, TOL)


def _run_ev(
    run_id: str, claim_id: str, value: float, *, assumptions: tuple[str, ...] = ()
) -> Evidence:
    result = ExperimentResult(
        claim_id=claim_id,
        run_id=run_id,
        exit_code=0,
        metrics={METRIC: value},
        log_ref=f"logs/{run_id}.log",
        config_digest=f"sha256:{run_id}",
        duration=3600.0,
        gpu_hours=1.0,
    )
    return make_run_evidence(
        result,
        spec_faithful=not assumptions,
        assumption_evidence_ids=assumptions,
        source=REPO,
    )


def test_m0_story(tmp_path: Path) -> None:
    spec = ReproSpec.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    store = RunStore(tmp_path / "resnet18-cifar10")

    # 1. spec -> evidence + UnknownOpened
    records, opened = spec_to_evidence(spec, PAPER)
    for ev in records:
        assert store.append_evidence(ev)
    for e in opened:
        store.append_event(e)
    assert [o.unknown_id for o in opened] == [WD]
    assert not any(store.append_evidence(ev) for ev in records)  # idempotent re-ingest
    ledger = _ledger(store, spec)
    assert {e.status for e in ledger.values()} == {"open"}
    assert ledger["c1"].open_unknowns == (WD,)

    # 2. LLM proposes weight_decay: inferred evidence + Assumption (does not resolve)
    llm_ev = make_evidence(
        kind="llm_proposal",
        source="llm:proposal",
        locator="weight_decay",
        content={"name": "weight_decay", "value": "5e-4", "rationale": "common CIFAR default"},
        confidence="inferred",
        authority="llm",
        links=(opened[0].evidence_id,),
    )
    store.append_evidence(llm_ev)
    store.append_event(Assumption(unknown_id=WD, value="5e-4", evidence_id=llm_ev.evidence_id))

    # 3. Run 1 with the assumption, within tolerance -> verified, fidelity inferred
    run1 = _run_ev("run-1", "c1", 95.1, assumptions=(llm_ev.evidence_id,))
    store.append_evidence(run1)
    entry = _ledger(store, spec)["c1"]
    assert (entry.status, entry.fidelity) == ("verified", "inferred")
    assert entry.open_unknowns == (WD,)

    # 4. Repo evidence resolves weight_decay
    repo_ev = make_evidence(
        kind="repo_file",
        source=REPO,
        locator="configs/cifar10.yaml:L12",
        content={"key": "weight_decay", "value": "5e-4"},
        confidence="observed",
        authority="repo",
    )
    store.append_evidence(repo_ev)
    store.append_event(resolve(opened[0], repo_ev, "5e-4"))
    states = fold_unknowns(store.load_events(), store.load_evidence())
    assert states[WD].status == "resolved"
    assert states[WD].resolved_by == repo_ev.evidence_id

    # 5. Run 2 spec-faithful, within tolerance -> verified, spec_faithful
    run2 = _run_ev("run-2", "c1", 95.4)
    store.append_evidence(run2)
    entry = _ledger(store, spec)["c1"]
    assert (entry.status, entry.fidelity) == ("verified", "spec_faithful")
    assert entry.open_unknowns == ()
    assert entry.attempts == 2

    # 6. c2: 3 faithful completed runs out of tolerance (93.1 +/- 0.5) -> contradicted
    for i, v in enumerate((91.0, 91.2, 90.8), start=1):
        store.append_evidence(_run_ev(f"c2-run-{i}", "c2", v))
    ledger = _ledger(store, spec)
    assert ledger["c2"].status == "contradicted"
    assert ledger["c2"].fidelity is None
    assert ledger["c2"].attempts == 3
    assert ledger["c1"].status == "verified"

    # 7. Fresh store on the same folder -> identical ledger
    fresh = RunStore(tmp_path / "resnet18-cifar10")
    assert _ledger(fresh, spec) == ledger
    assert fresh.load_evidence() == store.load_evidence()
    assert fresh.load_events() == store.load_events()
