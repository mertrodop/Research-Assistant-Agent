"""A Cutout-like store for the vault renderer: the real paper spec, an env plan with the torch
deviation, one run_plan + preflight + completed run per claim, paper unknowns (seed,
eval_checkpoint) assumed by the plans, env unknowns assumed by the env plan and a repo unknown
resolved by execution evidence. In file order, as RunStore.load_evidence returns it."""

from __future__ import annotations

from pathlib import Path

from reproducer.domain.contracts import ExperimentResult, ReproSpec
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import make_run_evidence, tolerance_evidence
from reproducer.domain.telemetry import CallRecord
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    spec_to_evidence,
    unknown_id_for,
)

SPEC = Path(__file__).resolve().parents[2] / "evals" / "cutout" / "spec.json"
PAPER = "paper:cutout"
REPO = "repo:uoguelph-mlrg/Cutout@287f934ea5fa00d4345c2cccecf3552e2b1c33e3"
C10, C100 = "c10plus_resnet18_cutout", "c100plus_resnet18_cutout"
IMAGE = "reproducer/cutout:424da3bbafa8"
COMMANDS = {
    C10: "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout "
    "--length 16",
    C100: "python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout "
    "--length 8",
}
RESULTS = {C10: (0.959, 4.100000000000003, 5804.9), C100: (0.784, 21.599999999999998, 6074.4)}

UnknownEventT = UnknownOpened | UnknownResolved | Assumption


def run_id(cid: str) -> str:
    return f"run-{cid}-20261124-1"


def cutout_store() -> tuple[dict[str, Evidence], list[UnknownEventT], list[CallRecord]]:
    spec = ReproSpec.model_validate_json(SPEC.read_text())
    paper, opened = spec_to_evidence(spec, PAPER)
    records: list[Evidence] = list(paper)
    events: list[UnknownEventT] = list(opened)

    claim_ids = {ev.content["claim_id"]: ev.evidence_id for ev in paper if ev.kind == "paper_claim"}
    for c in spec.claims:
        records.append(tolerance_evidence(c, claim_evidence_id=claim_ids[c.claim_id], source=PAPER))

    workflow = make_evidence(kind="workflow", source=REPO, locator="subtask:C",
                             content={"datasets": []}, confidence="inferred",
                             authority="llm")  # fmt: skip
    records.append(workflow)
    events.append(UnknownOpened(unknown_id=unknown_id_for("dataset_source"),
                                name="dataset_source",
                                evidence_id=workflow.evidence_id))  # fmt: skip
    env_facts = make_evidence(kind="env_spec", source=REPO, locator="subtask:A",
                              content={"torch": "0.4.0"}, confidence="derived",
                              authority="llm", links=(workflow.evidence_id,))  # fmt: skip
    records.append(env_facts)
    for name in ("python_version", "cuda_version"):
        events.append(UnknownOpened(unknown_id=unknown_id_for(name), name=name,
                                    evidence_id=env_facts.evidence_id))  # fmt: skip

    env = make_evidence(
        kind="env_spec", source=REPO, locator="env_plan",
        content={"plan": {
            "base_image": "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
            "python": "3.7", "torch": "1.12.1", "torchvision": "0.13.1", "cuda": "11.3",
            "deviations": [{"field": "torch", "stated": "0.4.0", "chosen": "1.12.1",
                            "reason": "no CUDA build for sm_89"}],
            "assumptions": [
                {"name": "python_version", "value": "3.7", "reason": "python of the image"},
                {"name": "cuda_version", "value": "11.3", "reason": "CUDA build of torch"},
                {"name": "unpinned:tqdm", "value": None, "reason": "unpinned"},
            ],
        }},
        confidence="derived", authority="rule", links=(env_facts.evidence_id,),
    )  # fmt: skip
    records.append(env)
    events += [Assumption(unknown_id="unk:python_version", value="3.7",
                          evidence_id=env.evidence_id),
               Assumption(unknown_id="unk:cuda_version", value="11.3",
                          evidence_id=env.evidence_id)]  # fmt: skip

    cache = make_evidence(kind="dataset_cache", source="data-cache@" + IMAGE, locator="cifar10",
                          content={"dataset": "cifar10", "verified": True},
                          confidence="observed", authority="execution")  # fmt: skip
    records.append(cache)
    events.append(UnknownResolved(unknown_id="unk:dataset_source", value="torchvision:CIFAR10",
                                  evidence_id=cache.evidence_id))  # fmt: skip

    for cid in (C10, C100):
        plan = make_evidence(
            kind="run_plan", source=REPO, locator=f"run_plan:{cid}",
            content={
                "claim_id": cid, "command": COMMANDS[cid], "deviations": [],
                "plan": {"depends_on_unknowns": ["unk:eval_checkpoint", "unk:seed"]},
                "assumptions": [
                    {"unknown_id": "unk:seed", "value": "0"},
                    {"unknown_id": "unk:eval_checkpoint", "value": "last"},
                ],
                "notes": [
                    {"field": "seed", "status": "assumption", "detail": "repo default --seed 0"},
                    {"field": "eval_checkpoint", "status": "assumption", "detail": "which=last"},
                ],
            },
            confidence="derived", authority="rule", links=(claim_ids[cid],),
        )  # fmt: skip
        records.append(plan)
        events += [Assumption(unknown_id="unk:seed", value="0", evidence_id=plan.evidence_id),
                   Assumption(unknown_id="unk:eval_checkpoint", value="last",
                              evidence_id=plan.evidence_id)]  # fmt: skip
        digest = ("a" if cid == C10 else "b") * 64
        pre = make_evidence(kind="preflight", source=REPO, locator=f"preflight:{cid}",
                            content={"claim_id": cid, "command": COMMANDS[cid], "image": IMAGE,
                                     "plan_evidence_id": plan.evidence_id,
                                     "run_digest": digest, "decision": "run"},
                            confidence="observed", authority="execution",
                            links=(plan.evidence_id,))  # fmt: skip
        records.append(pre)
        raw, value, duration = RESULTS[cid]
        result = ExperimentResult(
            claim_id=cid, run_id=run_id(cid), exit_code=0,
            metrics={"raw:test_acc": raw, "test_error": value},
            log_ref=f"runs/{run_id(cid)}/run.log", config_digest=digest,
            duration=duration, gpu_hours=round(duration / 3600, 4),
        )  # fmt: skip
        # C100: a new record with stored artifact refs; C10: an older record without the key
        stored = (
            [f"runs/{run_id(cid)}/artifacts/logs/cifar100_resnet18.csv"] if cid == C100 else None
        )
        records.append(make_run_evidence(
            result, spec_faithful=False, assumption_evidence_ids=(plan.evidence_id,),
            source=REPO, links=(env.evidence_id,), artifacts=stored,
        ))  # fmt: skip

    calls = [
        CallRecord(node="A", backend="anthropic", model="claude-haiku-5-5", input_tokens=900,
                   output_tokens=100, latency_s=1.0, prompt_sha256="0" * 64),
        CallRecord(node="B", backend="ollama", model="qwen", input_tokens=400,
                   output_tokens=50, latency_s=2.0, prompt_sha256="1" * 64),
    ]  # fmt: skip
    return {ev.evidence_id: ev for ev in records}, events, calls
