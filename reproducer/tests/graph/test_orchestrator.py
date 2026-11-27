"""Orchestrator node on a RunStore (FakeLLM), plus one live run on runs/cutout_e2e."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.evidence import make_evidence
from reproducer.domain.unknowns import (
    Assumption,
    UnknownResolved,
    fold_unknowns,
    spec_to_evidence,
)
from reproducer.graph.nodes.orchestrator import (
    ORCHESTRATOR_NODE,
    current_plans,
    load_current_plans,
    run_orchestrator,
)
from reproducer.graph.nodes.subtasks import (
    candidate_entrypoints,
    command_evidence,
    metric_line_evidence,
)
from reproducer.providers.llm import Budget, FakeLLM, llm_for_node
from reproducer.providers.store import RunStore

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "evals" / "cutout" / "spec.json"
SOURCE = "repo:uoguelph-mlrg/Cutout@test"
C10, C100 = "c10plus_resnet18_cutout", "c100plus_resnet18_cutout"
COMMAND = (
    "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
)
README = f"# Cutout\n\n```\n{COMMAND}\n{COMMAND.replace('cifar10', 'cifar100')[:-2]}8\n```\n"
TRAIN = """\
import argparse
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

dataset_options = ['cifar10', 'cifar100', 'svhn']
model_options = ['resnet18', 'wideresnet']
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18', choices=model_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--data_augmentation', action='store_true', default=False)
parser.add_argument('--cutout', action='store_true', default=False)
parser.add_argument('--n_holes', type=int, default=1)
parser.add_argument('--length', type=int, default=16)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

def test(loader):
    correct = 0.
    total = 0.
    for images, labels in loader:
        pred = torch.max(cnn(images).data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    val_acc = correct / total
    return val_acc

for epoch in range(args.epochs):
    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))
"""
EXPECTED = {
    C10: COMMAND,
    C100: COMMAND.replace("cifar10", "cifar100").replace("16", "8"),
}


def _make_store(root: Path, train_text: str = TRAIN) -> RunStore:
    store = RunStore(root)
    spec = ReproSpec.model_validate_json(SPEC.read_text())
    paper, opened = spec_to_evidence(spec, "arxiv:1708.04552")

    def repo(locator: str, content: dict[str, Any], **kw: Any):  # noqa: ANN202
        return make_evidence(
            kind=kw.pop("kind", "repo_file"),
            source=SOURCE,
            locator=locator,
            content=content,
            confidence=kw.pop("confidence", "observed"),
            authority="repo",
            **kw,
        )

    files = ["README.md", "train.py"]
    inv = repo(
        "inventory",
        {
            "commit": "0" * 40,
            "files": [{"path": p, "size": 1, "is_symlink": False} for p in files],
            "readme": "README.md",
            "requirement_files": [],
            "docker_files": [],
            "config_files": [],
            "notebooks": [],
            "python_files": ["train.py"],
            "has_submodules": False,
        },
    )
    readme = repo("README.md", {"path": "README.md", "text": README}, kind="readme")
    train = repo("train.py", {"path": "train.py", "head": train_text, "truncated": False})
    (cand,) = candidate_entrypoints({"train.py": train_text}).candidates
    parsed = repo(
        "train.py#parsed",
        {"parser": "entrypoint", "result": cand.model_dump(mode="json")},
        confidence="derived",
        links=(train.evidence_id,),
    )
    base = {ev.evidence_id: ev for ev in (inv, readme, train, parsed)}
    derived = (*command_evidence(base), *metric_line_evidence(base))
    for ev in (*paper, inv, readme, train, parsed, *derived):
        store.append_evidence(ev)
    for u in opened:
        store.append_event(u)
    return store


METRIC_LINE = "tqdm.write('test_acc: %.3f' % (test_acc))"


def _mapping(lr_schedule_consistent: bool = True, metric_line: str = METRIC_LINE) -> dict[str, Any]:
    metric = {
        "metric_line_ref": "E2",
        "metric_line_quote": {"evidence_id": "E2", "quote": metric_line},
        "conversion": "accuracy_fraction_to_error_percent",
        "scale_citation": {"evidence_id": "E1", "quote": "val_acc = correct / total"},
    }
    hyper = [{"setup_name": n, "flag": f"--{n}"} for n in ("epochs", "batch_size", "learning_rate")]
    return {
        "claims": [
            {
                "claim_id": cid,
                "claim_args": [
                    {"flag": "--dataset", "value": ds},
                    {"flag": "--model", "value": "resnet18"},
                ],
                "setup_mappings": [{"setup_name": f"cutout_length_{ds}", "flag": "--length"}]
                + hyper,
                "metric": metric,
            }
            for cid, ds in ((C10, "cifar10"), (C100, "cifar100"))
        ],
        "unmapped_setup": [{"setup_name": "num_runs", "reason": "no flag"}],
        "setup_checks": [
            {
                "setup_name": "weight_decay",
                "repo_citation": {"evidence_id": "E3", "quote": "weight_decay=5e-4"},
                "consistent": True,
                "note": "5e-4 in the SGD call",
            },
            {
                "setup_name": "lr_schedule",
                "repo_citation": {"evidence_id": "E3", "quote": "milestones=[60, 120, 160]"},
                "consistent": lr_schedule_consistent,
                "note": "same milestones" if lr_schedule_consistent else "gamma differs",
            },
        ],
    }


def _fake(data: dict[str, Any]) -> FakeLLM:
    return FakeLLM(lambda system, prompt: json.dumps(data))


def _lines(store: RunStore) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in store.evidence_path.read_text().splitlines()]


def test_plans_both_claims_and_writes_tolerance_first(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    out = run_orchestrator(store, _fake(_mapping()))
    assert out.ok, out.errors
    assert {p.claim.claim_id: p.command for p in out.plans} == EXPECTED
    for p in out.plans:
        assert p.plan.depends_on_unknowns == ("unk:eval_checkpoint", "unk:seed")
        assert p.fidelity == "inferred"  # seed / eval_checkpoint are assumed
        assert p.plan.metric_source.conversion == "accuracy_fraction_to_error_percent"
        assert not p.deviations
    c100 = {a.flag: a for a in out.plans[1].plan.args}
    assert out.plans[1].plan.base_command == EXPECTED[C100]  # the matching verified command
    assert (c100["--length"].value, c100["--length"].source) == ("8", "repo")

    kinds = [r["kind"] for r in _lines(store)]
    first_plan = kinds.index("run_plan")
    assert kinds.count("tolerance") == 2
    assert max(i for i, k in enumerate(kinds) if k == "tolerance") < kinds.index("claim_mapping")
    assert kinds.index("claim_mapping") < first_plan
    assert any(r["locator"] == "train.py#train_config" for r in _lines(store))

    evidence = store.load_evidence()
    plan_ev = out.plans[0].evidence
    assert (plan_ev.authority, plan_ev.confidence) == ("rule", "derived")
    assert set(plan_ev.links) == {
        out.mapping_evidence.evidence_id,  # type: ignore[union-attr]
        out.plans[0].tolerance.evidence_id,
        next(
            i
            for i, ev in evidence.items()
            if ev.kind == "paper_claim" and ev.content["claim_id"] == C10
        ),
    }
    assert plan_ev.content["command"] == EXPECTED[C10]
    mapping_ev = out.mapping_evidence
    assert mapping_ev is not None and mapping_ev.authority == "llm"
    assert out.train_config is not None and out.train_config.evidence_id in mapping_ev.links

    events = store.load_events()
    assumed = [e for e in events if isinstance(e, Assumption)]
    assert {(e.unknown_id, e.value) for e in assumed} == {
        ("unk:seed", "0"),
        ("unk:eval_checkpoint", "last"),
    }
    assert {e.evidence_id for e in assumed} == {p.evidence.evidence_id for p in out.plans}
    assert len(assumed) == 4  # 2 unknowns x 2 plans
    fold_unknowns(events, evidence)  # the event log stays consistent


def test_load_current_plans_is_read_only_and_matches_stored(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="run `plan`"):
        load_current_plans(store)  # nothing planned yet
    out = run_orchestrator(store, _fake(_mapping()))
    n_lines, n_events = len(_lines(store)), len(store.load_events())
    cur = load_current_plans(store)
    assert cur.input_digest == out.input_digest
    assert [p.evidence.evidence_id for p in cur.plans] == [
        p.evidence.evidence_id for p in out.plans
    ]
    assert {p.claim.claim_id: p.command for p in cur.plans} == EXPECTED
    assert (len(_lines(store)), len(store.load_events())) == (n_lines, n_events)  # no writes


def test_rerun_reuses_mapping_and_writes_no_duplicates(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = run_orchestrator(store, _fake(_mapping()))
    n_ev, n_events = len(_lines(store)), len(store.load_events())
    fake = _fake(_mapping())
    again = run_orchestrator(store, fake)
    assert again.ok and again.reused and again.attempts == 0 and fake.prompts == []
    assert again.input_digest == first.input_digest
    assert again.events_written == ()
    assert [p.evidence for p in again.plans] == [p.evidence for p in first.plans]
    assert len(_lines(store)) == n_ev
    assert len(store.load_events()) == n_events
    current = current_plans(store.load_evidence(), first.input_digest)
    assert {c: ev.evidence_id for c, ev in current.items()} == {
        p.claim.claim_id: p.evidence.evidence_id for p in first.plans
    }


def test_replan_calls_again_and_newest_plan_is_current(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    run_orchestrator(store, _fake(_mapping()))
    fake = _fake(_mapping(lr_schedule_consistent=False))
    out = run_orchestrator(store, fake, replan=True)
    assert out.ok and not out.reused and len(fake.prompts) == 1
    current = current_plans(store.load_evidence(), out.input_digest)
    assert {c: ev.evidence_id for c, ev in current.items()} == {
        p.claim.claim_id: p.evidence.evidence_id for p in out.plans
    }
    assert all(ev.content["deviations"] for ev in current.values())  # the replanned mapping


def test_changed_inputs_change_the_digest(tmp_path: Path) -> None:
    first = run_orchestrator(_make_store(tmp_path / "a"), _fake(_mapping()))
    other = run_orchestrator(
        _make_store(tmp_path / "b", TRAIN.replace("gamma=0.2", "gamma=0.1")), _fake(_mapping())
    )
    assert first.input_digest != other.input_digest  # train config evidence differs


TRAIN_AND_TEST = TRAIN.replace(
    "    tqdm.write('test_acc",
    "    tqdm.write('train_acc: %.3f' % (accuracy))\n    tqdm.write('test_acc",
)


@pytest.mark.parametrize("name", ["test_acc", "train_acc"])
def test_metric_line_is_the_quoted_one(tmp_path: Path, name: str) -> None:
    store = _make_store(tmp_path, TRAIN_AND_TEST)
    lines = [ln.strip() for ln in TRAIN_AND_TEST.splitlines() if "tqdm.write" in ln]
    assert len(lines) == 2  # both lines are verified metric lines, both allow the conversion
    line = next(ln for ln in lines if name in ln)
    out = run_orchestrator(store, _fake(_mapping(metric_line=line)))
    assert out.ok, out.errors
    for p in out.plans:
        assert p.plan.metric_source.pattern.startswith(rf"(?<!\w){name}:")


def test_inconsistent_setup_check_becomes_deviation(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    out = run_orchestrator(store, _fake(_mapping(lr_schedule_consistent=False)))
    assert out.ok
    for p in out.plans:
        (dev,) = p.deviations
        assert dev.field == "lr_schedule"
        assert dev.stated == "divide by 5 after epochs 60, 120, 160"
        assert dev.chosen == "milestones=[60, 120, 160]" and dev.reason == "gamma differs"
        status = {n.field: n.status for n in p.notes}
        assert status["lr_schedule"] == "deviation" and status["weight_decay"] == "consistent"
        assert status["num_runs"] == "open"
        assert p.fidelity == "inferred"
        assert p.evidence.content["deviations"][0]["field"] == "lr_schedule"


def test_resolved_unknown_gets_no_assumption(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    human = make_evidence(
        kind="human_input",
        source="user",
        locator="seed",
        content={"seed": "0"},
        confidence="observed",
        authority="human",
    )
    store.append_evidence(human)
    resolved = UnknownResolved(unknown_id="unk:seed", value="0", evidence_id=human.evidence_id)
    store.append_event(resolved)
    out = run_orchestrator(store, _fake(_mapping()))
    assert out.ok
    assert {e.unknown_id for e in out.events_written} == {"unk:eval_checkpoint"}
    for p in out.plans:  # the resolved value is used (= the repo default): nothing assumed
        assert p.plan.depends_on_unknowns == ("unk:eval_checkpoint",)
        seed = next(n for n in p.notes if n.field == "seed")
        assert seed.status == "consistent" and "repo default 0" in seed.detail
        assert "(human: human_input " in seed.detail
    fold_unknowns(store.load_events(), store.load_evidence())


def _answer(store: RunStore, name: str, value: str) -> str:
    from reproducer.domain.approval import answer_unknown

    ans = answer_unknown(store.load_events(), store.load_evidence(), name, value, "authors")
    assert ans.event is not None
    store.append_evidence(ans.evidence)
    store.append_event(ans.event)
    return ans.evidence.evidence_id


def _no_llm(*a: Any) -> Any:
    raise AssertionError("only a value changed: the stored mapping is reused, no LLM call")


def test_human_answer_replans_without_llm_call(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = run_orchestrator(store, _fake(_mapping()))
    assert first.ok and first.plan_digest == first.input_digest  # nothing resolved yet
    old = {p.claim.claim_id: p.evidence.evidence_id for p in first.plans}

    hid = _answer(store, "seed", "1")
    again = run_orchestrator(store, llm_factory=_no_llm)
    assert again.ok and again.reused and again.attempts == 0
    assert again.input_digest == first.input_digest  # the mapping digest is unchanged
    assert again.plan_digest != first.plan_digest  # the answer is part of the plan digest
    for p in again.plans:
        assert p.evidence.evidence_id != old[p.claim.claim_id]
        assert p.evidence.content["input_digest"] == again.plan_digest
        assert "--seed" in p.argv and p.argv[p.argv.index("--seed") + 1] == "1"
        (arg,) = [a for a in p.plan.args if a.flag == "--seed"]
        assert arg.value == "1" and arg.evidence_id == hid  # provenance: the human answer
        assert arg.source == "human"
        assert "unk:seed" not in p.plan.depends_on_unknowns
        assert all(a.unknown_id != "unk:seed" for a in p.assumptions)
        seed = next(n for n in p.notes if n.field == "seed")
        assert seed.status == "override"
        assert seed.detail == f"--seed 1 (human: human_input {hid[:12]}; repo default 0)"
    assert not [e for e in again.events_written if e.unknown_id == "unk:seed"]

    current = load_current_plans(store)  # the answered plans are the current ones
    assert current.input_digest == again.plan_digest
    assert {p.evidence.evidence_id for p in current.plans} == {
        p.evidence.evidence_id for p in again.plans
    }
    n = len(_lines(store))
    run_orchestrator(store, llm_factory=_no_llm)  # idempotent
    assert len(_lines(store)) == n


def test_mapping_records_its_gate_attempts(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    bad = copy.deepcopy(_mapping())
    bad["claims"][0]["claim_args"][0]["value"] = "imagenet"
    out = run_orchestrator(store, FakeLLM([json.dumps(bad), json.dumps(_mapping())]))
    assert out.ok and out.attempts == 2 and out.mapping_evidence is not None
    assert out.mapping_evidence.content["gate_attempts"] == 2


def test_rejected_mapping_writes_no_plan_but_tolerance(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    bad = copy.deepcopy(_mapping())
    bad["claims"][0]["claim_args"][0]["value"] = "imagenet"
    out = run_orchestrator(store, _fake(bad), max_retries=1)
    assert not out.ok and out.attempts == 2
    assert "not a choice" in out.errors[0]
    kinds = {r["kind"] for r in _lines(store)}
    assert "tolerance" in kinds and not kinds & {"claim_mapping", "run_plan"}
    assert not [e for e in store.load_events() if isinstance(e, Assumption)]


# --- live ----------------------------------------------------------------------------------

E2E = ROOT / "runs" / "cutout_e2e"


@pytest.mark.llm_live
def test_orchestrator_live_cutout(tmp_path: Path) -> None:
    if not (E2E / "evidence.jsonl").is_file():
        pytest.skip(f"{E2E} not configured")
    if not os.environ.get("REPRODUCER_LLM_ORCHESTRATOR") and not os.environ.get("REPRODUCER_LLM"):
        pytest.skip("no orchestrator LLM configured (source env.local.sh)")
    for name in ("evidence.jsonl", "unknown_events.jsonl"):
        shutil.copy(E2E / name, tmp_path / name)  # never write to the real run folder here
    store = RunStore(tmp_path)
    llm = llm_for_node(ORCHESTRATOR_NODE, Budget(max_tokens_total=60_000))
    start = time.monotonic()
    out = run_orchestrator(store, llm, repo_dir=E2E / "repo")
    wall = time.monotonic() - start
    records = llm.records()
    print(f"\nattempts={out.attempts} calls={len(records)} wall={wall:.1f}s")
    for r in records:
        print(f"  {r.model}: in={r.input_tokens} out={r.output_tokens} latency={r.latency_s:.1f}s")
    for e in out.errors:
        print(f"  rejected: {e.splitlines()[0]}")
    assert out.ok, out.errors
    assert out.mapping is not None
    print(json.dumps(out.mapping.model_dump(mode="json"), indent=1)[:4000])
    for p in out.plans:
        print(f"{p.claim.claim_id}: {p.command}")
        for n in p.notes:
            print(f"  {n.status:<10} {n.field}: {n.detail[:120]}")
    cmds = {p.claim.claim_id: p.argv for p in out.plans}
    assert set(cmds) == {C10, C100}
    for cid, (ds, length) in {C10: ("cifar10", "16"), C100: ("cifar100", "8")}.items():
        argv = list(cmds[cid])
        assert argv[argv.index("--dataset") + 1] == ds
        assert argv[argv.index("--length") + 1] == length
    for cm in out.mapping.claims:
        assert cm.metric.conversion == "accuracy_fraction_to_error_percent"
        assert "/" in cm.metric.scale_citation.quote
    tc_id = out.train_config.evidence_id  # type: ignore[union-attr]
    cited = {s.setup_name for s in out.mapping.setup_checks if s.repo_citation.evidence_id == tc_id}
    assert {"weight_decay", "lr_schedule"} <= cited
    assert any(n in cited for n in ("optimizer", "nesterov", "momentum"))
    kinds = [r["kind"] for r in _lines(store)]
    assert kinds.index("tolerance") < kinds.index("run_plan")


def test_cli_plan_prints_commands_and_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reproducer import cli
    from reproducer.providers.llm import MeteredLLM

    _make_store(tmp_path)
    fake = _fake(_mapping(lr_schedule_consistent=False))
    monkeypatch.setattr(cli, "llm_for_node", lambda node, budget: MeteredLLM(fake, budget, node))
    assert cli.main(["plan", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for cmd in EXPECTED.values():
        assert f"command:      {cmd}" in out
    assert "--length 8  [repo]" in out and "--dataset cifar100  [repo]" in out
    assert f"base command: {EXPECTED[C100]}" in out
    assert "depends on:   unk:eval_checkpoint, unk:seed" in out
    assert "conversion=accuracy_fraction_to_error_percent" in out
    assert "tolerance:    abs=0.26" in out  # 2 x the paper's ± 0.13
    assert "INCONSISTENT lr_schedule" in out and "num_runs: no flag" in out
    assert "assumptions:    4 event(s) written" in out
    assert len(RunStore(tmp_path).load_call_records()) == 1
    assert cli.main(["plan", "--out", str(tmp_path)]) == 0  # same inputs: reused
    assert "reused stored mapping (no LLM call)" in capsys.readouterr().out
    assert len(RunStore(tmp_path).load_call_records()) == 1
    assert cli.main(["plan", "--out", str(tmp_path), "--replan"]) == 0
    assert len(RunStore(tmp_path).load_call_records()) == 2


def test_llm_factory_is_called_only_when_a_call_is_needed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    made: list[int] = []

    def factory() -> FakeLLM:
        made.append(1)
        return _fake(_mapping())

    first = run_orchestrator(store, llm_factory=factory)
    assert first.ok and not first.reused and made == [1]

    def no_config() -> FakeLLM:
        raise AssertionError("the stored mapping is reused: no LLM may be made")

    again = run_orchestrator(store, llm_factory=no_config)
    assert again.ok and again.reused and made == [1]


def test_claim_that_does_not_plan_is_reported_and_the_other_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reproducer.graph.nodes.orchestrator as orch
    from reproducer.graph.nodes.pipeline import stored_plan_failures

    real = orch.build_run_plan

    def flaky(claim: Any, *a: Any, **kw: Any) -> Any:
        if claim.claim_id == C100:
            raise ValueError("metric line not found")
        return real(claim, *a, **kw)

    monkeypatch.setattr(orch, "build_run_plan", flaky)
    store = _make_store(tmp_path)
    out = run_orchestrator(store, _fake(_mapping()))
    assert [p.claim.claim_id for p in out.plans] == [C10] and not out.ok
    assert f"{C100}: metric line not found" in out.errors
    cur = load_current_plans(store, partial=True)
    assert [p.claim.claim_id for p in cur.plans] == [C10]
    assert dict(cur.failed) == {C100: "metric line not found"}
    with pytest.raises(ValueError, match="no longer plans"):
        load_current_plans(store)  # strict (the `run` command): every claim must plan
    assert stored_plan_failures(store, tmp_path) == {C100: "metric line not found"}


def test_resolved_non_setup_unknown_keeps_plans_current(tmp_path: Path) -> None:
    """An unknown that is no paper setup field (e.g. dataset_source, resolved by the data
    node) never changes the plan digest: stored plans stay current."""
    from reproducer.domain.unknowns import UnknownOpened

    store = _make_store(tmp_path)
    first = run_orchestrator(store, _fake(_mapping()))
    src = make_evidence(kind="dataset_cache", source="data", locator="dataset_source",
                        content={"value": "torchvision"}, confidence="observed",
                        authority="execution")  # fmt: skip
    store.append_evidence(src)
    store.append_event(UnknownOpened(unknown_id="unk:dataset_source", name="dataset_source",
                                     evidence_id=src.evidence_id))  # fmt: skip
    store.append_event(UnknownResolved(unknown_id="unk:dataset_source", value="torchvision",
                                       evidence_id=src.evidence_id))  # fmt: skip
    current = load_current_plans(store)
    assert current.input_digest == first.plan_digest
