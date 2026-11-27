"""ClaimMapping (orchestrator LLM output) validation and its RunPlans (Cutout fixture)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from conftest import C10, C100, C100_COMMAND, COMMAND, CUTOUT_ARGS, METRIC_LINE, SOURCE, by_locator
from pydantic import ValidationError

from reproducer.domain.contracts import ClaimMapping, ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.run_plan import ClaimMapping as PlanMapping
from reproducer.domain.run_plan import build_run_plan
from reproducer.domain.train_config import train_config_evidence
from reproducer.graph.nodes.orchestrator import (
    PlanTarget,
    build_orchestrator_context,
    pick_base_command,
    to_plan_mapping,
)

TRAIN_CODE = """\
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

tqdm.write('test_acc: %.3f' % (test_acc))
"""
HYPER = [
    {"setup_name": "epochs", "flag": "--epochs"},
    {"setup_name": "batch_size", "flag": "--batch_size"},
    {"setup_name": "learning_rate", "flag": "--learning_rate"},
]
METRIC = {
    "metric_line_ref": "E2",
    "metric_line_quote": {"evidence_id": "E2", "quote": METRIC_LINE},
    "conversion": "accuracy_fraction_to_error_percent",
    "scale_citation": {"evidence_id": "E1", "quote": "val_acc = correct / total"},
}
VALID: dict[str, Any] = {
    "claims": [
        {
            "claim_id": C10,
            "claim_args": [
                {"flag": "--dataset", "value": "cifar10"},
                {"flag": "--model", "value": "resnet18"},
            ],
            "setup_mappings": [{"setup_name": "cutout_length_cifar10", "flag": "--length"}, *HYPER],
            "metric": METRIC,
        },
        {
            "claim_id": C100,
            "claim_args": [
                {"flag": "--dataset", "value": "cifar100"},
                {"flag": "--model", "value": "resnet18"},
            ],
            "setup_mappings": [
                {"setup_name": "cutout_length_cifar100", "flag": "--length"},
                *HYPER,
            ],
            "metric": METRIC,
        },
    ],
    "unmapped_setup": [{"setup_name": "num_runs", "reason": "no flag: repeat the run"}],
    "setup_checks": [
        {
            "setup_name": "optimizer",
            "repo_citation": {"evidence_id": "E3", "quote": "nesterov=True"},
            "consistent": True,
            "note": "SGD with nesterov=True",
        },
        {
            "setup_name": "weight_decay",
            "repo_citation": {"evidence_id": "E3", "quote": "weight_decay=5e-4"},
            "consistent": True,
            "note": "5e-4 in the optimizer call",
        },
        {
            "setup_name": "lr_schedule",
            "repo_citation": {"evidence_id": "E3", "quote": "milestones=[60, 120, 160], gamma=0.2"},
            "consistent": True,
            "note": "x0.2 = divide by 5 at 60/120/160",
        },
    ],
}


@pytest.fixture
def evidence(cutout_evidence: dict[str, Evidence]) -> dict[str, Evidence]:
    """The S1a Cutout evidence with real code text for train.py and its '#train_config'."""
    out = {i: ev for i, ev in cutout_evidence.items() if ev.locator != "train.py"}
    code = make_evidence(
        kind="repo_file",
        source=SOURCE,
        locator="train.py",
        content={"path": "train.py", "head": TRAIN_CODE},
        confidence="observed",
        authority="repo",
    )
    tc = train_config_evidence("train.py", {"train.py": TRAIN_CODE}, source=SOURCE)
    return {**out, code.evidence_id: code, tc.evidence_id: tc}


@pytest.fixture
def target(cutout_evidence: dict[str, Evidence]) -> PlanTarget:
    return PlanTarget(
        source=SOURCE,
        entrypoint="train.py",
        argparse_spec=CUTOUT_ARGS,
        commands=tuple(
            (by_locator(cutout_evidence, "README.md#commands"), c) for c in (COMMAND, C100_COMMAND)
        ),
        repo_files=frozenset({"train.py", "README.md"}),
    )


@pytest.fixture
def ctx(evidence: dict[str, Evidence], target: PlanTarget) -> dict[str, Any]:
    tc = evidence[by_locator(evidence, "train.py#train_config")]
    context, prompt = build_orchestrator_context(evidence, target, train_config=tc)
    assert "[E3]" in prompt and "nesterov=True" in prompt and "--dataset/-d" in prompt
    assert "accuracy_fraction_to_error_percent" in prompt  # code-built allowed conversions
    assert "cutout_length_cifar10 = 16" in prompt and "seed = UNKNOWN" in prompt
    return context


def _validate(data: dict[str, Any], ctx: dict[str, Any]) -> ClaimMapping:
    return ClaimMapping.model_validate_json(json.dumps(data), context=ctx)


def test_valid_mapping_gives_the_s1a_plans_on_the_matching_command(
    ctx: dict[str, Any],
    evidence: dict[str, Evidence],
    target: PlanTarget,
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, PlanMapping],
    paper_setup: list[Evidence],
) -> None:
    mapping = _validate(VALID, ctx)
    metric_id = by_locator(evidence, "train.py#metric_lines")
    assert mapping.claims[0].metric.metric_line_ref == metric_id  # alias resolved
    assert mapping.claims[0].metric.scale_citation.evidence_id == by_locator(evidence, "train.py")
    assert mapping.setup_checks[2].repo_citation.evidence_id == by_locator(
        evidence, "train.py#train_config"
    )
    for cm in mapping.claims:  # c100 last: `got` is its plan after the loop
        cid = cm.claim_id
        want_mapping = cutout_mappings[cid]
        if cid == C100:  # code now runs the verified cifar100 command (shares both claim_args)
            want_mapping = want_mapping.model_copy(update={"base_command": C100_COMMAND})
        got = build_run_plan(
            cutout_claims[cid],
            to_plan_mapping(cm, target, ctx),
            evidence,
            CUTOUT_ARGS,
            paper_setup,
            repo_files=target.repo_files,
        )
        want = build_run_plan(
            cutout_claims[cid],
            want_mapping,
            evidence,
            CUTOUT_ARGS,
            paper_setup,
            repo_files=target.repo_files,
        )
        assert got[0] == want[0]  # the RunPlan
        assert got[1] == want[1]  # assumptions (seed, eval_checkpoint)
    c100 = mapping.claims[1]
    assert pick_base_command(c100, target)[1] == C100_COMMAND
    assert {a.source for a in got[0].args} == {"repo"}  # the c100 command already matches


def test_base_command_tie_takes_the_first(ctx: dict[str, Any], target: PlanTarget) -> None:
    data = copy.deepcopy(VALID)
    data["claims"][1]["claim_args"] = [{"flag": "--model", "value": "resnet18"}]  # both share 1
    cm = _validate(data, ctx).claims[1]
    assert pick_base_command(cm, target)[1] == COMMAND


def _mutate(path: list[Any], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    node: Any = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


BAD_CASES = [
    ("missing claim", _mutate(["claims"], VALID["claims"][:1]), "not mapped"),
    ("duplicate claim", _mutate(["claims"], [VALID["claims"][0]] * 2), "more than once"),
    (
        "value not in choices",
        _mutate(["claims", 0, "claim_args", 0, "value"], "imagenet"),
        "not a choice",
    ),
    (
        "wideresnet not in the claim text",
        _mutate(["claims", 1, "claim_args", 1, "value"], "wideresnet"),
        "'wideresnet' is not named by the claim",
    ),
    (
        "cifar10 is not cifar100",
        _mutate(["claims", 1, "claim_args", 0, "value"], "cifar10"),
        "'cifar10' is not named by the claim",
    ),
    (
        "flag without choices",
        _mutate(["claims", 0, "claim_args", 1], {"flag": "--epochs", "value": "200"}),
        "has no argparse choices",
    ),
    (
        "same argument twice",
        _mutate(["claims", 0, "claim_args", 1], {"flag": "-d", "value": "cifar10"}),
        "set more than once",
    ),
    ("no claim_args", _mutate(["claims", 0, "claim_args"], []), "at least 1 item"),
    (
        "metric line quote not in metric lines",
        _mutate(
            ["claims", 0, "metric", "metric_line_quote"],
            {"evidence_id": "E1", "quote": "val_acc = correct / total"},
        ),
        "must cite the metric_line_ref evidence",
    ),
    (
        "unknown setup name",
        _mutate(["claims", 0, "setup_mappings", 1, "setup_name"], "warmup"),
        "not a paper setup field",
    ),
    ("unknown flag", _mutate(["claims", 0, "setup_mappings", 1, "flag"], "--lr"), "not an arg"),
    (
        "paper value does not fit the flag",
        _mutate(["claims", 0, "setup_mappings", 1, "setup_name"], "optimizer"),
        "not a valid int",
    ),
    (
        "disallowed conversion",
        _mutate(["claims", 0, "metric", "conversion"], "identity"),
        "conversion 'identity' is not allowed",
    ),
    (
        "fabricated scale citation",
        _mutate(
            ["claims", 0, "metric", "scale_citation"],
            {"evidence_id": "E1", "quote": "acc = correct.float().mean() * 100"},
        ),
        "quote not found",
    ),
    (
        "scale citation of the wrong scale",
        _mutate(["claims", 0, "metric", "scale_citation"], {"evidence_id": "E1", "quote": "0."}),
        "needs a fraction",
    ),
    (
        "scale citation not code",
        _mutate(
            ["claims", 0, "metric", "scale_citation"],
            {"evidence_id": "E2", "quote": "tqdm.write('test_acc: %.3f' % (test_acc))"},
        ),
        "must quote the script's code",
    ),
    (
        "metric ref not metric lines",
        _mutate(["claims", 0, "metric", "metric_line_ref"], "E3"),
        "verified metric lines",
    ),
    (
        "setup check of an unknown",
        _mutate(["setup_checks", 0, "setup_name"], "seed"),
        "not stated by the paper",
    ),
    (
        "mapped and unmapped",
        _mutate(["unmapped_setup"], [{"setup_name": "epochs", "reason": "x"}]),
        "both mapped to a flag and unmapped",
    ),
]


@pytest.mark.parametrize(("data", "msg"), [c[1:] for c in BAD_CASES], ids=[c[0] for c in BAD_CASES])
def test_bad_mapping_rejected(data: dict[str, Any], msg: str, ctx: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=msg):
        _validate(data, ctx)


def test_missing_context_fails_closed(ctx: dict[str, Any]) -> None:
    partial = {k: v for k, v in ctx.items() if k != "metric_conversions"}
    with pytest.raises(ValidationError, match="requires validation context"):
        _validate(VALID, partial)
