"""Tests for reproducer.domain.run_plan (pure: Cutout fixture, no LLM, no Docker)."""

from __future__ import annotations

import math
import re
import shlex

import pytest
from conftest import C10, C100, COMMAND, CUTOUT_ARGS, METRIC_LINE, SOURCE, by_locator
from pydantic import ValidationError

from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.run_plan import (
    ClaimArg,
    ClaimMapping,
    SetupArg,
    allowed_conversions,
    build_run_plan,
    command_flags,
    convert,
    extract_metric,
    metric_direction,
    metric_pattern_from_line,
    parse_metric_line,
    render_command,
)
from reproducer.domain.unknowns import UnknownState

SMOKE_LOG = (
    "Files already downloaded and verified\n"
    "Epoch 0:  100%|##########| 391/391 [00:21<00:00, 18.66it/s, acc=0.283, xentropy=2.013]\n"
    "test_acc: 0.405\n"
    "/opt/conda/lib/python3.7/site-packages/torch/optim/lr_scheduler.py:156: UserWarning\n"
)

# --- metric pattern ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("tqdm.write('test_acc: %.3f' % (test_acc))", "test_acc"),
        ("print('test_acc: {:.2f}'.format(acc))", "test_acc"),
        ("print(f'test_acc: {test_acc:.3f}')", "test_acc"),
        ('print(f"test_acc: {100 * correct / total:.2f}")', "test_acc"),
        ("print('test_acc:', acc)", "test_acc"),
    ],
)
def test_metric_pattern_from_line(line: str, name: str) -> None:
    mp = parse_metric_line(line)
    assert mp.name == name
    assert mp.pattern == metric_pattern_from_line(line)
    m = re.search(mp.pattern, "foo\ntest_acc: 0.405\n")
    assert m is not None and float(m["value"]) == 0.405
    assert re.search(mp.pattern, "pretest_acc: 0.9") is None  # anchored on a word start


def test_metric_pattern_skips_other_placeholders_with_index() -> None:
    line = "print('epoch %d test_acc %.2f%%' % (e, acc))"
    with pytest.raises(ValueError, match="2 placeholders"):
        metric_pattern_from_line(line)
    p = metric_pattern_from_line(line, index=1)
    assert extract_metric("epoch 3 test_acc 91.50%", p, "last", "max") == 91.5


@pytest.mark.parametrize(
    ("line", "msg"),
    [
        ("tqdm.write(test_acc)", "no string literal"),
        ("print('done ' + str(acc))", "no placeholder"),
        ("print('%.3f' % acc)", "no literal name"),
    ],
)
def test_metric_pattern_fails_clearly(line: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        metric_pattern_from_line(line)


def test_extract_metric_last_and_best() -> None:
    p = metric_pattern_from_line(METRIC_LINE)
    log = "test_acc: 0.300\ntest_acc: 0.700\ntest_acc: 0.650\n"
    assert extract_metric(log, p, "last", "max") == 0.65
    assert extract_metric(log, p, "best", "max") == 0.7
    assert extract_metric(log, p, "best", "min") == 0.3
    assert extract_metric("no metric here\nacc=0.283", p, "last", "max") is None
    nan = extract_metric("test_acc: nan\n", p, "best", "max")
    assert nan is not None and math.isnan(nan)


def test_extract_from_smoke_log_converts_to_error_percent() -> None:
    p = metric_pattern_from_line(METRIC_LINE)
    acc = extract_metric(SMOKE_LOG, p, "last", metric_direction("test_acc"))
    assert acc == 0.405  # not the tqdm "acc=0.283"
    assert convert(acc, "accuracy_fraction_to_error_percent") == pytest.approx(59.5)


# --- conversion ----------------------------------------------------------------------------


def test_conversions() -> None:
    assert convert(0.405, "accuracy_fraction_to_error_percent") == pytest.approx(59.5)
    assert convert(96.01, "accuracy_percent_to_error_percent") == pytest.approx(3.99)
    assert convert(0.0399, "error_fraction_to_percent") == pytest.approx(3.99)
    assert convert(0.96, "fraction_to_percent") == pytest.approx(96.0)
    assert convert(3.99, "identity") == 3.99
    with pytest.raises(ValueError, match="outside"):
        convert(40.5, "accuracy_fraction_to_error_percent")  # a percent, not a fraction
    assert math.isnan(convert(float("nan"), "accuracy_fraction_to_error_percent"))


def test_allowed_conversions_table() -> None:
    assert allowed_conversions("test_acc", "test_error", "%") == {
        "accuracy_fraction_to_error_percent",
        "accuracy_percent_to_error_percent",
    }
    assert allowed_conversions("test_acc", "top1_accuracy", "%") == {
        "identity",
        "fraction_to_percent",
    }
    assert "error_fraction_to_percent" in allowed_conversions("val_err", "test_error", "%")
    assert allowed_conversions("test_acc", "test_acc", None) == {"identity"}
    assert allowed_conversions("train_loss", "test_error", "%") == frozenset()
    assert allowed_conversions("test_acc", "test_error", None) == frozenset()


# --- command parsing -----------------------------------------------------------------------


def test_command_flags() -> None:
    assert command_flags(
        "python train.py -d=svhn --cutout --length 8", "train.py", CUTOUT_ARGS
    ) == [
        ("-d", "svhn"),
        ("--cutout", None),
        ("--length", "8"),
    ]
    with pytest.raises(ValueError, match="does not run"):
        command_flags("python other.py --cutout", "train.py", CUTOUT_ARGS)
    with pytest.raises(ValueError, match="not an argparse option"):
        command_flags("python train.py --bogus 1", "train.py", CUTOUT_ARGS)


# --- build_run_plan (Cutout) ---------------------------------------------------------------

EXPECTED = {
    C10: "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout "
    "--length 16",
    C100: "python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout "
    "--length 8",
}


def _build(
    claim: ResultClaim,
    mapping: ClaimMapping,
    evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
):  # noqa: ANN202
    return build_run_plan(
        claim, mapping, evidence, CUTOUT_ARGS, paper_setup, repo_files={"train.py", "README.md"}
    )


@pytest.mark.parametrize("cid", [C10, C100])
def test_build_run_plan_cutout(
    cid: str,
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    claim = cutout_claims[cid]
    plan, assumptions, notes = _build(claim, cutout_mappings[cid], cutout_evidence, paper_setup)
    assert shlex.join(render_command(plan, CUTOUT_ARGS)) == EXPECTED[cid]
    assert plan.metric == "test_error" and plan.metric_unit == "%"
    assert plan.dataset == claim.dataset
    assert plan.metric_source.which == "last"
    assert plan.metric_source.conversion == "accuracy_fraction_to_error_percent"
    assert plan.base_command == COMMAND
    assert plan.depends_on_unknowns == ("unk:eval_checkpoint", "unk:seed")
    opened = {
        u: UnknownState(name=u[4:], claim_ids=(), status="open") for u in plan.depends_on_unknowns
    }
    assert plan.is_spec_faithful(opened) is False  # seed + eval_checkpoint still open
    resolved = {
        u: s.model_copy(update={"status": "resolved", "value": "x", "resolved_by": "ev"})
        for u, s in opened.items()
    }
    assert plan.is_spec_faithful(resolved) is True

    status = {n.field: n.status for n in notes}
    for f in ("epochs", "batch_size", "learning_rate"):
        assert status[f] == "consistent"  # equal to the repo default: no override
    flags = {a.flag for a in plan.args}
    assert not flags & {"--epochs", "--batch_size", "--learning_rate", "--seed"}

    parsed_id = by_locator(cutout_evidence, "train.py#parsed")
    metric_id = by_locator(cutout_evidence, "train.py#metric_lines")
    got = {(a.unknown_id, a.value, a.evidence_id) for a in assumptions}
    assert got == {("unk:seed", "0", parsed_id), ("unk:eval_checkpoint", "last", metric_id)}
    assert status["seed"] == status["eval_checkpoint"] == "assumption"


def test_build_run_plan_sources(
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    c10, _, notes10 = _build(cutout_claims[C10], cutout_mappings[C10], cutout_evidence, paper_setup)
    assert {a.source for a in c10.args} == {"repo"}  # everything consistent with the command
    assert {n.field for n in notes10 if n.status == "consistent"} >= {
        "claim:--dataset",
        "cutout_length_cifar10",
    }

    c100, _, _ = _build(cutout_claims[C100], cutout_mappings[C100], cutout_evidence, paper_setup)
    args = {a.flag: a for a in c100.args}
    assert (args["--dataset"].value, args["--dataset"].source) == ("cifar100", "claim")
    claim_ev = cutout_evidence[args["--dataset"].evidence_id]
    assert claim_ev.kind == "paper_claim" and claim_ev.content["claim_id"] == C100
    assert (args["--length"].value, args["--length"].source) == ("8", "paper")
    length_ev = cutout_evidence[args["--length"].evidence_id]
    assert length_ev.content["name"] == "cutout_length_cifar100"  # value copied from here
    assert args["--model"].source == "repo"
    assert [a.flag for a in c100.args] == [  # replaced in place, order kept
        "--dataset",
        "--model",
        "--data_augmentation",
        "--cutout",
        "--length",
    ]


def test_paper_value_differing_from_default_is_appended(
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    mapping = cutout_mappings[C10].model_copy(
        update={"setup_args": (SetupArg(setup="num_runs", flag="--n_holes"),)}
    )
    plan, _, notes = _build(cutout_claims[C10], mapping, cutout_evidence, paper_setup)
    assert render_command(plan, CUTOUT_ARGS)[-2:] == ["--n_holes", "5"]
    assert any(n.field == "num_runs" and n.status == "override" for n in notes)


@pytest.mark.parametrize(
    ("update", "msg"),
    [
        ({"claim_args": (ClaimArg(flag="--dataset", value="imagenet"),)}, "not a choice"),
        ({"setup_args": (SetupArg(setup="optimizer", flag="--epochs"),)}, "not a valid int"),
        ({"setup_args": (SetupArg(setup="momentum", flag="--cutout"),)}, "takes no value"),
        ({"setup_args": (SetupArg(setup="nope", flag="--epochs"),)}, "not in the paper setup"),
        ({"base_command": "python train.py --cutout"}, "not in the cited evidence"),
        ({"metric_line": "print('x: %d' % x)"}, "metric line is not in the cited evidence"),
        ({"claim_id": C100}, "mapping is for"),
    ],
)
def test_build_run_plan_rejects_bad_mapping(
    update: dict[str, object],
    msg: str,
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    mapping = cutout_mappings[C10].model_copy(update=update)
    with pytest.raises(ValueError, match=msg):
        _build(cutout_claims[C10], mapping, cutout_evidence, paper_setup)


def test_build_run_plan_rejects_disallowed_conversion(
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    mapping = cutout_mappings[C10].model_copy(update={"conversion": "identity"})
    with pytest.raises(ValidationError, match="conversion 'identity' not allowed"):
        _build(cutout_claims[C10], mapping, cutout_evidence, paper_setup)


def test_render_keeps_interpreter_and_env_prefix(
    cutout_claims: dict[str, ResultClaim],
    cutout_mappings: dict[str, ClaimMapping],
    cutout_evidence: dict[str, Evidence],
    paper_setup: list[Evidence],
) -> None:
    line = "CUDA_VISIBLE_DEVICES=0 python3 -u train.py --dataset cifar10 --cutout --length 16"
    cmds = make_evidence(
        kind="repo_file",
        source=SOURCE,
        locator="run.sh#commands",
        content={
            "parser": "commands",
            "result": [{"line": line, "path": "train.py", "source": "run.sh"}],
        },
        confidence="derived",
        authority="repo",
    )
    evidence = {**cutout_evidence, cmds.evidence_id: cmds}
    mapping = cutout_mappings[C100].model_copy(
        update={"base_command_evidence_id": cmds.evidence_id, "base_command": line}
    )
    plan, _, _ = _build(cutout_claims[C100], mapping, evidence, paper_setup)
    assert plan.base_command == line
    assert shlex.join(render_command(plan, CUTOUT_ARGS)) == (
        "CUDA_VISIBLE_DEVICES=0 python3 -u train.py --dataset cifar100 --cutout --length 8"
    )
    # the line must belong to the cited evidence
    wrong = mapping.model_copy(
        update={"base_command_evidence_id": cutout_mappings[C100].base_command_evidence_id}
    )
    with pytest.raises(ValueError, match="not in the cited evidence"):
        _build(cutout_claims[C100], wrong, evidence, paper_setup)
