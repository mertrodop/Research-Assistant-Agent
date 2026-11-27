"""Tests for reproducer.domain.contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import C100, C100_COMMAND, COMMAND, CUTOUT_ARGS, METRIC_LINE, by_locator
from pydantic import ValidationError

from reproducer.domain.contracts import (
    ArgOption,
    ArgSetting,
    ExperimentResult,
    Lesson,
    ReproSpec,
    ResultClaim,
    RunPlan,
    make_run_plan,
)
from reproducer.domain.evidence import Evidence
from reproducer.domain.run_plan import allowed_conversions, metric_pattern_from_line
from reproducer.domain.unknowns import UnknownState

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "repro_spec_example.json"


@pytest.fixture
def spec_data() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def spec() -> ReproSpec:
    return ReproSpec.model_validate_json(FIXTURE.read_text())


# --- ReproSpec -----------------------------------------------------------------------------


def test_fixture_loads(spec: ReproSpec) -> None:
    assert len(spec.claims) == 2
    assert len(spec.setup) == 3


def test_round_trip(spec: ReproSpec) -> None:
    assert ReproSpec.model_validate_json(spec.model_dump_json()) == spec


def test_duplicate_claim_id_rejected(spec_data: dict[str, Any]) -> None:
    spec_data["claims"][1]["claim_id"] = "c1"
    with pytest.raises(ValidationError, match="duplicate claim_id"):
        ReproSpec.model_validate(spec_data)


def test_duplicate_setup_name_rejected(spec_data: dict[str, Any]) -> None:
    spec_data["setup"][1]["name"] = "optimizer"
    with pytest.raises(ValidationError, match=r"duplicate setup name.*optimizer"):
        ReproSpec.model_validate(spec_data)


def test_extra_field_rejected(spec_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="extra"):
        ReproSpec.model_validate({**spec_data, "bogus": 1})


def test_extra_field_in_nested_claim_rejected(spec_data: dict[str, Any]) -> None:
    spec_data["claims"][0]["bogus"] = 1
    with pytest.raises(ValidationError, match="extra"):
        ReproSpec.model_validate(spec_data)


@pytest.mark.parametrize("field", ["code_links", "claims"])
def test_empty_required_sequences_rejected(spec_data: dict[str, Any], field: str) -> None:
    spec_data[field] = []
    with pytest.raises(ValidationError):
        ReproSpec.model_validate(spec_data)


def test_frozen_mutation_rejected(spec: ReproSpec) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        spec.claims[0].value = 99.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        spec.datasets = ()  # type: ignore[misc]
    assert isinstance(spec.claims, tuple)


def test_is_unknown(spec: ReproSpec) -> None:
    by_name = {s.name: s for s in spec.setup}
    assert by_name["weight_decay"].is_unknown is True
    assert by_name["optimizer"].is_unknown is False


# --- RunPlan (Cutout fixture, see conftest.py) ---------------------------------------------


@pytest.fixture
def ctx(
    cutout_claims: dict[str, ResultClaim], cutout_evidence: dict[str, Evidence]
) -> dict[str, Any]:
    metric_id = by_locator(cutout_evidence, "train.py#metric_lines")
    return {
        "repo_files": {"train.py", "README.md", "configs/base.yaml"},
        "config_keys": {"optim.lr"},
        "claims": cutout_claims,
        "evidence_ids": set(cutout_evidence),
        "argparse_spec": CUTOUT_ARGS,
        "command_evidence": {
            "train.py": {by_locator(cutout_evidence, "README.md#commands"): {COMMAND, C100_COMMAND}}
        },
        "metric_sources": {
            metric_id: {
                metric_pattern_from_line(METRIC_LINE): allowed_conversions(
                    "test_acc", "test_error", "%"
                )
            }
        },
    }


@pytest.fixture
def plan_data(cutout_evidence: dict[str, Evidence]) -> dict[str, Any]:
    cmd = by_locator(cutout_evidence, "README.md#commands")
    claim = next(
        i
        for i, ev in cutout_evidence.items()
        if ev.kind == "paper_claim" and ev.content["claim_id"] == C100
    )
    return {
        "claim_id": C100,
        "entrypoint": "train.py",
        "base_command_evidence_id": cmd,
        "base_command": COMMAND,
        "depends_on_unknowns": ["unk:seed"],
        "args": [
            {"flag": "--dataset", "value": "cifar100", "source": "claim", "evidence_id": claim},
            {"flag": "--cutout", "value": None, "source": "repo", "evidence_id": cmd},
            {"flag": "--length", "value": "8", "source": "repo", "evidence_id": cmd},
        ],
        "dataset": "CIFAR-100",
        "metric": "test_error",
        "metric_unit": "%",
        "metric_source": {
            "metric_line_evidence_id": by_locator(cutout_evidence, "train.py#metric_lines"),
            "pattern": metric_pattern_from_line(METRIC_LINE),
            "which": "last",
            "conversion": "accuracy_fraction_to_error_percent",
        },
    }


def test_runplan_valid_with_context(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    plan = RunPlan.model_validate(plan_data, context=ctx)
    assert plan.args[0].value == "cifar100" and plan.args[1].value is None
    assert plan.config is None and plan.config_overrides == ()
    assert RunPlan.model_validate_json(plan.model_dump_json(), context=ctx) == plan


def test_make_run_plan(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    plan = make_run_plan(plan_data, **ctx)
    assert plan == RunPlan.model_validate(plan_data, context=ctx)
    with pytest.raises(ValidationError, match="file not in repo"):
        make_run_plan({**plan_data, "entrypoint": "missing.py"}, **ctx)


def test_human_source_iff_human_input_evidence(
    plan_data: dict[str, Any], ctx: dict[str, Any]
) -> None:
    hid = "h" * 64
    ctx = {**ctx, "evidence_ids": {*ctx["evidence_ids"], hid}}
    seed = {"flag": "--seed", "value": "1", "source": "human", "evidence_id": hid}
    data = {**plan_data, "depends_on_unknowns": [], "args": [*plan_data["args"], seed]}
    plan = make_run_plan(data, **ctx, human_input_ids={hid})
    assert plan.args[-1].source == "human" and plan.is_spec_faithful({})
    with pytest.raises(ValidationError, match="source human needs a human_input evidence_id"):
        make_run_plan(data, **ctx)  # not a human_input id
    as_paper = {**data, "args": [*plan_data["args"], {**seed, "source": "paper"}]}
    with pytest.raises(ValidationError, match="a human_input value must have source human"):
        make_run_plan(as_paper, **ctx, human_input_ids={hid})


def test_runplan_model_copy_raises(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    plan = make_run_plan(plan_data, **ctx)
    with pytest.raises(TypeError, match="make_run_plan"):
        plan.model_copy()
    with pytest.raises(TypeError, match="make_run_plan"):
        plan.model_copy(update={"metric": "x"})


def test_runplan_fails_without_context(plan_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="requires validation context"):
        RunPlan.model_validate(plan_data)
    with pytest.raises(ValidationError, match="requires validation context"):
        RunPlan(**plan_data)


@pytest.mark.parametrize(
    "missing", ["config_keys", "argparse_spec", "metric_sources", "command_evidence"]
)
def test_runplan_fails_with_incomplete_context(
    plan_data: dict[str, Any], ctx: dict[str, Any], missing: str
) -> None:
    del ctx[missing]
    with pytest.raises(ValidationError, match="requires validation context"):
        RunPlan.model_validate(plan_data, context=ctx)


@pytest.mark.parametrize(
    ("arg", "msg"),
    [
        ({"flag": "--bogus", "value": "1"}, "not an argparse option"),
        ({"flag": "--dataset", "value": "imagenet"}, "not a choice"),
        ({"flag": "--epochs", "value": "2.5"}, "not a valid int"),
        ({"flag": "--learning_rate", "value": "fast"}, "not a valid float"),
        ({"flag": "--cutout", "value": "true"}, "value must be None"),
        ({"flag": "--length", "value": None}, "takes a value"),
    ],
)
def test_argsetting_validation(
    plan_data: dict[str, Any], ctx: dict[str, Any], arg: dict[str, Any], msg: str
) -> None:
    ev = plan_data["base_command_evidence_id"]
    with pytest.raises(ValidationError, match=msg):
        ArgSetting.model_validate({**arg, "source": "repo", "evidence_id": ev}, context=ctx)
    plan_data["args"] = [*plan_data["args"][1:], {**arg, "source": "repo", "evidence_id": ev}]
    with pytest.raises(ValidationError, match=msg):
        RunPlan.model_validate(plan_data, context=ctx)


def test_argsetting_needs_context_and_checkable_type() -> None:
    data = {"flag": "--epochs", "value": "1", "source": "repo", "evidence_id": "x"}
    with pytest.raises(ValidationError, match="requires validation context"):
        ArgSetting.model_validate(data)
    odd = (ArgOption(option_strings=("--epochs",), dest="epochs", type="str2int"),)
    with pytest.raises(ValidationError, match="cannot be checked"):
        ArgSetting.model_validate(data, context={"argparse_spec": odd, "evidence_ids": {"x"}})


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"config": "configs/missing.yaml"}, "file not in repo"),
        (
            {
                "config": "configs/base.yaml",
                "config_overrides": [{"key": "dropout", "value": 0.1, "source": "paper"}],
            },
            "override key",
        ),
        (
            {"config_overrides": [{"key": "optim.lr", "value": 0.1, "source": "paper"}]},
            "need a config",
        ),
        ({"metric": "top5_accuracy"}, "metric"),
        ({"metric_unit": None}, "metric_unit"),
        ({"claim_id": "c99"}, "unknown claim_id"),
        ({"base_command_evidence_id": "nope"}, "unknown evidence id"),
        ({"base_command": "python train.py --cutout"}, "not a command line of the cited"),
        ({"depends_on_unknowns": ["seed"]}, "depends_on_unknowns must be unique"),
        ({"depends_on_unknowns": ["unk:seed", "unk:seed"]}, "depends_on_unknowns must be unique"),
    ],
)
def test_runplan_rejections(
    plan_data: dict[str, Any], ctx: dict[str, Any], patch: dict[str, Any], msg: str
) -> None:
    with pytest.raises(ValidationError, match=msg):
        RunPlan.model_validate({**plan_data, **patch}, context=ctx)


def test_runplan_rejects_unverified_command_and_duplicate_dest(
    plan_data: dict[str, Any], ctx: dict[str, Any], cutout_evidence: dict[str, Evidence]
) -> None:
    readme = by_locator(cutout_evidence, "README.md")
    with pytest.raises(ValidationError, match="not a verified command"):
        RunPlan.model_validate({**plan_data, "base_command_evidence_id": readme}, context=ctx)
    dup = {"flag": "-d", "value": "svhn", "source": "repo", "evidence_id": readme}
    with pytest.raises(ValidationError, match=r"set more than once: \['dataset'\]"):
        RunPlan.model_validate({**plan_data, "args": [*plan_data["args"], dup]}, context=ctx)


def test_metric_source_must_be_code_built(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    src = plan_data["metric_source"]
    with pytest.raises(ValidationError, match="pattern must be the one code built"):
        RunPlan.model_validate(
            {**plan_data, "metric_source": {**src, "pattern": r"acc=(?P<value>\S+)"}}, context=ctx
        )
    with pytest.raises(ValidationError, match="conversion 'identity' not allowed"):
        RunPlan.model_validate(
            {**plan_data, "metric_source": {**src, "conversion": "identity"}}, context=ctx
        )


def test_is_spec_faithful(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    open_seed = {"unk:seed": UnknownState(name="seed", claim_ids=(), status="open")}
    resolved = {
        "unk:seed": UnknownState(
            name="seed", claim_ids=(), status="resolved", value="0", resolved_by="ev"
        )
    }
    plan = RunPlan.model_validate(plan_data, context=ctx)
    assert plan.is_spec_faithful({}) is False  # depends on an unknown not known to be resolved
    assert plan.is_spec_faithful(open_seed) is False
    assert plan.is_spec_faithful(resolved) is True
    ev = plan_data["base_command_evidence_id"]
    plan_data["args"].append(
        {"flag": "--seed", "value": "0", "source": "assumption", "evidence_id": ev}
    )
    assert RunPlan.model_validate(plan_data, context=ctx).is_spec_faithful(resolved) is False


# --- Lesson / ExperimentResult -------------------------------------------------------------


_LESSON_CTX: dict[str, Any] = {
    "claim_id": "c1",
    "trial": 1,
    "previous_step": None,
    "subject_ids": frozenset({"ev1"}),
    "evidence_ids": frozenset({"ev1"}),
    "texts": {},
    "evidence_authority": {"ev1": "execution"},
    "argparse_spec": (),
    "claim_flags": frozenset(),
    "open_plan_unknowns": frozenset(),
    "eval_protocol_table": {},
    "claim_unit": None,
    "open_unknown_names": frozenset(),
}


def _lesson(**kw: Any) -> Lesson:
    base: dict[str, Any] = {
        "claim_id": "c1",
        "trial": 1,
        "step": "eval_protocol",
        "evidence_ids": ["ev1"],
        "text": "Eval used val split; paper reports test.",
    }
    return Lesson.model_validate({**base, **kw}, context=_LESSON_CTX)


def test_lesson_fails_closed_without_context() -> None:
    data = {"claim_id": "c1", "trial": 1, "step": "eval_protocol", "evidence_ids": ["ev1"],
            "text": "t"}  # fmt: skip
    with pytest.raises(ValidationError, match="requires validation context"):
        Lesson.model_validate(data)


def test_lesson_valid_at_limit() -> None:
    assert _lesson(text=" ".join(["w"] * 75)).trial == 1


def test_lesson_rejects_long_text() -> None:
    with pytest.raises(ValidationError, match="words"):
        _lesson(text=" ".join(["w"] * 76))


def test_lesson_rejects_empty_evidence() -> None:
    with pytest.raises(ValidationError):
        _lesson(evidence_ids=[])


def test_lesson_rejects_trial_zero() -> None:
    with pytest.raises(ValidationError):
        _lesson(trial=0)


@pytest.mark.parametrize("field", ["duration", "gpu_hours"])
def test_experiment_result_rejects_negative(field: str) -> None:
    data: dict[str, Any] = {
        "claim_id": "c1",
        "run_id": "r1",
        "exit_code": 0,
        "metrics": {"top1_accuracy": 95.1},
        "log_ref": "logs/r1.txt",
        "config_digest": "abc",
        "duration": 10.0,
        "gpu_hours": 0.5,
    }
    assert ExperimentResult.model_validate(data).duration == 10.0
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate({**data, field: -1})
