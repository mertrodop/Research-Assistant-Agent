"""Tests for reproducer.domain.contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reproducer.domain.contracts import (
    ExperimentResult,
    Lesson,
    ReproSpec,
    RunPlan,
    make_run_plan,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "repro_spec_example.json"


@pytest.fixture
def spec_data() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def spec() -> ReproSpec:
    return ReproSpec.model_validate_json(FIXTURE.read_text())


@pytest.fixture
def ctx(spec: ReproSpec) -> dict[str, Any]:
    return {
        "repo_files": {"train.py", "configs/resnet18.yaml"},
        "config_keys": {"lr", "batch_size", "epochs"},
        "claims": {c.claim_id: c for c in spec.claims},
    }


@pytest.fixture
def plan_data() -> dict[str, Any]:
    return {
        "claim_id": "c1",
        "entrypoint": "train.py",
        "config": "configs/resnet18.yaml",
        "overrides": [
            {"key": "epochs", "value": 200, "source": "paper"},
            {"key": "batch_size", "value": 128, "source": "repo"},
        ],
        "dataset": "CIFAR-10",
        "metric": "top1_accuracy",
    }


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


# --- RunPlan -------------------------------------------------------------------------------


def test_runplan_valid_with_context(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    plan = RunPlan.model_validate(plan_data, context=ctx)
    assert plan.overrides[0].value == 200
    assert RunPlan.model_validate_json(plan.model_dump_json(), context=ctx) == plan


def test_make_run_plan(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    plan = make_run_plan(plan_data, **ctx)
    assert plan == RunPlan.model_validate(plan_data, context=ctx)
    with pytest.raises(ValidationError, match="entrypoint not in repo"):
        make_run_plan({**plan_data, "entrypoint": "missing.py"}, **ctx)


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


def test_runplan_fails_with_incomplete_context(
    plan_data: dict[str, Any], ctx: dict[str, Any]
) -> None:
    del ctx["config_keys"]
    with pytest.raises(ValidationError, match="requires validation context"):
        RunPlan.model_validate(plan_data, context=ctx)


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"entrypoint": "missing.py"}, "entrypoint not in repo"),
        ({"config": "configs/missing.yaml"}, "config not in repo"),
        (
            {"overrides": [{"key": "dropout", "value": 0.1, "source": "paper"}]},
            "override key",
        ),
        ({"metric": "top5_accuracy"}, "metric"),
        ({"claim_id": "c99"}, "unknown claim_id"),
    ],
)
def test_runplan_rejections(
    plan_data: dict[str, Any], ctx: dict[str, Any], patch: dict[str, Any], msg: str
) -> None:
    with pytest.raises(ValidationError, match=msg):
        RunPlan.model_validate({**plan_data, **patch}, context=ctx)


def test_is_spec_faithful(plan_data: dict[str, Any], ctx: dict[str, Any]) -> None:
    assert RunPlan.model_validate(plan_data, context=ctx).is_spec_faithful is True
    plan_data["overrides"].append({"key": "lr", "value": 0.1, "source": "unknown"})
    assert RunPlan.model_validate(plan_data, context=ctx).is_spec_faithful is False


# --- Lesson / ExperimentResult -------------------------------------------------------------


def _lesson(**kw: Any) -> Lesson:
    base: dict[str, Any] = {
        "claim_id": "c1",
        "trial": 1,
        "step": "eval_protocol",
        "evidence_ids": ["ev1"],
        "text": "Eval used val split; paper reports test.",
    }
    return Lesson.model_validate({**base, **kw})


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
