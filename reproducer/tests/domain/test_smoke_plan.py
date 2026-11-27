"""Tests for reproducer.domain.smoke_plan and the additive smoke evidence fields."""

from __future__ import annotations

import pytest

from reproducer.domain.ledger import make_smoke_evidence
from reproducer.domain.smoke_plan import (
    BudgetArg,
    Override,
    decide_signal,
    estimate_timing,
    metric_prefixes,
    plan_smoke,
    progress_seen,
)
from reproducer.domain.traceback_parse import ParsedTraceback

CUTOUT_CMD = (
    "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
)
CUTOUT_ARGS = (
    BudgetArg(option_strings=("--dataset", "-d"), dest="dataset"),
    BudgetArg(option_strings=("--batch_size",), dest="batch_size"),
    BudgetArg(option_strings=("--epochs",), dest="epochs"),
    BudgetArg(option_strings=("--cutout",), dest="cutout", action="store_true"),
    BudgetArg(option_strings=("--length",), dest="length"),
)


def test_cutout_plan_appends_epochs_1() -> None:
    plan = plan_smoke(CUTOUT_CMD, CUTOUT_ARGS)
    assert plan.command == CUTOUT_CMD + " --epochs 1"
    assert plan.overrides == (Override(dest="epochs", option="--epochs", value="1"),)


def test_existing_budget_values_replaced_in_place() -> None:
    args = (
        BudgetArg(option_strings=("--epochs", "-e"), dest="epochs"),
        BudgetArg(option_strings=("--max_steps",), dest="max_steps"),
    )
    plan = plan_smoke("python train.py -e 160 --max_steps=5000 --lr 0.1", args)
    assert plan.argv == ("python", "train.py", "-e", "1", "--max_steps=20", "--lr", "0.1")
    assert [(o.option, o.value, o.previous) for o in plan.overrides] == [
        ("-e", "1", "160"),
        ("--max_steps", "20", "5000"),
    ]


def test_no_budget_arg_no_overrides_unknown_flags_untouched() -> None:
    cmd = "python main.py --weird-flag x --lr=0.1 'quoted arg' --epochs 3"
    plan = plan_smoke(cmd, (BudgetArg(option_strings=("--lr",), dest="lr"),))
    assert plan.overrides == ()
    assert plan.argv == (
        "python",
        "main.py",
        "--weird-flag",
        "x",
        "--lr=0.1",
        "quoted arg",
        "--epochs",
        "3",
    )


def test_positional_or_flag_budget_args_are_not_overridden() -> None:
    args = (
        BudgetArg(option_strings=("epochs",), dest="epochs"),
        BudgetArg(option_strings=("--iterations",), dest="iterations", action="store_true"),
    )
    assert plan_smoke("python t.py 10", args).overrides == ()


def test_plan_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plan_smoke("   ", ())


# --- progress / timing -------------------------------------------------------------------


TQDM = (
    "Namespace(epochs=1)\n"
    "  0%|          | 0/391 [00:00<?, ?it/s]\r"
    " 50%|#####     | 196/391 [00:12<00:12, 15.8it/s, xentropy=1.9, acc=0.3]\r"
    "100%|##########| 391/391 [00:24<00:00, 16.1it/s, xentropy=1.6, acc=0.4]\n"
    "test_acc: 0.512\n"
)


def test_progress_detection() -> None:
    assert progress_seen(TQDM)
    assert progress_seen("step 3 | 2.50s/it\n")
    assert not progress_seen("loading...\nstill loading\n")
    prefixes = metric_prefixes(["tqdm.write('test_acc: %.3f' % (test_acc))", "print(args)"])
    assert prefixes == ("test_acc:",)
    assert progress_seen("test_acc: 0.4\n", prefixes)
    assert not progress_seen("test_acc: 0.4\n")


def test_timing_from_tqdm_and_wall() -> None:
    ep = (Override(dest="epochs", option="--epochs", value="1"),)
    t = estimate_timing(TQDM, overrides=ep, exit_code=0, duration_s=40.0)
    assert (t.sec_per_epoch, t.method, t.confidence) == (24.0, "tqdm", "medium")
    partial = " 25%|##  | 100/400 [01:00<03:00, 1.7it/s]\r"
    t = estimate_timing(partial, overrides=ep, exit_code=None, duration_s=1200)
    assert (t.sec_per_epoch, t.method, t.confidence) == (240.0, "tqdm", "low")
    t = estimate_timing("epoch done\n", overrides=ep, exit_code=0, duration_s=33.33)
    assert (t.sec_per_epoch, t.method, t.confidence) == (33.3, "wall", "low")
    t = estimate_timing("epoch done\n", overrides=(), exit_code=0, duration_s=33.3)
    assert t.sec_per_epoch is None and t.method == "none"


# --- decision table ----------------------------------------------------------------------


def _tb(kind: str = "python") -> ParsedTraceback:
    return ParsedTraceback(exc_type="E", message="m", raw="r", kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kw", "signal"),
    [
        ({"exit_code": 0, "timed_out": False, "progress": True, "traceback": None}, "passed_exit0"),
        (
            {"exit_code": 0, "timed_out": False, "progress": False, "traceback": None},
            "passed_exit0",
        ),
        (
            {"exit_code": 137, "timed_out": True, "progress": True, "traceback": None},
            "passed_progress",
        ),
        (
            {"exit_code": 137, "timed_out": True, "progress": True, "traceback": _tb("killed")},
            "passed_progress",
        ),
        ({"exit_code": 137, "timed_out": True, "progress": True, "traceback": _tb()}, "traceback"),
        (
            {"exit_code": 137, "timed_out": True, "progress": False, "traceback": None},
            "timeout_no_progress",
        ),
        ({"exit_code": 1, "timed_out": False, "progress": True, "traceback": _tb()}, "traceback"),
        (
            {"exit_code": 2, "timed_out": False, "progress": False, "traceback": _tb("exit")},
            "traceback",
        ),
        ({"exit_code": 1, "timed_out": False, "progress": True, "traceback": _tb("oom")}, "oom"),
        (
            {
                "exit_code": 137,
                "timed_out": False,
                "progress": True,
                "traceback": None,
                "oom_killed": True,
            },
            "oom",
        ),
        ({"exit_code": None, "timed_out": False, "progress": False, "traceback": None}, "infra"),
    ],
)
def test_decide_signal(kw: dict[str, object], signal: str) -> None:
    kw.setdefault("oom_killed", False)
    assert decide_signal(**kw) == signal  # type: ignore[arg-type]


# --- evidence ----------------------------------------------------------------------------


def test_smoke_evidence_additive_and_ledger_fields_kept() -> None:
    old = make_smoke_evidence(passed=True, attempts=1, capped=False, source="s", locator="x")
    assert old.content == {"passed": True, "attempts": 1, "capped": False}
    same = make_smoke_evidence(
        passed=True, attempts=1, capped=False, source="s", locator="x", details={}
    )
    assert same.evidence_id == old.evidence_id
    new = make_smoke_evidence(
        passed=False, attempts=2, capped=False, source="s", locator="y",
        details={"signal": "traceback", "traceback": None, "attempt": 2}, links=("e1",),
    )  # fmt: skip
    assert new.kind == "smoke_result" and new.authority == "execution"
    assert new.confidence == "observed" and new.links == ("e1",)
    assert new.content["passed"] is False and new.content["signal"] == "traceback"
    with pytest.raises(ValueError, match="redefine"):
        make_smoke_evidence(
            passed=True,
            attempts=1,
            capped=False,
            source="s",
            locator="z",
            details={"passed": False},
        )
