"""Tests for reproducer.rules.known_fixes: fixes, loop guard, fold, input digest / attempts."""

from __future__ import annotations

import pytest

from reproducer.domain.ledger import make_smoke_evidence
from reproducer.domain.smoke_plan import Override, plan_smoke
from reproducer.rules.classify import Classification
from reproducer.rules.known_fixes import (
    DataMountFix,
    FixState,
    apply_fix,
    count_attempts,
    fix_for,
    fold_fix_state,
    known_fix_evidence,
    smoke_input_digest,
)

LOGS = Classification(cls="known", pattern="missing_output_dir", params={"dir": "logs"}, reason="r")
OOM = Classification(
    cls="known",
    pattern="oom",
    params={"option": "--batch_size", "dest": "batch_size", "current": 128},
    reason="oom",
)


def test_prep_dir_no_deviation_and_duplicate_is_noop() -> None:
    a = fix_for(LOGS, FixState())
    assert a is not None and a.kind == "prep_dir" and a.params == {"dir": "logs"}
    assert a.deviation is None
    state = apply_fix(FixState(), a)
    assert state.prep_dirs == ("logs",)
    assert fix_for(LOGS, state) is None  # same action twice -> loop guard


def test_oom_halves_batch_with_deviation_never_below_1() -> None:
    a = fix_for(OOM, FixState())
    assert a is not None and a.kind == "override"
    assert a.params["overrides"] == [
        {"dest": "batch_size", "option": "--batch_size", "value": "64", "previous": "128"}
    ]
    assert a.deviation is not None
    assert (a.deviation.field, a.deviation.stated, a.deviation.chosen) == (
        "batch_size",
        "128",
        "64",
    )
    state = apply_fix(FixState(), a)
    assert state.overrides == (
        Override(dest="batch_size", option="--batch_size", value="64", previous="128"),
    )
    one = OOM.model_copy(update={"params": {**OOM.params, "current": 1}})
    assert fix_for(one, FixState()) is None
    three = OOM.model_copy(update={"params": {**OOM.params, "current": 3}})
    a3 = fix_for(three, FixState())
    assert a3 is not None and a3.params["overrides"][0]["value"] == "1"


def test_oom_with_accumulation_keeps_effective_batch() -> None:
    c = OOM.model_copy(
        update={
            "params": {
                **OOM.params,
                "accum_option": "--accum",
                "accum_dest": "accum",
                "accum_current": 2,
            }
        }
    )
    a = fix_for(c, FixState())
    assert a is not None and a.deviation is None
    assert [o["value"] for o in a.params["overrides"]] == ["64", "4"]


def test_override_replaces_previous_override_of_same_option() -> None:
    a1 = fix_for(OOM, FixState())
    assert a1 is not None
    s1 = apply_fix(FixState(), a1)
    c2 = OOM.model_copy(update={"params": {**OOM.params, "current": 64}})
    a2 = fix_for(c2, s1)
    assert a2 is not None
    s2 = apply_fix(s1, a2)
    assert [(o.option, o.value) for o in s2.overrides] == [("--batch_size", "32")]
    plan = plan_smoke("python train.py --batch_size 128", (), s2.overrides)
    assert plan.command == "python train.py --batch_size 32"


def test_add_package_and_mount() -> None:
    c = Classification(
        cls="known",
        pattern="missing_module",
        params={"module": "cv2", "package": "opencv-python"},
        reason="r",
    )
    a = fix_for(c, FixState())
    assert a is not None and a.kind == "add_package" and a.params["package"] == "opencv-python"
    s = apply_fix(FixState(), a)
    assert s.packages == ("opencv-python",) and fix_for(c, s) is None
    m = Classification(
        cls="known",
        pattern="missing_data_path",
        params={"name": "svhn", "container": "/work/s", "path": "x"},
        reason="r",
    )
    a = fix_for(m, FixState())
    assert a is not None and a.kind == "add_mount"
    s = apply_fix(FixState(), a)
    assert s.mounts == (DataMountFix(name="svhn", container="/work/s"),) and fix_for(m, s) is None


def test_fix_for_rejects_non_known() -> None:
    with pytest.raises(ValueError):
        fix_for(Classification(cls="critic", reason="r"), FixState())


def test_known_fix_evidence_and_fold_in_order() -> None:
    smoke = make_smoke_evidence(
        passed=False, attempts=1, capped=False, source="repo@x", locator="s1"
    )
    a1 = fix_for(LOGS, FixState())
    assert a1 is not None
    e1 = known_fix_evidence(a1, LOGS, source="repo@x", smoke_evidence_id=smoke.evidence_id)
    assert (e1.kind, e1.authority, e1.confidence, e1.links) == (
        "known_fix", "rule", "derived", (smoke.evidence_id,),
    )  # fmt: skip
    s1 = apply_fix(FixState(), a1)
    ckpt = LOGS.model_copy(update={"params": {"dir": "checkpoints"}})
    a2 = fix_for(ckpt, s1)
    assert a2 is not None
    e2 = known_fix_evidence(a2, ckpt, source="repo@x", smoke_evidence_id="f" * 64)
    assert fold_fix_state([smoke, e1, e2]).prep_dirs == ("logs", "checkpoints")
    assert fold_fix_state([smoke]) == FixState()


def test_input_digest_and_attempts() -> None:
    s0 = FixState()
    d0 = smoke_input_digest(image="img:1", state=s0, patches=(), command="python t.py --epochs 1")
    assert d0 == smoke_input_digest(
        image="img:1", state=FixState(), patches=[], command="python t.py --epochs 1"
    )
    assert count_attempts([], d0) == 1
    assert count_attempts([d0], d0) == 1  # identical inputs: no new attempt
    assert count_attempts([d0, d0, None], d0) == 1  # legacy records without a digest ignored
    s1 = FixState(prep_dirs=("logs",))
    changed = [
        smoke_input_digest(image="img:2", state=s0, patches=(), command="python t.py --epochs 1"),
        smoke_input_digest(image="img:1", state=s1, patches=(), command="python t.py --epochs 1"),
        smoke_input_digest(
            image="img:1", state=s0, patches=("diff",), command="python t.py --epochs 1"
        ),
        smoke_input_digest(image="img:1", state=s0, patches=(), command="python t.py --epochs 2"),
    ]
    assert len({d0, *changed}) == 5
    assert count_attempts([d0], changed[1]) == 2
