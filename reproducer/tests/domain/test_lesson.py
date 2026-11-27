"""Lesson / Change contract (M4-S2): every check is code, with the validation context."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import CUTOUT_ARGS
from pydantic import ValidationError

from reproducer.domain.contracts import Lesson, value_stated

RUN, CURVE, PLAN = "r" * 64, "c" * 64, "q" * 64
P_LR, P_SEED, REPO, LLM_EV, HUMAN = "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64
TEXTS = {
    P_LR: "The learning rate is initially set to 0.1 and we train for 200 epochs",
    P_SEED: "All runs use random seed 42",
    REPO: "parser.add_argument('--seed', type=int, default=0,\n"
    "parser.add_argument('--epochs', type=int, default=150,\n"
    "    val_acc = correct / total\n"
    "    tqdm.write('test_acc: %.3f' % (test_acc))  # the last epoch's accuracy\n",
    LLM_EV: "the mapping says epochs should be three hundred",
    HUMAN: "eval_checkpoint = best: the authors confirmed by email",
}
AUTHORITY = {
    RUN: "execution", CURVE: "rule", PLAN: "rule", P_LR: "paper", P_SEED: "paper",
    REPO: "repo", LLM_EV: "llm", HUMAN: "human",
}  # fmt: skip


def ctx(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "claim_id": "c10",
        "trial": 2,
        "previous_step": None,
        "subject_ids": frozenset({RUN, CURVE}),
        "evidence_ids": frozenset(AUTHORITY),
        "texts": TEXTS,
        "aliases": {
            "RUN": RUN,
            "CURVE": CURVE,
            "PLAN": PLAN,
            "P1": P_LR,
            "P2": P_SEED,
            "R1": REPO,
            "L1": LLM_EV,
            "H1": HUMAN,
        },  # fmt: skip
        "evidence_authority": AUTHORITY,
        "argparse_spec": CUTOUT_ARGS,
        "claim_flags": frozenset({"--dataset", "--model"}),
        "open_plan_unknowns": frozenset({"unk:seed", "unk:eval_checkpoint"}),
        "eval_protocol_table": {
            "metric_source.which": frozenset({"best"}),
            "metric_source.conversion": frozenset(
                {"accuracy_percent_to_error_percent", "accuracy_fraction_to_error_percent"}
            ),
        },  # fmt: skip
        "claim_unit": "%",
        "open_unknown_names": frozenset({"seed", "eval_checkpoint"}),
    }
    return {**base, **kw}


def lesson(change: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "claim_id": "c10", "trial": 2, "step": "eval_protocol", "evidence_ids": ["RUN", "CURVE"],
        "text": "Final epoch error is above the best epoch; the paper checkpoint is not stated.",
        "proposed_change": change,
    }  # fmt: skip
    return {**data, **kw}


def change(kind: str, target: str, new_value: str | None = None,
           support: tuple[str, str] | None = None) -> dict[str, Any]:  # fmt: skip
    return {
        "kind": kind, "target": target, "new_value": new_value,
        "support": None if support is None else {"evidence_id": support[0], "quote": support[1]},
        "rationale": "grounded in the cited text",
    }  # fmt: skip


def ok(data: dict[str, Any], **kw: Any) -> Lesson:
    return Lesson.model_validate(data, context=ctx(**kw))


def bad(data: dict[str, Any], match: str, **kw: Any) -> None:
    with pytest.raises(ValidationError, match=match):
        Lesson.model_validate(data, context=ctx(**kw))


# --- valid -----------------------------------------------------------------------------------


def test_valid_eval_protocol_lesson() -> None:
    les = ok(lesson(change("eval_protocol", "metric_source.which", "best")))
    assert les.evidence_ids == (RUN, CURVE)  # aliases resolved
    assert les.proposed_change is not None and les.proposed_change.new_value == "best"


def test_eval_protocol_conversion_needs_repo_scale() -> None:
    scale = ("R1", "val_acc = correct / total")
    frac = change("eval_protocol", "metric_source.conversion",
                  "accuracy_fraction_to_error_percent", scale)  # fmt: skip
    assert ok(lesson(frac)).proposed_change is not None
    pct = {**frac, "new_value": "accuracy_percent_to_error_percent"}
    bad(lesson(pct), "needs a percent")
    bad(lesson({**frac, "support": None}), "conversion change needs support")
    bad(lesson(change("eval_protocol", "metric_source.which", "last")), "one of")
    bad(lesson(change("eval_protocol", "metric_source.pattern", "x")), "target must be one of")


def test_valid_arg_change_citing_paper() -> None:
    ch = change("arg", "--epochs", "200", ("P1", "we train for 200 epochs"))
    les = ok(lesson(ch, step="config_vs_paper"))
    assert les.proposed_change is not None
    assert les.proposed_change.support is not None
    assert les.proposed_change.support.evidence_id == P_LR


def test_valid_assumption_change_citing_repo() -> None:
    ch = change("assumption", "unk:eval_checkpoint", "last",
                ("R1", "# the last epoch's accuracy"))  # fmt: skip
    for step in ("eval_protocol", "training"):
        assert ok(lesson(ch, step=step)).proposed_change is not None


def test_rejects_assumption_value_absent_from_the_quote() -> None:
    ch = change("assumption", "unk:eval_checkpoint", "best",
                ("R1", "# the last epoch's accuracy"))  # fmt: skip
    bad(lesson(ch), "does not state 'best'")
    seed = change("assumption", "unk:seed", "42", ("P2", "All runs use random seed 42"))
    assert ok(lesson(seed)).proposed_change is not None  # value present (paper seed)


def test_none_change_and_no_change() -> None:
    assert ok(lesson(change("none", "none"), step="training")).proposed_change is not None
    assert ok(lesson(None, step="training")).proposed_change is None
    bad(lesson(change("none", "none", "x")), "new_value must be null")


def test_seed_change_with_paper_seed_is_allowed() -> None:
    ch = change("arg", "--seed", "42", ("P2", "All runs use random seed 42"))
    assert ok(lesson(ch, step="config_vs_paper")).proposed_change is not None


# --- rejected --------------------------------------------------------------------------------


def test_rejects_step_out_of_order() -> None:
    bad(lesson(None, step="eval_protocol"), "goes back", previous_step="config_vs_paper")
    assert ok(lesson(None, step="config_vs_paper"), previous_step="config_vs_paper")
    assert ok(lesson(None, step="training"), previous_step="config_vs_paper")


def test_rejects_arg_change_citing_repo_only() -> None:
    ch = change("arg", "--epochs", "150", ("R1", "parser.add_argument('--epochs', type=int, "
                                                 "default=150,"))  # fmt: skip
    bad(lesson(ch, step="config_vs_paper"), "PAPER evidence")


def test_rejects_arg_value_not_stated_or_unparsable() -> None:
    ch = change("arg", "--epochs", "300", ("P1", "we train for 200 epochs"))
    bad(lesson(ch, step="config_vs_paper"), "does not state")
    ch = change("arg", "--epochs", "lots", ("P1", "we train for 200 epochs"))
    bad(lesson(ch, step="config_vs_paper"), "not a valid int")
    ch = change("arg", "--nope", "1", ("P1", "we train for 200 epochs"))
    bad(lesson(ch, step="config_vs_paper"), "not an argparse option")


def test_rejects_seed_change_without_paper_seed() -> None:
    arg = change("arg", "--seed", "1", ("P1", "The learning rate is initially set to 0.1"))
    bad(lesson(arg, step="config_vs_paper"), "seed fishing")
    unk = change("assumption", "unk:seed", "0", ("R1", "parser.add_argument('--seed', type=int, "
                                                       "default=0,"))  # fmt: skip
    bad(lesson(unk, step="training"), "seed fishing")


def test_rejects_support_citing_llm_evidence() -> None:
    ch = change("none", "none", None, ("L1", "the mapping says epochs should be three hundred"))
    bad(lesson(ch), "not 'llm' evidence")
    ch = change("assumption", "unk:eval_checkpoint", "best",
                ("L1", "the mapping says epochs should be three hundred"))  # fmt: skip
    bad(lesson(ch), "paper or repo")
    human = change("none", "none", None, ("H1", "the authors confirmed by email"))
    assert ok(lesson(human)).proposed_change is not None  # human is allowed as support


def test_rejects_two_changes() -> None:
    one = change("eval_protocol", "metric_source.which", "best")
    bad(lesson([one, one]), "at most one change")  # type: ignore[arg-type]
    data = lesson(None)
    data["proposed_changes"] = [one, one]
    bad(data, "at most one change")


def test_rejects_long_text() -> None:
    bad(lesson(None, text=" ".join(["w"] * 76)), "words")


def test_rejects_kind_step_mismatch() -> None:
    bad(lesson(change("eval_protocol", "metric_source.which", "best"), step="training"),
        "needs step 'eval_protocol'")  # fmt: skip
    ch = change("arg", "--epochs", "200", ("P1", "we train for 200 epochs"))
    bad(lesson(ch, step="eval_protocol"), "needs step 'config_vs_paper'")


def test_rejects_forbidden_targets() -> None:
    for target in ("tolerance", "claim value", "metric_line"):
        bad(lesson(change("assumption", target, "1", ("R1", "val_acc = correct / total"))),
            "touches the metric, claim or tolerance")  # fmt: skip
    ch = change("arg", "--dataset", "cifar100", ("P1", "we train for 200 epochs"))
    bad(lesson(ch, step="config_vs_paper"), "selects the claim")
    ch = change("assumption", "unk:python_version", "3.8", ("R1", "val_acc = correct / total"))
    bad(lesson(ch), "open unknown the plan depends on")


def test_rejects_evidence_without_the_run_and_wrong_ids() -> None:
    bad(lesson(None, evidence_ids=["P1"]), "must include the run_result or run_curve")
    bad(lesson(None, evidence_ids=["RUN", "nope"]), "unknown evidence id")
    bad(lesson(None, claim_id="c100"), "claim_id must be")
    bad(lesson(None, trial=1), "trial must be 2")


def test_value_stated() -> None:
    assert value_stated("5e-4", "weight decay of 5e-4")
    assert value_stated("0.0005", "weight decay of 5e-4")
    assert value_stated("16", "a cutout size of 16 × 16 pixels")
    assert not value_stated("1", "a cutout size of 16 × 16 pixels")
    assert value_stated("cifar10", "trained on CIFAR-10")
    assert not value_stated("", "anything")


# --- text grounding: never attribute an open unknown to the paper --------------------------------


def test_rejects_text_attributing_an_open_unknown_to_the_paper() -> None:
    haiku = ("Verified within tolerance. Best-epoch error is lower, but the paper's protocol is "
             "the final evaluation, so the claim stays as measured.")  # fmt: skip
    bad(lesson(None, step="training", text=haiku), "eval_checkpoint is an open unknown")
    bad(lesson(None, step="training", text="The paper uses seed 0."), "seed is an open unknown")


def test_text_grounding_accepts_hedged_or_unrelated_text() -> None:
    hedged = "The paper's checkpoint is not stated; we assumed the final epoch."
    assert ok(lesson(None, step="training", text=hedged))
    assert ok(lesson(None, step="training", text="The eval checkpoint is unknown to the paper."))
    assert ok(lesson(None, step="training", text="Epochs match the paper; error plateaued."))
    # a resolved (not open) unknown may be attributed
    closed = ctx(open_unknown_names=frozenset())
    assert Lesson.model_validate(
        lesson(None, step="training", text="The paper's protocol is the final evaluation."),
        context=closed,
    )
