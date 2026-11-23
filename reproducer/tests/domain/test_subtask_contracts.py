"""Tests for Citation (aliases + quote snapping) and the subtask result contracts A/B/C."""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from reproducer.domain.contracts import (
    CheckpointFact,
    Entrypoint,
    EntrypointList,
    EnvSpec,
    MetricOutput,
    PackageFact,
    UnknownName,
    UnknownProposal,
    WorkflowAnalysis,
)
from reproducer.domain.grounding import Citation, QuoteNotFound, snap_quote

README_ID = "r" * 64
REQS_ID = "q" * 64
TRAIN_ID = "t" * 64
README = (
    "## Requirements\n[PyTorch v0.4.0](http://pytorch.org/)\n\n"
    "Test error (%, flip/translation augmentation, mean of 5 runs)\n"
    "`python train.py --dataset   cifar10\n --cutout`"
)
TEXTS = {
    README_ID: README,
    REQS_ID: "torch==1.7.1\nnumpy\n",
    TRAIN_ID: "print('test_acc: %.3f' % test_acc)",
}


def _ctx(**kw: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "repo_files": {"train.py", "README.md", "configs/a.yaml"},
        "evidence_ids": set(TEXTS),
        "texts": TEXTS,
        "aliases": {"E1": README_ID, "E2": REQS_ID, "E3": TRAIN_ID},
        "evidence_kinds": {README_ID: "readme", REQS_ID: "repo_file", TRAIN_ID: "repo_file"},
        "entrypoint_args": {"train.py": frozenset({"--dataset", "--cutout"})},
        "allowed_unknowns": frozenset(get_args(UnknownName)),
        "metric_line_evidence": {},
        "command_evidence": {},
    }
    ctx.update(kw)
    return ctx


def _cit(quote: str, eid: str = "E1") -> dict[str, str]:
    return {"evidence_id": eid, "quote": quote}


# --------------------------------------------------------------------------- aliases


def test_alias_resolves_and_full_id_is_stored() -> None:
    c = Citation.model_validate(_cit("PyTorch v0.4.0"), context=_ctx())
    assert c.evidence_id == README_ID
    full = Citation.model_validate(_cit("PyTorch v0.4.0", README_ID), context=_ctx())
    assert full == c
    # the stored model re-validates unchanged (round trip from evidence content)
    assert Citation.model_validate(c.model_dump(), context=_ctx()) == c


@pytest.mark.parametrize("eid", ["E9", "e1", "x" * 64])
def test_unknown_alias_rejected(eid: str) -> None:
    with pytest.raises(ValidationError, match="unknown evidence id"):
        Citation.model_validate(_cit("PyTorch", eid), context=_ctx())


def test_citation_fails_closed_without_context() -> None:
    ctx = _ctx()
    del ctx["texts"]
    with pytest.raises(ValidationError, match="'texts'"):
        Citation.model_validate(_cit("PyTorch v0.4.0"), context=ctx)
    with pytest.raises(ValidationError, match="'texts'"):
        Citation.model_validate(_cit("PyTorch v0.4.0"))
    with pytest.raises(ValidationError, match="no text available"):
        Citation.model_validate(_cit("PyTorch v0.4.0"), context=_ctx(texts={}))
    no_alias = _ctx()
    del no_alias["aliases"]
    with pytest.raises(ValidationError, match="unknown evidence id"):
        Citation.model_validate(_cit("PyTorch v0.4.0"), context=no_alias)


# --------------------------------------------------------------------------- snapping


@pytest.mark.parametrize(
    ("quote", "verbatim"),
    [
        # (a) whitespace-normalized exact: stored with the source's own whitespace
        (
            "python train.py --dataset cifar10 --cutout",
            "python train.py --dataset   cifar10\n --cutout",
        ),
        # (b) case-insensitive
        ("pytorch V0.4.0", "PyTorch v0.4.0"),
        # (c) fuzzy: punctuation variant from the live qwen run
        (
            "Test error (%), flip/translation augmentation, mean of 5 runs",
            "Test error (%, flip/translation augmentation, mean of 5 runs",
        ),
    ],
)
def test_quote_snaps_to_verbatim_span(quote: str, verbatim: str) -> None:
    assert snap_quote(quote, README) == verbatim
    c = Citation.model_validate(_cit(quote), context=_ctx())
    assert c.quote == verbatim


def test_snap_rejects_changed_digits() -> None:
    with pytest.raises(QuoteNotFound):
        snap_quote("[PyTorch v1.0](http://pytorch.org/)", README)
    with pytest.raises(QuoteNotFound):  # near-identical except for the digit
        snap_quote("Test error (%, flip/translation augmentation, mean of 9 runs)", README)


def test_unrelated_quote_rejected_with_candidates() -> None:
    with pytest.raises(QuoteNotFound) as exc:
        snap_quote("the quick brown fox jumps over the lazy dog", README)
    assert len(exc.value.candidates) == 3
    assert all(c in " ".join(README.split()) for c in exc.value.candidates)
    with pytest.raises(ValidationError, match="closest spans in the source"):
        Citation.model_validate(_cit("the quick brown fox jumps"), context=_ctx())


@pytest.mark.parametrize(("quote", "fragment"), [("", "1..300"), ("a" * 301, "1..300")])
def test_quote_length_checked_on_model_text(quote: str, fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        Citation.model_validate(_cit(quote), context=_ctx())


def test_blank_quote_rejected() -> None:
    with pytest.raises(ValidationError, match="quote is blank"):
        Citation.model_validate(_cit("  \n "), context=_ctx())


# --------------------------------------------------------------------------- A: PackageFact


def test_package_source_derived_from_cited_evidence_kind() -> None:
    readme = PackageFact.model_validate(
        {"name": "torch", "version": "0.4.0", "citation": _cit("PyTorch v0.4.0")},
        context=_ctx(),
    )
    assert readme.source == "readme"
    req = PackageFact.model_validate(
        {"name": "torch", "version": "1.7.1", "citation": _cit("torch==1.7.1", "E2")},
        context=_ctx(),
    )
    assert req.source == "repo_file"
    inferred = PackageFact.model_validate({"name": "tqdm"}, context=_ctx())
    assert inferred.source == "inferred" and inferred.citation is None
    assert "source" not in PackageFact.model_json_schema()["properties"]


@pytest.mark.parametrize("version", ["0.4.0", "v0.4.0", "V0.4.0"])
def test_package_version_in_quote_accepts_leading_v(version: str) -> None:
    PackageFact.model_validate(
        {"name": "torch", "version": version, "citation": _cit("PyTorch v0.4.0")}, context=_ctx()
    )


@pytest.mark.parametrize("version", ["1.0", "0.4", "4.0", "0.4.0.1"])
def test_package_version_not_in_quote_rejected(version: str) -> None:
    with pytest.raises(ValidationError, match="does not appear in the cited quote"):
        PackageFact.model_validate(
            {"name": "torch", "version": version, "citation": _cit("PyTorch v0.4.0")},
            context=_ctx(),
        )


def test_package_source_given_must_match_derived() -> None:
    data = {"name": "torch", "version": "0.4.0", "citation": _cit("PyTorch v0.4.0")}
    PackageFact.model_validate({**data, "source": "readme"}, context=_ctx())
    with pytest.raises(ValidationError, match="derived from the citation"):
        PackageFact.model_validate({**data, "source": "repo_file"}, context=_ctx())


def test_package_citation_needs_evidence_kinds() -> None:
    ctx = _ctx()
    del ctx["evidence_kinds"]
    with pytest.raises(ValidationError, match="evidence_kinds"):
        PackageFact.model_validate(
            {"name": "torch", "citation": _cit("PyTorch v0.4.0")}, context=ctx
        )


def test_env_spec_round_trip_from_stored_content() -> None:
    data = {
        "packages": [{"name": "torch", "version": "0.4.0", "citation": _cit("pytorch v0.4.0")}],
        "unknowns": [{"name": "python_version", "why": "README does not state it"}],
    }
    env = EnvSpec.model_validate(data, context=_ctx())
    dumped = env.model_dump(mode="json")
    assert dumped["packages"][0]["source"] == "readme"
    assert dumped["packages"][0]["citation"] == {
        "evidence_id": README_ID,
        "quote": "PyTorch v0.4.0",
    }
    assert EnvSpec.model_validate(dumped, context=_ctx()) == env


# --------------------------------------------------------------------------- unknowns


@pytest.mark.parametrize("name", ["shake_shake_regularization_network", "numpy_version", ""])
def test_junk_unknown_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        UnknownProposal(name=name, why="x")  # type: ignore[arg-type]


def _unk(name: str, why: str = "w", **kw: Any) -> UnknownProposal:
    return UnknownProposal.model_validate({"name": name, "why": why}, context=_ctx(**kw))


def test_unknown_proposal_ok_and_why_capped() -> None:
    _unk("torch_version", "w" * 200)
    _unk("eval_protocol")
    with pytest.raises(ValidationError):
        _unk("python_version", "w" * 201)


@pytest.mark.parametrize(
    "name",
    [
        "epochs",
        "batch_size",
        "learning_rate",
        "lr_schedule",
        "weight_decay",
        "optimizer",
        "seed",
        "num_runs",
    ],
)
def test_hyperparameter_unknown_names_removed(name: str) -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        _unk(name)


def test_unknown_scoped_by_allowed_unknowns() -> None:
    a_only = frozenset({"python_version", "torch_version", "cuda_version"})
    _unk("cuda_version", allowed_unknowns=a_only)
    with pytest.raises(ValidationError, match="not allowed here"):
        _unk("dataset_source", allowed_unknowns=a_only)
    with pytest.raises(ValidationError, match="takes no unknowns"):
        _unk("python_version", allowed_unknowns=frozenset())
    with pytest.raises(ValidationError, match="allowed_unknowns"):
        UnknownProposal(name="python_version", why="w")  # no context: fail closed


# --------------------------------------------------------------------------- B: Entrypoints


def test_entrypoint_ok() -> None:
    e = Entrypoint.model_validate(
        {
            "path": "./train.py",
            "kind": "train",
            "args_used": ["--dataset", "--cutout"],
            "example": _cit("python train.py --dataset cifar10 --cutout"),
        },
        context=_ctx(),
    )
    assert e.path == "train.py"
    assert e.example is not None and e.example.evidence_id == README_ID
    assert e.example.quote == "python train.py --dataset   cifar10\n --cutout"  # verbatim


CUTOUT_CMD = (
    "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
)
CUTOUT_ARGS = frozenset(
    {"--dataset", "--model", "--data_augmentation", "--cutout", "--length", "--epochs"}
)


def _cutout_ctx() -> dict[str, Any]:
    texts = {**TEXTS, README_ID: f"`{CUTOUT_CMD}`"}
    return _ctx(texts=texts, entrypoint_args={"train.py": CUTOUT_ARGS})


def test_args_used_derived_from_example_in_order() -> None:
    e = Entrypoint.model_validate(
        {"path": "train.py", "kind": "train", "example": _cit(CUTOUT_CMD)}, context=_cutout_ctx()
    )
    assert e.args_used == ("--dataset", "--model", "--data_augmentation", "--cutout", "--length")
    assert "args_used" not in Entrypoint.model_json_schema()["properties"]
    # stored form re-validates (model-sent value equals the derived one)
    assert Entrypoint.model_validate(e.model_dump(), context=_cutout_ctx()) == e


def test_args_used_empty_without_example() -> None:
    e = Entrypoint.model_validate({"path": "train.py", "kind": "train"}, context=_ctx())
    assert e.args_used == ()


@pytest.mark.parametrize(
    "sent",
    [
        ["--dataset", "--cutout"],  # subset
        ["--model", "--dataset", "--data_augmentation", "--cutout", "--length"],  # order
        ["--dataset", "--model", "--data_augmentation", "--cutout", "--length", "--epochs"],
        [],
    ],
)
def test_mismatching_model_sent_args_used_rejected(sent: list[str]) -> None:
    data = {"path": "train.py", "kind": "train", "args_used": sent, "example": _cit(CUTOUT_CMD)}
    with pytest.raises(ValidationError, match="args_used is derived from the example command"):
        Entrypoint.model_validate(data, context=_cutout_ctx())


def test_args_used_without_example_must_be_empty_if_sent() -> None:
    Entrypoint.model_validate(
        {"path": "train.py", "kind": "train", "args_used": []}, context=_ctx()
    )
    with pytest.raises(ValidationError, match=r"derived from the example command \(\[\]\)"):
        Entrypoint.model_validate(
            {"path": "train.py", "kind": "train", "args_used": ["--dataset"]}, context=_ctx()
        )


def test_args_used_whole_tokens_only() -> None:
    texts = {**TEXTS, README_ID: "python train.py --cutout_size 8 --dataset=svhn --lr 0.1"}
    ctx = _ctx(texts=texts, entrypoint_args={"train.py": frozenset({"--cutout", "--dataset"})})
    data = {"path": "train.py", "kind": "train", "example": _cit(texts[README_ID])}
    assert Entrypoint.model_validate(data, context=ctx).args_used == ("--dataset",)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"path": "eval.py", "kind": "eval"}, "file not in repo"),
        ({"path": "README.md", "kind": "other"}, "not an entrypoint candidate"),
        ({"path": "train.py", "kind": "test"}, "Input should be"),
    ],
)
def test_entrypoint_rejected(data: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        Entrypoint.model_validate(data, context=_ctx())


def test_entrypoint_fails_closed_without_args_context() -> None:
    ctx = _ctx()
    del ctx["entrypoint_args"]
    with pytest.raises(ValidationError, match="entrypoint_args"):
        Entrypoint.model_validate({"path": "train.py", "kind": "train"}, context=ctx)


def test_entrypoint_list_needs_one_and_unique_paths() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        EntrypointList.model_validate({"entrypoints": []}, context=_ctx())
    dup = {
        "entrypoints": [
            {"path": "train.py", "kind": "train"},
            {"path": "./train.py", "kind": "eval"},
        ]
    }
    with pytest.raises(ValidationError, match="duplicate entrypoint path"):
        EntrypointList.model_validate(dup, context=_ctx())


# --------------------------------------------------------------------------- C: Workflow


def test_workflow_ok_and_grounded() -> None:
    data: dict[str, Any] = {
        "train_scripts": ["train.py"],
        "configs": ["configs/a.yaml"],
        "datasets": [{"name": "cifar10", "citation": _cit("--dataset cifar10")}],
        "metric_outputs": [
            {"path": "train.py", "metric": "test_acc", "citation": _cit("test_acc", "E3")}
        ],
    }
    wf = WorkflowAnalysis.model_validate(data, context=_ctx())
    assert wf.metric_outputs[0].citation.evidence_id == TRAIN_ID
    data["configs"] = ["configs/b.yaml"]
    with pytest.raises(ValidationError, match="file not in repo"):
        WorkflowAnalysis.model_validate(data, context=_ctx())


def test_checkpoint_requires_citation() -> None:
    with pytest.raises(ValidationError):
        CheckpointFact.model_validate({"name": "wrn", "url": None}, context=_ctx())


# --------------------------------------------------------------------------- relocation

A_ID, B_ID, C_ID = "a" * 64, "b" * 64, "c" * 64


def _reloc_ctx(texts: dict[str, str]) -> dict[str, Any]:
    return _ctx(
        texts=texts,
        evidence_ids=set(texts),
        aliases={f"E{i}": t for i, t in enumerate(texts, start=1)},
    )


def test_citation_relocated_to_single_other_text() -> None:
    texts = {A_ID: "def train():\n    pass", B_ID: "Test error (%, mean of 5 runs)", C_ID: "x"}
    c = Citation.model_validate(
        _cit("test ERROR (%, mean of 5 runs)", "E1"), context=_reloc_ctx(texts)
    )
    assert c.evidence_id == B_ID
    assert c.quote == "Test error (%, mean of 5 runs)"  # verbatim span of the new source


def test_citation_not_relocated_when_found_in_several_texts() -> None:
    texts = {
        A_ID: "nothing here",
        B_ID: "python train.py --cutout",
        C_ID: "run: python train.py --cutout",
    }
    with pytest.raises(ValidationError, match=r"appears in several texts \['E2', 'E3'\]"):
        Citation.model_validate(_cit("python train.py --cutout", "E1"), context=_reloc_ctx(texts))


def test_citation_not_relocated_on_fuzzy_only_match() -> None:
    texts = {
        A_ID: "def train():\n    pass",
        B_ID: "Test error (%, flip/translation augmentation, mean of 5 runs)",
    }
    quote = "Test error (%), flip/translation augmentation, mean of 5 runs"
    with pytest.raises(ValidationError, match=r"close but not exact match is in \['E2'\]"):
        Citation.model_validate(_cit(quote, "E1"), context=_reloc_ctx(texts))
    # cited correctly, the same fuzzy quote snaps (fuzzy is allowed in the cited text)
    assert Citation.model_validate(_cit(quote, "E2"), context=_reloc_ctx(texts)).evidence_id == B_ID


def test_citation_cited_text_match_wins_over_other_texts() -> None:
    texts = {A_ID: "python train.py", B_ID: "python train.py"}
    assert (
        Citation.model_validate(
            _cit("python train.py", "E1"), context=_reloc_ctx(texts)
        ).evidence_id
        == A_ID
    )


# --------------------------------------------------------------------------- metric lines

METRIC_ID = "m" * 64


def _metric_ctx() -> dict[str, Any]:
    texts = {**TEXTS, METRIC_ID: "print('test_acc: %.3f' % test_acc)"}
    return _ctx(
        texts=texts,
        evidence_ids=set(texts),
        aliases={"E1": README_ID, "E2": REQS_ID, "E3": TRAIN_ID, "E4": METRIC_ID},
        evidence_kinds={**_ctx()["evidence_kinds"], METRIC_ID: "repo_file"},
        metric_line_evidence={"train.py": METRIC_ID},
    )


def _metric(eid: str, path: str = "train.py") -> dict[str, Any]:
    return {"path": path, "metric": "test_acc", "citation": _cit("test_acc: %.3f", eid)}


def test_metric_output_must_cite_metric_lines_evidence() -> None:
    m = MetricOutput.model_validate(_metric("E4"), context=_metric_ctx())
    assert m.citation.evidence_id == METRIC_ID
    with pytest.raises(
        ValidationError, match="cite the verified metric line for train.py with evidence_id 'E4'"
    ):
        MetricOutput.model_validate(_metric("E3"), context=_metric_ctx())


def test_metric_output_any_citation_without_metric_lines_for_path() -> None:
    m = MetricOutput.model_validate(_metric("E3"), context=_ctx())  # no metric lines at all
    assert m.citation.evidence_id == TRAIN_ID
    ctx = _metric_ctx()
    ctx["repo_files"] = {*ctx["repo_files"], "eval.py"}
    m = MetricOutput.model_validate(_metric("E3", path="eval.py"), context=ctx)
    assert m.citation.evidence_id == TRAIN_ID


def test_metric_output_fails_closed_without_context_key() -> None:
    ctx = _ctx()
    del ctx["metric_line_evidence"]
    with pytest.raises(ValidationError, match="metric_line_evidence"):
        MetricOutput.model_validate(_metric("E3"), context=ctx)


def test_other_unknown_name_removed() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        _unk("other")


# --------------------------------------------------------------------------- verified commands

CMD_ID = "k" * 64


def _cmd_ctx() -> dict[str, Any]:
    texts = {**TEXTS, CMD_ID: "python train.py --dataset cifar10 --cutout"}
    return _ctx(
        texts=texts,
        evidence_ids=set(texts),
        aliases={"E1": README_ID, "E2": REQS_ID, "E3": TRAIN_ID, "E4": CMD_ID},
        command_evidence={"train.py": frozenset({CMD_ID})},
    )


def test_entrypoint_must_cite_verified_command_when_one_exists() -> None:
    data: dict[str, Any] = {
        "path": "train.py",
        "kind": "train",
        "args_used": ["--dataset", "--cutout"],
        "example": _cit("python train.py --dataset cifar10 --cutout", "E4"),
    }
    e = Entrypoint.model_validate(data, context=_cmd_ctx())
    assert e.example is not None and e.example.evidence_id == CMD_ID
    data["example"] = _cit("python train.py --dataset cifar10 --cutout", "E1")  # README itself
    with pytest.raises(
        ValidationError, match=r"cite a verified command for train.py: evidence_id one of \['E4'\]"
    ):
        Entrypoint.model_validate(data, context=_cmd_ctx())
    with pytest.raises(ValidationError, match="cite a verified command"):
        Entrypoint.model_validate({"path": "train.py", "kind": "train"}, context=_cmd_ctx())


def test_entrypoint_example_may_be_null_without_verified_commands() -> None:
    ctx = _cmd_ctx()
    ctx["command_evidence"] = {}
    Entrypoint.model_validate({"path": "train.py", "kind": "train"}, context=ctx)


def test_entrypoint_fails_closed_without_command_context() -> None:
    ctx = _ctx()
    del ctx["command_evidence"]
    with pytest.raises(ValidationError, match="command_evidence"):
        Entrypoint.model_validate({"path": "train.py", "kind": "train"}, context=ctx)
