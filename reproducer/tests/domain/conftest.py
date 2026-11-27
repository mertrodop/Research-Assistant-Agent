"""Cutout fixtures for run plans: the real paper spec, hand-written repo evidence matching the
uoguelph-mlrg/Cutout repo, its argparse spec and a hand-written claim mapping (the LLM
mapping comes later)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.contracts import ArgOption, ReproSpec, ResultClaim
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.run_plan import ClaimArg, ClaimMapping, SetupArg
from reproducer.domain.unknowns import spec_to_evidence

SPEC = Path(__file__).resolve().parents[2] / "evals" / "cutout" / "spec.json"
SOURCE = "uoguelph-mlrg/Cutout@test"
C10, C100 = "c10plus_resnet18_cutout", "c100plus_resnet18_cutout"
COMMAND = (
    "python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
)
C100_COMMAND = COMMAND.replace("cifar10", "cifar100").replace("16", "8")
METRIC_LINE = "tqdm.write('test_acc: %.3f' % (test_acc))"

CUTOUT_ARGS = (
    ArgOption(
        option_strings=("--dataset", "-d"),
        dest="dataset",
        default="cifar10",
        choices=("cifar10", "cifar100", "svhn"),
    ),
    ArgOption(
        option_strings=("--model", "-a"),
        dest="model",
        default="resnet18",
        choices=("resnet18", "wideresnet"),
    ),
    ArgOption(option_strings=("--batch_size",), dest="batch_size", type="int", default=128),
    ArgOption(option_strings=("--epochs",), dest="epochs", type="int", default=200),
    ArgOption(option_strings=("--learning_rate",), dest="learning_rate", type="float", default=0.1),
    ArgOption(
        option_strings=("--data_augmentation",),
        dest="data_augmentation",
        default=False,
        action="store_true",
    ),
    ArgOption(option_strings=("--cutout",), dest="cutout", default=False, action="store_true"),
    ArgOption(option_strings=("--n_holes",), dest="n_holes", type="int", default=1),
    ArgOption(option_strings=("--length",), dest="length", type="int", default=16),
    ArgOption(option_strings=("--no-cuda",), dest="no_cuda", default=False, action="store_true"),
    ArgOption(option_strings=("--seed",), dest="seed", type="int", default=0),
)


@pytest.fixture
def cutout_spec() -> ReproSpec:
    return ReproSpec.model_validate_json(SPEC.read_text())


@pytest.fixture
def cutout_claims(cutout_spec: ReproSpec) -> dict[str, ResultClaim]:
    return {c.claim_id: c for c in cutout_spec.claims}


@pytest.fixture
def cutout_evidence(cutout_spec: ReproSpec) -> dict[str, Evidence]:
    paper, _ = spec_to_evidence(cutout_spec, "arxiv:1708.04552")
    train = make_evidence(
        kind="repo_file",
        source=SOURCE,
        locator="train.py",
        content={"path": "train.py", "head": "..."},
        confidence="observed",
        authority="repo",
    )
    readme = make_evidence(
        kind="readme",
        source=SOURCE,
        locator="README.md",
        content={"path": "README.md", "text": COMMAND},
        confidence="observed",
        authority="repo",
    )
    repo = [
        train,
        readme,
        make_evidence(
            kind="repo_file",
            source=SOURCE,
            locator="train.py#parsed",
            content={"parser": "entrypoint", "result": {"path": "train.py"}},
            confidence="derived",
            authority="repo",
            links=(train.evidence_id,),
        ),
        make_evidence(
            kind="repo_file",
            source=SOURCE,
            locator="README.md#commands",
            content={
                "parser": "commands",
                "result": [
                    {"line": COMMAND, "path": "train.py", "source": "README.md"},
                    {"line": C100_COMMAND, "path": "train.py", "source": "README.md"},
                ],
            },
            confidence="derived",
            authority="repo",
            links=(readme.evidence_id,),
        ),
        make_evidence(
            kind="repo_file",
            source=SOURCE,
            locator="train.py#metric_lines",
            content={
                "parser": "metric_lines",
                "result": [{"line": METRIC_LINE, "line_no": 221, "path": "train.py"}],
            },
            confidence="derived",
            authority="repo",
            links=(train.evidence_id,),
        ),
    ]
    return {ev.evidence_id: ev for ev in (*paper, *repo)}


def by_locator(evidence: dict[str, Evidence], locator: str) -> str:
    return next(i for i, ev in evidence.items() if ev.locator == locator)


@pytest.fixture
def cutout_mappings(cutout_evidence: dict[str, Evidence]) -> dict[str, ClaimMapping]:
    common: dict[str, Any] = {
        "entrypoint": "train.py",
        "base_command_evidence_id": by_locator(cutout_evidence, "README.md#commands"),
        "base_command": COMMAND,
        "metric_line_evidence_id": by_locator(cutout_evidence, "train.py#metric_lines"),
        "metric_line": METRIC_LINE,
        "conversion": "accuracy_fraction_to_error_percent",
    }
    hyper = (
        SetupArg(setup="epochs", flag="--epochs"),
        SetupArg(setup="batch_size", flag="--batch_size"),
        SetupArg(setup="learning_rate", flag="--learning_rate"),
    )
    return {
        C10: ClaimMapping(
            claim_id=C10,
            claim_args=(ClaimArg(flag="--dataset", value="cifar10"),),
            setup_args=(SetupArg(setup="cutout_length_cifar10", flag="--length"), *hyper),
            **common,
        ),
        C100: ClaimMapping(
            claim_id=C100,
            claim_args=(ClaimArg(flag="--dataset", value="cifar100"),),
            setup_args=(SetupArg(setup="cutout_length_cifar100", flag="--length"), *hyper),
            **common,
        ),
    }


@pytest.fixture
def paper_setup(cutout_evidence: dict[str, Evidence]) -> list[Evidence]:
    return [ev for ev in cutout_evidence.values() if ev.kind == "paper_setup"]
