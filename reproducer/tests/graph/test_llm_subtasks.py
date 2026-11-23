"""Tests for the gated LLM subtasks (version scan, context building, run_subtask) with FakeLLM."""

from __future__ import annotations

import json
from typing import Any

import pytest

from reproducer.domain.evidence import Evidence
from reproducer.domain.inventory import FileInfo
from reproducer.domain.unknowns import fold_unknowns, unknown_id_for
from reproducer.graph.nodes.configure import classify_files, inventory_to_evidence
from reproducer.graph.nodes.subtasks import (
    build_subtask_context,
    command_evidence,
    extract_repo_facts,
    find_example_commands,
    find_metric_lines,
    find_version_mentions,
    load_repo_facts,
    metric_line_evidence,
    run_subtask,
    version_evidence,
)
from reproducer.providers.llm import FakeLLM

SRC = "repo:org/proj@" + "a" * 40
README = """# Cutout
## Dependencies
[PyTorch v0.4.0](http://pytorch.org/)
## Usage
`python train.py --dataset cifar10 --model resnet18 --cutout --length 16`
"""
TRAIN = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--length', type=int, default=16)

def main():
    print('test_acc: %.3f' % 0.9)

if __name__ == '__main__':
    main()
"""
UTIL = "def helper():\n    return 1\n"

Ctx = tuple[dict[str, Evidence], dict[str, Any], dict[str, str]]

# --------------------------------------------------------------------------- version scan


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[PyTorch v0.4.0](http://pytorch.org/)", [("torch", "0.4.0")]),
        ("Tested with python 3.6 on Ubuntu", [("python", "3.6")]),
        ("CUDA 9.0 and cuDNN 7", [("cuda", "9.0"), ("cudnn", "7")]),
        ("torch==1.7.1\ntorchvision>=0.8.2", [("torch", "1.7.1"), ("torchvision", "0.8.2")]),
        ("Requires Python3.8 and numpy version 1.21", [("python", "3.8"), ("numpy", "1.21")]),
        ("TensorFlow: v1.15", [("tensorflow", "1.15")]),
    ],
)
def test_find_version_mentions(text: str, expected: list[tuple[str, str]]) -> None:
    got = find_version_mentions(text)
    assert [(m.package, m.version) for m in got] == expected
    for m in got:
        assert m.quote in text and m.version in m.quote


@pytest.mark.parametrize(
    "text",
    [
        "We train for 200 epochs with batch size 128 and lr 0.1; error 3.6% (mean of 5 runs).",
        "see http://pytorch.org/ and torch.nn 3 layers",
        "pytorch_lightning 1.5 and torch2trt",
        "Python 1.2.3.4.5 is not a version",
        "",
    ],
)
def test_find_version_mentions_no_false_positives(text: str) -> None:
    assert find_version_mentions(text) == ()


# --------------------------------------------------------------------------- context


def _evidence() -> dict[str, Evidence]:
    files = {"README.md": README, "train.py": TRAIN, "util/misc.py": UTIL}
    inv = classify_files(
        [FileInfo(path=p, size=len(t), is_symlink=False) for p, t in files.items()], "a" * 40
    )
    texts = {p: (t, False) for p, t in files.items()}
    evs = (*inventory_to_evidence(inv, SRC, texts), *extract_repo_facts(inv, texts, SRC))
    out = {ev.evidence_id: ev for ev in evs}
    out.update((ev.evidence_id, ev) for ev in version_evidence(out))
    out.update((ev.evidence_id, ev) for ev in metric_line_evidence(out))
    out.update((ev.evidence_id, ev) for ev in command_evidence(out))
    return out


@pytest.fixture
def ctx() -> Ctx:
    evidence = _evidence()
    facts, scan = load_repo_facts(evidence)
    context, packed = build_subtask_context(evidence, facts, scan)
    return evidence, context, packed  # type: ignore[return-value]


def _id_of(evidence: dict[str, Evidence], locator: str) -> str:
    return next(
        ev.evidence_id
        for ev in evidence.values()
        if ev.locator == locator and ev.confidence == "observed"
    )


def test_version_evidence_derived_and_linked() -> None:
    evidence = _evidence()
    readme_id = _id_of(evidence, "README.md")
    (ev,) = [e for e in evidence.values() if e.locator.endswith("#versions")]
    assert (ev.locator, ev.kind, ev.confidence, ev.authority) == (
        "README.md#versions",
        "repo_file",
        "derived",
        "repo",
    )
    assert ev.links == (readme_id,)
    assert ev.content["result"] == [
        {"package": "torch", "version": "0.4.0", "quote": "[PyTorch v0.4.0](http://pytorch.org/)"}
    ]


def test_load_repo_facts_rebuilds_scan() -> None:
    facts, scan = load_repo_facts(_evidence())
    assert [c.path for c in scan.candidates] == ["train.py"]
    assert scan.candidates[0].arg_names == ("--dataset", "--model", "--cutout", "--length")
    assert all(ev.locator.endswith("#parsed") for ev in facts)


def test_build_context_aliases_and_texts(ctx: Ctx) -> None:
    evidence, context, packed = ctx
    readme_id, train_id = _id_of(evidence, "README.md"), _id_of(evidence, "train.py")
    assert context["repo_files"] == frozenset({"README.md", "train.py", "util/misc.py"})
    metric_id = next(i for i, e in evidence.items() if e.locator == "train.py#metric_lines")
    cmd_id = next(i for i, e in evidence.items() if e.locator == "README.md#commands")
    assert context["evidence_ids"] == frozenset({readme_id, train_id, metric_id, cmd_id})
    assert context["aliases"] == {"E1": readme_id, "E2": train_id, "E3": metric_id, "E4": cmd_id}
    assert context["evidence_kinds"] == {
        readme_id: "readme",
        train_id: "repo_file",
        metric_id: "repo_file",
        cmd_id: "repo_file",
    }
    assert context["command_evidence"] == {"train.py": frozenset({cmd_id})}
    cmd = "python train.py --dataset cifar10 --model resnet18 --cutout --length 16"
    assert context["texts"][cmd_id] == cmd
    assert context["metric_line_evidence"] == {"train.py": metric_id}
    assert context["texts"][metric_id] == "print('test_acc: %.3f' % 0.9)"
    assert context["texts"][readme_id] == README
    assert context["texts"][train_id] == TRAIN
    assert context["entrypoint_args"] == {
        "train.py": frozenset({"--dataset", "--model", "--cutout", "--length"})
    }
    for kind in ("A", "B", "C"):
        p = packed[kind]
        assert readme_id not in p and train_id not in p  # aliases only
        assert p.index("## README.md [E1]") < p.index("## Repo facts") < p.index("## train.py [E2]")
        assert "train.py: args ['--dataset', '--model', '--cutout', '--length']" in p
    assert "## Verified version mentions" in packed["A"]
    assert '- torch 0.4.0 [E1] quote: "[PyTorch v0.4.0](http://pytorch.org/)"' in packed["A"]
    assert "Verified version mentions" not in packed["B"] + packed["C"]
    assert "## Verified metric output lines" in packed["C"]
    assert "- train.py:9 [E3] quote: \"print('test_acc: %.3f' % 0.9)\"" in packed["C"]
    assert "Verified metric output lines" not in packed["A"] + packed["B"]
    assert f'## Verified example commands (found by code)\ntrain.py:\n- [E4] "{cmd}"' in packed["B"]
    assert "Verified example commands" not in packed["A"] + packed["C"]


def test_build_context_respects_max_chars() -> None:
    evidence = _evidence()
    facts, scan = load_repo_facts(evidence)
    _, packed = build_subtask_context(evidence, facts, scan, max_chars=300)
    for p in packed.values():
        assert len(p) <= 300
        assert p.startswith("## README.md [E1]")


def test_build_context_requires_single_inventory() -> None:
    with pytest.raises(ValueError, match="exactly one inventory"):
        build_subtask_context({}, (), load_repo_facts({})[1])


# --------------------------------------------------------------------------- run_subtask


def _run(kind: Any, responses: list[str], context: dict[str, Any], packed: str, **kw: Any) -> Any:
    return run_subtask(
        kind,
        FakeLLM(responses),
        context,
        packed,
        source=SRC,
        existing_unknown_ids=kw.pop("existing", set()),
        max_tokens=500,
        **kw,
    )


def test_subtask_a_ok_evidence_and_unknowns(ctx: Ctx) -> None:
    evidence, context, packed = ctx
    readme_id = _id_of(evidence, "README.md")
    answer = {
        "python": None,
        "packages": [
            {
                "name": "torch",
                "version": "v0.4.0",
                "citation": {"evidence_id": "E1", "quote": "pytorch v0.4.0"},
            }
        ],
        "gpu_required": None,
        "unknowns": [
            {"name": "python_version", "why": "not stated"},
            {"name": "cuda_version", "why": "not stated"},
            {"name": "python_version", "why": "duplicate in same answer"},
        ],
    }
    out = _run(
        "A",
        [f"```json\n{json.dumps(answer)}\n```"],
        context,
        packed["A"],
        existing={unknown_id_for("cuda_version")},
    )
    assert out.ok and out.attempts == 1 and out.errors == ()
    (ev,) = out.evidence
    assert (ev.kind, ev.locator, ev.confidence, ev.authority) == (
        "env_spec",
        "subtask:A",
        "derived",
        "llm",
    )
    assert ev.source == SRC and ev.links == (readme_id,)
    pkg = ev.content["packages"][0]
    assert pkg["source"] == "readme"
    assert pkg["citation"] == {"evidence_id": readme_id, "quote": "PyTorch v0.4.0"}  # verbatim
    assert [e.unknown_id for e in out.events] == ["unk:python_version"]
    assert out.events[0].evidence_id == ev.evidence_id
    states = fold_unknowns(out.events, {ev.evidence_id: ev})
    assert states["unk:python_version"].status == "open"


def test_subtask_a_feedback_has_example_shape(ctx: Ctx) -> None:
    _, context, packed = ctx
    bad = {"packages": [{"name": "torch", "version": "1.0", "citation": {"evidence_id": "E1"}}]}
    llm = FakeLLM([json.dumps(bad), json.dumps({"packages": []})])
    out = run_subtask(
        "A",
        llm,
        context,
        packed["A"],
        source=SRC,
        existing_unknown_ids=set(),
        max_tokens=500,
    )
    assert out.ok and out.attempts == 2
    retry_prompt = llm.prompts[1][1]
    assert "packages.0.citation.quote: Field required" in retry_prompt
    assert 'e.g. citation: {"evidence_id": "E1", "quote": "<copy exact text>"}' in retry_prompt
    assert retry_prompt.count("e.g. citation:") == 1


def test_subtask_b_retries_on_duplicates_and_invented_arg(ctx: Ctx) -> None:
    evidence, context, packed = ctx
    readme_id = _id_of(evidence, "README.md")
    ex = {"evidence_id": "E4", "quote": "python train.py --dataset cifar10"}
    dup = {
        "entrypoints": [
            {"path": "train.py", "kind": "train", "example": ex},
            {"path": "train.py", "kind": "eval", "example": ex},
        ]
    }
    bad_arg = {
        "entrypoints": [
            {"path": "train.py", "kind": "train", "args_used": ["--epochs"], "example": ex}
        ]
    }
    good = {
        "entrypoints": [
            {
                "path": "train.py",
                "kind": "train",
                "example": {
                    "evidence_id": "E4",
                    "quote": "python train.py --dataset cifar10 --model resnet18 --cutout",
                },
            }
        ],
        "unknowns": [],
    }
    out = _run("B", [json.dumps(d) for d in (dup, bad_arg, good)], context, packed["B"])
    assert out.ok and out.attempts == 3
    assert "duplicate entrypoint path" in out.errors[0]
    assert "args_used is derived from the example command (['--dataset'])" in out.errors[1]
    assert out.value.entrypoints[0].args_used == ("--dataset", "--model", "--cutout")
    assert out.evidence[0].content["entrypoints"][0]["args_used"] == [
        "--dataset",
        "--model",
        "--cutout",
    ]
    cmd_id = context["aliases"]["E4"]
    assert out.evidence[0].kind == "entrypoints" and out.evidence[0].links == (cmd_id,)
    assert readme_id not in out.evidence[0].links
    assert out.events == ()


def test_subtask_b_example_from_readme_rejected(ctx: Ctx) -> None:
    _, context, packed = ctx
    answer = {
        "entrypoints": [
            {
                "path": "train.py",
                "kind": "train",
                "example": {"evidence_id": "E1", "quote": "python train.py --dataset cifar10"},
            }
        ]
    }
    out = _run("B", [json.dumps(answer)], context, packed["B"], max_retries=0)
    assert not out.ok
    assert "cite a verified command for train.py: evidence_id one of ['E4']" in out.errors[0]


def test_subtask_b_unknowns_rejected(ctx: Ctx) -> None:
    _, context, packed = ctx
    for name, fragment in [
        ("shake_shake_regularization_network", "Input should be"),
        ("python_version", "takes no unknowns"),
    ]:
        junk = {
            "entrypoints": [{"path": "train.py", "kind": "train"}],
            "unknowns": [{"name": name, "why": "x"}],
        }
        out = _run("B", [json.dumps(junk)], context, packed["B"], max_retries=0)
        assert not out.ok and "unknowns.0" in out.errors[0] and fragment in out.errors[0]
        assert 'e.g. unknowns: {"name": "<an allowed name>"' in out.errors[0]


def test_subtask_c_rejects_env_unknown(ctx: Ctx) -> None:
    _, context, packed = ctx
    answer = {"unknowns": [{"name": "python_version", "why": "x"}]}
    out = _run("C", [json.dumps(answer)], context, packed["C"], max_retries=0)
    assert not out.ok and "not allowed here" in out.errors[0]
    assert "dataset_source" in out.errors[0]


def test_subtask_a_drops_version_unknown_when_version_cited(ctx: Ctx) -> None:
    _, context, packed = ctx
    answer = {
        "packages": [
            {
                "name": "PyTorch",
                "version": "0.4.0",
                "citation": {"evidence_id": "E1", "quote": "PyTorch v0.4.0"},
            },
            {"name": "tqdm"},
        ],
        "unknowns": [
            {"name": "torch_version", "why": "x"},
            {"name": "python_version", "why": "not stated"},
        ],
    }
    out = _run("A", [json.dumps(answer)], context, packed["A"])
    assert out.ok and out.dropped_unknowns == ("torch_version",)
    assert [u.name for u in out.value.unknowns] == ["python_version"]
    assert [u["name"] for u in out.evidence[0].content["unknowns"]] == ["python_version"]
    assert [e.unknown_id for e in out.events] == ["unk:python_version"]


def test_subtask_a_keeps_version_unknown_without_cited_version(ctx: Ctx) -> None:
    _, context, packed = ctx
    answer = {
        "packages": [{"name": "torch", "version": None, "citation": None}],
        "unknowns": [{"name": "torch_version", "why": "not pinned"}],
    }
    out = _run("A", [json.dumps(answer)], context, packed["A"])
    assert out.ok and out.dropped_unknowns == ()
    assert [e.unknown_id for e in out.events] == ["unk:torch_version"]


# --------------------------------------------------------------------------- metric lines

CUTOUT_LIKE = """import tqdm
# print('test_acc: old')
print(args)
for epoch in range(10):
    xentropy_loss_avg += loss.item()
    progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg))
    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))
    logger.info("val_loss %f", v)
    self.logger.debug('Top1 %s', t)
    writer.add_scalar('train/acc', acc, epoch)
    logging.warning("mAP = %.2f", m)
    print('saving model')
    x = np.log(err)
"""


def test_find_metric_lines_cutout_like() -> None:
    got = find_metric_lines({"train.py": CUTOUT_LIKE})
    assert [(m.line_no, m.line) for m in got] == [
        (8, "tqdm.write('test_acc: %.3f' % (test_acc))"),
        (9, 'logger.info("val_loss %f", v)'),
        (10, "self.logger.debug('Top1 %s', t)"),
        (11, "writer.add_scalar('train/acc', acc, epoch)"),
        (12, 'logging.warning("mAP = %.2f", m)'),
    ]
    assert all(m.path == "train.py" for m in got)


def test_find_metric_lines_sorted_by_path_and_empty() -> None:
    got = find_metric_lines({"b.py": "print('acc', a)", "a.py": "print('loss', l)", "c.py": ""})
    assert [(m.path, m.line_no) for m in got] == [("a.py", 1), ("b.py", 1)]


def test_metric_line_evidence_derived_and_linked() -> None:
    evidence = _evidence()
    train_id = _id_of(evidence, "train.py")
    (ev,) = [e for e in evidence.values() if e.locator.endswith("#metric_lines")]
    assert (ev.locator, ev.kind, ev.confidence, ev.authority, ev.links) == (
        "train.py#metric_lines",
        "repo_file",
        "derived",
        "repo",
        (train_id,),
    )
    assert ev.content["result"] == [
        {"path": "train.py", "line_no": 9, "line": "print('test_acc: %.3f' % 0.9)"}
    ]


def test_subtask_c_links_sorted_and_deduped(ctx: Ctx) -> None:
    evidence, context, packed = ctx
    readme_id, train_id = _id_of(evidence, "README.md"), _id_of(evidence, "train.py")
    answer = {
        "train_scripts": ["train.py"],
        "datasets": [
            {"name": "cifar10", "citation": {"evidence_id": "E1", "quote": "cifar10"}},
            {"name": "c10", "citation": {"evidence_id": readme_id, "quote": "--dataset cifar10"}},
        ],
        "metric_outputs": [
            {
                "path": "train.py",
                "metric": "test_acc",
                "citation": {"evidence_id": "E3", "quote": "print('test_acc: %.3f' % 0.9)"},
            }
        ],
        "unknowns": [{"name": "dataset_source", "why": "not given"}],
    }
    out = _run("C", [json.dumps(answer)], context, packed["C"])
    assert out.ok
    assert out.evidence[0].kind == "workflow"
    metric_id = context["metric_line_evidence"]["train.py"]
    assert out.evidence[0].links == tuple(sorted({readme_id, metric_id}))
    assert train_id not in out.evidence[0].links
    assert [e.name for e in out.events] == ["dataset_source"]


def test_subtask_c_metric_cited_from_script_rejected(ctx: Ctx) -> None:
    _, context, packed = ctx
    quote = "print('test_acc: %.3f' % 0.9)"
    answer = {
        "metric_outputs": [
            {
                "path": "train.py",
                "metric": "test_acc",
                "citation": {"evidence_id": "E2", "quote": quote},
            }
        ]
    }
    out = _run("C", [json.dumps(answer)], context, packed["C"], max_retries=0)
    assert not out.ok
    assert "cite the verified metric line for train.py with evidence_id 'E3'" in out.errors[0]


def test_subtask_fails_after_retries_no_evidence(ctx: Ctx) -> None:
    _, context, packed = ctx
    invented = json.dumps({"entrypoints": [{"path": "main.py", "kind": "train"}]})
    out = _run("B", ["not json", invented, "{}"], context, packed["B"])
    assert not out.ok and out.value is None
    assert out.evidence == () and out.events == ()
    assert out.attempts == 3 and len(out.errors) == 3
    assert "file not in repo" in out.errors[1]


@pytest.mark.parametrize("missing", ["texts", "aliases", "evidence_kinds", "entrypoint_args"])
def test_subtask_requires_mapping_context(ctx: Ctx, missing: str) -> None:
    _, context, packed = ctx
    broken = {k: v for k, v in context.items() if k != missing}
    with pytest.raises(ValueError, match=repr(missing)):
        _run("A", [], broken, packed["A"])


def test_subtask_unknown_kind(ctx: Ctx) -> None:
    _, context, packed = ctx
    with pytest.raises(ValueError, match="unknown subtask kind"):
        _run("D", [], context, packed["A"])


def test_feedback_examples_are_valid_shapes() -> None:
    from reproducer.graph.nodes.subtasks import SUBTASK_EXAMPLES

    ex = SUBTASK_EXAMPLES["entrypoints"]
    assert '"example": {"evidence_id": "E1"' in ex and '"example": null' in ex
    assert "args_used" not in ex


# --------------------------------------------------------------------------- example commands

CANDS = frozenset({"train.py", "tools/eval.py", "pkg/run/__main__.py", "pkg/mod.py"})


def _cmds(text: str, source: str = "README.md") -> list[tuple[str, str]]:
    return [(c.path, c.line) for c in find_example_commands({source: text}, CANDS)]


def test_commands_fenced_block_with_continuations() -> None:
    readme = """## Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \\
    --dataset cifar10 \\
    --cutout
$ python3 -u ./tools/eval.py --ckpt best.pt  # evaluate
```
"""
    assert _cmds(readme) == [
        ("train.py", "CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --cutout"),
        ("tools/eval.py", "python3 -u ./tools/eval.py --ckpt best.pt  # evaluate"),
    ]


def test_commands_plain_line_and_inline_code() -> None:
    text = (
        "python train.py --epochs 10\nThen run `python tools/eval.py --split test` to evaluate.\n"
    )
    assert _cmds(text) == [
        ("train.py", "python train.py --epochs 10"),
        ("tools/eval.py", "python tools/eval.py --split test"),
    ]


def test_commands_python_m_maps_to_candidate() -> None:
    text = "python -m pkg.run --x 1\npython -m pkg.mod\npython -m other.module\n"
    assert _cmds(text) == [
        ("pkg/run/__main__.py", "python -m pkg.run --x 1"),
        ("pkg/mod.py", "python -m pkg.mod"),
    ]


def test_commands_non_candidate_and_non_commands_ignored() -> None:
    text = (
        "python setup.py install\n"
        "pip install -r requirements.txt\n"
        "bash scripts/download.sh\n"
        "We use python train.py style scripts.\n"
        "train.py --dataset cifar10\n"
        "python ../train.py\n"
    )
    assert _cmds(text) == []


def test_commands_in_shell_script_relative_to_its_dir() -> None:
    sh = "#!/bin/bash\nset -e\nfor s in 1 2; do\n  python eval.py --seed $s\ndone\n"
    got = find_example_commands({"tools/run_eval.sh": sh}, CANDS)
    assert [(c.path, c.line, c.source) for c in got] == [
        ("tools/eval.py", "python eval.py --seed $s", "tools/run_eval.sh")
    ]


CUTOUT_README = """# Cutout
## Results and Usage
### CIFAR-10 and CIFAR-100
Train ResNet-18 on CIFAR-10 with data augmentation and cutout:
`python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16`

Train ResNet-18 on CIFAR-100 with data augmentation and cutout:
`python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8`
```
@article{devries2017cutout,
  title={Improved Regularization of Convolutional Neural Networks with Cutout},
}
```
"""


def test_commands_cutout_like_readme() -> None:
    got = find_example_commands({"README.md": CUTOUT_README}, frozenset({"train.py"}))
    flags = "--model resnet18 --data_augmentation --cutout --length"
    assert [c.line for c in got] == [
        f"python train.py --dataset cifar10 {flags} 16",
        f"python train.py --dataset cifar100 {flags} 8",
    ]
    assert {c.path for c in got} == {"train.py"} and {c.source for c in got} == {"README.md"}


def test_command_evidence_derived_and_linked() -> None:
    evidence = _evidence()
    readme_id = _id_of(evidence, "README.md")
    (ev,) = [e for e in evidence.values() if e.locator.endswith("#commands")]
    assert (ev.locator, ev.confidence, ev.authority, ev.links) == (
        "README.md#commands",
        "derived",
        "repo",
        (readme_id,),
    )
    assert ev.content["result"] == [
        {
            "path": "train.py",
            "line": "python train.py --dataset cifar10 --model resnet18 --cutout --length 16",
            "source": "README.md",
        }
    ]
