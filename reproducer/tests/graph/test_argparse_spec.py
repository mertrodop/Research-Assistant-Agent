"""argparse spec extraction in the entrypoint scan, and local import modules."""

from __future__ import annotations

from reproducer.domain.import_scan import local_import_modules
from reproducer.graph.nodes.subtasks import ArgSpec, EntrypointCandidate, candidate_entrypoints

TRAIN = """\
import argparse
from util.misc import CSVLogger
from model import resnet
import numpy as np

dataset_options = ['cifar10', 'cifar100', 'svhn']
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--cutout', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('-k', dest='top_k', type=int, default=None)
parser.add_argument('data')
parser.add_argument('--lr-sched', default=compute(), choices=sorted(x))
args = parser.parse_args()
"""


def _args() -> dict[str, ArgSpec]:
    (c,) = candidate_entrypoints({"train.py": TRAIN}).candidates
    return {a.option_strings[0]: a for a in c.arguments}


def test_type_default_action_choices() -> None:
    a = _args()
    assert a["--dataset"] == ArgSpec(
        option_strings=("--dataset", "-d"),
        dest="dataset",
        default="cifar10",
        choices=("cifar10", "cifar100", "svhn"),  # resolved through the module-level literal
    )
    assert (a["--epochs"].type, a["--epochs"].default, a["--epochs"].dest) == ("int", 200, "epochs")
    assert (a["--learning_rate"].type, a["--learning_rate"].default) == ("float", 0.1)
    assert (a["--cutout"].action, a["--cutout"].default) == ("store_true", False)
    assert a["--no-cuda"].dest == "no_cuda" and a["--no-cuda"].default is None
    assert a["-k"].dest == "top_k" and a["-k"].type == "int"
    assert a["data"].dest == "data" and a["data"].action is None
    assert a["--lr-sched"].default is None and a["--lr-sched"].choices is None  # not literals


def test_arguments_in_source_order_and_backward_compatible() -> None:
    (c,) = candidate_entrypoints({"train.py": TRAIN}).candidates
    assert [a.option_strings[0] for a in c.arguments][:3] == [
        "--dataset",
        "--epochs",
        "--learning_rate",
    ]
    old = EntrypointCandidate.model_validate(
        {"path": "t.py", "has_main_guard": False, "uses_argparse": True, "uses_click": False,
         "arg_names": ["--lr"]}
    )  # fmt: skip
    assert old.arguments == ()  # stored evidence without `arguments` still loads


def test_local_import_modules() -> None:
    paths = {"train.py", "util/__init__.py", "util/misc.py", "model/__init__.py", "model/resnet.py"}
    assert local_import_modules("train.py", TRAIN, paths) == ("util.misc", "model", "model.resnet")
    assert local_import_modules("train.py", "print 1", paths) == ()
