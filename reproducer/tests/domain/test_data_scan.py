"""Tests for reproducer.domain.data_scan (pure ast scan of torchvision dataset calls)."""

from __future__ import annotations

from reproducer.domain.data_scan import (
    DatasetCall,
    dataset_calls_evidence,
    find_dataset_calls,
    load_dataset_calls,
)

CUTOUT_LIKE = """\
import torch
from torchvision import datasets, transforms

if args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)
    test_dataset = datasets.CIFAR10(root='data/', train=False, download=True)
elif args.dataset == 'cifar100':
    train_dataset = datasets.CIFAR100(root='data/', train=True, download=True)
elif args.dataset == 'svhn':
    train_dataset = datasets.SVHN(root='data/', split='train', download=True)
loader = torch.utils.data.DataLoader(dataset=train_dataset)
"""


def _calls(text: str, path: str = "train.py") -> tuple[DatasetCall, ...]:
    return find_dataset_calls({path: text}).calls


def test_cutout_like_train_py() -> None:
    calls = _calls(CUTOUT_LIKE)
    assert [(c.cls, c.line, c.root, c.download) for c in calls] == [
        ("CIFAR10", 5, "data/", True),
        ("CIFAR10", 9, "data/", True),
        ("CIFAR100", 11, "data/", True),
        ("SVHN", 13, "data/", True),
    ]
    assert all(c.path == "train.py" and c.reason is None for c in calls)


def test_aliases() -> None:
    text = """\
import torchvision.datasets as dsets
import torchvision as tv
import torchvision
from torchvision import datasets as D
from torchvision.datasets import MNIST as M
from torchvision.datasets.cifar import CIFAR100

a = dsets.CIFAR10(root='a')
b = tv.datasets.STL10(root='b', split='train')
c = torchvision.datasets.FashionMNIST(root='c')
d = D.SVHN(root='d')
e = M(root='e')
f = CIFAR100(root='f')
g = dsets.utils.download_url('http://x', 'y')
"""
    got = [(c.cls, c.root) for c in _calls(text)]
    assert got == [
        ("CIFAR10", "a"),
        ("STL10", "b"),
        ("FashionMNIST", "c"),
        ("SVHN", "d"),
        ("MNIST", "e"),
        ("CIFAR100", "f"),
    ]


def test_import_torchvision_datasets_binds_torchvision() -> None:
    text = "import torchvision.datasets\nx = torchvision.datasets.MNIST('m', download=False)\n"
    (c,) = _calls(text)
    assert (c.cls, c.root, c.download) == ("MNIST", "m", False)


def test_positional_root_and_download_defaults() -> None:
    text = """\
from torchvision import datasets
a = datasets.CIFAR10('data/')
b = datasets.CIFAR10('data/', True, None, None, True)
"""
    a, b = _calls(text)
    assert (a.root, a.download) == ("data/", False)  # absent -> torchvision default
    assert (b.root, b.download) == ("data/", None)  # may be positional -> unknown


def test_non_literal_root_has_reason() -> None:
    text = """\
import os
from torchvision import datasets
a = datasets.CIFAR10(root=args.data_dir, download=flag)
b = datasets.CIFAR10(root=os.path.join('x', 'y'))
c = datasets.CIFAR10(f'{base}/data')
d = datasets.CIFAR10(train=True)
e = datasets.CIFAR10(**kw)
"""
    a, b, c, d, e = _calls(text)
    assert a.root is None and a.reason == "non-literal root: args.data_dir" and a.download is None
    assert b.root is None and b.reason == "non-literal root: os.path.join('x', 'y')"
    assert c.root is None and c.reason is not None and c.reason.startswith("non-literal root")
    assert d.root is None and d.reason == "root not given"
    assert e.root is None and e.reason == "root passed via *args/**kwargs"


def test_non_torchvision_calls_ignored() -> None:
    text = """\
import datasets
from mylib import datasets as D
from torchvision import transforms
a = datasets.CIFAR10(root='a')
b = D.MNIST(root='b')
c = transforms.ToTensor()
d = CIFAR10(root='d')
"""
    assert _calls(text) == ()


def test_unparsed_and_evidence_roundtrip() -> None:
    texts = {"train.py": CUTOUT_LIKE, "util/x.py": "x = 1\n", "py2.py": "print 1"}
    scan = find_dataset_calls(texts)
    assert scan.unparsed == ("py2.py",)
    assert scan.scanned == ("train.py", "util/x.py")
    evs = dataset_calls_evidence(scan, "repo@abc", {"train.py": "e" * 64})
    assert [e.locator for e in evs] == [
        "py2.py#dataset_calls",
        "train.py#dataset_calls",
        "util/x.py#dataset_calls",
    ]
    by_loc = {e.locator: e for e in evs}
    assert by_loc["train.py#dataset_calls"].links == ("e" * 64,)
    assert by_loc["util/x.py#dataset_calls"].content["result"] == []
    assert all(
        e.kind == "repo_file" and e.confidence == "derived" and e.authority == "repo" for e in evs
    )
    loaded = load_dataset_calls({e.evidence_id: e for e in evs}, "repo@abc")
    assert loaded == scan
    assert load_dataset_calls({e.evidence_id: e for e in evs}, "repo@other") is None
