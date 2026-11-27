"""Tests for reproducer.domain.data_plan (registry, spec x scan selection, mount plan)."""

from __future__ import annotations

import pytest

from reproducer.domain.data_plan import (
    KNOWN_DATASETS,
    container_root,
    dataset_cache_evidence,
    decide_dataset_source,
    normalize_dataset_name,
    plan_data_mounts,
    registry_name_for_class,
    select_datasets,
)
from reproducer.domain.data_scan import DatasetCall


def _call(cls: str, root: str | None, line: int = 1, reason: str | None = None) -> DatasetCall:
    return DatasetCall(path="train.py", line=line, cls=cls, root=root, reason=reason, download=True)


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("CIFAR-10", "cifar10"),
        ("CIFAR100", "cifar100"),
        ("cifar_100", "cifar100"),
        ("SVHN", "svhn"),
        ("Fashion-MNIST", "fashionmnist"),
        ("STL-10", "stl10"),
        (" MNIST ", "mnist"),
    ],
)
def test_normalize(raw: str, norm: str) -> None:
    assert normalize_dataset_name(raw) == norm
    assert norm in KNOWN_DATASETS


def test_registry() -> None:
    assert set(KNOWN_DATASETS) == {"cifar10", "cifar100", "svhn", "mnist", "fashionmnist", "stl10"}
    assert KNOWN_DATASETS["cifar10"].cls == "CIFAR10"
    assert KNOWN_DATASETS["cifar10"].splits == ({"train": True}, {"train": False})
    assert KNOWN_DATASETS["svhn"].splits == ({"split": "train"}, {"split": "test"})
    assert registry_name_for_class("CIFAR100") == "cifar100"
    assert registry_name_for_class("FashionMNIST") == "fashionmnist"
    assert registry_name_for_class("ImageFolder") is None


def test_select_spec_and_scan_with_warnings() -> None:
    calls = [
        _call("CIFAR10", "data/"),
        _call("CIFAR100", "data/"),
        _call("SVHN", "data/"),
        _call("ImageFolder", "imgs/"),
    ]
    sel = select_datasets(["CIFAR-10", "CIFAR-100", "ImageNet", "MNIST"], calls)
    assert sel.selected == ("cifar10", "cifar100")
    joined = "\n".join(sel.warnings)
    assert "SVHN (svhn) is not a spec dataset; not downloaded" in joined
    assert "ImageFolder is not in the dataset registry" in joined
    assert "spec dataset imagenet is not in the dataset registry" in joined
    assert "spec dataset mnist not found by the scan" in joined


def test_container_root() -> None:
    assert container_root("data/") == ("/work/data", "")
    assert container_root("./data/cifar") == ("/work/data/cifar", "")
    assert container_root("/datasets/cifar") == ("/datasets/cifar", "")
    assert container_root("../data") == ("/data", "")
    for bad in (".", "./", "..", "/", "/out", "/out/x", "/tmp", "~/data", "a:b", ""):
        target, why = container_root(bad)
        assert target is None and why, bad
    assert container_root(None, "non-literal root: x") == (None, "non-literal root: x")


def test_plan_relative_absolute_unresolved() -> None:
    calls = [
        _call("CIFAR10", "data/", 1),
        _call("CIFAR10", "data", 2),
        _call("CIFAR100", "/datasets", 3),
        _call("CIFAR100", None, 4, reason="non-literal root: args.root"),
        _call("SVHN", "svhn_data/", 5),  # not selected -> ignored
    ]
    plan = plan_data_mounts(calls, ["cifar10", "cifar100"], "/cache/tv")
    assert [(m.container, m.host, m.calls) for m in plan.mounts] == [
        ("/datasets", "/cache/tv", ("train.py:3 CIFAR100",)),
        ("/work/data", "/cache/tv", ("train.py:1 CIFAR10", "train.py:2 CIFAR10")),
    ]
    assert [(u.call, u.reason) for u in plan.unresolved] == [
        ("train.py:4 CIFAR100", "non-literal root: args.root")
    ]
    with pytest.raises(ValueError):
        plan.as_data_mounts()  # one host dir, two targets
    single = plan_data_mounts(calls[:2], ["cifar10"], "/cache/tv")
    assert single.as_data_mounts() == {"/cache/tv": "/work/data"}


def test_unresolved_only_gives_no_mount() -> None:
    plan = plan_data_mounts([_call("CIFAR10", None, reason="root not given")], ["cifar10"], "/c")
    assert plan.mounts == () and len(plan.unresolved) == 1


def test_dataset_cache_evidence() -> None:
    ev = dataset_cache_evidence(
        name="cifar10",
        files=[{"path": "cifar-10-batches-py/data_batch_1", "size": 1, "sha256": "x"}],
        verified=True,
        split_sizes=[50000, 10000],
        source="data-cache@img:tag",
    )
    assert (ev.kind, ev.authority, ev.confidence, ev.locator) == (
        "dataset_cache",
        "execution",
        "observed",
        "dataset_cache:cifar10",
    )
    assert ev.content["name"] == "cifar10" and ev.content["verified"] is True


def _cache(name: str, verified: bool = True):  # type: ignore[no-untyped-def]
    return dataset_cache_evidence(
        name=name, files=[], verified=verified, split_sizes=[1, 1], source="data-cache@img"
    )


CUTOUT_CALLS = [_call("CIFAR10", "data/", 1), _call("CIFAR100", "data/", 2), _call("SVHN", None)]
SPEC = ["CIFAR-10", "CIFAR-100"]


def test_dataset_source_all_ok_resolves() -> None:
    c10, c100 = _cache("cifar10"), _cache("cifar100")
    d = decide_dataset_source(SPEC, CUTOUT_CALLS, [c10, c100])
    assert d.value == "torchvision:CIFAR10,CIFAR100 (official download, checksum-verified)"
    assert d.resolver == c10.evidence_id and d.reasons == ()


def test_dataset_source_unresolved_root_stays_open() -> None:
    calls = [*CUTOUT_CALLS, _call("CIFAR100", None, 9, reason="non-literal root: args.root")]
    d = decide_dataset_source(SPEC, calls, [_cache("cifar10"), _cache("cifar100")])
    assert d.value is None and d.resolver is None
    assert d.reasons == ("cifar100: train.py:9 CIFAR100: non-literal root: args.root",)


def test_dataset_source_unverified_or_missing_cache_stays_open() -> None:
    d = decide_dataset_source(SPEC, CUTOUT_CALLS, [_cache("cifar10"), _cache("cifar100", False)])
    assert d.value is None and d.reasons == ("cifar100: no verified dataset_cache evidence",)
    d = decide_dataset_source(SPEC, CUTOUT_CALLS, [_cache("cifar10")])
    assert d.value is None
    d = decide_dataset_source(["CIFAR-10", "MNIST"], CUTOUT_CALLS, [_cache("cifar10")])
    assert "mnist: no scanned MNIST call" in d.reasons
