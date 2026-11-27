"""Dataset provisioning plan: trusted registry, spec x scan selection, container mount plan.

Pure domain: no IO. Only datasets in KNOWN_DATASETS can be downloaded, and only by our own
trusted script (providers/datasets.py); repo code never downloads anything (it runs with
--network none). Mount plan rules (repo cwd is /work):
  - relative literal root -> /work/<root> (normalized); absolute literal root -> as is;
  - a root that is not a str literal, is home-relative, resolves to /work itself, escapes into
    a reserved path (/, /out, /tmp, /work) or contains ':' / ',' -> NO mount; the call is
    listed as unresolved and the `dataset_source` Unknown stays open (never guessed).
All mounts are read-only (docker_runner.run_args mounts data dirs :ro).
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from reproducer.domain.data_scan import DatasetCall
from reproducer.domain.evidence import Evidence, make_evidence

WORK = "/work"
DATASET_SOURCE_UNKNOWN = "dataset_source"
_RESERVED = ("/", "/out", "/tmp", WORK)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetEntry(_Frozen):
    name: str  # normalized registry name
    cls: str  # torchvision.datasets class
    splits: tuple[dict[str, Any], ...]  # constructor kwargs per split (besides root/download)
    paths: tuple[str, ...]  # files / dirs under the shared root owned by this dataset


def normalize_dataset_name(name: str) -> str:
    """'CIFAR-10' -> 'cifar10', 'Fashion-MNIST' -> 'fashionmnist', 'CIFAR100' -> 'cifar100'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _entry(name: str, cls: str, splits: Sequence[dict[str, Any]], *paths: str) -> DatasetEntry:
    return DatasetEntry(name=name, cls=cls, splits=tuple(splits), paths=paths)


_TRAIN_TEST = ({"train": True}, {"train": False})
_SPLIT_TRAIN_TEST = ({"split": "train"}, {"split": "test"})
KNOWN_DATASETS: Mapping[str, DatasetEntry] = {
    e.name: e
    for e in (
        _entry("cifar10", "CIFAR10", _TRAIN_TEST, "cifar-10-batches-py", "cifar-10-python.tar.gz"),
        _entry("cifar100", "CIFAR100", _TRAIN_TEST, "cifar-100-python", "cifar-100-python.tar.gz"),
        _entry("svhn", "SVHN", _SPLIT_TRAIN_TEST, "train_32x32.mat", "test_32x32.mat"),
        _entry("mnist", "MNIST", _TRAIN_TEST, "MNIST"),
        _entry("fashionmnist", "FashionMNIST", _TRAIN_TEST, "FashionMNIST"),
        _entry("stl10", "STL10", _SPLIT_TRAIN_TEST, "stl10_binary", "stl10_binary.tar.gz"),
    )
}
_BY_CLASS: Mapping[str, str] = {e.cls: e.name for e in KNOWN_DATASETS.values()}


def registry_name_for_class(cls: str) -> str | None:
    return _BY_CLASS.get(cls)


class DatasetSelection(_Frozen):
    selected: tuple[str, ...]  # registry names: spec AND scan (downloadable)
    warnings: tuple[str, ...]


def select_datasets(spec_datasets: Iterable[str], calls: Iterable[DatasetCall]) -> DatasetSelection:
    """Spec datasets (normalized) intersected with the registry names of scanned classes.
    Everything else is a warning and is NOT downloaded."""
    spec = list(dict.fromkeys(normalize_dataset_name(d) for d in spec_datasets))
    scan_classes = list(dict.fromkeys(c.cls for c in calls))
    scan_names = {registry_name_for_class(c) for c in scan_classes} - {None}
    warnings: list[str] = []
    for cls in scan_classes:
        name = registry_name_for_class(cls)
        if name is None:
            warnings.append(f"scan class {cls} is not in the dataset registry; not downloaded")
        elif name not in spec:
            warnings.append(f"scan class {cls} ({name}) is not a spec dataset; not downloaded")
    selected: list[str] = []
    for name in spec:
        if name not in KNOWN_DATASETS:
            warnings.append(f"spec dataset {name} is not in the dataset registry; not downloaded")
        elif name not in scan_names:
            warnings.append(f"spec dataset {name} not found by the scan; not downloaded")
        else:
            selected.append(name)
    return DatasetSelection(selected=tuple(selected), warnings=tuple(warnings))


class DataMount(_Frozen):
    container: str
    host: str
    calls: tuple[str, ...]  # "path:line cls" that use this root


class UnresolvedRoot(_Frozen):
    call: str  # "path:line cls"
    reason: str


class DataMountPlan(_Frozen):
    mounts: tuple[DataMount, ...] = ()
    unresolved: tuple[UnresolvedRoot, ...] = ()

    def as_data_mounts(self) -> dict[str, str]:
        """host -> container for docker_runner (at most one target per host dir)."""
        if len({m.host for m in self.mounts}) != len(self.mounts):
            raise ValueError("several container targets for one host dir; mount them one by one")
        return {m.host: m.container for m in self.mounts}


def container_root(root: str | None, reason: str | None = None) -> tuple[str | None, str]:
    """Literal root -> (container path, '') or (None, why not)."""
    if root is None:
        return None, reason or "root unknown"
    if ":" in root or "," in root:
        return None, f"root {root!r} contains ':' or ','"
    if root.startswith("~"):
        return None, f"home-relative root {root!r}"
    if not root.strip():
        return None, "empty root"
    target = posixpath.normpath(root if posixpath.isabs(root) else posixpath.join(WORK, root))
    if target.startswith("//"):
        target = "/" + target.lstrip("/")
    if target in _RESERVED or target.startswith("/out/"):
        return None, f"root {root!r} resolves to reserved path {target}"
    return target, ""


def plan_data_mounts(
    calls: Iterable[DatasetCall], selected: Iterable[str], host_root: str
) -> DataMountPlan:
    """Mount plan for the calls whose class is a selected dataset: the shared cache root
    (host_root) mounted read-only at each distinct literal root. Unresolved -> no mount."""
    wanted = set(selected)
    targets: dict[str, list[str]] = {}
    unresolved: list[UnresolvedRoot] = []
    for c in calls:
        if registry_name_for_class(c.cls) not in wanted:
            continue
        label = f"{c.path}:{c.line} {c.cls}"
        target, why = container_root(c.root, c.reason)
        if target is None:
            unresolved.append(UnresolvedRoot(call=label, reason=why))
        else:
            targets.setdefault(target, []).append(label)
    mounts = tuple(
        DataMount(container=t, host=host_root, calls=tuple(labels))
        for t, labels in sorted(targets.items())
    )
    return DataMountPlan(mounts=mounts, unresolved=tuple(unresolved))


def dataset_cache_evidence(
    *,
    name: str,
    files: Sequence[Mapping[str, Any]],
    verified: bool,
    split_sizes: Sequence[int],
    source: str,
) -> Evidence:
    """Observed execution evidence that a dataset is in the shared cache. files: [{path (relative
    to the cache root), size, sha256}]; verified: torchvision's own integrity check passed in the
    trusted container (network none)."""
    entry = KNOWN_DATASETS[name]
    return make_evidence(
        kind="dataset_cache",
        source=source,
        locator=f"dataset_cache:{name}",
        content={
            "name": name,
            "cls": entry.cls,
            "files": [dict(f) for f in files],
            "verified": verified,
            "split_sizes": list(split_sizes),
        },
        confidence="observed",
        authority="execution",
    )


class DatasetSourceDecision(_Frozen):
    value: str | None  # None -> dataset_source stays open
    resolver: str | None  # evidence_id of the dataset_cache evidence that resolves it
    reasons: tuple[str, ...]  # why it stays open (empty when value is set)


def decide_dataset_source(
    spec_datasets: Iterable[str],
    calls: Iterable[DatasetCall],
    cache_evidence: Iterable[Evidence],
) -> DatasetSourceDecision:
    """Resolve dataset_source iff EVERY spec dataset (a) is in the registry and has at least
    one scanned call, all of whose roots are literal and mountable, and (b) has verified
    dataset_cache evidence (authority execution, observed). Resolver: the cache evidence of
    the first spec dataset (latest verified record per name)."""
    spec = list(dict.fromkeys(normalize_dataset_name(d) for d in spec_datasets))
    calls = list(calls)
    verified: dict[str, Evidence] = {}
    for ev in cache_evidence:
        if (
            ev.kind == "dataset_cache"
            and ev.authority == "execution"
            and ev.confidence == "observed"
            and ev.content.get("verified") is True
        ):
            verified[str(ev.content.get("name"))] = ev  # later records win
    reasons: list[str] = []
    if not spec:
        reasons.append("no spec datasets")
    for name in spec:
        entry = KNOWN_DATASETS.get(name)
        if entry is None:
            reasons.append(f"{name}: not in the dataset registry")
            continue
        mine = [c for c in calls if c.cls == entry.cls]
        if not mine:
            reasons.append(f"{name}: no scanned {entry.cls} call")
        for c in mine:
            target, why = container_root(c.root, c.reason)
            if target is None:
                reasons.append(f"{name}: {c.path}:{c.line} {c.cls}: {why}")
        if name not in verified:
            reasons.append(f"{name}: no verified dataset_cache evidence")
    if reasons:
        return DatasetSourceDecision(value=None, resolver=None, reasons=tuple(reasons))
    classes = ",".join(KNOWN_DATASETS[n].cls for n in spec)
    return DatasetSourceDecision(
        value=f"torchvision:{classes} (official download, checksum-verified)",
        resolver=verified[spec[0]].evidence_id,
        reasons=(),
    )
