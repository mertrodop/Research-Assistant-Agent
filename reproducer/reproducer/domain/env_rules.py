"""Environment planning rules: EnvSpec + GPU facts -> EnvPlan -> Dockerfile text.

Pure domain: no file, network or subprocess IO (nvidia-smi runs in providers/gpu.py).
Deterministic: the same EnvSpec and GPU always give the same plan. Nothing is silently
upgraded: every value that differs from what the repo states is a Deviation, and every value
the repo does not state is an EnvAssumption (it never resolves an Unknown).

Compatibility rule (COMPAT_RULE_SOURCE, NVIDIA Ada compatibility guide, section 1.2):
"A cubin generated for a certain compute capability is supported to run on any GPU with the
same major revision and same or higher minor revision of compute capability." So a torch build
is runnable on a GPU when its arch list has an entry with the GPU's major version and a minor
version <= the GPU's minor. PTX (+PTX) entries are deliberately ignored: JIT from PTX does not
cover prebuilt cuDNN/cuBLAS kernels, so we only count real cubin (sm_XY) targets.
Driver rule (DRIVER_TABLE_SOURCE): an entry is also skipped when the host driver is older than
MIN_DRIVER for its CUDA version.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from reproducer.domain.contracts import EnvSpec, PackageFact
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.import_scan import ImportedPackage, normalize_pip_name


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


ComputeCapability = tuple[int, int]

COMPAT_RULE_SOURCE = (
    "https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html"
    "#application-compatibility-on-the-nvidia-ada-gpu-architecture"
)


# --- GPU facts -----------------------------------------------------------------------------


class GpuInfo(_Frozen):
    name: str = Field(min_length=1)
    compute_capability: ComputeCapability
    driver_version: str = Field(min_length=1)
    memory_mb: int = Field(gt=0)


NVIDIA_SMI_QUERY = "name,compute_cap,driver_version,memory.total"
NVIDIA_SMI_ARGS = ("nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv")
_CC_RE = re.compile(r"^(\d+)\.(\d+)$")
_MEM_RE = re.compile(r"^(\d+)\s*MiB$")


def parse_nvidia_smi_csv(text: str) -> GpuInfo:
    """Parse `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total
    --format=csv` output (header line + one row per GPU). Several GPUs are accepted only if
    they share name and compute capability (the plan must hold for every GPU docker sees);
    the smallest memory is reported. Raises ValueError on anything unexpected."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"nvidia-smi CSV needs a header and at least one row, got {text!r}")
    header = [h.strip() for h in lines[0].split(",")]
    if [h.split(" [")[0] for h in header] != NVIDIA_SMI_QUERY.split(","):
        raise ValueError(f"unexpected nvidia-smi CSV header: {lines[0]!r}")
    gpus: list[GpuInfo] = []
    for row in lines[1:]:
        # name may not contain commas in practice, but split from the right to be safe.
        parts = [p.strip() for p in row.rsplit(",", 3)]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi CSV row: {row!r}")
        name, cc, driver, mem = parts
        cc_m, mem_m = _CC_RE.match(cc), _MEM_RE.match(mem)
        if cc_m is None:
            raise ValueError(f"unparseable compute capability {cc!r} in row {row!r}")
        if mem_m is None:
            raise ValueError(f"unparseable memory {mem!r} in row {row!r}")
        gpus.append(
            GpuInfo(
                name=name,
                compute_capability=(int(cc_m.group(1)), int(cc_m.group(2))),
                driver_version=driver,
                memory_mb=int(mem_m.group(1)),
            )
        )
    first = gpus[0]
    if any((g.name, g.compute_capability) != (first.name, first.compute_capability) for g in gpus):
        raise ValueError(f"mixed GPUs are not supported: {[g.name for g in gpus]}")
    return first.model_copy(update={"memory_mb": min(g.memory_mb for g in gpus)})


# --- Torch table ---------------------------------------------------------------------------


class TorchEntry(_Frozen):
    """One official pytorch/pytorch runtime image.

    python_versions: CPython versions the release supports (RELEASE.md compatibility matrix).
    image_python: the python the image actually ships (docker.Makefile PYTHON_VERSION at the
    release tag); the rendered Dockerfile asserts it at build time.
    torchvision: the torchvision release paired with this torch (torchvision README table for
    the minor, pytorch.org previous-versions install commands for the exact patch).
    arch_list: cubin targets of the CUDA build installed in the image (TORCH_CUDA_ARCH_LIST in
    the official binary build script for that release/CUDA; +PTX entries dropped).
    """

    torch: str
    torchvision: str
    python_versions: tuple[str, ...] = Field(min_length=1)
    image_python: str
    cuda: str
    base_image: str
    arch_list: tuple[ComputeCapability, ...] = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.image_python not in self.python_versions:
            raise ValueError(f"torch {self.torch}: image python {self.image_python} unsupported")
        if not all(s.startswith("https://") for s in self.sources):
            raise ValueError(f"torch {self.torch}: every source must be an https URL")
        return self

    @property
    def version_key(self) -> tuple[int, ...]:
        return version_key(self.torch)

    @property
    def wheel_index(self) -> str:
        """Official PyTorch wheel index for this entry's CUDA build."""
        return f"{WHEEL_INDEX_BASE}/cu{self.cuda.replace('.', '')}"

    @property
    def torchvision_wheel(self) -> str:
        """Exact torchvision wheel matching the image's CUDA torch build."""
        return f"torchvision=={self.torchvision}+cu{self.cuda.replace('.', '')}"


def version_key(version: str) -> tuple[int, ...]:
    """Release version -> int tuple ("1.12.1" -> (1, 12, 1)); "v" prefix, "+local" ignored."""
    v = version.strip().removeprefix("v").removeprefix("V").split("+", 1)[0]
    if not re.fullmatch(r"\d+(\.\d+)*", v):
        raise ValueError(f"not a release version: {version!r}")
    return tuple(int(x) for x in v.split("."))


_MATRIX = "https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix"
_VISION = "https://github.com/pytorch/vision#installation"
_PREVIOUS = "https://pytorch.org/get-started/previous-versions/"
# Checked 2026-11-23: <index>/torchvision/ lists torchvision-<ver>+cu<xy> for the image's
# python for every TORCH_TABLE entry (e.g. torchvision-0.13.1+cu113-cp37-cp37m-linux_x86_64).
WHEEL_INDEX_BASE = "https://download.pytorch.org/whl"
_HUB = "https://hub.docker.com/r/pytorch/pytorch/tags?name="
_GH = "https://github.com/pytorch"
_LEGACY_8X = ((3, 7), (5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6))


def _py(lo: int, hi: int) -> tuple[str, ...]:
    return tuple(f"3.{m}" for m in range(lo, hi + 1))


def _entry(
    torch: str, vision: str, py: tuple[int, int], image_py: str, cuda: str, tag: str,
    arch: tuple[ComputeCapability, ...], build_script: str,
) -> TorchEntry:  # fmt: skip
    return TorchEntry(
        torch=torch,
        torchvision=vision,
        python_versions=_py(*py),
        image_python=image_py,
        cuda=cuda,
        base_image=f"pytorch/pytorch:{tag}",
        arch_list=arch,
        sources=(
            _MATRIX,
            _VISION,
            _PREVIOUS,
            f"{_GH}/pytorch/blob/v{torch}/docker.Makefile",
            f"{_GH}/pytorch/blob/v{torch}/Dockerfile",
            build_script,
            _HUB + tag,
        ),
    )


# Verified 2026-11-23: every tag with `docker manifest inspect`; python/CUDA from the RELEASE.md
# matrix; torchvision from the vision README table + pytorch.org previous-versions commands
# ("torch==X torchvision==Y"); image python from docker.Makefile at the tag (the docker-release
# workflow does not override it); arch lists from the binary build script the image's install
# channel uses (<=2.1: conda channel -> pytorch/builder conda/pytorch-nightly/build.sh; >=2.4:
# pip wheels -> manywheel/build_cuda.sh). Sorted by torch version (oldest first).
TORCH_TABLE: tuple[TorchEntry, ...] = (
    _entry("1.12.1", "0.13.1", (7, 10), "3.7", "11.3", "1.12.1-cuda11.3-cudnn8-runtime",
           _LEGACY_8X, f"{_GH}/builder/blob/release/1.12/conda/pytorch-nightly/build.sh"),
    _entry("1.13.1", "0.14.1", (7, 10), "3.8", "11.6", "1.13.1-cuda11.6-cudnn8-runtime",
           _LEGACY_8X, f"{_GH}/builder/blob/release/1.13/conda/pytorch-nightly/build.sh"),
    _entry("2.0.1", "0.15.2", (8, 11), "3.10", "11.7", "2.0.1-cuda11.7-cudnn8-runtime",
           _LEGACY_8X, f"{_GH}/builder/blob/release/2.0/conda/pytorch-nightly/build.sh"),
    _entry("2.1.2", "0.16.2", (8, 11), "3.10", "11.8", "2.1.2-cuda11.8-cudnn8-runtime",
           ((3, 7), (5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)),
           f"{_GH}/builder/blob/release/2.1/conda/pytorch-nightly/build.sh"),
    _entry("2.4.1", "0.19.1", (8, 12), "3.11", "12.1", "2.4.1-cuda12.1-cudnn9-runtime",
           ((5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)),
           f"{_GH}/builder/blob/release/2.4/manywheel/build_cuda.sh"),
    _entry("2.7.1", "0.22.1", (9, 13), "3.11", "12.6", "2.7.1-cuda12.6-cudnn9-runtime",
           ((5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)),
           f"{_GH}/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh"),
    _entry("2.9.1", "0.24.1", (10, 14), "3.11", "12.8", "2.9.1-cuda12.8-cudnn9-runtime",
           ((7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)),
           f"{_GH}/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh"),
    _entry("2.11.0", "0.26.0", (10, 14), "3.11", "13.0", "2.11.0-cuda13.0-cudnn9-runtime",
           ((7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)),
           f"{_GH}/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh"),
)  # fmt: skip


# --- Driver table --------------------------------------------------------------------------

DRIVER_TABLE_SOURCE = (
    "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver"
)
# Minimum Linux x86_64 driver per CUDA toolkit (checked 2026-11-23). 11.x/12.x: Table 4
# "Corresponding Driver Versions", GA row. 13.0: Table 2 gives driver branch R580 (13.x
# needs >= 580). Conservative: we do not rely on minor-version compatibility (Table 3), which
# would allow older drivers but not PTX JIT.
MIN_DRIVER: Mapping[str, str] = {
    "11.3": "465.19.01",
    "11.6": "510.39.01",
    "11.7": "515.43.04",
    "11.8": "520.61.05",
    "12.1": "530.30.02",
    "12.6": "560.28.03",
    "12.8": "570.26",
    "13.0": "580",
}


def driver_key(version: str) -> tuple[int, ...]:
    """Driver version -> int tuple ("580.178.04" -> (580, 178, 4)); ValueError otherwise."""
    v = version.strip()
    if not re.fullmatch(r"\d+(\.\d+)*", v):
        raise ValueError(f"not a driver version: {version!r}")
    return tuple(int(x) for x in v.split("."))


def driver_supports(driver_version: str, cuda: str) -> bool:
    """True if the host driver meets MIN_DRIVER for this CUDA version. Unknown CUDA -> error."""
    if cuda not in MIN_DRIVER:
        raise ValueError(f"no minimum driver recorded for CUDA {cuda} (add it to MIN_DRIVER)")
    return driver_key(driver_version) >= driver_key(MIN_DRIVER[cuda])


def arch_covers(arch_list: Iterable[ComputeCapability], cc: ComputeCapability) -> bool:
    """Same major, listed minor <= GPU minor (see COMPAT_RULE_SOURCE)."""
    return any(major == cc[0] and minor <= cc[1] for major, minor in arch_list)


def runnable_entries(
    gpu: GpuInfo | None, table: tuple[TorchEntry, ...] = TORCH_TABLE
) -> tuple[TorchEntry, ...]:
    """Entries whose CUDA build covers the GPU's arch AND whose CUDA the host driver supports
    (all entries when gpu is None: CPU-only)."""
    if gpu is None:
        return table
    return tuple(
        e
        for e in table
        if arch_covers(e.arch_list, gpu.compute_capability)
        and driver_supports(gpu.driver_version, e.cuda)
    )


def find_entry(torch: str, table: tuple[TorchEntry, ...] = TORCH_TABLE) -> TorchEntry | None:
    """Table entry for a stated version: exact match, or the newest patch of a stated
    "major.minor" (e.g. "1.12" -> 1.12.1). None if the version is not in the table."""
    try:
        key = version_key(torch)
    except ValueError:
        return None
    hits = [e for e in table if e.version_key[: len(key)] == key]
    return max(hits, key=lambda e: e.version_key) if hits else None


# --- Plan ----------------------------------------------------------------------------------


class NoRunnableTorchError(ValueError):
    """No TORCH_TABLE entry covers the GPU's compute capability with a CUDA the driver supports."""


class Deviation(_Frozen):
    field: str
    stated: str
    chosen: str
    reason: str
    rule_source: tuple[str, ...] = Field(min_length=1)


class EnvAssumption(_Frozen):
    """A value the repo does not state. name is the unknown name ("python_version") or
    "unpinned:<pkg>" (value None: the package stays unpinned, no version is guessed)."""

    name: str
    value: str | None
    reason: str


class EnvPlan(_Frozen):
    base_image: str
    python: str
    torch: str
    torchvision: str  # pinned via the constraints file
    # True when the repo states or imports torchvision: the image must then contain it (the
    # build installs torchvision_wheel from wheel_index if the base image lacks that version).
    torchvision_required: bool = False
    torchvision_wheel: str = ""
    wheel_index: str = ""
    cuda: str | None  # None: CPU-only plan (the CUDA image still runs torch on CPU)
    pip_packages: tuple[str, ...] = ()
    deviations: tuple[Deviation, ...] = ()
    assumptions: tuple[EnvAssumption, ...] = ()


# Provided by the base image, never pip-installed. "cuda" stands for the toolkit.
_ALIASES = {"pytorch": "torch", "cudatoolkit": "cuda", "cuda-toolkit": "cuda", "python3": "python"}
_NOT_PIP = frozenset({"torch", "torchvision", "cuda", "python"})
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^(==|>=|<=|~=|!=|>|<)?\s*v?([A-Za-z0-9.*+!_-]+)$")


def _canonical(name: str) -> str:
    n = normalize_pip_name(name)
    return _ALIASES.get(n, n)


def _major_minor(version: str) -> str:
    key = version_key(version)
    if len(key) < 2:
        raise ValueError(f"python version needs major.minor, got {version!r}")
    return f"{key[0]}.{key[1]}"


def _requirement(name: str, version: str | None) -> str:
    """pip requirement. Name and version are validated so nothing from the (LLM-extracted)
    EnvSpec or the repo's import lines can inject shell or Dockerfile syntax."""
    name = name.strip()
    if not _PKG_NAME_RE.match(name):
        raise ValueError(f"invalid package name {name!r}")
    if version is None:
        return name
    m = _VERSION_RE.match(version.strip())
    if m is None:
        raise ValueError(f"invalid version {version!r} for package {name!r}")
    return f"{name}{m.group(1) or '=='}{m.group(2)}"


def _stated(env: EnvSpec, canonical: str) -> PackageFact | None:
    facts = [*([env.python] if env.python is not None else []), *env.packages]
    hits = [f for f in facts if _canonical(f.name) == canonical]
    versioned = [f for f in hits if f.version is not None]
    return (versioned or hits or [None])[0]


def _version_matches(stated: str, actual: str) -> bool:
    """Stated "0.13" matches 0.13.1; "0.13.1" matches only 0.13.1."""
    try:
        s = version_key(stated)
    except ValueError:
        return False
    return version_key(actual)[: len(s)] == s


def plan_environment(
    env: EnvSpec,
    gpu: GpuInfo | None,
    *,
    cpu_only: bool = False,
    imports: Iterable[ImportedPackage] = (),
    table: tuple[TorchEntry, ...] = TORCH_TABLE,
) -> EnvPlan:
    """Pick base image / python / torch / torchvision / CUDA for the repo by rule.

    Runnable entry = arch covers the GPU (module docstring) AND the host driver meets
    MIN_DRIVER for its CUDA. torch: stated and runnable -> used as is; stated but not runnable
    (or not in the table) -> OLDEST runnable entry (least API drift) + Deviation; not stated
    -> oldest runnable + torch_version assumption. When the driver filter changes the choice,
    the reason says so. torchvision: the entry's paired version; a different stated version is
    a Deviation. python: the chosen image's python; a stated python that differs is a
    Deviation, an unstated one a python_version assumption. CUDA likewise. Other packages keep
    their stated pins; unpinned ones stay unpinned (assumption "unpinned:<pkg>"). Packages in
    `imports` (third_party_imports) that are not stated are added unpinned; torch/torchvision
    always come from the table, never from imports.
    Raises NoRunnableTorchError if no entry is runnable, ValueError if there is no GPU and
    cpu_only is False, or if the repo requires a GPU and cpu_only is True.
    """
    imports = tuple(imports)
    if gpu is None and not cpu_only:
        raise ValueError("no GPU detected; pass cpu_only=True to plan a CPU-only environment")
    if cpu_only and env.gpu_required:
        raise ValueError("repo states it requires a GPU; refusing a CPU-only plan")
    target = None if cpu_only else gpu
    if not table:
        raise ValueError("TORCH_TABLE is empty")
    runnable = runnable_entries(target, table)
    arch_ok = runnable if target is None else runnable_entries_arch_only(target, table)
    driver_skipped = tuple(e for e in arch_ok if e not in runnable)
    if not runnable:
        assert target is not None
        cc = "{}.{}".format(*target.compute_capability)
        if arch_ok:
            raise NoRunnableTorchError(
                f"torch builds covering compute capability {cc} need a newer driver than "
                f"{target.driver_version}: "
                + ", ".join(f"{e.torch} (CUDA {e.cuda} >= {MIN_DRIVER[e.cuda]})" for e in arch_ok)
                + f" ({DRIVER_TABLE_SOURCE})"
            )
        raise NoRunnableTorchError(
            f"no torch build in TORCH_TABLE covers compute capability {cc} ({target.name}); "
            f"rule: same major, listed minor <= GPU minor ({COMPAT_RULE_SOURCE}). "
            "Add a verified table entry for this architecture."
        )
    oldest = min(runnable, key=lambda e: e.version_key)
    where = (
        "CPU-only"
        if target is None
        else "compute capability {}.{}".format(*target.compute_capability)
    )
    deviations: list[Deviation] = []
    assumptions: list[EnvAssumption] = []

    def driver_note(candidates: Iterable[TorchEntry]) -> str:
        """Why older entries were passed over for the driver; '' if the driver changed nothing."""
        skipped = [e for e in driver_skipped if e in tuple(candidates)]
        if target is None or not skipped:
            return ""
        return (
            f"; skipped torch {', '.join(e.torch for e in skipped)} (CUDA "
            f"{', '.join(sorted({e.cuda for e in skipped}))} needs a newer driver than "
            f"{target.driver_version})"
        )

    older_than_oldest = [e for e in driver_skipped if e.version_key < oldest.version_key]

    # torch
    torch_fact = _stated(env, "torch")
    if torch_fact is None or torch_fact.version is None:
        chosen = oldest
        assumptions.append(
            EnvAssumption(
                name="torch_version",
                value=chosen.torch,
                reason=f"not stated; oldest table entry runnable on {where}"
                + driver_note(older_than_oldest),
            )
        )
    else:
        stated_entry = find_entry(torch_fact.version, table)
        if stated_entry is not None and stated_entry in runnable:
            chosen = stated_entry
        else:
            chosen = oldest
            if stated_entry is None:
                why = (
                    f"torch {torch_fact.version} has no verified TORCH_TABLE entry with a CUDA "
                    f"build covering {where}"
                )
            elif stated_entry in driver_skipped:
                assert target is not None
                why = (
                    f"torch {stated_entry.torch} uses CUDA {stated_entry.cuda}, which needs "
                    f"driver >= {MIN_DRIVER[stated_entry.cuda]}; host driver is "
                    f"{target.driver_version}"
                )
            else:
                archs = ",".join(f"sm_{a}{b}" for a, b in stated_entry.arch_list)
                why = f"torch {stated_entry.torch} is built for [{archs}], not {where}"
            note = driver_note(e for e in older_than_oldest if e is not stated_entry)
            sources = (COMPAT_RULE_SOURCE, *chosen.sources)
            if driver_skipped and (note or stated_entry in driver_skipped):
                sources = (COMPAT_RULE_SOURCE, DRIVER_TABLE_SOURCE, *chosen.sources)
            deviations.append(
                Deviation(
                    field="torch",
                    stated=torch_fact.version,
                    chosen=chosen.torch,
                    reason=f"{why}; chose the oldest runnable entry (least API drift){note}",
                    rule_source=sources,
                )
            )

    # torchvision: always the entry's paired release
    vision_fact = _stated(env, "torchvision")
    if (
        vision_fact is not None
        and vision_fact.version is not None
        and not _version_matches(vision_fact.version, chosen.torchvision)
    ):
        deviations.append(
            Deviation(
                field="torchvision",
                stated=vision_fact.version,
                chosen=chosen.torchvision,
                reason=f"torchvision must match torch {chosen.torch} (paired release "
                f"{chosen.torchvision})",
                rule_source=(_VISION, _PREVIOUS),
            )
        )

    # python: the image's interpreter is the one that runs the code
    py_fact = _stated(env, "python")
    if py_fact is None or py_fact.version is None:
        assumptions.append(
            EnvAssumption(
                name="python_version",
                value=chosen.image_python,
                reason=f"not stated; python shipped by {chosen.base_image}",
            )
        )
    elif _major_minor(py_fact.version) != chosen.image_python:
        stated_py = _major_minor(py_fact.version)
        supported = f"supports {chosen.python_versions[0]}-{chosen.python_versions[-1]}"
        if stated_py in chosen.python_versions:
            why = (
                f"python {stated_py} is supported by torch {chosen.torch}, but "
                f"{chosen.base_image} ships {chosen.image_python}"
            )
        else:
            why = f"python {stated_py} is not supported by torch {chosen.torch} ({supported})"
        deviations.append(
            Deviation(
                field="python",
                stated=py_fact.version,
                chosen=chosen.image_python,
                reason=why,
                rule_source=chosen.sources,
            )
        )

    # CUDA
    cuda = None if target is None else chosen.cuda
    cuda_fact = _stated(env, "cuda")
    if cuda is not None:
        if cuda_fact is None or cuda_fact.version is None:
            assumptions.append(
                EnvAssumption(
                    name="cuda_version",
                    value=cuda,
                    reason=f"not stated; CUDA build of torch {chosen.torch} in {chosen.base_image}",
                )
            )
        elif version_key(cuda_fact.version)[:2] != version_key(cuda)[:2]:
            deviations.append(
                Deviation(
                    field="cuda",
                    stated=cuda_fact.version,
                    chosen=cuda,
                    reason=f"CUDA follows the chosen torch build ({chosen.base_image})",
                    rule_source=chosen.sources,
                )
            )

    # other packages: stated first, then imported-but-unstated
    pip: list[str] = []
    seen: set[str] = set()
    for fact in env.packages:
        canon = _canonical(fact.name)
        if canon in _NOT_PIP or canon in seen:
            continue
        seen.add(canon)
        pip.append(_requirement(fact.name, fact.version))
        if fact.version is None:
            assumptions.append(
                EnvAssumption(
                    name=f"unpinned:{fact.name.strip()}",
                    value=None,
                    reason="no version stated; left unpinned (pip resolves at build time)",
                )
            )
    for imp in imports:
        canon = _canonical(imp.pip_name)
        if canon in _NOT_PIP or canon in seen:
            continue
        seen.add(canon)
        pip.append(_requirement(imp.pip_name, None))
        assumptions.append(
            EnvAssumption(
                name=f"unpinned:{imp.pip_name}",
                value=None,
                reason=f"unpinned:{imp.pip_name} (imported by {imp.path}); not stated by the "
                "repo, left unpinned",
            )
        )

    return EnvPlan(
        base_image=chosen.base_image,
        python=chosen.image_python,
        torch=chosen.torch,
        torchvision=chosen.torchvision,
        torchvision_required=vision_fact is not None
        or any(_canonical(i.pip_name) == "torchvision" for i in imports),
        torchvision_wheel=chosen.torchvision_wheel,
        wheel_index=chosen.wheel_index,
        cuda=cuda,
        pip_packages=tuple(pip),
        deviations=tuple(deviations),
        assumptions=tuple(assumptions),
    )


def runnable_entries_arch_only(
    gpu: GpuInfo, table: tuple[TorchEntry, ...] = TORCH_TABLE
) -> tuple[TorchEntry, ...]:
    """Entries whose arch list covers the GPU, ignoring the driver."""
    return tuple(e for e in table if arch_covers(e.arch_list, gpu.compute_capability))


# --- Dockerfile ----------------------------------------------------------------------------

CONTAINER_USER = "repro"
CONTAINER_UID = 1000


def render_dockerfile(plan: EnvPlan) -> str:
    """Dockerfile for the plan. The repo is NOT copied in (it is mounted at /work at run
    time). The only network steps are pip. A constraints file pins torch and torchvision so pip
    cannot replace them. If torchvision is required and the base image lacks the planned
    version, the exact CUDA wheel is installed from the official index. A final no-network
    check asserts python, torch and torchvision (required: must import; otherwise: must match
    if present)."""
    py = tuple(int(x) for x in plan.python.split("."))
    tv = json.dumps(plan.torchvision)
    tv_check = (
        f'import torchvision; assert torchvision.__version__.split("+")[0] == {tv}, '
        "torchvision.__version__"
        if plan.torchvision_required
        else 'tv = importlib.util.find_spec("torchvision") and __import__("torchvision"); '
        f'assert not tv or tv.__version__.split("+")[0] == {tv}, tv.__version__'
    )
    check = (
        "import importlib.util, sys, torch; "
        f"assert sys.version_info[:2] == {py!r}, sys.version; "
        f'assert torch.__version__.split("+")[0] == {json.dumps(plan.torch)}, torch.__version__; '
        + tv_check
    )
    lines = [
        "# Generated by reproducer.domain.env_rules.render_dockerfile; do not edit.",
        f"FROM {plan.base_image}",
        "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1",
    ]
    steps: list[str] = []
    if plan.torchvision_required:
        has_tv = f'import sys, torchvision; sys.exit(torchvision.__version__.split("+")[0] != {tv})'
        steps.append(
            f"{{ python -c {shlex.quote(has_tv)} 2>/dev/null || python -m pip install "
            f"-c /tmp/constraints.txt --extra-index-url {shlex.quote(plan.wheel_index)} "
            f"{shlex.quote(plan.torchvision_wheel)}; }}"
        )
    if plan.pip_packages:
        reqs = " ".join(shlex.quote(p) for p in plan.pip_packages)
        steps.append(f"python -m pip install -c /tmp/constraints.txt {reqs}")
    if steps:
        pins = f"torch=={plan.torch}\\ntorchvision=={plan.torchvision}\\n"
        lines.append(f"RUN printf {shlex.quote(pins)} > /tmp/constraints.txt \\")
        lines += [f"    && {step} \\" for step in steps]
        lines.append("    && rm /tmp/constraints.txt")
    lines += [
        f"RUN python -c {shlex.quote(check)}",
        f"RUN useradd --create-home --uid {CONTAINER_UID} {CONTAINER_USER}",
        "WORKDIR /work",
        f"USER {CONTAINER_USER}",
    ]
    return "\n".join(lines) + "\n"


# --- Evidence ------------------------------------------------------------------------------


def env_plan_to_evidence(
    plan: EnvPlan, source: str, links: Iterable[str] = (), gpu: GpuInfo | None = None
) -> Evidence:
    """env_spec evidence for a plan: a deterministic rule decision (authority "rule",
    confidence "derived"), linked to the EnvSpec subtask evidence it was planned from. It never
    resolves an Unknown; its values enter the unknown log only as Assumptions."""
    content: dict[str, Any] = {"plan": plan.model_dump(mode="json")}
    if gpu is not None:
        content["gpu"] = gpu.model_dump(mode="json")
    return make_evidence(
        kind="env_spec",
        source=source,
        locator="env_plan",
        content=content,
        confidence="derived",
        authority="rule",
        links=links,
    )
