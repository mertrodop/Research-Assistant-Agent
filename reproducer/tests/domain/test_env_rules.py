from __future__ import annotations

from typing import Any

import pytest

from reproducer.domain.contracts import EnvSpec, PackageFact
from reproducer.domain.env_rules import (
    MIN_DRIVER,
    TORCH_TABLE,
    EnvPlan,
    GpuInfo,
    NoRunnableTorchError,
    arch_covers,
    driver_supports,
    env_plan_to_evidence,
    find_entry,
    parse_nvidia_smi_csv,
    plan_environment,
    render_dockerfile,
)
from reproducer.domain.evidence import make_evidence
from reproducer.domain.import_scan import ImportedPackage
from reproducer.domain.unknowns import UnknownOpened, resolve, unknown_id_for

SMI_CSV = (
    "name, compute_cap, driver_version, memory.total [MiB]\n"
    "NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 580.178.04, 8188 MiB\n"
)
RTX4060 = GpuInfo(
    name="NVIDIA GeForce RTX 4060 Laptop GPU",
    compute_capability=(8, 9),
    driver_version="580.178.04",
    memory_mb=8188,
)


def _pkg(name: str, version: str | None = None) -> PackageFact:
    # No citation -> no validation context needed; source is "inferred".
    return PackageFact.model_validate({"name": name, "version": version})


def _env(*packages: PackageFact, python: str | None = None, **kw: Any) -> EnvSpec:
    return EnvSpec(
        python=None if python is None else _pkg("python", python), packages=packages, **kw
    )


CUTOUT = _env(_pkg("torch", "0.4.0"), _pkg("tqdm"))


# --- nvidia-smi ----------------------------------------------------------------------------


def test_parse_real_gpu_line() -> None:
    assert parse_nvidia_smi_csv(SMI_CSV) == RTX4060


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_csv("No devices were found")
    with pytest.raises(ValueError, match="compute capability"):
        parse_nvidia_smi_csv(SMI_CSV.replace("8.9", "[N/A]"))


def test_parse_mixed_gpus_rejected() -> None:
    two = SMI_CSV + "NVIDIA A100-SXM4-40GB, 8.0, 580.178.04, 40960 MiB\n"
    with pytest.raises(ValueError, match="mixed"):
        parse_nvidia_smi_csv(two)


# --- table ---------------------------------------------------------------------------------


def test_table_integrity() -> None:
    keys = [e.version_key for e in TORCH_TABLE]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert keys[0][0] == 1 and keys[-1][0] == 2  # spans 1.x .. 2.x
    for e in TORCH_TABLE:
        assert e.sources and all(s.startswith("https://") for s in e.sources)
        assert any("hub.docker.com" in s for s in e.sources)
        assert e.base_image.startswith("pytorch/pytorch:") and e.base_image.endswith("-runtime")
        assert e.image_python in e.python_versions
        assert list(e.arch_list) == sorted(e.arch_list)
        assert e.cuda in MIN_DRIVER  # every entry has a sourced driver minimum
        assert e.torchvision.startswith("0.") and any("vision" in s for s in e.sources)


def test_arch_rule() -> None:
    assert arch_covers([(8, 6)], (8, 9))  # sm_86 cubin runs on 8.9 (same major, lower minor)
    assert not arch_covers([(8, 9)], (8, 6))  # higher minor does not run on lower
    assert not arch_covers([(8, 6)], (9, 0))  # never across majors
    assert not arch_covers([(7, 0), (7, 5)], (8, 9))


def test_find_entry() -> None:
    assert find_entry("1.12.1") is not None and find_entry("1.12.1").torch == "1.12.1"
    assert find_entry("v2.4.1").torch == "2.4.1"
    assert find_entry("1.12").torch == "1.12.1"
    assert find_entry("0.4.0") is None
    assert find_entry("not-a-version") is None


# --- plan ----------------------------------------------------------------------------------


def test_cutout_on_rtx4060_deviates_torch_and_assumes_python() -> None:
    plan = plan_environment(CUTOUT, RTX4060)
    oldest = min(
        (e for e in TORCH_TABLE if arch_covers(e.arch_list, (8, 9))), key=lambda e: e.version_key
    )
    assert (plan.torch, plan.base_image, plan.cuda) == (
        oldest.torch,
        oldest.base_image,
        oldest.cuda,
    )
    assert len(plan.deviations) == 1
    dev = plan.deviations[0]
    assert (dev.field, dev.stated, dev.chosen) == ("torch", "0.4.0", oldest.torch)
    assert "compute capability 8.9" in dev.reason
    assert any("nvidia.com" in s for s in dev.rule_source)
    assumed = {a.name: a.value for a in plan.assumptions}
    assert assumed["python_version"] == oldest.image_python == plan.python
    assert assumed["cuda_version"] == oldest.cuda
    assert "torch_version" not in assumed  # torch was stated -> deviation, not assumption
    assert assumed["unpinned:tqdm"] is None
    assert plan.pip_packages == ("tqdm",)


def test_stated_runnable_torch_kept_without_deviation() -> None:
    entry = next(e for e in TORCH_TABLE if e.torch == "2.1.2")
    plan = plan_environment(_env(_pkg("torch", "2.1.2"), python=entry.image_python), RTX4060)
    assert plan.torch == "2.1.2" and plan.base_image == entry.base_image
    assert plan.deviations == ()
    assert "python_version" not in {a.name for a in plan.assumptions}


def test_stated_torch_not_covering_gpu_names_its_archs() -> None:
    hopper = RTX4060.model_copy(update={"compute_capability": (9, 0)})
    plan = plan_environment(_env(_pkg("pytorch", "1.12.1")), hopper)
    dev = plan.deviations[0]
    assert dev.stated == "1.12.1" and dev.chosen != "1.12.1"
    assert "sm_86" in dev.reason and "9.0" in dev.reason


def test_no_stated_torch_is_an_assumption() -> None:
    plan = plan_environment(_env(), RTX4060)
    assert plan.deviations == ()
    assert {a.name: a.value for a in plan.assumptions}["torch_version"] == plan.torch


def test_stated_python_unsupported_is_deviation() -> None:
    plan = plan_environment(_env(_pkg("torch", "1.12.1"), python="3.6"), RTX4060)
    (dev,) = plan.deviations
    assert (dev.field, dev.stated, dev.chosen) == ("python", "3.6", plan.python)
    assert "not supported" in dev.reason


def test_pins_kept_and_validated() -> None:
    plan = plan_environment(_env(_pkg("numpy", "1.19.5"), _pkg("scipy", ">=1.5")), RTX4060)
    assert plan.pip_packages == ("numpy==1.19.5", "scipy>=1.5")
    with pytest.raises(ValueError, match="package name"):
        plan_environment(_env(_pkg("tqdm; rm -rf /")), RTX4060)
    with pytest.raises(ValueError, match="version"):
        plan_environment(_env(_pkg("tqdm", "1.0 && curl x")), RTX4060)


def test_cpu_only_plan() -> None:
    plan = plan_environment(CUTOUT, None, cpu_only=True)
    assert plan.cuda is None
    assert plan.torch == TORCH_TABLE[0].torch  # every entry runs on CPU -> oldest
    assert "CPU-only" in plan.deviations[0].reason
    assert "cuda_version" not in {a.name for a in plan.assumptions}


def test_no_gpu_without_cpu_only_errors() -> None:
    with pytest.raises(ValueError, match="cpu_only"):
        plan_environment(CUTOUT, None)


def test_cpu_only_refused_when_gpu_required() -> None:
    with pytest.raises(ValueError, match="requires a GPU"):
        plan_environment(_env(gpu_required=True), None, cpu_only=True)


@pytest.mark.parametrize("cc", [(3, 0), (13, 0)])
def test_unknown_arch_clear_error(cc: tuple[int, int]) -> None:
    gpu = RTX4060.model_copy(update={"compute_capability": cc, "name": "Future GPU"})
    with pytest.raises(NoRunnableTorchError, match=rf"compute capability {cc[0]}\.{cc[1]}"):
        plan_environment(CUTOUT, gpu)


# --- Dockerfile ----------------------------------------------------------------------------


def test_render_dockerfile() -> None:
    plan = plan_environment(
        _env(_pkg("torch", "1.12.1"), _pkg("numpy", "1.21.6"), _pkg("tqdm")), RTX4060
    )
    text = render_dockerfile(plan)
    lines = text.splitlines()
    assert lines[1] == f"FROM {plan.base_image}"
    assert "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1" in lines
    assert "numpy==1.21.6 tqdm" in text
    assert "torch==1.12.1" in text and "torchvision==0.13.1" in text  # constraints pins
    assert 'tv.__version__.split("+")[0] == "0.13.1"' in text  # build-time torchvision check
    assert "assert sys.version_info[:2] == (3, 7)" in text
    assert "WORKDIR /work" in lines
    assert lines[-1] == "USER repro" and any(ln.startswith("RUN useradd") for ln in lines)
    assert not any(ln.split()[0] in ("COPY", "ADD") for ln in lines if ln.strip())
    run_cmds = " ".join(ln for ln in lines if ln.startswith("RUN") or ln.startswith("    "))
    for net in ("apt-get", "curl", "wget", "git ", "conda "):
        assert net not in run_cmds


def test_render_dockerfile_without_packages_has_no_pip() -> None:
    text = render_dockerfile(plan_environment(_env(_pkg("torch", "2.4.1")), RTX4060))
    assert "pip install" not in text


# --- evidence / unknowns -------------------------------------------------------------------


def test_env_plan_evidence_is_rule_and_cannot_resolve() -> None:
    plan = plan_environment(CUTOUT, RTX4060)
    ev = env_plan_to_evidence(plan, "repo:x@abc", links=("e" * 64,), gpu=RTX4060)
    assert (ev.kind, ev.locator, ev.authority, ev.confidence) == (
        "env_spec",
        "env_plan",
        "rule",
        "derived",
    )
    assert ev.links == ("e" * 64,)
    assert EnvPlan.model_validate(ev.content["plan"]) == plan

    setup = make_evidence(
        kind="paper_setup",
        source="paper",
        locator="Sec 4",
        content={"name": "python_version", "value": None},
        confidence="observed",
        authority="paper",
    )
    opened = UnknownOpened(
        unknown_id=unknown_id_for("python_version"),
        name="python_version",
        evidence_id=setup.evidence_id,
    )
    with pytest.raises(ValueError, match="'rule' authority cannot resolve"):
        resolve(opened, ev, plan.python)


# --- torchvision ---------------------------------------------------------------------------


def test_torchvision_follows_table() -> None:
    plan = plan_environment(CUTOUT, RTX4060)
    assert plan.torchvision == find_entry(plan.torch).torchvision
    assert "torchvision" not in plan.pip_packages


def test_stated_torchvision_replaced_is_deviation() -> None:
    plan = plan_environment(_env(_pkg("torch", "0.4.0"), _pkg("torchvision", "0.2.1")), RTX4060)
    devs = {d.field: d for d in plan.deviations}
    assert devs["torchvision"].stated == "0.2.1"
    assert devs["torchvision"].chosen == plan.torchvision
    assert "torchvision" not in " ".join(plan.pip_packages)


def test_stated_matching_torchvision_no_deviation() -> None:
    plan = plan_environment(_env(_pkg("torch", "1.12.1"), _pkg("torchvision", "0.13")), RTX4060)
    assert plan.deviations == ()


# --- driver --------------------------------------------------------------------------------


def test_driver_supports() -> None:
    assert driver_supports("580.178.04", "13.0")
    assert driver_supports("465.19.01", "11.3")
    assert not driver_supports("465.19.00", "11.3")
    assert not driver_supports("525.60.13", "12.1")
    with pytest.raises(ValueError, match="MIN_DRIVER"):
        driver_supports("580.0", "10.2")
    with pytest.raises(ValueError, match="driver version"):
        driver_supports("[N/A]", "11.3")


def test_old_driver_skips_entries_and_says_so() -> None:
    # 512.x: runs CUDA 11.3 and 11.6 builds but not 11.7+.
    old = RTX4060.model_copy(update={"driver_version": "512.15"})
    plan = plan_environment(_env(_pkg("torch", "2.1.2")), old)
    assert plan.torch == "1.12.1"
    (dev,) = plan.deviations
    assert "CUDA 11.8" in dev.reason and "512.15" in dev.reason
    assert any("cuda-toolkit-release-notes" in s for s in dev.rule_source)


def test_driver_change_recorded_when_oldest_skipped() -> None:
    # 470.x: CUDA 11.3 (needs 465) ok, 11.6+ not; the oldest entry is still chosen -> no note.
    drv = RTX4060.model_copy(update={"driver_version": "470.42.01"})
    plan = plan_environment(_env(), drv)
    assert plan.torch == "1.12.1" and "skipped" not in plan.assumptions[0].reason
    too_old = RTX4060.model_copy(update={"driver_version": "460.32.03"})
    with pytest.raises(NoRunnableTorchError, match="newer driver than 460.32.03"):
        plan_environment(_env(), too_old)


def test_driver_skip_of_older_entries_noted_in_deviation() -> None:
    # Hypothetical table where the oldest arch-ok entry needs a newer driver than the host.
    t = (
        TORCH_TABLE[0].model_copy(update={"cuda": "12.8"}),  # 1.12.1 needing 570
        TORCH_TABLE[1],  # 1.13.1 / CUDA 11.6 needs 510
    )
    gpu = RTX4060.model_copy(update={"driver_version": "520.0"})
    plan = plan_environment(_env(_pkg("torch", "0.4.0")), gpu, table=t)
    assert plan.torch == "1.13.1"
    reason = plan.deviations[0].reason
    assert "skipped torch 1.12.1" in reason and "newer driver than 520.0" in reason


# --- imports -------------------------------------------------------------------------------


def test_imports_added_unpinned_torch_never_from_scan() -> None:
    imports = (
        ImportedPackage(pip_name="torch", module="torch", path="train.py"),
        ImportedPackage(pip_name="torchvision", module="torchvision", path="train.py"),
        ImportedPackage(pip_name="numpy", module="numpy", path="train.py"),
        ImportedPackage(pip_name="tqdm", module="tqdm", path="train.py"),
        ImportedPackage(pip_name="opencv-python", module="cv2", path="util/aug.py"),
    )
    plan = plan_environment(CUTOUT, RTX4060, imports=imports)
    assert plan.pip_packages == ("tqdm", "numpy", "opencv-python")  # stated first, no dupes
    reasons = {a.name: a.reason for a in plan.assumptions}
    assert "imported by train.py" in reasons["unpinned:numpy"]
    assert "imported by util/aug.py" in reasons["unpinned:opencv-python"]
    assert reasons["unpinned:tqdm"].startswith("no version stated")  # stated, not scanned
    assert not any(n.startswith("unpinned:torch") for n in reasons)


def test_stated_pin_wins_over_import() -> None:
    imports = (ImportedPackage(pip_name="numpy", module="numpy", path="train.py"),)
    plan = plan_environment(_env(_pkg("NumPy", "1.21.6")), RTX4060, imports=imports)
    assert plan.pip_packages == ("NumPy==1.21.6",)


# --- torchvision required ------------------------------------------------------------------


def test_torchvision_required_when_imported() -> None:
    imports = (ImportedPackage(pip_name="torchvision", module="torchvision", path="train.py"),)
    plan = plan_environment(CUTOUT, RTX4060, imports=imports)
    assert plan.torchvision_required
    assert plan.torchvision_wheel == "torchvision==0.13.1+cu113"
    assert plan.wheel_index == "https://download.pytorch.org/whl/cu113"
    text = render_dockerfile(plan)
    # install only if the image lacks that version, from the official CUDA index
    assert "|| python -m pip install -c /tmp/constraints.txt" in text
    assert (
        "--extra-index-url https://download.pytorch.org/whl/cu113 torchvision==0.13.1+cu113" in text
    )
    # build-time check REQUIRES torchvision (plain import, no find_spec escape hatch)
    check = next(ln for ln in text.splitlines() if ln.startswith("RUN python -c"))
    assert "import torchvision; assert torchvision.__version__" in check
    assert "find_spec" not in check.split("torch.__version__;", 1)[1]
    # the check runs after all installs
    lines = text.splitlines()
    assert lines.index(check) > max(i for i, ln in enumerate(lines) if "pip install" in ln)


def test_torchvision_required_when_stated() -> None:
    plan = plan_environment(_env(_pkg("torch", "1.12.1"), _pkg("torchvision")), RTX4060)
    assert plan.torchvision_required
    assert "torchvision==0.13.1+cu113" in render_dockerfile(plan)


def test_torchvision_not_required_no_install() -> None:
    plan = plan_environment(_env(_pkg("torch", "2.4.1")), RTX4060)
    assert not plan.torchvision_required
    text = render_dockerfile(plan)
    assert "extra-index-url" not in text and "find_spec" in text


def test_every_entry_has_a_cuda_wheel_index() -> None:
    for e in TORCH_TABLE:
        assert e.wheel_index.startswith("https://download.pytorch.org/whl/cu")
        assert e.torchvision_wheel.startswith(f"torchvision=={e.torchvision}+cu")
