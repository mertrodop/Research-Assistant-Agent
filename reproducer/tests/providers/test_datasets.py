"""Tests for reproducer.providers.datasets.

Pure tests (cache dir, payload, docker args, result parsing, file listing) run offline.
Tests marked `docker` need a local Docker daemon, network (MNIST download) and an image with
torchvision: $REPRODUCER_TEST_TV_IMAGE, else the first local reproducer/* image that imports
torchvision. Deselected by default:
    .venv/bin/pytest -q -m docker -rs
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from reproducer.domain.data_plan import KNOWN_DATASETS
from reproducer.providers.datasets import (
    CONTAINER_ROOT,
    TRUSTED_SCRIPT,
    cached_files,
    data_cache_dir,
    ensure_datasets,
    parse_result,
    trusted_args,
    trusted_payload,
)
from reproducer.providers.docker_runner import run_container


def _flag_values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


# --- pure ------------------------------------------------------------------------------------


def test_data_cache_dir_default_and_override(tmp_path: Path) -> None:
    assert data_cache_dir({}) == Path.home() / ".cache/reproducer/data/torchvision"
    assert data_cache_dir({"REPRODUCER_DATA_CACHE": str(tmp_path)}) == tmp_path
    assert data_cache_dir({"REPRODUCER_DATA_CACHE": "  "}) == data_cache_dir({})


def test_payload_is_registry_only() -> None:
    payload = json.loads(trusted_payload(["cifar10", "svhn"], download=True))
    assert payload == {
        "root": "/data",
        "download": True,
        "items": [
            {"name": "cifar10", "cls": "CIFAR10", "splits": [{"train": True}, {"train": False}]},
            {"name": "svhn", "cls": "SVHN", "splits": [{"split": "train"}, {"split": "test"}]},
        ],
    }
    with pytest.raises(KeyError):
        trusted_payload(["os.system('x')"], download=True)


@pytest.mark.parametrize(("mode", "network", "access"), [("check", "none", "ro"),
                                                         ("download", "bridge", "rw")])  # fmt: skip
def test_trusted_args(mode: str, network: str, access: str) -> None:
    args = trusted_args(
        "img:tag", mode=mode, cache_dir="/cache/tv", payload="{}", name="n1", uid=1000, gid=1000
    )  # type: ignore[arg-type]
    assert _flag_values(args, "--network") == [network]
    assert _flag_values(args, "-v") == [f"/cache/tv:{CONTAINER_ROOT}:{access}"]  # only mount
    assert _flag_values(args, "--user") == ["1000:1000"]
    assert _flag_values(args, "--cap-drop") == ["ALL"]
    assert "--read-only" in args and "--privileged" not in args
    assert args[-6:] == ["img:tag", "python", "-I", "-c", TRUSTED_SCRIPT, "{}"]


def test_trusted_args_rejects_bad_cache_dir() -> None:
    for bad in ("relative", "/a:b", "/a,b"):
        with pytest.raises(ValueError):
            trusted_args("i", mode="check", cache_dir=bad, payload="{}", name="n", uid=1, gid=1)


def test_parse_result() -> None:
    log = 'noise\r 50%|###\nREPRODUCER_RESULT {"results": {"mnist": {"ok": true}}}\n'
    assert parse_result(log) == {"results": {"mnist": {"ok": True}}}
    assert parse_result("REPRODUCER_RESULT {broken") is None
    assert parse_result("nothing") is None


def test_ensure_rejects_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registry"):
        ensure_datasets(["imagenet"], "img:tag", log_dir=tmp_path, cache_dir=tmp_path)


def test_cached_files(tmp_path: Path) -> None:
    (tmp_path / "cifar-10-batches-py").mkdir()
    (tmp_path / "cifar-10-batches-py" / "test_batch").write_bytes(b"abc")
    (tmp_path / "cifar-10-python.tar.gz").write_bytes(b"z")
    (tmp_path / "cifar-10-batches-py" / "link").symlink_to("/etc/passwd")
    (tmp_path / "train_32x32.mat").write_bytes(b"other dataset")
    files = cached_files(tmp_path, KNOWN_DATASETS["cifar10"])
    assert [(f["path"], f["size"]) for f in files] == [
        ("cifar-10-batches-py/test_batch", 3),
        ("cifar-10-python.tar.gz", 1),
    ]
    assert files[0]["sha256"].startswith("ba7816bf")


# --- docker ----------------------------------------------------------------------------------


def _has_torchvision(image: str) -> bool:
    r = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image, "python", "-c",
         "import torchvision"],
        capture_output=True, timeout=120, check=False,
    )  # fmt: skip
    return r.returncode == 0


@pytest.fixture(scope="module")
def tv_image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    env = os.environ.get("REPRODUCER_TEST_TV_IMAGE")
    if env:
        return env
    r = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "reproducer/*"],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    for image in r.stdout.split():
        if "test-tiny" not in image and _has_torchvision(image):
            return image
    pytest.skip("no local image with torchvision (set REPRODUCER_TEST_TV_IMAGE)")


@pytest.mark.docker
def test_docker_ensure_mnist_then_present_then_sandbox(tv_image: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = ensure_datasets(["mnist"], tv_image, log_dir=tmp_path / "logs", cache_dir=cache)
    st = first["mnist"]
    logs = "\n".join(p.read_text(errors="replace") for p in st.logs if p.exists())
    assert st.state == "downloaded" and st.verified, (st.error, logs[-3000:])
    assert st.split_sizes == (60000, 10000)
    assert st.files and all(f["path"].startswith("MNIST/") for f in st.files)

    second = ensure_datasets(["mnist"], tv_image, log_dir=tmp_path / "logs2", cache_dir=cache)
    assert second["mnist"].state == "present" and second["mnist"].verified
    assert second["mnist"].files == st.files

    # sandbox: network none, cache ro under /work, our script only
    work = tmp_path / "work"
    work.mkdir()
    code = (
        "import torchvision.datasets as d\n"
        "print('LEN', len(d.MNIST('/work/data', train=False, download=False)))\n"
        "try:\n"
        "    open('/work/data/probe', 'w'); print('DATA_WRITABLE')\n"
        "except OSError:\n"
        "    print('DATA_RO')\n"
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=3); print('NETWORK_OPEN')\n"
        "except OSError:\n"
        "    print('NET_BLOCKED')\n"
    )
    name = f"reproducer-test-{uuid.uuid4().hex[:8]}"
    res = run_container(
        tv_image,
        ["python", "-I", "-c", code],
        workdir=work,
        out_dir=tmp_path / "out",
        data_mounts={str(cache): "/work/data"},
        gpu=False,
        timeout_s=120,
        mem="2g",
        cpus=1.0,
        log_path=tmp_path / f"{name}.log",
        name=name,
    )
    log = res.log_path.read_text()
    assert res.exit_code == 0, log
    assert "LEN 10000" in log and "DATA_RO" in log and "NET_BLOCKED" in log
    assert "DATA_WRITABLE" not in log and "NETWORK_OPEN" not in log
    assert not (cache / "probe").exists()


@pytest.mark.docker
def test_docker_check_reports_missing_without_network(tv_image: str, tmp_path: Path) -> None:
    """An empty cache fails the network-none check (nothing is fetched by `check`)."""
    from reproducer.providers.datasets import run_trusted

    run = run_trusted(
        tv_image, ["mnist"], mode="check", cache_dir=tmp_path / "c",
        log_path=tmp_path / "check.log", timeout_s=120,
    )  # fmt: skip
    assert run.results["mnist"]["ok"] is False
    assert not any((tmp_path / "c").iterdir())
