"""Tests for reproducer.providers.docker_runner.

Pure tests (argument lists, tags, log capping, prepare_workdir with git apply) run offline.
Tests marked `docker` need a local Docker daemon and are deselected by default:
    .venv/bin/pytest -q -m docker -rs
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

from reproducer.domain.env_rules import EnvPlan
from reproducer.providers.docker_runner import (
    CappedLog,
    PatchError,
    build_args,
    build_image,
    image_exists,
    image_tag,
    prepare_workdir,
    run_args,
    run_container,
)

PLAN = EnvPlan(
    base_image="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
    python="3.7",
    torch="1.12.1",
    torchvision="0.13.1",
    cuda="11.3",
    pip_packages=("tqdm",),
)


def _run_args(**kw: object) -> list[str]:
    base: dict[str, object] = {
        "name": "reproducer-test",
        "workdir": "/runs/x/work",
        "out_dir": "/runs/x/out",
        "gpu": False,
        "uid": 1000,
        "gid": 1000,
    }
    base.update(kw)
    return run_args("img:tag", ["python", "train.py"], **base)  # type: ignore[arg-type]


def _flag_values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


# --- pure ------------------------------------------------------------------------------------


def test_run_args_security_flags() -> None:
    args = _run_args()
    assert args[:2] == ["docker", "run"]
    assert "--rm=false" in args
    assert _flag_values(args, "--network") == ["none"]
    assert _flag_values(args, "--cap-drop") == ["ALL"]
    assert _flag_values(args, "--security-opt") == ["no-new-privileges"]
    assert "--read-only" in args
    assert _flag_values(args, "--tmpfs") == ["/tmp:rw,size=2g"]
    assert _flag_values(args, "--user") == ["1000:1000"]
    assert "HOME=/tmp" in _flag_values(args, "-e")
    assert _flag_values(args, "--memory") == ["12g"]
    assert _flag_values(args, "--pids-limit") == ["512"]
    assert _flag_values(args, "--cpus") == ["4"]
    assert _flag_values(args, "-w") == ["/work"]
    assert _flag_values(args, "-v") == ["/runs/x/work:/work:rw", "/runs/x/out:/out:rw"]
    assert "--gpus" not in args
    assert not any(a.startswith("--network=") or a == "--net" or a == "--privileged" for a in args)
    assert args[-3:] == ["img:tag", "python", "train.py"]  # image then cmd, nothing after


def test_run_args_gpu_and_readonly_data_mounts() -> None:
    args = _run_args(gpu=True, data_mounts={"/data/cifar": "/data/cifar10", "/data/b": "/b"})
    assert _flag_values(args, "--gpus") == ["all"]
    vols = _flag_values(args, "-v")
    assert "/data/cifar:/data/cifar10:ro" in vols and "/data/b:/b:ro" in vols
    assert sum(v.endswith(":rw") for v in vols) == 2  # only /work and /out are writable


def test_run_args_accepts_readonly_data_under_work() -> None:
    args = _run_args(data_mounts={"/cache/tv": "/work/data"})
    vols = _flag_values(args, "-v")
    assert vols == ["/runs/x/work:/work:rw", "/runs/x/out:/out:rw", "/cache/tv:/work/data:ro"]


@pytest.mark.parametrize(
    "kw",
    [
        {"data_mounts": {"relative/path": "/data"}},
        {"data_mounts": {"/data": "/work"}},
        {"data_mounts": {"/data": "/work/"}},
        {"data_mounts": {"/data": "/out"}},
        {"data_mounts": {"/data": "/out/x"}},
        {"data_mounts": {"/data": "/work/../etc"}},
        {"data_mounts": {"/data": "/work/./data"}},
        {"data_mounts": {"/data": "//work/data"}},
        {"data_mounts": {"/data": "/tmp"}},
        {"data_mounts": {"/data": "/"}},
        {"data_mounts": {"/a:b": "/data"}},
        {"workdir": "runs/x"},
        {"name": "bad name; rm -rf /"},
    ],
)
def test_run_args_rejects_unsafe_input(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _run_args(**kw)


def test_build_args_no_context() -> None:
    args = build_args("reproducer/cutout:abc")
    assert args[:2] == ["docker", "build"] and args[-1] == "-"
    assert args[args.index("-t") + 1] == "reproducer/cutout:abc"


def test_image_tag_stable() -> None:
    tag = image_tag(PLAN, "cutout")
    assert tag == image_tag(PLAN, "cutout")
    assert tag.startswith("reproducer/cutout:") and len(tag.rsplit(":", 1)[1]) == 12
    assert image_tag(PLAN.model_copy(update={"pip_packages": ("numpy",)}), "cutout") != tag
    # assumptions/deviations do not change the Dockerfile -> same image
    assert image_tag(PLAN.model_copy(update={"assumptions": ()}), "cutout") == tag
    with pytest.raises(ValueError):
        image_tag(PLAN, "Cut Out")


def test_capped_log_keeps_head_and_tail(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    with CappedLog(path, max_bytes=100) as log:
        for i in range(100):
            log.write(f"{i:03d}\n".encode())  # 400 bytes total
    text = path.read_text()
    assert text.startswith("000\n001\n")
    assert text.endswith("098\n099\n")
    assert "[... 300 bytes omitted ...]" in text
    assert len(text.encode()) < 100 + 40


def test_capped_log_small_output_untouched(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    with CappedLog(path, max_bytes=1000) as log:
        log.write(b"hello\n")
        log.write(b"world\n")
    assert path.read_text() == "hello\nworld\n"


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "clone"
    (repo / "util").mkdir(parents=True)
    (repo / "train.py").write_text("lr = 0.1\nepochs = 200\n")
    (repo / "util" / "misc.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "marker").write_text("git internals")
    return repo


PATCH_OK = """\
--- a/train.py
+++ b/train.py
@@ -1,2 +1,2 @@
 lr = 0.1
-epochs = 200
+epochs = 1
"""

PATCH_BAD = """\
--- a/train.py
+++ b/train.py
@@ -1,2 +1,2 @@
 lr = 0.5
-epochs = 999
+epochs = 1
"""


def test_prepare_workdir_copies_without_git_and_applies_patch(clone: Path, tmp_path: Path) -> None:
    before = (clone / "train.py").read_text()
    dest = prepare_workdir(clone, [PATCH_OK], tmp_path / "runs" / "r1" / "work")
    assert (dest / "train.py").read_text() == "lr = 0.1\nepochs = 1\n"
    assert (dest / "util" / "misc.py").exists()
    assert not (dest / ".git").exists()
    assert (clone / "train.py").read_text() == before  # clone untouched


def test_prepare_workdir_failing_patch(clone: Path, tmp_path: Path) -> None:
    with pytest.raises(PatchError, match="patch 2/2 did not apply"):
        prepare_workdir(clone, [PATCH_OK, PATCH_BAD], tmp_path / "w")


def test_prepare_workdir_refuses_existing_dest(clone: Path, tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    with pytest.raises(FileExistsError):
        prepare_workdir(clone, [], tmp_path / "w")


# --- docker ----------------------------------------------------------------------------------

TINY_DOCKERFILE = "FROM python:3.10-slim\nENV PYTHONDONTWRITEBYTECODE=1\n"


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    tag = "reproducer/test-tiny:latest"
    log = tmp_path_factory.mktemp("build") / "build.log"
    result = build_image(TINY_DOCKERFILE, tag, timeout_s=600, log_path=log)
    assert result.ok, log.read_text() if log.exists() else "no log"
    return tag


def _run(image: str, code: str, tmp_path: Path, **kw: object):  # type: ignore[no-untyped-def]
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir(exist_ok=True)
    name = f"reproducer-test-{uuid.uuid4().hex[:8]}"
    params: dict[str, object] = {
        "workdir": work,
        "out_dir": out,
        "gpu": False,
        "timeout_s": 60,
        "mem": "512m",
        "cpus": 1.0,
        "log_path": tmp_path / f"{name}.log",
        "name": name,
    }
    params.update(kw)
    return run_container(image, ["python", "-c", code], **params)  # type: ignore[arg-type]


def _container_gone(name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", f"name=^{name}$"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() == ""


@pytest.mark.docker
def test_docker_build_cached(tiny_image: str, tmp_path: Path) -> None:
    again = build_image(TINY_DOCKERFILE, tiny_image, timeout_s=60, log_path=tmp_path / "b.log")
    assert again.ok and again.cached


@pytest.mark.docker
def test_docker_run_print(tiny_image: str, tmp_path: Path) -> None:
    res = _run(tiny_image, "print(1)", tmp_path)
    assert res.exit_code == 0 and not res.timed_out and not res.oom_killed
    assert res.log_path.read_text().strip() == "1"
    assert _container_gone(res.name)


@pytest.mark.docker
def test_docker_timeout_kills(tiny_image: str, tmp_path: Path) -> None:
    res = _run(tiny_image, "import time; time.sleep(60)", tmp_path, timeout_s=3)
    assert res.timed_out and res.exit_code != 0
    assert res.duration_s < 30
    assert _container_gone(res.name)


def _host_procs(pattern: str) -> list[str]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
    return r.stdout.split()


@pytest.mark.docker
def test_docker_build_timeout_cancels_buildkit(tiny_image: str, tmp_path: Path) -> None:
    """Killing the docker CLI on timeout cancels the BuildKit build: the RUN step's process,
    seen on the host while the build runs, is gone shortly after, and no image is tagged."""
    if shutil.which("pgrep") is None:
        pytest.skip("pgrep not found")
    secs = 900 + uuid.uuid4().int % 90  # unique-ish sleep length to find the RUN process
    marker = f"sleep {secs}"
    tag = f"reproducer/test-cancel:{uuid.uuid4().hex[:8]}"
    dockerfile = f"FROM {tiny_image}\nRUN echo {uuid.uuid4().hex} && {marker}\n"
    seen: list[bool] = []

    def watch() -> None:
        for _ in range(8):
            time.sleep(1)
            seen.append(bool(_host_procs(marker)))

    watcher = threading.Thread(target=watch)
    watcher.start()
    res = build_image(dockerfile, tag, timeout_s=10, log_path=tmp_path / "b.log")
    watcher.join()
    assert res.timed_out and not res.ok
    assert any(seen), "RUN step never observed; cannot test cancellation"
    deadline = time.monotonic() + 15
    while _host_procs(marker) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert _host_procs(marker) == [], "BuildKit build kept running after the client was killed"
    assert not image_exists(tag)


@pytest.mark.docker
def test_docker_network_none(tiny_image: str, tmp_path: Path) -> None:
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1', timeout=5)\n"
        "    print('NETWORK_OPEN')\n"
        "except OSError as e:\n"
        "    print('BLOCKED', type(e).__name__)\n"
    )
    res = _run(tiny_image, code, tmp_path)
    assert res.exit_code == 0
    assert "BLOCKED" in res.log_path.read_text() and "NETWORK_OPEN" not in res.log_path.read_text()


@pytest.mark.docker
def test_docker_readonly_root_writable_work_out(tiny_image: str, tmp_path: Path) -> None:
    code = (
        "import os\n"
        "try:\n"
        "    open('/usr/evil', 'w').write('x'); print('ROOT_WRITABLE')\n"
        "except OSError as e:\n"
        "    print('ROOT_RO', e.errno)\n"
        "open('/work/w.txt', 'w').write('w'); open('/out/o.txt', 'w').write('o')\n"
        "open('/tmp/t.txt', 'w').write('t')\n"
        "print('uid', os.getuid())\n"
    )
    res = _run(tiny_image, code, tmp_path)
    log = res.log_path.read_text()
    assert res.exit_code == 0, log
    assert "ROOT_RO" in log and "ROOT_WRITABLE" not in log
    assert "uid 0" not in log
    assert (tmp_path / "work" / "w.txt").read_text() == "w"
    assert (tmp_path / "out" / "o.txt").read_text() == "o"
    assert _container_gone(res.name)


@pytest.mark.docker
def test_docker_data_mount_is_readonly(tiny_image: str, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "x.txt").write_text("data")
    code = (
        "print(open('/data/x.txt').read())\n"
        "try:\n"
        "    open('/data/y.txt', 'w'); print('DATA_WRITABLE')\n"
        "except OSError:\n"
        "    print('DATA_RO')\n"
    )
    res = _run(tiny_image, code, tmp_path, data_mounts={str(data): "/data"})
    log = res.log_path.read_text()
    assert "data" in log and "DATA_RO" in log and not (data / "y.txt").exists()


@pytest.mark.docker
def test_docker_nested_readonly_data_mount_under_work(tiny_image: str, tmp_path: Path) -> None:
    """A ro bind mount nested inside the rw /work bind mount: readable, not writable, while
    /work itself stays writable. The mountpoint is created by us (host user), not dockerd."""
    data = tmp_path / "cache"
    data.mkdir()
    (data / "x.txt").write_text("cached")
    code = (
        "print(open('/work/data/x.txt').read())\n"
        "try:\n"
        "    open('/work/data/y.txt', 'w'); print('DATA_WRITABLE')\n"
        "except OSError:\n"
        "    print('DATA_RO')\n"
        "open('/work/w.txt', 'w').write('w')\n"
    )
    res = _run(tiny_image, code, tmp_path, data_mounts={str(data): "/work/data"})
    log = res.log_path.read_text()
    assert res.exit_code == 0, log
    assert "cached" in log and "DATA_RO" in log and "DATA_WRITABLE" not in log
    assert not (data / "y.txt").exists()
    assert (tmp_path / "work" / "w.txt").read_text() == "w"
    mountpoint = tmp_path / "work" / "data"
    assert mountpoint.is_dir() and mountpoint.stat().st_uid == os.getuid()
    assert list(mountpoint.iterdir()) == []  # the cache never leaked into the working copy


def test_run_container_refuses_symlinked_mountpoint(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "data").symlink_to("/etc")
    with pytest.raises(ValueError, match="symlink"):
        run_container(
            "img:tag",
            ["true"],
            workdir=work,
            out_dir=tmp_path / "out",
            data_mounts={str(tmp_path): "/work/data"},
            gpu=False,
            timeout_s=1,
            log_path=tmp_path / "x.log",
            name="reproducer-never-started",
        )


# --- known-fix support: unbuffered output, mount pairs, prep dirs ----------------------------


def test_run_args_unbuffered_python() -> None:
    assert "PYTHONUNBUFFERED=1" in _flag_values(_run_args(), "-e")


def test_run_args_mount_pairs_same_host_two_targets() -> None:
    args = _run_args(data_mounts=[("/cache", "/work/data"), ("/cache", "/datasets")])
    vols = _flag_values(args, "-v")
    assert "/cache:/work/data:ro" in vols and "/cache:/datasets:ro" in vols
    with pytest.raises(ValueError, match="duplicate"):
        _run_args(data_mounts=[("/a", "/work/data"), ("/b", "/work/data")])


def test_prepare_workdir_prep_dirs(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    (clone / "util").mkdir(parents=True)
    (clone / "train.py").write_text("x = 1\n")
    dest = prepare_workdir(clone, [], tmp_path / "w1", prep_dirs=["logs", "out/ckpt", "util"])
    assert (dest / "logs").is_dir() and (dest / "out" / "ckpt").is_dir()
    assert not (clone / "logs").exists()  # the clone is never modified
    for bad in ["/abs", "../up", "a/../b", ".", ""]:
        with pytest.raises(ValueError):
            prepare_workdir(clone, [], tmp_path / f"bad{abs(hash(bad))}", prep_dirs=[bad])
    (clone / "link").symlink_to("/etc")
    with pytest.raises(ValueError, match="symlink"):
        prepare_workdir(clone, [], tmp_path / "w2", prep_dirs=["link/x"])
