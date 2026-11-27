"""Tests for reproducer.providers.gpu (nvidia-smi is faked; nothing touches the host GPU)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from reproducer.domain.env_rules import NVIDIA_SMI_ARGS
from reproducer.providers import gpu

SMI_CSV = (
    "name, compute_cap, driver_version, memory.total [MiB]\n"
    "NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 580.178.04, 8188 MiB\n"
)


def _fake_run(returncode: int = 0, stdout: str = SMI_CSV, exc: Exception | None = None) -> Any:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def run(args: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kw))
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_query_gpu_parses_and_uses_arg_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_run()
    monkeypatch.setattr(gpu.subprocess, "run", fake)
    info = gpu.query_gpu()
    assert info is not None and info.compute_capability == (8, 9) and info.memory_mb == 8188
    args, kw = fake.calls[0]
    assert args == list(NVIDIA_SMI_ARGS) and kw["timeout"] > 0 and "shell" not in kw


@pytest.mark.parametrize(
    "fake",
    [
        _fake_run(exc=FileNotFoundError("nvidia-smi")),
        _fake_run(exc=subprocess.TimeoutExpired("nvidia-smi", 10)),
        _fake_run(returncode=9, stdout=""),
        _fake_run(stdout="No devices were found\n"),
    ],
)
def test_query_gpu_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    monkeypatch.setattr(gpu.subprocess, "run", fake)
    assert gpu.query_gpu() is None
