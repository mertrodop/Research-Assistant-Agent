"""GPU provider: query the host GPU with nvidia-smi (argument list, no shell, timeout)."""

from __future__ import annotations

import subprocess

from reproducer.domain.env_rules import NVIDIA_SMI_ARGS, GpuInfo, parse_nvidia_smi_csv

QUERY_TIMEOUT_S = 10.0


def query_gpu(timeout: float = QUERY_TIMEOUT_S) -> GpuInfo | None:
    """Host GPU facts, or None when nvidia-smi is missing, fails, times out or prints
    something unparseable (callers then plan CPU-only or stop)."""
    try:
        proc = subprocess.run(
            list(NVIDIA_SMI_ARGS),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return parse_nvidia_smi_csv(proc.stdout)
    except ValueError:
        return None
