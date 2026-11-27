"""Display formatting of metric values at the claim's precision (printouts only; stored
values are never rounded). Pure, no IO."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from reproducer.domain.contracts import ResultClaim

_NUM_RE = re.compile(r"(?<![\d.])\d+(?:\.(\d+))?(?![\d.])")
MAX_DECIMALS = 6


def _repr_decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "E" in text or "." not in text:
        return 0
    frac = text.split(".", 1)[1].rstrip("0")
    return min(len(frac), MAX_DECIMALS)


def metric_decimals(claim: ResultClaim) -> int:
    """Decimal places the paper reports the claim value with: taken from the value as written
    in the claim quote (so "4.10" -> 2 although the float is 4.1); if the quote never writes
    it, from the float itself (3.99 -> 2)."""
    written = {len(m[1] or "") for m in _NUM_RE.finditer(claim.quote) if float(m[0]) == claim.value}
    if written:
        return min(max(written), MAX_DECIMALS)
    return _repr_decimals(claim.value)


def format_metric(value: float | None, decimals: int) -> str:
    """`value` rounded to `decimals` places for display ("4.10", not 4.100000000000003)."""
    if value is None:
        return "-"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{decimals}f}"


RAW_PREFIX = "raw:"


def format_raw(value: float | None) -> str:
    """A raw (unconverted) log value: up to 6 significant digits, no float noise."""
    if value is None:
        return "-"
    return f"{value:.6g}"


def format_metrics(metrics: Mapping[str, float], decimals: int) -> str:
    """`name=value, ...` with the claim metric(s) at `decimals` places and raw:<name> values
    via format_raw."""
    return ", ".join(
        f"{k}={format_raw(v) if k.startswith(RAW_PREFIX) else format_metric(v, decimals)}"
        for k, v in metrics.items()
    )
