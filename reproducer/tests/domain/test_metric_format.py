"""Tests for reproducer.domain.metric_format (display precision only)."""

from __future__ import annotations

import math

import pytest

from reproducer.domain.contracts import ResultClaim
from reproducer.domain.metric_format import (
    format_metric,
    format_metrics,
    format_raw,
    metric_decimals,
)

QUOTE = "ResNet18 + cutout 9.31 ± 0.18 3.99 ± 0.13 34.98 ± 0.29 21.96 ± 0.24 -"


def _claim(value: float, quote: str = QUOTE) -> ResultClaim:
    return ResultClaim(
        claim_id="c", metric="test_error", value=value, dataset="cifar10", split="test",
        setting=None, location="Table 1", quote=quote,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("value", "quote", "decimals"),
    [
        (3.99, QUOTE, 2),
        (21.96, QUOTE, 2),
        (4.1, "error 4.10 ± 0.1", 2),  # as written in the quote, not the float repr (1)
        (4.1, "no number here", 1),  # fallback: the float itself
        (4.0, "error 4 %", 0),
        (96.5, "acc 96.500", 3),
        (12.0, "error 12.0 and 2.12", 1),
    ],
)
def test_metric_decimals(value: float, quote: str, decimals: int) -> None:
    assert metric_decimals(_claim(value, quote)) == decimals


def test_format_metric_uses_claim_precision_and_keeps_value() -> None:
    v = 100 * (1 - 0.959)  # 4.100000000000003
    assert str(v) != "4.10"
    assert format_metric(v, 2) == "4.10"
    assert format_metric(21.955, 2) in {"21.95", "21.96"}  # float rounding, display only
    assert format_metric(None, 2) == "-"
    assert format_metric(math.nan, 2) == "nan"


def test_format_metrics_claim_metric_and_raw() -> None:
    metrics = {"test_error": 100 * (1 - 0.959), "raw:test_acc": 0.959}
    assert format_metrics(metrics, 2) == "test_error=4.10, raw:test_acc=0.959"
    assert format_raw(0.1 + 0.2) == "0.3"
    assert format_raw(None) == "-"
