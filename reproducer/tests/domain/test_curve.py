"""Tests for reproducer.domain.curve (CSV / log points -> CurveSummary)."""

from __future__ import annotations

import math

import pytest

from reproducer.domain.curve import points_from_csv, points_from_log, summarize
from reproducer.domain.run_plan import parse_metric_line

PATTERN = parse_metric_line("tqdm.write('test_acc: %.3f' % (test_acc))").pattern
PREAMBLE = 'dataset,cifar10\nmodel,resnet18\nepochs,40\n""\n'


def _acc(epoch: int) -> float:
    """Cutout-like test accuracy: rises, then a flat noisy plateau from epoch 20; the best
    epoch (25) is a single lucky spike."""
    if epoch < 20:
        return 0.5 + 0.02 * epoch
    noise = (0.001, -0.001, 0.0005, -0.0005)[epoch % 4]
    return 0.95 + noise + (0.004 if epoch == 25 else 0.0)


def _csv(n: int = 40, acc=_acc) -> str:  # type: ignore[no-untyped-def]
    rows = [f"{e},{0.9:.4f},{acc(e):.4f}" for e in range(n)]
    return PREAMBLE + "epoch,train_acc,test_acc\n" + "\n".join(rows) + "\n"


def _summary(points, **kw):  # type: ignore[no-untyped-def]
    base = dict(metric="test_error", column="test_acc",
                conversion="accuracy_fraction_to_error_percent", direction="min",
                source="csv", source_ref="runs/x/artifacts/logs/c10.csv")  # fmt: skip
    return summarize(points, **(base | kw))


def test_points_from_csv_skips_preamble_and_picks_the_column() -> None:
    pts = points_from_csv(_csv(), "test_acc")
    assert pts is not None and len(pts) == 40
    assert pts[0] == (0, 0.5) and pts[-1][0] == 39
    assert points_from_csv(_csv(), "val_acc") is None  # no such column
    assert points_from_csv("a,b\n1,2\n", "b") is None  # no epoch column


def test_cutout_like_csv_summary() -> None:
    s = _summary(points_from_csv(_csv(), "test_acc"))
    assert s.n_points == 40 and not s.nonfinite
    # error % = (1 - acc) * 100; min is best
    assert s.best_epoch == 25 and s.best == pytest.approx(4.7)
    assert s.final_epoch == 39 and s.final == pytest.approx((1 - _acc(39)) * 100)
    assert s.best != s.final  # final vs best epoch differ
    assert s.k == 10 and s.last_k_mean == pytest.approx(5.0, abs=0.05)
    assert not s.still_improving
    # every point from epoch 20 on sits within 3 sigma of the last-10 mean; the lucky spike at
    # 25 is outside, so the plateau starts right after it
    assert s.plateau_from_epoch == 26


def test_still_improving_has_no_plateau() -> None:
    s = _summary(points_from_csv(_csv(n=20), "test_acc"))  # still on the rising part
    assert s.still_improving and s.plateau_from_epoch is None
    assert s.last_k_slope is not None and s.last_k_slope < 0  # error falling


def test_max_direction_and_identity_conversion() -> None:
    pts = [(e, 50.0 + e) for e in range(5)]
    s = _summary(pts, metric="test_acc", conversion="identity", direction="max")
    assert s.best == 54.0 and s.best_epoch == 4 and s.still_improving


def test_fallback_from_log_lines() -> None:
    log = "Epoch 0: 100%|###| 391/391\ntest_acc: 0.500\nEpoch 1\ntest_acc: 0.700\ntest_acc: 0.690\n"
    pts = points_from_log(log, PATTERN)
    assert pts == [(0, 0.5), (1, 0.7), (2, 0.69)]
    s = _summary(pts, source="log", source_ref="runs/x/run.log")
    assert s.source == "log" and s.best_epoch == 1 and s.best == pytest.approx(30.0)
    assert s.final == pytest.approx(31.0)


def test_nonfinite_detected_and_excluded_from_stats() -> None:
    log = "test_acc: 0.500\ntest_acc: 0.600\ntest_acc: nan\n"
    s = _summary(points_from_log(log, PATTERN), source="log")
    assert s.nonfinite and s.n_points == 3
    assert s.final == pytest.approx(40.0) and s.final_epoch == 1
    csv_pts = points_from_csv("epoch,test_acc\n0,0.5\n1,nan\n", "test_acc")
    assert csv_pts is not None and math.isnan(csv_pts[1][1])
    assert _summary(csv_pts).nonfinite


def test_all_nonfinite_gives_empty_stats() -> None:
    s = _summary([(0, float("nan"))])
    assert s.nonfinite and s.final is None and s.best is None and s.plateau_from_epoch is None


def test_conversion_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _summary([(0, 95.9)])  # a percent where a fraction is expected
