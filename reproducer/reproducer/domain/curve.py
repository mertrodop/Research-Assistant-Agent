"""Curve projection: a run's training curve -> a compact CurveSummary (M4).

Pure domain: text in, model out. The Reflexion step never sees raw logs, only this summary.

Sources, in order:
  csv  a per-epoch artifact (e.g. logs/<name>.csv): the first row with an "epoch" cell and
       the metric column is the header (rows before it, e.g. an args dump, are skipped); the
       data rows follow it (same width, numeric epoch). Rows that do not parse are skipped.
  log  every match of the plan's metric regex in the log, one point per match; epoch = the
       match index (0-based).
Values are converted with the plan's conversion (repo value -> claim metric), so best / final
are in the claim's metric and unit; best uses the claim metric's direction (max | min).

Statistics over the FINITE points only (nonfinite records whether any point was NaN / inf):
  final, final_epoch       the last point
  best, best_epoch         max / min by direction (first epoch on ties)
  last_k_mean / std        over the last k = 10 points (sample std; 0 for one point)
  last_k_slope             least-squares slope over the last k points (claim units / epoch)
  still_improving          the fitted change over the window, in the improving direction,
                           exceeds the window's noise: slope * (k - 1) > last_k_std
  plateau_from_epoch       None while still improving; else the first epoch from which every
                           later point stays within PLATEAU_SIGMA * last_k_std of last_k_mean
                           (a single lucky epoch does not move it, unlike a running best)
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.contracts import Conversion
from reproducer.domain.run_plan import Direction, convert

K = 10
PLATEAU_SIGMA = 3.0
DECIMALS = 6
EPOCH_COLUMN = "epoch"

CurveSource = Literal["csv", "log"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CurveSummary(_Frozen):
    metric: str  # the claim metric the values were converted to
    column: str  # the repo's raw metric name (CSV column / metric line name)
    conversion: str
    direction: Direction
    n_points: int  # all points, finite or not
    final: float | None
    final_epoch: int | None
    best: float | None
    best_epoch: int | None
    k: int
    last_k_mean: float | None
    last_k_std: float | None
    last_k_slope: float | None
    still_improving: bool
    plateau_from_epoch: int | None
    nonfinite: bool
    source: CurveSource
    source_ref: str  # the artifact / log ref the points came from


Point = tuple[int, float]  # (epoch, raw value)


def _num(cell: str) -> float | None:
    try:
        return float(cell.strip())
    except ValueError:
        return None


def points_from_csv(text: str, column: str) -> list[Point] | None:
    """(epoch, value) rows of `column` from a per-epoch CSV; None if no header row has both an
    epoch cell and `column` (case-insensitive)."""
    rows = list(csv.reader(io.StringIO(text)))
    for i, row in enumerate(rows):
        names = [c.strip().lower() for c in row]
        if EPOCH_COLUMN not in names or column.lower() not in names:
            continue
        ei, vi = names.index(EPOCH_COLUMN), names.index(column.lower())
        out: list[Point] = []
        for data in rows[i + 1 :]:
            if len(data) != len(row):
                continue
            epoch, value = _num(data[ei]), _num(data[vi])
            if epoch is None or value is None or not math.isfinite(epoch):
                continue
            if epoch != int(epoch):
                continue
            out.append((int(epoch), value))
        return out
    return None


def points_from_log(log_text: str, pattern: str) -> list[Point]:
    """One point per metric-regex match (named group `value`), epoch = match index."""
    return [(i, float(m["value"])) for i, m in enumerate(re.finditer(pattern, log_text))]


def _r(v: float) -> float:
    return round(v, DECIMALS)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den


def summarize(
    points: Sequence[Point],
    *,
    metric: str,
    column: str,
    conversion: Conversion,
    direction: Direction,
    source: CurveSource,
    source_ref: str,
    k: int = K,
) -> CurveSummary:
    """CurveSummary of the points (raw repo values, converted here). Raises ValueError if a
    finite value is outside the conversion's input range."""
    pts = sorted(points, key=lambda p: p[0])
    conv = [(e, convert(v, conversion)) for e, v in pts]
    finite = [(e, v) for e, v in conv if math.isfinite(v)]
    nonfinite = len(finite) != len(conv)
    base = dict(
        metric=metric, column=column, conversion=conversion, direction=direction,
        n_points=len(conv), k=k, nonfinite=nonfinite, source=source, source_ref=source_ref,
    )  # fmt: skip
    if not finite:
        return CurveSummary(
            **base, final=None, final_epoch=None, best=None, best_epoch=None,
            last_k_mean=None, last_k_std=None, last_k_slope=None,
            still_improving=False, plateau_from_epoch=None,
        )  # fmt: skip
    sign = 1.0 if direction == "max" else -1.0
    best_e, best_v = max(finite, key=lambda p: (sign * p[1], -p[0]))
    window = finite[-k:]
    ys = [v for _, v in window]
    mean = statistics.fmean(ys)
    std = statistics.stdev(ys) if len(ys) > 1 else 0.0
    slope = _slope([float(e) for e, _ in window], ys) if len(ys) > 1 else 0.0
    improving = len(ys) > 1 and sign * slope * (len(ys) - 1) > std
    plateau: int | None = None
    if not improving:
        band = PLATEAU_SIGMA * std
        for e, v in reversed(finite):  # walk back while the points stay in the band
            if abs(v - mean) > band:
                break
            plateau = e
    final_e, final_v = finite[-1]
    return CurveSummary(
        **base,
        final=_r(final_v),
        final_epoch=final_e,
        best=_r(best_v),
        best_epoch=best_e,
        last_k_mean=_r(mean),
        last_k_std=_r(std),
        last_k_slope=_r(slope),
        still_improving=improving,
        plateau_from_epoch=plateau,
    )
