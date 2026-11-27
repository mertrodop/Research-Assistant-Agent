"""Run outcome classification (code, never LLM): what happened to one full run (M4).

classify_run(evaluation, run evidence, log tail) -> RunOutcome{cls, direction, diff, tol,
reason}, checked in this order:
  not_completed     the run was aborted (Ctrl-C / exception after the container started)
  verified          exit 0, claim metric finite, |diff| <= tol
  out_of_tolerance  exit 0, claim metric finite, |diff| > tol (better or worse than the paper)
  diverged          no finite claim metric and: a non-finite metric (run evidence `diverged`,
                    or a NaN / inf point in the curve), or a loss explosion in the log tail
                    (a loss-like value that is NaN / inf or >= LOSS_EXPLOSION, "loss is nan")
  not_completed     exit 0 without a claim metric (no metric line / conversion failed); no
                    exit code (exit -1: timeout or the container vanished); a non-zero exit
                    before any progress
  crashed           a non-zero exit after progress (metric lines, a tqdm bar or curve points):
                    the log tail goes through traceback_parse + smoke_plan.decide_signal +
                    rules/classify.py exactly like a failed smoke (known fix / critic / infra
                    / unfixable), the Classification is attached
diff = ours - paper (signed); tol = max(abs, rel * |paper|), the pre-registered tolerance.
direction: worse | better vs the paper by the claim metric's direction (max: higher is
better; min: lower is better); None without a value, at diff 0, or an unknown direction.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.curve import CurveSummary
from reproducer.domain.evaluate import ClaimEvaluation, is_aborted, run_value
from reproducer.domain.evidence import Evidence
from reproducer.domain.run_plan import metric_direction
from reproducer.domain.smoke_plan import decide_signal, progress_seen
from reproducer.domain.traceback_parse import parse_last_traceback
from reproducer.rules.classify import Classification, ClassifyContext, classify

RunClass = Literal["verified", "out_of_tolerance", "diverged", "crashed", "not_completed"]
Direction = Literal["worse", "better"]

LOSS_EXPLOSION = 1e6
_LOSS_KEY = r"(?:loss|xentropy|nll|cross_?entropy)\w*"
_LOSS_NONFINITE_RE = re.compile(
    rf"(?i)\b{_LOSS_KEY}\s*[:=]\s*[-+]?(?:nan|inf(?:inity)?)\b|"
    r"(?i:\bloss\s+(?:is|became|=)\s+[-+]?(?:nan|inf))"
)
_LOSS_VALUE_RE = re.compile(
    rf"(?i)\b{_LOSS_KEY}\s*[:=]\s*(?P<v>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunOutcome(_Frozen):
    claim_id: str
    run_evidence_id: str
    cls: RunClass
    direction: Direction | None
    value: float | None  # ours, in the claim metric (completed runs only)
    paper: float
    diff: float | None  # ours - paper
    tol: float
    reason: str
    classification: Classification | None = None  # crashed only


def tolerance_bound(evaluation: ClaimEvaluation) -> float:
    t = evaluation.tolerance
    return max(t.abs or 0.0, (t.rel or 0.0) * abs(evaluation.paper_value))


def direction_vs_paper(metric: str, diff: float | None) -> Direction | None:
    if diff is None or diff == 0:
        return None
    try:
        d = metric_direction(metric)
    except ValueError:
        return None
    return "better" if (diff > 0) == (d == "max") else "worse"


def loss_explosion(log_tail: str) -> str | None:
    """The first loss-explosion evidence in the log tail, else None."""
    m = _LOSS_NONFINITE_RE.search(log_tail)
    if m:
        return m.group(0).strip()
    for m in _LOSS_VALUE_RE.finditer(log_tail):
        v = float(m["v"])
        if math.isfinite(v) and abs(v) >= LOSS_EXPLOSION:
            return m.group(0).strip()
    return None


def _progress(log_tail: str, metric_pattern: str | None, curve: CurveSummary | None) -> bool:
    if curve is not None and curve.n_points > 0:
        return True
    if metric_pattern and re.search(metric_pattern, log_tail):
        return True
    return progress_seen(log_tail)


def classify_run(
    evaluation: ClaimEvaluation,
    run: Evidence,
    log_tail: str,
    *,
    metric_pattern: str | None = None,
    curve: CurveSummary | None = None,
    ctx: ClassifyContext | None = None,
    command: str = "",
) -> RunOutcome:
    """Classify one run_result of the evaluated claim. Raises ValueError for another kind or
    another claim's run."""
    if run.kind != "run_result":
        raise ValueError(f"classify_run needs run_result evidence, got {run.kind!r}")
    c: Mapping[str, Any] = run.content
    if c.get("claim_id") != evaluation.claim_id:
        raise ValueError(f"run {run.evidence_id[:12]} is not a run of {evaluation.claim_id}")
    tol = tolerance_bound(evaluation)
    paper = evaluation.paper_value

    def out(cls: RunClass, reason: str, value: float | None = None,
            classification: Classification | None = None) -> RunOutcome:  # fmt: skip
        diff = None if value is None else value - paper
        return RunOutcome(
            claim_id=evaluation.claim_id, run_evidence_id=run.evidence_id, cls=cls,
            direction=direction_vs_paper(evaluation.metric, diff), value=value, paper=paper,
            diff=diff, tol=tol, reason=reason, classification=classification,
        )  # fmt: skip

    if is_aborted(c):
        return out("not_completed", f"aborted: {c.get('abort_reason') or 'interrupted'}")

    ours = run_value(c, evaluation.metric)
    if ours is not None:
        diff = ours - paper
        cmp = f"|diff| {abs(diff):.4g} {'<=' if abs(diff) <= tol else '>'} tol {tol:.4g}"
        if abs(diff) <= tol:
            return out("verified", f"completed, within tolerance ({cmp})", ours)
        side = direction_vs_paper(evaluation.metric, diff) or "different"
        return out("out_of_tolerance", f"completed, {side} than the paper ({cmp})", ours)

    if c.get("diverged"):
        return out("diverged", "non-finite metric in the run result")
    if curve is not None and curve.nonfinite:
        return out("diverged", f"non-finite {curve.column} in the curve ({curve.source})")
    boom = loss_explosion(log_tail)
    if boom is not None:
        return out("diverged", f"loss explosion in the log: {boom!r}")

    code = c.get("exit_code")
    if code == 0:
        return out("not_completed", "exit 0 but no claim metric (no metric line or "
                                    "conversion failed)")  # fmt: skip
    if not isinstance(code, int) or code == -1:
        return out("not_completed", "no exit code (timeout or the container vanished)")
    if not _progress(log_tail, metric_pattern, curve):
        return out("not_completed", f"exit {code} before any progress")

    tb = parse_last_traceback(log_tail, code)
    signal = decide_signal(
        exit_code=code, timed_out=False, oom_killed=False, progress=True, traceback=tb
    )
    smoke_like = {
        "passed": False,
        "signal": signal,
        "traceback": tb.model_dump(mode="json") if tb is not None else None,
        "command": command,
    }
    cl = classify(smoke_like, ctx or ClassifyContext(repo_files=frozenset()))
    return out("crashed", f"exit {code} after progress: {cl.cls} ({cl.reason})",
               classification=cl)  # fmt: skip
