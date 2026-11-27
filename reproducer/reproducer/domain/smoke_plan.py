"""Smoke test logic (pure): command plan, progress / timing detection, pass-fail signal.

- plan_smoke: the verified example command (shlex) plus budget overrides ONLY for argparse
  arguments whose dest is an epoch budget (-> 1) or a step budget (-> SMOKE_STEPS); an
  option already in the command gets its value replaced, otherwise it is appended. Nothing
  else in the command changes. No budget argument -> no overrides (the progress-based stop
  of the runner bounds the run).
- progress: tqdm rates ("12.3it/s", "1.5s/it") or a line starting with the literal prefix of
  a code-found metric line (e.g. "test_acc:").
- timing: the last tqdm bar with a total -> elapsed + remaining = seconds per bar (an epoch
  when the epoch budget was overridden); otherwise wall time of an exit-0 run with an epoch
  override; else unknown.
- decide_signal: the decision table (see its docstring).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.traceback_parse import ParsedTraceback

EPOCH_DESTS = frozenset({"epochs", "num_epochs", "n_epochs", "max_epochs"})
STEP_DESTS = frozenset({"max_steps", "max_iters", "iterations"})
SMOKE_EPOCHS = 1
SMOKE_STEPS = 20

Signal = Literal[
    "passed_exit0",
    "passed_progress",
    "traceback",
    "oom",
    "timeout_no_progress",
    "infra",
    "import_error",
]
PASSING_SIGNALS = frozenset({"passed_exit0", "passed_progress"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BudgetArg(_Frozen):
    """The parts of an argparse ArgSpec the plan needs (avoids a graph -> domain import)."""

    option_strings: tuple[str, ...]
    dest: str | None
    action: str | None = None
    default: Any = None  # literal argparse default (None if absent / not a literal)


class Override(_Frozen):
    dest: str
    option: str
    value: str
    previous: str | None = None  # value in the example command, if it was there


class SmokePlan(_Frozen):
    argv: tuple[str, ...]
    command: str  # shlex.join(argv)
    overrides: tuple[Override, ...] = ()


def _budget_value(dest: str | None) -> str | None:
    if dest in EPOCH_DESTS:
        return str(SMOKE_EPOCHS)
    if dest in STEP_DESTS:
        return str(SMOKE_STEPS)
    return None


def _set_option(
    argv: list[str], options: Sequence[str], option: str, value: str
) -> tuple[str, str | None]:
    """Replace the value of the first of `options` in argv (space or '=' form) in place, else
    append `option value`. Returns (option used, previous value or None)."""
    for i, tok in enumerate(argv):
        if tok in options and i + 1 < len(argv):
            previous = argv[i + 1]
            argv[i + 1] = value
            return tok, previous
        hit = next((o for o in options if tok.startswith(o + "=")), None)
        if hit is not None:
            previous = tok.split("=", 1)[1]
            argv[i] = f"{hit}={value}"
            return hit, previous
    argv += [option, value]
    return option, None


def plan_smoke(
    example: str, arguments: Iterable[BudgetArg], fixes: Sequence[Override] = ()
) -> SmokePlan:
    """Smoke argv from the verified example command line: budget overrides, then the known-fix
    overrides (e.g. a halved batch size), each replacing in place or appended. Raises
    ValueError if the command does not tokenize."""
    argv = shlex.split(example, comments=True)
    if not argv:
        raise ValueError("empty example command")
    overrides: list[Override] = []
    for arg in arguments:
        value = _budget_value(arg.dest)
        if value is None or arg.action not in (None, "store"):
            continue
        options = [o for o in arg.option_strings if o.startswith("-")]
        if not options:
            continue  # positional budget: never rewritten
        option = next((o for o in options if o.startswith("--")), options[0])
        option, previous = _set_option(argv, options, option, value)
        assert arg.dest is not None
        overrides.append(Override(dest=arg.dest, option=option, value=value, previous=previous))
    for fix in fixes:
        option, previous = _set_option(argv, (fix.option,), fix.option, fix.value)
        overrides.append(fix.model_copy(update={"option": option, "previous": previous}))
    return SmokePlan(argv=tuple(argv), command=shlex.join(argv), overrides=tuple(overrides))


# --- progress / timing -----------------------------------------------------------------------

_RATE_RE = re.compile(r"\d(?:\.\d+)?\s?(?:it/s|s/it)\b")
_TQDM_BAR_RE = re.compile(
    r"(?P<n>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:]+|\?)"
)
_LITERAL_RE = re.compile(
    r"""(?:print|write|info|debug|warning|log)\s*\(\s*[fbru]*(?P<q>['"])(?P<s>.*?)(?P=q)"""
)


def metric_prefixes(metric_lines: Iterable[str]) -> tuple[str, ...]:
    """Literal prefix of the first string in each code-found metric line, cut before the first
    format placeholder ('%' or '{'), stripped; kept if >= 3 chars with a letter."""
    out: list[str] = []
    for line in metric_lines:
        m = _LITERAL_RE.search(line)
        if not m:
            continue
        prefix = re.split(r"[%{]", m["s"], maxsplit=1)[0].strip()
        if len(prefix) >= 3 and re.search(r"[A-Za-z]", prefix) and prefix not in out:
            out.append(prefix)
    return tuple(out)


def _lines(log_text: str) -> list[str]:
    return log_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def progress_seen(log_text: str, prefixes: Sequence[str] = ()) -> bool:
    for line in _lines(log_text):
        if _RATE_RE.search(line):
            return True
        s = line.strip()
        if prefixes and any(s.startswith(p) for p in prefixes):
            return True
    return False


def _seconds(clock: str) -> int:
    total = 0
    for part in clock.split(":"):
        total = total * 60 + int(part)
    return total


class Timing(_Frozen):
    sec_per_epoch: float | None
    method: Literal["tqdm", "wall", "none"]
    confidence: Literal["medium", "low", "none"]
    detail: str = ""


def estimate_timing(
    log_text: str, *, overrides: Sequence[Override], exit_code: int | None, duration_s: float
) -> Timing:
    epoch_override = any(o.dest in EPOCH_DESTS for o in overrides)
    last = None
    for line in _lines(log_text):
        for m in _TQDM_BAR_RE.finditer(line):
            if int(m["total"]) > 0:
                last = m
    if last is not None:
        n, total = int(last["n"]), int(last["total"])
        elapsed = _seconds(last["elapsed"])
        if n >= total or last["remaining"] != "?":
            remaining = 0 if n >= total else _seconds(last["remaining"])
            complete = n >= total
            return Timing(
                sec_per_epoch=float(elapsed + remaining),
                method="tqdm",
                confidence="medium" if complete and epoch_override else "low",
                detail=f"last bar {n}/{total} [{last['elapsed']}<{last['remaining']}]",
            )
    if exit_code == 0 and epoch_override:
        return Timing(
            sec_per_epoch=round(duration_s, 1),
            method="wall",
            confidence="low",
            detail="wall time of a 1-epoch run (includes startup and evaluation)",
        )
    return Timing(sec_per_epoch=None, method="none", confidence="none")


# --- decision --------------------------------------------------------------------------------


def decide_signal(
    *,
    exit_code: int | None,
    timed_out: bool,
    oom_killed: bool,
    progress: bool,
    traceback: ParsedTraceback | None,
) -> Signal:
    """Decision table for the repo run (infra / import_error are decided before it):
    container OOM-killed, or a CUDA out-of-memory traceback      -> oom
    exit 0 (not timed out)                                       -> passed_exit0
    timed out: progress and no traceback                         -> passed_progress
    timed out: a traceback                                       -> traceback
    timed out: no progress                                       -> timeout_no_progress
    no exit code reported (container never ran / vanished)      -> infra
    any other non-zero exit                                      -> traceback
    """
    if oom_killed or (traceback is not None and traceback.kind == "oom"):
        return "oom"
    if timed_out:
        if traceback is not None and traceback.kind != "killed":
            return "traceback"
        return "passed_progress" if progress else "timeout_no_progress"
    if exit_code == 0:
        return "passed_exit0"
    if exit_code is None:
        return "infra"
    return "traceback"
