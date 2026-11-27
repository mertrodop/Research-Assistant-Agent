"""Training-config lines (pure, code-first): where a repo sets its optimizer, LR scheduler and
explicit learning-rate updates, found before any LLM runs so the orchestrator can cite them
when it checks the paper setup against the repo.

    optimizer   a call named like a torch optimizer (SGD, Adam, AdamW, ...), its keyword args
                (lr, momentum, nesterov, weight_decay, ...) as source text
    scheduler   a call named like an LR scheduler (MultiStepLR, StepLR, CosineAnnealingLR,
                ReduceLROnPlateau, ...), its keyword args (milestones, gamma, ...)
    lr_update   an assignment to x['lr'] (e.g. `param_group['lr'] = lr` in a manual schedule)

ast first; a file that does not parse (e.g. Python 2) falls back to a per-line regex.
The result is stored as derived repo evidence '<entrypoint>#train_config' (one per entrypoint,
covering the entrypoint and the local modules it imports). No file, network or subprocess IO.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.evidence import Evidence, make_evidence

ConfigKind = Literal["optimizer", "scheduler", "lr_update"]

OPTIMIZER_NAMES = frozenset(
    {
        "SGD",
        "Adam",
        "AdamW",
        "Adamax",
        "Adagrad",
        "Adadelta",
        "RMSprop",
        "NAdam",
        "RAdam",
        "ASGD",
        "Rprop",
        "LBFGS",
        "SparseAdam",
        "LARS",
        "LAMB",
        "Lamb",
        "Adafactor",
    }
)
_SCHEDULER_RE = re.compile(
    r"^(?:\w*LR|ReduceLROnPlateau|CosineAnnealingWarmRestarts|ChainedScheduler"
    r"|get_\w*schedule\w*)$"
)
MAX_LINE_CHARS = 500
_KWARG_RE = re.compile(
    r"\b(lr|momentum|nesterov|weight_decay|betas|eps|dampening|milestones|gamma|step_size"
    r"|T_max|T_0|eta_min|max_lr|total_steps)\s*=\s*(\[[^\]]*\]|\([^)]*\)|[^,()\s]+)"
)
_REGEX_CALL_RE = re.compile(r"(?<![\w])((?:[A-Za-z_]\w*\.)*)([A-Za-z_]\w*)\s*\(")
_LR_ASSIGN_RE = re.compile(r"\[\s*['\"]lr['\"]\s*\]\s*(?:[-+*/]?=)(?!=)")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrainConfigLine(_Frozen):
    path: str
    line_no: int  # 1-based first line of the statement part found
    end_line_no: int
    kind: ConfigKind
    name: str  # SGD, MultiStepLR, ... ; "lr" for lr_update
    line: str  # source lines line_no..end_line_no, stripped, joined with one space
    kwargs: dict[str, str] = {}  # keyword -> source text, e.g. {"momentum": "0.9"}


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kind_of(name: str) -> ConfigKind | None:
    if name in OPTIMIZER_NAMES:
        return "optimizer"
    if _SCHEDULER_RE.match(name):
        return "scheduler"
    return None


def _joined(lines: list[str], start: int, end: int) -> str:
    text = " ".join(ln.strip() for ln in lines[start - 1 : end] if ln.strip())
    return text[:MAX_LINE_CHARS]


def _is_lr_subscript(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "lr"
    )


def _scan_ast(path: str, text: str, tree: ast.Module) -> list[TrainConfigLine]:
    lines = text.splitlines()
    out: list[TrainConfigLine] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            kind = _kind_of(name) if name else None
            if kind is None or name is None:
                continue
            end = node.end_lineno or node.lineno
            kwargs = {
                kw.arg: ast.get_source_segment(text, kw.value) or ast.unparse(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            }
            out.append(
                TrainConfigLine(
                    path=path,
                    line_no=node.lineno,
                    end_line_no=end,
                    kind=kind,
                    name=name,
                    line=_joined(lines, node.lineno, end),
                    kwargs=kwargs,
                )
            )
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_lr_subscript(t) for t in targets):
                end = node.end_lineno or node.lineno
                value = ast.get_source_segment(text, node.value) or ast.unparse(node.value)
                out.append(
                    TrainConfigLine(
                        path=path,
                        line_no=node.lineno,
                        end_line_no=end,
                        kind="lr_update",
                        name="lr",
                        line=_joined(lines, node.lineno, end),
                        kwargs={"lr": value},
                    )
                )
    return out


def _scan_regex(path: str, text: str) -> list[TrainConfigLine]:
    out: list[TrainConfigLine] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        found: tuple[ConfigKind, str] | None = None
        for m in _REGEX_CALL_RE.finditer(line):
            kind = _kind_of(m.group(2))
            if kind is not None:
                found = (kind, m.group(2))
                break
        if found is None and _LR_ASSIGN_RE.search(line):
            found = ("lr_update", "lr")
        if found is None:
            continue
        out.append(
            TrainConfigLine(
                path=path,
                line_no=no,
                end_line_no=no,
                kind=found[0],
                name=found[1],
                line=line[:MAX_LINE_CHARS],
                kwargs={k: v for k, v in _KWARG_RE.findall(line)},
            )
        )
    return out


def find_training_config_lines(texts: Mapping[str, str]) -> tuple[TrainConfigLine, ...]:
    """Optimizer / scheduler calls and x['lr'] assignments in each text (ast; per-line regex
    for files that do not parse). Sorted by path, then line; one entry per statement part."""
    out: list[TrainConfigLine] = []
    for path in sorted(texts):
        text = texts[path]
        try:
            tree = ast.parse(text, filename=path)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            found = _scan_regex(path, text)
        else:
            found = _scan_ast(path, text, tree)
        seen: set[tuple[int, str]] = set()
        for t in sorted(found, key=lambda t: (t.line_no, t.end_line_no, t.name)):
            if (t.line_no, t.name) not in seen:
                seen.add((t.line_no, t.name))
                out.append(t)
    return tuple(out)


def train_config_evidence(
    entrypoint: str,
    texts: Mapping[str, str],
    *,
    source: str,
    links: Iterable[str] = (),
) -> Evidence:
    """Derived repo evidence '<entrypoint>#train_config' over the given texts (the entrypoint
    and its local modules), linked to the observed evidence of those files."""
    lines = find_training_config_lines(texts)
    return make_evidence(
        kind="repo_file",
        source=source,
        locator=f"{entrypoint}#train_config",
        content={
            "parser": "train_config",
            "scanned": sorted(texts),
            "result": [t.model_dump(mode="json") for t in lines],
        },
        confidence="derived",
        authority="repo",
        links=tuple(sorted(set(links))),
    )
