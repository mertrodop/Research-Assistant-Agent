"""Run plans (pure): claim -> RunPlan, metric pattern / extraction / conversion.

- metric_pattern_from_line: the regex for a code-found metric line, built by code from the
  printed string literal (printf / str.format / f-string placeholders -> a number group).
- extract_metric: the last match, or the best one by direction (max for accuracy-like,
  min for error / loss).
- convert / allowed_conversions: repo metric -> claim metric and unit (a fixed table).
- build_run_plan: the verified example command (flags: source repo), then the claim mapping
  (source claim), then paper setup values COPIED by code from the paper_setup evidence (source
  paper) -- each only where it differs from the effective value (command value, else argparse
  default); equal values are recorded as "consistent". Unknown setup fields become
  Assumptions (the command value or repo default of the matching flag; eval_checkpoint ->
  which="last"), unless the unknown is resolved (`answers`, e.g. a human answer): then the
  resolved value is applied like a paper value (source human for a human_input answer, else
  paper; evidence = the resolving evidence) and nothing is assumed. Then applied Reflexion
  lessons (`lessons`, LessonOverride, in order; a later lesson replaces an earlier one on the
  same flag): an arg change sets its flag (source paper, evidence = the lesson's paper
  support), an eval_protocol change sets metric_source.which / conversion. The mapping (claim ->
  flags / setup fields / metric line) is judgment and comes from the caller; every value in
  it is checked here and in the RunPlan contract.
- render_command: the plan's argv (verified command order, replaced flags in place, new ones
  appended).
No file, network or subprocess IO.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reproducer.domain.contracts import (
    ArgOption,
    ArgSource,
    Conversion,
    ResultClaim,
    RunPlan,
    find_option,
    make_run_plan,
    parse_arg_value,
)
from reproducer.domain.evidence import Evidence
from reproducer.domain.grounding import normalize_repo_path
from reproducer.domain.unknowns import Assumption, unknown_id_for

Direction = Literal["max", "min"]
MetricKind = Literal["accuracy", "error", "loss", "other"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- metric pattern ------------------------------------------------------------------------

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?i:nan|inf)"
_STR_PREFIXES = frozenset({"", "f", "r", "u", "b", "fr", "rf", "br", "rb"})
_STR_LIT_RE = re.compile(
    r"""(?<![\w'"])(?P<prefix>[A-Za-z]{0,2})(?P<q>'''|\"\"\"|'|")(?P<body>.*?)(?<!\\)(?P=q)"""
)
_PERCENT_RE = re.compile(
    r"%(?:\([^)]*\))?[-+ #0]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?[diouxXeEfFgGcrsa]"
)
_ESCAPES = {"n": " ", "t": " ", "r": " "}
_NAME_RE = re.compile(r"[A-Za-z][\w]*")


class MetricPattern(_Frozen):
    name: str  # the printed metric name, e.g. "test_acc"
    pattern: str  # regex with one named group "value"


def _string_literal(line: str) -> tuple[str, str, str]:
    """(prefix, decoded body, text after the literal) of the first string literal."""
    for m in _STR_LIT_RE.finditer(line):
        prefix = m["prefix"].lower()
        if prefix not in _STR_PREFIXES:
            continue
        body = m["body"]
        if "r" not in prefix:
            body = re.sub(r"\\(.)", lambda e: _ESCAPES.get(e[1], e[1]), body)
        return prefix, body, line[m.end() :]
    raise ValueError(f"no string literal in metric line {line!r}: cannot build a pattern")


def _brace_pieces(body: str) -> list[str | None]:
    """Literal text pieces and None for each {...} placeholder ({{ / }} are literal)."""
    pieces: list[str | None] = []
    buf: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c in "{}" and body[i : i + 2] in ("{{", "}}"):
            buf.append(c)
            i += 2
            continue
        if c == "{":
            depth, j = 1, i + 1
            while j < len(body) and depth:
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            if depth:
                raise ValueError(f"unbalanced '{{' in {body!r}")
            pieces += ["".join(buf), None]
            buf = []
            i = j
            continue
        buf.append(c)
        i += 1
    pieces.append("".join(buf))
    return pieces


def _percent_pieces(body: str) -> list[str | None]:
    pieces: list[str | None] = []
    pos = 0
    for m in re.finditer(r"%%|" + _PERCENT_RE.pattern, body):
        if m.group() == "%%":
            continue
        pieces += [body[pos : m.start()].replace("%%", "%"), None]
        pos = m.end()
    pieces.append(body[pos:].replace("%%", "%"))
    return pieces


_HOLE = "\x00"


def parse_metric_line(line: str, index: int | None = None) -> MetricPattern:
    """Pattern + name for a metric line. The first string literal is the format; its
    placeholders become numbers, the `index`-th (needed when there are several) the "value"
    group. A literal without placeholders followed by `,` (print('acc:', acc)) reads the
    next whitespace-separated number. Text after the value is not matched."""
    prefix, body, after = _string_literal(line)
    if "f" in prefix or re.match(r"\s*\.format\s*\(", after):
        pieces = _brace_pieces(body)
    else:
        pieces = _percent_pieces(body)
    n_holes = pieces.count(None)
    if n_holes == 0:
        if not re.match(r"\s*,", after):
            raise ValueError(f"metric line {line!r} has no placeholder for the value")
        pieces = [body.rstrip() + " ", None, ""]
        n_holes = 1
    if index is None:
        if n_holes > 1:
            raise ValueError(
                f"metric line {line!r} has {n_holes} placeholders: give the value's index"
            )
        index = 0
    if not 0 <= index < n_holes:
        raise ValueError(f"placeholder index {index} out of range (0..{n_holes - 1})")

    # text before the value; other placeholders as _HOLE (-> \S+), whitespace runs -> \s+
    lead_parts: list[str] = []
    hole = 0
    for piece in pieces:
        if piece is None:
            if hole == index:
                break
            lead_parts.append(_HOLE)
            hole += 1
        else:
            lead_parts.append(piece)
    lead = "".join(lead_parts).lstrip()
    out: list[str] = []
    for chunk in re.split(r"(\s+)", lead):
        if chunk.isspace():
            out.append(r"\s+")
        elif chunk:
            out.append(r"\S+".join(re.escape(p) for p in chunk.split(_HOLE)))
    names = _NAME_RE.findall(lead.replace(_HOLE, " "))
    if not names:
        raise ValueError(f"metric line {line!r}: no literal name before the value to anchor on")
    anchor = r"(?<!\w)" if re.match(r"\w", lead) else ""
    return MetricPattern(name=names[-1], pattern=f"{anchor}{''.join(out)}(?P<value>{NUMBER_RE})")


def metric_pattern_from_line(line: str, index: int | None = None) -> str:
    return parse_metric_line(line, index).pattern


def extract_metric(
    log_text: str, pattern: str, which: Literal["last", "best"], direction: Direction
) -> float | None:
    """The last match, or the best finite one (max / min by direction); None if no match.
    With "best" and only non-finite matches, the last (non-finite) value is returned."""
    values = [float(m["value"]) for m in re.finditer(pattern, log_text)]
    if not values:
        return None
    if which == "last":
        return values[-1]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return values[-1]
    return max(finite) if direction == "max" else min(finite)


# --- conversion ----------------------------------------------------------------------------


def metric_kind(name: str) -> MetricKind:
    n = name.lower()
    if "err" in n:
        return "error"
    if re.search(r"loss|xent|nll", n):
        return "loss"
    if re.search(r"acc|top-?[15]", n):
        return "accuracy"
    return "other"


def metric_direction(name: str) -> Direction:
    kind = metric_kind(name)
    if kind == "accuracy":
        return "max"
    if kind in ("error", "loss"):
        return "min"
    raise ValueError(f"no known direction for metric {name!r}")


def allowed_conversions(
    source_name: str, target_metric: str, target_unit: str | None
) -> frozenset[Conversion]:
    """Conversions from the repo's printed metric to the claim's metric and unit:
    accuracy -> error in %: accuracy_fraction_to_error_percent / accuracy_percent_to_error_percent;
    same kind in %: identity / fraction_to_percent (+ error_fraction_to_percent for errors);
    same kind otherwise: identity; anything else: none."""
    s, t = metric_kind(source_name), metric_kind(target_metric)
    pct = target_unit == "%"
    if s == t:
        if s == "other" and source_name.lower() != target_metric.lower():
            return frozenset()
        if pct and s == "accuracy":
            return frozenset({"identity", "fraction_to_percent"})
        if pct and s == "error":
            return frozenset({"identity", "fraction_to_percent", "error_fraction_to_percent"})
        return frozenset({"identity"})
    if s == "accuracy" and t == "error" and pct:
        return frozenset(
            {"accuracy_fraction_to_error_percent", "accuracy_percent_to_error_percent"}
        )
    return frozenset()


_RANGES: dict[str, tuple[float, float]] = {
    "fraction_to_percent": (0.0, 1.0),
    "accuracy_fraction_to_error_percent": (0.0, 1.0),
    "error_fraction_to_percent": (0.0, 1.0),
    "accuracy_percent_to_error_percent": (0.0, 100.0),
}


def convert(value: float, conversion: Conversion) -> float:
    """Repo value -> claim value. A finite value outside the input range of the conversion
    (e.g. 40.5 for a fraction) raises ValueError; non-finite values pass through."""
    if math.isfinite(value) and conversion in _RANGES:
        lo, hi = _RANGES[conversion]
        if not lo <= value <= hi:
            raise ValueError(f"{value} is outside [{lo}, {hi}] expected by {conversion}")
    if conversion == "identity":
        return value
    if conversion in ("fraction_to_percent", "error_fraction_to_percent"):
        return value * 100.0
    if conversion == "accuracy_fraction_to_error_percent":
        return (1.0 - value) * 100.0
    if conversion == "accuracy_percent_to_error_percent":
        return 100.0 - value
    raise ValueError(f"unknown conversion {conversion!r}")


def metric_conversions(
    metric_evidence: Iterable[Evidence], claims: Iterable[ResultClaim]
) -> dict[str, dict[str, dict[str, frozenset[Conversion]]]]:
    """'#metric_lines' evidence_id -> {line: {claim_id: allowed conversions}} for every line
    code can build a pattern from (lines with several placeholders are left out). The
    validation context of the orchestrator's ClaimMapping."""
    claims = tuple(claims)
    out: dict[str, dict[str, dict[str, frozenset[Conversion]]]] = {}
    for ev in metric_evidence:
        if not ev.locator.endswith("#metric_lines"):
            continue
        per_line: dict[str, dict[str, frozenset[Conversion]]] = {}
        for r in _result_lines(ev):
            try:
                name = parse_metric_line(r["line"]).name
            except ValueError:
                continue
            per_line[r["line"]] = {
                c.claim_id: allowed_conversions(name, c.metric, c.unit) for c in claims
            }
        out[ev.evidence_id] = per_line
    return out


# --- claim mapping (judgment; provided by the caller) ---------------------------------------


class ClaimArg(_Frozen):
    """A flag value the claim's setting states (e.g. CIFAR-100 -> --dataset cifar100)."""

    flag: str
    value: str


class SetupArg(_Frozen):
    """A paper setup field whose value sets a flag; the value is copied by code."""

    setup: str
    flag: str


class ClaimMapping(_Frozen):
    claim_id: str
    entrypoint: str
    base_command_evidence_id: str  # '<source>#commands' evidence
    base_command: str  # one command line of that evidence
    claim_args: tuple[ClaimArg, ...] = ()
    setup_args: tuple[SetupArg, ...] = ()
    metric_line_evidence_id: str  # '<entrypoint>#metric_lines' evidence
    metric_line: str  # one line of that evidence
    metric_index: int | None = None  # placeholder of the value, if the line has several
    conversion: Conversion


# deviation: the repo differs from a stated paper setup value that no flag can change
# (orchestrator setup check); the plan then has fidelity "inferred".
NoteStatus = Literal["consistent", "override", "assumption", "open", "deviation"]


class PlanNote(_Frozen):
    field: str
    status: NoteStatus
    detail: str


class LessonOverride(_Frozen):
    """An applied Reflexion lesson change, folded into the plan by code (orchestrator reads
    them from lesson_applied evidence). arg: target = flag, new_value (None for a flag without
    value), evidence_id = the paper support. eval_protocol: target = metric_source.which |
    metric_source.conversion, evidence_id = the lesson."""

    lesson_id: str
    kind: Literal["arg", "eval_protocol"]
    target: str
    new_value: str | None
    evidence_id: str


# --- build ---------------------------------------------------------------------------------

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_PYTHONS = frozenset({"python", "python3"})


def _result_lines(ev: Evidence, key: str = "line") -> list[Mapping[str, Any]]:
    result = ev.content.get("result")
    if not isinstance(result, list):
        return []
    return [r for r in result if isinstance(r, Mapping) and isinstance(r.get(key), str)]


def split_command(
    command: str, entrypoint: str, spec: Sequence[ArgOption]
) -> tuple[list[str], list[tuple[str, str | None]]]:
    """(prefix tokens up to and including the script: env assignments, interpreter and its
    options, path; (flag, value) pairs in order, value None for flags that take none).
    Raises ValueError for another script, python -m, positionals or unknown flags."""
    tokens = shlex.split(command, comments=True)
    i = 0
    while i < len(tokens) and _ENV_ASSIGN_RE.match(tokens[i]):
        i += 1
    if i >= len(tokens) or tokens[i] not in _PYTHONS:
        raise ValueError(f"not a python command: {command!r}")
    i += 1
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-m":
            raise ValueError(f"python -m commands are not supported: {command!r}")
        i += 1
    if i >= len(tokens) or normalize_repo_path(tokens[i]) != entrypoint:
        raise ValueError(f"command does not run {entrypoint}: {command!r}")
    rest = tokens[i + 1 :]
    out: list[tuple[str, str | None]] = []
    j = 0
    while j < len(rest):
        tok = rest[j]
        if not tok.startswith("-"):
            raise ValueError(f"positional argument {tok!r} in {command!r} is not supported")
        flag, eq, inline = tok.partition("=")
        opt = find_option(spec, flag)
        if opt is None:
            raise ValueError(f"flag {flag!r} of the command is not an argparse option")
        if opt.takes_value:
            if eq:
                value: str | None = inline
            elif j + 1 < len(rest):
                value = rest[j + 1]
                j += 1
            else:
                raise ValueError(f"flag {flag} has no value in {command!r}")
        elif eq:
            raise ValueError(f"flag {flag} takes no value in {command!r}")
        else:
            value = None
        out.append((flag, value))
        j += 1
    return tokens[: i + 1], out


def command_flags(
    command: str, entrypoint: str, spec: Sequence[ArgOption]
) -> list[tuple[str, str | None]]:
    """(flag, value) pairs of the command (see split_command)."""
    return split_command(command, entrypoint, spec)[1]


def _is_human(evidence_id: str, evidence: Mapping[str, Evidence]) -> bool:
    ev = evidence.get(evidence_id)
    return ev is not None and ev.kind == "human_input"


def _answer_label(answer: tuple[str, str], evidence: Mapping[str, Evidence]) -> str:
    """How a resolved unknown's value is described in plan notes: its source and evidence,
    e.g. "human: human_input 1a2b3c4d5e6f"."""
    ev = evidence.get(answer[1])
    kind = ev.kind if ev is not None else "evidence"
    who = "human" if _is_human(answer[1], evidence) else "paper"
    return f"{who}: {kind} {answer[1][:12]}"


class _Slot(_Frozen):
    flag: str
    value: str | None
    source: ArgSource
    evidence_id: str


def build_run_plan(
    claim: ResultClaim,
    mapping: ClaimMapping,
    evidence: Mapping[str, Evidence],
    argparse_spec: Sequence[ArgOption],
    paper_setup: Iterable[Evidence],
    *,
    repo_files: Set[str],
    config_keys: Set[str] = frozenset(),
    answers: Mapping[str, tuple[str, str]] | None = None,
    lessons: Sequence[LessonOverride] = (),
) -> tuple[RunPlan, tuple[Assumption, ...], tuple[PlanNote, ...]]:
    """RunPlan for one claim (see the module docstring). answers: setup name -> (value,
    evidence id) of RESOLVED unknowns; used instead of an assumption. lessons: applied lesson
    changes, in order (a paper value they conflict with raises). Raises ValueError when
    the mapping does not fit the evidence or the argparse spec (and ValidationError from the
    contract)."""
    answers = dict(answers or {})
    spec = tuple(argparse_spec)
    ep = normalize_repo_path(mapping.entrypoint)
    if mapping.claim_id != claim.claim_id:
        raise ValueError(f"mapping is for {mapping.claim_id!r}, not {claim.claim_id!r}")

    # verified commands of the entrypoint; the base command must be one of them
    command_lines = {
        ev.evidence_id: frozenset(r["line"] for r in _result_lines(ev) if r.get("path") == ep)
        for ev in evidence.values()
        if ev.locator.endswith("#commands")
    }
    command_lines = {i: lines for i, lines in command_lines.items() if lines}
    base_ev = evidence.get(mapping.base_command_evidence_id)
    if base_ev is None or base_ev.evidence_id not in command_lines:
        raise ValueError(f"base command evidence is not a '#commands' evidence of {ep}")
    if mapping.base_command not in command_lines[base_ev.evidence_id]:
        raise ValueError(f"base command is not in the cited evidence: {mapping.base_command!r}")

    claim_ev = next(
        (
            ev
            for ev in evidence.values()
            if ev.kind == "paper_claim" and ev.content.get("claim_id") == claim.claim_id
        ),
        None,
    )
    if claim_ev is None:
        raise ValueError(f"no paper_claim evidence for {claim.claim_id!r}")

    setup_ev = {str(ev.content.get("name")): ev for ev in paper_setup if ev.kind == "paper_setup"}
    notes: list[PlanNote] = []
    assumptions: list[Assumption] = []

    slots = [
        _Slot(flag=f, value=v, source="repo", evidence_id=base_ev.evidence_id)
        for f, v in command_flags(mapping.base_command, ep, spec)
    ]
    set_by: dict[str, tuple[str, Any]] = {}  # dest -> (field, typed value) set by claim/paper

    def slot_index(opt: ArgOption) -> int | None:
        for i, s in enumerate(slots):
            o = find_option(spec, s.flag)
            if o is not None and o.dest == opt.dest:
                return i
        return None

    def apply(
        field: str,
        flag: str,
        value: str,
        source: ArgSource,
        evidence_id: str,
        label: str | None = None,
    ) -> None:
        opt = find_option(spec, flag)
        if opt is None:
            raise ValueError(f"{field}: flag {flag!r} is not an argparse option")
        if not opt.takes_value:
            raise ValueError(f"{field}: {flag} takes no value, cannot set it to {value!r}")
        new = parse_arg_value(opt, value)
        dest = str(opt.dest)
        if dest in set_by and set_by[dest][1] != new:
            raise ValueError(f"{field} and {set_by[dest][0]} set {flag} to different values")
        set_by[dest] = (field, new)
        idx = slot_index(opt)
        if idx is not None:
            cur_raw = slots[idx].value
            current = parse_arg_value(opt, cur_raw) if cur_raw is not None else None
            where = f"verified command {slots[idx].flag} {cur_raw}"
        else:
            current = opt.default
            where = f"repo default {opt.default!r}"
        if current is not None and current == new:
            tag = f" ({label})" if label else ""
            notes.append(
                PlanNote(field=field, status="consistent", detail=f"{value} = {where}{tag}")
            )
            return
        slot = _Slot(flag=flag, value=value, source=source, evidence_id=evidence_id)
        if idx is not None:
            slots[idx] = slot.model_copy(update={"flag": slots[idx].flag})
        else:
            slots.append(slot)
        notes.append(
            PlanNote(
                field=field,
                status="override",
                detail=f"{flag} {value} ({label or source}; {where})",
            )
        )

    for ca in mapping.claim_args:
        apply(f"claim:{ca.flag}", ca.flag, ca.value, "claim", claim_ev.evidence_id)

    mapped = {sa.setup: sa.flag for sa in mapping.setup_args}
    for sa in mapping.setup_args:
        ev = setup_ev.get(sa.setup)
        if ev is None:
            raise ValueError(f"setup field {sa.setup!r} is not in the paper setup")
        value = ev.content.get("value")
        if value is not None:
            apply(sa.setup, sa.flag, str(value), "paper", ev.evidence_id)

    parsed_ev = next(
        (ev for ev in evidence.values() if ev.locator == f"{ep}#parsed"),
        None,
    )
    for name, ev in setup_ev.items():
        if ev.content.get("value") is not None or name == "eval_checkpoint":
            continue
        flag = mapped.get(name)
        if flag:
            opt = find_option(spec, flag)
        else:  # unmapped: the optional argument whose dest is the setup name
            opt = next((a for a in spec if a.dest == name and a.option_strings[0][:1] == "-"), None)
        answer = answers.get(name)
        if opt is None or not opt.takes_value:
            detail = "no argparse flag; stays open"
            if answer is not None:
                label = _answer_label(answer, evidence)
                detail = f"resolved to {answer[0]} ({label}); no argparse flag to set it"
            notes.append(PlanNote(field=name, status="open", detail=detail))
            continue
        if answer is not None:
            src: ArgSource = "human" if _is_human(answer[1], evidence) else "paper"
            apply(name, flag or opt.option_strings[0], answer[0], src, answer[1],
                  label=_answer_label(answer, evidence))  # fmt: skip
            continue
        idx = slot_index(opt)
        if idx is not None and slots[idx].value is not None:
            value, evid = slots[idx].value, slots[idx].evidence_id
            detail = f"{slots[idx].flag} {value} from the verified command"
        elif opt.default is not None and parsed_ev is not None:
            value, evid = str(opt.default), parsed_ev.evidence_id
            detail = f"repo default {opt.option_strings[0]} {value}"
        else:
            notes.append(PlanNote(field=name, status="open", detail="no repo value; stays open"))
            continue
        assumptions.append(
            Assumption(unknown_id=unknown_id_for(name), value=str(value), evidence_id=evid)
        )
        notes.append(PlanNote(field=name, status="assumption", detail=detail))

    # applied Reflexion lessons: arg changes (a later lesson replaces an earlier one)
    conversion: Conversion = mapping.conversion
    which_override: str | None = None
    for lo in lessons:
        label = f"lesson {lo.lesson_id[:12]}"
        if lo.kind == "eval_protocol":
            if lo.target == "metric_source.which":
                which_override = lo.new_value
            elif lo.target == "metric_source.conversion":
                conversion = lo.new_value  # type: ignore[assignment]  # MetricSource checks it
            else:
                raise ValueError(f"{label}: unknown eval_protocol target {lo.target!r}")
            notes.append(PlanNote(field=f"lesson:{lo.target}", status="override",
                                  detail=f"{lo.target} = {lo.new_value} ({label})"))  # fmt: skip
            continue
        opt = find_option(spec, lo.target)
        if opt is None:
            raise ValueError(f"{label}: flag {lo.target!r} is not an argparse option")
        prev = set_by.get(str(opt.dest))
        if prev is not None and prev[0].startswith("lesson:"):
            del set_by[str(opt.dest)]
        if opt.takes_value:
            if lo.new_value is None:
                raise ValueError(f"{label}: {lo.target} takes a value")
            apply(f"lesson:{lo.target}", lo.target, lo.new_value, "paper", lo.evidence_id,
                  label=label)  # fmt: skip
        elif slot_index(opt) is None:
            slots.append(_Slot(flag=lo.target, value=None, source="paper",
                               evidence_id=lo.evidence_id))  # fmt: skip
            notes.append(PlanNote(field=f"lesson:{lo.target}", status="override",
                                  detail=f"{lo.target} ({label})"))  # fmt: skip

    # metric source
    metric_ev = evidence.get(mapping.metric_line_evidence_id)
    if metric_ev is None or metric_ev.locator != f"{ep}#metric_lines":
        raise ValueError(f"metric line evidence is not the '#metric_lines' evidence of {ep}")
    if not any(r["line"] == mapping.metric_line for r in _result_lines(metric_ev)):
        raise ValueError(f"metric line is not in the cited evidence: {mapping.metric_line!r}")
    mp = parse_metric_line(mapping.metric_line, mapping.metric_index)

    which: Literal["last", "best"] = "last"
    ckpt = setup_ev.get("eval_checkpoint")
    ckpt_value = ckpt.content.get("value") if ckpt is not None else None
    ckpt_label = ""
    if ckpt_value is None and ckpt is not None and "eval_checkpoint" in answers:
        ckpt_value = answers["eval_checkpoint"][0]
        ckpt_label = f" ({_answer_label(answers['eval_checkpoint'], evidence)})"
    if ckpt_value is None:
        detail = "which=last: the final evaluation printed"
        if ckpt is not None:
            assumptions.append(
                Assumption(
                    unknown_id=unknown_id_for("eval_checkpoint"),
                    value="last",
                    evidence_id=metric_ev.evidence_id,
                )
            )
        else:
            detail += " (no eval_checkpoint field in the spec)"
        notes.append(PlanNote(field="eval_checkpoint", status="assumption", detail=detail))
    else:
        v = str(ckpt_value).strip().lower()
        if v in ("last", "final"):
            which = "last"
        elif v == "best":
            which = "best"
        else:
            raise ValueError(f"eval_checkpoint {ckpt_value!r} is neither last/final nor best")
        notes.append(
            PlanNote(
                field="eval_checkpoint", status="consistent", detail=f"which={which}{ckpt_label}"
            )
        )
    if which_override is not None:
        if which_override not in ("last", "best"):
            raise ValueError(f"metric_source.which {which_override!r} is neither last nor best")
        which = which_override  # type: ignore[assignment]

    data = {
        "claim_id": claim.claim_id,
        "entrypoint": ep,
        "base_command_evidence_id": base_ev.evidence_id,
        "base_command": mapping.base_command,
        "depends_on_unknowns": sorted({a.unknown_id for a in assumptions}),
        "args": [s.model_dump() for s in slots],
        "config": None,
        "config_overrides": [],
        "dataset": claim.dataset,
        "metric": claim.metric,
        "metric_unit": claim.unit,
        "metric_source": {
            "metric_line_evidence_id": metric_ev.evidence_id,
            "pattern": mp.pattern,
            "which": which,
            "conversion": conversion,
        },
    }
    plan = make_run_plan(
        data,
        repo_files=repo_files,
        config_keys=config_keys,
        claims={claim.claim_id: claim},
        evidence_ids=set(evidence),
        argparse_spec=spec,
        command_evidence={ep: command_lines},
        metric_sources={
            metric_ev.evidence_id: {
                mp.pattern: allowed_conversions(mp.name, claim.metric, claim.unit)
            }
        },
        human_input_ids={i for i, ev in evidence.items() if ev.kind == "human_input"},
    )
    return plan, tuple(assumptions), tuple(notes)


def render_command(plan: RunPlan, argparse_spec: Sequence[ArgOption]) -> list[str]:
    """argv from plan.base_command: its prefix (env assignments, interpreter, script) kept
    verbatim, then the plan's flags -- the base command's flags at their positions with the
    plan's values (store_true flags bare), new flags appended. Raises ValueError if the plan's
    args do not start with the base command's flags in order."""
    spec = tuple(argparse_spec)
    prefix, base = split_command(plan.base_command, plan.entrypoint, spec)

    def dest(flag: str) -> str | None:
        opt = find_option(spec, flag)
        if opt is None:
            raise ValueError(f"flag {flag!r} is not an argparse option")
        return opt.dest

    base_dests = [dest(f) for f, _ in base]
    plan_dests = [dest(a.flag) for a in plan.args]
    if plan_dests[: len(base_dests)] != base_dests:
        raise ValueError("plan args do not start with the base command's flags in order")
    argv = list(prefix)
    for a in plan.args:
        opt = find_option(spec, a.flag)
        assert opt is not None
        if opt.takes_value:
            if a.value is None:
                raise ValueError(f"{a.flag} needs a value")
            argv += [a.flag, a.value]
        else:
            if a.value is not None:
                raise ValueError(f"{a.flag} takes no value")
            argv.append(a.flag)
    return argv
