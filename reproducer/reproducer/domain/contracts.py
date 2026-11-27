"""Contracts: ReproSpec, RunPlan, ExperimentResult, Lesson.

Pure Pydantic v2 models. No file, network or subprocess IO.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Mapping, Set
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from reproducer.domain.grounding import Citation, EvidenceRef, RepoPath
from reproducer.domain.patching import (
    apply_line_edits,
    diff_changed_lines,
    range_text,
    split_lines,
    unified_diff,
    value_literals,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- ReproSpec -----------------------------------------------------------------------------


class ResultClaim(_Frozen):
    claim_id: str
    metric: str
    value: float
    # TODO: `unit` is a proposed addition pending confirmation from ingestion
    # (not yet in the CLAUDE.md ReproSpec contract).
    unit: str | None = None
    dataset: str
    split: str | None
    setting: str | None
    location: str
    quote: str


class SetupField(_Frozen):
    name: str
    value: str | None  # None -> Unknown (never guessed)
    location: str
    quote: str

    @property
    def is_unknown(self) -> bool:
        return self.value is None


class ReproSpec(_Frozen):
    code_links: tuple[str, ...] = Field(min_length=1)
    datasets: tuple[str, ...]
    claims: tuple[ResultClaim, ...] = Field(min_length=1)
    setup: tuple[SetupField, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        claim_counts = Counter(c.claim_id for c in self.claims)
        dupes = sorted(k for k, n in claim_counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate claim_id(s): {dupes}")
        name_counts = Counter(s.name for s in self.setup)
        dupe_names = sorted(k for k, n in name_counts.items() if n > 1)
        if dupe_names:
            raise ValueError(f"duplicate setup name(s): {dupe_names}")
        return self


# --- RunPlan -------------------------------------------------------------------------------
# A claim's full run: the verified example command with argparse flags set (ArgSetting) and,
# for config-driven repos, config overrides. Every value is checked in code against the
# entrypoint's argparse spec, the repo files, the config keys and the evidence (context).

OverrideSource = Literal["paper", "repo", "unknown"]
# human: a value a human answered for an unknown setup field (evidence_id -> human_input)
ArgSource = Literal["paper", "repo", "claim", "assumption", "human"]
Conversion = Literal[
    "identity",
    "fraction_to_percent",
    "accuracy_fraction_to_error_percent",
    "accuracy_percent_to_error_percent",
    "error_fraction_to_percent",
]

_RUNPLAN_CONTEXT_KEYS = (
    "repo_files",
    "config_keys",
    "claims",
    "evidence_ids",
    "argparse_spec",
    "command_evidence",
    "metric_sources",
)
# argparse actions that take no value on the command line
_FLAG_ACTIONS = frozenset({"store_true", "store_false", "store_const", "count"})
_VALUE_ACTIONS = frozenset({None, "store"})


class Override(_Frozen):
    key: str
    value: bool | int | float | str
    source: OverrideSource


class ArgOption(_Frozen):
    """argparse facts of one add_argument call (same fields as the scanner's ArgSpec, so
    ArgOption.model_validate(spec.model_dump()) converts one)."""

    option_strings: tuple[str, ...]
    dest: str | None
    type: str | None = None  # source name of type=, e.g. "int"
    default: Any = None
    action: str | None = None
    choices: tuple[Any, ...] | None = None

    @property
    def takes_value(self) -> bool:
        return self.action in _VALUE_ACTIONS


def _argparse_spec(info: ValidationInfo, type_name: str) -> tuple[ArgOption, ...]:
    ctx: Any = info.context
    spec = ctx.get("argparse_spec") if isinstance(ctx, Mapping) else None
    if not isinstance(spec, (tuple, list)) or not all(isinstance(a, ArgOption) for a in spec):
        raise ValueError(
            f"{type_name} requires validation context key 'argparse_spec' (ArgOption sequence)"
        )
    return tuple(spec)


def find_option(spec: tuple[ArgOption, ...] | list[ArgOption], flag: str) -> ArgOption | None:
    """The optional argument that `flag` names (positionals never match)."""
    if not flag.startswith("-"):
        return None
    return next((a for a in spec if flag in a.option_strings), None)


def parse_arg_value(opt: ArgOption, value: str) -> Any:
    """`value` parsed like argparse would (type=int/float/str or none), then checked against
    choices. Raises ValueError for a bad value or a type= that code cannot check."""
    t = opt.type
    parsers: dict[str | None, Any] = {None: str, "str": str, "int": int, "float": float}
    if t not in parsers:
        raise ValueError(f"argparse type {t!r} of {opt.option_strings[0]} cannot be checked")
    try:
        parsed = parsers[t](value)
    except ValueError:
        raise ValueError(f"{value!r} is not a valid {t} for {opt.option_strings[0]}") from None
    if opt.choices is not None and parsed not in opt.choices:
        raise ValueError(
            f"{value!r} is not a choice of {opt.option_strings[0]}: {list(opt.choices)}"
        )
    return parsed


class ArgSetting(_Frozen):
    """One command-line flag of the run. value None = a flag without value (store_true...).
    Checks (context argparse_spec + evidence_ids): the flag is an option string of the
    entrypoint; flags whose action takes no value have value None; other values parse with
    the argparse type (int/float/str) and are in choices. Missing context fails closed."""

    flag: str
    value: str | None
    source: ArgSource
    evidence_id: EvidenceRef

    @model_validator(mode="after")
    def _check_against_argparse(self, info: ValidationInfo) -> Self:
        spec = _argparse_spec(info, "ArgSetting")
        opt = find_option(spec, self.flag)
        if opt is None:
            known = sorted(o for a in spec for o in a.option_strings if o.startswith("-"))
            raise ValueError(f"flag {self.flag!r} is not an argparse option; options: {known}")
        if opt.action in _FLAG_ACTIONS:
            if self.value is not None:
                raise ValueError(f"{self.flag} is {opt.action}: value must be None")
        elif opt.action in _VALUE_ACTIONS:
            if self.value is None:
                raise ValueError(f"{self.flag} takes a value")
            parse_arg_value(opt, self.value)
        else:
            raise ValueError(f"argparse action {opt.action!r} of {self.flag} is not supported")
        return self


class MetricSource(_Frozen):
    """Where the run's metric is read: a code-found metric line, the regex code built from it,
    which match counts, and the conversion to the claim's metric/unit. Context
    "metric_sources": evidence_id -> {pattern: allowed conversions}, built by code
    (run_plan.build_run_plan); the pattern and conversion must be in it."""

    metric_line_evidence_id: EvidenceRef
    pattern: str
    which: Literal["last", "best"]
    conversion: Conversion

    @model_validator(mode="after")
    def _check_code_built(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        sources = ctx.get("metric_sources") if isinstance(ctx, Mapping) else None
        if not isinstance(sources, Mapping):
            raise ValueError("MetricSource requires validation context key 'metric_sources'")
        patterns = sources.get(self.metric_line_evidence_id)
        if not isinstance(patterns, Mapping) or self.pattern not in patterns:
            raise ValueError("pattern must be the one code built from the cited metric line")
        allowed = patterns[self.pattern]
        if self.conversion not in allowed:
            raise ValueError(
                f"conversion {self.conversion!r} not allowed here; allowed: {sorted(allowed)}"
            )
        return self


class RunPlan(_Frozen):
    """Validated against the repo and spec via validation context.

    Build with make_run_plan(...) (run_plan.build_run_plan does). Validation without the
    context fails (no silent pass), and model_copy is disabled so a validated plan cannot be
    altered unchecked.
    """

    claim_id: str
    entrypoint: RepoPath
    base_command_evidence_id: EvidenceRef  # the '<source>#commands' evidence of the command
    base_command: str  # the exact verified command line (one line of that evidence)
    depends_on_unknowns: tuple[str, ...] = ()  # unknown ids the plan assumes ("unk:seed")
    args: tuple[ArgSetting, ...]
    config: RepoPath | None = None
    config_overrides: tuple[Override, ...] = ()
    dataset: str
    metric: str
    metric_unit: str | None
    metric_source: MetricSource

    @model_validator(mode="after")
    def _check_against_repo(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        if not isinstance(ctx, dict) or any(k not in ctx for k in _RUNPLAN_CONTEXT_KEYS):
            raise ValueError(
                f"RunPlan requires validation context with keys {list(_RUNPLAN_CONTEXT_KEYS)}"
            )
        config_keys = ctx["config_keys"]
        claims = ctx["claims"]
        commands = ctx["command_evidence"]
        spec = _argparse_spec(info, "RunPlan")

        errors: list[str] = []
        verified = commands.get(self.entrypoint)
        lines = (
            verified.get(self.base_command_evidence_id) if isinstance(verified, Mapping) else None
        )
        if lines is None:
            errors.append(f"base command evidence is not a verified command of {self.entrypoint}")
        elif self.base_command not in lines:
            errors.append(
                f"base_command is not a command line of the cited evidence: {self.base_command!r}"
            )
        bad_unk = [u for u in self.depends_on_unknowns if not u.startswith("unk:")]
        if bad_unk or len(set(self.depends_on_unknowns)) != len(self.depends_on_unknowns):
            errors.append(f"depends_on_unknowns must be unique 'unk:<name>' ids, got {bad_unk}")
        dests = Counter(
            opt.dest for a in self.args if (opt := find_option(spec, a.flag)) is not None
        )
        dupes = sorted(str(d) for d, n in dests.items() if n > 1)
        if dupes:
            errors.append(f"argument(s) set more than once: {dupes}")
        # source "human" <-> evidence_id is a human_input evidence (context human_input_ids;
        # missing = none, so a human-sourced arg fails closed)
        human_ids = ctx.get("human_input_ids") or frozenset()
        for a in self.args:
            if a.source == "human" and a.evidence_id not in human_ids:
                errors.append(f"{a.flag}: source human needs a human_input evidence_id")
            elif a.source != "human" and a.evidence_id in human_ids:
                errors.append(f"{a.flag}: a human_input value must have source human")
        if self.config is None and self.config_overrides:
            errors.append("config_overrides need a config")
        bad_keys = [o.key for o in self.config_overrides if o.key not in config_keys]
        if bad_keys:
            errors.append(f"override key(s) not in config: {bad_keys}")
        claim = claims.get(self.claim_id)
        if claim is None:
            errors.append(f"unknown claim_id: {self.claim_id!r}")
        else:
            if self.metric != claim.metric:
                errors.append(f"metric {self.metric!r} != claim metric {claim.metric!r}")
            if self.metric_unit != claim.unit:
                errors.append(f"metric_unit {self.metric_unit!r} != claim unit {claim.unit!r}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def is_spec_faithful(self, unknown_states: Mapping[str, Any]) -> bool:
        """No assumed flag or unknown-sourced override, and every unknown the plan depends on
        is resolved in `unknown_states` (unknown id -> UnknownState; missing = open)."""
        if any(a.source == "assumption" for a in self.args):
            return False
        if any(o.source == "unknown" for o in self.config_overrides):
            return False
        return all(
            getattr(unknown_states.get(u), "status", None) == "resolved"
            for u in self.depends_on_unknowns
        )

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        raise TypeError("RunPlan must be built via make_run_plan")


def make_run_plan(
    data: dict[str, Any],
    *,
    repo_files: Set[str],
    config_keys: Set[str],
    claims: Mapping[str, ResultClaim],
    evidence_ids: Set[str],
    argparse_spec: tuple[ArgOption, ...],
    command_evidence: Mapping[str, Mapping[str, Set[str]]],
    metric_sources: Mapping[str, Mapping[str, Set[str]]],
    human_input_ids: Set[str] = frozenset(),
) -> RunPlan:
    """Build a RunPlan validated against repo files, config keys, spec claims, evidence ids,
    the entrypoint's argparse spec, verified commands (entrypoint -> {'#commands' evidence id:
    command lines})
    and code-built metric patterns (evidence_id -> {pattern: allowed conversions}).
    human_input_ids: ids of human_input evidence; an arg has source "human" iff its
    evidence_id is one of them."""
    return RunPlan.model_validate(
        data,
        context={
            "repo_files": repo_files,
            "config_keys": config_keys,
            "claims": claims,
            "evidence_ids": evidence_ids,
            "argparse_spec": tuple(argparse_spec),
            "command_evidence": command_evidence,
            "metric_sources": metric_sources,
            "human_input_ids": human_input_ids,
        },
    )


# --- ExperimentResult ----------------------------------------------------------------------


class ExperimentResult(_Frozen):
    claim_id: str
    run_id: str
    exit_code: int
    metrics: dict[str, float]
    log_ref: str
    config_digest: str
    duration: float = Field(ge=0)  # seconds
    gpu_hours: float = Field(ge=0)


# --- Lesson --------------------------------------------------------------------------------

LessonStep = Literal[
    "eval_protocol", "data_preprocessing", "config_vs_paper", "env_versions", "training"
]
# The fixed Reflexion localization order (CLAUDE.md error rulebook); lessons of a claim never
# go back in it.
LESSON_STEPS: tuple[str, ...] = (
    "eval_protocol", "data_preprocessing", "config_vs_paper", "env_versions", "training"
)  # fmt: skip

# Contract says text <= 100 tokens. To keep the domain pure (no tokenizer dependency) we
# approximate: English BPE averages ~0.75 words per token, so 100 tokens ~= 75 words.
LESSON_MAX_WORDS = 75

# A Lesson proposes at most one Change; nothing is applied here (M4-S3 applies it). Validation
# context (built by code, graph/nodes/reflect.py); missing context fails closed:
#   claim_id, trial        the claim and trial (1-based run index) the lesson is about
#   previous_step          step of this claim's previous lesson (None: first lesson)
#   subject_ids            ids of the run_result / run_curve the lesson reasons about
#   evidence_ids, texts, aliases   for evidence_ids and the support Citation
#   evidence_authority     evidence_id -> authority (support: paper | repo | human only)
#   argparse_spec          ArgOption sequence of the plan's entrypoint
#   claim_flags            flags the claim itself selects (dataset, model): never changed
#   open_plan_unknowns     'unk:<name>' ids the plan depends on that are still open
#   eval_protocol_table    eval_protocol target -> allowed new values (current value excluded)
#   claim_unit             the claim's unit (a conversion change must pass check_scale)
#   open_unknown_names     names of the OPEN unknowns (text grounding, see lesson_text_issue)
ChangeKind = Literal["eval_protocol", "arg", "assumption", "none"]
CHANGE_RATIONALE_MAX_CHARS = 200
EVAL_PROTOCOL_TARGETS = ("metric_source.which", "metric_source.conversion")
SUPPORT_AUTHORITIES = frozenset({"paper", "repo", "human"})
_LESSON_CONTEXT_KEYS = (
    "claim_id",
    "trial",
    "previous_step",
    "subject_ids",
    "evidence_ids",
    "evidence_authority",
    "argparse_spec",
    "claim_flags",
    "open_plan_unknowns",
    "eval_protocol_table",
    "claim_unit",
    "open_unknown_names",
)
# Lesson text grounding: the text may not attribute an open unknown to the paper. Topic
# keywords per unknown name (the name itself, with "_" as space, always counts too).
UNKNOWN_TOPICS: dict[str, tuple[str, ...]] = {
    "eval_checkpoint": ("checkpoint", "final", "best", "last epoch"),
    "seed": ("seed",),
    "num_runs": ("runs", "mean"),
}
HEDGES = ("assumed", "not stated", "unknown")
# targets touching the metric line / claim value / tolerance (eval_protocol targets are only
# those of the table and are checked before this)
_FORBIDDEN_TARGET_RE = re.compile(r"(?i)toleran|\btol\b|claim|paper|metric|pattern")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _lesson_context(info: ValidationInfo, type_name: str) -> Mapping[str, Any]:
    ctx: Any = info.context
    if not isinstance(ctx, Mapping) or any(k not in ctx for k in _LESSON_CONTEXT_KEYS):
        raise ValueError(
            f"{type_name} requires validation context with keys {list(_LESSON_CONTEXT_KEYS)}"
        )
    return ctx


def value_stated(value: str, quote: str) -> bool:
    """`value` is stated in `quote`: a number equal to one of the quote's numbers ("5e-4" in
    "weight decay of 5e-4", "0.1" in "set to 0.1"), else a token match (appears_in)."""
    v = value.strip()
    if not v:
        return False
    try:
        x = float(v)
    except ValueError:
        return appears_in(v, quote)
    for tok in _NUMBER_RE.findall(quote):
        try:
            if float(tok) == x:
                return True
        except ValueError:
            continue
    return False


class Change(_Frozen):
    """One proposed change (checks with the Lesson validation context, see above):
    eval_protocol  target + new_value from context eval_protocol_table; a conversion change
                   also cites repo code showing the printed value's scale (check_scale)
    arg            target an argparse flag of the entrypoint (not a claim flag); new_value
                   parses (None for flags without value); support cites PAPER evidence stating
                   new_value
    assumption     target an open unknown the plan depends on; new_value set and stated in
                   the support quote (value_stated); support cites paper or repo evidence
    none           new_value None; no support needed
    Always: support (if any) cites paper / repo / human evidence (never llm / rule); a target
    touching the metric, claim or tolerance is rejected; a seed target needs a paper quote
    containing the seed value (no seed fishing)."""

    kind: ChangeKind
    target: str = Field(min_length=1, max_length=100)
    new_value: str | None = None
    support: Citation | None = None
    rationale: str = Field(min_length=1, max_length=CHANGE_RATIONALE_MAX_CHARS)

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        ctx = _lesson_context(info, "Change")
        authority = ctx["evidence_authority"]
        if not isinstance(authority, Mapping):
            raise ValueError("Change context 'evidence_authority' must be a mapping")
        target = self.target.strip()
        sup = self.support
        sup_auth = authority.get(sup.evidence_id) if sup is not None else None
        errors: list[str] = []
        if sup is not None and sup_auth not in SUPPORT_AUTHORITIES:
            errors.append(
                f"support must cite paper, repo or human evidence, not {sup_auth!r} evidence"
            )
        if self.kind == "eval_protocol":
            table = ctx["eval_protocol_table"]
            if target not in table:
                errors.append(f"eval_protocol target must be one of {sorted(table)}")
            elif self.new_value not in table[target]:
                errors.append(
                    f"{target}: new_value must be one of {sorted(table[target])} "
                    f"(the current value is excluded), got {self.new_value!r}"
                )
            elif target == EVAL_PROTOCOL_TARGETS[1]:
                if sup is None or sup_auth != "repo":
                    errors.append(
                        "a conversion change needs support quoting the repo code that computes "
                        "the printed value (its scale)"
                    )
                else:
                    try:
                        check_scale(str(self.new_value), ctx["claim_unit"], sup.quote)
                    except ValueError as e:
                        errors.append(str(e))
        elif _FORBIDDEN_TARGET_RE.search(target):
            errors.append(
                f"target {target!r} touches the metric, claim or tolerance: never changed"
            )
        elif self.kind == "arg":
            spec = _argparse_spec(info, "Change")
            opt = find_option(spec, target)
            if opt is None:
                known = sorted(o for a in spec for o in a.option_strings if o.startswith("-"))
                errors.append(f"arg target {target!r} is not an argparse option; options: {known}")
            elif target in ctx["claim_flags"] or any(
                o in ctx["claim_flags"] for o in opt.option_strings
            ):
                errors.append(f"{target} selects the claim's experiment: never changed")
            elif opt.takes_value:
                if self.new_value is None:
                    errors.append(f"{target} takes a value: new_value is required")
                else:
                    try:
                        parse_arg_value(opt, self.new_value)
                    except ValueError as e:
                        errors.append(str(e))
            elif opt.action in _FLAG_ACTIONS:
                if self.new_value is not None:
                    errors.append(f"{target} is {opt.action}: new_value must be null")
            else:
                errors.append(f"argparse action {opt.action!r} of {target} is not supported")
            if sup is None or sup_auth != "paper":
                errors.append("an arg change needs support citing PAPER evidence")
            elif self.new_value is not None and not value_stated(self.new_value, sup.quote):
                errors.append(
                    f"the cited paper quote does not state {self.new_value!r}: {sup.quote[:80]!r}"
                )
        elif self.kind == "assumption":
            open_unk = ctx["open_plan_unknowns"]
            if target not in open_unk:
                errors.append(
                    f"assumption target must be an open unknown the plan depends on: "
                    f"{sorted(open_unk) or 'none'}"
                )
            if self.new_value is None:
                errors.append("an assumption change needs new_value")
            if sup is None or sup_auth not in ("paper", "repo"):
                errors.append("an assumption change needs support citing paper or repo evidence")
            elif self.new_value is not None and not value_stated(self.new_value, sup.quote):
                errors.append(
                    f"the cited quote does not state {self.new_value!r}: {sup.quote[:80]!r}"
                )
        elif self.new_value is not None:  # none
            errors.append("kind none changes nothing: new_value must be null")
        if "seed" in target.lower() and not (
            sup is not None
            and sup_auth == "paper"
            and self.new_value is not None
            and value_stated(self.new_value, sup.quote)
        ):
            errors.append(
                "a seed change needs support citing paper text containing the seed value "
                "(no seed fishing)"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self


def lesson_text_issue(text: str, open_unknown_names: Set[str]) -> str | None:
    """Why the lesson text is not grounded, else None: it mentions "paper" together with a
    topic keyword of an OPEN unknown but says nowhere that the value is assumed / not stated /
    unknown (it would state what the paper does without a quote)."""
    t = " ".join(text.lower().split())
    if not re.search(r"\bpaper", t) or any(h in t for h in HEDGES):
        return None
    for name in sorted(open_unknown_names):
        words = (*UNKNOWN_TOPICS.get(name, ()), name.lower(), name.lower().replace("_", " "))
        hit = next((w for w in words if re.search(rf"\b{re.escape(w)}", t)), None)
        if hit is not None:
            return (
                f"the text mentions the paper and {hit!r}, but {name} is an open unknown (the "
                "paper does not state it): say it is assumed / not stated / unknown, or do not "
                "attribute it to the paper"
            )
    return None


class Lesson(_Frozen):
    """A Reflexion lesson about one trial of a claim. Checks (context, see above): claim_id and
    trial are the context's; evidence_ids are known (aliases resolved) and include the
    run_result or run_curve reasoned about; the step never goes back in LESSON_STEPS from the
    claim's previous lesson; an eval_protocol change needs step eval_protocol, an arg change
    step config_vs_paper; at most one change; the text never attributes an open unknown to the
    paper (lesson_text_issue)."""

    claim_id: str
    trial: int = Field(ge=1)
    step: LessonStep
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    text: str
    proposed_change: Change | None = None

    @model_validator(mode="before")
    @classmethod
    def _one_change(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            if "proposed_changes" in data or isinstance(data.get("proposed_change"), list):
                raise ValueError("at most one change: proposed_change is one object or null")
        return data

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _aliases(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, (list, tuple)):
            return [_resolve_ref(v, info) for v in value]
        return value

    @field_validator("text")
    @classmethod
    def _text_len(cls, v: str) -> str:
        n = len(v.split())
        if n > LESSON_MAX_WORDS:
            raise ValueError(f"lesson text has {n} words (> {LESSON_MAX_WORDS}, ~100 tokens)")
        return v

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        ctx = _lesson_context(info, "Lesson")
        errors: list[str] = []
        if self.claim_id != ctx["claim_id"]:
            errors.append(f"claim_id must be {ctx['claim_id']!r}")
        if self.trial != ctx["trial"]:
            errors.append(f"trial must be {ctx['trial']}")
        prev = ctx["previous_step"]
        if prev is not None:
            if prev not in LESSON_STEPS:
                raise ValueError(f"Lesson context previous_step {prev!r} is not a step")
            if LESSON_STEPS.index(self.step) < LESSON_STEPS.index(prev):
                later = list(LESSON_STEPS[LESSON_STEPS.index(prev) :])
                errors.append(
                    f"step {self.step!r} goes back: the previous lesson was at {prev!r}; "
                    f"use one of {later}"
                )
        issue = lesson_text_issue(self.text, ctx["open_unknown_names"])
        if issue is not None:
            errors.append(issue)
        known = ctx["evidence_ids"]
        unknown = [i for i in self.evidence_ids if i not in known]
        if unknown:
            errors.append(f"unknown evidence id(s) {[u[:16] for u in unknown]}")
        if not set(self.evidence_ids) & set(ctx["subject_ids"]):
            errors.append("evidence_ids must include the run_result or run_curve reasoned about")
        ch = self.proposed_change
        if ch is not None:
            if ch.kind == "eval_protocol" and self.step != "eval_protocol":
                errors.append("an eval_protocol change needs step 'eval_protocol'")
            if ch.kind == "arg" and self.step != "config_vs_paper":
                errors.append("an arg change needs step 'config_vs_paper'")
        if errors:
            raise ValueError("; ".join(errors))
        return self


# --- Subtask results (A env, B entrypoints, C workflow) ------------------------------------
# LLM output, validated with context {repo_files, evidence_ids, texts[, entrypoint_args]}.
# The LLM classifies and locates; every path, quote and argument is checked here in code.


UnknownName = Literal[
    "python_version",
    "torch_version",
    "cuda_version",
    "dataset_source",
    "checkpoint_url",
    "eval_protocol",
]


class UnknownProposal(_Frozen):
    """Name must also be allowed for the running subtask: context["allowed_unknowns"]
    (a set; empty = this subtask proposes no unknowns). Missing context fails closed."""

    name: UnknownName
    why: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _allowed_here(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        allowed = ctx.get("allowed_unknowns") if isinstance(ctx, Mapping) else None
        if not isinstance(allowed, (set, frozenset)):
            raise ValueError("UnknownProposal requires validation context key 'allowed_unknowns'")
        if self.name not in allowed:
            if not allowed:
                raise ValueError("this task takes no unknowns: return an empty list")
            raise ValueError(f"unknown {self.name!r} not allowed here; allowed: {sorted(allowed)}")
        return self


PackageSource = Literal["repo_file", "readme", "inferred"]
_VERSION_SOURCE_KINDS = {"readme": "readme", "repo_file": "repo_file"}


def _version_in_quote(version: str, quote: str) -> bool:
    v = version.strip()
    if v[:1] in ("v", "V") and v[1:2].isdigit():
        v = v[1:]
    if not v:
        return False
    pattern = rf"(?<![\d.]){re.escape(v)}(?!\.?\d)"
    return re.search(pattern, quote, flags=re.IGNORECASE) is not None


class PackageFact(_Frozen):
    """`source` is not part of the LLM schema: it is derived in code from the kind of the
    cited evidence (context["evidence_kinds"]: id -> kind), or "inferred" without citation."""

    name: str = Field(min_length=1)
    version: str | None = None
    citation: Citation | None = None
    source: SkipJsonSchema[PackageSource] = "inferred"

    @model_validator(mode="wrap")
    @classmethod
    def _derive_source(cls, data: Any, handler: Any, info: ValidationInfo) -> Self:
        given = data.get("source") if isinstance(data, Mapping) else None
        fact = handler(data)
        if fact.citation is None:
            derived: PackageSource = "inferred"
        else:
            ctx: Any = info.context
            kinds = ctx.get("evidence_kinds") if isinstance(ctx, Mapping) else None
            if not isinstance(kinds, Mapping):
                raise ValueError("PackageFact requires validation context key 'evidence_kinds'")
            kind = kinds.get(fact.citation.evidence_id)
            if kind not in _VERSION_SOURCE_KINDS:
                raise ValueError(f"citation must point to a README or repo file, got kind {kind!r}")
            derived = _VERSION_SOURCE_KINDS[kind]  # type: ignore[assignment]
            if fact.version is not None and not _version_in_quote(
                fact.version, fact.citation.quote
            ):
                raise ValueError(
                    f"version {fact.version!r} does not appear in the cited quote "
                    f"{fact.citation.quote[:80]!r}"
                )
        if given is not None and given != derived:
            raise ValueError(f"source is derived from the citation ({derived!r}); omit it")
        return fact.model_copy(update={"source": derived})


class EnvSpec(_Frozen):
    python: PackageFact | None = None
    packages: tuple[PackageFact, ...] = ()
    gpu_required: bool | None = None
    unknowns: tuple[UnknownProposal, ...] = ()


def derive_args_used(quote: str, arg_names: Set[str]) -> tuple[str, ...]:
    """Tokens of `quote` that are parsed argument names of the script, in order, unique.
    "--dataset=svhn" counts as "--dataset"; "--cutout_size" never counts as "--cutout"."""
    out: list[str] = []
    for token in quote.split():
        name = token.strip("`'\"").split("=", 1)[0]
        if name in arg_names and name not in out:
            out.append(name)
    return tuple(out)


class Entrypoint(_Frozen):
    """`args_used` is not part of the LLM schema: code derives it from the cited example
    (derive_args_used with the script's parsed arg names; () without example). A model-sent
    value must equal the derived one. If code verified commands for `path`
    (context["command_evidence"]: path -> ids of '<source>#commands' evidence), example must
    cite one of them; otherwise it may be null."""

    path: RepoPath
    kind: Literal["train", "eval", "other"]
    example: Citation | None = None
    args_used: SkipJsonSchema[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _check_candidate_and_example(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        if not isinstance(ctx, Mapping) or not isinstance(ctx.get("entrypoint_args"), Mapping):
            raise ValueError("Entrypoint requires validation context key 'entrypoint_args' (dict)")
        known: Mapping[str, Any] = ctx["entrypoint_args"]
        if self.path not in known:
            raise ValueError(
                f"{self.path!r} is not an entrypoint candidate; candidates: {sorted(known)}"
            )
        commands = ctx.get("command_evidence")
        if not isinstance(commands, Mapping):
            raise ValueError("Entrypoint requires validation context key 'command_evidence'")
        verified = commands.get(self.path) or frozenset()
        if verified and (self.example is None or self.example.evidence_id not in verified):
            aliases = ctx.get("aliases") or {}
            names = sorted(a for a, i in aliases.items() if i in verified) or sorted(
                i[:12] for i in verified
            )
            raise ValueError(
                f"example must cite a verified command for {self.path}: evidence_id one of {names}"
            )
        return self

    @model_validator(mode="wrap")
    @classmethod
    def _derive_args_used(cls, data: Any, handler: Any, info: ValidationInfo) -> Self:
        given = data.get("args_used") if isinstance(data, Mapping) else None
        ep = handler(data)
        if ep.example is None:
            derived: tuple[str, ...] = ()
        else:
            arg_names = info.context["entrypoint_args"][ep.path]  # checked by the after-validator
            derived = derive_args_used(ep.example.quote, arg_names)
        if given is not None and tuple(given) != derived:
            raise ValueError(
                f"args_used is derived from the example command ({list(derived)}); omit it"
            )
        return ep.model_copy(update={"args_used": derived})


class EntrypointList(_Frozen):
    entrypoints: tuple[Entrypoint, ...] = Field(min_length=1)
    unknowns: tuple[UnknownProposal, ...] = ()

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        counts = Counter(e.path for e in self.entrypoints)
        dupes = sorted(p for p, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate entrypoint path(s) {dupes}: one entry per script")
        return self


class DatasetFact(_Frozen):
    name: str = Field(min_length=1)
    citation: Citation | None = None


class CheckpointFact(_Frozen):
    name: str = Field(min_length=1)
    url: str | None = None
    citation: Citation


class MetricOutput(_Frozen):
    """Where a script prints or logs a metric. If code found metric lines for `path`
    (context["metric_line_evidence"]: path -> evidence_id of "<path>#metric_lines"), the
    citation must cite that evidence; otherwise any valid citation."""

    path: RepoPath
    metric: str = Field(min_length=1)
    citation: Citation

    @model_validator(mode="after")
    def _cites_metric_lines(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        lines = ctx.get("metric_line_evidence") if isinstance(ctx, Mapping) else None
        if not isinstance(lines, Mapping):
            raise ValueError("MetricOutput requires validation context key 'metric_line_evidence'")
        required = lines.get(self.path)
        if required is not None and self.citation.evidence_id != required:
            aliases = ctx.get("aliases") or {}
            alias = next((a for a, i in aliases.items() if i == required), required[:12])
            raise ValueError(
                f"cite the verified metric line for {self.path} with evidence_id {alias!r}"
            )
        return self


class WorkflowAnalysis(_Frozen):
    train_scripts: tuple[RepoPath, ...] = ()
    eval_scripts: tuple[RepoPath, ...] = ()
    configs: tuple[RepoPath, ...] = ()
    datasets: tuple[DatasetFact, ...] = ()
    checkpoints: tuple[CheckpointFact, ...] = ()
    metric_outputs: tuple[MetricOutput, ...] = ()
    unknowns: tuple[UnknownProposal, ...] = ()


# --- CriticPatch (grounded critic, S5) -----------------------------------------------------
# LLM output for a `critic` failure, validated with context {repo_files, failing_file,
# file_text, metric_lines}; every check is code. Missing context fails closed.

CRITIC_MAX_CHANGED_LINES = 20
CRITIC_MAX_EDITS = 3
_CRITIC_CONTEXT_KEYS = ("failing_file", "file_text", "metric_lines")


class LineEdit(_Frozen):
    """Replace lines start_line..end_line (1-based, inclusive) of the file. `original` is
    those lines verbatim (no line numbers); an empty `replacement` deletes them."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    original: str
    replacement: str

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError(f"end_line {self.end_line} < start_line {self.start_line}")
        if self.original.rstrip("\n") == self.replacement.rstrip("\n"):
            raise ValueError("replacement equals original: the edit changes nothing")
        return self


class CriticPatch(_Frozen):
    """One minimal fix in the failing file. Checks (context): file == context["failing_file"];
    each `original` equals lines start..end of context["file_text"] (the current text after
    the patch stack); ranges within the file and non-overlapping; no edit touches a line in
    context["metric_lines"][file]; removed + added lines <= CRITIC_MAX_CHANGED_LINES; the
    multiset of numeric literals (subscript indices and int axis args excluded, see
    patching.value_literals) is unchanged; the patched file parses (ast.parse, never run)."""

    file: RepoPath
    rationale: str = Field(min_length=1, max_length=300)
    edits: tuple[LineEdit, ...] = Field(min_length=1, max_length=CRITIC_MAX_EDITS)

    @model_validator(mode="after")
    def _check_against_file(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        if not isinstance(ctx, Mapping) or any(k not in ctx for k in _CRITIC_CONTEXT_KEYS):
            raise ValueError(
                f"CriticPatch requires validation context with keys {list(_CRITIC_CONTEXT_KEYS)}"
            )
        failing, text, metric = ctx["failing_file"], ctx["file_text"], ctx["metric_lines"]
        if not isinstance(text, str) or not isinstance(metric, Mapping):
            raise ValueError("CriticPatch context: file_text must be str, metric_lines a mapping")
        if self.file != failing:
            raise ValueError(f"patch the failing file {failing!r} only, not {self.file!r}")
        lines = split_lines(text)
        n = len(lines)
        errors: list[str] = []
        prev_end = 0
        for i, e in enumerate(sorted(self.edits, key=lambda e: e.start_line)):
            if e.end_line > n:
                errors.append(f"edit {i}: lines {e.start_line}-{e.end_line} beyond end ({n})")
                continue
            if e.start_line <= prev_end:
                errors.append(f"edit {i}: lines {e.start_line}-{e.end_line} overlap another edit")
            prev_end = max(prev_end, e.end_line)
            actual = range_text(lines, e.start_line, e.end_line)
            if e.original.rstrip("\n") != actual:
                errors.append(
                    f"edit {i}: original does not match lines {e.start_line}-{e.end_line}; "
                    f"they are exactly: {actual[:400]!r}"
                )
            hit = sorted(ln for ln in metric.get(self.file, ()) if e.start_line <= ln <= e.end_line)
            if hit:
                errors.append(f"edit {i}: touches metric line(s) {hit}; metric code is off-limits")
        if errors:
            raise ValueError("; ".join(errors))
        after = apply_line_edits(text, self.edits)
        changed = diff_changed_lines(unified_diff(self.file, text, after))
        if changed > CRITIC_MAX_CHANGED_LINES:
            raise ValueError(
                f"patch changes {changed} lines (removed + added) > {CRITIC_MAX_CHANGED_LINES}"
            )
        try:
            after_tree = ast.parse(after)
        except SyntaxError as e:
            raise ValueError(f"patched file does not parse: line {e.lineno}: {e.msg}") from None
        try:
            before_lits = value_literals(ast.parse(text))
        except SyntaxError:
            before_lits = None  # cannot compare: fail closed below
        if before_lits is None:
            raise ValueError("current file does not parse; numeric literals cannot be checked")
        after_lits = value_literals(after_tree)
        if before_lits != after_lits:
            removed = sorted((before_lits - after_lits).elements())
            added = sorted((after_lits - before_lits).elements())
            raise ValueError(
                f"numeric literals changed (removed {removed}, added {added}): a patch may not "
                "change values (hyperparameters, constants); exempt: subscript indices and int "
                "dim=/axis= values or shape-method args (.view(-1), .size(0))"
            )
        return self


# --- ClaimMapping (orchestrator, M3-S1b) ---------------------------------------------------
# LLM output, one per paper: which flag selects each claim's dataset, which paper setup fields
# set which flags, where the metric is printed and on what scale, and whether the setup fields
# that are not flags match the repo code. The LLM never writes a hyperparameter value: code
# copies them from the paper_setup evidence (run_plan.build_run_plan). Validation context
# (texts: the cited '#metric_lines' text is one verified line per text line):
#   claims              claim_id -> ResultClaim
#   paper_setup         setup name -> paper value (None = Unknown)
#   argparse_spec       ArgOption sequence of the entrypoint
#   metric_conversions  '#metric_lines' evidence_id -> {line: {claim_id: allowed conversions}}
#                       (built by code: run_plan.metric_conversions)
#   code_evidence_ids   ids of observed code texts (scale citations cite one of them)
#   train_config_ids    ids of '#train_config' evidence (setup checks may cite them or code)
#   texts, aliases, evidence_ids  for citations and alias resolution
# Missing context fails closed.

_MAPPING_CONTEXT_KEYS = (
    "claims",
    "paper_setup",
    "argparse_spec",
    "metric_conversions",
    "code_evidence_ids",
    "train_config_ids",
)
_HUNDRED_RE = re.compile(r"(?<![\d.])100(?:\.0*)?(?!\d)")


def _mapping_context(info: ValidationInfo, type_name: str) -> Mapping[str, Any]:
    ctx: Any = info.context
    if not isinstance(ctx, Mapping) or any(k not in ctx for k in _MAPPING_CONTEXT_KEYS):
        raise ValueError(
            f"{type_name} requires validation context with keys {list(_MAPPING_CONTEXT_KEYS)}"
        )
    return ctx


def _resolve_ref(value: Any, info: ValidationInfo) -> Any:
    """Alias ("E3") -> full evidence id (EvidenceRef then checks it is known)."""
    ctx: Any = info.context
    if isinstance(value, str) and isinstance(ctx, Mapping):
        aliases = ctx.get("aliases") or {}
        return aliases.get(value.strip(), value.strip())
    return value


def _check_setup_name(name: str, ctx: Mapping[str, Any]) -> None:
    if name not in ctx["paper_setup"]:
        raise ValueError(
            f"setup_name {name!r} is not a paper setup field; fields: {sorted(ctx['paper_setup'])}"
        )


def metric_line_of(text: str, quote: str) -> str:
    """The one line of a '#metric_lines' text (one verified line per text line) containing the
    (already snapped) quote. Raises ValueError if none or several lines contain it."""
    hits = [ln for ln in text.split("\n") if quote.strip() and quote.strip() in ln]
    if len(hits) != 1:
        raise ValueError(
            f"metric_line_quote must lie within exactly one metric line, found {len(hits)}: "
            "copy the whole line you mean"
        )
    return hits[0]


def normalize_token(text: str) -> str:
    """Lowercase, letters and digits only: "CIFAR-100" -> "cifar100"."""
    return "".join(c for c in text.lower() if c.isalnum())


def appears_in(value: str, text: str) -> bool:
    """normalize_token(value) occurs in normalize_token(text), not glued to more digits on a
    digit edge ("cifar10" is not in "CIFAR-100", "resnet18" is in "ResNet18 + cutout")."""
    v, t = normalize_token(value), normalize_token(text)
    if not v:
        return False
    start = t.find(v)
    while start >= 0:
        end = start + len(v)
        before_ok = not (v[0].isdigit() and start > 0 and t[start - 1].isdigit())
        after_ok = not (v[-1].isdigit() and end < len(t) and t[end].isdigit())
        if before_ok and after_ok:
            return True
        start = t.find(v, start + 1)
    return False


def check_scale(conversion: str, claim_unit: str | None, quote: str) -> None:
    """The cited code must show the printed value's scale: a fraction (a division, no 100) for
    *_fraction_* conversions, a percent (a 100) for percent sources (incl. identity to %)."""
    has_100 = _HUNDRED_RE.search(quote) is not None
    if "fraction" in conversion:
        if "/" not in quote or has_100:
            raise ValueError(
                f"{conversion} needs a fraction: cite the code computing the printed value as a "
                f"division without 100 (e.g. 'correct / total'), got {quote[:80]!r}"
            )
    elif conversion == "accuracy_percent_to_error_percent" or (
        conversion == "identity" and claim_unit == "%"
    ):
        if not has_100:
            raise ValueError(
                f"{conversion} needs a percent: cite the code scaling the printed value by 100, "
                f"got {quote[:80]!r}"
            )


class ClaimFlag(_Frozen):
    """A flag choice the claim names (e.g. --dataset cifar100, --model resnet18)."""

    flag: str
    value: str


class SetupFlag(_Frozen):
    """A paper setup field whose value code copies into a flag (never the LLM)."""

    setup_name: str
    flag: str


class MetricMap(_Frozen):
    """metric_line_ref: the entrypoint's '#metric_lines' evidence (alias or id);
    metric_line_quote: a citation of the one line of it that prints the metric (code uses that
    line). scale_citation: repo code showing the printed value's scale."""

    metric_line_ref: EvidenceRef
    metric_line_quote: Citation
    conversion: Conversion
    scale_citation: Citation

    @field_validator("metric_line_ref", mode="before")
    @classmethod
    def _alias(cls, value: Any, info: ValidationInfo) -> Any:
        return _resolve_ref(value, info)

    @model_validator(mode="after")
    def _check_refs(self, info: ValidationInfo) -> Self:
        ctx = _mapping_context(info, "MetricMap")
        if self.metric_line_ref not in ctx["metric_conversions"]:
            raise ValueError("metric_line_ref must be the verified metric lines' alias")
        if self.metric_line_quote.evidence_id != self.metric_line_ref:
            raise ValueError("metric_line_quote must cite the metric_line_ref evidence")
        if self.scale_citation.evidence_id not in ctx["code_evidence_ids"]:
            raise ValueError("scale_citation must quote the script's code (its alias), not lines")
        return self


class ClaimMap(_Frozen):
    """Checks (context): known claim_id; each claim_args flag is a value-taking argparse option
    with choices, set once, its value one of them and named by the claim (appears_in its
    setting, dataset or quote); setup names are paper setup fields, each once; flags are
    value-taking options and a stated paper value parses with the flag's type and choices; the
    quoted metric line is one line of the cited evidence and the conversion is allowed from it;
    the scale citation matches the conversion."""

    claim_id: str
    claim_args: tuple[ClaimFlag, ...] = Field(min_length=1)
    setup_mappings: tuple[SetupFlag, ...] = ()
    metric: MetricMap

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        ctx = _mapping_context(info, "ClaimMap")
        spec = _argparse_spec(info, "ClaimMap")
        claim = ctx["claims"].get(self.claim_id)
        if claim is None:
            raise ValueError(f"unknown claim_id {self.claim_id!r}; claims: {sorted(ctx['claims'])}")
        errors: list[str] = []

        def option(flag: str, what: str) -> ArgOption | None:
            opt = find_option(spec, flag)
            if opt is None:
                known = sorted(o for a in spec for o in a.option_strings if o.startswith("-"))
                errors.append(f"{what}: flag {flag!r} is not an argparse option; options: {known}")
            elif not opt.takes_value:
                errors.append(f"{what}: {flag} takes no value")
                return None
            return opt

        claim_text = (claim.setting or "", claim.dataset, claim.quote)
        dests: Counter[str | None] = Counter()
        for ca in self.claim_args:
            opt = option(ca.flag, "claim_args")
            if opt is None:
                continue
            dests[opt.dest] += 1
            if opt.choices is None:
                errors.append(f"claim_args: {opt.option_strings[0]} has no argparse choices")
                continue
            try:
                parse_arg_value(opt, ca.value)
            except ValueError as e:
                errors.append(f"claim_args: {e}")
                continue
            if not any(appears_in(ca.value, t) for t in claim_text):
                errors.append(
                    f"claim_args: {ca.flag} {ca.value!r} is not named by the claim's setting, "
                    "dataset or quote"
                )
        twice = sorted(str(d) for d, n in dests.items() if n > 1)
        if twice:
            errors.append(f"claim_args: argument(s) set more than once: {twice}")
        names = Counter(m.setup_name for m in self.setup_mappings)
        dupes = sorted(n for n, k in names.items() if k > 1)
        if dupes:
            errors.append(f"setup_mappings: field(s) mapped more than once: {dupes}")
        for m in self.setup_mappings:
            try:
                _check_setup_name(m.setup_name, ctx)
            except ValueError as e:
                errors.append(f"setup_mappings: {e}")
                continue
            opt = option(m.flag, f"setup_mappings {m.setup_name}")
            value = ctx["paper_setup"][m.setup_name]
            if opt is not None and value is not None:
                try:
                    parse_arg_value(opt, str(value))
                except ValueError as e:
                    errors.append(f"setup_mappings {m.setup_name} (paper value {value!r}): {e}")
        try:
            ref = self.metric.metric_line_ref
            line = metric_line_of(ctx["texts"][ref], self.metric.metric_line_quote.quote)
            per_line = ctx["metric_conversions"][ref]
            if line not in per_line:
                raise ValueError(f"no metric pattern can be built from {line!r}")
            allowed = per_line[line].get(self.claim_id, frozenset())
            if self.metric.conversion not in allowed:
                raise ValueError(
                    f"conversion {self.metric.conversion!r} is not allowed from {line!r} for "
                    f"{self.claim_id}; allowed: {sorted(allowed)}"
                )
            check_scale(self.metric.conversion, claim.unit, self.metric.scale_citation.quote)
        except ValueError as e:
            errors.append(f"metric: {e}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class UnmappedSetup(_Frozen):
    setup_name: str
    reason: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _known(self, info: ValidationInfo) -> Self:
        _check_setup_name(self.setup_name, _mapping_context(info, "UnmappedSetup"))
        return self


class SetupCheck(_Frozen):
    """A stated paper setup field that is not a flag, checked against the repo: the citation
    must quote '#train_config' lines or the script's code."""

    setup_name: str
    repo_citation: Citation
    consistent: bool
    note: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> Self:
        ctx = _mapping_context(info, "SetupCheck")
        _check_setup_name(self.setup_name, ctx)
        if ctx["paper_setup"][self.setup_name] is None:
            raise ValueError(f"{self.setup_name} is not stated by the paper: nothing to check")
        allowed = set(ctx["train_config_ids"]) | set(ctx["code_evidence_ids"])
        if self.repo_citation.evidence_id not in allowed:
            raise ValueError("repo_citation must quote the train config lines or the script code")
        return self


class ClaimMapping(_Frozen):
    """Every claim_id of context["claims"] exactly once; a setup field at most once across
    setup_checks and unmapped_setup, and never both mapped to a flag and unmapped."""

    claims: tuple[ClaimMap, ...] = Field(min_length=1)
    unmapped_setup: tuple[UnmappedSetup, ...] = ()
    setup_checks: tuple[SetupCheck, ...] = ()

    @model_validator(mode="after")
    def _coverage(self, info: ValidationInfo) -> Self:
        ctx = _mapping_context(info, "ClaimMapping")
        counts = Counter(c.claim_id for c in self.claims)
        errors: list[str] = []
        missing = sorted(set(ctx["claims"]) - set(counts))
        dupes = sorted(c for c, n in counts.items() if n > 1)
        if missing:
            errors.append(f"claim(s) not mapped: {missing}")
        if dupes:
            errors.append(f"claim(s) mapped more than once: {dupes}")
        for label, items in (
            ("setup_checks", [s.setup_name for s in self.setup_checks]),
            ("unmapped_setup", [u.setup_name for u in self.unmapped_setup]),
        ):
            twice = sorted(n for n, k in Counter(items).items() if k > 1)
            if twice:
                errors.append(f"{label}: field(s) listed more than once: {twice}")
        mapped = {m.setup_name for c in self.claims for m in c.setup_mappings}
        both = sorted(mapped & {u.setup_name for u in self.unmapped_setup})
        if both:
            errors.append(f"field(s) both mapped to a flag and unmapped: {both}")
        if errors:
            raise ValueError("; ".join(errors))
        return self
