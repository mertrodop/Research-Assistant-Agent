"""Contracts: ReproSpec, RunPlan, ExperimentResult, Lesson.

Pure Pydantic v2 models. No file, network or subprocess IO.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Set
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from reproducer.domain.grounding import Citation, RepoPath


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

OverrideSource = Literal["paper", "repo", "unknown"]

_RUNPLAN_CONTEXT_KEYS = ("repo_files", "config_keys", "claims")


class Override(_Frozen):
    key: str
    value: bool | int | float | str
    source: OverrideSource


class RunPlan(_Frozen):
    """Validated against the repo and spec via validation context.

    Build with make_run_plan(...). Validation without the context fails (no silent pass),
    and model_copy is disabled so a validated plan cannot be altered unchecked.
    """

    claim_id: str
    entrypoint: str
    config: str
    overrides: tuple[Override, ...] = ()
    dataset: str
    metric: str

    @model_validator(mode="after")
    def _check_against_repo(self, info: ValidationInfo) -> Self:
        ctx: Any = info.context
        if not isinstance(ctx, dict) or any(k not in ctx for k in _RUNPLAN_CONTEXT_KEYS):
            raise ValueError(
                f"RunPlan requires validation context with keys {list(_RUNPLAN_CONTEXT_KEYS)}"
            )
        repo_files = ctx["repo_files"]
        config_keys = ctx["config_keys"]
        claims = ctx["claims"]

        errors: list[str] = []
        if self.entrypoint not in repo_files:
            errors.append(f"entrypoint not in repo: {self.entrypoint!r}")
        if self.config not in repo_files:
            errors.append(f"config not in repo: {self.config!r}")
        bad_keys = [o.key for o in self.overrides if o.key not in config_keys]
        if bad_keys:
            errors.append(f"override key(s) not in config: {bad_keys}")
        claim = claims.get(self.claim_id)
        if claim is None:
            errors.append(f"unknown claim_id: {self.claim_id!r}")
        elif self.metric != claim.metric:
            errors.append(f"metric {self.metric!r} != claim metric {claim.metric!r}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @property
    def is_spec_faithful(self) -> bool:
        return all(o.source != "unknown" for o in self.overrides)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        raise TypeError("RunPlan must be built via make_run_plan")


def make_run_plan(
    data: dict[str, Any],
    *,
    repo_files: set[str],
    config_keys: set[str],
    claims: dict[str, ResultClaim],
) -> RunPlan:
    """Build a RunPlan validated against repo files, config keys and spec claims."""
    return RunPlan.model_validate(
        data,
        context={"repo_files": repo_files, "config_keys": config_keys, "claims": claims},
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

# Contract says text <= 100 tokens. To keep the domain pure (no tokenizer dependency) we
# approximate: English BPE averages ~0.75 words per token, so 100 tokens ~= 75 words.
LESSON_MAX_WORDS = 75


class Lesson(_Frozen):
    claim_id: str
    trial: int = Field(ge=1)
    step: LessonStep
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    text: str

    @field_validator("text")
    @classmethod
    def _text_len(cls, v: str) -> str:
        n = len(v.split())
        if n > LESSON_MAX_WORDS:
            raise ValueError(f"lesson text has {n} words (> {LESSON_MAX_WORDS}, ~100 tokens)")
        return v


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
