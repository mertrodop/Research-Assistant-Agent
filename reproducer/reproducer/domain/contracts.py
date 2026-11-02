"""Contracts: ReproSpec, RunPlan, ExperimentResult, Lesson.

Pure Pydantic v2 models. No file, network or subprocess IO.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self


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


# --- SubtaskResult drafts ------------------------------------------------------------------
# DRAFT until M1


class EnvSpec(_Frozen):
    python_version: str
    base_image: str
    install_cmds: tuple[str, ...]
    requirements_files: tuple[str, ...]
    evidence_ids: tuple[str, ...]


class Entrypoint(_Frozen):
    path: str
    kind: Literal["train", "eval", "other"]
    args_hint: str | None = None


class EntrypointList(_Frozen):
    entrypoints: tuple[Entrypoint, ...]
    evidence_ids: tuple[str, ...]


class WorkflowAnalysis(_Frozen):
    train_scripts: tuple[str, ...]
    eval_scripts: tuple[str, ...]
    configs: tuple[str, ...]
    data_paths: tuple[str, ...]
    checkpoints: tuple[str, ...]
    evidence_ids: tuple[str, ...]
