"""Graph state for the configure graph, with its pure reducers.

Parallel branches (subtask_a/b/c) return partial updates; LangGraph merges each key with
its reducer. Reducers are pure and give the same result whatever order the branches
finish in (events: same set; their relative order is kept within one update).
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.evidence import Evidence
from reproducer.domain.unknowns import Assumption, UnknownOpened, UnknownResolved

if TYPE_CHECKING:
    from reproducer.providers.datasets import DatasetStatus
    from reproducer.providers.docker_runner import BuildResult, RunResult
    from reproducer.providers.llm import Budget, MeteredLLM
    from reproducer.providers.store import RunStore

UnknownEventT = UnknownOpened | UnknownResolved | Assumption
ConfigureStatus = Literal["running", "configured", "configure_failed"]
MAX_CONFIGURE_ATTEMPTS = 1  # M1: one pass; the M2 critic loop raises this


def merge_evidence(
    left: Iterable[Evidence] | None, right: Iterable[Evidence] | None
) -> list[Evidence]:
    """Union by evidence_id (ids are content hashes, so equal id = equal record), sorted by
    (kind, locator, evidence_id): the result does not depend on merge order."""
    by_id: dict[str, Evidence] = {}
    for ev in (*(left or ()), *(right or ())):
        by_id.setdefault(ev.evidence_id, ev)
    return sorted(by_id.values(), key=lambda e: (e.kind, e.locator, e.evidence_id))


def event_key(event: UnknownEventT) -> str:
    """Identity for exact-duplicate detection: event type + all fields."""
    return f"{type(event).__name__}:{event.model_dump_json()}"


def merge_events(
    left: Iterable[UnknownEventT] | None, right: Iterable[UnknownEventT] | None
) -> list[UnknownEventT]:
    """Append right to left, dropping events equal to one already present (exact duplicate:
    same type and same fields). Order is otherwise kept (opened before resolved)."""
    out: list[UnknownEventT] = []
    seen: set[str] = set()
    for event in (*(left or ()), *(right or ())):
        key = event_key(event)
        if key not in seen:
            seen.add(key)
            out.append(event)
    return out


def keep_first(left: str | None, right: str | None) -> str:
    """Reducer for the fallback keys: the first non-empty value wins (parallel branches may
    both fail; the first recorded stop is the one reported)."""
    return left or right or ""


class SubtaskSummary(BaseModel):
    """What one subtask branch reports to the fan-in (no payload: that is in evidence)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["A", "B", "C"]
    ok: bool
    attempts: int
    tokens: int
    latency_s: float
    errors: tuple[str, ...] = ()


class ConfigureState(TypedDict, total=False):
    # inputs (repo_url, or spec_path from which ingest_spec sets repo_url)
    out_dir: str
    repo_url: str
    spec_path: str
    commit: str | None
    max_chars: int
    max_tokens: int
    # produced by the graph
    spec: ReproSpec  # set by ingest_spec
    paper_source: str  # "paper:<spec folder name>", set by ingest_spec
    failure_reason: str  # set when ingest_spec fails (graph ends before configure)
    source: str
    evidence: Annotated[list[Evidence], merge_evidence]
    events: Annotated[list[UnknownEventT], merge_events]
    outcomes: Annotated[list[SubtaskSummary], operator.add]
    status: ConfigureStatus
    configure_attempts: int  # bounded by MAX_CONFIGURE_ATTEMPTS
    evidence_written: int  # set by persist
    events_written: int  # set by persist
    # default fallback route (routing.FALLBACK_NODE / build.guard_node): "fallback: <router or
    # node>: <unexpected value or error type>"; once set, every later node is a no-op and
    # every router goes to END (claims stay open)
    fallback_reason: Annotated[str, keep_first]
    fallback_detail: Annotated[str, keep_first]  # the error message (node exceptions)


class ConfigureDeps(TypedDict):
    """Injected IO. store: the single writer target (persist + telemetry); llm_for(node) ->
    MeteredLLM sharing `budget`."""

    store: RunStore
    llm_for: Callable[[str], MeteredLLM]
    budget: Budget


# --- setup graph (configure -> env -> data -> smoke, fixes / critic) --------------------------

SetupStatus = Literal[
    "running",
    "ready",  # smoke passed
    "infra",  # GPU / driver / container failure: stop, never the critic
    "capped",  # unfixable, a cap reached (incl. the critic token budget), or the critic gave up
    "llm_unavailable",  # critic backend down / not configured: retryable, claims stay open
    "configure_failed",
    "env_failed",
    "data_failed",
    "smoke_failed",  # nothing to smoke (no verified example command, image missing)
    "plan_failed",  # the orchestrator gave no valid mapping / plans
    "awaiting_approval",  # a gated action waits in the escalation queue (`queue`)
    "stopped",  # --until <stage> reached
    "fallback",  # default route: unexpected router state / node exception (fallback_reason)
]
UntilStage = Literal["configure", "env", "data", "smoke", "plan", "run", "vault"]


class SetupState(ConfigureState, total=False):
    """ConfigureState + the setup phases. Node payloads that are plain Python objects
    (EnvPlan, FixState, CriticState, SmokeStep, SmokeSetup, summaries) are typed Any here so the
    schema does not import the node modules; their types are documented per key."""

    # inputs
    until: UntilStage  # default (key absent): smoke
    no_vault: bool  # full pipeline: skip the vault node
    only_claim: str | None  # --claim: plan every claim; run / evaluate / reflect only this one
    vault_dir: str  # vault root (default <out_dir>/vault_out)
    no_checkpoints: bool  # full runs: do not keep *.pt / *.pth artifacts
    cpu_only: bool
    entrypoint: str | None
    smoke_timeout_s: float
    build_timeout_s: float
    data_timeout_s: float
    # env
    env_plan: Any  # EnvPlan
    image_tag: str
    env_ok: bool
    env_builds: int  # 1 + rebuilds after add_package fixes
    env_log: Annotated[list[Any], operator.add]  # EnvSummary per env run
    # data
    data: Any  # DataSummary
    data_ok: bool
    mount_plan: Any  # DataMountPlan
    # smoke / fixes / critic
    smoke_setup: Any  # SmokeSetup (stored facts for the smoke run)
    fix_state: Any  # FixState (known fixes applied so far, folded in order)
    critic_state: Any  # CriticState (patch stack, last signature, attempts)
    pending: Any  # SmokeStep waiting for apply_known_fix / critic to record it, or None
    last_smoke: Any  # routing.LastSmoke
    last_fix_kind: str
    last_critic_status: str
    smoke_iterations: int
    smoke_log: Annotated[list[Any], operator.add]  # IterationSummary per smoke run
    known_fixes_used: int  # <= routing.MAX_KNOWN_FIXES
    critic_attempts: int  # <= critic.MAX_CRITIC_ATTEMPTS (stored attempts count)
    # resume (full pipeline only): stages taken from the store instead of re-run
    resumed: list[str]  # "subtasks" (stored A/B/C output), "smoke" (a passed smoke with the
    # current smoke input digest)
    # plan -> per-claim workers -> fan-in -> evaluate -> vault
    plan_outcome: Any  # orchestrator.OrchestratorOutcome
    plans: list[str]  # claim ids of the current plans (spec claim order = GPU order)
    plan_failures: dict[str, str]  # claim_id -> why it did not plan (the claim stays open)
    worker_claim_id: str  # the one claim of a claim_worker task (Send payload)
    worker_order: list[str]  # the plans of the fan-out (Send payload): GPU turn order
    worker_results: Annotated[list[Any], operator.add]  # pipeline.WorkerResult per claim
    workers_done: int  # set by the fan-in (join)
    evaluations: Any  # dict claim_id -> ClaimEvaluation (read-only, from evidence)
    evaluate_error: str
    vault_counts: Any  # providers.vault.WriteCounts
    vault_paper_note: str
    vault_error: str
    vault_approval: Any  # domain.approval.GateResult of a gated shared vault write, if any
    # Reflexion loop (evaluate -> reflect -> apply_lesson -> rerun | reeval -> evaluate)
    reflect: bool  # input: run the loop (CLI default on; absent = off)
    reflect_info: dict[str, Any]  # claim_id -> newest trial, outcome, breaker (evaluate node)
    reflect_pending: list[dict[str, Any]]  # lessons to apply: claim_id, lesson_id, input_digest
    reflect_failures: dict[str, Any]  # claim_id -> {trial, reason}: no lesson for that trial
    rerun_claims: list[str]  # claims whose applied change needs a new run (worker fan-out)
    reflexion_log: Annotated[list[Any], operator.add]  # one line per reflect / apply step
    # outcome
    setup_status: SetupStatus
    stop_reason: str  # routing.StopReason, or <stage>_failed / until
    stop_detail: str


class SetupDeps(ConfigureDeps, total=False):
    """Injected IO for the setup graph (tests pass fakes). Required: plan_env(out, cpu_only),
    build_image, image_size, image_exists, runner (run_container), ensure_datasets,
    data_cache_dir, unload (free the GPU from Ollama). Optional: warn(msg)."""

    plan_env: Callable[[Any, bool], Any]
    build_image: Callable[..., BuildResult]
    image_size: Callable[[str], int | None]
    image_exists: Callable[[str], bool]
    runner: Callable[..., RunResult]
    ensure_datasets: Callable[..., dict[str, DatasetStatus]]
    data_cache_dir: Callable[[], Any]
    unload: Callable[[], Any]
    warn: Callable[[str], None]
    # pipeline (plan -> claim workers): optional
    make_critic: Callable[[Any], Any]  # SmokeSetup -> CriticFn | None (pre-flight smoke)
    env: Any  # Mapping[str, str] for the GPU-hour caps (default os.environ)
    gpu_lock: Any  # context manager; default pipeline.GPU_SEMAPHORE (one GPU job at a time)
    gpu_turns: Any  # pipeline.GpuTurns; default pipeline.GPU_TURNS (GPU in plan order)
    log: Callable[[str], None]  # per-claim pre-flight lines
    progress: Callable[[str], None]  # full-run progress lines
