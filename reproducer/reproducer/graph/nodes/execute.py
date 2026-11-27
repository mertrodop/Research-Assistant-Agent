"""Execute node (M3-S3): pre-flight + full runs of the current RunPlans. No LLM but the critic.

Per current plan (orchestrator.load_current_plans; mapping order):
  0. run digest = smoke_input_digest(image tag, fix state, critic patch stack, full command):
     everything that determines the run's inputs. A plan whose run digest already has a
     COMPLETED run_result (exit 0 and the claim metric present and finite) is skipped; an
     incomplete one (crashed / killed / no metric) is re-run from scratch (no resume).
  1. pre-flight: the smoke loop (graph/nodes/smoke.py) on the PLAN's rendered command with the
     budget overrides (epochs -> 1); failures go through the usual classify -> known fix /
     critic routing, bounded by the caps counted from stored evidence. A stored PASSED smoke
     with the same smoke input digest (same image, fix state, patches and command) is reused.
     Estimate: gpu_hours = (sec_per_epoch * planned_epochs + eval_overhead) / 3600 with
     planned_epochs = the effective epoch argument (plan arg, else the argparse default) and
     eval_overhead = planned_epochs * (smoke wall time - sec_per_epoch) when the timing came
     from a tqdm bar (the repo evaluates after every epoch; start-up is counted too, so the
     estimate errs high), else 0 (wall timing already includes it).
     Caps: REPRODUCER_RUN_GPU_HOURS (default 4) per run; REPRODUCER_PAPER_GPU_HOURS (default
     10) per paper minus the gpu_hours of the stored run_result evidence. Decision: run |
     skip_cap | skip_smoke_failed, stored as `preflight` evidence (authority execution,
     observed; links: run_plan + smoke_result).
  2. execute (decision run): fresh per-run copy (clone + patch stack + prep dirs), the full
     command (+ known-fix overrides, e.g. a halved batch) in the locked-down container
     (docker_runner: network none, data read-only, GPU), timeout = min(run cap, 1.5 x estimate
     + 10 min). Progress (epoch k/N) goes to stderr. After exit the repo's output dirs (the fix
     state's prep dirs, e.g. logs/, checkpoints/) are copied to <run dir>/artifacts (without
     *.pt / *.pth with no_checkpoints) BEFORE the working copy is deleted.
  3. metric: extract_metric(log, plan.metric_source) -> raw, convert -> value; ExperimentResult
     metrics {claim.metric: value, "raw:<name>": raw}; a nonzero exit, a timeout or a missing /
     unconvertible metric leaves the claim metric out (not completed).
  4. run_result evidence (ledger.make_run_evidence). spec_faithful = plan.is_spec_faithful
     AND no env deviation AND no plan deviation AND no known-fix deviation AND no critic patch
     AND no assumption evidence (the plan's open-unknown assumptions + env assumptions).
     config_digest = the run digest; links = assumption evidence + run_plan + preflight.
     The ledger is never written here (S4).
  2b. human approval (decision run only; domain/approval.py, env REPRODUCER_APPROVAL): when
     the policy gates the run (default: estimate_h >= REPRODUCER_APPROVE_GPU_HOURS), an
     approval_request (input digest over claim + run digest; links the preflight) is filed
     once and the claim is awaiting_approval (nothing executed). A rejected request leaves
     the claim open (rejected_by_human: <note>); an approved one runs. A pre-flight whose
     critic patch awaits approval is awaiting_approval too.
  4b. run_curve: for a run that exited 0, its curve summary (per-epoch CSV artifact, else the
     metric lines of the log) as run_curve evidence (graph/nodes/diagnose.py); for a crashed
     or diverged run with points, a partial one (content partial: true).
  5. abort: if the full run is interrupted (Ctrl-C, an exception) before its run_result is
     written, then in a finally block, and the interruption is re-raised afterwards:
     - after the container was started (the runner was called): a run_result with status
       "aborted" (exit_code None, never completed) and the duration / gpu_hours so far. It
       is a run attempt and counts toward the paper cap.
     - before that (workdir copy, patches, image check): a `setup_failed` note naming the
       stage. Not a run_result: no attempt, 0 GPU hours.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from reproducer.domain.approval import (
    ApprovalPolicy,
    GateResult,
    approval_request_evidence,
    gate,
    policy_from_env,
    request_digest,
)
from reproducer.domain.contracts import ArgOption, ExperimentResult, find_option
from reproducer.domain.env_rules import env_plan_to_evidence
from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.ledger import make_aborted_run_evidence, make_run_evidence
from reproducer.domain.metric_format import metric_decimals
from reproducer.domain.run_plan import (
    convert,
    extract_metric,
    metric_direction,
    parse_metric_line,
)
from reproducer.domain.smoke_plan import EPOCH_DESTS, plan_smoke
from reproducer.domain.unknowns import Assumption, UnknownState, fold_unknowns
from reproducer.graph.nodes.diagnose import CurveSpec, record_run_curve, wants_curve
from reproducer.graph.nodes.orchestrator import PlannedClaim, load_current_plans
from reproducer.graph.nodes.smoke import (
    Iteration,
    SmokeSetup,
    make_build,
    prepare_smoke,
    record_iteration,
    smoke_loop,
)
from reproducer.graph.routing import MAX_KNOWN_FIXES
from reproducer.providers.docker_runner import RunResult, new_run_dir, prepare_workdir
from reproducer.providers.store import RunStore, make_ref
from reproducer.rules.known_fixes import FixState, smoke_input_digest

RUN_CAP_ENV = "REPRODUCER_RUN_GPU_HOURS"
PAPER_CAP_ENV = "REPRODUCER_PAPER_GPU_HOURS"
DEFAULT_RUN_CAP_H = 4.0
DEFAULT_PAPER_CAP_H = 10.0
TIMEOUT_FACTOR = 1.5
TIMEOUT_SLACK_S = 600.0
PROGRESS_EVERY_S = 30.0
CHECKPOINT_PATTERNS = ("*.pt", "*.pth")

Decision = Literal["run", "skip_cap", "skip_smoke_failed"]
SetupStage = Literal["prepare_workdir", "image_check"]


class RunSetupError(RuntimeError):
    """The full run could not be started (e.g. its image is missing)."""


Runner = Callable[..., RunResult]


# --- estimate + caps (pure) ------------------------------------------------------------------


def planned_epochs(args: Iterable[tuple[str, str | None]], spec: Sequence[ArgOption]) -> int | None:
    """The effective epoch budget: the plan's epoch argument, else its argparse default; None
    if the script has no epoch argument or its value is not an int."""
    spec = tuple(spec)
    opt = next((a for a in spec if a.dest in EPOCH_DESTS and a.takes_value), None)
    if opt is None:
        return None
    value: Any = opt.default
    for flag, v in args:
        o = find_option(spec, flag)
        if o is not None and o.dest == opt.dest:
            value = v
    try:
        n = int(str(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def eval_overhead_s(
    *, sec_per_epoch: float, smoke_duration_s: float, timing_method: str, epochs: int
) -> float:
    """Per-epoch time the tqdm bar does not show (evaluation + start-up), times epochs. A wall
    timing already contains it -> 0."""
    if timing_method != "tqdm":
        return 0.0
    return max(0.0, smoke_duration_s - sec_per_epoch) * epochs


def estimate_gpu_hours(sec_per_epoch: float, epochs: int, overhead_s: float) -> float:
    return (sec_per_epoch * epochs + overhead_s) / 3600.0


@dataclass(frozen=True)
class Caps:
    run_h: float
    paper_h: float
    spent_h: float

    @property
    def paper_left_h(self) -> float:
        return max(0.0, self.paper_h - self.spent_h)

    def as_content(self) -> dict[str, float]:
        return {
            "run_h": self.run_h,
            "paper_h": self.paper_h,
            "spent_h": round(self.spent_h, 4),
            "paper_left_h": round(self.paper_left_h, 4),
        }


def _hours(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number of GPU hours") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name}={raw!r} must be finite and >= 0")
    return v


def caps_from_env(env: Mapping[str, str], spent_h: float) -> Caps:
    return Caps(
        run_h=_hours(env, RUN_CAP_ENV, DEFAULT_RUN_CAP_H),
        paper_h=_hours(env, PAPER_CAP_ENV, DEFAULT_PAPER_CAP_H),
        spent_h=spent_h,
    )


def spent_gpu_hours(evidence: Iterable[Evidence]) -> float:
    """GPU hours of every stored run_result (completed or not: the GPU was used)."""
    total = 0.0
    for ev in evidence:
        if ev.kind == "run_result":
            v = ev.content.get("gpu_hours")
            if isinstance(v, (int, float)) and math.isfinite(v):
                total += float(v)
    return total


def decide(smoke_passed: bool, estimate_h: float | None, caps: Caps) -> tuple[Decision, str]:
    if not smoke_passed:
        return "skip_smoke_failed", "pre-flight smoke did not pass"
    if estimate_h is None:
        return "skip_cap", "no GPU-hour estimate (no epoch argument or no smoke timing)"
    if estimate_h > caps.run_h:
        return "skip_cap", f"estimate {estimate_h:.2f} h > per-run cap {caps.run_h:g} h"
    if estimate_h > caps.paper_left_h:
        return "skip_cap", (
            f"estimate {estimate_h:.2f} h > paper budget left {caps.paper_left_h:.2f} h "
            f"({caps.paper_h:g} h cap - {caps.spent_h:.2f} h spent)"
        )
    return "run", f"estimate {estimate_h:.2f} h within caps"


def run_timeout_s(estimate_h: float, run_cap_h: float) -> float:
    return min(run_cap_h * 3600.0, TIMEOUT_FACTOR * estimate_h * 3600.0 + TIMEOUT_SLACK_S)


# --- run digest, completion, fidelity (pure) ------------------------------------------------


def full_argv(command: str, state: FixState) -> tuple[str, ...]:
    """The plan's command with the known-fix overrides (no budget overrides)."""
    return plan_smoke(command, (), state.overrides).argv


def run_digest(*, image: str, state: FixState, patches: Sequence[str], argv: Sequence[str]) -> str:
    """sha256 over the run's inputs (same scheme as the smoke input digest)."""
    return smoke_input_digest(
        image=image, state=state, patches=tuple(patches), command=shlex.join(argv)
    )


def is_completed(content: Mapping[str, Any], metric: str) -> bool:
    if content.get("exit_code") != 0:
        return False
    metrics = content.get("metrics")
    v = metrics.get(metric) if isinstance(metrics, Mapping) else None
    return isinstance(v, (int, float)) and math.isfinite(v)


def completed_run(evidence: Iterable[Evidence], digest: str, metric: str) -> Evidence | None:
    """The newest COMPLETED run_result for this run digest, if any."""
    hit = None
    for ev in evidence:
        if (
            ev.kind == "run_result"
            and ev.content.get("config_digest") == digest
            and is_completed(ev.content, metric)
        ):
            hit = ev
    return hit


def is_spec_faithful(
    *,
    plan_faithful: bool,
    env_deviations: int,
    plan_deviations: int,
    known_fix_deviations: int,
    critic_patches: int,
    assumption_evidence_ids: Sequence[str],
) -> bool:
    return (
        plan_faithful
        and env_deviations == 0
        and plan_deviations == 0
        and known_fix_deviations == 0
        and critic_patches == 0
        and not assumption_evidence_ids
    )


def plan_deviation_count(p: PlannedClaim) -> int:
    return len(p.deviations) + sum(1 for n in p.notes if n.status == "deviation")


def known_fix_deviation_count(evidence: Iterable[Evidence], source: str) -> int:
    return sum(
        1
        for ev in evidence
        if ev.kind == "known_fix"
        and ev.source == source
        and isinstance(ev.content.get("action"), Mapping)
        and ev.content["action"].get("deviation")
    )


def assumption_ids(
    p: PlannedClaim,
    events: Iterable[Any],
    states: Mapping[str, UnknownState],
    env_plan_id: str | None,
) -> tuple[str, ...]:
    """Evidence ids of the Assumption events this run depends on, for unknowns still open:
    the plan's (evidence = its run_plan) and the env plan's. A plan assumption without a
    stored event falls back to the evidence its value came from."""
    open_ids = {u for u, st in states.items() if st.status == "open"}
    deps = set(p.plan.depends_on_unknowns) & open_ids
    plan_id = p.evidence.evidence_id
    out: list[str] = []
    covered: set[str] = set()
    for e in events:
        if not isinstance(e, Assumption) or e.unknown_id not in open_ids:
            continue
        if e.evidence_id == plan_id and e.unknown_id in deps:
            out.append(e.evidence_id)
            covered.add(e.unknown_id)
        elif env_plan_id is not None and e.evidence_id == env_plan_id:
            out.append(e.evidence_id)
    out += [a.evidence_id for a in p.assumptions if a.unknown_id in deps - covered]
    return tuple(sorted(set(out)))


# --- metric -> ExperimentResult (pure) -------------------------------------------------------


def source_direction(conversion: str, claim_metric: str) -> Literal["max", "min"]:
    """Direction of the PRINTED metric (for which="best")."""
    if conversion.startswith("accuracy_"):
        return "max"
    try:
        return metric_direction(claim_metric)
    except ValueError:
        return "max"


def raw_metric_name(metric_lines: Iterable[str], pattern: str) -> str:
    for line in metric_lines:
        try:
            mp = parse_metric_line(line)
        except ValueError:
            continue
        if mp.pattern == pattern:
            return mp.name
    m = re.search(r"([A-Za-z_]\w*)", pattern.replace(r"(?<!\w)", ""))
    return m.group(1) if m else "metric"


@dataclass(frozen=True)
class MetricSpec:
    claim_metric: str
    pattern: str
    which: Literal["last", "best"]
    conversion: str
    raw_name: str


@dataclass(frozen=True)
class MetricReading:
    raw: float | None
    value: float | None  # converted, only for a completed run
    note: str


def read_metric(log_text: str, ms: MetricSpec, *, exit_code: int, timed_out: bool) -> MetricReading:
    raw = extract_metric(
        log_text, ms.pattern, ms.which, source_direction(ms.conversion, ms.claim_metric)
    )
    if raw is None:
        return MetricReading(None, None, "no metric line in the log")
    if exit_code != 0 or timed_out:
        return MetricReading(raw, None, "run did not finish (exit != 0 or timeout)")
    try:
        value = convert(raw, ms.conversion)  # type: ignore[arg-type]
    except ValueError as e:
        return MetricReading(raw, None, f"conversion failed: {e}")
    return MetricReading(raw, value, "ok")


def experiment_result(
    *,
    claim_id: str,
    run_id: str,
    exit_code: int | None,
    timed_out: bool,
    log_text: str,
    ms: MetricSpec,
    log_ref: str,
    config_digest: str,
    duration_s: float,
) -> tuple[ExperimentResult, MetricReading]:
    code = -1 if exit_code is None else exit_code
    if timed_out and code == 0:
        code = -1
    reading = read_metric(log_text, ms, exit_code=code, timed_out=timed_out)
    metrics: dict[str, float] = {}
    if reading.raw is not None:
        metrics[f"raw:{ms.raw_name}"] = reading.raw
    if reading.value is not None:
        metrics[ms.claim_metric] = reading.value
    duration = max(0.0, duration_s)
    return (
        ExperimentResult(
            claim_id=claim_id,
            run_id=run_id,
            exit_code=code,
            metrics=metrics,
            log_ref=log_ref,
            config_digest=config_digest,
            duration=round(duration, 1),
            gpu_hours=round(duration / 3600.0, 4),
        ),
        reading,
    )


# --- execute (IO) ----------------------------------------------------------------------------


def copy_artifacts(
    workdir: Path, dest: Path, dirs: Iterable[str], *, no_checkpoints: bool
) -> list[str]:
    """Copy the repo's output dirs from the per-run copy to dest (symlinks copied as links,
    never followed). Returns the copied files, relative to dest."""
    ignore = shutil.ignore_patterns(*CHECKPOINT_PATTERNS) if no_checkpoints else None
    copied: list[str] = []
    for d in dict.fromkeys(dirs):
        src = workdir / d
        if src.is_symlink() or not src.is_dir():
            continue
        shutil.copytree(src, dest / d, symlinks=True, ignore=ignore, dirs_exist_ok=True)
    if dest.is_dir():
        copied = sorted(
            str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() or p.is_symlink()
        )
    return copied


_EPOCH_RE = re.compile(r"[Ee]poch\s*[:#]?\s*(\d+)")
_BAR_RE = re.compile(r"(\d+)/(\d+)\s*\[")


def progress_line(log_tail: str, pattern: str, epochs: int | None) -> str | None:
    """'epoch k/N' from the number of metric lines printed (one per epoch) and the last tqdm
    bar; None when nothing is visible yet."""
    done = len(re.findall(pattern, log_tail))
    chunks = log_tail.replace("\r", "\n").split("\n")
    epoch = next((m[1] for c in reversed(chunks) if (m := _EPOCH_RE.search(c))), None)
    bar = next((f"{m[1]}/{m[2]}" for c in reversed(chunks) if (m := _BAR_RE.search(c))), None)
    if done == 0 and epoch is None and bar is None:
        return None
    total = f"/{epochs}" if epochs else ""
    parts = [f"epoch {done}{total} done"]
    if epoch is not None:
        parts.append(f"current 'Epoch {epoch}'")
    if bar is not None:
        parts.append(f"bar {bar}")
    return ", ".join(parts)


class _Progress:
    """Polls the run log and prints progress lines to `emit` when they change."""

    def __init__(
        self, log_path: Path, pattern: str, epochs: int | None, emit: Callable[[str], None],
        every_s: float,
    ) -> None:  # fmt: skip
        self.log_path, self.pattern, self.epochs = log_path, pattern, epochs
        self.emit, self.every_s = emit, every_s
        self._stop = threading.Event()
        self._last: str | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _poll(self) -> None:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(0)
                text = f.read(size).decode("utf-8", errors="replace")
        except OSError:
            return
        line = progress_line(text, self.pattern, self.epochs)
        if line is not None and line != self._last:
            self._last = line
            self.emit(line)

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self._poll()

    def __enter__(self) -> _Progress:
        if self.every_s > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


@dataclass(frozen=True)
class RunJob:
    """Everything one full run needs (built from the plan + the pre-flight state)."""

    claim_id: str
    source: str
    image: str
    clone_dir: Path
    argv: tuple[str, ...]
    patches: tuple[str, ...]
    prep_dirs: tuple[str, ...]
    data_mounts: tuple[tuple[str, str], ...]
    metric: MetricSpec
    run_digest: str
    timeout_s: float
    planned_epochs: int | None
    spec_faithful: bool
    assumption_evidence_ids: tuple[str, ...]
    gpu: bool = True
    mem: str = "12g"
    links: tuple[str, ...] = ()  # run_result links besides the assumptions (plan, preflight)


@dataclass(frozen=True)
class RunReport:
    result: ExperimentResult
    reading: MetricReading
    evidence: Evidence
    run_dir: Path
    artifacts: tuple[str, ...]
    timed_out: bool
    oom_killed: bool

    @property
    def completed(self) -> bool:
        return is_completed(self.evidence.content, self.result_metric)

    @property
    def result_metric(self) -> str:
        return next((k for k in self.result.metrics if not k.startswith("raw:")), "")


def execute_job(
    job: RunJob,
    out: Path,
    store: RunStore,
    *,
    runner: Runner,
    no_checkpoints: bool = False,
    progress: Callable[[str], None] = lambda s: None,
    progress_every_s: float = PROGRESS_EVERY_S,
    image_exists: Callable[[str], bool] | None = None,
) -> RunReport:
    """Run the job in the sandbox, keep artifacts, write run_result evidence. If anything
    (Ctrl-C included) interrupts it before that evidence is written: once the container was
    started an aborted run_result, before that a setup_failed note; then it is re-raised.
    image_exists (optional): checked before the container starts (RunSetupError if not)."""
    run_dir = new_run_dir(out / "runs", f"run-{job.claim_id}")
    work = run_dir / "work"
    log_path = run_dir / "run.log"
    artifacts: list[str] = []
    res: RunResult | None = None
    written = False
    started = False  # the runner (container) was called
    stage: SetupStage = "prepare_workdir"
    error: BaseException | None = None
    start = 0.0  # set when the container starts
    try:
        try:
            prepare_workdir(job.clone_dir, job.patches, work, job.prep_dirs)
            stage = "image_check"
            if image_exists is not None and not image_exists(job.image):
                raise RunSetupError(f"image {job.image} not found")
            with _Progress(log_path, job.metric.pattern, job.planned_epochs,
                           lambda s: progress(f"[{job.claim_id}] {s}"),
                           progress_every_s):  # fmt: skip
                start = time.monotonic()
                started = True
                res = runner(
                    job.image,
                    list(job.argv),
                    workdir=work,
                    out_dir=run_dir / "out",
                    data_mounts=list(job.data_mounts),
                    gpu=job.gpu,
                    timeout_s=job.timeout_s,
                    mem=job.mem,
                    log_path=log_path,
                    name=f"reproducer-{run_dir.name}",
                )
            artifacts = copy_artifacts(
                work, run_dir / "artifacts", job.prep_dirs, no_checkpoints=no_checkpoints
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        result, reading = experiment_result(
            claim_id=job.claim_id,
            run_id=run_dir.name,
            exit_code=res.exit_code,
            timed_out=res.timed_out,
            log_text=text,
            ms=job.metric,
            log_ref=make_ref(out, log_path),
            config_digest=job.run_digest,
            duration_s=res.duration_s,
        )
        ev = make_run_evidence(
            result,
            spec_faithful=job.spec_faithful,
            assumption_evidence_ids=job.assumption_evidence_ids,
            source=job.source,
            links=job.links,
            artifacts=[make_ref(out, run_dir / "artifacts" / a) for a in artifacts],
        )
        store.append_evidence(ev)
        written = True
        _record_curve(job, out, store, ev, progress)
    except BaseException as e:
        error = e
        raise
    finally:
        if not written and started:
            elapsed = res.duration_s if res is not None else time.monotonic() - start
            _record_aborted(job, out, store, run_dir, log_path, elapsed, error, progress)
        elif not written:
            _record_setup_failed(job, store, run_dir, stage, error, progress)
    return RunReport(
        result=result, reading=reading, evidence=ev, run_dir=run_dir,
        artifacts=tuple(artifacts), timed_out=res.timed_out, oom_killed=res.oom_killed,
    )  # fmt: skip


def _record_curve(
    job: RunJob, out: Path, store: RunStore, run: Evidence, progress: Callable[[str], None]
) -> None:
    """run_curve evidence for a run that exited 0, or a partial one for a crashed / diverged
    run (diagnose.py wants_curve); a failure is reported, never raised (the run_result is
    already stored; `diagnose` backfills it)."""
    if not wants_curve(run):
        return
    ms = job.metric
    spec = CurveSpec(claim_metric=ms.claim_metric, pattern=ms.pattern,
                     conversion=ms.conversion, raw_name=ms.raw_name)  # fmt: skip
    try:
        ev = record_run_curve(out, store, run, spec, {})
    except (ValueError, OSError) as e:
        progress(f"[{job.claim_id}] no run_curve: {e}")
        return
    if ev is not None:
        src = ev.content["summary"]["source"] + (", partial" if ev.content.get("partial") else "")
        progress(f"[{job.claim_id}] recorded run_curve {ev.evidence_id[:12]} (from {src})")


def _reason(error: BaseException | None) -> str:
    return "interrupted" if error is None else f"{type(error).__name__}: {error}".rstrip(": ")


def setup_failed_evidence(job: RunJob, run_id: str, stage: SetupStage, reason: str) -> Evidence:
    """setup_failed note: the full run never started (not a run_result, no attempt, 0 GPU h)."""
    return make_evidence(
        kind="setup_failed",
        source=job.source,
        locator=run_id,
        content={
            "claim_id": job.claim_id,
            "run_id": run_id,
            "stage": stage,
            "reason": reason,
            "config_digest": job.run_digest,
            "image": job.image,
        },
        confidence="observed",
        authority="execution",
        links=job.links,
    )


def _record_setup_failed(
    job: RunJob,
    store: RunStore,
    run_dir: Path,
    stage: SetupStage,
    error: BaseException | None,
    progress: Callable[[str], None],
) -> None:
    """Write the setup_failed note; a failure here is reported, never raised."""
    reason = _reason(error)
    try:
        ev = setup_failed_evidence(job, run_dir.name, stage, reason)
        store.append_evidence(ev)
        progress(f"[{job.claim_id}] full run not started ({stage}: {reason}); recorded "
                 f"setup_failed {ev.evidence_id[:12]} (no attempt, 0 GPU h)")  # fmt: skip
    except Exception as e:  # noqa: BLE001 - must not mask the original error
        progress(f"[{job.claim_id}] full run not started ({stage}: {reason}); "
                 f"could not record it: {e}")  # fmt: skip


def _record_aborted(
    job: RunJob,
    out: Path,
    store: RunStore,
    run_dir: Path,
    log_path: Path,
    duration_s: float,
    error: BaseException | None,
    progress: Callable[[str], None],
) -> None:
    """Write the aborted run_result; a failure here is reported, never raised (the original
    interruption is what propagates)."""
    reason = _reason(error)
    try:
        ev = make_aborted_run_evidence(
            claim_id=job.claim_id,
            run_id=run_dir.name,
            log_ref=make_ref(out, log_path),
            config_digest=job.run_digest,
            duration_s=duration_s,
            reason=reason,
            spec_faithful=job.spec_faithful,
            assumption_evidence_ids=job.assumption_evidence_ids,
            source=job.source,
            links=job.links,
        )
        store.append_evidence(ev)
        progress(f"[{job.claim_id}] aborted ({reason}); recorded run_result "
                 f"{ev.evidence_id[:12]} with {duration_s / 3600:.3f} GPU h")  # fmt: skip
    except Exception as e:  # noqa: BLE001 - must not mask the interruption
        progress(f"[{job.claim_id}] aborted ({reason}); could not record it: {e}")


# --- pre-flight + run per claim (IO) ---------------------------------------------------------


@dataclass(frozen=True)
class Preflight:
    smoke_passed: bool
    smoke_reused: bool
    smoke_evidence_id: str | None
    smoke_stop: str
    sec_per_epoch: float | None
    timing_method: str
    epochs: int | None
    overhead_s: float
    estimate_h: float | None
    caps: Caps
    decision: Decision
    reason: str
    timeout_s: float | None
    evidence: Evidence


@dataclass(frozen=True)
class ClaimRun:
    claim_id: str
    command: str  # the full command (with known-fix overrides)
    image: str
    run_digest: str
    status: Literal[
        "already_completed", "preflight_only", "skipped", "ran",
        "awaiting_approval",  # an approval_request waits for a human (nothing executed)
        "rejected",  # a human rejected the full run: the claim stays open
    ]  # fmt: skip
    preflight: Preflight | None = None
    report: RunReport | None = None
    existing: Evidence | None = None  # completed run_result (already_completed)
    spec_faithful: bool | None = None
    decimals: int = 2  # display precision of the claim metric (the paper's), printouts only
    approval: GateResult | None = None  # the full_run / critic gate, when one applied


def _stored_smoke(evidence: Mapping[str, Evidence], source: str, digest: str) -> Evidence | None:
    hit = None
    for ev in evidence.values():
        c = ev.content
        if (
            ev.kind == "smoke_result" and ev.source == source and c.get("input_digest") == digest
            and c.get("passed") is True and isinstance(c.get("sec_per_epoch"), (int, float))
        ):  # fmt: skip
            hit = ev
    return hit


def preflight_evidence(
    *,
    p: PlannedClaim,
    source: str,
    plan_digest: str,
    digest: str,
    argv: Sequence[str],
    image: str,
    pf: dict[str, Any],
    smoke_evidence_id: str | None,
) -> Evidence:
    links = [p.evidence.evidence_id] + ([smoke_evidence_id] if smoke_evidence_id else [])
    return make_evidence(
        kind="preflight",
        source=source,
        locator=f"preflight:{p.claim.claim_id}",
        content={
            "claim_id": p.claim.claim_id,
            "plan_evidence_id": p.evidence.evidence_id,
            "plan_input_digest": plan_digest,
            "run_digest": digest,
            "command": shlex.join(argv),
            "image": image,
            **pf,
        },
        confidence="observed",
        authority="execution",
        links=links,
    )


def full_run_request(
    p: PlannedClaim, *, source: str, digest: str, command: str, image: str, estimate_h: float,
    preflight_id: str, reason: str = "",
) -> Evidence:  # fmt: skip
    """The approval_request a gated full run files (input digest: claim + run digest)."""
    cid = p.claim.claim_id
    return approval_request_evidence(
        action="full_run",
        subject=cid,
        cost={"gpu_h": round(estimate_h, 3)},
        detail={"command": command, "image": image, "run_digest": digest},
        input_digest=request_digest("full_run", {"claim_id": cid, "run_digest": digest}),
        source=source,
        links=(preflight_id,),
        reason=reason,
    )


def full_run_gate(
    store: RunStore, request: Evidence, policy: ApprovalPolicy, estimate_h: float
) -> GateResult:
    """gate() + file the request when it is new (same digest -> the stored request)."""
    g = gate(store.load_evidence(), request, policy, estimate_h)
    if g.new_request and g.request is not None:
        filed = replace_reason(request, g.reason)
        store.append_evidence(filed)
        g = replace(g, request=filed)
    return g


def replace_reason(request: Evidence, reason: str) -> Evidence:
    c = request.content
    return approval_request_evidence(
        action=c["action"], subject=c["subject"], cost=c["cost"], detail=c["detail"],
        input_digest=c["input_digest"], source=request.source, links=request.links,
        reason=reason,
    )  # fmt: skip


def run_claims(
    out: Path,
    store: RunStore,
    deps: Mapping[str, Any],
    *,
    claim_id: str | None = None,
    preflight_only: bool = False,
    no_checkpoints: bool = False,
    partial_plans: bool = False,
    log: Callable[[str], None] = lambda s: None,
    progress: Callable[[str], None] = lambda s: None,
) -> list[ClaimRun]:
    """Pre-flight (+ run) every current plan, or only claim_id. deps: runner, plan_env,
    image_exists, build_image, unload, make_critic(SmokeSetup) -> CriticFn | None, env
    (caps), smoke_timeout_s, build_timeout_s. Raises ValueError when there is nothing to
    plan from or the claim is unknown. partial_plans: claims that do not plan are left out
    instead of failing every claim (load_current_plans(partial=True))."""
    current = load_current_plans(store, repo_dir=out / "repo", partial=partial_plans)
    plans = [p for p in current.plans if claim_id is None or p.claim.claim_id == claim_id]
    if not plans:
        known = [p.claim.claim_id for p in current.plans]
        raise ValueError(f"no current plan for claim {claim_id!r}; plans: {known}")
    spec = current.target.argparse_spec
    env: Mapping[str, str] = deps.get("env", os.environ)
    caps_from_env(env, 0.0)  # fail on a bad cap before any run
    policy = policy_from_env(env)  # and on a bad approval policy
    deps["unload"]()
    out_runs: list[ClaimRun] = []
    for p in plans:
        cid = p.claim.claim_id
        setup0 = prepare_smoke(out, store, p.plan.entrypoint)
        setup: SmokeSetup = replace(setup0, example=p.command)
        build = make_build(
            setup, out, store, plan_env=deps["plan_env"], image_exists=deps["image_exists"],
            builder=deps["build_image"], timeout_s=float(deps.get("smoke_timeout_s", 1200)),
            build_timeout=float(deps.get("build_timeout_s", 3600)), log=log,
        )  # fmt: skip
        state, cstate = setup.fix_state0, setup.critic_state0
        patches = tuple(a.diff for a in cstate.stack)
        inp = build(state)
        argv = full_argv(p.command, state)
        digest = run_digest(image=inp.image, state=state, patches=patches, argv=argv)
        evidence = store.load_evidence()
        dec = metric_decimals(p.claim)
        done = completed_run(evidence.values(), digest, p.claim.metric)
        if done is not None:
            out_runs.append(ClaimRun(cid, shlex.join(argv), inp.image, digest,
                                     "already_completed", existing=done,
                                     decimals=dec))  # fmt: skip
            continue

        # 1. pre-flight smoke on the plan's command (budget overrides)
        smoke_cmd = plan_smoke(p.command, inp.arguments, state.overrides).command
        smoke_digest = smoke_input_digest(
            image=inp.image, state=state, patches=patches, command=smoke_cmd
        )
        reused = _stored_smoke(evidence, setup.source, smoke_digest)
        smoke_id: str | None
        if reused is not None:
            c = reused.content
            passed, stop = True, "passed"
            sec, method = float(c["sec_per_epoch"]), str(c["timing"]["method"])
            wall = float(c.get("duration_s") or 0.0)
            smoke_id = reused.evidence_id
            image = inp.image
        else:
            recs: list[str] = []

            def record(it: Iteration, setup: SmokeSetup = setup, recs: list[str] = recs) -> None:
                rec = record_iteration(store, out, setup, it)
                recs.append(rec.smoke.evidence_id)
                r = it.result
                log(f"  smoke:        {r.signal} passed={r.passed} "
                    f"duration={r.duration_s:.1f}s evidence={rec.smoke.evidence_id[:12]}"
                    + (f" fix={it.action.kind} {it.action.params}" if it.action else "")
                    + (f" critic={it.critic.status}" if it.critic else ""))  # fmt: skip

            outcome = smoke_loop(
                build,
                setup.ctx,
                state=state,
                auto_fix=True,
                max_fixes=max(0, MAX_KNOWN_FIXES - setup.known_fixes0),
                record=record,
                runner=deps["runner"],
                critic=deps["make_critic"](setup),
                critic_state=cstate,
            )
            last = outcome.iterations[-1]
            passed, stop = last.result.passed, outcome.stop
            sec = last.result.timing.sec_per_epoch
            method = last.result.timing.method
            wall = last.result.duration_s
            smoke_id = recs[-1] if recs else None
            state, cstate, image = outcome.state, outcome.critic_state, last.image
            patches = tuple(a.diff for a in cstate.stack)
            argv = full_argv(p.command, state)
            digest = run_digest(image=image, state=state, patches=patches, argv=argv)

        # 2. estimate + caps
        evidence = store.load_evidence()
        epochs = planned_epochs([(a.flag, a.value) for a in p.plan.args], spec)
        overhead = 0.0
        estimate: float | None = None
        if passed and sec is not None and epochs is not None:
            overhead = eval_overhead_s(
                sec_per_epoch=sec, smoke_duration_s=wall, timing_method=method, epochs=epochs
            )
            estimate = estimate_gpu_hours(sec, epochs, overhead)
        caps = caps_from_env(env, spent_gpu_hours(evidence.values()))
        decision, reason = decide(passed, estimate, caps)
        if not passed:
            reason += f" ({stop})"
        timeout = run_timeout_s(estimate, caps.run_h) if estimate is not None else None
        pf_content = {
            "sec_per_epoch": sec,
            "timing_method": method,
            "smoke_wall_s": round(wall, 1),
            "planned_epochs": epochs,
            "eval_overhead_s": round(overhead, 1),
            "estimate_h": None if estimate is None else round(estimate, 3),
            "caps": caps.as_content(),
            "timeout_s": None if timeout is None else round(timeout),
            "decision": decision,
            "reason": reason,
            "smoke_reused": reused is not None,
        }
        pf_ev = preflight_evidence(
            p=p, source=setup.source, plan_digest=current.input_digest, digest=digest,
            argv=argv, image=image, pf=pf_content, smoke_evidence_id=smoke_id,
        )  # fmt: skip
        store.append_evidence(pf_ev)
        pf = Preflight(
            smoke_passed=passed, smoke_reused=reused is not None, smoke_evidence_id=smoke_id,
            smoke_stop=stop, sec_per_epoch=sec, timing_method=method, epochs=epochs,
            overhead_s=overhead, estimate_h=estimate, caps=caps, decision=decision,
            reason=reason, timeout_s=timeout, evidence=pf_ev,
        )  # fmt: skip

        # fidelity inputs
        events = store.load_events()
        states = fold_unknowns(events, evidence)
        pe = deps["plan_env"](out, False)
        env_ev = env_plan_to_evidence(
            pe.plan, pe.spec_evidence.source, links=(pe.spec_evidence.evidence_id,), gpu=pe.gpu
        )
        a_ids = assumption_ids(p, events, states, env_ev.evidence_id)
        faithful = is_spec_faithful(
            plan_faithful=p.plan.is_spec_faithful(states),
            env_deviations=len(pe.plan.deviations),
            plan_deviations=plan_deviation_count(p),
            known_fix_deviations=known_fix_deviation_count(evidence.values(), setup.source),
            critic_patches=len(cstate.stack),
            assumption_evidence_ids=a_ids,
        )
        command = shlex.join(argv)
        done = completed_run(evidence.values(), digest, p.claim.metric)
        if done is not None:  # the smoke led to a state that already has a completed run
            out_runs.append(ClaimRun(cid, command, image, digest, "already_completed", pf,
                                     existing=done, spec_faithful=faithful,
                                     decimals=dec))  # fmt: skip
            continue
        if not passed and stop == "awaiting_approval":  # a pre-flight critic patch waits
            out_runs.append(ClaimRun(cid, command, image, digest, "awaiting_approval", pf,
                                     spec_faithful=faithful, decimals=dec,
                                     approval=GateResult("pending", reason)))  # fmt: skip
            continue
        if preflight_only or decision != "run":
            status: Literal["preflight_only", "skipped"] = (
                "skipped" if decision != "run" else "preflight_only"
            )
            out_runs.append(ClaimRun(cid, command, image, digest, status, pf,
                                     spec_faithful=faithful, decimals=dec))  # fmt: skip
            continue

        # 2b. human approval of the full run
        assert estimate is not None
        request = full_run_request(p, source=setup.source, digest=digest, command=command,
                                   image=image, estimate_h=estimate,
                                   preflight_id=pf_ev.evidence_id)  # fmt: skip
        g = full_run_gate(store, request, policy, estimate)
        if g.status in ("pending", "rejected"):
            st: Literal["awaiting_approval", "rejected"] = (
                "awaiting_approval" if g.status == "pending" else "rejected"
            )
            out_runs.append(ClaimRun(cid, command, image, digest, st, pf,
                                     spec_faithful=faithful, decimals=dec,
                                     approval=g))  # fmt: skip
            continue

        # 3. execute
        assert timeout is not None and sec is not None
        ms = p.plan.metric_source
        metric_lines = [
            str(r.get("line"))
            for r in evidence[ms.metric_line_evidence_id].content.get("result", [])
            if isinstance(r, Mapping)
        ]
        job = RunJob(
            claim_id=cid,
            source=setup.source,
            image=image,
            clone_dir=setup.repo_dir,
            argv=argv,
            patches=patches,
            prep_dirs=state.prep_dirs,
            data_mounts=tuple(
                (*setup.base_mounts, *((setup.cache, m.container) for m in state.mounts))
            ),
            metric=MetricSpec(
                claim_metric=p.claim.metric,
                pattern=ms.pattern,
                which=ms.which,
                conversion=ms.conversion,
                raw_name=raw_metric_name(metric_lines, ms.pattern),
            ),
            run_digest=digest,
            timeout_s=timeout,
            planned_epochs=epochs,
            spec_faithful=faithful,
            assumption_evidence_ids=a_ids,
            links=(p.evidence.evidence_id, pf_ev.evidence_id),
        )
        deps["unload"]()
        log(f"  running:      {command} (timeout {timeout / 3600:.2f} h)")
        report = execute_job(
            job, out, store, runner=deps["runner"], no_checkpoints=no_checkpoints,
            progress=progress, image_exists=deps["image_exists"],
        )  # fmt: skip
        out_runs.append(ClaimRun(cid, command, image, digest, "ran", pf, report=report,
                                 spec_faithful=faithful, decimals=dec))  # fmt: skip
    return out_runs
