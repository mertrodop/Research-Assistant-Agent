"""Smoke test: the first execution of repo code, sandboxed. Deterministic, no LLM.

Three stages, each a separate locked-down container (docker_runner.run_container: network
none, read-only root, host uid, caps dropped), stopping at the first failure:
  1. probe   - our GPU probe (cuInit + matmul) in an EMPTY workdir: no repo code.
               Fail -> signal "infra".
  2. imports - import the third-party modules from the import scan and the local modules the
               entrypoint imports (never the entrypoint itself), /work on sys.path, in a fresh
               copy of the clone. Fail -> signal "import_error" with the parsed traceback.
  3. run     - the smoke command (verified example + budget overrides) in a fresh copy of the
               clone, data mounted read-only, GPU on, with a timeout. The log is analysed for
               progress and the last traceback; domain.smoke_plan.decide_signal decides.
Workdirs are removed afterwards (logs kept).

smoke_loop wraps run_smoke for --auto-fix as a loop over the same steps the setup graph runs
as nodes (smoke_step -> routing.decide_smoke -> apply_fix | critic_step): classify
(rules/classify.py, code only) -> known: apply the zero-token fix (rules/known_fixes.py) and
re-smoke; infra: stop; critic: with a critic (graph/nodes/critic.py) one validated patch per
attempt, pushed on the patch stack and re-smoked (max_critic LLM attempts, separate from
max_fixes). If the re-smoke after a patch keeps the error signature, the patch is reverted
(Iteration.reverted -> critic_revert evidence) and the critic is asked again with its diff;
otherwise a repeated signature = no new evidence -> stop. Without a critic: stop. unfixable,
a repeated fix (loop guard), max_fixes / max_critic reached, an invalid or skipped critic:
stop with capped=True. Patches are applied to each per-run copy in order (prepare_workdir),
never to the clone.

The second half (prepare_smoke, make_build, record_iteration, smoke_node) reads the stored
facts for a smoke run and persists each iteration; the CLI `smoke` command and the setup
graph share it.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reproducer.domain.data_plan import DataMountPlan
from reproducer.domain.evidence import Evidence
from reproducer.domain.import_scan import local_import_modules
from reproducer.domain.ledger import make_smoke_evidence
from reproducer.domain.patching import diff_sha256, error_signature
from reproducer.domain.smoke_plan import (
    PASSING_SIGNALS,
    BudgetArg,
    Override,
    Signal,
    SmokePlan,
    Timing,
    decide_signal,
    estimate_timing,
    metric_prefixes,
    plan_smoke,
    progress_seen,
)
from reproducer.domain.traceback_parse import ParsedTraceback, parse_last_traceback
from reproducer.graph.nodes.critic import (
    MAX_CRITIC_ATTEMPTS,
    AppliedPatch,
    CriticOutcome,
    CriticState,
    critic_patch_evidence,
    critic_revert_evidence,
    critic_trace_evidence,
    fold_critic_state,
)
from reproducer.graph.nodes.data import dataset_roots, latest_repo_evidence, stored_mount_plan
from reproducer.graph.nodes.env import PlanEnvFn, build_env_image, prepare_env
from reproducer.graph.nodes.subtasks import candidate_entrypoints
from reproducer.graph.routing import (
    MAX_KNOWN_FIXES,
    LastSmoke,
    SmokeDecision,
    StopReason,
    decide_smoke,
)
from reproducer.providers.datasets import data_cache_dir
from reproducer.providers.docker_runner import (
    BuildResult,
    DataMounts,
    RunResult,
    new_run_dir,
    prepare_workdir,
    run_container,
)
from reproducer.providers.git import list_files, read_text
from reproducer.providers.store import RunStore, make_ref
from reproducer.rules.classify import Classification, ClassifyContext, classify
from reproducer.rules.known_fixes import (
    FixAction,
    FixState,
    apply_fix,
    count_attempts,
    fix_for,
    fold_fix_state,
    known_fix_evidence,
    smoke_input_digest,
)

__all__ = ["StopReason"]  # re-exported: defined in graph/routing.py

DEFAULT_TIMEOUT_S = 20 * 60.0
IMPORT_TIMEOUT_S = 300.0
PROBE_TIMEOUT_S = 300.0
PROBE_OK = "REPRODUCER_PROBE_OK"
GPU_PROBE_CODE = (
    "import torch; torch.cuda.init(); x = torch.ones(256, 256, device='cuda'); "
    f"print({PROBE_OK!r}, torch.__version__, torch.cuda.get_device_name(0), "
    "float((x @ x).sum()))"
)
GPU_PROBE_CMD: tuple[str, ...] = ("python", "-c", GPU_PROBE_CODE)
IMPORTS_MARKER = "REPRODUCER_IMPORTS "
IMPORT_SCRIPT = f"""\
import importlib, json, sys, traceback
sys.path.insert(0, '/work')
mods = json.loads(sys.argv[1])
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except BaseException:
        traceback.print_exc()
        failed.append(m)
sys.stdout.flush()
print({IMPORTS_MARKER!r} + json.dumps({{'n': len(mods), 'failed': failed}}))
sys.exit(1 if failed else 0)
"""

Runner = Callable[..., RunResult]


@dataclass(frozen=True)
class SmokeInputs:
    image: str
    clone_dir: Path
    example: str  # the verified example command line
    arguments: tuple[BudgetArg, ...]  # argparse spec of the entrypoint
    third_party: tuple[str, ...]  # top-level modules from the import scan
    local_modules: tuple[str, ...]  # local modules the entrypoint imports (not itself)
    run_root: Path  # per-run dirs are created under it
    data_mounts: DataMounts = field(default_factory=dict)  # host -> container (ro), or pairs
    metric_prefixes: tuple[str, ...] = ()
    prep_dirs: tuple[str, ...] = ()  # known fix: dirs created in each per-run copy
    fix_overrides: tuple[Override, ...] = ()  # known fix: command overrides (after budget)
    patches: tuple[str, ...] = ()  # critic patch stack: unified diffs applied in order
    gpu: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    probe_cmd: tuple[str, ...] = GPU_PROBE_CMD
    mem: str = "12g"


@dataclass(frozen=True)
class SmokeResult:
    passed: bool
    signal: Signal
    stage: str  # probe | imports | run
    plan: SmokePlan
    duration_s: float  # of the repo run (0 if it never ran)
    timing: Timing
    progress_seen: bool
    traceback: ParsedTraceback | None
    exit_code: int | None
    timed_out: bool
    oom_killed: bool
    log_path: Path
    run_dir: Path
    stage_logs: dict[str, Path]
    patches_applied: tuple[str, ...] = ()  # patch ids (sha256 of each diff), in order

    def details(
        self, *, attempt: int, log_ref: str, image: str, input_digest: str | None = None
    ) -> dict[str, Any]:
        """Additive smoke_result evidence fields (JSON-compatible)."""
        return {
            "input_digest": input_digest,
            "signal": self.signal,
            "stage": self.stage,
            "command": self.plan.command,
            "overrides": [o.model_dump(mode="json") for o in self.plan.overrides],
            "duration_s": round(self.duration_s, 1),
            "sec_per_epoch": self.timing.sec_per_epoch,
            "timing": {
                "method": self.timing.method,
                "confidence": self.timing.confidence,
                "detail": self.timing.detail,
            },
            "progress_seen": self.progress_seen,
            "traceback": None if self.traceback is None else self.traceback.model_dump(mode="json"),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "log_ref": log_ref,
            "attempt": attempt,
            "image": image,
            "patches_applied": list(self.patches_applied),
        }


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def run_smoke(inp: SmokeInputs, *, runner: Runner = run_container) -> SmokeResult:
    plan = plan_smoke(inp.example, inp.arguments, inp.fix_overrides)  # ValueError: nothing ran
    run_dir = new_run_dir(inp.run_root, "smoke")
    out_dir = run_dir / "out"
    logs: dict[str, Path] = {}
    no_timing = Timing(sec_per_epoch=None, method="none", confidence="none")
    patch_ids = tuple(diff_sha256(d) for d in inp.patches)

    def container(stage: str, cmd: Sequence[str], workdir: Path, timeout: float, **kw: Any):  # type: ignore[no-untyped-def]
        logs[stage] = run_dir / f"{stage}.log"
        try:
            return runner(
                inp.image,
                list(cmd),
                workdir=workdir,
                out_dir=out_dir,
                gpu=inp.gpu,
                timeout_s=timeout,
                mem=inp.mem,
                log_path=logs[stage],
                name=f"reproducer-{run_dir.name}-{stage}",
                **kw,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def early(
        stage: str, signal: Signal, res: RunResult, tb: ParsedTraceback | None
    ) -> SmokeResult:
        return SmokeResult(
            passed=False, signal=signal, stage=stage, plan=plan, duration_s=0.0,
            timing=no_timing, progress_seen=False, traceback=tb, exit_code=res.exit_code,
            timed_out=res.timed_out, oom_killed=res.oom_killed, log_path=logs[stage],
            run_dir=run_dir, stage_logs=dict(logs), patches_applied=patch_ids,
        )  # fmt: skip

    # 1. infra probe: empty workdir, our code only
    probe_work = run_dir / "probe-work"
    probe_work.mkdir()
    res = container("probe", inp.probe_cmd, probe_work, min(PROBE_TIMEOUT_S, inp.timeout_s))
    if res.exit_code != 0 or PROBE_OK not in _read(logs["probe"]):
        tb = parse_last_traceback(_read(logs["probe"]), None if res.timed_out else res.exit_code)
        return early("probe", "infra", res, tb)

    # 2. imports: third-party + local modules, never the entrypoint
    modules = list(dict.fromkeys((*inp.third_party, *inp.local_modules)))
    work = prepare_workdir(inp.clone_dir, inp.patches, run_dir / "imports-work", inp.prep_dirs)
    res = container(
        "imports",
        ["python", "-c", IMPORT_SCRIPT, json.dumps(modules)],
        work,
        min(IMPORT_TIMEOUT_S, inp.timeout_s),
        data_mounts=inp.data_mounts,
    )
    if res.exit_code != 0 or res.timed_out:
        text = _read(logs["imports"])
        tb = parse_last_traceback(text, None if res.timed_out else res.exit_code)
        return early("imports", "import_error", res, tb)

    # 3. the repo run
    work = prepare_workdir(inp.clone_dir, inp.patches, run_dir / "run-work", inp.prep_dirs)
    start = time.monotonic()
    res = container("run", plan.argv, work, inp.timeout_s, data_mounts=inp.data_mounts)
    duration = res.duration_s or (time.monotonic() - start)
    text = _read(logs["run"])
    tb = parse_last_traceback(text, None if res.timed_out else res.exit_code)
    progress = progress_seen(text, inp.metric_prefixes)
    signal = decide_signal(
        exit_code=res.exit_code,
        timed_out=res.timed_out,
        oom_killed=res.oom_killed,
        progress=progress,
        traceback=tb,
    )
    timing = estimate_timing(
        text, overrides=plan.overrides, exit_code=res.exit_code, duration_s=duration
    )
    return SmokeResult(
        passed=signal in PASSING_SIGNALS,
        signal=signal,
        stage="run",
        plan=plan,
        duration_s=duration,
        timing=timing if progress or res.exit_code == 0 else no_timing,
        progress_seen=progress,
        traceback=None if signal in PASSING_SIGNALS else tb,
        exit_code=res.exit_code,
        timed_out=res.timed_out,
        oom_killed=res.oom_killed,
        log_path=logs["run"],
        run_dir=run_dir,
        stage_logs=dict(logs),
        patches_applied=patch_ids,
    )


# --- auto-fix loop ---------------------------------------------------------------------------

# (traceback, command, patch stack, last critic signature, reverted patches for this signature
#  when a revert just happened else ()) -> outcome
CriticFn = Callable[
    [ParsedTraceback, str, tuple[AppliedPatch, ...], str | None, tuple[AppliedPatch, ...]],
    CriticOutcome,
]


@dataclass(frozen=True)
class Iteration:
    result: SmokeResult
    image: str
    state: FixState  # the fix state this run was made with
    input_digest: str
    classification: Classification | None  # None when passed
    action: FixAction | None  # the fix applied after this run, if any
    capped: bool  # set only when the loop stops for good on this run
    critic: CriticOutcome | None = None  # the critic attempt made after this run, if any
    reverted: AppliedPatch | None = None  # patch dropped: this run kept its error signature


@dataclass(frozen=True)
class LoopOutcome:
    iterations: tuple[Iteration, ...]
    state: FixState  # final fix state
    stop: StopReason
    reason: str
    critic_state: CriticState = field(default_factory=CriticState)  # final patch stack


def failed_content(res: SmokeResult, image: str) -> dict[str, Any]:
    """The smoke evidence content shape that classify() reads."""
    return {"passed": res.passed, **res.details(attempt=0, log_ref="", image=image)}


@dataclass(frozen=True)
class SmokeStep:
    """One smoke run, classified and routed (routing.decide_smoke). iteration.capped is set
    iff the decision stops for good; action: the known fix to apply (route known); failed:
    reverted patches for this signature, for the critic prompt (route critic)."""

    iteration: Iteration
    critic_state: CriticState
    action: FixAction | None
    failed: tuple[AppliedPatch, ...]
    decision: SmokeDecision
    last: LastSmoke


def smoke_step(
    build: Callable[[FixState], SmokeInputs],
    ctx: ClassifyContext,
    state: FixState,
    cstate: CriticState,
    *,
    runner: Runner = run_container,
    auto_fix: bool = True,
    has_critic: bool = True,
    known_fixes_used: int = 0,
    max_fixes: int = MAX_KNOWN_FIXES,
    max_critic: int = MAX_CRITIC_ATTEMPTS,
) -> SmokeStep:
    """Run, classify (code), judge the pending critic patch (same signature after it -> revert)
    and decide. A known fix that would be a no-op becomes `unfixable` (loop guard)."""
    inp = replace(build(state), patches=tuple(p.diff for p in cstate.stack))
    res = run_smoke(inp, runner=runner)
    digest = smoke_input_digest(
        image=inp.image, state=state, patches=inp.patches, command=res.plan.command
    )
    c: Classification | None = None
    action: FixAction | None = None
    failed: tuple[AppliedPatch, ...] = ()
    reverted: AppliedPatch | None = None
    if not res.passed:
        mounted = ctx.mounted | {m.container for m in state.mounts}
        c = classify(failed_content(res, inp.image), ctx.model_copy(update={"mounted": mounted}))
        stops_here = c.cls == "infra" or (
            c.cls == "critic" and (not has_critic or res.traceback is None)
        )
        if auto_fix and not stops_here:
            pending = cstate.pending  # the patch made after the previous run, not yet judged
            cstate = replace(cstate, pending=None)
            if c.cls == "critic":
                assert res.traceback is not None
                sig = error_signature(res.traceback.model_dump(mode="json"))
                if pending is not None and sig == pending.error_signature:
                    # ineffective patch: drop it (the attempt stays counted) and show its diff
                    # to the critic (new evidence)
                    cstate, reverted = cstate.revert(pending.patch_id)
                    failed = cstate.failed_for(sig)
            elif c.cls == "known":
                action = fix_for(c, state)
                if action is None:
                    c = c.model_copy(
                        update={"cls": "unfixable", "reason": f"{c.reason}; fix already applied"}
                    )
    last = LastSmoke(
        passed=res.passed,
        signal=res.signal,
        cls=None if c is None else c.cls,
        reason="" if c is None else c.reason,
        has_traceback=res.traceback is not None,
    )
    decision = decide_smoke(
        last,
        known_fixes_used=known_fixes_used,
        critic_attempts=cstate.attempts,
        max_fixes=max_fixes,
        max_critic=max_critic,
        auto_fix=auto_fix,
        has_critic=has_critic,
    )
    it = Iteration(
        res, inp.image, state, digest, c, None, capped=decision.capped, reverted=reverted
    )
    return SmokeStep(it, cstate, action if decision.route == "known" else None, failed, decision,
                     last)  # fmt: skip


@dataclass(frozen=True)
class CriticStep:
    iteration: Iteration  # with critic set; capped iff stop
    critic_state: CriticState
    stop: StopReason | None  # None: patched, re-smoke
    reason: str


def critic_step(
    it: Iteration,
    cstate: CriticState,
    critic: CriticFn,
    failed: tuple[AppliedPatch, ...] = (),
    max_critic: int = MAX_CRITIC_ATTEMPTS,
) -> CriticStep:
    """One critic attempt for a `critic` iteration (cap checked by decide_smoke before).
    awaiting_approval: the patch waits for a human (not pushed, not capped: stop). A human
    rejection is a failed attempt (pushed); the critic is asked again with the rejection as
    new evidence while the cap allows."""
    res, c = it.result, it.classification
    assert res.traceback is not None and c is not None
    capped = replace(it, capped=True)
    out = critic(res.traceback, res.plan.command, cstate.stack, cstate.last_signature, failed)
    while out.status == "rejected":
        cstate = cstate.push(out)
        if cstate.attempts >= max_critic:
            return CriticStep(replace(capped, critic=out), cstate, "max_critic",
                              f"{max_critic} critic attempts used; {out.reason}")  # fmt: skip
        out = critic(res.traceback, res.plan.command, cstate.stack, cstate.last_signature, failed)
    if out.status == "awaiting_approval":
        return CriticStep(replace(it, critic=out), cstate, "awaiting_approval", out.reason)
    if out.status == "skipped":
        return CriticStep(replace(capped, critic=out), cstate, "no_new_evidence", out.reason)
    if out.status == "no_target":
        return CriticStep(
            replace(capped, critic=out), cstate, "unfixable", f"{c.reason}; {out.reason}"
        )
    cstate = cstate.push(out)
    if out.patch is None:
        return CriticStep(replace(capped, critic=out), cstate, "critic_failed", out.reason)
    return CriticStep(replace(it, critic=out), cstate, None, out.reason)


def smoke_loop(
    build: Callable[[FixState], SmokeInputs],
    ctx: ClassifyContext,
    *,
    state: FixState | None = None,
    auto_fix: bool = True,
    max_fixes: int = 5,
    record: Callable[[Iteration], None] | None = None,
    runner: Runner = run_container,
    critic: CriticFn | None = None,
    critic_state: CriticState | None = None,
    max_critic: int = MAX_CRITIC_ATTEMPTS,
) -> LoopOutcome:
    """Smoke until it passes or a stop condition; record(iteration) persists each run (and its
    fix or critic patch) before the next one starts. critic_state carries the patch stack,
    the last critic signature and the attempts made so far (from stored evidence).
    The setup graph runs the same steps (smoke_step / apply_fix / critic_step) as nodes."""
    state = state or FixState()
    cstate = critic_state or CriticState()
    iterations: list[Iteration] = []
    fixes = 0

    def keep(it: Iteration) -> None:
        iterations.append(it)
        if record is not None:
            record(it)

    while True:
        step = smoke_step(
            build, ctx, state, cstate, runner=runner, auto_fix=auto_fix,
            has_critic=critic is not None, known_fixes_used=fixes, max_fixes=max_fixes,
            max_critic=max_critic,
        )  # fmt: skip
        cstate, d = step.critic_state, step.decision
        if d.route == "known":
            assert step.action is not None
            keep(replace(step.iteration, action=step.action))
            state = apply_fix(state, step.action)
            fixes += 1
            continue
        final, stop, reason = step.iteration, d.stop, d.reason
        if d.route == "critic":
            assert critic is not None
            cs = critic_step(step.iteration, cstate, critic, step.failed, max_critic)
            cstate = cs.critic_state
            if cs.stop is None:
                keep(cs.iteration)
                continue
            final, stop, reason = cs.iteration, cs.stop, cs.reason
        assert stop is not None
        keep(final)
        return LoopOutcome(
            iterations=tuple(iterations), state=state, stop=stop, reason=reason,
            critic_state=cstate,
        )  # fmt: skip


# --- smoke setup from stored facts (CLI `smoke` and the setup graph) -------------------------


def _no_log(msg: str) -> None:
    pass


def smoke_entrypoint(
    evidence: Mapping[str, Evidence], source: str, path: str | None
) -> tuple[Evidence, str, str]:
    """(entrypoints evidence, entrypoint path, verified example command) from subtask B:
    the requested path, else the first train entrypoint with an example, else the first with
    one."""
    eps = [
        (ev, ep)
        for ev in evidence.values()
        if ev.kind == "entrypoints" and ev.source == source
        for ep in ev.content.get("entrypoints", [])
        if ep.get("example")
    ]
    if path is not None:
        eps = [(ev, ep) for ev, ep in eps if ep["path"] == path]
    eps.sort(key=lambda t: t[1].get("kind") != "train")
    if not eps:
        raise ValueError(
            "no entrypoint with a verified example command"
            + (f" for {path}" if path else "")
            + "; run `configure` first"
        )
    ev, ep = eps[0]
    return ev, str(ep["path"]), str(ep["example"]["quote"])


def entrypoint_arguments(
    evidence: Mapping[str, Evidence], source: str, path: str, text: str
) -> tuple[BudgetArg, ...]:
    """argparse spec of path: from the stored entrypoint fact if it has `arguments`, else
    parsed from the clone text (ast only)."""
    for ev in evidence.values():
        if ev.source == source and ev.locator == f"{path}#parsed":
            result = ev.content.get("result")
            if isinstance(result, Mapping) and "arguments" in result:
                return tuple(
                    BudgetArg.model_validate(
                        {k: a[k] for k in ("option_strings", "dest", "action")}
                    )
                    for a in result["arguments"]
                )
    scan = candidate_entrypoints({path: text})
    if not scan.candidates:
        return ()
    return tuple(
        BudgetArg(option_strings=a.option_strings, dest=a.dest, action=a.action, default=a.default)
        for a in scan.candidates[0].arguments
    )


@dataclass(frozen=True)
class SmokeSetup:
    """Everything a smoke run needs from the store and the clone (read once per entry)."""

    source: str
    entry_evidence: Evidence  # subtask B entrypoints (smoke_result links to it)
    path: str
    example: str
    repo_dir: Path
    repo_paths: frozenset[str]
    text: str  # entrypoint source
    arguments: tuple[BudgetArg, ...]
    mount_plan: DataMountPlan
    fix_state0: FixState  # from stored known_fix evidence
    known_fixes0: int  # distinct stored known_fix evidence of this source (all invocations)
    critic_state0: CriticState  # from stored critic_patch / critic_revert evidence
    metric_lines: tuple[str, ...]  # metric lines of the entrypoint
    metric_texts: dict[str, list[str]]  # path -> metric lines (critic protection)
    ctx: ClassifyContext
    cache: str

    @property
    def base_mounts(self) -> list[tuple[str, str]]:
        return [(m.host, m.container) for m in self.mount_plan.mounts]


def prepare_smoke(out: Path, store: RunStore, entrypoint: str | None = None) -> SmokeSetup:
    """Raises ValueError / OSError when there is nothing to smoke."""
    repo_dir = out / "repo"
    source, evidence = latest_repo_evidence(store)
    ep_ev, path, example = smoke_entrypoint(evidence, source, entrypoint)
    repo_paths = frozenset(f.path for f in list_files(repo_dir) if not f.is_symlink)
    text = read_text(repo_dir, path)[0]
    arguments = entrypoint_arguments(evidence, source, path, text)
    mount_plan = stored_mount_plan(store, source)
    stored = [e for e in store.load_evidence().values() if e.source == source]
    metric_lines = tuple(
        str(m.get("line", ""))
        for ev in evidence.values()
        if ev.locator == f"{path}#metric_lines"
        for m in ev.content.get("result", [])
    )
    metric_texts: dict[str, list[str]] = {}
    for ev in evidence.values():
        if ev.source == source and ev.locator.endswith("#metric_lines"):
            for m in ev.content.get("result", []):
                metric_texts.setdefault(str(m.get("path")), []).append(str(m.get("line", "")))
    ctx = ClassifyContext(
        repo_files=repo_paths,
        arguments=arguments,
        mounted=frozenset(m.container for m in mount_plan.mounts),
        dataset_roots=dataset_roots(store, source),
    )
    return SmokeSetup(
        source=source, entry_evidence=ep_ev, path=path, example=example, repo_dir=repo_dir,
        repo_paths=repo_paths, text=text, arguments=arguments, mount_plan=mount_plan,
        fix_state0=fold_fix_state(stored),
        known_fixes0=len({e.evidence_id for e in stored if e.kind == "known_fix"}),
        critic_state0=fold_critic_state(stored),
        metric_lines=metric_lines, metric_texts=metric_texts, ctx=ctx,
        cache=str(data_cache_dir().resolve()),
    )  # fmt: skip


def make_build(
    setup: SmokeSetup,
    out: Path,
    store: RunStore,
    *,
    plan_env: PlanEnvFn,
    image_exists: Callable[[str], bool],
    builder: Callable[..., BuildResult],
    image: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    build_timeout: float = 3600.0,
    log: Callable[[str], None] = _no_log,
) -> Callable[[FixState], SmokeInputs]:
    def build(state: FixState) -> SmokeInputs:
        """Inputs for one smoke run under the fix state; plans the env (incl. packages from
        add_package fixes, already in the store) and builds the image if it is new."""
        pe = plan_env(out, False)
        tag = image or pe.tag
        if not image_exists(tag):
            if image is not None or not state.packages:
                raise ValueError(f"image {tag} not built; run `env --out {out}` first")
            log(f"image:          {tag} (rebuilding for packages {list(state.packages)})")
            prep = prepare_env(out, store, pe, assumptions=False)
            built, _ = build_env_image(
                out, prep, build_timeout=build_timeout, builder=builder, size_of=lambda t: None
            )
            if not built.ok:
                raise ValueError(f"image build failed: {built.log_path}")
        extra = [(setup.cache, m.container) for m in state.mounts]
        return SmokeInputs(
            image=tag,
            clone_dir=setup.repo_dir,
            example=setup.example,
            arguments=setup.arguments,
            third_party=tuple(dict.fromkeys(p.module for p in pe.scan.packages)),
            local_modules=local_import_modules(setup.path, setup.text, setup.repo_paths),
            run_root=out / "smoke",
            data_mounts=[*setup.base_mounts, *extra],
            metric_prefixes=metric_prefixes(setup.metric_lines),
            timeout_s=timeout_s,
            prep_dirs=state.prep_dirs,
            fix_overrides=state.overrides,
        )

    return build


# --- persistence of one iteration ------------------------------------------------------------


@dataclass(frozen=True)
class IterationRecord:
    smoke: Evidence
    attempts: int
    fix: Evidence | None = None
    revert: Evidence | None = None
    critic: Evidence | None = None


def record_iteration(
    store: RunStore, out: Path, setup: SmokeSetup, it: Iteration
) -> IterationRecord:
    """Persist one iteration: smoke_result (+ known_fix, critic_revert, critic_patch), each
    linked to the smoke evidence. Attempts = distinct input digests so far."""
    source = setup.source
    res = it.result
    prior = [
        e.content.get("input_digest")
        for e in store.load_evidence().values()
        if e.kind == "smoke_result" and e.source == source
    ]
    attempts = count_attempts(prior, it.input_digest)
    log_ref = make_ref(out, res.log_path)
    smoke_ev = make_smoke_evidence(
        passed=res.passed,
        attempts=attempts,
        capped=it.capped,
        source=source,
        locator=f"smoke:{res.run_dir.name}",
        details=res.details(
            attempt=attempts, log_ref=log_ref, image=it.image, input_digest=it.input_digest
        ),
        links=(setup.entry_evidence.evidence_id,),
    )
    store.append_evidence(smoke_ev)
    fix_ev = rv_ev = cp_ev = None
    c = it.classification
    if it.action is not None and c is not None:
        fix_ev = known_fix_evidence(
            it.action, c, source=source, smoke_evidence_id=smoke_ev.evidence_id
        )
        store.append_evidence(fix_ev)
    if it.reverted is not None:
        patch_evs = [
            e
            for e in store.load_evidence().values()
            if e.kind == "critic_patch"
            and e.source == source
            and e.content.get("patch_id") == it.reverted.patch_id
        ]
        if not patch_evs:
            raise ValueError(f"no critic_patch evidence for patch {it.reverted.patch_id[:12]}")
        rv_ev = critic_revert_evidence(
            it.reverted,
            source=source,
            patch_evidence_id=patch_evs[-1].evidence_id,
            smoke_evidence_id=smoke_ev.evidence_id,
        )
        store.append_evidence(rv_ev)
    if it.critic is not None and it.critic.counts_as_attempt:
        cp_ev = critic_patch_evidence(
            it.critic, source=source, smoke_evidence_id=smoke_ev.evidence_id
        )
        store.append_evidence(cp_ev)
    if it.critic is not None and it.critic.trace is not None:
        store.append_evidence(
            critic_trace_evidence(it.critic, source=source, smoke_evidence_id=smoke_ev.evidence_id)
        )
    return IterationRecord(smoke_ev, attempts, fix_ev, rv_ev, cp_ev)


@dataclass(frozen=True)
class IterationSummary:
    """One smoke iteration of the setup graph, for the reproduce summary."""

    n: int
    signal: str
    passed: bool
    stage: str
    duration_s: float
    classification: str | None  # "<cls>[/<pattern>]: <reason>"
    fix: str | None  # "<kind> <params>"
    patch: str | None  # "<status> <file>: <rationale or reason>"
    reverted: str | None  # patch id
    capped: bool
    smoke_evidence: str
    log: str


def summarize_iteration(n: int, it: Iteration, rec: IterationRecord) -> IterationSummary:
    res, c, cr = it.result, it.classification, it.critic
    return IterationSummary(
        n=n,
        signal=res.signal,
        passed=res.passed,
        stage=res.stage,
        duration_s=res.duration_s,
        classification=None
        if c is None
        else f"{c.cls}" + (f"/{c.pattern}" if c.pattern else "") + f": {c.reason}",
        fix=None if it.action is None else f"{it.action.kind} {it.action.params}",
        patch=None
        if cr is None
        else f"{cr.status} {cr.file or '-'}: "
        + (cr.patch.rationale if cr.patch is not None else cr.reason),
        reverted=None if it.reverted is None else it.reverted.patch_id[:12],
        capped=it.capped,
        smoke_evidence=rec.smoke.evidence_id[:12],
        log=str(res.log_path),
    )


# --- graph node ------------------------------------------------------------------------------

SETUP_STATUS_BY_ROUTE = {"ready": "ready", "infra": "infra"}  # every other stop: capped


def finish_update(stop: str | None, reason: str, route: str) -> dict[str, Any]:
    """setup_status / stop_reason / stop_detail for a terminal decision."""
    return {
        "setup_status": SETUP_STATUS_BY_ROUTE.get(route, "capped"),
        "stop_reason": stop,
        "stop_detail": reason,
    }


def smoke_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """One smoke run (unload Ollama first: the GPU is for the container). deps: store, runner,
    plan_env, image_exists, build_image, unload. A terminal decision
    (ready / infra / capped) is recorded here; known / critic iterations are recorded by the
    apply_known_fix / critic node (they add the fix / patch to the same record)."""
    out = Path(state["out_dir"])
    store = deps["store"]
    n = state.get("smoke_iterations", 0) + 1
    try:
        setup = prepare_smoke(out, store, state.get("entrypoint"))
    except (ValueError, OSError) as e:
        return {"setup_status": "smoke_failed", "stop_reason": "smoke_failed",
                "stop_detail": str(e)}  # fmt: skip
    # first entry: the stored fix / critic state; afterwards the in-state one
    fix_state = state["fix_state"] if "fix_state" in state else setup.fix_state0
    cstate = state["critic_state"] if "critic_state" in state else setup.critic_state0
    # like critic_attempts: counted from the store, so the cap holds across invocations
    # (apply_known_fix records each fix before the next smoke entry)
    used = setup.known_fixes0
    deps["unload"]()
    build = make_build(
        setup,
        out,
        store,
        plan_env=deps["plan_env"],
        image_exists=deps["image_exists"],
        builder=deps["build_image"],
        timeout_s=float(state.get("smoke_timeout_s", DEFAULT_TIMEOUT_S)),
        build_timeout=float(state.get("build_timeout_s", 3600)),
    )
    try:
        step = smoke_step(build, setup.ctx, fix_state, cstate, runner=deps["runner"],
                          known_fixes_used=used)  # fmt: skip
    except ValueError as e:
        return {"setup_status": "smoke_failed", "stop_reason": "smoke_failed",
                "stop_detail": str(e)}  # fmt: skip
    update: dict[str, Any] = {
        "smoke_setup": setup,
        "fix_state": fix_state,
        "critic_state": step.critic_state,
        "critic_attempts": step.critic_state.attempts,
        "known_fixes_used": used,
        "last_smoke": step.last,
        "smoke_iterations": n,
        "pending": step,
    }
    d = step.decision
    if d.route in ("known", "critic"):
        return update
    rec = record_iteration(store, out, setup, step.iteration)
    summary = summarize_iteration(n, step.iteration, rec)
    update.update(pending=None, smoke_log=[summary], **finish_update(d.stop, d.reason, d.route))
    return update
