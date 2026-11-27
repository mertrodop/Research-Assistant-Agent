"""Pipeline nodes past the setup graph (orchestrator-workers), wiring existing functions only:

  resume -> (subtasks | persist)         full pipeline: reuse stored A/B/C output; skip
                                         env/data/smoke only when the newest smoke_result with
                                         the smoke input digest the current state would produce
                                         (image tag of the env plan, fix state, patch stack,
                                         command) passed (no LLM call, no container)
  plan                                   orchestrator.run_orchestrator (stored mapping reused
                                         when the input digest matches; the LLM is made only
                                         when a call is needed). Claims that do not plan are
                                         recorded (plan_failures) and the others continue;
                                         only "no claim planned" ends at plan_failed
  claim_worker  x N (Send per claim)     execute.run_claims(claim_id=...) holding the GPU in
                                         plan order (spec claim order, GPU_TURNS; a worker
                                         missing for GPU_TURN_GRACE_S is skipped) and
                                         GPU_SEMAPHORE: pre-flight + execute, idempotent by
                                         run digest
  join                                   fan-in (worker_results reducer: operator.add)
  evaluate                               evaluate_claims over the store (read-only); with the
                                         Reflexion loop on (state "reflect"), also the newest
                                         trial per claim: outcome, breaker, applied lesson
  reflect                                reflect.reflect_claim for every claim that
                                         routing.decide_reflect sends there (LLM node "reflect")
  apply_lesson                           apply_lesson.apply_change: code applies the lesson
                                         (re-plan without LLM, reeval or stop); claims to re-run
                                         go to claim_worker again, then join -> evaluate
  vault                                  render_vault + write_notes; a shared vault write
                                         the approval policy gates is skipped until a human
                                         approves its approval_request (awaiting_approval)

Caps and counters stay derived from stored evidence (run_claims reads the GPU hours spent so
far under the semaphore, so workers see each other's runs).
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reproducer.domain.approval import (
    ApprovalPolicy,
    GateResult,
    approval_request_evidence,
    gate,
    policy_from_env,
    request_digest,
)
from reproducer.domain.breaker import (
    breaker_state,
    max_trials_from_env,
    token_budget_from_env,
)
from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evaluate import ClaimEvaluation, evaluate_claims
from reproducer.domain.evidence import Evidence
from reproducer.domain.smoke_plan import plan_smoke
from reproducer.domain.telemetry import token_totals
from reproducer.domain.unknowns import fold_unknowns
from reproducer.domain.vault_render import render_vault
from reproducer.graph.nodes.apply_lesson import (
    applied_for_trial,
    apply_change,
    plan_changed_claims,
    run_command,
)
from reproducer.graph.nodes.diagnose import diagnose
from reproducer.graph.nodes.execute import ClaimRun, RunSetupError, caps_from_env, run_claims
from reproducer.graph.nodes.orchestrator import (
    ORCHESTRATOR_NODE,
    load_current_plans,
    paper_claims,
    plan_failures,
    run_orchestrator,
)
from reproducer.graph.nodes.reflect import REFLECT_NODE, newest_run, reflect_claim
from reproducer.graph.nodes.smoke import prepare_smoke
from reproducer.graph.nodes.subtasks import SUBTASK_KINDS
from reproducer.graph.routing import full_pipeline, reflect_decisions, selected_claims
from reproducer.graph.state import SubtaskSummary
from reproducer.providers.docker_runner import DockerError
from reproducer.providers.llm import LLM, BudgetExceeded, LLMError
from reproducer.providers.store import RunStore
from reproducer.providers.vault import WriteCounts, run_files, write_notes
from reproducer.rules.known_fixes import smoke_input_digest

GPU_SEMAPHORE = threading.Semaphore(1)  # one pre-flight / full run on the GPU at a time
WORKER_ERRORS = (ValueError, OSError, DockerError, LLMError, BudgetExceeded, RunSetupError)


GPU_TURN_GRACE_S = 5.0  # wait this long for the in-order worker before skipping ahead


@dataclass
class _Turn:
    free_since: float  # when the GPU was last released (or the fan-out started)
    done: set[str] = field(default_factory=set)
    waiting: set[str] = field(default_factory=set)
    holding: str | None = None
    broken: str | None = None


class GpuTurns:
    """Hands the GPU to the claim workers of one fan-out one at a time in plan order: when
    the GPU is free, the earliest unfinished claim in `order` gets it if its worker is
    waiting. If that worker has not arrived within `grace_s` of the GPU becoming free (e.g. a
    thread pool smaller than the fan-out runs workers in another order), the earliest-ordered
    waiting worker gets it instead, so it can never deadlock; with all workers present the
    order is exactly plan order. A worker that dies with an unexpected exception breaks its
    fan-out: the claims still waiting raise instead of hanging. A fan-out's record is dropped
    once all of its claims are done."""

    def __init__(
        self, grace_s: float = GPU_TURN_GRACE_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.grace_s = grace_s
        self._clock = clock
        self._cond = threading.Condition()
        self._turns: dict[tuple[str, ...], _Turn] = {}

    def _granted(self, turn: _Turn, key: tuple[str, ...], cid: str) -> float | None:
        """0 when cid may take the GPU now, else how long to wait before checking again."""
        if turn.broken is not None:
            return 0.0
        if turn.holding is not None:
            return self.grace_s
        pending = [c for c in key if c not in turn.done]
        if pending[0] == cid:
            return 0.0
        if pending[0] in turn.waiting:
            return self.grace_s  # the in-order worker is here: it goes first
        left = self.grace_s - (self._clock() - turn.free_since)
        if left > 0:
            return left
        first_waiting = next(c for c in pending if c in turn.waiting)
        return 0.0 if first_waiting == cid else self.grace_s

    @contextmanager
    def hold(self, order: Sequence[str], cid: str) -> Iterator[None]:
        key = tuple(order)
        if cid not in key:
            raise ValueError(f"claim {cid!r} not in the worker order {list(key)}")
        with self._cond:
            turn = self._turns.setdefault(key, _Turn(free_since=self._clock()))
            turn.waiting.add(cid)
            self._cond.notify_all()
            while (wait := self._granted(turn, key, cid)) != 0.0:
                self._cond.wait(timeout=max(wait, 0.01))
            turn.waiting.discard(cid)
            broken = turn.broken
            if broken is None:
                turn.holding = cid
        try:
            if broken is not None:
                raise RuntimeError(f"GPU turn of {cid} abandoned: {broken}")
            yield
        except BaseException:
            with self._cond:
                turn.broken = turn.broken or f"worker {cid} failed"
            raise
        finally:
            with self._cond:
                turn.done.add(cid)
                if turn.holding == cid:
                    turn.holding = None
                    turn.free_since = self._clock()
                if turn.done >= set(key) and self._turns.get(key) is turn:
                    del self._turns[key]
                self._cond.notify_all()


GPU_TURNS = GpuTurns()


# --- resume (full pipeline only) -------------------------------------------------------------


def stored_subtask_kinds(evidence: Mapping[str, Evidence], source: str) -> set[str]:
    """Subtask kinds (A/B/C) whose gated output (locator subtask:<kind>) is stored for source."""
    locs = {f"subtask:{k}": k for k in SUBTASK_KINDS}
    return {locs[ev.locator] for ev in evidence.values()
            if ev.source == source and ev.locator in locs}  # fmt: skip


def setup_smoke_digest(
    out: Path, store: RunStore, plan_env: Callable[[Path, bool], Any], entrypoint: str | None
) -> tuple[str, str]:
    """(source, smoke input digest) the setup smoke would run with now, read-only (no image
    build, no container): the env plan's image tag, the stored fix state, the stored critic
    patch stack and the smoke command, exactly as smoke.smoke_step computes it. Raises
    ValueError / OSError when there is nothing to smoke."""
    setup = prepare_smoke(out, store, entrypoint)
    tag = plan_env(out, False).tag
    state = setup.fix_state0
    command = plan_smoke(setup.example, setup.arguments, state.overrides).command
    patches = tuple(p.diff for p in setup.critic_state0.stack)
    digest = smoke_input_digest(image=tag, state=state, patches=patches, command=command)
    return setup.source, digest


def newest_smoke_for(evidence: Mapping[str, Evidence], source: str, digest: str) -> Evidence | None:
    """The newest smoke_result of source with this input digest (pre-flight smokes of other
    commands have other digests and are ignored)."""
    return next((ev for ev in reversed(list(evidence.values()))
                 if ev.kind == "smoke_result" and ev.source == source
                 and ev.content.get("input_digest") == digest), None)  # fmt: skip


def resume_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only. Full pipeline only (--until plan/run/vault): if the store holds A/B/C output
    for this source, report them as ok with 0 attempts (no LLM call); if moreover the newest
    smoke_result with the smoke input digest the current state would produce (env plan tag,
    fix state, patch stack, command) passed, the setup is ready (env/data/smoke skipped).
    Any change of those inputs (or no such smoke) -> env/data/smoke run as usual."""
    if not full_pipeline(state):
        return {}
    store = deps["store"]
    evidence = store.load_evidence()
    source = state["source"]
    if stored_subtask_kinds(evidence, source) != set(SUBTASK_KINDS):
        return {"resumed": []}
    update: dict[str, Any] = {
        "resumed": ["subtasks"],
        "outcomes": [
            SubtaskSummary(kind=k, ok=True, attempts=0, tokens=0, latency_s=0.0)
            for k in SUBTASK_KINDS
        ],  # fmt: skip
    }
    try:
        smoke_source, digest = setup_smoke_digest(
            Path(state["out_dir"]), store, deps["plan_env"], state.get("entrypoint")
        )
    except (ValueError, OSError):
        return update  # nothing to compare against: run env/data/smoke
    smoke = newest_smoke_for(evidence, smoke_source, digest)
    if smoke is not None and smoke.content.get("passed") is True:
        update["resumed"] = ["subtasks", "smoke"]
        update.update(setup_status="ready", stop_reason="resumed",
                      stop_detail=f"smoke_result {smoke.evidence_id[:12]} passed with the "
                                  f"current smoke inputs (digest {digest[:12]})")  # fmt: skip
    return update


# --- plan ------------------------------------------------------------------------------------


def plan_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """run_orchestrator on the store: reuses the stored claim_mapping for the current input
    digest (no LLM call; the orchestrator LLM is made only when a call is needed). A claim
    that does not plan is recorded in plan_failures (it stays open) and the others go on;
    no claim planned -> setup_status plan_failed (the graph ends). plans: spec claim order."""
    store = deps["store"]
    out = Path(state["out_dir"])

    def failed(detail: str, outcome: Any = None) -> dict[str, Any]:
        return {"setup_status": "plan_failed", "stop_reason": "plan_failed",
                "stop_detail": detail, "plan_outcome": outcome, "plans": []}  # fmt: skip

    def make_llm() -> LLM:
        llm = deps["llm_for"](ORCHESTRATOR_NODE)
        llm.on_record = store.append_call_record
        return llm

    try:
        outcome = run_orchestrator(store, llm_factory=make_llm, repo_dir=out / "repo")
    except (ValueError, LLMError, BudgetExceeded) as e:
        return failed(f"{type(e).__name__}: {e}")
    if not outcome.plans:
        return failed("; ".join(outcome.errors) or "no plans", outcome)
    stored = list(paper_claims(store.load_evidence()))  # store order
    spec = state.get("spec")
    in_spec = [c.claim_id for c in spec.claims] if spec is not None else []
    claim_ids = [c for c in in_spec if c in stored] + [c for c in stored if c not in in_spec]
    planned = {p.claim.claim_id for p in outcome.plans}
    return {
        "plan_outcome": outcome,
        "plans": [cid for cid in claim_ids if cid in planned],
        "plan_failures": plan_failures(claim_ids, sorted(planned), outcome.errors),
    }


def stored_plan_failures(store: RunStore, out_dir: Path) -> dict[str, str]:
    """claim_id -> why it has no current plan, recomputed read-only from the store (for the
    ledger); {} when nothing is planned yet."""
    try:
        return dict(load_current_plans(store, repo_dir=out_dir / "repo", partial=True).failed)
    except (ValueError, OSError):
        return {}


# --- claim workers + fan-in ------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerResult:
    claim_id: str
    runs: tuple[ClaimRun, ...]
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        return self.runs[-1].status if self.runs else "none"


def claim_worker(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """One claim: pre-flight + execute via run_claims(claim_id=...), holding the GPU for the
    whole job (the pre-flight smoke uses the GPU too): first its turn in plan order
    (worker_order), then the semaphore. An error is recorded in the result; the other
    workers continue."""
    cid = state["worker_claim_id"]
    order = state.get("worker_order") or [cid]
    run_deps = {
        "runner": deps["runner"],
        "plan_env": deps["plan_env"],
        "image_exists": deps["image_exists"],
        "build_image": deps["build_image"],
        "unload": deps["unload"],
        "make_critic": deps.get("make_critic", lambda setup: None),
        "env": deps.get("env", os.environ),
        "smoke_timeout_s": state.get("smoke_timeout_s", 1200),
        "build_timeout_s": state.get("build_timeout_s", 3600),
    }
    noop = lambda s: None  # noqa: E731
    with deps.get("gpu_turns", GPU_TURNS).hold(order, cid), deps.get("gpu_lock", GPU_SEMAPHORE):
        try:
            runs = run_claims(
                Path(state["out_dir"]), deps["store"], run_deps, claim_id=cid,
                no_checkpoints=bool(state.get("no_checkpoints", False)), partial_plans=True,
                log=deps.get("log", noop), progress=deps.get("progress", noop),
            )  # fmt: skip
        except WORKER_ERRORS as e:
            return {"worker_results": [WorkerResult(cid, (), f"{type(e).__name__}: {e}")]}
    return {"worker_results": [WorkerResult(cid, tuple(runs))]}


def join_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Fan-in: runs once after every claim_worker of the step (results merged by reducer)."""
    return {"workers_done": len(state.get("worker_results", ()))}


# --- evaluate + vault ------------------------------------------------------------------------


def evaluate_store(
    store: RunStore, env: Mapping[str, str] | None = None
) -> tuple[tuple[ResultClaim, ...], dict[str, ClaimEvaluation]]:
    """(claims, evaluations) recomputed from the store (read-only); the ledger's trial limit
    is REPRODUCER_MAX_TRIALS of env (default os.environ). Raises ValueError when there are
    no claims or no pre-registered tolerance."""
    evidence = store.load_evidence()
    unknowns = fold_unknowns(store.load_events(), evidence)
    claims = tuple(c for c, _ in paper_claims(evidence).values())
    if not claims:
        raise ValueError("no paper_claim evidence: configure with --spec first")
    max_tries = max_trials_from_env(os.environ if env is None else env)
    return claims, evaluate_claims(claims, evidence, unknowns, max_tries=max_tries)


def render_store_vault(
    store: RunStore,
    out_dir: Path,
    plan_failures: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """{note path: markdown} rendered from the store (read-only); ValueError on a bad note.
    The trial limit in the notes is REPRODUCER_MAX_TRIALS of env (default os.environ)."""
    env = os.environ if env is None else env
    if plan_failures is None:
        plan_failures = stored_plan_failures(store, out_dir)
    evidence = store.load_evidence()
    events = store.load_events()
    unknowns = fold_unknowns(events, evidence)
    claims, evals = evaluate_store(store, env)
    notes = render_vault(
        evidence, events, unknowns, evals,
        call_records=store.load_call_records(), run_files=run_files(out_dir, evidence),
        plan_failures=plan_failures, max_trials=max_trials_from_env(env),
    )  # fmt: skip
    return notes


def write_store_vault(
    store: RunStore,
    out_dir: Path,
    root: Path,
    plan_failures: Mapping[str, str] | None = None,
    notes: Mapping[str, str] | None = None,
) -> tuple[str, WriteCounts]:
    """Render the store into notes (or take `notes`) and write them under root. Returns
    (paper note path relative to root, counts). plan_failures: claim_id -> why it did not plan
    (default: recomputed from the store). Read-only on the store; ValueError on a bad note."""
    if notes is None:
        notes = render_store_vault(store, out_dir, plan_failures)
    counts = write_notes(root, notes)
    return next(p for p in notes if p.startswith("Papers/")), counts


def vault_request(
    store: RunStore, root: Path, notes: Mapping[str, str], reason: str = ""
) -> Evidence:
    """approval_request for writing into the vault at root: a standing approval per vault
    path (input digest: the resolved vault path only, so changed notes file no new request;
    links the newest stored evidence the first notes were rendered from)."""
    subject = str(Path(root).resolve())
    newest = list(store.load_evidence())[-1:]
    return approval_request_evidence(
        action="shared_vault_write",
        subject=subject,
        cost={"notes": len(notes)},
        detail={"notes": sorted(notes)},
        input_digest=request_digest("shared_vault_write", {"vault": subject}),
        source="vault",
        links=newest,
        reason=reason,
    )  # fmt: skip


def targets_shared_vault(root: Path, policy: ApprovalPolicy) -> bool:
    """root is the policy's shared vault (REPRODUCER_SHARED_VAULT), compared resolved."""
    return bool(policy.shared_vault) and (
        Path(root).resolve() == Path(str(policy.shared_vault)).resolve()
    )


def vault_gate(
    store: RunStore, root: Path, notes: Mapping[str, str], policy: ApprovalPolicy
) -> GateResult:
    """Gate a vault write; files the request when it is new. The shared vault path of the
    policy is compared resolved."""
    if policy.shared_vault:
        policy = dataclasses.replace(policy, shared_vault=str(Path(policy.shared_vault).resolve()))
    req = vault_request(store, root, notes)
    g = gate(store.load_evidence(), req, policy, float(len(notes)))
    if g.new_request:
        req = vault_request(store, root, notes, g.reason)
        store.append_evidence(req)
        g = dataclasses.replace(g, request=req)
    return g


def _diagnosis(
    state: Mapping[str, Any], deps: Mapping[str, Any], claims: Any, evals: Any
) -> tuple[list[Any], Any]:
    env = deps.get("env", os.environ)
    return diagnose(
        Path(state["out_dir"]), deps["store"], claims, evals,
        gpu_h_cap=caps_from_env(env, 0.0).paper_h, env=env,
    )  # fmt: skip


def current_commands(store: RunStore, out_dir: Path) -> dict[str, str]:
    """claim_id -> the rendered command of its current plan (read-only; {} if none plan)."""
    try:
        cur = load_current_plans(store, repo_dir=out_dir / "repo", partial=True)
    except (ValueError, OSError):
        return {}
    return {p.claim.claim_id: p.command for p in cur.plans}


def reflect_info(
    diags: Sequence[Any],
    breaker: Any,
    evidence: Mapping[str, Evidence],
    commands: Mapping[str, str] | None = None,
) -> dict:
    """claim_id -> {trial, outcome, can_try, tripped, applied, unchanged, changes}: the newest
    trial of each claim (routing.decide_reflect's input), from evidence (+ the current plan
    commands: unchanged = the current plan renders the newest run's command)."""
    out: dict[str, dict[str, Any]] = {}
    for d in diags:
        cid = d.claim.claim_id
        hit = newest_run(d)
        trial, outcome = (hit[0], hit[1].outcome.cls) if hit is not None else (0, None)
        applied = applied_for_trial(evidence, cid, trial) if trial else None
        mine = [t for t in breaker.tripped if not t.startswith("claim ") or
                t.startswith(f"claim {cid}:")]  # fmt: skip
        changes = [
            ev for ev in evidence.values()
            if ev.kind == "lesson_applied" and ev.content.get("claim_id") == cid
            and ev.content.get("trial") == trial
        ]  # fmt: skip
        cmd = (commands or {}).get(cid)
        unchanged = bool(
            hit is not None and cmd is not None and run_command(hit[1].run, evidence) == cmd
        )
        out[cid] = {
            "unchanged": unchanged,
            "changes": len(changes),
            "trial": trial,
            "outcome": outcome,
            "can_try": breaker.can_try(cid),
            "tripped": "; ".join(mine),
            "applied": applied.content.get("outcome") if applied is not None else None,
            "value": hit[1].outcome.value if hit is not None else None,
        }
    return out


def evaluate_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """evaluations (and, with the loop on, reflect_info) of the selected claims: all, or only
    the --claim one (the others are not evaluated, so never reflected on)."""
    try:
        claims, evals = evaluate_store(deps["store"], deps.get("env", os.environ))
    except ValueError as e:
        return {"evaluate_error": str(e)}
    keep = set(selected_claims(state, evals))
    claims = tuple(c for c in claims if c.claim_id in keep)
    evals = {cid: e for cid, e in evals.items() if cid in keep}
    update: dict[str, Any] = {"evaluations": evals}
    if state.get("reflect"):
        try:
            diags, breaker = _diagnosis(state, deps, claims, evals)
        except ValueError as e:
            return {**update, "evaluate_error": f"diagnose: {e}"}
        store = deps["store"]
        update["reflect_info"] = reflect_info(
            diags, breaker, store.load_evidence(), current_commands(store, Path(state["out_dir"]))
        )
    return update


def reflect_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """One lesson per claim that decide_reflect sends here (gated LLM call, stored). A claim
    without a valid lesson is recorded in reflect_failures for that trial (the loop then ends
    for it: decide_reflect never reflects twice on one trial)."""
    store = deps["store"]
    want = [cid for cid, (d, _) in reflect_decisions(state).items() if d == "reflect"]
    failures = dict(state.get("reflect_failures") or {})
    pending: list[dict[str, Any]] = []
    log: list[str] = []

    def make_llm() -> LLM:
        llm = deps["llm_for"](REFLECT_NODE)
        llm.on_record = store.append_call_record
        return llm

    made: list[LLM] = []

    def factory() -> LLM:
        if not made:
            made.append(make_llm())
        return made[0]

    claims, evals = evaluate_store(store, deps.get("env", os.environ))
    diags, breaker = _diagnosis(state, deps, claims, evals)
    by_id = {d.claim.claim_id: d for d in diags}
    for cid in want:
        trial = (state.get("reflect_info") or {}).get(cid, {}).get("trial")
        try:
            r = reflect_claim(store, by_id[cid], breaker, llm_factory=factory)
        except (ValueError, LLMError, BudgetExceeded) as e:
            failures[cid] = {"trial": trial, "reason": f"{type(e).__name__}: {e}"}
            log.append(f"[{cid}] trial {trial}: reflection failed: {type(e).__name__}: {e}")
            continue
        if r.status != "lesson" or r.evidence is None:
            failures[cid] = {"trial": trial, "reason": r.reason}
            log.append(f"[{cid}] trial {trial}: no lesson: {r.reason}")
            continue
        les = r.evidence.content["lesson"]
        ch = les.get("proposed_change") or {"kind": "none"}
        val = "" if ch.get("new_value") is None else f" -> {ch['new_value']}"
        how = "reused" if r.reused else f"attempts={r.attempts}"
        log.append(f"[{cid}] trial {r.trial}: lesson {r.evidence.evidence_id[:12]} ({how}) "
                   f"step {les['step']}, change [{ch['kind']}] {ch.get('target') or ''}{val}: "
                   f"{les['text']}")  # fmt: skip
        pending.append({"claim_id": cid, "lesson_id": r.evidence.evidence_id,
                        "input_digest": r.input_digest})  # fmt: skip
    return {"reflect_pending": pending, "reflect_failures": failures, "reflexion_log": log,
            "rerun_claims": []}  # fmt: skip


def apply_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Code applies each pending lesson (apply_lesson.apply_change): re-plan without an LLM
    call, then rerun (claim_worker again) / reeval (already stored) / stop."""
    store = deps["store"]
    out = Path(state["out_dir"])
    failures = dict(state.get("reflect_failures") or {})
    rerun: list[str] = []
    log: list[str] = []
    applied_now: list[Evidence] = []

    def make_llm() -> LLM:  # only if the mapping inputs changed (not for a lesson)
        llm = deps["llm_for"](ORCHESTRATOR_NODE)
        llm.on_record = store.append_call_record
        return llm

    evidence = store.load_evidence()
    for p in state.get("reflect_pending") or ():
        cid = p["claim_id"]
        lesson_ev = evidence.get(p["lesson_id"])
        trial = lesson_ev.content.get("trial") if lesson_ev is not None else None
        try:
            if lesson_ev is None:
                raise ValueError(f"lesson {p['lesson_id'][:12]} not in the store")
            res = apply_change(out, store, lesson_ev, input_digest=p["input_digest"],
                               llm_factory=make_llm)  # fmt: skip
        except (ValueError, LLMError, BudgetExceeded) as e:
            failures[cid] = {"trial": trial, "reason": f"apply: {e}"}
            log.append(f"[{cid}] trial {trial}: lesson not applied: {e}")
            continue
        extra = ""
        if res.reeval_run is not None:
            extra = f"; reeval run_result {res.reeval_run.evidence_id[:12]}"
        if res.reused:  # the same lesson again: no new input, no progress -> stop the loop
            failures[cid] = {"trial": trial, "reason": "lesson already applied (no new input)"}
            log.append(f"[{cid}] trial {trial}: lesson already applied; no new input")
            continue
        log.append(f"[{cid}] trial {trial}: {res.action}: {res.detail}{extra}")
        if res.action == "rerun":
            rerun.append(cid)
        if res.record is not None:
            applied_now.append(res.record)

    # cross-claim effects: other claims whose current plan changed (e.g. a resolved shared
    # unknown) and that have no completed run of it run too (gated; a trial of their own)
    if applied_now:
        evidence = store.load_evidence()
        env = deps.get("env", os.environ)
        tokens = token_totals(store.load_call_records())
        commands = current_commands(store, out)
        breaker = breaker_state(
            list(commands), evidence, gpu_h_cap=caps_from_env(env, 0.0).paper_h,
            token_budget=token_budget_from_env(env), tokens_used=tokens["api"] + tokens["local"],
            max_per_claim=max_trials_from_env(env),
        )  # fmt: skip
        causes = ", ".join(f"{ev.evidence_id[:12]} ({ev.content.get('claim_id')})"
                           for ev in applied_now)  # fmt: skip
        for cid in plan_changed_claims(evidence, commands):
            if cid in rerun:
                continue
            if not selected_claims(state, [cid]):
                log.append(f"[{cid}] plan changed by lesson_applied {causes}; not re-run: "
                           f"--claim {state.get('only_claim')}")  # fmt: skip
                continue
            if not breaker.can_try(cid):
                log.append(f"[{cid}] plan changed by lesson_applied {causes}; not re-run: "
                           f"breaker: {'; '.join(breaker.tripped)}")  # fmt: skip
                continue
            rerun.append(cid)
            log.append(f"[{cid}] rerun: its plan changed by lesson_applied {causes}")
    order = list(state.get("plans") or [])
    rerun = sorted(rerun, key=lambda c: order.index(c) if c in order else len(order))
    return {"rerun_claims": rerun, "reflect_pending": [], "reflect_failures": failures,
            "reflexion_log": log}  # fmt: skip


def vault_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Render, gate (shared_vault_write), write. Pending / rejected: nothing is written."""
    out = Path(state["out_dir"])
    root = Path(state.get("vault_dir") or out / "vault_out")
    store = deps["store"]
    try:
        env = deps.get("env", os.environ)
        notes = render_store_vault(store, out, state.get("plan_failures"), env)
        g = vault_gate(store, root, notes, policy_from_env(deps.get("env", os.environ)))
        if not g.proceed:
            return {"vault_approval": g}
        paper_note, counts = write_store_vault(store, out, root, notes=notes)
    except ValueError as e:
        return {"vault_error": str(e)}
    update: dict[str, Any] = {"vault_counts": counts, "vault_paper_note": str(root / paper_note)}
    if g.status is not None:
        update["vault_approval"] = g
    return update


def awaiting_approval(final: Mapping[str, Any]) -> bool:
    """Something in this run waits for a human: a critic patch (setup), a full run or a
    pre-flight critic patch (workers), or a shared vault write."""
    if final.get("setup_status") == "awaiting_approval":
        return True
    if any(r.status == "awaiting_approval"
           for w in final.get("worker_results") or () for r in w.runs):  # fmt: skip
        return True
    va = final.get("vault_approval")
    return va is not None and va.status == "pending"
