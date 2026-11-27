"""Command-line entry point.

    python -m reproducer.cli inventory <github-url> [--commit SHA] --out runs/<name>
    python -m reproducer.cli subtasks --out runs/<name>
    python -m reproducer.cli configure <github-url> [--commit SHA] --out runs/<name>
    python -m reproducer.cli configure --spec evals/<name>/spec.json [--commit SHA] \
        --out runs/<name> [--tol-abs 0.5]
    python -m reproducer.cli env --out runs/<name> [--cpu-only]
    python -m reproducer.cli gpu-check --out runs/<name>
    python -m reproducer.cli data --out runs/<name> [--spec evals/<name>/spec.json]
    python -m reproducer.cli smoke --out runs/<name> [--timeout 1200] [--auto-fix]
    python -m reproducer.cli reproduce --spec evals/<name>/spec.json --out runs/<name> \
        [--until configure|env|data|smoke]
    python -m reproducer.cli plan --out runs/<name> [--replan]
    python -m reproducer.cli run --out runs/<name> [--claim ID] [--preflight-only] \
        [--no-checkpoints]
    python -m reproducer.cli ledger --out runs/<name>
    python -m reproducer.cli reflect --out runs/<name> [--claim ID] [--dry-run] [--force]
    python -m reproducer.cli vault --out runs/<name> [--vault PATH]
    python -m reproducer.cli queue --out runs/<name>
    python -m reproducer.cli approve --out runs/<name> <request_id> [--note "..."]
    python -m reproducer.cli reject --out runs/<name> <request_id> --note "..."
    python -m reproducer.cli answer --out runs/<name> <unknown_name> <value> --note "..."

Clones into <out>/repo (never executes anything from it), classifies the files, reads the
README and requirement/docker files, and appends evidence to RunStore(<out>). Then runs
the deterministic extractors (requirements, setup.py, pyproject, environment.yml, config
keys, entrypoint candidates; ast/safe loaders only) and appends derived evidence. Shell
scripts (*.sh, at most MAX_SHELL_SCRIPTS) are stored as observed text.

`subtasks` loads that store, rebuilds the facts from stored evidence (re-running the
extractors on <out>/repo only if none are stored), and runs the gated LLM subtasks A (env),
B (entrypoints), C (workflow) sequentially with one shared Budget. Evidence, UnknownOpened
events and LLM call records are appended to the store.

`env` plans the Docker environment from the stored EnvSpec (subtask A), the host GPU
(nvidia-smi) and a static import scan of the entrypoints, stores the plan as env_spec/env_plan
evidence (authority "rule"), writes <out>/env/<hash>.Dockerfile and builds the image
reproducer/<name>:<hash> (cached if it exists). `gpu-check` runs a tiny torch/torchvision
matmul on the GPU inside that image (no network, locked-down container) to prove the arch
table and driver are right. Repo code never runs on the host.

`data` scans the repo for torchvision dataset calls (ast; stored as '<path>#dataset_calls'),
selects spec datasets that the scan also found and the registry knows, makes sure they are in
the shared cache (check with --network none; download only by our trusted script in the env
image), stores dataset_cache evidence, prints the read-only mount plan and proves the data is
readable from a network-none sandbox with our own script (no repo code).

`smoke` is the first execution of repo code (graph/nodes/smoke.py): GPU probe, import probe,
then the verified example command with budget overrides (epochs -> 1) in the sandbox; the
result is stored as smoke_result evidence with a signal, timing and the parsed traceback.
Without --auto-fix failures are recorded, not fixed. With --auto-fix known patterns get a
zero-token fix and a `critic` failure gets one validated patch from the grounded critic (LLM
node "critic", graph/nodes/critic.py), stored as critic_patch evidence and applied to the
per-run copies only; then the smoke is repeated.

`configure` runs both steps as the LangGraph configure graph (graph/build.py): clone or
reuse <out>/repo, collect evidence, run A/B/C in parallel, then one persist step writes
evidence + events. Status is configured only if all three subtasks succeed.
With --spec, the ReproSpec drives the run: ingest_spec adds paper evidence and opens an
unknown per null setup value, the repo is the spec's first GitHub code_link, and after
persist the ledger (recomputed read-only from the store) and the open unknowns are printed.

`reproduce` runs everything above as ONE LangGraph setup graph (graph/build.py
build_setup_graph): configure -> env -> data -> smoke, with deterministic routing by outcome
(graph/routing.py): known failures get their zero-token fix (add_package -> env rebuild),
`critic` failures the grounded critic, bounded by the known-fix and critic caps. env, data and
smoke above are thin wrappers over the same node functions (graph/nodes/env.py, data.py,
smoke.py, critic.py). Prints a setup summary and the ledger.

`plan` runs the orchestrator (graph/nodes/orchestrator.py, LLM node "orchestrator") on a
configured store: tolerance evidence per claim first (pre-registration), the code-found
'#train_config' lines, one gated ClaimMapping call, then one RunPlan per claim built by code
(values copied from the paper evidence), stored as run_plan evidence with Assumption events.
A mapping stored for the same inputs (digest) is reused without an LLM call; --replan forces
a new call. Nothing is executed.

`run` (graph/nodes/execute.py) takes the current plan per claim: pre-flight (the smoke loop on
the plan's command with epochs -> 1, known fixes / critic as in `smoke --auto-fix`), a GPU-hour
estimate checked against REPRODUCER_RUN_GPU_HOURS (default 4) and REPRODUCER_PAPER_GPU_HOURS
(default 10, minus hours already spent), stored as preflight evidence; then the full run in the
sandbox, its metric converted to the claim's, stored as run_result evidence. A plan whose run
inputs already have a completed run is skipped. --preflight-only stops before the full run.
The ledger is not written. One `run` per store at a time (<out>/run.lock); an interrupted full
run is stored as an aborted run_result (its GPU hours count toward the paper cap).

`ledger` is read-only: per claim the paper value ± its pre-registered tolerance, our best
completed run (raw + converted), |diff|, the code verdict, ledger status and fidelity, and the
code-built reasons the result is inferred (domain/evaluate.py); then GPU-hour and token totals.

`reflect` (graph/nodes/reflect.py, LLM node "reflect") writes one Reflexion lesson per (claim,
trial) for a claim whose newest run is out_of_tolerance or diverged and that has trial budget
left: compact observations (claim, our value, curve summary, plan, setup checks, deviations,
previous lessons; never raw logs) -> one gated Lesson with at most one grounded proposed
change, stored as lesson evidence. Nothing is applied. A stored lesson for the same (claim,
trial, inputs) is reused without an LLM call; changed inputs give a fresh lesson (the old one
stays, stale, never applied). --dry-run prints the lesson and writes nothing (no
evidence, no call records); --force (needs --dry-run) also reflects on other outcomes.
`vault` is read-only on the store: it renders the paper, claim, run and unknown notes
(domain/vault_render.py) and writes them into an Obsidian vault (--vault, else
$REPRODUCER_VAULT, else <out>/vault_out). Only the managed block of an existing note is
replaced; unchanged notes are not written; notes are never deleted.

Human in the loop (domain/approval.py; REPRODUCER_APPROVAL=strict|default|off): a gated full
run, critic patch or shared vault write is filed as approval_request evidence and the command
stops with exit 4 (awaiting_approval). `queue` lists the pending requests and the open unknowns
with the exact commands to answer them; `approve` / `reject` store a human approval_decision
(append-only; the LLM can never approve); `answer` resolves an open unknown with human_input
evidence. Re-running `reproduce` resumes and proceeds with the decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langgraph.errors import GraphRecursionError

from reproducer.domain.approval import (
    ApprovalPolicy,
    answer_unknown,
    approval_decision_evidence,
    cost_text,
    decision_for,
    decisions_for,
    detail_summary,
    escalation_queue,
    policy_from_env,
    policy_warning,
    resolve_request_id,
)
from reproducer.domain.breaker import BreakerState, max_trials_from_env, token_budget_from_env
from reproducer.domain.contracts import ReproSpec, ResultClaim
from reproducer.domain.data_plan import DATASET_SOURCE_UNKNOWN, select_datasets
from reproducer.domain.eval_report import (
    PaperReport,
    build_report,
    not_run,
    paper_report,
    render_markdown,
    report_json,
)
from reproducer.domain.evaluate import ClaimEvaluation
from reproducer.domain.evidence import Evidence
from reproducer.domain.inventory import RepoInventory
from reproducer.domain.ledger import Tolerance, ToleranceConfig, compute_ledger
from reproducer.domain.metric_format import (
    RAW_PREFIX,
    format_metric,
    format_metrics,
    format_raw,
    metric_decimals,
)
from reproducer.domain.telemetry import CallRecord, token_totals
from reproducer.domain.unknowns import UnknownOpened, UnknownState, fold_unknowns
from reproducer.domain.vault_render import applied_of, change_text, lessons_by_trial
from reproducer.graph.build import (
    SETUP_RECURSION_LIMIT,
    build_configure_graph,
    build_setup_graph,
)
from reproducer.graph.nodes.apply_lesson import trial_causes
from reproducer.graph.nodes.configure import (
    classify_files,
    inventory_to_evidence,
    parse_github_url,
    repo_source,
)
from reproducer.graph.nodes.critic import CRITIC_NODE, MAX_CRITIC_ATTEMPTS, make_critic_fn
from reproducer.graph.nodes.data import (
    DataScan,
    disk_usage,
    latest_repo_evidence,
    provision,
    resolve_mounts,
    sandbox_proof,
    sandbox_proof_code,
    scan_dataset_calls,
    spec_datasets,
    update_dataset_source,
)
from reproducer.graph.nodes.diagnose import ClaimDiagnosis, backfill_curves, diagnose
from reproducer.graph.nodes.env import (
    PlannedEnv,
    build_env_image,
    cleanup_workdir,
    env_assumption_events,
    plan_env,
    prepare_env,
)
from reproducer.graph.nodes.execute import (
    ClaimRun,
    RunSetupError,
    caps_from_env,
    run_claims,
    spent_gpu_hours,
)
from reproducer.graph.nodes.orchestrator import (
    DEFAULT_MAX_TOKENS as ORCH_MAX_TOKENS,
)
from reproducer.graph.nodes.orchestrator import (
    ORCHESTRATOR_NODE,
    OrchestratorOutcome,
    paper_claims,
    run_orchestrator,
    tolerance_summary,
)
from reproducer.graph.nodes.pipeline import (
    WorkerResult,
    awaiting_approval,
    evaluate_store,
    render_store_vault,
    stored_plan_failures,
    targets_shared_vault,
    vault_gate,
    write_store_vault,
)
from reproducer.graph.nodes.reflect import DEFAULT_MAX_TOKENS as REFLECT_MAX_TOKENS
from reproducer.graph.nodes.reflect import REFLECT_NODE, ReflectOutcome, reflect_claim
from reproducer.graph.nodes.smoke import (
    DEFAULT_TIMEOUT_S,
    Iteration,
    IterationRecord,
    SmokeResult,
    make_build,
    prepare_smoke,
    record_iteration,
    smoke_loop,
)
from reproducer.graph.nodes.subtasks import (
    MAX_SHELL_SCRIPTS,
    SUBTASK_KINDS,
    SubtaskOutcome,
    build_subtask_context,
    collect_texts,
    command_evidence,
    extract_repo_facts,
    file_evidence,
    is_shell_script,
    load_repo_facts,
    metric_line_evidence,
    run_subtask,
    version_evidence,
)
from reproducer.graph.routing import (
    MAX_KNOWN_FIXES,
    PIPELINE_STAGES,
    UNTIL_STAGES,
    fallback_reason,
    final_setup_status,
    reflect_decisions,
)
from reproducer.graph.state import SetupDeps, merge_evidence
from reproducer.providers.datasets import data_cache_dir, ensure_datasets
from reproducer.providers.docker_runner import (
    DockerError,
    build_image,
    image_exists,
    image_size_bytes,
    new_run_dir,
    prepare_workdir,
    run_container,
)
from reproducer.providers.git import (
    CloneError,
    clone_repo,
    head_sha,
    list_files,
    read_text,
    validate_commit,
)
from reproducer.providers.gpu import query_gpu
from reproducer.providers.llm import (
    Budget,
    BudgetExceeded,
    LLMError,
    OllamaLLM,
    OllamaOpenAILLM,
    llm_for_node,
    make_llm,
    parse_spec,
)
from reproducer.providers.store import (
    NotARunStore,
    RunLocked,
    RunStore,
    UnknownEventT,
    resolve_ref,
    run_lock,
)
from reproducer.providers.vault import run_files  # noqa: F401  (re-exported for callers)
from reproducer.rules.classify import ClassifyContext
from reproducer.rules.known_fixes import FixState

# re-exported for callers/tests of the pre-graph CLI helpers (logic lives in graph/nodes/*)
__all__ = [
    "cleanup_workdir",
    "env_assumption_events",
    "sandbox_proof_code",
    "update_dataset_source",
]


def _cmd_inventory(args: argparse.Namespace) -> int:
    try:
        parse_github_url(args.url)  # fail fast, before cloning
        if args.commit is not None:
            validate_commit(args.commit)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    repo_dir = out / "repo"
    try:
        if args.reuse_clone and repo_dir.exists():
            sha = head_sha(repo_dir)
            if args.commit is not None and sha != args.commit:
                raise CloneError(
                    f"--reuse-clone: {repo_dir} is at {sha}, not the requested --commit "
                    f"{args.commit}; use a fresh --out"
                )
            print(f"reusing existing clone {repo_dir}", file=sys.stderr)
        else:
            sha = clone_repo(args.url, repo_dir, commit=args.commit, timeout_s=args.timeout)
    except (CloneError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    inv = classify_files(list_files(repo_dir), sha)
    texts: dict[str, tuple[str, bool]] = {}
    wanted = ([inv.readme] if inv.readme else []) + [*inv.requirement_files, *inv.docker_files]
    for path in wanted:
        try:
            texts[path] = read_text(repo_dir, path)
        except (ValueError, OSError) as e:
            print(f"warning: skipped {path}: {e}", file=sys.stderr)

    source = repo_source(args.url, sha)
    evidence = inventory_to_evidence(inv, source, texts)
    store = RunStore(out)
    written = sum(store.append_evidence(ev) for ev in evidence)

    print(f"repo:          {args.url}")
    print(f"commit:        {sha}")
    print(f"files:         {len(inv.files)}")
    print(f"python:        {len(inv.python_files)}")
    print(f"notebooks:     {len(inv.notebooks)}")
    print(f"configs:       {len(inv.config_files)}")
    print(f"requirements:  {len(inv.requirement_files)} {list(inv.requirement_files)}")
    print(f"docker:        {len(inv.docker_files)} {list(inv.docker_files)}")
    print(f"submodules:    {inv.has_submodules}")
    print(f"readme:        {inv.readme}")
    print(f"evidence:      {written} written ({len(evidence)} built) -> {store.evidence_path}")

    facts = extract_repo_facts(inv, collect_texts(repo_dir, inv, args.max_py_files), source)
    derived = [ev for ev in facts if ev.confidence == "derived"]
    observed = [ev for ev in facts if ev.confidence == "observed"]
    obs_written = sum(store.append_evidence(ev) for ev in observed)
    der_written = sum(store.append_evidence(ev) for ev in derived)
    by_parser = Counter(ev.content["parser"] for ev in derived)
    counts = " ".join(f"{k}={by_parser[k]}" for k in sorted(by_parser)) or "none"
    print(f"extracted:     {counts}")
    print(
        f"derived:       {der_written} written ({len(derived)} built), "
        f"+{obs_written} source-file records"
    )

    # shell scripts: observed text, scanned for example commands by `subtasks`
    scripts = [f.path for f in inv.files if not f.is_symlink and is_shell_script(f.path)]
    sh_written = 0
    for path in scripts[:MAX_SHELL_SCRIPTS]:
        try:
            text, truncated = read_text(repo_dir, path)
        except (ValueError, OSError) as e:
            print(f"warning: skipped {path}: {e}", file=sys.stderr)
            continue
        sh_written += store.append_evidence(file_evidence(source, path, text, truncated))
    print(f"shell scripts: {len(scripts)} found, {sh_written} written (max {MAX_SHELL_SCRIPTS})")
    return 0


def _print_outcome(name: str, out: SubtaskOutcome, tokens: int, latency: float) -> None:
    line = (
        f"{name:<15} ok={out.ok} attempts={out.attempts} tokens={tokens} "
        f"latency={latency:.1f}s unknowns_opened={len(out.events)}"
    )
    if out.ok and out.kind == "B":
        paths = [e.path for e in out.value.entrypoints]  # type: ignore[union-attr]
        line += f" entrypoints={len(paths)} {paths}"
    print(line)
    v: Any = out.value
    if out.ok and out.kind == "A":
        pkgs = [p for p in (v.python, *v.packages) if p is not None]
        print(
            "  packages: " + (", ".join(f"{p.name}={p.version}({p.source})" for p in pkgs) or "-")
        )
    elif out.ok and out.kind == "B":
        for e in v.entrypoints:
            print(
                f"  {e.path} kind={e.kind} args={list(e.args_used)} example={e.example is not None}"
            )
    elif out.ok and out.kind == "C":
        print(
            f"  train={list(v.train_scripts)} eval={list(v.eval_scripts)} "
            f"configs={list(v.configs)} datasets={[d.name for d in v.datasets]} "
            f"checkpoints={len(v.checkpoints)} "
            f"metrics={[(m.path, m.metric) for m in v.metric_outputs]}"
        )
    for ev in out.events:
        print(f"  opened {ev.unknown_id}")
    for name in out.dropped_unknowns:
        print(f"  dropped {name} (version cited)")
    for i, err in enumerate(out.errors, start=1):
        label = "rejected" if out.ok else "error"
        print(f"  {label} attempt {i}: {err.splitlines()[0] if err else ''}")


def _cmd_subtasks(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if not (out_dir / "evidence.jsonl").exists():
        print(f"error: no run store at {out_dir}; run `inventory` first", file=sys.stderr)
        return 2
    store = RunStore(out_dir)
    try:
        source, evidence = latest_repo_evidence(store)
        facts, scan = load_repo_facts(evidence)
        if not facts:
            repo_dir = out_dir / "repo"
            if not repo_dir.is_dir():
                raise ValueError(f"no stored repo facts and no clone at {repo_dir}")
            inv_ev = next(ev for ev in evidence.values() if ev.locator == "inventory")
            inv = RepoInventory.model_validate(inv_ev.content)
            print("no stored facts: re-running extractors on the clone", file=sys.stderr)
            for ev in extract_repo_facts(inv, collect_texts(repo_dir, inv), source):
                store.append_evidence(ev)
            source, evidence = latest_repo_evidence(store)
            facts, scan = load_repo_facts(evidence)
        versions = version_evidence(evidence)
        metric_lines = metric_line_evidence(evidence)
        commands = command_evidence(evidence)
        for ev in (*versions, *metric_lines, *commands):
            store.append_evidence(ev)
            evidence[ev.evidence_id] = ev
        # same (kind, locator, id) order as the graph state, so aliases/prompts match it
        evidence = {ev.evidence_id: ev for ev in merge_evidence(evidence.values(), None)}
        facts, scan = load_repo_facts(evidence)
        context, packed = build_subtask_context(evidence, facts, scan, max_chars=args.max_chars)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    existing = {ev.unknown_id for ev in store.load_events() if isinstance(ev, UnknownOpened)}
    budget = Budget(max_tokens_total=args.budget_tokens)
    print(f"source:         {source}")
    n_mentions = sum(len(ev.content["result"]) for ev in versions)
    print(
        f"context:        {len(context['texts'])} texts, prompt chars A/B/C = "
        f"{len(packed['A'])}/{len(packed['B'])}/{len(packed['C'])}"
    )
    print(f"candidates:     {sorted(context['entrypoint_args'])}")
    print(f"versions:       {n_mentions} mention(s) found by code")
    n_lines = sum(len(ev.content["result"]) for ev in metric_lines)
    print(f"metric lines:   {n_lines} found by code")
    n_cmds = sum(len(ev.content["result"]) for ev in commands)
    print(f"commands:       {n_cmds} found by code")
    names = {"A": "A env", "B": "B entrypoints", "C": "C workflow"}
    failed = False
    for kind in SUBTASK_KINDS:
        try:
            llm = llm_for_node(f"subtask_{kind.lower()}", budget)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        llm.on_record = store.append_call_record
        try:
            outcome = run_subtask(
                kind,
                llm,
                context,
                packed[kind],
                source=source,
                existing_unknown_ids=existing,
                max_tokens=args.max_tokens,
            )
        except (BudgetExceeded, LLMError) as e:
            print(f"{names[kind]:<15} aborted: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        for ev in outcome.evidence:
            store.append_evidence(ev)
        for event in outcome.events:
            store.append_event(event)
            existing.add(event.unknown_id)
        records = llm.records()
        tokens = sum(r.input_tokens + r.output_tokens for r in records)
        _print_outcome(names[kind], outcome, tokens, sum(r.latency_s for r in records))
        failed |= not outcome.ok
    print(f"budget:         {budget.used_tokens} tokens used")
    return 1 if failed else 0


UNKNOWN_ORIGINS = ("paper setup", "repo subtasks")


def unknowns_by_origin(
    events: Sequence[UnknownEventT],
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
) -> dict[str, list[str]]:
    """Names of the open unknowns, grouped by where they were first opened: a paper_setup
    record (paper setup) or anything else (repo subtasks). Each group sorted."""
    first: dict[str, UnknownOpened] = {}
    for ev in events:
        if isinstance(ev, UnknownOpened):
            first.setdefault(ev.unknown_id, ev)
    groups: dict[str, list[str]] = {o: [] for o in UNKNOWN_ORIGINS}
    for uid, st in unknowns.items():
        if st.status != "open":
            continue
        kind = evidence[first[uid].evidence_id].kind
        groups["paper setup" if kind == "paper_setup" else "repo subtasks"].append(st.name)
    return {o: sorted(names) for o, names in groups.items()}


def _print_ledger(spec: ReproSpec, store: RunStore, tolerance: ToleranceConfig) -> None:
    """Read-only: ledger recomputed from the store, never written."""
    evidence = store.load_evidence()
    events = store.load_events()
    unknowns = fold_unknowns(events, evidence)
    ledger = compute_ledger(spec.claims, evidence, unknowns, tolerance)
    tol = tolerance.default
    print(f"ledger:         fallback tolerance abs={tol.abs} rel={tol.rel} "
          "(used only without tolerance evidence)")  # fmt: skip
    for claim in spec.claims:
        entry = ledger[claim.claim_id]
        unit = f" {claim.unit}" if claim.unit else ""
        names = [unknowns[uid].name for uid in entry.open_unknowns]
        print(
            f"  {claim.claim_id}  metric={claim.metric}  paper={claim.value}{unit}  "
            f"status={entry.status}  fidelity={entry.fidelity or '-'}  "
            f"attempts={entry.attempts}"
        )
        print(f"    tolerance: {tolerance_used(entry.tolerance, entry.tolerance_evidence_id)}")
        print(f"    open unknowns: {', '.join(names) or '-'}")
    print("open unknowns:")
    for origin, names in unknowns_by_origin(events, evidence, unknowns).items():
        print(f"  {origin + ':':<15} {', '.join(names) or '-'}")


def tolerance_used(tol: Tolerance | None, evidence_id: str | None) -> str:
    """Which tolerance the ledger used: the pre-registered evidence or the --tol-abs fallback."""
    if tol is None:
        return "-"
    where = (
        f"pre-registered tolerance evidence {evidence_id[:12]}"
        if evidence_id
        else "fallback --tol-abs (no tolerance evidence)"
    )
    return f"abs={tol.abs} rel={tol.rel} ({where})"


def _cmd_configure(args: argparse.Namespace) -> int:
    if (args.url is None) == (args.spec is None):
        print("error: give either a GitHub URL or --spec (not both)", file=sys.stderr)
        return 2
    try:
        if args.url is not None:
            parse_github_url(args.url)  # fail fast, before cloning
        if args.commit is not None:
            validate_commit(args.commit)
        tolerance = ToleranceConfig(default=Tolerance(abs=args.tol_abs))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    budget = Budget(max_tokens_total=args.budget_tokens)
    try:  # fail on missing LLM config before any clone or call (constructors do no IO)
        for kind in SUBTASK_KINDS:
            llm_for_node(f"subtask_{kind.lower()}", budget)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    store = RunStore(Path(args.out))
    graph = build_configure_graph(
        {"store": store, "llm_for": lambda node: llm_for_node(node, budget), "budget": budget}
    )
    inputs: dict[str, Any] = {
        "out_dir": args.out,
        "commit": args.commit,
        "max_chars": args.max_chars,
        "max_tokens": args.max_tokens,
        "status": "running",
        "configure_attempts": 0,
    }
    if args.spec is not None:
        inputs["spec_path"] = args.spec
    else:
        inputs["repo_url"] = args.url
    start = time.monotonic()
    try:
        final = graph.invoke(inputs)
    except (CloneError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except GraphRecursionError as e:
        final = {**inputs, **graph_fallback(e)}
    wall = time.monotonic() - start
    if final.get("fallback_reason"):  # default fallback route: nothing half-written
        print(f"status:         {final.get('status')}")
        print_fallback(final)
        return 1
    if "failure_reason" in final:  # ingest_spec failed: nothing cloned or written
        print(f"status:         {final['status']}")
        print(f"error: {final['failure_reason']}", file=sys.stderr)
        return 1
    if "paper_source" in final:
        print(f"paper:          {final['paper_source']} ({args.spec})")
        print(f"repo:           {final['repo_url']}")
    names = {"A": "A env", "B": "B entrypoints", "C": "C workflow"}
    print(f"status:         {final['status']}")
    print(f"source:         {final['source']}")
    for o in sorted(final["outcomes"], key=lambda o: o.kind):
        print(
            f"{names[o.kind]:<15} ok={o.ok} attempts={o.attempts} tokens={o.tokens} "
            f"latency={o.latency_s:.1f}s"
        )
        for i, err in enumerate(o.errors, start=1):
            label = "rejected" if o.ok else "error"
            print(f"  {label} attempt {i}: {err.splitlines()[0] if err else ''}")
    n_state = len(final["evidence"])
    print(f"evidence:       {final['evidence_written']} written ({n_state} in state)")
    print(f"events:         {final['events_written']} written")
    print(f"budget:         {budget.used_tokens} tokens used")
    print(f"wall time:      {wall:.1f}s")
    if "spec" in final:
        _print_ledger(final["spec"], store, tolerance)
    return 0 if final["status"] == "configured" else 1


def graph_fallback(e: BaseException) -> dict[str, str]:
    """A graph-level failure (GraphRecursionError) as a clean fallback stop."""
    return {"fallback_reason": fallback_reason("graph", type(e).__name__),
            "fallback_detail": str(e)[:500]}  # fmt: skip


def print_fallback(final: Mapping[str, Any]) -> None:
    detail = final.get("fallback_detail")
    print(f"stop:           fallback ({final['fallback_reason']}{': ' + detail if detail else ''})")
    print("claims:         not verified by this run; they stay open (re-run to resume)")


# --- env / gpu-check ------------------------------------------------------------------------
# Logic: graph/nodes/env.py (shared with the setup graph). These commands only print.

GPU_CHECK_CODE = (
    "import torch, torchvision; x=torch.ones(256,256,device='cuda'); "
    "print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0), "
    "float((x@x).sum()))"
)


def _plan_env(out: Path, cpu_only: bool) -> PlannedEnv:
    """plan_env with the live host GPU (one name to patch in tests)."""
    return plan_env(out, cpu_only, gpu_query=query_gpu)


def _cmd_env(args: argparse.Namespace) -> int:
    out = Path(args.out)
    try:
        pe = _plan_env(out, args.cpu_only)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    plan = pe.plan
    prep = prepare_env(out, RunStore(out), pe)
    ev = prep.evidence

    print(f"gpu:            {pe.gpu.name if pe.gpu else 'none'}"
          + (f" (cc {pe.gpu.compute_capability[0]}.{pe.gpu.compute_capability[1]}, "
             f"driver {pe.gpu.driver_version})" if pe.gpu else ""))  # fmt: skip
    print(f"imports:        {[p.pip_name for p in pe.scan.packages]}"
          + (f" unparsed={list(pe.scan.unparsed)}" if pe.scan.unparsed else ""))  # fmt: skip
    print(f"base image:     {plan.base_image}")
    print(
        f"torch:          {plan.torch}  torchvision: {plan.torchvision}"
        f"{' (required)' if plan.torchvision_required else ''}  python: {plan.python}  "
        f"cuda: {plan.cuda}"
    )
    print(f"pip:            {list(plan.pip_packages)}")
    for d in plan.deviations:
        print(f"deviation:      {d.field} {d.stated} -> {d.chosen}: {d.reason}")
    for a in plan.assumptions:
        print(f"assumption:     {a.name} = {a.value}: {a.reason}")
    print(f"evidence:       env_plan {ev.evidence_id[:12]} "
          f"({'written' if prep.written else 'exists'})")  # fmt: skip
    assumed = [f"{n}={e.value}" for n, e in zip(prep.assumed_names, prep.assumed, strict=True)]
    print(f"events:         {len(prep.assumed)} assumption(s) written"
          + (f" ({', '.join(assumed)})" if assumed else ""))  # fmt: skip
    print(f"dockerfile:     {prep.dockerfile_path}")
    result, size = build_env_image(
        out, prep, build_timeout=args.build_timeout, builder=build_image, size_of=image_size_bytes
    )
    print(f"image:          {result.tag}")
    print(f"cached:         {result.cached}")
    print(
        f"build time:     {result.duration_s:.1f}s" + (" (timed out)" if result.timed_out else "")
    )
    if size is not None:
        print(f"image size:     {size / 1e9:.2f} GB")
    if result.log_path is not None:
        print(f"build log:      {result.log_path}")
    if not result.ok:
        print("error: image build failed (see build log)", file=sys.stderr)
        return 1
    return 0


def unload_ollama(env: Mapping[str, str] | None = None) -> list[str]:
    """Best-effort: unload every Ollama model configured via REPRODUCER_LLM[_<NODE>] so the GPU
    is free for container runs (D28). Never raises; returns the specs it unloaded."""
    env = os.environ if env is None else env
    specs = sorted(
        {
            v.strip()
            for k, v in env.items()
            if (k == "REPRODUCER_LLM" or k.startswith("REPRODUCER_LLM_")) and v.strip()
        }
    )
    unloaded: list[str] = []
    for spec in specs:
        try:
            if parse_spec(spec)[0] != "ollama":
                continue
            llm = make_llm(spec, env)
            assert isinstance(llm, (OllamaLLM, OllamaOpenAILLM))
            ok = llm.unload()
            print(f"ollama:         unload {spec}: {'ok' if ok else 'not loaded'}", file=sys.stderr)
            unloaded.append(spec)
        except Exception as e:  # best-effort by design
            print(f"warning: ollama unload {spec!r} failed: {e}", file=sys.stderr)
    return unloaded


def _cmd_gpu_check(args: argparse.Namespace) -> int:
    out = Path(args.out)
    try:
        pe = _plan_env(out, cpu_only=False)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not image_exists(pe.tag):
        print(f"error: image {pe.tag} not built; run `env --out {out}` first", file=sys.stderr)
        return 1
    unload_ollama()
    run_dir = new_run_dir(out / "env", "gpu-check")
    workdir = prepare_workdir(out / "repo", [], run_dir / "work")
    try:
        result = run_container(
            pe.tag,
            ["python", "-c", GPU_CHECK_CODE],
            workdir=workdir,
            out_dir=run_dir / "out",
            gpu=True,
            timeout_s=args.timeout,
            log_path=run_dir / "run.log",
            name=f"reproducer-{run_dir.name}",
        )
    finally:
        cleanup_workdir(workdir, keep=args.keep_workdir)
    print(f"image:          {pe.tag}")
    print(f"exit code:      {result.exit_code}  timed_out={result.timed_out} "
          f"oom_killed={result.oom_killed}  duration={result.duration_s:.1f}s")  # fmt: skip
    print(f"log:            {result.log_path}")
    print(f"workdir:        {workdir} ({'kept' if args.keep_workdir else 'removed'})")
    print(result.log_path.read_text(errors="replace").rstrip())
    return 0 if result.exit_code == 0 else 1


# --- data -----------------------------------------------------------------------------------
# Logic: graph/nodes/data.py (shared with the setup graph). This command only prints.


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_data(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not (out / "evidence.jsonl").exists():
        print(f"error: no run store at {out}; run `configure` first", file=sys.stderr)
        return 2
    store = RunStore(out)
    try:
        source, evidence = latest_repo_evidence(store)
        scan, origin = scan_dataset_calls(out, store, source, evidence, _warn)
        datasets = spec_datasets(store, args.spec)
        image = args.image or _plan_env(out, cpu_only=False).tag
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    ds = DataScan(source, scan, origin, tuple(datasets), select_datasets(datasets, scan.calls))
    sel = ds.selection
    print(f"source:         {source}")
    print(f"scan:           {len(scan.calls)} call(s) in {len(scan.scanned)} file(s) ({origin})"
          + (f" unparsed={list(scan.unparsed)}" if scan.unparsed else ""))  # fmt: skip
    for c in scan.calls:
        root = repr(c.root) if c.root is not None else f"None ({c.reason})"
        print(f"  {c.path}:{c.line} {c.cls} root={root} download={c.download}")
    print(f"spec datasets:  {datasets}")
    print(f"selected:       {list(sel.selected)}")
    for w in sel.warnings:
        print(f"warning:        {w}")
    if not image_exists(image):
        print(f"error: image {image} not built; run `env --out {out}` first", file=sys.stderr)
        return 1
    cache = data_cache_dir().resolve()
    print(f"image:          {image}")
    print(f"cache:          {cache}")
    unload_ollama()
    data_dir = out / "data"
    try:
        prov = provision(
            store, sel.selected, image, log_dir=data_dir, timeout_s=args.timeout,
            ensure=ensure_datasets,
        )  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for p in prov:
        st = p.status
        size = sum(int(f["size"]) for f in st.files)
        print(f"  {p.name:<10} {st.state:<10} verified={st.verified} splits={list(st.split_sizes)} "
              f"files={len(st.files)} ({size / 1e6:.1f} MB)"
              + (f" error={st.error}" if st.error else ""))  # fmt: skip
        print(f"             logs: {', '.join(str(p) for p in dict.fromkeys(st.logs))}")
        if p.evidence is not None:
            print(
                f"             evidence: {p.evidence.evidence_id[:12]} "
                f"({'written' if p.written else 'exists'})"
            )
    ok = all(p.status.verified for p in prov)
    plan, status = resolve_mounts(store, ds, prov, cache)
    print("mount plan:")
    for m in plan.mounts:
        print(f"  {m.host} -> {m.container} (ro)  used by {list(m.calls)}")
    for u in plan.unresolved:
        print(f"  unresolved: {u.call}: {u.reason} -> no mount")
    if not plan.mounts and not plan.unresolved:
        print("  - (nothing to mount)")
    print(f"unknown:        {DATASET_SOURCE_UNKNOWN}: {status}")
    print(f"cache size:     {disk_usage(cache) / 1e6:.1f} MB")
    if not ok or not plan.mounts:
        print("sandbox proof:  skipped (no verified dataset with a resolved mount)")
        return 1 if not ok else 0

    names = [p.name for p in prov if p.status.verified]
    proof = sandbox_proof(
        image, names, plan, cache, data_dir=data_dir, timeout_s=args.timeout,
        runner=run_container,
    )  # fmt: skip
    print(f"sandbox proof:  network none, {cache} -> {proof.target} (ro), "
          f"exit={proof.result.exit_code}")  # fmt: skip
    for line in proof.log.splitlines()[-5:]:
        print(f"  {line}")
    return 0 if proof.passed else 1


# --- smoke ----------------------------------------------------------------------------------
# Logic: graph/nodes/smoke.py + critic.py (shared with the setup graph). This command only
# prints; each iteration is persisted by record_iteration before the next one starts.

SMOKE_LOG_TAIL = 30


def _print_smoke_result(res: SmokeResult) -> None:
    print(f"  command:        {res.plan.command}")
    ov = ", ".join(
        f"{o.option} {o.value}"
        + (f" (was {o.previous})" if o.previous is not None else " (appended)")
        for o in res.plan.overrides
    )
    print(f"  overrides:      {ov or '-'}")
    print(f"  signal:         {res.signal}  passed={res.passed}  stage={res.stage}")
    print(f"  exit code:      {res.exit_code}  timed_out={res.timed_out}  "
          f"oom_killed={res.oom_killed}  duration={res.duration_s:.1f}s")  # fmt: skip
    if res.progress_seen or res.timing.sec_per_epoch is not None:
        t = res.timing
        print(
            f"  sec_per_epoch:  {t.sec_per_epoch if t.sec_per_epoch is not None else '-'}"
            f"  (method={t.method}, confidence={t.confidence}"
            f"{', ' + t.detail if t.detail else ''})  progress_seen={res.progress_seen}"
        )
    if res.traceback is not None:
        tb = res.traceback
        rf = tb.repo_frame
        print(f"  traceback:      {tb.exc_type}: {tb.message}  (kind {tb.kind})")
        print(f"  repo frame:     {f'{rf.file}:{rf.line} in {rf.func}' if rf else '-'}")
    print(f"  log:            {res.log_path}")


def _print_critic_calls(calls: Sequence[CallRecord]) -> None:
    print(
        f"  critic llm:     {calls[0].backend}:{calls[0].model} calls={len(calls)} "
        f"tokens in={sum(c.input_tokens for c in calls)} "
        f"out={sum(c.output_tokens for c in calls)} "
        f"latency={sum(c.latency_s for c in calls):.1f}s"
    )


def _print_iteration(n: int, it: Iteration, rec: IterationRecord) -> None:
    res = it.result
    print(f"--- iteration {n} ---")
    _print_smoke_result(res)
    print(f"  evidence:       smoke_result {rec.smoke.evidence_id[:12]} attempts={rec.attempts} "
          f"capped={it.capped} digest={it.input_digest[:12]}")  # fmt: skip
    c = it.classification
    if c is not None:
        print(f"  classification: {c.cls}" + (f"/{c.pattern}" if c.pattern else "")
              + f": {c.reason}")  # fmt: skip
    if it.action is not None and rec.fix is not None:
        dev = it.action.deviation
        dev_text = f" [deviation {dev.field} {dev.stated} -> {dev.chosen}]" if dev else ""
        print(
            f"  fix:            {it.action.kind} {it.action.params}: {it.action.reason}{dev_text}"
        )
        print(f"  evidence:       known_fix {rec.fix.evidence_id[:12]}")
    if it.reverted is not None and rec.revert is not None:
        print(
            f"  reverted:       patch {it.reverted.patch_id[:12]} (same error signature "
            f"after it); evidence critic_revert {rec.revert.evidence_id[:12]}"
        )
    cr = it.critic
    if cr is not None:
        print(f"  critic:         {cr.status} (signature {cr.signature})")
        for err in cr.errors:
            print(f"  gate rejected:  {err[:300]}")
        if cr.patch is not None:
            print(f"  rationale:      {cr.patch.rationale}")
            print(cr.patch.diff.rstrip("\n"))
        elif cr.reason:
            print(f"  reason:         {cr.reason}")
        if rec.critic is not None:
            print(f"  evidence:       critic_patch {rec.critic.evidence_id[:12]}")


def _cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not (out / "evidence.jsonl").exists():
        print(f"error: no run store at {out}; run `configure` first", file=sys.stderr)
        return 2
    store = RunStore(out)
    try:
        setup = prepare_smoke(out, store, args.entrypoint)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    build = make_build(
        setup, out, store, plan_env=_plan_env, image_exists=image_exists, builder=build_image,
        image=args.image, timeout_s=args.timeout, build_timeout=args.build_timeout, log=print,
    )  # fmt: skip
    budget = Budget(max_tokens_total=args.critic_budget_tokens)
    policy = _approval_policy()
    if policy is None:
        return 2
    critic_fn = make_critic_fn(
        lambda: llm_for_node(CRITIC_NODE, budget),
        store,
        clone_dir=setup.repo_dir,
        repo_files=setup.repo_paths,
        metric_texts=setup.metric_texts,
        on_calls=_print_critic_calls,
        approval=policy,
        source=setup.source,
        trigger_ids=(setup.entry_evidence.evidence_id,),
    )
    counter = {"n": 0}

    def record(it: Iteration) -> None:
        counter["n"] += 1
        _print_iteration(counter["n"], it, record_iteration(store, out, setup, it))

    state0, critic0 = setup.fix_state0, setup.critic_state0
    print(f"source:         {setup.source}")
    print(f"entrypoint:     {setup.path}")
    print(f"example:        {setup.example}")
    for m in setup.mount_plan.mounts:
        print(f"data mount:     {m.host} -> {m.container} (ro)")
    for u in setup.mount_plan.unresolved:
        print(f"warning:        unresolved dataset root {u.call}: {u.reason} (no mount)")
    if state0 != FixState():
        print(f"fix state:      {state0.model_dump(mode='json')} (from stored known_fix evidence)")
    if critic0.attempts:
        print(
            f"critic state:   {len(critic0.stack)} patch(es) on the stack, "
            f"{critic0.attempts} attempt(s) (from stored critic_patch evidence)"
        )
    print(
        f"auto-fix:       {args.auto_fix} (max {args.max_fixes} fixes, "
        f"max {args.max_critic} critic attempts)"
    )
    unload_ollama()
    try:
        outcome = smoke_loop(
            build,
            setup.ctx,
            state=state0,
            auto_fix=args.auto_fix,
            max_fixes=args.max_fixes,
            record=record,
            critic=critic_fn if args.auto_fix else None,
            critic_state=critic0,
            max_critic=args.max_critic,
        )
    except (ValueError, LLMError, BudgetExceeded) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    last = outcome.iterations[-1].result
    print("=== result ===")
    print(f"stop:           {outcome.stop}: {outcome.reason}")
    print(f"fix state:      {outcome.state.model_dump(mode='json')}")
    cs = outcome.critic_state
    print(f"critic:         {len(cs.stack)} patch(es) applied, {cs.attempts} attempt(s)")
    if (
        outcome.stop in ("critic", "critic_failed", "no_new_evidence", "max_critic")
        and last.traceback is not None
    ):
        print("unresolved traceback:")
        print(last.traceback.raw)
    elif outcome.stop == "infra":
        print(f"infra failure, stopping (no critic): {outcome.reason}")
    lines = [
        ln for ln in last.log_path.read_text(errors="replace").replace("\r", "\n").splitlines()
        if ln.strip()
    ]  # fmt: skip
    print(f"--- last {min(SMOKE_LOG_TAIL, len(lines))} log lines ({last.log_path}) ---")
    for ln in lines[-SMOKE_LOG_TAIL:]:
        print(ln[:300])
    if outcome.stop == "awaiting_approval":
        print()
        print_queue(out, store)
        return EXIT_AWAITING_APPROVAL
    return 0 if outcome.stop == "passed" else 1


# --- reproduce: the setup graph -------------------------------------------------------------


def setup_deps(store: RunStore, budget: Budget, critic_budget: Budget) -> SetupDeps:
    """Real IO for the setup graph (names looked up here, so tests can patch them)."""
    return {
        "store": store,
        "llm_for": lambda node: llm_for_node(
            node, critic_budget if node == CRITIC_NODE else budget
        ),
        "budget": budget,
        "plan_env": _plan_env,
        "build_image": build_image,
        "image_size": image_size_bytes,
        "image_exists": image_exists,
        "runner": run_container,
        "ensure_datasets": ensure_datasets,
        "data_cache_dir": data_cache_dir,
        "unload": unload_ollama,
        "warn": _warn,
    }


def _progress(node: str, update: Mapping[str, Any]) -> None:
    """One stderr line per finished node (the run can take minutes)."""
    if not isinstance(update, Mapping):
        return
    detail = ""
    if node == "persist":
        detail = str(update.get("status", ""))
    elif node == "env":
        detail = f"{update.get('image_tag', '-')} ok={update.get('env_ok')}"
    elif node == "data":
        detail = f"ok={update.get('data_ok')}"
    elif node == "resume":
        detail = f"reused {', '.join(update.get('resumed', ())) or 'nothing'}"
    elif node == "plan":
        out = update.get("plan_outcome")
        how = "" if out is None else "reused mapping" if out.reused else f"attempts={out.attempts}"
        detail = f"{len(update.get('plans') or [])} plan(s) {how}".rstrip()
    elif node == "claim_worker":
        detail = ", ".join(f"{w.claim_id} {w.status}" for w in update.get("worker_results", ()))
    elif node == "evaluate":
        evals = update.get("evaluations") or {}
        detail = update.get("evaluate_error") or ", ".join(
            f"{cid} {e.status}" for cid, e in evals.items()
        )
    elif node in ("reflect", "apply_lesson"):
        detail = " | ".join(update.get("reflexion_log", ())) or "nothing to do"
    elif node == "vault":
        c = update.get("vault_counts")
        detail = update.get("vault_error") or (
            f"created={len(c.created)} updated={len(c.updated)} unchanged={len(c.unchanged)}"
            if c is not None
            else ""
        )
    for s in update.get("smoke_log", ()):
        detail += (
            f"#{s.n} {s.signal}"
            + (f" [{s.classification.split(':', 1)[0]}]" if s.classification else "")
            + (f" fix={s.fix}" if s.fix else "")
            + (f" critic={s.patch.split(':', 1)[0]}" if s.patch else "")
            + (f" reverted={s.reverted}" if s.reverted else "")
        )
    if update.get("stop_reason"):
        detail += f" -> stop {update['stop_reason']}"
    print(f"[{node}] {detail}".rstrip(), file=sys.stderr, flush=True)


def print_setup_summary(
    final: Mapping[str, Any], records: Sequence[CallRecord], wall: float
) -> None:
    names = {"A": "A env", "B": "B entrypoints", "C": "C workflow"}
    print("=== setup ===")
    if "paper_source" in final:
        print(f"paper:          {final['paper_source']} ({final.get('spec_path')})")
        print(f"repo:           {final['repo_url']}")
    print(f"configure:      {final.get('status')}")
    for o in sorted(final.get("outcomes", ()), key=lambda o: o.kind):
        reused = "  (stored output reused)" if o.ok and o.attempts == 0 else ""
        print(f"  {names[o.kind]:<15} ok={o.ok} attempts={o.attempts} tokens={o.tokens} "
              f"latency={o.latency_s:.1f}s{reused}")  # fmt: skip
    if final.get("resumed"):
        print(f"resumed:        {', '.join(final['resumed'])} (from the store; nothing re-run)")
    for i, env in enumerate(final.get("env_log", ()), start=1):
        label = "env:" if i == 1 else f"env (rebuild {i - 1}):"
        print(f"{label:<15} {env.image}  ok={env.ok} cached={env.cached} "
              f"build={env.build_s:.1f}s")  # fmt: skip
        print(f"  base image:   {env.base_image}  torch={env.torch} python={env.python} "
              f"cuda={env.cuda}")  # fmt: skip
        print(f"  pip:          {list(env.pip)}")
        if env.packages_from_fixes:
            print(f"  from fixes:   {list(env.packages_from_fixes)}")
        for d in env.deviations:
            print(f"  deviation:    {d}")
        for a in env.assumptions:
            print(f"  assumption:   {a}")
        if env.unpinned:
            print(f"  unpinned:     {', '.join(env.unpinned)} (no version stated; pip resolves)")
    data = final.get("data")
    if data is not None:
        print(f"data:           ok={data.ok} selected={list(data.selected)} proof={data.proof}")
        for d in data.datasets:
            print(f"  dataset:      {d}")
        for m in data.mounts or ("- (nothing to mount)",):
            print(f"  mount:        {m}")
        for u in data.unresolved:
            print(f"  unresolved:   {u}")
        for w in data.warnings:
            print(f"  warning:      {w}")
        print(f"  dataset_source: {data.dataset_source}")
    log = final.get("smoke_log", ())
    print(f"smoke:          {len(log)} iteration(s)")
    for s in log:
        print(f"  #{s.n} signal={s.signal} passed={s.passed} stage={s.stage} "
              f"duration={s.duration_s:.1f}s capped={s.capped} "
              f"evidence={s.smoke_evidence}")  # fmt: skip
        if s.classification:
            print(f"     classification: {s.classification}")
        if s.fix:
            print(f"     fix:            {s.fix}")
        if s.reverted:
            print(f"     reverted:       patch {s.reverted}")
        if s.patch:
            print(f"     critic:         {s.patch}")
        print(f"     log:            {s.log}")
    print(
        f"counters:       known_fixes_used={final.get('known_fixes_used', 0)}/{MAX_KNOWN_FIXES}"
        f"  critic_attempts={final.get('critic_attempts', 0)}/{MAX_CRITIC_ATTEMPTS}"
        f"  env_builds={final.get('env_builds', 0)}"
    )
    status = final_setup_status(final)
    reason = (
        final.get("fallback_reason")
        or final.get("stop_reason")
        or (f"--until {final.get('until')}" if status == "stopped" else final.get("failure_reason"))
    )
    detail = final.get("fallback_detail") if final.get("fallback_reason") else None
    detail = detail or final.get("stop_detail")
    print(f"stop:           {status} ({reason}{': ' + detail if detail else ''})")
    tokens = token_totals(records)
    print(f"tokens:         local={tokens['local']} api={tokens['api']} "
          f"calls={len(records)}")  # fmt: skip
    print(f"wall time:      {wall:.1f}s")


class Counted:
    """Wraps an IO callable and counts its calls (container runs, image builds)."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.fn(*args, **kwargs)


def print_pipeline(final: Mapping[str, Any], out_dir: Path, store: RunStore) -> None:
    """Plan line, per-claim pre-flight / run lines, ledger, vault (whatever the run reached)."""
    out = final.get("plan_outcome")
    if out is None and final.get("setup_status") != "plan_failed":
        return
    print("\n=== plan ===")
    if out is not None:
        how = "reused stored mapping (no LLM call)" if out.reused else f"attempts={out.attempts}"
        print(f"mapping:        ok={out.ok} {how} digest={out.input_digest[:12]}")
        order = {cid: i for i, cid in enumerate(final.get("plans") or [])}
        for p in sorted(out.plans, key=lambda p: order.get(p.claim.claim_id, len(order))):
            print(f"  {p.claim.claim_id}: {p.command}  [{p.fidelity}, "
                  f"run_plan {p.evidence.evidence_id[:12]}]")  # fmt: skip
    for cid, why in (final.get("plan_failures") or {}).items():
        print(f"  {cid}: not planned (stays open): plan_failed: {why}")
    if final.get("setup_status") == "plan_failed":
        print(f"plan failed:    {final.get('stop_detail')}")
        return
    results: list[WorkerResult] = final.get("worker_results") or []
    if results:
        order = {cid: i for i, cid in enumerate(final.get("plans") or [])}
        print("\n=== runs (pre-flight + execute, one worker per claim) ===")
        for w in sorted(results, key=lambda w: order.get(w.claim_id, len(order))):
            if w.error is not None:
                print(f"\nclaim {w.claim_id}\n  error:        {w.error}")
            for r in w.runs:
                print_claim_run(r, out_dir)
    if "evaluations" in final:
        print()
        claims = [c for c, _ in paper_claims(store.load_evidence()).values()
                  if c.claim_id in final["evaluations"]]  # fmt: skip
        print_ledger_report(out_dir, store, claims, final["evaluations"],
                            final.get("plan_failures"))  # fmt: skip
        if final.get("reflect"):
            print_reflexion(final, out_dir, store)
    elif final.get("evaluate_error"):
        print(f"\nevaluate error: {final['evaluate_error']}")
    va = final.get("vault_approval")
    if va is not None and not va.proceed:
        rid = va.request.evidence_id[:12] if va.request is not None else "-"
        what = (f"awaiting_approval (request {rid}: {va.reason})" if va.status == "pending"
                else f"{va.rejected_reason} (request {rid})")  # fmt: skip
        print(f"\nvault:          {what}; nothing written")
    counts = final.get("vault_counts")
    if counts is not None:
        print(f"\nvault:          created={len(counts.created)} updated={len(counts.updated)} "
              f"unchanged={len(counts.unchanged)}")  # fmt: skip
        print(f"paper note:     {final.get('vault_paper_note')}")
    elif final.get("vault_error"):
        print(f"\nvault error:    {final['vault_error']}")


def print_reflexion(final: Mapping[str, Any], out_dir: Path, store: RunStore) -> None:
    """Per claim the trial history: trial, outcome, value, the lesson (step, change, what code
    did with it); then why the loop ended for the claim (routing.decide_reflect)."""
    try:
        claims, evals = evaluate_store(store)
        diags, _ = diagnose(
            out_dir, store, claims, evals, gpu_h_cap=caps_from_env(os.environ, 0.0).paper_h,
        )  # fmt: skip
    except ValueError as e:
        print(f"\nreflexion:      cannot summarize: {e}")
        return
    evidence = store.load_evidence()
    decisions = reflect_decisions(final)
    print("\n=== reflexion (trial history per claim) ===")
    for line in final.get("reflexion_log") or ():
        print(f"  {line}")
    for d in diags:
        cid, claim = d.claim.claim_id, d.claim
        if final.get("only_claim") not in (None, cid):
            continue
        unit = f" {claim.unit}" if claim.unit else ""
        dec = metric_decimals(claim)
        state, why = decisions.get(cid, ("done" if evals[cid].status == "verified" else "open",
                                         evals[cid].status))  # fmt: skip
        print(f"\nclaim {cid}  (paper {format_metric(claim.value, dec)}{unit}; "
              f"{evals[cid].status}; loop: {state}{': ' + why if why else ''})")  # fmt: skip
        lessons = lessons_by_trial(evidence, cid)
        causes = trial_causes(evidence, cid)
        for k, r in enumerate(d.runs, start=1):
            o = r.outcome
            val = "-" if o.value is None else f"{format_metric(o.value, dec)}{unit}"
            reeval = " (reeval, 0 GPU h)" if r.run.content.get("reeval_of") else ""
            print(f"  trial {k}: {o.cls:<16} {val}{reeval}")
            for ev in causes.get(k, ()):
                print(f"    caused by:  lesson_applied {ev.evidence_id[:12]} of "
                      f"{ev.content.get('claim_id')} ({ev.content.get('detail')})")  # fmt: skip
            shown = [les for les in lessons.get(k, ())
                     if applied_of(evidence, les.evidence_id) is not None]  # fmt: skip
            if k in lessons and lessons[k][-1] not in shown:
                shown.append(lessons[k][-1])
            for les in shown:
                ap = applied_of(evidence, les.evidence_id)
                content = les.content["lesson"]
                done = f" ({ap.content.get('outcome')})" if ap is not None else ""
                print(f"    lesson:     {content['step']} "
                      f"{change_text(content.get('proposed_change'))}{done}")  # fmt: skip
        if not d.runs:
            print("  no runs")


def pipeline_failed(final: Mapping[str, Any]) -> bool:
    """A claim that did not plan, a worker error, a skipped claim or a run that did not
    complete; or evaluate / vault errors. A full run a human rejected is not a failure."""
    only = final.get("only_claim")
    if any(only in (None, cid) for cid in final.get("plan_failures") or {}):
        return True
    for w in final.get("worker_results") or ():
        if w.error is not None:
            return True
        for r in w.runs:
            if r.status == "skipped" or (r.report is not None and not r.report.completed):
                return True
    return bool(final.get("evaluate_error") or final.get("vault_error"))


def _cmd_reproduce(args: argparse.Namespace) -> int:
    try:
        if args.commit is not None:
            validate_commit(args.commit)
        tolerance = ToleranceConfig(default=Tolerance(abs=args.tol_abs))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.claim is not None:
        try:
            ids = [c.claim_id for c in ReproSpec.model_validate_json(
                Path(args.spec).read_text(encoding="utf-8")).claims]  # fmt: skip
        except (OSError, ValueError) as e:
            print(f"error: cannot read spec {args.spec}: {e}", file=sys.stderr)
            return 2
        if args.claim not in ids:
            print(f"error: unknown claim {args.claim!r} (spec claims: {', '.join(ids)})",
                  file=sys.stderr)  # fmt: skip
            return 2
    if _approval_policy() is None:
        return 2
    full = args.until in PIPELINE_STAGES
    budget = Budget(max_tokens_total=args.budget_tokens)
    critic_budget = Budget(max_tokens_total=args.critic_budget_tokens)
    # fail on missing LLM config before any clone or call (critic and orchestrator: lazily,
    # only when a call is needed; a reused mapping needs no orchestrator config)
    try:
        for kind in SUBTASK_KINDS:
            llm_for_node(f"subtask_{kind.lower()}", budget)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    try:
        with run_lock(out_dir):
            return _reproduce_locked(args, out_dir, full, budget, critic_budget, tolerance)
    except RunLocked as e:
        print(f"error: {e}; wait for it to finish (one writer per store)", file=sys.stderr)
        return 3


def _reproduce_locked(
    args: argparse.Namespace,
    out_dir: Path,
    full: bool,
    budget: Budget,
    critic_budget: Budget,
    tolerance: ToleranceConfig,
) -> int:
    store = RunStore(out_dir)
    n_calls = len(store.load_call_records())
    gpu0 = spent_gpu_hours(store.load_evidence().values())
    deps = setup_deps(store, budget, critic_budget)
    runner, builder = Counted(deps["runner"]), Counted(deps["build_image"])
    policy = policy_from_env(os.environ)  # validated by _cmd_reproduce
    deps.update(
        runner=runner, build_image=builder,
        make_critic=_run_critic_factory(store, critic_budget, policy), env=os.environ, log=print,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )  # fmt: skip
    graph = build_setup_graph(deps)
    inputs: dict[str, Any] = {
        "out_dir": args.out,
        "spec_path": args.spec,
        "commit": args.commit,
        "max_chars": args.max_chars,
        "max_tokens": args.max_tokens,
        "status": "running",
        "configure_attempts": 0,
        "until": args.until,
        "no_vault": args.no_vault,
        "only_claim": args.claim,
        "reflect": not args.no_reflect,
        "vault_dir": str(vault_root(out_dir, None)),
        "no_checkpoints": args.no_checkpoints,
        "cpu_only": False,
        "entrypoint": args.entrypoint,
        "smoke_timeout_s": args.timeout,
        "build_timeout_s": args.build_timeout,
        "data_timeout_s": args.data_timeout,
        "setup_status": "running",
        "known_fixes_used": 0,
        "smoke_iterations": 0,
        "env_builds": 0,
    }
    start = time.monotonic()
    final: dict[str, Any] = dict(inputs)
    try:
        for mode, chunk in graph.stream(
            inputs,
            config={"recursion_limit": SETUP_RECURSION_LIMIT},
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final = chunk
            else:
                for node, update in chunk.items():
                    _progress(node, update)
    except (CloneError, FileExistsError, ValueError, OSError, DockerError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    except GraphRecursionError as e:
        final = {**final, **graph_fallback(e)}
    wall = time.monotonic() - start
    records = store.load_call_records()[n_calls:]
    print_setup_summary(final, records, wall)
    if full:
        print_pipeline(final, out_dir, store)
    elif "spec" in final:
        _print_ledger(final["spec"], store, tolerance)
    if full:
        tokens = token_totals(records)
        gpu = spent_gpu_hours(store.load_evidence().values())
        reached = "" if args.until == "vault" else f" (stopped: --until {args.until})"
        print(f"\n=== totals (this invocation){reached} ===")
        print(f"GPU hours:      {gpu - gpu0:.3f} this run, {gpu:.3f} paper total")
        print(f"LLM:            {tokens['calls']} call(s), api={tokens['api']} "
              f"local={tokens['local']} tokens")  # fmt: skip
        print(f"containers:     {runner.calls} run(s), {builder.calls} image build call(s)")
        print(f"wall time:      {wall:.1f}s")
    status = final_setup_status(final)
    if status == "fallback":  # default fallback route (summary + ledger above)
        print()
        print_fallback(final)
        return 1
    if awaiting_approval(final):
        print("\n=== awaiting approval (final status awaiting_approval) ===")
        print_queue(out_dir, store)
        print("\nre-run the same `reproduce` command after answering: it resumes from the store")
        return EXIT_AWAITING_APPROVAL
    if full and status == "ready":
        return 1 if pipeline_failed(final) else 0
    return 0 if status in ("ready", "stopped") else 1


# --- plan (orchestrator) --------------------------------------------------------------------


def print_plan(out: OrchestratorOutcome, calls: Sequence[CallRecord]) -> None:
    """Per claim: command, args with sources, unknowns, metric source, tolerance; then the
    setup checks, unmapped setup fields and LLM usage."""
    t = out.target
    if t is not None:
        print(f"entrypoint:     {t.entrypoint} ({len(t.commands)} verified command(s))")
    if out.train_config is not None:
        n = len(out.train_config.content.get("result", []))
        print(f"train config:   {n} line(s) in {out.train_config.content.get('scanned')}")
    how = "reused stored mapping (no LLM call)" if out.reused else f"attempts={out.attempts}"
    print(f"mapping:        ok={out.ok} {how} digest={out.input_digest[:12]}")
    for i, err in enumerate(out.errors, start=1):
        print(f"  rejected/error {i}: {err.splitlines()[0] if err else ''}")
    for p in out.plans:
        c, plan = p.claim, p.plan
        unit = f" {c.unit}" if c.unit else ""
        print(f"\nclaim {c.claim_id}  ({c.metric} = {c.value}{unit}, {c.dataset})")
        print(f"  base command: {plan.base_command}")
        print(f"  command:      {p.command}")
        for a in plan.args:
            print(f"  arg           {a.flag} {a.value or ''}".rstrip() + f"  [{a.source}]")
        print(f"  depends on:   {', '.join(plan.depends_on_unknowns) or '-'}")
        ms = plan.metric_source
        print(
            f"  metric:       {ms.pattern}  which={ms.which}  conversion={ms.conversion}  "
            f"(evidence {ms.metric_line_evidence_id[:12]})"
        )
        print(f"  tolerance:    {tolerance_summary(c)}  [{p.tolerance.evidence_id[:12]}]")
        print(f"  fidelity:     {p.fidelity}  run_plan {p.evidence.evidence_id[:12]}")
        for n in p.notes:
            print(f"    {n.status:<10} {n.field}: {n.detail}")
    if out.mapping is not None:
        print("\nsetup checks:")
        for s in out.mapping.setup_checks:
            mark = "consistent" if s.consistent else "INCONSISTENT"
            print(f"  {mark:<12} {s.setup_name}: {s.note}")
            cit = s.repo_citation
            print(f"               cites {cit.evidence_id[:12]}: {cit.quote!r}")
        print("unmapped setup:")
        for u in out.mapping.unmapped_setup:
            print(f"  {u.setup_name}: {u.reason}")
    skipped = f"; skipped {list(out.skipped_assumptions)}" if out.skipped_assumptions else ""
    print(f"\nassumptions:    {len(out.events_written)} event(s) written{skipped}")
    tokens = sum(r.input_tokens + r.output_tokens for r in calls)
    latency = sum(r.latency_s for r in calls)
    models = sorted({r.model for r in calls})
    print(f"llm:            {len(calls)} call(s) {models} tokens={tokens} latency={latency:.1f}s")


def _cmd_plan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if not (out_dir / "evidence.jsonl").is_file():
        print(f"error: no store in {out_dir}: run `configure --spec` first", file=sys.stderr)
        return 2
    budget = Budget(max_tokens_total=args.budget_tokens)
    try:
        llm = llm_for_node(ORCHESTRATOR_NODE, budget)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    store = RunStore(out_dir)
    llm.on_record = store.append_call_record
    try:
        out = run_orchestrator(
            store, llm, repo_dir=out_dir / "repo", max_tokens=args.max_tokens,
            max_retries=args.max_retries, replan=args.replan,
        )  # fmt: skip
    except (ValueError, LLMError, BudgetExceeded) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print_plan(out, llm.records())
    return 0 if out.ok else 1


# --- run (pre-flight + execute) -------------------------------------------------------------


def _run_critic_factory(store: RunStore, budget: Budget, policy: ApprovalPolicy) -> Any:
    def make(setup: Any) -> Any:
        return make_critic_fn(
            lambda: llm_for_node(CRITIC_NODE, budget),
            store,
            clone_dir=setup.repo_dir,
            repo_files=setup.repo_paths,
            metric_texts=setup.metric_texts,
            on_calls=_print_critic_calls,
            approval=policy,
            source=setup.source,
            trigger_ids=(setup.entry_evidence.evidence_id,),
        )

    return make


def print_claim_run(r: ClaimRun, out_dir: Path) -> None:
    print(f"\nclaim {r.claim_id}")
    print(f"  command:      {r.command}")
    print(f"  image:        {r.image}")
    print(f"  run digest:   {r.run_digest[:12]}")
    pf = r.preflight
    if pf is not None:
        smoke = "reused stored smoke" if pf.smoke_reused else f"smoke loop stop={pf.smoke_stop}"
        sid = pf.smoke_evidence_id[:12] if pf.smoke_evidence_id else "-"
        print(f"  preflight:    smoke passed={pf.smoke_passed} ({smoke}, evidence {sid})")
        if pf.sec_per_epoch is not None:
            print(
                f"  timing:       {pf.sec_per_epoch:g} s/epoch ({pf.timing_method}) x "
                f"{pf.epochs} epochs + {pf.overhead_s:.0f} s eval/start-up overhead"
            )
        est = f"{pf.estimate_h:.2f} h" if pf.estimate_h is not None else "-"
        c = pf.caps
        print(
            f"  estimate:     {est}  caps: run {c.run_h:g} h, paper {c.paper_h:g} h "
            f"(spent {c.spent_h:.2f} h, left {c.paper_left_h:.2f} h)"
        )
        tmo = f"{pf.timeout_s / 3600:.2f} h" if pf.timeout_s is not None else "-"
        print(f"  decision:     {pf.decision}: {pf.reason}  (timeout {tmo})")
        print(f"  evidence:     preflight {pf.evidence.evidence_id[:12]}")
    if r.spec_faithful is not None:
        print(f"  spec_faithful: {r.spec_faithful}")
    if r.status == "already_completed" and r.existing is not None:
        c = r.existing.content
        print(f"  status:       already completed (run_result {r.existing.evidence_id[:12]}, "
              f"metrics {format_metrics(c.get('metrics') or {}, r.decimals)}); "
              "skipped")  # fmt: skip
    elif r.status == "preflight_only":
        print("  status:       pre-flight only (no full run started)")
    elif r.status == "skipped":
        print("  status:       skipped (see decision)")
    elif r.status == "awaiting_approval":
        g = r.approval
        if g is not None and g.request is not None:
            print(f"  status:       awaiting_approval (request {g.request.evidence_id[:12]}: "
                  f"{g.reason}); nothing executed, see `queue`")  # fmt: skip
        else:
            print("  status:       awaiting_approval (a pre-flight critic patch); see `queue`")
    elif r.status == "rejected" and r.approval is not None:
        print(f"  status:       rejected by human: {r.approval.note or '(no note)'} (request "
              f"{(r.approval.request.evidence_id if r.approval.request else '-')[:12]}); "
              "nothing executed, the claim stays open")  # fmt: skip
    elif r.report is not None:
        rep, res = r.report, r.report.result
        raw = next((f"{k}={format_raw(v)}" for k, v in res.metrics.items()
                    if k.startswith(RAW_PREFIX)), "-")  # fmt: skip
        conv = next((f"{k}={format_metric(v, r.decimals)}" for k, v in res.metrics.items()
                     if not k.startswith(RAW_PREFIX)), "-")  # fmt: skip
        state = "completed" if rep.completed else "NOT completed"
        print(f"  status:       {state}  exit={res.exit_code} timed_out={rep.timed_out} "
              f"oom_killed={rep.oom_killed}")  # fmt: skip
        print(f"  metric:       raw {raw} -> {conv}  ({rep.reading.note})")
        print(f"  duration:     {res.duration:.0f} s  gpu_hours={res.gpu_hours:.3f}")
        print(f"  evidence:     run_result {rep.evidence.evidence_id[:12]} "
              f"spec_faithful={rep.evidence.content['spec_faithful']}")  # fmt: skip
        print(f"  log:          {resolve_ref(out_dir, res.log_ref)}")
        print(f"  artifacts:    {rep.run_dir / 'artifacts'} ({len(rep.artifacts)} file(s))")


def _cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if not (out_dir / "evidence.jsonl").is_file():
        print(f"error: no store in {out_dir}: run `configure --spec` and `plan` first",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        with run_lock(out_dir):
            return _run_locked(args, out_dir)
    except RunLocked as e:
        print(f"error: {e}; wait for it to finish (one `run` per store)", file=sys.stderr)
        return 3


def _run_locked(args: argparse.Namespace, out_dir: Path) -> int:
    policy = _approval_policy()
    if policy is None:
        return 2
    store = RunStore(out_dir)
    budget = Budget(max_tokens_total=args.critic_budget_tokens)
    deps: dict[str, Any] = {
        "runner": run_container,
        "plan_env": _plan_env,
        "image_exists": image_exists,
        "build_image": build_image,
        "unload": unload_ollama,
        "make_critic": _run_critic_factory(store, budget, policy),
        "env": os.environ,
        "smoke_timeout_s": args.timeout,
        "build_timeout_s": args.build_timeout,
    }

    def progress(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    try:
        runs = run_claims(
            out_dir, store, deps, claim_id=args.claim, preflight_only=args.preflight_only,
            no_checkpoints=args.no_checkpoints, log=print, progress=progress,
        )  # fmt: skip
    except (ValueError, OSError, DockerError, LLMError, BudgetExceeded, RunSetupError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for r in runs:
        print_claim_run(r, out_dir)
    pending = [r for r in runs if r.status == "preflight_only"]
    if pending:
        print("\nfull runs (start them yourself; each runs in the sandbox):")
        for r in pending:
            print(f"  .venv/bin/python -m reproducer.cli run --out {out_dir} --claim {r.claim_id}")
    if any(r.status == "awaiting_approval" for r in runs):
        print()
        print_queue(out_dir, store)
        return EXIT_AWAITING_APPROVAL
    # a human rejection is a decision, not a failure: exit 0 (the claim stays open)
    bad = [r for r in runs if r.status == "skipped"
           or (r.report and not r.report.completed)]  # fmt: skip
    return 1 if bad else 0


VERDICT_TEXT = {
    "within_tolerance": "within tolerance",
    "outside_tolerance": "OUTSIDE tolerance",
    "no_completed_run": "no completed run",
}


def tolerance_bound(e: ClaimEvaluation) -> float:
    return max(e.tolerance.abs or 0.0, (e.tolerance.rel or 0.0) * abs(e.paper_value))


def print_evaluation(e: ClaimEvaluation, decimals: int, skip: str | None = None) -> None:
    unit = f" {e.unit}" if e.unit else ""
    bound = tolerance_bound(e)
    fmt = lambda v: format_metric(v, decimals)  # noqa: E731
    print(f"\nclaim {e.claim_id}  ({e.metric})")
    print(f"  paper:        {fmt(e.paper_value)}{unit} ± {fmt(bound)}  "
          f"(tolerance {e.tolerance_evidence_id[:12]}: {e.tolerance_reason})")  # fmt: skip
    if e.our_value is None:
        print("  ours:         -  (no completed run)")
    else:
        raw = ", ".join(f"{k.removeprefix(RAW_PREFIX)}={format_raw(v)}"
                        for k, v in e.our_raw.items()) or "-"  # fmt: skip
        print(f"  ours:         {fmt(e.our_value)}{unit}  (raw {raw}; "
              f"run_result {(e.best_run_evidence_id or '-')[:12]})")  # fmt: skip
    diff = "-" if e.abs_diff is None else fmt(e.abs_diff)
    print(f"  |diff|:       {diff}")
    cmp = (
        ""
        if e.abs_diff is None
        else f" ({diff} {'<=' if e.within_tolerance else '>'} {fmt(bound)})"
    )
    print(f"  verdict:      {VERDICT_TEXT[e.verdict]}{cmp}")
    print(f"  status:       {e.status}")
    fid = e.fidelity or f"- (best run: {e.run_fidelity or '-'})"
    print(f"  fidelity:     {fid}")
    print(f"  runs:         {e.completed_runs} completed / {e.attempts} attempt(s) "
          f"({e.aborted_runs} aborted), {e.gpu_hours:.3f} GPU h")  # fmt: skip
    if skip:
        print(f"  skipped:      {skip}")
    if e.inferred_reasons:
        print("  inferred because:")
        for r in e.inferred_reasons:
            ids = ", ".join(i[:12] for i in r.evidence_ids)
            print(f"    - [{r.kind}] {r.text}" + (f"  [{ids}]" if ids else ""))
    for r in e.info:
        ids = ", ".join(i[:12] for i in r.evidence_ids)
        print(f"  info:         {r.text}" + (f"  [{ids}]" if ids else ""))


def skip_reason(evidence: Mapping[str, Evidence], claim_id: str) -> str | None:
    """The newest pre-flight decision of the claim when it skipped the run (cap / smoke
    failed), or its full run awaits / was refused a human approval, and no run_result was
    stored after it; else None."""
    reason: str | None = None
    for ev in evidence.values():
        c = ev.content
        if ev.kind == "preflight" and c.get("claim_id") == claim_id:
            decision = c.get("decision")
            reason = None if decision == "run" else f"{decision}: {c.get('reason')}"
        elif (ev.kind == "approval_request" and c.get("action") == "full_run"
              and c.get("subject") == claim_id):  # fmt: skip
            dec = decision_for(evidence, ev.evidence_id)
            if dec is None:
                reason = f"awaiting_approval: request {ev.evidence_id[:12]} (see `queue`)"
            elif dec.content.get("decision") == "rejected":
                reason = f"rejected_by_human: {dec.content.get('note') or '(no note)'}"
        elif ev.kind == "run_result" and c.get("claim_id") == claim_id:
            reason = None
    return reason


def print_ledger_report(
    out_dir: Path,
    store: RunStore,
    claims: Sequence[ResultClaim],
    evals: Mapping[str, ClaimEvaluation],
    unplanned: Mapping[str, str] | None = None,
) -> None:
    """The `ledger` printout (also used by `reproduce`): per claim, then totals. unplanned:
    claim_id -> why it did not plan (default: recomputed from the store)."""
    evidence = store.load_evidence()
    records = store.load_call_records()
    if unplanned is None:
        unplanned = stored_plan_failures(store, out_dir)
    print(f"=== ledger: {out_dir} (read-only, recomputed from evidence) ===")
    for c in claims:
        cid = c.claim_id
        skip = f"plan_failed: {unplanned[cid]}" if cid in unplanned else None
        print_evaluation(evals[cid], metric_decimals(c), skip or skip_reason(evidence, cid))
    runs = [ev for ev in evidence.values() if ev.kind == "run_result"]
    tokens = token_totals(records)
    by_status = Counter(e.status for e in evals.values())
    print("\ntotals:")
    print(f"  claims:       {len(claims)} ("
          + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())) + ")")  # fmt: skip
    print(f"  GPU hours:    {spent_gpu_hours(evidence.values()):.3f} "
          f"({len(runs)} run_result(s))")  # fmt: skip
    print(f"  tokens:       api={tokens['api']} local={tokens['local']} "
          f"calls={len(records)}")  # fmt: skip


def _cmd_ledger(args: argparse.Namespace) -> int:
    """Read-only: nothing is written to the store."""
    out_dir = Path(args.out)
    try:
        store = RunStore(out_dir, create=False)
    except NotARunStore as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        claims, evals = evaluate_store(store)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print_ledger_report(out_dir, store, claims, evals)
    return 0


# --- diagnose (M4: run outcome, curve, trial budget; no LLM) ---------------------------------


def _signed(v: float, decimals: int) -> str:
    return ("+" if v >= 0 else "-") + format_metric(abs(v), decimals)


def print_diagnosis(d: ClaimDiagnosis, breaker: BreakerState) -> None:
    c, e = d.claim, d.evaluation
    dec = metric_decimals(c)
    fmt = lambda v: "-" if v is None else format_metric(v, dec)  # noqa: E731
    unit = f" {c.unit}" if c.unit else ""
    fid = f" ({e.fidelity})" if e.fidelity else ""
    print(f"\nclaim {c.claim_id}  ({c.metric})")
    print(f"  ledger:       {e.status}{fid}; paper {fmt(c.value)}{unit} "
          f"± {fmt(tolerance_bound(e))}")  # fmt: skip
    if not d.runs:
        print("  runs:         none")
    for r in d.runs:
        o, rc = r.outcome, r.run.content
        print(f"  run {rc.get('run_id')}  (run_result {r.run.evidence_id[:12]})")
        diff = "-" if o.diff is None else _signed(o.diff, dec)
        print(f"    outcome:    {o.cls}  direction {o.direction or '-'}  ours {fmt(o.value)}{unit}"
              f"  diff {diff}  tol {fmt(o.tol)}")  # fmt: skip
        print(f"    reason:     {o.reason}")
        if o.classification is not None:
            cl = o.classification
            print(f"    classify:   {cl.cls} {cl.pattern or ''}".rstrip())
        cv = r.curve
        if cv is None:
            print("    curve:      - (no run_curve)")
            continue
        print(f"    curve:      {cv.source} {cv.source_ref} ({cv.column} -> {cv.metric}, "
              f"{cv.conversion}; {cv.n_points} points; run_curve "
              f"{(r.curve_evidence_id or '-')[:12]})")  # fmt: skip
        print(f"      final:    {fmt(cv.final)}{unit} @ epoch {cv.final_epoch}   "
              f"best ({cv.direction}): {fmt(cv.best)}{unit} @ epoch {cv.best_epoch}")  # fmt: skip
        slope = "-" if cv.last_k_slope is None else f"{cv.last_k_slope:+.4f}/epoch"
        print(f"      last {cv.k}:  mean {fmt(cv.last_k_mean)} ± {fmt(cv.last_k_std)}, "
              f"slope {slope}")  # fmt: skip
        plateau = "-" if cv.plateau_from_epoch is None else f"epoch {cv.plateau_from_epoch}"
        yes = lambda b: "yes" if b else "no"  # noqa: E731
        print(f"      still improving: {yes(cv.still_improving)}   "
              f"plateau from: {plateau}   nonfinite: {yes(cv.nonfinite)}")  # fmt: skip
    b = breaker.per_claim[c.claim_id]
    can = "yes" if breaker.can_try(c.claim_id) else "no"
    print(f"  budget:       {b.trials}/{b.max_trials} trials used ({b.aborted} aborted), "
          f"{b.remaining} remaining; can try: {can}")  # fmt: skip


def print_breaker(b: BreakerState) -> None:
    print("\nbreaker:")
    print(f"  trials:       {b.global_used}/{b.global_max} used, {b.global_remaining} remaining")
    print(f"  GPU hours:    {b.gpu_h_spent:.3f}/{b.gpu_h_cap:g} spent, "
          f"{b.gpu_h_remaining:.3f} remaining")  # fmt: skip
    print(f"  tokens:       {b.tokens_used}/{b.token_budget} used, {b.tokens_remaining} remaining")
    print(f"  tripped:      {'; '.join(b.tripped) if b.tripped else 'none'}")


def _classify_ctx(out_dir: Path, store: RunStore) -> ClassifyContext | None:
    """The smoke's classify context, only when a run exited non-zero (crash triage)."""
    if not any(ev.kind == "run_result" and isinstance(ev.content.get("exit_code"), int)
               and ev.content["exit_code"] not in (0, -1)
               for ev in store.load_evidence().values()):  # fmt: skip
        return None
    try:
        return prepare_smoke(out_dir, store).ctx
    except (ValueError, OSError):
        return None


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Read-only except the run_curve backfill (append-only)."""
    out_dir = Path(args.out)
    store = _read_store(out_dir)
    if store is None:
        return 2
    try:
        gpu_cap = caps_from_env(os.environ, 0.0).paper_h
        new = backfill_curves(out_dir, store)
        claims, evals = evaluate_store(store)
        diags, breaker = diagnose(
            out_dir, store, claims, evals, gpu_h_cap=gpu_cap,
            token_budget=args.token_budget, ctx=_classify_ctx(out_dir, store),
        )  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"=== diagnose: {out_dir} (no LLM) ===")
    ids = ", ".join(ev.evidence_id[:12] for ev in new)
    print(f"run_curve:      {len(new)} backfilled" + (f" ({ids})" if ids else ""))
    for d in diags:
        print_diagnosis(d, breaker)
    print_breaker(breaker)
    return 0


# --- reflect (M4: Reflexion lesson; proposes, never applies) ----------------------------------


def print_reflection(r: ReflectOutcome) -> None:
    print(f"\nclaim {r.claim_id}")
    if r.status == "skipped":
        print(f"  skipped:      {r.reason}")
        return
    if r.status == "failed":
        print(f"  FAILED:       {r.reason} (trial {r.trial})")
        for i, err in enumerate(r.errors, start=1):
            print(f"  rejected {i}:   {err.splitlines()[0] if err else ''}")
        return
    assert r.evidence is not None
    how = "reused stored lesson (no LLM call)" if r.reused else f"attempts={r.attempts}"
    if r.stale_ids:
        ids = ", ".join(i[:12] for i in r.stale_ids)
        how += f"; stale (other inputs, never applied): {ids}"
    where = "stored" if r.stored or r.reused else "dry run, not stored"
    print(f"  trial:        {r.trial}  {how}  digest={r.input_digest[:12]}")
    if r.forced:
        print("  forced:       the newest run does not trigger reflection (demo only)")
    for i, err in enumerate(r.errors, start=1):
        print(f"  rejected {i}:   {err.splitlines()[0] if err else ''}")
    les = r.evidence.content["lesson"]
    print(f"  lesson:       {r.evidence.evidence_id[:12]} ({where})")
    print(f"  step:         {les['step']}")
    print(f"  evidence:     {', '.join(str(i)[:12] for i in les['evidence_ids'])}")
    print(f"  text:         {les['text']}")
    ch = les.get("proposed_change")
    if ch is None:
        print("  change:       none")
        return
    val = "" if ch.get("new_value") is None else f" -> {ch['new_value']}"
    print(f"  change:       [{ch['kind']}] {ch['target']}{val}")
    print(f"  rationale:    {ch['rationale']}")
    sup = ch.get("support")
    if sup:
        print(f"  support:      {str(sup['evidence_id'])[:12]}: {sup['quote']!r}")


def _cmd_reflect(args: argparse.Namespace) -> int:
    """Appends lesson evidence (and LLM call records) only without --dry-run."""
    if args.force and not args.dry_run:
        print("error: --force requires --dry-run (a forced lesson is never stored)",
              file=sys.stderr)  # fmt: skip
        return 2
    out_dir = Path(args.out)
    store = _read_store(out_dir)
    if store is None:
        return 2
    budget = Budget(max_tokens_total=args.budget_tokens)
    made: list[Any] = []

    def factory() -> Any:
        if not made:
            llm = llm_for_node(REFLECT_NODE, budget)
            if not args.dry_run:
                llm.on_record = store.append_call_record
            made.append(llm)
        return made[0]

    try:
        claims, evals = evaluate_store(store)
        if args.claim is not None and args.claim not in {c.claim_id for c in claims}:
            print(f"error: unknown claim {args.claim!r}", file=sys.stderr)
            return 2
        diags, breaker = diagnose(
            out_dir, store, claims, evals,
            gpu_h_cap=caps_from_env(os.environ, 0.0).paper_h, token_budget=args.token_budget,
        )  # fmt: skip
        mode = "dry run: nothing is stored" if args.dry_run else "stores lesson evidence"
        print(f"=== reflect: {out_dir} ({mode}) ===")
        results = [
            reflect_claim(
                store,
                d,
                breaker,
                llm_factory=factory,
                force=args.force,
                dry_run=args.dry_run,
                max_tokens=args.max_tokens,
            )  # fmt: skip
            for d in diags
            if args.claim is None or d.claim.claim_id == args.claim
        ]
    except (ValueError, LLMError, BudgetExceeded) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for r in results:
        print_reflection(r)
    calls = made[0].records() if made else ()
    tokens = sum(c.input_tokens + c.output_tokens for c in calls)
    latency = sum(c.latency_s for c in calls)
    models = sorted({c.model for c in calls})
    print(f"\nllm:            {len(calls)} call(s) {models} tokens={tokens} "
          f"(in {sum(c.input_tokens for c in calls)}, out {sum(c.output_tokens for c in calls)}) "
          f"latency={latency:.1f}s")  # fmt: skip
    return 1 if any(r.status == "failed" for r in results) else 0


VAULT_ENV = "REPRODUCER_VAULT"


def vault_root(out_dir: Path, arg: str | None, env: Mapping[str, str] | None = None) -> Path:
    """--vault, else $REPRODUCER_VAULT, else <out>/vault_out."""
    env = os.environ if env is None else env
    if arg:
        return Path(arg)
    if env.get(VAULT_ENV, "").strip():
        return Path(env[VAULT_ENV].strip())
    return out_dir / "vault_out"


def _cmd_vault(args: argparse.Namespace) -> int:
    """Renders notes and writes only into the vault; read-only on the store unless the vault
    is REPRODUCER_SHARED_VAULT and the policy gates it (then a standing approval_request is
    filed once: pending -> exit 4, rejected -> nothing written, exit 0)."""
    out_dir = Path(args.out)
    try:
        store = RunStore(out_dir, create=False)
    except NotARunStore as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    policy = _approval_policy()
    if policy is None:
        return 2
    root = vault_root(out_dir, args.vault)
    try:
        notes = render_store_vault(store, out_dir)
        if targets_shared_vault(root, policy):
            g = vault_gate(store, root, notes, policy)
            rid = g.request.evidence_id[:12] if g.request is not None else "-"
            if g.status == "pending":
                print(f"vault:          awaiting_approval (request {rid}: {g.reason}); "
                      "nothing written")  # fmt: skip
                print()
                print_queue(out_dir, store)
                return EXIT_AWAITING_APPROVAL
            if g.status == "rejected":
                print(f"vault:          rejected by human: {g.note or '(no note)'} "
                      f"(request {rid}); nothing written to {root}")  # fmt: skip
                return 0
        paper_note, counts = write_store_vault(store, out_dir, root, notes=notes)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"=== vault: {root} (from {out_dir}, store read-only) ===")
    print(f"notes:          {counts.total} rendered")
    print(f"created:        {len(counts.created)}")
    print(f"updated:        {len(counts.updated)}")
    print(f"unchanged:      {len(counts.unchanged)}")
    print(f"paper note:     {root / paper_note}")
    return 0


# --- human in the loop: queue / approve / reject / answer ------------------------------------

EXIT_AWAITING_APPROVAL = 4
CLI = ".venv/bin/python -m reproducer.cli"


def _approval_policy() -> ApprovalPolicy | None:
    """The policy from the environment (warning when off); None + an error on a bad value."""
    try:
        policy = policy_from_env(os.environ)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    warning = policy_warning(policy)
    if warning:
        print(warning, file=sys.stderr)
    return policy


def print_queue(out_dir: Path, store: RunStore) -> int:
    """The escalation queue: pending approval requests + open unknowns, each with the exact
    command to answer it. Read-only. Returns the number of items."""
    evidence = store.load_evidence()
    events = store.load_events()
    q = escalation_queue(evidence, fold_unknowns(events, evidence))
    opened = {e.unknown_id: e for e in events if isinstance(e, UnknownOpened)}
    out = shlex.quote(str(out_dir))
    print(f"=== escalation queue: {out_dir} ===")
    print(f"approval requests: {len(q.requests)} pending")
    for r in q.requests:
        c, rid = r.content, r.evidence_id[:12]
        print(f"\n  [{rid}] {c.get('action')}  subject={c.get('subject')}  cost={cost_text(c)}")
        print(f"      detail:  {detail_summary(c)}")
        if c.get("reason"):
            print(f"      why:     {c['reason']}")
        if c.get("action") == "critic_patch":
            for ln in str((c.get("detail") or {}).get("diff", "")).rstrip().splitlines():
                print(f"      | {ln}")
        print(f'      approve: {CLI} approve --out {out} {rid} --note "..."')
        print(f'      reject:  {CLI} reject --out {out} {rid} --note "..."')
    print(f"\nopen unknowns: {len(q.unknowns)}")
    for st in q.unknowns:
        uid = f"unk:{st.name}"
        assumed = list(dict.fromkeys(a.value for a in st.assumptions))
        print(f"\n  {st.name}" + (f"  (assumed so far: {', '.join(assumed)})" if assumed else ""))
        src = evidence.get(opened[uid].evidence_id) if uid in opened else None
        if src is not None and src.content.get("quote"):
            print(f"      paper:   {' '.join(str(src.content['quote']).split())[:160]}")
        name = shlex.quote(st.name)
        print(f'      answer:  {CLI} answer --out {out} {name} <value> --note "..."')
    if q.empty:
        print("\nnothing to answer")
    return len(q.requests) + len(q.unknowns)


def _read_store(out_dir: Path) -> RunStore | None:
    try:
        return RunStore(out_dir, create=False)
    except NotARunStore as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def _cmd_queue(args: argparse.Namespace) -> int:
    """Read-only: nothing is written to the store."""
    out_dir = Path(args.out)
    store = _read_store(out_dir)
    if store is None:
        return 2
    try:
        print_queue(out_dir, store)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _decide(args: argparse.Namespace, decision: str) -> int:
    out_dir = Path(args.out)
    store = _read_store(out_dir)
    if store is None:
        return 2
    try:
        evidence = store.load_evidence()
        req = resolve_request_id(evidence, args.request_id, decision)  # type: ignore[arg-type]
        revision = len(decisions_for(evidence, req.evidence_id))
        ev = approval_decision_evidence(req, decision, args.note or "",  # type: ignore[arg-type]
                                        revision=revision)  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    store.append_evidence(ev)
    c = req.content
    if revision:
        verb = "revoked" if decision == "rejected" else "restored"
        print(f"{verb}:        standing approval of {c.get('subject')} ({decision} again)")
    print(
        f"{decision}:       request {req.evidence_id[:12]} ({c.get('action')} "
        f"{c.get('subject')}) -> approval_decision {ev.evidence_id[:12]} (authority human)"
    )
    print(f"next:           re-run `reproduce` (or `run`) on {out_dir}: it resumes and proceeds")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    return _decide(args, "approved")


def _cmd_reject(args: argparse.Namespace) -> int:
    return _decide(args, "rejected")


def _cmd_answer(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    store = _read_store(out_dir)
    if store is None:
        return 2
    try:
        ans = answer_unknown(store.load_events(), store.load_evidence(), args.unknown,
                             args.value, args.note)  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if ans.event is None:
        print(f"unchanged:      {args.unknown} is already resolved to {args.value!r}; "
              "nothing written")  # fmt: skip
        return 0
    store.append_evidence(ans.evidence)
    store.append_event(ans.event)
    print(f"resolved:       {ans.event.unknown_id} = {args.value!r} by human_input "
          f"{ans.evidence.evidence_id[:12]} (authority human, observed)")  # fmt: skip
    print(f"next:           `ledger --out {out_dir}` shows it; re-run `reproduce` to use it")
    return 0


# --- eval-report (Lecture 6: outcome / trajectory / efficiency over every paper store) -------


def load_manifest(path: Path) -> list[dict[str, str]]:
    """The cases of evals/manifest.json: {name, spec, out, note} each (out required)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f'{path}: expected {{"cases": [...]}}')
    out: list[dict[str, str]] = []
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or not c.get("name") or not c.get("out"):
            raise ValueError(f"{path}: case {i} needs a name and an out")
        out.append({k: str(c.get(k) or "") for k in ("name", "spec", "out", "note")})
    return out


def case_report(case: Mapping[str, str], env: Mapping[str, str]) -> PaperReport:
    """One case, read-only: RunStore(create=False) never creates or writes a file."""
    try:
        store = RunStore(Path(case["out"]), create=False)
    except NotARunStore:
        return not_run(case, f"no store at {case['out']}")
    try:
        evidence = store.load_evidence()
        unknowns = fold_unknowns(store.load_events(), evidence)
        calls = store.load_call_records()
        claims = tuple(c for c, _ in paper_claims(evidence).values())
        return paper_report(
            case, claims, evidence, unknowns, calls,
            gpu_h_cap=caps_from_env(env, 0.0).paper_h,
            token_budget=token_budget_from_env(env), max_trials=max_trials_from_env(env),
        )  # fmt: skip
    except ValueError as e:
        return PaperReport(**{k: case.get(k, "") for k in ("name", "spec", "out", "note")},
                           status="error", reason=str(e))  # fmt: skip


def _cmd_eval_report(args: argparse.Namespace) -> int:
    """Read-only on every store; writes only <--out>/report.md and report.json."""
    try:
        cases = load_manifest(Path(args.manifest))
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report = build_report([case_report(c, os.environ) for c in cases])
    md = render_markdown(report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(report_json(report), encoding="utf-8")
    print(md, end="")
    print(f"\nwrote {out / 'report.md'} and {out / 'report.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reproducer")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="clone a repo and record a read-only inventory")
    inv.add_argument("url", help="GitHub repository URL")
    inv.add_argument("--commit", default=None, help="full 40-char commit sha to check out")
    inv.add_argument("--out", required=True, help="run folder, e.g. runs/clip")
    inv.add_argument(
        "--reuse-clone",
        action="store_true",
        help="if <out>/repo exists, verify it is a git checkout (at --commit if given) "
        "and rebuild the inventory from it instead of cloning",
    )
    inv.add_argument("--timeout", type=int, default=300, help="clone timeout in seconds")
    inv.add_argument(
        "--max-py-files",
        type=int,
        default=50,
        help="max Python files scanned for entrypoints (train/eval/main/run names first)",
    )
    inv.set_defaults(func=_cmd_inventory)

    st = sub.add_parser("subtasks", help="run the gated LLM subtasks A/B/C on a run folder")
    st.add_argument("--out", required=True, help="run folder created by `inventory`")
    st.add_argument("--max-chars", type=int, default=12_000, help="packed prompt size cap")
    st.add_argument("--max-tokens", type=int, default=1_500, help="max output tokens per call")
    st.add_argument(
        "--budget-tokens", type=int, default=200_000, help="token cap shared by A, B and C"
    )
    st.set_defaults(func=_cmd_subtasks)

    cf = sub.add_parser("configure", help="run the configure graph (inventory + A/B/C)")
    cf.add_argument("url", nargs="?", default=None, help="GitHub repository URL (or --spec)")
    cf.add_argument(
        "--spec",
        default=None,
        help="ReproSpec JSON, e.g. evals/cutout/spec.json; its first GitHub code_link is the "
        "repo, and the ledger is printed after persist",
    )
    cf.add_argument(
        "--tol-abs",
        type=float,
        default=0.5,
        help="fallback absolute tolerance, used only for claims without tolerance evidence",
    )
    cf.add_argument("--commit", default=None, help="full 40-char commit sha to check out")
    cf.add_argument("--out", required=True, help="run folder; <out>/repo is reused if present")
    cf.add_argument("--max-chars", type=int, default=12_000, help="packed prompt size cap")
    cf.add_argument("--max-tokens", type=int, default=1_500, help="max output tokens per call")
    cf.add_argument(
        "--budget-tokens", type=int, default=200_000, help="token cap shared by A, B and C"
    )
    cf.set_defaults(func=_cmd_configure)

    en = sub.add_parser("env", help="plan the Docker environment and build its image")
    en.add_argument("--out", required=True, help="run folder created by `configure`")
    en.add_argument("--cpu-only", action="store_true", help="plan without a GPU")
    en.add_argument("--build-timeout", type=float, default=3600, help="build timeout (s)")
    en.set_defaults(func=_cmd_env)

    gc = sub.add_parser("gpu-check", help="run a torch matmul on the GPU in the built image")
    gc.add_argument("--out", required=True, help="run folder with a built image (`env`)")
    gc.add_argument("--timeout", type=float, default=300, help="container timeout (s)")
    gc.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the per-run working copy (default: deleted after the run; logs and /out "
        "are always kept)",
    )
    gc.set_defaults(func=_cmd_gpu_check)

    da = sub.add_parser("data", help="provision spec datasets and prove the read-only mount")
    da.add_argument("--out", required=True, help="run folder with a built image (`env`)")
    da.add_argument(
        "--spec", default=None, help="ReproSpec JSON (default: datasets of stored paper claims)"
    )
    da.add_argument("--image", default=None, help="image to use (default: the planned env tag)")
    da.add_argument("--timeout", type=float, default=3600, help="download timeout (s)")
    da.set_defaults(func=_cmd_data)

    sm = sub.add_parser("smoke", help="first sandboxed run of the repo's verified command")
    sm.add_argument("--out", required=True, help="run folder after `env` and `data`")
    sm.add_argument("--entrypoint", default=None, help="entrypoint path (default: train one)")
    sm.add_argument("--image", default=None, help="image to use (default: the planned env tag)")
    sm.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="repo run timeout (s)")
    sm.add_argument(
        "--auto-fix",
        action="store_true",
        help="apply known zero-token fixes and grounded critic patches, re-smoke after each "
        "(stops at infra / unfixable / a cap)",
    )
    sm.add_argument("--max-fixes", type=int, default=5, help="max known fixes per invocation")
    sm.add_argument(
        "--max-critic",
        type=int,
        default=MAX_CRITIC_ATTEMPTS,
        help="max critic LLM attempts per repo (stored attempts count)",
    )
    sm.add_argument(
        "--critic-budget-tokens", type=int, default=60_000, help="token cap for the critic"
    )
    sm.add_argument("--build-timeout", type=float, default=3600, help="image rebuild timeout (s)")
    sm.set_defaults(func=_cmd_smoke)

    rp = sub.add_parser(
        "reproduce",
        help="the whole pipeline: configure -> env -> data -> smoke -> plan -> per-claim "
        "pre-flight + run -> evaluate -> vault",
    )
    rp.add_argument("--spec", required=True, help="ReproSpec JSON, e.g. evals/cutout/spec.json")
    rp.add_argument("--out", required=True, help="run folder; <out>/repo is reused if present")
    rp.add_argument(
        "--until",
        choices=UNTIL_STAGES,
        default="vault",
        help="last stage to run (default: vault, i.e. the whole pipeline; smoke = setup only)",
    )
    rp.add_argument("--no-vault", action="store_true", help="skip writing the vault notes")
    rp.add_argument(
        "--claim",
        default=None,
        help="plan every claim, but run / evaluate / reflect only this claim_id (the vault is "
        "still written from the whole store)",
    )
    rp.add_argument(
        "--no-reflect",
        action="store_true",
        help="skip the Reflexion loop (default: a claim out of tolerance / diverged gets a "
        "lesson, its grounded change is applied and the claim re-run, within the breaker)",
    )
    rp.add_argument(
        "--no-checkpoints", action="store_true", help="full runs: do not keep *.pt / *.pth"
    )
    rp.add_argument("--commit", default=None, help="full 40-char commit sha to check out")
    rp.add_argument("--entrypoint", default=None, help="entrypoint path (default: train one)")
    rp.add_argument(
        "--tol-abs",
        type=float,
        default=0.5,
        help="fallback absolute tolerance, used only for claims without tolerance evidence",
    )
    rp.add_argument("--max-chars", type=int, default=12_000, help="packed prompt size cap")
    rp.add_argument("--max-tokens", type=int, default=1_500, help="max output tokens per call")
    rp.add_argument(
        "--budget-tokens", type=int, default=200_000, help="token cap shared by A, B and C"
    )
    rp.add_argument(
        "--critic-budget-tokens", type=int, default=60_000, help="token cap for the critic"
    )
    rp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="smoke timeout (s)")
    rp.add_argument("--build-timeout", type=float, default=3600, help="image build timeout (s)")
    rp.add_argument("--data-timeout", type=float, default=3600, help="dataset download timeout (s)")
    rp.set_defaults(func=_cmd_reproduce)

    pl = sub.add_parser("plan", help="map claims onto the repo and build one RunPlan per claim")
    pl.add_argument("--out", required=True, help="run folder configured with `configure --spec`")
    pl.add_argument(
        "--max-tokens", type=int, default=ORCH_MAX_TOKENS, help="max output tokens per call"
    )
    pl.add_argument("--max-retries", type=int, default=2, help="gate retries after a rejection")
    pl.add_argument("--budget-tokens", type=int, default=60_000, help="token cap for the plan")
    pl.add_argument(
        "--replan",
        action="store_true",
        help="ask the LLM again even if a mapping for the same inputs is stored",
    )
    pl.set_defaults(func=_cmd_plan)

    rn = sub.add_parser("run", help="pre-flight and run the current plan of each claim")
    rn.add_argument("--out", required=True, help="run folder with current plans (`plan`)")
    rn.add_argument("--claim", default=None, help="only this claim_id")
    rn.add_argument(
        "--preflight-only", action="store_true", help="smoke + estimate + caps; no full run"
    )
    rn.add_argument(
        "--no-checkpoints", action="store_true", help="do not keep *.pt / *.pth artifacts"
    )
    rn.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="pre-flight smoke timeout (s)"
    )
    rn.add_argument("--build-timeout", type=float, default=3600, help="image rebuild timeout (s)")
    rn.add_argument(
        "--critic-budget-tokens", type=int, default=60_000, help="token cap for the critic"
    )
    rn.set_defaults(func=_cmd_run)

    lg = sub.add_parser("ledger", help="read-only claim ledger: paper vs ours, verdict, reasons")
    lg.add_argument("--out", required=True, help="run folder with run_result evidence (`run`)")
    lg.set_defaults(func=_cmd_ledger)

    dg = sub.add_parser(
        "diagnose", help="per claim: run outcome class, curve summary, trial budget (no LLM)"
    )
    dg.add_argument("--out", required=True, help="run folder with run_result evidence (`run`)")
    dg.add_argument(
        "--token-budget", type=int, default=None,
        help="paper token budget (LLM calls of the store count against it; default "
             "$REPRODUCER_TOKEN_BUDGET, else 200000)",
    )  # fmt: skip
    dg.set_defaults(func=_cmd_diagnose)

    rf = sub.add_parser(
        "reflect", help="Reflexion lesson for an out-of-tolerance / diverged claim (proposes only)"
    )
    rf.add_argument("--out", required=True, help="run folder with run_result evidence (`run`)")
    rf.add_argument("--claim", default=None, help="only this claim_id")
    rf.add_argument("--dry-run", action="store_true", help="print the lesson; store nothing")
    rf.add_argument(
        "--force", action="store_true",
        help="also reflect on a claim whose newest run does not trigger it (needs --dry-run)",
    )  # fmt: skip
    rf.add_argument(
        "--max-tokens", type=int, default=REFLECT_MAX_TOKENS, help="max output tokens per call"
    )
    rf.add_argument("--budget-tokens", type=int, default=30_000, help="token cap for reflect")
    rf.add_argument(
        "--token-budget", type=int, default=None,
        help="paper token budget for the breaker (LLM calls of the store count against it; "
             "default $REPRODUCER_TOKEN_BUDGET, else 200000)",
    )  # fmt: skip
    rf.set_defaults(func=_cmd_reflect)

    vt = sub.add_parser("vault", help="render the store into Obsidian notes (read-only store)")
    vt.add_argument("--out", required=True, help="run folder with the store")
    vt.add_argument(
        "--vault",
        default=None,
        help=f"vault root (default: ${VAULT_ENV}, else <out>/vault_out)",
    )
    vt.set_defaults(func=_cmd_vault)

    qu = sub.add_parser("queue", help="escalation queue: pending approvals + open unknowns")
    qu.add_argument("--out", required=True, help="run folder with the store")
    qu.set_defaults(func=_cmd_queue)
    ap = sub.add_parser("approve", help="approve a pending approval request (human decision)")
    ap.add_argument("--out", required=True, help="run folder with the store")
    ap.add_argument("request_id", help="request id (prefix, >= 6 chars) from `queue`")
    ap.add_argument("--note", default="", help="why (stored with the decision)")
    ap.set_defaults(func=_cmd_approve)
    rj = sub.add_parser("reject", help="reject a pending approval request (human decision)")
    rj.add_argument("--out", required=True, help="run folder with the store")
    rj.add_argument("request_id", help="request id (prefix, >= 6 chars) from `queue`")
    rj.add_argument("--note", required=True, help="why (the critic sees it for a patch)")
    rj.set_defaults(func=_cmd_reject)
    an = sub.add_parser("answer", help="resolve an open unknown with a human answer")
    an.add_argument("--out", required=True, help="run folder with the store")
    an.add_argument("unknown", help="unknown name, as `queue` lists it")
    an.add_argument("value", help="the value")
    an.add_argument("--note", required=True, help="where the value comes from")
    an.set_defaults(func=_cmd_answer)
    er = sub.add_parser(
        "eval-report", help="outcome / trajectory / efficiency report over every paper store"
    )
    er.add_argument("--manifest", default="evals/manifest.json", help="cases: name, spec, out")
    er.add_argument("--out", default="runs/eval_report", help="folder for report.md + .json")
    er.set_defaults(func=_cmd_eval_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
