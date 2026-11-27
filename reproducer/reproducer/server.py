"""MCP server (stdio): exposes the Reproducer to the team's assistant.

  tools (model-controlled, bounded)  reproduce_setup, start_run, status
  resources (read-only)              reproducer://{out_name}/ledger | unknowns | queue |
                                     lessons | paper_note
  prompt (user-controlled)           reproduce_paper

Human-only (Tier 2), deliberately NOT exposed: approve, reject, answer. A model can see the
escalation queue but never decide it; a human runs the CLI command `queue` prints.

Both job tools start the existing CLI (`python -m reproducer.cli reproduce ...`) as a
detached background job, so no pipeline logic lives here and the CLI's stdout never reaches
the MCP wire. A job's log ends with a `[job] exit <code>` line; status reads the stage (last
`[node]` line) and that exit code from it. Paths are confined: a relative spec_path must
resolve under evals/, an absolute one under $REPRODUCER_SPEC_ROOT; out_name is a slug mapped
to runs/<out_name>. Run with: .venv/bin/python -m reproducer.server
"""

from __future__ import annotations

import io
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp_types import ToolAnnotations

from reproducer import cli
from reproducer.domain.approval import escalation_queue
from reproducer.domain.unknowns import fold_unknowns
from reproducer.domain.vault_render import applied_of, lessons_by_trial
from reproducer.graph.nodes.diagnose import diagnose
from reproducer.graph.nodes.execute import caps_from_env
from reproducer.graph.nodes.ingest_spec import paper_source_for
from reproducer.graph.nodes.orchestrator import load_current_plans, paper_claims
from reproducer.graph.nodes.pipeline import evaluate_store, render_store_vault
from reproducer.providers.store import NotARunStore, RunStore, lock_holder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT_ENV = "REPRODUCER_SPEC_ROOT"
SLUG = re.compile(r"[a-z0-9_-]{1,64}")
JOBS_DIR = "mcp_jobs"  # <out>/mcp_jobs/<utc stamp>-<what>.log: one log per started job
TAIL_LINES = 15
START_TIMEOUT_S = 5.0  # a launched job must take run.lock (or exit) within this
JOB_EXIT = re.compile(r"\[job\] exit (-?\d+)")  # last line of a finished job's log
STAGE = re.compile(r"\[([a-z_]+)\]")  # a progress line of the CLI: "[node] detail"
EXIT_STATUS = {
    0: "ok",
    1: "failed",
    2: "config_error",
    3: "already running",
    cli.EXIT_AWAITING_APPROVAL: "awaiting_approval",
}

HUMAN_ONLY = (
    "Approvals and answers are human-only (Tier 2): approve, reject and answer are NOT "
    "exposed over MCP. When something is pending, show the user the `queue` resource; they "
    f"run `{cli.CLI} approve|reject|answer ...` themselves, then re-run start_run."
)
INSTRUCTIONS = (
    "Reproducer: reproduces a paper's reported results from its GitHub repo, deterministic "
    "code first, LLM only where judgment is needed. Workflow: reproduce_setup(spec_path, "
    "out_name) -> configure/env/data/smoke/plan; start_run(out_name) -> full pipeline "
    "(runs + reflexion + vault) in the background; status(out_name) and the "
    "reproducer://{out_name}/... resources are read-only. Both jobs run in the background: "
    "follow them with status (stage, exit code). A claim is verified only with a "
    "passing run; never report success the ledger does not show. " + HUMAN_ONLY
)

# argv, cwd, log file -> pid of the detached job
Spawner = Callable[[Sequence[str], Path, Path], int]

# runs "$@" and appends its exit code to stdout (the job log); argv is passed as positional
# parameters, never interpolated into the script
_JOB_SH = '"$@"; c=$?; echo "[job] exit $c"; exit $c'


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUNBUFFERED": "1"}  # the log streams while the job runs


def spawn_detached(argv: Sequence[str], cwd: Path, log: Path) -> int:
    """Start argv in a new session (survives the server and the host), stdout + stderr into
    log, which ends with `[job] exit <code>` when argv finishes. A daemon thread reaps it so
    no zombie stays while the server lives. Returns the pid."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        proc = subprocess.Popen(
            ["sh", "-c", _JOB_SH, "job", *argv],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            start_new_session=True,
            close_fds=True,
        )
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc.pid


@dataclass(frozen=True)
class ServerConfig:
    project_root: Path = PROJECT_ROOT
    spec_root: Path | None = None  # absolute spec paths: None -> $REPRODUCER_SPEC_ROOT
    python: str = sys.executable
    spawn: Spawner = field(default=spawn_detached)
    start_timeout_s: float = START_TIMEOUT_S

    @property
    def runs_root(self) -> Path:
        return self.project_root / "runs"

    def evals_root(self) -> Path:
        """Where relative spec paths must resolve."""
        return (self.project_root / "evals").resolve()

    def abs_spec_root(self) -> Path | None:
        """Where absolute spec paths must resolve: spec_root, else $REPRODUCER_SPEC_ROOT
        (relative values from the project root); None: absolute spec paths are refused."""
        root = self.spec_root
        if root is None:
            env = os.environ.get(SPEC_ROOT_ENV, "").strip()
            if not env:
                return None
            root = Path(env)
        if not root.is_absolute():
            root = self.project_root / root
        return root.resolve()


# --- path safety -----------------------------------------------------------------------------


def out_dir_for(cfg: ServerConfig, out_name: str) -> Path:
    """runs/<out_name>; out_name must be a slug ([a-z0-9_-], 1-64 chars)."""
    if not isinstance(out_name, str) or not SLUG.fullmatch(out_name):
        raise ValueError(f"invalid out_name {out_name!r}: use a slug of [a-z0-9_-] (1-64 chars)")
    return cfg.runs_root / out_name


def spec_file_for(cfg: ServerConfig, spec_path: str) -> Path:
    """An existing spec file, symlinks followed, without '..': a relative path (from the
    project root) must resolve under evals/; an absolute one under $REPRODUCER_SPEC_ROOT
    (refused when that is unset)."""
    p = Path(spec_path) if isinstance(spec_path, str) and spec_path.strip() else None
    if p is None or ".." in p.parts or "\x00" in spec_path:
        raise ValueError(
            f"invalid spec_path {spec_path!r}: e.g. evals/<paper>/spec.json, without '..'"
        )
    if p.is_absolute():
        root = cfg.abs_spec_root()
        if root is None:
            raise ValueError(
                f"absolute spec_path {spec_path!r} needs {SPEC_ROOT_ENV}; "
                "or pass a path relative to the project, under evals/"
            )
        full = p.resolve()
    else:
        root = cfg.evals_root()
        full = (cfg.project_root / p).resolve()
    if not full.is_relative_to(root):
        raise ValueError(f"spec_path {spec_path!r} is outside the spec root {root}")
    if not full.is_file():
        raise ValueError(f"spec_path {spec_path!r}: no such file")
    return full


def _rel(cfg: ServerConfig, path: Path) -> str:
    """path relative to the project root when inside it (how the CLI is usually invoked)."""
    try:
        return path.resolve().relative_to(cfg.project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


# --- read-side helpers (reuse only) ----------------------------------------------------------


def _store(out: Path) -> RunStore:
    return RunStore(out, create=False)


def _cli_argv(cfg: ServerConfig, *args: str) -> list[str]:
    return [cfg.python, "-m", "reproducer.cli", *args]


def _new_log(out: Path, what: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return out / JOBS_DIR / f"{stamp}-{what}.log"


def newest_log(out: Path) -> Path | None:
    logs = sorted((out / JOBS_DIR).glob("*.log"), key=lambda p: (p.stat().st_mtime, p.name))
    return logs[-1] if logs else None


def tail(path: Path | None, n: int = TAIL_LINES, max_bytes: int = 64_000) -> list[str]:
    """The last n non-blank lines of a log (only its last max_bytes are read; training logs
    can be large). tqdm's carriage returns count as line breaks."""
    if path is None or not path.is_file():
        return []
    with open(path, "rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - max_bytes))
        text = f.read().decode("utf-8", errors="replace")
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()][-n:]


def job_exit(log: Path | None) -> int | None:
    """The exit code a finished job wrote as its last log line; None while it runs (or if
    it died without writing it)."""
    last = tail(log, 1, max_bytes=4_096)
    m = JOB_EXIT.fullmatch(last[0]) if last else None
    return int(m.group(1)) if m else None


def job_stage(log: Path | None) -> str | None:
    """The last `[node] ...` progress line of a job log (the current / last stage)."""
    for line in reversed(tail(log, 10_000, max_bytes=256_000)):
        m = STAGE.match(line)
        if m and m.group(1) != "job":
            return line
    return None


def job_info(cfg: ServerConfig, log: Path | None) -> dict[str, Any] | None:
    if log is None:
        return None
    code = job_exit(log)
    return {
        "log": _rel(cfg, log),
        "kind": log.stem.split("-", 2)[-1],  # <date>-<time>-<what>
        "stage": job_stage(log),
        "finished": code is not None,
        "exit_code": code,
        "result": None if code is None else EXIT_STATUS.get(code, "failed"),
    }


def launch(cfg: ServerConfig, out: Path, argv: Sequence[str], what: str) -> dict[str, Any]:
    """Spawn a detached job, then wait up to cfg.start_timeout_s for it to take run.lock
    (lock_holder polling). started: the lock is held; already running: the job exited 3
    (another job took the lock first); finished / failed to start: it exited before that
    (exit 0 / other); starting: still alive without the lock at the deadline."""
    log = _new_log(out, what)
    pid = cfg.spawn(argv, cfg.project_root, log)
    info: dict[str, Any] = {"pid": pid, "command": shlex.join(argv[1:]), "log": _rel(cfg, log)}
    deadline = time.monotonic() + cfg.start_timeout_s
    while True:
        holder = lock_holder(out)
        if holder is not None:
            return {"status": "started", **info, "lock_pid": holder}
        code = job_exit(log)
        if code is not None:
            status = {0: "finished", 3: "already running"}.get(code, "failed to start")
            return {"status": status, **info, "exit_code": code, "log_tail": tail(log)}
        if time.monotonic() >= deadline:
            return {"status": "starting", **info,
                    "note": f"alive but run.lock not taken within {cfg.start_timeout_s:g}s; "
                    "follow it with status"}  # fmt: skip
        time.sleep(0.1)


def _queue(store: RunStore) -> dict[str, Any]:
    evidence = store.load_evidence()
    q = escalation_queue(evidence, fold_unknowns(store.load_events(), evidence))
    return {
        "pending_approvals": [
            {
                "request_id": r.evidence_id[:12],
                "action": r.content.get("action"),
                "subject": r.content.get("subject"),
                "reason": r.content.get("reason"),
            }
            for r in q.requests
        ],
        "open_unknowns": [st.name for st in q.unknowns],
    }


def _plans(out: Path, store: RunStore) -> dict[str, Any]:
    try:
        cur = load_current_plans(store, repo_dir=out / "repo", partial=True)
    except ValueError as e:
        return {"plans": [], "plans_error": str(e)}
    return {
        "plans": [
            {"claim_id": p.claim.claim_id, "command": p.command, "fidelity": p.fidelity}
            for p in cur.plans
        ],
        "plan_failures": dict(cur.failed),
    }


def _claims_and_breaker(out: Path, store: RunStore) -> dict[str, Any]:
    try:
        claims, evals = evaluate_store(store)
        _, b = diagnose(out, store, claims, evals, gpu_h_cap=caps_from_env(os.environ, 0.0).paper_h)
    except ValueError as e:
        return {"claims": [], "claims_error": str(e), "breaker": None}
    return {
        "claims": [
            {
                "claim_id": e.claim_id,
                "metric": e.metric,
                "unit": e.unit,
                "paper_value": e.paper_value,
                "our_value": e.our_value,
                "abs_diff": e.abs_diff,
                "status": e.status,
                "fidelity": e.fidelity,
                "run_fidelity": e.run_fidelity,
                "attempts": e.attempts,
                "within_tolerance": e.within_tolerance,
            }
            for e in evals.values()
        ],
        "breaker": {
            "trials_used": b.global_used,
            "trials_max": b.global_max,
            "gpu_h_spent": round(b.gpu_h_spent, 4),
            "gpu_h_cap": b.gpu_h_cap,
            "tokens_used": b.tokens_used,
            "token_budget": b.token_budget,
            "tripped": list(b.tripped),
            "per_claim": {
                cid: {
                    "trials": c.trials,
                    "max_trials": c.max_trials,
                    "remaining": c.remaining,
                    "can_try": b.can_try(cid),
                }
                for cid, c in b.per_claim.items()
            },
        },
    }


_CAPTURE = threading.Lock()  # redirect_stdout is process-wide: one capture at a time


def _captured(fn: Callable[[], object]) -> str:
    """stdout of a CLI printer as text (the same report a human sees)."""
    buf = io.StringIO()
    with _CAPTURE, redirect_stdout(buf):
        fn()
    return buf.getvalue()


# --- server ----------------------------------------------------------------------------------


def build_server(cfg: ServerConfig | None = None) -> MCPServer:
    cfg = cfg or ServerConfig()
    mcp = MCPServer("reproducer", instructions=INSTRUCTIONS)

    def tool_out(out_name: str) -> Path:
        try:
            return out_dir_for(cfg, out_name)
        except ValueError as e:
            raise ToolError(str(e)) from None

    def resource_store(out_name: str) -> tuple[Path, RunStore]:
        try:
            out = out_dir_for(cfg, out_name)
        except ValueError as e:
            raise ResourceError(str(e)) from None
        try:
            return out, _store(out)
        except NotARunStore as e:
            raise ResourceNotFoundError(str(e)) from None

    def resource_text(fn: Callable[[], str]) -> str:
        try:
            return fn()
        except ValueError as e:
            raise ResourceError(str(e)) from None

    @mcp.tool(
        description=(
            "Set up a reproduction: configure -> env (Docker image) -> data -> smoke -> plan "
            "(`reproduce --until plan`) as a detached background job (minutes; no full "
            "training run); returns once the job holds run.lock. Follow it with status "
            "(stage, exit code, plans, pending approvals, open unknowns). spec_path: a "
            "ReproSpec JSON, relative under evals/ (e.g. evals/cutout/spec.json) or absolute "
            "under $REPRODUCER_SPEC_ROOT. out_name: slug [a-z0-9_-]; the run folder is "
            "runs/<out_name>. One job per run folder. " + HUMAN_ONLY
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    )
    def reproduce_setup(spec_path: str, out_name: str) -> dict[str, Any]:
        out = tool_out(out_name)
        try:
            spec = spec_file_for(cfg, spec_path)
        except ValueError as e:
            raise ToolError(str(e)) from None
        holder = lock_holder(out)
        if holder is not None:
            return {"status": "already running", "pid": holder, "out_name": out_name,
                    "log": _rel_or_none(cfg, newest_log(out))}  # fmt: skip
        argv = _cli_argv(
            cfg, "reproduce", "--spec", _rel(cfg, spec), "--out", _rel(cfg, out), "--until", "plan"
        )
        return {"out_name": out_name, **launch(cfg, out, argv, "setup")}

    @mcp.tool(
        description=(
            "Start the full pipeline (pre-flight + runs + reflexion + vault) for runs/<out_name> "
            "as a detached background process; returns at once with pid and log path. Needs "
            "reproduce_setup first. The spec is found from the store, or pass spec_path. "
            "With claim_id: `reproduce --claim` (every claim planned; only that one runs, is "
            "evaluated and reflected on; the vault is still written). Waits until the job "
            "holds run.lock: started | already running | failed to start. One job per run "
            "folder. GPU-hour caps apply. " + HUMAN_ONLY
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    def start_run(
        out_name: str, claim_id: str | None = None, spec_path: str | None = None
    ) -> dict[str, Any]:
        out = tool_out(out_name)
        try:
            store = _store(out)
        except NotARunStore:
            raise ToolError(
                f"no run store in runs/{out_name}: call reproduce_setup first"
            ) from None
        claims = paper_claims(store.load_evidence())
        if claim_id is not None and claim_id not in claims:
            raise ToolError(f"unknown claim_id {claim_id!r}; known: {sorted(claims)}")
        try:
            spec = (
                spec_file_for(cfg, spec_path) if spec_path is not None else stored_spec(cfg, claims)
            )
        except ValueError as e:
            raise ToolError(str(e)) from None
        holder = lock_holder(out)
        if holder is not None:
            return {
                "status": "already running",
                "pid": holder,
                "out_name": out_name,
                "log": _rel_or_none(cfg, newest_log(out)),
            }
        argv = _cli_argv(cfg, "reproduce", "--spec", _rel(cfg, spec), "--out", _rel(cfg, out))
        what = "reproduce"
        if claim_id is not None:
            argv += ["--claim", claim_id]
            what = f"reproduce-{claim_id}" if SLUG.fullmatch(claim_id) else "reproduce-claim"
        return {"out_name": out_name, **launch(cfg, out, argv, what)}

    @mcp.tool(
        description=(
            "Read-only status of runs/<out_name>: whether a job is running (run.lock holder "
            "alive); the newest job (stage = its last [node] line, exit code once finished); "
            "current plans; per-claim ledger status/fidelity/value; breaker (trial / GPU-hour "
            "/ token budgets); pending approvals and open unknowns. Writes nothing."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def status(out_name: str) -> dict[str, Any]:
        out = tool_out(out_name)
        holder = lock_holder(out)
        base: dict[str, Any] = {
            "out_name": out_name,
            "running": holder is not None,
            "pid": holder,
            "job": job_info(cfg, newest_log(out)),
        }
        try:
            store = _store(out)
        except NotARunStore:
            return {**base, "exists": False}
        return {
            **base,
            "exists": True,
            **_plans(out, store),
            **_claims_and_breaker(out, store),
            **_queue(store),
        }

    # --- resources (read-only) ---------------------------------------------------------------

    @mcp.resource(
        "reproducer://{out_name}/ledger",
        mime_type="text/plain",
        description="Claim ledger (paper vs ours, verdict, reasons), recomputed "
        "from evidence; same text as `cli ledger`.",
    )
    def ledger(out_name: str) -> str:
        out, store = resource_store(out_name)

        def render() -> str:
            claims, evals = evaluate_store(store)
            return _captured(lambda: cli.print_ledger_report(out, store, claims, evals))

        return resource_text(render)

    @mcp.resource(
        "reproducer://{out_name}/unknowns",
        mime_type="application/json",
        description="Every unknown (open or resolved) with its value, resolver and "
        "assumptions, folded from the event log.",
    )
    def unknowns(out_name: str) -> list[dict[str, Any]]:
        _, store = resource_store(out_name)
        states = fold_unknowns(store.load_events(), store.load_evidence())
        return [st.model_dump(mode="json") for st in sorted(states.values(), key=lambda s: s.name)]

    @mcp.resource(
        "reproducer://{out_name}/queue",
        mime_type="text/plain",
        description="Escalation queue: pending approval requests and open unknowns, "
        "each with the CLI command a HUMAN runs to answer it (not exposed over MCP).",
    )
    def queue(out_name: str) -> str:
        out, store = resource_store(out_name)
        return resource_text(lambda: _captured(lambda: cli.print_queue(out, store)))

    @mcp.resource(
        "reproducer://{out_name}/lessons",
        mime_type="application/json",
        description="Reflexion lessons per claim and trial, with what code did with "
        "each (lesson_applied), in store order.",
    )
    def lessons(out_name: str) -> list[dict[str, Any]]:
        _, store = resource_store(out_name)
        evidence = store.load_evidence()
        rows: list[dict[str, Any]] = []
        for cid in paper_claims(evidence):
            for trial, evs in lessons_by_trial(evidence, cid).items():
                for ev in evs:
                    ap = applied_of(evidence, ev.evidence_id)
                    rows.append(
                        {
                            "evidence_id": ev.evidence_id,
                            "claim_id": cid,
                            "trial": trial,
                            "newest_of_trial": ev is evs[-1],
                            "lesson": ev.content.get("lesson"),
                            "applied": None if ap is None else ap.content,
                        }
                    )
        return rows

    @mcp.resource(
        "reproducer://{out_name}/paper_note",
        mime_type="text/markdown",
        description="The vault Papers note (markdown), rendered from the store "
        "read-only; nothing is written to the vault.",
    )
    def paper_note(out_name: str) -> str:
        out, store = resource_store(out_name)

        def render() -> str:
            notes = render_store_vault(store, out)
            return next(v for p, v in notes.items() if p.startswith("Papers/"))

        return resource_text(render)

    # --- prompt (user-controlled) ------------------------------------------------------------

    @mcp.prompt(description="Reproduce a paper's reported results end to end (human approves).")
    def reproduce_paper(spec_path: str = "evals/cutout/spec.json", out_name: str = "") -> str:
        name = out_name or "<out_name>"
        return (
            f"Reproduce the results claimed in the paper spec {spec_path} into runs/{name}.\n\n"
            "Workflow:\n"
            f"1. reproduce_setup(spec_path={spec_path!r}, out_name={name!r}) starts the setup "
            "in the background (configure, Docker env, data, smoke run, one plan per claim). "
            f"Poll status({name!r}) until its job is finished, then report the result, the "
            "plans and the pending approvals / open unknowns.\n"
            f"2. If anything is pending, read reproducer://{name}/queue and show it to me. "
            "Approvals, rejections and answers to unknowns are MINE: I run the printed "
            f"`{cli.CLI} approve|reject|answer` commands myself. You cannot and must not try "
            "to approve anything, and never guess a value for an unknown.\n"
            f"3. When nothing blocks, start_run(out_name={name!r}) starts the full pipeline in "
            f"the background. Follow it with status({name!r}); do not start a second job.\n"
            f"4. Finally summarize reproducer://{name}/ledger per claim (paper vs ours, "
            "status, fidelity) and link the lessons and paper_note resources. A claim counts "
            "as reproduced only if the ledger says verified; open/blocked claims stay open."
        )

    return mcp


def stored_spec(cfg: ServerConfig, claims: dict[str, Any]) -> Path:
    """The spec of the stored paper claims: paper:<dir> -> <spec root>/<dir>/spec.json, checked
    to give back the same paper source (ingest_spec.paper_source_for)."""
    sources = {cev.source for _, cev in claims.values()}
    if len(sources) != 1:
        raise ValueError(
            f"cannot tell the spec from the store (paper sources {sorted(sources)}); pass spec_path"
        )
    source = sources.pop()
    name = source.removeprefix("paper:")
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"cannot tell the spec from paper source {source!r}; pass spec_path")
    roots = [r for r in (cfg.evals_root(), cfg.abs_spec_root()) if r is not None]
    for root in roots:
        spec = (root / name / "spec.json").resolve()
        if spec.is_relative_to(root) and spec.is_file() and paper_source_for(spec) == source:
            return spec
    tried = ", ".join(str(r / name / "spec.json") for r in roots)
    raise ValueError(f"no spec for {source} (tried {tried}); pass spec_path")


def _rel_or_none(cfg: ServerConfig, path: Path | None) -> str | None:
    return None if path is None else _rel(cfg, path)


def main() -> None:
    os.chdir(PROJECT_ROOT)  # stored refs may be relative to the project root (resolve_ref)
    build_server().run("stdio")


if __name__ == "__main__":
    main()
