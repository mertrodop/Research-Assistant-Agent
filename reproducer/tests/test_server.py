"""Tests for the MCP server (reproducer.server) through the in-memory MCP client."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import anyio
import pytest
from domain.vault_fixture import C10, SPEC, cutout_store
from mcp import Client
from mcp.shared.exceptions import MCPError

from reproducer.graph.nodes.pipeline import evaluate_store
from reproducer.providers.store import RunStore, run_lock
from reproducer.server import ServerConfig, build_server, job_exit, launch, spawn_detached, tail

HUMAN_ONLY = {"approve", "reject", "answer"}


def _fill(out: Path) -> RunStore:
    evidence, events, calls = cutout_store()
    store = RunStore(out)
    for ev in evidence.values():
        store.append_evidence(ev)
    for e in events:
        store.append_event(e)
    for c in calls:
        store.append_call_record(c)
    return store


class FakeSpawn:
    """Records argv and plays the detached job: "lock" takes run.lock and keeps it (a job
    that started), ("exit", n) writes the job's exit line without the lock (exited at once),
    "silent" does neither (alive, no lock yet). `effect` runs first (e.g. writes a store)."""

    def __init__(self, mode: Any = "lock", effect: Callable[[Path], None] | None = None) -> None:
        self.mode, self.effect = mode, effect
        self.calls: list[tuple[list[str], Path, Path]] = []
        self.held = ExitStack()
        _SPAWNS.append(self)

    def __call__(self, argv: Sequence[str], cwd: Path, log: Path) -> int:
        self.calls.append((list(argv), cwd, log))
        out = log.parents[1]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("[configure] ready\n[plan] 2 plan(s) attempts=1\n")
        if self.effect is not None:
            self.effect(out)
        if self.mode == "lock":
            self.held.enter_context(run_lock(out))
        elif isinstance(self.mode, tuple):
            with open(log, "a") as f:
                f.write(f"error: boom\n[job] exit {self.mode[1]}\n")
        return 4242


_SPAWNS: list[FakeSpawn] = []


@pytest.fixture(autouse=True)
def _release_fake_locks() -> Any:
    yield
    while _SPAWNS:
        _SPAWNS.pop().held.close()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "evals" / "cutout").mkdir(parents=True)
    shutil.copy(SPEC, tmp_path / "evals" / "cutout" / "spec.json")
    (tmp_path / "runs").mkdir()
    (tmp_path / "secret.json").write_text("{}")
    return tmp_path


@pytest.fixture
def store_out(project: Path) -> Path:
    out = project / "runs" / "cutout"
    _fill(out)
    return out


def _cfg(project: Path, **kw: Any) -> ServerConfig:
    kw.setdefault("spawn", FakeSpawn())
    kw.setdefault("start_timeout_s", 0.3)
    return ServerConfig(project_root=project, python="python", **kw)


def _with_client(cfg: ServerConfig, fn: Callable[[Client], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        async with Client(build_server(cfg)) as c:
            return await fn(c)

    return anyio.run(main)


def _call(cfg: ServerConfig, tool: str, args: dict[str, Any]) -> tuple[bool, Any]:
    async def go(c: Client) -> tuple[bool, Any]:
        r = await c.call_tool(tool, args)
        return r.is_error, (r.content[0].text if r.is_error else r.structured_content)

    return _with_client(cfg, go)


def _read(cfg: ServerConfig, uri: str) -> str:
    async def go(c: Client) -> str:
        return (await c.read_resource(uri)).contents[0].text

    return _with_client(cfg, go)


# --- surface ---------------------------------------------------------------------------------


def test_lists_tools_resources_prompt_without_human_only_actions(project: Path) -> None:
    async def go(c: Client) -> tuple[Any, ...]:
        return (
            await c.list_tools(),
            await c.list_resources(),
            await c.list_resource_templates(),
            await c.list_prompts(),
            await c.get_prompt("reproduce_paper", {"out_name": "x"}),
        )

    tools, resources, templates, prompts, prompt = _with_client(_cfg(project), go)
    names = {t.name for t in tools.tools}
    assert names == {"reproduce_setup", "start_run", "status"}
    uris = {t.uri_template for t in templates.resource_templates}
    assert uris == {
        f"reproducer://{{out_name}}/{k}"
        for k in ("ledger", "unknowns", "queue", "lessons", "paper_note")
    }
    assert resources.resources == []
    assert [p.name for p in prompts.prompts] == ["reproduce_paper"]
    everything = names | uris | {p.name for p in prompts.prompts}
    assert not any(h in n for n in everything for h in HUMAN_ONLY)
    status = next(t for t in tools.tools if t.name == "status")
    assert status.annotations is not None and status.annotations.read_only_hint is True
    text = prompt.messages[0].content.text
    assert "approve|reject|answer" in text and "MINE" in text


def test_server_instructions_say_approvals_are_human_only() -> None:
    from reproducer.server import INSTRUCTIONS

    assert "human-only" in INSTRUCTIONS and "NOT exposed over MCP" in INSTRUCTIONS


# --- status + resources on a fake store ------------------------------------------------------


def test_status_on_fake_store_matches_the_ledger(project: Path, store_out: Path) -> None:
    (store_out / "mcp_jobs").mkdir()
    log = store_out / "mcp_jobs" / "20261125-000000-reproduce.log"
    log.write_text("[plan] 2 plan(s)\n[claim_worker] c10 done\nepoch 3 loss 0.1\n\n")
    before = {p.name: p.read_bytes() for p in store_out.glob("*.jsonl")}
    err, st = _call(_cfg(project), "status", {"out_name": "cutout"})
    assert not err
    assert st["exists"] is True and st["running"] is False and st["pid"] is None
    assert st["job"] == {
        "log": "runs/cutout/mcp_jobs/20261125-000000-reproduce.log", "kind": "reproduce",
        "stage": "[claim_worker] c10 done", "finished": False, "exit_code": None,
        "result": None,
    }  # fmt: skip
    _, evals = evaluate_store(RunStore(store_out, create=False))
    got = {c["claim_id"]: (c["status"], c["fidelity"], c["our_value"]) for c in st["claims"]}
    assert got == {cid: (e.status, e.fidelity, e.our_value) for cid, e in evals.items()}
    assert st["breaker"]["per_claim"][C10]["max_trials"] >= 1
    assert "seed" in st["open_unknowns"]
    assert st["pending_approvals"] == []
    assert "plans" in st
    assert {p.name: p.read_bytes() for p in store_out.glob("*.jsonl")} == before  # read-only
    assert not (store_out / "run.lock").exists()


def test_status_shows_finished_job_exit_code(project: Path, store_out: Path) -> None:
    (store_out / "mcp_jobs").mkdir()
    old = store_out / "mcp_jobs" / "20261125-000000-setup.log"
    old.write_text("[plan] 2 plan(s)\n[job] exit 0\n")
    new = store_out / "mcp_jobs" / "20261125-000100-reproduce-c10.log"
    new.write_text("[evaluate] c10 verified\n[vault] created=3\n[job] exit 4\n")
    os.utime(old, (1, 1))
    err, st = _call(_cfg(project), "status", {"out_name": "cutout"})
    assert not err
    job = st["job"]
    assert job["kind"] == "reproduce-c10" and job["stage"] == "[vault] created=3"
    assert (job["finished"], job["exit_code"], job["result"]) == (True, 4, "awaiting_approval")


def test_status_of_missing_store_and_running_job(project: Path, store_out: Path) -> None:
    err, st = _call(_cfg(project), "status", {"out_name": "nothing_here"})
    assert not err and st["exists"] is False and st["running"] is False and st["job"] is None
    assert not (project / "runs" / "nothing_here").exists()
    with run_lock(store_out):
        err, st = _call(_cfg(project), "status", {"out_name": "cutout"})
    assert st["running"] is True and st["pid"] == str(os.getpid())


def test_resources_read_the_store(project: Path, store_out: Path) -> None:
    cfg = _cfg(project)
    ledger = _read(cfg, "reproducer://cutout/ledger")
    assert "=== ledger:" in ledger and C10 in ledger and "totals:" in ledger
    unknowns = json.loads(_read(cfg, "reproducer://cutout/unknowns"))
    assert any(u["name"] == "seed" and u["status"] == "open" for u in unknowns)
    queue = _read(cfg, "reproducer://cutout/queue")
    assert "=== escalation queue" in queue and "answer --out" in queue
    assert json.loads(_read(cfg, "reproducer://cutout/lessons")) == []
    note = _read(cfg, "reproducer://cutout/paper_note")
    assert note.startswith("---") and C10 in note
    assert not (store_out / "vault_out").exists()  # rendered, never written


def test_resources_reject_bad_or_missing_out_name(project: Path) -> None:
    uris = (
        "reproducer://Bad/ledger",
        "reproducer://nothing/ledger",
        "reproducer://..%2Fsecret/ledger",
        "reproducer://a%2Fb/queue",
    )

    async def go(c: Client) -> list[str]:
        errors = []
        for uri in uris:
            with pytest.raises(MCPError) as e:
                await c.read_resource(uri)
            errors.append(str(e.value))
        return errors

    errors = _with_client(_cfg(project), go)
    assert "invalid out_name" in errors[0]
    assert "not a run store" in errors[1]


# --- reproduce_setup (detached) --------------------------------------------------------------


def test_reproduce_setup_launches_detached_until_plan(project: Path) -> None:
    spawn = FakeSpawn("lock", effect=_fill)
    err, out = _call(
        _cfg(project, spawn=spawn),
        "reproduce_setup",
        {"spec_path": "evals/cutout/spec.json", "out_name": "cutout"},
    )
    assert not err, out
    ((argv, cwd, log),) = spawn.calls
    assert argv == [
        "python", "-m", "reproducer.cli", "reproduce", "--spec", "evals/cutout/spec.json",
        "--out", "runs/cutout", "--until", "plan",
    ]  # fmt: skip
    assert cwd == project and log.parent == project / "runs" / "cutout" / "mcp_jobs"
    assert log.name.endswith("-setup.log")
    assert out["status"] == "started" and out["pid"] == 4242
    assert out["lock_pid"] == str(os.getpid())  # the job holds run.lock
    # status follows the background job
    err, st = _call(_cfg(project), "status", {"out_name": "cutout"})
    assert st["running"] is True and st["job"]["kind"] == "setup"
    assert st["job"]["stage"] == "[plan] 2 plan(s) attempts=1" and st["job"]["finished"] is False


def test_reproduce_setup_already_running_when_locked(project: Path, store_out: Path) -> None:
    spawn = FakeSpawn()
    with run_lock(store_out):
        err, out = _call(
            _cfg(project, spawn=spawn),
            "reproduce_setup",
            {"spec_path": "evals/cutout/spec.json", "out_name": "cutout"},
        )
    assert not err and out["status"] == "already running" and spawn.calls == []


# --- start_run -------------------------------------------------------------------------------


def test_start_run_launches_reproduce_with_stored_spec(project: Path, store_out: Path) -> None:
    spawn = FakeSpawn()
    err, out = _call(_cfg(project, spawn=spawn), "start_run", {"out_name": "cutout"})
    assert not err, out
    ((argv, cwd, log),) = spawn.calls
    assert argv == [
        "python", "-m", "reproducer.cli", "reproduce", "--spec", "evals/cutout/spec.json",
        "--out", "runs/cutout",
    ]  # fmt: skip
    assert cwd == project and log.parent == store_out / "mcp_jobs"
    assert out["status"] == "started" and out["pid"] == 4242
    assert out["log"].startswith("runs/cutout/mcp_jobs/") and out["log"].endswith("reproduce.log")


def test_start_run_one_claim_uses_reproduce_claim(project: Path, store_out: Path) -> None:
    spawn = FakeSpawn()
    err, out = _call(
        _cfg(project, spawn=spawn), "start_run", {"out_name": "cutout", "claim_id": C10}
    )
    assert not err and out["status"] == "started"
    assert spawn.calls[0][0][3:] == [
        "reproduce", "--spec", "evals/cutout/spec.json", "--out", "runs/cutout", "--claim", C10,
    ]  # fmt: skip
    assert out["log"].endswith(f"reproduce-{C10}.log")
    err, msg = _call(
        _cfg(project, spawn=spawn), "start_run", {"out_name": "cutout", "claim_id": "nope"}
    )
    assert err and "unknown claim_id" in msg and len(spawn.calls) == 1


def test_start_run_already_running_when_locked(project: Path, store_out: Path) -> None:
    spawn = FakeSpawn()
    with run_lock(store_out):
        err, out = _call(_cfg(project, spawn=spawn), "start_run", {"out_name": "cutout"})
    assert not err and out["status"] == "already running" and out["pid"] == str(os.getpid())
    assert spawn.calls == []


@pytest.mark.parametrize(
    ("mode", "status"),
    [(("exit", 3), "already running"), (("exit", 2), "failed to start"),
     (("exit", 0), "finished"), ("silent", "starting")],
)  # fmt: skip
def test_launch_waits_for_the_lock_and_reports_accurately(
    project: Path, store_out: Path, mode: Any, status: str
) -> None:
    spawn = FakeSpawn(mode)
    t0 = time.monotonic()
    err, out = _call(_cfg(project, spawn=spawn), "start_run", {"out_name": "cutout"})
    assert not err and out["status"] == status, out
    if isinstance(mode, tuple):
        assert out["exit_code"] == mode[1] and out["log_tail"][-1] == f"[job] exit {mode[1]}"
        assert time.monotonic() - t0 < 0.3  # no waiting once the job exited
    else:
        assert time.monotonic() - t0 >= 0.3  # polled until the deadline
        assert "not taken within" in out["note"]


def test_start_run_needs_a_store(project: Path) -> None:
    spawn = FakeSpawn()
    err, msg = _call(_cfg(project, spawn=spawn), "start_run", {"out_name": "fresh"})
    assert err and "reproduce_setup first" in msg and spawn.calls == []


def test_spawn_detached_new_session_and_exit_line(tmp_path: Path) -> None:
    log = tmp_path / "jobs" / "x.log"
    code = "import os, sys; print(os.getsid(0), flush=True); sys.exit(5)"
    pid = spawn_detached([sys.executable, "-c", code], tmp_path, log)
    deadline = time.monotonic() + 20
    while job_exit(log) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid > 0 and job_exit(log) == 5  # exit code recorded by the job wrapper
    # its own session, led by the job wrapper (pid): detached from the server and the host
    assert tail(log) == [str(pid), "[job] exit 5"] and pid != os.getsid(0)


def test_spawn_detached_passes_argv_without_shell_interpolation(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    arg = "$(touch pwned); `touch pwned2` 'q\" ; touch pwned3"
    spawn_detached([sys.executable, "-c", "import sys; print(sys.argv[1])", arg], tmp_path, log)
    deadline = time.monotonic() + 20
    while job_exit(log) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert tail(log) == [arg, "[job] exit 0"]
    assert not any(tmp_path.glob("pwned*"))


# --- path validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_name", ["../cutout", "/tmp/x", "Cutout", "a b", "", "a/b", "x" * 65, ".."]
)
def test_bad_out_name_is_rejected(project: Path, out_name: str) -> None:
    spawn = FakeSpawn()
    cfg = _cfg(project, spawn=spawn)
    for tool, args in (
        ("status", {}),
        ("start_run", {}),
        ("reproduce_setup", {"spec_path": "evals/cutout/spec.json"}),
    ):
        err, msg = _call(cfg, tool, {"out_name": out_name, **args})
        assert err and "invalid out_name" in msg, (tool, msg)
    assert spawn.calls == []
    assert sorted(p.name for p in (project / "runs").iterdir()) == []


@pytest.mark.parametrize(
    "spec_path",
    [
        "../evals/cutout/spec.json", "evals/../secret.json", "evals/cutout/../../secret.json",
        "secret.json", "evals/cutout/missing.json", "", "evals/link.json",
        "/etc/passwd", "ABS/evals/cutout/spec.json",  # absolute without REPRODUCER_SPEC_ROOT
    ],
)  # fmt: skip
def test_bad_spec_path_is_rejected(
    project: Path, spec_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REPRODUCER_SPEC_ROOT", raising=False)
    (project / "evals" / "link.json").symlink_to(project / "secret.json")  # escapes evals/
    spawn = FakeSpawn()
    spec_path = spec_path.replace("ABS", str(project))
    err, msg = _call(
        _cfg(project, spawn=spawn), "reproduce_setup", {"spec_path": spec_path, "out_name": "c"}
    )
    assert err and "spec_path" in msg, msg
    assert spawn.calls == []


def test_absolute_spec_path_only_under_spec_root(
    project: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path_factory.mktemp("specs")
    (root / "p").mkdir()
    shutil.copy(SPEC, root / "p" / "spec.json")
    outside = tmp_path_factory.mktemp("elsewhere")
    shutil.copy(SPEC, outside / "spec.json")
    (root / "p" / "link.json").symlink_to(outside / "spec.json")  # resolves outside the root
    (root / "q").symlink_to(root / "p")  # a symlink that stays inside is fine
    monkeypatch.setenv("REPRODUCER_SPEC_ROOT", str(root))
    spawn = FakeSpawn()
    cfg = _cfg(project, spawn=spawn)

    def setup(spec_path: str) -> tuple[bool, Any]:
        return _call(cfg, "reproduce_setup", {"spec_path": spec_path, "out_name": "p"})

    err, out = setup(str(root / "p" / "spec.json"))
    assert not err and out["status"] == "started"
    assert spawn.calls[-1][0][5] == str((root / "p" / "spec.json").resolve())
    spawn.held.close()
    err, out = setup(str(root / "q" / "spec.json"))
    assert not err and out["status"] == "started"
    spawn.held.close()
    for bad in (str(root / "p" / "link.json"), str(outside / "spec.json"),
                str(project / "evals" / "cutout" / "spec.json")):  # fmt: skip
        err, msg = setup(bad)
        assert err and "outside the spec root" in msg, (bad, msg)
    err, out = setup("evals/cutout/spec.json")  # relative paths: still under evals/
    assert not err and out["status"] == "started"
    spawn.held.close()
    err, msg = setup("p/spec.json")  # relative is never resolved against the spec root
    assert err and "spec_path" in msg
    assert len(spawn.calls) == 3


def test_launch_real_child_reports_started_once_it_holds_the_lock(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "x"
    out.mkdir(parents=True)
    hold = (
        "import sys, time\n"
        "from reproducer.providers.store import run_lock\n"
        "time.sleep(0.5)\n"  # the CLI imports before it takes the lock
        "with run_lock(sys.argv[1]):\n"
        "    time.sleep(3)\n"
    )
    cfg = ServerConfig(project_root=tmp_path, start_timeout_s=10)
    res = launch(cfg, out, [sys.executable, "-c", hold, str(out)], "hold")
    assert res["status"] == "started", res
    assert res["lock_pid"] not in (str(os.getpid()), str(res["pid"]))  # held by the job
    deadline = time.monotonic() + 20
    while job_exit(tmp_path / res["log"]) is None:
        assert time.monotonic() < deadline
        time.sleep(0.05)
