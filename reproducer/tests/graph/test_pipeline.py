"""The full pipeline graph (setup -> plan -> per-claim workers -> join -> evaluate -> vault)
with fakes. Real: configure/resume/env/data/smoke over the setup-graph fakes, routing, the
Send fan-out, the join, evaluate_claims and the vault. Faked at the pipeline module boundary:
run_orchestrator (writes tolerances + one claim_mapping, one LLM call record the first time)
and run_claims (pre-flight decision with the real caps rules, then a run_result)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END
from langgraph.types import Send
from test_setup_graph import Fakes, _ok, make_setup_fixture

import reproducer.graph.nodes.pipeline as pl
from reproducer.domain.contracts import ExperimentResult
from reproducer.domain.evidence import make_evidence
from reproducer.domain.ledger import make_run_evidence, tolerance_evidence
from reproducer.domain.telemetry import CallRecord
from reproducer.graph.build import SETUP_RECURSION_LIMIT, build_setup_graph
from reproducer.graph.nodes.execute import (
    ClaimRun,
    caps_from_env,
    completed_run,
    decide,
    spent_gpu_hours,
)
from reproducer.graph.nodes.orchestrator import paper_claims
from reproducer.graph.routing import final_setup_status, route_plan
from reproducer.providers.store import RunStore

CLAIMS = [
    {"claim_id": "c1", "metric": "test_acc", "value": 96.0, "dataset": "CIFAR-10",
     "split": "test", "setting": "s", "location": "Table 1", "quote": "96.0"},
    {"claim_id": "c2", "metric": "test_acc", "value": 80.0, "dataset": "CIFAR-10",
     "split": "test", "setting": "s", "location": "Table 1", "quote": "80.0"},
]  # fmt: skip
OURS = {"c1": 96.2, "c2": 80.1}
FIVE = [f"k{i}" for i in range(1, 6)]
CLAIMS5 = [{**CLAIMS[0], "claim_id": cid, "value": 90.0, "quote": "90.0"} for cid in FIVE]


class PipelineFakes:
    """Fake orchestrator + run_claims; counts LLM calls, full runs and GPU concurrency."""

    def __init__(self, estimates: dict[str, float] | None = None, fail: set[str] = frozenset(),
                 plan_ok: bool = True, unplanned: set[str] = frozenset()) -> None:  # fmt: skip
        self.estimates = estimates or {"c1": 0.5, "c2": 0.5}
        self.fail = fail
        self.plan_ok = plan_ok
        self.unplanned = unplanned
        self.llm_calls = 0
        self.llms_made = 0  # orchestrator LLMs created (lazily, only for a call)
        self.full_runs: list[str] = []
        self.gpu_order: list[str] = []  # claims in the order they got the GPU
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_orchestrator(self, store: RunStore, llm: Any = None, *, llm_factory: Any = None,
                         repo_dir: Path) -> Any:  # fmt: skip
        evidence = store.load_evidence()
        claims = paper_claims(evidence)
        for c, cev in claims.values():
            store.append_evidence(tolerance_evidence(c, claim_evidence_id=cev.evidence_id,
                                                     source=cev.source))  # fmt: skip
        reused = any(ev.kind == "claim_mapping" for ev in evidence.values())
        if not reused:  # the one LLM call, recorded like MeteredLLM does
            llm = llm or llm_factory()
            self.llms_made += 1
            self.llm_calls += 1
            llm.on_record(CallRecord(node="orchestrator", backend="anthropic", model="m",
                                     input_tokens=10, output_tokens=5, latency_s=0.1,
                                     prompt_sha256="0" * 64))  # fmt: skip
            store.append_evidence(make_evidence(
                kind="claim_mapping", source="repo:x@0", locator="orchestrator:claim_mapping",
                content={"input_digest": "I" * 64}, confidence="derived", authority="llm",
                links=("x" * 64,),
            ))  # fmt: skip
        plans = tuple(
            types.SimpleNamespace(claim=c, command=f"python train.py --claim {cid}",
                                  fidelity="inferred",
                                  evidence=types.SimpleNamespace(evidence_id=cid * 32))
            for cid, (c, _) in reversed(claims.items())  # mapping order != spec order
            if cid not in self.unplanned
        )  # fmt: skip
        errors = tuple(f"{cid}: metric line not found" for cid in sorted(self.unplanned))
        return types.SimpleNamespace(
            ok=self.plan_ok and not errors, plans=plans if self.plan_ok else (), reused=reused,
            attempts=0 if reused else 1, input_digest="I" * 64,
            errors=errors if self.plan_ok else ("mapping rejected",),
        )  # fmt: skip

    def run_claims(self, out: Path, store: RunStore, deps: dict[str, Any], *, claim_id: str,
                   no_checkpoints: bool, partial_plans: bool, log: Any,
                   progress: Any) -> list[ClaimRun]:  # fmt: skip
        assert partial_plans  # a claim that did not plan must not fail the others
        with self.lock:
            self.gpu_order.append(claim_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return self._run(store, deps, claim_id)
        finally:
            with self.lock:
                self.active -= 1

    def _run(self, store: RunStore, deps: dict[str, Any], cid: str) -> list[ClaimRun]:
        if cid in self.fail:
            raise ValueError(f"no current plan for claim {cid!r}")
        claim, _ = paper_claims(store.load_evidence())[cid]
        digest = hashlib.sha256(cid.encode()).hexdigest()
        evidence = store.load_evidence()
        done = completed_run(evidence.values(), digest, claim.metric)
        if done is not None:
            return [ClaimRun(cid, "cmd", "img", digest, "already_completed", existing=done)]
        caps = caps_from_env(deps["env"], spent_gpu_hours(evidence.values()))
        decision, reason = decide(True, self.estimates.get(cid, 0.5), caps)
        pf = make_evidence(kind="preflight", source="repo:x@0", locator=f"preflight:{cid}",
                           content={"claim_id": cid, "run_digest": digest,
                                    "decision": decision, "reason": reason},
                           confidence="observed", authority="execution")  # fmt: skip
        store.append_evidence(pf)
        if decision != "run":
            return [ClaimRun(cid, "cmd", "img", digest, "skipped")]
        self.full_runs.append(cid)
        result = ExperimentResult(
            claim_id=cid, run_id=f"run-{cid}", exit_code=0,
            metrics={claim.metric: OURS.get(cid, 90.1)}, log_ref=f"runs/run-{cid}/run.log",
            config_digest=digest, duration=3600.0, gpu_hours=self.estimates.get(cid, 0.5),
        )  # fmt: skip
        store.append_evidence(make_run_evidence(result, spec_faithful=False,
                                                assumption_evidence_ids=(), source="repo:x@0",
                                                links=(pf.evidence_id,), artifacts=[]))  # fmt: skip
        return [ClaimRun(cid, "cmd", "img", digest, "ran")]


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path]:
    out, spec = make_setup_fixture(tmp_path)
    data = json.loads(spec.read_text())
    data["claims"] = CLAIMS
    spec.write_text(json.dumps(data))
    return out, spec


def _install(monkeypatch: pytest.MonkeyPatch, pf: PipelineFakes) -> None:
    monkeypatch.setattr(pl, "run_orchestrator", pf.run_orchestrator)
    monkeypatch.setattr(pl, "run_claims", pf.run_claims)


def run_pipeline(out: Path, spec: Path, fakes: Fakes, env: dict[str, str] | None = None,
                 extra_deps: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
                 **inputs: Any) -> tuple[dict[str, Any], list[str]]:  # fmt: skip
    """(final state, names of the nodes that ran, in order)."""
    deps = fakes.deps(RunStore(out))
    deps.update(env=env or {}, gpu_lock=threading.Semaphore(1), gpu_turns=pl.GpuTurns(),
                **(extra_deps or {}))  # fmt: skip
    graph = build_setup_graph(deps)  # type: ignore[arg-type]
    init: dict[str, Any] = {
        "out_dir": str(out), "spec_path": str(spec), "commit": None, "max_chars": 12_000,
        "max_tokens": 1_500, "status": "running", "configure_attempts": 0,
        "setup_status": "running", "known_fixes_used": 0, "smoke_iterations": 0,
        "env_builds": 0, "until": "vault", "vault_dir": str(out / "vault_out"), **inputs,
    }  # fmt: skip
    final: dict[str, Any] = init
    nodes: list[str] = []
    cfg = {"recursion_limit": SETUP_RECURSION_LIMIT, **(config or {})}
    for mode, chunk in graph.stream(init, config=cfg,  # type: ignore[arg-type]
                                    stream_mode=["updates", "values"]):  # fmt: skip
        if mode == "values":
            final = chunk
        else:
            nodes += list(chunk)
    return final, nodes


def _ok_fakes() -> Fakes:
    return Fakes(lambda stage, image, work, n: _ok(stage))


# --- full pipeline ---------------------------------------------------------------------------


def test_full_pipeline_two_claims_run_evaluated_and_vaulted(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    final, nodes = run_pipeline(out, spec, _ok_fakes())
    assert final_setup_status(final) == "ready" and final["stop_reason"] == "passed"
    assert final["plans"] == ["c1", "c2"]  # spec claim order (the fake maps c2 first)
    assert pf.gpu_order == ["c1", "c2"] and not final["plan_failures"]
    assert nodes.count("claim_worker") == 2  # one worker per claim (Send fan-out)
    assert nodes.count("join") == 1 and nodes.count("evaluate") == 1  # fan-in runs once
    assert nodes[-3:] == ["join", "evaluate", "vault"]
    assert sorted(w.claim_id for w in final["worker_results"]) == ["c1", "c2"]
    assert {w.status for w in final["worker_results"]} == {"ran"}
    assert sorted(pf.full_runs) == ["c1", "c2"] and pf.max_active == 1  # GPU semaphore 1
    evals = final["evaluations"]
    assert {cid: (e.status, e.fidelity) for cid, e in evals.items()} == {
        "c1": ("verified", "inferred"), "c2": ("verified", "inferred")}  # fmt: skip
    counts = final["vault_counts"]
    assert counts.changed == len(counts.created) > 0 and not counts.updated
    paper = Path(final["vault_paper_note"])
    assert paper == out / "vault_out" / "Papers" / "proj.md" and paper.is_file()
    assert (out / "vault_out" / "Claims" / "proj__c2.md").is_file()


def test_only_claim_runs_one_claim_plans_all_and_vaults(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    final, nodes = run_pipeline(out, spec, _ok_fakes(), only_claim="c2")
    assert final_setup_status(final) == "ready"
    assert final["plans"] == ["c1", "c2"]  # every claim is planned
    assert pf.full_runs == ["c2"] and nodes.count("claim_worker") == 1
    assert [w.claim_id for w in final["worker_results"]] == ["c2"]
    assert set(final["evaluations"]) == {"c2"}
    runs = [ev for ev in RunStore(out).load_evidence().values() if ev.kind == "run_result"]
    assert {ev.content["claim_id"] for ev in runs} == {"c2"}  # c1 untouched
    assert nodes[-1] == "vault"
    assert (out / "vault_out" / "Claims" / "proj__c1.md").is_file()  # vault: whole store


def test_cap_skips_one_claim_other_still_runs_and_ledger_shows_reason(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    from reproducer import cli

    out, spec = fixture
    pf = PipelineFakes(estimates={"c1": 0.5, "c2": 5.0})
    _install(monkeypatch, pf)
    final, _ = run_pipeline(out, spec, _ok_fakes(), env={"REPRODUCER_RUN_GPU_HOURS": "1"})
    status = {w.claim_id: w.status for w in final["worker_results"]}
    assert status == {"c1": "ran", "c2": "skipped"} and pf.full_runs == ["c1"]
    evals = final["evaluations"]
    assert (evals["c1"].status, evals["c2"].status) == ("verified", "open")  # evaluate ran
    assert "vault_counts" in final
    capsys.readouterr()
    assert cli.main(["ledger", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "skipped:      skip_cap: estimate 5.00 h > per-run cap 1 h" in text
    assert text.count("skipped:") == 1  # c1 ran: no skip line


def test_worker_error_is_recorded_and_others_continue(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    pf = PipelineFakes(fail={"c1"})
    _install(monkeypatch, pf)
    final, nodes = run_pipeline(out, spec, _ok_fakes())
    by = {w.claim_id: w for w in final["worker_results"]}
    assert by["c1"].status == "error" and "no current plan" in (by["c1"].error or "")
    assert by["c2"].status == "ran" and "evaluate" in nodes


def test_plan_failure_ends_plan_failed(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    _install(monkeypatch, PipelineFakes(plan_ok=False))
    final, nodes = run_pipeline(out, spec, _ok_fakes())
    assert final_setup_status(final) == "plan_failed" and nodes[-1] == "plan"
    assert "mapping rejected" in final["stop_detail"]
    assert not final.get("worker_results") and "evaluations" not in final


# --- stop points -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("until", "no_vault", "last", "keys", "absent"),
    [
        ("plan", False, "plan", ["plans"], ["worker_results", "evaluations", "vault_counts"]),
        ("run", False, "join", ["worker_results"], ["evaluations", "vault_counts"]),
        ("vault", True, "evaluate", ["evaluations"], ["vault_counts"]),
        ("vault", False, "vault", ["evaluations", "vault_counts"], []),
        ("smoke", False, "smoke", ["smoke_log"], ["plans", "plan_outcome"]),
    ],
)
def test_until_stop_points(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, until: str, no_vault: bool,
    last: str, keys: list[str], absent: list[str],
) -> None:  # fmt: skip
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    final, nodes = run_pipeline(out, spec, _ok_fakes(), until=until, no_vault=no_vault)
    assert nodes[-1] == last
    assert all(final.get(k) for k in keys) and not any(final.get(k) for k in absent)
    assert sorted(pf.full_runs) == ([] if until in ("plan", "smoke") else ["c1", "c2"])
    assert (out / "vault_out").exists() is (last == "vault")


# --- idempotent re-run -----------------------------------------------------------------------


def test_rerun_is_idempotent(fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    first_fakes = _ok_fakes()
    first, _ = run_pipeline(out, spec, first_fakes)
    calls = len(RunStore(out).load_call_records())
    assert calls == 4 and pf.llm_calls == 1  # A, B, C + the orchestrator
    assert sum(first_fakes.runs.values()) > 0  # data proof / smoke containers ran

    second_fakes = _ok_fakes()
    second, nodes = run_pipeline(out, spec, second_fakes)
    assert len(RunStore(out).load_call_records()) == calls  # 0 LLM calls
    assert pf.llm_calls == 1 and second["plan_outcome"].reused
    assert second_fakes.runs == {} and second_fakes.builds == []  # 0 containers, 0 builds
    assert sorted(pf.full_runs) == ["c1", "c2"]  # 0 new full runs
    assert second["resumed"] == ["subtasks", "smoke"] and second["stop_reason"] == "resumed"
    assert "passed with the current smoke inputs" in second["stop_detail"]
    assert not {"subtask_a", "subtask_b", "subtask_c", "env", "data", "smoke"} & set(nodes)
    assert {w.status for w in second["worker_results"]} == {"already_completed"}
    assert second["evaluations"] == first["evaluations"]
    assert second["vault_counts"].changed == 0
    assert final_setup_status(second) == "ready"


def test_setup_only_rerun_still_smokes(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is for the full pipeline only: --until smoke re-checks the setup as before."""
    out, spec = fixture
    _install(monkeypatch, PipelineFakes())
    run_pipeline(out, spec, _ok_fakes())
    again = _ok_fakes()
    final, nodes = run_pipeline(out, spec, again, until="smoke")
    assert "smoke" in nodes and again.runs and "resumed" not in final


# --- routing ---------------------------------------------------------------------------------


def test_route_plan_fans_out_one_send_per_claim() -> None:
    state = {"plans": ["a", "b"], "out_dir": "o", "smoke_timeout_s": 5.0, "until": "vault",
             "setup_status": "ready", "spec": object()}  # fmt: skip
    sends = route_plan(state)
    base = {"out_dir": "o", "smoke_timeout_s": 5.0, "worker_order": ["a", "b"]}
    assert sends == [
        Send("claim_worker", {**base, "worker_claim_id": "a"}),
        Send("claim_worker", {**base, "worker_claim_id": "b"}),
    ]
    assert route_plan({**state, "setup_status": "plan_failed"}) == END
    assert route_plan({**state, "until": "plan"}) == END


# --- CLI -------------------------------------------------------------------------------------


def test_reproduce_cli_full_then_idempotent(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    from reproducer import cli
    from reproducer.providers.llm import Budget

    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    fakes = _ok_fakes()
    budget = Budget()
    monkeypatch.setattr(cli, "llm_for_node", lambda node, b: fakes.llm_for(budget)(node))
    for name, fn in {"_plan_env": fakes.plan_env, "build_image": fakes.build_image,
                     "image_size_bytes": lambda tag: 1, "image_exists": fakes.image_exists,
                     "run_container": fakes.runner, "ensure_datasets": fakes.ensure_datasets,
                     "unload_ollama": fakes.unload}.items():  # fmt: skip
        monkeypatch.setattr(cli, name, fn)
    monkeypatch.delenv("REPRODUCER_VAULT", raising=False)
    args = ["reproduce", "--spec", str(spec), "--out", str(out)]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "=== plan ===" in text and "=== runs (pre-flight + execute" in text
    assert "=== ledger:" in text and "status:       verified" in text
    assert "vault:          created=" in text and "Papers/proj.md" in text
    assert "LLM:            4 call(s)" in text

    def no_orchestrator(node: str, b: Any) -> Any:  # 2nd run: no orchestrator config at all
        if node == "orchestrator":
            raise ValueError("no LLM configured for node 'orchestrator'")
        return fakes.llm_for(budget)(node)

    monkeypatch.setattr(cli, "llm_for_node", no_orchestrator)
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "resumed:        subtasks, smoke" in text
    assert "reused stored mapping (no LLM call)" in text
    assert text.count("already completed") == 2
    assert "vault:          created=0 updated=0 unchanged=" in text
    assert "LLM:            0 call(s)" in text
    assert "containers:     0 run(s), 0 image build call(s)" in text


# --- resume only with the current smoke inputs ----------------------------------------------


def _assert_no_smoke_resume(final: dict[str, Any], nodes: list[str], fakes: Fakes) -> None:
    assert final["resumed"] == ["subtasks"]  # A/B/C reused, the setup is checked again
    assert "smoke" in nodes and fakes.runs  # smoke container(s) ran
    assert final_setup_status(final) == "ready" and final["stop_reason"] == "passed"


def test_changed_env_plan_means_no_smoke_resume(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    _install(monkeypatch, PipelineFakes())
    run_pipeline(out, spec, _ok_fakes())
    again = _ok_fakes()
    base = again.plan_env
    again.plan_env = lambda o, cpu: dataclasses.replace(base(o, cpu), tag="reproducer/proj:v2")  # type: ignore[method-assign]
    final, nodes = run_pipeline(out, spec, again)
    _assert_no_smoke_resume(final, nodes, again)
    assert "env" in nodes and again.builds == ["reproducer/proj:v2"]
    # the new inputs passed: the next re-run resumes again, with no container
    third = _ok_fakes()
    third.plan_env = again.plan_env  # type: ignore[method-assign]
    third.built = set(again.built)
    final, nodes = run_pipeline(out, spec, third)
    assert final["resumed"] == ["subtasks", "smoke"] and third.runs == {}


def test_changed_patch_stack_means_no_smoke_resume(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    _install(monkeypatch, PipelineFakes())
    first, _ = run_pipeline(out, spec, _ok_fakes())
    diff = ("--- a/README.md\n+++ b/README.md\n@@ -1,2 +1,3 @@\n # demo\n"
            " Trains a small model.\n+patched\n")  # fmt: skip
    RunStore(out).append_evidence(make_evidence(
        kind="critic_patch", source=first["source"], locator="critic:README.md",
        content={"ok": True, "file": "README.md", "diff": diff, "edits": [],
                 "rationale": "test", "error_signature": "sig"},
        confidence="inferred", authority="llm",
    ))  # fmt: skip
    again = _ok_fakes()
    final, nodes = run_pipeline(out, spec, again)
    _assert_no_smoke_resume(final, nodes, again)


# --- GPU in plan order ---------------------------------------------------------------------


def test_gpu_turns_follow_plan_order_whatever_the_start_order() -> None:
    turns = pl.GpuTurns()
    order = ["c1", "c2", "c3"]
    got: list[str] = []

    def worker(cid: str) -> None:
        with turns.hold(order, cid):
            got.append(cid)
            time.sleep(0.01)

    for _ in range(3):  # also: the record resets after each fan-out
        got.clear()
        threads = [threading.Thread(target=worker, args=(c,)) for c in reversed(order)]
        for t in threads:  # c3 first, then c2, then c1
            t.start()
            time.sleep(0.02)
        for t in threads:
            t.join(timeout=5)
        assert got == order


def test_gpu_turns_skip_a_missing_worker_after_the_grace_period() -> None:
    """c1 (first in order) arrives late: after the grace period the earliest waiting worker
    (c2) gets the GPU; c1, arriving while c2 holds it, is next again (plan order among the
    present workers), then c3."""
    turns = pl.GpuTurns(grace_s=0.2)
    order = ["c1", "c2", "c3"]
    got: list[str] = []

    def worker(cid: str, hold_s: float) -> None:
        with turns.hold(order, cid):
            got.append(cid)
            time.sleep(hold_s)

    threads = {c: threading.Thread(target=worker, args=(c, 0.6 if c == "c2" else 0.01))
               for c in order}  # fmt: skip
    threads["c3"].start()
    time.sleep(0.02)
    threads["c2"].start()
    time.sleep(0.4)  # c2 has had the GPU since ~0.2 s
    threads["c1"].start()
    for t in threads.values():
        t.join(timeout=5)
    assert not any(t.is_alive() for t in threads.values())
    assert got == ["c2", "c1", "c3"]


def _run_with_hang_guard(fn: Any, timeout_s: float = 60.0) -> Any:
    """Run fn in a thread; fail (instead of hanging the suite) if it does not finish."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    assert not t.is_alive(), f"pipeline hung for {timeout_s} s"
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.mark.parametrize("max_concurrency", [1, None])
def test_five_claims_complete_without_hang_in_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_concurrency: int | None
) -> None:
    """LangGraph max_concurrency=1 (one thread for five workers) must not deadlock the GPU
    turns; with every worker present (default pool) the GPU goes in plan order."""
    out, spec = make_setup_fixture(tmp_path)
    data = json.loads(spec.read_text())
    data["claims"] = CLAIMS5
    spec.write_text(json.dumps(data))
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    config = {} if max_concurrency is None else {"max_concurrency": max_concurrency}
    final, nodes = _run_with_hang_guard(
        lambda: run_pipeline(out, spec, _ok_fakes(), config=config)  # fresh GpuTurns
    )
    assert final["plans"] == FIVE and nodes.count("claim_worker") == 5
    assert pf.gpu_order == FIVE and pf.max_active == 1
    assert {e.status for e in final["evaluations"].values()} == {"verified"}
    assert nodes[-1] == "vault"


def test_gpu_turns_broken_by_a_dead_worker_do_not_hang() -> None:
    turns = pl.GpuTurns()
    errors: list[str] = []

    def second() -> None:
        try:
            with turns.hold(["a", "b"], "b"):
                errors.append("ran")
        except RuntimeError as e:
            errors.append(str(e))

    t = threading.Thread(target=second)
    t.start()
    with pytest.raises(KeyError):
        with turns.hold(["a", "b"], "a"):
            raise KeyError("boom")
    t.join(timeout=5)
    assert not t.is_alive() and "abandoned" in errors[0]
    with turns.hold(["a", "b"], "a"):  # a fresh fan-out works again
        pass


# --- per-claim planning ----------------------------------------------------------------------


def test_claim_that_does_not_plan_stays_open_others_run(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    from reproducer import cli

    out, spec = fixture
    pf = PipelineFakes(unplanned={"c1"})
    _install(monkeypatch, pf)
    final, nodes = run_pipeline(out, spec, _ok_fakes())
    assert final["plans"] == ["c2"]
    assert final["plan_failures"] == {"c1": "metric line not found"}
    assert [w.claim_id for w in final["worker_results"]] == ["c2"] and pf.full_runs == ["c2"]
    evals = final["evaluations"]
    assert (evals["c1"].status, evals["c2"].status) == ("open", "verified")
    assert nodes[-1] == "vault" and cli.pipeline_failed(final)
    note = (out / "vault_out" / "Claims" / "proj__c1.md").read_text()
    assert "- Plan: plan_failed: metric line not found (no current run plan)" in note
    assert "plan_failed" not in (out / "vault_out" / "Claims" / "proj__c2.md").read_text()
    capsys.readouterr()
    cli.print_pipeline(final, out, RunStore(out))
    text = capsys.readouterr().out
    assert "c1: not planned (stays open): plan_failed: metric line not found" in text
    assert "skipped:      plan_failed: metric line not found" in text


def test_no_claim_planned_ends_plan_failed(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    _install(monkeypatch, PipelineFakes(unplanned={"c1", "c2"}))
    final, nodes = run_pipeline(out, spec, _ok_fakes())
    assert final_setup_status(final) == "plan_failed" and nodes[-1] == "plan"
    assert "c1: metric line not found" in final["stop_detail"]


# --- lazy orchestrator LLM -------------------------------------------------------------------


def test_orchestrator_llm_made_only_when_a_call_is_needed(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    run_pipeline(out, spec, _ok_fakes())
    assert pf.llms_made == 1

    fakes = _ok_fakes()
    base = fakes.deps(RunStore(out))["llm_for"]

    def no_orchestrator(node: str) -> Any:
        if node == "orchestrator":
            raise ValueError("no LLM configured for node 'orchestrator'")
        return base(node)

    final, _ = run_pipeline(out, spec, fakes, extra_deps={"llm_for": no_orchestrator})
    assert final["plan_outcome"].reused and pf.llms_made == 1  # reused: no LLM made
    assert final_setup_status(final) == "ready" and "vault_counts" in final


def test_missing_orchestrator_config_fails_plan_only_when_a_call_is_needed(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = fixture
    _install(monkeypatch, PipelineFakes())
    fakes = _ok_fakes()
    base = fakes.deps(RunStore(out))["llm_for"]

    def no_orchestrator(node: str) -> Any:
        if node == "orchestrator":
            raise ValueError("no LLM configured for node 'orchestrator'")
        return base(node)

    final, nodes = run_pipeline(out, spec, fakes, extra_deps={"llm_for": no_orchestrator})
    assert final_setup_status(final) == "plan_failed" and nodes[-1] == "plan"
    assert "no LLM configured for node 'orchestrator'" in final["stop_detail"]


def test_verified_claims_never_reflect_and_reruns_stay_idempotent(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Reflexion loop on (as `reproduce` runs it by default) with every claim within
    tolerance (like Cutout): no reflect node, no reflect LLM, and a re-run adds nothing."""
    out, spec = fixture
    pf = PipelineFakes()
    _install(monkeypatch, pf)
    fakes = _ok_fakes()
    final, nodes = run_pipeline(out, spec, fakes, reflect=True)
    assert {e.status for e in final["evaluations"].values()} == {"verified"}
    assert "reflect" not in nodes and "apply_lesson" not in nodes
    assert {i["outcome"] for i in final["reflect_info"].values()} == {"verified"}
    before = (out / "evidence.jsonl").read_text()
    again = _ok_fakes()
    again.built = set(fakes.built)
    final2, nodes2 = run_pipeline(out, spec, again, reflect=True)
    assert "reflect" not in nodes2 and "reflect" not in fakes.llm_nodes + again.llm_nodes
    assert sorted(pf.full_runs) == ["c1", "c2"]  # the first invocation only
    assert (out / "evidence.jsonl").read_text() == before
