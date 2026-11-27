"""Human-approval gates with fakes: full_run (run_claims), critic_patch (setup graph) and
shared_vault_write (pipeline vault node); stop -> human decision -> resume."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_execute import METRIC_LINE, PATTERN, SPEC, FakeRunner
from test_pipeline import CLAIMS, PipelineFakes, _install, _ok_fakes, run_pipeline
from test_setup_graph import (
    Fakes,
    _index_scenario,
    _item_patch,
    _kinds,
    make_setup_fixture,
    run_setup,
)

import reproducer.graph.nodes.execute as ex
from reproducer.domain.approval import approval_decision_evidence, pending_requests
from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evidence import make_evidence
from reproducer.graph.nodes.critic import MAX_CRITIC_ATTEMPTS, CriticState
from reproducer.graph.nodes.pipeline import awaiting_approval
from reproducer.graph.routing import final_setup_status
from reproducer.providers.store import RunStore
from reproducer.rules.known_fixes import FixState


def _decide(out: Path, decision: str, note: str = "") -> str:
    store = RunStore(out)
    (req,) = pending_requests(store.load_evidence())
    store.append_evidence(approval_decision_evidence(req, decision, note))  # type: ignore[arg-type]
    return req.evidence_id


def _requests(out: Path) -> list[Any]:
    return [e for e in RunStore(out).load_evidence().values() if e.kind == "approval_request"]


# --- full_run --------------------------------------------------------------------------------


def _claims_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sec_per_epoch: float):  # type: ignore[no-untyped-def]
    """run_claims over one fake plan; the stored smoke gives the estimate
    (sec_per_epoch x 200 argparse-default epochs)."""
    out = tmp_path / "out"
    store = RunStore(out)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('hi')\n")
    metric_ev = make_evidence(kind="repo_file", source="repo:x@0", locator="train.py#metric_lines",
                              content={"result": [{"line": METRIC_LINE}]}, confidence="derived",
                              authority="repo")  # fmt: skip
    plan_ev = make_evidence(kind="run_plan", source="repo:x@0", locator="run_plan:c10",
                            content={"claim_id": "c10"}, confidence="derived", authority="rule",
                            links=("m" * 64,))  # fmt: skip
    smoke_ev = make_evidence(kind="smoke_result", source="repo:x@0", locator="smoke:1",
                             content={"passed": True, "attempts": 1, "capped": False,
                                      "sec_per_epoch": sec_per_epoch, "timing": {"method": "wall"},
                                      "duration_s": 1.0}, confidence="observed",
                             authority="execution")  # fmt: skip
    for ev in (metric_ev, plan_ev, smoke_ev):
        store.append_evidence(ev)
    claim = ResultClaim(claim_id="c10", metric="test_error", value=4.0, unit="%",
                        dataset="CIFAR-10", split="test", setting=None, location="T1",
                        quote="4.0")  # fmt: skip
    plan = types.SimpleNamespace(
        entrypoint="train.py", args=(), depends_on_unknowns=(),
        metric_source=types.SimpleNamespace(metric_line_evidence_id=metric_ev.evidence_id,
                                            pattern=PATTERN, which="last",
                                            conversion="accuracy_fraction_to_error_percent"),
        is_spec_faithful=lambda states: False,
    )  # fmt: skip
    planned = types.SimpleNamespace(claim=claim, plan=plan, evidence=plan_ev, notes=(),
                                    deviations=(), assumptions=(),
                                    command="python train.py")  # fmt: skip
    current = types.SimpleNamespace(target=types.SimpleNamespace(argparse_spec=SPEC),
                                    input_digest="I" * 64, plans=(planned,))  # fmt: skip

    @dataclass(frozen=True)
    class FakeSetup:
        example: str = ""
        source: str = "repo:x@0"
        repo_dir: Path = repo
        fix_state0: FixState = FixState(prep_dirs=("logs", "checkpoints"))
        critic_state0: CriticState = field(default_factory=CriticState)
        known_fixes0: int = 0
        cache: str = str(tmp_path / "cache")
        base_mounts: tuple[tuple[str, str], ...] = ()

    env_ev = make_evidence(kind="env_spec", source="repo:x@0", locator="env_plan",
                           content={"plan": {}}, confidence="derived", authority="rule",
                           links=("e" * 64,))  # fmt: skip
    monkeypatch.setattr(ex, "load_current_plans", lambda store, repo_dir, partial=False: current)
    monkeypatch.setattr(ex, "prepare_smoke", lambda out, store, entrypoint: FakeSetup())
    monkeypatch.setattr(ex, "make_build", lambda setup, out, store, **kw: (
        lambda state: types.SimpleNamespace(image="img:tag", arguments=())))  # fmt: skip
    monkeypatch.setattr(ex, "_stored_smoke", lambda evidence, source, digest: smoke_ev)
    monkeypatch.setattr(ex, "env_plan_to_evidence", lambda *a, **kw: env_ev)
    plan_env = types.SimpleNamespace(
        plan=types.SimpleNamespace(deviations=()), gpu=None,
        spec_evidence=types.SimpleNamespace(source="repo:x@0", evidence_id="s" * 64),
    )  # fmt: skip
    runner = CountingRunner()

    def run(env: dict[str, str] | None = None) -> list[Any]:
        deps = {
            "runner": runner, "plan_env": lambda out, cpu: plan_env,
            "image_exists": lambda image: True, "build_image": None, "unload": lambda: None,
            "make_critic": lambda setup: None, "env": env or {},
        }  # fmt: skip
        return ex.run_claims(out, store, deps)

    return out, run, runner


class CountingRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, *a: Any, **kw: Any) -> Any:  # type: ignore[override]
        self.calls += 1
        return super().__call__(*a, **kw)


def test_full_run_pending_then_approved_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, run, runner = _claims_harness(tmp_path, monkeypatch, sec_per_epoch=36.0)  # 2.0 h
    (r,) = run()
    assert r.status == "awaiting_approval" and runner.calls == 0  # nothing executed
    assert r.approval is not None and r.approval.request is not None
    (req,) = _requests(out)
    c = req.content
    assert c["action"] == "full_run" and c["subject"] == "c10" and c["cost"] == {"gpu_h": 2.0}
    assert c["detail"]["command"] == r.command and req.links == (r.preflight.evidence.evidence_id,)
    assert not [e for e in RunStore(out).load_evidence().values() if e.kind == "run_result"]

    (again,) = run()  # re-run: same input digest -> same request, no duplicate
    assert again.status == "awaiting_approval" and runner.calls == 0 and len(_requests(out)) == 1

    _decide(out, "approved", "go")
    (done,) = run()
    assert done.status == "ran" and runner.calls == 1 and done.report is not None
    assert len(_requests(out)) == 1
    (final,) = run()  # completed: skipped by run digest, no new request
    assert final.status == "already_completed" and runner.calls == 1


def test_full_run_rejected_stays_open_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reproducer.cli import skip_reason

    out, run, runner = _claims_harness(tmp_path, monkeypatch, sec_per_epoch=36.0)
    run()
    _decide(out, "rejected", "too expensive for this week")
    (r,) = run()
    assert r.status == "rejected" and runner.calls == 0
    assert r.approval.rejected_reason == "rejected_by_human: too expensive for this week"  # type: ignore[union-attr]
    assert skip_reason(RunStore(out).load_evidence(), "c10") == (
        "rejected_by_human: too expensive for this week"
    )
    assert len(_requests(out)) == 1


def test_full_run_below_threshold_or_off_needs_no_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, run, runner = _claims_harness(tmp_path, monkeypatch, sec_per_epoch=36.0)
    (r,) = run({"REPRODUCER_APPROVAL": "off"})
    assert r.status == "ran" and runner.calls == 1 and _requests(out) == []


def test_full_run_strict_gates_a_cheap_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, run, runner = _claims_harness(tmp_path, monkeypatch, sec_per_epoch=1.0)  # 0.06 h
    assert run()[0].status == "ran"  # default: below the 1 h threshold
    out2, run2, runner2 = _claims_harness(tmp_path / "b", monkeypatch, sec_per_epoch=1.0)
    assert run2({"REPRODUCER_APPROVAL": "strict"})[0].status == "awaiting_approval"


# --- critic_patch (setup graph) --------------------------------------------------------------


@pytest.fixture
def setup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    return make_setup_fixture(tmp_path)


def _gated(critic: Any = _item_patch) -> Fakes:
    return Fakes(_index_scenario, critic=critic, approval="default")


def test_critic_patch_pending_keeps_patch_off_the_stack_then_approve_applies(
    setup_fixture: tuple[Path, Path],
) -> None:
    out, spec = setup_fixture
    fakes = _gated()
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "awaiting_approval"
    assert final["stop_reason"] == "awaiting_approval" and awaiting_approval(final)
    assert final["critic_state"].stack == () and final.get("critic_attempts", 0) == 0
    assert _kinds(out).count("critic_patch") == 0  # not stored, not applied
    (req,) = _requests(out)
    assert req.content["action"] == "critic_patch" and req.content["subject"] == "train.py"
    assert ".item()" in req.content["detail"]["diff"] and req.content["cost"]["changed_lines"] == 2
    assert len(fakes.critic_prompts) == 1

    again = _gated()
    final = run_setup(out, spec, again)  # re-run: same request, no LLM call, no duplicate
    assert final_setup_status(final) == "awaiting_approval"
    assert again.critic_prompts == [] and len(_requests(out)) == 1

    _decide(out, "approved")
    after = _gated()
    final = run_setup(out, spec, after)
    assert final_setup_status(final) == "ready" and after.critic_prompts == []  # stored patch
    assert len(final["critic_state"].stack) == 1 and final["critic_attempts"] == 1
    assert _kinds(out).count("critic_patch") == 1 and len(_requests(out)) == 1


def test_critic_patch_rejected_is_a_failed_attempt_and_the_note_reaches_the_critic(
    setup_fixture: tuple[Path, Path],
) -> None:
    out, spec = setup_fixture
    run_setup(out, spec, _gated())
    _decide(out, "rejected", "do not hide the bug, use .item()")
    fakes = _gated()
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "awaiting_approval"  # a new patch for the human
    assert len(fakes.critic_prompts) == 1
    assert "human rejected this patch: do not hide the bug, use .item()" in fakes.critic_prompts[0]
    assert final["critic_attempts"] == 1  # the rejection counts
    rejected = [e for e in RunStore(out).load_evidence().values()
                if e.kind == "critic_patch" and e.content.get("rejected_by_human")]  # fmt: skip
    assert len(rejected) == 1 and rejected[0].content["ok"] is False
    assert len(_requests(out)) == 2 and final["critic_state"].stack == ()


def test_critic_rejections_hit_the_same_cap(setup_fixture: tuple[Path, Path]) -> None:
    out, spec = setup_fixture
    run_setup(out, spec, _gated())
    prompts = 0
    for i in range(MAX_CRITIC_ATTEMPTS):
        _decide(out, "rejected", f"no {i}")
        fakes = _gated()
        final = run_setup(out, spec, fakes)
        prompts += len(fakes.critic_prompts)
    assert final_setup_status(final) == "capped" and final["stop_reason"] == "max_critic"
    assert final["critic_attempts"] == MAX_CRITIC_ATTEMPTS
    assert prompts == MAX_CRITIC_ATTEMPTS - 1  # no critic call once the cap is reached
    assert pending_requests(RunStore(out).load_evidence()) == []


# --- shared_vault_write ----------------------------------------------------------------------


@pytest.fixture
def pipeline_fixture(tmp_path: Path) -> tuple[Path, Path]:
    import json

    out, spec = make_setup_fixture(tmp_path)
    data = json.loads(spec.read_text())
    data["claims"] = CLAIMS
    spec.write_text(json.dumps(data))
    return out, spec


def test_shared_vault_write_waits_for_approval(
    pipeline_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, spec = pipeline_fixture
    _install(monkeypatch, PipelineFakes())
    vault = out / "vault_out"
    env = {"REPRODUCER_SHARED_VAULT": str(vault)}
    final, _ = run_pipeline(out, spec, _ok_fakes(), env=env)
    assert final["vault_approval"].status == "pending" and awaiting_approval(final)
    assert "vault_counts" not in final and not vault.exists()  # nothing written
    (req,) = _requests(out)
    assert req.content["action"] == "shared_vault_write"
    assert req.content["subject"] == str(vault.resolve())

    final, _ = run_pipeline(out, spec, _ok_fakes(), env=env)  # same notes -> same request
    assert final["vault_approval"].status == "pending" and len(_requests(out)) == 1

    _decide(out, "approved")
    final, _ = run_pipeline(out, spec, _ok_fakes(), env=env)
    assert final["vault_counts"].created and not awaiting_approval(final)
    assert len(_requests(out)) == 1


def test_private_vault_is_not_gated(
    pipeline_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out, spec = pipeline_fixture
    _install(monkeypatch, PipelineFakes())
    env = {"REPRODUCER_SHARED_VAULT": str(tmp_path / "team_vault")}
    final, _ = run_pipeline(out, spec, _ok_fakes(), env=env)
    assert final["vault_counts"].created and "vault_approval" not in final
    assert _requests(out) == []


# --- CLI: reproduce exit 4 + queue / approve / reject / answer ------------------------------


def _patch_cli(monkeypatch: pytest.MonkeyPatch, fakes: Fakes) -> Any:
    from reproducer import cli
    from reproducer.providers.llm import Budget

    budget = Budget()
    monkeypatch.setattr(cli, "llm_for_node", lambda node, b: fakes.llm_for(budget)(node))
    for name, fn in {"_plan_env": fakes.plan_env, "build_image": fakes.build_image,
                     "image_size_bytes": lambda tag: 1, "image_exists": fakes.image_exists,
                     "run_container": fakes.runner, "ensure_datasets": fakes.ensure_datasets,
                     "unload_ollama": fakes.unload}.items():  # fmt: skip
        monkeypatch.setattr(cli, name, fn)
    monkeypatch.delenv("REPRODUCER_VAULT", raising=False)
    monkeypatch.delenv("REPRODUCER_APPROVAL", raising=False)
    return cli


def test_cli_reproduce_awaits_then_approve_resumes(
    pipeline_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    out, spec = pipeline_fixture
    _install(monkeypatch, PipelineFakes())
    cli = _patch_cli(monkeypatch, _ok_fakes())
    monkeypatch.setenv("REPRODUCER_SHARED_VAULT", str(out / "vault_out"))
    args = ["reproduce", "--spec", str(spec), "--out", str(out)]
    assert cli.main(args) == 4
    text = capsys.readouterr().out
    assert "=== awaiting approval (final status awaiting_approval) ===" in text
    assert "approval requests: 1 pending" in text and "shared_vault_write" in text
    assert "vault:          awaiting_approval (request" in text

    (req,) = _requests(out)
    rid = req.evidence_id[:12]
    assert f"approve --out {out} {rid}" in text and f"reject --out {out} {rid}" in text
    assert cli.main(["approve", "--out", str(out), "ffffffffffff"]) == 1  # unknown id
    assert cli.main(["approve", "--out", str(out), rid, "--note", "ok"]) == 0
    assert cli.main(["approve", "--out", str(out), rid, "--note", "again"]) == 1  # no change
    capsys.readouterr()
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "vault:          created=" in text and "awaiting approval" not in text

    # standing approval: a later reject revokes it (no new request), a later approve restores
    assert cli.main(["reject", "--out", str(out), rid, "--note", "late"]) == 0
    assert "revoked:" in capsys.readouterr().out
    assert cli.main(args) == 0  # a human rejection is not a failure
    text = capsys.readouterr().out
    assert "rejected_by_human: late" in text and "nothing written" in text
    assert cli.main(["approve", "--out", str(out), rid, "--note", "ok"]) == 0
    assert "restored:" in capsys.readouterr().out
    assert cli.main(args) == 0
    assert "vault:          created=" in capsys.readouterr().out
    assert len(_requests(out)) == 1


def test_cli_queue_lists_unknowns_and_answer_resolves(
    setup_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    import json

    from reproducer.domain.unknowns import fold_unknowns

    out, spec = setup_fixture
    data = json.loads(spec.read_text())
    momentum = {"name": "momentum", "value": None, "location": "Sec 4", "quote": "SGD"}
    data["setup"] = [*data.get("setup", []), momentum]
    spec.write_text(json.dumps(data))
    cli = _patch_cli(monkeypatch, _gated())
    assert cli.main(["reproduce", "--spec", str(spec), "--out", str(out), "--until", "smoke"]) == 4
    capsys.readouterr()

    store = RunStore(out)
    before = store.load_evidence()
    assert cli.main(["queue", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert store.load_evidence() == before  # read-only
    assert "approval requests: 1 pending" in text and "critic_patch" in text
    assert "| " in text and ".item()" in text  # the diff is shown for review
    assert "open unknowns:" in text and "\n  momentum" in text
    assert f"answer --out {out} momentum <value>" in text

    assert cli.main(["answer", "--out", str(out), "momentum", "0.9", "--note", "authors"]) == 0
    states = fold_unknowns(store.load_events(), store.load_evidence())
    assert states["unk:momentum"].status == "resolved" and states["unk:momentum"].value == "0.9"
    assert cli.main(["answer", "--out", str(out), "momentum", "0.5", "--note", "x"]) == 1
    capsys.readouterr()
    cli.main(["queue", "--out", str(out)])
    assert "\n  momentum" not in capsys.readouterr().out


# --- HITL review fixes: standing vault approval, vault command gate, rejected run exit 0 -----


def test_shared_vault_approval_is_standing_per_path(
    pipeline_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from reproducer.domain.approval import ApprovalPolicy, decisions_for
    from reproducer.graph.nodes.pipeline import vault_gate

    out, _ = pipeline_fixture
    store = RunStore(out)
    store.append_evidence(make_evidence(kind="repo_file", source="t", locator="x", content={},
                                        confidence="derived", authority="rule"))  # fmt: skip
    vault = out / "team_vault"
    policy = ApprovalPolicy(shared_vault=str(vault))
    g = vault_gate(store, vault, {"Papers/a.md": "v1"}, policy)
    assert g.status == "pending" and g.new_request
    rid = _decide(out, "approved", "ok")
    # changed content: same standing request, still approved, no new request
    g = vault_gate(store, vault, {"Papers/a.md": "v2", "Claims/b.md": "x"}, policy)
    assert g.status == "approved" and not g.new_request and len(_requests(out)) == 1
    # a later reject revokes it; a later approve restores it (newest decision counts)
    req = RunStore(out).load_evidence()[rid]
    store.append_evidence(approval_decision_evidence(req, "rejected", "stop", revision=1))
    assert vault_gate(store, vault, {"Papers/a.md": "v3"}, policy).status == "rejected"
    store.append_evidence(approval_decision_evidence(req, "approved", "ok", revision=2))
    assert vault_gate(store, vault, {"Papers/a.md": "v4"}, policy).status == "approved"
    assert len(_requests(out)) == 1 and len(decisions_for(store.load_evidence(), rid)) == 3
    # another vault path is its own standing request
    other = out / "team_vault2"
    g = vault_gate(store, other, {"Papers/a.md": "v4"}, ApprovalPolicy(shared_vault=str(other)))
    assert g.status == "pending" and len(_requests(out)) == 2


def test_cli_vault_command_is_gated_for_the_shared_vault(
    pipeline_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    out, spec = pipeline_fixture
    _install(monkeypatch, PipelineFakes())
    run_pipeline(out, spec, _ok_fakes(), env={})  # a store with notes to render
    cli = _patch_cli(monkeypatch, _ok_fakes())
    team = out.parent / "team_vault"
    monkeypatch.setenv("REPRODUCER_SHARED_VAULT", str(team))
    capsys.readouterr()

    # a private vault is not gated (and nothing is written to the store)
    before = RunStore(out).load_evidence()
    assert cli.main(["vault", "--out", str(out), "--vault", str(out.parent / "mine")]) == 0
    assert RunStore(out).load_evidence() == before

    args = ["vault", "--out", str(out), "--vault", str(team)]
    assert cli.main(args) == 4
    text = capsys.readouterr().out
    assert "awaiting_approval" in text and "nothing written" in text and not team.exists()
    (req,) = [r for r in _requests(out) if r.content["action"] == "shared_vault_write"]
    rid = req.evidence_id[:12]
    assert f"approve --out {out} {rid}" in text
    assert cli.main(args) == 4 and len(_requests(out)) == 1  # same request, no duplicate

    assert cli.main(["approve", "--out", str(out), rid]) == 0
    capsys.readouterr()
    assert cli.main(args) == 0 and (team / "Papers").is_dir()
    assert "created:" in capsys.readouterr().out

    assert cli.main(["reject", "--out", str(out), rid, "--note", "freeze"]) == 0  # revoke
    capsys.readouterr()
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "rejected by human: freeze" in text and "nothing written" in text
    assert len(_requests(out)) == 1


def test_rejected_full_run_exits_0_with_clear_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reproducer import cli
    from reproducer.cli import pipeline_failed
    from reproducer.domain.approval import GateResult
    from reproducer.graph.nodes.pipeline import WorkerResult

    out = tmp_path / "out"
    RunStore(out).append_evidence(make_evidence(kind="repo_file", source="t", locator="x",
                                                content={}, confidence="derived",
                                                authority="rule"))  # fmt: skip
    req = make_evidence(kind="approval_request", source="repo", locator="approval:full_run:c10",
                        content={"action": "full_run", "subject": "c10"}, confidence="derived",
                        authority="rule", links=("p" * 64,))  # fmt: skip
    dec = approval_decision_evidence(req, "rejected", "too expensive")
    rejected = ex.ClaimRun("c10", "python train.py", "img", "d" * 64, "rejected",
                           approval=GateResult("rejected", "too expensive", req, dec))  # fmt: skip
    monkeypatch.setattr(cli, "run_claims", lambda *a, **kw: [rejected])
    monkeypatch.delenv("REPRODUCER_APPROVAL", raising=False)
    assert cli.main(["run", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "status:       rejected by human: too expensive" in text
    assert "the claim stays open" in text
    assert not pipeline_failed({"worker_results": [WorkerResult("c10", (rejected,))]})
    skipped = ex.ClaimRun("c10", "python train.py", "img", "d" * 64, "skipped")
    assert pipeline_failed({"worker_results": [WorkerResult("c10", (skipped,))]})
