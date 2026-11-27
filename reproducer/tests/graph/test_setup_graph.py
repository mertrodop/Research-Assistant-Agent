"""The setup graph (graph/build.py build_setup_graph) end to end with fakes, no Docker / LLM:
spec -> ingest -> configure (FakeLLM A/B/C on a local fixture repo) -> env (fake build) ->
data (fake provider) -> smoke (fake container runner that "runs" the per-run copy by reading
it) -> known fixes / critic (FakeLLM) -> END. Counters are checked after every step.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.env_rules import EnvAssumption, EnvPlan
from reproducer.domain.import_scan import ImportedPackage, ImportScan
from reproducer.domain.ledger import Tolerance, ToleranceConfig, compute_ledger
from reproducer.domain.unknowns import fold_unknowns
from reproducer.graph.build import SETUP_RECURSION_LIMIT, build_setup_graph
from reproducer.graph.nodes.env import PlannedEnv, stored_env_spec
from reproducer.graph.nodes.smoke import PROBE_OK
from reproducer.graph.routing import MAX_CRITIC_ATTEMPTS, MAX_KNOWN_FIXES, final_setup_status
from reproducer.providers.docker_runner import BuildResult, RunResult
from reproducer.providers.git import clone_repo
from reproducer.providers.llm import Budget, BudgetExceeded, FakeLLM, LLMError, MeteredLLM
from reproducer.providers.store import RunStore
from reproducer.rules.known_fixes import fold_fix_state

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, env=_GIT_ENV, check=True)


ANSWERS: dict[str, Callable[[dict[str, str]], Any]] = {
    "subtask_a": lambda ids: {
        "packages": [
            {
                "name": "numpy",
                "version": None,
                "citation": {"evidence_id": ids["requirements.txt"], "quote": "numpy"},
            }
        ],
        "unknowns": [{"name": "python_version", "why": "not stated"}],
    },
    "subtask_c": lambda ids: {"unknowns": [{"name": "dataset_source", "why": "not given"}]},
}

TRAIN = (
    "import argparse\n"
    "p = argparse.ArgumentParser()\n"
    "p.add_argument('--epochs', type=int, default=10)\n"
    "args = p.parse_args()\n"
    "import torch\n"
    "loss = torch.tensor(1.5)\n"
    "x = loss.data[0]\n"
    "open('logs/x.csv', 'w').write('epoch,acc\\n')\n"
    "print('test_acc: %.3f' % (loss.item()))\n"
)
BAD_LINE = TRAIN.split("\n").index("x = loss.data[0]") + 1
WRITE_LINE = TRAIN.split("\n").index("open('logs/x.csv', 'w').write('epoch,acc\\n')") + 1
SPEC = {
    "code_links": ["https://github.com/org/proj"],  # never contacted: <out>/repo exists
    "datasets": ["CIFAR-10"],
    "claims": [
        {"claim_id": "c1", "metric": "test_acc", "value": 96.0, "dataset": "CIFAR-10",
         "split": "test", "setting": "s", "location": "Table 1", "quote": "96.0"}
    ],
    "setup": [{"name": "batch_size", "value": None, "location": "Sec 4", "quote": "q"}],
}  # fmt: skip
PASS_LOG = "1/1 epoch\ntest_acc: 0.960\n"
INDEX_MSG = "invalid index of a 0-dim tensor. Use `tensor.item()` in Python"
BASE_TAG = "reproducer/proj:base"


def _tb(line: int, src: str, exc: str) -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "/work/train.py", line {line}, in <module>\n'
        f"    {src}\n"
        f"{exc}\n"
    )


INDEX_TB = _tb(BAD_LINE, "x = loss.data[0]", f"IndexError: {INDEX_MSG}")


def _no_dir_tb(d: str) -> str:
    return _tb(
        WRITE_LINE,
        f"open('{d}/x.csv', 'w').write('epoch,acc\\n')",
        f"FileNotFoundError: [Errno 2] No such file or directory: '{d}/x.csv'",
    )


NO_YAML_TB = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 7, in <module>\n'
    '  File "/opt/conda/lib/python3.10/importlib/__init__.py", line 126, in import_module\n'
    "ModuleNotFoundError: No module named 'yaml'\n"
)


# --- fixture repo + spec ----------------------------------------------------------------------


def make_setup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "README.md").write_text("# demo\nTrains a small model.\n")
    (src / "requirements.txt").write_text("numpy\n")
    (src / "train.py").write_text(TRAIN)
    (src / "scripts" / "run.sh").write_text("python train.py --epochs 3\n")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "init")
    out = tmp_path / "runs" / "proj"
    clone_repo(src.as_uri(), out / "repo")
    spec = tmp_path / "evals" / "proj" / "spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(SPEC))
    return out, spec


def _b_answer(prompt: str) -> dict[str, Any]:
    m = re.search(r"- \[(E\d+)\] \"python train.py --epochs 3\"", prompt)
    assert m, "the run.sh command must be offered to subtask B"
    example = {"evidence_id": m.group(1), "quote": "python train.py --epochs 3"}
    return {"entrypoints": [{"path": "train.py", "kind": "train", "example": example}]}


# --- fakes ------------------------------------------------------------------------------------

CriticAnswer = Callable[[str, int], str]  # (prompt, call index) -> JSON answer


class Fakes:
    """All injected IO for one graph run. scenario(stage, image, workdir, n) -> (exit, log)."""

    def __init__(
        self,
        scenario: Callable[[str, str, Path, int], tuple[int | None, str]],
        critic: CriticAnswer | None = None,
        critic_budget: Budget | None = None,
        critic_configured: bool = True,
        approval: str = "off",  # REPRODUCER_APPROVAL; the approval gate tests set it
    ) -> None:
        self.scenario = scenario
        self.approval = approval
        self.critic = critic
        self.critic_budget = critic_budget
        self.critic_configured = critic_configured
        self.built: set[str] = set()
        self.builds: list[str] = []
        self.ensure_calls: list[list[str]] = []
        self.llm_nodes: list[str] = []
        self.critic_prompts: list[str] = []
        self.unloads = 0
        self.runs: dict[str, int] = {}

    # docker
    def build_image(self, dockerfile: str, tag: str, *, timeout_s: float, log_path: Path):  # type: ignore[no-untyped-def]
        cached = tag in self.built
        self.built.add(tag)
        self.builds.append(tag)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("built\n")
        return BuildResult(ok=True, tag=tag, cached=cached, duration_s=0.1, log_path=log_path)

    def image_exists(self, tag: str) -> bool:
        return tag in self.built

    def runner(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        stage = name.rsplit("-", 1)[1]
        self.runs[stage] = self.runs.get(stage, 0) + 1
        code, log = self.scenario(stage, image, Path(workdir), self.runs[stage])
        Path(log_path).write_text(log)
        return RunResult(
            exit_code=code, timed_out=False, oom_killed=False, duration_s=1.0,
            log_path=Path(log_path), name=name,
        )  # fmt: skip

    # env plan: base tag, + one tag per set of add_package packages (from stored known_fix)
    def plan_env(self, out: Path, cpu_only: bool) -> PlannedEnv:
        store = RunStore(out)
        spec_ev = stored_env_spec(store)
        pkgs = fold_fix_state(store.load_evidence().values()).packages
        scan = ImportScan(
            packages=(
                ImportedPackage(pip_name="torch", module="torch", path="train.py"),
                *(ImportedPackage(pip_name=p, module=p, path="known_fix") for p in pkgs),
            )
        )
        plan = EnvPlan(
            base_image="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime", python="3.7",
            torch="1.12.1", torchvision="0.13.1", cuda="11.3", pip_packages=tuple(pkgs),
            assumptions=(
                EnvAssumption(name="python_version", value="3.7", reason="base image"),
                EnvAssumption(name="unpinned:tqdm", value=None, reason="no version"),
            ),
        )  # fmt: skip
        tag = BASE_TAG if not pkgs else f"reproducer/proj:{'-'.join(pkgs).lower()}"
        return PlannedEnv(plan, None, spec_ev, scan, tag)

    def ensure_datasets(self, names, image, *, log_dir, timeout_s):  # type: ignore[no-untyped-def]
        self.ensure_calls.append(list(names))
        return {}

    def llm_for(self, budget: Budget) -> Callable[[str], MeteredLLM]:
        def factory(node: str) -> MeteredLLM:
            self.llm_nodes.append(node)
            if node == "critic" and not self.critic_configured:
                raise ValueError("no LLM configured for node 'critic'")

            def respond(system: str, prompt: str) -> str:
                if node == "critic":
                    assert self.critic is not None, "critic must not be called"
                    self.critic_prompts.append(prompt)
                    return self.critic(prompt, len(self.critic_prompts))
                if node == "subtask_b":
                    return json.dumps(_b_answer(prompt))
                ids = dict(re.findall(r"## (\S+) \[(E\d+)\]", prompt))
                return json.dumps(ANSWERS[node](ids))

            b = self.critic_budget if node == "critic" and self.critic_budget else budget
            return MeteredLLM(FakeLLM(respond), b, node)

        return factory

    def unload(self) -> list[str]:
        self.unloads += 1
        return []

    def deps(self, store: RunStore) -> dict[str, Any]:
        budget = Budget()
        return {
            "store": store, "llm_for": self.llm_for(budget), "budget": budget,
            "plan_env": self.plan_env, "build_image": self.build_image,
            "image_size": lambda tag: 1, "image_exists": self.image_exists,
            "runner": self.runner, "ensure_datasets": self.ensure_datasets,
            "data_cache_dir": lambda: Path("/tmp/reproducer-test-cache"), "unload": self.unload,
            "env": {"REPRODUCER_APPROVAL": self.approval},
        }  # fmt: skip


def run_setup(out: Path, spec: Path, fakes: Fakes, **inputs: Any) -> dict[str, Any]:
    """Stream the graph, asserting the counter caps after EVERY step; returns the final state."""
    graph = build_setup_graph(fakes.deps(RunStore(out)))  # type: ignore[arg-type]
    init: dict[str, Any] = {
        "out_dir": str(out), "spec_path": str(spec), "commit": None, "max_chars": 12_000,
        "max_tokens": 1_500, "status": "running", "configure_attempts": 0,
        "setup_status": "running", "known_fixes_used": 0, "smoke_iterations": 0,
        "env_builds": 0, **inputs,
    }  # fmt: skip
    final: dict[str, Any] = init
    for state in graph.stream(
        init, config={"recursion_limit": SETUP_RECURSION_LIMIT}, stream_mode="values"
    ):
        assert state.get("known_fixes_used", 0) <= MAX_KNOWN_FIXES
        assert state.get("critic_attempts", 0) <= MAX_CRITIC_ATTEMPTS
        final = state
    return final


def _ok(stage: str) -> tuple[int, str]:
    return (
        (0, PROBE_OK)
        if stage == "probe"
        else (0, "REPRODUCER_IMPORTS {}" if stage == "imports" else PASS_LOG)
    )


def _kinds(out: Path) -> list[str]:
    """Evidence kinds; env plans (kind env_spec, locator env_plan) count as "env_plan"."""
    return [
        "env_plan" if e.locator == "env_plan" else e.kind
        for e in RunStore(out).load_evidence().values()
    ]


def _ledger_status(out: Path, spec: Path) -> str:
    from reproducer.domain.contracts import ReproSpec

    store = RunStore(out)
    evidence = store.load_evidence()
    claims = ReproSpec.model_validate_json(spec.read_text()).claims
    unknowns = fold_unknowns(store.load_events(), evidence)
    ledger = compute_ledger(claims, evidence, unknowns, ToleranceConfig(default=Tolerance(abs=0.5)))
    return ledger["c1"].status


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path]:
    return make_setup_fixture(tmp_path)


# --- scenarios --------------------------------------------------------------------------------


def test_pass_first_time_is_ready(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    fakes = Fakes(lambda stage, image, work, n: _ok(stage))
    final = run_setup(out, spec, fakes)
    assert final["status"] == "configured"
    assert final_setup_status(final) == "ready" and final["stop_reason"] == "passed"
    assert final["image_tag"] == BASE_TAG and fakes.builds == [BASE_TAG]
    env = final["env_log"][0]
    assert env.unpinned == ("tqdm",) and env.assumptions == ("python_version = 3.7: base image",)
    assert final["data_ok"] and fakes.ensure_calls == [[]]  # no dataset call in the repo
    assert [s.signal for s in final["smoke_log"]] == ["passed_exit0"]
    assert final["known_fixes_used"] == 0 and final.get("critic_attempts", 0) == 0
    assert "critic" not in fakes.llm_nodes and fakes.unloads >= 2  # before data and smoke
    kinds = _kinds(out)
    assert kinds.count("smoke_result") == 1 and kinds.count("env_plan") == 1
    assert _ledger_status(out, spec) == "open"  # smoke passing verifies nothing


def test_known_fix_then_pass(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        if stage == "run" and not (work / "logs").is_dir():
            return 1, _no_dir_tb("logs")
        return _ok(stage)

    fakes = Fakes(scenario)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "ready"
    log = final["smoke_log"]
    assert [s.signal for s in log] == ["traceback", "passed_exit0"]
    assert log[0].classification.startswith("known/missing_output_dir")
    assert log[0].fix == "prep_dir {'dir': 'logs'}"
    assert final["known_fixes_used"] == 1 and final["fix_state"].prep_dirs == ("logs",)
    assert fakes.builds == [BASE_TAG]  # prep_dir does not touch the env
    assert _kinds(out).count("known_fix") == 1
    assert "critic" not in fakes.llm_nodes


def test_add_package_rebuilds_env_then_passes(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        if stage == "imports" and image == BASE_TAG:
            return 1, NO_YAML_TB
        return _ok(stage)

    fakes = Fakes(scenario)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "ready"
    log = final["smoke_log"]
    assert [s.signal for s in log] == ["import_error", "passed_exit0"]
    assert log[0].fix is not None and log[0].fix.startswith("add_package")
    assert final["last_fix_kind"] == "add_package"
    assert final["env_builds"] == 2 and len(fakes.builds) == 2
    assert fakes.builds[0] == BASE_TAG and final["image_tag"] == fakes.builds[1] != BASE_TAG
    assert final["env_log"][1].packages_from_fixes == final["fix_state"].packages
    assert len(fakes.ensure_calls) == 1  # data is not provisioned again after the rebuild
    assert _kinds(out).count("env_plan") == 2


def _item_patch(prompt: str, n: int) -> str:
    """Grounded fake critic: patches the line the prompt points at (.data[0] -> .item())."""
    line = int(re.search(r"innermost repo frame is line (\d+)", prompt).group(1))  # type: ignore[union-attr]
    src = re.search(rf"^\s*>>\s+{line}\| (.*)$", prompt, re.MULTILINE).group(1)  # type: ignore[union-attr]
    return _patch(line, src, src.replace(".data[0]", ".item()"))


def _noop_patch(prompt: str, n: int) -> str:
    """Ineffective but valid: a comment on the `loss = ...` line (different each attempt)."""
    m = re.search(r"^\s*(\d+)\| (loss = torch\.tensor\(1\.5\).*)$", prompt, re.MULTILINE)
    assert m, prompt
    return _patch(int(m.group(1)), m.group(2), f"loss = torch.tensor(1.5)  # try {'abc'[n - 1]}")


def _patch(line: int, original: str, replacement: str) -> str:
    return json.dumps(
        {"file": "train.py", "rationale": "fix the crash",
         "edits": [{"start_line": line, "end_line": line, "original": original,
                    "replacement": replacement}]}
    )  # fmt: skip


def _index_scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
    if stage == "run" and "loss.data[0]" in (work / "train.py").read_text():
        return 1, INDEX_TB
    return _ok(stage)


def test_critic_patch_then_pass(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    fakes = Fakes(_index_scenario, critic=_item_patch)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "ready"
    log = final["smoke_log"]
    assert [s.signal for s in log] == ["traceback", "passed_exit0"]
    assert log[0].classification.startswith("critic") and log[0].patch.startswith("patched")
    assert final["critic_attempts"] == 1 and len(final["critic_state"].stack) == 1
    assert len(fakes.critic_prompts) == 1 and fakes.llm_nodes.count("critic") == 1
    assert _kinds(out).count("critic_patch") == 1
    assert (out / "repo" / "train.py").read_text() == TRAIN  # the clone is never patched
    records = RunStore(out).load_call_records()
    assert [r.node for r in records].count("critic") == 1


def test_ineffective_patch_reverted_then_second_patch_passes(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    fakes = Fakes(
        _index_scenario, critic=lambda p, n: (_noop_patch if n == 1 else _item_patch)(p, n)
    )
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "ready"
    log = final["smoke_log"]
    assert [s.signal for s in log] == ["traceback", "traceback", "passed_exit0"]
    assert log[0].reverted is None and log[1].reverted is not None
    assert "Previous patch 1" in fakes.critic_prompts[1]
    assert "did not change the error" in fakes.critic_prompts[1]
    assert final["critic_attempts"] == 2 and len(final["critic_state"].stack) == 1
    kinds = _kinds(out)
    assert kinds.count("critic_patch") == 2 and kinds.count("critic_revert") == 1


def test_infra_stops_and_never_calls_the_critic(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        assert stage == "probe", "repo code must not run after an infra failure"
        return 1, "RuntimeError: CUDA driver initialization failed\n"

    fakes = Fakes(scenario)  # critic=None: any critic call fails the test
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "infra" and final["stop_reason"] == "infra"
    assert [s.signal for s in final["smoke_log"]] == ["infra"]
    assert "critic" not in fakes.llm_nodes and fakes.critic_prompts == []
    assert final["smoke_log"][0].capped is False


def test_critic_cap_is_capped_and_blocks_claims(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    fakes = Fakes(_index_scenario, critic=_noop_patch)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "capped" and final["stop_reason"] == "max_critic"
    assert final["critic_attempts"] == MAX_CRITIC_ATTEMPTS
    assert len(fakes.critic_prompts) == MAX_CRITIC_ATTEMPTS
    log = final["smoke_log"]
    assert len(log) == MAX_CRITIC_ATTEMPTS + 1 and log[-1].capped
    assert all(s.reverted for s in log[1:])  # every patch was ineffective
    assert _ledger_status(out, spec) == "blocked"


def test_known_fix_cap_is_capped(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    dirs = [f"d{i}" for i in range(MAX_KNOWN_FIXES + 1)]

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        missing = [d for d in dirs if not (work / d).is_dir()]
        if stage == "run" and missing:
            return 1, _no_dir_tb(missing[0])
        return _ok(stage)

    fakes = Fakes(scenario)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "capped" and final["stop_reason"] == "max_fixes"
    assert final["known_fixes_used"] == MAX_KNOWN_FIXES
    assert len(final["smoke_log"]) == MAX_KNOWN_FIXES + 1 and final["smoke_log"][-1].capped
    assert _kinds(out).count("known_fix") == MAX_KNOWN_FIXES
    assert _ledger_status(out, spec) == "blocked"


def test_until_env_stops_before_data(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    fakes = Fakes(lambda stage, image, work, n: pytest.fail("nothing may run"))  # type: ignore[arg-type,return-value]
    final = run_setup(out, spec, fakes, until="env")
    assert final_setup_status(final) == "stopped" and final["env_ok"]
    assert fakes.ensure_calls == [] and not final.get("smoke_log")


def test_bad_spec_ends_configure_failed(tmp_path: Path) -> None:
    out, _ = make_setup_fixture(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    fakes = Fakes(lambda stage, image, work, n: pytest.fail("nothing may run"))  # type: ignore[arg-type,return-value]
    final = run_setup(out, bad, fakes)
    assert final_setup_status(final) == "configure_failed"
    assert fakes.builds == [] and fakes.llm_nodes == []


# --- the reproduce CLI over the same fakes ----------------------------------------------------


def test_reproduce_cli_prints_summary_and_ledger(
    fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reproducer import cli

    out, spec = fixture

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        if stage == "run" and not (work / "logs").is_dir():
            return 1, _no_dir_tb("logs")
        return _ok(stage)

    fakes = Fakes(scenario)
    budget = Budget()
    monkeypatch.setattr(cli, "llm_for_node", lambda node, b: fakes.llm_for(budget)(node))
    monkeypatch.setattr(cli, "_plan_env", fakes.plan_env)
    monkeypatch.setattr(cli, "build_image", fakes.build_image)
    monkeypatch.setattr(cli, "image_size_bytes", lambda tag: 1)
    monkeypatch.setattr(cli, "image_exists", fakes.image_exists)
    monkeypatch.setattr(cli, "run_container", fakes.runner)
    monkeypatch.setattr(cli, "ensure_datasets", fakes.ensure_datasets)
    monkeypatch.setattr(cli, "unload_ollama", fakes.unload)
    assert cli.main(["reproduce", "--spec", str(spec), "--out", str(out), "--until", "smoke"]) == 0
    captured = capsys.readouterr()
    text = captured.out
    assert "configure:      configured" in text
    assert f"env:            {BASE_TAG}  ok=True" in text
    assert "smoke:          2 iteration(s)" in text
    assert "fix:            prep_dir {'dir': 'logs'}" in text
    assert "counters:       known_fixes_used=1/5  critic_attempts=0/3" in text
    assert "stop:           ready (passed: smoke passed (passed_exit0))" in text
    assert re.search(r"tokens:         local=0 api=\d+ calls=3", text)  # fake = not local
    assert "ledger:" in text and "c1  metric=test_acc" in text
    assert "[apply_known_fix] #1 traceback [known/missing_output_dir] fix=prep_dir" in captured.err
    assert "[smoke] #2 passed_exit0 -> stop passed" in captured.err


# --- critic cannot run: retryable vs cap (M2-S6 review) ---------------------------------------


def _backend_down(prompt: str, n: int) -> str:
    raise LLMError("backend unreachable", status=503)


@pytest.mark.parametrize(
    "fakes_kw",
    [{"critic": _backend_down}, {"critic": _item_patch, "critic_configured": False}],
    ids=["backend_error", "not_configured"],
)
def test_critic_llm_unavailable_is_retryable_not_capped(
    fixture: tuple[Path, Path], fakes_kw: dict[str, Any]
) -> None:
    out, spec = fixture
    final = run_setup(out, spec, Fakes(_index_scenario, **fakes_kw))
    assert final_setup_status(final) == "llm_unavailable"
    assert final["stop_reason"] == "llm_unavailable"
    log = final["smoke_log"]
    assert len(log) == 1 and log[0].capped is False
    assert _kinds(out).count("critic_patch") == 0  # no attempt was made
    assert final.get("critic_attempts", 0) == 0
    assert _ledger_status(out, spec) == "open"  # claims stay open: retry later


def test_critic_budget_exceeded_is_a_cap(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    tiny = Budget(max_tokens_total=10)  # the critic reserves CRITIC_MAX_TOKENS: refused
    fakes = Fakes(_index_scenario, critic=_item_patch, critic_budget=tiny)
    final = run_setup(out, spec, fakes)
    assert final_setup_status(final) == "capped" and final["stop_reason"] == "budget"
    assert "BudgetExceeded" in final["stop_detail"]
    assert final["smoke_log"][-1].capped is True
    assert fakes.critic_prompts == []  # refused before any call
    assert _ledger_status(out, spec) == "blocked"
    assert issubclass(BudgetExceeded, RuntimeError)


def test_known_fix_cap_holds_across_reproduce_runs(fixture: tuple[Path, Path]) -> None:
    """Two runs on the same out dir: the cap counts stored known_fix evidence, so the second
    run applies no further fix (5 in total) and stops at max_fixes right away."""
    out, spec = fixture
    dirs = [f"d{i}" for i in range(MAX_KNOWN_FIXES + 3)]

    def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
        missing = [d for d in dirs if not (work / d).is_dir()]
        if stage == "run" and missing:
            return 1, _no_dir_tb(missing[0])
        return _ok(stage)

    first = run_setup(out, spec, Fakes(scenario))
    assert first["stop_reason"] == "max_fixes"
    assert first["known_fixes_used"] == MAX_KNOWN_FIXES
    second = run_setup(out, spec, Fakes(scenario))
    assert second["stop_reason"] == "max_fixes"
    assert second["known_fixes_used"] == MAX_KNOWN_FIXES
    assert len(second["smoke_log"]) == 1  # one smoke, no fix
    assert second["fix_state"].prep_dirs == tuple(dirs[:MAX_KNOWN_FIXES])  # stored fixes kept
    assert _kinds(out).count("known_fix") == MAX_KNOWN_FIXES


def test_partial_fixes_then_second_run_uses_the_remaining_cap(fixture: tuple[Path, Path]) -> None:
    out, spec = fixture
    dirs = [f"d{i}" for i in range(MAX_KNOWN_FIXES + 3)]

    def make(limit: int):  # type: ignore[no-untyped-def]
        def scenario(stage: str, image: str, work: Path, n: int) -> tuple[int, str]:
            missing = [d for d in dirs[:limit] if not (work / d).is_dir()]
            if stage == "run" and missing:
                return 1, _no_dir_tb(missing[0])
            return _ok(stage)

        return scenario

    first = run_setup(out, spec, Fakes(make(3)))  # 3 fixes, then passes
    assert final_setup_status(first) == "ready" and first["known_fixes_used"] == 3
    second = run_setup(out, spec, Fakes(make(len(dirs))))  # needs 5 more: only 2 left
    assert second["stop_reason"] == "max_fixes"
    assert second["known_fixes_used"] == MAX_KNOWN_FIXES
    assert _kinds(out).count("known_fix") == MAX_KNOWN_FIXES


def test_summary_prints_unpinned_on_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    from reproducer.cli import print_setup_summary
    from reproducer.graph.nodes.env import EnvSummary

    env = EnvSummary(
        image="img:1", base_image="b", torch="1.12.1", python="3.7", cuda="11.3",
        pip=("tqdm", "numpy"), deviations=(), assumptions=("python_version = 3.7: r",),
        cached=True, ok=True, build_s=0.0, log=None, unpinned=("tqdm", "numpy"),
    )  # fmt: skip
    print_setup_summary({"env_log": [env], "status": "configured"}, [], 0.0)
    text = capsys.readouterr().out
    assert "  unpinned:     tqdm, numpy (no version stated; pip resolves)" in text
    assert "= None" not in text and text.count("assumption:") == 1


def test_reproduce_cli_rejects_unknown_claim_before_anything_runs(
    fixture: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    from reproducer import cli

    out, spec = fixture
    before = sorted(p.name for p in out.parent.iterdir()) if out.parent.exists() else []
    argv = ["reproduce", "--spec", str(spec), "--out", str(out), "--claim", "nope"]
    assert cli.main(argv) == 2
    assert "unknown claim 'nope'" in capsys.readouterr().err
    after = sorted(p.name for p in out.parent.iterdir()) if out.parent.exists() else []
    assert after == before and not (out / "evidence.jsonl").exists()


def test_node_exception_is_a_clean_fallback_stop(fixture: tuple[Path, Path]) -> None:
    """An unhandled exception in a node (here: unloading Ollama before data) is not a crash:
    fallback "fallback: <node>: <error type>", later nodes are no-ops, no smoke ran, the claim
    stays open."""
    out, spec = fixture
    fakes = Fakes(lambda stage, image, work, n: _ok(stage))

    def broken_unload() -> list[str]:
        raise RuntimeError("ollama socket vanished")

    fakes.unload = broken_unload  # type: ignore[method-assign]
    final = run_setup(out, spec, fakes)
    assert final["fallback_reason"] == "fallback: data: RuntimeError"
    assert final["fallback_detail"] == "ollama socket vanished"
    assert final_setup_status(final) == "fallback"
    assert "smoke_result" not in _kinds(out) and fakes.runs.get("run", 0) == 0
    assert _ledger_status(out, spec) == "open"
