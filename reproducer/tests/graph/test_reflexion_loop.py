"""The Reflexion loop (M4-S3) on the SYNTHETIC toy paper (evals/toy_reflexion), end to end:
spec -> configure (local snapshot) -> env -> data -> smoke -> plan -> run -> evaluate ->
reflect -> apply_lesson -> re-run / re-evaluate -> evaluate -> vault.

Real: every node, the orchestrator (claim mapping reused: one LLM call), build_run_plan with
the applied lessons, execute (pre-flight + run digest), evaluate, diagnose, breaker, reflect
(gate + Lesson contract), apply_lesson, vault. Faked: Docker (the runner emulates the toy's
deterministic formula from the command; untrusted code never runs on the host) and the LLMs
(scripted JSON per node; reflect answers per (claim, trial))."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_pipeline import run_pipeline
from test_setup_graph import Fakes

from reproducer.domain.evidence import Evidence
from reproducer.graph.nodes.apply_lesson import apply_change, trial_causes
from reproducer.graph.nodes.smoke import PROBE_OK
from reproducer.graph.routing import decide_reflect, route_reflect
from reproducer.providers.docker_runner import RunResult
from reproducer.providers.llm import Budget, FakeLLM, MeteredLLM
from reproducer.providers.store import RunStore

TOY = Path(__file__).resolve().parents[2] / "evals" / "toy_reflexion"
BASE_ERR = {"toyA": 2.0, "toyB": 9.0}
METRIC_LINE = 'print("test_err: %.2f" % (test_err), flush=True)'
SCALE = "test_err = 100.0 * (BASE_ERR[args.dataset] / 100.0"
NOISE_QUOTE = "we add Gaussian input noise with a standard deviation of 0.05"


# --- route_reflect: the decision table -------------------------------------------------------


def _info(outcome: str | None, *, can_try: bool = True, applied: str | None = None,
          trial: int = 1, tripped: str = "") -> dict[str, Any]:  # fmt: skip
    return {"trial": trial, "outcome": outcome, "can_try": can_try, "tripped": tripped,
            "applied": applied}  # fmt: skip


@pytest.mark.parametrize(
    ("info", "failed", "decision", "reason"),
    [
        (_info(None, trial=0), None, "open", "no completed trial"),
        (_info("verified"), None, "done", "verified"),
        (_info("out_of_tolerance"), None, "reflect", ""),
        (_info("diverged"), None, "reflect", ""),
        (_info("crashed"), None, "open", "crashed: not reflected"),
        (_info("not_completed"), None, "open", "not_completed: not reflected"),
        (_info("out_of_tolerance", can_try=False, tripped="claim c: 3/3 trials used"), None,
         "open", "breaker: claim c: 3/3 trials used"),
        (_info("out_of_tolerance", can_try=False, tripped="gpu: 10 h spent"), None, "open",
         "breaker: gpu"),
        (_info("out_of_tolerance", applied="stopped_no_change"), None, "open",
         "reflexion stopped: no grounded change"),
        (_info("out_of_tolerance", applied="applied"), None, "open", "no new run of it yet"),
        (_info("out_of_tolerance"), {"trial": 1, "reason": "gate"}, "open", "reflection failed"),
        (_info("out_of_tolerance", trial=2), {"trial": 1, "reason": "x"}, "reflect", ""),
        (_info("verified", applied="applied", trial=2), None, "done", "verified"),
    ],
)  # fmt: skip
def test_decide_reflect_table(
    info: dict[str, Any], failed: dict[str, Any] | None, decision: str, reason: str
) -> None:
    d, why = decide_reflect(info, failed)
    assert d == decision and reason in why


def test_route_reflect() -> None:
    oot = {"c": _info("out_of_tolerance")}
    assert route_reflect({"reflect": True, "reflect_info": oot}) == "reflect"
    assert route_reflect({"reflect": False, "reflect_info": oot}) == "vault"
    assert route_reflect({"reflect_info": oot}) == "vault"  # absent = off
    assert route_reflect({"reflect": True, "reflect_info": oot, "no_vault": True}) == "reflect"
    done = {"c": _info("verified")}
    assert route_reflect({"reflect": True, "reflect_info": done, "no_vault": True}) == "__end__"
    assert route_reflect({"reflect": True, "reflect_info": done}) == "vault"
    assert route_reflect({"reflect": True, "reflect_info": oot, "evaluate_error": "x"}) == "__end__"


# --- the toy paper with fakes ----------------------------------------------------------------


def toy_output(cmd: list[str], *, log: Callable[[str, int, float], list[float]] | None) -> str:
    """The toy's deterministic stdout for this command (emulated; nothing runs on the host)."""
    tokens = shlex.split(" ".join(cmd))
    i = next(i for i, t in enumerate(tokens) if t.endswith("train.py"))
    args = dict(zip(tokens[i + 1 :: 2], tokens[i + 2 :: 2], strict=False))
    dataset, epochs = args.get("--dataset", "toyA"), int(args.get("--epochs", "3"))
    noise = float(args.get("--noise", "0.2"))
    errs = [100.0 * (BASE_ERR[dataset] / 100.0 + abs(noise - 0.05) + 0.01 * (epochs - 1 - e))
            for e in range(epochs)]  # fmt: skip
    if log is not None:
        errs = log(dataset, epochs, noise)
    return "".join(f"test_err: {v:.2f}\n" for v in errs)


def mapping_answer(prompt: str) -> dict[str, Any]:
    claims = [
        {"claim_id": cid, "claim_args": [{"flag": "--dataset", "value": ds}],
         "setup_mappings": [{"setup_name": "epochs", "flag": "--epochs"}],
         "metric": {"metric_line_ref": "E2",
                    "metric_line_quote": {"evidence_id": "E2", "quote": METRIC_LINE},
                    "conversion": "identity",
                    "scale_citation": {"evidence_id": "E1", "quote": SCALE}}}
        for cid, ds in (("toyA_err", "toyA"), ("toyB_err", "toyB"))
    ]  # fmt: skip
    unmapped = [{"setup_name": "noise", "reason": "the value is prose, not a --noise float"}]
    if "- noise = UNKNOWN" in prompt:
        unmapped = []
    return {"claims": claims, "unmapped_setup": unmapped, "setup_checks": []}


def b_answer(prompt: str) -> dict[str, Any]:
    m = re.search(r'- \[(E\d+)\] "(python train.py --dataset toyA[^"]*)"', prompt)
    assert m, "the README command must be offered to subtask B"
    example = {"evidence_id": m.group(1), "quote": m.group(2)}
    return {"entrypoints": [{"path": "train.py", "kind": "train", "example": example}]}


Reflector = Callable[[str, int, str], dict[str, Any]]  # (claim_id, trial, prompt) -> lesson


def lesson(cid: str, trial: int, step: str, text: str,
           change: dict[str, Any] | None) -> dict[str, Any]:  # fmt: skip
    return {"claim_id": cid, "trial": trial, "step": step, "evidence_ids": ["RUN"],
            "text": text, "proposed_change": change}  # fmt: skip


def alias_of(prompt: str, needle: str) -> str:
    m = re.search(rf"\[([PRH]\d+)\] [^\n]*{re.escape(needle)}", prompt)
    assert m, f"{needle!r} must be citable in the reflect prompt"
    return m.group(1)


def noise_fix(cid: str, trial: int, prompt: str) -> dict[str, Any]:
    p = alias_of(prompt, NOISE_QUOTE)
    ch = {
        "kind": "arg",
        "target": "--noise",
        "new_value": "0.05",
        "support": {"evidence_id": p, "quote": NOISE_QUOTE},
        "rationale": "the paper states noise std 0.05; the run used the README's 0.2",
    }
    return lesson(cid, trial, "config_vs_paper",
                  "Error is about 15 points above the paper; the run used --noise 0.2 but the "
                  "paper states 0.05.", ch)  # fmt: skip


def stop(cid: str, trial: int, prompt: str) -> dict[str, Any]:
    ch = {
        "kind": "none",
        "target": "none",
        "new_value": None,
        "support": None,
        "rationale": "nothing in the paper or repo explains the remaining gap",
    }
    return lesson(cid, trial, "training", "Noise now matches the paper; the remaining gap has "
                  "no grounded explanation in the observations.", ch)  # fmt: skip


def default_reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
    return noise_fix(cid, trial, prompt) if trial == 1 else stop(cid, trial, prompt)


class ToyFakes(Fakes):
    """Fakes for the toy: subtask / orchestrator / reflect answers and the emulated runner."""

    def __init__(self, reflector: Reflector = default_reflector,
                 log: Callable[[str, int, float], list[float]] | None = None) -> None:  # fmt: skip
        super().__init__(lambda *a: (0, ""))
        self.reflector = reflector
        self.log = log
        self.reflect_prompts: list[str] = []
        self.full_runs: list[str] = []

    def runner(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        stage = name.rsplit("-", 1)[1]
        if stage == "probe":
            text = PROBE_OK
        elif stage == "imports":
            text = "REPRODUCER_IMPORTS {}"
        else:
            if "-smoke-" not in name and not name.endswith("-smoke"):
                self.full_runs.append(" ".join(cmd))
            text = toy_output(list(cmd), log=self.log)
        Path(log_path).write_text(text)
        return RunResult(exit_code=0, timed_out=False, oom_killed=False, duration_s=1.0,
                         log_path=Path(log_path), name=name)  # fmt: skip

    def llm_for(self, budget: Budget) -> Callable[[str], MeteredLLM]:
        def factory(node: str) -> MeteredLLM:
            self.llm_nodes.append(node)

            def respond(system: str, prompt: str) -> str:
                if node == "subtask_a":
                    return json.dumps({"packages": [], "unknowns": []})
                if node == "subtask_b":
                    return json.dumps(b_answer(prompt))
                if node == "subtask_c":
                    return json.dumps({"unknowns": []})
                if node == "orchestrator":
                    return json.dumps(mapping_answer(prompt))
                if node == "reflect":
                    self.reflect_prompts.append(prompt)
                    m = re.search(r"## Claim (\S+) \(claim_id \S+, trial (\d+)\)", prompt)
                    assert m
                    return json.dumps(self.reflector(m.group(1), int(m.group(2)), prompt))
                raise AssertionError(f"unexpected LLM node {node}")

            return MeteredLLM(FakeLLM(respond), budget, node)

        return factory


def make_toy(tmp_path: Path, *, setup: dict[str, Any] | None = None,
             noise_help: str | None = None) -> tuple[Path, Path]:  # fmt: skip
    """A copy of the toy spec (its file: link pointing at a copy of toyrepo) -> (out, spec)."""
    repo = tmp_path / "toyrepo"
    shutil.copytree(TOY / "toyrepo", repo)
    if noise_help is not None:
        train = repo / "train.py"
        train.write_text(train.read_text().replace("std of the Gaussian input noise", noise_help))
    data = json.loads((TOY / "spec.json").read_text())
    data["code_links"] = [f"file:{repo}"]
    if setup is not None:
        data["setup"] = [s for s in data["setup"] if s["name"] not in setup] + [
            {"location": "-", **v, "name": k} for k, v in setup.items()
        ]  # fmt: skip
    spec = tmp_path / "evals" / "toy_reflexion" / "spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(data))
    return tmp_path / "runs" / "toy", spec


def run_toy(out: Path, spec: Path, fakes: ToyFakes, **inputs: Any) -> dict[str, Any]:
    final, _ = run_pipeline(out, spec, fakes, env={"REPRODUCER_APPROVAL": "off"},
                            reflect=True, **inputs)  # fmt: skip
    return final


def kinds(store: RunStore, kind: str) -> list[Evidence]:
    return [ev for ev in store.load_evidence().values() if ev.kind == kind]


def statuses(final: dict[str, Any]) -> dict[str, str]:
    return {cid: e.status for cid, e in final["evaluations"].items()}


def values(store: RunStore, cid: str) -> list[float | None]:
    return [ev.content["metrics"].get("test_error") for ev in kinds(store, "run_result")
            if ev.content["claim_id"] == cid]  # fmt: skip


def test_toy_miss_then_paper_grounded_arg_lesson_then_verified(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)
    fakes = ToyFakes()
    final = run_toy(out, spec, fakes)
    store = RunStore(out)
    assert statuses(final) == {"toyA_err": "verified", "toyB_err": "open"}
    assert values(store, "toyA_err") == pytest.approx([17.0, 2.0])  # 2 trials
    assert values(store, "toyB_err") == pytest.approx([24.0, 9.0])  # the fix, then no more
    assert fakes.llm_nodes.count("orchestrator") == 1  # re-plans reuse the stored mapping
    assert len(fakes.reflect_prompts) == 3  # toyA t1, toyB t1, toyB t2
    applied = {(ev.content["claim_id"], ev.content["trial"]): ev.content["outcome"]
               for ev in kinds(store, "lesson_applied")}  # fmt: skip
    assert applied == {("toyA_err", 1): "applied", ("toyB_err", 1): "applied",
                       ("toyB_err", 2): "stopped_no_change"}  # fmt: skip
    # the second toyA run is the plan with the lesson's paper value, everything else unchanged
    assert any("--dataset toyA --epochs 3 --noise 0.05" in c for c in fakes.full_runs)
    plans = [ev for ev in kinds(store, "run_plan") if ev.content["claim_id"] == "toyA_err"]
    arg = next(a for a in plans[-1].content["plan"]["args"] if a["flag"] == "--noise")
    assert arg["value"] == "0.05" and arg["source"] == "paper"
    les = next(ev for ev in kinds(store, "lesson") if ev.content["claim_id"] == "toyA_err")
    assert arg["evidence_id"] == les.content["lesson"]["proposed_change"]["support"]["evidence_id"]
    # the loop's reasons
    log = "\n".join(final["reflexion_log"])
    assert "toyB_err] trial 2: stop: reflexion stopped: no grounded change" in log
    # trial-2 prompt: previous lesson labeled and the authoritative unknowns listed
    p2 = next(p for p in fakes.reflect_prompts if "trial 2)" in p)
    assert "LLM-written, unverified" in p2 and "- eval_checkpoint: open (not stated" in p2
    assert "trial 1 (config_vs_paper)" in p2
    # vault: lesson notes + the open reason on the claim note
    vault = out / "vault_out"
    note = (vault / "Lessons" / "toy_reflexion__toyA_err__t1.md").read_text()
    assert "LLM-written, unverified" in note and NOISE_QUOTE in note and "- applied:" in note
    claim_b = (vault / "Claims" / "toy_reflexion__toyB_err.md").read_text()
    assert "Open after 2 trial(s)" in claim_b
    assert "reflexion stopped: no grounded change" in claim_b
    assert (vault / "Lessons" / "toy_reflexion__toyB_err__t2.md").is_file()

    # idempotent: a re-run reflects on nothing new, runs nothing, calls no LLM
    again = ToyFakes(reflector=lambda *a: pytest.fail("no reflection expected"))
    again.built = set(fakes.built)
    final2 = run_toy(out, spec, again)
    assert statuses(final2) == statuses(final) and again.full_runs == []
    assert "reflect" not in again.llm_nodes and "orchestrator" not in again.llm_nodes


def test_eval_protocol_lesson_reevaluates_without_running(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)

    def log(dataset: str, epochs: int, noise: float) -> list[float]:
        return [2.05, 2.0, 5.0] if dataset == "toyA" else [9.0, 9.0, 9.0]

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        if cid == "toyA_err" and trial == 1:
            ch = {
                "kind": "eval_protocol",
                "target": "metric_source.which",
                "new_value": "best",
                "support": None,
                "rationale": "the best epoch matches; the checkpoint is open",
            }
            return lesson(cid, trial, "eval_protocol", "The final epoch jumps to 5.0 while the "
                          "best epoch is 2.0; the eval checkpoint is not stated (assumed last).",
                          ch)  # fmt: skip
        return stop(cid, trial, prompt)

    fakes = ToyFakes(reflector=reflector, log=log)
    final = run_toy(out, spec, fakes)
    store = RunStore(out)
    assert statuses(final)["toyA_err"] == "verified"
    runs = [ev for ev in kinds(store, "run_result") if ev.content["claim_id"] == "toyA_err"]
    assert len(runs) == 2 and len([c for c in fakes.full_runs if "toyA" in c]) == 1
    reeval = runs[1].content
    assert reeval["reeval_of"] == runs[0].evidence_id and reeval["gpu_hours"] == 0.0
    assert reeval["metrics"]["test_error"] == pytest.approx(2.0)
    les = next(ev for ev in kinds(store, "lesson") if ev.content["claim_id"] == "toyA_err")
    assert les.evidence_id in runs[1].links and runs[0].evidence_id in runs[1].links
    plan = [ev for ev in kinds(store, "run_plan") if ev.content["claim_id"] == "toyA_err"][-1]
    assert plan.content["plan"]["metric_source"]["which"] == "best"


def test_assumption_lesson_resolves_the_unknown_then_reruns(tmp_path: Path) -> None:
    # variant: the paper leaves the noise open; the repo's help text states the value
    out, spec = make_toy(tmp_path, setup={"noise": {"value": None, "quote": ""}},
                         noise_help="std of the input noise (the paper uses 0.05)")  # fmt: skip
    hint = "the paper uses 0.05"

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        if trial == 1 and cid == "toyA_err":
            r = alias_of(prompt, "train.py code")
            ch = {
                "kind": "assumption",
                "target": "unk:noise",
                "new_value": "0.05",
                "support": {"evidence_id": r, "quote": hint},
                "rationale": "the repo help text states 0.05; 0.2 was assumed",
            }
            return lesson(cid, trial, "config_vs_paper", "The noise was assumed 0.2 from the "
                          "README; the repo states 0.05 for the paper runs.", ch)  # fmt: skip
        return stop(cid, trial, prompt)

    fakes = ToyFakes(reflector=reflector)
    final = run_toy(out, spec, fakes)
    store = RunStore(out)
    assert statuses(final)["toyA_err"] == "verified"
    assert values(store, "toyA_err") == pytest.approx([17.0, 2.0])
    resolved = [e for e in store.load_events() if e.event == "resolved"]
    assert [(e.unknown_id, e.value) for e in resolved] == [("unk:noise", "0.05")]
    sup = store.load_evidence()[resolved[0].evidence_id]
    assert sup.authority == "repo"  # resolved by repo evidence, never by the LLM
    # cross-claim: toyB's own lesson was "none", but the resolved (shared) unknown changed its
    # plan, so it re-runs with the resolved value as its trial 2, caused by toyA's change
    assert values(store, "toyB_err") == pytest.approx([24.0, 9.0])
    plan_b = [ev for ev in kinds(store, "run_plan") if ev.content["claim_id"] == "toyB_err"][-1]
    assert "--noise 0.05" in plan_b.content["command"]
    toy_a_applied = next(ev for ev in kinds(store, "lesson_applied")
                         if ev.content["claim_id"] == "toyA_err")  # fmt: skip
    causes = trial_causes(store.load_evidence(), "toyB_err")
    assert [ev.evidence_id for ev in causes[2]] == [toy_a_applied.evidence_id]
    assert trial_causes(store.load_evidence(), "toyA_err") == {}  # its own lesson
    log = "\n".join(final["reflexion_log"])
    assert (
        f"[toyB_err] rerun: its plan changed by lesson_applied {toy_a_applied.evidence_id[:12]}"
        in log
    )
    assert final["reflect_info"]["toyB_err"]["trial"] == 2


def test_only_claim_runs_and_reflects_one_claim_others_untouched(tmp_path: Path) -> None:
    """--claim: every claim is planned, only the selected one runs / is evaluated / reflected.
    toyB ran once (alone); then --claim toyA_err: toyA's lesson resolves a shared unknown that
    changes toyB's plan, but toyB is not re-run (logged) and gets no new evidence."""
    out, spec = make_toy(tmp_path, setup={"noise": {"value": None, "quote": ""}},
                         noise_help="std of the input noise (the paper uses 0.05)")  # fmt: skip
    reflected: list[str] = []

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        reflected.append(cid)
        if cid != "toyA_err":
            return stop(cid, trial, prompt)
        r = alias_of(prompt, "train.py code")
        ch = {"kind": "assumption", "target": "unk:noise", "new_value": "0.05",
              "support": {"evidence_id": r, "quote": "the paper uses 0.05"},
              "rationale": "the repo help text states 0.05; 0.2 was assumed"}  # fmt: skip
        return lesson(cid, trial, "config_vs_paper", "The noise was assumed 0.2; the repo "
                      "states 0.05 for the paper runs.", ch)  # fmt: skip

    fakes = ToyFakes(reflector=reflector)
    first = run_toy(out, spec, fakes, only_claim="toyB_err")
    store = RunStore(out)
    assert set(first["evaluations"]) == {"toyB_err"} and set(reflected) == {"toyB_err"}
    assert values(store, "toyA_err") == [] and values(store, "toyB_err") == pytest.approx([24.0])
    toy_b = {i for i, ev in store.load_evidence().items()
             if ev.content.get("claim_id") == "toyB_err"}  # fmt: skip

    reflected.clear()
    final = run_toy(out, spec, fakes, only_claim="toyA_err")
    assert final["plans"] == ["toyA_err", "toyB_err"]  # both planned
    assert statuses(final) == {"toyA_err": "verified"}  # only the selected claim evaluated
    assert set(final["reflect_info"]) == {"toyA_err"} and set(reflected) == {"toyA_err"}
    assert values(store, "toyA_err") == pytest.approx([17.0, 2.0])
    assert values(store, "toyB_err") == pytest.approx([24.0])  # not re-run
    log = "\n".join(final["reflexion_log"])
    assert "[toyB_err] plan changed by lesson_applied" in log
    assert "not re-run: --claim toyA_err" in log
    after = {i for i, ev in store.load_evidence().items()
             if ev.content.get("claim_id") == "toyB_err"
             and ev.kind in ("run_result", "lesson", "lesson_applied", "preflight")}  # fmt: skip
    assert after <= toy_b  # no new run / lesson / pre-flight evidence for toyB
    assert (out / "vault_out" / "Claims" / "toy_reflexion__toyB_err.md").is_file()  # vault: all


def test_breaker_trips_after_three_trials(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        if cid == "toyA_err" or trial == 2:
            return noise_fix(cid, trial, prompt)
        ch = {
            "kind": "eval_protocol",
            "target": "metric_source.which",
            "new_value": "best",
            "support": None,
            "rationale": "try the best epoch; the checkpoint is not stated",
        }
        return lesson(cid, trial, "eval_protocol", "The checkpoint is not stated; the best "
                      "epoch was not tried yet.", ch)  # fmt: skip

    fakes = ToyFakes(reflector=reflector)
    final = run_toy(out, spec, fakes)
    store = RunStore(out)
    assert statuses(final) == {"toyA_err": "verified", "toyB_err": "open"}
    # toyB: trial 1 (24.0), reeval best epoch (trial 2, 24.0), noise fix (trial 3, 9.0)
    assert values(store, "toyB_err") == pytest.approx([24.0, 24.0, 9.0])
    d, why = decide_reflect(final["reflect_info"]["toyB_err"])
    assert d == "open" and why == "breaker: claim toyB_err: 3/3 trials used"
    assert len([p for p in fakes.reflect_prompts if "Claim toyB_err" in p]) == 2
    claim_b = (out / "vault_out" / "Claims" / "toy_reflexion__toyB_err.md").read_text()
    assert "Open after 3 trial(s)" in claim_b and "trial budget used (3/3)" in claim_b


def test_step_order_is_enforced_across_trials(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        if trial == 1:
            return noise_fix(cid, trial, prompt)
        ch = {
            "kind": "eval_protocol",
            "target": "metric_source.which",
            "new_value": "best",
            "support": None,
            "rationale": "x",
        }
        return lesson(cid, trial, "eval_protocol", "The checkpoint is not stated.", ch)

    final = run_toy(out, spec, ToyFakes(reflector=reflector))
    d, why = decide_reflect(final["reflect_info"]["toyB_err"],
                            final["reflect_failures"].get("toyB_err"))  # fmt: skip
    assert d == "open" and "reflection failed" in why
    assert not [ev for ev in kinds(RunStore(out), "lesson") if ev.content["trial"] == 2]


def test_stale_lesson_is_never_applied(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)
    run_toy(out, spec, ToyFakes())
    store = RunStore(out)
    les = next(ev for ev in kinds(store, "lesson") if ev.content["claim_id"] == "toyB_err"
               and ev.content["trial"] == 2)  # fmt: skip
    before = (out / "evidence.jsonl").read_text()
    with pytest.raises(ValueError, match="stale lesson is never applied"):
        apply_change(out, store, les, input_digest="0" * 64)
    assert (out / "evidence.jsonl").read_text() == before
    # the current one is applied at most once (stored lesson_applied returned as is)
    again = apply_change(out, store, les, input_digest=les.content["input_digest"])
    assert again.reused and again.action == "stop"
    assert (out / "evidence.jsonl").read_text() == before


def test_no_reflect_skips_the_loop(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)
    fakes = ToyFakes(reflector=lambda *a: pytest.fail("--no-reflect"))
    final, nodes = run_pipeline(out, spec, fakes, env={"REPRODUCER_APPROVAL": "off"},
                                reflect=False)  # fmt: skip
    assert "reflect" not in nodes and statuses(final) == {"toyA_err": "open", "toyB_err": "open"}


# --- applied changes that leave the command unchanged -----------------------------------------


def epochs_noop(cid: str, trial: int, prompt: str) -> dict[str, Any]:
    p = alias_of(prompt, "we train for 3 epochs")
    ch = {
        "kind": "arg",
        "target": "--epochs",
        "new_value": "3",
        "support": {"evidence_id": p, "quote": "we train for 3 epochs"},
        "rationale": "the paper states 3 epochs",
    }
    return lesson(cid, trial, "config_vs_paper", "Set epochs to the paper's 3.", ch)  # fmt: skip


def test_unchanged_command_uses_no_trial_then_a_fresh_lesson(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)
    seen: list[tuple[str, int]] = []

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        seen.append((cid, trial))
        if cid == "toyB_err":
            return stop(cid, trial, prompt)
        if seen.count((cid, 1)) == 1:
            return epochs_noop(cid, trial, prompt)  # already 3: the command does not change
        assert "applied, the command did not change" in prompt  # memory of this trial
        return noise_fix(cid, trial, prompt)

    fakes = ToyFakes(reflector=reflector)
    final = run_toy(out, spec, fakes)
    store = RunStore(out)
    assert statuses(final)["toyA_err"] == "verified"
    assert values(store, "toyA_err") == pytest.approx([17.0, 2.0])  # the no-op used no trial
    assert seen.count(("toyA_err", 1)) == 2  # a fresh lesson for the same trial
    applied = [ev.content for ev in kinds(store, "lesson_applied")
               if ev.content["claim_id"] == "toyA_err"]  # fmt: skip
    assert [(a["trial"], a["target"]) for a in applied] == [(1, "--epochs"), (1, "--noise")]
    log = "\n".join(final["reflexion_log"])
    assert "[toyA_err] trial 1: evaluate:" in log and "command is unchanged" in log


def test_repeated_unchanged_commands_stop_the_loop(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        return epochs_noop(cid, trial, prompt) if cid == "toyA_err" else stop(cid, trial, prompt)

    fakes = ToyFakes(reflector=reflector)
    final = run_toy(out, spec, fakes)
    assert values(RunStore(out), "toyA_err") == pytest.approx([17.0])
    d, why = decide_reflect(final["reflect_info"]["toyA_err"])
    assert d == "open" and "2 applied change(s) of trial 1 left the command unchanged" in why


# --- env limits ---------------------------------------------------------------------------------


def test_max_trials_env_limits_the_loop(tmp_path: Path) -> None:
    out, spec = make_toy(tmp_path)

    def reflector(cid: str, trial: int, prompt: str) -> dict[str, Any]:
        if cid == "toyA_err" or trial == 2:
            return noise_fix(cid, trial, prompt)
        ch = {
            "kind": "eval_protocol",
            "target": "metric_source.which",
            "new_value": "best",
            "support": None,
            "rationale": "the checkpoint is not stated",
        }
        return lesson(cid, trial, "eval_protocol", "The checkpoint is not stated.", ch)

    env = {"REPRODUCER_APPROVAL": "off", "REPRODUCER_MAX_TRIALS": "2"}
    final, _ = run_pipeline(out, spec, ToyFakes(reflector=reflector), reflect=True, env=env)
    assert len(values(RunStore(out), "toyB_err")) == 2  # trial 1 + reeval, then the breaker
    d, why = decide_reflect(final["reflect_info"]["toyB_err"])
    assert d == "open" and why.startswith("breaker: claim toyB_err: 2/2 trials used")
    note = (out / "vault_out" / "Claims" / "toy_reflexion__toyB_err.md").read_text()
    assert "trial budget used (2/2)" in note
