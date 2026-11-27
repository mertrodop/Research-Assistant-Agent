"""ReAct critic (graph/nodes/critic_react.py + critic_tools.py), FakeLLM with scripted tool
calls. Marked: llm_live + docker, the IndexError fixture in the Cutout image in react mode."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest
from test_critic import (
    BAD_LINE,
    FIXTURE_TRAIN,
    METRIC_TEXTS,
    PROGRESS,
    VALID,
    PatchAwareRunner,
    _docker_loop,
    _inputs,
    _patch_json,
    _repo,
    _tb,
    cutout_image,  # noqa: F401  (fixture)
)
from test_setup_graph import Fakes, _index_scenario, _kinds, fixture, run_setup  # noqa: F401

from reproducer.domain.traceback_parse import Frame, ParsedTraceback
from reproducer.graph.nodes.critic import (
    REACT_SYSTEM,
    SYSTEM,
    critic_mode_from_env,
    critic_trace_evidence,
    run_critic,
)
from reproducer.graph.nodes.critic_react import ReactLimits, react_tools
from reproducer.graph.nodes.critic_tools import (
    GREP_CHECK_EVERY,
    GREP_DEADLINE_S,
    GREP_MATCH_LINE_CHARS,
    MAX_GREP_MATCHES,
    MAX_RESULT_CHARS,
    GrepArgs,
    RepoView,
    view_for,
)
from reproducer.graph.nodes.smoke import smoke_loop
from reproducer.providers.llm import Budget, FakeLLM, MeteredLLM, llm_for_node
from reproducer.providers.store import RunStore
from reproducer.rules.classify import ClassifyContext

READ = ("read_file", {"path": "train.py", "start_line": 1, "end_line": 20})
GREP = ("grep", {"pattern": r"\.data\[0\]"})
SUBMIT_OK = ("submit_patch", json.loads(VALID))
SUBMIT_BAD = (
    "submit_patch",
    json.loads(_patch_json(BAD_LINE, "print(loss.data[1])", "print(loss.item())")),
)


def _critic(llm: Any, repo: Path, **kw: Any):  # type: ignore[no-untyped-def]
    return run_critic(
        llm,
        traceback=kw.pop("traceback", _tb(BAD_LINE)),
        command="python train.py",
        clone_dir=repo,
        repo_files=kw.pop("repo_files", frozenset({"train.py"})),
        metric_line_texts=METRIC_TEXTS,
        **kw,
    )


def _tool_results(llm: FakeLLM) -> list[str]:
    """Tool observations the model saw, in order (from the last chat's messages)."""
    msgs = llm.chats[-1][0]
    return [m["content"] for m in msgs if m["role"] == "tool"]


# --- loop ------------------------------------------------------------------------------------


def test_read_grep_submit_in_three_steps_applied_as_today(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM([], tool_script=[READ, GREP, SUBMIT_OK])
    runner = PatchAwareRunner()

    def critic(tb, command, stack, last, failed):  # type: ignore[no-untyped-def]
        return run_critic(
            llm, traceback=tb, command=command, clone_dir=repo,
            repo_files=frozenset({"train.py"}), stack=stack, metric_line_texts=METRIC_TEXTS,
            last_signature=last, failed_patches=failed,
        )  # fmt: skip

    out = smoke_loop(lambda s: _inputs(tmp_path, repo),
                     ClassifyContext(repo_files=frozenset({"train.py"})), runner=runner,
                     critic=critic)  # fmt: skip
    assert out.stop == "passed" and len(out.iterations) == 2
    c = out.iterations[0].critic
    assert c is not None and c.status == "patched" and c.patch is not None
    assert "-print(loss.data[0])\n+print(loss.item())\n" in c.patch.diff
    assert out.iterations[1].result.patches_applied == (c.patch.patch_id,)
    assert c.gate_attempts == 1 and c.errors == ()
    t = c.trace
    assert t is not None and t.outcome == "patched" and t.fallback_reason is None
    assert [s.tool for s in t.steps] == ["read_file", "grep", "submit_patch"]
    assert t.llm_calls == 3 and llm.prompts == []  # no single-shot call
    obs = _tool_results(llm)
    assert f"{BAD_LINE}| print(loss.data[0])" in obs[0] and "[METRIC]" in obs[0]
    assert obs[1].startswith(f"train.py:{BAD_LINE}:print(loss.data[0])")
    first = llm.chats[0][0]  # same grounded inputs as single-shot, plus the ReAct system
    assert first[0]["content"] == REACT_SYSTEM and _tb(BAD_LINE).raw in first[1]["content"]
    assert re.search(rf">>\s+{BAD_LINE}\| print\(loss.data\[0\]\)", first[1]["content"])
    assert {t["function"]["name"] for t in llm.chats[0][1]} == {
        "read_file", "grep", "list_dir", "submit_patch",
    }  # fmt: skip
    assert (repo / "train.py").read_text() == FIXTURE_TRAIN


def test_invalid_submit_gets_the_error_and_is_corrected(tmp_path: Path) -> None:
    llm = FakeLLM([], tool_script=[SUBMIT_BAD, SUBMIT_OK])
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "patched" and out.gate_attempts == 2 and len(out.errors) == 1
    assert "does not match" in out.errors[0]
    obs = _tool_results(llm)
    assert obs[0].startswith("submit_patch rejected") and "does not match" in obs[0]
    assert [s.note for s in out.trace.steps] == ["invalid patch", "accepted"]  # type: ignore[union-attr]


def test_path_escapes_are_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (tmp_path / "secret.txt").write_text("TOP SECRET\n")
    (repo / "link.py").symlink_to(tmp_path / "secret.txt")
    view = RepoView(clone_dir=repo, repo_files=frozenset({"train.py", "link.py"}))
    for name, args in [
        ("read_file", {"path": "../secret.txt", "start_line": 1, "end_line": 1}),
        ("read_file", {"path": "/etc/passwd", "start_line": 1, "end_line": 1}),
        ("read_file", {"path": "sub/../../secret.txt", "start_line": 1, "end_line": 1}),
        ("read_file", {"path": "link.py", "start_line": 1, "end_line": 1}),  # symlink escape
        ("read_file", {"path": ".git/config", "start_line": 1, "end_line": 1}),
        ("list_dir", {"path": ".."}),
        ("list_dir", {"path": "/"}),
        ("read_file", {"path": "train.py", "start_line": 1, "end_line": 1, "mode": "w"}),
    ]:
        r = view.call(name, json.dumps(args))
        assert not r.ok and r.text.startswith("error:"), (name, args, r.text)
        assert "TOP SECRET" not in r.text
    grep = view.call("grep", json.dumps({"pattern": "SECRET"}))
    assert grep.ok and "TOP SECRET" not in grep.text  # the escaping link is not searched
    # the loop: an escape is an error observation, never a crash or a read
    llm = FakeLLM([], tool_script=[("read_file", {"path": "../secret.txt", "start_line": 1,
                                                  "end_line": 3}), SUBMIT_OK])  # fmt: skip
    out = _critic(llm, repo)
    assert out.status == "patched" and not out.trace.steps[0].ok  # type: ignore[union-attr]
    assert "escapes the repo root" in _tool_results(llm)[0]


def test_results_are_truncated_and_line_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.py").write_text("".join(f"x_{i} = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n"
                                         for i in range(500)))  # fmt: skip
    view = RepoView(clone_dir=repo, repo_files=frozenset({"big.py"}))
    r = view.call("read_file", json.dumps({"path": "big.py", "start_line": 1, "end_line": 400}))
    assert r.ok and r.truncated and len(r.text) <= MAX_RESULT_CHARS and r.chars > MAX_RESULT_CHARS
    assert "[truncated:" in r.text
    small = view.call("read_file", json.dumps({"path": "big.py", "start_line": 301,
                                               "end_line": 800}))  # fmt: skip
    assert "lines 301-500 of 500" in small.text  # clamped to the file end (200 lines max)
    clamp = view.call("read_file", json.dumps({"path": "big.py", "start_line": 1,
                                               "end_line": 450}))  # fmt: skip
    assert "lines 1-200 of 500" in clamp.text
    g = view.call("grep", json.dumps({"pattern": r"^x_"}))
    full = view.grep(GrepArgs(pattern=r"^x_"))  # uncapped: the 50-match limit
    assert full.count("\n") == MAX_GREP_MATCHES and "stopped at 50 matches" in full
    assert len(g.text) <= MAX_RESULT_CHARS
    # in the loop, the trace records size + truncated
    llm = FakeLLM([], tool_script=[("read_file", {"path": "big.py", "start_line": 1,
                                                  "end_line": 200})])  # fmt: skip
    from reproducer.graph.nodes.critic_react import run_react

    res = run_react(llm, system="s", prompt="p", view=view, context={}, max_tokens=10,
                    limits=ReactLimits(max_steps=1))  # fmt: skip
    (step,) = res.trace.steps
    assert step.truncated and step.result_chars > MAX_RESULT_CHARS
    assert all(len(c) <= MAX_RESULT_CHARS for c in _tool_results(llm))


def test_repeated_call_rejected_twice_stops_and_falls_back(tmp_path: Path) -> None:
    llm = FakeLLM([VALID], tool_script=[READ, READ, READ])
    out = _critic(llm, _repo(tmp_path))
    t = out.trace
    assert t is not None and t.outcome == "fallback" and "loop detected" in str(t.fallback_reason)
    assert [s.note for s in t.steps] == ["", "rejected: repeated call", "rejected: repeated call"]
    assert "identical call already made at step 1" in _tool_results(llm)[1]
    assert out.status == "patched" and t.fallback_status == "patched"  # single-shot once
    assert len(llm.prompts) == 1 and llm.prompts[0][0] == SYSTEM


def test_five_step_cap_falls_back_to_single_shot(tmp_path: Path) -> None:
    script = [("read_file", {"path": "train.py", "start_line": i, "end_line": i + 1})
              for i in range(1, 6)]  # fmt: skip
    llm = FakeLLM([VALID], tool_script=script)
    out = _critic(llm, _repo(tmp_path))
    t = out.trace
    assert t is not None and len(t.steps) == 5 and t.llm_calls == 5
    assert t.fallback_reason == "no valid patch after 5 steps"
    assert out.status == "patched" and len(llm.prompts) == 1
    assert "steps left: 1; call submit_patch now" in _tool_results(llm)[3]


def test_no_tool_support_falls_back_with_reason(tmp_path: Path) -> None:
    llm = FakeLLM([VALID])  # no tool_script: chat() is unsupported
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "patched" and out.trace is not None
    assert out.trace.fallback_reason is not None
    assert out.trace.fallback_reason.startswith("no tool support")
    assert out.trace.steps == () and out.trace.llm_calls == 0


def test_text_replies_without_tool_calls_are_rejected(tmp_path: Path) -> None:
    llm = FakeLLM([VALID], tool_script=["I think it is .data[0].", "Use .item()."])
    out = _critic(llm, _repo(tmp_path))
    t = out.trace
    assert t is not None and "rejected turns" in str(t.fallback_reason) and len(t.steps) == 2
    assert t.thoughts == ("I think it is .data[0].", "Use .item().")
    assert out.status == "patched"


def test_fallback_invalid_counts_as_one_attempt(tmp_path: Path) -> None:
    llm = FakeLLM(["{}"] * 3, tool_script=[SUBMIT_BAD] + [READ, GREP, ("list_dir", {}),
                                                         ("list_dir", {"path": "."})])  # fmt: skip
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "invalid" and out.counts_as_attempt
    assert out.gate_attempts == 1 + 3 and len(out.errors) == 4  # ReAct submit + 3 answers
    assert out.trace is not None and out.trace.fallback_status == "invalid"


def test_wall_clock_and_token_caps(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    ticks = iter([0.0, 0.0, 999.0, 999.0])
    slow = ReactLimits(wall_s=10, clock=lambda: next(ticks))
    out = _critic(FakeLLM([VALID], tool_script=[READ, GREP]), repo, react_limits=slow)
    assert "wall-clock cap" in str(out.trace.fallback_reason)  # type: ignore[union-attr]
    assert len(out.trace.steps) == 1  # type: ignore[union-attr]
    llm = FakeLLM([VALID], tool_script=[READ, GREP])
    out = _critic(llm, repo, react_limits=ReactLimits(token_cap=1))
    assert "token cap" in str(out.trace.fallback_reason)  # type: ignore[union-attr]


def test_view_shows_the_patch_stack_and_never_writes(tmp_path: Path) -> None:
    train = "import torch\nx = torch.tensor(1.0)\na = x.data[0]\nb = x.data[0]\n"
    repo = _repo(tmp_path, train)
    first = _critic(FakeLLM([_patch_json(3, "a = x.data[0]", "a = x.item()")]), repo,
                    traceback=_tb(3, src="a = x.data[0]"))  # fmt: skip
    assert first.patch is not None
    view = view_for(repo, {"train.py"}, (first.patch,))
    r = view.call("read_file", json.dumps({"path": "train.py", "start_line": 1, "end_line": 4}))
    assert "3| a = x.item()" in r.text and "4| b = x.data[0]" in r.text
    assert view.call("grep", json.dumps({"pattern": "item"})).text == "train.py:3:a = x.item()"
    assert view.call("list_dir", "{}").text == "train.py"
    assert (repo / "train.py").read_text() == train


def test_single_mode_never_chats(tmp_path: Path) -> None:
    llm = FakeLLM([VALID], tool_script=[READ])
    out = _critic(llm, _repo(tmp_path), mode="single")
    assert out.status == "patched" and out.trace is None and llm.chats == []
    assert critic_mode_from_env({}) == "react"
    assert critic_mode_from_env({"REPRODUCER_CRITIC_MODE": "single"}) == "single"
    with pytest.raises(ValueError):
        critic_mode_from_env({"REPRODUCER_CRITIC_MODE": "agentic"})


def test_tool_schemas_are_self_contained() -> None:
    for t in react_tools():
        assert "$ref" not in json.dumps(t) and "$defs" not in json.dumps(t)
        assert t["function"]["parameters"]["type"] == "object"


# --- trace evidence --------------------------------------------------------------------------


def test_trace_evidence_content(tmp_path: Path) -> None:
    llm = FakeLLM([VALID], tool_script=[READ, READ, READ])
    out = _critic(llm, _repo(tmp_path))
    ev = critic_trace_evidence(out, source="r@c", smoke_evidence_id="a" * 64)
    assert (ev.kind, ev.authority, ev.confidence, ev.links) == (
        "critic_trace", "llm", "derived", ("a" * 64,),
    )  # fmt: skip
    c = ev.content
    assert c["mode"] == "react" and c["status"] == "patched" and c["file"] == "train.py"
    assert c["outcome"] == "fallback" and "loop detected" in c["fallback_reason"]
    assert c["fallback_status"] == "patched" and c["llm_calls"] == 3
    assert c["steps"][0] == {
        "step": 1, "tool": "read_file",
        "args": '{"end_line":20,"path":"train.py","start_line":1}',
        "result_chars": c["steps"][0]["result_chars"], "truncated": False, "ok": True, "note": "",
    }  # fmt: skip
    assert c["steps"][0]["result_chars"] > 0
    single = _critic(FakeLLM([VALID]), _repo(tmp_path / "s"), mode="single")
    with pytest.raises(ValueError):
        critic_trace_evidence(single, source="r@c", smoke_evidence_id="a" * 64)


def _react_item_patch(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
    """read the failing line, then submit .data[0] -> .item() for it."""
    prompt = messages[1]["content"]
    line = int(re.search(r"innermost repo frame is line (\d+)", prompt).group(1))  # type: ignore[union-attr]
    if sum(m["role"] == "tool" for m in messages) == 0:
        return ("read_file", {"path": "train.py", "start_line": line, "end_line": line})
    src = re.search(rf"^\s*>>\s+{line}\| (.*)$", prompt, re.MULTILINE).group(1)  # type: ignore[union-attr]
    return ("submit_patch", json.loads(_patch_json(line, src, src.replace(".data[0]", ".item()"))))


class ReactFakes(Fakes):
    def llm_for(self, budget: Budget):  # type: ignore[no-untyped-def]
        base = super().llm_for(budget)

        def factory(node: str) -> MeteredLLM:
            m = base(node)
            if node == "critic":
                m.inner = FakeLLM([], tool_script=_react_item_patch)
            return m

        return factory


def test_setup_graph_stores_critic_trace(fixture: tuple[Path, Path]) -> None:  # noqa: F811
    out, spec = fixture
    fakes = ReactFakes(_index_scenario, critic=lambda p, n: pytest.fail("single-shot called"))
    final = run_setup(out, spec, fakes)
    assert final["setup_status"] == "ready" and final["critic_attempts"] == 1
    kinds = _kinds(out)
    assert kinds.count("critic_patch") == 1 and kinds.count("critic_trace") == 1
    evs = RunStore(out).load_evidence()
    (tr,) = [e for e in evs.values() if e.kind == "critic_trace"]
    smoke = evs[tr.links[0]]
    assert smoke.kind == "smoke_result" and not smoke.content.get("passed")
    assert [s["tool"] for s in tr.content["steps"]] == ["read_file", "submit_patch"]
    records = RunStore(out).load_call_records()
    assert [r.node for r in records].count("critic") == 2  # one record per ReAct turn


# --- live -------------------------------------------------------------------------------------


@pytest.mark.llm_live
@pytest.mark.docker
def test_live_react_critic_fixes_index_error(cutout_image: str, tmp_path: Path) -> None:  # noqa: F811
    spec = os.environ.get("REPRODUCER_LLM_CRITIC") or os.environ.get("REPRODUCER_LLM")
    if not spec:
        pytest.skip("set REPRODUCER_LLM_CRITIC (e.g. openrouter:anthropic/claude-haiku-5.5)")
    if os.environ.get("REPRODUCER_CRITIC_MODE", "react") != "react":
        pytest.skip("REPRODUCER_CRITIC_MODE is not react")
    llm = llm_for_node("critic", Budget(max_tokens_total=60_000))
    start = time.monotonic()
    out, repo = _docker_loop(tmp_path, cutout_image, llm)
    first = out.iterations[0]
    c = first.critic
    print("\n=== live ReAct critic trace ===")
    print(f"backend: {spec}")
    print("--- traceback ---")
    print(first.result.traceback.raw if first.result.traceback else None)
    assert c is not None
    t = c.trace
    if t is not None:
        print(json.dumps(t.as_json(), indent=1))
    print(f"--- critic: {c.status}, submits+answers {c.gate_attempts}, errors {list(c.errors)}")
    if c.patch is not None:
        print(f"rationale: {c.patch.rationale}")
        print(c.patch.diff)
    for r in llm.records():
        print(f"call: {r.backend}:{r.model} in={r.input_tokens} out={r.output_tokens} "
              f"latency={r.latency_s:.2f}s")  # fmt: skip
    last = out.iterations[-1].result
    print(f"--- re-smoke: signal={last.signal} passed={last.passed} "
          f"patches_applied={[p[:12] for p in last.patches_applied]}")  # fmt: skip
    print(f"total wall: {time.monotonic() - start:.1f}s; stop={out.stop}: {out.reason}")
    assert c.status == "patched" and c.patch is not None and t is not None
    assert out.stop == "passed" and last.passed
    assert (repo / "train.py").read_text() == FIXTURE_TRAIN


# --- grep guards -----------------------------------------------------------------------------


def _grep(view: RepoView, pattern: str) -> str:
    return view.call("grep", json.dumps({"pattern": pattern})).text


def _clock_after(n_ok: int):  # type: ignore[no-untyped-def]
    """A clock that is past the grep deadline from its (n_ok + 1)-th reading on."""
    n = [0]

    def clock() -> float:
        n[0] += 1
        return 0.0 if n[0] <= n_ok else GREP_DEADLINE_S + 1

    return clock


def test_grep_deadline_between_files_and_every_500_lines(tmp_path: Path) -> None:
    assert GREP_CHECK_EVERY == 500
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("a.py", "b.py"):
        (repo / name).write_text("x = 1\n" * 1200)
    files = frozenset({"a.py", "b.py"})
    # readings: deadline set, before a.py, a.py lines 500 + 1000, before b.py -> past
    view = RepoView(clone_dir=repo, repo_files=files, clock=_clock_after(4))
    out = _grep(view, r"^y")
    assert out.startswith("(incomplete: deadline 5s reached after 1 file(s))"), out
    # readings: deadline set, before a.py, a.py line 500 -> past at line 1000
    view = RepoView(clone_dir=repo, repo_files=files, clock=_clock_after(3))
    out = _grep(view, r"^y")
    assert out.startswith("(incomplete: deadline 5s reached in a.py at line 1000)"), out
    assert out.endswith("no matches for '^y'")


def test_grep_long_lines_matched_on_first_2000_chars(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("s = '" + "a" * 2500 + "NEEDLE'\nt = 'NEEDLE'\n")
    view = RepoView(clone_dir=repo, repo_files=frozenset({"a.py"}))
    out = _grep(view, "NEEDLE")
    lines = out.split("\n")
    assert lines[0] == (f"(1 line(s) longer than {GREP_MATCH_LINE_CHARS} chars were matched "
                        "on their first part only)")  # fmt: skip
    assert lines[1:] == ["a.py:2:t = 'NEEDLE'"]  # the needle past char 2000 is not matched


def test_grep_scan_limit_5mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import reproducer.graph.nodes.critic_tools as ct

    assert ct.GREP_MAX_BYTES == 5_000_000
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(7):  # 7 x ~0.9 MB > 5 MB
        (repo / f"f{i}.py").write_text("z = 'q' * 3\n" * 75_000)
    view = RepoView(clone_dir=repo, repo_files=frozenset(f"f{i}.py" for i in range(7)))
    out = _grep(view, "NOPE")
    assert out.startswith("(incomplete: scan limit 5000000 bytes reached in f5.py at line")
    assert out.endswith("no matches for 'NOPE'")
    monkeypatch.setattr(ct, "GREP_MAX_BYTES", 100)
    out = _grep(view, "z =")
    assert out.split("\n")[0].startswith("(incomplete: scan limit 100 bytes reached in f0.py")
    assert out.count("f0.py:") == 100 // len("z = 'q' * 3\n")


def test_grep_notes_survive_the_result_cap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("".join(f"value_{i} = '{'b' * 150}'\n" for i in range(80)))
    view = RepoView(clone_dir=repo, repo_files=frozenset({"a.py"}))
    r = view.call("grep", json.dumps({"pattern": "value_"}))
    assert r.truncated and r.text.startswith("(incomplete: stopped at 50 matches)")


# --- investigation fixture: the fix in train.py needs the helper's code --------------------

INV_HELPER = (
    '"""Batch accuracy statistics (v2 API)."""\n'
    "import torch\n"
    "\n"
    "\n"
    "def batch_stats(output, target):\n"
    '    """Counts for one batch as 0-dim tensors / ints; keys renamed in the v2 refactor."""\n'
    "    pred = output.argmax(dim=1)\n"
    '    return {"num_correct": (pred == target).sum(), "num_seen": target.size(0)}\n'
)
INV_TRAIN = (
    "import torch\n"
    "from utils.metrics import batch_stats\n" + PROGRESS + "torch.manual_seed(0)\n"
    "output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])\n"
    "target = torch.tensor([1, 0, 0, 0])\n"
    "stats = batch_stats(output, target)\n"
    "correct, total = stats['correct'].item(), stats['total']\n"
    "print('test_acc: %.3f' % (100.0 * correct / total))\n"
)
INV_LINE = (
    INV_TRAIN.split("\n").index("correct, total = stats['correct'].item(), stats['total']") + 1
)
INV_FIXED = "correct, total = stats['num_correct'].item(), stats['num_seen']"
INV_FILES = frozenset({"train.py", "utils/__init__.py", "utils/metrics.py"})
INV_METRIC = {"train.py": ["print('test_acc: %.3f' % (100.0 * correct / total))"]}


def _inv_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "utils").mkdir(parents=True)
    (repo / "train.py").write_text(INV_TRAIN)
    (repo / "utils" / "__init__.py").write_text("")
    (repo / "utils" / "metrics.py").write_text(INV_HELPER)
    return repo


def _inv_tb(line: int = INV_LINE) -> ParsedTraceback:
    raw = (
        "Traceback (most recent call last):\n"
        f'  File "train.py", line {line}, in <module>\n'
        "    correct, total = stats['correct'].item(), stats['total']\n"
        "KeyError: 'correct'"
    )
    frame = Frame(file="train.py", line=line, func="<module>")
    return ParsedTraceback(exc_type="KeyError", message="'correct'", frames=(frame,),
                           repo_frame=frame, raw=raw)  # fmt: skip


def test_investigation_fixture_needs_the_helper(tmp_path: Path) -> None:
    repo = _inv_repo(tmp_path)
    fix = {"file": "train.py", "rationale": "use the v2 keys of batch_stats",
           "edits": [{"start_line": INV_LINE, "end_line": INV_LINE,
                      "original": INV_TRAIN.split("\n")[INV_LINE - 1],
                      "replacement": INV_FIXED}]}  # fmt: skip
    llm = FakeLLM([], tool_script=[
        ("grep", {"pattern": r"def batch_stats"}),
        ("read_file", {"path": "utils/metrics.py", "start_line": 1, "end_line": 8}),
        ("submit_patch", fix),
    ])  # fmt: skip
    out = run_critic(llm, traceback=_inv_tb(), command="python train.py", clone_dir=repo,
                     repo_files=INV_FILES, metric_line_texts=INV_METRIC)  # fmt: skip
    first_prompt = llm.chats[0][0][1]["content"]
    assert "num_correct" not in first_prompt and "num_seen" not in first_prompt  # not in window
    assert out.status == "patched" and out.patch is not None
    assert f"+{INV_FIXED}" in out.patch.diff
    assert [s.tool for s in out.trace.steps] == ["grep", "read_file", "submit_patch"]  # type: ignore[union-attr]
    obs = _tool_results(llm)
    assert obs[0].startswith("utils/metrics.py:5:def batch_stats(output, target):\n(steps left")
    assert '"num_correct"' in obs[1]
    helper_fix = dict(fix, file="utils/metrics.py")  # the failing-file rule still holds
    llm2 = FakeLLM(["{}"] * 3, tool_script=[("submit_patch", helper_fix)] + [READ, GREP] * 2)
    bad = run_critic(llm2, traceback=_inv_tb(), command="python train.py", clone_dir=repo,
                     repo_files=INV_FILES, metric_line_texts=INV_METRIC)  # fmt: skip
    assert "patch the failing file 'train.py' only" in bad.errors[0]


@pytest.mark.llm_live
@pytest.mark.docker
def test_live_react_critic_investigates_helper(cutout_image: str, tmp_path: Path) -> None:  # noqa: F811
    spec = os.environ.get("REPRODUCER_LLM_CRITIC") or os.environ.get("REPRODUCER_LLM")
    if not spec:
        pytest.skip("set REPRODUCER_LLM_CRITIC (e.g. openrouter:anthropic/claude-haiku-5.5)")
    llm = llm_for_node("critic", Budget(max_tokens_total=80_000))
    repo = _inv_repo(tmp_path)

    def critic(tb, command, stack, last, failed):  # type: ignore[no-untyped-def]
        return run_critic(
            llm, traceback=tb, command=command, clone_dir=repo, repo_files=INV_FILES,
            stack=stack, metric_line_texts=INV_METRIC, last_signature=last,
            failed_patches=failed, mode="react",
        )  # fmt: skip

    start = time.monotonic()
    out = smoke_loop(
        lambda s: _inputs(tmp_path, repo, image=cutout_image, third_party=("torch",)),
        ClassifyContext(repo_files=INV_FILES),
        critic=critic,
    )
    print("\n=== live ReAct critic: investigation fixture ===")
    print(f"backend: {spec}")
    for k, it in enumerate(out.iterations, 1):
        r = it.result
        print(f"--- smoke {k}: signal={r.signal} passed={r.passed} "
              f"patches={[p[:12] for p in r.patches_applied]}")  # fmt: skip
        if r.traceback is not None:
            print(r.traceback.raw)
        c = it.critic
        if c is None:
            continue
        print(f"critic: {c.status}, submits+answers {c.gate_attempts}, errors {list(c.errors)}")
        if c.trace is not None:
            print(json.dumps(c.trace.as_json(), indent=1))
        if c.patch is not None:
            print(f"rationale: {c.patch.rationale}\n{c.patch.diff}")
    for r in llm.records():
        print(f"call: {r.backend}:{r.model} in={r.input_tokens} out={r.output_tokens} "
              f"latency={r.latency_s:.2f}s")  # fmt: skip
    last = out.iterations[-1].result
    print(f"--- final: stop={out.stop} ({out.reason}); wall {time.monotonic() - start:.1f}s")
    print(last.log_path.read_text()[-300:])
    first = out.iterations[0].critic
    assert first is not None and first.trace is not None
    tools = [s.tool for s in first.trace.steps]
    assert "submit_patch" in tools
    assert any(t in ("read_file", "grep") for t in tools[: tools.index("submit_patch")]), tools
    assert out.stop == "passed" and "test_acc: 75.000" in last.log_path.read_text()
    assert (repo / "train.py").read_text() == INV_TRAIN
