import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ra_agent.agent.context import estimate_tokens, fit_context
from ra_agent.agent.memory import Episode, EpisodicMemory, ProceduralMemory, memory_prompt


def ep(paper, outcome="finish", failure=""):
    return Episode("task", paper, "2026-01-01", [], outcome, failure, [])


def test_recall_prefers_same_paper_then_failures(tmp_path):
    mem = EpisodicMemory(tmp_path / "episodes.jsonl")
    for e in [ep("a"), ep("b", "fallback", "ocr failed"), ep("c"), ep("a", "fallback", "x")]:
        mem.append(e)
    got = mem.recall("a", limit=3)
    assert [e.paper for e in got[:2]] == ["a", "a"]
    assert got[2].paper == "b"


def test_reload_and_corrupt_line(tmp_path):
    path = tmp_path / "episodes.jsonl"
    EpisodicMemory(path).append(ep("a", "fallback", "boom"))
    with path.open("a") as f:
        f.write("{not json\n")
    got = EpisodicMemory(path).recall("a")
    assert len(got) == 1 and got[0].failure == "boom"
    assert EpisodicMemory(tmp_path / "none.jsonl").recall("a") == []


def test_procedures_render_and_prompt(tmp_path):
    (tmp_path / "b.md").write_text("B" * 100)
    (tmp_path / "a.md").write_text("A" * 100)
    pm = ProceduralMemory(tmp_path)
    assert [n for n, _ in pm.all()] == ["a", "b"]
    assert "B" not in pm.render(max_chars=130) and "A" in pm.render(max_chars=130)
    assert ProceduralMemory(tmp_path / "missing").render() == ""
    out = memory_prompt([ep("a", "fallback", "ocr failed")], pm.all())
    assert "Past episodes:" in out and "fallback: ocr failed" in out and "Procedures:" in out


def _msgs():
    big = json.dumps({"handle": "h1", "text": "x" * 5000})
    return [
        SystemMessage(content="sys"), HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content=big, tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(content="y" * 5000, tool_call_id="2"),
        AIMessage(content="done"),
    ]


def test_fit_context_evicts_oldest_tool_message():
    msgs = _msgs()
    out, evicted = fit_context(msgs, budget=200, keep_last=3)
    assert len(out) == len(msgs) and evicted == [msgs[3].content]
    assert out[3].content.startswith("[evicted] ") and "(full result: h1)" in out[3].content
    assert out[3].tool_call_id == "1"
    assert out[5].content == msgs[5].content  # inside the last 3
    assert out[:3] == msgs[:3] and out[4:] == msgs[4:]
    assert estimate_tokens(out) < estimate_tokens(msgs)  # protected tail may stay over budget


def test_fit_context_noop_and_no_handle():
    msgs = _msgs()
    assert fit_context(msgs, budget=10**6, keep_last=4) == (msgs, [])
    out, _ = fit_context(msgs, budget=200, keep_last=1)
    assert "(full result: in episode)" in out[5].content


def test_fit_context_idempotent():
    out, ev = fit_context(_msgs(), budget=200, keep_last=1)
    again, ev2 = fit_context(out, budget=200, keep_last=1)
    assert ev and ev2 == []
    assert [m.content for m in again] == [m.content for m in out]
    assert "(full result: h1)" in again[3].content
