import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ra_agent.agent.graph import build_graph
from ra_agent.agent.guard import FailureReport, validate_call
from ra_agent.agent.scripted import ScriptedChatModel
from ra_agent.tools import TOOL_KINDS


def call(tool, i=0, **args):
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"c{i}"}])


def local_tools(ws):
    return [StructuredTool.from_function(getattr(ws, n), name=n, description=n) for n in TOOL_KINDS]


def run(ws, responses, task="ingest paper.pdf", paper="paper.pdf", tools=None, resume=None, thread="t"):
    model = ScriptedChatModel(responses=responses)
    graph = build_graph(model, tools or local_tools(ws), ws.settings, ws.fallback_ingest, InMemorySaver())
    config = {"configurable": {"thread_id": thread}}

    async def go():
        out = await graph.ainvoke({"task": task, "paper": paper}, config)
        for answer in resume or []:
            out = await graph.ainvoke(Command(resume=answer), config)
        return out

    return asyncio.run(go()), model, graph, config


def episodes(ws):
    path = Path(ws.settings.paths.memory_dir) / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def vault_files(ws):
    root = Path(ws.settings.paths.vault_dir) / "Papers"  # the Inbox is not a note
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_finish(ws):
    out, model, _, _ = run(ws, [call("list_papers"), AIMessage(content="done")])
    assert out["outcome"] == "finish" and out["steps"] == 2 and model.calls == 2
    assert out["failure"] == ""
    assert episodes(ws)[-1]["outcome"] == "finish"


def test_step_budget(ws):
    assert ws.settings.agent.max_steps == 12
    script = [call("list_papers", i, dummy=str(i)) for i in range(20)]
    out, model, _, _ = run(ws, script)
    assert model.calls == 12 and out["steps"] == 12
    assert out["outcome"] == "fallback" and "step_budget" in out["failure"]
    assert episodes(ws)[-1]["failure"] == out["failure"]


def test_repeated_action(ws):
    out, _, _, _ = run(ws, [call("list_papers"), call("list_papers")])
    assert out["outcome"] == "fallback" and "repeated_action" in out["failure"]
    assert "list_papers was proposed twice" in out["failure"]


def test_invalid_calls(ws):
    out, model, _, _ = run(ws, [call("nope", 0), call("nope", 1, x="2")])
    assert out["outcome"] == "fallback" and "invalid_calls" in out["failure"]
    assert out["invalid"] == 2 and model.calls == 2


def test_path_guard(ws, tmp_path, monkeypatch):
    # "../../.env" from cwd a/b is the real sentinel tmp_path/.env
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    secret = tmp_path / ".env"
    secret.write_text("KEY=1")
    assert Path("../../.env").resolve() == secret.resolve()
    before = secret.stat().st_mtime_ns
    out, _, _, _ = run(ws, [call("ingest_paper", pdf="../../.env"), AIMessage(content="ok")])
    assert out["invalid"] == 1 and out["tried"] == []
    assert any("outside" in str(m.content) for m in out["messages"])
    assert secret.read_text() == "KEY=1" and secret.stat().st_mtime_ns == before


def test_validate_call(ws):
    tools = {t.name: t for t in local_tools(ws)}
    paths = ws.settings.paths
    assert validate_call("ingest_paper", {"pdf": "paper.pdf"}, tools, paths) is None
    assert "unknown" in validate_call("nope", {}, tools, paths)
    assert validate_call("reprocess_page", {"paper": "x", "page": "a", "engine": "granite"}, tools, paths)
    assert validate_call("ingest_paper", {"pdf": "/etc/passwd"}, tools, paths)
    assert validate_call("ingest_paper", {"pdf": "..\\x.pdf"}, tools, paths)
    inside = [paths.inbox_dir, paths.vault_dir, paths.staging_dir]
    assert all(validate_call("ingest_paper", {"pdf": f"{d}/a.pdf"}, tools, paths) is None for d in inside)
    assert validate_call("ingest_paper", {"pdf": f"{paths.vault_dir}/../elsewhere/a.pdf"}, tools, paths)
    # free text may contain slashes
    assert validate_call("save_procedure", {"name": "n", "text": "use granite/docling"}, tools, paths) is None


def test_failure_sentence():
    report = FailureReport("repeated_action", 7, "paperbench", "commit_note",
                           "commit_note was proposed twice with the same arguments",
                           ["ingest_paper", "reprocess_page(4, granite)", "stage_note"])
    assert report.sentence() == (
        "Stopped by repeated_action at step 7 on paperbench: commit_note was proposed twice "
        "with the same arguments. Tried: ingest_paper, reprocess_page(4, granite), stage_note."
    )


@pytest.fixture
def slug(ws):
    return ws.ingest_paper("paper.pdf")["slug"]


def test_approval_interrupt_reject_and_approve(ws, slug):
    script = [call("commit_note", paper=slug), AIMessage(content="done")]
    out, _, graph, config = run(ws, script, paper=slug)
    (pending,) = out["__interrupt__"]
    assert pending.value["tool"] == "commit_note" and pending.value["diff"].startswith("<<<DATA")
    assert vault_files(ws) == []

    out, _, _, _ = run(ws, script, paper=slug, resume=["reject"], thread="r")
    assert out["outcome"] == "finish" and vault_files(ws) == []
    assert any(m.content == "rejected by human" for m in out["messages"])

    out, _, _, _ = run(ws, script, paper=slug, resume=["approve"], thread="a")
    assert out["outcome"] == "finish" and out["tried"] == ["commit_note"]
    assert ws.vault.note(ws.vault.root, slug).is_file()


def test_edit_during_interrupt_survives_resume(ws, slug):
    script = [call("commit_note", paper=slug), AIMessage(content="done")]
    _, _, graph, config = run(ws, script, paper=slug)
    staged = ws.vault.note(ws.vault.staging, slug)
    staged.write_text(staged.read_text() + "\nhuman edit\n")
    asyncio.run(graph.ainvoke(Command(resume="approve"), config))
    assert "human edit" in ws.vault.note(ws.vault.root, slug).read_text()


def test_reject_then_same_write_is_repeated_action(ws, slug):
    out, model, _, _ = run(ws, [call("commit_note", paper=slug)], paper=slug, resume=["reject"])
    assert "__interrupt__" not in out
    assert out["outcome"] == "fallback" and "repeated_action" in out["failure"]
    assert vault_files(ws) == [] and model.calls == 2
    staged = ws.vault.note(ws.vault.staging, slug).read_text()
    assert "needs-review" in staged


def test_fallback_ingest_from_pdf(ws):
    result = ws.fallback_ingest("paper.pdf")
    assert result["ok"] and result["status"] == "needs-review"
    assert "needs-review" in Path(result["handle"]).read_text()
    assert vault_files(ws) == []


def test_eviction(ws):
    ws.settings.agent.context_budget = 300
    ws.settings.agent.keep_last_turns = 2
    big = StructuredTool.from_function(lambda n: {"ok": True, "summary": "x" * 400, "n": "y" * 2000},
                                       name="big", description="big")
    script = [call("big", 0, n=1), call("big", 1, n=2), call("big", 2, n=3), AIMessage(content="done")]
    # "big" is outside TOOL_KINDS, so each call waits for approval (fail closed)
    out, _, _, _ = run(ws, script, tools=local_tools(ws) + [big], resume=["approve"] * 3)
    assert out["evicted"] and episodes(ws)[-1]["evicted"]
    assert any(str(m.content).startswith("[evicted] ") for m in out["messages"])


def test_procedure_in_next_system_message(ws):
    run(ws, [call("save_procedure", name="retry", text="rerun with granite"), AIMessage(content="ok")],
        resume=["approve"])
    out, _, _, _ = run(ws, [AIMessage(content="ok")], thread="next")
    system = next(m for m in out["messages"] if isinstance(m, SystemMessage))
    assert "rerun with granite" in system.content and "DATA>>>" in system.content


def test_llm_error_goes_to_fallback(ws):
    class Broken(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("endpoint down")

    graph = build_graph(Broken(responses=[]), local_tools(ws), ws.settings, ws.fallback_ingest)
    out = asyncio.run(graph.ainvoke({"task": "t", "paper": "paper.pdf"}))
    assert out["outcome"] == "fallback" and "llm_error" in out["failure"]


def test_tool_outside_tool_kinds_needs_approval(ws):
    ran = []
    extra = StructuredTool.from_function(lambda x: ran.append(x) or {"ok": True, "summary": "ran"},
                                         name="delete_all", description="unknown kind")
    out, _, _, _ = run(ws, [call("delete_all", x="1"), AIMessage(content="ok")],
                       tools=local_tools(ws) + [extra])
    (pending,) = out["__interrupt__"]
    assert pending.value["tool"] == "delete_all" and ran == []


def test_failing_stage_still_interrupts_with_reason(ws):
    out, _, _, _ = run(ws, [call("commit_note", paper="never-ingested"), AIMessage(content="ok")],
                       paper="never-ingested")
    (pending,) = out["__interrupt__"]
    assert pending.value["diff"].startswith("diff unavailable:") and "never-ingested" in pending.value["diff"]

    tools = [t for t in local_tools(ws) if t.name != "stage_note"]  # missing tool: no KeyError
    out, _, _, _ = run(ws, [call("commit_note", paper="x"), AIMessage(content="ok")], tools=tools, thread="m")
    (pending,) = out["__interrupt__"]
    assert pending.value["diff"].startswith("diff unavailable: KeyError")
    assert vault_files(ws) == []


def test_broken_json_counts_as_invalid_call(ws):
    broken = [
        AIMessage(content="", invalid_tool_calls=[
            {"name": "ingest_paper", "args": '{"pdf": "paper.pdf"', "id": f"b{i}", "error": "bad json"}
        ])
        for i in range(2)
    ]
    out, model, _, _ = run(ws, broken + [AIMessage(content="done")])
    assert out["outcome"] == "fallback" and "invalid_calls" in out["failure"]
    assert out["invalid"] == 2 and model.calls == 2


def test_limit_after_successful_commit_finishes_without_fallback(ws, slug):
    script = [call("stage_note", 0, paper=slug), call("commit_note", 1, paper=slug)]
    script += [call("vault_search", i, query=f"q{i}") for i in range(2, 30)]
    out, model, _, _ = run(ws, script, paper=slug, resume=["approve"])
    assert out["outcome"] == "finish" and out["failure"] == "" and out["committed"]
    assert model.calls == 12 and "step_budget" in out["messages"][-1].content
    assert ws.vault.note(ws.vault.root, slug).is_file()
    assert not (Path(ws.settings.paths.staging_dir) / f"{slug}.json").exists()  # no fallback restage
    assert episodes(ws)[-1]["outcome"] == "finish"
