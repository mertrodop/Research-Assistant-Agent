from __future__ import annotations

import json

import pytest

from ra_agent.agent.loop import Agent, parse_action
from ra_agent.agent.tools import Tool, build_tools, call_tool
from ra_agent.config import AgentSettings, Settings


class ScriptedLlm:
    """Replays a fixed list of replies, one per step."""

    model = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("model called more times than scripted")
        return self.replies.pop(0)

    def describe(self, image_path, prompt):  # pragma: no cover
        raise AssertionError("not used")


def action(name: str, args: dict | None = None, thought: str = "") -> str:
    return json.dumps({"thought": thought, "action": name, "args": args or {}})


def finish(answer: str = "done") -> str:
    return json.dumps({"thought": "finished", "action": "finish", "answer": answer})


def echo_tool() -> dict[str, Tool]:
    from pydantic import BaseModel

    class EchoArgs(BaseModel):
        text: str = ""

    return {
        "echo": Tool(
            "echo",
            "Echo the text back.",
            EchoArgs,
            lambda args: {"echoed": args.text},
        )
    }


def test_parse_action_plain_json():
    parsed = parse_action(action("echo", {"text": "hi"}, "because"))
    assert parsed["action"] == "echo"
    assert parsed["args"] == {"text": "hi"}
    assert parsed["thought"] == "because"


def test_parse_action_strips_code_fence():
    raw = "```json\n" + action("echo", {"text": "hi"}) + "\n```"
    assert parse_action(raw)["action"] == "echo"


def test_parse_action_repairs_truncated_json():
    raw = '{"thought": "x", "action": "echo", "args": {"text": "hi"'
    parsed = parse_action(raw)
    assert parsed["action"] == "echo"
    assert parsed["args"] == {"text": "hi"}


def test_parse_action_rejects_missing_action():
    with pytest.raises(ValueError):
        parse_action('{"thought": "x"}')


def test_parse_action_rejects_non_object_args():
    with pytest.raises(ValueError):
        parse_action('{"action": "echo", "args": [1, 2]}')


def test_loop_runs_tool_then_finishes():
    llm = ScriptedLlm([action("echo", {"text": "hi"}), finish("echoed hi")])
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=4)).run("echo hi")
    assert run.ok
    assert run.stopped == "finish"
    assert run.answer == "echoed hi"
    assert run.steps[0].observation == {"ok": True, "result": {"echoed": "hi"}}


def test_loop_stops_on_repeated_action():
    llm = ScriptedLlm([action("echo", {"text": "hi"}), action("echo", {"text": "hi"})])
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=6)).run("loop")
    assert run.stopped == "repeated_action"
    assert not run.ok
    assert len(run.steps) == 2


def test_loop_stops_on_step_budget():
    replies = [action("echo", {"text": str(index)}) for index in range(5)]
    llm = ScriptedLlm(replies)
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=3)).run("never finish")
    assert run.stopped == "step_budget"
    assert len(run.steps) == 3


def test_loop_stops_on_unknown_tools():
    llm = ScriptedLlm([action("nope"), action("nope2")])
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=6, max_consecutive_errors=2)).run("x")
    assert run.stopped == "unknown_tools"


def test_loop_stops_on_invalid_actions():
    llm = ScriptedLlm(["not json", "still not json"])
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=6, max_consecutive_errors=2)).run("x")
    assert run.stopped == "invalid_actions"


def test_loop_recovers_after_one_bad_reply():
    llm = ScriptedLlm(["garbage", action("echo", {"text": "hi"}), finish("ok")])
    run = Agent(llm, echo_tool(), AgentSettings(max_steps=6)).run("x")
    assert run.ok
    assert run.steps[0].error.startswith("invalid action JSON")


def test_loop_reports_llm_failure():
    class Broken:
        model = "broken"

        def complete(self, prompt):
            from ra_agent.llm.client import VisionError

            raise VisionError("HTTP 401")

        def describe(self, image_path, prompt):  # pragma: no cover
            raise AssertionError

    run = Agent(Broken(), echo_tool(), AgentSettings()).run("x")
    assert run.stopped == "llm_error"
    assert "401" in run.steps[0].error


def test_loop_prompt_carries_history():
    llm = ScriptedLlm([action("echo", {"text": "hi"}), finish("ok")])
    Agent(llm, echo_tool(), AgentSettings(max_steps=4)).run("echo hi")
    assert "echo hi" in llm.prompts[0]
    assert "(empty)" in llm.prompts[0]
    assert "echoed" in llm.prompts[1]


def test_run_saves_trace(tmp_path):
    llm = ScriptedLlm([finish("ok")])
    run = Agent(llm, echo_tool(), AgentSettings()).run("x")
    path = run.save(tmp_path / "trace.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stopped"] == "finish"
    assert payload["answer"] == "ok"


def test_trace_names_are_unique_per_run():
    llm = ScriptedLlm([finish("ok"), finish("ok")])
    first = Agent(llm, echo_tool(), AgentSettings()).run("same task")
    second = Agent(llm, echo_tool(), AgentSettings()).run("same task")
    assert first.trace_name() != second.trace_name()
    assert first.trace_name().endswith(".json")
    assert "same-task" in first.trace_name()


def test_call_tool_validates_arguments():
    tools = echo_tool()
    result = call_tool(tools["echo"], {"unknown": 1})
    assert result["ok"] is True  # extra keys are ignored by pydantic


def test_call_tool_reports_bad_types():
    from pydantic import BaseModel

    class StrictArgs(BaseModel):
        count: int

    tool = Tool("strict", "needs an int", StrictArgs, lambda args: {"count": args.count})
    result = call_tool(tool, {"count": "not a number"})
    assert result["ok"] is False
    assert "invalid arguments" in result["error"]


def test_call_tool_turns_exceptions_into_observations():
    from pydantic import BaseModel

    class Empty(BaseModel):
        pass

    def boom(_):
        raise RuntimeError("disk on fire")

    tool = Tool("boom", "always fails", Empty, boom)
    result = call_tool(tool, {})
    assert result["ok"] is False
    assert "disk on fire" in result["error"]


def test_build_tools_exposes_the_catalog():
    tools = build_tools(Settings())
    assert set(tools) == {
        "list_papers",
        "ocr",
        "gate",
        "references",
        "read_note",
        "write_report",
    }
    assert all(tool.args_model.model_json_schema() for tool in tools.values())


def test_list_papers_tool_reads_raw_dir(tmp_path):
    settings = Settings()
    settings.paths.raw_dir = str(tmp_path)
    tools = build_tools(settings)
    result = call_tool(tools["list_papers"], {})
    assert result["ok"] is True
    assert result["result"]["papers"] == []


def test_read_note_reports_missing(tmp_path):
    settings = Settings()
    settings.paths.staging_dir = str(tmp_path)
    tools = build_tools(settings)
    result = call_tool(tools["read_note"], {"sha": "deadbeef"})
    assert result["ok"] is False
    assert "no processed note" in result["error"]


def test_write_report_writes_file(tmp_path):
    settings = Settings()
    settings.paths.staging_dir = str(tmp_path)
    tools = build_tools(settings)
    result = call_tool(tools["write_report"], {"name": "summary", "body": "# hi"})
    assert result["ok"] is True
    assert (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8") == "# hi"
