"""OpenAI function calling in providers.llm (chat with tools), against the stub server.

Response shapes are the ones verified live (2026-11-27): OpenRouter anthropic/claude-haiku-5.5
returns message.tool_calls (content null, finish_reason "tool_calls", ids "toolu_..."); Ollama
0.18.0 qwen2.5-coder:7b returns the call as bare JSON in message.content, tool_calls absent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_llm import Stub, no_sleep, stub  # noqa: F401  (fixtures)

from reproducer.providers.llm import (
    Budget,
    ChatResponse,
    FakeLLM,
    LLMError,
    MeteredLLM,
    OllamaLLM,
    OpenAICompatLLM,
    ToolProtocolError,
    ToolsUnsupported,
    assistant_message,
    tool_call_from_content,
    tool_message,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
PATH = "/v1/chat/completions"


def _haiku_tool_call() -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": None,
                    "refusal": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "index": 0,
                            "id": "toolu_01C7",
                            "function": {"name": "read_file", "arguments": '{"path": "train.py"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 448, "completion_tokens": 95},
    }


def _llm(stub: Stub) -> OpenAICompatLLM:  # noqa: F811
    return OpenAICompatLLM(base_url=f"{stub.url}/v1", model="m", env={})


def test_chat_payload_and_tool_calls(stub: Stub) -> None:  # noqa: F811
    stub.add("POST", PATH, 200, _haiku_tool_call())
    resp = _llm(stub).chat(MSGS, TOOLS, max_tokens=100)
    (req,) = stub.posts(PATH)
    body = req["json"]
    assert body["tools"] == TOOLS and body["messages"] == MSGS and body["tool_choice"] == "auto"
    assert "response_format" not in body and body["max_tokens"] == 100
    assert resp.finish_reason == "tool_calls" and resp.text is None and not resp.from_content
    (call,) = resp.tool_calls
    assert (call.id, call.name, json.loads(call.arguments)) == (
        "toolu_01C7",
        "read_file",
        {"path": "train.py"},
    )
    assert (resp.input_tokens, resp.output_tokens) == (448, 95)


def test_assistant_and_tool_messages_roundtrip(stub: Stub) -> None:  # noqa: F811
    stub.add("POST", PATH, 200, _haiku_tool_call())
    stub.add("POST", PATH, 200, {"choices": [{"message": {"content": "done"}}]})
    llm = _llm(stub)
    r1 = llm.chat(MSGS, TOOLS, max_tokens=50)
    msgs = [*MSGS, assistant_message(r1), tool_message(r1.tool_calls[0].id, "1| x")]
    r2 = llm.chat(msgs, TOOLS, max_tokens=50)
    sent = stub.posts(PATH)[1]["json"]["messages"]
    assert sent[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "toolu_01C7",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "train.py"}'},
            }
        ],
    }
    assert sent[3] == {"role": "tool", "tool_call_id": "toolu_01C7", "content": "1| x"}
    assert r2.text == "done" and r2.tool_calls == ()


def test_qwen_style_content_json_becomes_a_tool_call(stub: Stub) -> None:  # noqa: F811
    content = '{"name": "read_file", "arguments": {"path": "train.py"}}'
    stub.add("POST", PATH, 200, {"choices": [{"finish_reason": "stop",
                                              "message": {"content": content}}]})  # fmt: skip
    resp = _llm(stub).chat(MSGS, TOOLS, max_tokens=50)
    assert resp.from_content and resp.text is None
    assert resp.tool_calls[0].name == "read_file"
    assert json.loads(resp.tool_calls[0].arguments) == {"path": "train.py"}


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "rm_rf", "arguments": {}}',  # not a declared tool
        '{"name": "read_file", "arguments": {}, "x": 1}',  # extra keys
        'I will call {"name": "read_file", "arguments": {}}',  # free text
        '{"file": "train.py", "edits": []}',
        None,
    ],
)
def test_content_recovery_is_narrow(content: str | None) -> None:
    assert tool_call_from_content(content, TOOLS) is None


def test_fenced_content_recovered() -> None:
    c = tool_call_from_content('```json\n{"name": "read_file", "arguments": "{}"}\n```', TOOLS)
    assert c is not None and c.arguments == "{}"


def test_provider_refusing_tools_is_tools_unsupported(stub: Stub) -> None:  # noqa: F811
    stub.add("POST", PATH, 404, {"error": {"message": "No endpoints found that support tool use"}})
    with pytest.raises(ToolsUnsupported):
        _llm(stub).chat(MSGS, TOOLS, max_tokens=50)


def test_other_4xx_stays_llm_error(stub: Stub) -> None:  # noqa: F811
    stub.add("POST", PATH, 401, {"error": "bad key"})
    with pytest.raises(LLMError) as ei:
        _llm(stub).chat(MSGS, TOOLS, max_tokens=50)
    assert not isinstance(ei.value, ToolsUnsupported)


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": [{"message": {"content": None, "tool_calls": "nope"}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}]},
    ],
)
def test_bad_shapes_are_protocol_errors(stub: Stub, body: Any) -> None:  # noqa: F811
    stub.add("POST", PATH, 200, body)
    with pytest.raises(ToolProtocolError):
        _llm(stub).chat(MSGS, TOOLS, max_tokens=50)


def test_metered_chat_records_and_budget() -> None:
    fake = FakeLLM([], tool_script=[("read_file", {"path": "a.py"})])
    budget = Budget(max_tokens_total=10_000)
    m = MeteredLLM(fake, budget, "critic")
    resp = m.chat(MSGS, TOOLS, max_tokens=100)
    assert isinstance(resp, ChatResponse) and resp.tool_calls[0].name == "read_file"
    (rec,) = m.records()
    assert rec.node == "critic" and budget.used_tokens == resp.input_tokens + resp.output_tokens


def test_metered_chat_without_inner_chat_is_unsupported() -> None:
    m = MeteredLLM(OllamaLLM("http://x", "m"), Budget(), "critic")
    with pytest.raises(ToolsUnsupported):
        m.chat(MSGS, TOOLS, max_tokens=10)
    assert m.budget.used_tokens == 0 and m.records() == ()


def test_fake_without_script_is_unsupported() -> None:
    with pytest.raises(ToolsUnsupported):
        FakeLLM(["x"]).chat(MSGS, TOOLS, max_tokens=10)
