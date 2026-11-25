"""MemGPT-style eviction: shrink old tool results, never drop messages."""
from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage


def estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum(len(str(m.content)) for m in messages) // 4


def _handle(content: str) -> str:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("handle"):
            return str(data["handle"])
    except ValueError:
        pass
    return "in episode"


def fit_context(
    messages: list[BaseMessage], budget: int, keep_last: int
) -> tuple[list[BaseMessage], list[str]]:
    out, evicted = list(messages), []
    if estimate_tokens(out) <= budget:
        return out, evicted
    first_human = next((i for i, m in enumerate(out) if isinstance(m, HumanMessage)), -1)
    protected_from = len(out) - keep_last
    for i, m in enumerate(out):
        if estimate_tokens(out) <= budget:
            break
        if i >= protected_from or i == first_human or not isinstance(m, ToolMessage):
            continue
        text = str(m.content)
        stub = f"[evicted] {text[:120]} (full result: {_handle(text)})"
        out[i] = m.model_copy(update={"content": stub})
        evicted.append(text)
    return out, evicted
