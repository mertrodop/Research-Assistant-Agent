"""A chat model that replays scripted replies: tests and evals without network."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        # a fresh id per call: a repeated reply must append, not replace, in the graph state
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy(update={"id": None}))])
