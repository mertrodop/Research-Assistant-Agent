"""Context eviction with retry for the gated LLM nodes (subtasks A/B/C, orchestrator, critic,
reflect).

A node passes a prompt builder instead of a fixed prompt: build(budget) -> (system, prompt,
evicted_titles). budget None (or the node's own default) builds the normal prompt, unchanged,
so normal-sized calls send the same prompts (same hashes) as before. When the call hits the
context limit, either the client-side guard (ContextTooLong: estimated size > num_ctx, no
request made) or the provider (HTTP 400 naming the context / token limit), the budget is
halved and the prompt rebuilt: pack_context drops the lowest-priority sections first, then
truncates. At most MAX_EVICTIONS rebuilds; then ContextEvictionFailed, whose message starts
with "fallback: context:" so the node stops cleanly with that reason. The evicted titles go
into each CallRecord (MeteredLLM.evicted) and into the node's evidence ("evicted").

Worst case per node call: (gate retries + 1) x (eviction retries + 1) = (2 + 1) x (2 + 1) = 9
attempts, since each rebuilt prompt restarts gated_call. Client-side refusals (ContextTooLong)
send no request and spend no tokens; provider rejections are released from the reservation;
the shared Budget caps the tokens of every attempt that does run (BudgetExceeded propagates).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from reproducer.graph.gate import GateResult, gated_call
from reproducer.providers.llm import LLM, ContextTooLong, LLMError

if TYPE_CHECKING:  # subtasks imports this module: its packing helpers are imported lazily
    from reproducer.graph.nodes.subtasks import ContextSection

T = TypeVar("T", bound=BaseModel)

MAX_EVICTIONS = 2
PromptBuilder = Callable[[int | None], tuple[str, str, tuple[str, ...]]]

_CTX_WORD = re.compile(r"context|token", re.IGNORECASE)
_LIMIT_WORD = re.compile(
    r"length|limit|exceed|too long|too large|too many|maximum|window", re.IGNORECASE
)


class ContextEvictionFailed(ContextTooLong):
    """The prompt still did not fit after MAX_EVICTIONS halvings (clean fallback stop)."""


def is_context_error(exc: BaseException) -> bool:
    """ContextTooLong (client-side guard), or a provider HTTP 400 whose body names the
    context / token limit (e.g. OpenAI "context_length_exceeded", OpenRouter "maximum
    context length")."""
    if isinstance(exc, ContextTooLong):
        return True
    if not isinstance(exc, LLMError) or exc.status != 400:
        return False
    text = f"{exc.body} {exc}"
    return bool(_CTX_WORD.search(text) and _LIMIT_WORD.search(text))


@dataclass(frozen=True)
class EvictedGate(Generic[T]):
    result: GateResult[T]
    evicted: tuple[str, ...]  # section titles dropped / truncated in the prompt that answered
    budget: int | None  # the budget of that prompt (None: the normal prompt)
    evictions: int  # rebuilds needed (0: the normal prompt fit)


def _set_evicted(llm: LLM, evicted: tuple[str, ...]) -> None:
    if hasattr(llm, "evicted"):
        llm.evicted = evicted  # type: ignore[attr-defined]


def call_with_eviction(
    llm: LLM,
    build_prompt: PromptBuilder,
    *,
    max_tokens: int,
    budget: int | None = None,
    max_evictions: int = MAX_EVICTIONS,
    **gate_kwargs: Any,
) -> EvictedGate[Any]:
    """gated_call(llm, system, prompt, max_tokens, **gate_kwargs) on build_prompt(budget);
    on a context error, halve the budget (from the prompt length when None) and rebuild, at
    most max_evictions times. Other errors (BudgetExceeded, other LLMError) propagate."""
    evictions = 0
    while True:
        system, prompt, evicted = build_prompt(budget)
        _set_evicted(llm, tuple(evicted))
        try:
            result = gated_call(
                llm, system=system, prompt=prompt, max_tokens=max_tokens, **gate_kwargs
            )
        except LLMError as e:
            if not is_context_error(e):
                raise
            if evictions >= max_evictions:
                raise ContextEvictionFailed(
                    f"fallback: context: prompt still too long after {evictions} eviction(s) "
                    f"(budget {budget} chars, evicted {list(evicted)}): {e}"
                ) from e
            budget = (len(prompt) if budget is None else budget) // 2
            evictions += 1
            continue
        finally:
            _set_evicted(llm, ())
        return EvictedGate(result, tuple(evicted), budget, evictions)


# --- prompt shrinking for prompts that are not packed already -----------------------------

_HEADER = re.compile(r"(?m)^## ")


def markdown_sections(text: str, priority: Callable[[int, str], int]) -> list[ContextSection]:
    """Split a '## <title>\\n<body>' prompt into ContextSections. priority(index, title);
    text before the first header becomes a section titled "(preamble)"."""
    from reproducer.graph.nodes.subtasks import ContextSection

    starts = [m.start() for m in _HEADER.finditer(text)]
    out: list[ContextSection] = []
    head = text[: starts[0]] if starts else text
    if head.strip():
        out.append(ContextSection(title="(preamble)", text=head.strip("\n"), priority=1000))
    for i, start in enumerate(starts):
        chunk = text[start + 3 : starts[i + 1] if i + 1 < len(starts) else len(text)]
        title, _, body = chunk.partition("\n")
        out.append(
            ContextSection(title=title.strip(), text=body.strip("\n"), priority=priority(i, title))
        )
    return out


def shrink_prompt(
    full: str, sections: Sequence[ContextSection], budget: int | None, tail: str = ""
) -> tuple[str, tuple[str, ...]]:
    """`full` unchanged when budget is None or it fits; else the sections packed into
    budget (minus the tail) + tail, and the evicted titles."""
    if budget is None or len(full) <= budget:
        return full, ()
    from reproducer.graph.nodes.subtasks import pack_context_report

    room = max(0, budget - len(tail) - 2)
    packed, evicted = pack_context_report(sections, room)
    return (f"{packed}\n\n{tail}" if tail else packed), evicted
