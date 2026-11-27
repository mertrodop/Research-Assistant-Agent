"""Context eviction with retry (graph/eviction.py): oversized context -> halve the budget and
re-pack (lowest priority first), at most 2 times, then a clean "fallback: context:" failure.
Both the client-side guard (ContextTooLong) and a provider HTTP 400 context error take it."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from reproducer.graph.eviction import (
    MAX_EVICTIONS,
    ContextEvictionFailed,
    call_with_eviction,
    is_context_error,
    markdown_sections,
    shrink_prompt,
)
from reproducer.graph.nodes.orchestrator import SYSTEM as ORCH_SYSTEM
from reproducer.graph.nodes.orchestrator import orchestrator_prompt_builder
from reproducer.graph.nodes.reflect import SYSTEM as REFLECT_SYSTEM
from reproducer.graph.nodes.reflect import reflect_prompt_builder
from reproducer.graph.nodes.subtasks import ContextSection, pack_context, pack_context_report
from reproducer.providers.llm import (
    Budget,
    ContextTooLong,
    FakeLLM,
    LLMError,
    LLMResponse,
    MeteredLLM,
)


class Ok(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ok: bool


PROVIDER_400 = (
    '{"error": {"message": "This model\'s maximum context length is 8192 tokens. However, '
    'your messages resulted in 9100 tokens.", "code": "context_length_exceeded"}}'
)


class LimitedLLM:
    """FakeLLM that rejects prompts longer than `limit` chars: like the client-side guard
    (ContextTooLong, no call) or like a provider (HTTP 400 naming the context length)."""

    backend = "fake"
    model = "fake"

    def __init__(self, limit: int, *, provider: bool = False) -> None:
        self.limit = limit
        self.provider = provider
        self.inner = FakeLLM(lambda s, p: '{"ok": true}')
        self.sizes: list[int] = []

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, json_schema: dict | None = None
    ) -> LLMResponse:
        self.sizes.append(len(prompt))
        if len(prompt) > self.limit:
            if self.provider:
                raise LLMError("request to x failed", status=400, body=PROVIDER_400)
            raise ContextTooLong(f"estimated too long: {len(prompt)} > {self.limit}")
        return self.inner.complete(system, prompt, max_tokens=max_tokens, json_schema=json_schema)


SECTIONS = [
    ContextSection(title="README [E1]", text="r" * 100, priority=100),
    ContextSection(title="Repo facts", text="f" * 400, priority=90),
    ContextSection(title="train.py [E2]", text="t" * 400, priority=80),
    ContextSection(title="util.py [E3]", text="u" * 400, priority=60),
]
FULL = pack_context(SECTIONS, 10_000)


def _builder(b: int | None) -> tuple[str, str, tuple[str, ...]]:
    text, evicted = (FULL, ()) if b is None else pack_context_report(SECTIONS, b)
    return "sys", text, evicted


def _call(llm: Any) -> Any:
    return call_with_eviction(
        llm, _builder, schema=Ok, context={}, max_tokens=8, required_context=()
    )


def test_normal_prompt_is_sent_unchanged() -> None:
    llm = MeteredLLM(LimitedLLM(10_000), Budget(), "n")
    out = _call(llm)
    assert out.result.ok and (out.evictions, out.evicted, out.budget) == (0, (), None)
    assert llm.inner.inner.prompts == [("sys", FULL)]  # type: ignore[attr-defined]
    (rec,) = llm.records()
    assert rec.evicted == ()


@pytest.mark.parametrize("provider", [False, True])
def test_one_eviction_drops_lowest_priority_first(provider: bool) -> None:
    inner = LimitedLLM(len(FULL) * 3 // 4, provider=provider)
    llm = MeteredLLM(inner, Budget(), "n")
    out = _call(llm)
    assert out.result.ok and out.evictions == 1 and out.budget == len(FULL) // 2
    assert out.evicted == ("train.py [E2]", "util.py [E3]")  # lowest priority first
    assert inner.sizes[0] == len(FULL) and inner.sizes[1] <= len(FULL) // 2
    (rec,) = llm.records()  # only the successful call is recorded
    assert rec.evicted == out.evicted
    assert llm.evicted == ()  # reset after the call


def test_two_evictions_then_success() -> None:
    inner = LimitedLLM(len(FULL) // 3)
    llm = MeteredLLM(inner, Budget(), "n")
    out = _call(llm)
    assert out.result.ok and out.evictions == 2 and out.budget == len(FULL) // 4
    assert out.evicted == ("Repo facts", "train.py [E2]", "util.py [E3]")  # README kept
    assert llm.records()[0].evicted == out.evicted


@pytest.mark.parametrize("provider", [False, True])
def test_still_too_large_after_two_is_a_clean_fallback(provider: bool) -> None:
    budget = Budget()
    inner = LimitedLLM(10, provider=provider)
    llm = MeteredLLM(inner, budget, "n")
    with pytest.raises(ContextEvictionFailed, match=r"^fallback: context: .*2 eviction"):
        _call(llm)
    assert len(inner.sizes) == 1 + MAX_EVICTIONS
    assert llm.records() == () and budget.used_tokens == 0  # nothing spent / recorded


def test_other_errors_propagate_without_eviction() -> None:
    class Down:
        def complete(self, *a: Any, **k: Any) -> LLMResponse:
            raise LLMError("bad request", status=400, body='{"error": "invalid model id"}')

    with pytest.raises(LLMError, match="invalid model") as ei:
        _call(Down())
    assert not isinstance(ei.value, ContextEvictionFailed)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ContextTooLong("x"), True),
        (LLMError("x", status=400, body=PROVIDER_400), True),
        (
            LLMError("x", status=400, body="prompt is too long: 210000 tokens > 200000 maximum"),
            True,
        ),
        (LLMError("x", status=400, body="Input token limit exceeded"), True),
        (LLMError("x", status=400, body="invalid json schema"), False),
        (LLMError("x", status=500, body="maximum context length"), False),
        (LLMError("x", status=401, body="token expired"), False),
        (ValueError("context length"), False),
    ],
)
def test_is_context_error(exc: BaseException, expected: bool) -> None:
    assert is_context_error(exc) is expected


# --- prompt builders of the orchestrator / reflect (critic: test_critic.py) ----------------

ORCH_PROMPT = "\n\n".join(
    [
        "## Paper claims (claim_id: metric = value on dataset (split); setting)",
        "- c1: acc = 96.0",
        "## Paper setup (name = value)",
        "- lr = 0.1",
        "## Entrypoint train.py: argparse options",
        "- --lr: type float",
        "## Verified commands (code runs the one sharing the most claim_args)",
        "python train.py",
        "## Verified metric output lines [E2] (found by code)",
        '- train.py:9 "print(acc)"',
        "## Training config lines [E3] (found by code)",
        "c" * 300,
        "## Script code train.py [E1]",
        "x" * 2000,
        "Return the JSON now.",
    ]
)


def test_orchestrator_builder_unchanged_then_drops_code_first() -> None:
    build = orchestrator_prompt_builder(ORCH_PROMPT)
    assert build(None) == (ORCH_SYSTEM, ORCH_PROMPT, ())
    assert build(len(ORCH_PROMPT)) == (ORCH_SYSTEM, ORCH_PROMPT, ())
    system, prompt, evicted = build(len(ORCH_PROMPT) // 2)
    assert system == ORCH_SYSTEM and len(prompt) <= len(ORCH_PROMPT) // 2
    assert evicted == ("Script code train.py [E1]",)
    assert "## Paper claims" in prompt and prompt.endswith("Return the JSON now.")
    _, prompt, evicted = build(400)
    assert "Training config lines [E3] (found by code)" in evicted
    assert "- c1: acc = 96.0" in prompt  # the claims survive longest


def test_reflect_builder_unchanged_then_drops_previous_lessons_first() -> None:
    prompt = "\n".join(
        [
            "## Claim c1 (claim_id c1, trial 1)",
            "paper: acc = 96.0",
            "\n## Our run, trial 1 [RUN]",
            "outcome: out_of_tolerance",
            "\n## Previous lessons for this claim (episodic memory; LLM-written, unverified)",
            "l" * 1500,
            "\n## Citable evidence (support quotes one of these)",
            "[E1] lr = 0.1",
            "\nReturn the JSON now.",
        ]
    )
    build = reflect_prompt_builder(prompt)
    assert build(None) == (REFLECT_SYSTEM, prompt, ())
    _, small, evicted = build(len(prompt) // 2)
    assert evicted == (
        "Previous lessons for this claim (episodic memory; LLM-written, unverified)",
    )
    assert "outcome: out_of_tolerance" in small and small.endswith("Return the JSON now.")


def test_markdown_sections_and_shrink() -> None:
    secs = markdown_sections("intro\n## A\na\n## B\nbb", lambda i, t: 10 - i)
    assert [(s.title, s.text, s.priority) for s in secs] == [
        ("(preamble)", "intro", 1000),
        ("A", "a", 10),
        ("B", "bb", 9),
    ]
    assert shrink_prompt("full", secs, None) == ("full", ())
    assert shrink_prompt("full", secs, 4) == ("full", ())
