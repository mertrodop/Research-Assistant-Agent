"""Bounded ReAct loop for the grounded critic (Thought -> Action -> Observation).

The model sees the same grounded inputs as the single-shot critic and may call read-only
tools (critic_tools: read_file / grep / list_dir on the virtual repo view) before it calls
submit_patch with a CriticPatch. submit_patch is validated by the SAME code as today
(CriticPatch.model_validate_json with the critic context); a validation error goes back as
the tool result, so the model can correct it within the step budget.

Hard limits (ReactLimits): max_steps tool calls in total (a reply without a tool call also
uses a step), a loop-local token cap and a wall-clock cap; the shared Budget still applies to
every call (BudgetExceeded propagates). Loop detection: a tool call identical to an earlier
one (same name + canonical args) is not run but rejected; a reply without a tool call is
rejected too; 2 rejections stop the loop.

The loop never returns an unvalidated patch. When it ends without one it returns a
fallback_reason: no tool support (ToolsUnsupported), a protocol error (ToolProtocolError), a
context-limit error, the step / token / wall-clock caps, or the rejection limit. The caller
(critic.run_critic) then runs the single-shot critic once and records the reason.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import ValidationError

from reproducer.domain.contracts import CriticPatch
from reproducer.graph.eviction import is_context_error
from reproducer.graph.gate import MAX_ERROR_CHARS, summarize_errors
from reproducer.graph.nodes.critic_tools import (
    MAX_RESULT_CHARS,
    RepoView,
    canonical_args,
    cap,
    read_tool_specs,
    tool_spec,
)
from reproducer.providers.llm import (
    LLMError,
    ToolProtocolError,
    ToolsUnsupported,
    assistant_message,
    tool_message,
)

SUBMIT = "submit_patch"
MAX_REJECTIONS = 2
MAX_TRACE_ARGS_CHARS = 1000
MAX_THOUGHT_CHARS = 500
NO_TOOL_NUDGE = (
    "Reply with a tool call: inspect with read_file / grep / list_dir, or finish with submit_patch."
)


@dataclass(frozen=True)
class ReactLimits:
    max_steps: int = 5
    token_cap: int = 40_000  # input + output tokens of the loop's own calls
    wall_s: float = 180.0
    clock: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class TraceStep:
    step: int
    tool: str  # tool name; "(none)" for a reply without a tool call
    args: str  # canonical JSON, cut to MAX_TRACE_ARGS_CHARS
    result_chars: int  # observation length before truncation
    truncated: bool
    ok: bool
    note: str = ""  # rejected / invalid patch / accepted / from_content

    def as_json(self) -> dict[str, Any]:
        return {"step": self.step, "tool": self.tool, "args": self.args,
                "result_chars": self.result_chars, "truncated": self.truncated, "ok": self.ok,
                "note": self.note}  # fmt: skip


@dataclass(frozen=True)
class CriticTrace:
    """The ReAct trajectory of one critic attempt (stored as critic_trace evidence)."""

    steps: tuple[TraceStep, ...] = ()
    outcome: str = "fallback"  # patched | fallback
    fallback_reason: str | None = None
    fallback_status: str | None = None  # the single-shot critic's status after a fallback
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts: tuple[str, ...] = ()  # model text that came with its turns (cut)

    def as_json(self) -> dict[str, Any]:
        return {
            "mode": "react",
            "steps": [s.as_json() for s in self.steps],
            "outcome": self.outcome,
            "fallback_reason": self.fallback_reason,
            "fallback_status": self.fallback_status,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thoughts": list(self.thoughts),
        }


@dataclass(frozen=True)
class ReactResult:
    patch: CriticPatch | None  # validated; None -> trace.fallback_reason is set
    submits: int  # submit_patch calls validated by code
    errors: tuple[str, ...]  # validation errors of rejected submits
    trace: CriticTrace


def submit_spec() -> dict[str, Any]:
    return tool_spec(
        SUBMIT,
        "Submit the final patch for the failing file (validated by code; a rejection comes "
        "back as the result, then fix it and submit again).",
        CriticPatch,
    )


def react_tools() -> list[dict[str, Any]]:
    return [*read_tool_specs(), submit_spec()]


@dataclass
class _Loop:
    limits: ReactLimits
    steps: list[TraceStep] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    seen: dict[tuple[str, str], int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    rejections: int = 0
    submits: int = 0
    calls: int = 0
    tok_in: int = 0
    tok_out: int = 0

    @property
    def step(self) -> int:
        return len(self.steps)

    def trace(self, outcome: str, reason: str | None) -> CriticTrace:
        return CriticTrace(
            steps=tuple(self.steps), outcome=outcome, fallback_reason=reason,
            llm_calls=self.calls, input_tokens=self.tok_in, output_tokens=self.tok_out,
            thoughts=tuple(self.thoughts),
        )  # fmt: skip

    def result(self, reason: str, patch: CriticPatch | None = None) -> ReactResult:
        trace = self.trace("patched" if patch else "fallback", None if patch else reason)
        return ReactResult(patch, self.submits, tuple(self.errors), trace)


def _validate(arguments: str, context: Mapping[str, Any]) -> tuple[CriticPatch | None, str]:
    try:
        return CriticPatch.model_validate_json(arguments or "{}", context=dict(context)), ""
    except ValidationError as e:
        return None, summarize_errors(e)[:MAX_ERROR_CHARS]


def run_react(
    llm: Any,
    *,
    system: str,
    prompt: str,
    view: RepoView,
    context: Mapping[str, Any],
    max_tokens: int,
    limits: ReactLimits | None = None,
) -> ReactResult:
    """The bounded ReAct loop (module docstring). `llm` needs chat(messages, tools, ...)
    (providers.llm.MeteredLLM / OpenAICompatLLM / FakeLLM with a tool_script)."""
    lim = limits or ReactLimits()
    st = _Loop(lim)
    t0 = lim.clock()
    tools = react_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    chat = getattr(llm, "chat", None)
    if chat is None:
        return st.result("no tool support: the LLM client has no chat()")
    while True:
        if st.step >= lim.max_steps:
            return st.result(f"no valid patch after {lim.max_steps} steps")
        if lim.clock() - t0 > lim.wall_s:
            return st.result(f"wall-clock cap {lim.wall_s:.0f}s reached after {st.step} steps")
        if st.tok_in + st.tok_out >= lim.token_cap:
            return st.result(f"token cap {lim.token_cap} reached after {st.step} steps")
        try:
            resp = chat(messages, tools, max_tokens=max_tokens)
        except ToolsUnsupported as e:
            return st.result(f"no tool support: {e}")
        except ToolProtocolError as e:
            return st.result(f"protocol error: {e}")
        except LLMError as e:
            if is_context_error(e):
                return st.result(f"context: {e}")
            raise
        st.calls += 1
        st.tok_in += resp.input_tokens
        st.tok_out += resp.output_tokens
        if resp.text and resp.text.strip():
            st.thoughts.append(resp.text.strip()[:MAX_THOUGHT_CHARS])
        messages.append(assistant_message(resp))
        if not resp.tool_calls:
            st.rejections += 1
            st.steps.append(TraceStep(st.step + 1, "(none)", "", 0, False, False,
                                      "rejected: reply without a tool call"))  # fmt: skip
            if st.rejections >= MAX_REJECTIONS:
                return st.result(f"{MAX_REJECTIONS} rejected turns (last: no tool call)")
            messages.append({"role": "user", "content": NO_TOOL_NUDGE})
            continue
        for call in resp.tool_calls:
            if st.step >= lim.max_steps:  # every tool_call_id needs an answer
                messages.append(tool_message(call.id, "error: step cap reached; not run"))
                continue
            n = st.step + 1
            left = lim.max_steps - n
            note_tail = (f"\n(steps left: {left}" + ("; call submit_patch now)" if left == 1
                         else ")"))  # fmt: skip
            key = (call.name, canonical_args(call.arguments))
            args_trace = key[1][:MAX_TRACE_ARGS_CHARS]
            src = "from_content; " if resp.from_content else ""
            if key in st.seen:
                st.rejections += 1
                body = (f"error: identical call already made at step {st.seen[key]}; its "
                        "result is above. Do something else.")  # fmt: skip
                st.steps.append(TraceStep(n, call.name, args_trace, len(body), False, False,
                                          f"{src}rejected: repeated call"))  # fmt: skip
                messages.append(tool_message(call.id, body + note_tail))
                continue
            st.seen[key] = n
            if call.name == SUBMIT:
                st.submits += 1
                patch, err = _validate(call.arguments, context)
                if patch is not None:
                    st.steps.append(TraceStep(n, SUBMIT, args_trace, 0, False, True,
                                              f"{src}accepted"))  # fmt: skip
                    return st.result("", patch)
                st.errors.append(err)
                raw = f"submit_patch rejected (fix it and call submit_patch again):\n{err}"
                text, trunc = cap(raw, MAX_RESULT_CHARS - len(note_tail))
                st.steps.append(TraceStep(n, SUBMIT, args_trace, len(raw), trunc, False,
                                          f"{src}invalid patch"))  # fmt: skip
                messages.append(tool_message(call.id, text + note_tail))
                continue
            res = view.call(call.name, call.arguments, limit=MAX_RESULT_CHARS - len(note_tail))
            st.steps.append(TraceStep(n, call.name, args_trace, res.chars, res.truncated, res.ok,
                                      src.rstrip("; ")))  # fmt: skip
            messages.append(tool_message(call.id, res.text + note_tail))
        if st.rejections >= MAX_REJECTIONS:
            return st.result(f"{MAX_REJECTIONS} rejected calls (loop detected)")


def with_fallback(trace: CriticTrace, status: str) -> CriticTrace:
    return replace(trace, fallback_status=status)
