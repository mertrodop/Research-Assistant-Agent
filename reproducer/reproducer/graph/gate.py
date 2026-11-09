"""Pydantic gate: LLM output must pass schema + code-level grounding, else retry with feedback.

Deterministic: the only feedback is the real ValidationError (or JSON error) text plus the
previous answer. No LLM self-critique. BudgetExceeded / LLMError propagate immediately.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from reproducer.providers.llm import LLM

T = TypeVar("T", bound=BaseModel)

MAX_ERROR_CHARS = 1000
MAX_PREVIOUS_CHARS = 2000
_FENCE_RE = re.compile(r"^```[ \t]*(?:json)?[ \t]*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class GateResult(Generic[T]):
    ok: bool
    value: T | None
    attempts: int
    errors: tuple[str, ...]  # one per failed attempt, each <= MAX_ERROR_CHARS


def strip_fence(text: str) -> str:
    """Remove one surrounding ``` / ```json fence and outer whitespace."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def summarize_errors(exc: ValidationError) -> str:
    """One 'loc: msg' line per error (loc dotted; '(root)' for model-level errors)."""
    lines = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"{loc}: {err['msg']}")
    return "\n".join(lines)


def _feedback_prompt(prompt: str, error: str, previous: str) -> str:
    return (
        f"{prompt}\n\nYour previous answer was rejected:\n{error}\n"
        f"{previous[:MAX_PREVIOUS_CHARS]}\nReturn ONLY corrected JSON matching the schema."
    )


def gated_call(
    llm: LLM,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    context: Mapping[str, Any],
    max_tokens: int,
    max_retries: int = 2,
    required_context: Iterable[str] = ("repo_files",),
) -> GateResult[T]:
    """Call llm until its JSON validates as `schema` under `context` (1 + max_retries tries).

    Missing `required_context` keys are a programming error: ValueError before any LLM call.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    required = sorted(set(required_context))
    missing = [k for k in required if k not in context]
    if missing:
        raise ValueError(f"gated_call: validation context is missing required key(s) {missing}")
    not_sets = [k for k in required if not isinstance(context[k], (set, frozenset))]
    if not_sets:
        raise ValueError(f"gated_call: validation context key(s) {not_sets} must be set/frozenset")
    json_schema = schema.model_json_schema()
    errors: list[str] = []
    current = prompt
    for attempt in range(1, max_retries + 2):
        resp = llm.complete(system, current, max_tokens=max_tokens, json_schema=json_schema)
        try:
            value = schema.model_validate_json(strip_fence(resp.text), context=dict(context))
        except ValidationError as e:  # covers invalid JSON (json_invalid) and grounding
            errors.append(summarize_errors(e)[:MAX_ERROR_CHARS])
            current = _feedback_prompt(prompt, errors[-1], resp.text)
            continue
        return GateResult(ok=True, value=value, attempts=attempt, errors=tuple(errors))
    return GateResult(ok=False, value=None, attempts=max_retries + 1, errors=tuple(errors))
