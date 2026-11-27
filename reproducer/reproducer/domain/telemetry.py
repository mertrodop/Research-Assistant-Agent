"""LLM call telemetry record (pure: no IO). The prompt is kept only as its sha256."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

LOCAL_BACKENDS = frozenset({"ollama"})


class CallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    prompt_sha256: str
    # titles of the prompt sections dropped / truncated by context eviction for this call
    # (additive; empty -> not written to llm_calls.jsonl, so older lines stay identical)
    evicted: tuple[str, ...] = ()


def token_totals(records: Sequence[CallRecord]) -> dict[str, int]:
    """Tokens (in + out) by where they ran: local (ollama) vs api (everything else), plus the
    number of calls."""
    out = {"api": 0, "local": 0, "calls": len(records)}
    for r in records:
        where = "local" if r.backend in LOCAL_BACKENDS else "api"
        out[where] += r.input_tokens + r.output_tokens
    return out
