"""LLM call telemetry record (pure: no IO). The prompt is kept only as its sha256."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    prompt_sha256: str
