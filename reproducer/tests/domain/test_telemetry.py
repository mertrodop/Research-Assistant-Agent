"""Tests for reproducer.domain.telemetry.token_totals (the single token-count helper)."""

from __future__ import annotations

from reproducer.domain.telemetry import CallRecord, token_totals


def _rec(backend: str, tin: int, tout: int) -> CallRecord:
    return CallRecord(node="n", backend=backend, model="m", input_tokens=tin,
                      output_tokens=tout, latency_s=0.1, prompt_sha256="0" * 64)  # fmt: skip


def test_token_totals_split_and_count() -> None:
    records = [_rec("anthropic", 100, 10), _rec("ollama", 50, 5), _rec("openrouter", 1, 2)]
    assert token_totals(records) == {"api": 113, "local": 55, "calls": 3}


def test_token_totals_empty() -> None:
    assert token_totals([]) == {"api": 0, "local": 0, "calls": 0}


def test_cli_uses_the_domain_helper() -> None:
    import reproducer.cli as cli

    assert cli.token_totals is token_totals
    assert not hasattr(cli, "token_split")
