from __future__ import annotations

import json

import pytest

from ra_agent.config import CriticSettings
from ra_agent.ocr.critic import critique_page, critique_pages
from ra_agent.schema import PageResult, Route

GOOD = json.dumps(
    {
        "fidelity": 5,
        "completeness": 4,
        "structure": 5,
        "no_hallucination": 5,
        "evidence": "We evaluate three baselines",
        "reason": "faithful",
    }
)


class FakeLlm:
    model = "fake"

    def __init__(self, reply: str = GOOD) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        assert "fidelity" in prompt
        return self.reply

    def describe(self, image_path, prompt):  # pragma: no cover - unused here
        raise AssertionError("not used")


def page(no: int = 1, markdown: str = "Body text about baselines.") -> PageResult:
    return PageResult(page_no=no, route=Route.NATIVE_TEXT, markdown=markdown)


def test_high_scores_pass():
    check = critique_page("Body text.", FakeLlm(), CriticSettings())
    assert check.passed
    assert check.severity.value == "info"
    assert "fidelity=5" in check.detail


def test_low_dimension_blocks():
    reply = json.dumps(
        {
            "fidelity": 2,
            "completeness": 5,
            "structure": 5,
            "no_hallucination": 5,
            "evidence": "x",
            "reason": "invented sentence",
        }
    )
    check = critique_page("Body text.", FakeLlm(reply), CriticSettings())
    assert not check.passed
    assert check.severity.value == "blocker"


def test_low_average_blocks():
    reply = json.dumps(
        {
            "fidelity": 3,
            "completeness": 3,
            "structure": 3,
            "no_hallucination": 3,
            "evidence": "x",
            "reason": "mediocre",
        }
    )
    check = critique_page("Body text.", FakeLlm(reply), CriticSettings())
    assert not check.passed


def test_invalid_json_is_warn_not_crash():
    check = critique_page("Body text.", FakeLlm("no json"), CriticSettings())
    assert not check.passed
    assert check.severity.value == "warn"
    assert "invalid JSON" in check.detail


def test_empty_page_blocks():
    check = critique_page("   ", FakeLlm(), CriticSettings())
    assert not check.passed
    assert check.severity.value == "blocker"


def test_critique_pages_marks_failures():
    reply = json.dumps(
        {
            "fidelity": 1,
            "completeness": 1,
            "structure": 1,
            "no_hallucination": 1,
            "evidence": "x",
            "reason": "bad",
        }
    )
    updated = critique_pages([page()], FakeLlm(reply), CriticSettings(enabled=True))
    assert updated[0].status == "needs-review"
    assert "critic" in updated[0].review_reasons


def test_critique_pages_skips_disabled():
    llm = FakeLlm()
    updated = critique_pages([page()], llm, CriticSettings(enabled=False))
    assert llm.calls == 0
    assert updated[0].status == "verified"


def test_critique_pages_keeps_existing_failures():
    broken = page().model_copy(update={"status": "needs-review"})
    llm = FakeLlm()
    updated = critique_pages([broken], llm, CriticSettings(enabled=True))
    assert llm.calls == 0
    assert updated[0].status == "needs-review"
