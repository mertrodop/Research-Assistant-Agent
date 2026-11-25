from __future__ import annotations

import json

from ra_agent.config import VisionExtra
from ra_agent.ocr.gate import check_figure_grounding
from ra_agent.llm.client import VisionError
from ra_agent.ocr.interpret import describe_figure, describe_figures, figure_block
from ra_agent.schema import FigureInterpretation, FigureRef

GOOD_JSON = json.dumps(
    {
        "kind": "chart",
        "summary": "Bar chart of replication scores.",
        "markdown": "| agent | score |\n|---|---|\n| best | 21% |",
    }
)


class FakeClient:
    model = "fake-vision"

    def __init__(self, reply: str = GOOD_JSON) -> None:
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    def describe(self, image_path, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        return self.reply


class DownClient:
    model = "fake-vision"

    def describe(self, image_path, prompt):
        raise VisionError("endpoint down")


def figure(caption: str = "Figure 1. Replication scores.") -> FigureRef:
    return FigureRef(
        page_no=1,
        index=1,
        filename="fig-1.png",
        width=400,
        height=300,
        caption=caption,
        label="Figure 1",
    )


def describe(client, page_text="The best agent reaches 21%.", tmp_path=None):
    (tmp_path / "fig-1.png").write_bytes(b"png")
    extra = VisionExtra(cache_dir=str(tmp_path / "cache"))
    return describe_figure(figure(), tmp_path, page_text, client, extra, "abc").interpretation


def test_prompt_asks_for_required_keys(tmp_path):
    client = FakeClient()
    describe(client, tmp_path=tmp_path)
    for key in ("kind", "summary", "markdown"):
        assert f'"{key}"' in client.prompts[0]


def test_grounded_table_has_no_ungrounded(tmp_path):
    reply = json.dumps({"kind": "table", "summary": "s", "markdown": "| a |\n|---|\n| 21.0 |"})
    reading = describe(FakeClient(reply), "scores of 21.0 overall", tmp_path)
    assert reading.kind == "table"
    assert reading.ungrounded == []
    assert reading.ok


def test_numbers_missing_from_page_text_are_ungrounded(tmp_path):
    reply = json.dumps({"kind": "table", "summary": "s", "markdown": "| a |\n|---|\n| 21.0 |"})
    reading = describe(FakeClient(reply), "no figures here", tmp_path)
    assert reading.ungrounded == ["21.0"]


def test_vision_error_gives_unavailable_block(tmp_path):
    reading = describe(DownClient(), tmp_path=tmp_path)
    assert reading.error
    block = figure_block(figure().model_copy(update={"interpretation": reading}), "figures")
    assert "⚠ description unavailable" in block


def test_strips_code_fence(tmp_path):
    reading = describe(FakeClient(f"```json\n{GOOD_JSON}\n```"), tmp_path=tmp_path)
    assert reading.error == ""
    assert reading.summary


def test_invalid_json_is_recorded_not_raised(tmp_path):
    reading = describe(FakeClient("I cannot read this."), tmp_path=tmp_path)
    assert "invalid JSON" in reading.error


def test_unknown_kind_becomes_other(tmp_path):
    reply = json.dumps({"kind": "screenshot", "summary": "s", "markdown": ""})
    assert describe(FakeClient(reply), tmp_path=tmp_path).kind == "other"


def test_missing_image_is_recorded(tmp_path):
    extra = VisionExtra(cache_dir=str(tmp_path / "cache"))
    out = describe_figures([figure()], tmp_path, {1: "21%"}, FakeClient(), extra, "abc")
    assert out[0].interpretation.error == "image file missing"


def test_cache_prevents_second_call(tmp_path):
    (tmp_path / "fig-1.png").write_bytes(b"png")
    extra = VisionExtra(cache_dir=str(tmp_path / "cache"))
    client = FakeClient()
    first = describe_figures([figure()], tmp_path, {1: "21%"}, client, extra, "abc")
    second = describe_figures([figure()], tmp_path, {1: "21%"}, client, extra, "abc")
    assert client.calls == 1
    assert first[0].interpretation.summary == second[0].interpretation.summary


def test_truncated_json_is_repaired(tmp_path):
    truncated = '{"kind": "chart", "summary": "chart of 21%", "markdown": "| a |'
    reading = describe(FakeClient(truncated), tmp_path=tmp_path)
    assert reading.error == ""
    assert reading.summary == "chart of 21%"


def test_trailing_comma_is_repaired(tmp_path):
    reading = describe(FakeClient('{"kind": "chart", "summary": "x",}'), tmp_path=tmp_path)
    assert reading.error == ""


def test_retry_after_unparseable_reply(tmp_path):
    class RetryClient(FakeClient):
        def describe(self, image_path, prompt):
            self.calls += 1
            return "not json at all" if self.calls == 1 else GOOD_JSON

    client = RetryClient()
    reading = describe(client, tmp_path=tmp_path)
    assert client.calls == 2
    assert reading.error == ""


def test_figure_block_format():
    reading = FigureInterpretation(
        kind="chart",
        summary="Scores.",
        markdown="| a |\n|---|",
        ungrounded=["12.5", "0.3"],
    )
    fig = figure("Figure 1. Replication scores. More text.").model_copy(
        update={"interpretation": reading}
    )
    assert figure_block(fig, "figs") == (
        "![Figure 1: Replication scores.](figs/fig-1.png)\n\n"
        "> **chart**: Scores.\n>\n> | a |\n> |---|\n"
        "> ⚠ not in page text: 12.5, 0.3"
    )


def test_gate_flags_ungrounded_figure():
    bad = figure().model_copy(
        update={
            "interpretation": FigureInterpretation(
                summary="reaches 87%", ungrounded=["87"], model="fake"
            )
        }
    )
    check = check_figure_grounding([bad])
    assert not check.passed
    assert check.severity.value == "warn"
    assert "87" in check.evidence


def test_gate_passes_grounded_figure():
    good = figure().model_copy(
        update={"interpretation": FigureInterpretation(summary="21%", model="fake")}
    )
    assert check_figure_grounding([good]).passed


def test_gate_reports_failed_reading():
    failed = figure().model_copy(
        update={"interpretation": FigureInterpretation(error="HTTP 401", model="fake")}
    )
    check = check_figure_grounding([failed])
    assert not check.passed
    assert "HTTP 401" in check.evidence
