from __future__ import annotations

import json

from ra_agent.config import VisionSettings
from ra_agent.ocr.gate import check_figure_grounding
from ra_agent.ocr.interpret import ground, interpret_figure, interpret_figures
from ra_agent.schema import FigureInterpretation, FigureRef

GOOD_JSON = json.dumps(
    {
        "summary": "Bar chart of replication scores reaching 21% for the best agent.",
        "kind": "chart",
        "key_values": [{"label": "best agent", "value": "21%"}],
        "takeaways": ["human baseline is far higher"],
        "confidence": 0.8,
    }
)


class FakeClient:
    model = "fake-vision"

    def __init__(self, reply: str = GOOD_JSON) -> None:
        self.reply = reply
        self.calls = 0

    def describe(self, image_path, prompt):
        self.calls += 1
        assert "Caption" in prompt
        return self.reply


def figure(caption: str = "Figure 1. Replication scores.") -> FigureRef:
    return FigureRef(
        page_no=1,
        index=1,
        filename="p001_fig1.png",
        width=400,
        height=300,
        caption=caption,
        label="Figure 1",
    )


def test_ground_accepts_numbers_from_page_text():
    reading = FigureInterpretation(
        summary="best agent reaches 21%", key_values=[{"label": "score", "value": "21%"}]
    )
    assert ground(reading, "Figure 1. Scores. The best agent reaches 21%.") == []


def test_ground_flags_invented_number():
    reading = FigureInterpretation(
        summary="best agent reaches 87%", key_values=[{"label": "score", "value": "87%"}]
    )
    assert ground(reading, "Figure 1. Scores. The best agent reaches 21%.") == ["87"]


def test_interpret_figure_parses_json(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")
    reading = interpret_figure(
        figure(), image, "The best agent reaches 21%.", FakeClient(), VisionSettings()
    )
    assert reading.kind == "chart"
    assert reading.key_values[0]["value"] == "21%"
    assert reading.ungrounded == []
    assert reading.ok


def test_interpret_figure_strips_code_fence(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")
    client = FakeClient(f"```json\n{GOOD_JSON}\n```")
    reading = interpret_figure(
        figure(), image, "21%", client, VisionSettings()
    )
    assert reading.error == ""
    assert reading.summary


def test_invalid_json_is_recorded_not_raised(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")
    reading = interpret_figure(
        figure(), image, "21%", FakeClient("I cannot read this."), VisionSettings()
    )
    assert "invalid JSON" in reading.error
    assert not reading.ok


def test_missing_image_is_recorded(tmp_path):
    reading = interpret_figures(
        [figure()],
        tmp_path,
        {1: "21%"},
        FakeClient(),
        VisionSettings(cache_dir=str(tmp_path / "cache")),
        sha="abc",
    )[0].interpretation
    assert reading.error == "image file missing"


def test_cache_prevents_second_call(tmp_path):
    image_dir = tmp_path / "figs"
    image_dir.mkdir()
    (image_dir / "p001_fig1.png").write_bytes(b"png")
    settings = VisionSettings(cache_dir=str(tmp_path / "cache"))
    client = FakeClient()

    first = interpret_figures([figure()], image_dir, {1: "21%"}, client, settings, sha="abc")
    second = interpret_figures([figure()], image_dir, {1: "21%"}, client, settings, sha="abc")

    assert client.calls == 1
    assert first[0].interpretation.summary == second[0].interpretation.summary


def test_truncated_json_is_repaired(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")
    truncated = '{"summary": "chart of 21%", "kind": "chart", "key_values": [{"label": "best", "value": "21%"'
    reading = interpret_figure(
        figure(), image, "21%", FakeClient(truncated), VisionSettings()
    )
    assert reading.error == ""
    assert reading.summary == "chart of 21%"


def test_trailing_comma_is_repaired(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")
    trailing = '{"summary": "chart of 21%", "kind": "chart", "takeaways": ["a",],}'
    reading = interpret_figure(
        figure(), image, "21%", FakeClient(trailing), VisionSettings()
    )
    assert reading.error == ""
    assert reading.takeaways == ["a"]


def test_retry_after_unparseable_reply(tmp_path):
    image = tmp_path / "p001_fig1.png"
    image.write_bytes(b"png")

    class RetryClient:
        model = "fake-vision"

        def __init__(self):
            self.calls = 0

        def describe(self, image_path, prompt):
            self.calls += 1
            return "not json at all" if self.calls == 1 else GOOD_JSON

    client = RetryClient()
    reading = interpret_figure(figure(), image, "21%", client, VisionSettings())
    assert client.calls == 2
    assert reading.error == ""
    assert reading.summary


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
