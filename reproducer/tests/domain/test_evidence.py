"""Tests for reproducer.domain.evidence."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from reproducer.domain.evidence import (
    Evidence,
    canonical_json,
    dedupe,
    evidence_from_jsonl_line,
    make_evidence,
    to_jsonl_line,
)


def _ev(**overrides: Any) -> Evidence:
    kwargs: dict[str, Any] = {
        "kind": "repo_file",
        "source": "example-lab/resnet18@abc123",
        "locator": "train.py:L42",
        "content": {"lr": 0.1, "nested": {"b": 2, "a": 1}},
        "confidence": "observed",
        "authority": "repo",
    }
    kwargs.update(overrides)
    return make_evidence(**kwargs)


def test_canonical_json_exact() -> None:
    assert canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'


def test_id_stable_across_key_order() -> None:
    a = _ev(content={"x": 1, "y": {"p": 1, "q": 2}})
    b = _ev(content={"y": {"q": 2, "p": 1}, "x": 1})
    assert a.evidence_id == b.evidence_id
    assert len(a.evidence_id) == 64


def test_different_content_different_id() -> None:
    assert _ev(content={"x": 1}).evidence_id != _ev(content={"x": 2}).evidence_id


def test_non_hashed_fields_do_not_change_id() -> None:
    base = _ev()
    other = _ev(confidence="inferred", authority="llm", links=("e1", "e2"))
    assert base.evidence_id == other.evidence_id


def test_tampered_id_rejected_on_reload() -> None:
    data = json.loads(to_jsonl_line(_ev()))
    data["evidence_id"] = "0" * 64
    with pytest.raises(ValidationError, match="evidence_id mismatch"):
        evidence_from_jsonl_line(json.dumps(data))


def test_tampered_content_rejected_on_reload() -> None:
    data = json.loads(to_jsonl_line(_ev()))
    data["content"]["lr"] = 0.2
    with pytest.raises(ValidationError, match="evidence_id mismatch"):
        evidence_from_jsonl_line(json.dumps(data))


def test_direct_construction_with_bad_id_rejected() -> None:
    good = _ev()
    data = good.model_dump()
    data["evidence_id"] = "deadbeef"
    with pytest.raises(ValidationError):
        Evidence(**data)


def test_jsonl_round_trip() -> None:
    ev = _ev(links=("a", "b"))
    line = to_jsonl_line(ev)
    assert "\n" not in line
    assert evidence_from_jsonl_line(line + "\n") == ev


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        _ev(content={"x": bad})


def test_non_json_content_rejected() -> None:
    with pytest.raises(ValueError):
        _ev(content={"x": {1, 2}})


def test_content_is_detached_copy() -> None:
    content: dict[str, Any] = {"x": [1, 2], "y": {"z": 1}}
    ev = _ev(content=content)
    before_id = ev.evidence_id
    before = ev.content
    content["x"].append(3)
    content["y"]["z"] = 99
    content["new"] = True
    assert ev.content == {"x": [1, 2], "y": {"z": 1}}
    assert ev.content == before
    assert ev.evidence_id == before_id
    assert evidence_from_jsonl_line(to_jsonl_line(ev)) == ev


def test_dedupe_keeps_first() -> None:
    first = _ev(confidence="observed")
    dup = _ev(confidence="inferred")
    other = _ev(content={"x": 1})
    out = dedupe([first, other, dup])
    assert out == (first, other)
    assert out[0].confidence == "observed"


@pytest.mark.parametrize(
    ("authority", "confidence"),
    [("repo", "observed"), ("llm", "observed"), ("repo", "inferred")],
)
def test_llm_proposal_requires_llm_inferred(authority: str, confidence: str) -> None:
    with pytest.raises(ValidationError, match="llm_proposal evidence requires"):
        _ev(kind="llm_proposal", authority=authority, confidence=confidence)


def test_llm_proposal_llm_inferred_accepted() -> None:
    ev = _ev(kind="llm_proposal", authority="llm", confidence="inferred")
    assert evidence_from_jsonl_line(to_jsonl_line(ev)) == ev
