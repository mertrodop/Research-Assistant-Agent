"""Tests for reproducer.domain.grounding."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from reproducer.domain.grounding import EvidenceRef, RepoPath, normalize_repo_path

REPO = {"train.py", "configs/a.yaml", "README.md"}
EVIDENCE = {"e" * 64, "f" * 64}


class Paths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    main: RepoPath
    extra: list[RepoPath] = []


class Refs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: EvidenceRef


def _ctx(**kw: Any) -> dict[str, Any]:
    return {"repo_files": REPO, "evidence_ids": EVIDENCE, **kw}


def test_repo_path_ok() -> None:
    p = Paths.model_validate({"main": "train.py", "extra": ["configs/a.yaml"]}, context=_ctx())
    assert (p.main, p.extra) == ("train.py", ["configs/a.yaml"])


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("./train.py", "train.py"),
        ("configs//a.yaml", "configs/a.yaml"),
        ("configs/../train.py", "train.py"),
        ("  README.md ", "README.md"),
    ],
)
def test_repo_path_normalized(raw: str, norm: str) -> None:
    assert Paths.model_validate({"main": raw}, context=_ctx()).main == norm


def test_repo_path_missing_file_with_suggestion() -> None:
    with pytest.raises(ValidationError) as ei:
        Paths.model_validate({"main": "trian.py"}, context=_ctx())
    (err,) = ei.value.errors()
    assert err["loc"] == ("main",)
    assert "file not in repo: 'trian.py'" in err["msg"]
    assert "train.py" in err["msg"]  # did-you-mean


def test_repo_path_error_loc_in_list() -> None:
    with pytest.raises(ValidationError) as ei:
        Paths.model_validate(
            {"main": "train.py", "extra": ["README.md", "nope.txt"]}, context=_ctx()
        )
    assert [e["loc"] for e in ei.value.errors()] == [("extra", 1)]


@pytest.mark.parametrize("bad", ["../train.py", "configs/../../x", "/etc/passwd", "", ".", ".."])
def test_repo_path_rejects_escape_and_absolute(bad: str) -> None:
    with pytest.raises(ValidationError):
        Paths.model_validate({"main": bad}, context=_ctx(repo_files=REPO | {bad}))


def test_normalize_repo_path_direct() -> None:
    assert normalize_repo_path("a/./b/../c.py") == "a/c.py"
    with pytest.raises(ValueError):
        normalize_repo_path("../a")


@pytest.mark.parametrize(
    "ctx", [None, {}, {"evidence_ids": EVIDENCE}, {"repo_files": ["train.py"]}, "repo"]
)
def test_repo_path_missing_or_bad_context_fails_closed(ctx: Any) -> None:
    with pytest.raises(ValidationError) as ei:
        Paths.model_validate({"main": "train.py"}, context=ctx)
    assert "repo_files" in str(ei.value)


def test_evidence_ref_ok_and_missing() -> None:
    assert Refs.model_validate({"ref": "e" * 64}, context=_ctx()).ref == "e" * 64
    with pytest.raises(ValidationError, match="unknown evidence id"):
        Refs.model_validate({"ref": "a" * 64}, context=_ctx())


def test_evidence_ref_missing_context_fails_closed() -> None:
    with pytest.raises(ValidationError, match="evidence_ids"):
        Refs.model_validate({"ref": "e" * 64})


def test_json_schema_is_plain_string() -> None:
    props = Paths.model_json_schema()["properties"]
    assert props["main"] == {"title": "Main", "type": "string"}
