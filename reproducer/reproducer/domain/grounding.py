"""Grounded string types for LLM output, checked against Pydantic validation context.

Pure: no IO. Same pattern as RunPlan: the facts to ground against are passed as
`model_validate*(..., context={...})`. Missing or malformed context fails closed with a
ValueError (surfaced as a ValidationError), never a silent pass.

    RepoPath     normalized posix relative path; must be in context["repo_files"]
    EvidenceRef  must be in context["evidence_ids"]
"""

from __future__ import annotations

import difflib
import posixpath
from collections.abc import Mapping, Set
from typing import Annotated, Any

from pydantic import AfterValidator, ValidationInfo

_MAX_SUGGESTIONS = 3


def _context_set(info: ValidationInfo, key: str, type_name: str) -> Set[str]:
    ctx: Any = info.context
    if not isinstance(ctx, Mapping) or key not in ctx:
        raise ValueError(f"{type_name} requires validation context with key {key!r}")
    values = ctx[key]
    if not isinstance(values, Set):
        raise ValueError(f"validation context {key!r} must be a set, got {type(values).__name__}")
    return values


def _suggest(value: str, choices: Set[str]) -> str:
    close = difflib.get_close_matches(value, sorted(choices), n=_MAX_SUGGESTIONS, cutoff=0.6)
    return f"; did you mean one of {close}?" if close else ""


def normalize_repo_path(value: str) -> str:
    """'./src/../train.py' -> 'train.py'. Rejects absolute paths and paths leaving the repo."""
    raw = value.strip()
    if not raw:
        raise ValueError("path is empty")
    if raw.startswith("/"):
        raise ValueError(f"path must be relative to the repo root, got {value!r}")
    norm = posixpath.normpath(raw)
    if norm == "." or norm == ".." or norm.startswith("../"):
        raise ValueError(f"path escapes the repo root: {value!r}")
    return norm


def _check_repo_path(value: str, info: ValidationInfo) -> str:
    repo_files = _context_set(info, "repo_files", "RepoPath")
    norm = normalize_repo_path(value)
    if norm not in repo_files:
        raise ValueError(f"file not in repo: {norm!r}{_suggest(norm, repo_files)}")
    return norm


def _check_evidence_ref(value: str, info: ValidationInfo) -> str:
    evidence_ids = _context_set(info, "evidence_ids", "EvidenceRef")
    if value not in evidence_ids:
        raise ValueError(f"unknown evidence id: {value!r}")
    return value


RepoPath = Annotated[str, AfterValidator(_check_repo_path)]
EvidenceRef = Annotated[str, AfterValidator(_check_evidence_ref)]
