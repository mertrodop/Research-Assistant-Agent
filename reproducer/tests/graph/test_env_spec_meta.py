"""Subtask A evidence carries node metadata (gate_attempts, evicted) next to the EnvSpec
fields: env planning strips it before re-validating, and the same spec stored again with
other metadata is still one spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from reproducer.domain.evidence import make_evidence
from reproducer.graph.nodes.env import plan_env, stored_env_spec
from reproducer.providers.store import RunStore

SRC = "repo:x@1"


def _spec(store: RunStore, **meta: object) -> None:
    store.append_evidence(make_evidence(
        kind="env_spec", source=SRC, locator="subtask:A",
        content={"python": None, "packages": [], "gpu_required": False, "unknowns": [],
                 **meta},
        confidence="derived", authority="llm",
    ))  # fmt: skip


def test_same_spec_with_other_metadata_is_one(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _spec(store)  # an older record without gate_attempts
    _spec(store, gate_attempts=2)
    _spec(store, gate_attempts=1, evicted=["README.md"])
    assert "gate_attempts" not in stored_env_spec(store).content  # the first stored
    store.append_evidence(make_evidence(
        kind="env_spec", source=SRC, locator="subtask:A",
        content={"python": None, "packages": [], "gpu_required": True, "unknowns": [],
                 "gate_attempts": 1},
        confidence="derived", authority="llm",
    ))  # fmt: skip
    with pytest.raises(ValueError, match="expected exactly one"):
        stored_env_spec(store)


def test_plan_env_accepts_metadata_keys(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _spec(store, gate_attempts=3, evicted=["README.md"])
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "train.py").write_text("print('hi')\n")
    planned = plan_env(tmp_path, True, gpu_query=lambda: None)
    assert planned.spec_evidence.content["gate_attempts"] == 3
