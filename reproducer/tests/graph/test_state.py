"""Tests for the configure-graph reducers (pure)."""

from __future__ import annotations

import itertools
import operator
from functools import reduce

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.unknowns import UnknownOpened, UnknownResolved
from reproducer.graph.state import SubtaskSummary, merge_events, merge_evidence


def _ev(kind: str, locator: str, n: int = 0) -> Evidence:
    return make_evidence(
        kind=kind,  # type: ignore[arg-type]
        source="repo:o/p@" + "a" * 40,
        locator=locator,
        content={"n": n},
        confidence="derived",
        authority="llm",
    )


CONFIGURE = [_ev("repo_file", "inventory"), _ev("readme", "README.md")]
A = [_ev("env_spec", "subtask:A"), CONFIGURE[0]]  # overlaps with configure
B = [_ev("entrypoints", "subtask:B")]
C = [_ev("workflow", "subtask:C"), _ev("repo_file", "inventory", 1)]  # same locator, new id


def test_merge_evidence_union_dedupes_by_id() -> None:
    merged = merge_evidence(CONFIGURE, A)
    assert [e.evidence_id for e in merged].count(CONFIGURE[0].evidence_id) == 1
    assert len(merged) == 3


def test_merge_evidence_sorted_by_kind_locator_id() -> None:
    merged = reduce(merge_evidence, [CONFIGURE, A, B, C], [])
    keys = [(e.kind, e.locator, e.evidence_id) for e in merged]
    assert keys == sorted(keys)
    assert [e.kind for e in merged] == [
        "entrypoints",
        "env_spec",
        "readme",
        "repo_file",
        "repo_file",
        "workflow",
    ]


def test_merge_evidence_order_independent_across_branches() -> None:
    results = {
        tuple(e.evidence_id for e in reduce(merge_evidence, perm, list(CONFIGURE)))
        for perm in itertools.permutations([A, B, C])
    }
    assert len(results) == 1  # A,B,C == C,B,A == ...


def test_merge_evidence_handles_none_and_empty() -> None:
    assert merge_evidence(None, None) == []
    assert merge_evidence([], B) == B
    assert merge_evidence(B, None) == B


OPEN_PY = UnknownOpened(unknown_id="unk:python_version", name="python_version", evidence_id="e1")
OPEN_DS = UnknownOpened(unknown_id="unk:dataset_source", name="dataset_source", evidence_id="e2")
RESOLVED = UnknownResolved(unknown_id="unk:python_version", value="3.8", evidence_id="e3")


def test_merge_events_appends_and_drops_exact_duplicates() -> None:
    merged = merge_events([OPEN_PY], [OPEN_PY, OPEN_DS, RESOLVED, OPEN_DS])
    assert merged == [OPEN_PY, OPEN_DS, RESOLVED]


def test_merge_events_keeps_same_unknown_with_different_fields() -> None:
    other = UnknownOpened(unknown_id="unk:python_version", name="python_version", evidence_id="x")
    assert merge_events([OPEN_PY], [other]) == [OPEN_PY, other]  # not an exact duplicate


def test_merge_events_same_set_in_any_branch_order() -> None:
    branches = [[OPEN_PY], [], [OPEN_DS, OPEN_PY]]
    results = [reduce(merge_events, perm, []) for perm in itertools.permutations(branches)]
    assert all(set(map(repr, r)) == {repr(OPEN_PY), repr(OPEN_DS)} for r in results)
    assert all(len(r) == 2 for r in results)
    assert merge_events(None, None) == []


def test_outcomes_reducer_concatenates() -> None:
    a = SubtaskSummary(kind="A", ok=True, attempts=1, tokens=10, latency_s=0.1)
    c = SubtaskSummary(kind="C", ok=False, attempts=3, tokens=30, latency_s=0.3, errors=("x",))
    assert operator.add([a], [c]) == [a, c]
