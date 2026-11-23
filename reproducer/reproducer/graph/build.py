"""Graph construction.

Configure graph: START -> ingest_spec -> configure -> {subtask_a, subtask_b, subtask_c}
(parallel) -> persist (runs once, after all three) -> END. ingest_spec is a no-op without
spec_path; if it fails (status configure_failed) the graph ends there, before any clone.
Only persist writes evidence/events; the subtask branches write LLM telemetry through
MeteredLLM.on_record (lock-safe appends).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from reproducer.graph.nodes.configure import configure
from reproducer.graph.nodes.ingest_spec import ingest_spec
from reproducer.graph.nodes.persist import persist
from reproducer.graph.nodes.subtasks import subtask_a, subtask_b, subtask_c
from reproducer.graph.state import ConfigureDeps, ConfigureState

SUBTASK_NODES = {"subtask_a": subtask_a, "subtask_b": subtask_b, "subtask_c": subtask_c}


def _after_ingest(state: Mapping[str, Any]) -> str:
    return END if state.get("status") == "configure_failed" else "configure"


def build_configure_graph(deps: ConfigureDeps) -> Any:
    """Compiled graph; call .invoke(initial_state) with the ConfigureState inputs."""
    graph = StateGraph(ConfigureState)
    graph.add_node("ingest_spec", partial(ingest_spec, deps=deps))
    graph.add_node("configure", partial(configure, deps=deps))
    for name, fn in SUBTASK_NODES.items():
        graph.add_node(name, partial(fn, deps=deps))
    graph.add_node("persist", partial(persist, deps=deps))
    graph.add_edge(START, "ingest_spec")
    graph.add_conditional_edges("ingest_spec", _after_ingest, ["configure", END])
    for name in SUBTASK_NODES:
        graph.add_edge("configure", name)
    graph.add_edge(list(SUBTASK_NODES), "persist")  # fan-in: waits for all three
    graph.add_edge("persist", END)
    return graph.compile()
