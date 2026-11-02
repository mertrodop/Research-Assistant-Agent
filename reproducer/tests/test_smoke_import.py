"""Smoke test: every scaffolded module imports."""

import importlib

import pytest

MODULES = [
    "reproducer",
    "reproducer.cli",
    "reproducer.server",
    "reproducer.domain",
    "reproducer.domain.evidence",
    "reproducer.domain.unknowns",
    "reproducer.domain.ledger",
    "reproducer.domain.contracts",
    "reproducer.graph",
    "reproducer.graph.state",
    "reproducer.graph.build",
    "reproducer.graph.nodes",
    "reproducer.graph.nodes.configure",
    "reproducer.graph.nodes.subtasks",
    "reproducer.graph.nodes.smoke",
    "reproducer.graph.nodes.orchestrator",
    "reproducer.graph.nodes.execute",
    "reproducer.graph.nodes.evaluator",
    "reproducer.graph.nodes.breaker",
    "reproducer.graph.nodes.reflect",
    "reproducer.providers",
    "reproducer.providers.git",
    "reproducer.providers.docker_runner",
    "reproducer.providers.llm",
    "reproducer.providers.vault",
    "reproducer.providers.store",
    "reproducer.rules",
    "reproducer.rules.known_fixes",
    "reproducer.rules.classify",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)
