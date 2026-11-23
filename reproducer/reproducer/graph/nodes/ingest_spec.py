"""ingest_spec node: the paper side of the configure graph.

Loads a ReproSpec JSON (stand-in for the ingestion module's output), turns it into
paper_claim / paper_setup evidence plus UnknownOpened events for null setup values, and
picks the repo to configure from its code_links. Reads only the spec file; writes nothing
(persist is the single writer). Without state["spec_path"] it is a no-op (URL-only entry).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.unknowns import spec_to_evidence
from reproducer.graph.nodes.configure import parse_github_url


def paper_source_for(spec_path: str | Path) -> str:
    """evals/cutout/spec.json -> 'paper:cutout' (the spec file's parent folder name)."""
    return f"paper:{Path(spec_path).resolve().parent.name}"


def first_github_link(code_links: tuple[str, ...]) -> str | None:
    for link in code_links:
        try:
            parse_github_url(link)
        except ValueError:
            continue
        return link.strip()
    return None


def _failed(reason: str) -> dict[str, Any]:
    return {"status": "configure_failed", "failure_reason": reason}


def ingest_spec(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Graph node. Invalid/unreadable spec or no GitHub code_link -> status
    configure_failed with failure_reason (the graph then ends: nothing is cloned)."""
    spec_path = state.get("spec_path")
    if not spec_path:
        return {}
    try:
        raw = Path(spec_path).read_text(encoding="utf-8")
    except OSError as e:
        return _failed(f"cannot read spec {spec_path}: {e}")
    try:
        spec = ReproSpec.model_validate_json(raw)
    except ValidationError as e:
        return _failed(f"invalid ReproSpec {spec_path}: {e}")
    url = first_github_link(spec.code_links)
    if url is None:
        return _failed(
            f"no GitHub repository in code_links {list(spec.code_links)} "
            "(only GitHub is supported for now)"
        )
    source = paper_source_for(spec_path)
    evidence, opened = spec_to_evidence(spec, source)
    return {
        "spec": spec,
        "paper_source": source,
        "repo_url": url,
        "evidence": list(evidence),
        "events": list(opened),
    }
