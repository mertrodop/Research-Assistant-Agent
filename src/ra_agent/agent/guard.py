"""First line of defense: check a proposed tool call before anything runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
from langchain_core.tools import BaseTool

from ..config import PathSettings

# free-text arguments: slashes in prose are not paths
FREE_TEXT = {"text", "query"}


def _looks_like_path(value: str) -> bool:
    return any(mark in value for mark in ("/", "\\", ".pdf", ".."))


def _outside(value: str, paths: PathSettings) -> bool:
    path = Path(value.replace("\\", "/"))
    if len(path.parts) == 1:  # bare name: the workspace reads it from raw_dir
        path = Path(paths.raw_dir) / path
    resolved = path.resolve()
    roots = (paths.raw_dir, paths.vault_dir, paths.staging_dir)
    return not any(resolved.is_relative_to(Path(root).resolve()) for root in roots)


def validate_call(
    name: str, args: dict, tools: dict[str, BaseTool], paths: PathSettings
) -> str | None:
    """Return why the call must not run, or None."""
    tool = tools.get(name)
    if tool is None:
        return f"unknown tool {name!r}; available: {', '.join(sorted(tools))}"
    if not isinstance(args, dict):
        return f"{name} arguments must be an object"
    schema = tool.args_schema
    try:
        if isinstance(schema, dict):
            jsonschema.validate(args, schema)
        elif schema is not None:
            schema.model_validate(args)
    except Exception as exc:  # noqa: BLE001 - pydantic or jsonschema error, both readable
        return f"arguments do not match {name}: {exc}"[:300]
    for key, value in args.items():
        if key in FREE_TEXT or not isinstance(value, str) or not _looks_like_path(value):
            continue
        if _outside(value, paths):
            return f"{key}={value!r} is outside the workspace folders"
    return None


def signature(name: str, args: dict) -> str:
    return name + json.dumps(args, sort_keys=True, default=str)


@dataclass
class FailureReport:
    limit: str
    step: int
    paper: str
    tool: str
    evidence: str
    tried: list[str] = field(default_factory=list)

    def sentence(self) -> str:
        tried = ", ".join(self.tried) or "nothing"
        return (
            f"Stopped by {self.limit} at step {self.step} on {self.paper or 'no paper'}: "
            f"{self.evidence}. Tried: {tried}."
        )
