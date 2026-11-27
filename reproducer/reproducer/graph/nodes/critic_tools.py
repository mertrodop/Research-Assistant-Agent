"""Read-only tools for the ReAct critic, on a VIRTUAL view of the repo.

The view = the host clone at the pinned sha + the current critic patch stack, folded in memory
(the same fold as critic.fold_text). Nothing here writes a file, executes repo code or follows
a path out of the repo: every path is normalized like RepoPath (grounding.normalize_repo_path)
and must name a listed repo file (or, for list_dir, a directory of one); reads go through
providers.git.read_text (symlink / .git escape refused). Every result is capped at
MAX_RESULT_CHARS (marked "[truncated ...]").

  read_file(path, start_line, end_line)  <= MAX_READ_LINES lines, '<n>| <code>' numbered,
                                         metric lines tagged [METRIC]
  grep(pattern, path_glob=None)          Python regex over the view, <= MAX_GREP_MATCHES
                                         'path:line:text' matches (pattern <= 200 chars).
                                         Guards: a GREP_DEADLINE_S deadline checked between
                                         files and every GREP_CHECK_EVERY lines; each line is
                                         matched on its first GREP_MATCH_LINE_CHARS chars; at
                                         most GREP_MAX_BYTES (UTF-8) scanned per call. A stop
                                         or truncation is noted at the TOP of the result (so
                                         the result cap never cuts it). A single pathological
                                         regex on one <= 2000-char line is not interruptible.
  list_dir(path)                         the entries of one directory ('dir/' for dirs)
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reproducer.domain.grounding import normalize_repo_path
from reproducer.domain.patching import Edit, apply_line_edits, line_body, split_lines
from reproducer.providers.git import UnsafePathError, read_text

MAX_READ_LINES = 200
MAX_GREP_MATCHES = 50
MAX_RESULT_CHARS = 2000
MAX_PATTERN_CHARS = 200
MAX_GREP_LINE_CHARS = 200
MAX_VIEW_FILE_BYTES = 1_000_000
GREP_DEADLINE_S = 5.0
GREP_CHECK_EVERY = 500  # lines between deadline checks inside a file
GREP_MATCH_LINE_CHARS = 2000  # a longer line is matched on its first part only
GREP_MAX_BYTES = 5_000_000  # total UTF-8 bytes scanned per grep call

READ_TOOLS = ("read_file", "grep", "list_dir")


class ToolError(ValueError):
    """A tool call the view refuses (bad path, bad args): its message is the tool result."""


class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReadFileArgs(_Args):
    path: str = Field(description="Repo-relative file path, e.g. 'train.py'.")
    start_line: int = Field(ge=1, description="First line (1-based).")
    end_line: int = Field(ge=1, description=f"Last line (inclusive); <= {MAX_READ_LINES} lines.")


class GrepArgs(_Args):
    pattern: str = Field(max_length=MAX_PATTERN_CHARS, description="Python regular expression.")
    path_glob: str | None = Field(
        default=None, description="Optional fnmatch glob on repo paths, e.g. '*.py' or 'models/*'."
    )


class ListDirArgs(_Args):
    path: str = Field(default="", description="Repo-relative directory; '' or '.' = repo root.")


ARG_MODELS: dict[str, type[_Args]] = {
    "read_file": ReadFileArgs,
    "grep": GrepArgs,
    "list_dir": ListDirArgs,
}
DESCRIPTIONS = {
    "read_file": f"Read lines of a repo file (current view: pinned commit + applied patches), "
    f"numbered '<n>| <code>'. At most {MAX_READ_LINES} lines per call.",
    "grep": f"Search the repo view with a Python regex; returns up to {MAX_GREP_MATCHES} "
    "'path:line:text' matches.",
    "list_dir": "List the entries of a repo directory ('name/' marks a subdirectory).",
}


@dataclass(frozen=True)
class ToolResult:
    text: str  # what the model sees (<= MAX_RESULT_CHARS)
    ok: bool
    chars: int  # length before truncation
    truncated: bool


def cap(text: str, limit: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    """text cut to <= limit chars, with a '[truncated ...]' marker inside the limit."""
    if len(text) <= limit:
        return text, False
    marker = f"\n[truncated: {len(text)} chars, showing the first part]"
    return text[: max(0, limit - len(marker))] + marker, True


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """A pydantic JSON schema with '#/$defs/...' refs inlined (function `parameters` must be
    a self-contained object schema for some providers)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return walk(defs[ref.split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def tool_spec(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _inline_refs(model.model_json_schema()),
        },
    }


def read_tool_specs() -> list[dict[str, Any]]:
    return [tool_spec(n, DESCRIPTIONS[n], ARG_MODELS[n]) for n in READ_TOOLS]


@dataclass
class RepoView:
    """Clone + patch stack, read lazily and cached; never written.

    stack: (file, edits) in application order. metric_line_texts: file -> metric line texts
    (tagged [METRIC] by text, so earlier patches' line shifts do not matter)."""

    clone_dir: Path
    repo_files: Set[str]
    stack: Sequence[tuple[str, Sequence[Edit]]] = ()
    metric_line_texts: Mapping[str, Sequence[str]] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic  # grep deadline
    _cache: dict[str, str | None] = field(default_factory=dict, repr=False)

    # --- paths -----------------------------------------------------------------------

    def file_path(self, path: str) -> str:
        try:
            norm = normalize_repo_path(path)
        except ValueError as e:
            raise ToolError(str(e)) from None
        if norm not in self.repo_files:
            raise ToolError(f"file not in repo: {norm!r}")
        return norm

    def dir_path(self, path: str) -> str:
        """'' for the root; else a normalized directory that contains a repo file."""
        raw = path.strip()
        if raw in ("", ".", "./"):
            return ""
        try:
            norm = normalize_repo_path(raw.rstrip("/") or ".")
        except ValueError as e:
            raise ToolError(str(e)) from None
        if not any(f.startswith(norm + "/") for f in self.repo_files):
            raise ToolError(f"not a directory in the repo: {norm!r}")
        return norm

    # --- content ---------------------------------------------------------------------

    def text(self, path: str) -> str | None:
        """The current text of a (validated) repo file; None if unreadable / binary / huge."""
        if path in self._cache:
            return self._cache[path]
        value: str | None
        try:
            raw, truncated = read_text(self.clone_dir, path, max_bytes=MAX_VIEW_FILE_BYTES)
        except (OSError, UnsafePathError):
            value = None
        else:
            value = None if truncated or "\x00" in raw else raw
        if value is not None:
            for file, edits in self.stack:
                if file == path:
                    value = apply_line_edits(value, edits)
        self._cache[path] = value
        return value

    def metric_lines(self, path: str, lines: Sequence[str]) -> frozenset[int]:
        wanted = {t.strip() for t in self.metric_line_texts.get(path, ()) if t.strip()}
        return frozenset(i for i, ln in enumerate(lines, 1) if ln.strip() in wanted)

    # --- tools -----------------------------------------------------------------------

    def read_file(self, args: ReadFileArgs) -> str:
        path = self.file_path(args.path)
        text = self.text(path)
        if text is None:
            raise ToolError(f"cannot read {path!r} (binary, unreadable or > 1 MB)")
        lines = split_lines(text)
        n = len(lines)
        if args.start_line > n:
            raise ToolError(f"start_line {args.start_line} beyond end of {path} ({n} lines)")
        if args.end_line < args.start_line:
            raise ToolError(f"end_line {args.end_line} < start_line {args.start_line}")
        end = min(args.end_line, n, args.start_line + MAX_READ_LINES - 1)
        marked = self.metric_lines(path, lines)
        width = len(str(end))
        out = [f"{path} lines {args.start_line}-{end} of {n}:"]
        for i in range(args.start_line, end + 1):
            tag = "[METRIC] " if i in marked else ""
            out.append(f"{tag}{i:>{width}}| {line_body(lines[i - 1])}")
        if end < min(args.end_line, n):
            out.append(f"(clamped to {MAX_READ_LINES} lines; read from {end + 1} for more)")
        return "\n".join(out)

    def grep(self, args: GrepArgs) -> str:
        try:
            rx = re.compile(args.pattern)
        except re.error as e:
            raise ToolError(f"invalid regex {args.pattern!r}: {e}") from None
        files = sorted(self.repo_files)
        if args.path_glob:
            files = [f for f in files if fnmatch.fnmatchcase(f, args.path_glob)]
        deadline = self.clock() + GREP_DEADLINE_S
        hits: list[str] = []
        stop: str | None = None
        scanned = long_lines = files_done = 0
        for f in files:
            if self.clock() > deadline:
                stop = f"deadline {GREP_DEADLINE_S:.0f}s reached after {files_done} file(s)"
                break
            text = self.text(f)
            if text is None:
                files_done += 1
                continue
            for i, ln in enumerate(split_lines(text), 1):
                if i % GREP_CHECK_EVERY == 0 and self.clock() > deadline:
                    stop = f"deadline {GREP_DEADLINE_S:.0f}s reached in {f} at line {i}"
                    break
                size = len(ln.encode("utf-8", "replace"))
                if scanned + size > GREP_MAX_BYTES:
                    stop = f"scan limit {GREP_MAX_BYTES} bytes reached in {f} at line {i}"
                    break
                scanned += size
                body = line_body(ln)
                if len(body) > GREP_MATCH_LINE_CHARS:
                    long_lines += 1
                    body = body[:GREP_MATCH_LINE_CHARS]
                if rx.search(body):
                    if len(hits) == MAX_GREP_MATCHES:
                        stop = f"stopped at {MAX_GREP_MATCHES} matches"
                        break
                    hits.append(f"{f}:{i}:{body.strip()[:MAX_GREP_LINE_CHARS]}")
            if stop is not None:
                break
            files_done += 1
        notes = [f"(incomplete: {stop})"] if stop else []
        if long_lines:
            notes.append(
                f"({long_lines} line(s) longer than {GREP_MATCH_LINE_CHARS} chars were "
                "matched on their first part only)"
            )
        if not hits:
            where = f" in {args.path_glob!r}" if args.path_glob else ""
            return "\n".join([*notes, f"no matches for {args.pattern!r}{where}"])
        return "\n".join([*notes, *hits])

    def list_dir(self, args: ListDirArgs) -> str:
        d = self.dir_path(args.path)
        prefix = d + "/" if d else ""
        entries: set[str] = set()
        for f in self.repo_files:
            if not f.startswith(prefix):
                continue
            head, sep, _ = f[len(prefix) :].partition("/")
            entries.add(head + "/" if sep else head)
        return "\n".join(sorted(entries)) or "(empty)"

    def call(self, name: str, arguments: str, limit: int = MAX_RESULT_CHARS) -> ToolResult:
        """Run one read tool on raw JSON arguments; refusals become ok=False results. The
        result is capped at min(limit, MAX_RESULT_CHARS) chars."""
        model = ARG_MODELS.get(name)
        try:
            if model is None:
                raise ToolError(f"unknown tool {name!r}; tools: {[*READ_TOOLS, 'submit_patch']}")
            try:
                args = model.model_validate_json(arguments or "{}")
            except ValidationError as e:
                errs = "; ".join(
                    f"{'.'.join(map(str, x['loc'])) or '(root)'}: {x['msg']}"
                    for x in e.errors(include_url=False)
                )
                raise ToolError(f"bad arguments for {name}: {errs}") from None
            fn: Callable[[Any], str] = getattr(self, name)
            raw, ok = fn(args), True
        except ToolError as e:
            raw, ok = f"error: {e}", False
        text, truncated = cap(raw, min(limit, MAX_RESULT_CHARS))
        return ToolResult(text=text, ok=ok, chars=len(raw), truncated=truncated)


def canonical_args(arguments: str) -> str:
    """Arguments as canonical JSON (loop detection); the raw string when not JSON."""
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        return arguments


def view_for(
    clone_dir: Path,
    repo_files: Set[str],
    stack: Iterable[Any],
    metric_line_texts: Mapping[str, Sequence[str]] | None = None,
) -> RepoView:
    """A RepoView over the clone and critic AppliedPatch-like objects (.file, .edits)."""
    return RepoView(
        clone_dir=clone_dir,
        repo_files=frozenset(repo_files),
        stack=tuple((p.file, p.edits) for p in stack),
        metric_line_texts=dict(metric_line_texts or {}),
    )


__all__ = [
    "ARG_MODELS",
    "MAX_GREP_MATCHES",
    "MAX_READ_LINES",
    "MAX_RESULT_CHARS",
    "READ_TOOLS",
    "RepoView",
    "ToolResult",
    "canonical_args",
    "cap",
    "read_tool_specs",
    "tool_spec",
    "view_for",
]
