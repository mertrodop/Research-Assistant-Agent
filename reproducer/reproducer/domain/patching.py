"""Line edits on repo text, unified diffs and error signatures for the grounded critic.

Pure domain: text in, text out. Nothing here executes repo code (ast.parse only).

  split_lines / apply_line_edits  lines are split on "\\n" only (git's notion); an edit
                                  replaces lines start..end (1-based, inclusive)
  unified_diff                    git-apply compatible diff (a/<file> b/<file>)
  diff_changed_lines              removed + added lines of a unified diff
  value_literals                  multiset of numeric literals of a module, EXCEPT subscript
                                  indices / slice bounds (x[0], x[:, 1]) and INTEGER axis
                                  literals: the value of a dim= / axis= keyword and positional
                                  args of shape methods (.view .reshape .squeeze .unsqueeze
                                  .permute .transpose .flatten .size). These address tensor
                                  axes, they are not hyperparameters. Floats are never exempt.
  error_signature                 exc_type | normalized message | repo frame file:line
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from reproducer.domain.traceback_parse import WORK, is_repo_file


class Edit(Protocol):
    start_line: int
    end_line: int
    original: str
    replacement: str


def split_lines(text: str) -> list[str]:
    """Lines with their "\\n" kept; the last one may lack it. '' -> []."""
    if not text:
        return []
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def line_body(line: str) -> str:
    """A line without its line ending ("\\n" or "\\r\\n")."""
    return line[:-1].removesuffix("\r") if line.endswith("\n") else line


def range_text(lines: Sequence[str], start: int, end: int) -> str:
    """Lines start..end (1-based, inclusive) joined with "\\n", without the final ending."""
    return "\n".join(line_body(ln) for ln in lines[start - 1 : end])


def _replacement_lines(replacement: str, eol: str, last_has_eol: bool) -> list[str]:
    if replacement == "":
        return []
    body = replacement[:-1] if replacement.endswith("\n") else replacement
    out = [ln + eol for ln in body.split("\n")]
    if not last_has_eol:
        out[-1] = out[-1][: -len(eol)]
    return out


def apply_line_edits(text: str, edits: Sequence[Edit]) -> str:
    """Apply non-overlapping edits (checked by the caller) bottom-up. An empty replacement
    deletes the lines. Line endings follow the replaced range's first line."""
    lines = split_lines(text)
    for e in sorted(edits, key=lambda e: e.start_line, reverse=True):
        first = lines[e.start_line - 1]
        eol = "\r\n" if first.endswith("\r\n") else "\n"
        last_has_eol = lines[e.end_line - 1].endswith("\n")
        lines[e.start_line - 1 : e.end_line] = _replacement_lines(e.replacement, eol, last_has_eol)
    return "".join(lines)


_NO_EOL = "\\ No newline at end of file\n"


def unified_diff(path: str, before: str, after: str, context: int = 3) -> str:
    """Unified diff of path for `git apply` (-p1). '' when before == after."""
    out: list[str] = []
    for ln in difflib.unified_diff(
        split_lines(before), split_lines(after), f"a/{path}", f"b/{path}", n=context
    ):
        if ln.startswith(("--- ", "+++ ")):
            out.append(ln.rstrip("\n") + "\n")
        elif ln.endswith("\n"):
            out.append(ln)
        else:
            out.append(ln + "\n" + _NO_EOL)
    return "".join(out)


def diff_changed_lines(diff: str) -> int:
    """Removed + added lines (headers and context lines excluded)."""
    return sum(
        1
        for ln in diff.split("\n")
        if ln[:1] in "+-" and ln and not ln.startswith(("+++ ", "--- "))
    )


def diff_sha256(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _index_constants(tree: ast.AST) -> set[int]:
    """ids of numeric Constants used as a subscript index or slice bound."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        stack: list[ast.AST] = [node.slice]
        while stack:
            s = stack.pop()
            if isinstance(s, ast.Constant):
                out.add(id(s))
            elif isinstance(s, ast.UnaryOp) and isinstance(s.op, (ast.USub, ast.UAdd)):
                stack.append(s.operand)
            elif isinstance(s, ast.Tuple):
                stack.extend(s.elts)
            elif isinstance(s, ast.Slice):
                stack.extend(x for x in (s.lower, s.upper, s.step) if x is not None)
    return out


AXIS_KEYWORDS = frozenset({"dim", "axis"})
SHAPE_METHODS = frozenset(
    {"view", "reshape", "squeeze", "unsqueeze", "permute", "transpose", "flatten", "size"}
)


def _int_constants(node: ast.AST) -> list[ast.Constant]:
    """Int Constants of an axis argument: 1, -1, (0, 2, 1), [0, -1]."""
    stack, out = [node], []
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Constant) and type(n.value) is int:
            out.append(n)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            stack.append(n.operand)
        elif isinstance(n, (ast.Tuple, ast.List)):
            stack.extend(n.elts)
    return out


def _axis_constants(tree: ast.AST) -> set[int]:
    """ids of int Constants used as dim= / axis= values or positional args of shape methods."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        args: list[ast.AST] = [k.value for k in node.keywords if k.arg in AXIS_KEYWORDS]
        if isinstance(node.func, ast.Attribute) and node.func.attr in SHAPE_METHODS:
            args.extend(node.args)
        for a in args:
            out.update(id(c) for c in _int_constants(a))
    return out


def value_literals(tree: ast.AST) -> Counter[str]:
    """Numeric literals (int / float / complex, not bool) by repr; subscript indices and integer
    axis literals excluded (module docstring). A negative literal counts as its absolute value
    (unary minus is an operator)."""
    skip = _index_constants(tree) | _axis_constants(tree)
    return Counter(
        repr(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float, complex))
        and not isinstance(n.value, bool)
        and id(n) not in skip
    )


# --- error signature ---------------------------------------------------------------------

_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_DIGITS_RE = re.compile(r"\d+")


def normalize_message(message: str) -> str:
    """First line, hex addresses and digit runs masked, whitespace collapsed."""
    first = message.strip().split("\n", 1)[0]
    first = _HEX_RE.sub("<addr>", first)
    first = _DIGITS_RE.sub("N", first)
    return " ".join(first.split())[:300]


def repo_relative(path: str) -> str | None:
    """Repo-relative path of a traceback frame file in repo code (cwd /work), else None."""
    if not is_repo_file(path):
        return None
    if path.startswith(WORK + "/"):
        path = path[len(WORK) + 1 :]
    return path.removeprefix("./")


def error_signature(tb: Mapping[str, Any]) -> str:
    """exc_type | normalized message | repo frame file:line (parsed traceback as a dict)."""
    frame = tb.get("repo_frame") or {}
    where = ""
    if frame:
        file = str(frame.get("file", ""))
        where = f"{repo_relative(file) or file}:{frame.get('line')}"
    return f"{tb.get('exc_type', '')}|{normalize_message(str(tb.get('message', '')))}|{where}"
