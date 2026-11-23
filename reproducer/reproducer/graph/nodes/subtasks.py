"""Subtasks node: deterministic repo-fact extractors.

Pure functions (text in, models out). Repo code is NEVER executed, imported or eval'd:
Python is read with `ast`, YAML with `yaml.safe_load`, TOML with tomllib/tomli, JSON with
`json`. The only IO is `collect_texts`, which reads files through providers/git.read_text.
These facts feed the LLM subtasks later; anything a parser cannot handle is reported,
never dropped silently.
"""

from __future__ import annotations

import ast
import configparser
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.inventory import RepoInventory
from reproducer.graph.nodes.configure import MAX_CONFIG_BYTES

if sys.version_info >= (3, 11):  # pragma: no cover - project targets 3.10
    import tomllib
else:
    import tomli as tomllib


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError)


# --------------------------------------------------------------------------- requirements.txt


class Requirement(_Frozen):
    name: str
    extras: tuple[str, ...] = ()
    spec: str | None = None  # e.g. ">=1.0,<2" (whitespace removed)
    marker: str | None = None  # e.g. 'python_version < "3.8"'
    raw: str
    line_no: int


class Include(_Frozen):
    kind: Literal["requirement", "constraint"]
    path: str
    raw: str
    line_no: int


class UnsupportedLine(_Frozen):
    raw: str
    reason: str  # "url" | "editable" | "path" | "option" | "unparsable"
    line_no: int


class RequirementsFacts(_Frozen):
    requirements: tuple[Requirement, ...]
    includes: tuple[Include, ...]
    unsupported: tuple[UnsupportedLine, ...]


_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_REQ_RE = re.compile(
    rf"^(?P<name>{_NAME})\s*(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>[^;]*?)\s*"
    r"(?:;\s*(?P<marker>.*\S))?\s*$"
)
_CLAUSE = r"\s*(?:===|~=|==|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+\s*"
_SPEC_RE = re.compile(rf"^\(?{_CLAUSE}(?:,{_CLAUSE})*\)?$")
_INCLUDE_RE = re.compile(
    r"^(?P<flag>-r|--requirement|-c|--constraint)(?:\s*=\s*|\s+|(?=\S))(?P<path>\S+)$"
)
_VCS_PREFIXES = ("git+", "hg+", "svn+", "bzr+")
_ARCHIVE_SUFFIXES = (".whl", ".tar.gz", ".tgz", ".zip", ".tar.bz2")


def _strip_comment(line: str) -> str:
    """pip rule: '#' starts a comment at line start or after whitespace."""
    if line.lstrip().startswith("#"):
        return ""
    return re.split(r"\s#", line, maxsplit=1)[0].strip()


def _logical_lines(text: str) -> Iterable[tuple[int, str]]:
    """Join backslash continuations; yield (first physical line number, stripped line)."""
    buf: list[str] = []
    start = 0
    for i, physical in enumerate(text.splitlines(), start=1):
        if not buf:
            start = i
        if physical.endswith("\\"):
            buf.append(physical[:-1])
            continue
        buf.append(physical)
        yield start, _strip_comment(" ".join(buf))
        buf = []
    if buf:
        yield start, _strip_comment(" ".join(buf))


def _parse_requirement(line: str, line_no: int) -> Requirement | UnsupportedLine:
    def bad(reason: str) -> UnsupportedLine:
        return UnsupportedLine(raw=line, reason=reason, line_no=line_no)

    if line.startswith(_VCS_PREFIXES) or "://" in line or re.search(r"\s@\s|^\S+@\s", line):
        return bad("url")
    if line.startswith((".", "/", "~")) or line.split(";")[0].strip().endswith(_ARCHIVE_SUFFIXES):
        return bad("path")
    req_part = re.split(r"\s--", line, maxsplit=1)[0]  # drop per-line options like --hash
    m = _REQ_RE.match(req_part)
    if not m:
        return bad("unparsable")
    spec = m.group("spec").strip()
    if spec and not _SPEC_RE.match(spec):
        return bad("unparsable")
    extras = tuple(e.strip() for e in (m.group("extras") or "").split(",") if e.strip())
    return Requirement(
        name=m.group("name"),
        extras=extras,
        spec=re.sub(r"[\s()]", "", spec) or None,
        marker=m.group("marker"),
        raw=line,
        line_no=line_no,
    )


def parse_requirements_txt(text: str) -> RequirementsFacts:
    """pip requirements file -> requirements, -r/-c includes, and unsupported lines."""
    reqs: list[Requirement] = []
    includes: list[Include] = []
    unsupported: list[UnsupportedLine] = []
    for line_no, line in _logical_lines(text):
        if not line:
            continue
        inc = _INCLUDE_RE.match(line)
        if inc:
            kind: Literal["requirement", "constraint"] = (
                "constraint" if inc.group("flag") in ("-c", "--constraint") else "requirement"
            )
            includes.append(Include(kind=kind, path=inc.group("path"), raw=line, line_no=line_no))
        elif line.startswith(("-e ", "--editable")):
            unsupported.append(UnsupportedLine(raw=line, reason="editable", line_no=line_no))
        elif line.startswith("-"):
            unsupported.append(UnsupportedLine(raw=line, reason="option", line_no=line_no))
        else:
            parsed = _parse_requirement(line, line_no)
            if isinstance(parsed, Requirement):
                reqs.append(parsed)
            else:
                unsupported.append(parsed)
    return RequirementsFacts(
        requirements=tuple(reqs), includes=tuple(includes), unsupported=tuple(unsupported)
    )


# --------------------------------------------------------------------------- setup.py


class SetupPyFacts(_Frozen):
    install_requires: tuple[str, ...] | None = None
    python_requires: str | None = None
    extras_require: dict[str, tuple[str, ...]] | None = None
    unresolved: tuple[str, ...] = ()  # "unresolved: <field>"
    error: str | None = None


def _str_list(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return tuple(s.strip() for s in value.splitlines() if s.strip())
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    return None


def _str_list_dict(value: Any) -> dict[str, tuple[str, ...]] | None:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        return None
    out: dict[str, tuple[str, ...]] = {}
    for k, v in value.items():
        items = _str_list(v)
        if items is None:
            return None
        out[k] = items
    return out


def _is_setup_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "setup") or (
        isinstance(f, ast.Attribute) and f.attr == "setup"
    )


_MUTATORS = frozenset(
    {"append", "extend", "insert", "remove", "pop", "clear", "update", "setdefault", "sort"}
)


def _single_literal_binding(tree: ast.Module, name: str, before_line: int) -> ast.expr | None:
    """Value node of `name = <expr>` if that is the name's ONLY binding in the module, is a
    plain top-level statement (not inside if/try/for/def), precedes the setup() call, and the
    name is never mutated in place (REQS.append(...)). Otherwise None. ast only."""
    stores = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
            stores += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores += node.name == name
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            stores += sum((a.asname or a.name.split(".")[0]) == name for a in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            stores += 2 * (name in node.names)  # rebinding from elsewhere: give up
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and node.func.attr in _MUTATORS
        ):
            return None
    if stores != 1:
        return None
    for stmt in tree.body:
        if stmt.lineno >= before_line:
            break
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            return stmt.value
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == name
            and stmt.value is not None
        ):
            return stmt.value
    return None


def parse_setup_py(text: str) -> SetupPyFacts:
    """Read setup() keyword literals via ast. Never executes the file."""
    try:
        tree = ast.parse(text)
    except _PARSE_ERRORS as e:
        return SetupPyFacts(error=f"{type(e).__name__}: {e}")
    calls = [n for n in ast.walk(tree) if _is_setup_call(n)]
    if not calls:
        return SetupPyFacts(error="no setup() call found")
    call = calls[0]
    assert isinstance(call, ast.Call)
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
    unresolved: list[str] = []
    if any(k.arg is None for k in call.keywords):
        unresolved.append("unresolved: **kwargs")

    def literal(field: str, convert: Any) -> Any:
        if field not in kw:
            return None
        node = kw[field]
        if isinstance(node, ast.Name):
            node = _single_literal_binding(tree, node.id, before_line=call.lineno)
        try:
            if node is None:
                raise ValueError("unresolvable name")
            value = convert(ast.literal_eval(node))
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            value = None
        if value is None:
            unresolved.append(f"unresolved: {field}")
        return value

    return SetupPyFacts(
        install_requires=literal("install_requires", _str_list),
        python_requires=literal("python_requires", lambda v: v if isinstance(v, str) else None),
        extras_require=literal("extras_require", _str_list_dict),
        unresolved=tuple(unresolved),
    )


# --------------------------------------------------------------------------- setup.cfg


def _cfg_list(value: str) -> tuple[str, ...]:
    """setuptools requirement list: one per line (';' starts a marker), comments dropped."""
    return tuple(s for s in (_strip_comment(line) for line in value.splitlines()) if s)


def parse_setup_cfg(text: str) -> SetupPyFacts:
    """[options] install_requires / python_requires and [options.extras_require] (configparser,
    no interpolation). `file:`/`attr:` directives are reported as unresolved, never followed."""
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text)
    except configparser.Error as e:
        return SetupPyFacts(error=f"{type(e).__name__}: {e}")
    unresolved: list[str] = []

    def directive(field: str, value: str) -> bool:
        if value.strip().startswith(("file:", "attr:")):
            unresolved.append(f"unresolved: {field}")
            return True
        return False

    install: tuple[str, ...] | None = None
    python: str | None = None
    if cp.has_section("options"):
        opts = cp["options"]
        if "install_requires" in opts and not directive(
            "install_requires", opts["install_requires"]
        ):
            install = _cfg_list(opts["install_requires"])
        if "python_requires" in opts and not directive("python_requires", opts["python_requires"]):
            python = opts["python_requires"].strip() or None
    extras: dict[str, tuple[str, ...]] | None = None
    if cp.has_section("options.extras_require"):
        extras = {}
        for k, v in cp["options.extras_require"].items():
            if not directive(f"extras_require.{k}", v):
                extras[k] = _cfg_list(v)
    return SetupPyFacts(
        install_requires=install,
        python_requires=python,
        extras_require=extras,
        unresolved=tuple(unresolved),
    )


# --------------------------------------------------------------------------- Pipfile


class PipfileFacts(_Frozen):
    packages: dict[str, str | None] = {}  # name -> spec ("*" -> None)
    dev_packages: dict[str, str | None] = {}
    python_version: str | None = None  # [requires] python_version or python_full_version
    unsupported: tuple[str, ...] = ()  # git/path/file/editable entries, non-string specs
    error: str | None = None


def _pipfile_section(table: Any, section: str, unsupported: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if table is None:
        return out
    if not isinstance(table, dict):
        unsupported.append(f"{section}: not a table")
        return out
    for name, spec in table.items():
        if isinstance(spec, str):
            out[name] = None if spec.strip() in ("", "*") else spec.strip()
        elif isinstance(spec, dict) and not (spec.keys() & {"git", "path", "file", "editable"}):
            v = spec.get("version")
            out[name] = None if not isinstance(v, str) or v.strip() == "*" else v.strip()
        else:
            unsupported.append(f"{section}.{name}: {json.dumps(spec, default=str, sort_keys=True)}")
    return out


def parse_pipfile(text: str) -> PipfileFacts:
    """Pipfile (TOML): [packages], [dev-packages], [requires] python_version."""
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as e:
        return PipfileFacts(error=f"{type(e).__name__}: {e}")
    unsupported: list[str] = []
    packages = _pipfile_section(data.get("packages"), "packages", unsupported)
    dev = _pipfile_section(data.get("dev-packages"), "dev-packages", unsupported)
    requires = data.get("requires")
    python: str | None = None
    if isinstance(requires, dict):
        v = requires.get("python_version", requires.get("python_full_version"))
        python = v if isinstance(v, str) else None
    return PipfileFacts(
        packages=packages,
        dev_packages=dev,
        python_version=python,
        unsupported=tuple(unsupported),
    )


# --------------------------------------------------------------------------- pyproject.toml


class PyprojectFacts(_Frozen):
    dependencies: tuple[str, ...] | None = None
    requires_python: str | None = None
    optional_dependencies: dict[str, tuple[str, ...]] | None = None
    notes: tuple[str, ...] = ()
    error: str | None = None


def parse_pyproject(text: str) -> PyprojectFacts:
    """[project] dependencies / requires-python / optional-dependencies (PEP 621)."""
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as e:
        return PyprojectFacts(error=f"{type(e).__name__}: {e}")
    notes: list[str] = []
    project = data.get("project")
    if not isinstance(project, dict):
        project = {}
        notes.append("no [project] table")
    dynamic = project.get("dynamic")
    if isinstance(dynamic, list):
        notes.extend(f"dynamic: {d}" for d in dynamic if isinstance(d, str))
    tool = data.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("poetry"), dict):
        notes.append("tool.poetry present (not parsed)")

    deps = _str_list(project["dependencies"]) if "dependencies" in project else None
    if "dependencies" in project and deps is None:
        notes.append("unresolved: dependencies")
    rp = project.get("requires-python")
    if rp is not None and not isinstance(rp, str):
        notes.append("unresolved: requires-python")
        rp = None
    opt = (
        _str_list_dict(project["optional-dependencies"])
        if "optional-dependencies" in project
        else None
    )
    if "optional-dependencies" in project and opt is None:
        notes.append("unresolved: optional-dependencies")
    return PyprojectFacts(
        dependencies=deps, requires_python=rp, optional_dependencies=opt, notes=tuple(notes)
    )


# --------------------------------------------------------------------------- environment.yml


class EnvironmentFacts(_Frozen):
    name: str | None = None
    channels: tuple[str, ...] = ()
    python: str | None = None  # version spec, e.g. "3.8" or ">=3.7"
    conda_dependencies: tuple[str, ...] = ()
    pip_dependencies: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    error: str | None = None


_CONDA_NAME_RE = re.compile(r"^(?:[^:\s]+::)?(?P<name>[A-Za-z0-9_.-]+)\s*(?P<spec>.*)$")


def parse_environment_yml(text: str) -> EnvironmentFacts:
    """conda environment file: python version, conda deps, pip deps (yaml.safe_load only)."""
    try:
        data = yaml.safe_load(text)
    except (yaml.YAMLError, RecursionError) as e:
        return EnvironmentFacts(error=f"{type(e).__name__}: {e}")
    if not isinstance(data, dict):
        return EnvironmentFacts(error=f"top level is {type(data).__name__}, expected mapping")
    unsupported: list[str] = []
    channels = _str_list(data.get("channels") or []) or ()
    conda: list[str] = []
    pip: list[str] = []
    python: str | None = None
    deps = data.get("dependencies") or []
    if not isinstance(deps, list):
        deps = []
        unsupported.append("dependencies is not a list")
    for d in deps:
        if isinstance(d, (str, int, float)) and not isinstance(d, bool):
            d = str(d)
            conda.append(d)
            m = _CONDA_NAME_RE.match(d.strip())
            if m and m.group("name").lower() == "python" and python is None:
                spec = m.group("spec").strip()
                python = spec[1:] if spec.startswith("=") and not spec.startswith("==") else spec
                python = python or None
        elif isinstance(d, dict) and set(d) == {"pip"} and _str_list(d["pip"] or []) is not None:
            pip.extend(_str_list(d["pip"] or []) or ())
        else:
            unsupported.append(json.dumps(d, default=str, sort_keys=True))
    name = data.get("name")
    return EnvironmentFacts(
        name=name if isinstance(name, str) else None,
        channels=channels,
        python=python,
        conda_dependencies=tuple(conda),
        pip_dependencies=tuple(pip),
        unsupported=tuple(unsupported),
    )


# --------------------------------------------------------------------------- config keys

CONFIG_KEY_SUFFIXES = (".yaml", ".yml", ".json", ".toml")
MAX_CONFIG_KEYS = 10_000
_MAX_DEPTH = 64


class _TooBig(Exception):
    pass


def _flatten(obj: Any, prefix: str, out: set[str], depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise _TooBig(f"nesting deeper than {_MAX_DEPTH}")
    if len(out) > MAX_CONFIG_KEYS:
        raise _TooBig(f"more than {MAX_CONFIG_KEYS} keys")
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        if prefix:
            out.add(prefix)  # the list itself is overridable as a whole
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                _flatten(v, f"{prefix}.{i}" if prefix else str(i), out, depth + 1)
    elif prefix:
        out.add(prefix)


def config_keys(text: str, suffix: str) -> frozenset[str] | str:
    """Flattened dotted keys ("model.lr", "layers.0.dim") or an error string. Never raises."""
    suffix = suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            data = tomllib.loads(text)
        else:
            return f"unsupported config suffix: {suffix!r}"
        out: set[str] = set()
        _flatten(data, "", out, 0)
    except (yaml.YAMLError, ValueError, RecursionError, MemoryError, _TooBig) as e:
        return f"{type(e).__name__}: {e}"
    return frozenset(out)


# --------------------------------------------------------------------------- entrypoints


class EntrypointCandidate(_Frozen):
    path: str
    has_main_guard: bool
    uses_argparse: bool
    uses_click: bool
    arg_names: tuple[str, ...]  # literal option/argument names, in source order


class EntrypointScan(_Frozen):
    candidates: tuple[EntrypointCandidate, ...]
    unparsable: dict[str, str]  # path -> error (e.g. Python 2 syntax)


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    t = node.test
    if len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq):
        return False
    sides = [t.left, t.comparators[0]]
    has_name = any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
    has_main = any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return has_name and has_main


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _scan_module(path: str, tree: ast.Module) -> EntrypointCandidate | None:
    has_main = any(_is_main_guard(s) for s in tree.body)
    uses_argparse = uses_click = False
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            uses_click |= any(a.name.split(".")[0] == "click" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            uses_click |= node.module.split(".")[0] == "click"
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "ArgumentParser":
                uses_argparse = True
            elif name in ("add_argument", "option", "argument"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        if a.value not in names:
                            names.append(a.value)
    if not (has_main or uses_argparse or uses_click):
        return None
    return EntrypointCandidate(
        path=path,
        has_main_guard=has_main,
        uses_argparse=uses_argparse,
        uses_click=uses_click,
        arg_names=tuple(names),
    )


def candidate_entrypoints(texts: Mapping[str, str]) -> EntrypointScan:
    """Python files with a main guard, argparse or click; sorted by path. ast only."""
    candidates: list[EntrypointCandidate] = []
    errors: dict[str, str] = {}
    for path in sorted(texts):
        try:
            tree = ast.parse(texts[path])
        except _PARSE_ERRORS as e:
            errors[path] = f"{type(e).__name__}: {e}"
            continue
        c = _scan_module(path, tree)
        if c is not None:
            candidates.append(c)
    return EntrypointScan(candidates=tuple(candidates), unparsable=errors)


# --------------------------------------------------------------------------- context packing


class ContextSection(_Frozen):
    title: str
    text: str
    priority: int


_SEP = "\n\n"


def _block(title: str, text: str) -> str:
    return f"## {title}\n{text}"


def pack_context(sections: Iterable[ContextSection | Mapping[str, Any]], max_chars: int) -> str:
    """Deterministic prompt context: highest priority first (ties keep input order).

    A section that does not fit is cut with "[truncated N chars]"; sections that get no room
    at all are summarized by a final "[omitted K sections]" line. Never exceeds max_chars.
    """
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    secs = [s if isinstance(s, ContextSection) else ContextSection(**s) for s in sections]
    ordered = [s for _, s in sorted(enumerate(secs), key=lambda p: (-p[1].priority, p[0]))]
    full = _SEP.join(_block(s.title, s.text) for s in ordered)
    if len(full) <= max_chars:
        return full

    reserve = len(f"{_SEP}[omitted {len(ordered)} sections]")
    budget = max_chars - reserve
    parts: list[str] = []
    used = 0
    omitted = 0
    for s in ordered:
        sep = len(_SEP) if parts else 0
        room = budget - used - sep
        block = _block(s.title, s.text)
        if len(block) <= room:
            parts.append(block)
            used += sep + len(block)
            continue
        header = len(_block(s.title, ""))
        marker_max = len(f"\n[truncated {len(s.text)} chars]")
        keep = room - header - marker_max
        if keep > 0:
            cut = len(s.text) - keep
            parts.append(_block(s.title, s.text[:keep]) + f"\n[truncated {cut} chars]")
            used += sep + len(parts[-1])
            budget = used  # nothing after a truncated section
        else:
            omitted += 1
    out = _SEP.join(parts)
    if omitted:
        out += f"{_SEP if out else ''}[omitted {omitted} sections]"
    return out[:max_chars]


# --------------------------------------------------------------------------- evidence


def _jsonable(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, (frozenset, set)):
        return sorted(result)
    if isinstance(result, Mapping):
        return {k: _jsonable(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_jsonable(v) for v in result]
    return result


def file_evidence(source: str, path: str, text: str, truncated: bool) -> Evidence:
    """Observed evidence for a repo file; same content shape as configure.inventory_to_evidence,
    so a requirement file read the same way gets the same id (deduped by the store)."""
    content = {
        "path": path,
        "text": text,
        "truncated": truncated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return make_evidence(
        kind="repo_file",
        source=source,
        locator=path,
        content=content,
        confidence="observed",
        authority="repo",
    )


DIGEST_HEAD_CHARS = 20_000


def digest_evidence(source: str, path: str, text: str, truncated: bool, size: int) -> Evidence:
    """Observed evidence for a config or Python file without storing it whole:
    {path, size (bytes on disk), sha256 (of the text read), head (first 20_000 chars),
    truncated (head is not the whole file: read was capped or text longer than head)}."""
    content = {
        "path": path,
        "size": size,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "head": text[:DIGEST_HEAD_CHARS],
        "truncated": truncated or len(text) > DIGEST_HEAD_CHARS,
    }
    return make_evidence(
        kind="repo_file",
        source=source,
        locator=path,
        content=content,
        confidence="observed",
        authority="repo",
    )


def facts_to_evidence(
    *, source: str, path: str, parser: str, result: Any, source_evidence_id: str
) -> Evidence:
    """Derived evidence for parsed facts, linked to the observed evidence of the source file."""
    return make_evidence(
        kind="repo_file",
        source=source,
        locator=f"{path}#parsed",
        content={"parser": parser, "result": _jsonable(result)},
        confidence="derived",
        authority="repo",
        links=(source_evidence_id,),
    )


def _requirement_parser(path: str) -> str | None:
    name = path.rsplit("/", 1)[-1].lower()
    if name.startswith("requirements") and name.endswith(".txt"):
        return "requirements_txt"
    if name == "setup.py":
        return "setup_py"
    if name == "setup.cfg":
        return "setup_cfg"
    if name == "pipfile":
        return "pipfile"
    if name == "pyproject.toml":
        return "pyproject"
    if name in ("environment.yml", "environment.yaml"):
        return "environment_yml"
    return None


_REQ_PARSERS: dict[str, Any] = {
    "requirements_txt": parse_requirements_txt,
    "setup_py": parse_setup_py,
    "setup_cfg": parse_setup_cfg,
    "pipfile": parse_pipfile,
    "pyproject": parse_pyproject,
    "environment_yml": parse_environment_yml,
}


def extract_repo_facts(
    inv: RepoInventory, texts: Mapping[str, tuple[str, bool]], source: str
) -> tuple[Evidence, ...]:
    """Run every extractor over texts (path -> (text, truncated)); pure.

    Returns observed evidence for each parsed source file followed by its derived evidence.
    Requirement files keep full-text evidence (same ids as inventory_to_evidence); configs
    and Python files get digest evidence. Files under .github/ and files without a parser
    (.cfg/.ini configs) are skipped.
    """
    out: list[Evidence] = []
    sizes = {f.path: f.size for f in inv.files}
    texts = {p: t for p, t in texts.items() if not is_skipped_path(p)}

    def emit(path: str, parser: str, result: Any, full_text: bool = False) -> None:
        text, truncated = texts[path]
        if full_text:
            src = file_evidence(source, path, text, truncated)
        else:
            size = sizes.get(path, len(text.encode("utf-8")))
            src = digest_evidence(source, path, text, truncated, size)
        out.append(src)
        out.append(
            facts_to_evidence(
                source=source,
                path=path,
                parser=parser,
                result=result,
                source_evidence_id=src.evidence_id,
            )
        )

    for path in inv.requirement_files:
        parser = _requirement_parser(path)
        if parser is not None and path in texts:
            emit(path, parser, _REQ_PARSERS[parser](texts[path][0]), full_text=True)
    for path in inv.config_files:
        suffix = Path(path).suffix.lower()
        if suffix in CONFIG_KEY_SUFFIXES and path in texts:
            keys = config_keys(texts[path][0], suffix)
            emit(path, "config_keys", {"error": keys} if isinstance(keys, str) else {"keys": keys})
    py = {p: texts[p][0] for p in inv.python_files if p in texts}
    scan = candidate_entrypoints(py)
    for c in scan.candidates:
        emit(c.path, "entrypoint", c)
    for path, err in sorted(scan.unparsable.items()):
        emit(path, "entrypoint", {"error": err})
    return tuple(out)


# --------------------------------------------------------------------------- IO helper

_ENTRY_WORDS = frozenset({"train", "eval", "evaluate", "test", "main", "run"})
_SKIP_DIRS = frozenset({".github"})


def is_skipped_path(path: str) -> bool:
    """CI/tooling folders (e.g. .github/ workflows) are not repo facts for reproduction."""
    return any(part in _SKIP_DIRS for part in path.split("/")[:-1])


def _py_priority(path: str) -> tuple[int, int, str]:
    stem = Path(path).stem.lower()
    named = bool(_ENTRY_WORDS & set(re.split(r"[_\-.]", stem)))
    return (0 if named else 1, path.count("/"), path)


def collect_texts(
    repo_dir: Path, inventory: RepoInventory, max_files: int = 50
) -> dict[str, tuple[str, bool]]:
    """Read requirement files, configs (<= 1 MB) and up to max_files Python files.

    Python files are prioritized by entrypoint-like names (train/eval/test/main/run), then
    by depth. Anything under .github/ and unreadable/unsafe paths are skipped (read_text
    refuses repo escapes).
    """
    from reproducer.providers.git import read_text  # IO stays out of the pure functions

    sizes = {f.path: f.size for f in inventory.files if not f.is_symlink}
    wanted: list[tuple[str, int]] = [(p, 50_000) for p in inventory.requirement_files]
    wanted += [
        (p, MAX_CONFIG_BYTES) for p in inventory.config_files if sizes.get(p, 0) <= MAX_CONFIG_BYTES
    ]
    py = sorted(
        (p for p in inventory.python_files if p in sizes and not is_skipped_path(p)),
        key=_py_priority,
    )
    wanted += [(p, MAX_CONFIG_BYTES) for p in py[: max(0, max_files)]]
    out: dict[str, tuple[str, bool]] = {}
    for path, limit in wanted:
        if path in out or is_skipped_path(path):
            continue
        try:
            out[path] = read_text(repo_dir, path, max_bytes=limit)
        except (ValueError, OSError):
            continue
    return out
