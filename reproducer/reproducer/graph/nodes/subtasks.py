"""Subtasks node: deterministic repo-fact extractors + the gated LLM subtasks A/B/C.

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
import posixpath
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from reproducer.domain.contracts import EntrypointList, EnvSpec, WorkflowAnalysis
from reproducer.domain.evidence import Evidence, EvidenceKind, canonical_json, make_evidence
from reproducer.domain.grounding import CITATION_MAX_QUOTE_CHARS, Citation, normalize_repo_path
from reproducer.domain.inventory import RepoInventory
from reproducer.domain.unknowns import UnknownOpened, unknown_id_for
from reproducer.graph.eviction import call_with_eviction
from reproducer.graph.nodes.configure import MAX_CONFIG_BYTES
from reproducer.graph.state import SubtaskSummary
from reproducer.providers.llm import LLM, BudgetExceeded, LLMError

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


class ArgSpec(_Frozen):
    """One argparse add_argument call (ast only). Values that are not literals (or a
    module-level single literal binding such as `choices=dataset_options`) are None."""

    option_strings: tuple[str, ...]  # e.g. ("--dataset", "-d"); positional: ("data",)
    dest: str | None  # explicit dest=, else derived like argparse
    type: str | None = None  # source name of type=, e.g. "int"
    default: Any = None  # literal default (None if absent or not a literal)
    action: str | None = None  # literal action=, e.g. "store_true"
    choices: tuple[Any, ...] | None = None


class EntrypointCandidate(_Frozen):
    path: str
    has_main_guard: bool
    uses_argparse: bool
    uses_click: bool
    arg_names: tuple[str, ...]  # literal option/argument names, in source order
    arguments: tuple[ArgSpec, ...] = ()  # argparse spec per add_argument (additive)


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


def _literal(tree: ast.Module, node: ast.expr) -> tuple[bool, Any]:
    """(True, value) for a literal, or a Name with a single module-level literal binding."""
    if isinstance(node, ast.Name):
        bound = _single_literal_binding(tree, node.id, node.lineno)
        if bound is None:
            return False, None
        node = bound
    try:
        return True, ast.literal_eval(node)
    except _PARSE_ERRORS:
        return False, None


def _derived_dest(options: Sequence[str]) -> str | None:
    """argparse's rule: first long option, else first short option, else the positional
    name; leading dashes stripped and '-' -> '_'."""
    if not options:
        return None
    longs = [o for o in options if o.startswith("--")]
    shorts = [o for o in options if o.startswith("-") and not o.startswith("--")]
    pick = longs[0] if longs else (shorts[0] if shorts else options[0])
    return pick.lstrip("-").replace("-", "_") or None


def _arg_spec(tree: ast.Module, call: ast.Call) -> ArgSpec | None:
    options = tuple(
        a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
    )
    if not options:
        return None
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
    dest: str | None = _derived_dest(options)
    if "dest" in kw:
        ok, v = _literal(tree, kw["dest"])
        dest = v if ok and isinstance(v, str) else None
    type_name: str | None = None
    if "type" in kw:
        type_name = ast.unparse(kw["type"])[:100]
    default: Any = None
    if "default" in kw:
        ok, v = _literal(tree, kw["default"])
        default = v if ok and isinstance(v, (str, int, float, bool, type(None))) else None
    action: str | None = None
    if "action" in kw:
        ok, v = _literal(tree, kw["action"])
        action = v if ok and isinstance(v, str) else ast.unparse(kw["action"])[:100]
    choices: tuple[Any, ...] | None = None
    if "choices" in kw:
        ok, v = _literal(tree, kw["choices"])
        if ok and isinstance(v, (list, tuple, set, frozenset)):
            items = sorted(v, key=repr) if isinstance(v, (set, frozenset)) else list(v)
            if all(isinstance(i, (str, int, float, bool)) for i in items):
                choices = tuple(items)
    return ArgSpec(
        option_strings=options,
        dest=dest,
        type=type_name,
        default=default,
        action=action,
        choices=choices,
    )


def _scan_module(path: str, tree: ast.Module) -> EntrypointCandidate | None:
    has_main = any(_is_main_guard(s) for s in tree.body)
    uses_argparse = uses_click = False
    names: list[str] = []
    arguments: list[tuple[int, int, ArgSpec]] = []
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
                if name == "add_argument":
                    spec = _arg_spec(tree, node)
                    if spec is not None:
                        arguments.append((node.lineno, node.col_offset, spec))
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
        arguments=tuple(a for *_, a in sorted(arguments, key=lambda t: (t[0], t[1]))),
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
    return pack_context_report(sections, max_chars)[0]


def pack_context_report(
    sections: Iterable[ContextSection | Mapping[str, Any]], max_chars: int
) -> tuple[str, tuple[str, ...]]:
    """pack_context plus the titles of the sections it truncated or omitted (priority
    order: a smaller budget drops the lowest-priority sections first, then truncates)."""
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    secs = [s if isinstance(s, ContextSection) else ContextSection(**s) for s in sections]
    ordered = [s for _, s in sorted(enumerate(secs), key=lambda p: (-p[1].priority, p[0]))]
    full = _SEP.join(_block(s.title, s.text) for s in ordered)
    if len(full) <= max_chars:
        return full, ()

    reserve = len(f"{_SEP}[omitted {len(ordered)} sections]")
    budget = max_chars - reserve
    parts: list[str] = []
    cut_titles: list[str] = []
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
        cut_titles.append(s.title)
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
    return out[:max_chars], tuple(cut_titles)


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


# --------------------------------------------------------------------------- version scan


class VersionMention(_Frozen):
    package: str  # canonical: python, torch, torchvision, cuda, cudnn, tensorflow, numpy
    version: str  # as written, without a leading "v"
    quote: str  # the whole source line, stripped


_VERSION_PACKAGES = {
    "python": "python",
    "pytorch": "torch",
    "torch": "torch",
    "torchvision": "torchvision",
    "cuda": "cuda",
    "cudnn": "cudnn",
    "tensorflow": "tensorflow",
    "numpy": "numpy",
}
_VERSION_RE = re.compile(
    r"(?<![\w.-])(python|pytorch|torchvision|torch|cudnn|cuda|tensorflow|numpy)"
    r"(?:\s+version\s+|\s*(?:===|==|>=|<=|~=|>|<|=|:)?\s*)"
    r"v?(\d+(?:\.\d+){0,2})(?!\w|\.\d)",
    re.IGNORECASE,
)


def find_version_mentions(text: str) -> tuple[VersionMention, ...]:
    """Versions of python/torch/torchvision/cuda/cudnn/tensorflow/numpy stated in text
    (case-insensitive, optional "v", 1-3 dot-separated parts). Pure regex; deduplicated,
    in text order. A bare number is never a mention: the package name must precede it."""
    out: list[VersionMention] = []
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        for m in _VERSION_RE.finditer(line):
            key = (_VERSION_PACKAGES[m.group(1).lower()], m.group(2), line.strip())
            if key not in seen:
                seen.add(key)
                out.append(VersionMention(package=key[0], version=key[1], quote=key[2]))
    return tuple(out)


def version_evidence(evidence: Mapping[str, Evidence]) -> tuple[Evidence, ...]:
    """Derived '<path>#versions' evidence for every observed README / requirement-file text
    (pure), linked to the scanned file's evidence."""
    out: list[Evidence] = []
    for ev in evidence.values():
        if ev.confidence != "observed" or not isinstance(ev.content.get("text"), str):
            continue
        path = str(ev.content.get("path", ev.locator))
        if ev.kind != "readme" and not (ev.kind == "repo_file" and _requirement_parser(path)):
            continue
        mentions = find_version_mentions(ev.content["text"])
        out.append(
            make_evidence(
                kind="repo_file",
                source=ev.source,
                locator=f"{path}#versions",
                content={"parser": "versions", "result": _jsonable(mentions)},
                confidence="derived",
                authority="repo",
                links=(ev.evidence_id,),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- metric lines


class MetricLine(_Frozen):
    path: str
    line_no: int  # 1-based
    line: str  # stripped source line


_METRIC_CALL_RE = re.compile(
    r"(?:^|[^\w.])(?:print|tqdm\.write|logging\.\w+|(?:\w+\.)*logger\.\w+|[\w.]+\.add_scalars?)"
    r"\s*\("
)
_METRIC_WORD_RE = re.compile(r"acc|err|loss|top1|top5|test_|val_|f1|bleu|map", re.IGNORECASE)
_MAX_METRIC_LINES_SHOWN = 40


def find_metric_lines(texts: Mapping[str, str]) -> tuple[MetricLine, ...]:
    """Lines that print/log a metric-like value (pure regex; comments skipped): a call to
    print, tqdm.write, logging.*, *logger.* or *.add_scalar(s) on a line containing
    acc|err|loss|top1|top5|test_|val_|f1|bleu|map (case-insensitive). Sorted by path."""
    out: list[MetricLine] = []
    for path in sorted(texts):
        for no, raw in enumerate(texts[path].splitlines(), start=1):
            line = raw.strip()
            if line.startswith("#") or not _METRIC_CALL_RE.search(line):
                continue
            if _METRIC_WORD_RE.search(line):
                out.append(MetricLine(path=path, line_no=no, line=line))
    return tuple(out)


def metric_line_evidence(evidence: Mapping[str, Evidence]) -> tuple[Evidence, ...]:
    """Derived '<path>#metric_lines' evidence for every entrypoint candidate whose head is
    stored (pure), linked to that file's observed evidence."""
    entry_paths = {c.path for c in load_repo_facts(evidence)[1].candidates}
    out: list[Evidence] = []
    for ev in evidence.values():
        path = str(ev.content.get("path", ev.locator))
        head = ev.content.get("head")
        if ev.confidence != "observed" or path not in entry_paths or not isinstance(head, str):
            continue
        lines = find_metric_lines({path: head})
        out.append(
            make_evidence(
                kind="repo_file",
                source=ev.source,
                locator=f"{path}#metric_lines",
                content={"parser": "metric_lines", "result": _jsonable(lines)},
                confidence="derived",
                authority="repo",
                links=(ev.evidence_id,),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- example commands


class ExampleCommand(_Frozen):
    path: str  # the entrypoint candidate it runs
    line: str  # the command (backslash continuations joined)
    source: str  # README or shell script it was found in


_INTERPRETERS = frozenset({"python", "python3", "bash", "sh"})
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_PROMPT_PREFIXES = ("$ ", "> ")
MAX_SHELL_SCRIPTS = 20


def is_shell_script(path: str) -> bool:
    return path.lower().endswith(".sh") and not is_skipped_path(path)


def _continued_lines(text: str) -> Iterable[str]:
    """Physical lines with trailing-backslash continuations joined by a single space."""
    buf: list[str] = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1].strip())
            continue
        buf.append(stripped.strip())
        yield " ".join(p for p in buf if p)
        buf = []
    if buf:
        yield " ".join(p for p in buf if p)


def _command_target(segment: str, candidates: Set[str], source_dir: str) -> str | None:
    """Candidate path run by `segment` (env vars, python/bash/sh, -m module), else None."""
    try:
        tokens = shlex.split(segment, comments=True)
    except ValueError:
        tokens = segment.split()
    i = 0
    while i < len(tokens) and _ENV_ASSIGN_RE.match(tokens[i]):
        i += 1
    if i >= len(tokens) or tokens[i] not in _INTERPRETERS:
        return None
    i += 1
    while i < len(tokens) and tokens[i].startswith("-") and tokens[i] != "-m":
        i += 1
    if i >= len(tokens):
        return None
    if tokens[i] == "-m":
        if i + 1 >= len(tokens):
            return None
        mod = tokens[i + 1].replace(".", "/")
        options = [f"{mod}.py", f"{mod}/__main__.py"]
    else:
        options = [tokens[i]]
        if source_dir:
            options.append(f"{source_dir}/{tokens[i]}")
    for opt in options:
        try:
            norm = normalize_repo_path(opt)
        except ValueError:
            continue
        if norm in candidates:
            return norm
    return None


def find_example_commands(
    texts: Mapping[str, str], candidates: Set[str]
) -> tuple[ExampleCommand, ...]:
    """Command lines that run an entrypoint candidate (pure), from README / shell texts
    (source path -> text): fenced blocks and plain lines alike, inline `code` spans checked
    separately, backslash continuations joined, leading "$ " / "> " prompts and env var
    assignments skipped, then python/python3/bash/sh followed by a candidate path (or
    "python -m pkg.mod" mapping to pkg/mod.py or pkg/mod/__main__.py). Deduplicated."""
    out: list[ExampleCommand] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sorted(texts):
        source_dir = posixpath.dirname(source)
        for line in _continued_lines(texts[source]):
            segments = [m.group(1).strip() for m in _INLINE_CODE_RE.finditer(line)] or [line]
            for seg in segments:
                for prefix in _PROMPT_PREFIXES:
                    seg = seg.removeprefix(prefix)
                seg = seg.strip()
                target = _command_target(seg, candidates, source_dir) if seg else None
                key = (target or "", seg, source)
                if target is not None and key not in seen:
                    seen.add(key)
                    out.append(ExampleCommand(path=target, line=seg, source=source))
    return tuple(out)


def command_evidence(evidence: Mapping[str, Evidence]) -> tuple[Evidence, ...]:
    """Derived '<source>#commands' evidence for every observed README / shell-script text
    (pure), linked to that file's evidence."""
    candidates = {c.path for c in load_repo_facts(evidence)[1].candidates}
    out: list[Evidence] = []
    for ev in evidence.values():
        text = ev.content.get("text")
        path = str(ev.content.get("path", ev.locator))
        if ev.confidence != "observed" or not isinstance(text, str):
            continue
        if ev.kind != "readme" and not (ev.kind == "repo_file" and is_shell_script(path)):
            continue
        commands = find_example_commands({path: text}, candidates)
        out.append(
            make_evidence(
                kind="repo_file",
                source=ev.source,
                locator=f"{path}#commands",
                content={"parser": "commands", "result": _jsonable(commands)},
                confidence="derived",
                authority="repo",
                links=(ev.evidence_id,),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- LLM subtasks
# A env, B entrypoints, C workflow. The LLM classifies and locates; the contracts' validators
# (grounding against repo_files / evidence texts / entrypoint args) verify in code.

PROMPT_HEAD_CHARS = 4_000  # per py/config head shown in the prompt (texts keep the full head)
_FACT_CHARS = 1_500
_MAX_LISTED_FILES = 200
# Unknown names each subtask may propose (enforced via context["allowed_unknowns"]).
ALLOWED_UNKNOWNS: dict[str, frozenset[str]] = {
    "A": frozenset({"python_version", "torch_version", "cuda_version"}),
    "B": frozenset(),
    "C": frozenset({"dataset_source", "checkpoint_url", "eval_protocol"}),
}
# Package names whose "<canonical>_version" unknown is dropped when the version is cited.
_PKG_CANONICAL = {"pytorch": "torch", "cudatoolkit": "cuda", "python3": "python"}

_CITE = """Citation = {"evidence_id": "E1", "quote": "<text copied from that source>"}
Each source text is headed by an alias like [E1]; use that alias as evidence_id.
Copy quotes character by character from the source (max 300 chars)."""


def _unknowns_rule(kind: str) -> str:
    names = sorted(ALLOWED_UNKNOWNS[kind])
    if not names:
        return 'Always return "unknowns": []. Never invent paths, versions or arguments.'
    return (
        "If one of these is needed but not stated, add an unknown instead of guessing:\n"
        f'{{"name": one of {names}, "why": "short reason"}}. Only these names.\n'
        "Never invent paths, versions or arguments."
    )


SYSTEM_A = f"""You extract the software environment of a Python research repo.
Return ONLY JSON:
{{"python": PackageFact or null, "packages": [PackageFact], "gpu_required": true/false/null,
 "unknowns": [Unknown]}}
PackageFact = {{"name": "torch", "version": "0.4.0" or null, "citation": Citation or null}}
{_CITE}
Prefer the "Verified version mentions": cite their quote with the alias shown.
A version must appear in its quote. No citation -> version null.
{_unknowns_rule("A")}"""

SYSTEM_B = f"""You find the scripts used to run a Python research repo.
Return ONLY JSON:
{{"entrypoints": [{{"path": "train.py", "kind": "train" | "eval" | "other",
   "example": Citation or null}}],
 "unknowns": []}}
{_CITE}
Rules:
- One entry per script: each path at most once. path must be a listed entrypoint candidate.
- kind: "train" if the script trains (even if it also evaluates), "eval" only for
  evaluation-only scripts.
- example: pick the verified command for each script from "Verified example commands" and
  cite it with the alias shown there; copy the whole command exactly, flags included.
  If a script has no verified command, example is null.
- Do not list the flags separately: code reads them from your example.
{_unknowns_rule("B")}"""

SYSTEM_C = f"""You map the experiment workflow of a Python research repo.
Return ONLY JSON:
{{"train_scripts": ["path"], "eval_scripts": ["path"], "configs": ["path"],
 "datasets": [{{"name": "...", "citation": Citation or null}}],
 "checkpoints": [{{"name": "...", "url": "..." or null, "citation": Citation}}],
 "metric_outputs": [{{"path": "script.py", "metric": "test_acc", "citation": Citation}}],
 "unknowns": [Unknown]}}
{_CITE}
Rules:
- Paths must be repo files listed in the context; each path once. Empty list if none.
- metric_outputs: pick from the "Verified metric output lines"; cite that line with the
  alias shown there (not the script's own alias).
{_unknowns_rule("C")}"""

_CIT_EX = '{"evidence_id": "E1", "quote": "<copy exact text>"}'
SUBTASK_EXAMPLES: dict[str, str] = {
    "citation": _CIT_EX,
    "example": _CIT_EX,
    "packages": f'{{"name": "torch", "version": "0.4.0", "citation": {_CIT_EX}}}',
    "python": f'{{"name": "python", "version": "3.6", "citation": {_CIT_EX}}}',
    "unknowns": '{"name": "<an allowed name>", "why": "<short reason>"}',
    "entrypoints": '{"path": "train.py", "kind": "train", '
    '"example": {"evidence_id": "E1", "quote": "python train.py --epochs 10"}} '
    '(no verified command -> "example": null)',
    "datasets": '{"name": "cifar10", "citation": null}',
    "checkpoints": f'{{"name": "model", "url": null, "citation": {_CIT_EX}}}',
    "metric_outputs": f'{{"path": "train.py", "metric": "test_acc", "citation": {_CIT_EX}}}',
}

SubtaskKind = Literal["A", "B", "C"]
SUBTASK_KINDS: tuple[SubtaskKind, ...] = ("A", "B", "C")
# node metadata stored next to the schema fields in A/B/C evidence content (not schema data):
# strip them before re-validating the content with the schema (extra="forbid")
SUBTASK_META_KEYS = frozenset({"evicted", "gate_attempts"})
_SUBTASKS: dict[str, tuple[type[BaseModel], str, EvidenceKind]] = {
    "A": (EnvSpec, SYSTEM_A, "env_spec"),
    "B": (EntrypointList, SYSTEM_B, "entrypoints"),
    "C": (WorkflowAnalysis, SYSTEM_C, "workflow"),
}
_REQUIRED_SET_CONTEXT = ("repo_files", "evidence_ids")
_REQUIRED_MAP_CONTEXT = (
    "texts",
    "entrypoint_args",
    "aliases",
    "evidence_kinds",
    "metric_line_evidence",
    "command_evidence",
)


def load_repo_facts(
    evidence: Mapping[str, Evidence],
) -> tuple[tuple[Evidence, ...], EntrypointScan]:
    """Rebuild the extractor output from stored evidence (pure).

    Returns (derived '#parsed' facts in input order, EntrypointScan from the 'entrypoint'
    facts). '#versions' / '#metric_lines' evidence is not a fact here (build_subtask_context
    reads it).
    """
    facts = tuple(
        ev
        for ev in evidence.values()
        if ev.kind == "repo_file"
        and ev.confidence == "derived"
        and ev.locator.endswith("#parsed")
        and "parser" in ev.content
    )
    candidates: list[EntrypointCandidate] = []
    unparsable: dict[str, str] = {}
    for ev in facts:
        if ev.content["parser"] != "entrypoint":
            continue
        path = ev.locator.removesuffix("#parsed")
        result = ev.content["result"]
        if isinstance(result, Mapping) and "error" in result:
            unparsable[path] = str(result["error"])
        else:
            candidates.append(EntrypointCandidate.model_validate(result))
    candidates.sort(key=lambda c: c.path)
    return facts, EntrypointScan(candidates=tuple(candidates), unparsable=unparsable)


def _inventory(evidence: Mapping[str, Evidence]) -> RepoInventory:
    invs = {
        ev.evidence_id: ev
        for ev in evidence.values()
        if ev.kind == "repo_file" and ev.locator == "inventory"
    }
    if len(invs) != 1:
        raise ValueError(f"expected exactly one inventory evidence, found {len(invs)}")
    return RepoInventory.model_validate(next(iter(invs.values())).content)


def _fact_line(ev: Evidence) -> str:
    path = ev.locator.removesuffix("#parsed")
    body = canonical_json(ev.content["result"])
    if len(body) > _FACT_CHARS:
        body = body[:_FACT_CHARS] + " ...[cut]"
    return f"- {path} ({ev.content['parser']}): {body}"


def build_subtask_context(
    evidence: Mapping[str, Evidence],
    facts: Iterable[Evidence],
    scan: EntrypointScan,
    *,
    max_chars: int = 12_000,
) -> tuple[dict[str, Any], dict[SubtaskKind, str]]:
    """Validation context + one packed prompt per subtask (pure); see subtask_sections."""
    context, sections = subtask_sections(evidence, facts, scan)
    return context, {k: pack_context(secs, max_chars) for k, secs in sections.items()}


def subtask_sections(
    evidence: Mapping[str, Evidence],
    facts: Iterable[Evidence],
    scan: EntrypointScan,
) -> tuple[dict[str, Any], dict[SubtaskKind, list[ContextSection]]]:
    """Validation context + the prompt sections of each subtask (pure, unpacked).

    Context: repo_files (frozenset, from the single inventory record), evidence_ids
    (frozenset of the ids in texts), texts (evidence_id -> README full text, requirement /
    docker file text, or head of a py/config file), entrypoint_args (path -> frozenset of
    arg names), aliases ("E1".. -> evidence_id, in prompt order), evidence_kinds
    (evidence_id -> kind), metric_line_evidence (path -> id of its non-empty
    '<path>#metric_lines' evidence, which is citable: its text is the verified lines),
    command_evidence (candidate path -> ids of the non-empty '<source>#commands' evidence
    holding a command for it; also citable, text = the verified commands).
    Prompts show aliases only, README + facts + entrypoint heads
    first; A also gets the "Verified version mentions" ('#versions' evidence), B the
    "Verified example commands" ('#commands') and C the "Verified metric output lines"
    ('#metric_lines'), all found by code.
    """
    inv = _inventory(evidence)
    entry_paths = {c.path for c in scan.candidates}
    entries: list[tuple[int, int, Evidence, str, str, str]] = []
    for idx, ev in enumerate(evidence.values()):
        if ev.confidence != "observed" or ev.kind not in ("readme", "repo_file"):
            continue
        path = str(ev.content.get("path", ev.locator))
        if isinstance(ev.content.get("text"), str):  # README, requirement and docker files
            text = ev.content["text"]
            prompt_text = text
            priority = 100 if ev.kind == "readme" else 70
        elif isinstance(ev.content.get("head"), str):  # py / config digest
            text = ev.content["head"]
            prompt_text = text[:PROMPT_HEAD_CHARS]
            if len(text) > PROMPT_HEAD_CHARS:
                prompt_text += f"\n[... {len(text) - PROMPT_HEAD_CHARS} more chars]"
            priority = 80 if path in entry_paths else 60
        else:
            continue
        entries.append((priority, idx, ev, path, text, prompt_text))
    entries.sort(key=lambda e: (-e[0], e[1]))

    texts: dict[str, str] = {}
    aliases: dict[str, str] = {}
    alias_of: dict[str, str] = {}
    sections: list[ContextSection] = []
    for priority, _, ev, path, text, prompt_text in entries:
        if ev.evidence_id in texts:
            continue
        alias = f"E{len(aliases) + 1}"
        aliases[alias] = ev.evidence_id
        alias_of[ev.evidence_id] = alias
        texts[ev.evidence_id] = text
        sections.append(
            ContextSection(title=f"{path} [{alias}]", text=prompt_text, priority=priority)
        )

    cand_lines = [
        f"- {c.path}: args {list(c.arg_names)}" + ("" if c.has_main_guard else " (no main guard)")
        for c in scan.candidates
    ] or ["(none found)"]
    listed = [*inv.python_files, *inv.config_files][:_MAX_LISTED_FILES]
    other_facts = [_fact_line(ev) for ev in facts if ev.content.get("parser") != "entrypoint"]
    fact_text = "\n".join(
        [
            "Entrypoint candidates (path: argument names):",
            *cand_lines,
            "Python and config files in the repo:",
            *(f"- {p}" for p in listed),
            *(["Parsed facts:", *other_facts] if other_facts else []),
        ]
    )
    sections.append(
        ContextSection(title="Repo facts (parsed by code)", text=fact_text, priority=90)
    )

    version_lines: list[str] = []
    for ev in evidence.values():
        if not (ev.locator.endswith("#versions") and ev.content.get("parser") == "versions"):
            continue
        alias = next((alias_of[i] for i in ev.links if i in alias_of), None)
        if alias is None:
            continue
        for m in ev.content["result"]:
            quote = str(m["quote"])[:CITATION_MAX_QUOTE_CHARS]
            version_lines.append(f'- {m["package"]} {m["version"]} [{alias}] quote: "{quote}"')
    version_section = ContextSection(
        title="Verified version mentions (found by code)",
        text="\n".join(version_lines) or "(none found)",
        priority=95,
    )

    metric_lines: list[str] = []
    metric_line_evidence: dict[str, str] = {}  # path -> '<path>#metric_lines' evidence_id
    for ev in evidence.values():
        if not (
            ev.locator.endswith("#metric_lines") and ev.content.get("parser") == "metric_lines"
        ):
            continue
        result = ev.content["result"]
        if not result or not any(i in alias_of for i in ev.links) or ev.evidence_id in texts:
            continue
        # citable on its own: text = the verified lines, one per line, with a new alias
        alias = f"E{len(aliases) + 1}"
        aliases[alias] = ev.evidence_id
        alias_of[ev.evidence_id] = alias
        texts[ev.evidence_id] = "\n".join(str(m["line"]) for m in result)
        metric_line_evidence[ev.locator.removesuffix("#metric_lines")] = ev.evidence_id
        for m in result:
            line = str(m["line"])[:CITATION_MAX_QUOTE_CHARS]
            metric_lines.append(f'- {m["path"]}:{m["line_no"]} [{alias}] quote: "{line}"')
    by_script: dict[str, list[str]] = {}
    command_evidence_ids: dict[str, set[str]] = {}  # candidate path -> '#commands' ids
    for ev in evidence.values():
        if not (ev.locator.endswith("#commands") and ev.content.get("parser") == "commands"):
            continue
        result = ev.content["result"]
        if not result or not any(i in alias_of for i in ev.links) or ev.evidence_id in texts:
            continue
        alias = f"E{len(aliases) + 1}"
        aliases[alias] = ev.evidence_id
        alias_of[ev.evidence_id] = alias
        texts[ev.evidence_id] = "\n".join(str(c["line"]) for c in result)
        for c in result:
            command_evidence_ids.setdefault(str(c["path"]), set()).add(ev.evidence_id)
            by_script.setdefault(str(c["path"]), []).append(
                f'- [{alias}] "{str(c["line"])[:CITATION_MAX_QUOTE_CHARS]}"'
            )
    command_lines: list[str] = []
    for script in sorted(by_script):
        command_lines += [f"{script}:", *by_script[script]]
    shown_cmds = command_lines[:_MAX_METRIC_LINES_SHOWN]
    if len(command_lines) > len(shown_cmds):
        shown_cmds.append(f"[... {len(command_lines) - len(shown_cmds)} more lines]")
    command_section = ContextSection(
        title="Verified example commands (found by code)",
        text="\n".join(shown_cmds) or "(none found)",
        priority=95,
    )

    shown = metric_lines[:_MAX_METRIC_LINES_SHOWN]
    if len(metric_lines) > len(shown):
        shown.append(f"[... {len(metric_lines) - len(shown)} more lines]")
    metric_section = ContextSection(
        title="Verified metric output lines (found by code)",
        text="\n".join(shown) or "(none found)",
        priority=95,
    )

    context: dict[str, Any] = {
        "repo_files": frozenset(f.path for f in inv.files),
        "evidence_ids": frozenset(texts),
        "texts": texts,
        "entrypoint_args": {c.path: frozenset(c.arg_names) for c in scan.candidates},
        "aliases": aliases,
        "evidence_kinds": {i: evidence[i].kind for i in texts},
        "metric_line_evidence": metric_line_evidence,
        "command_evidence": {p: frozenset(ids) for p, ids in command_evidence_ids.items()},
    }
    per_kind: dict[SubtaskKind, list[ContextSection]] = {
        "A": [*sections, version_section],
        "B": [*sections, command_section],
        "C": [*sections, metric_section],
    }
    return context, per_kind


@dataclass(frozen=True)
class SubtaskOutcome:
    kind: SubtaskKind
    ok: bool
    value: EnvSpec | EntrypointList | WorkflowAnalysis | None
    evidence: tuple[Evidence, ...]
    events: tuple[UnknownOpened, ...]
    errors: tuple[str, ...]
    attempts: int
    dropped_unknowns: tuple[str, ...] = ()  # "<pkg>_version" dropped: version was cited


def _cited_ids(obj: Any) -> set[str]:
    if isinstance(obj, Citation):
        return {obj.evidence_id}
    if isinstance(obj, BaseModel):
        return set().union(*(_cited_ids(getattr(obj, f)) for f in type(obj).model_fields))
    if isinstance(obj, (tuple, list)):
        return set().union(*(_cited_ids(v) for v in obj))
    return set()


def _drop_cited_version_unknowns(value: BaseModel) -> tuple[Any, tuple[str, ...]]:
    """Drop "<pkg>_version" unknowns whose package has a cited version in the same output."""
    if not isinstance(value, EnvSpec):
        return value, ()
    cited = {
        _PKG_CANONICAL.get(p.name.strip().lower(), p.name.strip().lower())
        for p in (value.python, *value.packages)
        if p is not None and p.version is not None and p.citation is not None
    }
    keep = tuple(u for u in value.unknowns if u.name.removesuffix("_version") not in cited)
    dropped = tuple(u.name for u in value.unknowns if u not in keep)
    return (value.model_copy(update={"unknowns": keep}) if dropped else value), dropped


def run_subtask(
    kind: SubtaskKind,
    llm: LLM,
    context: Mapping[str, Any],
    packed: str,
    *,
    source: str,
    existing_unknown_ids: set[str],
    max_tokens: int,
    max_retries: int = 2,
    repack: Callable[[int], tuple[str, tuple[str, ...]]] | None = None,
    budget: int | None = None,
) -> SubtaskOutcome:
    """One gated LLM subtask. On success: one derived/llm Evidence linked to every cited
    evidence_id (full ids; aliases are resolved during validation), plus UnknownOpened for
    each proposal not already open (first wins). Unknown names are limited per subtask
    (ALLOWED_UNKNOWNS, passed as context["allowed_unknowns"]); "<pkg>_version" unknowns are
    dropped when the same output cites that package's version. Context eviction: on a
    context-limit error the context is re-packed by repack(budget) at half the budget
    (`budget` = the max_chars `packed` was made with), at most twice; the evicted section
    titles go into the evidence ("evicted"). BudgetExceeded / LLMError (incl. ContextTooLong
    after the evictions) propagate."""
    if kind not in _SUBTASKS:
        raise ValueError(f"unknown subtask kind {kind!r}")
    for key in _REQUIRED_MAP_CONTEXT:
        if not isinstance(context.get(key), Mapping):
            raise ValueError(f"run_subtask: validation context key {key!r} must be a mapping")
    schema, system, ev_kind = _SUBTASKS[kind]

    def build(b: int | None) -> tuple[str, str, tuple[str, ...]]:
        text, evicted = (packed, ()) if b is None or b == budget or repack is None else repack(b)
        return system, f"{text}\n\nReturn the JSON now.", evicted

    call = call_with_eviction(
        llm,
        build,
        budget=budget,
        schema=schema,
        context={**context, "allowed_unknowns": ALLOWED_UNKNOWNS[kind]},
        max_tokens=max_tokens,
        max_retries=max_retries,
        required_context=_REQUIRED_SET_CONTEXT,
        examples=SUBTASK_EXAMPLES,
    )
    result = call.result
    if not result.ok or result.value is None:
        return SubtaskOutcome(kind, False, None, (), (), result.errors, result.attempts)

    value, dropped = _drop_cited_version_unknowns(result.value)
    content = value.model_dump(mode="json")
    if call.evicted:
        content["evicted"] = list(call.evicted)
    content["gate_attempts"] = result.attempts  # additive: tries of the gate that answered
    ev = make_evidence(
        kind=ev_kind,
        source=source,
        locator=f"subtask:{kind}",
        content=content,
        confidence="derived",
        authority="llm",
        links=sorted(_cited_ids(value)),
    )
    seen = set(existing_unknown_ids)
    events: list[UnknownOpened] = []
    for prop in value.unknowns:  # type: ignore[attr-defined]
        uid = unknown_id_for(prop.name)
        if uid in seen:
            continue
        seen.add(uid)
        events.append(UnknownOpened(unknown_id=uid, name=prop.name, evidence_id=ev.evidence_id))
    return SubtaskOutcome(
        kind,
        True,
        value,  # type: ignore[arg-type]
        (ev,),
        tuple(events),
        result.errors,
        result.attempts,
        dropped,
    )


# --------------------------------------------------------------------------- graph nodes


def _subtask_node(
    kind: SubtaskKind, state: Mapping[str, Any], deps: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one gated subtask on the configure evidence in the state. Returns evidence,
    events and one SubtaskSummary; writes only LLM telemetry (on_record). BudgetExceeded /
    LLMError become a failed summary so the other branches' evidence is still persisted."""
    evidence = {ev.evidence_id: ev for ev in state["evidence"]}
    facts, scan = load_repo_facts(evidence)
    max_chars = state.get("max_chars", 12_000)
    context, sections = subtask_sections(evidence, facts, scan)
    packed = pack_context(sections[kind], max_chars)
    existing = {e.unknown_id for e in state.get("events", ()) if isinstance(e, UnknownOpened)}
    llm = deps["llm_for"](f"subtask_{kind.lower()}")
    llm.on_record = deps["store"].append_call_record
    try:
        out = run_subtask(
            kind,
            llm,
            context,
            packed,
            source=state["source"],
            existing_unknown_ids=existing,
            max_tokens=state.get("max_tokens", 1_500),
            repack=lambda b: pack_context_report(sections[kind], b),
            budget=max_chars,
        )
        ok, attempts, errors = out.ok, out.attempts, out.errors
        new_evidence, events = list(out.evidence), list(out.events)
    except (BudgetExceeded, LLMError) as e:
        ok, attempts, errors = False, len(llm.records()) + 1, (f"{type(e).__name__}: {e}",)
        new_evidence, events = [], []
    records = llm.records()
    summary = SubtaskSummary(
        kind=kind,
        ok=ok,
        attempts=attempts,
        tokens=sum(r.input_tokens + r.output_tokens for r in records),
        latency_s=sum(r.latency_s for r in records),
        errors=errors,
    )
    return {"evidence": new_evidence, "events": events, "outcomes": [summary]}


def subtask_a(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    return _subtask_node("A", state, deps)


def subtask_b(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    return _subtask_node("B", state, deps)


def subtask_c(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    return _subtask_node("C", state, deps)
