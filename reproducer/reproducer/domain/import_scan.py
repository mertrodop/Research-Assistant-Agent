"""Static import scan: which third-party pip packages do the entrypoints import?

Pure domain: works on texts already read by a provider; nothing is imported or executed
(ast only). Rules:
  - every `import x` / `from x import y` in an entrypoint (anywhere in the file, including
    inside functions and try/except) counts by its top-level module name;
  - imports of LOCAL modules are followed exactly one level: the local .py module/package files
    they name are scanned too, their own local imports are not followed further;
  - local modules = top-level .py files / directories of the repo root or the entrypoint's
    directory; relative imports are local;
  - stdlib modules (sys.stdlib_module_names of the running 3.10 interpreter) are dropped;
  - module -> pip name via MODULE_TO_PIP, identity otherwise.
Files that do not parse (e.g. Python 2 syntax) are reported in ImportScan.unparsed, not guessed.
"""

from __future__ import annotations

import ast
import posixpath
import re
import sys
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


MODULE_TO_PIP: Mapping[str, str] = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
}
STDLIB_MODULES = frozenset(sys.stdlib_module_names)


class ImportedPackage(_Frozen):
    pip_name: str
    module: str  # top-level module name as imported
    path: str  # repo file that imports it (first seen)


class ImportScan(_Frozen):
    packages: tuple[ImportedPackage, ...] = ()
    unparsed: tuple[str, ...] = ()  # repo paths whose source did not parse


def normalize_pip_name(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _local_names(repo_paths: set[str], base: str) -> set[str]:
    prefix = f"{base}/" if base else ""
    out: set[str] = set()
    for p in repo_paths:
        if not p.startswith(prefix):
            continue
        rest = p[len(prefix) :]
        head, sep, _ = rest.partition("/")
        if sep:
            out.add(head)  # a directory (package or namespace package)
        elif head.endswith(".py"):
            out.add(head[:-3])
    return out


def _imports(tree: ast.AST) -> list[tuple[str, int, tuple[str, ...]]]:
    """(dotted module, relative level, imported names) per import statement, in source order."""
    found: list[tuple[int, int, tuple[str, int, tuple[str, ...]]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.append((node.lineno, node.col_offset, (a.name, 0, ())))
        elif isinstance(node, ast.ImportFrom):
            names = tuple(a.name for a in node.names)
            found.append((node.lineno, node.col_offset, (node.module or "", node.level, names)))
    found.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in found]


def _candidate_files(base: str, module: str, names: tuple[str, ...]) -> list[str]:
    parts = [p for p in module.split(".") if p]
    cands: list[str] = []
    for i in range(1, len(parts) + 1):
        cands.append(posixpath.join(base, *parts[:i], "__init__.py"))
    if parts:
        cands.append(posixpath.join(base, *parts) + ".py")
    for n in names:
        if n != "*":
            cands.append(posixpath.join(base, *parts, n) + ".py")
            cands.append(posixpath.join(base, *parts, n, "__init__.py"))
    return cands


def third_party_imports(
    entrypoints: Mapping[str, str], repo_files: Mapping[str, str]
) -> ImportScan:
    """Scan entrypoint sources (path -> text) for third-party imports.

    repo_files maps every repo path to its text; the text is only needed for local .py
    modules that may be followed (other paths may map to ""). Result order: entrypoints in
    sorted path order, imports in source order, each pip package once (first importer wins).
    """
    repo_paths = set(repo_files) | set(entrypoints)
    found: dict[str, ImportedPackage] = {}
    unparsed: list[str] = []
    scanned: set[str] = set()

    def parse(path: str, text: str) -> ast.AST | None:
        try:
            return ast.parse(text, filename=path)
        except (SyntaxError, ValueError):
            unparsed.append(path)
            return None

    def record(top: str, path: str) -> None:
        if top in STDLIB_MODULES or top == "__main__":
            return
        pip = MODULE_TO_PIP.get(top, top)
        found.setdefault(
            normalize_pip_name(pip), ImportedPackage(pip_name=pip, module=top, path=path)
        )

    def scan(path: str, text: str, follow: bool) -> None:
        if path in scanned:
            return
        scanned.add(path)
        tree = parse(path, text)
        if tree is None:
            return
        here = posixpath.dirname(path)
        local = _local_names(repo_paths, "") | _local_names(repo_paths, here)
        for module, level, names in _imports(tree):
            if level > 0:
                if follow:
                    base = here
                    for _ in range(level - 1):
                        base = posixpath.dirname(base)
                    for cand in _candidate_files(base, module, names):
                        if cand in repo_files and cand.endswith(".py"):
                            scan(cand, repo_files[cand], follow=False)
                continue
            top = module.split(".", 1)[0]
            if top in local:
                if follow:
                    for base in dict.fromkeys((here, "")):
                        for cand in _candidate_files(base, module, names):
                            if cand in repo_files and cand.endswith(".py"):
                                scan(cand, repo_files[cand], follow=False)
                continue
            record(top, path)

    for path in sorted(entrypoints):
        scan(path, entrypoints[path], follow=True)
    return ImportScan(packages=tuple(found.values()), unparsed=tuple(unparsed))


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def local_import_modules(entry_path: str, entry_text: str, repo_paths: set[str]) -> tuple[str, ...]:
    """Dotted names of the LOCAL modules the entrypoint imports (absolute imports only), for an
    import probe: `import a.b` -> a.b; `from a.b import c` -> a.b, plus a.b.c when that is a
    repo module file/package. Local = top-level name is a repo-root or entrypoint-dir .py file
    or directory. Unparsable entrypoint -> (). Source order, deduplicated."""
    try:
        tree = ast.parse(entry_text, filename=entry_path)
    except (SyntaxError, ValueError):
        return ()
    here = posixpath.dirname(entry_path)
    local = _local_names(repo_paths, "") | _local_names(repo_paths, here)
    out: list[str] = []
    for module, level, names in _imports(tree):
        if level > 0 or not module or module.split(".", 1)[0] not in local:
            continue
        found = [module]
        base = module.replace(".", "/")
        for n in names:
            if n == "*":
                continue
            for root in dict.fromkeys(("", here)):
                prefix = f"{root}/" if root else ""
                if f"{prefix}{base}/{n}.py" in repo_paths or f"{prefix}{base}/{n}/__init__.py" in (
                    repo_paths
                ):
                    found.append(f"{module}.{n}")
        out.extend(m for m in found if _IDENT_RE.match(m))
    return tuple(dict.fromkeys(out))
