"""Static dataset-call scan: which torchvision dataset classes does the repo build, and where?

Pure domain: works on texts already read by a provider; nothing is imported or executed
(ast only). Rules:
  - torchvision bindings are collected from every import in the file (module level or not):
    `import torchvision [as tv]`, `import torchvision.datasets [as d]`,
    `from torchvision import datasets [as d]`, `from torchvision.datasets[.<mod>] import X [as Y]`;
  - a call counts when its callee resolves to `torchvision.datasets[.<mod>].<Name>` with a
    capitalized Name (a class; helpers such as utils.download_url are skipped). Calls through
    any other module (e.g. a local `datasets` package) are ignored;
  - root: the `root=` keyword or the first positional argument, only if it is a str literal;
    otherwise root=None and `reason` says why (never guessed);
  - download: the `download=` keyword if it is a bool literal; absent -> False (the torchvision
    default) unless extra positional args or **kwargs could carry it -> None.
Files that do not parse are reported in DatasetCallScan.unparsed.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from reproducer.domain.evidence import Evidence, make_evidence

TV_DATASETS = "torchvision.datasets"
DATASET_CALLS_SUFFIX = "#dataset_calls"
DATASET_CALLS_PARSER = "dataset_calls"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetCall(_Frozen):
    path: str
    line: int  # 1-based
    cls: str  # torchvision class name, e.g. "CIFAR10"
    root: str | None  # str literal only
    reason: str | None = None  # why root is None
    download: bool | None


class DatasetCallScan(_Frozen):
    calls: tuple[DatasetCall, ...] = ()
    scanned: tuple[str, ...] = ()  # paths that parsed (with or without calls)
    unparsed: tuple[str, ...] = ()


def _bindings(tree: ast.AST) -> dict[str, str]:
    """Local name -> fully qualified torchvision name it refers to."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name != "torchvision" and not a.name.startswith("torchvision."):
                    continue
                if a.asname:
                    out[a.asname] = a.name
                else:
                    out["torchvision"] = "torchvision"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mod = node.module
            if mod != "torchvision" and not mod.startswith("torchvision."):
                continue
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{mod}.{a.name}"
    return out


def _qualified(func: ast.expr, bindings: Mapping[str, str]) -> str | None:
    attrs: list[str] = []
    while isinstance(func, ast.Attribute):
        attrs.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in bindings:
        return None
    return ".".join([bindings[func.id], *reversed(attrs)])


def _class_name(qualified: str | None) -> str | None:
    if qualified is None or not qualified.startswith(TV_DATASETS + "."):
        return None
    name = qualified.rsplit(".", 1)[1]
    return name if name[:1].isupper() else None


def _root(call: ast.Call) -> tuple[str | None, str | None]:
    expr: ast.expr | None = None
    for kw in call.keywords:
        if kw.arg == "root":
            expr = kw.value
    if expr is None and call.args and not isinstance(call.args[0], ast.Starred):
        expr = call.args[0]
    if expr is None:
        if any(kw.arg is None for kw in call.keywords) or call.args:
            return None, "root passed via *args/**kwargs"
        return None, "root not given"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value, None
    return None, f"non-literal root: {ast.unparse(expr)[:200]}"


def _download(call: ast.Call) -> bool | None:
    for kw in call.keywords:
        if kw.arg == "download":
            v = kw.value
            return v.value if isinstance(v, ast.Constant) and isinstance(v.value, bool) else None
    if len(call.args) > 1 or any(kw.arg is None for kw in call.keywords):
        return None  # may be passed positionally / via **kwargs
    return False


def find_dataset_calls(texts: Mapping[str, str]) -> DatasetCallScan:
    """Scan path -> source text. Calls sorted by (path, line, column)."""
    calls: list[tuple[str, int, int, DatasetCall]] = []
    scanned: list[str] = []
    unparsed: list[str] = []
    for path in sorted(texts):
        try:
            tree = ast.parse(texts[path], filename=path)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            unparsed.append(path)
            continue
        scanned.append(path)
        bindings = _bindings(tree)
        if not bindings:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            cls = _class_name(_qualified(node.func, bindings))
            if cls is None:
                continue
            root, reason = _root(node)
            call = DatasetCall(
                path=path,
                line=node.lineno,
                cls=cls,
                root=root,
                reason=reason,
                download=_download(node),
            )
            calls.append((path, node.lineno, node.col_offset, call))
    calls.sort(key=lambda t: t[:3])
    return DatasetCallScan(
        calls=tuple(c for *_, c in calls), scanned=tuple(scanned), unparsed=tuple(unparsed)
    )


def dataset_calls_evidence(
    scan: DatasetCallScan, source: str, links: Mapping[str, str]
) -> tuple[Evidence, ...]:
    """One derived '<path>#dataset_calls' repo evidence per scanned or unparsed path (empty
    result included, so "scanned, none found" is distinguishable from "not scanned"),
    linked to links[path] (the file's observed evidence id) when given."""
    out: list[Evidence] = []
    for path in sorted({*scan.scanned, *scan.unparsed}):
        if path in scan.unparsed:
            result: object = {"error": "source did not parse"}
        else:
            result = [c.model_dump(mode="json") for c in scan.calls if c.path == path]
        out.append(
            make_evidence(
                kind="repo_file",
                source=source,
                locator=path + DATASET_CALLS_SUFFIX,
                content={"parser": DATASET_CALLS_PARSER, "result": result},
                confidence="derived",
                authority="repo",
                links=(links[path],) if path in links else (),
            )
        )
    return tuple(out)


def load_dataset_calls(evidence: Mapping[str, Evidence], source: str) -> DatasetCallScan | None:
    """Rebuild the scan from stored '#dataset_calls' evidence of source; None if none stored."""
    stored = [
        ev
        for ev in evidence.values()
        if ev.source == source
        and ev.kind == "repo_file"
        and ev.locator.endswith(DATASET_CALLS_SUFFIX)
        and ev.content.get("parser") == DATASET_CALLS_PARSER
    ]
    if not stored:
        return None
    calls: list[DatasetCall] = []
    scanned: set[str] = set()
    unparsed: set[str] = set()
    for ev in stored:
        path = ev.locator.removesuffix(DATASET_CALLS_SUFFIX)
        result = ev.content["result"]
        if isinstance(result, Mapping):
            unparsed.add(path)
            continue
        scanned.add(path)
        calls.extend(DatasetCall.model_validate(c) for c in result)
    calls.sort(key=lambda c: (c.path, c.line))
    return DatasetCallScan(
        calls=tuple(dict.fromkeys(calls)),
        scanned=tuple(sorted(scanned)),
        unparsed=tuple(sorted(unparsed)),
    )
