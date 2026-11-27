"""Failure classification (code, never LLM).

classify(smoke evidence content, context) -> Classification{cls, pattern, params, reason}:
  infra    signal "infra", or a CUDA init / driver / arch error (cuInit, "CUDA unknown error",
           "no kernel image is available", ...): stop, never the critic.
  known    a pattern with a zero-token code fix (rules/known_fixes.py):
             missing_output_dir  FileNotFoundError raised by a WRITE call (open mode w/a/x,
                                 torch.save, np.save, .to_csv, .savefig, ...) on a path that
                                 is relative or under /work, not under /work/data or another
                                 data mount, whose parent dir has no file in the fresh clone;
             oom                 signal oom / CUDA out of memory, with a batch argument in the
                                 argparse spec (dest batch_size | bs | train_batch_size);
             missing_module      ModuleNotFoundError "No module named X" for a non-local X;
             missing_data_path   FileNotFoundError reading under a scanned dataset root that is
                                 not mounted, when a registry dataset is called at that root.
  critic   any other real traceback (kind python / cuda), with its repo_frame (S5).
  unfixable no traceback and no known pattern (timeout without progress, Killed, exit codes),
           an OOM without a batch argument, a write into a read-only data mount, ...
The write call is found on the source line of the innermost frame or, when that is library
code (torch.save -> open(name, mode)), of the innermost repo frame (traceback raw text).
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence, Set
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reproducer.domain.import_scan import MODULE_TO_PIP
from reproducer.domain.smoke_plan import BudgetArg
from reproducer.domain.traceback_parse import is_repo_file

WORK = "/work"
BATCH_DESTS = ("batch_size", "bs", "train_batch_size")
ACCUM_DESTS = (
    "grad_accum",
    "grad_accum_steps",
    "gradient_accumulation_steps",
    "accumulate_grad_batches",
    "accum_iter",
    "accumulation_steps",
)

ClassKind = Literal["infra", "known", "critic", "unfixable"]

_INFRA_RE = re.compile(
    r"cuInit|CUDA unknown error|CUDA error: unknown error|CUDA driver initialization failed|"
    r"no CUDA-capable device|CUDA driver version is insufficient|NVIDIA driver|"
    r"no kernel image is available|CUDA error: initialization error|"
    r"Found no NVIDIA driver",
    re.IGNORECASE,
)
_NO_FILE_RE = re.compile(r"No such file or directory: '(?P<path>[^']+)'")
_NO_MODULE_RE = re.compile(r"No module named '(?P<mod>[\w.]+)'")
_FRAME_RE = re.compile(r'^\s*File "[^"]+", line \d+')
_OPEN_WRITE_RE = re.compile(
    r"""\bopen\s*\((?:[^()]|\([^()]*\))*?(?:,\s*|mode\s*=\s*)[rbt]*['"][rbt+]*[wax][rbt+]*['"]"""
)
_WRITE_CALL_RE = re.compile(
    r"\b(?:torch\.save|np\.save[a-z]*|numpy\.save[a-z]*|pickle\.dump|json\.dump|"
    r"imageio\.imwrite|cv2\.imwrite|plt\.savefig|"
    r"\w+\.(?:to_csv|to_json|to_parquet|to_pickle|savefig|save_pretrained|save|write_text|"
    r"write_bytes|savez))\s*\("
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Classification(_Frozen):
    cls: ClassKind
    pattern: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str


class ClassifyContext(_Frozen):
    """What classification may look at besides the smoke evidence (all from stored facts)."""

    repo_files: frozenset[str]  # inventory paths of the fresh clone
    arguments: tuple[BudgetArg, ...] = ()  # argparse spec of the entrypoint
    mounted: frozenset[str] = frozenset()  # container paths of read-only data mounts
    dataset_roots: Mapping[str, str | None] = Field(default_factory=dict)  # container root ->
    # registry name of a dataset called at that root (None: not in the registry)


_FRAME_FILE_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line \d+')


def _frame_sources(raw: str) -> list[tuple[str, str]]:
    """(file, source line) per frame of the traceback text, outermost first ('' if the
    source line is not shown)."""
    lines = raw.split("\n")
    out: list[tuple[str, str]] = []
    for i, ln in enumerate(lines):
        m = _FRAME_FILE_RE.match(ln)
        if not m:
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        src = nxt.strip() if nxt.startswith((" ", "\t")) and not _FRAME_RE.match(nxt) else ""
        out.append((m["file"], src))
    return out


def write_call_line(raw: str) -> str | None:
    """The source line of the write call: the innermost frame's line, else the innermost REPO
    frame's line (library frames such as torch.save's open(name, mode) do not show the write
    mode). None if neither is a write call."""
    frames = _frame_sources(raw)
    candidates = [frames[-1][1]] if frames else []
    repo = [src for f, src in frames if is_repo_file(f)]
    if repo:
        candidates.append(repo[-1])
    return next((src for src in candidates if src and is_write_call(src)), None)


def is_write_call(source_line: str) -> bool:
    return bool(_OPEN_WRITE_RE.search(source_line) or _WRITE_CALL_RE.search(source_line))


def container_path(path: str) -> str:
    """Normalized container path of a path as the repo used it (cwd /work)."""
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(WORK, path))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _arg(arguments: Sequence[BudgetArg], dests: Sequence[str]) -> BudgetArg | None:
    for d in dests:
        for a in arguments:
            if (
                a.dest == d
                and a.action in (None, "store")
                and any(o.startswith("-") for o in a.option_strings)
            ):
                return a
    return None


def _current_value(command: str, arg: BudgetArg) -> str | None:
    """Value of arg in the command (space or '=' form), else its argparse default."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()
    for i, tok in enumerate(argv):
        if tok in arg.option_strings and i + 1 < len(argv):
            return argv[i + 1]
        for o in arg.option_strings:
            if tok.startswith(o + "="):
                return tok.split("=", 1)[1]
    return None if arg.default is None else str(arg.default)


def _option(arg: BudgetArg) -> str:
    opts = [o for o in arg.option_strings if o.startswith("-")]
    return next((o for o in opts if o.startswith("--")), opts[0])


def _local_tops(repo_files: Set[str]) -> set[str]:
    tops: set[str] = set()
    for p in repo_files:
        head, sep, _ = p.partition("/")
        tops.add(head if sep else head.removesuffix(".py"))
    return tops


def classify(smoke: Mapping[str, Any], ctx: ClassifyContext) -> Classification:
    """Classify a FAILED smoke result (its evidence content). Raises ValueError if passed."""
    if smoke.get("passed"):
        raise ValueError("classify() is for failed smoke results")
    signal = smoke.get("signal")
    tb: Mapping[str, Any] | None = smoke.get("traceback")
    exc = str(tb.get("exc_type", "")) if tb else ""
    msg = str(tb.get("message", "")) if tb else ""
    raw = str(tb.get("raw", "")) if tb else ""
    kind = tb.get("kind") if tb else None

    if signal == "infra" or (tb and _INFRA_RE.search(f"{exc}: {msg}\n{raw}")):
        return Classification(
            cls="infra",
            pattern="infra",
            reason=f"infrastructure failure ({signal}): {exc}: {msg}".rstrip(": "),
        )

    if signal == "oom" or kind == "oom":
        batch = _arg(ctx.arguments, BATCH_DESTS)
        if batch is None:
            return Classification(
                cls="unfixable", pattern="oom", reason="out of memory and no batch argument"
            )
        current = _current_value(str(smoke.get("command", "")), batch)
        if current is None or not current.isdigit():
            return Classification(
                cls="unfixable", pattern="oom",
                reason=f"out of memory; current {_option(batch)} value unknown ({current!r})",
            )  # fmt: skip
        accum = _arg(ctx.arguments, ACCUM_DESTS)
        params: dict[str, Any] = {
            "option": _option(batch),
            "dest": batch.dest,
            "current": int(current),
        }
        if accum is not None:
            acc = _current_value(str(smoke.get("command", "")), accum)
            params |= {
                "accum_option": _option(accum),
                "accum_dest": accum.dest,
                "accum_current": int(acc) if acc and acc.isdigit() else 1,
            }
        return Classification(
            cls="known", pattern="oom", params=params,
            reason=f"out of memory at {_option(batch)} {current}",
        )  # fmt: skip

    if tb is None or kind not in ("python", "cuda"):
        return Classification(
            cls="unfixable",
            reason=f"no traceback and no known pattern (signal {signal}"
            + (f", {exc}: {msg}" if tb else "")
            + ")",
        )

    if exc in ("ModuleNotFoundError", "ImportError"):
        m = _NO_MODULE_RE.search(msg)
        if m:
            top = m["mod"].split(".", 1)[0]
            if top not in _local_tops(ctx.repo_files):
                return Classification(
                    cls="known", pattern="missing_module",
                    params={"module": top, "package": MODULE_TO_PIP.get(top, top)},
                    reason=f"module {top!r} is not installed",
                )  # fmt: skip

    if exc == "FileNotFoundError":
        m = _NO_FILE_RE.search(msg)
        path = m["path"] if m else None
        if path is not None:
            target = container_path(path)
            source_line = write_call_line(raw)
            in_mount = any(_under(target, r) for r in ctx.mounted) or _under(target, "/work/data")
            if source_line is not None:
                if in_mount:
                    return Classification(
                        cls="unfixable", pattern="write_to_data_mount",
                        params={"path": path},
                        reason=f"write into the read-only data mount: {target}",
                    )  # fmt: skip
                if _under(target, WORK + "/"):
                    rel_dir = posixpath.dirname(target[len(WORK) + 1 :])
                    if rel_dir and not any(f.startswith(rel_dir + "/") for f in ctx.repo_files):
                        return Classification(
                            cls="known", pattern="missing_output_dir",
                            params={"dir": rel_dir, "path": path, "line": source_line},
                            reason=f"output dir {rel_dir}/ does not exist in the clone",
                        )  # fmt: skip
            else:
                for root, name in sorted(ctx.dataset_roots.items()):
                    if _under(target, root) and root not in ctx.mounted:
                        if name is not None:
                            return Classification(
                                cls="known", pattern="missing_data_path",
                                params={"name": name, "container": root, "path": path},
                                reason=f"dataset root {root} is not mounted ({name})",
                            )  # fmt: skip
                        return Classification(
                            cls="unfixable", pattern="missing_data_path",
                            params={"container": root, "path": path},
                            reason=f"dataset root {root} not mounted and not in the registry",
                        )  # fmt: skip

    frame = tb.get("repo_frame")
    where = f" at {frame['file']}:{frame['line']}" if frame else ""
    return Classification(
        cls="critic",
        params={"repo_frame": frame},
        reason=f"{exc}: {msg.splitlines()[0] if msg else ''}{where}",
    )
