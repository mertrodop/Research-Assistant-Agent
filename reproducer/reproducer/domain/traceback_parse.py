"""Parse the LAST Python traceback out of a container log.

Pure domain: text in, model out. Rules:
  - carriage returns (tqdm redraws) are treated as line breaks, and a header that starts
    mid-line (after a progress bar without a trailing newline) starts the block;
  - chained exceptions ("During handling of the above exception..." / "The above exception
    was the direct cause...") -> the last "Traceback (most recent call last):" block wins;
  - frames: `File "<file>", line <n>, in <func>` lines of that block, outermost first;
  - the exception line is the first non-indented line after the frames; message = that line
    only (stdout and stderr interleave in container logs, so following lines may be
    unrelated output); up to MAX_CONTEXT_LINES following lines (until a blank line) are kept
    in raw as context, e.g. the second line of a CUDA error;
  - repo_frame: innermost frame in repo code: an absolute path under /work (not /work/data)
    or a relative path (python was started from /work, e.g. `File "train.py"`);
  - kind: "oom" (CUDA out of memory / OutOfMemoryError), "cuda" (other CUDA errors),
    "python" otherwise; with no traceback, a "Killed" line or exit code 137 gives kind
    "killed", and another non-zero exit code gives kind "exit" (last log line as message).
raw keeps the block (at most RAW_MAX bytes: its tail when longer).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

RAW_MAX = 4096
MAX_CONTEXT_LINES = 20  # lines after the exception line kept in raw (not in message)
WORK = "/work"
_HEADER = "Traceback (most recent call last):"
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?$')
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s?(?P<msg>.*))?$")
_CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)
_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE)
_CUDA_RE = re.compile(r"\bCUDA\b|cuDNN|CUBLAS|NCCL|cudaError", re.IGNORECASE)
_KILLED_RE = re.compile(r"^\s*Killed\s*$", re.MULTILINE)

TracebackKind = Literal["python", "oom", "cuda", "killed", "exit"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Frame(_Frozen):
    file: str
    line: int
    func: str


class ParsedTraceback(_Frozen):
    exc_type: str
    message: str
    frames: tuple[Frame, ...] = ()
    repo_frame: Frame | None = None
    raw: str
    kind: TracebackKind = "python"


def is_repo_file(path: str) -> bool:
    if path.startswith("<"):  # <string>, <frozen importlib._bootstrap>
        return False
    if not path.startswith("/"):
        return True  # relative to the /work cwd
    return path.startswith(WORK + "/") and not (
        path == "/work/data" or path.startswith("/work/data/")
    )


def _cap(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= RAW_MAX:
        return text
    return data[-RAW_MAX:].decode("utf-8", errors="ignore")


def _kind(exc_type: str, message: str) -> TracebackKind:
    text = f"{exc_type}: {message}"
    if _OOM_RE.search(text):
        return "oom"
    if _CUDA_RE.search(text):
        return "cuda"
    return "python"


def parse_last_traceback(log_text: str, exit_code: int | None = None) -> ParsedTraceback | None:
    """The last traceback in log_text; else a Killed / non-zero-exit record; else None."""
    text = log_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if _HEADER in ln and ln.strip() != _HEADER:
            lines[i] = ln[ln.index(_HEADER) :]  # e.g. a tqdm bar without a trailing newline
    starts = [i for i, ln in enumerate(lines) if ln.strip() == _HEADER]
    if not starts:
        return _without_traceback(lines, exit_code)
    i = starts[-1] + 1
    frames: list[Frame] = []
    while i < len(lines):
        ln = lines[i]
        m = _FRAME_RE.match(ln)
        if m:
            frames.append(
                Frame(file=m["file"], line=int(m["line"]), func=(m["func"] or "").strip())
            )
            i += 1
        elif ln.startswith((" ", "\t")) or not ln.strip():
            i += 1  # source line / caret line under a frame
        else:
            break
    exc_type, message, end = "UnknownError", "", i
    if i < len(lines):
        m = _EXC_RE.match(lines[i].strip())
        if m:
            exc_type, message = m["type"], (m["msg"] or "").strip()
        else:
            message = lines[i].strip()
        end = i + 1
        extra: list[str] = []
        while (
            end < len(lines)
            and len(extra) < MAX_CONTEXT_LINES
            and lines[end].strip()
            and lines[end].strip() not in (*_CHAIN_MARKERS, _HEADER)
        ):
            extra.append(lines[end].rstrip())
            end += 1
    repo = next((f for f in reversed(frames) if is_repo_file(f.file)), None)
    raw = "\n".join(lines[starts[-1] : end]).rstrip()
    return ParsedTraceback(
        exc_type=exc_type,
        message=message,
        frames=tuple(frames),
        repo_frame=repo,
        raw=_cap(raw),
        kind=_kind(exc_type, message),
    )


def _without_traceback(lines: list[str], exit_code: int | None) -> ParsedTraceback | None:
    text = "\n".join(lines)
    if _KILLED_RE.search(text) or exit_code == 137:
        return ParsedTraceback(
            exc_type="Killed",
            message="process killed (SIGKILL / exit 137)",
            raw="Killed" if _KILLED_RE.search(text) else "exit code 137",
            kind="killed",
        )
    if exit_code not in (None, 0):
        tail = [ln.rstrip() for ln in lines if ln.strip()][-5:]
        return ParsedTraceback(
            exc_type="SystemExit",
            message=tail[-1].strip() if tail else f"exit code {exit_code}",
            raw=_cap("\n".join(tail)),
            kind="exit",
        )
    return None
