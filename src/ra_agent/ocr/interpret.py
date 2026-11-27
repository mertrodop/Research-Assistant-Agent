from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import VisionExtra
from ..llm.client import LlmClient, VisionError
from ..schema import FigureInterpretation, FigureRef

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_LABEL = re.compile(r"^(?:figure|fig\.?)\s*\d+\s*[.:]?\s*", re.I)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)

PROMPT = """You are reading one figure from a scientific paper.

Caption (may be empty):
{caption}

Surrounding page text (for context only):
{context}

Describe only what is visible in the image. Rules:
- Never invent numbers, axis labels, model names or trends.
- Every number you report must be readable in the image.
- If a value is unreadable, omit it instead of guessing.
- "markdown" holds the figure content as Markdown: a table as a GFM table,
  a chart as a table of the values you can read, a diagram as a short list of
  nodes and edges, an equation as LaTeX. Use an empty string for a plain photo.

Answer with strict JSON only, no prose and no code fences:
{{
  "kind": "chart|table|diagram|photo|equation|other",
  "summary": "one or two sentences on what the figure shows",
  "markdown": "figure content as Markdown"
}}
"""


COMPACT_HINT = """

Your previous answer was not valid JSON. Reply again with compact JSON only:
keep markdown under 15 lines and escape newlines inside strings.
"""


def _numbers(text: str) -> set[str]:
    found: set[str] = set()
    for token in _NUMBER.findall(text or ""):
        found.add(token.replace(",", ""))
    return found


def _parse_json(raw: str) -> dict:
    cleaned = _FENCE.sub("", raw or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0:
        raise ValueError("no JSON object in response")
    candidate = cleaned[start : end + 1] if end > start else cleaned[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_repair_json(candidate))


def _repair_json(text: str) -> str:
    """Close a truncated object and drop trailing commas, the two common failures."""
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


_KINDS = {"chart", "table", "diagram", "photo", "equation", "other"}


def ground(interpretation: FigureInterpretation, evidence: str) -> list[str]:
    """Numbers the model reported that the caption or page text never mentions."""
    known = _numbers(evidence)
    reported = _numbers(interpretation.summary) | _numbers(interpretation.markdown)
    return sorted(number for number in reported if number not in known)


def _read(figure: FigureRef, image_path: Path, page_text: str, client: LlmClient) -> FigureInterpretation:
    prompt = PROMPT.format(
        caption=figure.caption or "(none)",
        context=(page_text or "")[:2500],
    )
    try:
        raw = client.describe(image_path, prompt)
    except VisionError as exc:
        return FigureInterpretation(model=client.model, error=str(exc)[:300])

    try:
        data = _parse_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        try:
            raw = client.describe(image_path, prompt + COMPACT_HINT)
            data = _parse_json(raw)
        except (VisionError, ValueError, json.JSONDecodeError):
            return FigureInterpretation(
                model=client.model, error=f"invalid JSON: {exc}"[:300]
            )

    kind = str(data.get("kind", "other"))
    reading = FigureInterpretation(
        kind=kind if kind in _KINDS else "other",
        summary=str(data.get("summary", ""))[:600],
        markdown=str(data.get("markdown", "")).strip()[:4000],
        model=client.model,
    )
    reading.ungrounded = ground(reading, f"{figure.caption}\n{page_text}")
    return reading


def _cache_path(root: Path, sha: str, filename: str, model: str, image: bytes = b"") -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    prompt = hashlib.sha256((PROMPT + COMPACT_HINT).encode()).hexdigest()[:8]  # a new prompt is a miss
    pixels = hashlib.sha256(image).hexdigest()[:8]  # same name, new crop: a miss
    return root / sha / f"{Path(filename).stem}__{pixels}__{slug}__{prompt}.json"


def describe_figure(
    figure: FigureRef,
    image_dir: str | Path,
    page_text: str,
    client: LlmClient,
    extra: VisionExtra,
    sha: str,
) -> FigureRef:
    """Attach the vision reading to one figure, using the per-figure disk cache."""
    image_path = Path(image_dir) / figure.filename
    image = image_path.read_bytes() if image_path.is_file() else b""
    cache = _cache_path(Path(extra.cache_dir), sha, figure.filename, client.model, image)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return figure.model_copy(
                update={"interpretation": FigureInterpretation.model_validate(data)}
            )
        except ValueError:
            pass  # stale or corrupt cache entry: read the figure again

    if not image_path.is_file():
        reading = FigureInterpretation(model=client.model, error="image file missing")
        return figure.model_copy(update={"interpretation": reading})
    reading = _read(figure, image_path, page_text, client)
    if not reading.error:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(reading.model_dump_json(indent=1), encoding="utf-8")
    return figure.model_copy(update={"interpretation": reading})


def describe_figures(
    figures: list[FigureRef],
    image_dir: str | Path,
    texts: dict[int, str],
    client: LlmClient,
    extra: VisionExtra,
    sha: str,
) -> list[FigureRef]:
    """Describe every figure in parallel, keeping reading order."""
    if not figures:
        return figures
    with ThreadPoolExecutor(max_workers=max(1, extra.concurrency)) as pool:
        return list(
            pool.map(
                lambda f: describe_figure(
                    f, image_dir, texts.get(f.page_no, ""), client, extra, sha
                ),
                figures,
            )
        )


def _alt(figure: FigureRef) -> str:
    caption = _LABEL.sub("", figure.caption or "").strip()
    sentence = re.split(r"(?<=[.!?])\s", caption, maxsplit=1)[0].replace("]", ")")
    return f"Figure {figure.index}: {sentence}" if sentence else f"Figure {figure.index}"


def _content_rows(markdown: str) -> list[str]:
    """Drop Markdown tables whose cells are all empty (a model reading of nothing)."""
    rows, run = [], []
    for row in markdown.splitlines() + [""]:
        if row.lstrip().startswith("|"):
            run.append(row)
            continue
        if any(re.search(r"[^\s|:-]", r) for r in run) or not run:
            rows.extend(run)
        run = []
        rows.append(row)
    return rows[:-1]


def figure_block(figure: FigureRef, rel_dir: str, caption: str | None = None) -> str:
    """Image, the caption verbatim when given, then a quoted reading; a failed reading
    gets a single warning line."""
    lines = [f"![{_alt(figure)}]({rel_dir}/{figure.filename})"]
    if caption:
        lines += ["", caption]
    reading = figure.interpretation
    if reading is None:
        return "\n".join(lines)
    lines.append("")
    if reading.error:
        lines.append("> ⚠ description unavailable")
        return "\n".join(lines)
    lines.append(f"> [!figure] {reading.kind}")
    lines.append(f"> {reading.summary}")
    rows = _content_rows(reading.markdown)
    if any(r.strip() for r in rows):
        lines.append(">")
        lines.extend(f"> {row}".rstrip() for row in rows)
    if reading.ungrounded:
        lines += [
            "",
            "> [!warning]- Numbers not found in page text",
            f"> {', '.join(reading.ungrounded)} — read from the image; check against the figure.",
        ]
    return "\n".join(lines)
