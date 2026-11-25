from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import VisionExtra
from ..llm.client import LlmClient, VisionError
from ..schema import FigureInterpretation, FigureRef

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
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
- If the image is a logo, icon or decoration, set kind to "decoration" and keep it short.

Answer with strict JSON only, no prose and no code fences:
{{
  "summary": "one or two sentences on what the figure shows",
  "kind": "chart|diagram|table|screenshot|photo|decoration|other",
  "key_values": [{{"label": "what is measured", "value": "number as printed"}}],
  "takeaways": ["short factual observation"],
  "confidence": 0.0
}}
"""


COMPACT_HINT = """

Your previous answer was not valid JSON. Reply again with compact JSON only:
at most 3 key_values, at most 3 takeaways, no newlines inside strings.
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


def ground(interpretation: FigureInterpretation, evidence: str) -> list[str]:
    """Numbers the model reported that the caption or page text never mentions."""
    known = _numbers(evidence)
    reported: set[str] = set()
    for item in interpretation.key_values:
        reported |= _numbers(str(item.get("value", "")))
    reported |= _numbers(interpretation.summary)
    for takeaway in interpretation.takeaways:
        reported |= _numbers(takeaway)
    return sorted(number for number in reported if number not in known)


def interpret_figure(
    figure: FigureRef,
    image_path: str | Path,
    page_text: str,
    client: LlmClient,
    settings: VisionExtra,
) -> FigureInterpretation:
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

    key_values = []
    for item in data.get("key_values") or []:
        if isinstance(item, dict) and item.get("value") not in (None, ""):
            key_values.append(
                {
                    "label": str(item.get("label", ""))[:120],
                    "value": str(item.get("value"))[:60],
                }
            )

    interpretation = FigureInterpretation(
        summary=str(data.get("summary", ""))[:600],
        kind=str(data.get("kind", ""))[:40],
        key_values=key_values[:12],
        takeaways=[str(item)[:160] for item in (data.get("takeaways") or [])][:6],
        confidence=float(data.get("confidence") or 0.0),
        model=client.model,
    )
    interpretation.ungrounded = ground(
        interpretation, f"{figure.caption}\n{page_text}"
    )
    return interpretation


def _cache_path(root: Path, sha: str, filename: str, model: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    return root / sha / f"{Path(filename).stem}__{slug}.json"


def interpret_figures(
    figures: list[FigureRef],
    image_dir: str | Path,
    page_texts: dict[int, str],
    client: LlmClient,
    settings: VisionExtra,
    sha: str = "",
) -> list[FigureRef]:
    """Interpret every figure in parallel, with a per-figure disk cache."""
    if not figures:
        return figures
    image_dir = Path(image_dir)
    cache_root = Path(settings.cache_dir)

    def run(figure: FigureRef) -> FigureRef:
        cache = _cache_path(cache_root, sha, figure.filename, client.model)
        if cache.is_file():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                return figure.model_copy(
                    update={"interpretation": FigureInterpretation.model_validate(data)}
                )
            except (json.JSONDecodeError, ValueError):
                pass

        image_path = image_dir / figure.filename
        if not image_path.is_file():
            return figure.model_copy(
                update={
                    "interpretation": FigureInterpretation(
                        model=client.model, error="image file missing"
                    )
                }
            )
        interpretation = interpret_figure(
            figure, image_path, page_texts.get(figure.page_no, ""), client, settings
        )
        if not interpretation.error:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                interpretation.model_dump_json(indent=1), encoding="utf-8"
            )
        return figure.model_copy(update={"interpretation": interpretation})

    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
        return list(pool.map(run, figures))
