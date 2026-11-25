from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import CriticSettings
from ..llm.client import LlmClient, VisionError
from ..schema import CheckResult, Severity

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)

IMAGE_NOTE = """The attached image is the original page. Judge the OCR text against it:
text that is not on the page is a hallucination, and a number that differs from the
page is a fidelity error. Ignore reading order and line breaks.

"""

RUBRIC = """You are a strict reviewer of OCR output for one page of a scientific paper.

Page text produced by OCR:
---
{markdown}
---

Score four dimensions from 1 to 5:
- fidelity: does the text say what the page says, without invented content?
- completeness: are all sections, tables and equations present?
- structure: are headings, lists and tables well formed?
- no_hallucination: are there no fabricated sentences, numbers or citations?

Rules:
- Every error must quote its "ocr_span" exactly as it appears in the OCR text.
- Never invent content that is not in the text above.
- If a dimension cannot be judged, score it 3 and say why.

Answer with strict JSON only, no prose and no code fences:
{{
  "fidelity": <integer 1-5>,
  "completeness": <integer 1-5>,
  "structure": <integer 1-5>,
  "no_hallucination": <integer 1-5>,
  "errors": [
    {{"type": "hallucination | wrong_number | missing | structure",
      "ocr_span": "exact words copied from the OCR text, 4-15 words",
      "page_says": "what the page shows instead, empty for hallucination"}}
  ],
  "reason": "one sentence"
}}
5 means no problem at all, 1 means the dimension is badly wrong.
List every concrete error in "errors"; use [] when there is none. Keep the answer short.
"""

DIMENSIONS = ("fidelity", "completeness", "structure", "no_hallucination")
BLOCKING_ERRORS = {"hallucination", "wrong_number"}


def _in_text(span: str, text: str) -> bool:
    words = re.findall(r"\w+", span.lower())
    return len(words) >= 3 and " ".join(words) in " ".join(re.findall(r"\w+", text.lower()))


def _parse(raw: str) -> dict:
    cleaned = _FENCE.sub("", raw or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0:
        raise ValueError("no JSON object in response")
    candidate = cleaned[start : end + 1] if end > start else cleaned[start:]
    return json.loads(candidate)


def _clamp(value: object) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(1, min(5, number))


def critique_page(
    markdown: str,
    client: LlmClient,
    settings: CriticSettings,
    page_no: int = 0,
    image_path: str | Path | None = None,
    text_layer: str = "",
) -> CheckResult:
    """Tier 2 rubric critic. Returns a warn/blocker check, never raises.

    With image_path the critic compares the text to the page itself; without it
    the critic only sees the OCR text and cannot judge fidelity. With text_layer,
    a claimed error whose span is in the PDF text layer is discarded: that text is
    on the page, so the claim is the critic's own mistake.
    """
    if not markdown.strip():
        return CheckResult(
            name="critic",
            passed=False,
            severity=Severity.BLOCKER,
            detail="empty page, nothing to critique",
        )
    prompt = RUBRIC.format(markdown=markdown[:6000])
    try:
        if image_path is not None:
            raw = client.describe(image_path, IMAGE_NOTE + prompt)
        else:
            raw = client.complete(prompt)
    except VisionError as exc:
        return CheckResult(
            name="critic",
            passed=False,
            severity=Severity.WARN,
            detail=f"critic unavailable: {str(exc)[:200]}",
        )
    try:
        data = _parse(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="critic",
            passed=False,
            severity=Severity.WARN,
            detail=f"critic returned invalid JSON: {str(exc)[:160]}",
        )

    scores = {name: _clamp(data.get(name)) for name in DIMENSIONS}
    lowest = min(scores.values())
    average = sum(scores.values()) / len(scores)
    # Fix list (semantic gradient): an error counts only when its quoted span is
    # really in the OCR text, so the critic cannot block a page by hallucinating.
    errors = [
        e
        for e in (data.get("errors") or [])
        if isinstance(e, dict) and str(e.get("type", "")) in BLOCKING_ERRORS
    ]
    grounded = [
        e
        for e in errors
        if _in_text(str(e.get("ocr_span", "")), markdown)
        and not (text_layer and _in_text(str(e.get("ocr_span", "")), text_layer))
    ]
    failed = (
        lowest < settings.min_dimension
        or average < settings.min_average
        or bool(grounded)
    )
    detail = " ".join(f"{name}={score}" for name, score in scores.items())
    detail += f" avg={average:.2f} min={lowest} errors={len(grounded)}/{len(errors)} grounded"
    evidence = "; ".join(
        f"{e.get('type')}: {str(e.get('ocr_span'))[:80]!r} -> {str(e.get('page_says', ''))[:60]!r}"
        for e in grounded[:3]
    )
    return CheckResult(
        name="critic",
        passed=not failed,
        severity=Severity.BLOCKER if failed else Severity.INFO,
        detail=detail,
        evidence=evidence[:300],
    )


def critique_pages(
    pages: list,
    client: LlmClient,
    settings: CriticSettings,
) -> list:
    """Attach a critic check to every page that already passed Tier 1."""
    if not settings.enabled:
        return pages
    updated = []
    for page in pages:
        if page.status != "verified":
            updated.append(page)
            continue
        result = critique_page(page.markdown, client, settings, page.page_no)
        reasons = page.review_reasons + ([] if result.passed else [result.name])
        updated.append(
            page.model_copy(
                update={
                    "checks": page.checks + [result],
                    "review_reasons": reasons,
                    "status": "verified" if result.passed else "needs-review",
                }
            )
        )
    return updated


def write_report(checks: list, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "reference": check.reference.raw[:300],
            "status": check.status,
            "score": check.score,
            "matched_title": check.matched_title,
            "matched_year": check.matched_year,
            "detail": check.detail,
        }
        for check in checks
    ]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target
