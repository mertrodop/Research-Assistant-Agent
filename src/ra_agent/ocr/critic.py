from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import CriticSettings
from ..llm.client import LlmClient, VisionError
from ..schema import CheckResult, Severity

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)

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
- Quote the exact span you are judging in "evidence".
- Never invent content that is not in the text above.
- If a dimension cannot be judged, score it 3 and say why.

Answer with strict JSON only, no prose and no code fences:
{{
  "fidelity": 1,
  "completeness": 1,
  "structure": 1,
  "no_hallucination": 1,
  "evidence": "short quoted span",
  "reason": "one sentence"
}}
"""

DIMENSIONS = ("fidelity", "completeness", "structure", "no_hallucination")


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
) -> CheckResult:
    """Tier 2 rubric critic. Returns a warn/blocker check, never raises."""
    if not markdown.strip():
        return CheckResult(
            name="critic",
            passed=False,
            severity=Severity.BLOCKER,
            detail="empty page, nothing to critique",
        )
    prompt = RUBRIC.format(markdown=markdown[:6000])
    try:
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
    failed = lowest < settings.min_dimension or average < settings.min_average
    detail = " ".join(f"{name}={score}" for name, score in scores.items())
    detail += f" avg={average:.2f} min={lowest}"
    return CheckResult(
        name="critic",
        passed=not failed,
        severity=Severity.BLOCKER if failed else Severity.INFO,
        detail=detail,
        evidence=str(data.get("evidence", ""))[:200],
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
