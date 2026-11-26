"""Code repository links of a paper: PDF link annotations plus URLs in the text layer."""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import CheckResult, Severity

_HOSTS = ("github.com", "gitlab.com", "huggingface.co", "bitbucket.org", "zenodo.org", "codeocean.com")
_URL = re.compile(
    r"(?<![\w.@-])(?:https?://)?(?:www\.)?(?:%s)/[^\s<>\"')\]]+" % "|".join(map(re.escape, _HOSTS)), re.I
)
_REFS = re.compile(r"^[ #]*(?:\d+\.?\s*)?(references|bibliography)(?:\s+\d+)?\s*$", re.I | re.M)
MAX_LINKS = 5


def normalize(url: str) -> str:
    url = url.split("#", 1)[0].rstrip(".,;:)]}'\"/")
    url = re.sub(r"\.git$", "", url).rstrip("/")
    url = re.sub(r"^(?:https?://)?(?:www\.)?([^/]+)", lambda m: "https://" + m.group(1).lower(), url, flags=re.I)
    return url


def _is_code(url: str) -> bool:
    host = url.split("/")[2].lower() if url.count("/") >= 2 else ""
    return host in _HOSTS and url.count("/") >= 4  # host plus owner/repo at least


def code_links(pdf_path: str | Path, limit: int = MAX_LINKS) -> list[str]:
    """Ordered by first page seen; URLs inside the references section are skipped.

    ponytail: the section runs from the first "References" heading to the end,
    so a link in an appendix after the references is missed.
    """
    import fitz

    found: list[str] = []
    in_refs = False
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text, y_cut = page.get_text("text"), float("inf")
            if in_refs:
                text, y_cut = "", 0.0
            elif heading := _REFS.search(text):
                in_refs = True
                hits = page.search_for(heading.group().strip())
                text, y_cut = text[: heading.start()], hits[-1].y0 if hits else 0.0
            raw = [l["uri"] for l in page.get_links() if l.get("uri") and l["from"].y0 < y_cut]
            found += [u for u in map(normalize, raw + _URL.findall(text)) if _is_code(u)]
    return list(dict.fromkeys(found))[:limit]


def code_links_check(links: list[str]) -> CheckResult:
    return CheckResult(
        name="code_links", passed=True, severity=Severity.INFO,
        detail=", ".join(links) or "none found",
    )
