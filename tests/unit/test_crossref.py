from __future__ import annotations

import json

from ra_agent.ocr.crossref import (
    CrossrefClient,
    Reference,
    check_references,
    split_references,
)

MARKDOWN = """# Paper

Body text.

## References

[1] Tosun, E. (2024). Grounded writing assistants for scientific papers. Journal of Agents, 12(3), 1-20.
[2] Tuygan, E. (2023). Citation validation for large language models. In Proceedings of ACL, 100-110.
"""


def test_split_references_finds_entries():
    refs = split_references(MARKDOWN)
    assert len(refs) == 2
    assert refs[0].year == "2024"
    assert "Grounded writing assistants" in refs[0].title


def test_split_references_without_section():
    assert split_references("# Paper\n\nNo bibliography here.") == []


def test_split_references_stops_at_next_heading():
    text = MARKDOWN + "\n## Appendix\n\n[3] Not a reference at all but long enough to pass.\n"
    refs = split_references(text)
    assert len(refs) == 2


def test_verified_when_title_and_year_match(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    client._fetch = lambda query: [
        {
            "title": ["Grounded writing assistants for scientific papers"],
            "issued": {"date-parts": [[2024]]},
        }
    ]
    check = client.check(
        Reference(raw="x", title="Grounded writing assistants for scientific papers", year="2024")
    )
    assert check.status == "verified"
    assert check.score > 0.9


def test_mismatch_on_year(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    client._fetch = lambda query: [
        {
            "title": ["Grounded writing assistants for scientific papers"],
            "issued": {"date-parts": [[2019]]},
        }
    ]
    check = client.check(
        Reference(raw="x", title="Grounded writing assistants for scientific papers", year="2024")
    )
    assert check.status == "mismatch"
    assert "year" in check.detail


def test_mismatch_on_title(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    client._fetch = lambda query: [
        {"title": ["A completely different study of protein folding"], "issued": {"date-parts": [[2024]]}}
    ]
    check = client.check(
        Reference(raw="x", title="Grounded writing assistants for scientific papers", year="2024")
    )
    assert check.status == "mismatch"


def test_unresolved_when_no_hits(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    client._fetch = lambda query: []
    check = client.check(Reference(raw="x", title="Some plausible paper title here", year="2024"))
    assert check.status == "unresolved"


def test_short_title_is_skipped(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    check = client.check(Reference(raw="x", title="Short", year="2024"))
    assert check.status == "skipped"


def test_cache_roundtrip(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    items = [{"title": ["Cached title"], "issued": {"date-parts": [[2024]]}}]
    cache = client._cache_path("query")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items), encoding="utf-8")
    assert client._fetch("query") == items


def test_check_references_uses_client(tmp_path):
    client = CrossrefClient(cache_dir=tmp_path)
    client._fetch = lambda query: [
        {"title": ["Grounded writing assistants for scientific papers"], "issued": {"date-parts": [[2024]]}}
    ]
    checks = check_references(MARKDOWN, client)
    assert len(checks) == 2
    assert checks[0].status == "verified"
