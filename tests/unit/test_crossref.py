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


def test_wrapped_author_list_stays_one_entry():
    markdown = """References

Cai, D., Modi, C., Pillaud-Vivien, L., Margossian, C.,
Gower, R. M., Blei, D., and Saul, L. K. Batch and match: black-box variational inference. 2024.
Gloeckler, M., Deistler, M., Weilbach, C. D., Wood, F., and
Macke, J. H. All-in-one simulation-based inference. 2024.
"""
    from ra_agent.ocr.crossref import split_references

    entries = split_references(markdown)
    assert len(entries) == 2
    assert entries[0].raw.startswith("Cai, D.") and "Batch and match" in entries[0].raw
    assert "All-in-one" in entries[1].raw


def test_list_does_not_stop_at_initial_or_shebang():
    from ra_agent.ocr.crossref import split_references

    markdown = """References

Sanchez, G., Spangher, A., Fan, H., Levi, E., and Biderman,
S. Stay on Topic with Classifier-Free Guidance. 2024.

#!/ usr/bin/env bash

Zhou, H., Zhang, Q. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge. 2024.
"""
    entries = split_references(markdown)
    assert any("Stay on Topic" in e.raw for e in entries)
    assert any("MLLM-as-a-Judge" in e.raw for e in entries)


def test_layout_blocks_are_trusted_and_appendix_heading_ends_list():
    from ra_agent.ocr.crossref import split_references

    markdown = """References

Bradley Brown, Jordan Juravsky, and Azalia
Mirhoseini. Large Language Monkeys: Scaling Inference Compute, 2024.

Edward Raff. A Step Toward Quantifying Independently Reproducible Machine Learning Research, 2019.

Matei Zaharia, Omar Khattab, and Lingjiao Chen. The Shift from Models to Compound AI Systems, 2024.

A Benchmark Details
To obtain a dataset of all capsules on CodeOcean we used the public API in 2024.
"""
    entries = split_references(markdown)
    assert len(entries) == 3
    assert "Large Language Monkeys" in entries[0].raw
