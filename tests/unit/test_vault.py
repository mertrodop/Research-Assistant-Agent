from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from ra_agent.schema import (
    CheckResult,
    FigureRef,
    NoteFrontmatter,
    OcrDocument,
    PageResult,
    PaperMeta,
    ReferenceCheck,
    Reference,
    Route,
)
from ra_agent.vault.note import render_note
from ra_agent.vault.store import Vault, slugify

PAPERBENCH = "PaperBench: Evaluating AI’s Ability to Replicate AI Research"
CORE = (
    "CORE-Bench: Fostering the Credibility of Published Research Through a "
    "Computational Reproducibility Agent Benchmark"
)


def doc(title=PAPERBENCH, sha="a" * 64, body="We replicate research.", **kw) -> OcrDocument:
    page = PageResult(page_no=1, route=Route.DOCLING, markdown=body, engine="docling")
    return OcrDocument(
        source_pdf=kw.pop("source_pdf", "data/raw/x.pdf"),
        sha256=sha,
        title=title,
        pages=[page],
        markdown=f"# {title}\n\n{body}",
        **kw,
    )


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "vault", tmp_path / "staging")


def test_slugify():
    assert (
        slugify(PAPERBENCH, "x")
        == "paperbench-evaluating-ais-ability-to-replicate-ai-research"
    )
    assert slugify("", "2504.01848") == "2504-01848"
    slug = slugify(CORE, "x")
    assert slug == (
        "core-bench-fostering-the-credibility-of-published-research-through-a-"
        "computational-reproducibility"
    )
    assert len(slug) == 98
    assert len(slugify("word " * 60, "x")) <= 100


def test_titleless_document_uses_pdf_stem(vault):
    staged = vault.stage(doc(title="", source_pdf="data/raw/My Paper_v2.pdf"))
    assert staged.slug == "my-paper-v2"


def test_collisions_get_suffix_and_restage_reuses(vault):
    first = vault.stage(doc(sha="a" * 64))
    vault.commit(first.slug)
    second = vault.stage(doc(sha="b" * 64))
    assert second.slug == first.slug + "-2"
    assert vault.stage(doc(sha="b" * 64)).slug == second.slug  # same sha, same slug
    vault.commit(second.slug)
    assert vault.stage(doc(sha="c" * 64)).slug == first.slug + "-3"
    assert vault.stage(doc(sha="a" * 64)).slug == first.slug  # reuse across vault


def test_commit_moves_note_and_figures(vault, tmp_path):
    src = tmp_path / "pipeline_figs"
    src.mkdir()
    (src / "fig-1.png").write_bytes(b"png")
    figure = FigureRef(page_no=1, index=1, filename="fig-1.png", width=10, height=10)
    document = doc(body="text\n\n![Figure 1](figures/fig-1.png)")
    document.pages[0].figures = [figure]
    staged = vault.stage(document, figures_dir=src)
    text = staged.note_path.read_text(encoding="utf-8")
    assert "](../figures/fig-1.png)" in text
    assert (staged.note_path.parent / "../figures/fig-1.png").read_bytes() == b"png"  # link matches file
    assert yaml.safe_load(text.split("---")[1])["figure_files"] == ["figures/fig-1.png"]

    note = vault.commit(staged.slug)
    paper = vault.root / "Papers" / staged.slug
    assert note == paper / "md" / f"{staged.slug}.md"
    assert (note.parent / "../figures/fig-1.png").is_file()
    assert not staged.note_path.exists() and not staged.figure_dir.exists()


def test_commit_without_stage_raises(vault):
    with pytest.raises(FileNotFoundError):
        vault.commit("nothing")


def test_diff(vault):
    document = doc()
    staged = vault.stage(document)
    assert "title:" in vault.diff(staged.slug)  # new note: full text
    vault.commit(staged.slug)
    assert vault.diff(vault.stage(document).slug) == ""
    changed = vault.stage(doc(body="different body"))
    assert "-We replicate research." in vault.diff(changed.slug)


def test_search_ranks_by_hits_and_read(vault):
    one = vault.stage(doc(sha="a" * 64, body="replicate replicate replicate"))
    two = vault.stage(doc(title="Other Paper", sha="b" * 64, body="we replicate once"))
    vault.commit(one.slug)
    vault.commit(two.slug)
    hits = vault.search("REPLICATE")
    assert [h["slug"] for h in hits] == [one.slug, two.slug]
    assert set(hits[0]) == {"slug", "title", "line"}
    assert vault.search("zzzz") == []
    assert vault.read(one.slug, max_chars=10) == vault.note(vault.root, one.slug).read_text(encoding="utf-8")[:10]


def test_frontmatter_round_trips():
    document = doc(meta=PaperMeta(title="T", authors=["A"], year="2025", source="arxiv", code=["https://github.com/a/b"]))
    text = render_note(document)
    front = NoteFrontmatter(**yaml.safe_load(text.split("---")[1]))
    assert front.metadata == "verified"
    assert front.engines == {"docling": 1, "granite": 0, "vlm": 0, "text_layer": 0}
    assert front.authors == ["A"]
    assert front.code == ["https://github.com/a/b"]
    assert doc().model_dump() and "unverified" in render_note(doc())


def test_invalid_frontmatter_writes_nothing(vault):
    bad = doc().model_copy(update={"status": "ok"})
    with pytest.raises(ValueError):
        vault.stage(bad)
    assert not vault.staging.exists()


def test_note_lists_only_failed_pages_and_unverified_refs():
    ok = PageResult(page_no=1, route=Route.DOCLING, markdown="a")
    bad = PageResult(
        page_no=2,
        route=Route.GRANITE,
        markdown="b",
        status="needs-review",
        review_reasons=["coverage"],
        checks=[CheckResult(name="coverage", passed=False)],
    )
    refs = [
        ReferenceCheck(reference=Reference(raw="good ref"), status="verified"),
        ReferenceCheck(reference=Reference(raw="bad ref"), status="unresolved"),
    ]
    text = render_note(doc().model_copy(update={"pages": [ok, bad], "references": refs}))
    assert "| 2 | granite" in text and "| 1 | docling" not in text
    assert "bad ref" in text and "good ref" not in text


def _source(note: Path) -> str:
    return yaml.safe_load(note.read_text(encoding="utf-8").split("---")[1])["source_pdf"]


def test_pdf_moves_to_raw_on_commit(vault, tmp_path):
    pdf = tmp_path / "inbox" / "My Paper_v2.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF")
    staged = vault.stage(doc(source_pdf=str(pdf)))
    note = vault.commit(staged.slug)
    raw = vault.root / "Papers" / staged.slug / "raw" / f"{staged.slug}.pdf"
    assert raw.read_bytes() == b"%PDF" and not pdf.exists()
    assert _source(note) == f"Papers/{staged.slug}/raw/{staged.slug}.pdf"


def test_raw_pdf_never_overwritten(vault, tmp_path, caplog):
    pdf = tmp_path / "raw.pdf"
    pdf.write_bytes(b"new")
    staged = vault.stage(doc(source_pdf=str(pdf)))
    raw = vault.root / "Papers" / staged.slug / "raw" / f"{staged.slug}.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"old")
    with caplog.at_level(logging.WARNING):
        vault.commit(staged.slug)
    assert raw.read_bytes() == b"old" and pdf.read_bytes() == b"new"
    assert "already exists" in caplog.text


def test_recommit_of_a_paper_in_papers_keeps_its_raw_pdf(vault):
    raw = vault.root / "Papers" / "an-example" / "raw" / "an-example.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"%PDF")
    staged = vault.stage(doc(title="An Example", source_pdf=str(raw)))
    assert staged.slug == "an-example"
    vault.commit(staged.slug)
    assert raw.read_bytes() == b"%PDF"


def test_commit_cleans_pipeline_figures_and_staging(vault):
    document = doc(sha="d" * 64)
    pipeline_dir = vault.staging / "figures" / ("d" * 12)
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "fig-1.png").write_bytes(b"png")
    document.pages[0].figures = [
        FigureRef(page_no=1, index=1, filename="fig-1.png", width=1, height=1)
    ]
    vault.commit(vault.stage(document, figures_dir=pipeline_dir).slug)
    assert not [p for p in vault.staging.rglob("*") if p.is_file()]
    assert not pipeline_dir.exists()


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "A b", ""])
def test_bad_slugs_rejected(vault, bad):
    for call in (vault.read, vault.diff, vault.commit):
        with pytest.raises(ValueError):
            call(bad)


def test_note_drops_heading_that_repeats_the_title():
    document = doc()
    document.markdown = "## PaperBench: Evaluating AI's Ability to Replicate AI Research\n\nBody."
    text = render_note(document)
    assert "## PaperBench" not in text and "\n# PaperBench" in text and "Body." in text
    document.markdown = "## Abstract\n\nBody."
    assert "## Abstract" in render_note(document)


def test_validation_lists_verified_page_with_no_text_layer_warning():
    document = doc()
    document.pages[0].review_reasons = ["no_text_layer"]
    assert "| 1 | docling | 1 | verified | no_text_layer |" in render_note(document)


def test_commit_resolves_relative_source_against_the_vault(vault, tmp_path, monkeypatch):
    raw = vault.root / "Papers" / "an-example" / "raw" / "an-example.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"%PDF")
    staged = vault.stage(doc(title="An Example", source_pdf="Papers/an-example/raw/an-example.pdf"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    note = vault.commit(staged.slug)
    assert raw.read_bytes() == b"%PDF" and _source(note) == "Papers/an-example/raw/an-example.pdf"
