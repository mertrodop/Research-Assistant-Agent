from __future__ import annotations

import logging

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
    link = f"{staged.slug}/fig-1.png"
    assert f"]({link})" in text
    assert (staged.note_path.parent / link).read_bytes() == b"png"  # link matches file
    assert yaml.safe_load(text.split("---")[1])["figure_files"] == [link]

    note = vault.commit(staged.slug)
    assert note == vault.root / "Papers" / f"{staged.slug}.md"
    assert (vault.root / "Papers" / link).is_file()
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
    assert vault.read(one.slug, max_chars=10) == (vault.root / "Papers" / f"{one.slug}.md").read_text(encoding="utf-8")[:10]


def test_frontmatter_round_trips():
    document = doc(meta=PaperMeta(title="T", authors=["A"], year="2025", source="arxiv"))
    text = render_note(document, "slug")
    front = NoteFrontmatter(**yaml.safe_load(text.split("---")[1]))
    assert front.metadata == "verified"
    assert front.engines == {"docling": 1, "granite": 0, "text_layer": 0}
    assert front.authors == ["A"]
    assert doc().model_dump() and "unverified" in render_note(doc(), "s")


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
    text = render_note(doc().model_copy(update={"pages": [ok, bad], "references": refs}), "s")
    assert "| 2 | granite" in text and "| 1 | docling" not in text
    assert "bad ref" in text and "good ref" not in text


def test_raw_pdf_renamed_on_commit(vault, tmp_path):
    pdf = tmp_path / "raw.pdf"
    pdf.write_bytes(b"%PDF")
    staged = vault.stage(doc(source_pdf=str(pdf)))
    vault.commit(staged.slug)
    renamed = tmp_path / f"{staged.slug}.pdf"
    assert renamed.is_file() and not pdf.exists()
    note = (vault.root / "Papers" / f"{staged.slug}.md").read_text(encoding="utf-8")
    assert yaml.safe_load(note.split("---")[1])["source_pdf"] == str(renamed)


def test_raw_pdf_rename_never_overwrites(vault, tmp_path, caplog):
    pdf = tmp_path / "raw.pdf"
    pdf.write_bytes(b"new")
    staged = vault.stage(doc(source_pdf=str(pdf)))
    target = tmp_path / f"{staged.slug}.pdf"
    target.write_bytes(b"old")
    with caplog.at_level(logging.WARNING):
        vault.commit(staged.slug)
    assert target.read_bytes() == b"old" and pdf.read_bytes() == b"new"
    note = (vault.root / "Papers" / f"{staged.slug}.md").read_text(encoding="utf-8")
    assert yaml.safe_load(note.split("---")[1])["source_pdf"] == str(pdf)
    assert "already exists" in caplog.text


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
