from __future__ import annotations

from ra_agent.ocr.assemble import assemble, section_tree, strip_margins
from ra_agent.schema import FigureInterpretation, FigureRef, PageResult, Route


def page(no: int, markdown: str) -> PageResult:
    return PageResult(page_no=no, route=Route.TEXT_LAYER, markdown=markdown)


def test_repeated_headers_and_footers_are_stripped():
    pages = [
        page(1, "Journal of Agents\nBody one\nPage 1"),
        page(2, "Journal of Agents\nBody two\nPage 2"),
        page(3, "Journal of Agents\nBody three\nPage 3"),
    ]
    cleaned = strip_margins(pages)
    assert [p.markdown for p in cleaned] == ["Body one", "Body two", "Body three"]


def test_short_documents_keep_margins():
    pages = [page(1, "Header\nBody\nFooter"), page(2, "Header\nBody two\nFooter")]
    assert strip_margins(pages)[0].markdown == "Header\nBody\nFooter"


def test_dehyphenation_across_page_break():
    pages = [page(1, "a trans-\nformer"), page(2, "model")]
    assert "transformer" in assemble(pages)


def test_pages_are_ordered():
    pages = [page(2, "second"), page(1, "first")]
    assert assemble(pages) == "first\n\nsecond"


def test_section_tree():
    tree = section_tree("# A\ntext\n## B\n### C\n")
    assert tree == [(1, "A"), (2, "B"), (3, "C")]


def fig(index: int = 1) -> FigureRef:
    return FigureRef(
        page_no=1, index=index, filename=f"fig-{index}.png", width=400, height=300
    )


def test_figure_replaces_image_marker():
    markdown = assemble([page(1, "A\n\n<!-- image -->\n\nB")], {1: [fig()]}, "figures")
    assert "<!-- image -->" not in markdown
    assert markdown.index("A") < markdown.index("![Figure 1") < markdown.index("B")
    assert "(figures/fig-1.png)" in markdown


def test_extra_figure_is_appended_and_extra_marker_removed():
    text = "A\n\n<!-- image -->\n\n<!-- image -->\n\nB"
    markdown = assemble([page(1, text)], {1: [fig(1)]}, "figures")
    assert markdown.count("![Figure") == 1
    assert "<!-- image -->" not in markdown
    markdown = assemble([page(1, "A\n\nB")], {1: [fig(1), fig(2)]}, "figures")
    assert markdown.endswith("(figures/fig-2.png)")
    assert markdown.index("B") < markdown.index("![Figure 1")


def test_unavailable_description_is_rendered():
    failed = fig().model_copy(update={"interpretation": FigureInterpretation(error="down")})
    markdown = assemble([page(1, "A\n\n<!-- image -->")], {1: [failed]}, "figures")
    assert "> ⚠ description unavailable" in markdown


def test_figure_sits_at_its_caption_with_the_caption_verbatim():
    figure = FigureRef(
        page_no=1, index=1, filename="fig-1.png", width=10, height=10, label="Figure 7",
        caption="Figure 7: Multi-token answers.",
        interpretation=FigureInterpretation(kind="chart", summary="Lines rise.", markdown="| | |\n|---|---|\n| | |"),
    )
    text = "Intro.\n\nFigure 6: other.\n\nFigure 7: Multi-token\nanswers (Gemma).\n\nFigure 7 shows more."
    markdown = assemble([page(1, text)], {1: [figure]}, "figures")
    assert markdown.index("Figure 6: other.") < markdown.index("![") < markdown.index("Figure 7: Multi-token answers (Gemma).")
    assert markdown.index("Figure 7: Multi-token answers (Gemma).") < markdown.index("> [!figure] chart") < markdown.index("Figure 7 shows more.")
    assert markdown.count("Figure 7: Multi-token") == 1  # the caption paragraph is not duplicated
    assert "|---|" not in markdown  # an empty table reading is dropped


def test_text_layer_line_breaks_are_joined_inside_paragraphs():
    text = "## Limitations\n\nWe test three\nmodels in English\nonly.\n\n| a | b |\n| 1 | 2 |"
    markdown = assemble([page(1, text).model_copy(update={"engine": "text_layer"})], None, "figures")
    assert "We test three models in English only." in markdown
    assert "| a | b |\n| 1 | 2 |" in markdown


def test_references_become_one_bullet_per_entry():
    text = (
        "## References\n\nAndy Arditi and Neel Nanda.\n2024. Refusal in language models. Advances in NeurIPS, 37:1-9.\n\n"
        "- Collin Burns and Jacob Steinhardt. 2023. Discovering latent knowledge. In ICLR.\n\n"
        "Xander Davies and David Bau. 2023. Discovering variable binding. arXiv:2307.03637.\n\n## A Appendix\n\nText."
    )
    out = assemble([page(1, text)], None, "figures")
    assert out.count("\n- ") == 3
    assert "- Andy Arditi and Neel Nanda. 2024. Refusal" in out
    assert out.index("- Xander") < out.index("## A Appendix")


def test_fragment_heading_joins_the_paragraph_above():
    out = assemble([page(1, "Intro text (Tezuka\n\n## and Inoue, 2025).\n\n## Next\n\nBody")], None, "figures")
    assert "(Tezuka and Inoue, 2025)." in out and "## Next" in out and "## and" not in out


def test_hyphens_that_docling_dropped_come_back(tmp_path):
    import fitz

    from ra_agent.ocr.assemble import restore_hyphens

    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "We patch the last-token residual and the lasting one.", fontsize=10)
    pdf = tmp_path / "h.pdf"
    doc.save(pdf)
    out = restore_hyphens("the lasttoken residual, lasting", pdf)
    assert out == "the last-token residual, lasting"


def test_paragraph_cut_by_a_column_break_is_rejoined():
    from ra_agent.ocr.assemble import reflow

    text = "## Ethics Statement\n\nThe stimuli are machine-rendered and the\n\nscaffold is translated. Done.\n\nNew paragraph."
    assert reflow(text) == "## Ethics Statement\n\nThe stimuli are machine-rendered and the scaffold is translated. Done.\n\nNew paragraph."
