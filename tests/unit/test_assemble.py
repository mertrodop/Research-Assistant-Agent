from __future__ import annotations

from ra_agent.ocr.assemble import assemble, dehyphenate, section_tree, strip_margins
from ra_agent.schema import FigureInterpretation, FigureRef, PageResult, Route


def page(no: int, markdown: str) -> PageResult:
    return PageResult(page_no=no, route=Route.NATIVE_TEXT, markdown=markdown)


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


def test_figure_image_is_inserted_above_its_caption():
    figures = {
        1: [
            FigureRef(
                page_no=1,
                index=1,
                filename="p001_fig1.png",
                width=400,
                height=300,
                caption="Figure 1. A bar chart.",
                label="Figure 1",
            )
        ]
    }
    markdown = assemble([page(1, "Body text.\nFigure 1. A bar chart.")], figures)
    assert "![Figure 1](figures/p001_fig1.png)" in markdown
    assert markdown.index("![Figure 1]") < markdown.index("Figure 1. A bar chart.")


def test_figure_without_caption_is_appended():
    figures = {
        1: [
            FigureRef(
                page_no=1,
                index=1,
                filename="p001_fig1.png",
                width=400,
                height=300,
                caption="",
                label="",
            )
        ]
    }
    markdown = assemble([page(1, "Body text.")], figures)
    assert markdown.startswith("Body text.")
    assert "![Figure 1](figures/p001_fig1.png)" in markdown


def test_interpretation_is_rendered_under_image():
    figures = {
        1: [
            FigureRef(
                page_no=1,
                index=1,
                filename="p001_fig1.png",
                width=400,
                height=300,
                caption="Figure 1. Scores.",
                label="Figure 1",
                interpretation=FigureInterpretation(
                    summary="Bar chart of scores.",
                    kind="chart",
                    key_values=[{"label": "best", "value": "21%"}],
                    takeaways=["human baseline is higher"],
                    ungrounded=["87"],
                    model="fake",
                ),
            )
        ]
    }
    markdown = assemble([page(1, "Figure 1. Scores.")], figures)
    assert "![Figure 1](figures/p001_fig1.png)" in markdown
    assert "> **chart** — Bar chart of scores." in markdown
    assert "> - best: 21%" in markdown
    assert "> - human baseline is higher" in markdown
    assert "not corroborated by page text: 87" in markdown
    assert markdown.index("![Figure 1]") < markdown.index("> **chart**")


def test_failed_reading_is_shown_as_warning():
    figures = {
        1: [
            FigureRef(
                page_no=1,
                index=1,
                filename="p001_fig1.png",
                width=400,
                height=300,
                caption="Figure 1. Scores.",
                label="Figure 1",
                interpretation=FigureInterpretation(error="HTTP 401", model="fake"),
            )
        ]
    }
    markdown = assemble([page(1, "Figure 1. Scores.")], figures)
    assert "> reading failed: HTTP 401" in markdown


def test_shared_caption_numbers_panels():
    figures = {
        1: [
            FigureRef(
                page_no=1,
                index=1,
                filename="p001_fig1.png",
                width=400,
                height=300,
                caption="Figure 4. Two panels.",
                label="Figure 4",
            ),
            FigureRef(
                page_no=1,
                index=2,
                filename="p001_fig2.png",
                width=400,
                height=300,
                caption="Figure 4. Two panels.",
                label="Figure 4",
            ),
        ]
    }
    markdown = assemble([page(1, "Figure 4. Two panels.")], figures)
    assert "![Figure 4 (1/2)](figures/p001_fig1.png)" in markdown
    assert "![Figure 4 (2/2)](figures/p001_fig2.png)" in markdown


def test_dehyphenate_keeps_real_hyphens():
    assert dehyphenate("state-of-the-art") == "state-of-the-art"


def test_dehyphenate_keeps_compound_split_across_lines():
    words = frozenset({"state", "of", "the", "art"})
    assert dehyphenate("state-\nof-the-art", words) == "state-of-the-art"


def test_dehyphenate_joins_broken_word():
    words = frozenset({"state", "of", "the", "art"})
    assert dehyphenate("trans-\nformer", words) == "transformer"


def test_dehyphenate_without_wordlist_still_joins():
    assert dehyphenate("docu-\nment", frozenset()) == "document"
