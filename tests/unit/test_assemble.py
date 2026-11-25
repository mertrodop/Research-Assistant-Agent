from __future__ import annotations

from ra_agent.ocr.assemble import assemble, dehyphenate, section_tree, strip_margins
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
