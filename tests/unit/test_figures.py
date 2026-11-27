from __future__ import annotations

import fitz

from ra_agent.ocr.figures import extract_figures


def make_pdf_with_figure(path, caption: str = "Figure 1. A bar chart of scores.") -> str:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 60, 545, 120), "Body text before the figure.", fontsize=11)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 300), 0)
    pix.set_rect(pix.irect, (30, 90, 200))
    page.insert_image(fitz.Rect(80, 150, 480, 450), pixmap=pix)
    page.insert_textbox(fitz.Rect(80, 460, 520, 520), caption, fontsize=10)
    doc.save(path)
    doc.close()
    return str(path)


def test_extracts_raster_figure_and_links_caption(tmp_path):
    pdf = make_pdf_with_figure(tmp_path / "fig.pdf")
    out = tmp_path / "figures"
    figures = extract_figures(pdf, 1, out)
    assert len(figures) == 1
    figure = figures[0]
    assert figure.kind == "raster"
    assert figure.label == "Figure 1"
    assert "bar chart" in figure.caption
    assert (out / figure.filename).is_file()


def test_tiny_images_are_skipped(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
    page.insert_image(fitz.Rect(80, 150, 100, 170), pixmap=pix)
    page.insert_textbox(fitz.Rect(80, 200, 520, 260), "Figure 1. Tiny icon.", fontsize=10)
    pdf = tmp_path / "tiny.pdf"
    doc.save(pdf)
    doc.close()

    figures = extract_figures(str(pdf), 1, tmp_path / "out", page_render_fallback=False)
    assert figures == []


def test_vector_only_page_is_cropped_to_its_caption(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(80, 150, 480, 450), color=(0, 0, 0), fill=(0.4, 0.6, 0.9))
    page.insert_textbox(fitz.Rect(80, 460, 520, 520), "Figure 2. Vector diagram.", fontsize=10)
    pdf = tmp_path / "vector.pdf"
    doc.save(pdf)
    doc.close()

    figures = extract_figures(str(pdf), 1, tmp_path / "out")
    assert len(figures) == 1
    assert figures[0].kind == "vector"
    assert figures[0].height < 400  # the drawing, not the page
    assert figures[0].label == "Figure 2"
    assert (tmp_path / "out" / figures[0].filename).is_file()


def test_page_without_figures_returns_empty(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 60, 545, 400), "Only text here.", fontsize=11)
    pdf = tmp_path / "text.pdf"
    doc.save(pdf)
    doc.close()
    assert extract_figures(str(pdf), 1, tmp_path / "out") == []


def test_two_vector_figures_on_a_page_get_one_crop_each(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(80, 80, 480, 240), fill=(0.4, 0.6, 0.9))
    page.insert_textbox(fitz.Rect(80, 250, 520, 280), "Figure 1. First.", fontsize=10)
    page.draw_rect(fitz.Rect(80, 400, 480, 560), fill=(0.9, 0.6, 0.4))
    page.insert_textbox(fitz.Rect(80, 570, 520, 600), "Figure 2. Second.", fontsize=10)
    pdf = tmp_path / "two.pdf"
    doc.save(pdf)
    doc.close()

    figures = extract_figures(str(pdf), 1, tmp_path / "out")
    assert [f.label for f in figures] == ["Figure 1", "Figure 2"]
    assert [f.filename for f in figures] == ["fig-1.png", "fig-2.png"]
    assert figures[0].bbox[3] < figures[1].bbox[1]


def test_text_layer_marks_bold_larger_lines_as_headings(tmp_path):
    from ra_agent.ocr.engine import NativeTextEngine

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Ethics Statement", fontname="hebo", fontsize=12)
    page.insert_text((72, 130), "This work uses no human subjects.", fontname="helv", fontsize=10)
    page.insert_text((300, 820), "9", fontname="helv", fontsize=10)
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    doc.close()

    text, _ = NativeTextEngine().ocr_page(pdf, 1)
    assert text.startswith("## Ethics Statement\n\nThis work uses no human subjects.")
    assert not text.rstrip().endswith("9")  # the page number is dropped
