import fitz

from ra_agent.ocr.codelinks import code_links, code_links_check, normalize


def _pdf(tmp_path):
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Code release", fontsize=11)
    p1.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 90, 200, 110),
                    "uri": "https://github.com/openai/preparedness.git#readme"})
    p2 = doc.new_page()
    p2.insert_text((72, 100), "See https://gitlab.com/a/b), and huggingface.co/x/y.", fontsize=11)
    p2.insert_text((72, 300), "References", fontsize=11)
    p2.insert_text((72, 330), "[1] https://github.com/cited/paper", fontsize=11)
    p3 = doc.new_page()
    p3.insert_text((72, 100), "https://github.com/later/ref", fontsize=11)
    path = tmp_path / "p.pdf"
    doc.save(path)
    return path


def test_links_from_annotation_and_text_skip_references(tmp_path):
    assert code_links(_pdf(tmp_path)) == [
        "https://github.com/openai/preparedness",
        "https://gitlab.com/a/b",
        "https://huggingface.co/x/y",
    ]


def test_cap_and_normalize(tmp_path):
    assert normalize("http://www.github.com/a/b.git.") == "https://github.com/a/b"
    assert len(code_links(_pdf(tmp_path), limit=2)) == 2


def test_check_detail():
    assert code_links_check([]).detail == "none found"
    assert code_links_check(["https://github.com/a/b"]).passed


def test_gist_host_is_not_github():
    from ra_agent.ocr.codelinks import _URL

    assert _URL.findall("see gist.github.com/a/b and https://gist.github.com/a/b") == []
    assert _URL.findall("at (github.com/a/b)") == ["github.com/a/b"]


def test_normalize_git_suffix_and_host_case():
    assert normalize("https://GitHub.com/A/B.git/") == "https://github.com/A/B"
    assert normalize("HTTPS://www.GitHub.com/a/b.git") == "https://github.com/a/b"


def test_numbered_references_heading(tmp_path):
    for heading in ("7 References", "References 12", "## 7. Bibliography"):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Code at github.com/a/code", fontsize=11)
        page.insert_text((72, 300), heading, fontsize=11)
        page.insert_text((72, 330), "[1] github.com/c/cited", fontsize=11)
        path = tmp_path / "r.pdf"
        doc.save(path)
        assert code_links(path) == ["https://github.com/a/code"], heading
