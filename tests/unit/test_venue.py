from types import SimpleNamespace

from ra_agent.ocr import venue
from ra_agent.schema import PaperMeta

META = PaperMeta(title="Test, then Route: How Language Models Execute Rules", arxiv_id="2601.00001")
XREF = SimpleNamespace(cache_dir=None, timeout_s=1.0)


class Web:
    model = "web"

    def __init__(self, reply, urls):
        self.reply, self.urls, self.calls = reply, urls, 0

    def search(self, prompt):
        self.calls += 1
        return self.reply, self.urls


def run(monkeypatch, tmp_path, cands, verifier=None, meta=META):
    monkeypatch.setattr(venue, "candidates", lambda *a: cands)
    return venue.refine(meta, verifier, SimpleNamespace(cache_dir=tmp_path, timeout_s=1.0), 0.9)


def rec(source, venue_="", doi=""):
    return {"title": META.title, "venue": venue_, "year": "2026", "doi": doi, "source": source}


def test_record_venue_wins_and_the_web_model_is_not_called(monkeypatch, tmp_path):
    web = Web("", [])
    out = run(monkeypatch, tmp_path, [rec("semanticscholar", "ICLR", "10.1/x"), rec("openalex", "CoRR")], web)
    assert (out.venue, out.doi, web.calls) == ("ICLR", "10.1/x", 0)


def test_no_record_venue_falls_back_to_a_cited_web_answer(monkeypatch, tmp_path):
    url = "https://openreview.net/forum?id=abc"
    out = run(monkeypatch, tmp_path, [rec("openalex", "arXiv")], Web(f'{{"venue": "ICLR 2026", "url": "{url}"}}', [url]))
    assert out.venue == "ICLR 2026" and out.doi == "10.48550/arXiv.2601.00001"


def test_web_answer_without_a_cited_source_is_dropped(monkeypatch, tmp_path):
    out = run(monkeypatch, tmp_path, [], Web('{"venue": "NeurIPS 2026", "url": "https://made.up/x"}', ["https://other.org"]))
    assert out.venue == ""  # the current venue stays


def test_semantic_scholar_fields_reach_the_meta(monkeypatch, tmp_path):
    s2 = {**rec("semanticscholar", "ICLR", "10.1/x"), "abstract": "We show.", "citation_count": 7}
    out = run(monkeypatch, tmp_path, [s2])
    assert (out.abstract, out.citation_count) == ("We show.", 7)
