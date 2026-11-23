from __future__ import annotations

import json

from ra_agent.config import ReferenceSettings
from ra_agent.ocr.crossref import CrossrefClient
from ra_agent.ocr.refagent import (
    ReferenceAgent,
    extract_identity,
    llm_identity,
    resolve_references,
)
from ra_agent.schema import Reference

TITLE = "Grounded writing assistants for scientific papers"
ITEM = {"title": [TITLE], "issued": {"date-parts": [[2024]]}}


class StubCrossref(CrossrefClient):
    """Records which strategy was used and returns scripted answers."""

    def __init__(self, doi_item=None, search_items=None, tmp_path=None):
        super().__init__(cache_dir=tmp_path or ".cache/test-crossref")
        self.doi_item = doi_item
        self.search_items = search_items or []
        self.doi_calls: list[str] = []
        self.search_calls: list[str] = []

    def lookup_doi(self, doi: str):
        self.doi_calls.append(doi)
        return self.doi_item

    def search(self, query: str):
        self.search_calls.append(query)
        return self.search_items


class FakeLlm:
    model = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.reply

    def describe(self, image_path, prompt):  # pragma: no cover
        raise AssertionError("not used")


def ref(raw: str, title: str = TITLE, year: str = "2024") -> Reference:
    return Reference(raw=raw, title=title, year=year)


def test_doi_is_extracted_from_raw_entry():
    identity = extract_identity(ref("Tosun, E. 2024. doi:10.1234/abcd.5678"))
    assert identity.doi == "10.1234/abcd.5678"
    assert identity.source == "regex"


def test_arxiv_id_is_extracted():
    identity = extract_identity(ref("Tosun, E. arXiv:2404.13161v2"))
    assert identity.arxiv == "2404.13161"


def test_doi_strategy_wins_first(tmp_path):
    stub = StubCrossref(doi_item=ITEM, tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun. doi:10.1234/abcd"))
    assert check.status == "verified"
    assert check.strategy == "doi"
    assert stub.search_calls == []


def test_title_strategy_when_no_identifier(tmp_path):
    stub = StubCrossref(search_items=[ITEM], tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun, E. 2024. Grounded writing."))
    assert check.status == "verified"
    assert check.strategy == "title"


def test_arxiv_strategy_uses_doi_prefix(tmp_path):
    stub = StubCrossref(doi_item=ITEM, tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun. arXiv:2404.13161"))
    assert check.status == "verified"
    assert check.strategy == "arxiv"
    assert stub.doi_calls == ["10.48550/arXiv.2404.13161"]


def test_circuit_breaker_stops_after_budget(tmp_path):
    stub = StubCrossref(search_items=[], tmp_path=tmp_path)
    check = ReferenceAgent(stub, ReferenceSettings(max_attempts=2)).resolve(
        ref("Tosun, E. 2024. Grounded writing.")
    )
    assert check.status == "unresolved"
    assert check.strategy == "exhausted"
    assert len(stub.search_calls) == 2


def test_attempts_are_recorded(tmp_path):
    stub = StubCrossref(search_items=[], tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun, E. 2024. Grounded writing."))
    assert any(attempt.startswith("title:") for attempt in check.attempts)
    assert any(attempt.startswith("title_year:") for attempt in check.attempts)


def test_llm_identity_rescues_author_only_entry(tmp_path):
    stub = StubCrossref(search_items=[ITEM], tmp_path=tmp_path)
    llm = FakeLlm(
        json.dumps(
            {
                "title": TITLE,
                "authors": "Tosun, E.",
                "year": "2024",
                "doi": "",
                "arxiv": "",
            }
        )
    )
    agent = ReferenceAgent(stub, ReferenceSettings(max_attempts=4), llm)
    check = agent.resolve(ref("Tosun, E., Tuygan, E., Rodop, M.", title="Tosun, E., Tuygan, E."))
    assert llm.calls == 1
    assert check.status == "verified"
    assert check.strategy == "llm_title"


def test_llm_identity_rejects_unusable_reply():
    llm = FakeLlm(json.dumps({"title": "", "authors": "", "year": "", "doi": "", "arxiv": ""}))
    assert llm_identity(ref("Tosun, E."), llm) is None


def test_llm_identity_handles_bad_json():
    llm = FakeLlm("not json")
    assert llm_identity(ref("Tosun, E."), llm) is None


def test_no_llm_means_no_llm_call(tmp_path):
    stub = StubCrossref(search_items=[], tmp_path=tmp_path)
    check = ReferenceAgent(stub, ReferenceSettings(max_attempts=4)).resolve(
        ref("Tosun, E. 2024. Grounded writing.")
    )
    assert check.strategy == "exhausted"


def test_best_mismatch_survives_exhaustion(tmp_path):
    near = {"title": ["Grounded writing assistants for scientific papers"], "issued": {"date-parts": [[2019]]}}
    stub = StubCrossref(search_items=[near], tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun, E. 2024. Grounded writing."))
    assert check.status == "mismatch"
    assert check.strategy == "exhausted"
    assert check.score > 0.9
    assert check.attempts


def test_weak_candidate_becomes_unresolved(tmp_path):
    weak = {"title": ["A study of protein folding dynamics"], "issued": {"date-parts": [[2024]]}}
    stub = StubCrossref(search_items=[weak], tmp_path=tmp_path)
    check = ReferenceAgent(stub).resolve(ref("Tosun, E. 2024. Grounded writing."))
    assert check.status == "unresolved"
    assert any(attempt.startswith("floor:") for attempt in check.attempts)


def test_resolve_references_maps_all(tmp_path):
    stub = StubCrossref(search_items=[ITEM], tmp_path=tmp_path)
    checks = resolve_references(
        [ref("a. doi:10.1/x"), ref("b. Grounded writing.")], stub
    )
    assert len(checks) == 2
    assert all(check.status == "verified" for check in checks)
