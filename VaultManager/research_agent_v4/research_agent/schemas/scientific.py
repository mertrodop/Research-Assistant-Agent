"""Persistent scientific records constructed by Python, not the LLM."""
from typing import Literal
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .extraction import Verdict

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class Record(BaseModel):
    model_config = ConfigDict(extra='forbid')

class EvidenceRef(Record):
    section_id: str
    unit_id: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode='after')
    def ordered(self):
        if self.line_end < self.line_start:
            raise ValueError('Reversed evidence span')
        return self

class Fact(Record):
    fact_id: str
    text: str
    origin_section_id: str
    origin_unit_id: str
    evidence: list[EvidenceRef]
    source_candidate_id: str | None = None
    verification_status: Literal['checker_supported_provisional','verified','disputed','superseded'] = 'checker_supported_provisional'
    derived_from_fact_ids: list[str] = Field(default_factory=list)
    supplemented_by: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    checked_by: str
    created_at: str = Field(default_factory=utc_now)

class CandidateRecord(Record):
    candidate_id: str
    text: str
    origin_section_id: str
    origin_unit_id: str
    evidence: list[EvidenceRef]
    verdict: Verdict
    reason: str
    accepted_fact_id: str | None = None
    created_at: str = Field(default_factory=utc_now)

class CandidateAnchor(Record):
    candidate_id: str
    accepted_fact_id: str | None
    verdict: Verdict
    checker_reason: str

class Question(Record):
    question_id: str
    question: str
    origin_section_id: str
    origin_unit_id: str
    motivating_candidate_ids: list[str]
    motivating_fact_ids: list[str]
    motivating_candidates: list[CandidateAnchor]
    status: Literal['open','investigating','answered','partial','unanswerable'] = 'open'
    answer: str | None = None
    answer_fact_ids: list[str] = Field(default_factory=list)
    answer_evidence: list[EvidenceRef] = Field(default_factory=list)
    related_question_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)

class KnowledgeStore(Record):
    paper_id: str
    facts: dict[str, Fact] = Field(default_factory=dict)
    candidate_facts: dict[str, CandidateRecord] = Field(default_factory=dict)
    questions: dict[str, Question] = Field(default_factory=dict)
    # Reserved collections; their Stage II schemas are intentionally not implemented.
    clusters: dict = Field(default_factory=dict)
    contradictions: dict = Field(default_factory=dict)
