from pydantic import Field
from .scientific import Record, KnowledgeStore, utc_now
from ..stage1.source import ReadingUnit
from .extraction import NewQuestion

class RejectedQuestion(Record):
    question_index: int
    proposal: NewQuestion
    reason: str

class UnitReview(Record):
    unit_id: str
    section_id: str
    accepted_fact_ids: list[str]
    candidate_fact_ids: list[str]
    new_question_ids: list[str]
    rejected_questions: list[RejectedQuestion]
    reader_log: str
    checker_log: str
    completed_at: str = Field(default_factory=utc_now)

class ReadingCheckpoint(Record):
    schema_version: str = 'stage1-v2.1'
    run_fingerprint: str
    source_sha256: str
    reading_plan: list[ReadingUnit]
    skipped_sections: list[dict]
    reviews: dict[str,UnitReview] = Field(default_factory=dict)
    knowledge: KnowledgeStore
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def completed_unit_ids(self):
        return [u.unit_id for u in self.reading_plan if u.unit_id in self.reviews]

    @property
    def reading_complete(self):
        return len(self.completed_unit_ids)==len(self.reading_plan)
