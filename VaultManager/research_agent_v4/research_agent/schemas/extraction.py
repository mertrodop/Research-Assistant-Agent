"""Only LLM-facing schemas; indices are local, zero-based."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal['supported', 'partial', 'unsupported']

class StrictOutput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class CandidateFact(StrictOutput):
    text: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode='after')
    def ordered(self):
        if self.line_end < self.line_start:
            raise ValueError('Reversed evidence span')
        return self

class ReaderOutput(StrictOutput):
    facts: list[CandidateFact]

class FactCheck(StrictOutput):
    fact_index: int = Field(ge=0)
    verdict: Verdict
    reason: str

class NewQuestion(StrictOutput):
    question: str
    motivated_by_fact_indices: list[int]

class CheckerOutput(StrictOutput):
    fact_checks: list[FactCheck]
    new_questions: list[NewQuestion]
