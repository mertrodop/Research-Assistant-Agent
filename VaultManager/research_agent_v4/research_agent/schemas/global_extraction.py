"""Global reader contract. Checker reuses the original local-index contract."""
from pydantic import Field, model_validator
from .extraction import StrictOutput

class EvidenceSpan(StrictOutput):
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode='after')
    def ordered(self):
        if self.line_end < self.line_start:
            raise ValueError('Reversed evidence span')
        return self

class GlobalCandidateFact(StrictOutput):
    text: str = Field(min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)

class GlobalReaderOutput(StrictOutput):
    facts: list[GlobalCandidateFact]
