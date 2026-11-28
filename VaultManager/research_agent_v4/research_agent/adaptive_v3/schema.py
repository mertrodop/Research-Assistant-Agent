"""Compact wire records; Python builds provenance and durable knowledge."""
from typing import Literal
from pydantic import Field, create_model
from ..schemas.extraction import StrictOutput
from ..schemas.scientific import Record, KnowledgeStore
from ..core.config import ModelConfig

Action = Literal['CONSOLIDATE','REASON','INVESTIGATE','QUESTION']
Kind = Literal['fact','derive','merge','correct','answer','partial_answer','question','boundary','not_found','contradiction']
class Task(StrictOutput):
    # Internal, resolved task. Python partitions the wire references.
    action: Action
    prompt: str
    facts: list[str]
    questions: list[str]
    candidates: list[str]
    assessments: list[str] = Field(default_factory=list)
    sections: list[str]

class TaskWire(StrictOutput):
    action: str
    prompt: str
    refs: list[str]
    sections: list[str]

class Plan(StrictOutput):
    tasks: list[TaskWire]
    defer: list[str]
    reason: str

class Span(StrictOutput):
    a: int
    b: int

# Internal records only; not sent as the executor schema.
class Proposal(StrictOutput):
    kind: Kind
    text: str
    facts: list[str]
    questions: list[str]
    candidates: list[str]
    assessments: list[str] = Field(default_factory=list)
    spans: list[Span]

class Result(StrictOutput):
    proposals: list[Proposal]
    note: str = Field(max_length=300)

def result_schema():
    """Separate findings from resolutions; Python validates each item, not whole batches."""
    # Keep wire types simple; semantic validation rejects individual invalid items.
    Finding=create_model('Finding',__base__=StrictOutput,kind=(str,...),text=(str,...),refs=(list[str],...),lines=(list[Span],...))
    Resolution=create_model('QuestionResolution',__base__=StrictOutput,q=(str,...),status=(str,...),text=(str,...),refs=(list[str],...),lines=(list[Span],...))
    return create_model('ScopedResult',__base__=StrictOutput,findings=(list[Finding],...),resolutions=(list[Resolution],...))

class Check(StrictOutput):
    i: int
    ok: bool
    reason: str

class Review(StrictOutput):
    checks: list[Check]

class Config(Record):
    objective: str = 'Improve scientific completeness, correctness, conciseness and depth within the available budget.'
    planner: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=6000,temperature=.2,extra_body={'reasoning':{'effort':'low'}}))
    executor: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=6000,temperature=.2,extra_body={'reasoning':{'effort':'low'}}))
    verifier: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=4000,temperature=0,extra_body={'reasoning':{'effort':'low'}}))
    max_rounds: int = Field(default=6,ge=1)
    max_tasks_per_round: int = Field(default=2,ge=1,le=5)
    max_sections_per_task: int = Field(default=3,ge=1)
    max_selected_facts: int = Field(default=12,ge=2)
    max_selected_questions: int = Field(default=6,ge=1)
    max_proposals: int = Field(default=5,ge=1)
    max_llm_calls: int = Field(default=36,ge=1)
    max_reported_tokens: int = Field(default=90000,ge=1)
    max_output_repairs: int = Field(default=0,ge=0)
    max_no_progress: int = Field(default=2,ge=1)
    max_context_chars: int = Field(default=24000,ge=1000)
    max_planner_chars: int = Field(default=45000,ge=1000)
    recent_episodes: int = Field(default=6,ge=1)
    max_new_questions: int = Field(default=12,ge=0)
    allow_partial_stage1: bool = False

class State(Record):
    knowledge: KnowledgeStore
    revision: int = 0
    round: int = 0
    pending: list[Task] = Field(default_factory=list)
    cursor: int = 0
    round_start_revision: int = 0
    no_progress: int = 0
    termination: str = ''
    types: dict[str,str] = Field(default_factory=dict)
    assessments: dict[str,dict] = Field(default_factory=dict)
    assessment_dependencies: dict[str,list[str]] = Field(default_factory=dict)
    question_evidence: dict[str,list[dict]] = Field(default_factory=dict)
    question_triage: dict[str,dict] = Field(default_factory=dict)
    episodes: list[dict] = Field(default_factory=list)
    plans: list[dict] = Field(default_factory=list)
