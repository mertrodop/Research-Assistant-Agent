"""Small wire schemas; durable records retain complete provenance."""
from typing import Literal
from pydantic import Field
from ..schemas.extraction import StrictOutput
from ..schemas.scientific import Record, KnowledgeStore
from ..core.config import ModelConfig

Action = Literal['MERGE','ANSWER','INVESTIGATE','QUESTION','AUDIT','REPAIR','COVERAGE','CONTRADICTION']
Kind = Literal['fact','derive','merge','correct','answer','partial_answer','question','boundary','not_found','contradiction']
class Task(StrictOutput):
    action: Action
    prompt: str = Field(min_length=1, max_length=1200)
    facts: list[str]
    questions: list[str]
    candidates: list[str]
    sections: list[str]

class Plan(StrictOutput):
    tasks: list[Task]
    reason: str = Field(max_length=600)
    # Empty tasks means stop. Python assigns IDs.

class Span(StrictOutput):
    a: int = Field(ge=1)
    b: int = Field(ge=1)

class Proposal(StrictOutput):
    kind: Kind
    text: str = Field(min_length=1,max_length=2400)
    facts: list[str]
    questions: list[str]
    candidates: list[str]
    spans: list[Span]

class Result(StrictOutput):
    proposals: list[Proposal]
    note: str = Field(max_length=900)

class Check(StrictOutput):
    i: int = Field(ge=0)
    ok: bool
    reason: str = Field(max_length=600)

class Review(StrictOutput):
    checks: list[Check]

class Config(Record):
    objective: str = "Improve completeness, correctness and scientific depth; prioritize consequential uncertainties and knowledge boundaries."
    planner: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=4000,temperature=.2,extra_body={'reasoning':{'effort':'low'}}))
    executor: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=4000,temperature=.2,extra_body={'reasoning':{'effort':'low'}}))
    verifier: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=2400,temperature=0,extra_body={'reasoning':{'enabled':False}}))
    max_rounds: int = Field(default=6,ge=1)
    max_tasks_per_round: int = Field(default=3,ge=1,le=5)
    max_sections_per_task: int = Field(default=3,ge=1)
    max_selected_facts: int = Field(default=16,ge=2)
    max_proposals: int = Field(default=6,ge=1)
    max_llm_calls: int = Field(default=43,ge=1)
    max_reported_tokens: int = Field(default=120000,ge=1)
    max_output_repairs: int = Field(default=1,ge=0)
    max_no_progress: int = Field(default=2,ge=1)
    max_context_chars: int = Field(default=24000,ge=1000)
    max_planner_chars: int = Field(default=45000,ge=1000)
    recent_episodes: int = Field(default=6,ge=1)
    max_new_questions: int = Field(default=18,ge=0)
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
    # Full durable records are not repeatedly sent to the model.
    types: dict[str,str] = Field(default_factory=dict)
    assessments: dict[str,dict] = Field(default_factory=dict)
    question_evidence: dict[str,list[dict]] = Field(default_factory=dict)
    episodes: list[dict] = Field(default_factory=list)
    plans: list[dict] = Field(default_factory=list)
