"""Round-based planning. A strategy is revisable; only current-round tasks execute."""
from typing import Literal
from pydantic import Field
from .extraction import StrictOutput

TaskAction = Literal['MERGE_FACTS','CHECK_CONTRADICTION','ANSWER_QUESTION',
    'INVESTIGATE_SOURCE','CHECK_COVERAGE']

MAX_SECTIONS_PER_TASK = 3

class Task(StrictOutput):
    task_id: str = Field(pattern=r'^T[0-9]+$')
    action: TaskAction
    task_prompt: str = Field(min_length=1)
    rationale: str
    success_criteria: list[str] = Field(min_length=1)
    fact_ids: list[str]
    question_ids: list[str]
    section_ids: list[str] = Field(max_length=MAX_SECTIONS_PER_TASK)

class StrategyStep(StrictOutput):
    objective: str
    rationale: str
    status: Literal['pending','in_progress','completed','abandoned']

class InvestigationStrategy(StrictOutput):
    overall_objective: str
    steps: list[StrategyStep]

class PlannerDecision(StrictOutput):
    strategy: InvestigationStrategy
    current_round_tasks: list[Task]
    terminate: bool
    termination_reason: str | None

class TaskEvaluation(StrictOutput):
    task_id: str
    accepted_proposal_indices: list[int]
    rationale: str
    task_satisfied: bool

class RoundEvaluation(StrictOutput):
    task_evaluations: list[TaskEvaluation]
    round_observation: str

class SynthesisClaim(StrictOutput):
    text: str
    fact_ids: list[str]

class SynthesisSection(StrictOutput):
    title: str
    claims: list[SynthesisClaim]

class Synthesis(StrictOutput):
    sections: list[SynthesisSection]
