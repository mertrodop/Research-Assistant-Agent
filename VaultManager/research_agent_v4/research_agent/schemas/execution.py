"""Structured worker outputs, persistent records, and minimal round state."""
from typing import Literal
from typing_extensions import TypedDict
from pydantic import Field
from .extraction import StrictOutput
from .scientific import Record, EvidenceRef, KnowledgeStore, utc_now
from .planning import Task, MAX_SECTIONS_PER_TASK

class InformationRequest(StrictOutput):
    reason: str
    fact_ids: list[str]
    question_ids: list[str]
    section_ids: list[str] = Field(max_length=MAX_SECTIONS_PER_TASK)

class BaseExecutorResult(StrictOutput):
    status: Literal['success','partial','failed','insufficient_context']
    task_assessment: Literal['appropriate','inappropriate_grouping','insufficient_context','unnecessary','other_problem']
    outcome_summary: str
    evidence_refs: list[EvidenceRef]
    remaining_uncertainties: list[str]
    information_requests: list[InformationRequest]
    failure_reason: str | None

class FactProposal(StrictOutput):
    text: str
    evidence: list[EvidenceRef]
    derived_from_fact_ids: list[str]

class MergeProposal(FactProposal):
    supersede_ids: list[str]

class FactIssue(StrictOutput):
    fact_id: str
    explanation: str
    evidence: list[EvidenceRef]

class ContradictionProposal(StrictOutput):
    fact_ids: list[str]
    verdict: Literal['contradiction','context_difference','consistent','extraction_error','unresolved']
    explanation: str
    evidence: list[EvidenceRef]

class QuestionResolution(StrictOutput):
    question_id: str
    status: Literal['answered','partial','unresolved','unanswerable']
    explanation: str
    answer_fact_ids: list[str]
    evidence: list[EvidenceRef]

class GroupAnswer(StrictOutput):
    answer: str
    resolutions: list[QuestionResolution]

class QuestionProposal(StrictOutput):
    question: str
    motivating_fact_ids: list[str]
    evidence: list[EvidenceRef]

class CoverageGap(StrictOutput):
    section_id: str
    description: str
    evidence: list[EvidenceRef]

class ConsolidationResult(BaseExecutorResult):
    merge_disposition: Literal['merged','partially_merged','not_mergeable','potential_contradiction','unsupported','insufficient_context']
    merged_facts: list[MergeProposal]
    fact_issues: list[FactIssue]

class ConsistencyResult(BaseExecutorResult):
    contradictions: list[ContradictionProposal]
    fact_issues: list[FactIssue]

class QuestionResult(BaseExecutorResult):
    answers: list[GroupAnswer]

class InvestigationResult(BaseExecutorResult):
    new_facts: list[FactProposal]
    new_questions: list[QuestionProposal]

class CoverageResult(BaseExecutorResult):
    gaps: list[CoverageGap]
    new_facts: list[FactProposal]

class ProposalCheck(StrictOutput):
    proposal_index: int = Field(ge=0)
    verdict: Literal['supported','partially_supported','unsupported','contradicted']
    reason: str

class VerificationResult(StrictOutput):
    checks: list[ProposalCheck]

class VerificationRecord(Record):
    # Python creates this record, not the verifier LLM.
    status: Literal['completed','skipped','unavailable']
    checks: list[ProposalCheck] = Field(default_factory=list)
    log: str | None = None
    reason: str = ''
    input_hash: str

    @property
    def eligible_indices(self):
        if self.status != 'completed': return set()
        return {c.proposal_index for c in self.checks if c.verdict == 'supported'}

class Contradiction(Record):
    contradiction_id: str
    assessment: ContradictionProposal
    task_id: str

class Episode(Record):
    attempt_id: str
    round_number: int
    task_id: str
    action: str
    objective: str
    outcome_summary: str
    status: str
    task_assessment: str
    evaluation: str
    evidence_refs: list[EvidenceRef]
    information_requests: list[InformationRequest]
    remaining_uncertainties: list[str]
    failure_reason: str | None
    accepted_updates: list[str]
    verification: VerificationRecord
    created_at: str = Field(default_factory=utc_now)

class Stage2Knowledge(KnowledgeStore):
    contradictions: dict[str, Contradiction] = Field(default_factory=dict)
    question_evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    coverage_gaps: dict[str, CoverageGap] = Field(default_factory=dict)
    answer_groups: dict[str, GroupAnswer] = Field(default_factory=dict)
    fact_issues: dict[str, FactIssue] = Field(default_factory=dict)

class Ledger(Record):
    knowledge: Stage2Knowledge
    episodes: dict[str, Episode] = Field(default_factory=dict)
    round_commits: dict[str, list[str]] = Field(default_factory=dict)
    revision: int = 0

class TaskRecord(Record):
    task: Task
    round_number: int
    status: Literal['pending','running','observed','completed','partial','failed','cancelled'] = 'pending'
    attempt_id: str | None = None
    outcome: str = ''

class OrchestratorState(TypedDict):
    objective: str
    strategy: dict | None
    round_number: int
    round_task_ids: list[str]
    round_observation: str
    tasks: dict[str, dict]
    active_task_id: str | None
    attempt_id: str | None
    attempts: int
    no_progress: int
    context_path: str | None
    result_paths: dict[str, str]
    evaluation: dict | None
    route: str
    termination_reason: str
    summary_path: str | None
