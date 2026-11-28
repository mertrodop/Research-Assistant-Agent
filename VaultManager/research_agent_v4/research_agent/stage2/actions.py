"""Single action registry: contracts and specialized executor instructions."""
from dataclasses import dataclass
from ..schemas.planning import MAX_SECTIONS_PER_TASK as MAX_SECTIONS
from ..schemas.execution import (ConsolidationResult, ConsistencyResult, QuestionResult,
    InvestigationResult, CoverageResult)

@dataclass(frozen=True)
class ActionSpec:
    executor: str
    schema: type
    min_facts: int
    min_questions: int
    min_sections: int
    max_sections: int
    proposals: frozenset
    description: str
    worker_instruction: str

ACTIONS = {
    'MERGE_FACTS': ActionSpec('consolidation',ConsolidationResult,2,0,1,MAX_SECTIONS,frozenset({'merged_fact','fact_issue'}),
        'Assess compatible claims for grounded consolidation. Select all essential evidence sections. A merge is optional; preserve distinct claims, merge a subset, or flag extraction errors.',
        '''Compare every selected target with its original source. Identify overlapping and complementary claims, conditions, metrics, scope and qualifications. Merge only when the full scientific meaning of each superseded fact is preserved. Do not force a merge or assume compatibility from the task prompt. Return exact source evidence, derived_from_fact_ids and only justified supersede_ids. If a claim is unsupported, propose a fact issue with evidence; if evidence is insufficient, request context. A grounded finding that no merge is appropriate can be a successful assessment. Never claim a proposal has been committed.'''),
    'CHECK_CONTRADICTION': ActionSpec('consistency',ConsistencyResult,2,0,1,MAX_SECTIONS,frozenset({'contradiction','fact_issue'}),
        'Assess suspected contradictions using original sections. Distinguish actual conflict, different conditions, consistency, extraction error and unresolved.',
        '''Compare the original source for each selected claim, including datasets, metrics, experiment conditions, notation, qualifiers and scope. Distinguish genuine conflicts from context differences, extraction errors, consistency and insufficient evidence. Explain the specific relationship and cite evidence for both sides. Do not infer external truth or flag an extraction error solely because a manuscript claim is surprising. Return fact issues only when source supports the issue.'''),
    'ANSWER_QUESTION': ActionSpec('question',QuestionResult,1,1,0,0,frozenset({'answer'}),
        'Reason over selected existing facts, including mathematical/logical derivations. No source retrieval. Use this directly when sufficient facts exist; do not schedule acquisition unnecessarily.',
        '''Your responsibility is scientific reasoning, not source extraction. For EACH assigned question identify its essential components, combine complementary selected facts across sections and perform valid logical or mathematical derivations. State assumptions and distinguish derived conclusions from explicit manuscript claims. Use selected facts as knowledge without demanding their original sections; cite all essential supporting fact IDs and inherit their evidence only. Do not upgrade their verification status. Answered means all essential components supported; partial means at least one meaningful component supported and another unresolved; unresolved means no meaningful answer established. Do not add requirements not present in the original question. Return exactly one resolution per question even when grouping is inappropriate or context is insufficient; separate unrelated questions into different groups. Shared answer must not introduce claims absent from individual resolutions. Do not mark a question unanswerable for the entire manuscript from facts alone. Describe precisely what additional knowledge is needed.'''),
    'INVESTIGATE_SOURCE': ActionSpec('investigation',InvestigationResult,0,0,1,MAX_SECTIONS,frozenset({'new_fact','new_question'}),
        'Acquire targeted missing scientific knowledge from complete source sections. Optional questions guide acquisition; selected facts are reference knowledge for identifying gaps and avoiding duplicates. Never resolve questions here; schedule ANSWER_QUESTION after useful facts are committed.',
        '''Your responsibility is knowledge acquisition, not question answering. Identify the missing scientific information using the task prompt and optional questions. Examine all supplied complete sections for relevant definitions, equations, results, limitations and qualifications. Compare with selected reference facts to avoid redundant extraction and identify missing details. Every proposed fact must be directly supported by cited original source passages; reference facts are not source evidence. Extract only genuinely new reusable scientific facts, with precise line references; do not invent formulations or add speculative mechanisms. Return important new questions only if grounded in actual source and not redundant with supplied questions. Do not return question answers or resolutions. Report what was examined, what was found, what is already known and what remains missing. A negative finding is scoped to the parsed sections. Request a plausible unexamined section only when useful.'''),
    'CHECK_COVERAGE': ActionSpec('coverage',CoverageResult,0,0,1,MAX_SECTIONS,frozenset({'gap','new_fact'}),
        'Compare complete sections against all associated active facts. Explicitly select every active fact with evidence in any selected section; Python validates completeness. Identify material omissions and directly recover simple facts.',
        '''Compare supplied complete source sections systematically against the selected active facts. A coverage gap is material information actually present in source but missing from memory, not a detail the manuscript fails to report. Consider equations, conditions, results, comparisons, limitations and author qualifications. Propose straightforward missing facts directly with exact source evidence; reserve gaps for omissions requiring further work. Avoid duplicate proposals and do not invent absent material. A well-grounded finding of no material omission can be a successful assessment.'''),
}

def action_catalog():
    return '\n'.join(f'{name} [{a.executor}]: {a.description} Minimum facts={a.min_facts}, minimum questions={a.min_questions}, complete sections={a.min_sections}–{a.max_sections}.' for name,a in ACTIONS.items())
