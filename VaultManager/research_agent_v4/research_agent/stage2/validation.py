"""Structural safeguards; semantic entailment remains an LLM judgment."""
import re
from ..schemas.execution import *
from ..schemas.planning import PlannerDecision, RoundEvaluation
from .actions import ACTIONS

def unique(values,name):
    if len(values)!=len(set(values)): raise ValueError('Duplicate '+name)

def subset(values,allowed,name,minimum=0):
    unique(values,name)
    if len(values)<minimum or not set(values)<=set(allowed): raise ValueError(f'Invalid {name}: {values}')

def nonblank(s,name):
    if not s.strip(): raise ValueError('Empty '+name)

def task_signature(t):
    norm=lambda s:re.sub(r'\s+',' ',s.strip().casefold())
    return (t.action,tuple(sorted(t.fact_ids)),tuple(sorted(t.question_ids)),tuple(sorted(t.section_ids)),norm(t.task_prompt))

def validate_task(t,k,bundle):
    nonblank(t.task_prompt,'task prompt'); nonblank(t.rationale,'rationale')
    for criterion in t.success_criteria: nonblank(criterion,'success criterion')
    spec=ACTIONS[t.action]
    active={fid for fid,f in k.facts.items() if f.verification_status!='superseded'}
    if t.action=='ANSWER_QUESTION': active={fid for fid in active if k.facts[fid].verification_status!='disputed'}
    subset(t.fact_ids,active,'fact IDs',spec.min_facts)
    subset(t.question_ids,k.questions,'question IDs',spec.min_questions)
    subset(t.section_ids,bundle.source.sections,'section IDs',spec.min_sections)
    if len(t.section_ids)>spec.max_sections: raise ValueError(f'{t.action} permits at most {spec.max_sections} sections')
    if t.action=='CHECK_COVERAGE':
        # Validate completeness; never silently add facts to the planner selection.
        targets=set(t.section_ids)
        required={fid for fid,f in k.facts.items()
                  if f.verification_status not in ('superseded','disputed')
                  and any(e.section_id in targets for e in f.evidence)}
        missing=required-set(t.fact_ids)
        if missing: raise ValueError('CHECK_COVERAGE must explicitly select all active facts associated with the selected sections; missing: '+str(sorted(missing)))
    if t.action in ('MERGE_FACTS','CHECK_CONTRADICTION'):
        needed={e.section_id for fid in t.fact_ids for e in k.facts[fid].evidence}
        if not needed<=set(t.section_ids): raise ValueError('Grounded operation requires all target fact evidence sections: '+str(sorted(needed)))
    return t

def validate_decision(decision,k,bundle,config,state):
    nonblank(decision.strategy.overall_objective,'strategy objective')
    for step in decision.strategy.steps:
        nonblank(step.objective,'step objective'); nonblank(step.rationale,'step rationale')
    if decision.terminate:
        if decision.current_round_tasks or not (decision.termination_reason or '').strip(): raise ValueError('Termination needs a reason and no tasks')
        return
    if decision.termination_reason is not None: raise ValueError('Non-termination requires null termination_reason')
    tasks=decision.current_round_tasks
    if not 1<=len(tasks)<=min(config.max_tasks_per_round,config.max_task_attempts-state['attempts']): raise ValueError('Current round task budget exceeded or empty')
    if not decision.strategy.steps: raise ValueError('Nonempty strategy required')
    unique([t.task_id for t in tasks],'task IDs')
    old=[TaskRecord.model_validate(r).task for r in state['tasks'].values()]
    signatures={task_signature(t) for t in old}; writers=set(); fact_writers=set()
    for t in tasks:
        if t.task_id in state['tasks']: raise ValueError('Task ID already used: '+t.task_id)
        validate_task(t,k,bundle)
        sig=task_signature(t)
        if sig in signatures: raise ValueError('Repeated task without new context or approach: '+t.task_id)
        signatures.add(sig)
        if t.action == 'ANSWER_QUESTION':
            if writers & set(t.question_ids): raise ValueError('Multiple tasks may not answer the same question in one round')
            writers.update(t.question_ids)
        if t.action in ('MERGE_FACTS','CHECK_CONTRADICTION'):
            if fact_writers & set(t.fact_ids): raise ValueError('Overlapping grounded fact write targets in one round')
            fact_writers.update(t.fact_ids)

PROPOSAL_FIELDS=[('merged_facts','merged_fact'),('fact_issues','fact_issue'),('contradictions','contradiction'),('new_facts','new_fact'),('answers','answer'),('new_questions','new_question'),('gaps','gap')]
def proposals_of(result):
    return [(kind,p) for field,kind in PROPOSAL_FIELDS for p in getattr(result,field,[])]

def validate_result(task,result,context,bundle,k):
    props=proposals_of(result)
    if context.get('error') and props: raise ValueError('Context unavailable')
    nonblank(result.outcome_summary,'outcome summary')
    if result.status in ('partial','failed','insufficient_context') and not (result.failure_reason or result.remaining_uncertainties or result.information_requests): raise ValueError('Explain incomplete outcome')
    if result.status=='failed' and props: raise ValueError('Failed task cannot propose updates')
    facts={f['fact_id']:f for f in context['facts']}; fids=set(facts); qids={q['question_id'] for q in context['questions']}
    spans=[EvidenceRef.model_validate(x['ref']) for x in context['passages']]
    def evidence(refs,required=True,allowed=None):
        if required and not refs: raise ValueError('Missing evidence')
        pool=spans if allowed is None else allowed
        for e in refs:
            bundle.check_evidence(e)
            if not any(s.unit_id==e.unit_id and s.section_id==e.section_id and s.line_start<=e.line_start<=e.line_end<=s.line_end for s in pool): raise ValueError('Evidence not available in supplied context: '+str(e))
    def inherited(ids): return [EvidenceRef.model_validate(e) for fid in ids for e in facts[fid]['evidence']]
    fact_only=task.action=='ANSWER_QUESTION'
    evidence(result.evidence_refs,False,inherited(fids) if fact_only else None)
    if not {s.section_id for s in spans}<=set(task.section_ids): raise ValueError('Unexpected source sections')
    for req in result.information_requests:
        nonblank(req.reason,'information request reason'); subset(req.fact_ids,fids,'request fact IDs'); subset(req.question_ids,qids,'request question IDs'); subset(req.section_ids,set(bundle.source.sections)-set(task.section_ids),'request unexamined section IDs')
    answer_targets=set(); superseded=set(); issues=set()
    for kind,p in props:
        if kind not in ACTIONS[task.action].proposals: raise ValueError(f'{kind} not allowed for {task.action}')
        if kind in ('new_fact','merged_fact'):
            nonblank(p.text,'fact'); evidence(p.evidence); subset(p.derived_from_fact_ids,fids,'derived facts',2 if kind=='merged_fact' else 0)
            if kind=='merged_fact':
                subset(p.derived_from_fact_ids,task.fact_ids,'merge inputs',2); subset(p.supersede_ids,p.derived_from_fact_ids,'superseded facts',1)
                if superseded & set(p.supersede_ids): raise ValueError('Conflicting supersession')
                if any(k.facts[i].verification_status=='superseded' for i in p.supersede_ids): raise ValueError('Already superseded')
                superseded.update(p.supersede_ids)
        elif kind=='fact_issue':
            subset([p.fact_id],task.fact_ids,'flagged fact',1); nonblank(p.explanation,'fact issue'); evidence(p.evidence)
            if p.fact_id in issues: raise ValueError('Repeated fact issue')
            issues.add(p.fact_id)
        elif kind=='contradiction':
            subset(p.fact_ids,task.fact_ids,'contradiction facts',2); nonblank(p.explanation,'explanation'); evidence(p.evidence)
        elif kind=='answer':
            nonblank(p.answer,'group answer'); subset([r.question_id for r in p.resolutions],set(task.question_ids)&qids,'answer questions',1)
            for r in p.resolutions:
                nonblank(r.explanation,'resolution explanation'); subset(r.answer_fact_ids,fids,'answer facts')
                if r.question_id in answer_targets: raise ValueError('Multiple resolutions for one question')
                answer_targets.add(r.question_id)
                if r.status in ('answered','partial') and fact_only and not r.answer_fact_ids: raise ValueError('Facts-only answer requires supporting facts')
                if fact_only and any(facts[i]['status'] in ('superseded','disputed') for i in r.answer_fact_ids): raise ValueError('Cannot answer using disputed/superseded facts')
                if r.status=='unanswerable':
                    substantive={s['section_id'] for s in bundle.source.catalog if s['kind'] not in ('bibliography','back_matter') and bundle.source.read_section(s['section_id'],labels=False).strip()}
                    if fact_only or not substantive<=set(task.section_ids): raise ValueError('Use unresolved/partial: current source scope cannot establish manuscript-wide absence')
                evidence(r.evidence,r.status!='unresolved',inherited(r.answer_fact_ids) if fact_only else None)
        elif kind=='new_question':
            nonblank(p.question,'question'); subset(p.motivating_fact_ids,fids,'motivating facts'); evidence(p.evidence)
        elif kind=='gap':
            subset([p.section_id],set(task.section_ids),'coverage section',1); nonblank(p.description,'gap'); evidence(p.evidence)
            if not all(e.section_id==p.section_id for e in p.evidence): raise ValueError('Gap evidence must be from named section')
    if superseded & issues: raise ValueError('Cannot simultaneously supersede and dispute a fact')
    if task.action=='MERGE_FACTS':
        if result.merge_disposition in ('merged','partially_merged') and not result.merged_facts: raise ValueError('Merge disposition requires proposed merge')
        if result.merge_disposition not in ('merged','partially_merged') and result.merged_facts: raise ValueError('Merge proposals contradict disposition')
        if result.merge_disposition=='merged' and superseded!=set(task.fact_ids): raise ValueError('Full merge must cover all selected facts')
    if task.action=='ANSWER_QUESTION' and result.status!='failed' and not context.get('error'):
        if answer_targets!=set(task.question_ids): raise ValueError('Every question requires exactly one resolution, even if unresolved')
        statuses=[q.status for g in result.answers for q in g.resolutions]
        expected='success' if all(x=='answered' for x in statuses) else 'partial' if any(x in ('answered','partial') for x in statuses) else 'insufficient_context'
        if result.status!=expected: raise ValueError('Answer task status must match dispositions: '+expected)
    return props

def validate_evaluation(ev,items,verifications):
    by={t.task_id:r for t,ctx,r in items}
    subset([x.task_id for x in ev.task_evaluations],by,'evaluation task IDs',len(by))
    nonblank(ev.round_observation,'round observation')
    for x in ev.task_evaluations:
        r=by[x.task_id]; props=proposals_of(r)
        subset(x.accepted_proposal_indices,range(len(props)),'accepted proposal indices')
        verify=verifications[x.task_id]
        validate_verification_record(verify,props)
        subset(x.accepted_proposal_indices,verify.eligible_indices,'verified eligible proposal indices')
        nonblank(x.rationale,'evaluation rationale')
        if x.task_satisfied and (r.status!='success' or set(x.accepted_proposal_indices)!=set(range(len(props)))): raise ValueError('Satisfied task requires success and acceptance of all proposals')
        if x.task_satisfied and any(kind=='answer' and any(q.status in ('partial','unresolved') for q in p.resolutions) for kind,p in props): raise ValueError('Unresolved/partial questions cannot satisfy full answer task')

    flagged={p.fact_id for x in ev.task_evaluations for i,(kind,p) in enumerate(proposals_of(by[x.task_id])) if i in x.accepted_proposal_indices and kind=='fact_issue'}
    for x in ev.task_evaluations:
        for i,(kind,p) in enumerate(proposals_of(by[x.task_id])):
            if i not in x.accepted_proposal_indices: continue
            if kind=='answer' and any(flagged.intersection(r.answer_fact_ids) for r in p.resolutions): raise ValueError('Accepted answer uses a fact disputed in this round')
            if kind=='merged_fact' and flagged.intersection(p.derived_from_fact_ids): raise ValueError('Accepted merge uses a fact disputed in this round')


def validate_verification_record(record,props):
    if record.status=='completed':
        subset([c.proposal_index for c in record.checks],range(len(props)),'verification proposal indices',len(props))
        if not props or not record.log: raise ValueError('Completed verification requires proposals and log')
        for c in record.checks: nonblank(c.reason,'verification reason')
    elif record.checks or record.log or (record.status=='skipped' and props) or (record.status=='unavailable' and not props):
        raise ValueError('Invalid skipped/unavailable verification record')
    nonblank(record.input_hash,'verification input hash')
    if record.status!='completed': nonblank(record.reason,'verification status reason')
