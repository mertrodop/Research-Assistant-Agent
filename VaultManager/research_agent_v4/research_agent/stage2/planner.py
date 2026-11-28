"""Planner strategy and working memory rebuilt each round, with explicit context limits."""
import json
from ..stage1.pipeline import atomic_json
from ..schemas.planning import PlannerDecision, RoundEvaluation
from ..schemas.execution import TaskRecord
from ..tools.retrieval import dump
from .validation import validate_decision,proposals_of,validate_evaluation
from .prompts import PLANNER,EVALUATOR
from .llm import StructuredCallFailed

def records_from(state): return {i:TaskRecord.model_validate(r) for i,r in state['tasks'].items()}
def records_dump(records): return {i:dump(r) for i,r in records.items()}

def compact_fact(f):
    """Planning projection only; preserve complete text and section provenance."""
    return {'fact_id': f['fact_id'], 'text': f['text'], 'status': f['status'],
            'section_ids': sorted({e['section_id'] for e in f['evidence']})}


def planner_fact_view(f):
    return {'fact_id': f.fact_id, 'text': f.text, 'status': f.verification_status,
            'section_ids': sorted({e.section_id for e in f.evidence})}


def compact_question(q):
    """Keep question meaning, status, answer and fact associations, not line spans."""
    out={key:q[key] for key in ('question_id','question','status','origin_section_id')}
    for key in ('motivating_fact_ids','answer','answer_fact_ids','related_question_ids'):
        if q.get(key): out[key]=q[key]
    # Preserve section-level answer provenance even for answers without fact anchors.
    sections=sorted({e['section_id'] for e in q.get('answer_evidence',[])})
    if sections: out['answer_section_ids']=sections
    return out


def episode_view(e):
    """Compact working memory, not a change to the durable Episode record."""
    v=e.verification
    out={'attempt_id':e.attempt_id,'round_number':e.round_number,'task_id':e.task_id,
         'action':e.action,'objective':e.objective,'status':e.status,
         'task_assessment':e.task_assessment,'outcome_summary':e.outcome_summary,
         'evaluation':e.evaluation,'accepted_updates':e.accepted_updates,
         'remaining_uncertainties':e.remaining_uncertainties,
         'information_requests':[dump(r) for r in e.information_requests],
         'verification_status':v.status}
    if e.evidence_refs: out['evidence_section_ids']=sorted({x.section_id for x in e.evidence_refs})
    if e.failure_reason: out['failure_reason']=e.failure_reason
    if v.reason: out['verification_reason']=v.reason
    # Retain feedback needed for replanning without successful checks or audit hashes.
    rejected=[dump(c) for c in v.checks if c.verdict!='supported']
    if rejected: out['rejected_checks']=rejected
    out['supported_proposal_count']=sum(c.verdict=='supported' for c in v.checks)
    return out


def evaluation_payload(state,items,verifications):
    """Strategic evaluation projection. Full contexts remain in task artifacts.

    Deduplicate selected fact/question records and preserve task membership by ID.
    Independently checked proposals do not require a third reading of raw sections.
    For unverified no-proposal source assessments, retain complete scoped passages
    once per distinct reading unit/span (not a new verifier call).
    """
    facts={}; questions={}; tasks=[]; observation_passages={}
    for t,ctx,r in items:
        v=verifications[t.task_id]; props=proposals_of(r)
        for f in ctx['facts']: facts[f['fact_id']]=compact_fact(f)
        for q in ctx['questions']: questions[q['question_id']]=compact_question(q)
        item={'task':dump(t),'status':r.status,'task_assessment':r.task_assessment,
              'outcome_summary':r.outcome_summary,
              'remaining_uncertainties':r.remaining_uncertainties,
              'information_requests':[dump(x) for x in r.information_requests],
              'failure_reason':r.failure_reason,
              'numbered_proposals':[{'index':i,'kind':kind,'proposal':dump(p)} for i,(kind,p) in enumerate(props)],
              'verification':{'status':v.status,'checks':[dump(c) for c in v.checks],'reason':v.reason},
              'eligible_proposal_indices':sorted(v.eligible_indices)}
        if hasattr(r,'merge_disposition'): item['merge_disposition']=r.merge_disposition
        if ctx.get('error'): item['context_error']=ctx['error']
        # A verifier skipping a negative finding is not evidence that it is correct.
        if v.status=='skipped' and not props and r.status!='failed' and ctx['passages']:
            ids=[]
            for passage in ctx['passages']:
                ref=passage['ref']; key=f"{ref['section_id']}:{ref['unit_id']}:{ref['line_start']}-{ref['line_end']}"
                if key in observation_passages and observation_passages[key]!=passage:
                    raise ValueError('Conflicting source passages for evaluation: '+key)
                observation_passages[key]=passage; ids.append(key)
            item['unverified_observation_passage_ids']=ids
            item['observation_evidence_refs']=[dump(e) for e in r.evidence_refs]
        tasks.append(item)
    payload={'goal':state['objective'],'round_number':state['round_number'],
             'strategy':state['strategy'],'facts':list(facts.values()),
             'questions':list(questions.values()),'tasks':tasks}
    if observation_passages: payload['unverified_observation_passages']=observation_passages
    return payload


class PlannerContext:
    def __init__(self,store,config,audit): self.store=store; self.config=config; self.audit=audit
    def build(self,state):
        l=self.store.load(); k=l.knowledge; bundle=self.store.bundle
        completed=set(bundle.raw['completed_unit_ids']); planned=bundle.raw['reading_plan']
        sections=[]
        # Questions may originate from a multi-section candidate, including rejected candidates.
        question_sections={qid: {q.origin_section_id} | {e.section_id for cid in q.motivating_candidate_ids if cid in k.candidate_facts for e in k.candidate_facts[cid].evidence} | {e.section_id for fid in q.motivating_fact_ids if fid in k.facts for e in k.facts[fid].evidence} for qid,q in k.questions.items()}
        for s in bundle.source.catalog:
            ids={u['unit_id'] for u in planned if u['section_id']==s['section_id']}
            coverage='not_scheduled' if not ids else 'processed' if ids<=completed else 'partial' if ids&completed else 'unprocessed'
            sections.append({**{key:s[key] for key in ('section_id','title','kind','line_start','local_end')},'stage1_status':coverage,
                'fact_ids':sorted(fid for fid,f in k.facts.items() if f.verification_status not in ('superseded','disputed') and any(e.section_id==s['section_id'] for e in f.evidence)),
                'question_ids':sorted(qid for qid,q in k.questions.items() if s['section_id'] in question_sections[qid])})
        records=records_from(state)
        tasks=[{'task_id':tid,'round':r.round_number,'action':r.task.action,'task_prompt':r.task.task_prompt,'fact_ids':r.task.fact_ids,'question_ids':r.task.question_ids,'section_ids':r.task.section_ids,'status':r.status,'outcome':r.outcome[:600]} for tid,r in records.items()]
        usage=self.audit.usage()
        payload={'goal':state['objective'],'paper_title':bundle.raw.get('paper_title'),'stage1_complete':bundle.raw['reading_complete'],'knowledge_revision':l.revision,'next_round':state['round_number']+1,'strategy':state['strategy'],'last_round_observation':state['round_observation'],'task_records':tasks,'questions':[compact_question(dump(q)) for q in k.questions.values()],'sections':sections,'facts':[],'recent_episodes':[],'related_episodes':[],'contradictions':[dump(c) for c in k.contradictions.values()],'fact_issues':{i:dump(x) for i,x in k.fact_issues.items()},'coverage_gaps':{i:dump(x) for i,x in k.coverage_gaps.items()},'remaining_rounds':self.config.max_rounds-state['round_number'],'remaining_task_attempts':self.config.max_task_attempts-state['attempts'],'max_tasks_this_round':min(self.config.max_tasks_per_round,self.config.max_task_attempts-state['attempts']),'remaining_llm_calls':max(0,self.config.max_llm_calls-usage['calls']),'remaining_reported_tokens':max(0,self.config.max_reported_tokens-usage['reported_tokens']),'omitted_fact_ids':[],'omitted_episode_ids':[]}
        episodes=list(l.episodes.values()); n=max(self.config.recent_episodes,len(state['round_task_ids'])); recent=episodes[-n:]
        payload['recent_episodes']=[episode_view(e) for e in recent]
        open_questions=[q for q in k.questions.values() if q.status in ('open','partial','investigating')]
        anchors={f for q in open_questions for f in q.motivating_fact_ids}
        for fid in list(anchors):
            seen=set()
            while fid in k.facts and k.facts[fid].superseded_by and fid not in seen:
                seen.add(fid); fid=k.facts[fid].superseded_by; anchors.add(fid)
        related=[]; qids={q.question_id for q in open_questions}
        for e in reversed(episodes[:-n]):
            old=records.get(e.task_id)
            if old and (anchors.intersection(old.task.fact_ids) or qids.intersection(old.task.question_ids)): related.append(e)
        payload['omitted_episode_ids']=[e.attempt_id for e in episodes if e not in recent]
        facts=sorted((f for f in k.facts.values() if f.verification_status!='superseded'),key=lambda f:(f.fact_id not in anchors,f.fact_id))
        payload['omitted_fact_ids']=[f.fact_id for f in facts]
        size=lambda:len(json.dumps(payload,ensure_ascii=False))
        if size()>self.config.max_planner_chars: raise ValueError('Pinned planner context exceeds max_planner_chars; increase limit or narrow run')
        for key,omitted,seq,view,idkey in [('facts','omitted_fact_ids',facts,planner_fact_view,'fact_id'),('related_episodes','omitted_episode_ids',related[:n],episode_view,'attempt_id')]:
            for item in seq:
                ident=getattr(item,idkey); payload[key].append(view(item)); payload[omitted].remove(ident)
                if size()>self.config.max_planner_chars: payload[key].pop(); payload[omitted].append(ident)
        return payload

class Planner:
    def __init__(self,store,config,audit): self.store=store; self.config=config; self.audit=audit; self.context=PlannerContext(store,config,audit)
    def plan(self,state):
        k=self.store.load().knowledge
        path=self.store.root/'plan_history'/f'round_{state["round_number"]+1:03d}_input.json'
        if path.exists():
            payload=json.loads(path.read_text())
            if payload['knowledge_revision']!=self.store.load().revision: raise ValueError('Plan input snapshot mismatch')
        else:
            payload=self.context.build(state); atomic_json(path,payload)
        visible_facts={f['fact_id'] for f in payload['facts']}
        def validate(p):
            validate_decision(p,k,self.store.bundle,self.config,state)
            for t in p.current_round_tasks:
                if not set(t.fact_ids)<=visible_facts: raise ValueError('Select facts visible in current planner context')
        return self.audit.call(f'plan_round_{state["round_number"]+1:03d}',PLANNER,payload,PlannerDecision,self.config.planner,validate)
    def evaluate(self,state,items,verifications):
        payload=evaluation_payload(state,items,verifications)
        if len(json.dumps(payload,ensure_ascii=False))>self.config.max_evaluation_chars: raise StructuredCallFailed('Round evaluation context exceeds max_evaluation_chars; no updates approved')
        return self.audit.call(f'evaluate_round_{state["round_number"]:03d}',EVALUATOR,payload,RoundEvaluation,self.config.planner,lambda e:validate_evaluation(e,items,verifications))
