"""Adaptive planner / scoped executor / verifier. Serial, resumable, no hidden rereads."""
import json
from pathlib import Path
from .schema import Config, State, Plan, Task, Result, Review, Proposal, Check, result_schema
from .prompts import PLANNER, executor_prompt, VERIFIER, COMMON, RULES
from ..schemas.scientific import KnowledgeStore, Fact, Question, EvidenceRef, CandidateAnchor
from ..stage2.store import PaperBundle
from .audit import AuditedLLM, BudgetExceeded, StructuredCallFailed
from ..stage1.pipeline import atomic_json, fingerprint

VERSION = 'adaptive-v3.1'
def dump(x): return x.model_dump(mode='json')
def compact(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def normalized(s): return ' '.join(s.casefold().split())
def next_id(d,prefix,width):
    return prefix+str(max([int(k[len(prefix):]) for k in d if k.startswith(prefix) and k[len(prefix):].isdigit()] or [0])+1).zfill(width)
def unique(xs): return list(dict.fromkeys(xs))
def active(f): return f.verification_status not in ('superseded','disputed')

ACTION_KINDS = {
    'CONSOLIDATE':{'merge','question'},
    'REASON':{'answer','partial_answer','derive','boundary','question','contradiction'},
    'INVESTIGATE':{'fact','answer','partial_answer','not_found','question','boundary','derive','correct','contradiction'},
    'QUESTION':{'question'},
}
def allowed_kinds(t):
    kinds=set(ACTION_KINDS[t.action])
    if not t.questions: kinds-={'answer','partial_answer'}
    if not t.facts: kinds-={'correct'}
    if not (t.facts or t.assessments): kinds-={'derive'}
    if len(t.facts)<2: kinds-={'merge','contradiction'}
    if not t.sections: kinds-={'fact','correct','not_found'}
    return sorted(kinds)

def wire_result(w,t):
    """Convert wire records without inventing associations. Invalid items are reported separately."""
    proposals=[]; errors=[]
    items=[(f.kind,f.text,f.refs,f.lines,None) for f in w.findings]
    items += [(f.status,f.text,f.refs,f.lines,f.q) for f in w.resolutions]
    for i,(kind,text,refs,lines,q) in enumerate(items):
        try:
            refs=unique(refs)
            if not set(refs)<=set(t.facts+t.questions+t.candidates+t.assessments): raise ValueError('Unselected reference')
            if q is not None:
                if q not in t.questions or kind not in ('answer','partial_answer','not_found'): raise ValueError('Invalid resolution question or status')
                if set(refs)&set(t.questions)-{q}: raise ValueError('Resolution refers to another question')
                refs=unique(refs+[q])
            elif kind in ('answer','partial_answer') or (kind=='not_found' and set(refs)&set(t.questions)):
                raise ValueError('Use resolutions with explicit q for question outcomes')
            proposals.append(Proposal(kind=kind,text=text,facts=[i for i in refs if i in t.facts],questions=[i for i in refs if i in t.questions],candidates=[i for i in refs if i in t.candidates],assessments=[i for i in refs if i in t.assessments],spans=lines))
        except ValueError as exc: errors.append(f'Wire item {i}: {exc}')
    return Result(proposals=proposals,note=''),errors

def wire_proposal(p):
    data={'kind':p.kind,'text':p.text,'refs':p.facts+p.questions+p.candidates+p.assessments,'lines':[dump(x) for x in p.spans]}
    if p.kind in ('answer','partial_answer','not_found') and p.questions:
        data['q']=p.questions[0]
    return data

class AdaptiveAgent:
    def __init__(self,handoff_path,config,llm,run_dir,source_paths=None):
        self.config=config; self.root=Path(run_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.bundle=PaperBundle(handoff_path,source_paths,config.allow_partial_stage1)
        self.source=self.bundle.source
        self.path=self.root/'state.json'; self.manifest=self.root/'manifest.json'
        identity=fingerprint({'version':VERSION,'bundle':self.bundle.identity,'config':dump(config),'llm':llm.identity,'prompts':[PLANNER,COMMON,RULES,VERIFIER], 'schemas':[x.model_json_schema() for x in (Plan,Result,Review)]+[result_schema().model_json_schema()]})
        if self.manifest.exists():
            if json.loads(self.manifest.read_text())['fingerprint']!=identity: raise ValueError('Checkpoint mismatch: use a new run directory')
            if not self.path.exists(): raise ValueError('Missing checkpoint state')
        else:
            if self.path.exists() or (self.root/'call_logs').exists(): raise ValueError('Unbound run directory; use a new directory')
            k=KnowledgeStore.model_validate({key:self.bundle.raw[key] for key in KnowledgeStore.model_fields})
            self.save(State(knowledge=k,types={fid:'extracted' for fid in k.facts}))
            atomic_json(self.manifest,{'fingerprint':identity,'version':VERSION,'config':dump(config),'handoff':str(self.bundle.path)})
        self.audit=AuditedLLM(llm,config,self.root)
        (self.root/'plan_inputs').mkdir(exist_ok=True)
        # Count local substantive nonblank lines, not child-section lines.
        self.section_lengths={sid:self.substantive_line_count(sid) for sid in self.source.sections}
        self.scientific_sections={x['section_id'] for x in self.source.catalog if x['kind']!='bibliography' and self.section_lengths[x['section_id']]>0}

    def substantive_line_count(self,sid):
        return len({i for b in self.source.blocks.get(sid,[]) if b['kind'] not in ('heading','rule') for i in range(b['line_start'],b['line_end']+1) if self.source.lines[i-1].strip()})

    def load(self): return State.model_validate_json(self.path.read_text(encoding='utf-8'))
    def save(self,s): atomic_json(self.path,dump(s))
    def sections_for(self,f): return sorted({e.section_id for e in f.evidence})
    def inspected(self,s,qid):
        return sorted({sid for a in s.assessments.values() if a['kind']=='not_found' and not a.get('stale') and qid in a['questions'] for sid in a['sections']})
    def planner_payload(self,s):
        k=s.knowledge; usage=self.audit.usage()
        facts={fid:[f.text,self.sections_for(f),s.types.get(fid,'extracted')] for fid,f in k.facts.items() if active(f)}
        questions={}; deferred={}; answers={}
        for qid,q in k.questions.items():
            if q.status=='answered':
                # Answers are scientific knowledge too; keep their content accessible without repeating full question text.
                answers[qid]=[q.answer or '',[i for i in q.answer_fact_ids if i in facts],sorted({e.section_id for e in q.answer_evidence})]
                continue
            record={'text':q.question,'facts':[i for i in q.motivating_fact_ids if i in facts],'searched':self.inspected(s,qid)}
            if q.answer: record['partial']=q.answer
            if qid in s.question_triage:
                deferred[qid]=[q.question[:120]+('…' if len(q.question)>120 else ''),record['searched']]
            else: questions[qid]=record
        candidates={cid:[c.text,c.reason,self.sections_for(c)] for cid,c in k.candidate_facts.items() if c.verdict!='supported' and not any(cid in a.get('candidates',[]) and a['kind']=='repair' for a in s.assessments.values())}
        sections={x['section_id']:[x['title'],self.section_lengths[x['section_id']]] for x in self.source.catalog}
        assessments={aid:[a['kind'],a['text'],a['questions'],a['sections']] for aid,a in s.assessments.items() if not a.get('stale')}
        history=[]
        for e in s.episodes[-self.config.recent_episodes:]:
            selected=e['selected']; history.append([e['id'],e['action'],selected['questions'],selected['sections'],e['accepted'],(e.get('error') or '; '.join(e.get('validation_errors',[])))[:160],len(e['rejected'])])
        p={'goal':self.config.objective,'round':s.round+1,'facts':facts,'questions':questions,'sections':sections,'budget':{'rounds_left':self.config.max_rounds-s.round,'calls_left':self.config.max_llm_calls-usage['calls'],'tokens_left':max(0,self.config.max_reported_tokens-usage['reported_tokens'])},'limits':{'tasks':self.config.max_tasks_per_round,'sections':self.config.max_sections_per_task,'facts':self.config.max_selected_facts,'questions':self.config.max_selected_questions}}
        for key,value in [('candidates',candidates),('deferred',deferred),('answers',answers),('assessments',assessments),('history',history)]:
            if value: p[key]=value
        if s.plans and s.plans[-1].get('validation_errors'): p['last_plan_errors']=[x[:160] for x in s.plans[-1]['validation_errors'][:3]]
        while len(compact(p))>self.config.max_planner_chars and history: history.pop(0)
        if len(compact(p))>self.config.max_planner_chars: raise ValueError('Planner context cap exceeded; no hidden truncation. Increase cap or narrow scope.')
        return p

    def validate_task(self,t,s,planning=False):
        k=s.knowledge
        for name,ids,allowed in [('facts',t.facts,{i for i,f in k.facts.items() if active(f)}),('questions',t.questions,set(k.questions)),('candidates',t.candidates,{i for i,c in k.candidate_facts.items() if c.verdict!='supported'}),('sections',t.sections,set(self.source.sections)),('assessments',t.assessments,{i for i,a in s.assessments.items() if not a.get('stale')})]:
            if not set(ids)<=allowed: raise ValueError('Unknown or inactive '+name)
        if not t.prompt.strip(): raise ValueError('Empty task prompt')
        if len(t.sections)>self.config.max_sections_per_task or len(t.facts)+len(t.assessments)>self.config.max_selected_facts or len(t.questions)>self.config.max_selected_questions: raise ValueError('Task scope exceeds limit')
        if t.action in ('REASON','QUESTION') and t.sections: raise ValueError('No manuscript for memory-only task')
        if t.action=='INVESTIGATE' and not t.sections: raise ValueError('INVESTIGATE requires selected sections')
        if t.action=='CONSOLIDATE' and len(t.facts)<2: raise ValueError('CONSOLIDATE needs >=2 facts')
        if t.action in ('REASON','QUESTION') and not (t.facts or t.assessments): raise ValueError('Reasoning or questioning needs scientific records')
        if t.candidates and t.action not in ('INVESTIGATE','QUESTION'): raise ValueError('Candidates are investigation leads only')

    def apply_plan(self,s,plan,payload=None):
        payload=payload or self.planner_payload(s)
        tasks=[]; errors=[]; seen=set(); selected=set()
        qs=set(payload['questions'])|set(payload.get('deferred',{}))|set(payload.get('answers',{}))
        for i,w in enumerate(plan.tasks):
            try:
                ids=unique(w.refs)
                if not set(ids)<=set(payload['facts'])|qs|set(payload.get('candidates',{}))|set(payload.get('assessments',{})): raise ValueError('Unknown/unavailable reference')
                t=Task(action=w.action,prompt=w.prompt,facts=[x for x in ids if x in payload['facts']],questions=[x for x in ids if x in qs],candidates=[x for x in ids if x in payload.get('candidates',{})],assessments=[x for x in ids if x in payload.get('assessments',{})],sections=unique(w.sections))
                self.validate_task(t,s,planning=True)
                h=fingerprint(dump(t))
                if h in seen: raise ValueError('Duplicate task')
                if len(tasks)>=self.config.max_tasks_per_round: raise ValueError('Task limit; omitted')
                seen.add(h); tasks.append(t); selected.update(t.questions)
            except ValueError as exc: errors.append(f'Task {i+1}: {exc}')
        for qid in unique(plan.defer):
            if qid not in qs: errors.append(f'Unknown deferred question: {qid}'); continue
            if qid in selected or s.knowledge.questions[qid].status=='answered': continue
            s.question_triage[qid]={'decision':'defer','round':s.round}
        for qid in selected: s.question_triage.pop(qid,None)
        return tasks,errors

    def context(self,t,s):
        self.validate_task(t,s); k=s.knowledge
        return {'facts':{i:[k.facts[i].text,s.types.get(i,'extracted')] for i in t.facts},'questions':{i:[k.questions[i].question,k.questions[i].status,k.questions[i].answer or ''] for i in t.questions},'candidates':{i:[k.candidate_facts[i].text,k.candidate_facts[i].reason] for i in t.candidates},'assessments':{i:{key:s.assessments[i][key] for key in ('kind','text','questions','sections','facts')} for i in t.assessments},'source':{sid:self.source.read_section(sid) for sid in t.sections}}

    def resolve_spans(self,spans,sections):
        refs=[]
        for sp in spans:
            if sp.a<1 or sp.a>sp.b: raise ValueError('Invalid or reversed span')
            units=sorted((u for u in self.bundle.units.values() if u.section_id in sections and u.line_start<=sp.b and u.line_end>=sp.a),key=lambda u:u.line_start)
            cursor=sp.a
            for u in units:
                a,b=max(sp.a,u.line_start),min(sp.b,u.line_end)
                if a!=cursor: raise ValueError('Evidence not in selected sections')
                e=EvidenceRef(section_id=u.section_id,unit_id=u.unit_id,line_start=a,line_end=b); self.bundle.check_evidence(e); refs.append(e); cursor=b+1
            if cursor!=sp.b+1: raise ValueError('Evidence outside selected sections')
        return self.dedup_refs(refs)

    @staticmethod
    def dedup_refs(refs):
        return list({(e.section_id,e.unit_id,e.line_start,e.line_end):e for e in refs}.values())

    def validate_result(self,r,t):
        if len(r.proposals)>self.config.max_proposals: raise ValueError('Too many proposals')
        allowed=set(allowed_kinds(t))
        seen=set(); answers=set(); replaces=set(); corrections=set()
        for i,p in enumerate(r.proposals):
            if p.kind not in allowed or not p.text.strip(): raise ValueError(f'Proposal {i}: kind {p.kind} not permitted for this task or blank text; allowed kinds={sorted(allowed)}')
            for name,ids,scope in [('facts',p.facts,t.facts),('questions',p.questions,t.questions),('candidates',p.candidates,t.candidates),('assessments',p.assessments,t.assessments)]:
                unknown=sorted(set(ids)-set(scope))
                if unknown: raise ValueError(f'Proposal {i}: unselected {name} IDs {unknown}; permitted IDs {scope}')
                if len(ids)!=len(set(ids)): raise ValueError(f'Proposal {i}: duplicate {name} IDs {ids}')
            self.resolve_spans(p.spans,t.sections)
            if not (p.spans or p.facts or p.candidates or p.assessments):
                if p.kind!='not_found': raise ValueError('Proposal must have grounded anchors')
            if p.kind in ('fact','correct') and not p.spans: raise ValueError('Explicit or corrected fact needs source evidence')
            if p.kind in ('derive','boundary','answer','partial_answer','contradiction') and not (p.facts or p.spans or p.assessments): raise ValueError('Rejected candidates cannot serve as premises')
            if p.kind=='derive' and not (p.facts or p.assessments): raise ValueError('Derivation requires scientific premises')
            if p.kind in ('derive','answer','partial_answer','boundary','contradiction','merge') and p.candidates: raise ValueError('Rejected candidates cannot be used as scientific premises')
            if p.kind in ('merge','contradiction') and len(p.facts)<2: raise ValueError('Need >=2 facts')
            if p.kind=='correct' and len(p.facts)!=1: raise ValueError('Correct one fact at a time')
            if p.kind in ('merge','correct'):
                if replaces.intersection(p.facts): raise ValueError('Overlapping replacements')
                replaces.update(p.facts)
                if p.kind=='correct': corrections.update(p.facts)
            if p.kind in ('answer','partial_answer') or (p.kind=='not_found' and p.questions):
                if len(p.questions)!=1: raise ValueError(f'Proposal {i}: {p.kind} requires exactly one selected Q ID; got {p.questions}; selected={t.questions}')
                if answers.intersection(p.questions): raise ValueError(f'Proposal {i}: question {p.questions[0]} already has a resolution in this result; combine subanswers into one answer OR partial_answer OR not_found')
                answers.update(p.questions)
            if p.kind=='not_found' and not t.sections: raise ValueError('Negative search requires sections')
            key=(p.kind,normalized(p.text),tuple(p.questions))
            if key in seen: raise ValueError('Duplicate proposal')
            seen.add(key)

        for p in r.proposals:
            if p.kind not in ('correct','merge') and corrections.intersection(p.facts): raise ValueError('Do not use a corrected premise in another proposal from the same task; replan after commit')

    def filter_result(self,r,t):
        valid=[]; errors=[]
        for i,p in enumerate(r.proposals):
            try:
                self.validate_result(Result(proposals=valid+[p],note=''),t)
                valid.append(p)
            except ValueError as exc: errors.append(f'Proposal {i}: {exc}')
        return Result(proposals=valid,note=r.note),errors

    def executor_request(self,t,s):
        ctx=self.context(t,s); kinds=allowed_kinds(t)
        schema=result_schema()
        payload={'action':t.action,'prompt':t.prompt,'max_proposals':self.config.max_proposals,**ctx}
        return executor_prompt(kinds,bool(t.questions)),payload,schema

    def verify_payload(self,ctx,r):
        scientific={str(i):wire_proposal(p) for i,p in enumerate(r.proposals) if p.kind!='question'}
        # Always retain the complete selected scientific context, including uncited sections.
        return {**{k:ctx[k] for k in ('facts','questions','candidates','assessments','source') if k in ctx},'proposals':scientific}

    def normalize_review(self,v,r):
        checks=[]; errors=[]
        expected=[i for i,p in enumerate(r.proposals) if p.kind!='question']
        for i in expected:
            found=[c for c in v.checks if c.i==i]
            if len(found)!=1 or not found[0].reason.strip():
                errors.append(f'Invalid or missing verifier check for {i}')
                checks.append(Check(i=i,ok=False,reason='Missing, duplicate or unexplained verifier decision'))
            else: checks.append(found[0])
        for c in v.checks:
            if c.i not in expected: errors.append(f'Unexpected verifier index {c.i}')
        return Review(checks=checks),errors

    def invalidate_dependents(self,s,old_ids,replacement):
        """A verified correction can invalidate descendants; never erase history.
        Old merged aliases also count as premises of the corrected canonical fact.
        Reopened answers remain in audit logs but leave active question memory.
        """
        k=s.knowledge; invalid=set(old_ids); changed=[]
        stale={aid for aid,a in s.assessments.items() if a.get('stale')}
        while True:
            prev=(set(invalid),set(stale))
            for fid,f in k.facts.items():
                if fid==replacement: continue
                if f.superseded_by in invalid or set(f.derived_from_fact_ids)&invalid or set(s.assessment_dependencies.get(fid,[]))&stale:
                    invalid.add(fid)
                    if active(f): f.verification_status='disputed'; changed.append(fid)
            for aid,a in s.assessments.items():
                if (a['kind']!='not_found' and set(a['facts'])&invalid) or set(s.assessment_dependencies.get(aid,[]))&stale:
                    if not a.get('stale'): changed.append(aid)
                    a['stale']=True; stale.add(aid)
            if (invalid,stale)==prev: break
        for q in k.questions.values():
            if set(q.answer_fact_ids)&invalid or set(s.assessment_dependencies.get(q.question_id,[]))&stale:
                q.status='open'; q.answer=None; q.answer_fact_ids=[]; q.answer_evidence=[]; s.assessment_dependencies.pop(q.question_id,None); changed.append(q.question_id)
        return unique(changed)

    def commit(self,s,t,r,v,task_id):
        k=s.knowledge; updates=[]; checks={c.i:c for c in v.checks}; original={i:k.facts[i] for i in t.facts}
        # Snapshot original references so same-result proposals remain valid even after a merge.
        for i,p in enumerate(r.proposals):
            if p.kind!='question' and not checks[i].ok: continue
            if any(s.assessments[aid].get('stale') for aid in p.assessments): continue
            refs=self.resolve_spans(p.spans,t.sections)
            if p.kind in ('derive','merge','answer','partial_answer','boundary','contradiction','question'):
                refs=self.dedup_refs(refs+[e for fid in p.facts for e in original[fid].evidence])
            inherited_facts=unique(p.facts+[fid for aid in p.assessments for fid in s.assessments[aid]['facts']])
            for aid in p.assessments:
                a=s.assessments[aid]
                # An assessment's inspected scope is inherited provenance, not a fresh citation.
                inherited=[EvidenceRef.model_validate(e) for e in a['evidence']]
                if not inherited:
                    inherited=[EvidenceRef(section_id=u.section_id,unit_id=u.unit_id,line_start=u.line_start,line_end=u.line_end) for u in self.bundle.units.values() if u.section_id in a['sections']]
                refs=self.dedup_refs(refs+inherited)
            if p.kind=='question':
                if len(k.questions)-len(self.bundle.initial.questions)>=self.config.max_new_questions: continue
                if any(normalized(q.question)==normalized(p.text) for q in k.questions.values()): continue
                cids=p.candidates; anchors=[CandidateAnchor(candidate_id=cid,accepted_fact_id=k.candidate_facts[cid].accepted_fact_id,verdict=k.candidate_facts[cid].verdict,checker_reason=k.candidate_facts[cid].reason) for cid in cids]
                if not refs: refs=self.dedup_refs([e for cid in cids for e in k.candidate_facts[cid].evidence])
                if not refs: continue
                qid=next_id(k.questions,'Q',4); e=refs[0]
                k.questions[qid]=Question(question_id=qid,question=p.text,origin_section_id=e.section_id,origin_unit_id=e.unit_id,motivating_candidate_ids=cids,motivating_fact_ids=inherited_facts,motivating_candidates=anchors,related_question_ids=p.questions)
                s.question_evidence[qid]=[dump(x) for x in refs]
                if p.assessments: s.assessment_dependencies[qid]=p.assessments
                updates.append(qid)
            elif p.kind in ('fact','derive','merge','correct','boundary'):
                if p.kind not in ('merge','correct') and any(active(f) and normalized(f.text)==normalized(p.text) for f in k.facts.values()): continue
                if not refs: continue
                fid=next_id(k.facts,'F',5); e=refs[0]
                k.facts[fid]=Fact(fact_id=fid,text=p.text,origin_section_id=e.section_id,origin_unit_id=e.unit_id,evidence=refs,verification_status=('checker_supported_provisional' if p.kind=='merge' and any(original[i].verification_status=='checker_supported_provisional' for i in p.facts) else 'verified'),derived_from_fact_ids=[] if p.kind=='correct' else inherited_facts,checked_by=task_id)
                s.types[fid]={'derive':'derived','merge':'merged','correct':'corrected','fact':'extracted','boundary':'boundary'}[p.kind]
                if p.kind in ('merge','correct'):
                    for old in p.facts: k.facts[old].verification_status='superseded'; k.facts[old].superseded_by=fid
                    # Retain historical candidate anchors; add replacement to active question associations.
                    for q in k.questions.values():
                        if set(q.motivating_fact_ids)&set(p.facts): q.motivating_fact_ids=unique(q.motivating_fact_ids+[fid])
                    if p.kind=='correct': updates.extend(self.invalidate_dependents(s,p.facts,fid))
                if p.assessments: s.assessment_dependencies[fid]=p.assessments
                updates.append(fid)
                if p.candidates:
                    aid=next_id(s.assessments,'A',5); s.assessments[aid]={'kind':'repair','text':p.text,'facts':[fid],'questions':p.questions,'candidates':p.candidates,'sections':t.sections,'evidence':[dump(e) for e in refs],'task':task_id,'target':t.prompt}; updates.append(aid)
            elif p.kind in ('answer','partial_answer'):
                q=k.questions[p.questions[0]]; status='answered' if p.kind=='answer' else 'partial'
                if q.status=='answered' and status=='partial': continue
                if q.status==status and q.answer==p.text: continue
                q.status=status; q.answer=p.text; q.answer_fact_ids=inherited_facts; q.answer_evidence=refs; s.assessment_dependencies[q.question_id]=p.assessments; updates.append(q.question_id)
            else:
                if any(a['kind']==p.kind and normalized(a['text'])==normalized(p.text) and a['questions']==p.questions and set(a['sections'])==set(t.sections) and (p.questions or a.get('target')==t.prompt) for a in s.assessments.values()): continue
                # No global absence claim is generated by the LLM. Scoped record is explicit.
                aid=next_id(s.assessments,'A',5); s.assessments[aid]={'kind':p.kind,'text':p.text,'facts':inherited_facts,'questions':p.questions,'candidates':p.candidates,'sections':t.sections,'evidence':[dump(e) for e in refs],'task':task_id,'target':t.prompt}; updates.append(aid)
                if p.assessments: s.assessment_dependencies[aid]=p.assessments
                # A non-finding never resolves a question, even with exhaustive Markdown coverage.
                # Coverage and scoped assessments remain available for planning/reporting.
        if updates: s.revision+=1
        return updates

    def step(self):
        s=self.load()
        if s.termination: return {'event':'terminated','reason':s.termination}
        if s.cursor>=len(s.pending):
            if s.pending:
                s.no_progress=s.no_progress+1 if s.revision==s.round_start_revision else 0
                s.pending=[]; s.cursor=0
            if s.round>=self.config.max_rounds or s.no_progress>=self.config.max_no_progress:
                s.termination='Round limit' if s.round>=self.config.max_rounds else 'No verified progress'; self.save(s); return {'event':'terminated','reason':s.termination}
            # Persist bookkeeping before a call; replay sees same state and cached log.
            self.save(s)
            try:
                snapshot=self.root/'plan_inputs'/f'R{s.round+1:03d}.json'
                if snapshot.exists(): p=json.loads(snapshot.read_text(encoding='utf-8'))
                else:
                    if not self.audit.available(): raise BudgetExceeded('API budget reached')
                    p=self.planner_payload(s); atomic_json(snapshot,p)
                plan,log=self.audit.call(f'R{s.round+1:03d}_planner',PLANNER,p,Plan,self.config.planner,lambda x:None)
            except (BudgetExceeded,StructuredCallFailed,ValueError) as exc:
                s.termination=f'Planner stopped: {exc}'; self.save(s); return {'event':'terminated','reason':s.termination}
            s.round+=1; s.round_start_revision=s.revision; s.pending,issues=self.apply_plan(s,plan,p); s.plans.append({'round':s.round,'plan':dump(plan),'resolved_tasks':[dump(t) for t in s.pending],'validation_errors':issues,'log':log}); s.cursor=0
            if plan.tasks and not s.pending: s.no_progress+=1
            if not plan.tasks: s.termination=plan.reason or 'Planner terminated'
            self.save(s); return {'event':'plan','round':s.round,**dump(plan),'tasks':[dump(t) for t in s.pending],'validation_errors':issues}
        t=s.pending[s.cursor]; ident=f'R{s.round:03d}_T{s.cursor+1:02d}'
        e={'id':ident,'action':t.action,'prompt':t.prompt,'selected':{'facts':t.facts,'questions':t.questions,'candidates':t.candidates,'assessments':t.assessments,'sections':t.sections},'note':'','accepted':[],'rejected':[],'logs':[]}
        try:
            system,payload,schema=self.executor_request(t,s)
            ctx={k:payload[k] for k in ('facts','questions','candidates','assessments','source') if k in payload}
            if len(compact(payload))>self.config.max_context_chars: raise ValueError('Task context exceeds cap; select fewer sections/IDs')
            # Same exact context with no scientific change is not repeated.
            h=fingerprint(payload); e['context_hash']=h
            if any(old.get('context_hash')==h for old in s.episodes): raise ValueError('Identical investigation already attempted; change context or approach')
            w,log=self.audit.call(ident+'_executor',system,payload,schema,self.config.executor,lambda x:None)
            r,issues=wire_result(w,t); r,more=self.filter_result(r,t); e['validation_errors']=issues+more; e['logs'].append(log); e['note']=r.note; e['proposals']=[dump(p) for p in r.proposals]
            if r.proposals:
                if any(p.kind!='question' for p in r.proposals):
                    v,log=self.audit.call(ident+'_verifier',VERIFIER,self.verify_payload(ctx,r),Review,self.config.verifier,lambda x:None); e['logs'].append(log); v,issues=self.normalize_review(v,r); e['validation_errors'].extend(issues)
                else: v=Review(checks=[])
                e['checks']=[dump(c) for c in v.checks]; e['rejected']=[dump(c) for c in v.checks if not c.ok]
                e['unverified_questions']=[i for i,p in enumerate(r.proposals) if p.kind=='question']
                updated=s.model_copy(deep=True); e['accepted']=self.commit(updated,t,r,v,ident); s=updated
        except (ValueError,BudgetExceeded,StructuredCallFailed) as exc:
            e['note']=(e['note']+' | '+str(exc)).strip(' |'); e['error']=str(exc)
            if isinstance(exc,BudgetExceeded): s.termination='API budget reached; unverified proposals were not committed'
        s.episodes.append(e); s.cursor+=1; self.save(s); return {'event':'task',**e}

    def run(self,callback=print):
        while True:
            event=self.step(); callback(event)
            if event['event']=='terminated': return self.load()

    def diagnostics(self):
        rows=[]; state=self.load(); used={p['log'] for p in state.plans}|{log for e in state.episodes for log in e['logs']}
        for path in sorted(self.audit.root.glob('*.json')):
            d=json.loads(path.read_text()); raw=d.get('raw') or {}; u=raw.get('usage') or {}; msg=d.get('messages',[]); user=next((m['content'] for m in msg if m['role']=='user'),''); payload=json.loads(user) if user else {}; content=((raw.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            rows.append({'key':d['key'],'status':d['status'],'used':str(path.relative_to(self.root)) in used,'provider':raw.get('provider'),'input':u.get('prompt_tokens'),'output':u.get('completion_tokens'),'reasoning':(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),'total':u.get('total_tokens'),'cost':u.get('cost'),'system_chars':sum(len(m['content']) for m in msg if m['role']=='system'),'payload_chars':len(user),'source_chars':sum(len(x) for x in payload.get('source',{}).values()),'field_chars':{k:len(compact(x)) for k,x in payload.items()},'schema_chars':len(compact(d['schema_json'])),'response_chars':len(content),'duration':d.get('duration_seconds'),'error':d.get('error')})
        return rows

    def export(self):
        s=self.load(); atomic_json(self.root/'knowledge_store.json',dump(s.knowledge)); atomic_json(self.root/'knowledge_types.json',s.types); atomic_json(self.root/'assessments.json',s.assessments); atomic_json(self.root/'assessment_dependencies.json',s.assessment_dependencies); atomic_json(self.root/'question_evidence.json',s.question_evidence); atomic_json(self.root/'question_triage.json',s.question_triage); atomic_json(self.root/'episodes.json',s.episodes); atomic_json(self.root/'plans.json',s.plans); atomic_json(self.root/'usage.json',{'totals':self.audit.usage(),'calls':self.diagnostics()}); self.write_report(s); return self.root

    def write_report(self,s):
        k=s.knowledge; lines=['# Verified scientific knowledge ledger','',f'Termination: {s.termination or "not finished"}', '', 'Stage I facts are provisional source-grounded author statements. Stage II verification is LLM-based, not independent scientific validation.', '', '## Active knowledge']
        for fid,f in k.facts.items():
            if not active(f): continue
            refs=', '.join(f'{e.section_id}:L{e.line_start}–{e.line_end}' for e in f.evidence)
            lines.extend([f'### {fid} — {s.types.get(fid,"extracted")} ({f.verification_status})',f.text,f'Evidence / inherited premise evidence: {refs}',f'Premises: {", ".join(f.derived_from_fact_ids+s.assessment_dependencies.get(fid,[])) or "—"}',''])
        lines.extend(['## Questions'])
        for qid,q in k.questions.items():
            lines.extend([f'### {qid} — {q.status}',q.question,q.answer or 'Unresolved.',f'Planning triage (not verified knowledge): {s.question_triage.get(qid, {})}',f'Answer facts: {", ".join(q.answer_fact_ids) or "—"}',f'Verified negative-search sections: {", ".join(self.inspected(s,qid)) or "—"}',f'All catalogued substantive sections searched unsuccessfully: {bool(self.scientific_sections and self.scientific_sections<=set(self.inspected(s,qid)))}',''])
        lines.append('## Scoped assessments and repair history')
        for aid,a in s.assessments.items():
            lines.extend([f'### {aid} — {a["kind"]}'+(' (stale)' if a.get('stale') else ''),a['text'],f'Inspected sections: {", ".join(a["sections"]) or "none"}',f'Related facts: {", ".join(a["facts"]) or "—"}',f'Questions: {", ".join(a["questions"]) or "—"}',''])
        lines.extend(['## Archived and disputed records'])
        for fid,f in k.facts.items():
            if not active(f): lines.extend([f'- {fid} [{f.verification_status}] {f.text} (replacement: {f.superseded_by or "—"})'])
        lines.extend(['','## Original rejected candidates'])
        for cid,c in k.candidate_facts.items():
            if c.verdict!='supported': lines.extend([f'- {cid} [{c.verdict}] {c.text} — {c.reason}'])
        (self.root/'scientific_knowledge.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
