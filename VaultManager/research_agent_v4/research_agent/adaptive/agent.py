"""Adaptive planner / scoped executor / verifier. Serial, resumable, no hidden rereads."""
import json
from pathlib import Path
from .schema import Config, State, Plan, Task, Result, Review
from .prompts import PLANNER, EXECUTOR, VERIFIER
from ..schemas.scientific import KnowledgeStore, Fact, Question, EvidenceRef, CandidateAnchor
from ..stage2.store import PaperBundle
from ..stage2.llm import AuditedLLM, BudgetExceeded, StructuredCallFailed
from ..stage1.pipeline import atomic_json, fingerprint

VERSION = 'adaptive-v1.1'
def dump(x): return x.model_dump(mode='json')
def compact(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def normalized(s): return ' '.join(s.casefold().split())
def next_id(d,prefix,width):
    return prefix+str(max([int(k[len(prefix):]) for k in d if k.startswith(prefix) and k[len(prefix):].isdigit()] or [0])+1).zfill(width)
def unique(xs): return list(dict.fromkeys(xs))
def active(f): return f.verification_status not in ('superseded','disputed')

ACTION_KINDS = {'MERGE':{'merge','question'},'ANSWER':{'answer','partial_answer','derive','boundary','question'},'INVESTIGATE':{'fact','answer','partial_answer','not_found','question','boundary','derive'},'QUESTION':{'question'},'AUDIT':{'boundary','derive','answer','partial_answer','question','contradiction','fact','not_found'},'REPAIR':{'fact','correct','question','boundary'},'COVERAGE':{'fact','question'},'CONTRADICTION':{'contradiction','correct','question','boundary'}}

def allowed_kinds(t):
    kinds=set(ACTION_KINDS[t.action])
    if not t.questions: kinds-={'answer','partial_answer','not_found'}
    if not t.facts: kinds.discard('derive')
    if len(t.facts)<2: kinds-={'merge','contradiction'}
    if not t.facts: kinds.discard('correct')
    if not t.sections: kinds-={'fact','correct','merge','not_found'}
    return sorted(kinds)

class AdaptiveAgent:
    def __init__(self,handoff_path,config,llm,run_dir,source_paths=None):
        self.config=config; self.root=Path(run_dir); self.root.mkdir(parents=True,exist_ok=True)
        self.bundle=PaperBundle(handoff_path,source_paths,config.allow_partial_stage1)
        self.source=self.bundle.source
        self.path=self.root/'state.json'; self.manifest=self.root/'manifest.json'
        identity=fingerprint({'version':VERSION,'bundle':self.bundle.identity,'config':dump(config),'llm':llm.identity,'prompts':[PLANNER,EXECUTOR,VERIFIER], 'schemas':[x.model_json_schema() for x in (Plan,Result,Review)]})
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
        # Local section IDs with non-heading content; used only for explicit search coverage.
        self.scientific_sections={s['section_id'] for s in self.source.catalog if s['kind']!='bibliography' and any(b['kind'] not in ('heading','rule') for b in self.source.blocks.get(s['section_id'],[]))}

    def load(self): return State.model_validate_json(self.path.read_text(encoding='utf-8'))
    def save(self,s): atomic_json(self.path,dump(s))
    def sections_for(self,f): return sorted({e.section_id for e in f.evidence})
    def inspected(self,s,qid):
        return sorted({sid for a in s.assessments.values() if a['kind']=='not_found' and not a.get('stale') and qid in a['questions'] for sid in a['sections']})
    def planner_payload(self,s):
        k=s.knowledge; usage=self.audit.usage()
        # Compact keyed dictionaries, no repeated IDs, timestamps, candidate backlinks or line spans.
        facts={fid:[f.text,self.sections_for(f),s.types.get(fid,'extracted')] for fid,f in k.facts.items() if active(f)}
        questions={qid:[q.question,q.status,[f for f in q.motivating_fact_ids if f in facts],q.answer or '',self.inspected(s,qid), sorted({q.origin_section_id}|{e.section_id for cid in q.motivating_candidate_ids if cid in k.candidate_facts for e in k.candidate_facts[cid].evidence})] for qid,q in k.questions.items()}
        candidates={cid:[c.text,c.verdict,c.reason,self.sections_for(c)] for cid,c in k.candidate_facts.items() if c.verdict!='supported' and not any(cid in a.get('candidates',[]) and a['kind']=='repair' for a in s.assessments.values())}
        sections={x['section_id']:[x['title'],x['kind'],x['line_start'],x['local_end']] for x in self.source.catalog}
        assessments={aid:[a['kind'],a['text'],a['facts'],a['questions'],a['sections']] for aid,a in s.assessments.items() if not a.get('stale')}
        recent=[{key:e[key] for key in ('id','action','prompt','selected','note','accepted','rejected') if key in e} for e in s.episodes[-self.config.recent_episodes:]]
        p={'goal':self.config.objective,'round':s.round+1,'limits':{'tasks':self.config.max_tasks_per_round,'sections_per_task':self.config.max_sections_per_task,'facts_per_task':self.config.max_selected_facts,'rounds_left':self.config.max_rounds-s.round,'calls_left':self.config.max_llm_calls-usage['calls'],'reported_tokens_left':self.config.max_reported_tokens-usage['reported_tokens'],'new_questions_left':max(0,self.config.max_new_questions-(len(k.questions)-len(self.bundle.initial.questions)))},'legend':{'facts':'[text,sections,type]','questions':'[text,status,active motivating facts,answer,verified negative-search sections,origin sections]','candidates':'[text,verdict,reason,sections]','sections':'[title,kind,first line,last line]','assessments':'[kind,text,facts,questions,sections]'},'facts':facts,'questions':questions,'candidates':candidates,'sections':sections,'assessments':assessments,'recent':recent}
        # Bound planner memory explicitly; retain all question anchors and all IDs, omit older episode details first.
        while len(compact(p))>self.config.max_planner_chars and p['recent']: p['recent'].pop(0)
        if len(compact(p))>self.config.max_planner_chars: raise ValueError('Planner context cap exceeded; no hidden truncation. Narrow scope or increase cap.')
        return p

    def validate_task(self,t,s):
        k=s.knowledge
        for name,ids,allowed in [('facts',t.facts,{i for i,f in k.facts.items() if active(f)}),('questions',t.questions,set(k.questions)),('candidates',t.candidates,{i for i,c in k.candidate_facts.items() if c.verdict!='supported'}),('sections',t.sections,set(self.source.sections))]:
            if len(ids)!=len(set(ids)) or not set(ids)<=allowed: raise ValueError('Unknown, inactive or duplicate '+name)
        if len(t.sections)>self.config.max_sections_per_task or len(t.facts)>self.config.max_selected_facts: raise ValueError('Task scope exceeds limit')
        if t.action in ('ANSWER','QUESTION') and t.sections: raise ValueError('No manuscript for memory-only task')
        if t.action in ('MERGE','INVESTIGATE','REPAIR','COVERAGE','CONTRADICTION') and not t.sections: raise ValueError('Source sections required')
        if t.action in ('MERGE','CONTRADICTION') and len(t.facts)<2: raise ValueError('At least two facts required')
        if t.action=='ANSWER' and (not t.questions or not t.facts): raise ValueError('ANSWER requires existing question IDs and fact IDs')
        if t.action=='INVESTIGATE' and not t.questions: raise ValueError('INVESTIGATE requires existing question IDs; use QUESTION to create new questions, or COVERAGE for source-only discovery')
        if t.action=='QUESTION' and not t.facts: raise ValueError('Question needs facts')
        if t.action=='AUDIT' and not (t.facts or t.questions): raise ValueError('Audit needs a claim or question')
        if t.action=='REPAIR' and not (t.candidates or t.facts): raise ValueError('Repair needs a target')
        if t.candidates and t.action not in ('REPAIR','INVESTIGATE','QUESTION','AUDIT'): raise ValueError('Rejected candidates are investigation leads only')

    def validate_plan(self,p,s,payload):
        if len(p.tasks)>self.config.max_tasks_per_round: raise ValueError('Too many tasks')
        if not p.tasks and not p.reason.strip(): raise ValueError('Explain termination')
        seen=set()
        for i,t in enumerate(p.tasks,1):
            try: self.validate_task(t,s)
            except ValueError as exc: raise ValueError(f'Task {i} ({t.action}): {exc}') from exc
            if not set(t.facts)<=set(payload['facts']) or not set(t.candidates)<=set(payload['candidates']): raise ValueError('Select visible IDs')
            h=fingerprint(dump(t))
            if h in seen: raise ValueError('Duplicate task')
            seen.add(h)

    def context(self,t,s):
        self.validate_task(t,s); k=s.knowledge
        return {'facts':{i:[k.facts[i].text,s.types.get(i,'extracted'),[[e.line_start,e.line_end] for e in k.facts[i].evidence]] for i in t.facts},'questions':{i:[k.questions[i].question,k.questions[i].status,k.questions[i].answer or ''] for i in t.questions},'candidates':{i:[k.candidate_facts[i].text,k.candidate_facts[i].reason,[[e.line_start,e.line_end] for e in k.candidate_facts[i].evidence]] for i in t.candidates},'source':{sid:self.source.read_section(sid) for sid in t.sections}}

    def resolve_spans(self,spans,sections):
        refs=[]
        for sp in spans:
            if sp.a>sp.b: raise ValueError('Reversed span')
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
            for name,ids,scope in [('facts',p.facts,t.facts),('questions',p.questions,t.questions),('candidates',p.candidates,t.candidates)]:
                unknown=sorted(set(ids)-set(scope))
                if unknown: raise ValueError(f'Proposal {i}: unselected {name} IDs {unknown}; permitted IDs {scope}')
                if len(ids)!=len(set(ids)): raise ValueError(f'Proposal {i}: duplicate {name} IDs {ids}')
            self.resolve_spans(p.spans,t.sections)
            if not (p.spans or p.facts or p.candidates):
                if p.kind!='not_found': raise ValueError('Proposal must have grounded anchors')
            if p.kind in ('fact','correct','merge') and not p.spans: raise ValueError('Explicit or replacement fact needs source evidence')
            if p.kind in ('derive','boundary','answer','partial_answer','contradiction') and not (p.facts or p.spans): raise ValueError('Rejected candidates cannot serve as premises')
            if p.kind=='derive' and not p.facts: raise ValueError('Derivation requires facts')
            if p.kind in ('merge','contradiction') and len(p.facts)<2: raise ValueError('Need >=2 facts')
            if p.kind=='correct' and len(p.facts)!=1: raise ValueError('Correct one fact at a time')
            if p.kind in ('merge','correct'):
                if replaces.intersection(p.facts): raise ValueError('Overlapping replacements')
                replaces.update(p.facts)
                if p.kind=='correct': corrections.update(p.facts)
            if p.kind in ('answer','partial_answer','not_found'):
                if len(p.questions)!=1: raise ValueError(f'Proposal {i}: {p.kind} requires exactly one selected Q ID; got {p.questions}; selected={t.questions}')
                if answers.intersection(p.questions): raise ValueError(f'Proposal {i}: question {p.questions[0]} already has a resolution in this result; combine subanswers into one answer OR partial_answer OR not_found')
                answers.update(p.questions)
            if p.kind=='not_found' and not t.sections: raise ValueError('Negative search requires sections')
            key=(p.kind,normalized(p.text),tuple(p.questions))
            if key in seen: raise ValueError('Duplicate proposal')
            seen.add(key)

        for p in r.proposals:
            if p.kind not in ('correct','merge') and corrections.intersection(p.facts): raise ValueError('Do not use a corrected premise in another proposal from the same task; replan after commit')

    def verify_payload(self,ctx,r):
        # All scoped sections for absence, otherwise only sections containing cited lines.
        needs_all=any(p.kind=='not_found' for p in r.proposals)
        source=ctx['source'] if needs_all else {sid:text for sid,text in ctx['source'].items() if any(self.source.sections[sid]['line_start']<=sp.b and self.source.sections[sid]['local_end']>=sp.a for p in r.proposals for sp in p.spans)}
        return {**{k:ctx[k] for k in ('facts','questions','candidates')},'source':source,'proposals':[dump(p) for p in r.proposals]}

    def validate_review(self,v,r):
        if sorted(c.i for c in v.checks)!=list(range(len(r.proposals))): raise ValueError('Exactly one check per proposal')
        if any(not c.ok and not c.reason.strip() for c in v.checks): raise ValueError('Rejected proposal needs reason')

    def invalidate_dependents(self,s,old_ids,replacement):
        """A verified correction can invalidate descendants; never erase history.
        Old merged aliases also count as premises of the corrected canonical fact.
        Reopened answers remain in audit logs but leave active question memory.
        """
        k=s.knowledge; invalid=set(old_ids); changed=[]
        while True:
            prev=set(invalid)
            for fid,f in k.facts.items():
                if fid==replacement: continue
                if f.superseded_by in invalid or set(f.derived_from_fact_ids)&invalid:
                    invalid.add(fid)
                    if active(f): f.verification_status='disputed'; changed.append(fid)
            if invalid==prev: break
        for q in k.questions.values():
            if set(q.answer_fact_ids)&invalid:
                q.status='open'; q.answer=None; q.answer_fact_ids=[]; q.answer_evidence=[]; changed.append(q.question_id)
        for aid,a in s.assessments.items():
            if a['kind']!='not_found' and set(a['facts'])&invalid:
                a['stale']=True; changed.append(aid)
        return unique(changed)

    def commit(self,s,t,r,v,task_id):
        k=s.knowledge; updates=[]; checks={c.i:c for c in v.checks}; original={i:k.facts[i] for i in t.facts}
        # Snapshot original references so same-result proposals remain valid even after a merge.
        for i,p in enumerate(r.proposals):
            if not checks[i].ok: continue
            refs=self.resolve_spans(p.spans,t.sections)
            if p.kind in ('derive','merge','answer','partial_answer','boundary','contradiction','question'):
                refs=self.dedup_refs(refs+[e for fid in p.facts for e in original[fid].evidence])
            if p.kind=='question':
                if len(k.questions)-len(self.bundle.initial.questions)>=self.config.max_new_questions: continue
                if any(normalized(q.question)==normalized(p.text) for q in k.questions.values()): continue
                cids=p.candidates; anchors=[CandidateAnchor(candidate_id=cid,accepted_fact_id=k.candidate_facts[cid].accepted_fact_id,verdict=k.candidate_facts[cid].verdict,checker_reason=k.candidate_facts[cid].reason) for cid in cids]
                if not refs: refs=self.dedup_refs([e for cid in cids for e in k.candidate_facts[cid].evidence])
                if not refs: continue
                qid=next_id(k.questions,'Q',4); e=refs[0]
                k.questions[qid]=Question(question_id=qid,question=p.text,origin_section_id=e.section_id,origin_unit_id=e.unit_id,motivating_candidate_ids=cids,motivating_fact_ids=p.facts,motivating_candidates=anchors,related_question_ids=p.questions)
                s.question_evidence[qid]=[dump(x) for x in refs]
                updates.append(qid)
            elif p.kind in ('fact','derive','merge','correct','boundary'):
                if p.kind not in ('merge','correct') and any(active(f) and normalized(f.text)==normalized(p.text) for f in k.facts.values()): continue
                if not refs: continue
                fid=next_id(k.facts,'F',5); e=refs[0]
                k.facts[fid]=Fact(fact_id=fid,text=p.text,origin_section_id=e.section_id,origin_unit_id=e.unit_id,evidence=refs,verification_status='verified',derived_from_fact_ids=p.facts,checked_by=task_id)
                s.types[fid]={'derive':'derived','merge':'merged','correct':'corrected','fact':'extracted','boundary':'boundary'}[p.kind]
                if p.kind in ('merge','correct'):
                    for old in p.facts: k.facts[old].verification_status='superseded'; k.facts[old].superseded_by=fid
                    # Retain historical candidate anchors; add replacement to active question associations.
                    for q in k.questions.values():
                        if set(q.motivating_fact_ids)&set(p.facts): q.motivating_fact_ids=unique(q.motivating_fact_ids+[fid])
                    if p.kind=='correct': updates.extend(self.invalidate_dependents(s,p.facts,fid))
                updates.append(fid)
                if p.candidates:
                    aid=next_id(s.assessments,'A',5); s.assessments[aid]={'kind':'repair','text':p.text,'facts':[fid],'questions':p.questions,'candidates':p.candidates,'sections':t.sections,'evidence':[dump(e) for e in refs],'task':task_id}; updates.append(aid)
            elif p.kind in ('answer','partial_answer'):
                q=k.questions[p.questions[0]]; status='answered' if p.kind=='answer' else 'partial'
                if q.status=='answered' and status=='partial': continue
                if q.status==status and q.answer==p.text: continue
                q.status=status; q.answer=p.text; q.answer_fact_ids=p.facts; q.answer_evidence=refs; updates.append(q.question_id)
            else:
                if any(a['kind']==p.kind and normalized(a['text'])==normalized(p.text) and a['questions']==p.questions and a['sections']==t.sections for a in s.assessments.values()): continue
                # No global absence claim is generated by the LLM. Scoped record is explicit.
                aid=next_id(s.assessments,'A',5); s.assessments[aid]={'kind':p.kind,'text':p.text,'facts':p.facts,'questions':p.questions,'candidates':p.candidates,'sections':t.sections,'evidence':[dump(e) for e in refs],'task':task_id}; updates.append(aid)
                if p.kind=='not_found':
                    qid=p.questions[0]; q=k.questions[qid]; inspected=set(self.inspected(s,qid))
                    if self.scientific_sections and self.scientific_sections<=inspected and q.status=='open':
                        q.status='unanswerable'; q.answer='No answer located in the verified searches of all catalogued non-bibliography sections with substantive text. This is a search outcome, not proof of absolute absence.'; updates.append(qid)
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
                plan,log=self.audit.call(f'R{s.round+1:03d}_planner',PLANNER,p,Plan,self.config.planner,lambda x:self.validate_plan(x,s,p))
            except (BudgetExceeded,StructuredCallFailed,ValueError) as exc:
                s.termination=f'Planner stopped: {exc}'; self.save(s); return {'event':'terminated','reason':s.termination}
            s.round+=1; s.plans.append({'round':s.round,'plan':dump(plan),'log':log}); s.pending=plan.tasks; s.cursor=0; s.round_start_revision=s.revision
            if not plan.tasks: s.termination=plan.reason
            self.save(s); return {'event':'plan','round':s.round,**dump(plan)}
        t=s.pending[s.cursor]; ident=f'R{s.round:03d}_T{s.cursor+1:02d}'
        e={'id':ident,'action':t.action,'prompt':t.prompt,'selected':{'facts':t.facts,'questions':t.questions,'candidates':t.candidates,'sections':t.sections},'note':'','accepted':[],'rejected':[],'logs':[]}
        try:
            ctx=self.context(t,s); payload={'action':t.action,'prompt':t.prompt,'max_proposals':self.config.max_proposals,'allowed_kinds':allowed_kinds(t),**ctx}
            if len(compact(payload))>self.config.max_context_chars: raise ValueError('Task context exceeds cap; select fewer sections/IDs')
            h=fingerprint(payload); e['context_hash']=h
            if any(old.get('context_hash')==h for old in s.episodes): raise ValueError('Identical investigation already attempted; change context or approach')
            r,log=self.audit.call(ident+'_executor',EXECUTOR,payload,Result,self.config.executor,lambda x:self.validate_result(x,t)); e['logs'].append(log); e['note']=r.note; e['proposals']=[dump(p) for p in r.proposals]
            if r.proposals:
                v,log=self.audit.call(ident+'_verifier',VERIFIER,self.verify_payload(ctx,r),Review,self.config.verifier,lambda x:self.validate_review(x,r)); e['logs'].append(log); e['checks']=[dump(c) for c in v.checks]; e['rejected']=[dump(c) for c in v.checks if not c.ok]
                # Work on a copy; never partially commit on exception.
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
            rows.append({'key':d['key'],'status':d['status'],'used':str(path.relative_to(self.root)) in used,'provider':raw.get('provider'),'input':u.get('prompt_tokens'),'output':u.get('completion_tokens'),'reasoning':(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),'total':u.get('total_tokens'),'cost':u.get('cost'),'system_chars':sum(len(m['content']) for m in msg if m['role']=='system'),'payload_chars':len(user),'source_chars':sum(len(x) for x in payload.get('source',{}).values()),'field_chars':{k:len(compact(x)) for k,x in payload.items()},'schema_chars':len(compact({'Plan':Plan,'Result':Result,'Review':Review}[d['schema']].model_json_schema())),'response_chars':len(content),'duration':d.get('duration_seconds'),'error':d.get('error')})
        return rows

    def export(self):
        s=self.load(); atomic_json(self.root/'knowledge_store.json',dump(s.knowledge)); atomic_json(self.root/'knowledge_types.json',s.types); atomic_json(self.root/'assessments.json',s.assessments); atomic_json(self.root/'question_evidence.json',s.question_evidence); atomic_json(self.root/'episodes.json',s.episodes); atomic_json(self.root/'plans.json',s.plans); atomic_json(self.root/'usage.json',{'totals':self.audit.usage(),'calls':self.diagnostics()}); self.write_report(s); return self.root

    def write_report(self,s):
        k=s.knowledge; lines=['# Verified scientific knowledge ledger','',f'Termination: {s.termination or "not finished"}', '', 'Stage I facts are provisional source-grounded author statements. Stage II verification is LLM-based, not independent scientific validation.', '', '## Active knowledge']
        for fid,f in k.facts.items():
            if not active(f): continue
            refs=', '.join(f'{e.section_id}:L{e.line_start}–{e.line_end}' for e in f.evidence)
            lines.extend([f'### {fid} — {s.types.get(fid,"extracted")} ({f.verification_status})',f.text,f'Evidence / inherited premise evidence: {refs}',f'Premises: {", ".join(f.derived_from_fact_ids) or "—"}',''])
        lines.extend(['## Questions'])
        for qid,q in k.questions.items():
            lines.extend([f'### {qid} — {q.status}',q.question,q.answer or 'Unresolved.',f'Answer facts: {", ".join(q.answer_fact_ids) or "—"}',f'Verified negative-search sections: {", ".join(self.inspected(s,qid)) or "—"}',''])
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
