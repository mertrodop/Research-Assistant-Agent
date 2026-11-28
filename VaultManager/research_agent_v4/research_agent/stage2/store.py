"""Read a Stage I handoff; persist Stage II separately with idempotent commits."""
import json
from pathlib import Path
from ..stage1.source import PaperSource, ReadingUnit
from ..stage1.pipeline import atomic_json, fingerprint
from ..schemas.scientific import Fact, Question, EvidenceRef
from ..schemas.execution import *

class PaperBundle:
    def __init__(self, handoff_path, source_paths=None, allow_partial=False):
        self.path=Path(handoff_path).resolve()
        self.raw=json.loads(self.path.read_text(encoding='utf-8'))
        if self.raw.get('schema_version')!='stage1-agent-handoff-v2.1':
            raise ValueError('Expected no-summary Stage I v2.1 handoff')
        if not self.raw['reading_complete'] and not allow_partial:
            raise ValueError('Stage I incomplete. Finish extraction or explicitly allow_partial_stage1.')
        p=self.raw['source_paths']
        self.source=PaperSource(*(source_paths or (p['markdown'],p['parsed_paper'],p['section_catalog'])))
        if self.source.sha256!=self.raw['source_sha256'] or self.source.paper_id!=self.raw['paper_id']:
            raise ValueError('Stage I handoff/source mismatch')
        if self.raw['sections']!=self.source.catalog:
            raise ValueError('Stage I handoff/catalog mismatch')
        self.units={u['unit_id']:ReadingUnit.model_validate(u) for u in self.raw['reading_plan']}
        # Skipped sections can be retrieved, using section ID as their evidence unit.
        for sid,s in self.source.sections.items():
            if not any(u.section_id==sid for u in self.units.values()) and s['line_start']<=s['local_end']:
                self.units[sid]=ReadingUnit(unit_id=sid,section_id=sid,title=s['title'],kind=s['kind'],heading_path=s.get('heading_path',[]),line_start=s['line_start'],line_end=s['local_end'],oversized_atomic_block=False)
        for uid,u in self.units.items():
            s=self.source.sections[u.section_id]
            if not s['line_start']<=u.line_start<=u.line_end<=s['local_end']:
                raise ValueError('Invalid reading unit boundaries: '+uid)
        for sid,s in self.source.sections.items():
            if s['line_start']>s['local_end']: continue
            spans=sorted((u.line_start,u.line_end) for u in self.units.values() if u.section_id==sid)
            cursor=s['line_start']
            for a,z in spans:
                if a!=cursor: raise ValueError('Reading units do not cover complete local section '+sid)
                cursor=z+1
            if cursor!=s['local_end']+1: raise ValueError('Incomplete local section '+sid)
        self.initial=Stage2Knowledge.model_validate({k:self.raw[k] for k in ('paper_id','facts','candidate_facts','questions','clusters','contradictions')})
        self.validate_initial()
        self.identity=fingerprint({'handoff':self.raw,'parsed':self.source.parsed,'catalog':self.source.catalog})

    def check_evidence(self,e):
        u=self.units.get(e.unit_id)
        if not u or u.section_id!=e.section_id or not u.line_start<=e.line_start<=e.line_end<=u.line_end:
            raise ValueError(f'Invalid evidence reference: {e}')

    def validate_initial(self):
        k=self.initial
        for fid,f in k.facts.items():
            if fid!=f.fact_id or not f.evidence: raise ValueError('Invalid fact: '+fid)
            for e in f.evidence: self.check_evidence(e)
            if f.source_candidate_id not in k.candidate_facts: raise ValueError('Missing source candidate')
        for cid,c in k.candidate_facts.items():
            if cid!=c.candidate_id or (c.verdict=='supported')!=(c.accepted_fact_id in k.facts): raise ValueError('Invalid candidate mapping: '+cid)
            if c.accepted_fact_id and k.facts[c.accepted_fact_id].source_candidate_id!=cid: raise ValueError('Invalid candidate backlink')
            for e in c.evidence: self.check_evidence(e)
        for qid,q in k.questions.items():
            if qid!=q.question_id or not set(q.motivating_candidate_ids)<=k.candidate_facts.keys() or not set(q.motivating_fact_ids)<=k.facts.keys(): raise ValueError('Invalid question anchors: '+qid)
            if set(q.motivating_candidate_ids)!={a.candidate_id for a in q.motivating_candidates}: raise ValueError('Invalid anchor list')
            for a in q.motivating_candidates:
                c=k.candidate_facts[a.candidate_id]
                if (a.verdict,a.accepted_fact_id,a.checker_reason)!=(c.verdict,c.accepted_fact_id,c.reason): raise ValueError('Invalid anchor snapshot')

class Stage2Store:
    def __init__(self,bundle,run_dir):
        self.bundle=bundle; self.root=Path(run_dir).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'ledger.json'
        if not self.path.exists(): self.save(Ledger(knowledge=bundle.initial))
        else: self.load()
    def load(self): return Ledger.model_validate_json(self.path.read_text(encoding='utf-8'))
    def save(self,ledger):
        atomic_json(self.path,ledger.model_dump(mode='json'))
        self.export(ledger)
    def export(self,ledger=None):
        l=ledger or self.load()
        atomic_json(self.root/'knowledge_store.json',l.knowledge.model_dump(mode='json'))
        atomic_json(self.root/'episodic_memory.json',[x.model_dump(mode='json') for x in l.episodes.values()])
    def commit_round(self,round_number,items,evaluation,log_ref,verifications):
        """Commit all accepted updates atomically; replay a saved round without writing twice."""
        l=self.load(); key=str(round_number)
        if key in l.round_commits:
            self.export(l); return [l.episodes[i] for i in l.round_commits[key]]
        decisions={e.task_id:e for e in evaluation.task_evaluations}
        from .validation import validate_verification_record, subset
        # Enforce eligibility at the persistent write boundary too.
        for aid,task,result,proposals in items:
            v=verifications[task.task_id]; validate_verification_record(v,proposals)
            subset(decisions[task.task_id].accepted_proposal_indices,v.eligible_indices,'verified eligible proposal indices')
        episodes=[]
        for aid,task,result,proposals in items:
            episodes.append(self._apply(l,round_number,aid,task,result,decisions[task.task_id],proposals,log_ref,verifications[task.task_id]))
        l.round_commits[key]=[e.attempt_id for e in episodes]; l.revision+=1
        self.save(l); return episodes

    def _apply(self,l,round_number,attempt_id,task,result,evaluation,proposals,log_ref,verification):
        k=l.knowledge; changed=[]
        def new_id(prefix,mapping):
            n=max([int(x[len(prefix):]) for x in mapping if x.startswith(prefix) and x[len(prefix):].isdigit()] or [0])+1
            width=4 if prefix=='Q' else 5
            return f'{prefix}{n:0{width}d}'
        for i in evaluation.accepted_proposal_indices:
            kind,p=proposals[i]
            if kind in ('new_fact','merged_fact'):
                existing=next((f for f in k.facts.values() if f.verification_status!='superseded' and f.text.strip().casefold()==p.text.strip().casefold()),None)
                if existing and kind=='new_fact': continue
                fid=new_id('F',k.facts); first=p.evidence[0]
                k.facts[fid]=Fact(fact_id=fid,text=p.text,origin_section_id=first.section_id,origin_unit_id=first.unit_id,evidence=p.evidence,derived_from_fact_ids=p.derived_from_fact_ids,verification_status='verified',checked_by=verification.log or log_ref)
                if kind=='merged_fact':
                    for old in p.supersede_ids:
                        k.facts[old].verification_status='superseded'; k.facts[old].superseded_by=fid
                else:
                    for old in p.derived_from_fact_ids:
                        if fid not in k.facts[old].supplemented_by: k.facts[old].supplemented_by.append(fid)
                changed.append(fid)
            elif kind=='fact_issue':
                existing=k.fact_issues.get(p.fact_id)
                if existing==p: continue
                k.fact_issues[p.fact_id]=p
                if k.facts[p.fact_id].verification_status!='superseded': k.facts[p.fact_id].verification_status='disputed'
                changed.append(p.fact_id)
                # Keep historical answers in logs/group records, but reopen answers relying on disputed facts.
                for q in k.questions.values():
                    if p.fact_id in q.answer_fact_ids and q.status in ('answered','partial','unanswerable'):
                        q.status='open'; q.answer=None; q.answer_fact_ids=[]; q.answer_evidence=[]; changed.append(q.question_id)
            elif kind=='contradiction':
                # Same target pair: replace assessment only if changed; old assessment remains in task log.
                cid=next((i for i,c in k.contradictions.items() if set(c.assessment.fact_ids)==set(p.fact_ids)),None)
                if cid and k.contradictions[cid].assessment==p: continue
                cid=cid or new_id('CT',k.contradictions); k.contradictions[cid]=Contradiction(contradiction_id=cid,assessment=p,task_id=task.task_id); changed.append(cid)
            elif kind=='answer':
                # One shared answer record, with individual question dispositions; original IDs preserved.
                if any(r.status!='unresolved' for r in p.resolutions) and p not in k.answer_groups.values():
                    gid=new_id('A',k.answer_groups); k.answer_groups[gid]=p; changed.append(gid)
                for r in p.resolutions:
                    q=k.questions[r.question_id]
                    if r.status=='unresolved': continue # preserve existing supported answer/status
                    answer=r.explanation
                    if (q.status,q.answer,q.answer_fact_ids,q.answer_evidence)==(r.status,answer,r.answer_fact_ids,r.evidence): continue
                    q.status=r.status; q.answer=answer; q.answer_fact_ids=r.answer_fact_ids; q.answer_evidence=r.evidence; changed.append(q.question_id)
            elif kind=='new_question':
                if any(q.question.strip().casefold()==p.question.strip().casefold() for q in k.questions.values()): continue
                qid=new_id('Q',k.questions); e=p.evidence[0]
                # Stage I archive fields stay intact; Stage II creates fact/source-anchored questions.
                k.questions[qid]=Question(question_id=qid,question=p.question,origin_section_id=e.section_id,origin_unit_id=e.unit_id,motivating_candidate_ids=[],motivating_fact_ids=p.motivating_fact_ids,motivating_candidates=[])
                k.question_evidence[qid]=p.evidence; changed.append(qid)
            elif kind=='gap':
                if any(g.section_id==p.section_id and g.description.casefold()==p.description.casefold() for g in k.coverage_gaps.values()): continue
                gid=new_id('G',k.coverage_gaps); k.coverage_gaps[gid]=p; changed.append(gid)
        episode=Episode(attempt_id=attempt_id,task_id=task.task_id,action=task.action,round_number=round_number,objective=task.task_prompt,outcome_summary=result.outcome_summary,status=result.status,task_assessment=result.task_assessment,evaluation=evaluation.rationale,evidence_refs=result.evidence_refs,information_requests=result.information_requests,remaining_uncertainties=result.remaining_uncertainties,failure_reason=result.failure_reason,accepted_updates=changed,verification=verification)
        l.episodes[attempt_id]=episode
        return episode
