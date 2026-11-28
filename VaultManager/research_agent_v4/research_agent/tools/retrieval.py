"""Deterministic read-only retrieval; same implementation behind MCP and offline tests."""
import json, re
from ..schemas.planning import Task
from ..schemas.scientific import EvidenceRef


def dump(x):
    return x.model_dump(mode='json')

def fact_view(f):
    return {'fact_id':f.fact_id,'text':f.text,'status':f.verification_status,'evidence':[dump(e) for e in f.evidence]}

def candidate_view(c):
    return {'candidate_id':c.candidate_id,'text':c.text,'verdict':c.verdict,'reason':c.reason,'accepted_fact_id':c.accepted_fact_id,'evidence':[dump(e) for e in c.evidence]}

def question_view(q):
    return {k:v for k,v in dump(q).items() if k not in ('created_at','motivating_candidates','motivating_candidate_ids')}

class Retrieval:
    def __init__(self,store,max_search_hits=6):
        self.store=store; self.bundle=store.bundle; self.source=self.bundle.source; self.max_search_hits=max_search_hits
    def get_facts(self,ids):
        k=self.store.load().knowledge; return [fact_view(k.facts[i]) for i in ids]
    def get_candidates(self,ids):
        k=self.store.load().knowledge; return [candidate_view(k.candidate_facts[i]) for i in ids]
    def get_questions(self,ids):
        k=self.store.load().knowledge; return [question_view(k.questions[i]) for i in ids]
    def get_section_catalog(self):
        return [{'section_id':s['section_id'],'title':s['title'],'kind':s['kind'],'line_start':s['line_start'],'local_end':s['local_end']} for s in self.source.catalog]
    def read_lines(self,span):
        e=EvidenceRef.model_validate(span); self.bundle.check_evidence(e)
        return {'ref':dump(e),'text':self.source.original_lines(e.line_start,e.line_end)}
    def section_spans(self,section_id):
        if section_id not in self.source.sections: raise ValueError('Unknown section '+section_id)
        return [EvidenceRef(section_id=u.section_id,unit_id=u.unit_id,line_start=u.line_start,line_end=u.line_end) for u in sorted(self.bundle.units.values(),key=lambda u:u.line_start) if u.section_id==section_id]
    def read_section(self,section_id):
        return [self.read_lines(dump(e)) for e in self.section_spans(section_id)]
    def object_spans(self,kind,object_id):
        key=kind+'_id'; objs=self.source.parsed.get(kind+'s',[])
        obj=next((o for o in objs if o[key]==object_id),None)
        if obj is None: raise ValueError('Unknown '+kind+' '+object_id)
        a=min(obj['line_start'],obj.get('caption_line') or obj['line_start']); z=obj['line_end']
        spans=[]
        for u in self.bundle.units.values():
            first,last=max(a,u.line_start),min(z,u.line_end)
            if first<=last: spans.append(EvidenceRef(section_id=u.section_id,unit_id=u.unit_id,line_start=first,line_end=last))
        if not spans: raise ValueError('Object has no valid source spans')
        return spans
    def read_table(self,table_id):
        return [self.read_lines(dump(e)) for e in self.object_spans('table',table_id)]
    def read_equation(self,equation_id):
        return [self.read_lines(dump(e)) for e in self.object_spans('equation',equation_id)]
    def search_paper(self,query,top_k=6):
        words=set(re.findall(r'\w+',query.casefold()))-{'the','of','a','an','in','and','is','to','for','what','how','are','was','with'}
        if not words: return []
        hits=[]
        for b in self.source.parsed['blocks']:
            if b['kind']=='heading' or self.source.sections[b['section_id']]['kind']=='bibliography': continue
            a,z=b['line_start'],b['line_end']; text=self.source.original_lines(a,z,False).casefold()
            score=sum(w in text for w in words)+3*(query.casefold() in text)
            if not score: continue
            for u in self.bundle.units.values():
                if u.section_id==b['section_id'] and u.line_start<=a<=z<=u.line_end:
                    hits.append((score,a,EvidenceRef(section_id=u.section_id,unit_id=u.unit_id,line_start=a,line_end=z))); break
        return [self.read_lines(dump(e)) for _,_,e in sorted(hits,key=lambda x:(-x[0],x[1]))[:min(top_k,self.max_search_hits)]]
    def search_facts(self,query,top_k=10):
        words=set(re.findall(r'\w+',query.casefold())); k=self.store.load().knowledge
        ranked=[(sum(w in f.text.casefold() for w in words),fid) for fid,f in k.facts.items() if f.verification_status!='superseded']
        return [fact_view(k.facts[fid]) for score,fid in sorted(ranked,key=lambda x:(-x[0],x[1]))[:top_k] if score]
    def get_task_history(self,ids):
        l=self.store.load(); return [dump(e) for e in l.episodes.values() if e.task_id in ids or e.attempt_id in ids]
    def build_context(self,task,max_chars):
        """Retrieve exactly the planner's selected IDs from one ledger revision.

        The full lightweight catalog is navigation metadata, not source evidence.
        No question-anchor expansion, candidate retrieval, or implicit source text.
        """
        from ..stage2.validation import validate_task
        task=Task.model_validate(task); l=self.store.load(); k=l.knowledge
        validate_task(task,k,self.bundle)
        context={
            'task':dump(task),
            'knowledge_revision':l.revision,
            'facts':[fact_view(k.facts[i]) for i in task.fact_ids],
            'questions':[question_view(k.questions[i]) for i in task.question_ids],
            'passages':[p for sid in task.section_ids for p in self.read_section(sid)],
            'source_sections':[{'section_id':sid,'title':self.source.sections[sid]['title']} for sid in task.section_ids],
            'section_catalog':self.get_section_catalog(),
        }
        size=len(json.dumps(context,ensure_ascii=False))
        if size>max_chars:
            return {'error':f'Context {size} characters exceeds {max_chars}; select fewer IDs or one complete section. No source was truncated.',
                    'task':dump(task),'knowledge_revision':l.revision,'facts':[],'questions':[],'passages':[],
                    'source_sections':[],'section_catalog':[]}
        return context
