"""Stage I only: deterministic reader → checker → commit. No Stage II agents."""
import hashlib, json, time, uuid
from pathlib import Path
from pydantic import BaseModel
from ..core.llm_client import parse_completion
from ..schemas.extraction import ReaderOutput, CheckerOutput
from ..schemas.scientific import KnowledgeStore, utc_now
from ..schemas.checkpoint import ReadingCheckpoint
from ..memory.knowledge_store import commit_unit
from .prompts import reader_messages, checker_messages, READER_SYSTEM, CHECKER_SYSTEM
from .validation import validate_reader, validate_checker


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def atomic_json(path, obj):
    path=Path(path); tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

class UnitResult(BaseModel):
    unit_id: str
    reader: ReaderOutput
    checker: CheckerOutput
    reader_log: str
    checker_log: str
    accepted_fact_ids: list[str]
    candidate_fact_ids: list[str]
    new_question_ids: list[str]
    cached_reader: bool

class Stage1Pipeline:
    def __init__(self, source, config, llm, run_dir):
        self.source=source; self.config=config; self.llm=llm
        self.run_dir=Path(run_dir); self.run_dir.mkdir(parents=True,exist_ok=True)
        self.log_dir=self.run_dir/'call_logs'; self.log_dir.mkdir(exist_ok=True)
        self.state_path=self.run_dir/'stage1_reading_state.json'; self.handoff_path=self.run_dir/'stage1_agent_handoff.json'
        self.units,self.skips=source.build_reading_plan(config)
        self.unit_index={u.unit_id:u for u in self.units}
        self.run_fingerprint=fingerprint({'version':'stage1-v2.1','source':source.sha256,'parsed':source.parsed,'catalog':source.catalog,'config':config.model_dump(mode='json'),'transport':llm.identity,'units':[u.model_dump() for u in self.units],'prompts':[READER_SYSTEM,CHECKER_SYSTEM],'schemas':[ReaderOutput.model_json_schema(),CheckerOutput.model_json_schema()]})
        if self.state_path.exists():
            saved=json.loads(self.state_path.read_text(encoding='utf-8'))
            if saved.get('run_fingerprint')!=self.run_fingerprint: raise ValueError('Checkpoint mismatch; use a NEW run_dir for changes in source/config/schema/prompts/provider.')
            self.state=ReadingCheckpoint.model_validate(saved)
        else:
            self.state=ReadingCheckpoint(run_fingerprint=self.run_fingerprint,source_sha256=source.sha256,reading_plan=self.units,skipped_sections=self.skips,knowledge=KnowledgeStore(paper_id=source.paper_id))
        self.save()

    def save(self):
        atomic_json(self.state_path,self.state.model_dump(mode='json'))
        self.export_handoff()

    def export_handoff(self):
        s=self.state; k=s.knowledge.model_dump(mode='json')
        result={'schema_version':'stage1-agent-handoff-v2.1','source_sha256':self.source.sha256,'paper_title':self.source.parsed.get('metadata',{}).get('title'),'bibliographic_metadata':self.source.metadata,'source_paths':{'markdown':str(self.source.markdown.resolve()),'parsed_paper':str(self.source.parsed_path.resolve()),'section_catalog':str(self.source.catalog_path.resolve())},'reading_complete':s.reading_complete,'completed_unit_ids':s.completed_unit_ids,'total_units':len(self.units),'skipped_sections':self.skips,'sections':self.source.catalog,'reading_plan':[u.model_dump() for u in self.units],**k,'notes':'Checker-supported facts are provisional manuscript-grounded claims, not independently verified truth. Partial/unsupported candidates are investigation leads only. No cross-section consolidation or question answering performed.'}
        atomic_json(self.handoff_path,result)
        return result

    def next_unit(self): return next((u for u in self.units if u.unit_id not in self.state.reviews),None)

    def _unit(self, uid): return self.unit_index[uid]

    def _call(self, unit, stage, messages, schema, model, validator):
        # Log every attempt; the original raw completion is never overwritten.
        path=self.log_dir/f'{unit.unit_id}_{stage}_{time.time_ns()}_{uuid.uuid4().hex[:6]}.json'
        log={'unit_id':unit.unit_id,'stage':stage,'at':utc_now(),'run_fingerprint':self.run_fingerprint,'prompt_hash':fingerprint(messages),'messages':messages,'model':model.model_dump(mode='json'),'schema':schema.__name__,'raw':None,'parsed':None,'error':None,'status':'started'}
        atomic_json(path,log)
        start=time.monotonic()
        try:
            raw=self.llm.complete(messages,schema,model); log['raw']=raw
            # Persist raw data before any semantic/structural checks.
            atomic_json(path,log)
            result=parse_completion(raw,schema); log['parsed']=result.model_dump(mode='json')
            validator(result); log['status']='success'
            return result,str(path.relative_to(self.run_dir))
        except Exception as exc:
            log['error']=f'{type(exc).__name__}: {exc}'; log['status']='failed'
            raise
        finally:
            log['duration_seconds']=round(time.monotonic()-start,3)
            atomic_json(path,log)

    def _cached_reader(self, unit, messages):
        if not self.config.reuse_reader_cache: return None
        for path in sorted(self.log_dir.glob(f'{unit.unit_id}_reader_*.json'),reverse=True):
            d=json.loads(path.read_text(encoding='utf-8'))
            if d['run_fingerprint']==self.run_fingerprint and d['prompt_hash']==fingerprint(messages) and d['status']=='success':
                r=ReaderOutput.model_validate(d['parsed']); validate_reader(self.source,unit,r,self.config)
                return r,str(path.relative_to(self.run_dir))
        return None

    def read(self, unit_id):
        """Debuggable intermediate step. Does not commit scientific memory."""
        unit=self._unit(unit_id); messages=reader_messages(self.source,unit,self.config)
        cached=self._cached_reader(unit,messages)
        if cached: return cached[0],cached[1],True
        r,path=self._call(unit,'reader',messages,ReaderOutput,self.config.reader,lambda r:validate_reader(self.source,unit,r,self.config))
        return r,path,False

    def check(self, unit_id, reader):
        """Debuggable intermediate step. Does not commit scientific memory."""
        unit=self._unit(unit_id); validate_reader(self.source,unit,reader,self.config)
        return self._call(unit,'checker',checker_messages(self.source,unit,self.config,reader),CheckerOutput,self.config.checker,lambda c:validate_checker(reader,c))

    def commit(self, unit_id, reader, checker, reader_log, checker_log, cached_reader=False):
        unit=self._unit(unit_id)
        if self.next_unit() is None or self.next_unit().unit_id!=unit_id: raise ValueError('Commit must follow reading order; unit already committed or earlier unit pending')
        validate_reader(self.source,unit,reader,self.config)
        # Bind the committed objects to exact successful call logs, including prompt identity.
        for rel,stage,obj,messages in [(reader_log,'reader',reader,reader_messages(self.source,unit,self.config)),(checker_log,'checker',checker,checker_messages(self.source,unit,self.config,reader))]:
            p=(self.run_dir/rel).resolve()
            if p.parent!=self.log_dir.resolve(): raise ValueError('Log must be in this run call_logs directory')
            d=json.loads(p.read_text(encoding='utf-8'))
            if d['unit_id']!=unit_id or d['stage']!=stage or d['status']!='success' or d['run_fingerprint']!=self.run_fingerprint or d['prompt_hash']!=fingerprint(messages) or d['parsed']!=obj.model_dump(mode='json'):
                raise ValueError('Output/log mismatch: rerun the corresponding stage')
        nxt=commit_unit(self.state,unit,reader,checker,self.config,reader_log,checker_log)
        atomic_json(self.state_path,nxt.model_dump(mode='json')) # canonical commit
        self.state=nxt; self.export_handoff() # derived export regenerated on restart
        review=nxt.reviews[unit_id]
        return UnitResult(unit_id=unit_id,reader=reader,checker=checker,reader_log=reader_log,checker_log=checker_log,accepted_fact_ids=review.accepted_fact_ids,candidate_fact_ids=review.candidate_fact_ids,new_question_ids=review.new_question_ids,cached_reader=cached_reader)

    def run_next(self):
        unit=self.next_unit()
        if unit is None: return None
        reader,rl,cached=self.read(unit.unit_id)
        checker,cl=self.check(unit.unit_id,reader)
        return self.commit(unit.unit_id,reader,checker,rl,cl,cached)

    def run_remaining(self,max_units=None):
        count=0
        while self.next_unit() is not None and (max_units is None or count<max_units):
            result=self.run_next() # stop on error; never silently skip
            count+=1
            yield result

    def usage_summary(self):
        records=[json.loads(p.read_text(encoding='utf-8')) for p in self.log_dir.glob('*.json')]
        records=[r for r in records if r.get('run_fingerprint')==self.run_fingerprint]
        usage=[(r.get('raw') or {}).get('usage') or {} for r in records]
        costs=[u.get('cost') for u in usage if isinstance(u.get('cost'),(int,float))]
        return {'attempts':len(records),'responses':sum(r.get('raw') is not None for r in records),'successful_calls':sum(r['status']=='success' for r in records),'failed_or_interrupted':sum(r['status']!='success' for r in records),'input_tokens':sum(u.get('prompt_tokens') or 0 for u in usage),'output_tokens':sum(u.get('completion_tokens') or 0 for u in usage),'reasoning_tokens':sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for u in usage),'responses_with_reasoning_token_count':sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens') is not None for u in usage),'total_tokens':sum(u.get('total_tokens') or 0 for u in usage),'reported_cost_usd':sum(costs) if costs else None,'responses_with_reported_cost':len(costs)}
