"""Structured completion, durable audit logs, bounded repair, call budgets."""
import json,time,uuid
from pathlib import Path
from ..core.llm_client import parse_completion
from ..stage1.pipeline import atomic_json,fingerprint
from ..schemas.scientific import utc_now

class BudgetExceeded(RuntimeError): pass
class StructuredCallFailed(RuntimeError): pass

class AuditedLLM:
    def __init__(self,llm,config,root):
        self.llm=llm; self.config=config; self.root=Path(root)/'call_logs'; self.root.mkdir(exist_ok=True)
    def usage(self):
        logs=[json.loads(p.read_text(encoding='utf-8')) for p in self.root.glob('*.json')]
        us=[(l.get('raw') or {}).get('usage') or {} for l in logs]
        return {'calls':len(logs),'successful_calls':sum(l['status']=='success' for l in logs),'failed_calls':sum(l['status']!='success' for l in logs),'reported_tokens':sum(u.get('total_tokens') or 0 for u in us),'reasoning_tokens':sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for u in us),'reported_cost_usd':sum(u['cost'] for u in us if isinstance(u.get('cost'),(float,int))),'responses_with_usage':sum(bool(u) for u in us)}
    def available(self):
        u=self.usage(); return u['calls']<self.config.max_llm_calls and u['reported_tokens']<self.config.max_reported_tokens
    def call(self,key,system,payload,schema,model,validator=lambda x:None):
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
        h=fingerprint({'messages':messages,'schema':schema.model_json_schema(),'model':model.model_dump(mode='json')})
        prior=[]
        for p in sorted(self.root.glob(f'{key}_*.json')):
            d=json.loads(p.read_text(encoding='utf-8'))
            if d.get('request_hash')!=h: continue
            prior.append(d)
            if d['status']=='success':
                obj=schema.model_validate(d['parsed']); validator(obj); return obj,str(p.relative_to(self.root.parent))
        last_error=prior[-1].get('error') if prior else None
        # Failed attempts are persisted and count toward the repair limit on resume.
        for attempt in range(len(prior),1+self.config.max_output_repairs):
            if not self.available(): raise BudgetExceeded('LLM call/reported-token budget reached')
            msg=messages if not last_error else [*messages,{'role':'user','content':'Previous attempt invalid. Correct the following problem and return complete JSON: '+last_error[:1200]}]
            log={'key':key,'request_hash':h,'messages':msg,'schema':schema.__name__,'schema_json':schema.model_json_schema(),'model':model.model_dump(mode='json'),'at':utc_now(),'status':'started','raw':None,'parsed':None,'error':None}
            path=self.root/f'{key}_{time.time_ns()}_{uuid.uuid4().hex[:6]}.json'; atomic_json(path,log); start=time.monotonic()
            try:
                raw=self.llm.complete(msg,schema,model); log['raw']=raw; atomic_json(path,log)
                obj=parse_completion(raw,schema); log['parsed']=obj.model_dump(mode='json'); validator(obj); log['status']='success'
                return obj,str(path.relative_to(self.root.parent))
            except Exception as exc:
                last_error=f'{type(exc).__name__}: {exc}'; log['error']=last_error; log['status']='failed'
                # A length failure is not committed; bounded correction only (no silent increase of budgets).
            finally:
                log['duration_seconds']=round(time.monotonic()-start,3); atomic_json(path,log)
        raise StructuredCallFailed(f'{key}: {last_error or "repair budget exhausted"}')
