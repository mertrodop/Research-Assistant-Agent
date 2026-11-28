"""Stateless specialist configurations, one LLM generation per attempt."""
from .actions import ACTIONS
from .prompts import EXECUTOR_PROMPTS
from .validation import validate_result

ACTION_MAP={a:(spec.executor,spec.schema) for a,spec in ACTIONS.items()}

def empty_result(task,status,reason):
    schema=ACTION_MAP[task.action][1]
    d={'status':status,'task_assessment':'insufficient_context' if status=='insufficient_context' else 'other_problem','outcome_summary':reason,'evidence_refs':[],'remaining_uncertainties':[reason],'information_requests':[],'failure_reason':reason}
    if task.action=='MERGE_FACTS': d['merge_disposition']='insufficient_context'
    for key in schema.model_fields:
        if key not in d: d[key]=[]
    return schema.model_validate(d)

class ExecutorAgent:
    def __init__(self,audit,model,store,answerer=None): self.audit=audit; self.model=model; self.store=store; self.answerer=answerer or model
    def execute(self,task,context,attempt_id):
        if context.get('error'): return empty_result(task,'insufficient_context',context['error']),None
        _,schema=ACTION_MAP[task.action]
        return self.audit.call(f'{attempt_id}_executor',EXECUTOR_PROMPTS[task.action],context,schema,self.answerer if task.action=='ANSWER_QUESTION' else self.model,lambda r:validate_result(task,r,context,self.store.bundle,self.store.load().knowledge))
