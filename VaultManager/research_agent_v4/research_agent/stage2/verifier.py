"""One verification component, with source and reasoning modes. No model writes memory."""
from ..schemas.execution import VerificationRecord, VerificationResult
from ..stage1.pipeline import fingerprint
from ..tools.retrieval import dump
from .validation import proposals_of, subset, nonblank, validate_verification_record
from .prompts import GROUNDED_VERIFIER, REASONING_VERIFIER
from .llm import BudgetExceeded, StructuredCallFailed


def verification_payload(task,context,result):
    # Deliberately exclude task prompt, rationale, strategy, executor outcome and hypotheses.
    # Include only the evidence, original question texts and numbered claims to be assessed.
    return {'action':task.action,
            'facts':context['facts'],
            'questions':[{'question_id':q['question_id'],'question':q['question']} for q in context['questions']],
            'passages':context['passages'],
            'numbered_proposals':[{'index':i,'kind':kind,'proposal':dump(p)} for i,(kind,p) in enumerate(proposals_of(result))]}


def verification_input_hash(task,context,result):
    # Bind to exact task, snapshot, result and proposals (including source contents).
    return fingerprint({'task':dump(task),'context':context,'result':dump(result)})


class VerifierAgent:
    def __init__(self,audit,model): self.audit=audit; self.model=model
    def verify(self,task,context,result,attempt_id):
        props=proposals_of(result); h=verification_input_hash(task,context,result)
        if not props:
            return VerificationRecord(status='skipped',reason='No scientific proposals; scoped observation reviewed by planner.',input_hash=h)
        def validate(v):
            subset([c.proposal_index for c in v.checks],range(len(props)),'verification proposal indices',len(props))
            for c in v.checks: nonblank(c.reason,'verification reason')
        system=REASONING_VERIFIER if task.action=='ANSWER_QUESTION' else GROUNDED_VERIFIER
        try:
            v,log=self.audit.call(f'{attempt_id}_verifier',system,verification_payload(task,context,result),VerificationResult,self.model,validate)
            record=VerificationRecord(status='completed',checks=v.checks,log=log,input_hash=h)
        except (BudgetExceeded,StructuredCallFailed) as exc:
            record=VerificationRecord(status='unavailable',reason=str(exc),input_hash=h)
        validate_verification_record(record,props)
        return record
