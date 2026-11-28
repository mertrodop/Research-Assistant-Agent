"""Scripted demo: real contracts, no API calls; not an assessment of model quality."""
import json
from pathlib import Path
from ..stage1.demo import DemoLLM,make_demo_source
from ..stage1.pipeline import Stage1Pipeline
from ..stage1.source import PaperSource
from ..core.config import Stage1Config
from ..schemas.planning import PlannerDecision,RoundEvaluation,Synthesis,Task
from ..schemas.execution import QuestionResult,InvestigationResult,CoverageResult,VerificationResult

def make_stage2_demo_handoff(root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    stage1=Stage1Pipeline(PaperSource(*make_demo_source(root/'source')),Stage1Config(),DemoLLM(),root/'stage1')
    list(stage1.run_remaining()); return stage1.handoff_path

def task(tid,action,prompt,**kwargs):
    return Task(task_id=tid,action=action,task_prompt=prompt,rationale=kwargs.pop('rationale','Improve manuscript-grounded knowledge.'),success_criteria=kwargs.pop('success_criteria',['Assess evidence and report the outcome.']),fact_ids=kwargs.pop('fact_ids',[]),question_ids=kwargs.pop('question_ids',[]),section_ids=kwargs.pop('section_ids',[]),**kwargs).model_dump(mode='json')
def ref(sid,a,z): return {'section_id':sid,'unit_id':sid,'line_start':a,'line_end':z}
def outcome(status,summary,**kwargs):
    return {'status':status,'task_assessment':kwargs.pop('task_assessment','appropriate'),'outcome_summary':summary,'evidence_refs':kwargs.pop('evidence_refs',[]),'remaining_uncertainties':kwargs.pop('remaining_uncertainties',[]),'information_requests':kwargs.pop('information_requests',[]),'failure_reason':kwargs.pop('failure_reason',None),**kwargs}

def decision(goal,tasks,phase,terminate=False):
    return {'strategy':{'overall_objective':goal,'steps':[{'objective':'Resolve dataset identity','rationale':'Important evaluation detail','status':'abandoned' if phase>=3 else 'in_progress'},{'objective':'Check coverage of test results and limitations','rationale':'Recover omitted claims','status':'completed' if phase>=3 else 'in_progress'}]},'current_round_tasks':tasks,'terminate':terminate,'termination_reason':'The synthetic paper does not name the dataset; reported results and limitations have been assessed.' if terminate else None}

class Stage2DemoLLM:
    identity='stage2-rounds-scripted-v3'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        self.calls.append(schema.__name__); payload=json.loads(messages[1]['content'])
        if schema is PlannerDecision:
            n=payload['next_round']
            if n==1: data=decision(payload['goal'],[task('T001','ANSWER_QUESTION','Try to identify the test dataset from existing facts without assuming its identity.',fact_ids=['F00001'],question_ids=['Q0001']),task('T002','CHECK_COVERAGE','Check test results coverage and recover simple missing facts.',fact_ids=['F00001'],section_ids=['S001'])],n)
            elif n==2: data=decision(payload['goal'],[task('T003','INVESTIGATE_SOURCE','Inspect both original sections and the newly committed accuracy fact. Determine whether dataset identity is stated.',fact_ids=['F00003'],question_ids=['Q0001'],section_ids=['S001','S002'])],n)
            else: data=decision(payload['goal'],[],n,True)
        elif schema is QuestionResult:
            data=outcome('insufficient_context','The available fact does not identify the dataset.',task_assessment='insufficient_context',remaining_uncertainties=['Dataset identity'],information_requests=[{'reason':'Inspect original sections','fact_ids':[],'question_ids':['Q0001'],'section_ids':['S001','S002']}],failure_reason=None,answers=[{'answer':'The selected fact does not identify the test dataset.','resolutions':[{'question_id':'Q0001','status':'unresolved','explanation':'The model architecture does not establish the test dataset.','answer_fact_ids':[],'evidence':[]}]}])
        elif schema is CoverageResult:
            e=ref('S001',3,3)
            data=outcome('success','Accuracy was omitted; the original section states 80%.',evidence_refs=[e],gaps=[],new_facts=[{'text':'The reported test accuracy is 80%.','evidence':[e],'derived_from_fact_ids':[]}])
        elif schema is InvestigationResult:
            e1,e2=ref('S001',3,3),ref('S002',6,6)
            data=outcome('success','Both synthetic sections were assessed; neither names the test dataset. The accuracy fact is already known.',evidence_refs=[e1,e2],remaining_uncertainties=['Dataset identity remains unknown.'],new_facts=[],new_questions=[])
        elif schema is VerificationResult:
            data={'checks':[{'proposal_index':p['index'],'verdict':'supported','reason':'Scripted fixture: proposal matches supplied evidence or accurately reports a limitation.'} for p in payload['numbered_proposals']]}
        elif schema is RoundEvaluation:
            data={'task_evaluations':[{'task_id':t['task']['task_id'],'accepted_proposal_indices':t['eligible_proposal_indices'] if t['status']=='success' else [],'rationale':'Supported by the supplied source or explicit limitation.','task_satisfied':t['status']=='success' and len(t['eligible_proposal_indices'])==len(t['numbered_proposals'])} for t in payload['tasks']],'round_observation':'Review the source scope and the accepted updates before planning again.'}
        elif schema is Synthesis:
            data={'sections':[{'title':'Model and results','claims':[{'text':f['text'],'fact_ids':[f['fact_id']]} for f in payload['facts']]}]}
        else: raise ValueError('Unexpected demo schema '+schema.__name__)
        return {'id':f'round-demo-{len(self.calls)}','model':'scripted-demo','choices':[{'finish_reason':'stop','message':{'content':json.dumps(data),'refusal':None}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'completion_tokens_details':{'reasoning_tokens':0},'cost':0}}
