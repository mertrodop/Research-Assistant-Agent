"""Offline fixture: scripted decisions, not a scientific benchmark."""
import json
from pathlib import Path
from .schema import Plan, Review
from ..stage1.global_pipeline import GlobalStage1Pipeline, GlobalStage1Config
from ..stage1.global_demo import GlobalDemoLLM, make_demo_source
from ..stage1.source import PaperSource

def demo_handoff(root):
    root=Path(root)
    pipeline=GlobalStage1Pipeline(PaperSource(*make_demo_source(root/'source')),GlobalStage1Config(),GlobalDemoLLM(),root/'stage1')
    r,log,_,cs,gs=pipeline.read(); pipeline.commit(r,log,pipeline.check_all(gs,cs)); return pipeline.handoff_path

def task(action,prompt,refs=(),sections=()):
    return dict(action=action,prompt=prompt,refs=list(refs),sections=list(sections))
def finding(kind,text,refs=(),lines=()):
    return dict(kind=kind,text=text,refs=list(refs),lines=[dict(a=a,b=b) for a,b in lines])
def resolution(q,status,text,refs=(),lines=()):
    return dict(q=q,status=status,text=text,refs=list(refs),lines=[dict(a=a,b=b) for a,b in lines])

class DemoLLM:
    identity='adaptive-v4-scripted-demo'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        p=json.loads(messages[1]['content']); self.calls.append((schema.__name__,p))
        if schema is Plan:
            defer=[]
            if p['round']==1:
                tasks=[task('INVESTIGATE','Challenge the scope of the reported result.',refs=['F00002']),task('INVESTIGATE','Does reported accuracy establish external validation?',refs=['F00002','F00003'])]
            elif p['round']==2:
                tasks=[task('CONSOLIDATE','Combine duplicate dataset questions.',refs=['Q0001','Q0002']),task('INVESTIGATE','Compute the percentage of incorrect predictions.',refs=['F00002'])]
            elif p['round']==3:
                tasks=[task('INVESTIGATE','Look for the named test dataset.',refs=['Q0003'],sections=['S001']),task('INVESTIGATE','Inspect remaining limitations for dataset name.',refs=['Q0003'],sections=['S002'])]
            else: tasks=[]
            d={'tasks':tasks,'defer':defer,'reason':'Investigate missing details and scope.' if tasks else 'No further productive investigation in fixture.'}
        elif schema.__name__=='ScopedResult':
            a=p['action']; q=list(p['questions']); f=list(p['facts']); findings=[]; resolutions=[]
            if a=='CONSOLIDATE': findings=[finding('merge_questions','Which dataset was used for testing?',refs=q)]
            elif p['source'] and q: resolutions=[resolution(q[0],'not_found','No test dataset name located in the selected section.')]
            elif p['source']: findings=[finding('not_found','Inference latency was not located in the selected section.')]
            elif 'Challenge' in p['prompt']: findings=[finding('question','What is the identity of the evaluation dataset?',refs=f)]
            elif 'incorrect predictions' in p['prompt']: findings=[finding('derive','An accuracy of 80% corresponds to an error rate of 20%.',refs=f)]
            elif a=='INVESTIGATE': findings=[finding('boundary','The reported accuracy alone does not establish external generalization.',refs=f)]
            d={'findings':findings,'resolutions':resolutions}
        elif schema is Review: d={'checks':[{'i':int(i),'ok':True,'reason':'Scripted fixture acceptance.'} for i in p['proposals']]}
        else: raise ValueError('Unexpected demo schema')
        return {'model':'scripted-demo','provider':'offline','choices':[{'finish_reason':'stop','message':{'content':json.dumps(d)}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'completion_tokens_details':{'reasoning_tokens':0},'cost':0}}
