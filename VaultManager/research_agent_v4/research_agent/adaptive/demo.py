"""Scripted demo: shows repair, scoped negative search, and dependent reasoning.
Not evidence of real model quality. No API access.
"""
import json
from pathlib import Path
from .schema import Plan, Result, Review
from ..stage1.global_pipeline import GlobalStage1Pipeline, GlobalStage1Config
from ..stage1.global_demo import GlobalDemoLLM, make_demo_source
from ..stage1.source import PaperSource

def demo_handoff(root):
    root=Path(root)
    pipeline=GlobalStage1Pipeline(PaperSource(*make_demo_source(root/'source')),GlobalStage1Config(),GlobalDemoLLM(),root/'stage1')
    r,log,_,cs,gs=pipeline.read(); pipeline.commit(r,log,pipeline.check_all(gs,cs)); return pipeline.handoff_path

def task(action,prompt,facts=(),questions=(),candidates=(),sections=()): return dict(action=action,prompt=prompt,facts=list(facts),questions=list(questions),candidates=list(candidates),sections=list(sections))
def prop(kind,text,facts=(),questions=(),candidates=(),spans=()): return dict(kind=kind,text=text,facts=list(facts),questions=list(questions),candidates=list(candidates),spans=[dict(a=a,b=b) for a,b in spans])

class DemoLLM:
    identity='adaptive-scripted-demo-v1'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        p=json.loads(messages[1]['content']); self.calls.append((schema.__name__,p))
        if schema is Plan:
            if p['round']==1:
                tasks=[task('INVESTIGATE','Find whether a test dataset is named.',questions=['Q0001'],sections=['S001']),task('QUESTION','Challenge generalization from the reported result.',facts=['F00002'])]
            elif p['round']==2:
                q=next(i for i in p['questions'] if i!='Q0001')
                tasks=[task('INVESTIGATE','Inspect the remaining limitations section for dataset identification.',questions=['Q0001'],sections=['S002']),task('ANSWER','Determine whether the result establishes external validation.',facts=['F00002','F00003'],questions=[q])]
            elif p['round']==3:
                tasks=[task('AUDIT','Explain the scope of reported accuracy.',facts=['F00002','F00003'])]
            else: tasks=[]
            d={'tasks':tasks,'reason':'Investigate missing dataset details and scope of results.' if tasks else 'No further productive investigation in the fixture.'}
        elif schema is Result:
            a=p['action']; q=list(p['questions']); f=list(p['facts'])
            if a=='INVESTIGATE': proposals=[prop('not_found','No test dataset name located in this selected section.',questions=q)]
            elif a=='QUESTION': proposals=[prop('question','Does the reported test accuracy establish external validation?',facts=f)]
            elif a=='ANSWER': proposals=[prop('answer','No. External datasets were not evaluated, so the accuracy cannot establish external validation.',facts=f,questions=q),prop('derive','The reported accuracy cannot establish performance on external datasets.',facts=f)]
            elif a=='AUDIT': proposals=[prop('boundary','A single reported test accuracy does not establish generalization to external datasets.',facts=f)]
            else: proposals=[]
            d={'proposals':proposals,'note':'Scripted demonstration.'}
        elif schema is Review: d={'checks':[{'i':i,'ok':True,'reason':''} for i,_ in enumerate(p['proposals'])]}
        else: raise ValueError('Unexpected demo schema')
        return {'model':'scripted-demo','provider':'offline','choices':[{'finish_reason':'stop','message':{'content':json.dumps(d)}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'completion_tokens_details':{'reasoning_tokens':0},'cost':0}}
