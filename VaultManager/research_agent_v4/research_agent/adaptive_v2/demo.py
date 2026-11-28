"""Offline fixture exercises decisions and contracts; does not test scientific LLM ability."""
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


def task(action,prompt,facts=(),questions=(),candidates=(),sections=(),ask='',calc=()):
    return dict(action=action,prompt=prompt,facts=list(facts),questions=list(questions),candidates=list(candidates),sections=list(sections),ask=ask,calc=list(calc))
def finding(kind,text,refs=(),lines=()):
    return dict(kind=kind,text=text,refs=list(refs),lines=[dict(a=a,b=b) for a,b in lines])

class DemoLLM:
    identity='adaptive-v2-scripted-demo'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        p=json.loads(messages[1]['content']); self.calls.append((schema.__name__,p))
        if schema is Plan:
            triage=[]
            if p['round']==1:
                tasks=[task('INVESTIGATE','Look for the named test dataset.',questions=['Q0001'],sections=['S001']),task('QUESTION','Challenge the scope of the reported result.',facts=['F00002'])]
            elif p['round']==2:
                tasks=[task('INVESTIGATE','Inspect remaining limitations for dataset name.',questions=['Q0001'],sections=['S002']),task('REASON','Does the result establish external validation?',facts=['F00002','F00003'],ask='Does the reported accuracy establish external validation?')]
            elif p['round']==3:
                tasks=[task('REASON','Compute the percentage of incorrect predictions.',facts=['F00002'],calc=['100-80'])]
                q=next((i for i in p['questions'] if i!='Q0001' and 'latency' in p['questions'][i][0]),None)
                if q: triage=[dict(q=q,decision='defer',related='',reason='Peripheral to the reported accuracy.')]
            else: tasks=[]
            d={'tasks':tasks,'triage':triage,'reason':'Investigate missing details and scope.' if tasks else 'No further productive investigation in the fixture.'}
        elif schema.__name__=='ScopedResult':
            a=p['action']; q=list(p['questions']); f=list(p['facts'])
            if a=='INVESTIGATE': proposals=[finding('not_found','No test dataset name located in the selected section.',refs=q)]
            elif a=='QUESTION': proposals=[finding('question','Is inference latency reported?',refs=f)]
            elif p.get('calculations'): proposals=[finding('derive','An accuracy of 80% corresponds to an error rate of 20%.',refs=f)]
            elif a=='REASON': proposals=[finding('answer','No. External datasets were not evaluated.',refs=f+q),finding('boundary','The reported accuracy alone does not establish external generalization.',refs=f)]
            else: proposals=[]
            d={'findings':proposals,'note':'Scripted demonstration.'}
        elif schema is Review: d={'checks':[{'i':int(i),'ok':True,'reason':'Scripted fixture acceptance.'} for i in p['proposals']]}
        else: raise ValueError('Unexpected demo schema')
        return {'model':'scripted-demo','provider':'offline','choices':[{'finish_reason':'stop','message':{'content':json.dumps(d)}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'completion_tokens_details':{'reasoning_tokens':0},'cost':0}}
