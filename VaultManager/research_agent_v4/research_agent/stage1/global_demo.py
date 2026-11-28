"""Synthetic multi-section fixture, including a rejected candidate and duplicate question."""
import json
from .demo import make_demo_source
from ..schemas.global_extraction import GlobalReaderOutput
from ..schemas.extraction import CheckerOutput

class GlobalDemoLLM:
    identity='global-stage1-scripted-v1'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        self.calls.append(schema.__name__)
        if schema is GlobalReaderOutput:
            data={'facts':[
                {'text':'The model uses two layers and no external datasets were evaluated.','evidence':[{'line_start':2,'line_end':2},{'line_start':6,'line_end':6}]},
                {'text':'The test accuracy is 80% on external datasets.','evidence':[{'line_start':3,'line_end':3}]},
                {'text':'The reported test accuracy is 80%.','evidence':[{'line_start':3,'line_end':3}]},
                {'text':'The authors did not evaluate external datasets.','evidence':[{'line_start':6,'line_end':6}]},
            ]}
        elif schema is CheckerOutput:
            payload=json.loads(messages[1]['content']); checks=[]; questions=[]
            for f in payload['candidate_facts']:
                i=f['fact_index']; bad='80% on external datasets' in f['text']
                checks.append({'fact_index':i,'verdict':'partial' if bad else 'supported','reason':'80% is reported, but the cited line does not identify external data.' if bad else ''})
                if bad or 'reported test accuracy' in f['text']:
                    questions.append({'question':'Which dataset was used for testing?','motivated_by_fact_indices':[i]})
            data={'fact_checks':checks,'new_questions':questions}
        else: raise ValueError('Unexpected demo schema')
        return {'id':f'global-demo-{len(self.calls)}','model':'scripted-demo','provider':'offline','choices':[{'finish_reason':'stop','message':{'content':json.dumps(data),'refusal':None}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'completion_tokens_details':{'reasoning_tokens':0},'cost':0}}
