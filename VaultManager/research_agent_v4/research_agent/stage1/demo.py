"""Tiny synthetic fixture for debugging without a real paper or API key."""
from pathlib import Path
import json, hashlib
from ..schemas.extraction import ReaderOutput


def make_demo_source(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    text='# Example Study\nThe model uses two layers.\nThe test accuracy is 80%.\n\n## Limitations\nThe authors did not evaluate external datasets.\n'
    md=root/'demo.md'; md.write_text(text,encoding='utf-8')
    sections=[dict(section_id='S001',title='Example Study',kind='front_matter',heading_path=['Example Study'],parent_id=None,line_start=1,local_end=4,subtree_end=6,chunk_ids=[]),dict(section_id='S002',title='Limitations',kind='section',heading_path=['Example Study','Limitations'],parent_id='S001',line_start=5,local_end=6,subtree_end=6,chunk_ids=[])]
    blocks=[dict(section_id='S001',kind='heading',line_start=1,line_end=1),dict(section_id='S001',kind='paragraph',line_start=2,line_end=3),dict(section_id='S002',kind='heading',line_start=5,line_end=5),dict(section_id='S002',kind='paragraph',line_start=6,line_end=6)]
    parsed={'source':{'sha256':hashlib.sha256(md.read_bytes()).hexdigest(),'line_count':6},'metadata':{'title':'Example Study','title_source_line':1},'sections':sections,'blocks':blocks}
    p=root/'ParsedPaper.json'; c=root/'SectionCatalog.json'
    p.write_text(json.dumps(parsed),encoding='utf-8'); c.write_text(json.dumps(sections),encoding='utf-8')
    return md,p,c

class DemoLLM:
    """Scripted completions; deliberately includes a partial and invalid question."""
    identity='offline-demo-v2'
    def __init__(self): self.calls=[]
    def complete(self,messages,schema,config):
        self.calls.append(schema.__name__)
        first='CURRENT UNIT: S001 |' in messages[-1]['content']
        if schema is ReaderOutput:
            data={'facts':[{'text':'The model uses two layers.','line_start':2,'line_end':2},{'text':'The test accuracy is 80% on external datasets.','line_start':3,'line_end':3}]} if first else {'facts':[{'text':'The authors did not evaluate external datasets.','line_start':6,'line_end':6}]}
        elif first:
            data={'fact_checks':[{'fact_index':0,'verdict':'supported','reason':''},{'fact_index':1,'verdict':'partial','reason':'80% is stated, external datasets are not.'}], 'new_questions':[{'question':'Which dataset was used for testing?','motivated_by_fact_indices':[1]},{'question':'Invalid reference example','motivated_by_fact_indices':[99]}]}
        else:
            data={'fact_checks':[{'fact_index':0,'verdict':'supported','reason':''}],'new_questions':[]}
        return {'id':f'demo-{len(self.calls)}','model':'scripted-demo','choices':[{'finish_reason':'stop','message':{'content':json.dumps(data),'refusal':None}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}


def make_compact_demo_source(root):
    """Separate fixture: add a repeated claim to demonstrate cross-section dedup."""
    md,p,c=make_demo_source(root)
    with md.open('a',encoding='utf-8') as f: f.write('The model uses two layers.\n')
    parsed=json.loads(p.read_text()); sections=parsed['sections']
    sections[0]['subtree_end']=7; sections[1]['local_end']=7; sections[1]['subtree_end']=7
    parsed['blocks'].append(dict(section_id='S002',kind='paragraph',line_start=7,line_end=7))
    parsed['source']={'sha256':hashlib.sha256(md.read_bytes()).hexdigest(),'line_count':7}
    p.write_text(json.dumps(parsed),encoding='utf-8'); c.write_text(json.dumps(sections),encoding='utf-8')
    return md,p,c

class CompactDemoLLM(DemoLLM):
    identity='offline-compact-demo-v1'
    def complete(self,messages,schema,config):
        from .deduplication import DeduplicationResult
        from ..schemas.extraction import CheckerOutput
        if schema is DeduplicationResult:
            self.calls.append(schema.__name__)
            data={'duplicate_groups':[{'canonical_fact_id':'F00001','redundant_fact_ids':['F00003'],'relationship':'equivalent','reason':'Both specify two layers for the model.'}]}
        elif 'CURRENT UNIT: S001 |' in messages[-1]['content']:
            return super().complete(messages,schema,config)
        else:
            self.calls.append(schema.__name__)
            if schema is ReaderOutput:
                data={'facts':[{'text':'No external dataset evaluation.','line_start':6,'line_end':6},{'text':'Model: two layers.','line_start':7,'line_end':7}]}
            elif schema is CheckerOutput:
                data={'fact_checks':[{'fact_index':i,'verdict':'supported','reason':''} for i in range(2)],'new_questions':[{'question':'What types of layers are used?','motivated_by_fact_indices':[1]}]}
            else: raise ValueError('Unexpected demo schema')
        return {'id':f'compact-demo-{len(self.calls)}','model':'scripted-demo','choices':[{'finish_reason':'stop','message':{'content':json.dumps(data),'refusal':None}}],'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}
