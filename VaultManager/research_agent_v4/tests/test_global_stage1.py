import json,tempfile,unittest
from pathlib import Path
from research_agent.stage1.global_pipeline import (GlobalStage1Pipeline,GlobalStage1Config,GroupFailures,resolve_candidates,build_groups,validate_group)
from research_agent.stage1.global_demo import GlobalDemoLLM,make_demo_source
from research_agent.stage1.source import PaperSource
from research_agent.schemas.global_extraction import GlobalReaderOutput
from research_agent.schemas.extraction import CheckerOutput
from research_agent.stage2.store import PaperBundle
from research_agent.stage2.graph import Stage2Pipeline
from research_agent.stage2.config import Stage2Config
from research_agent.stage2.demo import Stage2DemoLLM,task

class GlobalStage1Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        self.source=PaperSource(*make_demo_source(self.root/'source'))
        self.llm=GlobalDemoLLM(); self.p=GlobalStage1Pipeline(self.source,GlobalStage1Config(),self.llm,self.root/'run')
    def run_pipeline(self):
        reader,log,cached,candidates,groups=self.p.read()
        results=self.p.check_all(groups,candidates,max_workers=3)
        return self.p.commit(reader,log,results)
    def test_group_partition_multispan_and_handoff(self):
        h=self.run_pipeline(); self.assertEqual(len(self.llm.calls),4)
        self.assertEqual(len(h['facts']),3); self.assertEqual(len(h['candidate_facts']),4); self.assertEqual(len(h['questions']),1)
        self.assertEqual([e['section_id'] for e in h['facts']['F00001']['evidence']],['S001','S002'])
        self.assertEqual(h['candidate_facts']['C00002']['verdict'],'partial')
        q=h['questions']['Q0001']; self.assertEqual(q['motivating_candidate_ids'],['C00002','C00003']); self.assertEqual(q['motivating_fact_ids'],['F00002'])
        self.assertTrue(h['reading_complete']); self.assertEqual(len(h['global_initialization']['groups']),3)
        b=PaperBundle(self.p.handoff_path); self.assertEqual(len(b.initial.facts),3)
        p=Stage2Pipeline(self.p.handoff_path,Stage2Config(),Stage2DemoLLM(),self.root/'stage2',transport='inprocess'); self.addCleanup(p.close)
        ctx=p.planner.context.build(p.initial_state()); self.assertIn('F00001',next(s for s in ctx['sections'] if s['section_id']=='S002')['fact_ids'])
        retrieved=p.retrieval.build_context(task('T001','INVESTIGATE_SOURCE','Read',fact_ids=['F00001'],section_ids=['S001','S002']),60000)
        self.assertEqual(len(retrieved['passages']),2); self.assertEqual(len(retrieved['facts'][0]['evidence']),2)
    def test_resume_no_new_calls_or_changed_handoff(self):
        self.run_pipeline(); before=self.p.handoff_path.read_bytes()
        self.p=GlobalStage1Pipeline(self.source,GlobalStage1Config(),self.llm,self.root/'run')
        self.run_pipeline(); self.assertEqual(len(self.llm.calls),4); self.assertEqual(before,self.p.handoff_path.read_bytes())
        self.assertEqual(self.p.usage_summary()['total_tokens'],120); self.assertTrue(all(r['committed'] for r in self.p.call_diagnostics()))
    def test_failed_group_reuses_successes(self):
        class Flaky(GlobalDemoLLM):
            def __init__(self): super().__init__(); self.failed=False
            def complete(self,messages,schema,config):
                if schema is CheckerOutput and json.loads(messages[1]['content'])['group_id']=='G002' and not self.failed:
                    self.failed=True
                    return {'choices':[{'finish_reason':'length','message':{'content':' ' * 30}}],'usage':{'prompt_tokens':10,'completion_tokens':30,'total_tokens':40}}
                return super().complete(messages,schema,config)
        llm=Flaky(); p=GlobalStage1Pipeline(self.source,GlobalStage1Config(),llm,self.root/'flaky')
        r,log,_,cs,gs=p.read()
        with self.assertRaises(GroupFailures): p.check_all(gs,cs,max_workers=3)
        self.assertFalse(p.handoff_path.exists()); self.assertEqual(p.usage_summary()['attempts'],4)
        rs=p.check_all(gs,cs,max_workers=3); self.assertEqual(sum(x.cached for x in rs),2)
        p.commit(r,log,rs); self.assertEqual(p.usage_summary()['attempts'],5)
        self.assertEqual(sum(x['committed'] for x in p.call_diagnostics()),4)
    def test_cross_section_span_and_bounds(self):
        r=GlobalReaderOutput.model_validate({'facts':[{'text':'Both statements','evidence':[{'line_start':3,'line_end':6}]}]})
        cs=resolve_candidates(self.source,r); self.assertEqual([(e.section_id,e.line_start,e.line_end) for e in cs[0].evidence],[('S001',3,4),('S002',5,6)])
        r.facts[0].evidence[0].line_end=99
        with self.assertRaisesRegex(ValueError,'out-of-source'): resolve_candidates(self.source,r)
    def test_checker_indices_and_commit_binding(self):
        r,log,_,cs,gs=self.p.read(); rs=self.p.check_all(gs,cs)
        bad=rs[0].checker.model_copy(deep=True); bad.fact_checks=[]
        with self.assertRaises(ValueError): validate_group(gs[0],bad,4)
        with self.assertRaises(ValueError): self.p.commit(r,log,rs[:-1])
        rs[0].checker.fact_checks[0].reason='changed'
        with self.assertRaisesRegex(ValueError,'Output/log mismatch'): self.p.commit(r,log,rs)
    def test_new_config_requires_new_directory(self):
        with self.assertRaisesRegex(ValueError,'Checkpoint mismatch'): GlobalStage1Pipeline(self.source,GlobalStage1Config(max_questions_per_group=2),self.llm,self.root/'run')
    def test_bounded_groups(self):
        r,_,_,cs,_=self.p.read(); gs=build_groups(cs,1); self.assertEqual(len(gs),4)
        self.assertEqual(sorted(i for g in gs for i in g.candidate_indices),list(range(4)))
