import json, tempfile, unittest
from pathlib import Path
from research_agent.stage1.demo import make_demo_source, DemoLLM
from research_agent.stage1.source import PaperSource
from research_agent.stage1.pipeline import Stage1Pipeline
from research_agent.core.config import Stage1Config
from research_agent.schemas.extraction import ReaderOutput, CheckerOutput

class Stage1Test(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=PaperSource(*make_demo_source(self.root/'source'))
        self.llm=DemoLLM(); self.cfg=Stage1Config(); self.run=self.root/'run'
        self.p=Stage1Pipeline(self.source,self.cfg,self.llm,self.run)

    def test_full_pipeline_and_resume(self):
        result=self.p.run_next(); k=self.p.state.knowledge
        self.assertEqual(result.accepted_fact_ids,['F00001']); self.assertEqual(len(k.candidate_facts),2)
        self.assertEqual(k.candidate_facts['C00002'].verdict,'partial')
        self.assertEqual(k.questions['Q0001'].motivating_candidate_ids,['C00002'])
        self.assertEqual(k.questions['Q0001'].motivating_fact_ids,[])
        self.assertEqual(len(self.p.state.reviews['S001'].rejected_questions),1)
        self.assertNotIn('unit_summary',result.checker.model_dump())
        self.assertEqual(len(self.llm.calls),2)
        p2=Stage1Pipeline(self.source,self.cfg,self.llm,self.run)
        list(p2.run_remaining()); self.assertTrue(p2.state.reading_complete); self.assertEqual(len(self.llm.calls),4)
        self.assertEqual(list(p2.run_remaining()),[])
        handoff=json.loads(p2.handoff_path.read_text()); self.assertTrue(handoff['reading_complete'])
        self.assertEqual(len(handoff['facts']),2); self.assertEqual(p2.usage_summary()['total_tokens'],120)

    def test_checker_failure_reuses_reader(self):
        original=self.llm.complete
        def broken(messages,schema,config):
            raw=original(messages,schema,config)
            if schema is CheckerOutput: raw['choices'][0]['finish_reason']='length'
            return raw
        self.llm.complete=broken
        with self.assertRaises(ValueError): self.p.run_next()
        self.assertFalse(self.p.state.reviews); self.assertFalse(self.p.state.knowledge.facts)
        self.llm.complete=original
        p2=Stage1Pipeline(self.source,self.cfg,self.llm,self.run)
        r=p2.run_next(); self.assertTrue(r.cached_reader)
        self.assertEqual(self.llm.calls,['ReaderOutput','CheckerOutput','CheckerOutput'])
        self.assertEqual(p2.usage_summary()['attempts'],3)

    def test_legacy_summary_field_rejected(self):
        original=self.llm.complete
        def wrong(messages,schema,config):
            raw=original(messages,schema,config)
            if schema is CheckerOutput:
                d=json.loads(raw['choices'][0]['message']['content']); d['unit_summary']={'summary':'legacy','supporting_fact_indices':[0]}; raw['choices'][0]['message']['content']=json.dumps(d)
            return raw
        self.llm.complete=wrong
        with self.assertRaises(ValueError): self.p.run_next()
        self.assertFalse(self.p.state.reviews)
        self.assertFalse(self.p.state.knowledge.facts)

    def test_out_of_span_reader_fails(self):
        original=self.llm.complete
        def wrong(messages,schema,config):
            raw=original(messages,schema,config)
            if schema is ReaderOutput:
                d=json.loads(raw['choices'][0]['message']['content']); d['facts'][0]['line_end']=99; raw['choices'][0]['message']['content']=json.dumps(d)
            return raw
        self.llm.complete=wrong
        with self.assertRaises(ValueError): self.p.run_next()
        self.assertEqual(len(self.llm.calls),1); self.assertFalse(self.p.state.reviews)

    def test_bad_commit_and_config_mismatch(self):
        r,rl,cached=self.p.read('S001'); c,cl=self.p.check('S001',r)
        r.facts[0].text='modified fact'
        with self.assertRaises(ValueError): self.p.commit('S001',r,c,rl,cl,cached)
        with self.assertRaises(ValueError): Stage1Pipeline(self.source,Stage1Config(max_facts_per_unit=40),self.llm,self.run)

    def test_remaining_yields_incrementally(self):
        results=self.p.run_remaining()
        self.assertEqual(next(results).unit_id,'S001')
        self.assertEqual(self.p.state.completed_unit_ids,['S001'])
        self.assertEqual(next(results).unit_id,'S002')
        with self.assertRaises(StopIteration): next(results)

    def test_source_hash(self):
        self.source.markdown.write_text('changed')
        with self.assertRaises(ValueError): PaperSource(self.source.markdown,self.source.parsed_path,self.source.catalog_path)

if __name__=='__main__': unittest.main()
