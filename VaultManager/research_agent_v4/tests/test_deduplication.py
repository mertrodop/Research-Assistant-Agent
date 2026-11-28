import tempfile, unittest, json
from pathlib import Path
from research_agent.stage1.demo import CompactDemoLLM, make_compact_demo_source
from research_agent.stage1.source import PaperSource
from research_agent.stage1.pipeline import Stage1Pipeline
from research_agent.core.config import Stage1Config, ModelConfig
from research_agent.stage1.deduplication import DeduplicationPipeline, DeduplicationResult, DuplicateGroup, apply_groups
from research_agent.stage2.store import PaperBundle, Stage2Store
from research_agent.stage2.planner import PlannerContext
from research_agent.stage2.config import Stage2Config
from types import SimpleNamespace

class DedupTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        self.llm=CompactDemoLLM(); self.source=PaperSource(*make_compact_demo_source(self.root/'source'))
        self.p=Stage1Pipeline(self.source,Stage1Config(),self.llm,self.root/'stage1')
        list(self.p.run_remaining())
        self.model=ModelConfig(model='demo',max_tokens=1000)
        self.d=DeduplicationPipeline(self.p.handoff_path,self.model,self.llm,self.root/'dedup')

    def test_dedup_preserves_provenance_and_stage2_compatibility(self):
        original=self.p.handoff_path.read_bytes()
        result=self.d.propose(); report=self.d.commit(result)
        self.assertEqual(report['active_facts_before'],3); self.assertEqual(report['active_facts_after'],2)
        self.assertEqual(self.p.handoff_path.read_bytes(),original)
        k=PaperBundle(self.d.output_path).initial
        self.assertEqual(len(k.facts),3)
        self.assertEqual(k.facts['F00003'].superseded_by,'F00001')
        self.assertEqual(k.facts['F00001'].text,self.p.state.knowledge.facts['F00001'].text)
        self.assertEqual(len(k.facts['F00001'].evidence),2)
        self.assertEqual(k.facts['F00001'].verification_status,'checker_supported_provisional')
        self.assertEqual(k.questions['Q0002'].motivating_fact_ids,['F00001'])
        self.assertEqual(k.questions['Q0002'].motivating_candidates[0].accepted_fact_id,'F00003')
        store=Stage2Store(PaperBundle(self.d.output_path),self.root/'stage2')
        audit=SimpleNamespace(usage=lambda:{'calls':0,'reported_tokens':0})
        state={'tasks':{},'round_number':0,'objective':'demo','strategy':None,'round_observation':'','attempts':0,'round_task_ids':[]}
        payload=PlannerContext(store,Stage2Config(),audit).build(state)
        self.assertEqual({f['fact_id'] for f in payload['facts']},{'F00001','F00002'})
        self.assertIn('F00001',next(s for s in payload['sections'] if s['section_id']=='S002')['fact_ids'])

    def test_replay_is_cached(self):
        r=self.d.propose(); self.d.commit(r); n=len(self.llm.calls)
        d=DeduplicationPipeline(self.p.handoff_path,self.model,self.llm,self.root/'dedup')
        self.assertEqual(d.propose(),r); d.commit(r); self.assertEqual(len(self.llm.calls),n)

    def test_overlap_cycle_and_unknown(self):
        def group(c,rs): return DuplicateGroup(canonical_fact_id=c,redundant_fact_ids=rs,relationship='equivalent',reason='demo')
        for groups in [[group('F00001',['F00099'])],[group('F00001',['F00001'])],[group('F00001',['F00003']),group('F00003',['F00002'])],[group('F00001',['F00003','F00003'])]]:
            with self.assertRaises(ValueError): apply_groups(self.d.knowledge,DeduplicationResult(duplicate_groups=groups))
        self.assertEqual(self.p.state.knowledge.facts['F00003'].verification_status,'checker_supported_provisional')

    def test_changed_output_refused(self):
        r=self.d.propose(); r.duplicate_groups[0].reason='changed'
        with self.assertRaises(ValueError): self.d.commit(r)
        self.assertFalse(self.d.output_path.exists())

    def test_incomplete_and_size_limit(self):
        raw=json.loads(self.p.handoff_path.read_text()); raw['reading_complete']=False
        path=self.root/'partial.json'; path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError): DeduplicationPipeline(path,self.model,self.llm,self.root/'partial')
        with self.assertRaises(ValueError): DeduplicationPipeline(self.p.handoff_path,self.model,self.llm,self.root/'large',max_input_chars=1)

    def test_invalid_model_result_is_logged_without_commit(self):
        complete=self.llm.complete
        def broken(*args):
            raw=complete(*args); raw['choices'][0]['finish_reason']='length'; return raw
        self.llm.complete=broken
        with self.assertRaises(ValueError): self.d.propose()
        self.assertEqual(json.loads(self.d.log_path.read_text())['status'],'failed')
        self.assertFalse(self.d.output_path.exists())

if __name__=='__main__': unittest.main()
