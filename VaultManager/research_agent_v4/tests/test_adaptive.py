import tempfile, unittest
from pathlib import Path
from research_agent.adaptive.agent import AdaptiveAgent, allowed_kinds
from research_agent.adaptive.schema import Config, Task, Result, Review
from research_agent.adaptive.demo import demo_handoff, DemoLLM, task, prop

class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup); self.root=Path(t.name)
        self.h=demo_handoff(self.root/'fixture'); self.llm=DemoLLM(); self.config=Config(); self.a=AdaptiveAgent(self.h,self.config,self.llm,self.root/'run')
    def test_full_run_and_resume(self):
        s=self.a.run(lambda e:None); self.assertEqual(s.knowledge.questions['Q0001'].status,'unanswerable'); self.assertEqual(s.knowledge.questions['Q0002'].status,'answered')
        self.assertIn('derived',s.types.values()); self.assertIn('boundary',s.types.values()); self.assertEqual(len(s.assessments),2)
        self.assertTrue(s.termination); self.assertEqual(s.round,4); n=len(self.llm.calls)
        self.a=AdaptiveAgent(self.h,self.config,self.llm,self.root/'run'); self.a.run(lambda e:None); self.assertEqual(n,len(self.llm.calls))
        for schema,p in self.llm.calls:
            if schema=='Plan': self.assertNotIn('source',p)
            if schema=='Result' and p['action'] in ('ANSWER','QUESTION'): self.assertFalse(p['source'])
            if schema=='Review': self.assertNotIn('prompt',p)
        self.assertTrue(all(row['used'] for row in self.a.diagnostics())); self.a.export()
    def test_scoped_absence_not_global(self):
        self.a.step(); self.a.step(); s=self.a.load(); self.assertEqual(s.knowledge.questions['Q0001'].status,'open'); self.assertEqual(self.a.inspected(s,'Q0001'),['S001'])
    def test_invalid_scope_and_evidence(self):
        s=self.a.load(); t=Task.model_validate(task('INVESTIGATE','Read',questions=['Q0001'],sections=['S001']))
        r=Result.model_validate({'proposals':[prop('fact','Wrong section',spans=[(6,6)])],'note':''})
        with self.assertRaises(ValueError): self.a.validate_result(r,t)
        with self.assertRaises(ValueError): self.a.context(Task.model_validate(task('ANSWER','Bad',facts=['F00002'],questions=['Q0001'],sections=['S001'])),s)
    def test_verifier_rejection_no_commit(self):
        s=self.a.load(); t=Task.model_validate(task('REPAIR','repair',candidates=['C00002'],sections=['S001']))
        r=Result.model_validate({'proposals':[prop('fact','Test set accuracy was 80% on external datasets.',candidates=['C00002'],spans=[(3,3)])],'note':''})
        self.a.validate_result(r,t); v=Review.model_validate({'checks':[{'i':0,'ok':False,'reason':'External data not supported.'}]}); self.a.validate_review(v,r)
        self.assertFalse(self.a.commit(s,t,r,v,'test')); self.assertEqual(s.revision,0)
    def test_repair_and_merge(self):
        s=self.a.load(); t=Task.model_validate(task('REPAIR','repair',candidates=['C00002'],sections=['S001']))
        r=Result.model_validate({'proposals':[prop('fact','The reported test accuracy is 80 percent.',candidates=['C00002'],spans=[(3,3)])],'note':''}); self.a.validate_result(r,t)
        v=Review.model_validate({'checks':[{'i':0,'ok':True,'reason':''}]}); self.a.commit(s,t,r,v,'test'); self.assertEqual(s.assessments['A00001']['kind'],'repair')
        t=Task.model_validate(task('MERGE','Merge equivalent accuracy facts',facts=['F00002','F00004'],sections=['S001']))
        r=Result.model_validate({'proposals':[prop('merge','Reported test accuracy is 80%.',facts=t.facts,spans=[(3,3)])],'note':''}); self.a.validate_result(r,t); self.a.commit(s,t,r,v,'test2')
        self.assertEqual(s.knowledge.facts['F00002'].superseded_by,'F00005'); self.assertIn('F00005',s.knowledge.questions['Q0001'].motivating_fact_ids)
    def test_resume_midround_and_config_guard(self):
        self.a.step(); self.a.step(); self.a=AdaptiveAgent(self.h,self.config,self.llm,self.root/'run'); self.a.run(lambda e:None)
        self.assertEqual(sum(1 for schema,_ in self.llm.calls if schema=='Plan'),4)
        with self.assertRaisesRegex(ValueError,'Checkpoint mismatch'): AdaptiveAgent(self.h,Config(max_rounds=7),self.llm,self.root/'run')
    def test_budget_stops_without_unverified_commit(self):
        a=AdaptiveAgent(self.h,Config(max_llm_calls=2),DemoLLM(),self.root/'budget'); s=a.run(lambda e:None)
        self.assertEqual(s.revision,0); self.assertEqual(a.audit.usage()['calls'],2)

    def test_unanchored_investigation_rejected_before_execution(self):
        from research_agent.adaptive.schema import Plan
        s=self.a.load()
        p=Plan.model_validate({'tasks':[task('INVESTIGATE','Find decoder formula',sections=['S001'])],'reason':'Find formula'})
        with self.assertRaisesRegex(ValueError,'Task 1.*existing question IDs'):
            self.a.validate_plan(p,s,self.a.planner_payload(s))
        self.assertFalse(self.llm.calls)

    def test_source_only_coverage_and_invalid_ids(self):
        s=self.a.load(); t=Task.model_validate(task('COVERAGE','Find missing details',sections=['S001']))
        self.a.validate_task(t,s); self.assertEqual(allowed_kinds(t),['fact','question'])
        r=Result.model_validate({'proposals':[prop('fact','Test accuracy was 80%.',spans=[(3,3)],questions=['Q0001'])],'note':''})
        with self.assertRaisesRegex(ValueError,'Proposal 0: unselected questions IDs'):
            self.a.validate_result(r,t)
        r=Result.model_validate({'proposals':[prop('fact','Test accuracy was 80%.',spans=[(3,3)])],'note':''})
        self.a.validate_result(r,t)

    def test_single_resolution_per_question(self):
        t=Task.model_validate(task('INVESTIGATE','Find test dataset',questions=['Q0001'],sections=['S001']))
        r=Result.model_validate({'proposals':[prop('partial_answer','Test accuracy is reported, but the dataset is unidentified.',questions=['Q0001'],spans=[(3,3)]),prop('not_found','Dataset name not found.',questions=['Q0001'])],'note':''})
        with self.assertRaisesRegex(ValueError,'already has a resolution'):
            self.a.validate_result(r,t)
        r.proposals.pop(); self.a.validate_result(r,t)

    def test_planner_repair_before_executors(self):
        import json
        from research_agent.adaptive.schema import Plan
        class InvalidFirstPlan(DemoLLM):
            def complete(self,messages,schema,config):
                if schema is Plan and not self.calls:
                    self.calls.append(('invalid_plan',{}))
                    data={'tasks':[task('INVESTIGATE','Find test dataset',sections=['S001'])],'reason':'Test'}
                    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}]}
                if schema is Plan and len(self.calls)==1:
                    assert 'existing question IDs' in messages[-1]['content']
                return super().complete(messages,schema,config)
        llm=InvalidFirstPlan(); a=AdaptiveAgent(self.h,Config(),llm,self.root/'repair_plan')
        event=a.step(); self.assertEqual(event['event'],'plan'); self.assertEqual(event['tasks'][0]['questions'],['Q0001']); self.assertEqual(a.audit.usage()['calls'],2)
        self.assertEqual(a.audit.usage()['failed_calls'],1); self.assertEqual(a.load().revision,0)
        event=a.step(); self.assertEqual(event['event'],'task'); self.assertTrue(event['accepted'])
