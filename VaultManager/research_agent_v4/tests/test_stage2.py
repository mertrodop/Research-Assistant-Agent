"""Offline regressions and optional real LangGraph/MCP integration."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from research_agent.stage2.demo import make_stage2_demo_handoff,Stage2DemoLLM,task,ref,outcome,decision
from research_agent.stage2.config import Stage2Config
from research_agent.stage2.graph import Stage2Pipeline
from research_agent.stage2.validation import validate_decision,validate_result,validate_evaluation,proposals_of
from research_agent.schemas.planning import Task,PlannerDecision,RoundEvaluation,TaskEvaluation
from research_agent.schemas.execution import InvestigationResult,ConsolidationResult,ConsistencyResult,QuestionResult,CoverageResult,VerificationRecord,VerificationResult
from research_agent.stage2.store import PaperBundle
from research_agent.stage2.executor import ACTION_MAP
from research_agent.stage2.prompts import PLANNER

class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.handoff=make_stage2_demo_handoff(self.root/'fixture')
    def pipeline(self,config=None,transport='inprocess',llm=None,name='run'):
        p=Stage2Pipeline(self.handoff,config or Stage2Config(),llm or Stage2DemoLLM(),self.root/name,transport=transport)
        self.addCleanup(p.close); return p
    def test_round_loop_and_new_ids(self):
        old=self.handoff.read_bytes(); p=self.pipeline(); states=list(p.run_direct_for_tests()); state=states[-1][1]
        self.assertEqual(state['route'],'end'); self.assertEqual(state['round_number'],2); self.assertEqual(state['attempts'],3)
        self.assertEqual(state['tasks']['T001']['status'],'failed'); self.assertEqual(state['tasks']['T002']['status'],'completed'); self.assertEqual(state['tasks']['T003']['status'],'completed')
        self.assertEqual(p.store.load().knowledge.questions['Q0001'].status,'open')
        self.assertEqual(p.store.load().knowledge.facts['F00003'].text,'The reported test accuracy is 80%.')
        self.assertEqual(len(p.store.load().episodes),3); self.assertEqual(p.store.load().revision,2); self.assertEqual(old,self.handoff.read_bytes())
        self.assertEqual(p.audit.usage()['successful_calls'],11); self.assertIn('F00003',Path(state['summary_path']).read_text())
        self.assertEqual(len(p.store.load().knowledge.answer_groups),0)
        self.assertTrue((p.root/'planner_strategy.json').exists())
        contexts=[json.loads(x.read_text()) for x in (p.root/'task_results').glob('*_context.json')]
        self.assertEqual(sorted(c['knowledge_revision'] for c in contexts),[0,0,1])
        self.assertEqual([n for n,s in states if n=='evaluate_round'],['evaluate_round']*2)
    def test_action_registry_and_schema(self):
        self.assertEqual(len(ACTION_MAP),5)
        for name in ACTION_MAP: self.assertIn(name,PLANNER)
        for bad in ('search_queries','depends_on','evidence_spans','table_ids','equation_ids','candidate_ids'):
            self.assertNotIn(bad,Task.model_fields)
        with self.assertRaises(ValueError): Task.model_validate({**task('T001','CHECK_COVERAGE','x',section_ids=['S001']), 'search_queries':['x']})
    def test_decision_constraints(self):
        p=self.pipeline(); k=p.store.load().knowledge; state=p.initial_state()
        t=task('T001','ANSWER_QUESTION','Answer',fact_ids=['F00001'],question_ids=['Q0001'])
        for tasks in ([t,{**t,'task_id':'T002','task_prompt':'another'}], [{**t,'fact_ids':['F99999']}],[{**t,'section_ids':['S001']}], [task('T001','MERGE_FACTS','Merge',fact_ids=['F00001','F00002'],section_ids=['S001'])]):
            with self.assertRaises(ValueError): validate_decision(PlannerDecision.model_validate(decision('goal',tasks,1)),k,p.bundle,p.config,state)
        with self.assertRaises(ValueError): Task.model_validate({**t,'section_ids':['S001','S002','S003','S004']})
    def test_section_only_context(self):
        p=self.pipeline(); t=task('T001','INVESTIGATE_SOURCE','Inspect',question_ids=['Q0001'],section_ids=['S002'])
        ctx=p.retrieval.build_context(t,60000)
        self.assertNotIn('candidates',ctx); self.assertEqual(ctx['facts'],[]); self.assertEqual({x['ref']['section_id'] for x in ctx['passages']},{'S002'})
        self.assertEqual(ctx['passages'][0]['text'],p.bundle.source.read_section('S002'))
        tiny=p.retrieval.build_context(t,50); self.assertIn('error',tiny); self.assertEqual(tiny['passages'],[])
        answer=p.retrieval.build_context(task('T002','ANSWER_QUESTION','Answer',question_ids=['Q0001'],fact_ids=['F00001']),60000)
        self.assertEqual(answer['passages'],[])
        self.assertEqual([f['fact_id'] for f in answer['facts']],['F00001'])
        self.assertEqual(ctx['section_catalog'],p.retrieval.get_section_catalog())
        self.assertNotIn('motivating_candidate_ids',ctx['questions'][0])
        self.assertNotIn('motivating_candidates',ctx['questions'][0])
    def test_explicit_context_no_expansion(self):
        p=self.pipeline(); l=p.store.load(); q=l.knowledge.questions['Q0001']
        q.motivating_fact_ids=['F00001']; q.answer_fact_ids=['F00002']; p.store.save(l)
        t=task('T001','INVESTIGATE_SOURCE','Inspect',question_ids=['Q0001'],section_ids=['S002'])
        ctx=p.retrieval.build_context(t,60000)
        self.assertEqual(ctx['facts'],[])
        self.assertEqual([x['question_id'] for x in ctx['questions']],['Q0001'])
        self.assertNotIn('candidates',ctx)
        extract=p.retrieval.build_context(task('T002','INVESTIGATE_SOURCE','Inspect',section_ids=['S001']),60000)
        self.assertEqual(extract['facts'],[])
        planner=p.planner.context.build(p.initial_state())
        self.assertNotIn('rejected_candidates',planner)
        self.assertNotIn('omitted_candidate_ids',planner)
        self.assertNotIn('motivating_candidate_ids',planner['questions'][0])
        self.assertTrue(p.store.load().knowledge.candidate_facts)
    def test_coverage_requires_explicit_facts(self):
        p=self.pipeline(); t=task('T001','CHECK_COVERAGE','Assess',section_ids=['S001'])
        with self.assertRaisesRegex(ValueError,'missing:.*F00001'):
            p.retrieval.build_context(t,60000)
        t['fact_ids']=['F00001']
        self.assertEqual([f['fact_id'] for f in p.retrieval.build_context(t,60000)['facts']],['F00001'])
        l=p.store.load(); l.knowledge.facts['F00001'].verification_status='disputed'; p.store.save(l)
        t['fact_ids']=[]
        self.assertEqual(p.retrieval.build_context(t,60000)['facts'],[])
    def test_new_questions_without_candidates(self):
        p=self.pipeline(); t=Task.model_validate(task('T001','INVESTIGATE_SOURCE','Inspect',fact_ids=['F00001'],section_ids=['S001']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=InvestigationResult.model_validate(outcome('success','New uncertainty',new_facts=[],new_questions=[{'question':'What kind of layers?','motivating_fact_ids':['F00001'],'evidence':[ref('S001',2,2)]}]))
        props=validate_result(t,r,ctx,p.bundle,p.store.load().knowledge)
        ev=RoundEvaluation(task_evaluations=[TaskEvaluation(task_id='T001',accepted_proposal_indices=[0],rationale='Supported question',task_satisfied=True)],round_observation='New question')
        p.store.commit_round(1,[('A0001_T001',t,r,props)],ev,'test',{'T001':supported_record(1)})
        q=p.store.load().knowledge.questions['Q0002']
        self.assertEqual(q.motivating_candidate_ids,[])
        self.assertEqual(q.motivating_candidates,[])
        self.assertEqual(q.motivating_fact_ids,['F00001'])
    def test_evidence_restriction(self):
        p=self.pipeline(); t=Task.model_validate(task('T001','INVESTIGATE_SOURCE','Check results',section_ids=['S001']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=InvestigationResult.model_validate(outcome('success','bad evidence',new_facts=[{'text':'Not in context','evidence':[ref('S002',6,6)],'derived_from_fact_ids':[]}],new_questions=[]))
        with self.assertRaisesRegex(ValueError,'not available'): validate_result(t,r,ctx,p.bundle,p.store.load().knowledge)
    def test_grounded_merge_and_rejection(self):
        p=self.pipeline(); t=Task.model_validate(task('T001','MERGE_FACTS','Assess merge',fact_ids=['F00001','F00002'],section_ids=['S001','S002'])); ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000); k=p.store.load().knowledge
        reject=ConsolidationResult.model_validate(outcome('success','Different topics, no merge.',task_assessment='inappropriate_grouping',merge_disposition='not_mergeable',merged_facts=[],fact_issues=[]))
        self.assertEqual(validate_result(t,reject,ctx,p.bundle,k),[])
        merge=ConsolidationResult.model_validate(outcome('success','Merge supported',merge_disposition='merged',merged_facts=[{'text':'The model has two layers and no external datasets were evaluated.','evidence':[ref('S001',2,2),ref('S002',6,6)],'derived_from_fact_ids':['F00001','F00002'],'supersede_ids':['F00001','F00002']}],fact_issues=[]))
        ev=RoundEvaluation(task_evaluations=[TaskEvaluation(task_id='T001',accepted_proposal_indices=[0],rationale='Source support',task_satisfied=True)],round_observation='Merged')
        p.store.commit_round(1,[('A0001_T001',t,merge,validate_result(t,merge,ctx,p.bundle,k))],ev,'test',{'T001':supported_record(1)})
        self.assertEqual(p.store.load().knowledge.facts['F00001'].superseded_by,'F00003'); self.assertEqual(len(p.store.load().knowledge.facts['F00003'].evidence),2)
    def test_grouped_answer_and_inherited_evidence(self):
        p=self.pipeline(); l=p.store.load(); k=l.knowledge
        q=k.questions['Q0001'].model_copy(deep=True); q.question_id='Q0002'; q.question='Was external data evaluated?'; k.questions[q.question_id]=q; p.store.save(l)
        t=Task.model_validate(task('T001','ANSWER_QUESTION','Answer group',fact_ids=['F00002'],question_ids=['Q0001','Q0002'])); ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=QuestionResult.model_validate(outcome('partial','One answer known',remaining_uncertainties=['Dataset identity'],answers=[{'answer':'No external datasets were evaluated; dataset identity unknown.','resolutions':[{'question_id':'Q0001','status':'unresolved','explanation':'Not known from these facts.','answer_fact_ids':[],'evidence':[]},{'question_id':'Q0002','status':'answered','explanation':'No external datasets were evaluated.','answer_fact_ids':['F00002'],'evidence':[ref('S002',6,6)]}]}]))
        props=validate_result(t,r,ctx,p.bundle,k); self.assertEqual(len(props),1)
        ev=RoundEvaluation(task_evaluations=[TaskEvaluation(task_id='T001',accepted_proposal_indices=[0],rationale='Supported scope',task_satisfied=False)],round_observation='One unresolved')
        p.store.commit_round(1,[('A0001_T001',t,r,props)],ev,'test',{'T001':supported_record(1)}); kn=p.store.load().knowledge
        self.assertEqual(kn.questions['Q0001'].status,'open'); self.assertEqual(kn.questions['Q0002'].status,'answered'); self.assertEqual(len(kn.answer_groups),1)
        bad=r.model_copy(deep=True); bad.answers[0].resolutions[1].evidence=[ref_model('S001',2,2)]
        with self.assertRaises(ValueError): validate_result(t,bad,ctx,p.bundle,k)
    def test_commit_replay_and_plan_replay(self):
        p=self.pipeline(); start=p.initial_state(); first=p.plan_round(start); calls=p.audit.usage()['calls']; self.assertEqual(first,p.plan_round(start)); self.assertEqual(calls,p.audit.usage()['calls'])
        states=list(p.run_direct_for_tests()); before=p.store.load().model_dump(mode='json'); s=next(s for n,s in states if n=='evaluate_round'); p.commit_round(s); self.assertEqual(before,p.store.load().model_dump(mode='json'))
    def test_budgets(self):
        p=self.pipeline(Stage2Config(max_rounds=1)); state=list(p.run_direct_for_tests())[-1][1]; self.assertEqual(state['round_number'],1); self.assertIn('budget',state['termination_reason'])
        p=self.pipeline(Stage2Config(max_llm_calls=1),name='calls'); state=list(p.run_direct_for_tests())[-1][1]; self.assertEqual(p.audit.usage()['calls'],1); self.assertIn('fallback',Path(state['summary_path']).read_text()); self.assertEqual(len(p.store.load().knowledge.facts),2)
    def test_partial_handoff_and_fingerprint(self):
        self.pipeline()
        with self.assertRaisesRegex(ValueError,'mismatch'): self.pipeline(Stage2Config(max_rounds=2))
        raw=json.loads(self.handoff.read_text()); raw['reading_complete']=False; raw['completed_unit_ids']=raw['completed_unit_ids'][:1]; partial=self.root/'partial.json'; partial.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'incomplete'): PaperBundle(partial)
        p=Stage2Pipeline(partial,Stage2Config(allow_partial_stage1=True),Stage2DemoLLM(),self.root/'partialrun',transport='inprocess')
        ctx=p.planner.context.build(p.initial_state()); self.assertFalse(ctx['stage1_complete']); self.assertIn('unprocessed',{s['stage1_status'] for s in ctx['sections']})
    def test_round_evaluation_requires_every_task(self):
        p=self.pipeline(); t=Task.model_validate(task('T001','CHECK_COVERAGE','Check',section_ids=['S001'])); r=CoverageResult.model_validate(outcome('success','No gap',gaps=[],new_facts=[]))
        with self.assertRaises(ValueError): validate_evaluation(RoundEvaluation(task_evaluations=[],round_observation='Missing'),[(t,{},r)],{})
    def test_three_sections_and_separate_investigation_contract(self):
        # Three IDs accepted structurally, four refused (actual fixture has only two sections).
        from research_agent.schemas.execution import InformationRequest
        from research_agent.schemas.planning import MAX_SECTIONS_PER_TASK
        self.assertEqual(MAX_SECTIONS_PER_TASK,3)
        t=Task.model_validate(task('T001','INVESTIGATE_SOURCE','Inspect',section_ids=['S001','S002','S003']))
        self.assertEqual(len(t.section_ids),3)
        self.assertNotIn('answers',InvestigationResult.model_fields)
        self.assertEqual(len(InformationRequest(reason='More',fact_ids=[],question_ids=[],section_ids=t.section_ids).section_ids),3)
        with self.assertRaises(ValueError): InformationRequest(reason='More',fact_ids=[],question_ids=[],section_ids=[*t.section_ids,'S004'])
        with self.assertRaises(ValueError): InvestigationResult.model_validate(outcome('success','Wrong contract',new_facts=[],new_questions=[],answers=[]))

    def test_section_index_and_exact_selection(self):
        p=self.pipeline(); ctx=p.planner.context.build(p.initial_state())
        by={s['section_id']:s for s in ctx['sections']}
        self.assertIn('F00001',by['S001']['fact_ids']); self.assertIn('F00002',by['S002']['fact_ids'])
        self.assertIn('Q0001',by['S001']['question_ids'])
        l=p.store.load(); l.knowledge.facts['F00001'].verification_status='disputed'; p.store.save(l)
        by={s['section_id']:s for s in p.planner.context.build(p.initial_state())['sections']}
        self.assertNotIn('F00001',by['S001']['fact_ids'])

    def test_verifier_payload_independence_and_replay(self):
        from research_agent.stage2.verifier import verification_payload
        p=self.pipeline(); gen=p.run_direct_for_tests()
        for node,state in gen:
            if node=='execute' and state['active_task_id']=='T002': break
        path=Path(state['result_paths']['T002']); d=json.loads(path.read_text())
        t=Task.model_validate(d['task']); r=CoverageResult.model_validate(d['result'])
        payload=verification_payload(t,d['context'],r)
        self.assertEqual(set(payload),{'action','facts','questions','passages','numbered_proposals'})
        self.assertNotIn(t.task_prompt,json.dumps(payload)); self.assertEqual(payload['passages'],d['context']['passages'])
        first=p.verify(state); calls=p.audit.usage()['calls']; self.assertEqual(first,p.verify(state)); self.assertEqual(calls,p.audit.usage()['calls'])
        self.assertEqual(p.store.load().revision,0)
        report=json.loads(path.with_name(path.stem+'_verification.json').read_text()); self.assertEqual(report['checks'][0]['verdict'],'supported')
        # A result edited after verification must be rejected even if its proposal indices are unchanged.
        d['result']['new_facts'][0]['text']='Different claim'; path.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'snapshot mismatch'): p.verify(state)
        gen.close()

    def test_verifier_rejection_blocks_planner_and_store(self):
        p=self.pipeline(); t=Task.model_validate(task('T001','MERGE_FACTS','Assess merge',fact_ids=['F00001','F00002'],section_ids=['S001','S002']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=ConsolidationResult.model_validate(outcome('success','Proposed merge',merge_disposition='merged',merged_facts=[{'text':'The model has two layers.','evidence':[ref('S001',2,2)],'derived_from_fact_ids':['F00001','F00002'],'supersede_ids':['F00001','F00002']}],fact_issues=[]))
        props=validate_result(t,r,ctx,p.bundle,p.store.load().knowledge)
        report=VerificationRecord(status='completed',checks=[{'proposal_index':0,'verdict':'partially_supported','reason':'Drops the external dataset limitation'}],log='test-verifier',input_hash='test')
        ev=RoundEvaluation(task_evaluations=[TaskEvaluation(task_id='T001',accepted_proposal_indices=[0],rationale='Try to override rejection',task_satisfied=True)],round_observation='Bad approval')
        with self.assertRaisesRegex(ValueError,'verified eligible'): validate_evaluation(ev,[(t,ctx,r)],{'T001':report})
        with self.assertRaisesRegex(ValueError,'verified eligible'): p.store.commit_round(1,[('A0001_T001',t,r,props)],ev,'test',{'T001':report})
        self.assertEqual(p.store.load().revision,0); self.assertEqual(p.store.load().knowledge.facts['F00002'].verification_status,'checker_supported_provisional')

    def test_rejected_proposal_end_to_end(self):
        class RejectAccuracy(Stage2DemoLLM):
            def complete(self,messages,schema,config):
                raw=super().complete(messages,schema,config)
                if schema is VerificationResult:
                    payload=json.loads(messages[1]['content'])
                    if payload['action']=='CHECK_COVERAGE':
                        raw['choices'][0]['message']['content']=json.dumps({'checks':[{'proposal_index':0,'verdict':'unsupported','reason':'Scripted rejection to test hard gate'}]})
                return raw
        p=self.pipeline(Stage2Config(max_rounds=1),llm=RejectAccuracy())
        list(p.run_direct_for_tests()); l=p.store.load()
        self.assertNotIn('F00003',l.knowledge.facts)
        self.assertEqual(l.episodes['A0002_T002'].verification.checks[0].verdict,'unsupported')
        self.assertEqual(l.episodes['A0002_T002'].accepted_updates,[])

    def test_verifier_budget_fail_closed(self):
        p=self.pipeline(Stage2Config(max_llm_calls=4))
        list(p.run_direct_for_tests()); l=p.store.load()
        self.assertEqual(l.episodes['A0002_T002'].verification.status,'unavailable')
        self.assertEqual(len(l.knowledge.facts),2)
        self.assertEqual(p.audit.usage()['calls'],4)

    def test_answer_status_and_information_requests(self):
        p=self.pipeline(); k=p.store.load().knowledge
        t=Task.model_validate(task('T001','ANSWER_QUESTION','Answer',fact_ids=['F00001'],question_ids=['Q0001']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=QuestionResult.model_validate(outcome('insufficient_context','Not known',remaining_uncertainties=['Dataset'],answers=[]))
        with self.assertRaisesRegex(ValueError,'Every question'): validate_result(t,r,ctx,p.bundle,k)
        t=Task.model_validate(task('T002','INVESTIGATE_SOURCE','Inspect',section_ids=['S001']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=InvestigationResult.model_validate(outcome('insufficient_context','Not known',remaining_uncertainties=['Dataset'],information_requests=[{'reason':'Again','fact_ids':[],'question_ids':[],'section_ids':['S001']}],new_facts=[],new_questions=[]))
        with self.assertRaisesRegex(ValueError,'unexamined section'): validate_result(t,r,ctx,p.bundle,k)
        r.information_requests[0].section_ids=['S002']; r.information_requests[0].fact_ids=['F00001']
        with self.assertRaisesRegex(ValueError,'request fact IDs'): validate_result(t,r,ctx,p.bundle,k)

    def test_compact_planner_projection_preserves_memory(self):
        p=self.pipeline(); before=p.store.path.read_bytes()
        ctx=p.planner.context.build(p.initial_state())
        self.assertEqual(p.store.path.read_bytes(),before)
        k=p.store.load().knowledge
        self.assertEqual({f['fact_id'] for f in ctx['facts']},set(k.facts))
        for f in ctx['facts']:
            stored=k.facts[f['fact_id']]
            self.assertEqual(f['text'],stored.text)
            self.assertEqual(f['status'],stored.verification_status)
            self.assertEqual(f['section_ids'],sorted({e.section_id for e in stored.evidence}))
            self.assertNotIn('evidence',f)
        for q in ctx['questions']:
            self.assertEqual(q['question'],k.questions[q['question_id']].question)
            self.assertNotIn('answer_evidence',q)
            self.assertNotIn('origin_unit_id',q)
        list(p.run_direct_for_tests())
        from research_agent.stage2.planner import episode_view
        for e in p.store.load().episodes.values():
            v=episode_view(e)
            self.assertEqual(v['outcome_summary'],e.outcome_summary)
            self.assertEqual(v['accepted_updates'],e.accepted_updates)
            self.assertNotIn('verification',v)
            self.assertNotIn('created_at',v)
            self.assertNotIn('evidence_refs',v)
            self.assertTrue(e.verification.input_hash)

    def test_compact_evaluation_preserves_proposals_and_scope(self):
        from research_agent.stage2.planner import evaluation_payload
        p=self.pipeline()
        t=Task.model_validate(task('T001','CHECK_COVERAGE','Check',fact_ids=['F00001'],section_ids=['S001']))
        ctx=p.retrieval.build_context(t.model_dump(mode='json'),60000)
        r=CoverageResult.model_validate(outcome('success','Accuracy omitted',new_facts=[{'text':'The reported test accuracy is 80%.','evidence':[ref('S001',3,3)],'derived_from_fact_ids':[]}],gaps=[]))
        state=p.initial_state()
        payload=evaluation_payload(state,[(t,ctx,r)],{'T001':supported_record(1)})
        entry=payload['tasks'][0]
        self.assertNotIn('context',entry); self.assertNotIn('result',entry)
        self.assertNotIn('unverified_observation_passages',payload)
        self.assertNotIn('section_catalog',json.dumps(payload))
        self.assertEqual(entry['numbered_proposals'][0]['proposal'],r.new_facts[0].model_dump(mode='json'))
        self.assertEqual(entry['eligible_proposal_indices'],[0])
        self.assertEqual(entry['verification']['checks'][0]['verdict'],'supported')
        # Two unverified source assessments of the same section share one stored passage.
        t2=t.model_copy(update={'task_id':'T002'})
        ctx2={**ctx,'task':t2.model_dump(mode='json')}
        empty=CoverageResult.model_validate(outcome('success','No gap',gaps=[],new_facts=[]))
        skipped=VerificationRecord(status='skipped',reason='No proposals',input_hash='fixture')
        payload=evaluation_payload(state,[(t,ctx,empty),(t2,ctx2,empty)],{'T001':skipped,'T002':skipped})
        self.assertEqual(len(payload['facts']),1)
        self.assertEqual(len(payload['unverified_observation_passages']),len(ctx['passages']))
        self.assertEqual(payload['tasks'][0]['unverified_observation_passage_ids'],payload['tasks'][1]['unverified_observation_passage_ids'])
        self.assertEqual(list(payload['unverified_observation_passages'].values()),ctx['passages'])
        # Verification unavailable is not equivalent to supported.
        unavailable=VerificationRecord(status='unavailable',reason='Budget',input_hash='fixture')
        payload=evaluation_payload(state,[(t,ctx,r)],{'T001':unavailable})
        self.assertEqual(payload['tasks'][0]['eligible_proposal_indices'],[])
        self.assertEqual(payload['tasks'][0]['verification']['status'],'unavailable')

    @unittest.skipUnless(importlib.util.find_spec('langgraph') and importlib.util.find_spec('mcp'),'LangGraph/MCP not installed')
    def test_real_mcp_and_langgraph(self):
        p=self.pipeline(transport='mcp'); self.assertIn('build_context',p.client.list_tools()); list(p.stream()); self.assertEqual(p.get_state()['route'],'end'); self.assertEqual(p.store.load().revision,2)
    @unittest.skipUnless(importlib.util.find_spec('langgraph') and importlib.util.find_spec('mcp'),'LangGraph/MCP not installed')
    def test_real_resume(self):
        p=self.pipeline(transport='mcp'); stream=p.stream()
        for event in stream:
            if 'commit_round' in event: break
        stream.close(); p.close(); p=self.pipeline(transport='mcp'); list(p.stream()); self.assertEqual(p.get_state()['route'],'end'); self.assertEqual(p.store.load().revision,2)

def ref_model(s,a,z):
    from research_agent.schemas.scientific import EvidenceRef
    return EvidenceRef.model_validate(ref(s,a,z))


def supported_record(n):
    return VerificationRecord(status='completed',checks=[{'proposal_index':i,'verdict':'supported','reason':'Unit test fixture'} for i in range(n)],log='test-verifier',input_hash='test-fixture')
