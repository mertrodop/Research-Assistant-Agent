import json
import pytest
from pydantic import ValidationError
from research_agent.adaptive_v2.agent import AdaptiveAgent, allowed_kinds, wire_result
from research_agent.adaptive_v2.demo import demo_handoff, DemoLLM, task, finding
from research_agent.adaptive_v2.schema import Config, Task, Plan, Review, Result, Proposal, result_schema
from research_agent.adaptive_v2.calculator import calculate

@pytest.fixture
def agent(tmp_path):
    return AdaptiveAgent(demo_handoff(tmp_path/'fixture'),Config(),DemoLLM(),tmp_path/'run')

def test_end_to_end(agent):
    s=agent.run(lambda _:None)
    assert s.termination and s.revision>=5
    assert s.knowledge.questions['Q0001'].status=='unanswerable'
    assert s.knowledge.questions['Q0003'].status=='answered'
    assert s.question_triage['Q0002']['decision']=='defer'
    assert len(s.assessments)==2
    assert agent.audit.usage()['failed_calls']==0
    assert agent.audit.usage()['calls']==13
    assert len(s.episodes[1]['logs'])==1 # no verifier for question-only task
    assert s.episodes[1]['unverified_questions']==[0]
    assert any('20%' in f.text for f in s.knowledge.facts.values())
    assert (agent.export()/'question_triage.json').exists()
    assert all('schema_json' in json.loads(p.read_text()) for p in agent.audit.root.glob('*.json'))

def test_scope_and_schema(agent):
    s=agent.load(); t=Task(**task('INVESTIGATE','Recover details',sections=['S001']))
    sys,p,sch=agent.executor_request(t,s)
    assert list(p['source'])==['S001'] and not p['facts']
    assert 'not_found' not in allowed_kinds(t) and 'derive' not in allowed_kinds(t)
    assert 'derive:' not in sys
    with pytest.raises(ValidationError): sch.model_validate({'findings':[finding('derive','Unjustified')],'note':''})
    with pytest.raises(ValidationError): sch.model_validate({'findings':[finding('fact','Incorrect ID',refs=['F99999'],lines=[(2,2)])],'note':''})
    w=sch.model_validate({'findings':[finding('fact','The model uses two layers.',lines=[(2,2)])],'note':''})
    agent.validate_result(wire_result(w,t),t)

def test_new_ask_immediate(agent):
    s=agent.load(); p=agent.planner_payload(s)
    plan=Plan(tasks=[task('INVESTIGATE','Locate the number of training examples.',sections=['S001'],ask='How many training examples were used?')],triage=[],reason='Missing data')
    agent.validate_plan(plan,s,p); t=agent.apply_plan(s,plan)[0]
    assert not t.ask and len(t.questions)==1
    assert 'not_found' in allowed_kinds(t)
    assert agent.context(t,s)['questions'][t.questions[0]][0]=='How many training examples were used?'

def test_triage_reversible(agent):
    s=agent.load(); p=agent.planner_payload(s)
    plan=Plan(tasks=[],triage=[{'q':'Q0001','decision':'defer','related':'','reason':'Low priority'}],reason='stop')
    agent.validate_plan(plan,s,p); agent.apply_plan(s,plan)
    assert s.knowledge.questions['Q0001'].status=='open'
    assert 'Q0001' in s.question_triage
    plan=Plan(tasks=[task('INVESTIGATE','Find test set',questions=['Q0001'],sections=['S001'])],triage=[],reason='Revisit')
    agent.apply_plan(s,plan)
    assert 'Q0001' not in s.question_triage

def test_review_mixed(agent):
    t=Task(**task('REASON','Reason',facts=['F00002']))
    _,ctx,sch=agent.executor_request(t,agent.load())
    w=sch.model_validate({'findings':[finding('question','What was the sample size?',refs=['F00002']),finding('derive','The error rate was 20%.',refs=['F00002'])],'note':''})
    r=wire_result(w,t); agent.validate_result(r,t)
    payload=agent.verify_payload(ctx,r)
    assert list(payload['proposals'])==['1']
    with pytest.raises(ValueError): agent.validate_review(Review(checks=[{'i':0,'ok':True,'reason':'wrong index'}]),r)
    with pytest.raises(ValueError): agent.validate_review(Review(checks=[{'i':1,'ok':True,'reason':''}]),r)
    agent.validate_review(Review(checks=[{'i':1,'ok':True,'reason':'80% accuracy implies 20% error.'}]),r)

def test_full_scope_nonfinding(agent):
    s=agent.load(); t=Task(**task('INVESTIGATE','Locate dataset',questions=['Q0001'],sections=['S001','S002']))
    _,ctx,sch=agent.executor_request(t,s)
    r=wire_result(sch.model_validate({'findings':[finding('not_found','No test dataset name in selected sections.',refs=['Q0001'])],'note':''}),t)
    agent.validate_result(r,t)
    assert set(agent.verify_payload(ctx,r)['source'])=={'S001','S002'}

def test_negative_scope_validation(agent):
    t=Task(**task('INVESTIGATE','Find details',questions=['Q0001'],sections=['S001']))
    _,ctx,sch=agent.executor_request(t,agent.load())
    r=wire_result(sch.model_validate({'findings':[finding('fact','Unjustified lines.',lines=[(6,6)])],'note':''}),t)
    with pytest.raises(ValueError): agent.validate_result(r,t)

def test_multiple_resolution_rejected(agent):
    t=Task(**task('INVESTIGATE','Find details',questions=['Q0001'],sections=['S001']))
    _,ctx,sch=agent.executor_request(t,agent.load())
    r=wire_result(sch.model_validate({'findings':[finding('not_found','Missing A',refs=['Q0001']),finding('not_found','Missing B',refs=['Q0001'])],'note':''}),t)
    with pytest.raises(ValueError): agent.validate_result(r,t)

def test_resume_and_compact_history(agent):
    agent.step(); agent.step()
    a=AdaptiveAgent(agent.bundle.path,agent.config,DemoLLM(),agent.root)
    assert a.load().cursor==1
    p=a.planner_payload(a.load())
    assert 'Scripted demonstration' not in json.dumps(p['recent'])
    assert a.run(lambda _:None).termination

def test_arithmetic():
    assert calculate('0.619-0.604')=='3/200 (approx 0.015)'
    assert calculate('100-80')=='20'
    for x in ('__import__("os")','1/0','2**1000','(1).__class__','x+1'):
        with pytest.raises(ValueError): calculate(x)

def test_merge_inherits_evidence(agent):
    s=agent.load(); t=Task(**task('CONSOLIDATE','Consolidate only overlapping reports',facts=['F00001','F00003']))
    _,ctx,sch=agent.executor_request(t,s)
    r=wire_result(sch.model_validate({'findings':[finding('merge','The model uses two layers and no external datasets were evaluated.',refs=t.facts)],'note':''}),t)
    agent.validate_result(r,t)
    v=Review(checks=[{'i':0,'ok':True,'reason':'Fixture preserves both details.'}]); agent.validate_review(v,r)
    ids=agent.commit(s,t,r,v,'test')
    f=s.knowledge.facts[ids[0]]
    assert f.verification_status=='checker_supported_provisional'
    assert len(f.evidence)>=2
    assert all(s.knowledge.facts[i].verification_status=='superseded' for i in t.facts)
