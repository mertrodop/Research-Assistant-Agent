import json
import pytest
from research_agent.adaptive_v4.agent import AdaptiveAgent, wire_result
from research_agent.adaptive_v4.demo import DemoLLM, demo_handoff, task, finding, resolution
from research_agent.adaptive_v4.schema import Config, Plan, Review, result_schema

@pytest.fixture
def agent(tmp_path):
    return AdaptiveAgent(demo_handoff(tmp_path/'fixture'),Config(),DemoLLM(),tmp_path/'run')

def make_task(a,action,prompt,refs=(),sections=()):
    s=a.load(); ts,err=a.apply_plan(s,Plan(tasks=[task(action,prompt,refs,sections)],defer=[],reason='test'))
    assert not err
    return ts[0]

def make_result(a,t,findings=(),resolutions=()):
    w=result_schema().model_validate({'findings':list(findings),'resolutions':list(resolutions)})
    r,err=wire_result(w,t); r,err2=a.filter_result(r,t); return r,err+err2

def approve(a,r):
    v,e=a.normalize_review(Review(checks=[{'i':i,'ok':True,'reason':'fixture'} for i,p in enumerate(r.proposals) if p.kind!='question']),r)
    assert not e; return v

def test_demo(agent):
    s=agent.run(lambda e:None)
    assert s.knowledge.questions['Q0001'].status=='open'
    assert agent.inspected(s,'Q0003')==['S001','S002']
    assert len(s.assessments)==2 and s.revision==6
    assert s.question_merges=={'Q0001':'Q0003','Q0002':'Q0003'}
    assert agent.audit.usage()['failed_calls']==0
    assert agent.audit.usage()['calls']==14
    assert all(not e.get('validation_errors') for e in s.episodes)
    assert (agent.export()/'scientific_knowledge.md').exists()

def test_independent_plan_normalization(agent):
    s=agent.load()
    p=Plan(tasks=[task('INVESTIGATE','bad',['F99999'],['S001']),task('INVESTIGATE','good',['Q0001','Q0001'],['S001','S001'])],defer=['Q0001','Q9999','Q9999'],reason='test')
    ts,err=agent.apply_plan(s,p)
    assert len(ts)==1 and ts[0].questions==['Q0001'] and ts[0].sections==['S001']
    assert len(err)==2 and not s.question_triage
    assert set(Plan.model_json_schema()['$defs']['TaskWire']['properties'])=={'action','prompt','refs','sections'}

def test_reversible_deferral(agent):
    s=agent.load(); q=s.knowledge.questions['Q0001']; original=q.question
    agent.apply_plan(s,Plan(tasks=[],defer=['Q0001'],reason='defer'))
    p=agent.planner_payload(s)
    assert 'Q0001' not in p['questions'] and 'Q0001' in p['deferred']
    assert 'triage' not in p and q.status=='open' and q.question==original
    ts,err=agent.apply_plan(s,Plan(tasks=[task('INVESTIGATE','revisit',['Q0001'],['S001'])],defer=[],reason='revisit'),p)
    assert not err and 'Q0001' not in s.question_triage
    assert agent.context(ts[0],s)['questions']['Q0001'][0]==original

def test_full_context_for_uncited_boundary_and_answer(agent):
    t=make_task(agent,'INVESTIGATE','review reporting',['F00002','Q0001'],['S001','S002'])
    _,ctx,_=agent.executor_request(t,agent.load())
    r,err=make_result(agent,t,[finding('boundary','The reported accuracy alone cannot establish external generalization.',['F00002'])],[resolution('Q0001','answer','The test data were not named.',['F00002'])])
    assert not err
    p=agent.verify_payload(ctx,r)
    assert p['source']==ctx['source'] and set(p['source'])=={'S001','S002'}
    assert p['facts']==ctx['facts'] and p['questions']==ctx['questions']
    assert 'prompt' not in p and p['proposals']['1']['q']=='Q0001'
    # Script an independent rejection of the mislabeled answer; no closure.
    v,e=agent.normalize_review(Review(checks=[{'i':0,'ok':True,'reason':'fixture'},{'i':1,'ok':False,'reason':'Scoped non-finding, not answer'}]),r)
    s=agent.load(); agent.commit(s,t,r,v,'test')
    assert s.knowledge.questions['Q0001'].status=='open'

def test_nonfinding_preserves_partial(agent):
    s=agent.load(); q=s.knowledge.questions['Q0001']; q.status='partial'; q.answer='Some details known'; q.answer_fact_ids=['F00002']
    t=make_task(agent,'INVESTIGATE','Locate details',['Q0001'],['S001','S002'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','not_found','Exact name not located in selected sections.')]); assert not err
    agent.commit(s,t,r,approve(agent,r),'test')
    assert q.status=='partial' and q.answer=='Some details known' and q.answer_fact_ids==['F00002']
    assert agent.inspected(s,'Q0001')==['S001','S002']

def test_open_ended_nonfinding(agent):
    t=make_task(agent,'INVESTIGATE','Locate runtime',sections=['S001'])
    r,err=make_result(agent,t,[finding('not_found','Runtime not located in S001.')]); assert not err
    s=agent.load(); ids=agent.commit(s,t,r,approve(agent,r),'T1')
    a=s.assessments[ids[0]]; assert a['questions']==[] and a['target']=='Locate runtime' and a['sections']==['S001']

def test_explicit_resolution_and_invalid_item_salvage(agent):
    t=make_task(agent,'INVESTIGATE','Investigate',['Q0001','F00002'],['S001'])
    r,err=make_result(agent,t,[finding('fact','Invalid citation',lines=[(999,999)]),finding('derive','Invalid ref',['F99999'])],[resolution('Q0001','not_found','Name not located in S001.')])
    assert len(err)==2 and len(r.proposals)==1 and r.proposals[0].questions==['Q0001']
    r,err=make_result(agent,t,resolutions=[resolution('Q9999','not_found','bad'),resolution('Q0001','not_found','good')]); assert len(err)==1 and len(r.proposals)==1

def test_no_source_no_nonfinding(agent):
    t=make_task(agent,'INVESTIGATE','reason',['F00002','Q0001'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','not_found','not located')]); assert err and not r.proposals

def test_review_missing_duplicate_and_blank_rejected(agent):
    t=make_task(agent,'INVESTIGATE','reason',['F00002'])
    r,err=make_result(agent,t,[finding('derive','20% error',['F00002']),finding('boundary','Accuracy alone does not establish generalization',['F00002'])]); assert not err
    v,e=agent.normalize_review(Review(checks=[{'i':0,'ok':True,'reason':'supported'},{'i':1,'ok':True,'reason':''}]),r)
    assert v.checks[0].ok and not v.checks[1].ok and e
    v,e=agent.normalize_review(Review(checks=[{'i':0,'ok':True,'reason':'supported'},{'i':0,'ok':True,'reason':'duplicate'}]),r)
    assert not any(c.ok for c in v.checks) and len(e)==2

def test_question_not_verified_and_resume(agent):
    agent.step(); agent.step(); agent.step()
    assert len(agent.load().episodes[1]['logs'])==2
    a=AdaptiveAgent(agent.bundle.path,agent.config,DemoLLM(),agent.root)
    assert a.run(lambda e:None).termination
    assert a.audit.usage()['calls']==14

def test_explicit_answer_closes_question(agent):
    t=make_task(agent,'INVESTIGATE','reason',['F00002','Q0001'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','answer','Fixture answer.',['F00002'])]); assert not err
    s=agent.load(); agent.commit(s,t,r,approve(agent,r),'test'); assert s.knowledge.questions['Q0001'].status=='answered'
    p=agent.planner_payload(s); assert 'Q0001' not in p['questions'] and 'Q0001' in p['answers']


def test_memory_context_has_no_inherited_line_numbers(agent):
    t=make_task(agent,'INVESTIGATE','reason from known facts',['F00002'])
    s=agent.load(); ctx=agent.context(t,s)
    assert ctx['facts']['F00002']==[s.knowledge.facts['F00002'].text,'extracted']
    assert ctx['source']=={}
    r,err=make_result(agent,t,[finding('derive','An accuracy of 80% corresponds to an error rate of 20%.',['F00002'])])
    assert not err
    assert agent.verify_payload(ctx,r)['facts']==ctx['facts']
    ids=agent.commit(s,t,r,approve(agent,r),'test')
    assert len(ids)==1 and s.knowledge.facts[ids[0]].evidence==s.knowledge.facts['F00002'].evidence


def test_section_catalog_counts_local_substantive_lines(agent):
    p=agent.planner_payload(agent.load())
    for sid,(title,count) in p['sections'].items():
        assert title==agent.source.sections[sid]['title']
        expected={i for b in agent.source.blocks.get(sid,[]) if b['kind'] not in ('heading','rule') for i in range(b['line_start'],b['line_end']+1) if agent.source.lines[i-1].strip()}
        assert count==len(expected)
    assert p['sections']['S001'][1]==2
    assert p['sections']['S002'][1]==1
    # A local heading without its own substantive block contributes zero lines.
    agent.source.blocks['S001']=[b for b in agent.source.blocks['S001'] if b['kind']=='heading']
    assert agent.substantive_line_count('S001')==0


def test_verified_assessment_retrieval_and_provenance(agent):
    s=agent.load()
    t=make_task(agent,'INVESTIGATE','Locate runtime',sections=['S001'])
    r,err=make_result(agent,t,[finding('not_found','Runtime not located in S001.')]); assert not err
    aid=agent.commit(s,t,r,approve(agent,r),'T1')[0]
    agent.save(s)
    p=agent.planner_payload(s)
    assert p['assessments'][aid]==['not_found','Runtime not located in S001.',[],['S001']]
    t=make_task(agent,'INVESTIGATE','Evaluate reporting scope',[aid])
    ctx=agent.context(t,s)
    assert ctx['assessments'][aid]['sections']==['S001'] and not ctx['source']
    r,err=make_result(agent,t,[finding('boundary','S001 was inspected without locating runtime; the remaining sections are not covered.',[aid])]); assert not err
    vp=agent.verify_payload(ctx,r)
    assert vp['assessments']==ctx['assessments'] and vp['proposals']['0']['refs']==[aid]
    fid=agent.commit(s,t,r,approve(agent,r),'T2')[0]
    assert s.assessment_dependencies[fid]==[aid]
    assert {e.section_id for e in s.knowledge.facts[fid].evidence}=={'S001'}
    s.assessments[aid]['stale']=True
    assert aid not in agent.planner_payload(s).get('assessments',{})
    agent.invalidate_dependents(s,[],None)
    assert s.knowledge.facts[fid].verification_status=='disputed'


def test_source_line_numbers_remain_available(agent):
    t=make_task(agent,'INVESTIGATE','Inspect source',sections=['S001'])
    ctx=agent.context(t,agent.load())
    assert '[L' in ctx['source']['S001']
    unit=next(u for u in agent.bundle.units.values() if u.section_id=='S001')
    r,err=make_result(agent,t,[finding('fact','Fixture source observation.',lines=[(unit.line_start,unit.line_start)])])
    assert not err and r.proposals[0].spans[0].a==unit.line_start


def test_candidate_ids_are_not_model_facing(agent):
    s=agent.load(); assert s.knowledge.candidate_facts
    p=agent.planner_payload(s); assert 'candidates' not in p
    cid=next(iter(s.knowledge.candidate_facts))
    ts,err=agent.apply_plan(s,Plan(tasks=[task('INVESTIGATE','bad',[cid],['S001'])],defer=[],reason='test'))
    assert not ts and err
    t=make_task(agent,'INVESTIGATE','reason',['F00002']); assert 'candidates' not in agent.context(t,s)


def test_question_consolidation_preserves_scopes(agent):
    s=agent.load(); q=s.knowledge.questions['Q0001']; q2=q.model_copy(deep=True); q2.question_id='Q0002'; q2.question='How large was the test dataset?'; q2.status='partial'; q2.answer='No size established yet'; s.knowledge.questions['Q0002']=q2
    t=make_task(agent,'INVESTIGATE','Locate name',['Q0001'],['S001'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','not_found','Dataset name not found in S001.')]); assert not err
    agent.commit(s,t,r,approve(agent,r),'search'); agent.save(s)
    t=make_task(agent,'CONSOLIDATE','Combine related questions',['Q0001','Q0002'])
    r,err=make_result(agent,t,[finding('merge_questions','Which dataset and how many test samples were used?',['Q0001','Q0002'])]); assert not err
    vp=agent.verify_payload(agent.context(t,s),r); assert vp['proposals']['0']['kind']=='merge_questions'
    ids=agent.commit(s,t,r,approve(agent,r),'merge'); assert ids==['Q0003']
    assert s.question_merges=={'Q0001':'Q0003','Q0002':'Q0003'}
    assert agent.inspected(s,'Q0003')==[] and agent.inspected(s,'Q0001')==['S001']
    p=agent.planner_payload(s); assert set(p['questions'])=={'Q0003'}
    assert p['questions']['Q0003']['member_history']['Q0002']['answer']=='No size established yet'
    agent.save(s); t=make_task(agent,'INVESTIGATE','Revisit',['Q0003'],['S002']); ctx=agent.context(t,s)
    assert ctx['question_member_history']['Q0003']['Q0001']['searched']==['S001']
    assert agent.verify_payload(ctx,r)['question_member_history']==ctx['question_member_history']


def test_empty_sections_not_selectable(agent):
    agent.section_lengths['S001']=0
    s=agent.load(); assert 'S001' not in agent.planner_payload(s)['sections']
    ts,err=agent.apply_plan(s,Plan(tasks=[task('INVESTIGATE','empty',[],['S001'])],defer=[],reason='test'))
    assert not ts and err


def test_negative_search_and_new_question_not_substantive(agent):
    s=agent.load(); t=make_task(agent,'INVESTIGATE','search',['Q0001'],['S001'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','not_found','Name not located.')]); assert not err
    agent.commit(s,t,r,approve(agent,r),'search'); assert s.revision==1 and s.substantive_revision==0
    t=make_task(agent,'INVESTIGATE','question',['F00002']); r,err=make_result(agent,t,[finding('question','What is the sample size?',['F00002'])]); assert not err
    agent.commit(s,t,r,Review(checks=[]),'question'); assert s.revision==2 and s.substantive_revision==0
    t=make_task(agent,'INVESTIGATE','reason',['F00002']); r,err=make_result(agent,t,[finding('derive','20% error rate.',['F00002'])]); assert not err
    agent.commit(s,t,r,approve(agent,r),'reason'); assert s.substantive_revision==1


def test_fact_consolidation_preserves_evidence(agent):
    s=agent.load(); t=make_task(agent,'CONSOLIDATE','Merge overlapping facts',['F00001','F00003'])
    r,err=make_result(agent,t,[finding('merge','The model uses two layers and the authors did not evaluate external datasets.',['F00001','F00003'])]); assert not err
    old={(e.section_id,e.unit_id,e.line_start,e.line_end) for f in t.facts for e in s.knowledge.facts[f].evidence}
    fid=agent.commit(s,t,r,approve(agent,r),'merge')[0]
    assert all(s.knowledge.facts[f].superseded_by==fid for f in t.facts)
    assert {(e.section_id,e.unit_id,e.line_start,e.line_end) for e in s.knowledge.facts[fid].evidence}==old


def test_stagnation_stops_even_with_nonfinding_updates(agent):
    s=agent.load(); t=make_task(agent,'INVESTIGATE','search',['Q0001'],['S001'])
    r,err=make_result(agent,t,resolutions=[resolution('Q0001','not_found','Name not located.')]); assert not err
    agent.commit(s,t,r,approve(agent,r),'search')
    s.round=2; s.no_progress=1; s.pending=[t]; s.cursor=1; s.round_start_substantive=0
    agent.save(s); event=agent.step()
    assert event['event']=='terminated' and 'No substantive progress' in event['reason']
    assert agent.audit.usage()['calls']==0


def test_v41_explicit_record_repair(agent):
    s=agent.load()
    plan=Plan(tasks=[task('INVESTIGATE','Use F00002 and F00003 to assess Q0001; F99999 is not available.',['Q0001'])],defer=[],reason='test')
    ts,err=agent.apply_plan(s,plan)
    assert not err and ts[0].facts==['F00002','F00003']
    assert agent.plan_normalizations==[{'task':1,'explicit_prompt_refs_added':['F00002','F00003'],'motivating_facts_added':[]}]
    assert plan.tasks[0].refs==['Q0001'] # raw planner output is preserved
    assert set(agent.context(ts[0],s)['facts'])=={'F00002','F00003'}


def test_v41_motivating_fact_fallback(agent):
    s=agent.load(); q=s.knowledge.questions['Q0001']; q.motivating_fact_ids=['F00002']
    ts,err=agent.apply_plan(s,Plan(tasks=[task('INVESTIGATE','Investigate the selected question.',['Q0001'],['S001'])],defer=[],reason='test'))
    assert not err and ts[0].facts==['F00002']
    assert agent.plan_normalizations[0]['motivating_facts_added']==['F00002']
    # An explicit fact selection takes precedence over fallback.
    ts,err=agent.apply_plan(s,Plan(tasks=[task('INVESTIGATE','Investigate.',['Q0001','F00003'])],defer=[],reason='test'))
    assert not err and ts[0].facts==['F00003']


def test_v41_no_proposal_count_discard(agent):
    t=make_task(agent,'INVESTIGATE','Reason',['F00002','Q0001'])
    fs=[finding('derive',f'Fixture proposition {i}.',['F00002']) for i in range(6)]
    r,err=make_result(agent,t,fs,[resolution('Q0001','partial_answer','Fixture partial answer.',['F00002'])])
    assert not err and len(r.proposals)==7
    assert len(agent.verify_payload(agent.context(t,agent.load()),r)['proposals'])==7
    assert r.proposals[-1].questions==['Q0001']
    assert 'max_proposals' not in Config.model_fields
    assert 'max_proposals' not in agent.executor_request(t,agent.load())[1]


def test_v41_compact_history_preserves_archive(agent):
    s=agent.load(); long_prompt='Find the reported results. '+('Additional detail. '*100)
    s.episodes.append({'id':'R001_T01','action':'INVESTIGATE','prompt':long_prompt,'selected':{'facts':['F00002'],'questions':['Q0001'],'assessments':[],'sections':['S001']},'accepted':[],'error':'No answer'})
    history=agent.planner_payload(s)['history'][0]
    assert len(history['objective_preview'])<=200 and long_prompt not in json.dumps(history)
    assert history['refs']==['F00002','Q0001'] and history['sections']==['S001']
    assert history['error']=='No answer' and s.episodes[0]['prompt']==long_prompt


def test_v41_empty_member_history_omitted(agent):
    s=agent.load(); q=s.knowledge.questions['Q0001']
    q2=q.model_copy(deep=True); q2.question_id='Q0002'; q2.question='Another dataset question?'; s.knowledge.questions['Q0002']=q2
    agent.save(s)
    t=make_task(agent,'CONSOLIDATE','Merge questions',['Q0001','Q0002'])
    r,err=make_result(agent,t,[finding('merge_questions','Which dataset was used?',['Q0001','Q0002'])]); assert not err
    qid=agent.commit(s,t,r,approve(agent,r),'test')[0]; agent.save(s)
    assert agent.member_history(s,qid)=={}
    assert 'member_history' not in agent.planner_payload(s)['questions'][qid]
    t=make_task(agent,'INVESTIGATE','Investigate consolidated question',[qid])
    assert 'question_member_history' not in agent.context(t,s)
    assert s.question_merges=={'Q0001':qid,'Q0002':qid}


def test_v41_three_round_ceiling(agent):
    assert agent.config.max_rounds==3 and agent.config.verifier.max_tokens==6000
    s=agent.run(lambda e:None)
    assert s.round==3 and s.termination=='Round limit'
    assert len(s.plans)==3 and len(s.episodes)==6
    assert agent.planner_payload(s)['budget']['rounds_left']==0


def test_v41_planner_can_stop_early(agent):
    class StopLLM(DemoLLM):
        identity='stop-early-fixture'
        def complete(self,messages,schema,config):
            out=super().complete(messages,schema,config)
            if schema is Plan:
                out['choices'][0]['message']['content']=json.dumps({'tasks':[],'defer':[],'reason':'No justifiable scientific contribution remains.'})
            return out
    a=AdaptiveAgent(agent.bundle.path,Config(),StopLLM(),agent.root.parent/'early_stop')
    s=a.run(lambda e:None)
    assert s.round==1 and not s.episodes and 'No justifiable' in s.termination
