"""Strategy + Rounds: fixed LangGraph control loop, no executable future DAG."""
import json,sqlite3
from pathlib import Path
from ..schemas.execution import OrchestratorState,TaskRecord,VerificationResult,VerificationRecord
from ..schemas.planning import PlannerDecision,RoundEvaluation,TaskEvaluation,Synthesis
from ..stage1.pipeline import atomic_json,fingerprint
from ..tools.retrieval import Retrieval,dump
from ..tools.mcp_client import LocalMCPClient,InProcessClient,ContextBuilder
from .store import PaperBundle,Stage2Store
from .planner import Planner,records_from,records_dump
from .executor import ExecutorAgent,ACTION_MAP,empty_result
from .validation import validate_result,validate_evaluation,proposals_of,validate_verification_record
from .llm import AuditedLLM,BudgetExceeded,StructuredCallFailed
from .synthesis import write_summary
from .prompts import PLANNER,EVALUATOR,EXECUTOR_PROMPTS,SYNTHESIS,GROUNDED_VERIFIER,REASONING_VERIFIER
from .verifier import VerifierAgent,verification_input_hash

DEFAULT_OBJECTIVE='Refine manuscript-grounded scientific knowledge: consolidate redundancies, resolve important uncertainties and contradictions, and recover missing methodological/results information for final synthesis.'

class Stage2Pipeline:
    def __init__(self,handoff_path,config,llm,run_dir,source_paths=None,*,transport='mcp',objective=DEFAULT_OBJECTIVE):
        self.config=config; self.root=Path(run_dir).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.bundle=PaperBundle(handoff_path,source_paths,allow_partial=config.allow_partial_stage1)
        if transport not in ('mcp','inprocess'): raise ValueError('Unknown retrieval transport')
        self.objective=objective; self.transport=transport
        manifest={'version':'stage2-rounds-v3.1-compact-verified','handoff_identity':self.bundle.identity,'stage1_handoff':str(self.bundle.path),'config':dump(config),'llm':llm.identity,'transport':transport,'objective':objective,'prompts':fingerprint([PLANNER,EVALUATOR,EXECUTOR_PROMPTS,SYNTHESIS,GROUNDED_VERIFIER,REASONING_VERIFIER]),'schemas':fingerprint([x.model_json_schema() for x in [PlannerDecision,RoundEvaluation,Synthesis,VerificationResult,*[v[1] for v in ACTION_MAP.values()]]])}
        mp=self.root/'config.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest: raise ValueError('Stage II checkpoint mismatch; use a new run directory for Strategy + Rounds')
        if not mp.exists(): atomic_json(mp,manifest)
        self.store=Stage2Store(self.bundle,self.root); self.store.export()
        self.audit=AuditedLLM(llm,config,self.root); self.retrieval=Retrieval(self.store)
        self.client=LocalMCPClient(self.store,6) if transport=='mcp' else InProcessClient(self.retrieval)
        self.context_builder=ContextBuilder(self.client,config.max_context_chars,self.root)
        self.planner=Planner(self.store,config,self.audit); self.executor=ExecutorAgent(self.audit,config.executor,self.store,config.answerer); self.verifier=VerifierAgent(self.audit,config.verifier)
        self.result_dir=self.root/'task_results'; self.result_dir.mkdir(exist_ok=True)
        self.history=self.root/'plan_history'; self.history.mkdir(exist_ok=True)
        self.round_dir=self.root/'round_results'; self.round_dir.mkdir(exist_ok=True)
        self.conn=None; self.graph=None
        self.nodes={name:getattr(self,name) for name in ('plan_round','dispatch','prepare_context','execute','verify','evaluate_round','commit_round','synthesize')}

    def initial_state(self):
        return OrchestratorState(objective=self.objective,strategy=None,round_number=0,round_task_ids=[],round_observation='',tasks={},active_task_id=None,attempt_id=None,attempts=0,no_progress=0,context_path=None,result_paths={},evaluation=None,route='plan_round',termination_reason='',summary_path=None)
    def _stop(self,reason): return {'route':'synthesize','termination_reason':reason}
    def _limit(self,s):
        if s['round_number']>=self.config.max_rounds: return 'Round budget reached'
        if s['attempts']>=self.config.max_task_attempts: return 'Task-attempt budget reached'
        if s['no_progress']>=self.config.max_no_progress: return 'No-progress round limit reached'
        if not self.audit.available(): return 'LLM call/reported-token budget reached'
        return ''
    def plan_round(self,state):
        limit=self._limit(state)
        if limit: return self._stop(limit)
        n=state['round_number']+1
        try: decision,log=self.planner.plan(state)
        except (BudgetExceeded,StructuredCallFailed,ValueError) as exc: return self._stop('Round planning unavailable: '+str(exc))
        atomic_json(self.history/f'revision_{n:03d}.json',{'decision':dump(decision),'log':log})
        atomic_json(self.root/'planner_strategy.json',dump(decision.strategy))
        if decision.terminate: return {'strategy':dump(decision.strategy),**self._stop(decision.termination_reason)}
        records=records_from(state); ids=[]
        for task in decision.current_round_tasks:
            records[task.task_id]=TaskRecord(task=task,round_number=n); ids.append(task.task_id)
        return {'strategy':dump(decision.strategy),'round_number':n,'round_task_ids':ids,'tasks':records_dump(records),'result_paths':{},'evaluation':None,'route':'dispatch'}
    def dispatch(self,state):
        records=records_from(state); pending=[records[i] for i in state['round_task_ids'] if records[i].status=='pending']
        if not pending: return {'route':'evaluate_round'}
        # Never skip the review gate; if budget expires, remaining workers return failed observations.
        rec=pending[0]; tid=rec.task.task_id; aid=f'A{state["attempts"]+1:04d}_{tid}'
        rec.status='running'; rec.attempt_id=aid
        return {'tasks':records_dump(records),'active_task_id':tid,'attempt_id':aid,'attempts':state['attempts']+1,'context_path':None,'route':'prepare_context'}
    def prepare_context(self,state):
        task=records_from(state)[state['active_task_id']].task; aid=state['attempt_id']; path=self.result_dir/f'{aid}_context.json'
        if path.exists(): ctx=json.loads(path.read_text())
        else:
            try: ctx=self.context_builder.build(task,aid)
            except Exception as exc: ctx={'task':dump(task),'knowledge_revision':self.store.load().revision,'error':f'MCP retrieval failed: {exc}','facts':[],'questions':[],'passages':[],'source_sections':[],'section_catalog':[]}
            atomic_json(path,ctx)
        if ctx['task']!=dump(task) or ctx['knowledge_revision']!=self.store.load().revision: raise ValueError('Context cache/snapshot mismatch')
        return {'context_path':str(path),'route':'execute'}
    def execute(self,state):
        records=records_from(state); rec=records[state['active_task_id']]; task=rec.task; aid=state['attempt_id']; path=self.result_dir/f'{aid}.json'
        ctx=json.loads(Path(state['context_path']).read_text())
        if path.exists():
            d=json.loads(path.read_text())
            if d['task']!=dump(task) or d['context']!=ctx: raise ValueError('Task result cache mismatch')
        else:
            try: result,log=self.executor.execute(task,ctx,aid)
            except (BudgetExceeded,StructuredCallFailed) as exc: result,log=empty_result(task,'failed',str(exc)),None
            d={'task':dump(task),'context':ctx,'result':dump(result),'executor_log':log}; atomic_json(path,d)
        return {'result_paths':{**state['result_paths'],task.task_id:str(path)},'route':'verify'}
    def verify(self,state):
        records=records_from(state); rec=records[state['active_task_id']]; task=rec.task
        path=Path(state['result_paths'][task.task_id]); d=json.loads(path.read_text())
        ctx=d['context']; result=ACTION_MAP[task.action][1].model_validate(d['result'])
        vp=path.with_name(path.stem+'_verification.json')
        if vp.exists(): verification=VerificationRecord.model_validate_json(vp.read_text())
        else:
            verification=self.verifier.verify(task,ctx,result,state['attempt_id'])
            atomic_json(vp,dump(verification))
        self._check_verification(task,ctx,result,verification)
        rec.status='observed'; rec.outcome=result.outcome_summary
        return {'tasks':records_dump(records),'active_task_id':None,'attempt_id':None,'context_path':None,'route':'dispatch'}
    def _check_verification(self,task,ctx,result,verification):
        if verification.input_hash!=verification_input_hash(task,ctx,result): raise ValueError('Verification snapshot mismatch')
        validate_verification_record(verification,proposals_of(result))
    def _verifications(self,state,items):
        out={}
        for task,ctx,result in items:
            path=Path(state['result_paths'][task.task_id]); vp=path.with_name(path.stem+'_verification.json')
            v=VerificationRecord.model_validate_json(vp.read_text())
            self._check_verification(task,ctx,result,v); out[task.task_id]=v
        return out
    def _items(self,state):
        records=records_from(state); items=[]
        for tid in state['round_task_ids']:
            task=records[tid].task; d=json.loads(Path(state['result_paths'][tid]).read_text())
            if d['task']!=dump(task): raise ValueError('Task artifact mismatch')
            items.append((task,d['context'],ACTION_MAP[task.action][1].model_validate(d['result'])))
        if len({ctx['knowledge_revision'] for _,ctx,_ in items})!=1: raise ValueError('Round contexts used different knowledge snapshots')
        return items
    def evaluate_round(self,state):
        items=self._items(state)
        try: ev,log=self.planner.evaluate(state,items,self._verifications(state,items))
        except (BudgetExceeded,StructuredCallFailed) as exc:
            reason='Evaluation unavailable; no updates approved: '+str(exc)
            ev=RoundEvaluation(task_evaluations=[TaskEvaluation(task_id=t.task_id,accepted_proposal_indices=[],rationale=reason,task_satisfied=False) for t,_,_ in items],round_observation=reason); log=None
        out={'decision':dump(ev),'log':log}; atomic_json(self.round_dir/f'round_{state["round_number"]:03d}_evaluation.json',out)
        return {'evaluation':out,'route':'commit_round'}
    def commit_round(self,state):
        items=self._items(state); ev=RoundEvaluation.model_validate(state['evaluation']['decision']); verifications=self._verifications(state,items); validate_evaluation(ev,items,verifications)
        ledger=self.store.load(); records=records_from(state); key=str(state['round_number'])
        if key in ledger.round_commits: episodes=[ledger.episodes[i] for i in ledger.round_commits[key]]; self.store.export(ledger)
        else:
            if any(ctx['knowledge_revision']!=ledger.revision for _,ctx,_ in items): raise ValueError('Round knowledge changed before commit')
            prepared=[(records[t.task_id].attempt_id,t,r,validate_result(t,r,ctx,self.bundle,ledger.knowledge)) for t,ctx,r in items]
            episodes=self.store.commit_round(state['round_number'],prepared,ev,state['evaluation']['log'] or 'no-accepted-updates',verifications)
        decisions={x.task_id:x for x in ev.task_evaluations}
        for ep in episodes:
            rec=records[ep.task_id]; d=decisions[ep.task_id]
            rec.status='completed' if d.task_satisfied else 'partial' if ep.accepted_updates else 'failed'
            rec.outcome=ep.outcome_summary+' | Evaluation: '+d.rationale
        # A valid negative result is useful progress even when no fact changes.
        progress=any(ep.accepted_updates for ep in episodes) or any(d.task_satisfied for d in ev.task_evaluations)
        out={'tasks':records_dump(records),'round_observation':ev.round_observation,'no_progress':0 if progress else state['no_progress']+1,'evaluation':None}
        atomic_json(self.round_dir/f'round_{state["round_number"]:03d}_commit.json',{'episodes':[dump(e) for e in episodes],'observation':ev.round_observation,'knowledge_revision':self.store.load().revision})
        if state['evaluation']['log'] is None: return {**out,**self._stop(ev.round_observation)}
        limit=self._limit({**state,**out})
        return {**out,**(self._stop(limit) if limit else {'route':'plan_round'})}
    def synthesize(self,state):
        p=write_summary(self.store,self.audit,self.config,state['termination_reason']); return {'summary_path':p,'route':'end'}
    def compile(self):
        if self.graph is not None: return self.graph
        from langgraph.graph import StateGraph,START,END
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.conn=sqlite3.connect(str(self.root/'checkpoints.sqlite'),check_same_thread=False)
        builder=StateGraph(OrchestratorState)
        for name,node in self.nodes.items(): builder.add_node(name,node)
        builder.add_edge(START,'plan_round')
        routes={'plan_round':['dispatch','synthesize'],'dispatch':['prepare_context','evaluate_round'],'prepare_context':['execute'],'execute':['verify'],'verify':['dispatch'],'evaluate_round':['commit_round'],'commit_round':['plan_round','synthesize'],'synthesize':['end']}
        for name,targets in routes.items(): builder.add_conditional_edges(name,lambda state:state['route'],{r:END if r=='end' else r for r in targets})
        self.graph=builder.compile(checkpointer=SqliteSaver(self.conn)); return self.graph
    @property
    def graph_config(self): return {'configurable':{'thread_id':'stage2-rounds'},'recursion_limit':6*self.config.max_task_attempts+5*self.config.max_rounds+20}
    def stream(self):
        g=self.compile(); cfg=self.graph_config; snap=g.get_state(cfg)
        if snap.values and not snap.next: return
        yield from g.stream(None if snap.values else self.initial_state(),cfg,stream_mode='updates',durability='sync')
    def get_state(self): return self.compile().get_state(self.graph_config).values
    def close(self):
        if self.conn is not None: self.conn.close(); self.conn=None; self.graph=None
    def run_direct_for_tests(self,state=None):
        state=dict(state or self.initial_state())
        for _ in range(self.graph_config['recursion_limit']):
            node=state['route']
            if node=='end': return
            state={**state,**self.nodes[node](state)}; yield node,state
        raise RuntimeError('Direct test circuit breaker')
