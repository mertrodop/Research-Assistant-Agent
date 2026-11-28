"""One global read, section-set grouped grounded checks, deterministic commit.

Successful calls are content-addressed by run and prompt fingerprints. The final
handoff is published only when all groups pass structural validation. No automatic
retry, semantic repair, Stage I.5, or knowledge mutation by checker workers.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import json
from pydantic import BaseModel, Field
from ..core.config import ModelConfig
from ..schemas.global_extraction import GlobalReaderOutput
from ..schemas.extraction import CheckerOutput
from ..schemas.scientific import (Record, EvidenceRef, KnowledgeStore, Fact,
    CandidateRecord, CandidateAnchor, Question)
from .source import ReadingUnit
from .pipeline import Stage1Pipeline, atomic_json, fingerprint
from .global_prompts import reader_messages, checker_messages, GLOBAL_READER_SYSTEM, GROUP_CHECKER_SYSTEM


class GlobalStage1Config(BaseModel):
    reader: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=16000,temperature=0.2,extra_body={'reasoning':{'effort':'medium'}}))
    checker: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=3000,temperature=0,extra_body={'reasoning':{'enabled':False}}))
    max_facts_per_group: int = Field(default=12,gt=0)
    max_questions_per_group: int = Field(default=4,ge=0)


class ResolvedCandidate(Record):
    candidate_id: str
    text: str
    evidence: list[EvidenceRef]


class CheckerGroup(Record):
    group_id: str
    section_ids: list[str]
    candidate_indices: list[int]


class GroupResult(Record):
    group_id: str
    checker: CheckerOutput
    log: str
    cached: bool


class GroupFailures(RuntimeError):
    def __init__(self, failures):
        self.failures=failures
        super().__init__('Checker groups failed; successful groups are cached. Rerun with same config to retry only failures: '+str(failures))


def resolve_candidates(source, reader):
    """Split a cross-section range at local-section boundaries; preserve every line.
    No semantic inference from evidence locations. Reject unmapped/overlapping ownership.
    """
    sections=sorted(source.sections.values(),key=lambda s:(s['line_start'],s['section_id']))
    owners={}
    for s in sections:
        for n in range(s['line_start'],s['local_end']+1):
            if not 1<=n<=len(source.lines) or n in owners:
                raise ValueError(f'Invalid or overlapping Stage 0 local section at line {n}')
            owners[n]=s['section_id']
    result=[]
    for i,f in enumerate(reader.facts):
        if not f.text.strip(): raise ValueError(f'Empty candidate {i}')
        refs=set()
        for span in f.evidence:
            a,z=span.line_start,span.line_end
            if not 1<=a<=z<=len(source.lines): raise ValueError(f'Candidate {i}: out-of-source evidence {a}-{z}')
            start=a; sid=owners.get(a)
            if sid is None: raise ValueError(f'Candidate {i}: unmapped evidence line {a}')
            for n in range(a+1,z+1):
                nxt=owners.get(n)
                if nxt is None: raise ValueError(f'Candidate {i}: unmapped evidence line {n}')
                if nxt!=sid:
                    refs.add((sid,start,n-1)); start=n; sid=nxt
            refs.add((sid,start,z))
        evidence=[EvidenceRef(section_id=s,unit_id=s,line_start=a,line_end=z) for s,a,z in sorted(refs,key=lambda x:(x[1],x[2],x[0]))]
        result.append(ResolvedCandidate(candidate_id=f'C{i+1:05d}',text=f.text.strip(),evidence=evidence))
    return result


def build_groups(candidates, max_facts_per_group):
    """Exact supporting-section-set grouping. Split only for configured batch size."""
    by_sections=defaultdict(list)
    for i,c in enumerate(candidates):
        signature=tuple(dict.fromkeys(e.section_id for e in c.evidence))
        by_sections[signature].append(i)
    groups=[]
    for sections,indices in by_sections.items():
        for start in range(0,len(indices),max_facts_per_group):
            groups.append(CheckerGroup(group_id=f'G{len(groups)+1:03d}',section_ids=list(sections),candidate_indices=indices[start:start+max_facts_per_group]))
    assert sorted(i for g in groups for i in g.candidate_indices)==list(range(len(candidates)))
    return groups


def validate_group(group, output, cap):
    n=len(group.candidate_indices)
    if sorted(c.fact_index for c in output.fact_checks)!=list(range(n)):
        raise ValueError('Checker must check each local candidate exactly once')
    if len(output.new_questions)>cap:
        raise ValueError('Checker exceeded question cap')
    for q in output.new_questions:
        if not q.question.strip() or not q.motivated_by_fact_indices or len(q.question)>650:
            raise ValueError('Question must be nonempty, anchored and at most 650 characters')
        if len(set(q.motivated_by_fact_indices))!=len(q.motivated_by_fact_indices) or any(not 0<=i<n for i in q.motivated_by_fact_indices):
            raise ValueError('Question has duplicate or invalid local candidate indices')


class GlobalStage1Pipeline:
    # Reuse the audited transport (raw response persisted before validation).
    _call=Stage1Pipeline._call
    usage_summary=Stage1Pipeline.usage_summary

    def __init__(self, source, config, llm, run_dir):
        self.source=source; self.config=config; self.llm=llm
        self.run_dir=Path(run_dir).resolve(); self.run_dir.mkdir(parents=True,exist_ok=True)
        self.log_dir=self.run_dir/'call_logs'; self.log_dir.mkdir(exist_ok=True)
        self.handoff_path=self.run_dir/'stage1_agent_handoff.json'
        self.manifest_path=self.run_dir/'global_manifest.json'
        self.run_fingerprint=fingerprint({'version':'global-stage1-v1','source':source.sha256,'parsed':source.parsed,'catalog':source.catalog,'config':config.model_dump(mode='json'),'transport':llm.identity,'prompts':[GLOBAL_READER_SYSTEM,GROUP_CHECKER_SYSTEM],'schemas':[GlobalReaderOutput.model_json_schema(),CheckerOutput.model_json_schema()]})
        if self.manifest_path.exists():
            if json.loads(self.manifest_path.read_text())['run_fingerprint']!=self.run_fingerprint:
                raise ValueError('Checkpoint mismatch: use a NEW run directory after changing source, model, prompts or config.')
        elif any(self.log_dir.glob('*.json')) or self.handoff_path.exists():
            raise ValueError('Existing unrelated run artifacts: use a NEW run directory')
        else:
            atomic_json(self.manifest_path,{'run_fingerprint':self.run_fingerprint,'source_sha256':source.sha256,'config':config.model_dump(mode='json')})

    def _cached_call(self, key, stage, messages, schema, model, validator):
        for p in sorted(self.log_dir.glob(f'{key}_{stage}_*.json'),reverse=True):
            d=json.loads(p.read_text(encoding='utf-8'))
            if d.get('run_fingerprint')==self.run_fingerprint and d.get('prompt_hash')==fingerprint(messages) and d.get('status')=='success' and d.get('schema')==schema.__name__ and d.get('model')==model.model_dump(mode='json'):
                result=schema.model_validate(d['parsed']); validator(result)
                return result,str(p.relative_to(self.run_dir)),True
        result,log=self._call(SimpleNamespace(unit_id=key),stage,messages,schema,model,validator)
        return result,log,False

    def read(self):
        reader,log,cached=self._cached_call('GLOBAL','reader',reader_messages(self.source),GlobalReaderOutput,self.config.reader,lambda r:resolve_candidates(self.source,r))
        candidates=resolve_candidates(self.source,reader)
        groups=build_groups(candidates,self.config.max_facts_per_group)
        atomic_json(self.run_dir/'global_candidates.json',[c.model_dump(mode='json') for c in candidates])
        atomic_json(self.run_dir/'checker_groups.json',[g.model_dump(mode='json') for g in groups])
        return reader,log,cached,candidates,groups

    def check(self, group, candidates):
        messages=checker_messages(self.source,group,candidates,self.config.max_questions_per_group)
        checker,log,cached=self._cached_call(group.group_id,'checker',messages,CheckerOutput,self.config.checker,lambda r:validate_group(group,r,self.config.max_questions_per_group))
        return GroupResult(group_id=group.group_id,checker=checker,log=log,cached=cached)

    def check_all(self, groups, candidates, max_workers=1, on_result=None):
        if max_workers<1: raise ValueError('max_workers must be positive')
        results={}; failures={}
        # Drain every submitted future even after a failure; successful responses remain cached.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures={pool.submit(self.check,g,candidates):g.group_id for g in groups}
            for future in as_completed(futures):
                gid=futures[future]
                try: r=future.result()
                except Exception as exc: failures[gid]=f'{type(exc).__name__}: {exc}'
                else:
                    results[gid]=r
                    if on_result: on_result(r)
        if failures: raise GroupFailures(failures)
        return [results[g.group_id] for g in groups]

    def _verify_log(self, rel, key, stage, messages, obj):
        p=(self.run_dir/rel).resolve()
        if p.parent!=self.log_dir.resolve(): raise ValueError('Log outside current run')
        d=json.loads(p.read_text(encoding='utf-8'))
        if (d.get('run_fingerprint')!=self.run_fingerprint or d.get('unit_id')!=key or d.get('stage')!=stage or d.get('status')!='success' or d.get('prompt_hash')!=fingerprint(messages) or d.get('parsed')!=obj.model_dump(mode='json')):
            raise ValueError('Output/log mismatch: '+key)

    def commit(self, reader, reader_log, results):
        """Stable IDs by original reader order; no partial publish, no unchecked facts."""
        candidates=resolve_candidates(self.source,reader)
        groups=build_groups(candidates,self.config.max_facts_per_group)
        self._verify_log(reader_log,'GLOBAL','reader',reader_messages(self.source),reader)
        result_map={r.group_id:r for r in results}
        if len(result_map)!=len(results) or set(result_map)!={g.group_id for g in groups}: raise ValueError('Missing or duplicated checker groups')
        checks={}
        for g in groups:
            r=result_map[g.group_id]; validate_group(g,r.checker,self.config.max_questions_per_group)
            self._verify_log(r.log,g.group_id,'checker',checker_messages(self.source,g,candidates,self.config.max_questions_per_group),r.checker)
            for c in r.checker.fact_checks: checks[g.candidate_indices[c.fact_index]]=(c,r.log)
        selected={'reader':reader_log,'checkers':{g.group_id:result_map[g.group_id].log for g in groups}}
        # Preserve timestamps and exact handoff fingerprint on notebook re-run.
        if self.handoff_path.exists():
            old=json.loads(self.handoff_path.read_text(encoding='utf-8'))
            if old.get('global_initialization',{}).get('selected_logs')!=selected: raise ValueError('Existing committed handoff uses different call logs; use a new run directory')
            return old
        k=KnowledgeStore(paper_id=self.source.paper_id)
        for i,c in enumerate(candidates):
            check,log=checks[i]; first=c.evidence[0]; fid=None
            if check.verdict=='supported':
                fid=f'F{len(k.facts)+1:05d}'
                k.facts[fid]=Fact(fact_id=fid,text=c.text,origin_section_id=first.section_id,origin_unit_id=first.unit_id,evidence=c.evidence,source_candidate_id=c.candidate_id,checked_by=log)
            k.candidate_facts[c.candidate_id]=CandidateRecord(candidate_id=c.candidate_id,text=c.text,origin_section_id=first.section_id,origin_unit_id=first.unit_id,evidence=c.evidence,verdict=check.verdict,reason=check.reason,accepted_fact_id=fid)
        # Exact normalized question duplicates only: union their candidate anchors.
        questions={}
        for g in groups:
            for q in result_map[g.group_id].checker.new_questions:
                key=' '.join(q.question.casefold().split())
                item=questions.setdefault(key,{'question':q.question.strip(),'ids':[]})
                for i in q.motivated_by_fact_indices:
                    cid=candidates[g.candidate_indices[i]].candidate_id
                    if cid not in item['ids']: item['ids'].append(cid)
        for item in questions.values():
            cs=[k.candidate_facts[cid] for cid in item['ids']]; first=cs[0]; qid=f'Q{len(k.questions)+1:04d}'
            k.questions[qid]=Question(question_id=qid,question=item['question'],origin_section_id=first.origin_section_id,origin_unit_id=first.origin_unit_id,motivating_candidate_ids=item['ids'],motivating_fact_ids=list(dict.fromkeys(c.accepted_fact_id for c in cs if c.accepted_fact_id)),motivating_candidates=[CandidateAnchor(candidate_id=c.candidate_id,accepted_fact_id=c.accepted_fact_id,verdict=c.verdict,checker_reason=c.reason) for c in cs])
        # Retrieval units are complete local sections, not checker groups. All sections
        # were seen by global reader; processed does not mean exhaustive fact coverage.
        units=[ReadingUnit(unit_id=s['section_id'],section_id=s['section_id'],title=s['title'],kind=s['kind'],heading_path=s.get('heading_path',[]),line_start=s['line_start'],line_end=s['local_end'],oversized_atomic_block=False) for s in self.source.catalog if s['line_start']<=s['local_end']]
        handoff={'schema_version':'stage1-agent-handoff-v2.1','source_sha256':self.source.sha256,'paper_title':self.source.parsed.get('metadata',{}).get('title'),'bibliographic_metadata':self.source.metadata,'source_paths':{'markdown':str(self.source.markdown.resolve()),'parsed_paper':str(self.source.parsed_path.resolve()),'section_catalog':str(self.source.catalog_path.resolve())},'reading_complete':True,'completed_unit_ids':[u.unit_id for u in units],'total_units':len(units),'skipped_sections':[],'sections':self.source.catalog,'reading_plan':[u.model_dump(mode='json') for u in units],**k.model_dump(mode='json'),'global_initialization':{'version':'global-stage1-v1','run_fingerprint':self.run_fingerprint,'grouping':'exact supporting section set, bounded candidate batches','groups':[g.model_dump(mode='json') for g in groups],'selected_logs':selected},'notes':'All original lines supplied to global reader. Sections with no candidates were not checked; processed does not imply exhaustive coverage. Checker-supported facts are provisional manuscript-grounded claims. Partial/unsupported candidates are retained as investigation leads. No Stage I.5 LLM deduplication.'}
        atomic_json(self.handoff_path,handoff)
        return handoff

    def call_diagnostics(self):
        selected=set()
        if self.handoff_path.exists():
            logs=json.loads(self.handoff_path.read_text())['global_initialization']['selected_logs']; selected={logs['reader'],*logs['checkers'].values()}
        rows=[]
        for p in sorted(self.log_dir.glob('*.json')):
            d=json.loads(p.read_text()); raw=d.get('raw') or {}; usage=raw.get('usage') or {}; choice=(raw.get('choices') or [{}])[0]; content=(choice.get('message') or {}).get('content') or ''; parsed=d.get('parsed') or {}; rel=str(p.relative_to(self.run_dir))
            if d.get('run_fingerprint')!=self.run_fingerprint: continue
            rows.append({'group':d['unit_id'],'stage':d['stage'],'status':d['status'],'committed':rel in selected,'model':raw.get('model') or d['model']['model'],'provider':raw.get('provider'),'finish_reason':choice.get('finish_reason'),'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'reasoning_tokens':(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),'total_tokens':usage.get('total_tokens'),'cost':usage.get('cost'),'output_chars':len(content),'whitespace_chars':sum(ch.isspace() for ch in content),'trailing_whitespace_chars':len(content)-len(content.rstrip()),'facts':len(parsed.get('facts',[])),'checks':len(parsed.get('fact_checks',[])),'questions':len(parsed.get('new_questions',[])),'duration_seconds':d.get('duration_seconds'),'error':d.get('error'),'file':rel})
        return rows
