"""Final paper-level synthesis (not Stage I section summaries)."""
import json
from pathlib import Path
from ..schemas.planning import Synthesis
from ..tools.retrieval import fact_view
from .prompts import SYNTHESIS
from .llm import BudgetExceeded, StructuredCallFailed


def write_summary(store,audit,config,reason):
    k=store.load().knowledge; active={fid:f for fid,f in k.facts.items() if f.verification_status not in ('superseded','disputed')}
    title=store.bundle.raw.get('paper_title') or 'Scientific paper'
    lines=[f'# {title}','',f'Investigation termination: {reason}','', 'These are manuscript-grounded claims; they are not independently verified scientific truth.','']
    if not store.bundle.raw['reading_complete']:
        lines.extend(['**Warning:** Stage I extraction was incomplete; conclusions and coverage are provisional.',''])
    def cite(ids):
        refs=[]
        for fid in ids:
            for e in active[fid].evidence:
                ref=f'{fid}, {e.section_id}/{e.unit_id}, L{e.line_start}–L{e.line_end}'
                if ref not in refs: refs.append(ref)
        return ' ('+'; '.join(refs)+')'
    def validate(s):
        if active and not any(sec.claims for sec in s.sections): raise ValueError('Empty summary')
        for sec in s.sections:
            if not sec.title.strip(): raise ValueError('Empty heading')
            for c in sec.claims:
                if not c.text.strip() or not c.fact_ids or not set(c.fact_ids)<=active.keys(): raise ValueError('Unsupported citation IDs')
    payload={'paper_title':title,'facts':[fact_view(f) for f in active.values()],'contradictions':[c.model_dump(mode='json') for c in k.contradictions.values()]}
    try:
        if len(json.dumps(payload,ensure_ascii=False))>config.max_planner_chars: raise BudgetExceeded('Synthesis context cap')
        s,_=audit.call('synthesis',SYNTHESIS,payload,Synthesis,config.planner,validate)
        for sec in s.sections:
            lines.extend(['## '+sec.title,'']); lines.extend('- '+c.text+cite(c.fact_ids) for c in sec.claims); lines.append('')
    except (BudgetExceeded,StructuredCallFailed) as exc:
        lines.extend(['## Accepted knowledge (deterministic fallback)','',f'LLM synthesis unavailable: {exc}.',''])
        for fid,f in active.items(): lines.append('- '+f.text+cite([fid]))
        lines.append('')
    lines.extend(['## Investigation questions',''])
    for q in k.questions.values():
        lines.append(f'- **{q.question_id} [{q.status}]** {q.question}')
        if q.answer:
            spans='; '.join(f'{e.section_id}/{e.unit_id}:L{e.line_start}–L{e.line_end}' for e in q.answer_evidence)
            lines.append(f'  - {q.answer}'+(f' ({spans})' if spans else ''))
    if k.contradictions:
        lines.extend(['','## Consistency assessments',''])
        for c in k.contradictions.values():
            e='; '.join(f'{x.section_id}:L{x.line_start}–L{x.line_end}' for x in c.assessment.evidence)
            lines.append(f'- {c.contradiction_id} [{c.assessment.verdict}]: {c.assessment.explanation} ({e})')
    if k.fact_issues:
        lines.extend(['','## Flagged extraction issues',''])
        lines.extend(f'- {fid}: {issue.explanation}' for fid,issue in k.fact_issues.items())
    if k.coverage_gaps:
        lines.extend(['','## Recorded coverage gaps (historical findings)',''])
        lines.extend(f'- {g.section_id}: {g.description}' for g in k.coverage_gaps.values())
    p=store.root/'final_summary.md'; tmp=p.with_suffix('.tmp'); tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); tmp.replace(p); return str(p)
