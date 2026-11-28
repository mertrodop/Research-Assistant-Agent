"""Commit an entire reading unit transactionally; immutable source provenance."""
from ..schemas.scientific import (Fact, CandidateRecord, EvidenceRef, Question, CandidateAnchor, utc_now)
from ..schemas.checkpoint import UnitReview, RejectedQuestion
from ..stage1.validation import validate_checker, question_error

def unique(xs): return list(dict.fromkeys(xs))

def _new_id(prefix, mapping, width):
    # Preserve monotonic IDs even if a future consumer removes records.
    return f'{prefix}{max((int(k[len(prefix):]) for k in mapping),default=0)+1:0{width}d}'

def commit_unit(checkpoint, unit, reader, checker, config, reader_log, checker_log):
    if unit.unit_id in checkpoint.reviews: raise ValueError('Unit already committed')
    supported=validate_checker(reader,checker)
    by_index={x.fact_index:x for x in checker.fact_checks}
    nxt=checkpoint.model_copy(deep=True); store=nxt.knowledge
    candidates={}; accepted={}
    for i,f in enumerate(reader.facts):
        c=by_index[i]; cid=_new_id('C',store.candidate_facts,5)
        ev=EvidenceRef(section_id=unit.section_id,unit_id=unit.unit_id,line_start=f.line_start,line_end=f.line_end)
        fid=None
        if i in supported:
            fid=_new_id('F',store.facts,5)
            store.facts[fid]=Fact(fact_id=fid,text=f.text,origin_section_id=unit.section_id,origin_unit_id=unit.unit_id,evidence=[ev],source_candidate_id=cid,checked_by=checker_log)
            accepted[i]=fid
        record=CandidateRecord(candidate_id=cid,text=f.text,origin_section_id=unit.section_id,origin_unit_id=unit.unit_id,evidence=[ev],verdict=c.verdict,reason=c.reason,accepted_fact_id=fid)
        store.candidate_facts[cid]=record; candidates[i]=record
    questions=[]; rejected=[]
    for i,q in enumerate(checker.new_questions):
        error=question_error(q,i,len(candidates),config.max_questions_per_unit)
        if error:
            rejected.append(RejectedQuestion(question_index=i,proposal=q,reason=error)); continue
        anchors=[candidates[j] for j in unique(q.motivated_by_fact_indices)]
        qid=_new_id('Q',store.questions,4)
        store.questions[qid]=Question(question_id=qid,question=q.question.strip(),origin_section_id=unit.section_id,origin_unit_id=unit.unit_id,motivating_candidate_ids=[a.candidate_id for a in anchors],motivating_fact_ids=[a.accepted_fact_id for a in anchors if a.accepted_fact_id],motivating_candidates=[CandidateAnchor(candidate_id=a.candidate_id,accepted_fact_id=a.accepted_fact_id,verdict=a.verdict,checker_reason=a.reason) for a in anchors])
        questions.append(qid)
    nxt.reviews[unit.unit_id]=UnitReview(unit_id=unit.unit_id,section_id=unit.section_id,accepted_fact_ids=list(accepted.values()),candidate_fact_ids=[c.candidate_id for c in candidates.values()],new_question_ids=questions,rejected_questions=rejected,reader_log=reader_log,checker_log=checker_log)
    nxt.updated_at=utc_now()
    return nxt
