"""Deterministic checks, not semantic verification."""
def validate_reader(source, unit, output, config):
    if len(output.facts)>config.max_facts_per_unit: raise ValueError('Reader exceeded fact cap')
    # Empty output is allowed for non-informative material; coverage remains a Stage II task.
    for i,f in enumerate(output.facts):
        if not unit.line_start<=f.line_start<=f.line_end<=unit.line_end: raise ValueError(f'Fact {i}: out-of-unit evidence')
        if not f.text.strip() or len(f.text)>850: raise ValueError(f'Fact {i}: empty or too long')

def validate_checker(reader, checker):
    if sorted(c.fact_index for c in checker.fact_checks)!=list(range(len(reader.facts))):
        raise ValueError('Checker must check each candidate exactly once')
    return {c.fact_index for c in checker.fact_checks if c.verdict=='supported'}

def question_error(q, i, count, cap):
    if i>=cap: return 'above configured question cap'
    if not q.question.strip() or not q.motivated_by_fact_indices: return 'empty question or no motivating indices'
    if len(q.question)>650: return 'question exceeds 650 characters'
    if any(not 0<=x<count for x in q.motivated_by_fact_indices): return 'invalid candidate index'
    return None
