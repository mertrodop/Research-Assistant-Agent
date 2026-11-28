"""Read the existing Stage 0 artifacts; do not reparse or rewrite the paper."""
from pathlib import Path
import hashlib, json
from pydantic import Field
from ..schemas.scientific import Record
from ..core.config import Stage1Config

class ReadingUnit(Record):
    unit_id: str
    section_id: str
    title: str
    kind: str
    heading_path: list[str]
    line_start: int
    line_end: int
    oversized_atomic_block: bool

class PaperSource:
    def __init__(self, markdown, parsed, catalog):
        self.markdown, self.parsed_path, self.catalog_path = map(Path, (markdown, parsed, catalog))
        raw = self.markdown.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.lines = raw.decode('utf-8-sig').splitlines()
        self.parsed = json.loads(self.parsed_path.read_text(encoding='utf-8'))
        self.catalog = json.loads(self.catalog_path.read_text(encoding='utf-8'))
        if self.sha256 != self.parsed['source']['sha256'] or len(self.lines) != self.parsed['source']['line_count']:
            raise ValueError('Stage 0 source hash or line count mismatch; rerun preprocessing.')
        self.paper_id = 'paper_' + self.sha256[:16]
        self.sections = {s['section_id']:s for s in self.parsed['sections']}
        if len(self.sections) != len(self.parsed['sections']) or len(self.catalog) != len(self.sections) or {s['section_id'] for s in self.catalog} != set(self.sections):
            raise ValueError('Stage 0 section catalog mismatch')
        for s in self.catalog:
            original = self.sections[s['section_id']]
            for k in ('title','kind','line_start','local_end','subtree_end'):
                if s[k] != original[k]: raise ValueError(f'Stale section catalog: {s["section_id"]} {k}')
        self.blocks = {}
        for b in self.parsed['blocks']:
            self.blocks.setdefault(b['section_id'], []).append(b)
        self.metadata = self._metadata()

    def original_lines(self, start, end, labels=True):
        if not 1 <= start <= end <= len(self.lines): raise ValueError(f'Invalid span {start}–{end}')
        return '\n'.join(f'[L{i}] {self.lines[i-1]}' if labels else self.lines[i-1] for i in range(start,end+1))

    def read_section(self, section_id, include_children=False, labels=True):
        s = self.sections[section_id]
        end = s['subtree_end'] if include_children else s['local_end']
        return self.original_lines(s['line_start'],end,labels) if end >= s['line_start'] else ''

    def _metadata(self):
        md = self.parsed.get('metadata', {})
        front = self.sections.get('S001', {})
        def cite(line, text):
            if not isinstance(line,int) or not 1 <= line <= len(self.lines) or not text or text.casefold() not in self.lines[line-1].casefold():
                return []
            owner = next((s['section_id'] for s in self.sections.values() if s['line_start'] <= line <= s['local_end']),None)
            return [{'section_id':owner,'line_start':line,'line_end':line}] if owner else []
        title = md.get('title'); authors = md.get('authors_candidate') or []
        return {'title':{'value':title,'evidence':cite(md.get('title_source_line'),title)},
                'authors':[{'value':a,'evidence':cite(md.get('authors_source_line'),a)} for a in authors],
                'doi_candidate':md.get('doi'), 'extraction_origin':'Stage 0 parser; evidence attached only where literal source match exists',
                'front_matter_source_lines':[{'line':i,'text':self.lines[i-1]} for i in range(front.get('line_start',1),front.get('local_end',0)+1) if self.lines[i-1].strip()]}

    def build_reading_plan(self, config: Stage1Config):
        units, skips = [], []
        for s in sorted(self.sections.values(),key=lambda x:x['line_start']):
            sid = s['section_id']; first=s['line_start']; last=s['local_end']
            blocks = sorted((b for b in self.blocks.get(sid,[]) if b['kind'] not in {'heading','rule'}),key=lambda b:b['line_start'])
            reason = None
            if sid=='S000' and not self.read_section(sid,labels=False).strip(): reason='empty unsectioned preamble'
            elif s['kind']=='bibliography' and not config.include_references: reason='bibliography excluded; retained for retrieval'
            elif s['kind']=='back_matter' and not config.include_back_matter: reason='back matter disabled'
            elif not blocks: reason='heading-only or no substantive blocks'
            if reason:
                skips.append({'section_id':sid,'title':s['title'],'reason':reason}); continue
            groups=[]; start=first; previous_end=None
            for b in blocks:
                if not first <= b['line_start'] <= b['line_end'] <= last: raise ValueError(f'Invalid block span in {sid}')
                if previous_end is not None and b['line_start'] <= previous_end: raise ValueError(f'Overlapping blocks in {sid}')
                if previous_end is not None and len(self.original_lines(start,b['line_end'],False)) > config.max_chars_per_unit:
                    # Include inter-block blank/raw lines, so no source lines are lost.
                    groups.append((start,b['line_start']-1)); start=b['line_start']
                previous_end=b['line_end']
            groups.append((start,last))
            for i,(a,z) in enumerate(groups,1):
                units.append(ReadingUnit(unit_id=sid if len(groups)==1 else f'{sid}_U{i:02d}',section_id=sid,title=s['title'],kind=s['kind'],heading_path=s.get('heading_path',[]),line_start=a,line_end=z,oversized_atomic_block=len(self.original_lines(a,z,False))>config.max_chars_per_unit))
        if len({u.unit_id for u in units})!=len(units): raise ValueError('Duplicate reading units')
        return units,skips
