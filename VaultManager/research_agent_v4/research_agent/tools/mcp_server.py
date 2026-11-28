"""MCP v1 SDK stdio server; read-only. Started by the Python host, not the LLM."""
import argparse,json
from .retrieval import Retrieval
from ..stage2.store import PaperBundle, Stage2Store

def create_server(retrieval):
    from mcp.server.fastmcp import FastMCP
    mcp=FastMCP('Paper Knowledge')
    # Return JSON text for stable decoding across v1 SDK releases.
    def encode(v): return json.dumps(v,ensure_ascii=False)
    @mcp.tool()
    def get_facts(ids:list[str])->str: return encode(retrieval.get_facts(ids))
    @mcp.tool()
    def get_candidates(ids:list[str])->str: return encode(retrieval.get_candidates(ids))
    @mcp.tool()
    def get_questions(ids:list[str])->str: return encode(retrieval.get_questions(ids))
    @mcp.tool()
    def get_section_catalog()->str: return encode(retrieval.get_section_catalog())
    @mcp.tool()
    def read_section(section_id:str)->str: return encode(retrieval.read_section(section_id))
    @mcp.tool()
    def read_lines(span:dict)->str: return encode(retrieval.read_lines(span))
    @mcp.tool()
    def read_table(table_id:str)->str: return encode(retrieval.read_table(table_id))
    @mcp.tool()
    def read_equation(equation_id:str)->str: return encode(retrieval.read_equation(equation_id))
    @mcp.tool()
    def search_paper(query:str,top_k:int=6)->str: return encode(retrieval.search_paper(query,top_k))
    @mcp.tool()
    def search_facts(query:str,top_k:int=10)->str: return encode(retrieval.search_facts(query,top_k))
    @mcp.tool()
    def get_task_history(ids:list[str])->str: return encode(retrieval.get_task_history(ids))
    @mcp.tool()
    def build_context(task:dict,max_chars:int)->str: return encode(retrieval.build_context(task,max_chars))
    return mcp

def main():
    p=argparse.ArgumentParser(); p.add_argument('--handoff',required=True); p.add_argument('--run-dir',required=True); p.add_argument('--source-paths',nargs=3,required=True); p.add_argument('--search-hits',type=int,default=6)
    a=p.parse_args(); b=PaperBundle(a.handoff,a.source_paths,allow_partial=True)
    # Read-only: the orchestrator must initialize the ledger before spawning this server.
    from pathlib import Path
    if not (Path(a.run_dir)/'ledger.json').is_file(): raise RuntimeError('Missing initialized Stage II ledger')
    create_server(Retrieval(Stage2Store(b,a.run_dir),a.search_hits)).run(transport='stdio')

if __name__=='__main__': main()
