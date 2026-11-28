"""Synchronous host bridge; runs async MCP in a fresh worker thread (Jupyter-safe)."""
import asyncio, json, sys, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ..stage1.pipeline import atomic_json

class LocalMCPClient:
    identity='mcp-stdio-v1'
    def __init__(self,store,max_search_hits=6):
        self.store=store; self.max_search_hits=max_search_hits
    async def _request(self,name,args):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        b=self.store.bundle; s=b.source
        root=Path(__file__).resolve().parents[2]
        env={**os.environ,'PYTHONPATH':str(root)+os.pathsep+os.environ.get('PYTHONPATH','')}
        # Retrieval subprocess does not need API credentials.
        for key in ('OPENAI_API_KEY','OPENROUTER_API_KEY'): env.pop(key,None)
        params=StdioServerParameters(command=sys.executable,args=['-m','research_agent.tools.mcp_server','--handoff',str(b.path),'--run-dir',str(self.store.root),'--source-paths',*[str(p.resolve()) for p in (s.markdown,s.parsed_path,s.catalog_path)],'--search-hits',str(self.max_search_hits)],env=env)
        async with stdio_client(params) as (r,w):
            async with ClientSession(r,w) as session:
                await session.initialize()
                if name=='__list__': return [t.name for t in (await session.list_tools()).tools]
                result=await session.call_tool(name,arguments=args)
                if result.isError: raise RuntimeError(str(result.content))
                texts=[x.text for x in result.content if x.type=='text']
                if texts:
                    obj=json.loads(texts[0])
                    if isinstance(obj,str): return json.loads(obj)
                    if isinstance(obj,dict) and set(obj)=={'result'} and isinstance(obj['result'],str): return json.loads(obj['result'])
                    return obj
                structured=getattr(result,'structuredContent',None)
                if isinstance(structured,dict) and isinstance(structured.get('result'),str): return json.loads(structured['result'])
                raise RuntimeError('MCP returned no JSON text')
    def call(self,name,**args):
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(asyncio.wait_for(self._request(name,args),timeout=90))).result()
    def list_tools(self): return self.call('__list__')

class InProcessClient:
    """Explicit test adapter: no actual MCP transport. Same retrieval functions."""
    identity='in-process-test'
    def __init__(self,retrieval): self.retrieval=retrieval
    def call(self,name,**args): return getattr(self.retrieval,name)(**args)
    def list_tools(self): return ['get_facts','get_candidates','get_questions','get_section_catalog','read_section','read_lines','read_table','read_equation','search_paper','search_facts','get_task_history','build_context']

class ContextBuilder:
    def __init__(self,client,max_chars,run_dir): self.client=client; self.max_chars=max_chars; self.root=Path(run_dir)/'tool_logs'; self.root.mkdir(exist_ok=True)
    def build(self,task,attempt_id):
        args={'task':task.model_dump(mode='json'),'max_chars':self.max_chars}
        try:
            obs=self.client.call('build_context',**args)
            atomic_json(self.root/f'{attempt_id}.json',{'tool':'build_context','arguments':args,'observation':obs,'transport':self.client.identity})
            return obs
        except Exception as exc:
            atomic_json(self.root/f'{attempt_id}.json',{'tool':'build_context','arguments':args,'error':str(exc),'transport':self.client.identity})
            raise
