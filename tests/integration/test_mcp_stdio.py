import asyncio
import sys
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from ra_agent.tools import TOOL_KINDS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_mcp_tools_and_list_papers():
    async def go():
        client = MultiServerMCPClient(
            {
                "ra": {
                    "command": sys.executable,
                    "args": ["-m", "ra_agent.mcp_server"],
                    "transport": "stdio",
                    "cwd": str(ROOT),
                    "env": {"PYTHONPATH": str(ROOT / "src")},
                }
            }
        )
        tools = await client.get_tools()
        listing = await next(t for t in tools if t.name == "list_papers").ainvoke({})
        return {t.name for t in tools}, str(listing)

    names, listing = asyncio.run(asyncio.wait_for(go(), 30))
    assert names == set(TOOL_KINDS)
    assert "paperbench-evaluating" in listing and "core-bench-fostering" in listing
