"""MCP stdio server: every Workspace method is a tool with the same name."""

import argparse

from mcp.server.fastmcp import FastMCP

from .config import Settings, load_env_file
from .tools import TOOL_KINDS, Workspace


def build_server(settings: Settings) -> FastMCP:
    server = FastMCP("research-assistant")
    workspace = Workspace(settings)
    for name in TOOL_KINDS:
        server.tool(name=name)(getattr(workspace, name))
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ra_agent.mcp_server")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    load_env_file()
    build_server(Settings.load(args.config)).run("stdio")


if __name__ == "__main__":
    main()
