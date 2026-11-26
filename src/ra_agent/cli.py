"""`ra`: run the agent with human approval, ingest a paper, list papers, serve MCP."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from .config import Settings, load_env_file
from .llm.client import resolve_api_key
from .ocr.render import page_count
from .schema import NoteFrontmatter
from .vault import Vault
from .vault.store import _front

DECISIONS = {"a": "approve", "e": "edit", "r": "reject"}


def parse_decision(text: str) -> str | None:
    return DECISIONS.get(text.strip().lower())


def edit_and_validate(path: Path, editor_cmd: str) -> str:
    """Open the note in the editor; return the frontmatter error, or "" when it is valid."""
    subprocess.run([*shlex.split(editor_cmd), str(path)], check=False)
    try:
        NoteFrontmatter.model_validate(_front(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001 - a broken edit must come back as text, not a crash
        return str(exc)
    return ""


def decide(payload: dict, vault: Vault, input_fn=input) -> str:
    """Show a pending write and return "approve" or "reject"."""
    tool, args = payload["tool"], payload["args"]
    print(f"\nThe agent wants to run {tool} with {json.dumps(args, ensure_ascii=False)}\n")
    print(payload["diff"])
    note = vault.note(vault.staging, args.get("paper", ""))
    while True:
        choice = parse_decision(input_fn("[a]pprove, [e]dit, [r]eject? "))
        if choice == "edit" and tool == "commit_note" and note.is_file():
            while error := edit_and_validate(note, os.environ.get("EDITOR", "nano")):
                print(f"frontmatter is not valid, opening the editor again:\n{error}")
            print(vault.diff(args["paper"]))
        elif choice in {"approve", "reject"}:
            return choice


def note_path(messages) -> str:
    """Where the last commit_note (path) or stage_note (handle) put the note."""
    asked = {c["id"]: c["name"] for m in messages for c in getattr(m, "tool_calls", [])}
    found = ""
    for m in messages:
        key = {"commit_note": "path", "stage_note": "handle"}.get(asked.get(getattr(m, "tool_call_id", "")))
        if key:
            try:
                found = json.loads(m.content).get(key) or found
            except (ValueError, AttributeError):
                pass
    return found


async def run_agent(task: str, paper: str, settings: Settings, config: str | None) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from .agent.graph import build_chat_model, build_graph
    from .tools import Workspace

    args = ["-m", "ra_agent.mcp_server"] + (["--config", config] if config else [])
    server = {"command": sys.executable, "args": args, "transport": "stdio", "env": dict(os.environ)}
    tools = await MultiServerMCPClient({"ra": server}).get_tools()
    graph = build_graph(
        build_chat_model(settings.models["agent"]), tools, settings,
        lambda name: Workspace(settings).fallback_ingest(name), InMemorySaver(),
    )
    vault = Vault(Path(settings.paths.vault_dir), Path(settings.paths.staging_dir))
    run = {"configurable": {"thread_id": uuid.uuid4().hex}}  # state accumulates per thread
    out = await graph.ainvoke({"task": task, "paper": paper}, run)
    while "__interrupt__" in out:
        answer = decide(out["__interrupt__"][0].value, vault)
        out = await graph.ainvoke(Command(resume=answer), run)
    return out


def agent_command(args: argparse.Namespace, task: str, paper: str) -> int:
    settings = Settings.load(args.config)
    settings.agent.max_steps = args.max_steps or settings.agent.max_steps
    load_env_file()
    key_env = settings.models["agent"].api_key_env
    if not resolve_api_key(settings.models["agent"]):
        print(f"Set {key_env} (in the environment or .env) or point models.agent at a local endpoint.")
        return 2
    out = asyncio.run(run_agent(task, paper, settings, args.config))
    print(f"\n{out['messages'][-1].content}")
    if path := note_path(out["messages"]):
        print(f"Note: {path}")
    return 0 if out["outcome"] == "finish" else 1


def papers_command(args: argparse.Namespace) -> int:
    for pdf in Settings.load(args.config).paths.list_pdfs():
        print(f"{pdf.stem}  ({page_count(pdf)} pages)")
    return 0


def _find_pdf(pdf: str, config: str | None) -> str:
    """A bare file name lives in the Inbox, or in a paper's raw/ folder."""
    if pdf and len(Path(pdf).parts) == 1:
        return str(Settings.load(config).paths.find_pdf(pdf))
    return pdf


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML settings, e.g. config/pipeline.yaml")
    common.add_argument("--max-steps", type=int, default=None)
    parser = argparse.ArgumentParser(prog="ra", description="Research assistant agent")
    sub = parser.add_subparsers(dest="command", required=True)
    agent = sub.add_parser("agent", parents=[common], help="run the agent on a task")
    agent.add_argument("task")
    agent.add_argument("--paper", default="", help="PDF the task is about")
    sub.add_parser("ingest", parents=[common], help="ingest a PDF into the vault").add_argument("pdf")
    sub.add_parser("papers", parents=[common], help="list the PDFs in the Inbox and the vault")
    sub.add_parser("serve", parents=[common], help="run the MCP server on stdio")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "papers":
        return papers_command(args)
    if args.command == "serve":
        from .mcp_server import main as serve

        serve(["--config", args.config] if args.config else [])
        return 0
    if args.command == "ingest":
        pdf = _find_pdf(args.pdf, args.config)
        return agent_command(args, f"Ingest {Path(pdf).name} into the vault", pdf)
    return agent_command(args, args.task, _find_pdf(args.paper, args.config))


if __name__ == "__main__":
    sys.exit(main())
