"""Agent loop: the LLM proposes one action, code validates and executes it.

The loop is deliberately small and deterministic in its control flow:

    perceive -> propose -> validate -> act -> observe -> reflect -> repeat

The model never decides whether to stop on its own terms: a circuit breaker in
code ends the run on step budget, repeated actions or consecutive failures.
Every step is recorded, so a run can be replayed and audited.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import AgentSettings
from ..llm.client import LlmClient, VisionError
from .tools import Tool, call_tool, catalog

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)

SYSTEM = """You are the orchestrator of a research assistant that ingests papers.

You can call exactly one tool per step. Available tools:
{catalog}

Answer with strict JSON only, no prose and no code fences. Either call a tool:
{{"thought": "why this step", "action": "tool_name", "args": {{...}}}}

or finish:
{{"thought": "why the task is done", "action": "finish", "answer": "short summary"}}

Rules:
- Use only the tools listed above and only their declared arguments.
- Never invent a file path, a sha or a page number; read them from observations.
- If an observation reports an error, change your approach instead of repeating it.
- Finish as soon as the task is answered; do not call tools for decoration.
"""


@dataclass
class Step:
    index: int
    thought: str = ""
    action: str = ""
    args: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class AgentRun:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    stopped: str = "finish"
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.stopped == "finish" and bool(self.answer)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "stopped": self.stopped,
            "answer": self.answer,
            "steps": [
                {
                    "index": step.index,
                    "thought": step.thought,
                    "action": step.action,
                    "args": step.args,
                    "observation": step.observation,
                    "error": step.error,
                }
                for step in self.steps
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return target

    def trace_name(self) -> str:
        """Unique per run so two runs never overwrite each other's trace."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.task.lower())[:40].strip("-") or "task"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"{stamp}-{self.stopped}-{slug}.json"


def parse_action(raw: str) -> dict:
    """Parse one model reply into an action dict, repairing common damage."""
    cleaned = _FENCE.sub("", raw or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0:
        raise ValueError("no JSON object in response")
    candidate = cleaned[start : end + 1] if end > start else cleaned[start:]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = json.loads(_repair(candidate))
    if not isinstance(data, dict):
        raise ValueError("action must be a JSON object")
    if not str(data.get("action", "")).strip():
        raise ValueError("action is missing")
    args = data.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    return {
        "thought": str(data.get("thought", ""))[:400],
        "action": str(data["action"]).strip(),
        "args": args,
        "answer": str(data.get("answer", ""))[:2000],
    }


def _repair(text: str) -> str:
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _signature(action: str, args: dict) -> str:
    return f"{action}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


class Agent:
    """Runs the loop. Code owns every stop condition."""

    def __init__(
        self,
        client: LlmClient,
        tools: dict[str, Tool],
        settings: AgentSettings | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.settings = settings or AgentSettings()

    def _prompt(self, run: AgentRun) -> str:
        history = []
        for step in run.steps:
            history.append(
                json.dumps(
                    {
                        "step": step.index,
                        "thought": step.thought,
                        "action": step.action,
                        "args": step.args,
                        "observation": step.observation,
                        "error": step.error,
                    },
                    ensure_ascii=False,
                )
            )
        return (
            SYSTEM.format(catalog=catalog(self.tools))
            + f"\nTask: {run.task}\n\nHistory:\n"
            + ("\n".join(history) if history else "(empty)")
            + "\n\nNext action as strict JSON:"
        )

    def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task, model=getattr(self.client, "model", ""))
        seen: set[str] = set()
        consecutive_errors = 0

        for index in range(1, self.settings.max_steps + 1):
            try:
                raw = self.client.complete(self._prompt(run))
            except VisionError as exc:
                run.stopped = "llm_error"
                run.steps.append(
                    Step(index=index, error=f"llm unavailable: {str(exc)[:200]}")
                )
                return run

            try:
                action = parse_action(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                consecutive_errors += 1
                run.steps.append(
                    Step(index=index, error=f"invalid action JSON: {str(exc)[:160]}")
                )
                if consecutive_errors >= self.settings.max_consecutive_errors:
                    run.stopped = "invalid_actions"
                    return run
                continue

            if action["action"] == "finish":
                run.answer = action["answer"]
                run.steps.append(
                    Step(index=index, thought=action["thought"], action="finish")
                )
                run.stopped = "finish"
                return run

            tool = self.tools.get(action["action"])
            if tool is None:
                consecutive_errors += 1
                run.steps.append(
                    Step(
                        index=index,
                        thought=action["thought"],
                        action=action["action"],
                        args=action["args"],
                        error=f"unknown tool: {action['action']}",
                    )
                )
                if consecutive_errors >= self.settings.max_consecutive_errors:
                    run.stopped = "unknown_tools"
                    return run
                continue

            signature = _signature(action["action"], action["args"])
            if signature in seen:
                run.steps.append(
                    Step(
                        index=index,
                        thought=action["thought"],
                        action=action["action"],
                        args=action["args"],
                        error="repeated action with identical arguments",
                    )
                )
                run.stopped = "repeated_action"
                return run
            seen.add(signature)

            observation = call_tool(tool, action["args"])
            step = Step(
                index=index,
                thought=action["thought"],
                action=action["action"],
                args=action["args"],
                observation=observation,
            )
            if not observation.get("ok", False):
                step.error = str(observation.get("error", ""))[:300]
                consecutive_errors += 1
            else:
                consecutive_errors = 0
            run.steps.append(step)

            if consecutive_errors >= self.settings.max_consecutive_errors:
                run.stopped = "tool_errors"
                return run

        run.stopped = "step_budget"
        return run
