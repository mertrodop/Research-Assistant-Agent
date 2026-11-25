"""Episodic (past runs) and procedural (saved how-tos) memory."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Episode:
    task: str
    paper: str
    started: str
    steps: list[dict] = field(default_factory=list)
    outcome: Literal["finish", "fallback"] = "finish"
    failure: str = ""
    evicted: list[str] = field(default_factory=list)


class EpisodicMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, e: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def _load(self) -> list[Episode]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                out.append(Episode(**json.loads(line)))
            except (ValueError, TypeError):  # corrupt line: skip
                continue
        return out

    def recall(self, paper: str, limit: int = 3) -> list[Episode]:
        """Same paper first, then the most recent episodes that had a failure."""
        eps = self._load()[::-1]  # newest first
        same = [e for e in eps if e.paper == paper]
        failed = [e for e in eps if e.paper != paper and e.failure]
        return (same + failed)[:limit]


class ProceduralMemory:
    def __init__(self, dir: str | Path):
        self.dir = Path(dir)

    def all(self) -> list[tuple[str, str]]:
        if not self.dir.is_dir():
            return []
        return [(p.stem, p.read_text(encoding="utf-8", errors="ignore").strip())
                for p in sorted(self.dir.glob("*.md"))]

    def render(self, max_chars: int = 1500) -> str:
        """Join procedures, cutting at a procedure boundary when possible."""
        blocks = [f"- {name}: {text}" for name, text in self.all()]
        out = ""
        for b in blocks:
            if len(out) + len(b) + 1 > max_chars:
                break
            out += b + "\n"
        if not out and blocks:  # first one alone is too long: hard cut
            out = blocks[0][:max_chars]
        return out.strip()


def memory_prompt(episodes: list[Episode], procedures: list[tuple[str, str]]) -> str:
    lines = ["Past episodes:"]
    lines += [f"- {e.outcome}: {e.failure or 'no failure'}" for e in episodes] or ["- none"]
    lines.append("Procedures:")
    lines += [f"- {n}: {t}" for n, t in procedures] or ["- none"]
    return "\n".join(lines)
