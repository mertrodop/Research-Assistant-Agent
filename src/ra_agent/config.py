from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class RenderSettings(BaseModel):
    dpi: int = 144
    high_dpi: int = 216
    max_dpi: int = 288
    ink_threshold: int = 200


class RouterSettings(BaseModel):
    native_text_min_chars: int = 200


class EngineSettings(BaseModel):
    device: str = "cpu"  # auto OOMs on small GPUs


class GateSettings(BaseModel):
    min_chars: int = 40
    min_printable_ratio: float = 0.95
    max_ngram_repeat: int = 6
    ngram_size: int = 8
    min_coverage: float = 0.35
    max_coverage: float = 3.0
    sparse_ink: float = 0.02


class PipelineSettings(BaseModel):
    concurrency: int = 2
    max_attempts: int = 3
    cache_dir: str = ".cache/ocr"


class PathSettings(BaseModel):
    inbox_dir: str = "vault/Inbox"  # new PDFs; commit moves one to Papers/<slug>/raw/
    vault_dir: str = "vault"
    staging_dir: str = "staging"
    memory_dir: str = "memory"

    def list_pdfs(self) -> list[Path]:
        """PDFs the tools may read: the Inbox and each paper's raw/ folder."""
        found = [*Path(self.inbox_dir).glob("*.pdf"), *Path(self.vault_dir, "Papers").glob("*/raw/*.pdf")]
        return sorted(found)

    def find_pdf(self, name: str) -> Path:
        """A bare file name: the Inbox first, then a paper's raw/ folder."""
        found = next((p for p in self.list_pdfs() if p.name == name), None)
        if found is None:
            raise FileNotFoundError(f"no PDF {name!r} in {self.inbox_dir} or in {Path(self.vault_dir, 'Papers')}/*/raw")
        return found


class FigureSettings(BaseModel):
    enabled: bool = True
    min_side: int = 100
    min_area_ratio: float = 0.01
    max_caption_gap: float = 140.0
    page_render_fallback: bool = True
    render_dpi: int = 150
    describe: bool = True  # interpret figures with the vision model


class ModelSettings(BaseModel):
    """One OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, LM Studio)."""

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3-vl-8b-instruct"
    api_key_env: str = "OPEN_ROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2000  # reasoning models spend part of it before answering
    timeout_s: float = 90.0


VERIFIER = ModelSettings(model="google/gemini-2.5-flash:online")  # web-searching fallback for what the record APIs lack


class VisionExtra(BaseModel):
    cache_dir: str = ".cache/figures"
    concurrency: int = 2


class CriticSettings(BaseModel):
    """Tier 2 rubric critic. Off by default: it needs an LLM endpoint."""

    enabled: bool = False
    min_dimension: int = 3
    min_average: float = 3.5
    dpi: int = 110  # page image shown to the critic; readable, few tokens


class CrossrefSettings(BaseModel):
    enabled: bool = False
    cache_dir: str = ".cache/crossref"
    timeout_s: float = 20.0
    mailto: str = ""
    min_score: float = 0.82
    year_tolerance: int = 1


class ReferenceSettings(BaseModel):
    """Agentic reference resolution: strategy ladder plus circuit breaker."""

    max_attempts: int = 4
    use_llm: bool = False
    mismatch_floor: float = 0.5


class MetaSettings(BaseModel):
    """Document evaluator-optimizer. Runs with crossref: it needs external records."""

    max_iter: int = 3
    min_title_score: float = 0.9
    min_author_overlap: float = 0.6
    max_unresolved_refs: float = 0.3


class AgentSettings(BaseModel):
    """Agent loop limits. Code owns the stop conditions, the model only proposes."""

    max_steps: int = 12
    max_consecutive_errors: int = 2
    max_invalid_calls: int = 2
    context_budget: int = 6000
    keep_last_turns: int = 4


class Settings(BaseModel):
    render: RenderSettings = Field(default_factory=RenderSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    figures: FigureSettings = Field(default_factory=FigureSettings)
    models: dict[str, ModelSettings] = Field(
        default_factory=lambda: {"agent": ModelSettings(), "vision": ModelSettings(), "verifier": VERIFIER}
    )
    vision_extra: VisionExtra = Field(default_factory=VisionExtra)
    critic: CriticSettings = Field(default_factory=CriticSettings)
    crossref: CrossrefSettings = Field(default_factory=CrossrefSettings)
    references: ReferenceSettings = Field(default_factory=ReferenceSettings)
    meta: MetaSettings = Field(default_factory=MetaSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @field_validator("models")
    @classmethod
    def _fill_roles(cls, models: dict[str, ModelSettings]) -> dict[str, ModelSettings]:
        """A YAML that sets one role keeps the code defaults for the others."""
        return {"agent": ModelSettings(), "vision": ModelSettings(), "verifier": VERIFIER} | models

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        if path is None:  # the repo's config/pipeline.yaml (found from this file, not cwd), else code defaults
            path = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
            if not path.is_file():
                return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


def load_env_file(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment wins."""
    import os

    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
