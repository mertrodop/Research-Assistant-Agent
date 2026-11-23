from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RenderSettings(BaseModel):
    dpi: int = 200
    high_dpi: int = 300
    max_dpi: int = 400
    ink_threshold: int = 200


class RouterSettings(BaseModel):
    native_text_min_chars: int = 200
    ink_ratio_vlm: float = 0.02


class EngineSettings(BaseModel):
    name: Literal["granite_docling", "native"] = "granite_docling"
    model_id: str = "ibm-granite/granite-docling-258M"
    device: str = "auto"
    page_batch_size: int = 1


class GateSettings(BaseModel):
    min_chars: int = 40
    min_printable_ratio: float = 0.95
    max_ngram_repeat: int = 6
    ngram_size: int = 8
    min_coverage: float = 0.35
    max_coverage: float = 3.0
    numeric_tolerance: float = 0.0
    sparse_ink: float = 0.02


class PipelineSettings(BaseModel):
    concurrency: int = 2
    max_attempts: int = 3
    cache_dir: str = ".cache/ocr"
    vault_dir: str = "vault"


class DataSettings(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class FigureSettings(BaseModel):
    enabled: bool = True
    min_side: int = 100
    min_area_ratio: float = 0.01
    max_caption_gap: float = 140.0
    page_render_fallback: bool = True
    render_dpi: int = 150


class VisionSettings(BaseModel):
    """OpenAI-compatible vision endpoint. Off by default: it costs tokens."""

    enabled: bool = False
    model: str = "google/gemini-2.5-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPEN_ROUTER_API_KEY"
    timeout_s: float = 90.0
    max_tokens: int = 700
    temperature: float = 0.0
    concurrency: int = 2
    cache_dir: str = ".cache/figures"
    max_image_side: int = 1400


class CriticSettings(BaseModel):
    """Tier 2 rubric critic. Off by default: it needs an LLM endpoint."""

    enabled: bool = False
    min_dimension: int = 3
    min_average: float = 3.5


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


class AgentSettings(BaseModel):
    """Tool-calling loop. Code owns the stop conditions, the model only proposes."""

    enabled: bool = False
    max_steps: int = 8
    max_consecutive_errors: int = 2
    trace_dir: str = "data/processed/traces"


class Settings(BaseModel):
    render: RenderSettings = Field(default_factory=RenderSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    figures: FigureSettings = Field(default_factory=FigureSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    critic: CriticSettings = Field(default_factory=CriticSettings)
    crossref: CrossrefSettings = Field(default_factory=CrossrefSettings)
    references: ReferenceSettings = Field(default_factory=ReferenceSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        if path is None:
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
