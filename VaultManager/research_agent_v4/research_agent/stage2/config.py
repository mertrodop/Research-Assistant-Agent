from pydantic import Field
from ..schemas.scientific import Record
from ..core.config import ModelConfig

class Stage2Config(Record):
    planner: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=16000,temperature=0.2,extra_body={'reasoning':{'effort':'high'}}))
    executor: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=6000,temperature=0.2,extra_body={'reasoning':{'enabled':False}}))
    verifier: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b',max_tokens=6000,temperature=0.0,extra_body={'reasoning':{'enabled':False}}))
    # Optional distinct reasoning configuration for the question answerer.
    answerer: ModelConfig | None = None
    max_rounds: int = Field(default=10,ge=1)
    max_tasks_per_round: int = Field(default=3,ge=1)
    max_task_attempts: int = Field(default=30,ge=1)
    max_no_progress: int = Field(default=3,ge=1) # rounds, not individual tasks
    max_llm_calls: int = Field(default=100,ge=1)
    max_reported_tokens: int = Field(default=300000,ge=1)
    max_output_repairs: int = Field(default=1,ge=0)
    max_context_chars: int = Field(default=40000,ge=1000)
    max_planner_chars: int = Field(default=110000,ge=1000)
    max_evaluation_chars: int = Field(default=180000,ge=1000)
    recent_episodes: int = Field(default=6,ge=1)
    allow_partial_stage1: bool = False
