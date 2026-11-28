from typing import Literal
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model: str
    max_tokens: int = Field(gt=0)
    temperature: float | None = 0
    response_mode: Literal['json_schema','json_object'] = 'json_schema'
    extra_body: dict = Field(default_factory=dict)

class Stage1Config(BaseModel):
    reader: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b', max_tokens=2800))
    checker: ModelConfig = Field(default_factory=lambda: ModelConfig(model='qwen/qwen3.8-27b', max_tokens=4500))
    max_facts_per_unit: int = Field(default=30, gt=0)
    max_questions_per_unit: int = Field(default=5, ge=0)
    max_chars_per_unit: int = Field(default=9000, gt=0)
    include_references: bool = False
    include_back_matter: bool = True
    reuse_reader_cache: bool = True
