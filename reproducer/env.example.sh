# Template for local LLM settings. Committed: NEVER put a real key in this file.
#   cp env.example.sh env.local.sh && chmod 600 env.local.sh   # env.local.sh is gitignored
#   edit env.local.sh, then in each new terminal:  source env.local.sh   (not `bash env.local.sh`)

# OpenRouter API key: paste it only into env.local.sh. Empty -> make_llm raises a clear error.
export OPENROUTER_API_KEY=""
# JSON mode sent to OpenRouter: schema (Claude supports structured outputs) | object | none.
export OPENROUTER_JSON_MODE=schema

# Default backend for every node without its own override: local Ollama.
export REPRODUCER_LLM=ollama:qwen2.5-coder:7b
# Local Ollama server used by the ollama: backend.
export OLLAMA_HOST=http://localhost:11434
# Context window sent to Ollama on every call (tokens). Calls whose estimated size
# (prompt chars / 3.5 + max_tokens) exceeds it fail fast with ContextTooLong.
export OLLAMA_NUM_CTX=8192

# Per-node overrides (REPRODUCER_LLM_<NODE>): Claude via OpenRouter. Model ids are
# OpenRouter's ("anthropic/claude-haiku-5.5"), not Anthropic's ("claude-haiku-5-5").
# Development (pipeline testing, many runs): Haiku 5.5, $0.10 / $0.50 per 1M in/out.
# Final eval runs (M5): switch critic + orchestrator to anthropic/claude-sonnet-5.5
# ($2 / $10). Opus is not used: its output tokens cost ~40x Haiku's for this work.
# Critic (M2): one patch <= 20 lines from a traceback.
export REPRODUCER_LLM_CRITIC=openrouter:anthropic/claude-haiku-5.5
# Orchestrator (M3): plans runs from paper + repo evidence.
export REPRODUCER_LLM_ORCHESTRATOR=openrouter:anthropic/claude-haiku-5.5

# Model for `pytest -m llm_live -k openrouter` (one small paid call).
export REPRODUCER_LIVE_OPENROUTER_MODEL=anthropic/claude-haiku-5.5

# LangSmith tracing (pulled in by langgraph) stays OFF: we may process unpublished repos,
# so prompts and repo text must never be sent to an external tracing service.
export LANGSMITH_TRACING=false
