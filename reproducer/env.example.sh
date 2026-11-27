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
# Ollama API: openai (default: OpenAI-compatible {OLLAMA_HOST}/v1) | native (/api/chat).
export REPRODUCER_OLLAMA_API=openai
# Client-side context guard (tokens): calls whose estimated size (prompt chars / 3.5 +
# max_tokens) exceeds it fail fast with ContextTooLong, and the node evicts context and
# retries. Via /v1 the server window is OLLAMA_CONTEXT_LENGTH on the Ollama service (keep
# both equal; the server silently truncates longer prompts). The native API sends it as num_ctx.
export OLLAMA_NUM_CTX=8192

# Per-node overrides (REPRODUCER_LLM_<NODE>): Claude via OpenRouter. Model ids are
# OpenRouter's ("anthropic/claude-haiku-5.5"), not Anthropic's ("claude-haiku-5-5").
# Development (pipeline testing, many runs): Haiku 5.5, $0.10 / $0.50 per 1M in/out.
# Final eval runs (M5): switch critic + orchestrator to anthropic/claude-sonnet-5.5
# ($2 / $10). Opus is not used: its output tokens cost ~40x Haiku's for this work.
# Critic (M2): one patch <= 20 lines from a traceback.
export REPRODUCER_LLM_CRITIC=openrouter:anthropic/claude-haiku-5.5
# Critic mode: react (bounded ReAct loop with read-only repo tools, default) | single (one
# gated call). react falls back to single once when the model has no tool support.
export REPRODUCER_CRITIC_MODE=react
# Orchestrator (M3): plans runs from paper + repo evidence.
export REPRODUCER_LLM_ORCHESTRATOR=openrouter:anthropic/claude-haiku-5.5
# Reflect (M4): one Reflexion lesson per out-of-tolerance / diverged trial. Rare (at most a few
# calls per claim) and judgment-heavy (localize the gap, ground one change), so a stronger
# model than the local default is worth its small cost.
export REPRODUCER_LLM_REFLECT=openrouter:anthropic/claude-haiku-5.5

# Reflexion loop limits (breaker, ledger, vault notes and CLI read them).
# Paper token budget: LLM calls recorded in the store count against it.
export REPRODUCER_TOKEN_BUDGET=200000
# Trials (full runs + re-evaluations) per claim; at the limit a claim stays open.
export REPRODUCER_MAX_TRIALS=3

# Model for `pytest -m llm_live -k openrouter` (one small paid call).
export REPRODUCER_LIVE_OPENROUTER_MODEL=anthropic/claude-haiku-5.5

# LangSmith tracing (pulled in by langgraph) stays OFF: we may process unpublished repos,
# so prompts and repo text must never be sent to an external tracing service.
export LANGSMITH_TRACING=false
