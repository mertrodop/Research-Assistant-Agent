# reproducer

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
```

## LLM settings (local, not committed)
1. `cp env.example.sh env.local.sh && chmod 600 env.local.sh` (env.local.sh is gitignored).
2. Edit env.local.sh: paste your OPENROUTER_API_KEY and pick model ids.
3. Run `source env.local.sh` in each new terminal before running the CLI or live tests.

## Test
```bash
.venv/bin/pytest -q
```

## Lint
```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```
