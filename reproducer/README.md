# reproducer

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
```

## Test
```bash
.venv/bin/pytest -q
```

## Lint
```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```
