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

## MCP server (team assistant)
`reproducer/server.py` exposes the Reproducer over MCP (stdio; mcp SDK 2.x `MCPServer`, the
renamed FastMCP). Every tool shells out to the CLI, so the server holds no pipeline logic.

Register it in an MCP host. The command sources `env.local.sh` first (LLM keys, caps) so
no secret goes into the host config; `cwd` must be this folder:
```json
{
  "mcpServers": {
    "reproducer": {
      "command": "bash",
      "args": ["-c", "cd /path/to/reproducer && source env.local.sh && exec .venv/bin/python -m reproducer.server"]
    }
  }
}
```
Claude Code: `claude mcp add reproducer -- bash -c 'cd /path/to/reproducer && source env.local.sh && exec .venv/bin/python -m reproducer.server'`.
If the host passes env itself, the plain command is `.venv/bin/python -m reproducer.server`
with the variables of `env.local.sh` in the host's `env` map. Smoke test:
`npx @modelcontextprotocol/inspector .venv/bin/python -m reproducer.server`.

Tools (model-controlled):
- `reproduce_setup(spec_path, out_name)`: `reproduce --until plan` (configure, env, data,
  smoke, plan) as a detached background job.
- `start_run(out_name, claim_id=None, spec_path=None)`: `reproduce` (runs + reflexion +
  vault) as a detached background job; the spec is found from the store unless given. With
  `claim_id`: `reproduce --claim` (every claim planned; only that one runs, is evaluated and
  reflected on; the vault is still written).
- Both jobs log to `runs/<out>/mcp_jobs/<stamp>-<kind>.log` (last line `[job] exit <code>`)
  and return after waiting up to ~5 s for the job to hold `run.lock`: `started`,
  `already running`, `failed to start` (exited early; exit code + log tail), `finished`, or
  `starting` (alive, lock not taken yet).
- `status(out_name)`: read-only: running (lock holder alive), the newest job (stage = its
  last `[node]` line, exit code once finished), current plans, per-claim status / fidelity /
  value, breaker, pending approvals, open unknowns.

Resources (read-only): `reproducer://{out_name}/ledger`, `/unknowns`, `/queue`, `/lessons`,
`/paper_note` (the vault Papers note, rendered without writing the vault).

Prompt (user-controlled): `reproduce_paper(spec_path, out_name)`: the workflow above.

Paths: a relative `spec_path` (from this folder) must resolve under `evals/`; an absolute one
must resolve, symlinks included, under `$REPRODUCER_SPEC_ROOT` (refused when unset). No
`..`. `out_name` is a slug `[a-z0-9_-]` (1-64 chars) mapped to `runs/<out_name>`.

CLI equivalent of `start_run(claim_id=...)`:
`.venv/bin/python -m reproducer.cli reproduce --spec evals/cutout/spec.json --out runs/cutout --claim <claim_id>`.

**Approvals are human-only (Tier 2).** `approve`, `reject` and `answer` are NOT exposed over
MCP: the assistant can read the `queue` resource and show it, but a human runs the printed
`.venv/bin/python -m reproducer.cli approve|reject|answer ...` commands, then `start_run`
again (it resumes from the store).
