# Research-Assistant-Agent

An agent that turns research PDFs into checked notes in an Obsidian vault. A language
model decides what to do next, code checks every step, and a human approves every write.

```
ra ingest paper.pdf
  -> agent (LangGraph ReAct) -> guard -> tools over MCP -> agent ...
                                  |
                      write tools wait for your approval
```

Nothing is written to `vault/` or `memory/procedures/` without your approval.

## Install

```bash
pip install --user -e ".[dev]"
echo "OPEN_ROUTER_API_KEY=..." > .env      # never committed
```

The default vision and agent model is Qwen3-VL-8B (`qwen/qwen3-vl-8b-instruct`) on
OpenRouter. Any OpenAI-compatible endpoint works, a local one too (vLLM, Ollama, LM Studio):
set `base_url`, `model` and `api_key_env` under `models.agent` and `models.vision` in
`config/pipeline.yaml`. A role you leave out keeps its default.

## Use

```bash
ra papers                                   # PDFs in data/raw
ra ingest paperbench-evaluating-ais-ability-to-replicate-ai-research.pdf
ra agent "Which ingested papers mention agents?"
ra agent "Re-check the references of this paper" --paper core-bench-fostering-....pdf
ra serve                                    # the same tools as an MCP server on stdio
```

Options for `ingest` and `agent`: `--config FILE`, `--max-steps N`, `--yes-to-reads`.
A bare PDF name is looked up in `data/raw`. Exit code: `0` when the agent finished,
`1` when a limit stopped it and the deterministic fallback ran, `2` when no API key is set.
The run ends with the answer (or the failure sentence) and the path of the note.

## What the agent decides

The tools are the methods of `Workspace` in `src/ra_agent/tools.py`, served over MCP.

| Kind | Tools | Approval |
|---|---|---|
| read | `list_papers`, `vault_search`, `vault_read` | no |
| compute | `ingest_paper`, `reprocess_page`, `describe_figure`, `check_references`, `critique_page`, `stage_note` | no |
| write | `commit_note`, `save_procedure` | always |

`ingest_paper` runs the OCR pipeline and returns a list of `problems` (a page that failed
the gate, a figure number the page text does not corroborate, an unresolved reference).
The agent chooses per problem: reprocess the page with another engine, describe the figure
again, accept it, or finish and explain. It stages the note, and only `commit_note` moves
it into the vault.

## Limits

Code owns the stop conditions; the model only proposes. Values live under `agent:` in
`config/pipeline.yaml`.

| Limit | Default | When it trips |
|---|---|---|
| `max_steps` | 12 | the agent used all its steps |
| `max_consecutive_errors` | 2 | two tool errors in a row |
| `max_invalid_calls` | 2 | two calls with unknown tools or bad arguments |
| repeated action | fixed | the same tool with the same arguments twice |
| `context_budget` | 6000 | old tool results are evicted, the last `keep_last_turns` (4) stay |

When a limit trips, the run ends with a one-sentence failure report, the fallback stages a
deterministic `needs-review` note (never committed), and the episode is logged.

## Memory

| Memory | Where | Written by |
|---|---|---|
| Working | the message list, trimmed to `context_budget` | code |
| Episodic | `memory/episodes.jsonl`: task, steps, outcome, failure | code, after every run |
| Procedural | `memory/procedures/*.md`: lessons the agent saved | `save_procedure`, with approval |

Recent episodes for the same paper and all saved procedures go into the system prompt.

## Approval

For every write the CLI prints the tool, its arguments and the diff against the vault, then asks:

```
[a]pprove, [e]dit, [r]eject?
```

`e` opens the staged note in `$EDITOR` (default `nano`). If the frontmatter is not valid
the error is shown and the editor opens again. After a valid edit you see the new diff and
are asked again. A rejected write goes back to the agent as "rejected by human".

## OCR pipeline

`ingest_paper` runs: render and route each page (text layer or Granite-Docling-258M),
assemble, extract and describe figures, run 11 deterministic gate checks per page, retry a
failing page up the DPI ladder (144, 216, 288; at most 3 attempts), resolve references
against arXiv and Crossref, then check the document identity. Code gates; the language model
proposes. Modules are in `src/ra_agent/ocr/`, notes in `src/ra_agent/vault/`.

## Layout

```
data/raw/     source PDFs, never modified
staging/      notes and OCR state waiting for approval (git-ignored)
vault/        approved notes, vault/Papers/<slug>.md (git-ignored)
memory/       episodes and procedures (git-ignored)
```

## Evaluation

```bash
python eval/run_eval.py gate            # Tier 1 gate fault injection, offline
python eval/run_eval.py critic --n 4    # rubric critic, needs an API key
python eval/run_eval.py granite --n 3   # Granite-Docling against the text layer
python eval/run_eval.py document        # references and metadata, needs network
```

Results land in `eval/results/<part>.json`.

| Experiment | Result |
|---|---|
| Agent evaluation | filled in Task 11 |

## Tests

```bash
/usr/bin/python3 -m pytest -q                 # unit tests, no network, no model calls
/usr/bin/python3 -m pytest -q -m integration  # spawns the MCP server over stdio
```
