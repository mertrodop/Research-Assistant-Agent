# Research-Assistant-Agent

Research PDFs in, checked notes in an Obsidian vault out. A fixed OCR pipeline does the work;
an agent steps in only when problems are left, code checks every step and a human approves
every write.

```
ra ingest paper.pdf
  -> OCR pipeline (engines, gate, vision-LLM rereads, figures, references, metadata)
  -> problems left?  yes -> agent (LangGraph ReAct, tools over MCP: reprocess / describe / check refs)
  -> status verified? yes -> diff -> your approval -> commit to vault
                      no  -> stays in staging as needs-review
```

Nothing is written to `vault/` without your approval.

## Install

```bash
pip install --user -e ".[dev]"
echo "OPEN_ROUTER_API_KEY=..." > .env      # never committed
```

The default vision and agent model is Qwen3-VL-8B (`qwen/qwen3-vl-8b-instruct`) on
OpenRouter. Any OpenAI-compatible endpoint works, a local one too (vLLM, Ollama, LM Studio):
set `base_url`, `model` and `api_key_env` under `models.agent` and `models.vision` in
`config/pipeline.yaml`. It is read by default (found from the repo root, whatever the
current directory); `--config FILE` picks another. A role you leave out keeps its default,
and without the file the code defaults apply.

## Use

```bash
ra papers                                   # PDFs in vault/Inbox and vault/Papers/*/raw
ra ingest my-paper.pdf                      # pipeline first, the agent only for what is left; drop the PDF into vault/Inbox first
ra agent "Which ingested papers mention agents?"
ra agent "Re-check the references of this paper" --paper core-bench-fostering-the-credibility-of-published-research-through-a-computational-reproducibility.pdf
ra serve                                    # the same tools as an MCP server on stdio
```

Options for `ingest` and `agent`: `--config FILE`, `--max-steps N`.
A bare PDF name is looked up in `vault/Inbox`, then in `vault/Papers/*/raw` (re-ingest). Exit codes of `ingest`: `0` committed, `1` left in staging (needs-review, or you rejected it), `2` the PDF could not be ingested.
`ra agent` keeps its own codes: `0` finished, `1` a limit stopped it and the deterministic fallback ran, `2` no API key.

## What the agent decides

The tools are the methods of `Workspace` in `src/ra_agent/tools.py`, served over MCP.

| Kind | Tools | Approval |
|---|---|---|
| read | `list_papers`, `vault_search` | no |
| compute | `ingest_paper`, `reprocess_page`, `describe_figure`, `check_references`, `stage_note` | no |
| write | `commit_note` | always |

`ingest_paper` runs the OCR pipeline and returns `problems` (page blockers only: a page
that failed a blocking gate check) and `warnings` (every other failed check, figure grounding
included). Reference results are not part of its output; `check_references` reports them.
After `ra ingest` the agent only sees `reprocess_page`, `describe_figure` and `check_references`
and chooses per problem: reprocess the page with another engine, describe again, accept, or
finish and explain. Staging and `commit_note` stay with the CLI and your approval. A free
`ra agent "..."` task gets every tool.

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
deterministic `needs-review` note (never committed).

## Approval

For every write the CLI prints the tool, its arguments and the diff against the vault, then asks:

```
[a]pprove, [e]dit, [r]eject?
```

`e` opens the staged note in `$EDITOR` (default `nano`). If the frontmatter is not valid
the error is shown and the editor opens again. After a valid edit you see the new diff and
are asked again. A rejected write goes back to the agent as "rejected by human".

## OCR pipeline

`ingest_paper` runs: render and route each page (text pages: Docling, then
Granite-Docling-258M, then the text layer; low-text and scanned pages: the vision model, then Granite
at 216 and 288 dpi (low-text pages end on the text layer, scans on Granite); without an
API key the vision rung is skipped and Granite runs at 144, 216 and 288 dpi). A page under
200 text-layer characters carries a `no_text_layer` warning: it cannot be checked against the PDF. Then
assemble, extract and describe figures, run 13 deterministic gate checks per page, resolve
references against arXiv and Crossref, then check the document identity. A Docling table that
still fails its table checks after row repair is read again by the vision model; the reading
replaces the table only if every word and number is in the PDF text layer (check marks cannot
be grounded and get a warning callout), otherwise Docling's table and its blocker stay. Code gates; the language model
proposes. Modules are in `src/ra_agent/ocr/`, notes in `src/ra_agent/vault/`.

## Layout

```
vault/Inbox/                 new PDFs wait here (tracked via .gitkeep)
vault/Papers/<slug>/raw/<slug>.pdf     the source PDF, moved here by commit_note
vault/Papers/<slug>/md/<slug>.md       the note; figure links are ../figures/fig-N.png
vault/Papers/<slug>/figures/fig-N.png
staging/      notes and OCR state waiting for approval, same layout (git-ignored)
eval/data/    generated eval fixtures: scanned copies, injection PDFs (git-ignored)
```

## Evaluation

```bash
python eval/run_eval.py gate            # Tier 1 gate fault injection, offline
python eval/run_eval.py ingest          # engines, full ingest, unit tests on the notes
```

Results land in `eval/results/<part>.json`. Dev set: two 30-page arXiv papers
(CORE-Bench, PaperBench). Vision and agent model: Qwen3-VL-8B on OpenRouter.

| Experiment | Result |
|---|---|
| Gate, 59 text pages x 10 injected faults | 8 of 10 fault types blocked on every page; dropped numbers blocked on 8.5% (flagged on all); a hallucinated sentence is never blocked. Clean pages: 0% falsely blocked |
| Engines vs text layer (6 pages) | docling WER 0.089, 2.4 s/page; Granite-Docling WER 0.081, 215 s/page on CPU, 2 pages gate-blocked |
| Scanned pages through Granite | bag-of-words F1 0.94-0.98 on 5 of 6 pages; one page (PaperBench p4) bad and gate-blocked |
| Figure models (6 figures) | Qwen3-VL-8B and Gemini 2.5 Flash: 6/6 valid JSON, 35% of reported numbers grounded in the page; Gemma free tier: 0/6, rate-limited (HTTP 429) |
| References and identity | CORE-Bench 29/55 references verified, PaperBench 34/44; both identities verified against arXiv |
| Agent scenarios and end-to-end run | measured with `eval/agent_eval.py`, removed later (git history). Fault-injected pages: the fixed workflow and the agent both recovered 10/10, all through the code ladder, so the agent adds nothing to page recovery; that is why `ra ingest` now runs the pipeline first and calls the agent only for what is left |

The scanned-page and figure-model rows were measured with eval parts that were removed later (see git history). The rubric critic was removed too: it blocked at most 1 of 6 injected faults.

Weak spots, plainly:
- The free Gemma model is rate-limited (HTTP 429).
- The identity check accepted a Crossref match ("Taking Research Notes", 2003) for an
  invented one-page test paper.

## Tests

```bash
/usr/bin/python3 -m pytest -q                 # unit tests, no network, no model calls
/usr/bin/python3 -m pytest -q -m integration  # spawns the MCP server over stdio
```
