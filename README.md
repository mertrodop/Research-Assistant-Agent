# Research-Assistant-Agent

Agentic research assistant. This repository currently contains the **OCR pipeline**:
the path a new paper takes into the vault, processing pages in parallel, verifying every
element, and retrying only the pages that fail.

The design from slide 6 is implemented directly:

```
parallel pages -> routing -> code gate -> evaluator/optimizer
Tier 1: deterministic unit tests   Tier 2: rubric critic + Crossref
circuit breaker: <= 3 attempts     output: verified note | #needs-review
```

## Install

```bash
pip install --user -e ".[dev]"        # core + tests
pip install --user -e ".[vlm]"        # Granite-Docling-258M engine (docling)
```

Without `docling` the pipeline still runs with `--engine native` (text-layer PDFs only).

## Usage

```bash
ra batch                                   # data/raw -> staging
ra batch --engine native                   # fast path for text-layer PDFs
ra batch --figures-llm                     # interpret figures with a vision model
ra ingest data/raw/paperbench-evaluating-ais-ability-to-replicate-ai-research.pdf          # single PDF
ra ingest paper.pdf --pages 1-3,7          # page selection
ra ingest paper.pdf --device cpu           # for models that do not fit in 2 GB VRAM
ra ingest paper.pdf --vault ./vault        # Obsidian vault root
ra ingest paper.pdf --json                 # machine-readable output
ra ingest paper.pdf --no-figures           # disable figure extraction
```

Exit code: `0` when every page is verified, `1` when at least one page is `#needs-review`.

## Data layout

```
data/
  raw/          source PDFs, never modified
  processed/    OCR output, <sha256[:12]>.md
  processed/figures/<sha12>/   extracted figure PNGs
```

`raw/` is append-only: if a PDF changes its sha256 changes, so the cache and the output
file get a new name and the old record is never overwritten. `processed/` is fully
reproducible: delete it and rerun `ra batch`. Details: `data/README.md`.

The repository ships two real papers: `paperbench-evaluating-ais-ability-to-replicate-ai-research.pdf` (PaperBench) and
`core-bench-...-reproducibility.pdf` (CORE-Bench), both cited in the motivation on slide 2.

Every output carries YAML frontmatter (`sha256`, `ocr_engine`, `pages`, `verified_pages`,
`status`), a markdown body, and a per-page `## Validation` table. Use `--vault` for the
Obsidian layout: `vault/Resources/Parsed/<sha12>.md`.

## Pipeline

| Stage | Module | Who decides |
|---|---|---|
| Render + stats | `ocr/render.py` | code (PyMuPDF) |
| Routing | `ocr/router.py` | code: text layer >= 200 chars -> native, else VLM |
| OCR | `ocr/engine.py` | Granite-Docling-258M, per page, `temperature=0` |
| Assembly | `ocr/assemble.py` | code: repeated headers/footers, dictionary-backed dehyphenation |
| Figures | `ocr/figures.py` | code: raster extraction, caption matching, page render for vectors |
| Figure reading | `ocr/interpret.py` + `llm/client.py` | LLM proposes, code verifies (grounded critic) |
| Tier 1 gate | `ocr/gate.py` | code: 11 deterministic checks |
| References | `ocr/refagent.py` + `ocr/crossref.py` | code runs the loop, LLM only proposes identities |
| Tier 2 critic | `ocr/critic.py` | LLM lists errors from the page image, code verifies each quote |
| Retry / circuit breaker | `ocr/pipeline.py` | code: DPI ladder 144 -> 216 -> 288, <= 3 attempts |
| Document evaluator-optimizer | `ocr/synth.py` | code: draft identity, compare to arXiv/Crossref, apply fix list, <= 3 iterations |
| Note | `vault/note.py` | code |
| Agent loop | `agent/loop.py` + `agent/tools.py` | LLM proposes one action, code validates and stops |

Tier 1 checks: `non_empty`, `printable_ratio`, `repetition_loop`, `coverage`,
`heading_levels`, `table_columns`, `table_header`, `equation_delimiters`,
`doctags_balance`, `numeric_agreement`, `reference_fields`. A page with a failing
`blocker` is retried; a page with a failing `warn` still counts as verified but the reason
is recorded in the note.

`numeric_agreement` compares numbers as a multiset against the PDF text layer. A number
the text layer does not have, or has fewer times, is a `blocker`: that is how a misread
value (`13.2 ± 0.3` -> `0.9`), a dropped decimal point (`0.45` -> `0 45`) and a duplicated
table look. Missing numbers (margin stamps, page numbers) stay a `warn`.

`table_header` was added because of a real failure in this pipeline: Granite-Docling drops
the table header and promotes the first data row (`| 1 | 0.91 |` followed by the
separator). Losing header text is losing content, so it is a `blocker` and the page goes
back through the DPI ladder.

## Figures

Raster figures are found with `page.get_image_info(xrefs=True)`, written as PNG under
`processed/figures/<sha12>/pNNN_figK.png`, and placed in the markdown next to their
caption:

```markdown
![Figure 1](figures/p002_fig1.png)

Figure 1. PaperBench is a benchmark for evaluating AI agents' abilities ...
```

- Caption matching: the text block starting with `Figure N` / `Fig. N` closest to the
  figure box (within `max_caption_gap` points) wins; it may sit above or below.
- Filter: images smaller than `min_side` pixels or than `min_area_ratio` of the page area
  (icons, logos, rules) are skipped.
- Panels sharing one caption are numbered `Figure 4 (1/2)`, `Figure 4 (2/2)`.
- Vector figures carry no raster object; when a page has a caption but no raster, the page
  is rendered as `pNNN_page.png` (`kind: page_render`). This is an approximation and does
  not crop.
- Frontmatter records the `figures` count and the `figure_files` list.

## Figure interpretation (optional)

With `--figures-llm` every figure is sent to a vision model and appears in the note as a
blockquote:

```markdown
![Figure 1](figures/p002_fig1.png)

> **screenshot** — The image displays a sequence of thought processes ...
> - Experiments are run with 'python model_training.py --bs 32'.
> ⚠ not corroborated by page text: 32
```

The design follows the "code decides, LLM only proposes" principle from the slides:

| Step | Who decides |
|---|---|
| Prompt + strict JSON schema | code (`ocr/interpret.py`) |
| JSON parsing, field truncation | code, `pydantic` validation |
| **Grounded critic** | code: every number the model reports is looked up in caption + page text |
| `figure_grounding` check | code: `warn` when a number is not corroborated |
| Rendering in the note | code: the warning line sits directly under the quote |

The `ungrounded` label does not mean "hallucination", it means "not corroborated by the
page text": a number printed inside the chart (for example `--bs 32`) never appears in the
text, so it lands here. That is why it is a `warn`, not a `blocker`.

The client speaks the OpenAI-compatible `/chat/completions` protocol; the same code works
with OpenRouter, OpenAI, vLLM, Ollama and LM Studio. The key is read from the environment
or `.env` through `vision.api_key_env` (default `OPEN_ROUTER_API_KEY`) and never written
into the code.

```bash
ra ingest paper.pdf --figures-llm
ra batch --figures-llm --vision-model google/gemini-2.5-flash
ra ingest paper.pdf --figures-llm --vision-base-url http://localhost:11434/v1
```

Readings are cached under `.cache/figures/<sha12>/<figure>__<model>.json`, so the same
figure and model never costs a second API call.

Broken JSON is recovered in two stages: first the truncated object is closed and trailing
commas are removed, then the model gets one "compact JSON" retry. The permanent fix is
constrained decoding (`guided_json`), which prevents invalid JSON from appearing at all.

### Open-source model options

| Role | Model | License | 4-bit |
|---|---|---|---|
| OCR | `ibm-granite/granite-docling-258M` | Apache 2.0 | ~0.5 GB |
| Vision | `ibm-granite/granite-vision-3.1-2b` | Apache 2.0 | ~1.3 GB |
| Vision | `Qwen/Qwen2.5-VL-3B-Instruct` | Apache 2.0 | ~2 GB |
| Agentic | `ibm-granite/granite-4.0-micro` | Apache 2.0 | ~2 GB |
| Agentic | `Qwen/Qwen3-4B-Instruct-2507` | Apache 2.0 | ~2.5 GB |

Llama and Gemma licenses are not OSI-approved, so they are not open source. Determinism
needs `temperature=0`, a fixed `seed`, `batch_size=1`, the same quantization file and the
same runtime version.

```bash
vllm serve ibm-granite/granite-vision-3.1-2b --guided-json --seed 42
ra batch --figures-llm --vision-model ibm-granite/granite-vision-3.1-2b \
             --vision-base-url http://localhost:8000/v1
```

## Tier 2: rubric critic

A page that passes Tier 1 goes to a rubric critic when `--critic` is given. The critic sees
the rendered page image (`critic.dpi`) next to the OCR text, scores four dimensions 1-5
(`fidelity`, `completeness`, `structure`, `no_hallucination`) and returns a fix list:
`{"type": "hallucination | wrong_number | missing | structure", "ocr_span", "page_says"}`.

Code decides what counts:

- a `hallucination` or `wrong_number` error blocks the page only when its `ocr_span` is
  really in the OCR text (the critic cannot block a page with its own invention);
- when the page has a text layer, an error whose span is in the text layer is discarded:
  that text is on the page, so the claim is wrong (running headers, figure labels);
- a low score (`min_dimension`, `min_average`) also blocks.

If the critic is unreachable or returns broken JSON the check degrades to `warn` and the
run does not crash.

## Document evaluator-optimizer

With `--crossref`, the assembled document goes through the right half of slide 6:

```
draft identity (page 1) -> evaluator: OCR · Citation · DOI · Author · Venue <-> arXiv / Crossref
        ^                                   |
        +---- JSON fix list, iter < 3 ------+  -> verified note | #needs-review
```

| Check | Severity | Passes when |
|---|---|---|
| `identity` | blocker | an arXiv/Crossref record exists and the draft converges to it within `meta.max_iter` |
| `ocr_pages` | blocker | every page is verified |
| `citations` | warn | at most `meta.max_unresolved_refs` of the checked references are unresolved or mismatched |

The draft (title, authors, year, DOI, arXiv id) is read from page 1 without any model. A fix
is taken only from an external record, never invented; the loop repeats because a fixed field
can unlock a better lookup (an arXiv record that carries a journal DOI leads to the Crossref
record with the venue). The result goes into the note frontmatter (`authors`, `year`,
`venue`, `doi`, `arxiv_id`) and a `## Document checks` table.

## Agentic reference resolution

`--crossref` verifies bibliography entries against `api.crossref.org`. The first version
asked one bibliographic query per entry and gave up when the answer did not match, which
produced a high mismatch rate for reasons unrelated to fabricated citations: OCR merges
entries, titles get confused with author lists, and a single query is a weak probe.

`ocr/refagent.py` replaces that with the same loop the OCR pipeline uses: perceive, act,
evaluate, reflect, retry, stop at a circuit breaker.

| Stage | What happens | Who decides |
|---|---|---|
| Perceive | DOI and arXiv identifiers are extracted by regex; year from the entry | code |
| Act | One strategy per attempt: `doi` -> `arxiv` -> `title` -> `arxiv_title` -> `title_year` | code |
| Evaluate | Title similarity (`rapidfuzz`) plus year tolerance decides the status | code |
| Reflect | The failed attempt is appended to the trace with its score | code |
| Rescue | If the budget is not spent, the LLM proposes a title/DOI as strict JSON | LLM proposes, code validates |
| Stop | `max_attempts` network calls, then `unresolved` with the full trace | code |

`arxiv_title` exists because ML venues (PMLR, OpenReview) publish without DOIs, and a
Crossref title search then returns an unrelated record. arXiv is queried with at most one
call per 3 seconds, as its API terms ask.

Entry boundaries come from the PDF layout: the native engine inserts a blank line where a
hanging indent starts a new bibliography entry, so a wrapped author list
(`Margossian, C.,` / `Gower, R. M., ...`) stays one entry.

Strategies that cannot run are skipped without consuming budget, so a reference without a
DOI does not waste an attempt. Every check records its `strategy` and its `attempts` trace,
and the note shows both in the `## References check` table.

```bash
ra batch --crossref
ra batch --crossref --refs-llm --refs-attempts 5
ra ingest paper.pdf --crossref --json
```

Results are cached under `.cache/crossref`, so reruns stay offline.

## Agent loop

`ra agent` puts the LLM at the centre of the system as a tool-calling
orchestrator. The model proposes one action per step; code validates it, runs it, and
decides when to stop.

```
perceive -> propose -> validate -> act -> observe -> reflect -> repeat
```

```bash
ra agent "List the papers under data/raw and report the one with the most pages."
ra agent "Run the Tier 1 gate on page 1 of data/raw/paperbench-evaluating-ais-ability-to-replicate-ai-research.pdf." --engine native
ra agent "..." --max-steps 6 --json
```

The model answers with strict JSON, either a tool call or a finish:

```json
{"thought": "why this step", "action": "gate", "args": {"pdf": "data/raw/x.pdf", "page": 1}}
{"thought": "why the task is done", "action": "finish", "answer": "short summary"}
```

| Tool | Arguments | Returns |
|---|---|---|
| `list_papers` | - | PDFs under `data/raw` with sha, page count, size |
| `ocr` | `pdf`, `pages` | verification status, review pages, char count |
| `gate` | `pdf`, `page` | failing checks, blockers, warnings |
| `references` | `pdf`, `max_attempts` | Crossref status counts and unresolved entries |
| `read_note` | `sha` | processed note path and head |
| `write_report` | `name`, `body` | markdown report under `staging/reports` |

Every tool declares a pydantic argument model, so a malformed proposal is rejected by
code before anything runs. A tool never raises: failures come back as observations, which
is what lets the loop reflect and change approach.

Circuit breakers, all in code:

| Condition | `stopped` value |
|---|---|
| `finish` action with an answer | `finish` |
| `max_steps` reached | `step_budget` |
| Same tool with identical arguments twice | `repeated_action` |
| `max_consecutive_errors` invalid JSON replies | `invalid_actions` |
| `max_consecutive_errors` unknown tools | `unknown_tools` |
| `max_consecutive_errors` failing tools | `tool_errors` |
| LLM endpoint unreachable | `llm_error` |

Each run writes a replayable trace to `staging/traces/<timestamp>-<stopped>-<slug>.json`
with every thought, action, argument and observation. Exit code is `0` only when the run
finished with an answer.

## Settings

Values in `config/pipeline.yaml` are loaded into the `Settings` model:

```bash
ra ingest paper.pdf --config config/pipeline.yaml
```

On machines with 2 GB VRAM start with `engine.page_batch_size: 1` and
`pipeline.concurrency: 2`. If Granite-Docling-258M does not fit in 2 GB the engine raises
`CUDA out of memory`; the pipeline records it as `engine_error` and marks the page
`#needs-review` instead of crashing. Use `--device cpu` to run on CPU.

## Tests

```bash
pytest -q
```

`tests/unit/test_gate.py` exercises every check with synthetic broken output,
`tests/unit/test_pipeline.py` covers the end-to-end flow, the cache and the circuit
breaker, and `tests/unit/test_refagent.py` covers the reference strategy ladder.

## Next steps

- Paper summary / reproduction / vault-context pipelines (slides 7-9)
- Crop vector figures instead of rendering the whole page
- LangGraph adapter: `pipeline.py` is built from pure nodes, so one wrapper is enough
