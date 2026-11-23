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
ra-ocr batch                                   # data/raw -> data/processed
ra-ocr batch --engine native                   # fast path for text-layer PDFs
ra-ocr batch --figures-llm                     # interpret figures with a vision model
ra-ocr ingest data/raw/2504.01848.pdf          # single PDF
ra-ocr ingest paper.pdf --pages 1-3,7          # page selection
ra-ocr ingest paper.pdf --device cpu           # for models that do not fit in 2 GB VRAM
ra-ocr ingest paper.pdf --vault ./vault        # Obsidian vault root
ra-ocr ingest paper.pdf --json                 # machine-readable output
ra-ocr ingest paper.pdf --no-figures           # disable figure extraction
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
reproducible: delete it and rerun `ra-ocr batch`. Details: `data/README.md`.

The repository ships two real papers: `2504.01848.pdf` (PaperBench) and
`2409.11363.pdf` (CORE-Bench), both cited in the motivation on slide 2.

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
| Tier 2 critic | `ocr/critic.py` | LLM scores, code applies the threshold |
| Retry / circuit breaker | `ocr/pipeline.py` | code: DPI ladder, <= 3 attempts |
| Note | `vault/note.py` | code |

Tier 1 checks: `non_empty`, `printable_ratio`, `repetition_loop`, `coverage`,
`heading_levels`, `table_columns`, `table_header`, `equation_delimiters`,
`doctags_balance`, `numeric_agreement`, `reference_fields`. A page with a failing
`blocker` is retried; a page with a failing `warn` still counts as verified but the reason
is recorded in the note.

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
ra-ocr ingest paper.pdf --figures-llm
ra-ocr batch --figures-llm --vision-model google/gemini-2.5-flash
ra-ocr ingest paper.pdf --figures-llm --vision-base-url http://localhost:11434/v1
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
ra-ocr batch --figures-llm --vision-model ibm-granite/granite-vision-3.1-2b \
             --vision-base-url http://localhost:8000/v1
```

## Tier 2: rubric critic

A page that passes Tier 1 goes to a rubric critic when `--critic` is given. Four dimensions
are scored 1-5 (`fidelity`, `completeness`, `structure`, `no_hallucination`); if the lowest
dimension is below `min_dimension` or the average is below `min_average`, the page becomes
`needs-review` and re-enters the DPI ladder. If the critic is unreachable the check degrades
to `warn` and the run does not crash.

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
| Act | One strategy per attempt: `doi` -> `arxiv` -> `title` -> `title_year` | code |
| Evaluate | Title similarity (`rapidfuzz`) plus year tolerance decides the status | code |
| Reflect | The failed attempt is appended to the trace with its score | code |
| Rescue | If the budget is not spent, the LLM proposes a title/DOI as strict JSON | LLM proposes, code validates |
| Stop | `max_attempts` network calls, then `unresolved` with the full trace | code |

Strategies that cannot run are skipped without consuming budget, so a reference without a
DOI does not waste an attempt. Every check records its `strategy` and its `attempts` trace,
and the note shows both in the `## References check` table.

```bash
ra-ocr batch --crossref
ra-ocr batch --crossref --refs-llm --refs-attempts 5
ra-ocr ingest paper.pdf --crossref --json
```

Results are cached under `.cache/crossref`, so reruns stay offline.

## Settings

Values in `config/pipeline.yaml` are loaded into the `Settings` model:

```bash
ra-ocr ingest paper.pdf --config config/pipeline.yaml
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
