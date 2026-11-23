# data/

Input and output layer of the pipeline. The code reads these directories through
`data.raw_dir` and `data.processed_dir` in `config/pipeline.yaml`.

```
data/
  raw/          source PDFs, never modified
  processed/    OCR pipeline output, <sha256[:12]>.md
  processed/figures/<sha12>/   extracted figure PNGs
```

## Rules

- `raw/` is append-only. If a PDF changes its sha256 changes, so the cache and the output
  file get a new name and the old record is never silently overwritten.
- `processed/` is fully reproducible. Delete it and rebuild with `ra-ocr batch`.
- Every output file carries YAML frontmatter: `sha256`, `ocr_engine`, `pages`,
  `verified_pages`, `status`, `figures`, `figure_files`, `figure_readings`. A file with
  `status: needs-review` must be inspected before it enters the vault.
- Figure PNGs live under `processed/figures/<sha12>/`; the markdown links them with a
  relative path (`figures/pNNN_figK.png`) so they also open in Obsidian.

## Papers in the repository

| file | title | pages |
|---|---|---|
| `raw/2504.01848.pdf` | PaperBench: Evaluating AI's Ability to Replicate AI Research | 30 |
| `raw/2409.11363.pdf` | CORE-Bench: Fostering the Credibility of Published Research | 30 |

Both are cited in the motivation on slide 2.

## Usage

```bash
ra-ocr batch                       # data/raw -> data/processed
ra-ocr batch --engine native       # fast path for text-layer PDFs
ra-ocr ingest data/raw/2504.01848.pdf --pages 1-5
```
