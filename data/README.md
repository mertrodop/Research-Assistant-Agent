# data/

Input layer. The code reads it through `paths.raw_dir` in `config/pipeline.yaml`.

```
data/
  raw/    source PDFs, named by paper title, never modified
```

`raw/` is append-only. Notes, figures and reports are generated into `vault/` and
`staging/` (git-ignored), never into `data/`.

| file | title |
|---|---|
| `raw/paperbench-evaluating-ais-ability-to-replicate-ai-research.pdf` | PaperBench: Evaluating AI's Ability to Replicate AI Research |
| `raw/core-bench-fostering-the-credibility-of-published-research-through-a-computational-reproducibility.pdf` | CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark |
