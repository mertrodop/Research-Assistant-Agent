# data/

Pipeline'ın girdi ve çıktı katmanı. Kod bu dizinlere `config/pipeline.yaml` içindeki
`data.raw_dir` ve `data.processed_dir` üzerinden bakar.

```
data/
  raw/          kaynak PDF'ler, hiç değiştirilmez
  processed/    OCR pipeline çıktısı, <sha256[:12]>.md
  processed/figures/<sha12>/   çıkarılan figür PNG'leri
```

## Kurallar

- `raw/` yalnızca eklenir. Bir PDF değişirse sha256'sı değişir, dolayısıyla cache ve
  çıktı dosyası da yeni bir ad alır; eski kayıt sessizce ezilmez.
- `processed/` tamamen üretilebilir. Silinip `ra-ocr batch` ile yeniden oluşturulabilir.
- Her çıktı dosyası YAML frontmatter taşır: `sha256`, `ocr_engine`, `pages`,
  `verified_pages`, `status`, `figures`, `figure_files`, `figure_readings`.
  `status: needs-review` olan dosya vault'a alınmadan önce incelenmelidir.
- Figür PNG'leri `processed/figures/<sha12>/` altında durur; markdown göreli yolla
  (`figures/pNNN_figK.png`) bağlar, böylece Obsidian'da da açılır.

## Mevcut makaleler

| dosya | başlık | sayfa |
|---|---|---|
| `raw/2504.01848.pdf` | PaperBench: Evaluating AI's Ability to Replicate AI Research | 30 |
| `raw/2409.11363.pdf` | CORE-Bench: Fostering the Credibility of Published Research | 30 |

İkisi de slayt 2'deki motivasyonda atıf verilen çalışmalar.

## Kullanım

```bash
ra-ocr batch                       # data/raw -> data/processed
ra-ocr batch --engine native       # metin katmanı olan PDF'ler için hızlı yol
ra-ocr ingest data/raw/2504.01848.pdf --pages 1-5
```
