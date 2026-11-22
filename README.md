# Research-Assistant-Agent

Agentic research assistant. Bu depo şu an **OCR pipeline**'ını içerir: yeni bir makale
vault'a girerken sayfaları paralel işleyen, her elementi doğrulayan ve yalnızca başarısız
sayfaları yeniden deneyen hat.

Slayt 6'daki tasarım birebir uygulanır:

```
paralel sayfalar -> routing -> code gate -> evaluator/optimizer
Tier 1: deterministik unit testler   Tier 2: rubric critic + Crossref
circuit breaker: <= 3 deneme         çıktı: verified note | #needs-review
```

## Kurulum

```bash
pip install --user -e ".[dev]"        # çekirdek + testler
pip install --user -e ".[vlm]"        # Granite-Docling-258M motoru (docling)
```

`docling` yoksa hat `--engine native` ile de çalışır (yalnızca metin katmanı olan PDF'ler).

## Kullanım

```bash
ra-ocr batch                                   # data/raw -> data/processed
ra-ocr batch --engine native                   # metin katmanı olan PDF'ler için hızlı yol
ra-ocr batch --figures-llm                     # figürleri görsel modelle yorumla
ra-ocr ingest data/raw/2504.01848.pdf          # tek PDF
ra-ocr ingest paper.pdf --pages 1-3,7          # sayfa seçimi
ra-ocr ingest paper.pdf --device cpu           # 2 GB VRAM'e sığmayan model için
ra-ocr ingest paper.pdf --vault ./vault        # Obsidian vault kökü
ra-ocr ingest paper.pdf --json                 # makine okunur çıktı
ra-ocr ingest paper.pdf --no-figures           # figür çıkarımını kapat
```

Çıkış kodu: `0` tüm sayfalar doğrulandı, `1` en az bir sayfa `#needs-review`.

## Veri düzeni

```
data/
  raw/          kaynak PDF'ler, hiç değiştirilmez
  processed/    OCR çıktısı, <sha256[:12]>.md
  processed/figures/<sha12>/   çıkarılan figür PNG'leri
```

`raw/` yalnızca eklenir: PDF değişirse sha256 değişir, cache ve çıktı yeni ad alır, eski
kayıt ezilmez. `processed/` tamamen üretilebilir, silinip `ra-ocr batch` ile yeniden
oluşturulur. Ayrıntı: `data/README.md`.

Depoda iki gerçek makale var: `2504.01848.pdf` (PaperBench) ve `2409.11363.pdf`
(CORE-Bench), ikisi de slayt 2'deki motivasyonda atıf verilen çalışmalar.

Her çıktı YAML frontmatter taşır (`sha256`, `ocr_engine`, `pages`, `verified_pages`,
`status`), gövde markdown ve sayfa bazlı `## Validation` tablosu içerir. Obsidian vault
düzeni için `--vault` kullanılır: `vault/Resources/Parsed/<sha12>.md`.

## Akış

| Aşama | Modül | Karar veren |
|---|---|---|
| Render + istatistik | `ocr/render.py` | kod (PyMuPDF) |
| Routing | `ocr/router.py` | kod: metin katmanı ≥ 200 karakter → native, değilse VLM |
| OCR | `ocr/engine.py` | Granite-Docling-258M, sayfa başına, `temperature=0` |
| Birleştirme | `ocr/assemble.py` | kod: tekrar eden header/footer, sözlük destekli tire kırılması |
| Figürler | `ocr/figures.py` | kod: raster çıkarımı, caption eşleme, vektör için sayfa render'ı |
| Figür yorumu | `ocr/interpret.py` + `llm/client.py` | LLM önerir, kod doğrular (grounded critic) |
| Tier 1 gate | `ocr/gate.py` | kod: 10 deterministik kontrol |
| Retry / circuit breaker | `ocr/pipeline.py` | kod: DPI merdiveni, ≤ 3 deneme |
| Not | `vault/note.py` | kod |

Tier 1 kontrolleri: `non_empty`, `printable_ratio`, `repetition_loop`, `coverage`,
`heading_levels`, `table_columns`, `table_header`, `equation_delimiters`,
`doctags_balance`, `numeric_agreement`, `reference_fields`. `blocker` düşen sayfa yeniden
denenir; `warn` düşen sayfa doğrulanmış sayılır ama gerekçesi notta görünür.

`table_header` bu hatta gerçek bir hatayı yakalamak için eklendi: Granite-Docling tablo
başlığını düşürüp ilk veri satırını başlık yapıyor (`| 1 | 0.91 |` + ayırıcı). Başlık
metni kaybı içerik kaybı olduğu için `blocker` sayılır ve sayfa DPI merdiveniyle yeniden
denenir.

## Figürler

Sayfa başına raster figürler `page.get_image_info(xrefs=True)` ile bulunur, PNG olarak
`processed/figures/<sha12>/pNNN_figK.png` altına yazılır ve caption'ına göre markdown'a
yerleştirilir:

```markdown
![Figure 1](figures/p002_fig1.png)

Figure 1. PaperBench is a benchmark for evaluating AI agents' abilities ...
```

- Caption eşleme: `Figure N` / `Fig. N` ile başlayan metin bloğu, figür kutusuna en yakın
  olan (≤ `max_caption_gap` punto) seçilir; üstte veya altta olabilir.
- Filtre: `min_side` pikselden küçük veya sayfa alanının `min_area_ratio`'sundan küçük
  görseller (ikon, logo, çizgi) atlanır.
- Aynı caption'ı paylaşan paneller `Figure 4 (1/2)`, `Figure 4 (2/2)` olarak numaralanır.
- Vektör figürlerde raster nesne yoktur; sayfada caption var ama raster yoksa sayfa
  `pNNN_page.png` olarak render edilir (`kind: page_render`). Bu bir yaklaşımdır, kırpma
  yapmaz.
- Frontmatter `figures` sayısını ve `figure_files` listesini taşır.

## Figür yorumlama (opsiyonel)

`--figures-llm` ile her figür bir görsel modele gönderilir ve notta alıntı bloğu olarak
görünür:

```markdown
![Figure 1](figures/p002_fig1.png)

> **screenshot** — The image displays a sequence of thought processes ...
> - Experiments are run with 'python model_training.py --bs 32'.
> ⚠ not corroborated by page text: 32
```

Tasarım slayttaki "code decides, LLM only proposes" ilkesine uyar:

| Adım | Kim karar verir |
|---|---|
| Prompt + strict JSON şeması | kod (`ocr/interpret.py`) |
| JSON ayrıştırma, alan kırpma | kod, `pydantic` doğrulaması |
| **Grounded critic** | kod: modelin bildirdiği her sayı caption + sayfa metninde aranır |
| `figure_grounding` kontrolü | kod: doğrulanamayan sayı varsa `warn` |
| Notta gösterim | kod: uyarı satırı doğrudan alıntının altında |

`ungrounded` etiketi "halüsinasyon" demek değil, "sayfa metninde teyit edilemedi" demek:
grafiğin içindeki bir sayı (ör. `--bs 32`) metinde geçmediği için de buraya düşer. Bu
yüzden `warn`, `blocker` değil.

İstemci OpenAI-uyumlu `/chat/completions` konuşur; OpenRouter, OpenAI, vLLM, Ollama ve
LM Studio ile aynı kod çalışır. Anahtar `vision.api_key_env` (varsayılan
`OPEN_ROUTER_API_KEY`) üzerinden ortamdan veya `.env`'den okunur, koda yazılmaz.

```bash
ra-ocr ingest paper.pdf --figures-llm
ra-ocr batch --figures-llm --vision-model google/gemini-2.5-flash
ra-ocr ingest paper.pdf --figures-llm --vision-base-url http://localhost:11434/v1
```

Okumalar `.cache/figures/<sha12>/<figür>__<model>.json` altında saklanır; aynı figür ve
model için ikinci kez API çağrısı yapılmaz.

Bozuk JSON iki kademede kurtarılır: önce kesik nesne kapatılır ve sondaki virgüller
temizlenir, olmazsa modele bir kez "compact JSON" düzeltici isteği giderilir. Kalıcı çözüm
constrained decoding'dir (`guided_json`), o zaman geçersiz JSON hiç oluşmaz.

### Açık kaynak model seçenekleri

| Rol | Model | Lisans | 4-bit |
|---|---|---|---|
| OCR | `ibm-granite/granite-docling-258M` | Apache 2.0 | ~0.5 GB |
| Görsel | `ibm-granite/granite-vision-3.1-2b` | Apache 2.0 | ~1.3 GB |
| Görsel | `Qwen/Qwen2.5-VL-3B-Instruct` | Apache 2.0 | ~2 GB |
| Agentic | `ibm-granite/granite-4.0-micro` | Apache 2.0 | ~2 GB |
| Agentic | `Qwen/Qwen3-4B-Instruct-2507` | Apache 2.0 | ~2.5 GB |

Llama ve Gemma lisansları OSI onaylı değildir; "açık kaynak" sayılmaz. Determinizm için
`temperature=0`, sabit `seed`, `batch_size=1`, aynı quantization dosyası ve aynı runtime
sürümü gerekir.

```bash
vllm serve ibm-granite/granite-vision-3.1-2b --guided-json --seed 42
ra-ocr batch --figures-llm --vision-model ibm-granite/granite-vision-3.1-2b \
             --vision-base-url http://localhost:8000/v1
```

## Ayarlar

`config/pipeline.yaml` içindeki değerler `Settings` modeline yüklenir:

```bash
ra-ocr ingest paper.pdf --config config/pipeline.yaml
```

2 GB VRAM'li makinelerde `engine.page_batch_size: 1` ve `pipeline.concurrency: 2` ile
başlayın. Granite-Docling-258M 2 GB'a sığmazsa motor `CUDA out of memory` verir; hat bunu
`engine_error` olarak kaydeder ve sayfayı `#needs-review` yapar, süreç çökmez. CPU için
`--device cpu` kullanın.

## Testler

```bash
pytest -q
```

`tests/unit/test_gate.py` her kontrolü sentetik bozuk çıktıyla sınar,
`tests/unit/test_pipeline.py` uçtan uca akışı, cache'i ve circuit breaker'ı doğrular.

## Sıradaki adımlar

- Tier 2 rubric critic (görüntü + metin, strict JSON) ve Crossref doğrulaması
- Paper summary / reproduction / vault-context pipeline'ları (slayt 7-9)
- LangGraph adaptörü: `pipeline.py` saf düğümlerden oluştuğu için tek sarmalayıcı yeterli
