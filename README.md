# paperless-cf-ocr

A [Paperless-ngx](https://docs.paperless-ngx.com/) 3.x (3.0.5+) parser plugin that OCRs scanned
documents with vision language models on **Cloudflare Workers AI**, called through
**Cloudflare AI Gateway**. It replaces Tesseract's text with a much more accurate
transcription. It reads handwriting, form fields and degraded copies. Paperless keeps
everything else: thumbnails, searchable archive PDFs, workflows and AI suggestions.

- Works with **native and Docker installs**, with no custom Paperless image and no sidecar
  service.
- **Default model: DeepSeek V4.1 Flash** (`@cf/deepseek-ai/deepseek-v4.1-flash-1017`, with
  reasoning off). Any Workers AI vision model,
  or third-party models routed through the gateway, can be selected.
- **Confidence-aware.** Each page gets a confidence score from token probabilities. Pages
  the model is unsure about can optionally be re-read by a second model, such as Claude
  Sonnet 5.5.
- **Asked not to guess.** The transcription prompt tells the model to write `[illegible]`
  for unreadable text instead of making it up. In testing, models without this rule
  invented plausible text, including a wrong drug name on a medical form.
- **Reuses Paperless's AI settings** when they already point at the same AI Gateway.

> Third-party plugins are not supported by the Paperless-ngx maintainers. Reproduce any
> Paperless problem with the plugin removed before reporting it upstream.

## How it works

1. Paperless picks this parser for PDFs and images (`score` 30, above Tesseract's 10 and
   the built-in Azure OCR's 20) whenever the plugin is configured. With
   `PAPERLESS_REMOTE_OCR_MODE=workflow_only`, it is only used for documents a workflow
   marks for remote OCR, as for any remote parser.
2. PDF pages are rendered at 200 dpi (at most 2048 px) and images are normalised. Pages
   are sent concurrently to the gateway with a strict transcription prompt.
3. Confidence comes from token log-probabilities. A page counts as low-confidence if more
   than 1% of its words have p < 0.5, any word has p < 0.2, or the model wrote
   `[illegible]`. If a second-opinion model is configured, such pages are re-run on it and
   its output is used.
4. **Hybrid archive.** When Paperless asks for a searchable archive PDF, Paperless's own
   Tesseract/ocrmypdf pipeline builds it, so text positions, PDF/A, deskew and rotation
   follow your normal OCR settings. The document *content* (search, AI, matching) comes
   from the vision model.
5. Pages the gateway cannot transcribe fall back to local Tesseract, and the document is
   still consumed.

### Which PDFs are OCR'd

| Situation | Behaviour |
|---|---|
| `PAPERLESS_OCR_MODE=off` | never OCR'd; embedded text is kept |
| `PAPERLESS_OCR_MODE=redo` / `force` | always OCR'd |
| no real embedded text | OCR'd |
| scan that already has a (scanner) OCR text layer | OCR'd again, because pages are full-page images (`EXISTING_TEXT=auto`) |
| born-digital PDF (text, no full-page scan images) | embedded text kept, no API call |

Set `PAPERLESS_CF_OCR_EXISTING_TEXT=trust` to keep any existing text layer, as Paperless
does. Set it to `ignore` to always OCR.

## Results from testing

These are measured on three real documents: a typed letter, an NHS GOS 2 form with
handwriting, and a poorly photocopied allergy action plan with handwriting. Scores are the
share of key values (names, dates, phone numbers, prescriptions, handwritten notes)
transcribed correctly.

| | Typed letter (word F1) | Form + handwriting | Degraded copy + handwriting | Speed (median / page) | Cost / 1,000 pages |
|---|---|---|---|---|---|
| Existing scanner OCR (Tesseract) | 0.980 | 24/26 | 10/21 | – | – |
| **DeepSeek V4.1 Flash** (default, reasoning off) | 0.998 | 25/26 | 18–19/21 | 4 s | ~$1 |
| Qwen 3.8 27B (previous default) | 0.997 | 26/26 | 18–19/21 | 16 s | ~$3.50 |
| Mistral Small 3.1 | 0.996 | 25/26 | 18/21 | 15 s | ~$1 |
| Claude Sonnet 5.5 | 0.998 | 26/26 | 21/21 | 5 s | ~$15 |

Workers AI latency varies a lot under load, with occasional pages taking up to about
100 s (Qwen); DeepSeek's slowest page in testing took 18 s. The plugin processes 4 pages at a
time. Models occasionally get stuck repeating one line until the output limit (DeepSeek did
once in 11 reads of the hardest page); the plugin detects this and retries the page.

## Requirements

- Paperless-ngx **3.0.5 or newer**. Every 3.x minor release from 3.0.5 to 3.5.1 is tested
  end-to-end in the official Docker image. Before 3.1 there is no
  `PAPERLESS_REMOTE_OCR_MODE`, so the plugin handles every supported document.
- A Cloudflare account with an **AI Gateway**. New gateways require authentication.
- The **Workers Paid** plan for the default model (DeepSeek V4.1 Flash is paid-only). On the
  free plan, pick another model such as `@cf/qwen/qwen3.8-27b`.
- A Cloudflare **API token** with *Workers AI: Read* and *AI Gateway: Run*. With both
  permissions, this one token also passes an authenticated gateway. A separate gateway
  token (`PAPERLESS_CF_OCR_GATEWAY_TOKEN`) is optional.

> **Privacy.** Page images leave your server and are processed by Cloudflare Workers AI.
> With a third-party second-opinion model such as `anthropic/...`, they are also processed
> by that provider. AI Gateway **request logging stores the full requests and responses**,
> which means your document images and their text. Disable log collection on the gateway
> you use for OCR, or restrict logging to metadata.

## Installation

### Docker: installer image (recommended)

A small, versioned installer image copies the plugin into a volume that Paperless loads
through `PYTHONPATH`. It needs no internet access at Paperless startup and no init script,
and it works with rootless Paperless containers. Add the parts marked `# --- add` in
[`deploy/docker/compose.example.yml`](deploy/docker/compose.example.yml) to your compose
file, create `cf-ocr.env` from
[`deploy/docker/cf-ocr.env.example`](deploy/docker/cf-ocr.env.example), then run:

```sh
docker compose pull && docker compose up -d
docker compose logs webserver | grep -i "third-party parser"
# Loaded third-party parser 'Cloudflare AI Gateway OCR' v0.1.0 ...
```

To upgrade or roll back, change the installer image tag.

### Docker: install from PyPI at startup

Mount [`deploy/docker/init/`](deploy/docker/init/) at `/custom-cont-init.d` (read-only,
owned by root). The script runs `pip install paperless-cf-ocr` with `--no-deps` on every
container start. It needs root and internet access. Pin the version with
`PAPERLESS_CF_OCR_VERSION`.

### Native (bare metal)

Install into the Python environment that runs Paperless, then restart all Paperless
services:

```sh
sudo -Hu paperless pip3 install --no-deps paperless-cf-ocr
```

`--no-deps` is safe because the plugin only imports libraries that Paperless already ships
(httpx, Pillow, pypdfium2). Put the settings into `paperless.conf`.

## Configuration

Only a gateway and a credential are required.

| Variable | Default | |
|---|---|---|
| `PAPERLESS_CF_OCR_ACCOUNT_ID`, `PAPERLESS_CF_OCR_GATEWAY_ID` | | gateway location… |
| `PAPERLESS_CF_OCR_GATEWAY_URL` | | …or `https://gateway.ai.cloudflare.com/v1/<account>/<gateway>` |
| `PAPERLESS_CF_OCR_API_TOKEN` | | sent as `Authorization: Bearer` |
| `PAPERLESS_CF_OCR_GATEWAY_TOKEN` | | sent as `cf-aig-authorization` (authenticated gateways) |
| `PAPERLESS_CF_OCR_MODEL` | `@cf/deepseek-ai/deepseek-v4.1-flash-1017` | `@cf/...` models use the Workers AI route; `provider/model` uses the gateway's OpenAI-compatible route |
| `PAPERLESS_CF_OCR_SECOND_OPINION_MODEL` | off | e.g. `anthropic/claude-sonnet-5-5` |
| `PAPERLESS_CF_OCR_SECOND_OPINION` | `low-confidence` | `always`: send every page to the second model (primary as fallback) |
| `PAPERLESS_CF_OCR_LOW_CONFIDENCE_RATIO` | `0.01` | share of words with p < 0.5 that makes a page low-confidence |
| `PAPERLESS_CF_OCR_LOW_CONFIDENCE_MIN` | `0.2` | any word below this makes a page low-confidence |
| `PAPERLESS_CF_OCR_EXISTING_TEXT` | `auto` | `auto` / `trust` / `ignore`, see above |
| `PAPERLESS_CF_OCR_ARCHIVE` | `hybrid` | `none`: never build archive PDFs (saves Tesseract CPU) |
| `PAPERLESS_CF_OCR_FALLBACK` | `tesseract` | `fail`: fail the document when a page cannot be OCR'd |
| `PAPERLESS_CF_OCR_PROMPT` / `_PROMPT_FILE` | built-in strict prompt | override the transcription instructions |
| `PAPERLESS_CF_OCR_EXTRA_PARAMS` | `{}` | JSON merged into primary-model requests; overrides the default model's `{"reasoning_effort": "none"}` |
| `PAPERLESS_CF_OCR_TIMEOUT` | `180` | seconds per request |
| `PAPERLESS_CF_OCR_MAX_RETRIES` | `4` | retries on 408/429/5xx, network errors, empty answers, repetition loops |
| `PAPERLESS_CF_OCR_CONCURRENCY` | `4` | pages in flight per document |
| `PAPERLESS_CF_OCR_DPI` / `_MAX_IMAGE_PX` | `200` / `2048` | page rendering |
| `PAPERLESS_CF_OCR_MAX_PAGES` | `200` | also limited by `PAPERLESS_OCR_PAGES` |
| `PAPERLESS_CF_OCR_MAX_TOKENS` | `8192` | output limit per page |
| `PAPERLESS_CF_OCR_SCORE` | `30` | parser priority |
| `PAPERLESS_CF_OCR_ENABLED` | `true` | `false` disables the plugin without uninstalling it |

### Reusing Paperless's AI settings

If Paperless's AI backend is `openai-like` and its endpoint is the gateway's `compat` or
`workers-ai` route, for example
`PAPERLESS_AI_LLM_ENDPOINT=https://gateway.ai.cloudflare.com/v1/<account>/<gateway>/compat`,
the plugin reuses that gateway and API key. Values set in the Paperless UI take precedence
over environment variables, as in Paperless itself. Provider routes such as `/openai` carry
that provider's key, so set `PAPERLESS_CF_OCR_API_TOKEN` in that case. `PAPERLESS_CF_OCR_*`
variables always override reused settings.

## Troubleshooting

The plugin logs to `paperless.parsing.cf_ocr`:

- `Cloudflare OCR configured: gateway=… model=… auth=…`: the effective settings, without
  secrets.
- `page 2: @cf/deepseek-ai/deepseek-v4.1-flash-1017 confidence mean=0.957 low=2.1% min=0.17 …`: the
  per-page confidence.
- `Cloudflare OCR: 4 page(s), 1 low-confidence, 1 second opinion(s), 0 failed`: the
  per-document summary.
- `HTTP 401 from AI Gateway`: check the tokens. Gateways with authentication need
  `PAPERLESS_CF_OCR_GATEWAY_TOKEN` or a token with AI Gateway permissions.

If the plugin is not configured, Paperless silently uses its built-in parsers, and the
reason is logged as a warning.

## Development

```sh
python3 -m venv .venv-dev && . .venv-dev/bin/activate
pip install -e '.[dev]'
pytest                                   # unit tests (Paperless is stubbed)

# integration tests against real Paperless parsing code + Tesseract/ocrmypdf
# (needs tesseract, ghostscript, poppler, unpaper):
pip install -c tests_integration/constraints-paperless-3.5.1.txt ocrmypdf pikepdf pillow pypdfium2
tests_integration/fetch_paperless.sh && pytest tests_integration
```

Run the two suites in separate `pytest` invocations. `tools/lab/` holds the model
evaluation tooling used to choose the defaults.

## License

GPL-3.0-only, matching Paperless-ngx.
