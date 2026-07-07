# Official KB Sync CLI

Run the official knowledge-base sync pipeline from the project root:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

The sync CLI only ingests approved official sources and writes explicit
embeddings into the Chroma knowledge-base collection.

## Environment

Set environment variables directly or place them in a local `.env` file.

Required:

- `OPENAI_API_KEY` or `CCA_OPENAI_API_KEY`

Optional:

- `CCA_OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `CCA_LOG_LEVEL` (default: `INFO`)

## Approved Source IDs

Three.js:

- `three_docs`
- `three_manual`
- `three_examples`

React Three Fiber:

- `r3f_introduction`
- `r3f_canvas_api`
- `r3f_hooks_api`

p5.js:

- `p5_reference`
- `p5_tutorials`
- `p5_examples`

GLSL:

- `glsl_language_spec_460`
- `glsl_es_language_spec_320`

## Common Usage

Sync all approved sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

Sync selected approved sources:

```bash
.venv/bin/python scripts/sync_official_kb.py \
  --source-id three_docs \
  --source-id r3f_canvas_api \
  --source-id glsl_language_spec_460
```

Continue after a source-level failure:

```bash
.venv/bin/python scripts/sync_official_kb.py --all --continue-on-error
```

Print the final summary as JSON:

```bash
.venv/bin/python scripts/sync_official_kb.py --all --summary-format json
```

The CLI rejects `--all` together with `--source-id`.

## Exit Codes

- `0`: all requested sources synced successfully
- `1`: an unexpected failure occurred, or a continued run finished with failed
  sources
- `2`: configuration or validation failure, such as missing embedding
  configuration or an unknown source ID
