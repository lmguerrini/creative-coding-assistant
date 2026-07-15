# Official KB Sync CLI

Run the official knowledge-base sync pipeline from the project root:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

The sync CLI only ingests approved official sources and writes explicit
embeddings into the Chroma knowledge-base collection.

## Network and replacement safety

Registered source fetches must use absolute HTTPS URLs whose host resolves only
to globally routable addresses. Each response is limited to 5 MiB. Redirects
are followed only when they preserve the original HTTPS host and port;
cross-origin redirects are rejected before redirected I/O.

After a successful fetch, normalize, and embedding pass, one source-replacement
operation upserts current records and deletes superseded records. A valid
zero-chunk replacement clears the old source snapshot so stale chunks cannot
remain retrievable. Batch and rebuild failure handling still reports the source
failure and applies the surrounding backup/restore contract where configured.

## Environment

Set environment variables directly or place them in a local `.env` file.

Required:

- `OPENAI_API_KEY` or `CCA_OPENAI_API_KEY`

Optional:

- `CCA_OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `CCA_LOG_LEVEL` (default: `INFO`)

## Approved source inventory

The committed registry in
`src/creative_coding_assistant/rag/sources.py` is the canonical source-ID and
URL inventory. The in-product Knowledge Base surface and CLI validation read
that registry directly; this page intentionally avoids a hand-maintained list
that can drift as sources are added or retired.

Inspect the current command surface without fetching or mutating data:

```bash
.venv/bin/python scripts/sync_official_kb.py --help
```

## Common Usage

Sync all approved sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

Sync selected approved sources:

```bash
.venv/bin/python scripts/sync_official_kb.py \
  --source-id three_docs \
  --source-id p5_reference
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
