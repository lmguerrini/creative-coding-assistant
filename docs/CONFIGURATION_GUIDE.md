# Configuration Guide

Creative Coding Assistant (CCA) loads backend configuration from environment
variables and a root `.env` file. The browser uses `NEXT_PUBLIC_*` variables at
build/start time. Keep configuration minimal, local by default, and explicit
about every external-provider boundary.

## Minimal local configuration

No secret should appear in a committed file. For live OpenAI generation, a
minimal uncommitted `.env` is:

```dotenv
OPENAI_API_KEY=replace-with-a-real-secret
CCA_OPENAI_MODEL=gpt-5-mini
CCA_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LANGSMITH_TRACING=false
```

`CCA_OPENAI_API_KEY` is an alias for `OPENAI_API_KEY`; when both exist,
`OPENAI_API_KEY` wins. A missing provider key should be represented as an
unavailable provider-dependent path, not disguised as a successful provider
call.

## Backend settings

Defaults below describe the current settings model. Environment-specific paths
may resolve relative to the repository.

| Variable | Current default or status | Purpose and boundary |
|---|---|---|
| `CCA_ENVIRONMENT` | `local` | Selects environment posture. Production requires explicit origins and an appropriate process server. |
| `OPENAI_API_KEY` | unset, secret | Authorizes OpenAI generation/evaluation calls. Never log or commit it. |
| `CCA_OPENAI_API_KEY` | unset, secret alias | Used only when `OPENAI_API_KEY` is absent. |
| `CCA_OPENAI_MODEL` | `gpt-5-mini` | Default generation model; availability and price remain provider-dependent. |
| `CCA_OPENAI_MAX_OUTPUT_TOKENS` | `4000` | Bounded from 64 through 8000 by settings validation. |
| `CCA_OPENAI_TIMEOUT_SECONDS` | `90` | Provider request timeout, validated from 1 through 300 seconds. |
| `CCA_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for provider-assisted indexing/retrieval work. A model change can require a compatible rebuild. |
| `CCA_DEFAULT_GENERATION_PROVIDER` | `openai` | Default generation provider identifier; it does not prove a call succeeded. |
| `CCA_DEFAULT_DOMAIN` / `CCA_DEFAULT_MODE` | `three_js` / `generate` | Local runtime defaults; they do not add domain or workflow capabilities. |
| `CCA_CORS_ALLOWED_ORIGINS` | `http://127.0.0.1:3000,http://localhost:3000` | Comma-separated browser origins. Local settings are loopback-only by default; production requires explicit deployed origins and rejects the wildcard. Aliases: `CCA_CORS_ALLOW_ORIGINS`, `CCA_ALLOWED_ORIGINS`. |
| `CCA_CHROMA_PERSIST_DIR` | `data/chroma` | Stores the local vector index. It may contain source and conversation excerpts and must be handled as local application data. |
| `CCA_WORKSPACE_SESSION_DB_PATH` | `data/workspace_sessions.sqlite3` | Stores compacted workspace/session snapshots; image attachments are not restored into the composer. Alias: `CCA_WORKSPACE_DB_PATH`. |
| `CCA_ARTIFACT_DIR` | `data/artifacts` | Reserved backend artifact path. The current product persists artifacts in workspace snapshots and creates browser-mediated downloads; it does not automatically write generated artifacts here. |
| `CCA_EVAL_DATA_PATH` | `data/eval/live_sessions.jsonl` | Raw live evaluation rows; can contain private prompt/context data and are not public fixtures. |
| `CCA_EVAL_RAGAS_RESULTS_PATH` | `data/eval/ragas_results.jsonl` | Default local RAGAS result path. Do not treat it as approved public evidence. |
| `CCA_EVAL_RAGAS_MODEL` | `gpt-4o-mini` | Evaluator model for RAGAS, not the default generation model. |
| `CCA_EVAL_RAGAS_TIMEOUT_SECONDS` | `180` | Per-evaluation timeout, minimum 1 second. |
| `CCA_EVAL_RAGAS_MAX_RETRIES` / `CCA_EVAL_RAGAS_MAX_WORKERS` | `2` / `2` | Evaluation retry and concurrency bounds; both can affect provider cost and load. |
| `CCA_LOG_LEVEL` / `CCA_LOG_FORMAT` | `INFO` / `text` | Format is `text` or `json`. Increase verbosity only when needed and redact private/secret-bearing data before sharing. |
| `LANGSMITH_TRACING` | application default `false` | Enables optional external tracing when deliberately configured. Keep false or unset unless approved. |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | unset / `creative-coding-assistant` / unset | External telemetry boundary; key is secret. `LANGCHAIN_*` and `CCA_LANGSMITH_*` aliases are supported. |
| `CCA_LANGSMITH_TIMEOUT_MS` / `CCA_LANGSMITH_SAMPLING_RATE` | `1500` / `1.0` | Optional tracing timeout and sampling fraction (0 through 1); they matter only after tracing is intentionally authorized. |

Treat `.env.example` values as placeholders. It keeps tracing disabled and its
LangSmith key blank; copy only the settings that the local run actually needs.

## Frontend endpoint settings

The local UI defaults to the Python API on port 8000:

| Variable | Local default |
|---|---|
| `NEXT_PUBLIC_ASSISTANT_STREAM_URL` | `http://localhost:8000/api/assistant/stream` |
| `NEXT_PUBLIC_WORKSPACE_SESSION_URL` | `http://localhost:8000/api/workspace/session` |
| `NEXT_PUBLIC_DOMAIN_EXPERIENCE_URL` | `http://localhost:8000/api/domain-experience` |
| `NEXT_PUBLIC_KNOWLEDGE_BASE_URL` | `http://localhost:8000/api/knowledge-base` |
| `NEXT_PUBLIC_EVALUATION_RUN_URL` | `http://localhost:8000/api/evaluation/run` |

Each endpoint has its own override. When changing hosts or ports, configure and
verify every UI route in the checked-out frontend rather than assuming one
variable rewrites them all.

Because `NEXT_PUBLIC_*` values are delivered to the browser, they must never
contain credentials. Restart the Next.js process after changing them.

## Provider and privacy boundaries

- Generation sends the submitted prompt, current request context, and any image
  attached to that submission to the configured provider.
- Image bytes are request-scoped. They are not restored by normal session
  persistence, but a project bundle exported before submission may contain the
  queued image.
- Knowledge-base refresh can download registered source material and send text
  for embedding, depending on configuration.
- When conversation recording is configured, a successful non-error turn can
  send its prompt and answer for embeddings before their vectors are stored
  locally. Recording failure does not prove the response failed.
- Provider-scored RAGAS evaluation sends the approved evaluation payload to an
  external evaluator. Raw local session JSONL and local Chroma excerpts are not
  implicitly approved for that transfer.
- Optional tracing is an additional external data boundary, not a harmless UI
  switch.

See [ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md) before using
real user data.

## Storage and export

CCA uses local paths for Chroma, SQLite workspace/session snapshots, a reserved
artifact directory, and evaluation records. Current artifact persistence is in
workspace snapshots and explicit browser downloads; the backend does not
automatically write generated artifacts to `CCA_ARTIFACT_DIR`. The browser also
keeps a compact localStorage fallback and workspace identity so it can recover
bounded state when the session API is unavailable. “Local” is a placement
claim, not automatic encryption, retention, backup, or deletion. Apply
workstation access controls and inspect ignored runtime directories and browser
storage before archiving or handing off a workstation.

Exports are user-mediated handoff packages. They can contain prompt-derived
code, metadata, source attribution, and—in the pre-submit queued-image
case—image bytes. Export does not prove that an external tool executed,
validated, or deployed the package.

## Knowledge-base configuration

After reviewing provider and source boundaries, refresh all registered sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

The in-product Knowledge Base API also supports review and explicit update or
rebuild operations. Mutating operations require confirmation and selected
source IDs and use backup/restore behavior on failure. The exact operational
model is in [DATA_AND_KB.md](DATA_AND_KB.md).

## Evaluation configuration

Prepare the current seven-case benchmark without provider calls:

```bash
.venv/bin/python -m creative_coding_assistant.eval.current_product_cli \
  --scope rag \
  --dry-run
```

A live run requires the optional evaluation dependencies and the CLI's explicit
`--allow-provider-calls` flag. First inspect the benchmark, model, output path,
expected cost, knowledge-base state, and privacy boundary. The separate
`eval_live_sessions.py` runner is retained for historical or independently
approved recorded-session fixtures. Current commands and evidence boundaries
are in [eval.md](eval.md).

## Production warning

The local Python development server is for local development. A production
deployment needs explicit CORS origins, secret injection, TLS at the serving
edge, durable storage/backup policy, dependency and vulnerability review,
logging/redaction controls, and a supported process server. A successful local
smoke test is not evidence that those controls exist.
