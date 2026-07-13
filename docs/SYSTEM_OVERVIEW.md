# System Overview

Creative Coding Assistant is a local-first, two-process application: a Next.js
workstation presents the product, and a Python WSGI service owns validation,
orchestration, retrieval, generation, evaluation actions, and persistence APIs.
The normal runtime is deliberately bounded: the browser never calls a model
directly, the backend exposes a fixed route set, and external calls occur only
through explicit provider, embedding, source-sync, evaluation, or optional
tracing adapters.

## System context

```mermaid
flowchart TB
    reviewer["Creative coder / reviewer"]

    subgraph browser["Browser process"]
        next["Next.js workstation"]
        dashboard["Dashboard + Inspector"]
        renderer["Controlled preview runtimes"]
        local_fallback["Browser storage fallback"]
        next --> dashboard
        next --> renderer
        next --> local_fallback
    end

    subgraph backend["Python process"]
        wsgi["Exact-path WSGI dispatcher"]
        service["AssistantService"]
        langgraph["Compiled LangGraph"]
        kb_api["KB and evaluation actions"]
        session_api["Workspace session service"]
        wsgi --> service --> langgraph
        wsgi --> kb_api
        wsgi --> session_api
    end

    subgraph local["Local durable state"]
        chroma["Chroma\nofficial docs + memory"]
        sqlite["SQLite\nworkspace sessions"]
        files["JSONL / JSON / artifact files"]
    end

    subgraph external["Explicit external boundaries"]
        openai_gen["OpenAI Responses"]
        openai_embed["OpenAI embeddings"]
        official["Approved official source URLs"]
        evaluator["RAGAS evaluator provider"]
        traces["Optional LangSmith"]
    end

    reviewer --> next
    next -->|"HTTP: NDJSON + JSON"| wsgi
    langgraph --> chroma
    session_api --> sqlite
    service --> files
    langgraph --> openai_gen
    langgraph --> openai_embed
    kb_api --> official
    kb_api --> openai_embed
    kb_api --> evaluator
    langgraph -. "safe trace metadata when enabled" .-> traces
```

## Component responsibilities

| Component | Owns | Does not own |
|---|---|---|
| Next.js workstation | Composer, Dashboard, Inspector, sessions UI, artifact source, exports, preview controls, fullscreen, Demo Mode | Model credentials, retrieval ranking, workflow decisions |
| WSGI API | HTTP validation, CORS, contract versions, fixed dispatch, NDJSON serialization | A generic web framework router or arbitrary plugin endpoints |
| `AssistantService` | Dependency composition, stream lifecycle, provider adapter, eval and memory recording boundaries | Browser rendering |
| LangGraph | Node state, conditional transitions, one bounded refinement loop, terminal failure normalization | Parallel agent workers or dynamic tool installation |
| Chroma | Official-document vectors and separate memory collections | Workspace layout/session snapshot storage |
| SQLite session store | Workstation snapshots and session lifecycle | Model memory retrieval or official-source embeddings |
| OpenAI adapters | Generation and embeddings behind provider-neutral contracts | Automatic provider switching; OpenAI is the only implemented provider factory route |
| Preview runtime | Executes source that passes the current browser contract | General HTML, React Three Fiber, remote modules, or external DCC execution |
| Evaluation action | Dry-run or explicitly authorized RAGAS over approved committed datasets | Scoring raw private local sessions by default |

The canonical source locations are the
[Next.js page](../clients/nextjs/src/app/page.tsx),
[WSGI dispatcher](../src/creative_coding_assistant/api/dev_server.py),
[service composition](../src/creative_coding_assistant/app/bootstrap.py),
[runtime graph builder](../src/creative_coding_assistant/orchestration/runtime/graph_builder.py),
and [workflow node registry](../src/creative_coding_assistant/orchestration/runtime/nodes/registry.py).

## Browser product surface

The root Next.js page loads a workspace snapshot and renders one workstation
shell. That shell coordinates:

- prompt, domain, mode, workflow, generation-control, and image-reference input;
- streamed run state and recoverable subsystem errors;
- generated answer, artifact registry, code inspection, refinement, comparison,
  and export;
- controlled p5.js, Three.js, GLSL, and Tone.js preview routes;
- Dashboard and compact Inspector projections of the same published state;
- SQLite-backed session create, rename, restore, update, and delete operations,
  with a browser storage fallback;
- curated Demo Mode prompts that still use the normal request path.

The Dashboard's knowledge inventory is not inferred from one retrieval result.
The domain API returns registered-source metadata and read-only local Chroma
counts separately. Likewise, a preview card is not proof that source executed:
the renderer route, runtime status, visible output, and errors remain distinct.

## HTTP boundary

The development server uses `wsgiref`; the production entrypoint is WSGI for a
compatible server such as Gunicorn. It is not a FastAPI or ASGI application.
The dispatcher matches exact paths:

| Path | Methods | Purpose |
|---|---|---|
| `/api/assistant/stream` | `POST`, `OPTIONS` | Validate one assistant request and return newline-delimited stream events |
| `/api/workspace/session` | `GET`, `POST`, `PUT`, `DELETE`, `OPTIONS` | List, restore, create/update, and delete local workspace sessions |
| `/api/domain-experience` | `GET`, `OPTIONS` | Public-safe domain delivery contracts and read-only KB inventory |
| `/api/evaluation/run` | `POST`, `OPTIONS` | Dry-run or explicitly authorize approved-dataset RAGAS evaluation |
| `/api/knowledge-base` | `GET`, `POST`, `OPTIONS` | Inventory, check, validate, update, or rebuild selected official sources |
| `/api/health` | `GET`, `OPTIONS` | General service and configuration status |
| `/api/health/live` | `GET`, `OPTIONS` | Process liveness |
| `/api/health/ready` | `GET`, `OPTIONS` | Dependency and configuration readiness |

The stream request is capped at 8 MiB. It forbids unknown top-level fields and
validates the query, domain alignment, workflow mode, controls, artifact
refinement, and attachments. Each image reference is limited to 1 MiB; at most
four are accepted; PNG, JPEG, WebP, and GIF MIME declarations must match the
decoded file signature.

## Assistant runtime

The graph has a shared front half and two real execution routes:

```mermaid
flowchart LR
    common["intake → routing → memory"] --> route{"resolved route"}
    route -->|"Single"| single["skip retrieval → context → prompt → generation"]
    route -->|"Multi"| multi["retrieval → context → planning → Director → reasoning → generation"]
    single --> artifacts["extract → preview metadata"]
    multi --> artifacts
    artifacts -->|"Single"| final["finalize"]
    artifacts -->|"Multi"| review["critique → review"]
    review -->|"pass"| final
    review -->|"one bounded retry"| refine["refine → generate again"] --> artifacts
```

`Auto` can resolve to Single only for a lightweight Explain or Debug request
with no official-document capability, no attachment, and at most one domain.
The current default route map assigns the official-document capability to every
task mode, so ordinary Auto requests currently resolve to Multi. The route
selection is published in the stream payload so the UI does not guess it.

Multi Agent is sequential. Its published responsibilities are planner,
researcher, generator, critic, and reviewer, but only the generation stage
crosses the model-provider boundary. Planning, Director, reasoning, artifact
critique, review, and final self-evaluation are typed application logic. A
review can request no more than one additional generation pass.

See the [Architecture Walkthrough](ARCHITECTURE_WALKTHROUGH.md) and
[runtime workflow graph](../architecture/workflow_graph.md) for exact
transitions and skip conditions.

## Retrieval and knowledge

Knowledge ingestion and request retrieval are separate operations:

1. The approved source registry defines source IDs, publishers, URLs, domains,
   priorities, and tags.
2. An explicit sync fetches selected official pages, normalizes and chunks
   them, sends chunk text to the embedding provider, and upserts records into
   `kb_official_docs`.
3. A Multi Agent request sends its query to the embedding provider, searches
   local Chroma, applies requested-domain filtering and bounded diversity
   post-processing, and returns ranked chunks with provenance.
4. Context assembly isolates retrieved excerpts as untrusted reference text
   before prompt rendering.
5. The selected excerpts may enter the provider generation prompt. Registration
   or indexing alone does not mean a source grounded a particular answer.

Retrieval failure is recoverable: the service emits an explicit error payload,
continues with an empty retrieval context, and never invents citations. The
fixed seven-query retrieval report is read-only and stores non-text lineage so
ranking changes can be audited without publishing the retrieved excerpts.

## Persistence

### Chroma collections

| Collection | Responsibility |
|---|---|
| `kb_official_docs` | Official documentation chunks and sync metadata |
| `conversation_turns` | Durable user and assistant turns |
| `conversation_summaries` | Running conversation summaries when available |
| `project_memory` | Stable project preferences, goals, and decisions |
| `eval_traces` | Evaluation trace foundations |
| `preview_artifacts_index` | Preview manifest and capture metadata foundations |

The runtime actively reads official docs, conversation turns, summaries, and
project memory through the composed gateways. A successful conversation turn
is written only when a conversation ID, non-error final answer, and OpenAI
embedding configuration exist. The prompt and answer are embedded externally,
then stored locally with their vectors.

### Workspace SQLite

`data/workspace_sessions.sqlite3` stores saved workstation snapshots. The UI
compacts messages and artifacts before persistence and applies a request-scoped
boundary to multimodal state: queued image bytes are not restored as durable
session attachments.

### Files

- `data/eval/` contains private local live-session and RAGAS rows and remains
  untracked.
- `demo/evaluation/` contains reviewed committed public-safe fixtures and
  machine evidence.
- `data/artifacts/` is the configurable local artifact path.

## External-call matrix

| Trigger | Destination | Data sent | Authorization posture |
|---|---|---|---|
| Assistant generation | OpenAI Responses | Configured request can contain rendered prompt messages and explicitly submitted images | User submits a live request; credentials required; current tests establish payload construction, not live provider receipt/use/image influence |
| Query retrieval | OpenAI embeddings | User query text | Happens only when Multi retrieval is selected and configured |
| Conversation recording | OpenAI embeddings | Successful user query and assistant answer | Configured memory recorder; failure does not fail the response |
| KB sync/update | Official URL, then OpenAI embeddings | Approved page fetch; normalized chunk text for embeddings | Explicit CLI or confirmed Dashboard action |
| RAGAS evaluation | OpenAI evaluator/embeddings | Selected committed sanitized or redacted fixture | Dry-run by default; live action requires provider-call opt-in |
| LangSmith | Configured LangSmith endpoint | Run metadata such as mode, IDs, lengths, counts, route lineage | Disabled unless tracing and key are configured |

## Deployment and security posture

The repository includes a Gunicorn WSGI entrypoint, Docker image, health probes,
durable path configuration, structured logging, and production CORS guards.
Those are deployment foundations, not evidence that a public instance is
running. Authentication, authorization, rate limiting, TLS/WAF termination,
managed secrets, tenant isolation, and managed backups remain platform or
future product responsibilities.

For a local review, bind both services to loopback. For any networked use, set
`CCA_ENVIRONMENT=production`, configure explicit
`CCA_CORS_ALLOWED_ORIGINS`, place an authentication/rate-limit boundary in
front of the WSGI app, and mount the `data/` paths on protected durable
storage.

## Evidence rule

Resolve conflicting claims in this order:

1. behavior observed in the current running application;
2. current deterministic or browser validation;
3. current machine-readable evaluation evidence;
4. current architecture and product documentation;
5. clearly dated historical evidence;
6. clearly labelled future work.

Blocked or missing evidence is neither zero nor a pass. Continue with the
[Capability Matrix](CAPABILITY_MATRIX.md) for the public claim inventory.
