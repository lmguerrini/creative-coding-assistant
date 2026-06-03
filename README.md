# Creative Coding Assistant

Creative Coding Assistant is an AI-powered creative workflow platform for
creative coding, generative graphics, and realtime media work. It combines a
LangGraph-orchestrated Python backend with a Next.js workstation that streams
assistant output into code, multiple artifacts, live previews, comparison tools,
refinement controls, and runtime inspectors.

The application is a multi-artifact generation environment: requests can route
through domain-aware retrieval and generation, produce several creative
candidates, rank them with structured critique, compare alternatives, refine a
selected artifact, and inspect the resulting runtime behavior from the same
workstation.

![Creative Coding Assistant workstation preview](assets/preview.png)

## Highlights

- AI-powered workstation UX with chat, a compact bottom preview shelf, and
  focused inspector tabs for overview, preview, runtime, code, workflow,
  telemetry, artifacts, and retrieval
- Domain-aware generation for p5.js, Three.js, React Three Fiber, GLSL, and
  broader creative-coding knowledge domains
- LangGraph workflow runtime with routing, memory, retrieval, prompt assembly,
  generation, artifact extraction, preview preparation, artifact critique,
  deterministic review, bounded refinement, and terminal failure handling
- Graph-owned preview pipeline that prepares preview metadata before the
  frontend mounts supported outputs in controlled browser runtimes
- Streaming NDJSON bridge between the Next.js client and the Python assistant
  runtime
- Live artifact hydration that turns final assistant output into active code,
  artifact, preview, comparison, and refinement state
- Multi-artifact generation support for comparing, selecting, refining, and
  previewing multiple creative candidates from one run
- Structured artifact critique metadata for quality ranking, recommended
  candidates, runtime suitability, preview readiness, and refinement guidance
- Artifact comparison and selected-artifact refinement flows that preserve the
  original candidate while adding refined versions
- Controlled preview runtimes for p5.js sketches, Three.js and React Three
  Fiber scene outputs, and GLSL fragment shaders
- Runtime console with live renderer status, FPS/frame telemetry, reload events,
  diagnostics, and error surfacing
- Multimodal image references for grounding creative requests visually
- Project bundle export with artifacts, runtime metadata, retrieval context, and
  session state
- Provider/runtime telemetry, LangSmith trace metadata when available, RAGAs
  evaluation lineage, renderer diagnostics, and subsystem error surfaces
- Local workspace persistence with browser restore and SQLite-backed backend
  session storage

## Creative Workflow

The workstation is designed around an iterative creative coding loop:

`Generate -> Critique -> Compare -> Select -> Refine -> Preview -> Runtime diagnostics`

- Generate: the assistant resolves the effective creative-coding domains,
  retrieves official-source context, assembles the prompt, and streams generated
  output.
- Critique: generated artifacts receive structured scores for prompt alignment,
  creative quality, runtime suitability, code quality, preview readiness, and
  domain appropriateness.
- Compare: multi-artifact runs expose ranked candidates, recommendation
  rationale, runtime support, previewability, and refinement guidance.
- Select: choosing an artifact keeps the code inspector, preview shelf,
  artifacts inspector, and comparison state synchronized.
- Refine: the selected artifact can be sent back with explicit refinement
  instructions while preserving the source candidate and adding a labeled
  refined version.
- Preview: supported artifacts mount in the bottom preview shelf with visible
  play, reload, collapse, and fullscreen controls.
- Runtime diagnostics: the Preview, Runtime, Workflow, Telemetry, Artifacts, and
  Retrieval inspectors expose the runtime state without crowding the visual
  output.

## Architecture

### Frontend Workstation

The primary interface lives in `clients/nextjs/`. The workstation shell
provides:

- a main creative session area for conversation and streaming output
- a lower preview shelf that appears when previewable output is available
- a right-side inspector with `Overview`, `Code`, `Workflow`, `Artifacts`, and
  `Retrieval` tabs
- session, layout, and theme persistence across reloads
- artifact selection, copy, download, and export actions
- local approval checkpoints for runtime resets, destructive actions, and file
  transfer flows

The initial shell opens on a clean creative workspace, then hydrates code,
multiple artifacts, preview, retrieval, and workflow state through live stream
events from the backend bridge.

### Backend Runtime

The Python backend lives under `src/creative_coding_assistant/` and centers on
an implemented LangGraph workflow in
`src/creative_coding_assistant/orchestration/workflow_graph.py`.

The current workflow order is:

`intake -> routing -> memory -> retrieval -> context_assembly -> prompt_input -> prompt_rendering -> generation -> artifact_extraction -> preview_preparation -> artifact_critique -> review -> refinement -> finalization -> failure`

Key backend capabilities include:

- domain-aware routing and request shaping
- curated official-source retrieval and query grounding
- conversation memory and memory recording
- prompt input assembly and rendered provider prompts
- streamed generation, lifecycle, review, retry, artifact, and preview events with workflow metadata
- multi-artifact extraction metadata including source order, default selection,
  runtime, and preview eligibility
- structured artifact critique metadata including per-artifact scores, ranks,
  rationales, recommended candidates, and refinement guidance
- deterministic review checks with at most one refinement retry
- structured terminal failure handling
- live session recording and evaluation support

Architecture documentation for the current workflow graph is available in
[`architecture/workflow_graph.md`](architecture/workflow_graph.md).

### Preview, Runtime, And Safety Model

Preview handling is driven by workstation runtime surfaces and artifact state.
The client routes previewable outputs into controlled runtime adapters rather
than executing arbitrary generated application code directly.

Implemented runtime surfaces include:

- p5.js-style canvas previews
- bounded GLSL fragment shader previews
- controlled Three.js-style WebGL scene previews

The workstation also exposes:

- preview runtime health, FPS, frame-time, and diagnostics overlays
- provider/model/tokens/latency/cost telemetry summaries
- retrieval inspectors with source quality, freshness, and chunk context
- local HITL-style approval flows for export/runtime/reset actions

## Feature Areas

### Creative Workstation

- Streaming conversation workflow designed for iterative creative sessions
- Inspector surfaces for overview, preview metadata, runtime console, code,
  workflow state, telemetry, artifacts, and retrieval
- Live artifact selection, comparison, refinement, and code-focused inspection
- Compact bottom preview shelf that stays out of the chat flow until previewable
  output exists
- Workspace-level session restore with active tab, artifact, preview, layout,
  and preference state

### Artifact Generation And Refinement

- Domain-aware prompt shaping for the effective creative-coding ecosystem
- Multi-artifact extraction with source order, default selection, runtime, and
  preview eligibility metadata
- Structured artifact critique for ranking, recommendation, and refinement
  guidance
- Artifact comparison UI for previewable, code-only, and unsupported-runtime
  candidates
- Selected-artifact refinement that carries source code, critique context,
  runtime metadata, and the user instruction into the next generation request

### Retrieval And Grounding

- Retrieval over curated official documentation sources
- Domain and source metadata registries spanning a broad creative coding tool
  ecosystem
- Retrieval summaries with request parameters, source scoring, freshness, and
  chunk-level grounding context
- Validation coverage for source registry, retrieval foundation, and retrieval
  integration boundaries

### Memory And Evaluation

- Conversation memory repositories and memory retrieval adapters
- Live session recording for later evaluation
- Offline evaluation helpers and RAGAs-oriented evaluation runner
- In-workstation telemetry surfaces for stream events, provider usage,
  workflow runtime, preview health, retrieval quality, optional LangSmith trace
  metadata, and RAGAs evaluation lineage
- Tests covering memory behavior, session evaluation foundations, and live
  session flows

### Export And Multimodal

- Image reference attachments for PNG, JPEG, WebP, and GIF inputs
- Frontend-side validation for attachment size/count/type
- Project bundle export containing:
  - generated artifacts
  - workspace session snapshot
  - workflow summary
  - retrieval summary
  - preview routing/runtime metadata
  - operator approval summary
  - multimodal image metadata
  - bundle manifest and bundled README

## Repository Layout

```text
.
├── architecture/                # Workflow graph docs and Mermaid source
├── assets/                      # README assets
├── clients/
│   ├── nextjs/                  # Primary workstation UI
│   │   ├── src/app/             # Next.js app entrypoints and global styles
│   │   ├── src/components/      # Workstation shell, preview surfaces, callouts
│   │   └── src/lib/             # Frontend runtime models, export, persistence
│   └── streamlit/               # Earlier reference client
├── data/                        # Local runtime data (Chroma, eval, artifacts, SQLite)
├── docs/                        # Local ignored implementation/planning material
├── scripts/                     # KB sync and evaluation utilities
├── src/creative_coding_assistant/
│   ├── api/                     # Streaming and persistence bridge apps
│   ├── app/                     # Top-level service composition
│   ├── artifacts/               # Artifact contracts
│   ├── contracts/               # Assistant request and event contracts
│   ├── domains/                 # Domain registry and metadata
│   ├── eval/                    # Live-session evaluation support
│   ├── llm/                     # Provider adapters
│   ├── orchestration/           # LangGraph runtime and service flow
│   ├── preview/                 # Preview contracts
│   ├── rag/                     # Source registry, sync, embeddings, retrieval
│   ├── vectorstore/             # Chroma persistence helpers
│   └── workspace/               # Workspace session contracts and persistence
├── tests/                       # Backend and bridge tests
├── README.md
└── pyproject.toml
```

## Key Reference Files

- Workflow graph docs:
  [`architecture/workflow_graph.md`](architecture/workflow_graph.md)
- Workflow graph Mermaid source:
  [`architecture/workflow_graph.mmd`](architecture/workflow_graph.mmd)
- Next.js workstation shell:
  [`clients/nextjs/src/components/workstation-shell.tsx`](clients/nextjs/src/components/workstation-shell.tsx)
- Frontend workstation tests:
  [`clients/nextjs/src/components/workstation-shell.test.tsx`](clients/nextjs/src/components/workstation-shell.test.tsx)
- Streaming bridge:
  [`src/creative_coding_assistant/api/streaming.py`](src/creative_coding_assistant/api/streaming.py)
- Workspace persistence bridge:
  [`src/creative_coding_assistant/api/workspace_sessions.py`](src/creative_coding_assistant/api/workspace_sessions.py)
- Workflow integration tests:
  [`tests/test_langgraph_workflow_integration.py`](tests/test_langgraph_workflow_integration.py)

## Domain Coverage

The request/domain registry and approved source registry cover a broad creative
coding surface. Core domains exercised most directly by the current workstation
and preview/runtime flow include:

- Three.js
- React Three Fiber
- p5.js
- GLSL
- Processing
- Canvas 2D
- WebGPU / WGSL

The broader registry also covers ecosystems such as GSAP, Tone.js, PixiJS,
Matter.js, Rapier, Hydra, Shadertoy, TouchDesigner, Houdini, Blender,
openFrameworks, OPENRNDR, SuperCollider, Sonic Pi, TensorFlow.js, ComfyUI,
Runway, Unreal, Unity, and more.

## Setup

### Python Environment

Create and activate a virtual environment, then install the Python project with
dev dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Copy the environment template and fill in local values:

```bash
cp .env.example .env
```

Required for live generation and embeddings:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Useful optional Python settings:

```bash
CCA_OPENAI_API_KEY=
CCA_OPENAI_MODEL=gpt-5-mini
CCA_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CCA_LOG_LEVEL=INFO
CCA_CHROMA_PERSIST_DIR=data/chroma
CCA_EVAL_DATA_PATH=data/eval/live_sessions.jsonl
CCA_WORKSPACE_SESSION_DB_PATH=data/workspace_sessions.sqlite3
```

### Next.js Workstation

Install the frontend dependencies:

```bash
cd clients/nextjs
npm install
```

Optional frontend overrides:

```bash
NEXT_PUBLIC_ASSISTANT_STREAM_URL=http://localhost:8000/api/assistant/stream
NEXT_PUBLIC_WORKSPACE_SESSION_URL=http://localhost:8000/api/workspace/session
```

## Running The System

Start the Next.js workstation:

```bash
cd clients/nextjs
npm run dev
```

Then open:

```text
http://localhost:3000
```

The frontend expects two backend bridge endpoints by default:

- `POST /api/assistant/stream` for assistant NDJSON streaming
- `GET/POST /api/workspace/session` for workspace save/restore

These bridges are exposed as importable WSGI applications in
`creative_coding_assistant.api`:

- `create_assistant_streaming_app`
- `create_workspace_session_app`

Frontend defaults:

```text
http://localhost:8000/api/assistant/stream
http://localhost:8000/api/workspace/session
```

### Streamlit Reference Client

An earlier Streamlit client is still included as a lightweight reference
interface:

```bash
.venv/bin/streamlit run clients/streamlit/app.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Knowledge Base Sync

The retrieval stack depends on a synced local Chroma knowledge base. Sync all
approved sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

Sync selected sources:

```bash
.venv/bin/python scripts/sync_official_kb.py \
  --source-id three_docs \
  --source-id r3f_canvas_api \
  --source-id p5_examples \
  --source-id glsl_mdn_webgl_examples \
  --source-id processing_reference \
  --source-id canvas2d_context_api \
  --source-id webgpu_mdn_api \
  --source-id wgsl_spec
```

## Evaluation

Live sessions are recorded locally for later evaluation. The repository
includes offline evaluation helpers today, including a retrieval-focused
RAGAs-oriented runner. In-app evaluation dashboards and richer RAGAs/observability
surfacing are intentionally separate concerns and belong to a later
observability layer.

Evaluate the latest eligible samples:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path data/eval/live_sessions.jsonl \
  --output-path data/eval/ragas_latest4_context_precision.jsonl \
  --latest 4 \
  --metric context_precision
```

Or use the helper:

```bash
scripts/run_eval_latest.sh 4
```

## Validation

Python checks:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src clients tests scripts
.venv/bin/python -m compileall -q src clients tests scripts
```

Frontend checks:

```bash
cd clients/nextjs
npm run typecheck
npm run lint
npm run test
npm run build
```
