# Creative Coding Assistant

Creative Coding Assistant is an AI-native creative translation workstation for
professional creative coding workflows. It combines a LangGraph-orchestrated
Python backend with a Next.js workstation for audio, visual, and audiovisual
generation, comparison, refinement, preview, observability, validation, and
export preparation.

The current product scope is a creative coding platform and workstation rather
than a generic chat assistant. Requests can be translated from intent, symbols,
and concepts into structured creative guidance, grounded with retrieval when
useful, expanded into multiple candidate artifacts, compared and refined, then
observed through live preview and runtime inspection from the same interface.

![Creative Coding Assistant workstation preview](assets/preview.png)

## Highlights

- AI-native workstation UX with chat, a compact bottom preview shelf, and
  focused inspector tabs for overview, preview, runtime, code, workflow,
  telemetry, artifacts, and retrieval
- Creative Translation Engine that converts user intent into bounded modality,
  symbolism, geometry, mood, movement, runtime, and refinement guidance
- Sacred Geometry Prompt Layer for explicit concepts such as mandalas, yantras,
  torus forms, spirals, Fibonacci structures, and related geometric motifs
- Multi-artifact generation, critique, comparison, and selected-artifact
  refinement inside one continuous workstation flow
- Controlled live runtimes for p5.js, Three.js, React Three Fiber, GLSL, Hydra,
  and Tone.js outputs
- Runtime diagnostics, provider observability, workflow timeline inspection,
  retrieval intelligence, evaluation surfaces, and creative cost visibility
- Multimodal image references, local session persistence, and project bundle
  export

## Creative Workflow

The workstation is designed around the current V2 creative loop:

`Intent / symbol / concept -> Creative Translation -> Generate -> Compare -> Refine -> Preview -> Observe -> Validate -> Export`

- Creative Translation: the assistant derives bounded intent, symbolic,
  geometric, mood, movement, runtime, and refinement guidance before
  generation.
- Generate: the backend resolves effective domains, retrieves official-source
  context when useful, assembles the prompt, and streams creative output.
- Compare: multi-artifact runs expose ranked candidates, recommendation
  rationale, runtime support, previewability, and refinement guidance.
- Refine: the selected artifact can be sent back with explicit refinement
  instructions while preserving the source candidate and adding a labeled
  refined version.
- Preview: supported artifacts mount in the bottom preview shelf with visible
  play, reload, collapse, and fullscreen controls.
- Observe: the Preview, Runtime, Workflow, Telemetry, Artifacts, and Retrieval
  inspectors expose runtime, workflow, cost, and grounding state without
  crowding the visual output.
- Validate: critique scores, runtime diagnostics, retrieval quality, evaluation
  traces, and workflow review state help verify the result before export.
- Export: the current export surface is the project bundle workflow. Advanced
  Blueprint Export and pro DAW/runtime pipeline exports remain roadmap items.

## Implemented Capabilities

### Creative Core

- Creative Translation Engine with deterministic prompt guidance
- Sacred Geometry Prompt Layer integrated into the existing translation flow
- Domain-aware generation across supported preview runtimes and broader
  creative-coding knowledge domains
- Multi-artifact generation, artifact critique, artifact comparison, and
  selected-artifact refinement
- Multimodal image references for visually grounded generation requests

### Runtime And Preview

- p5.js live runtime
- Three.js live runtime
- React Three Fiber live runtime
- GLSL live runtime
- Hydra live runtime
- Tone.js live runtime
- Runtime console and preview runtime diagnostics with lifecycle, error, and
  renderer telemetry surfaces

### Retrieval, Evaluation, And Observability

- Retrieval Inspector Advanced and Retrieval Source Explorer
- Retrieval Quality Deep Dive and KB Source Health Dashboard
- Provider Observability Deep Dive
- Evaluation Session Dashboard
- Workflow Timeline Explorer
- LangSmith Trace Deep Dive when trace metadata is available
- Creative Cost Intelligence Dashboard
- Local workspace persistence and project bundle export

## Architecture

### Frontend Workstation

The primary interface lives in `clients/nextjs/`. The workstation shell
provides:

- a main creative session area for conversation and streaming output
- a lower preview shelf that appears when previewable output is available
- a right-side inspector with `Overview`, `Preview`, `Runtime`, `Code`,
  `Workflow`, `Telemetry`, `Artifacts`, and `Retrieval` tabs
- session, layout, and theme persistence across reloads
- artifact selection, comparison, refinement, copy, download, and export actions
- runtime console surfaces for renderer lifecycle, diagnostics, reloads, and
  frame telemetry
- telemetry panels for provider usage, workflow runtime, preview health,
  retrieval quality, LangSmith metadata, and RAGAs evaluation lineage
- local operator approvals for runtime resets, destructive actions, and file
  transfer flows

The initial shell opens on a clean creative workspace, then hydrates code,
multiple artifacts, comparison, refinement, preview, runtime, retrieval,
telemetry, and workflow state through live stream events from the backend
bridge.

### Backend Runtime

The Python backend lives under `src/creative_coding_assistant/` and centers on
an implemented LangGraph workflow in
`src/creative_coding_assistant/orchestration/workflow_graph.py`.

The current workflow order is:

`intake -> routing -> memory -> retrieval -> context_assembly -> prompt_input -> prompt_rendering -> generation -> artifact_extraction -> preview_preparation -> artifact_critique -> review -> refinement -> finalization -> failure`

Key backend capabilities include:

- domain-aware routing, generation-domain resolution, and request shaping
- curated official-source retrieval and query grounding
- conversation memory and memory recording
- prompt input assembly and rendered provider prompts
- streamed generation, lifecycle, review, retry, artifact, and preview events
  carrying workflow and telemetry metadata
- multi-artifact extraction metadata including source order, default selection,
  runtime, and preview eligibility
- structured artifact critique metadata including per-artifact scores, ranks,
  rationales, recommended candidates, and refinement guidance
- deterministic review checks with at most one refinement retry
- selected-artifact refinement context threaded into request contracts, prompt
  inputs, and rendered prompts
- structured terminal failure handling
- live session recording, optional LangSmith runtime metadata, and offline
  evaluation support

Architecture documentation for the current workflow graph is available in
[`architecture/workflow_graph.md`](architecture/workflow_graph.md).

### Preview, Runtime, And Safety Model

Preview handling is split between the backend graph and the frontend runtime
surfaces. The LangGraph workflow owns artifact extraction and preview metadata
preparation. The Next.js workstation then routes previewable outputs into
controlled runtime adapters rather than executing arbitrary generated
application code directly.

Current live preview/runtime support includes:

- p5.js sketches through a controlled p5-compatible 2D canvas runtime
- Three.js scenes through a controlled Three-compatible WebGL runtime
- React Three Fiber artifacts routed through the Three-compatible preview
  surface when they match the supported browser runtime contract
- GLSL fragment shaders through a bounded WebGL shader runtime
- Hydra live-coded visuals through a bounded Hydra-compatible browser runtime
- Tone.js audio outputs through a controlled user-activated Web Audio runtime

The workstation also exposes:

- preview runtime health, FPS, frame-time, and diagnostics overlays
- a runtime console for lifecycle events, reload requests, renderer errors, and
  latest runtime messages
- provider/model/tokens/latency/cost telemetry summaries
- retrieval inspectors with source quality, freshness, and chunk context
- artifact comparison rows with quality rank, recommendation state, runtime
  support, and preview actions
- local operator approval flows for export/runtime/reset actions

## Feature Areas

### Workstation

- Streaming conversation workflow designed for iterative creative sessions
- Inspector surfaces for overview, preview metadata, runtime console, code,
  workflow state, telemetry, artifacts, and retrieval
- Live artifact selection, comparison, refinement, and code-focused inspection
- Compact bottom preview shelf that stays out of the chat flow until previewable
  output exists
- Workspace-level session restore with active tab, artifact, preview, layout,
  and preference state

### Generation, Retrieval, And Evaluation

- Creative Translation prompt shaping for the effective creative-coding
  ecosystem
- Multi-artifact extraction with source order, default selection, runtime, and
  preview eligibility metadata
- Structured artifact critique for ranking, recommendation, and refinement
  guidance
- Retrieval over curated official documentation sources with chunk-level
  grounding context
- Live session recording, offline evaluation helpers, and RAGAs-oriented
  evaluation runner

### Export

- Image reference attachments for PNG, JPEG, WebP, and GIF inputs
- Frontend-side validation for attachment size/count/type
- Project bundle export containing generated artifacts, workspace session state,
  workflow summary, retrieval summary, preview/runtime metadata, operator
  approval summary, multimodal image metadata, and a bundle manifest

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
coding surface. Current live preview/runtime support includes:

- p5.js
- Three.js
- React Three Fiber
- GLSL
- Hydra
- Tone.js

The broader generation and retrieval domain registry also covers code-oriented
or documentation-grounded support for:

- Processing
- Canvas 2D
- WebGPU / WGSL
- GSAP
- PixiJS
- Matter.js
- Rapier
- Shadertoy
- Houdini
- Blender
- openFrameworks
- OPENRNDR
- SuperCollider
- Sonic Pi
- TensorFlow.js
- ComfyUI
- Runway
- Unreal
- Unity
- additional curated creative-coding ecosystems

Those broader domains are available for routing, prompt guidance, retrieval, or
code inspection depending on available source coverage. They should not be read
as live browser preview runtimes unless they are listed in the current live
runtime support list above.

## Roadmap

### High-Priority Creative Core

- Advanced Shader Presets
- Visual Style System
- GSAP Runtime Support
- SVG / Canvas Runtime Expansion
- Multi-Preview Comparison Workspace
- Dynamic Param Control Panel
- Audio-Reactive Visual Engine
- Creative Quality Critic
- Sacred Consistency Evaluator

### Later Product And Pro Workflow

- Export Bundle Enhancements
- Performance Blueprint Export
- MIDI / OSC Mapping Export
- Session Timeline Explorer
- Prompt Evolution Explorer
- Artifact Lineage Graph
- Final UI Polish
- Audit & Refactor
- V2 Freeze
- Local Model Provider Support
- Hybrid Model Router
- HoloMind Integration Bridge after HoloMind V1

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
RAGAs-oriented runner. The workstation can surface RAGAs lineage and optional
LangSmith metadata when that data is present; richer in-app RAGAs scoring
dashboards remain planned for a later observability layer.

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
