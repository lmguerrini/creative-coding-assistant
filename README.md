# Creative Coding Assistant

Creative Coding Assistant is a V3.1 AI-native creative translation
workstation designed to transform symbolic, conceptual, geometric, stylistic,
and multimodal intent into professional audio, visual, and audiovisual
creative systems.

It combines a LangGraph-orchestrated Python backend with a Next.js workstation
for the full V2.5 Creative Core, the V3.1 Creative Cognition Core,
multi-artifact generation, multi-preview comparison, controlled runtime
preview, critique and scoring, parameter control, observability, validation,
and export preparation.

The current product scope is a creative coding platform rather than a generic
chat assistant. Requests can be translated from intent, symbols, geometry,
style, shader language, motion systems, audio-reactive mappings, and visual
references into structured creative guidance, grounded with retrieval when
useful, expanded into multiple candidate artifacts, compared, critiqued,
refined, and observed through live preview and runtime inspection from the
same interface.

![Creative Coding Assistant](assets/preview_current.png)

## Highlights

- AI-native workstation UX with chat, a compact bottom preview shelf, and
  focused inspector tabs for overview, preview, runtime, code, workflow,
  telemetry, artifacts, and retrieval
- Complete V2.5 Creative Core plus the implemented V3.1 Creative Cognition
  Core for intent decomposition, hierarchy, strategy, technique, constraints,
  runtime reasoning, trade-offs, quality prediction, narrative, composition,
  and creative reasoning
- Multi-artifact generation, multi-preview comparison, dynamic parameter
  control, and HITL candidate selection inside one continuous workstation flow
- Controlled live runtimes for p5.js, Three.js, React Three Fiber, GLSL,
  Hydra, Tone.js, GSAP, SVG, and Canvas outputs
- Runtime diagnostics, provider observability, workflow timeline inspection,
  retrieval intelligence, evaluation surfaces, and creative cost visibility
- Multimodal image references, local session persistence, and project bundle
  export

## Current Branch Status

The current branch, `feature/creative-intelligence-core`, implements the
V3.1 Creative Cognition Core.

V3.1 adds bounded internal metadata and reasoning layers that run before
generation and feed planning, Director guidance, Creative Reasoning synthesis,
and prompt rendering. These capabilities are inspectable and typed, but they
are not all separate LangGraph runtime nodes.

Implemented V3.1 capabilities:

- Creative Intent Decomposer
- Creative Hierarchy Planner
- Creative Strategy Engine
- Creative Technique Selector
- Creative Constraint Solver
- Runtime Capability Reasoner
- Creative Trade-off Explorer
- Creative Constraint Prioritizer
- Creative Quality Predictor
- Symbolic Narrative Planner
- Creative Composition Planner
- Creative Reasoning Engine

The architecture intentionally documents two layers:

- the real runtime graph in
  [`architecture/workflow_graph.md`](architecture/workflow_graph.md)
- the internal Creative Intelligence capability graph in
  [`architecture/creative_intelligence_graph.md`](architecture/creative_intelligence_graph.md)

The runtime graph remains compact and truthful. The internal capability graph
documents the V3.1 sub-capabilities and acts as the blueprint for future V4
multi-agent decomposition rather than claiming that V4 already exists.

## Creative Workflow

The workstation still centers on the V2.5 creative loop, now enriched by the
V3.1 Creative Cognition Core before generation:

`Intent -> HITL Clarification -> Creative Translation -> Reference Fusion -> Creative Planning -> Generation -> Preview -> Critique -> Sacred Consistency -> Calibrated Quality -> Multi-Pass Refinement -> Export Preparation`

- HITL Clarification: the assistant can ask targeted questions to lock
  modality, intent, symbolism, geometry, runtime, or refinement direction
  before generation.
- Creative Translation: the assistant derives bounded modality, symbolic,
  geometric, mood, movement, runtime, visual style, shader, audio-reactive,
  and refinement guidance.
- Reference Fusion: multimodal image references and retrieved context are fused
  into the structured creative brief when they improve grounding.
- Creative Planning: the planning layer organizes generation strategy, runtime
  fit, and artifact expectations before provider execution.
- Creative Cognition Core: deterministic intent, hierarchy, strategy,
  technique, constraints, runtime, trade-off, quality, narrative, composition,
  Director, and Reasoning layers refine the final creative brief without
  inflating the public runtime graph into dozens of nodes.
- Generation: the backend resolves effective domains, assembles the prompt,
  and streams one or more creative artifacts.
- Preview: supported artifacts mount in controlled preview runtimes with
  visible play, reload, collapse, fullscreen, start, stop, or mute controls
  depending on runtime type.
- Critique: generated artifacts receive structured quality review, ranking,
  recommendation, and refinement guidance.
- Sacred Consistency: symbolic, geometric, and style coherence can be checked
  as a distinct evaluation layer.
- Calibrated Quality: critique, consistency, runtime health, and workflow
  review metadata are combined into a more legible decision surface.
- Multi-Pass Refinement: selected artifacts can be refined through iterative
  review with parameter control and explicit refinement serialization.
- Export Preparation: the current export surface is the project bundle
  workflow. Blueprint Export, MIDI / OSC, lineage, and live performance
  workflows remain future roadmap items.

## Implemented Capabilities

### V3.1 Creative Cognition Core

- Creative Intent Decomposer for atomic creative-intent dimensions
- Creative Hierarchy Planner for ranked creative priorities
- Creative Strategy Engine for high-level artistic strategy
- Creative Technique Selector for bounded implementation technique guidance
- Creative Constraint Solver for intent/runtime/safety/performance/cost/HITL
  tensions
- Runtime Capability Reasoner for supported runtime fit evaluation
- Creative Trade-off Explorer for structured creative-versus-technical
  consequences
- Creative Constraint Prioritizer for non-negotiable, flexible, and
  sacrificial constraint tiers
- Creative Quality Predictor for pre-generation readiness and likely failure
  modes
- Symbolic Narrative Planner for symbolic arc and phase structure
- Creative Composition Planner for focal hierarchy, density, balance, and
  transition structure
- Creative Reasoning Engine for final synthesis of the stored V3.1 metadata

These V3.1 capabilities are internal metadata/reasoning layers. They enrich the
existing workflow state, Director, Reasoning, and rendered prompt, but they do
not each become a separate LangGraph node in the runtime graph.

### Creative Core

- Creative Translation Engine with deterministic prompt guidance
- Sacred Geometry Prompt Layer integrated into the existing translation flow
- Advanced Shader Presets for glow, aura, plasma, bloom-like emission,
  refraction, glass, volumetric atmosphere, fractal fields, kaleidoscopic
  symmetry, and sacred light aesthetics
- Visual Style System with deterministic style guidance and artifact/refinement
  persistence
- Reference Fusion Layer for multimodal image references and grounded context
  incorporation
- Creative Planning Agent for generation strategy, runtime fit, and artifact
  shaping
- Advanced HITL Questioning Engine for targeted clarification before generation
- Audio-Reactive Visual Engine for bounded relationships such as
  amplitude-to-glow, bass-to-pulse, rhythm-to-rotation, and drone-to-atmosphere
- Creative Quality Critic for structured artifact ranking, recommendation, and
  refinement guidance
- Sacred Consistency Evaluator for symbolic, geometric, and style-coherence
  review
- Calibrated Quality Scoring for combining critique, consistency, and runtime
  review into clearer selection signals
- Domain-aware generation across supported preview runtimes and broader
  creative-coding knowledge domains
- Multi-artifact generation, artifact critique, artifact comparison, and
  selected-artifact refinement
- Multi-Preview Comparison Workspace for HITL selection across runtime-capable
  visual, audio, audiovisual, and code-only candidates
- Dynamic Parameter Control Panel with preview-safe local edits and explicit
  refinement serialization
- Multi-Pass Creative Refinement with structured review feedback threaded back
  into generation and refinement
- Multimodal image references for visually grounded generation requests

### Runtime And Preview

- p5.js live runtime
- Three.js live runtime
- React Three Fiber live runtime
- GLSL live runtime
- Hydra live runtime
- Tone.js live runtime
- GSAP live runtime
- SVG live runtime
- Canvas live runtime
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

## Creative Core Layers

The assistant progressively enriches user requests through bounded translation
layers before and after generation. These layers are deterministic, additive,
and metadata-driven rather than separate generation pipelines.

Current layers include:

- Creative Translation Engine for modality, intent, mood, movement, runtime,
  and refinement guidance
- Sacred Geometry Prompt Layer for bounded symbolic and geometric generation
  cues
- Advanced Shader Presets for practical visual effect guidance
- Visual Style System for coherent aesthetic identity across artifacts and
  refinements
- Reference Fusion Layer for multimodal grounding and context-aware creative
  shaping
- Creative Planning Agent for pre-generation strategy and runtime fit
- Audio-Reactive Visual Engine for audiovisual relationships between sound
  features and visual behavior
- Runtime Recommendation Layer for matching generated artifacts to supported
  preview/runtime surfaces
- Creative Quality Critic, Sacred Consistency Evaluator, and Calibrated
  Quality Scoring for structured post-generation review
- Dynamic Parameter Control Layer for artifact-specific local controls and
  parameter-guided refinement
- Multi-Pass Creative Refinement for iterative improvement across generation
  and refinement cycles

These layers allow the app to move beyond direct prompt-to-code generation and
toward a structured creative translation workflow.

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

`intake -> routing -> memory -> retrieval -> context_assembly -> prompt_input -> planning -> director -> reasoning -> prompt_rendering -> generation -> artifact_extraction -> preview_preparation -> artifact_critique -> review -> refinement -> finalization -> failure`

Key backend capabilities include:

- domain-aware routing, generation-domain resolution, and request shaping
- curated official-source retrieval and query grounding
- conversation memory and memory recording
- prompt input assembly, deterministic creative planning, bounded Director
  guidance, synthesized creative reasoning, and rendered provider prompts
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
[`architecture/workflow_graph.md`](architecture/workflow_graph.md). The
corresponding internal Creative Intelligence capability graph is documented in
[`architecture/creative_intelligence_graph.md`](architecture/creative_intelligence_graph.md).

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
- GSAP motion studies through a bounded DOM motion runtime
- SVG artifacts through a sanitized inline SVG runtime surface
- Canvas artifacts through a bounded Canvas 2D runtime surface

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
├── architecture/                # Runtime graph docs plus internal capability graph docs
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
- Creative Intelligence graph docs:
  [`architecture/creative_intelligence_graph.md`](architecture/creative_intelligence_graph.md)
- Creative Intelligence Mermaid source:
  [`architecture/creative_intelligence_graph.mmd`](architecture/creative_intelligence_graph.mmd)
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
- GSAP
- SVG
- Canvas

The broader generation and retrieval domain registry also covers code-oriented
or documentation-grounded support for:

- Processing
- WebGPU / WGSL
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

### Program Increment Versioning

V3 now follows Program Increment style versioning rather than one long,
undifferentiated milestone.

- `v3.1.0`: Creative Cognition Core
- `v3.2.0`: next macro-capability family after V3.1 closeout
- `v3.3.0`: subsequent macro-capability family after V3.2 closeout

### V3.1.0 Status

- V2.5 remains the base workstation/runtime milestone.
- V3.1.0 adds the implemented Creative Cognition Core on the current branch.
- The runtime graph remains the compact LangGraph workflow documented in
  `architecture/workflow_graph.md`.
- The internal Creative Intelligence graph documents the V3.1 sub-capabilities
  in `architecture/creative_intelligence_graph.md`.
- That internal graph is the blueprint for future V4 multi-agent work, but it
  is not yet a V4 runtime implementation.

### Macro-Capability Lifecycle

Each V3 macro-capability is expected to close through the same bounded
lifecycle:

1. Implementation
2. Architecture Update
3. Documentation Update
4. Capability Validation
5. Create Version Tag
6. Merge & Push

### Continuing Direction

- close `v3.1.0` cleanly before opening a larger new increment
- keep the runtime graph truthful instead of mapping every internal helper to a
  public runtime node
- keep using bounded internal creative-intelligence metadata as the bridge
  toward future V4 multi-agent decomposition
- continue later V3 increments around deeper creative intelligence,
  export/performance workflows, and supporting platform improvements

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
