# Creative Coding Assistant

Creative Coding Assistant is an AI-native creative translation workstation for
the completed V3 platform, designed to transform symbolic, conceptual,
geometric, stylistic, and multimodal intent into professional audio, visual,
and audiovisual creative systems.

It combines a LangGraph-orchestrated Python backend with a Next.js workstation
for the full V2.5 Creative Core, the V3.1 Creative Cognition Core, the V3.2
Generative Design Core, the V3.3 Artifact Intelligence stack, and the V3.4
Creative Evaluation layer. V3.5 adds the Creative Workstation surface layer:
workstation state, session intelligence, workflow exploration, provenance,
creative timeline, V3 inspector panels, dashboard summaries, and surface
contracts over existing metadata. The product also includes multi-artifact
generation, multi-preview comparison, controlled runtime preview, critique and
scoring, parameter control, observability, validation, and export preparation.
The stabilized V3 surface hardens workflow graph assembly, stream and payload
boundaries, local backend mounting, shared orchestration utilities, and
documentation alignment without expanding generation behavior or
future-version runtime scope.

The product scope is a creative coding platform rather than a generic chat
assistant. Requests can be translated from intent, symbols, geometry, style,
shader language, motion systems, audio-reactive mappings, and visual references
into structured creative guidance, grounded with retrieval when useful,
expanded into multiple candidate artifacts, compared, critiqued, refined, and
observed through live preview and runtime inspection from the same interface.

![Creative Coding Assistant](assets/preview_current.png)

## Highlights

- AI-native workstation UX with chat, a compact bottom preview shelf, and
  focused inspector tabs for overview, preview, runtime, code, workflow,
  telemetry, artifacts, and retrieval
- Complete V2.5 Creative Core plus V3.1 Creative Cognition, V3.2 Generative
  Design, V3.3 Artifact Intelligence, and V3.4 Creative Evaluation metadata for
  intent decomposition, hierarchy, strategy, technique, constraints, runtime
  reasoning, trade-offs, quality prediction, narrative, composition,
  procedural structure, generative structure, motifs, emotional consistency,
  cross-modality, scene design, artifact planning, compatibility, critique,
  refinement, merge planning, export intelligence, confidence, scoring,
  consistency validation, reports, and engine contracts
- V3.5 Creative Workstation surfaces for shared workstation state, session
  intelligence, workflow exploration, provenance, creative timeline, V3
  inspector panels, dashboard cards, and metadata-only surface contracts
- Multi-artifact generation, multi-preview comparison, dynamic parameter
  control, and HITL candidate selection inside one continuous workstation flow
- Controlled live runtimes for p5.js, Three.js, React Three Fiber, GLSL,
  Hydra, Tone.js, GSAP, SVG, and Canvas outputs
- Runtime diagnostics, provider observability, workflow timeline inspection,
  retrieval intelligence, evaluation surfaces, and creative cost visibility
- Multimodal image references, local session persistence, and project bundle
  export

## Capability Scope

Artifact Intelligence extends the bounded planning pass with inspectable
compatibility, critique/refinement, merge, export, and engine-contract metadata
that feed Director guidance, Creative Reasoning synthesis, prompt rendering,
workflow serialization, and Next.js stream hydration. These capabilities remain
internal helpers inside the existing runtime graph rather than separate
LangGraph runtime nodes.

Creative Evaluation extends the same bounded metadata pattern into critic,
self-evaluation, improvement, reflection, confidence, score, consistency,
report, and evaluation engine-contract metadata. These summaries are serialized
for inspection without changing evaluation logic, workflow ordering, routing,
runtime selection, previews, retries, or generated output.

Creative Workstation turns existing workflow, artifact, evaluation,
provenance, and session metadata into inspectable product surfaces. Workstation
state, session intelligence, workflow explorer, provenance engine, creative
timeline, V3 inspector panels, dashboard layer, and workstation contracts make
the metadata easier to review without changing generation behavior or adding
future V4/V5/V6 systems.

Artifact Intelligence capabilities:

- Artifact Planner
- Artifact Dependency Graph
- Runtime Compatibility Engine
- Artifact Capability Matrix
- Multi-Artifact Strategy
- Artifact Critic
- Artifact Refiner
- Artifact Intelligence Synthesis
- Artifact Merge Planner
- Artifact Export Intelligence
- Artifact Engine Contracts

Creative Evaluation capabilities:

- Creative Critic Engine
- Self Evaluation Engine
- Creative Improvement Planner
- Reflection Loop Engine
- Creative Confidence Engine
- Creative Score Engine
- Consistency Validation Engine
- Evaluation Reports
- Evaluation Engine Contracts

Creative Workstation surfaces:

- Workstation State
- Session Intelligence
- Workflow Explorer
- Provenance Engine
- Creative Timeline
- V3 Inspector Panels
- Workstation Dashboard
- Workstation Engine Contracts

The architecture documents six complementary views:

- the real runtime graph in
  [`architecture/workflow_graph.md`](architecture/workflow_graph.md)
- the human-readable Creative Intelligence pipeline in
  [`architecture/creative_intelligence_graph.md`](architecture/creative_intelligence_graph.md)
- the V3.2 Generative Design dependency graph in
  [`architecture/generative_design_graph.md`](architecture/generative_design_graph.md)
- the V3.3 Artifact Intelligence dependency graph in
  [`architecture/artifact_intelligence_graph.md`](architecture/artifact_intelligence_graph.md)
- the V3.5 Creative Workstation surface graph in
  [`architecture/workstation_surface_graph.md`](architecture/workstation_surface_graph.md)
- the cross-cutting engine matrix in
  [`architecture/engine_matrix.md`](architecture/engine_matrix.md)

The runtime graph remains compact and truthful. The internal pipeline and
dependency views remain blueprints for future V4 Agentic Studio decomposition
rather than claims that a V4 multi-agent runtime already exists.

## Creative Workflow

The workstation still centers on the V2.5 creative loop, now enriched by the
V3.1 Creative Cognition Core, V3.2 Generative Design Core, V3.3 Artifact
Intelligence, V3.4 Creative Evaluation metadata, and V3.5 Creative
Workstation inspection surfaces:

`Intent -> HITL Clarification -> Creative Translation -> Reference Fusion -> Creative Planning -> Generation -> Preview -> Critique -> Creative Evaluation -> Calibrated Quality -> Multi-Pass Refinement -> Export Preparation`

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
- Generative Design Core: procedural structure, generative systems, semantic
  motifs, emotional continuity, cross-modality alignment, and scene scaffolds
  extend the stored creative brief as metadata and design guidance.
- Artifact Intelligence: artifact planning, dependency mapping, runtime
  compatibility, capability fit, multi-artifact strategy, artifact critique,
  refinement guidance, synthesis, merge planning, export intelligence, and
  engine contracts extend the stored brief as metadata. Artifact profiles can
  inform Director, Reasoning, and prompt rendering; engine contracts remain
  workflow/stream metadata and are not rendered into provider prompts.
- Creative Evaluation: critic, self-evaluation, improvement, reflection,
  confidence, score, consistency, report, and evaluation contract metadata
  summarize quality signals without changing routing, runtime selection,
  previews, retries, or generated output.
- Creative Workstation: state, session intelligence, workflow explorer,
  provenance, timeline, inspector panels, dashboard cards, and surface
  contracts make existing metadata reviewable without changing backend
  generation behavior.
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

### V3.2 Generative Design Core

- Procedural Structure Planner for selecting bounded structural systems and
  layout grammars before generation
- Generative Structure Engine for deriving inspectable generative-system
  blueprints from upstream creative intent and composition metadata
- Semantic Motif Engine for recurring symbolic motif systems, recurrence rules,
  and thematic anchors
- Emotional Consistency Engine for emotional tone continuity across palette,
  motion, composition pressure, and atmosphere
- Cross-Modality Composer for alignment across audio, visual, motion, shader,
  and camera behaviors
- Audio-Visual Scene System for phase, cue, transition, climax, and timing
  scaffolds across complex audiovisual outputs

These V3.2 capabilities are internal metadata/design-guidance layers. They
enrich the stored workflow state, Director, Reasoning, and rendered prompt, but
they do not become separate LangGraph runtime nodes or runtime auto-selection
logic.

### V3.3 Artifact Intelligence

- Artifact Planner for artifact intent, type, family, requirements, risks, and
  prompt guidance before generation
- Artifact Dependency Graph for metadata nodes, dependency edges, upstream
  requirements, downstream consumers, and dependency risks
- Runtime Compatibility Engine for advisory runtime compatibility,
  portability, interoperability, limitations, and implementation risks
- Artifact Capability Matrix for target strengths, weaknesses, export fit,
  portability fit, interoperability fit, and capability risks
- Multi-Artifact Strategy for primary/supporting artifacts, sequencing,
  priorities, groups, and handoff points
- Artifact Critic for pre-generation metadata strengths, weaknesses, runtime
  concerns, risk assessment, and HITL questions
- Artifact Refiner for metadata-level refinement focus, recommended
  improvements, risk reductions, and trade-off notes
- Artifact Intelligence Synthesis for implementation readiness, risk,
  recommended artifact path, coordination notes, and prompt guidance
- Artifact Merge Planner for merge strategy, artifact boundaries, join points,
  integration order, rejected merge paths, and merge risks
- Artifact Export Intelligence for advisory export targets, formats,
  readiness, constraints, documentation requirements, downstream handoffs, and
  rejected export paths
- Artifact Engine Contracts for the shared metadata contract registry across
  all ten Artifact Intelligence engines

These V3.3 capabilities are internal metadata/artifact-guidance layers. They
enrich workflow state, prompt input metadata, Director, Reasoning, rendered
artifact profile sections, final payload serialization, and Next.js stream
hydration. They are metadata-only: they do not execute artifacts, modify
artifacts, export artifacts, select runtimes, change provider routing, change
previews, trigger retries, or implement future V4/V5/V6 systems.

### V3.4 Creative Evaluation

- Creative Critic Engine for metadata-only strengths, weaknesses, quality
  scores, risk assessment, missing information, and HITL questions
- Self Evaluation Engine for generated-response alignment, completeness,
  ambiguity, risks, gaps, and advisory prompt guidance
- Creative Improvement Planner for metadata-level improvement priorities,
  opportunities, low-risk improvements, trade-offs, and future candidates
- Reflection Loop Engine for theoretical reflection value, unresolved
  questions, refinement candidates, stop conditions, and HITL posture
- Creative Confidence Engine for component confidence, uncertainties,
  reliability, execution readiness, and review need
- Creative Score Engine for advisory scoring, score breakdown, calibration,
  explainability, penalties, strengths, weaknesses, and evidence
- Consistency Validation Engine for consistency status, conflicts,
  contradiction level, unsupported conclusions, integrity, and HITL posture
- Evaluation Reports for executive, quality, confidence, consistency,
  improvement, score, trace, provenance, dependency, evidence, and
  recommendation metadata
- Evaluation Engine Contracts for the shared metadata contract registry across
  all eight Creative Evaluation engines

These V3.4 capabilities are internal metadata/evaluation layers. They enrich
workflow state, prompt input metadata, Director and Reasoning metadata, final
payload serialization, and Next.js stream hydration. They are metadata-only:
they do not change evaluation logic, scoring, confidence, reflection, reports,
prompts, routing, runtime selection, previews, retries, or future V4/V5/V6
systems.

### V3.5 Creative Workstation

- Workstation State for shared session, current run, selection, panel,
  readiness, and metadata status
- Session Intelligence for session readiness, completion status, warnings, and
  recommended operator actions
- Workflow Explorer for workflow nodes, edges, active step, progress, and
  runtime status
- Provenance Engine for evidence, dependency, artifact, evaluation, final
  payload, and missing-source visibility
- Creative Timeline for ordered request, planning, retrieval, creative,
  artifact, evaluation, and final metadata stages
- V3 Inspector Panels for creative intelligence, generative design, artifact
  intelligence, creative evaluation, and provenance records
- Workstation Dashboard for quality, confidence, consistency, artifact
  readiness, runtime fit, evaluation, workflow health, and HITL summary cards
- Workstation Engine Contracts for the metadata-only V3.5 surface registry and
  future V4/V5/V6 hook names

These V3.5 surfaces are product inspection layers over existing workflow,
stream, artifact, evaluation, and session metadata. They are metadata-only:
they do not change provider routing, runtime selection, workflow ordering,
preview execution, artifact execution, artifact modification, retries,
autonomous decisions, or generated output.

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
- workstation state, session intelligence, workflow explorer, provenance,
  creative timeline, V3 inspector panels, and dashboard summaries over the
  existing stream and workspace metadata
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

Workflow node order:

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
corresponding internal Creative Intelligence pipeline, V3.2 Generative Design
dependency graph, V3.3 Artifact Intelligence dependency graph, V3.5
workstation surface graph, and engine matrix are documented in
[`architecture/creative_intelligence_graph.md`](architecture/creative_intelligence_graph.md),
[`architecture/generative_design_graph.md`](architecture/generative_design_graph.md),
[`architecture/artifact_intelligence_graph.md`](architecture/artifact_intelligence_graph.md),
[`architecture/workstation_surface_graph.md`](architecture/workstation_surface_graph.md),
and [`architecture/engine_matrix.md`](architecture/engine_matrix.md).

### Preview, Runtime, And Safety Model

Preview handling is split between the backend graph and the frontend runtime
surfaces. The LangGraph workflow owns artifact extraction and preview metadata
preparation. The Next.js workstation then routes previewable outputs into
controlled runtime adapters rather than executing arbitrary generated
application code directly.

Live preview/runtime support includes:

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
- provenance, timeline, V3 metadata inspector panels, and workstation dashboard
  cards for reviewing generated artifacts and evaluation metadata
- artifact comparison rows with quality rank, recommendation state, runtime
  support, and preview actions
- local operator approval flows for export/runtime/reset actions

## Feature Areas

### Workstation

- Streaming conversation workflow designed for iterative creative sessions
- Inspector surfaces for overview, preview metadata, runtime console, code,
  workflow state, telemetry, artifacts, and retrieval
- Workstation state, session intelligence, workflow explorer, provenance,
  timeline, V3 inspector panels, and dashboard summaries for reviewing existing
  metadata without changing generation behavior
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
- Generative Design graph docs:
  [`architecture/generative_design_graph.md`](architecture/generative_design_graph.md)
- Generative Design Mermaid source:
  [`architecture/generative_design_graph.mmd`](architecture/generative_design_graph.mmd)
- Artifact Intelligence graph docs:
  [`architecture/artifact_intelligence_graph.md`](architecture/artifact_intelligence_graph.md)
- Artifact Intelligence Mermaid source:
  [`architecture/artifact_intelligence_graph.mmd`](architecture/artifact_intelligence_graph.mmd)
- Workstation surface graph docs:
  [`architecture/workstation_surface_graph.md`](architecture/workstation_surface_graph.md)
- Workstation surface Mermaid source:
  [`architecture/workstation_surface_graph.mmd`](architecture/workstation_surface_graph.mmd)
- Engine matrix:
  [`architecture/engine_matrix.md`](architecture/engine_matrix.md)
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

## Product Direction

Future product direction remains focused on deeper creative workstation
ergonomics, agentic collaboration patterns, production intelligence, and
long-horizon creative memory. Those directions are product roadmap context, not
a claim that every listed system is already implemented or exposed in the
current runtime.

- Creative Workstation: continued operator-flow polish, inspection clarity, and
  production-ready creative review.
- Agentic Studio: more collaborative decomposition of creative strategy,
  critique, and refinement when future runtime boundaries support it.
- Execution Optimization: stronger production telemetry, runtime policy, and
  cost/performance intelligence.
- HoloGenesis Core OS: long-horizon creative lineage, feedback, memory, and
  system-level continuity.

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

Start the local backend bridge:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server
```

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

- `create_backend_dev_app`
- `create_assistant_streaming_app`
- `create_workspace_session_app`
- `run_backend_dev_server`

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
