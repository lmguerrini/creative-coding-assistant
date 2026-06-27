# Engine Matrix

This document explains the cross-cutting architecture layers that sit alongside
the versioned roadmap. Versions are chronological delivery increments. Engines
are cross-cutting architecture layers that span multiple versions at once.

Use this matrix together with:

- [workflow_graph.md](workflow_graph.md) for the real LangGraph runtime graph
- [creative_intelligence_graph.md](creative_intelligence_graph.md) for the
  human-readable internal capability pipeline
- [generative_design_graph.md](generative_design_graph.md) for the V3.2
  Generative Design dependency graph and matrix
- [artifact_intelligence_graph.md](artifact_intelligence_graph.md) for the
  V3.3 Artifact Intelligence dependency graph and engine contract registry
- [workstation_surface_graph.md](workstation_surface_graph.md) for the V3.5
  Creative Workstation surface graph and contract boundary

## Engine Layers

| Engine | Purpose | Current scope | Key examples |
| --- | --- | --- | --- |
| Core Engine | Owns creative translation, planning, cognition, generative design, artifact intelligence, creative evaluation, critique, and final prompt construction. | Active and implemented through V2.5, V3.1, V3.2, V3.3, V3.4, and the V3.5 workstation metadata consumers; V3.6 stabilizes shared utilities without expanding behavior. | Creative Translation, Creative Planning, Creative Cognition Core, Generative Design Core, Artifact Intelligence, Creative Evaluation, Director, Creative Reasoning |
| Knowledge Engine | Owns retrieval, source grounding, memory, and future knowledge reasoning interfaces. | Active and implemented for retrieval and memory; future HoloMind integration remains outside the current runtime. | Source registry, KB retrieval, prompt memory, grounded prompt input |
| Execution Engine | Owns workflow orchestration, provider execution, validation, artifact extraction, preview preparation, metadata serialization, backend dev mounting, and future optimization. | Active and implemented for the bounded LangGraph runtime; V3.6 stabilizes graph assembly and serialization seams, and V5 expands this layer into Execution Optimization & Production Intelligence. | Workflow graph, generation, review gate, refinement loop, artifact extraction, preview preparation, workflow metadata payloads, backend bridge |
| Experience Layer | Owns workstation UX, preview surfaces, inspector views, comparison, export, stream hydration, operator controls, and workstation surface contracts. | Active and implemented in the Next.js workstation; V3.6 aligns the documented surface boundary, and V4 expands this layer into Agentic Studio collaboration patterns. | Workstation shell, preview shelf, inspectors, comparison workspace, provenance, timeline, dashboard, export surfaces, V3 metadata hydration |

## Version Vs Engine View

| Version | Core Engine | Knowledge Engine | Execution Engine | Experience Layer |
| --- | --- | --- | --- | --- |
| V1 | Backend service boundaries, request contracts, initial prompt path | Initial Chroma-backed retrieval and source sync | Streaming service, provider adapter, baseline workflow scaffolding | Streamlit reference client |
| V2.5 | Creative Core, critique, calibration, refinement | Retrieval grounding and session context | Multi-artifact flow, preview metadata, bounded review loop | Next.js workstation, preview runtimes, artifact comparison |
| V3.1 | Creative Cognition Core | Retrieval and memory remain the grounding substrate for cognition metadata | Compact runtime graph with richer stored planning metadata | Workflow inspector visibility for cognition-derived state |
| V3.2 | Generative Design Core extends the stored creative brief with design metadata | Retrieval and memory continue to ground higher-level design guidance | Same compact runtime graph; no runtime auto-selection or provider routing added | Existing workstation surfaces now expose richer creative metadata |
| V3.3 | Artifact Intelligence extends the stored creative/design brief with artifact planning, compatibility, critique, refinement, synthesis, merge, export intelligence, and engine contract metadata | Retrieval and memory continue to ground upstream planning; no new knowledge runtime is introduced | Same compact runtime graph; artifact metadata is serialized through workflow payloads without export execution, runtime auto-selection, provider routing, retries, or preview changes | Next.js stream hydration reads artifact summaries and the engine contract registry for inspector/workflow surfaces |
| V3.4 | Creative Evaluation adds metadata-only critic, self-evaluation, improvement, reflection, confidence, score, consistency, report, and evaluation engine-contract metadata | Retrieval and memory continue to ground evaluation context without introducing a new knowledge runtime | Evaluation metadata is serialized through workflow payloads without changing provider routing, runtime selection, artifact execution, autonomous retries, or preview behavior | Next.js stream hydration reads evaluation summaries and the engine contract registry for future inspector/workflow surfaces |
| V3.5 | Creative Workstation exposes state, session, workflow, provenance, timeline, inspector, dashboard, and workstation contract metadata without changing generation behavior | Knowledge surfaces become more operator-legible without changing retrieval ownership | Execution metadata is exposed more clearly without adding hidden runtime behavior | Workstation surfaces become the primary focus for usability, inspection, and operator flow |
| V3.6 | Stabilization & Refactor Pass hardens the completed V3 surface without adding new generation behavior | Knowledge boundaries are simplified without changing source-of-truth ownership | Runtime contracts, validation seams, backend dev mounting, and serialization paths are stabilized without feature expansion | Experience surfaces and documentation are aligned without changing capability scope |
| V4 | Agentic Studio decomposes more internal creative work into bounded collaborative systems | Deeper agent-facing knowledge packets may emerge here | More inspectable orchestration paths may appear here | Agentic Studio becomes the main collaboration surface |
| V5 | Core Engine remains creative-first but hands more optimization work outward | Knowledge signals can guide execution optimization and production policy | Execution Optimization & Production Intelligence becomes the primary expansion | Experience surfaces emphasize production telemetry and operational controls |
| V6 | HoloGenesis Core OS can unify long-horizon creative strategy, lineage, and system identity | Long-horizon knowledge and memory adaptation move into the future OS direction | Execution can learn from prior runs without replacing bounded workflow control | Experience surfaces expose lineage, feedback, and evolving operator guidance |

## Reading The Matrix

- Versions answer "when did this capability family land?"
- Engines answer "which architectural layer owns this responsibility?"
- A single version can strengthen several engines at once.
- A single engine can span many versions without requiring a rename of the
  roadmap.

## Implemented Contract Registries

The current V3 product exposes three source-of-truth contract registries for
implemented artifact, evaluation, and workstation metadata surfaces. Artifact
and evaluation contracts are serialized through workflow metadata payloads;
workstation contracts describe client-side surface hydration. They are static
metadata descriptions, not dynamic engine routers or future-version behavior.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| Artifact Intelligence contracts | `src/creative_coding_assistant/orchestration/artifact_engine_contracts.py` | 10 | `artifact_engine_contract_registry.v1` | Describes V3.3 artifact metadata dependencies, signals, cacheability, parallelization support, estimated cost, and estimated latency |
| Creative Evaluation contracts | `src/creative_coding_assistant/orchestration/evaluation_engine_contracts.py` | 8 | `evaluation_engine_contract_registry.v1` | Describes V3.4 evaluation metadata dependencies, evidence expectations, signals, cacheability, parallelization support, estimated cost, and estimated latency |
| Creative Workstation contracts | `src/creative_coding_assistant/orchestration/workstation_contracts.py` | 7 | `workstation_engine_contract_registry.v1` | Describes V3.5 workstation surface inputs, exposed metadata, stability signals, hydration mode, estimated cost, and estimated latency |

Artifact and evaluation registries expose confidence, ambiguity, risk, and
dependency signals. The workstation registry exposes stability and
missing-metadata behavior for client-side surfaces. All three registries keep
future hooks descriptive only: they do not invoke agents, route providers,
select runtimes, execute artifacts, repair previews, trigger retries, or change
generated output.

## V3.6 Audit And Future-Readiness Metadata

V3.6 also exposes passive metadata registries for architecture audit,
consistency checks, and future-readiness handoff. These registries are explicit
Python metadata APIs only. They are not serialized into provider prompts, they
are not additional LangGraph runtime nodes, and they do not enter workflow
payloads as runtime behavior.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| Agent capability readiness | `src/creative_coding_assistant/orchestration/agent_capabilities.py` | 6 | `agent_capability_registry.v1` | Describes future agent capability readiness metadata without creating agents or changing workflow control |
| Escalation policy metadata | `src/creative_coding_assistant/orchestration/escalation_policy.py` | 5 | `escalation_policy_registry.v1` | Describes advisory escalation policy metadata without evaluating policy, triggering escalation, or routing providers |
| Hybrid agentic workflow readiness | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 | `hybrid_workflow_registry.v1` | Maps existing V3 workflow nodes to future readiness stages without mutating the graph |
| Engine contract consistency audit | `src/creative_coding_assistant/orchestration/engine_contract_consistency.py` | 3 families | `engine_contract_consistency_registry.v1` | Normalizes artifact, evaluation, and workstation contract surfaces for audit without changing runtime behavior |

These V3.6 registries remain export-only metadata surfaces. Tests assert that
they do not alter provider/model routing, runtime selection, prompt rendering,
workflow payloads, retry behavior, artifact execution, preview execution,
generated output, or the V3 node order.

## V3.5 Workstation Contracts

The Creative Workstation exposes a metadata-only
`workstation_engine_contract_registry.v1` registry for seven stable V3.5
surfaces. The registry describes existing workstation surface inputs, exposed
metadata, exposed signals, missing-metadata behavior, and named future hooks
for V4, V5, and V6 consumers.

| Contract | Primary surface | Downstream boundary |
| --- | --- | --- |
| `workstation_state` | Session, run, selection, panel, readiness, and metadata status | Shared context packet for workstation-aware review surfaces |
| `session_intelligence` | Session readiness, completion, warnings, and operator next actions | Advisory context for future agentic studio handoff |
| `workflow_explorer` | Workflow nodes, edges, active step, runtime status, and progress | Workflow context for future agentic review without changing graph control |
| `provenance_engine` | Evidence, dependency, artifact, evaluation, final payload, and missing-source provenance | Lineage context for future creative evolution without external fetching |
| `creative_timeline` | Ordered request, planning, retrieval, creative, artifact, evaluation, and final stages | Timeline context for future lineage and learning signals |
| `v3_inspector_panels` | Creative intelligence, generative design, artifact intelligence, evaluation, and provenance records | Review context for future agentic and adaptive execution consumers |
| `workstation_dashboard` | Quality, confidence, consistency, readiness, runtime fit, evaluation, workflow health, and HITL cards | Operator policy signal for future adaptive execution without autonomous action |

These contracts do not implement V4 agents, V5 execution optimization, V6
learning, provider routing, runtime selection, autonomous retries, preview
execution, artifact modification, or generated output changes. They define the
metadata boundary that future systems can consume without making the current
workstation responsible for future behavior.

## Current Boundary

- V3.5 is still metadata projection, workstation inspection, workflow
  visibility, provenance visibility, timeline organization, dashboard
  summarization, and surface contract exposure, not agent behavior, execution
  optimization, learning behavior, artifact execution, artifact modification,
  artifact export, runtime selection, runtime repair, provider/model routing,
  autonomous retries, or preview behavior changes.
- V3.6 is the current stabilization layer over V3.5, not a new runtime feature
  family. After V3.6, the roadmap remains V4 Agentic Studio, V5 Execution
  Optimization & Production Intelligence, and V6 HoloGenesis Core OS.
- The current runtime graph remains the source of truth for execution order.
- The matrix is a planning and architecture aid, not a claim that every engine
  is already a separate runtime subsystem.
