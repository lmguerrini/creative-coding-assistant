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

## Engine Layers

| Engine | Purpose | Current V3.2 scope | Key examples |
| --- | --- | --- | --- |
| Core Engine | Owns creative translation, planning, cognition, generative design, critique, and final prompt construction. | Active and implemented through V2.5, V3.1, and V3.2. | Creative Translation, Creative Planning, Creative Cognition Core, Generative Design Core, Director, Creative Reasoning |
| Knowledge Engine | Owns retrieval, source grounding, memory, and future knowledge reasoning interfaces. | Active and implemented for retrieval and memory; future HoloMind integration remains outside the current runtime. | Source registry, KB retrieval, prompt memory, grounded prompt input |
| Execution Engine | Owns workflow orchestration, provider execution, validation, artifact extraction, preview preparation, and future optimization. | Active and implemented for the bounded LangGraph runtime; V5 expands this layer into Execution Optimization & Production Intelligence. | Workflow graph, generation, review gate, refinement loop, artifact extraction, preview preparation |
| Experience Layer | Owns workstation UX, preview surfaces, inspector views, comparison, export, and operator controls. | Active and implemented in the Next.js workstation; V4 expands this layer into Agentic Studio collaboration patterns. | Workstation shell, preview shelf, inspectors, comparison workspace, export surfaces |

## Version Vs Engine View

| Version | Core Engine | Knowledge Engine | Execution Engine | Experience Layer |
| --- | --- | --- | --- | --- |
| V1 | Backend service boundaries, request contracts, initial prompt path | Initial Chroma-backed retrieval and source sync | Streaming service, provider adapter, baseline workflow scaffolding | Streamlit reference client |
| V2.5 | Creative Core, critique, calibration, refinement | Retrieval grounding and session context | Multi-artifact flow, preview metadata, bounded review loop | Next.js workstation, preview runtimes, artifact comparison |
| V3.1 | Creative Cognition Core | Retrieval and memory remain the grounding substrate for cognition metadata | Compact runtime graph with richer stored planning metadata | Workflow inspector visibility for cognition-derived state |
| V3.2 | Generative Design Core extends the stored creative brief with design metadata | Retrieval and memory continue to ground higher-level design guidance | Same compact runtime graph; no runtime auto-selection or provider routing added | Existing workstation surfaces now expose richer creative metadata |
| V4 | Agentic Studio decomposes more internal creative work into bounded collaborative systems | Deeper agent-facing knowledge packets may emerge here | More inspectable orchestration paths may appear here | Agentic Studio becomes the main collaboration surface |
| V5 | Core Engine remains creative-first but hands more optimization work outward | Knowledge signals can guide execution optimization and production policy | Execution Optimization & Production Intelligence becomes the primary expansion | Experience surfaces emphasize production telemetry and operational controls |
| V6 | Learning loops can refine creative strategy and design heuristics over time | Learning & Evolution strengthens long-horizon knowledge and memory adaptation | Execution learns from prior runs without replacing bounded workflow control | Experience surfaces expose lineage, feedback, and evolving operator guidance |

## Reading The Matrix

- Versions answer "when did this capability family land?"
- Engines answer "which architectural layer owns this responsibility?"
- A single version can strengthen several engines at once.
- A single engine can span many versions without requiring a rename of the
  roadmap.

## Current Boundary

- V3.2 is still metadata and design guidance, not code generation execution,
  runtime repair, or provider/model routing.
- The current runtime graph remains the source of truth for execution order.
- The matrix is a planning and architecture aid, not a claim that every engine
  is already a separate runtime subsystem.
