# Project Context

## Product

Creative Coding Assistant is a domain-specific creative coding workstation. It
translates symbolic, geometric, stylistic, audiovisual, and multimodal intent
into structured creative guidance, generated artifacts, previewable outputs,
inspection metadata, critique, and refinement context.

## Current Platform Baseline

- V2.5 Creative Core for generation, artifact extraction, preview preparation,
  critique, and bounded refinement.
- V3.1 Creative Cognition Core for deterministic intent, hierarchy, strategy,
  technique, constraint, runtime, trade-off, quality, narrative, composition,
  Director, and Reasoning metadata.
- V3.2 Generative Design Core for procedural structure, generative systems,
  motifs, emotional continuity, cross-modality, and scene scaffolding.
- V3.3 Artifact Intelligence for artifact planning, dependency, compatibility,
  capability, strategy, critique, refinement, synthesis, merge, export, and
  engine-contract metadata.
- V3.4 Creative Evaluation for critic, self-evaluation, improvement,
  reflection, confidence, score, consistency, report, and evaluation-contract
  metadata.
- V3.5 Creative Workstation for workstation state, session intelligence,
  workflow explorer, provenance, timeline, inspector panels, dashboard cards,
  and workstation contracts.
- V3.6 stabilization for graph registration, stream payload helpers, workflow
  serialization, local backend mounting, and documentation alignment.
- V4.1 Multi-Agent Core for passive agent identities, roles, contracts, memory
  boundaries, authority boundaries, and advisory metadata.
- V4.2 Agent Orchestration for passive routing, blackboard, shared context,
  dependency, scheduling, coordination, debate, consensus, lifecycle,
  synchronization, handoff, escalation signal, and integration metadata.
- V4.3 Hybrid Agentic Workflow for passive V3 backbone, conditional
  escalation, specialist-loop, gate, creative policy, reflection, debate,
  voting, confidence, provenance, trace, budget, normalization, return
  handoff, HITL, threshold, ambiguity, risk, quality, adaptive escalation, and
  hybrid workflow integration metadata.

## Supported Creative Domains

- Three.js and React Three Fiber
- p5.js and Canvas
- GLSL and shader studies
- Hydra, Tone.js, GSAP, SVG, and browser-friendly audiovisual systems
- multimodal visual references and curated creative-coding source grounding

## Current Architecture Constraints

- The Python backend owns the compact LangGraph runtime and provider-facing
  workflow.
- The Next.js workstation owns preview, inspection, comparison, export, and
  operator-facing surfaces.
- Chroma remains the persistent retrieval and memory database.
- V3 metadata enriches workflow state and stream hydration without expanding
  the runtime graph into every internal helper.
- V4.1, V4.2, and V4.3 registries are passive product and architecture
  metadata. They are inspectable Python APIs and documentation surfaces.

## V4.3 Passive Boundary

V4.3 Hybrid Agentic Workflow is passive hybrid workflow metadata. It includes
the Adaptive Multi-Agent Escalation Registry and Hybrid Workflow Integration
source coverage, but it does not execute escalation, invoke agents, change
LangGraph node order, change provider/model routing, select runtimes, trigger
retries, mutate prompts, write storage, or modify generated output.

## Non-Goals For V4.3

- active multi-agent execution
- autonomous escalation
- provider or model routing changes
- runtime auto-selection
- generated-output mutation
- prompt rendering changes
- storage or blackboard runtime behavior
- V5 execution optimization
- V6 learning or long-horizon adaptation
