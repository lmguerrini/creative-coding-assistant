# Architecture Decisions

## Runtime Ownership

- Keep the Python backend as the source of truth for request handling,
  retrieval, memory, planning metadata, provider execution, artifact metadata,
  review, bounded refinement, finalization, and failure handling.
- Keep the LangGraph workflow compact. Internal creative intelligence,
  generative design, artifact intelligence, evaluation, workstation, and V4
  registry layers are metadata surfaces, not additional runtime nodes.
- Keep the Next.js workstation responsible for product inspection, preview,
  comparison, export, telemetry, workflow visibility, and operator controls.

## Persistence

Chroma remains the only persistent retrieval and memory database. Passive V4.1,
V4.2, and V4.3 registries do not create storage backends, write blackboard
state, or introduce runtime synchronization behavior.

## Metadata Layering

- V3.1 through V3.4 metadata is derived inside the existing planning and
  evaluation flow and serialized for downstream prompt, stream, and workstation
  consumers.
- V3.5 workstation contracts describe inspection surfaces over existing
  metadata without changing generation behavior.
- V4.1 Multi-Agent Core registries describe passive agent roles and contracts.
- V4.2 Agent Orchestration registries describe passive orchestration readiness.
- V4.3 Hybrid Agentic Workflow registries describe passive hybrid escalation,
  threshold, handoff, adaptive, and integration metadata over the stable V3
  backbone and V4 contracts.

## V4.3 Boundary Decision

V4.3 Hybrid Agentic Workflow is an inspectable metadata layer only. It may
expose V3 backbone mode, conditional escalation, specialist loop, gate, policy,
reflection, debate, voting, confidence, provenance, trace, exploration budget,
normalization, return handoff, HITL, threshold, ambiguity, risk, quality,
adaptive escalation, and source integration metadata.

It must not execute escalation, invoke agents, run debates, vote, fuse
confidence, record provenance, emit traces, enforce budgets, normalize outputs,
perform runtime handoffs, trigger HITL, evaluate thresholds, evaluate
ambiguity/risk/quality, orchestrate agents, change workflow order, route
providers or models, select runtimes, trigger retries, mutate prompts, write
storage, or modify generated output.

## Documentation Decision

Documentation should make passive metadata visible without implying active
runtime behavior. Product and architecture docs should continue to distinguish:

- the implemented compact LangGraph workflow
- internal V3 metadata derivation
- V3.5 workstation inspection surfaces
- passive V4.1 agent contracts
- passive V4.2 orchestration contracts
- passive V4.3 hybrid workflow metadata
- future active V4 Agentic Studio, V5 execution optimization, and V6 learning
  work

## Code Quality Rules

- Keep runtime behavior changes separate from metadata and documentation
  updates.
- Keep registry/source lists exhaustive for their capability scope.
- Add tests when docs claim source coverage or passive boundaries.
- Do not overstate passive registries as active orchestration behavior.
