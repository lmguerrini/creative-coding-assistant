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
V4.2, V4.3, V4.4, and V4.5 registries do not create storage backends, write
blackboard state, write replay storage, or introduce runtime synchronization
behavior.

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
- V4.4 Hybrid Studio registries describe passive local/cloud model, hybrid
  execution, Studio surface, HITL, profile, comparison, workspace, snapshot,
  replay, and integration metadata over the V4.1-V4.3 contract layers.
- V4.5 Multimodal Studio registries describe passive preview, canvas,
  workspace, collaboration, provenance, lineage, history, branching, creative
  evolution, workflow visualization, and integration metadata over the V4.4
  Studio inspection layer.

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

## V4.4 Boundary Decision

V4.4 Hybrid Studio is an inspectable metadata layer only. It may expose local
model, cloud model, hybrid execution, Auto Mode, Studio Mode, HITL decision,
provider selection, execution simulator, model profile, cost profile, quality
profile, local/cloud comparison, agent workspace, agent conversation,
workspace snapshot, session replay, execution replay, and Hybrid Studio
Integration source coverage.

It must not activate Studio runtime, execute providers, invoke agents, control
workflows, request human input, select providers or models, route providers or
models, select runtimes, execute simulations, execute replay, capture live
workspace state, persist conversations, write replay storage, mutate storage,
trigger retries, mutate prompts, change workflow order, or modify generated
output. More generally, it does not execute providers, does not activate
Studio runtime, does not change provider/model routing, and does not modify
generated output.

## V4.5 Boundary Decision

V4.5 Multimodal Studio is an inspectable metadata layer only. It may expose live
preview, multi preview, interactive canvas, visual workspace, runtime
collaboration, artifact collaboration, artifact provenance, artifact lineage,
cross-agent workspace, shared artifact board, workspace history, branching
timeline, creative evolution timeline, real-time workflow visualization, and
Multimodal Studio Integration source coverage.

It must not execute rendering, activate Studio runtime, control workflows,
request human input, select providers or models, route providers or models,
select runtimes, trigger retries, mutate artifacts, modify generated output,
persist collaboration storage, subscribe to live streams, open networking,
bind canvas inputs, mutate canvas contexts, reconstruct timelines, create
branches, record provenance, or change workflow order. More generally, it does
not execute rendering, does not activate Studio runtime, does not change
provider/model routing, and does not modify generated output.

## Documentation Decision

Documentation should make passive metadata visible without implying active
runtime behavior. Product and architecture docs should continue to distinguish:

- the implemented compact LangGraph workflow
- internal V3 metadata derivation
- V3.5 workstation inspection surfaces
- passive V4.1 agent contracts
- passive V4.2 orchestration contracts
- passive V4.3 hybrid workflow metadata
- passive V4.4 hybrid studio metadata
- passive V4.5 multimodal studio metadata
- future active V4 Agentic Studio, V5 execution optimization, and V6 learning
  work

## Code Quality Rules

- Keep runtime behavior changes separate from metadata and documentation
  updates.
- Keep registry/source lists exhaustive for their capability scope.
- Add tests when docs claim source coverage or passive boundaries.
- Do not overstate passive registries as active orchestration behavior.
