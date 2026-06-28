# Implementation Roadmap

This roadmap summarizes product capability layers. It is not a promise that
future systems are active in the current runtime.

## Delivered Platform Layers

- V1 established backend service boundaries, retrieval, memory, tools, preview
  scaffolding, analytics, and evaluation foundations.
- V2.5 delivered the Creative Core loop: creative translation, generation,
  artifact extraction, preview preparation, critique, and bounded refinement.
- V3.1 delivered Creative Cognition metadata for intent, hierarchy, strategy,
  technique, constraints, runtime fit, trade-offs, quality, narrative,
  composition, Director guidance, and Reasoning synthesis.
- V3.2 delivered Generative Design metadata for procedural structure,
  generative systems, motifs, emotional continuity, cross-modality, and scene
  scaffolding.
- V3.3 delivered Artifact Intelligence metadata for planning, dependencies,
  runtime compatibility, capability mapping, strategy, critique, refinement,
  synthesis, merge, export, and engine contracts.
- V3.4 delivered Creative Evaluation metadata for critic, self-evaluation,
  improvement, reflection, confidence, score, consistency, reports, and
  evaluation engine contracts.
- V3.5 delivered Creative Workstation surfaces for state, session intelligence,
  workflow exploration, provenance, timeline, inspector panels, dashboard
  cards, and workstation contracts.
- V3.6 stabilized graph assembly, stream payloads, workflow serialization,
  local backend mounting, shared utilities, and documentation alignment.

## Passive Agentic Metadata Layers

- V4.1 Multi-Agent Core defines passive agent identities, contracts, roles,
  memory boundaries, authority boundaries, and advisory metadata.
- V4.2 Agent Orchestration defines passive routing, blackboard, shared context,
  dependency, scheduling, coordination, debate, consensus, lifecycle,
  synchronization, workflow handoff, escalation signal, and integration
  metadata.
- V4.3 Hybrid Agentic Workflow defines passive V3 backbone, conditional
  escalation, specialist-loop, escalation gate, creative policy, reflection,
  debate, voting, confidence fusion, decision provenance, escalation trace,
  exploration budget, result normalization, return-to-workflow handoff, HITL
  gate, confidence/cost/latency threshold, ambiguity, risk, quality, adaptive
  multi-agent escalation, and hybrid workflow integration metadata.
- V4.4 Hybrid Studio defines passive local model, cloud model, hybrid
  execution, Auto Mode, Studio Mode, HITL decision, provider selection,
  execution simulator, model profile, cost profile, quality profile,
  local/cloud comparison, agent workspace, agent conversation, workspace
  snapshot, session replay, execution replay, and hybrid studio integration
  metadata.
- V4.5 Multimodal Studio defines passive live preview, multi preview,
  interactive canvas, visual workspace, runtime collaboration, artifact
  collaboration, artifact provenance, artifact lineage, cross-agent workspace,
  shared artifact board, workspace history, branching timeline, creative
  evolution timeline, real-time workflow visualization, and multimodal studio
  integration metadata.

The V4.3 layer is a completed passive metadata layer. It does not execute
agents, run autonomous escalation, change workflow order, route providers or
models, select runtimes, trigger retries, mutate prompts, write storage, or
modify generated output.

The V4.4 layer is a completed passive metadata layer. It does not activate
Studio runtime, execute providers, invoke agents, control workflows, request
human input, change provider/model routing, select runtimes, trigger retries,
mutate storage, write replay storage, or modify generated output.

The V4.5 layer is a completed passive metadata layer. It does not execute
rendering, activate Studio runtime, control workflows, request human input,
change provider/model routing, select runtimes, trigger retries, mutate
artifacts, modify generated output, persist collaboration storage, subscribe
to live streams, open networking, or change LangGraph node order.

## Future Product Directions

- V4 Agentic Studio remains future active collaboration work. The current
  V4.1, V4.2, V4.3, V4.4, and V4.5 registries provide inspection and contract
  metadata for that direction, but not active collaboration behavior or active
  Studio runtime.
- V5 Execution Optimization & Production Intelligence remains future runtime
  policy, production telemetry, and cost/performance work.
- V6 HoloGenesis Core OS remains future long-horizon creative lineage,
  feedback, memory, and system continuity work.

## Documentation Contract

Product docs should describe delivered metadata layers without overstating
future behavior. Architecture docs may show passive registry boundaries and
source coverage, but must keep the current compact LangGraph workflow as the
execution source of truth.
