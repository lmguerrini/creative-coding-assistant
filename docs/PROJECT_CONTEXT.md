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
- V4.4 Hybrid Studio for passive local model, cloud model, hybrid execution,
  Auto Mode, Studio Mode, HITL decision, provider selection, execution
  simulator, model profile, cost profile, quality profile, local/cloud
  comparison, agent workspace, agent conversation, workspace snapshot, session
  replay, execution replay, and hybrid studio integration metadata.
- V4.5 Multimodal Studio for passive live preview, multi preview, interactive
  canvas, visual workspace, runtime collaboration, artifact collaboration,
  artifact provenance, artifact lineage, cross-agent workspace, shared artifact
  board, workspace history, branching timeline, creative evolution timeline,
  real-time workflow visualization, and multimodal studio integration metadata.
- V4.6 Agentic Studio Hardening for passive contract, policy, workflow,
  registry, memory/context, collaboration, diversity, explainability,
  reliability, determinism, telemetry, cost, performance, architecture
  consistency, final hardening, and LangGraph error-path audit metadata.
- V5.1 Execution Optimization Engine for advisory graph, cost, complexity,
  creative complexity, context budget, exploration budget, context routing,
  compression, summarization, cache, reuse, pruning, forecasting, path
  optimization, strategy selection, architecture consistency, and failure audit
  metadata.
- V5.2 Intelligent Model Routing Engine for advisory model routing,
  local/cloud routing, hybrid routing, quality/cost optimization, cost
  estimation, budget policies, HITL budget gates, runtime recommendation,
  execution policy, model recommendation, model/provider capability matrices,
  quality/cost prediction, creative quality/diversity/consistency prediction,
  routing explainability, architecture consistency, and failure audit metadata.
- V5.3 Performance Engine for advisory parallel scheduling, latency
  optimization, async execution, streaming optimization, retry policies, load
  balancing, execution profiling, workflow replay, execution replay,
  bottleneck detection, throughput optimization, performance prediction,
  performance benchmarking, reasoning budget optimization, performance
  regression detection, resource utilization optimization, architecture
  consistency, and failure audit metadata.
- V5.4 Production Observability for read-only token, cost, quality,
  performance, production telemetry, workflow diagnostics, agent diagnostics,
  routing diagnostics, escalation diagnostics, failure analysis, error
  intelligence, workflow health monitoring, system health monitoring, creative
  analytics, confidence analytics, creative diversity analytics, runtime
  timeline, workflow explainability, architecture consistency, and failure
  audit metadata.
- V5.5 Adaptive Execution Intelligence for advisory hybrid workflow,
  escalation, agent activation, adaptive cost/quality and latency, dynamic
  execution strategy, dynamic agent/resource allocation, workflow self-tuning,
  execution confidence, workflow risk, creative exploration, emergence, agent
  diversity, reflection budget, adaptive policy explainability, architecture
  consistency, and failure audit metadata.

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
- V4.1, V4.2, V4.3, V4.4, V4.5, V4.6, V5.1, V5.2, V5.3, V5.4, and V5.5
  registries and helpers are passive, advisory, or read-only product and
  architecture metadata. They are inspectable Python APIs and documentation
  surfaces.

## V4.3 Passive Boundary

V4.3 Hybrid Agentic Workflow is passive hybrid workflow metadata. It includes
the Adaptive Multi-Agent Escalation Registry and Hybrid Workflow Integration
source coverage, but it does not execute escalation, invoke agents, change
LangGraph node order, change provider/model routing, select runtimes, trigger
retries, mutate prompts, write storage, or modify generated output.

## V4.4 Passive Boundary

V4.4 Hybrid Studio is passive hybrid studio metadata. It includes the Local
Model Registry, Cloud Model Registry, Hybrid Execution Registry, Auto Mode
Registry, Studio Mode Registry, HITL Decision Registry, Provider Selection
Registry, Execution Simulator Registry, Model Profile Registry, Cost Profile
Registry, Quality Profile Registry, Local/Cloud Comparison Registry, Agent
Workspace Registry, Agent Conversation View Registry, Workspace Snapshot
Registry, Session Replay Registry, Execution Replay Registry, and Hybrid
Studio Integration Registry with Hybrid Studio Integration source coverage,
but it does not activate Studio runtime, execute providers, invoke agents,
control workflows, request human input, change provider/model routing, select
runtimes, trigger retries, mutate storage, write replay storage, or modify
generated output.

The Hybrid Studio Integration Registry exposes Hybrid Studio Integration
source coverage for audit without activating Studio runtime.

## V4.5 Passive Boundary

V4.5 Multimodal Studio is passive multimodal studio metadata. It includes the
Live Preview Registry, Multi Preview Registry, Interactive Canvas Registry,
Visual Workspace Registry, Runtime Collaboration Registry, Artifact
Collaboration Registry, Artifact Provenance Registry, Artifact Lineage
Registry, Cross-Agent Workspace Registry, Shared Artifact Board Registry,
Workspace History Registry, Branching Timeline Registry, Creative Evolution
Timeline Registry, Real-Time Workflow Visualization Registry, and Multimodal
Studio Integration Registry with Multimodal Studio Integration source
coverage, but it does not execute rendering, activate Studio runtime, control
workflows, request human input, change provider/model routing, select
runtimes, trigger retries, mutate artifacts, modify generated output, persist
collaboration storage, subscribe to live streams, open networking, or change
LangGraph node order.

The Multimodal Studio Integration Registry exposes Multimodal Studio
Integration source coverage for audit without activating Studio runtime or
executing rendering.

## V4.6 Passive Boundary

V4.6 Agentic Studio Hardening is passive agentic studio hardening metadata. It
includes the Agent Contract Audit Registry, Escalation Policy Audit Registry,
Hybrid Workflow Audit Registry, Agent Registry Audit Registry, Blackboard Audit
Registry, Shared Context Audit Registry, Agent Collaboration Audit Registry,
Creative Diversity Audit Registry, Agent Explainability Audit Registry, Agent
Reliability Audit Registry, Agent Determinism Audit Registry, Agent Telemetry
Foundation Registry, Agent Cost Tracking Foundation Registry, Agent Performance
Tracking Foundation Registry, Architecture Consistency Pass Registry, Final V4 Hardening Registry, and LangGraph Error Path Audit, but it does not execute a
hardening engine, add LangGraph nodes, bypass failure normalization, activate
passive registries, control workflows, change provider/model routing, select
runtimes, trigger retries, invoke agents, mutate storage, execute artifacts, or
modify generated output.

The LangGraph Error Path Audit documents and tests terminal failure coverage
for the existing graph and backend/frontend boundaries without introducing new
runtime behavior.

## V5.1 Advisory Boundary

V5.1 Execution Optimization Engine is advisory execution optimization metadata
over the existing compact LangGraph workflow. It includes graph analysis,
workflow cost and complexity analysis, creative complexity analysis, context
budget planning, exploration budget planning, context routing, prompt and
retrieval compression metadata, memory summarization metadata, deterministic
in-memory cache metadata, context reuse planning, workflow pruning candidates,
execution cost forecasts, path optimization candidates, strategy selection,
architecture consistency coverage, and runtime failure-path audit coverage.

It must not add LangGraph nodes, compile or execute alternate graphs, enforce
budgets, apply pruning, select runtime paths, apply execution strategies, route
providers or models, trigger retries, mutate prompts, write persistent cache
storage, change retrieval or memory ownership, control workflows, activate
passive registries, or modify generated output.

## V5.2 Advisory Boundary

V5.2 Intelligent Model Routing Engine is advisory model-routing metadata over
the existing compact LangGraph workflow. It includes model routing candidates,
local/cloud and hybrid routing posture, quality/cost optimization, cost
estimation, budget policy metadata, HITL budget gate posture, runtime
recommendations, execution policy posture, model recommendations, model and
provider capability matrices, quality and cost prediction, creative quality,
creative diversity, creative consistency, routing explainability, architecture
consistency coverage, and runtime failure-path audit coverage.

It must not apply routing, switch providers or models, execute providers,
enforce budgets, emit HITL requests, request human input, select runtimes,
control workflows, trigger retries, mutate prompts, write persistent storage,
activate passive registries, apply Runtime Evolution, or modify generated
output.

## V5.3 Advisory Boundary

V5.3 Performance Engine is advisory performance metadata over the existing
compact LangGraph workflow. It includes parallel scheduling candidates,
latency optimization posture, async execution readiness, streaming
optimization posture, retry policy posture, load balancing posture, execution
profiling candidates, workflow replay planning, execution replay planning,
bottleneck detection, throughput optimization, performance prediction,
performance benchmarking, reasoning budget recommendations, performance
regression signals, resource utilization recommendations, architecture
consistency coverage, and runtime failure-path audit coverage.

It must not measure live performance, install profilers, collect traces,
execute workflows, execute benchmarks, execute replay, allocate resources,
autoscale, enforce capacity or budgets, select runtimes, control workflows,
trigger retries, route providers or models, mutate prompts, write persistent
storage, activate passive registries, apply Runtime Evolution, or modify
generated output.

## V5.4 Read-Only Observability Boundary

V5.4 Production Observability is read-only observability metadata over the
existing compact LangGraph workflow. It includes token dashboards, cost
dashboards, quality dashboards, performance dashboards, production telemetry,
workflow diagnostics, agent diagnostics, routing diagnostics, escalation
diagnostics, failure analysis, error intelligence, workflow health monitoring,
system health monitoring, creative analytics, confidence analytics, creative
diversity analytics, runtime timeline, workflow explainability, architecture
consistency coverage, and runtime failure-path audit coverage.

It must not collect live metrics, emit telemetry or alerts, capture traces,
execute health checks, classify live errors, remediate failures, reconstruct
timelines, record provenance, generate explanations, request human review,
execute or control workflows, trigger retries, route providers or models,
mutate prompts, write persistent storage, activate passive registries, apply
Runtime Evolution, or modify generated output.

## V5.5 Advisory Adaptive Execution Boundary

V5.5 Adaptive Execution Intelligence is advisory adaptive execution metadata
over the existing compact LangGraph workflow. It includes adaptive hybrid
workflow optimization, adaptive escalation optimization, agent activation
optimization, adaptive cost/quality optimization, adaptive latency
optimization, dynamic execution strategy selection, dynamic agent allocation,
dynamic resource allocation, workflow self-tuning policy posture, execution
confidence signals, workflow risk factors, creative exploration optimization,
emergence optimization, agent diversity optimization, reflection budget
optimization, adaptive policy explainability, architecture consistency
coverage, and runtime failure-path audit coverage.

It must not apply adaptive policies or strategies, apply routing, switch
providers or models, execute providers, instantiate or invoke agents, activate
agents, allocate agents or resources, measure runtime resources, enforce
budgets, emit HITL requests, request human input, compile graphs, execute or
control workflows, mutate workflow graphs, trigger retries or refinements,
mutate prompts, write persistent storage, activate passive registries, apply
Runtime Evolution, or modify generated output.

## Non-Goals For Current Baseline

- active multi-agent execution
- autonomous escalation
- active Studio runtime
- rendering execution
- preview runtime execution
- canvas input binding or canvas mutation
- workflow visualization execution
- runtime hardening execution
- active execution optimization
- active performance optimization
- runtime performance measurement
- live production telemetry emission
- active observability remediation
- benchmark execution
- resource allocation or autoscaling
- LangGraph node additions
- failure normalization bypasses
- passive registry activation
- provider execution
- provider/model recommendation application
- agent invocation
- human-input requests
- provider or model routing changes
- runtime auto-selection
- collaboration storage persistence
- replay persistence or storage mutation
- artifact mutation
- generated-output mutation
- prompt rendering changes
- storage or blackboard runtime behavior
- provider pricing lookup or budget enforcement
- active adaptive runtime policy
- V6 learning or long-horizon adaptation
