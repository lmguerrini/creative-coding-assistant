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
- V5.5 Adaptive Execution Intelligence for controlled adaptive execution
  policy and simulation, advisory hybrid workflow, escalation, agent
  activation, adaptive cost/quality and latency, dynamic execution strategy,
  dynamic agent/resource allocation, workflow self-tuning, execution
  confidence, workflow risk, creative exploration, emergence, agent diversity,
  reflection budget, adaptive policy explainability, architecture consistency,
  and failure audit metadata.
- V5.6 Production Release for production final optimization, packaging
  readiness, release-candidate posture, demo assets, deployment assumptions,
  production readiness, creative readiness, architecture freeze, release
  audit, final hardening, architecture consistency, and runtime failure-path
  metadata that prepares the existing V5 system for demo/release review
  without adding new core architecture.
- V6.1 Adaptive Learning Engine for advisory learning signals, workflow
  success tracking, failure tracking, strategy learning, technique learning,
  runtime learning, routing learning, artifact learning, evaluation learning,
  continuous improvement signals, success pattern discovery, failure pattern
  discovery, learning replay metadata, learning confidence calibration,
  creative success learning, creative failure learning, learning governance,
  and runtime failure-path audit metadata without memory persistence, replay
  execution, model training, feedback application, policy mutation, live
  outcome observation, workflow control, provider execution, storage writes,
  preference mutation, automatic remediation, or output mutation.
- V6.2 Creative Memory Engine for advisory long-term creative memory, user
  preferences, style profiles, project memory, Creative DNA, personalization
  posture, session memory evolution, artifact history, creative lineage,
  creative ontology, preference learning posture, user modeling, memory
  consolidation posture, memory retrieval intelligence, memory retrieval
  planning, memory conflict resolution posture, memory explainability, memory
  safety policy posture, creative taste modeling, creative preference
  evolution, creative memory governance, and runtime failure-path audit
  metadata without memory storage writes, memory retrieval execution,
  preference learning execution, personalization application, Creative DNA
  application, lineage inference, ontology relationship inference, safety
  policy enforcement, HITL request emission, automation activation, workflow
  control, provider execution, Runtime Evolution, or output mutation.
- V6.3 Knowledge Evolution Engine for advisory automatic KB updates,
  documentation intelligence, embedding refresh, retrieval evolution, ranking
  optimization, knowledge health monitoring, knowledge quality scoring,
  knowledge gap detection, knowledge conflict resolver, knowledge drift
  detection, source reliability, knowledge consolidation, lifecycle
  management, provenance evolution, versioning, snapshot, rollback, freshness,
  trust score, governance, and runtime failure-path audit metadata without
  automatic KB update execution, documentation fetch execution, embedding
  refresh execution, retrieval execution, retrieval configuration mutation,
  ranking mutation, quality or trust score computation, source record updates,
  KB storage writes, provenance graph mutation, version graph mutation,
  snapshot execution, rollback execution, freshness scans, policy enforcement,
  HITL request emission, automation activation, workflow control, provider
  execution, Runtime Evolution, or output mutation.

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
- V4.1, V4.2, V4.3, V4.4, V4.5, V4.6, V5.1, V5.2, V5.3, V5.4, V5.5, V5.6,
  V6.1, V6.2, and V6.3 registries and helpers are passive, advisory,
  read-only, controlled-policy, advisory-learning, advisory creative-memory,
  or advisory knowledge evolution product and architecture metadata. They are
  inspectable Python APIs and documentation surfaces.

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

## V5.5 Controlled Adaptive Execution Boundary

V5.5 Adaptive Execution Intelligence is a controlled adaptive execution policy
and simulation layer over the existing compact LangGraph workflow. It includes
task-aware allow/confirm/block decisions, provider/model path readiness,
pre-run tradeoff simulation, hybrid workflow policy, fallback and escalation
policy, Manual/Assisted/Auto mode semantics, adaptive hybrid workflow
optimization, adaptive escalation optimization, agent activation optimization,
adaptive cost/quality optimization, adaptive latency optimization, dynamic
execution strategy selection, dynamic agent allocation, dynamic resource
allocation, workflow self-tuning policy posture, execution confidence signals,
workflow risk factors, creative exploration optimization, emergence
optimization, agent diversity optimization, reflection budget optimization,
adaptive policy explainability, architecture consistency coverage, and runtime
failure-path audit coverage.

It must not mutate configured provider/model routing, silently switch
providers or models, execute providers, instantiate or invoke agents, activate
agents, allocate agents or resources, measure runtime resources, enforce
budgets, emit HITL requests, request human input on its own, compile graphs,
execute or control workflows, mutate workflow graphs, trigger retries or
refinements, mutate prompts, write persistent storage, activate passive
registries, apply Runtime Evolution, download local models, provision
providers, install runtimes, or modify generated output.

## V5.6 Production Release Boundary

V5.6 Production Release is production-readiness metadata over the existing V5
system. It includes final optimization posture, packaging readiness,
release-candidate posture, demo asset readiness, deployment assumption
metadata, provider/environment/API-key diagnostics, Manual/Assisted/Auto safety
boundaries, explainability fields, deterministic failure review, production
readiness review, creative readiness review, architecture freeze, release
audit, guarded final hardening posture, architecture consistency coverage,
runtime failure-path audit coverage, and explicit local-demo versus
external-deployment assumptions.

It must not introduce new core architecture, mutate configured provider/model
routing, silently switch providers or models, execute providers, generate or
mutate assets, execute retrieval, run package builds, install dependencies,
deploy services, create containers, provision providers, install runtimes,
download local models, emit HITL requests, execute or control workflows,
mutate workflow graphs, write persistent storage, create release artifacts,
execute hardening actions, create runtime failure handlers, mutate terminal
routing, merge, push, tag, apply Runtime Evolution, or modify generated output.

## V6.1 Adaptive Learning Boundary

V6.1 Adaptive Learning Engine begins as controlled learning metadata over the
V5 decision engine. The initial learning engine derives adaptive learning
signals from execution confidence, workflow risk, and workflow self-tuning
metadata, including priority scores, pattern tags, evidence, and review
posture. Workflow success tracking derives success indicators from adaptive
learning signals without observing live outcomes or persisting metrics.
Failure tracking derives failure indicators from read-only failure analysis
and adaptive learning metadata without observing, classifying, routing,
handling, or repairing live failures. Strategy learning derives strategy
patterns from advisory adaptive execution strategy metadata and adaptive
learning signals without applying or mutating strategy selection. Technique
learning derives technique patterns from read-only creative technique metadata
and adaptive learning signals without rendering prompts, applying techniques,
selecting runtimes, or executing artifacts. Runtime learning derives runtime
patterns from read-only runtime capability metadata and adaptive learning
signals without selecting runtimes, probing local runtimes, installing
dependencies, or changing preview behavior. Routing learning derives route
patterns from read-only task-aware routing metadata and adaptive learning
signals without applying routes, switching providers or models, executing
providers, probing local runtimes, or assuming credentials. Artifact learning
derives artifact-shape, capability, and risk patterns from read-only artifact
planning and artifact capability metadata without selecting, mutating,
generating, executing, merging, exporting, or previewing artifacts.
Evaluation learning derives evaluation patterns from read-only evaluation
engine contract metadata without running evaluations, evaluating generated
output, mutating scores or confidence, executing reflection loops, generating
reports, or changing workflow order. Continuous improvement signals
synthesize read-only V6.1 success, failure, artifact, and evaluation learning
metadata into advisory improvement candidates without applying feedback,
persisting learning memory, updating policies, observing runtime outcomes, or
changing workflows. Success pattern discovery derives candidate success
patterns from read-only workflow success and continuous improvement metadata
without observing live success, collecting telemetry, persisting success
metrics, applying patterns, or applying feedback. Failure pattern discovery
derives guarded failure patterns from read-only failure tracking and
continuous improvement metadata without observing live failures, classifying
live errors, routing terminal failures, handling or repairing failures, or
mutating terminal routing. Learning governance describes memory, feedback,
policy, HITL, explainability, and no-automation boundaries without persisting
memory, applying feedback, updating or enforcing policies, emitting HITL
requests, or requesting human input.
Learning replay records replay scenarios, source learning signal references,
expected replay insight, replay confidence, and replay safety boundaries
without executing learning replay, workflow replay, provider calls, storage
writes, generated-output mutation, or Runtime Evolution. Learning confidence
calibration maps confidence before/after bands from existing learning signals
and V5 execution/confidence metadata with rationale, uncertainty factors, and
HITL requirements for low or risky confidence without model training,
feedback application, storage writes, or runtime mutation. Creative success
learning specializes success patterns for creative coding artifact, aesthetic,
usefulness, and originality dimensions with explainability and without
generated-output mutation, automatic preference mutation, or storage writes.
Creative failure learning specializes failure patterns for artifact, preview,
runtime, aesthetic, prompt, and retrieval failure modes with explainability and
without generated-output mutation, automatic remediation, or storage writes.
The V6.1 runtime failure path audit verifies these learning surfaces against
the runtime failure checklist without creating failure handlers, routing
terminal failures, changing provider/model routing, writing storage, mutating
generated output, or applying Runtime Evolution.

It must not persist learning memory, execute learning replay, execute workflow
replay, train models, apply feedback, update policies, mutate strategies,
change provider/model routing, execute providers, probe local runtimes,
download models, invoke agents, allocate resources, emit HITL requests,
enforce budgets, collect telemetry, observe runtime success, observe runtime
failures, classify live errors, route terminal failures, handle or repair
failures, evaluate generated output, execute or control workflows, mutate
workflow graphs, trigger retries or refinements, mutate strategy selection,
mutate preferences, apply techniques, select runtimes, execute artifacts,
probe local runtimes, install dependencies, change preview behavior, compile
graphs, render or mutate prompts, write persistent storage, persist success
metrics, automatically remediate failures, modify generated output, or apply
Runtime Evolution.

## V6.2 Creative Memory Boundary

V6.2 Creative Memory Engine continues the V6 cognitive layer as advisory
creative-memory metadata over V6.1 learning and the V5 decision foundations.
Long-term creative memory records governed memory posture from existing
creative, artifact, and learning metadata without creating, updating,
deleting, or retrieving memory records. User preferences, style profiles,
project memory, Creative DNA, personalization, session memory evolution,
artifact history, creative lineage, and creative ontology expose inspectable
signals from prior metadata without learning preferences, applying
personalization, applying Creative DNA, reconstructing artifacts, inferring
lineage, or mutating ontology relationships. The secondary creative memory
surface covers preference learning, user modeling, memory consolidation,
retrieval intelligence, retrieval planning, memory conflict resolution,
memory explainability, memory safety policies, creative taste modeling, and
creative preference evolution as advisory posture only. Governance and safety
metadata records memory-storage, preference-learning, HITL, explainability,
safety, and no-automation boundaries, while the runtime failure path audit
verifies the V6.2 surfaces against runtime failure checklist coverage.

It must not write creative memory storage, execute memory retrieval, execute
memory consolidation, create or update user models, execute preference
learning, mutate preferences, apply personalization, apply Creative DNA,
persist artifact history, infer creative lineage, infer ontology
relationships, materialize semantic graphs, enforce governance or safety
policies, emit HITL requests, request human input, activate automation,
change provider/model routing, execute providers, execute or control
workflows, mutate workflow graphs, trigger retries or refinements, write
persistent storage, mutate generated output, or apply Runtime Evolution.

## V6.3 Knowledge Evolution Boundary

V6.3 Knowledge Evolution Engine continues the V6 cognitive layer as advisory
knowledge evolution metadata over V6.2 creative memory, V6.1 learning, V5
decision foundations, and existing retrieval/source metadata. Automatic KB
Updates, Documentation Intelligence, Embedding Refresh, Retrieval Evolution,
Ranking Optimization, Knowledge Health Monitoring, Knowledge Quality Scoring,
Knowledge Gap Detection, Knowledge Conflict Resolver, Knowledge Drift
Detection, Source Reliability Engine, Knowledge Consolidation, Knowledge
Lifecycle Management, Knowledge Provenance Evolution, Knowledge Versioning,
Knowledge Snapshot Engine, Knowledge Rollback, Knowledge Freshness Tracking,
and Knowledge Trust Score remain individually traceable roadmap surfaces. The
core surface, secondary surface, governance/safety metadata, and runtime
failure path audit aggregate those surfaces for inspection and audit without
grouping away their individual roadmap identities.

It must not execute automatic KB updates, fetch documentation, refresh
embeddings, execute retrieval, mutate retrieval configuration, mutate ranking,
run health monitoring, compute quality or trust scores, execute gap detection,
resolve conflicts, detect drift, score source reliability, consolidate
knowledge, manage lifecycle state, mutate provenance graphs, mutate version
graphs, execute snapshots, execute rollback, run freshness scans, write KB
storage, update source records, enforce governance or safety policies, emit
HITL requests, request human input, activate automation, change provider/model
routing, execute providers, execute or control workflows, mutate workflow
graphs, trigger retries or refinements, write persistent storage, mutate
generated output, or apply Runtime Evolution.

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
- active V6 learning, memory persistence, or long-horizon adaptation
