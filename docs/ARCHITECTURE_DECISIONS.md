# Architecture Decisions

## Runtime Ownership

- Keep the Python backend as the source of truth for request handling,
  retrieval, memory, planning metadata, provider execution, artifact metadata,
  review, bounded refinement, finalization, and failure handling.
- Keep the LangGraph workflow compact. Internal creative intelligence,
  generative design, artifact intelligence, evaluation, workstation, V4
  registry layers, V5.1 execution optimization helpers, V5.2 model-routing
  helpers, V5.3 performance helpers, V5.4 production observability helpers,
  V5.5 adaptive execution policy helpers, and V5.6 production-release
  readiness helpers, V6.1 adaptive learning helpers, V6.2 creative memory
  helpers, V6.3 knowledge evolution helpers, V6.4 autonomous research
  helpers, V6.5 self-evolution governance helpers, and V6.6 cognitive OS
  core helpers, V7.1 runtime graph consolidation helpers, V7.2 typed failure
  contracts, V7.3 registry consolidation helpers, V7.4 quality gates, and
  V7.5 production API/runtime contracts are metadata, validation, or bridge
  stabilization surfaces, not additional runtime nodes. V7.8 decomposes the
  workflow runtime implementation into builder, registry, and node handler
  modules plus isolated transition selector exports while preserving the same
  compact node set.
- Keep the Next.js workstation responsible for product inspection, preview,
  comparison, export, telemetry, workflow visibility, and operator controls.

## Persistence

Chroma remains the only persistent retrieval and memory database. Passive V4.1,
V4.2, V4.3, V4.4, V4.5, and V4.6 registries, advisory V5.1 optimization
helpers, advisory V5.2 model-routing helpers, advisory V5.3 performance
helpers, read-only V5.4 production observability helpers, and controlled V5.5
adaptive execution policy helpers, V5.6 production-release readiness helpers,
V6.1 adaptive learning helpers, V6.2 creative memory helpers, V6.3 knowledge
evolution helpers, V6.4 autonomous research helpers, and V6.5 self-evolution
governance helpers do not create storage backends, write blackboard state,
write replay storage, persist cache entries, persist learning memory, write
creative memory storage, write research memory storage, write KB storage,
update source records, discover sources, fetch external research sources,
write telemetry or trace stores, emit monitoring events, allocate resources,
create release artifacts, generate report artifacts, apply proposals, rewrite
prompts, mutate workflows, mutate routing, mutate memory or retrieval,
execute rollback, activate the Cognitive OS, apply execution graphs,
schedule, plan, route, write blackboard state, enforce governance or safety
policies, emit HITL requests, apply HITL decisions, run deployment storage
writes, mutate runtime graph contracts, apply graph diffs, persist trace
records, or introduce runtime synchronization behavior.
V7.5 production API/runtime contracts reuse the existing Chroma retrieval
database and workspace-session persistence boundary. They do not create new
storage backends, write telemetry stores, persist traces, mutate retrieval
ownership, or change workspace storage ownership.

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
- V4.6 Agentic Studio Hardening registries describe passive audit and
  hardening coverage across the V4.1-V4.5 Agentic Studio stack.
- V5.1 Execution Optimization Engine helpers describe advisory execution graph,
  cost, complexity, creative complexity, context budget, exploration, routing,
  compression, summarization, cache, reuse, pruning, forecasting, path,
  strategy, consistency, and failure-path metadata over the existing compact
  workflow.
- V5.2 Intelligent Model Routing Engine helpers describe advisory model routing,
  local/cloud and hybrid routing, quality/cost optimization, cost estimation,
  budget policy, HITL budget gate, runtime recommendation, execution policy,
  model recommendation, capability matrix, provider matrix, quality prediction,
  cost prediction, creative quality/diversity/consistency prediction,
  explainability, architecture consistency, and failure-path metadata over the
  existing compact workflow.
- V5.3 Performance Engine helpers describe advisory parallel scheduling,
  latency, async execution, streaming, retry policy, load balancing, execution
  profiling, workflow replay, execution replay, bottleneck detection,
  throughput, performance prediction, performance benchmarking, reasoning
  budget, performance regression, resource utilization, architecture
  consistency, and failure-path metadata over the existing compact workflow.
- V5.4 Production Observability helpers describe read-only token, cost,
  quality, performance, production telemetry, workflow diagnostics, agent
  diagnostics, routing diagnostics, escalation diagnostics, failure analysis,
  error intelligence, workflow health, system health, creative analytics,
  confidence analytics, creative diversity analytics, runtime timeline,
  workflow explainability, architecture consistency, and failure-path metadata
  over the existing compact workflow.
- V5.5 Adaptive Execution Intelligence helpers describe controlled adaptive
  execution policy and simulation plus advisory hybrid workflow optimization,
  escalation optimization, agent activation, adaptive cost/quality and latency
  posture, dynamic execution strategy, dynamic agent and resource allocation,
  workflow self-tuning, execution confidence, workflow risk, creative
  exploration, emergence, agent diversity, reflection budget, adaptive policy
  explainability, architecture consistency, and failure-path metadata over the
  existing compact workflow.
- V5.6 Production Release helpers describe final optimization, packaging,
  release-candidate, demo asset, deployment, production readiness, creative
  readiness, architecture freeze, release audit, final hardening, architecture
  consistency, and failure-path metadata over the existing V5 architecture
  without adding deployment automation or new core runtime architecture.
- V6.1 Adaptive Learning Engine helpers describe advisory learning,
  success/failure, strategy, technique, runtime, routing, artifact,
  evaluation, continuous improvement, pattern discovery, replay, confidence,
  creative success/failure, governance, and failure-path metadata over the V5
  decision foundations without applying feedback or training models.
- V6.2 Creative Memory Engine helpers describe advisory memory, preferences,
  style, project, Creative DNA, personalization, session, artifact, lineage,
  ontology, preference learning, user modeling, consolidation, retrieval,
  conflict, explainability, safety, taste, preference evolution, governance,
  and failure-path metadata over V6.1 learning and V5 decision foundations
  without writing creative memory, applying personalization, or changing
  runtime behavior.
- V6.3 Knowledge Evolution Engine helpers describe advisory automatic KB
  updates, documentation intelligence, embedding refresh, retrieval evolution,
  ranking optimization, health, quality, gap, conflict, drift, source
  reliability, consolidation, lifecycle, provenance, versioning, snapshot,
  rollback, freshness, trust, governance, and failure-path metadata over V6.2
  creative memory, V6.1 learning, V5 decision foundations, and existing
  retrieval/source metadata without writing KB storage, updating source
  records, mutating retrieval configuration, applying policies, or changing
  runtime behavior.
- V6.4 Autonomous Research Engine helpers describe advisory research planning,
  decomposition, paper/web research posture, cross-source comparison,
  distillation, KB enrichment posture, reports, memory, validation,
  credibility, contradiction, confidence, gap, recommendation, execution
  policy, HITL policy, creative research, cross-domain inspiration, core,
  secondary, governance, and failure-path metadata over V6.3 knowledge
  evolution, V6.2 creative memory, V6.1 learning, V5 decision foundations,
  and existing retrieval/source metadata without uncontrolled web access,
  external downloads, source discovery, KB writes, storage writes, provider
  execution, workflow control, HITL emission, or runtime behavior changes.
- V6.5 Self Evolution Engine helpers describe advisory prompt, workflow,
  benchmark, quality, cost, optimization, architecture, strategy, agent,
  routing, memory, retrieval, self-improvement, creative, taste, reasoning,
  ranking, cost/benefit, risk, expected-impact, rollback, core, secondary,
  governance, and failure-path metadata over V6.1 learning, V6.2 creative
  memory, V6.3 knowledge evolution, and V6.4 autonomous research signals
  without proposal application, prompt rewriting, workflow mutation, routing
  mutation, memory or retrieval mutation, storage writes, provider execution,
  report artifact generation, Runtime Evolution, or output mutation.
- V6.6 Cognitive Operating System Core helpers describe advisory unified
  graph, registry, learning, optimization, state, profile, reasoning,
  planning, governance, creative cognition, identity, emergence, scheduler,
  planner, router, blackboard, explanation, safety, HITL, execution graph,
  consolidation, core, secondary, governance/safety, and failure-path metadata
  over V5 Decision Engine and the V6.1 learning, V6.2 memory, V6.3
  knowledge, V6.4 research, and V6.5 self-evolution sequence without OS
  activation, execution graph application, scheduler/planner/router
  application, blackboard storage writes, governance or safety enforcement,
  HITL emission, HITL decision application, provider execution, Runtime
  Evolution, or output mutation.
- V7.1 Runtime Graph Consolidation helpers describe read-only graph
  contracts, subgraph boundaries, node handler references, state
  normalization, visualization, validation, invariant checks, trace records,
  explainability, diffing, determinism, and static relative cost/latency
  profiling over the existing compact LangGraph workflow. They preserve the
  V6.6 unified execution graph as read-only metadata and do not add
  LangGraph nodes, compile alternate graphs, invoke handlers, change node
  order, change provider/model routing, execute traces, apply graph diffs,
  write storage, apply Runtime Evolution, or mutate generated output.
- V7.2 Typed Failure Taxonomy helpers describe passive typed failure
  definitions, node-specific models, planning sub-helper models,
  provider/stream models, serialization models, workstation/client boundary
  models, event contracts, recovery invariants, regression scenarios,
  recovery strategies, explainability, severity and root-cause
  classification, analytics, reproducibility, ownership, fix
  recommendations, and knowledge-base metadata. They do not classify live
  failures, intercept exceptions, execute recovery, trigger retries, route
  providers/models, execute providers, control workflows, subscribe to
  streams, write storage, apply Runtime Evolution, or mutate generated
  output.
- V7.3 Registry & Contract Consolidation helpers describe passive registry
  family split, shared builders, shared passive boundary base models,
  inventory, coverage, normalized schema, import/export, review,
  compatibility, migration, explainability, dependency graph, diff, and
  simplification metadata without applying migrations, rewriting imports,
  changing provider routing, controlling workflows, writing storage, or
  mutating generated output.
- V7.4 E2E Quality & CI Hardening describes Playwright, Vitest, backend log,
  docs, dashboard, CI, release-checklist, performance-budget, and workstation
  regression gates over existing surfaces without changing backend API
  contracts, provider/model routing, workflow control, storage ownership,
  Runtime Evolution, generated output, or release operations.
- V7.5 Production API & Runtime Stabilization describes versioned API, error,
  stream, workspace-session, and health contracts; route manifest hardening;
  production configuration validation; Chroma dependency health reporting;
  health/live/ready endpoints; telemetry-ready API events; structured logging
  configuration; configuration migration aliases; and release checklist
  generation over the existing backend bridge. It preserves provider/model
  routing, LangGraph workflow order, prompt/Jinja rendering, retrieval
  ownership, workspace storage ownership, and generated output semantics.
- V7.8 Workflow Runtime Decomposition describes the internal module split for
  the live workflow runtime: `runtime.graph_builder` owns LangGraph
  construction, `runtime.nodes.registry` owns node/edge registrations, and
  `runtime.nodes.transitions` owns transition selector logic. V7.10 preserves
  that runtime shape while moving grouped node execution, shared state helpers,
  and stream emission helpers into focused `runtime.nodes` modules, with
  `runtime.nodes.handlers` retained as a compatibility facade. It preserves the
  same graph topology, node ordering, state transitions, provider routing,
  streaming payloads, workspace behavior, compatibility imports, and generated
  outputs.

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

## V4.6 Boundary Decision

V4.6 Agentic Studio Hardening is passive audit and hardening metadata over the
V4.1-V4.5 Agentic Studio stack. It records coverage for agent contracts,
policy, hybrid workflow, registry discoverability, memory/context boundaries,
collaboration, creative diversity, explainability, reliability, determinism,
telemetry, cost, performance, architecture consistency, final hardening, and
LangGraph error paths.

It must not execute hardening checks, add LangGraph nodes, bypass failure
normalization, activate passive registries, control workflows, invoke agents,
select providers or models, route providers or models, select runtimes, trigger
retries, mutate storage, execute artifacts, mutate generated output, or change
workflow order. More generally, it does not change provider/model routing,
does not introduce runtime recovery behavior, and does not modify generated
output.

## V5.1 Boundary Decision

V5.1 Execution Optimization Engine is an advisory metadata layer only. It may
expose graph topology analysis, workflow cost and complexity estimates,
creative complexity pressure, context and exploration budget plans, context
route recommendations, compression and summarization metadata, deterministic
in-memory cache status, context reuse opportunities, pruning candidates,
forecast scenarios, path optimization candidates, selected strategy metadata,
architecture consistency checks, and runtime failure-path audit coverage.

It must not add LangGraph nodes, compile or execute alternate graphs, enforce
budgets, apply pruning, select runtime paths, apply strategies, route providers
or models, trigger retries, mutate prompts, write persistent cache storage,
change retrieval or memory ownership, control workflows, activate passive
registries, or modify generated output. More generally, it does not change
provider/model routing, does not introduce autonomous execution optimization
behavior, and does not modify generated output.

## V5.2 Boundary Decision

V5.2 Intelligent Model Routing Engine is an advisory model-routing metadata
layer only. It may expose model route candidates, local/cloud posture, hybrid
routing posture, quality/cost optimization candidates, cost estimates, budget
policy posture, HITL budget gate posture, runtime recommendations, execution
policy posture, model recommendations, model and provider capability matrices,
quality and cost predictions, creative quality/diversity/consistency
predictions, routing explanations, architecture consistency coverage, and
runtime failure-path audit coverage.

It must not apply routing, select or switch providers or models, execute
providers, enforce budgets, emit HITL requests, request human input, select
runtimes, control workflows, trigger retries, mutate prompts, write persistent
storage, activate passive registries as runtime behavior, apply Runtime
Evolution, or modify generated output. More generally, it does not change
provider/model routing, does not introduce autonomous model selection behavior,
and does not modify generated output.

## V5.3 Boundary Decision

V5.3 Performance Engine is an advisory performance metadata layer only. It may
expose parallel scheduling candidates, latency optimization posture, async
execution readiness, streaming optimization posture, retry policy posture,
load balancing posture, execution profiling candidates, workflow replay
planning, execution replay planning, bottleneck detection, throughput
optimization, performance predictions, benchmark scenarios, reasoning budget
recommendations, performance regression signals, resource utilization
recommendations, architecture consistency coverage, and runtime failure-path
audit coverage.

It must not measure live performance, install profilers, collect traces,
execute workflows, execute benchmarks, execute replay, allocate resources,
autoscale, enforce capacity or budgets, select runtimes, control workflows,
trigger retries, route providers or models, mutate prompts, write persistent
storage, activate passive registries as runtime behavior, apply Runtime
Evolution, or modify generated output. More generally, it does not change
provider/model routing, does not introduce autonomous performance optimization
behavior, and does not modify generated output.

## V5.4 Boundary Decision

V5.4 Production Observability is a read-only observability metadata layer only.
It may expose token, cost, quality, performance, production telemetry,
workflow diagnostic, agent diagnostic, routing diagnostic, escalation
diagnostic, failure, error, workflow health, system health, creative analytics,
confidence analytics, creative diversity analytics, runtime timeline, workflow
explainability, architecture consistency, and runtime failure-path audit
coverage.

It must not collect live metrics, emit telemetry or alerts, capture traces,
execute health checks, classify live errors, remediate failures, reconstruct
timelines, record provenance, generate explanations, request human review,
execute or control workflows, trigger retries, route providers or models,
mutate prompts, write persistent storage, activate passive registries as
runtime behavior, apply Runtime Evolution, or modify generated output. More
generally, it does not change provider/model routing, does not introduce an
active telemetry or remediation runtime, and does not modify generated output.

## V5.5 Boundary Decision

V5.5 Adaptive Execution Intelligence is a controlled adaptive execution policy
and simulation layer. It may evaluate task-aware options, provider/model path
readiness, configured credential metadata, local runtime/model availability
metadata, hybrid workflow policy, fallback and escalation policy, and
Manual/Assisted/Auto mode semantics into actionable allow/confirm/block
decisions.

It must not execute providers, mutate configured provider/model routing,
silently switch providers or models, instantiate or invoke agents, allocate
agents or resources, measure runtime resources, enforce budgets, emit HITL
requests, request human input on its own, compile graphs, execute or control
workflows, mutate workflow graphs, trigger retries or refinements, mutate
prompts, write persistent storage, activate passive registries as runtime
behavior, apply Runtime Evolution, or modify generated output. Local model
download, provider provisioning, runtime installation, and Runtime Evolution
automation remain manual/HITL-only surfaces.

## V5.6 Boundary Decision

V5.6 Production Release is a production-readiness metadata layer. It may
inspect existing package metadata, environment templates, runtime path
assumptions, release-candidate posture, demo assets, deployment assumptions,
execution-mode safety, explainability fields, deterministic failure posture,
production readiness, creative readiness, architecture freeze posture, release
audit posture, guarded final hardening actions, architecture consistency, and
runtime failure-path coverage.

It must not introduce new core architecture, execute providers, mutate
configured provider/model routing, silently switch providers or models,
generate assets, execute retrieval, run package builds, install dependencies,
deploy services, create containers, provision providers, install runtimes,
download local models, emit HITL requests, compile graphs, execute or control
workflows, mutate workflow graphs, write persistent storage, create release
artifacts, execute hardening actions, create runtime failure handlers, mutate
terminal routing, merge, push, tag, apply Runtime Evolution, or modify
generated output. External deployment manifests are explicit guarded
assumptions, not automatic deployment work.

## V6.1 Boundary Decision

V6.1 Adaptive Learning Engine starts the V6 cognitive layer as controlled
learning metadata over the V5 decision and adaptive execution foundations. Its
initial learning engine may combine execution confidence, workflow risk, and
workflow self-tuning metadata into inspectable learning candidates, priority
posture, pattern tags, evidence, and review actions. Workflow success
tracking may derive success indicators from those learning candidates without
observing live outcomes or persisting metrics. Failure tracking may derive
failure indicators from read-only failure analysis and adaptive learning
metadata without observing, classifying, routing, handling, or repairing live
failures. Strategy learning may derive strategy patterns from advisory
adaptive execution strategy metadata and adaptive learning signals without
applying or mutating strategy selection. Technique learning may derive
technique patterns from read-only creative technique metadata and adaptive
learning signals without rendering prompts, applying techniques, selecting
runtimes, or executing artifacts. Runtime learning may derive runtime patterns
from read-only runtime capability metadata and adaptive learning signals
without selecting runtimes, probing local runtimes, installing dependencies,
or changing preview behavior. Routing learning may derive route patterns from
read-only task-aware routing metadata and adaptive learning signals without
applying routes, switching providers or models, executing providers, probing
local runtimes, or assuming credentials. Artifact learning may derive
artifact-shape, capability, and risk patterns from read-only artifact
planning and artifact capability metadata without selecting, mutating,
generating, executing, merging, exporting, or previewing artifacts.
Evaluation learning may derive evaluation patterns from read-only evaluation
engine contract metadata without running evaluations, evaluating generated
output, mutating scores or confidence, executing reflection loops, generating
reports, or changing workflow order. Continuous improvement signals may
synthesize read-only V6.1 success, failure, artifact, and evaluation learning
metadata into advisory improvement candidates without applying feedback,
persisting learning memory, updating policies, observing runtime outcomes, or
changing workflows. Success pattern discovery may derive candidate success
patterns from read-only workflow success and continuous improvement metadata
without observing live success, collecting telemetry, persisting success
metrics, applying patterns, or applying feedback. Failure pattern discovery
may derive guarded failure patterns from read-only failure tracking and
continuous improvement metadata without observing live failures, classifying
live errors, routing terminal failures, handling or repairing failures, or
mutating terminal routing. Learning governance may describe memory, feedback,
policy, HITL, explainability, and no-automation boundaries without persisting
memory, applying feedback, updating or enforcing policies, emitting HITL
requests, or requesting human input.
Learning replay may record replay scenarios, source learning signal
references, expected replay insight, replay confidence, and replay safety
boundaries without executing learning replay, workflow replay, provider calls,
storage writes, generated-output mutation, or Runtime Evolution. Learning
confidence calibration may map confidence before/after bands from existing
learning signals and V5 execution/confidence metadata with rationale,
uncertainty factors, and HITL requirements for low or risky confidence without
model training, feedback application, storage writes, or runtime mutation.
Creative success learning may specialize success patterns for creative coding
artifact, aesthetic, usefulness, and originality dimensions with
explainability and without generated-output mutation, automatic preference
mutation, or storage writes. Creative failure learning may specialize failure
patterns for artifact, preview, runtime, aesthetic, prompt, and retrieval
failure modes with explainability and without generated-output mutation,
automatic remediation, or storage writes.
The V6.1 runtime failure path audit verifies those learning surfaces against
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

## V6.2 Boundary Decision

V6.2 Creative Memory Engine is an advisory creative-memory metadata layer
only. It may expose long-term creative memory, user preferences, style
profiles, project memory, Creative DNA, personalization posture, session
memory evolution, artifact history, creative lineage, creative ontology,
preference learning posture, user modeling, memory consolidation posture,
memory retrieval intelligence, memory retrieval planning, conflict
resolution posture, memory explainability, memory safety policy posture,
creative taste modeling, creative preference evolution, governance, and
runtime failure-path audit coverage.

It must not write creative memory storage, execute memory retrieval, execute
memory consolidation, create or update user models, execute preference
learning, mutate preferences, apply personalization, apply Creative DNA,
persist artifact history, infer creative lineage, infer ontology
relationships, materialize semantic graphs, enforce governance or safety
policies, emit HITL requests, request human input, activate automation,
change provider/model routing, execute providers, execute or control
workflows, mutate workflow graphs, trigger retries or refinements, write
persistent storage, activate passive registries as runtime behavior, apply
Runtime Evolution, or modify generated output. More generally, it does not
change provider/model routing, does not introduce automatic personalization
or active memory behavior, and does not modify generated output.

## V6.3 Boundary Decision

V6.3 Knowledge Evolution Engine is an advisory knowledge evolution metadata
layer only. It may expose Automatic KB Updates, Documentation Intelligence,
Embedding Refresh, Retrieval Evolution, Ranking Optimization, Knowledge Health
Monitoring, Knowledge Quality Scoring, Knowledge Gap Detection, Knowledge
Conflict Resolver, Knowledge Drift Detection, Source Reliability Engine,
Knowledge Consolidation, Knowledge Lifecycle Management, Knowledge Provenance
Evolution, Knowledge Versioning, Knowledge Snapshot Engine, Knowledge
Rollback, Knowledge Freshness Tracking, Knowledge Trust Score, core surface,
secondary surface, governance/safety, and runtime failure-path audit coverage.
The 19 contractual roadmap items must remain individually traceable for
roadmap coverage verification, Codex Engineering Audit classification, and
future capability-scoped fixes.

It must not execute automatic KB updates, fetch documentation, refresh
embeddings, execute retrieval, mutate retrieval configuration, mutate ranking,
run health monitoring, compute quality or trust scores, execute gap detection,
resolve conflicts, detect drift, score source reliability, consolidate
knowledge, manage lifecycle state, mutate provenance graphs, mutate version
graphs, execute snapshots, execute rollback, run freshness scans, write KB
storage, update source records, enforce governance or safety policies, emit
HITL requests, request human input, activate automation, change provider/model
routing, execute providers, execute or control workflows, mutate workflow
graphs, trigger retries or refinements, write persistent storage, activate
passive registries as runtime behavior, apply Runtime Evolution, or modify
generated output. More generally, it does not change provider/model routing,
does not introduce active knowledge evolution behavior, and does not modify
generated output.

## V6.4 Boundary Decision

V6.4 Autonomous Research Engine is an advisory research metadata layer only.
It may expose Research Planner, Research Decomposer, Paper Research, Web
Research, Cross-source Comparison, Knowledge Distillation, Automatic KB
Enrichment, Research Reports, Research Memory, Source Validation Engine,
Source Credibility Engine, Contradiction Detection, Research Confidence
Engine, Research Gap Discovery, Research Recommendation Engine, Research
Execution Policy, Research HITL Policies, Creative Research Engine,
Cross-domain Inspiration Discovery, core surface, secondary surface,
governance/safety, and runtime failure-path audit coverage. The 19
contractual roadmap items must remain individually traceable for roadmap
coverage verification, Codex Engineering Audit classification, and future
capability-scoped fixes.

It must not execute research plans, create research tasks, perform
uncontrolled web access, browse the web, crawl sites, fetch external sources,
download papers, execute paper or web research, execute cross-source
comparison, execute knowledge distillation, execute KB enrichment, write KB
storage, generate research reports, write research memory, execute source
validation, score source credibility, execute contradiction detection, score
research confidence, execute gap discovery, generate or execute research
recommendations, apply research execution policy, emit HITL requests, request
human input, apply HITL decisions, generate creative output, execute active
cross-domain inspiration discovery, perform live cross-domain search, enforce
governance or safety policies, activate automation, change provider/model
routing, execute providers, execute or control workflows, mutate workflow
graphs, trigger retries or refinements, write persistent storage, activate
passive registries as runtime behavior, apply Runtime Evolution, or modify
generated output. More generally, it does not change provider/model routing,
does not introduce autonomous web or paper research behavior, and does not
modify generated output.

## V6.5 Boundary Decision

V6.5 Self Evolution Engine is an advisory self-evolution orchestration and
governance metadata layer only. It may expose Prompt Evolution, Workflow
Evolution, Benchmark Engine, Quality Trends, Cost Trends, Autonomous
Optimization Suggestions, Architecture Evolution Engine, Workflow Mutation
Engine, Strategy Evolution Engine, Agent Evolution Policies, Routing Evolution
Policies, Memory Evolution Policies, Retrieval Evolution Policies,
Self-Improvement Proposals, Creative Evolution Policies, Taste Evolution
Engine, Reasoning Evolution Engine, Improvement Ranking Engine, Cost /
Benefit Analysis, Risk Analysis, Expected Impact Estimator, Rollback Strategy
Generator, core surface, secondary surface, governance/safety, and runtime
failure-path audit coverage. The 22 contractual roadmap items must remain
individually traceable for roadmap coverage verification, Codex Engineering
Audit classification, and future capability-scoped fixes.

It may read deterministic metadata signals from V6.1 Adaptive Learning, V6.2
Creative Memory, V6.3 Knowledge Evolution, and V6.4 Autonomous Research,
compare improvement opportunities across capabilities, rank proposals by
impact, cost, risk, confidence, dependencies, and rollback feasibility,
explain why a proposal exists, identify upstream signal sources, identify
downstream systems that may be affected, and prepare advisory evolution
report metadata under strict HITL governance.

It must not apply Runtime Evolution, autonomously self-modify code, rewrite
prompts, mutate workflows, mutate routing, mutate memory, mutate retrieval,
write storage, generate report artifacts, execute rollback, enforce
governance or safety policies, emit HITL requests, request human input, apply
HITL decisions, activate automation, mutate cost/risk/ranking/impact policy,
perform live pricing lookup, enforce budgets, execute providers, invoke
agents, execute or control workflows, mutate workflow graphs, activate
passive registries as runtime behavior, or modify generated output. More
generally, it does not change provider/model routing, does not introduce
hidden autonomous self-improvement behavior, and does not modify generated
output.

## V6.6 Boundary Decision

V6.6 Cognitive Operating System Core is an advisory cognitive OS metadata
layer only. It may expose the Unified Cognitive Graph, Unified Memory Graph,
Unified Knowledge Graph, Unified Agent Registry, Unified Capability Registry,
Cross-System Learning Layer, Cross-System Optimization Layer, Cognitive State
Engine, Cognitive Profile Engine, Meta-Reasoning Layer, Meta-Planning Layer,
Cognitive Governance Layer, Creative Cognition Layer, Creative Identity
Layer, Emergent Creativity Layer, Cognitive Scheduler, Cognitive Planner,
Cognitive Router, Cognitive Blackboard, Cognitive Explanation Engine,
Cognitive Safety Layer, Cognitive HITL Layer, Unified Execution Graph, Core
OS Consolidation, core surface, secondary surface, governance/safety, and
runtime failure-path audit coverage. The 24 contractual roadmap items must
remain individually traceable for roadmap coverage verification, Codex
Engineering Audit classification, Cross-Capability Governance Audit, Unified
Cognitive System Verification, and future capability-scoped fixes.

It may read deterministic metadata from V5 Decision Engine, V6.1 Adaptive
Learning, V6.2 Creative Memory, V6.3 Knowledge Evolution, V6.4 Autonomous
Research, and V6.5 Self Evolution, compose the Learning -> Memory ->
Knowledge -> Research -> Self Evolution -> Cognitive Core sequence through
the Unified Cognitive Graph, and expose ownership, dependency traceability,
governance, explainability, safety, and HITL contracts for review.

It must not activate the Cognitive OS, execute graph nodes, traverse
execution edges, apply scheduling, planning, routing, blackboard,
explanation, governance, safety, or HITL behavior, enforce governance or
safety policies, emit HITL requests, request human input, apply HITL
decisions, activate automation, write storage, invoke agents, execute
providers, mutate workflows, prompts, routing, memory, retrieval, generated
output, runtime state, or apply Runtime Evolution. More generally, it does
not change provider/model routing, does not introduce hidden autonomous
cognitive execution behavior, and does not modify generated output.

## V7.1 Runtime Graph Consolidation Boundary

V7.1 Runtime Graph Consolidation is a read-only graph contract and
diagnostics layer over the current assistant workflow. The live execution
source remains `creative_coding_assistant.orchestration.workflow_graph`; the
static topology source remains `execution_graph_analyzer`; the Cognitive OS
projection remains `unified_execution_graph`; and V7.1 owns
`runtime_graph_consolidation` for contracts, validation, invariants, static
traces, explainability, diffing, determinism, visualization, and relative
cost/latency profiles.

It must not change user-visible behavior, add LangGraph nodes, compile or
execute alternate graphs, invoke node handlers from diagnostics, mutate
workflow order, route providers or models, enforce budgets, persist trace or
profile storage, apply graph diffs, apply Runtime Evolution, start V7.2
failure taxonomy work, or mutate generated output.

## V7.2 Typed Failure Taxonomy Boundary

V7.2 Typed Failure Taxonomy is a passive typed failure contract layer over the
existing workflow graph, stream event contracts, planning helper payload specs,
provider/stream boundaries, serialization surfaces, and client/workstation
boundaries. It owns `typed_failure_taxonomy` for stable failure type ids,
node-specific failure models, planning sub-helper failure models,
provider/stream failure models, serialization failure models, workstation/client
boundary failure models, event contracts, recovery invariants, regression
scenarios, recovery strategies, explainability, severity classification,
analytics contracts, root-cause classification, reproducibility records,
ownership records, fix recommendations, and in-memory knowledge-base entries.

It must not classify live failures, intercept exceptions, execute recovery,
trigger retries, route providers or models, execute providers, execute or
control workflows, mutate workflow graphs, subscribe to streams, write
persistent storage, apply Runtime Evolution, or mutate generated output. All
18 contractual V7.2 roadmap items remain individually traceable for roadmap
coverage verification, Codex Engineering Audit classification, and future
capability-scoped fixes.

## V7.3 Registry & Contract Consolidation Boundary

V7.3 Registry & Contract Consolidation is a passive registry and schema
consolidation layer over existing source registries. It owns
`registry_contract_consolidation` for registry family split metadata, shared
registry builders, shared passive boundary base models, source registry
inventory, coverage reports, normalized schema records, public export audits,
Pydantic/Jinja2/style/comment/logging review findings, registry package and
contract simplification posture, metadata-to-code ratio review, integrity
verification, compatibility checks, schema evolution planning, version
migration descriptors, explainability records, dependency graphs, diff reports,
and architecture simplification review.

It must not move mature registry modules, rewrite schemas, apply migrations,
change import behavior, route providers or models, execute providers, execute
or control workflows, mutate workflow graphs, mutate prompt rendering or Jinja
templates, change logging configuration, write persistent storage, apply
Runtime Evolution, or mutate generated output. All 24 contractual V7.3 roadmap
items remain individually traceable for roadmap coverage verification, Codex
Engineering Audit classification, and future capability-scoped fixes.

## V7.4 E2E Quality & CI Hardening Boundary

V7.4 E2E Quality & CI Hardening is a quality infrastructure layer over the
existing backend, Next.js workstation, preview, persistence, stream, retrieval,
and documentation surfaces. It adds Playwright configuration and browser E2E
tests, deterministic mocked assistant and workspace-session responses, browser
console and request gates, backend log scanning, docs/Mermaid linting,
GitHub Actions CI orchestration, release checklist generation, performance
budgets, and test coverage dashboard metadata.

It does not change provider/model routing, execute providers, change backend
API contracts, alter the LangGraph workflow, mutate workflow graphs, change
preview runtime behavior, write workspace storage outside existing
persistence contracts, change retrieval or KB ownership, apply Runtime
Evolution, mutate generated output, start V7.5 production API/runtime work, or
perform merge, push, or tag operations. All 23 contractual V7.4 roadmap items
remain traceable to explicit quality gates and validation evidence.

## V7.5 Production API & Runtime Stabilization Boundary

V7.5 Production API & Runtime Stabilization is a bridge-runtime hardening layer
over the existing Python backend WSGI surfaces. It owns shared versioned API
contracts, error response contracts, assistant stream contract headers,
workspace-session contract headers, health/live/ready endpoint contracts,
route manifest stabilization, production configuration validation, dependency
health reports, telemetry-ready API events, structured logging configuration,
configuration migration aliases, and V7.5 release checklist generation.

It may normalize browser-facing API errors, preserve legacy top-level error
fields for compatibility, expose deterministic liveness/readiness payloads,
guard the local wsgiref bridge from accidental production use, and report
dependency/configuration readiness without importing external services. It
must not change provider/model routing, execute providers differently, change
LangGraph workflow order, mutate workflow graphs, alter prompt or Jinja
rendering, change retrieval or KB ownership, change workspace storage
ownership, emit external telemetry, deploy services, apply Runtime Evolution,
mutate generated output, or perform merge, push, or tag operations. All 22
contractual V7.5 roadmap items remain traceable to explicit API/runtime
contracts and validation evidence.

## V7.6 Orchestration Package Decomposition Boundary

V7.6 Orchestration Package Decomposition separates the flat orchestration
package into canonical runtime, metadata, governance, audit, contract, and
advisory boundary packages. Legacy root module imports remain compatibility
shims, and root package exports remain stable for existing callers.

It must not change provider/model routing, LangGraph workflow order, prompt
rendering, generated output semantics, persistence ownership, retry behavior,
stream subscriptions, telemetry emission, frontend behavior, merge, push, tag,
freeze, or V8 start state.

## V7.7 Production Deployment Foundation Boundary

V7.7 Production Deployment Foundation is deployment infrastructure around the
existing browser-facing WSGI backend bridge. It owns the production WSGI
entrypoint, Gunicorn recommendation, Dockerfile, optional docker-compose,
environment-aware CORS policy, deployment documentation, health-check/runtime
checklist guidance, CI coverage reporting, CI dependency security scanning,
Chroma posture verification, production configuration validation, and
release/deployment readiness checklist reporting.

It may tighten production CORS behavior, report guarded production
configuration, expose a Gunicorn import target, and add deployment artifacts.
It must not change creative generation behavior, provider/model routing,
LangGraph workflow order, prompt rendering, generated output semantics,
workspace persistence semantics, frontend UI behavior, auth/rate-limit
enforcement, merge, push, tag, freeze, or V8 start state.

## V7.8 Workflow Runtime Decomposition Boundary

V7.8 Workflow Runtime Decomposition is an internal runtime maintainability
refactor over the existing assistant LangGraph workflow. It owns the split from
the former monolithic runtime workflow graph module into a compatibility shim,
graph builder, node registration layer, transition selector module, and node
handler module.

It may move node execution code into `runtime.nodes.handlers`, move LangGraph
construction into `runtime.graph_builder`, expose registration helpers through
`runtime.nodes.registry`, expose transition selector helpers through
`runtime.nodes.transitions`, and leave compatibility shims for existing root
and runtime imports. It must not change graph topology, node order, transition
selector behavior, retry policy, failure routing, provider/model routing,
prompt rendering, stream event payloads, workspace behavior, generated output
semantics, storage ownership, merge, push, tag, freeze, or V8 start state.

## V7.10 Workflow Node Handler Decomposition Boundary

V7.10 Workflow Node Handler Decomposition is a behavior-preserving internal
runtime refactor over the V7.8 workflow runtime split. It owns the extraction of
live node handlers from `runtime.nodes.handlers` into focused modules for
intake, routing, memory, retrieval, context assembly, planning, generation,
artifacts, review, refinement, finalization, shared workflow state helpers,
stream emission helpers, and transition logic.

It may update node registration to point at the focused modules, keep
`runtime.nodes.handlers` as a compatibility facade, and align tests and
architecture documentation with the new ownership. It must not change graph
topology, node order, state transition outcomes, retry policy, failure routing,
provider/model routing, prompt rendering, stream event payloads, workspace
behavior, generated output semantics, storage ownership, merge, push, tag,
freeze, or V8 start state.

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
- passive V4.6 agentic studio hardening metadata
- advisory V5.1 execution optimization metadata
- advisory V5.2 model-routing metadata
- advisory V5.3 performance metadata
- V7.8 workflow runtime builder, registry, transition, and node handler boundaries
- read-only V5.4 production observability metadata
- controlled V5.5 adaptive execution policy/simulation
- V5.6 production release readiness metadata
- advisory V6.1 adaptive learning metadata
- advisory V6.2 creative memory metadata
- advisory V6.3 knowledge evolution metadata
- advisory V6.4 autonomous research metadata
- advisory V6.5 self-evolution governance metadata
- advisory V6.6 cognitive OS core metadata
- read-only V7.1 runtime graph consolidation contracts and diagnostics
- passive V7.2 typed failure taxonomy contracts and registries
- passive V7.3 registry and contract consolidation metadata
- V7.4 E2E/CI quality gates and validation infrastructure
- V7.5 production API/runtime contracts and bridge stabilization
- V7.6 orchestration package boundaries and compatibility shims
- V7.7 production deployment foundation and CORS/deployment readiness gates
- future active V4 Agentic Studio, live adaptive runtime control, and later
  HoloMind / HoloGenesis Cognitive OS work

## Code Quality Rules

- Keep runtime behavior changes separate from metadata and documentation
  updates.
- Keep registry/source lists exhaustive for their capability scope.
- Add tests when docs claim source coverage or passive boundaries.
- Do not overstate passive registries as active orchestration behavior.
