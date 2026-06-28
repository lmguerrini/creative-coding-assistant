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
| V4.1 | Multi-Agent Core defines passive agent identities, contracts, roles, boundaries, and metadata over completed V3 capabilities without changing generation behavior | Agent-facing knowledge and memory access are described as metadata contracts only; no blackboard, shared context view, or retrieval side effect is implemented | Agent contracts remain export-only metadata and do not enter workflow control, provider routing, runtime selection, retries, or execution | Agentic Studio can inspect role and boundary contracts later, but no collaborative studio behavior is implemented here |
| V4.2 | Agent Orchestration defines passive orchestration contracts over V4.1 agents without activating multi-agent execution | Blackboard, shared context view, state synchronization, and handoff surfaces are metadata contracts only; no memory storage, runtime synchronization, or retrieval side effect is implemented | Orchestration registries remain export-only metadata and do not enter workflow control, provider routing, runtime selection, retries, storage behavior, or execution | Agentic Studio can inspect orchestration readiness later, but no active coordination, debate, consensus, or lifecycle UI behavior is implemented here |
| V4.3 | Hybrid Agentic Workflow defines passive V3 backbone, escalation, specialist loop, gate, debate, voting, confidence, provenance, trace, budget, normalization, handoff, threshold, quality, adaptive, and integration metadata without changing generation behavior | Agent-facing ambiguity, risk, quality, HITL, and handoff context is described as metadata only; no knowledge runtime, blackboard storage, or retrieval side effect is implemented | Hybrid workflow registries remain export-only metadata and do not enter workflow control, provider routing, model routing, runtime selection, retries, prompt rendering, storage behavior, or execution | Agentic Studio can inspect hybrid workflow readiness later, but no active escalation, agent execution, or autonomous orchestration UI behavior is implemented here |
| V4.4 | Hybrid Studio defines passive local/cloud model, hybrid execution, Auto Mode, Studio Mode, HITL, profile, comparison, workspace, snapshot, replay, and integration metadata without changing generation behavior | Agent-facing Studio workspace, conversation, snapshot, and replay context is described as metadata only; no knowledge runtime, memory storage, blackboard storage, or retrieval side effect is implemented | Hybrid Studio registries remain export-only metadata and do not activate Studio runtime, enter workflow control, route providers/models, select runtimes, trigger retries, mutate storage, persist replay data, or execute providers | Agentic Studio can inspect Hybrid Studio readiness later, but no active Studio runtime, provider execution, agent invocation, replay execution, or autonomous operator UI behavior is implemented here |
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

## V4.1 Multi-Agent Core Registries

V4.1 adds a passive Multi-Agent Core on top of the completed V3 platform. It
defines the agent society as static metadata: identities, contracts, memory
access boundaries, role ordering, role-specific authority boundaries, and
advisory operational metadata. These registries are product architecture
surfaces for inspection and future orchestration consumption; they are not
active multi-agent orchestration.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| Agent contracts | `src/creative_coding_assistant/orchestration/agent_contracts.py` | 12 | `agent_contract_registry.v1` | Describes per-agent passive inputs, outputs, capabilities, memory posture, cost, latency, and future orchestration hooks |
| Agent identities | `src/creative_coding_assistant/orchestration/agent_identities.py` | 12 | `agent_identity_registry.v1` | Defines stable agent names, role families, purposes, capability classes, authority scope, visibility, and version metadata |
| Agent memory contracts | `src/creative_coding_assistant/orchestration/agent_memory_contracts.py` | 12 | `agent_memory_contract_registry.v1` | Describes session, artifact, evaluation, provenance, and future blackboard read/write/reference boundaries without storage |
| Agent roles | `src/creative_coding_assistant/orchestration/agent_roles.py` | 12 | `agent_role_registry.v1` | Exposes static role order, lookup, role-family grouping, and capability-family grouping |
| Agent boundaries | `src/creative_coding_assistant/orchestration/agent_boundaries.py` | 12 | `agent_boundary_registry.v1` | Records allowed inputs, allowed outputs, forbidden behaviors, and role-specific boundary rationale |
| Agent metadata | `src/creative_coding_assistant/orchestration/agent_metadata.py` | 12 | `agent_metadata_registry.v1` | Provides advisory cacheability, parallelization, observability, auditability, cost, latency, and future-readiness metadata |

All V4.1 agent registries remain export-only metadata surfaces. They do not
create agents, route tasks dynamically, implement blackboard memory, materialize
shared context views, render agent contract text into provider prompts, enter
workflow event payloads, change LangGraph node order, alter provider/model
routing, select runtimes, trigger retries, execute artifacts, change final
response generation, or modify generated output.

## V4.2 Agent Orchestration Registries

V4.2 adds passive orchestration metadata on top of the V4.1 Multi-Agent Core.
The registries describe orchestration readiness, boundaries, and consistency
relationships for future Agentic Studio consumers. They are not active
multi-agent orchestration and do not create agents, coordinate live agents,
run debates, build consensus, synchronize runtime state, mutate blackboard
storage, change workflow payloads, alter prompts, route providers/models,
trigger retries, select runtimes, or modify generated output.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| Agent routing | `src/creative_coding_assistant/orchestration/agent_routing.py` | 12 profiles | `agent_routing_registry.v1` | Describes advisory route candidates per agent without changing provider/model routing, workflow routing, retries, or output |
| Blackboard memory | `src/creative_coding_assistant/orchestration/blackboard_memory.py` | 12 channels / 12 permissions | `blackboard_memory_registry.v1` | Describes planned blackboard channels and permissions without persistence, runtime reads, runtime writes, or storage backends |
| Shared context views | `src/creative_coding_assistant/orchestration/shared_context_views.py` | 12 views | `shared_context_view_registry.v1` | Describes scoped context visibility without materializing context, exposing global state, or mutating context |
| Agent dependency graph | `src/creative_coding_assistant/orchestration/agent_dependency_graph.py` | 30 nodes | `agent_dependency_graph.v1` | Describes static dependency relationships without scheduling traversal or executing agents |
| Parallel scheduling | `src/creative_coding_assistant/orchestration/agent_parallel_scheduling.py` | 6 groups | `parallel_scheduling_registry.v1` | Describes future concurrency groups without parallel execution, agent invocation, or workflow control |
| Agent coordination | `src/creative_coding_assistant/orchestration/agent_coordination.py` | 5 handoff channels | `coordination_registry.v1` | Describes responsibilities, handoff channels, and events without coordinating live agents |
| Agent debate | `src/creative_coding_assistant/orchestration/agent_debate.py` | 4 rounds | `agent_debate_registry.v1` | Describes bounded advisory debate metadata without running debate or triggering retries |
| Consensus builder | `src/creative_coding_assistant/orchestration/agent_consensus.py` | 4 voting inputs | `consensus_builder_registry.v1` | Describes voting inputs and agreement surfaces without voting, selecting outputs, or changing workflow control |
| Capability alignment | `src/creative_coding_assistant/orchestration/agent_capability_alignment.py` | 12 profiles | `agent_capability_alignment_registry.v1` | Maps V4.1 roles to V4.2 capabilities without activating capabilities or routing work |
| Escalation signals | `src/creative_coding_assistant/orchestration/agent_escalation_signals.py` | 7 signals | `agent_escalation_signal_registry.v1` | Describes advisory escalation thresholds without performing escalation, routing providers, or triggering HITL |
| Agent lifecycle | `src/creative_coding_assistant/orchestration/agent_lifecycle.py` | 12 profiles / 10 transitions | `agent_lifecycle_registry.v1` | Describes planned lifecycle states and transitions without a lifecycle engine or workflow state mutation |
| Agent state synchronization | `src/creative_coding_assistant/orchestration/agent_state_synchronization.py` | 12 profiles / 5 checkpoints | `agent_state_sync_registry.v1` | Describes sync checkpoints, stale warnings, and conflict surfaces without runtime synchronization or conflict resolution |
| Workflow-agent handoff | `src/creative_coding_assistant/orchestration/workflow_agent_handoff.py` | 5 surfaces / 12 profiles | `workflow_agent_handoff_registry.v1` | Maps V3 workflow surfaces to V4 agents without changing workflow graph, prompts, payloads, or agent execution |
| Orchestration integration | `src/creative_coding_assistant/orchestration/orchestration_contract_integration.py` | 13 registry entries | `orchestration_contract_integration.v1` | Makes V4.2 registries discoverable against V4.1 contracts without creating an active orchestration path |

All V4.2 orchestration registries remain export-only metadata surfaces. Boundary
tests assert that they do not enter provider/model routing, prompt rendering,
workflow node order, generated outputs, retry behavior, storage behavior, or
active runtime orchestration.

## V4.3 Hybrid Agentic Workflow Registries

V4.3 adds passive hybrid workflow metadata over the stable V3 runtime graph and
the V4.1/V4.2 agent contract layers. The registries describe future escalation
readiness, source coverage, and return-to-workflow context. They are not active
hybrid orchestration and do not execute escalation, invoke agents, evaluate
thresholds, route providers/models, select runtimes, trigger retries, mutate
prompts, write storage, or modify generated output.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| V3 backbone mode | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 18 node profiles | `v3_backbone_mode_registry.v1` | Declares the preserved V3 workflow backbone without changing graph order |
| Conditional multi-agent escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 conditions | `conditional_multi_agent_escalation_registry.v1` | Describes advisory escalation candidates without evaluating conditions or invoking agents |
| Specialist agent loops | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 loops | `specialist_agent_loop_registry.v1` | Describes bounded future loop candidates without executing loops or coordinating agents |
| Escalation gates | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 gates | `escalation_gate_registry.v1` | Describes advisory gates without evaluating, approving, or executing escalation |
| Creative escalation policy | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 rules | `creative_escalation_policy_registry.v1` | Describes creative-domain escalation policy without evaluating policy or triggering escalation |
| Reflection escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 profiles | `reflection_escalation_registry.v1` | Maps reflection posture without running reflection or refinement |
| Hybrid agent debate loop | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `hybrid_agent_debate_loop_registry.v1` | Describes debate readiness without running debate loops |
| Hybrid agent voting | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `hybrid_agent_voting_registry.v1` | Describes voting readiness without voting or selecting outputs |
| Agent confidence fusion | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `agent_confidence_fusion_registry.v1` | Describes confidence fusion context without calculating or weighting confidence |
| Decision provenance | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `decision_provenance_registry.v1` | Describes future decision lineage without recording provenance or writing memory |
| Escalation trace | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `escalation_trace_registry.v1` | Describes future trace context without capturing or emitting traces |
| Creative exploration budget | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `creative_exploration_budget_registry.v1` | Describes advisory exploration posture without enforcing budgets or generating variants |
| Result normalization | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `result_normalization_registry.v1` | Describes future result packet surfaces without transforming or rewriting outputs |
| Return-to-workflow handoff | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `return_to_workflow_handoff_registry.v1` | Describes handoff context without changing workflow control or prompts |
| HITL escalation gate | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `hitl_escalation_gate_registry.v1` | Describes human-review visibility without requesting review or approving escalation |
| Confidence threshold routing | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `confidence_threshold_routing_registry.v1` | Describes confidence bands without routing by confidence |
| Cost threshold routing | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `cost_threshold_routing_registry.v1` | Describes cost bands without routing by cost or enforcing budgets |
| Latency threshold routing | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `latency_threshold_routing_registry.v1` | Describes latency bands without routing by latency or selecting runtimes |
| Ambiguity escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `ambiguity_escalation_registry.v1` | Describes ambiguity posture without evaluating ambiguity or requesting clarification |
| Risk escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `risk_escalation_registry.v1` | Describes risk posture without evaluating risk or applying mitigation |
| Quality escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `quality_escalation_registry.v1` | Describes quality posture without evaluating quality or triggering refinement |
| Adaptive multi-agent escalation | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 4 profiles | `adaptive_multi_agent_escalation_registry.v1` | Describes adaptive escalation posture without orchestrating agents or executing escalation |
| Hybrid workflow integration | `src/creative_coding_assistant/orchestration/hybrid_agentic_workflow.py` | 5 stages / 43 source registries | `hybrid_workflow_registry.v1` | Exposes V4.3 source coverage without adding runtime behavior |

All V4.3 hybrid workflow registries remain export-only metadata surfaces.
Boundary tests assert that they do not enter provider/model routing, prompt
rendering, workflow node order, generated outputs, retry behavior, storage
behavior, runtime selection, or active multi-agent orchestration.

## V4.4 Hybrid Studio Registries

V4.4 adds passive hybrid studio metadata over the V4.1-V4.3 contract layers.
The registries describe future local/cloud model inspection, Studio mode
visibility, manual HITL review surfaces, simulation/replay context, workspace
inspection, and source coverage. They are not active Studio runtime and do not
activate Studio runtime, execute providers, invoke agents, control workflows,
request human input, change provider/model routing, select runtimes, trigger
retries, mutate storage, write replay storage, or modify generated output.

| Registry | Source module | Count | Serialization version | Current boundary |
| --- | --- | ---: | --- | --- |
| Local Model Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 surfaces | `local_model_registry.v1` | Describes local model candidate surfaces without local runtime discovery, local provider execution, provider/model routing, automatic model selection, retries, replay storage, or output mutation |
| Cloud Model Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 surfaces | `cloud_model_registry.v1` | Describes cloud model candidate surfaces without cloud provider execution, provider/model routing, automatic model selection, cost/latency optimization, retries, replay storage, or output mutation |
| Hybrid Execution Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `hybrid_execution_registry.v1` | Describes advisory local/cloud coordination without hybrid execution, provider execution, fallback execution, parallel model calls, provider/model routing, or automatic model selection |
| Auto Mode Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `auto_mode_registry.v1` | Describes advisory Auto Mode postures without workflow execution, automatic provider/model selection, hybrid execution, human-input requests, retries, or output mutation |
| Studio Mode Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `studio_mode_registry.v1` | Describes Studio mode surfaces without Studio runtime control, workflow control, hybrid execution, provider/model routing, artifact execution, human-input requests, or output mutation |
| HITL Decision Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `hitl_decision_registry.v1` | Describes HITL decision visibility without requesting human input, approving escalation, interrupting workflows, workflow control, provider/model routing, retries, or output mutation |
| Provider Selection Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `provider_selection_registry.v1` | Describes provider-candidate visibility without selecting providers, automatic model selection, model switching, provider execution, workflow control, human-input requests, or routing |
| Execution Simulator Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `execution_simulator_registry.v1` | Describes simulation metadata without simulation runtime execution, provider execution, artifact execution, workflow transition execution, human-input requests, retries, or output mutation |
| Model Profile Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `model_profile_registry.v1` | Describes advisory model profiles without model selection, provider execution, cost scoring, quality scoring, execution optimization, retries, or output mutation |
| Cost Profile Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `cost_profile_registry.v1` | Describes advisory cost postures without pricing lookup, cost scoring, budget enforcement, cost-based routing, provider execution, model selection, or output mutation |
| Quality Profile Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `quality_profile_registry.v1` | Describes advisory quality postures without quality scoring, quality evaluation, quality escalation, refinement triggering, workflow control, human-input requests, or output mutation |
| Local/Cloud Comparison Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `local_cloud_comparison_registry.v1` | Describes comparison metadata without local/cloud provider execution, parallel model execution, model selection, winner selection, fallback execution, cost scoring, or quality scoring |
| Agent Workspace Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `agent_workspace_registry.v1` | Describes agent workspace visibility without workspace execution, agent instantiation, agent invocation, multi-agent orchestration, memory writes, workflow control, or output mutation |
| Agent Conversation View Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `agent_conversation_view_registry.v1` | Describes conversation visibility without conversation execution, conversation persistence, agent message generation, agent invocation, memory writes, workflow control, or output mutation |
| Workspace Snapshot Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `workspace_snapshot_registry.v1` | Describes snapshot context without snapshot capture, snapshot persistence, live workspace reads, conversation recording, memory reads/writes, workspace mutation, or workflow control |
| Session Replay Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `session_replay_registry.v1` | Describes session replay context without session replay execution, session recording, timeline reconstruction, replay persistence, conversation persistence, snapshot capture, or agent invocation |
| Execution Replay Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles | `execution_replay_registry.v1` | Describes execution replay context without provider execution, local/cloud execution, model selection, execution trace reconstruction, replay persistence, cost scoring, quality scoring, or workflow control |
| Hybrid Studio Integration Registry | `src/creative_coding_assistant/orchestration/hybrid_studio.py` | 4 profiles / 17 source registries | `hybrid_studio_integration_registry.v1` | Exposes V4.4 Hybrid Studio Integration source coverage without activating Studio runtime, selecting providers/models, executing providers, invoking agents, controlling workflows, mutating storage, or writing replay storage |

All V4.4 Hybrid Studio registries remain export-only metadata surfaces.
Boundary tests assert that passive hybrid studio metadata does not activate
Studio runtime, enter provider/model routing, select runtimes, execute
providers, invoke agents, control workflows, request human input, mutate
storage, write replay storage, trigger retries, or modify generated output.

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
  family. After V4.4, active V4 Agentic Studio, V5 Execution Optimization &
  Production Intelligence, and V6 HoloGenesis Core OS remain future work.
- The current runtime graph remains the source of truth for execution order.
- The matrix is a planning and architecture aid, not a claim that every engine
  is already a separate runtime subsystem.
