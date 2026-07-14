# Single and Multi Runtime Routes

This document is the exact route-level view of the compiled LangGraph executed
by the Python backend. It separates executable nodes from the repository's
larger inventory of typed planning, advisory, and audit contracts. A class,
registry, or role name is not a runtime agent unless the registered graph calls
it.

The implementation sources are
[`runtime/graph_builder.py`](../src/creative_coding_assistant/orchestration/runtime/graph_builder.py),
[`runtime/nodes/registry.py`](../src/creative_coding_assistant/orchestration/runtime/nodes/registry.py),
[`runtime/nodes/transitions.py`](../src/creative_coding_assistant/orchestration/runtime/nodes/transitions.py),
and the grouped handlers under
[`runtime/nodes/`](../src/creative_coding_assistant/orchestration/runtime/nodes/).
The standalone [Mermaid source](workflow_graph.mmd) mirrors the Multi overview
below for slide rendering. Failure behavior is decomposed into
[Error and Recovery Paths](error_and_recovery_paths.md).

## Route summary

Every request enters `intake`, then `routing`, then `memory`. The execution plan
published by routing controls later branches:

| Resolved route | Retrieval | Prompt preparation | After preview metadata | Executable refinement budget |
|---|---|---|---|---:|
| Single Agent | Handler runs but records a skip | `prompt_input` goes directly to `prompt_rendering` | Goes directly to `finalization` | 0 |
| Multi Agent | Requests official-source retrieval when configured | Runs `planning`, `director`, and `reasoning` before `prompt_rendering` | Runs `artifact_critique` and `review` | Up to 2; artifact stop rules can end earlier |
| Auto | Resolves to Single or Multi, then follows that route | Follows the resolved route | Follows the resolved route | Follows the resolved route |

Auto is a selector, not a third execution graph. It resolves to Single exactly
when routing selected Explain or Debug, the request has no attachment, and the
route decision has no resolved domain. Otherwise it resolves to Multi. The
client waits for and displays the backend's published resolution.

Multi Agent is a sequential multi-node runtime graph with planner, researcher,
generator, critic, and reviewer responsibilities. It is not a true multi-agent
swarm: planning, Director guidance, reasoning, critique, and review are typed
application logic, while only `generation` invokes the configured generation
provider. A refinement appends guidance to the existing rendered prompt and
re-enters `generation`; it does not rerun retrieval, planning, Director,
reasoning, or prompt rendering.

The executable review limit is two attempts. The published
`execution.max_refinement_loops` field still reports one for Multi, and the
Inspector displays that published value. That field is not consulted by the
review transition, so this is a current contract drift rather than a runtime
limit. This documentation task records the mismatch without changing product
behavior or public APIs.

## Single-Agent workflow

**Purpose.** Show only the resolved Single path.

```mermaid
flowchart TB
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef skipped fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:5 3
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef external fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px
    classDef terminal fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    subgraph shared_single["1 · Shared request stages"]
        direction LR
        intake["intake"]:::runtime --> routing["routing<br/>publish Single"]:::runtime
        routing --> memory["memory"]:::runtime --> retrieval["retrieval<br/>explicit skip"]:::skipped
        retrieval --> context["context_assembly"]:::runtime --> input["prompt_input"]:::runtime
    end

    subgraph direct_single["2 · Direct generation"]
        direction LR
        clarify{"clarification?"}:::decision -->|"no"| render["prompt_rendering"]:::runtime
        render --> generation["generation"]:::runtime --> explain{"Explain route?"}:::decision
        generation -. "provider call" .-> openai["OpenAI Responses"]:::external
    end

    subgraph finish_single["3 · Artifact path, completion, and failure"]
        direction LR
        extract["artifact_extraction"]:::runtime --> preview["preview_preparation"]:::runtime
        preview --> final["finalization"]:::terminal --> event["final event<br/>completed"]:::terminal
        pending["pending failure<br/>from any node"]:::decision -.-> failure["failure"]:::failure
        failure --> failed_event["final event<br/>failed"]:::failure
    end

    input --> clarify
    clarify -->|"yes"| final
    explain -->|"yes"| final
    explain -->|"no"| extract
```

**What to notice.** Memory still has a real handler; retrieval is entered and
recorded as skipped. Single does not execute planning, Director, reasoning,
artifact critique, review, or refinement. Explain responses skip artifact and
preview nodes after generation.

**Truth boundary.** Preview preparation creates backend metadata only. Browser
execution happens after the final event. Any pending node failure branches to
the shared `failure` node shown in the dedicated error document.

## Multi-Agent workflow overview

**Purpose.** Show the sequential Multi responsibilities and the bounded review
loop without rendering every failure edge.

```mermaid
flowchart TB
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef external fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px
    classDef terminal fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    subgraph top_half[" "]
    direction LR
    subgraph shared["1 · Shared request stages"]
        direction TB
        subgraph shared_a[" "]
            direction LR
            intake["intake"]:::runtime --> routing["routing<br/>publish Multi"]:::runtime --> memory["memory"]:::runtime
        end
        subgraph shared_b[" "]
            direction LR
            retrieval["retrieval<br/>Researcher"]:::runtime --> context["context_assembly"]:::runtime --> prompt_input["prompt_input"]:::runtime
        end
        memory --> retrieval
    end

    subgraph plan["2 · Plan and generate"]
        direction TB
        subgraph plan_a[" "]
            direction LR
            clarify{"clarification?"}:::decision -->|"no"| planning["planning<br/>Planner"]:::runtime --> director["director"]:::runtime
        end
        subgraph plan_b[" "]
            direction LR
            reasoning["reasoning"]:::runtime --> prompt_rendering["prompt_rendering"]:::runtime --> generation["generation<br/>Generator"]:::runtime
        end
        director --> reasoning
        generation -. "only generation-provider call" .-> openai["OpenAI Responses"]:::external
    end
    end

    subgraph bottom_half[" "]
    direction LR
    subgraph artifacts["3 · Artifacts and review"]
        direction TB
        subgraph artifacts_a[" "]
            direction LR
            explain{"Explain route?"}:::decision -->|"no"| artifact_extraction["artifact_extraction"]:::runtime --> preview_preparation["preview_preparation"]:::runtime
        end
        subgraph artifacts_b[" "]
            direction LR
            artifact_critique["artifact_critique<br/>Critic"]:::runtime --> review["review<br/>Reviewer"]:::runtime --> gate{"review outcome"}:::decision
        end
        preview_preparation --> artifact_critique
    end

    subgraph completion["4 · Complete, refine, or fail"]
        direction LR
        refinement["refinement"]:::runtime --> return_generation(["↺ generation<br/>same node above"]):::decision
        finalization["finalization"]:::terminal --> event["final event<br/>completed"]:::terminal
        pending["pending failure<br/>from any node"]:::decision -.-> failure["failure"]:::failure
        failure --> failed_event["final event<br/>failed"]:::failure
    end
    end

    prompt_input --> clarify
    clarify -->|"yes"| finalization
    generation --> explain
    explain -->|"yes"| finalization
    gate -->|"pass / stop"| finalization
    gate -->|"refine; budget remains"| refinement

    style shared_a fill:none,stroke:none
    style shared_b fill:none,stroke:none
    style plan_a fill:none,stroke:none
    style plan_b fill:none,stroke:none
    style artifacts_a fill:none,stroke:none
    style artifacts_b fill:none,stroke:none
    style top_half fill:none,stroke:none
    style bottom_half fill:none,stroke:none

    %% Executable review logic currently permits up to two refinement attempts.
    %% Browser runtime telemetry occurs after finalization and is not critique input.
    %% Executable Multi spine: prompt_input --> planning --> director --> reasoning --> prompt_rendering.
    %% V4.3 hybrid workflow metadata boundary: no runtime escalation.
    %% V4.4 hybrid studio metadata boundary: no Studio runtime.
    %% V4.5 multimodal studio metadata boundary: no rendering execution.
    %% V4.6 agentic studio hardening metadata boundary: terminal failure audit only.
    %% V5.2 model routing metadata boundary: no provider/model switching.
    %% V5.4 production observability metadata boundary: no live telemetry emission.
    %% V5.5 adaptive execution policy boundary: no provider execution or routing mutation.
```

**What to notice.** The roles are sequential responsibilities, not parallel LLM
workers. Retrieval uses the configured query-embedding and local Chroma path;
only generation calls the text provider. Refinement loops directly back to
generation. See [Multi-Agent Role Zooms](multi_agent_roles.md) for inputs,
evidence, outputs, and failure boundaries.

**Truth boundary.** An Explain route still runs Multi planning but goes from
generation directly to finalization. Backend artifact critique reads prepared
preview metadata, not live browser telemetry. The diagram omits repeated
failure arrows; every pending normalized node failure goes to `failure`.

## Registered node order

`ASSISTANT_WORKFLOW_NODE_ORDER` is the source of truth for node ordering:

1. `intake`
2. `routing`
3. `memory`
4. `retrieval`
5. `context_assembly`
6. `prompt_input`
7. `planning`
8. `director`
9. `reasoning`
10. `prompt_rendering`
11. `generation`
12. `artifact_extraction`
13. `preview_preparation`
14. `artifact_critique`
15. `review`
16. `refinement`
17. `finalization`
18. `failure`

Current transition rules:

- Registry order is not a claim that every node runs for every request. The
  full ordered spine is
  `start --> intake --> routing --> memory --> retrieval --> context_assembly --> prompt_input --> planning --> director --> reasoning --> prompt_rendering --> generation`.
- After `prompt_input`, a clarification response goes to `finalization`, Single
  goes to `prompt_rendering`, and Multi goes to `planning`.
- `planning --> director --> reasoning --> prompt_rendering --> generation` is
  the Multi prompt-preparation path.
- An Explain route goes from `generation` to `finalization`. Other successful
  generation goes to `artifact_extraction --> preview_preparation`.
- After preview preparation, Single goes to `finalization`; Multi goes to
  `artifact_critique --> review`.
- Review passes go to `finalization`. A permitted retry goes to
  `refinement --> generation`. The executable default allows two attempts and
  can stop earlier on artifact pass rules. Exhaustion with a missing required
  deliverable goes to `failure`; other exhausted quality issues finalize with
  the review evidence preserved.
- A pending normalized failure at any conditional boundary goes to `failure`.
  `failure` emits the terminal failure response and ends the graph.

## Runtime and provider boundaries

The WSGI layer validates the request before the graph and streams
newline-delimited JSON events after it starts. Retrieval can send query text to
the OpenAI embedding adapter and reads selected official chunks from local
Chroma. Generation constructs an OpenAI Responses payload from the rendered
prompt and validated image data; live receipt and image use remain run-specific
evidence. No other graph node calls the text-generation provider.

Extracted code stays in application state. Preview preparation publishes
metadata, then the browser applies its own source preflight and controlled
runtime. SQLite session persistence, post-stream Chroma memory recording,
live-evaluation JSONL recording, and explicit RAGAS evaluation all sit outside
the graph. Request-scoped image bytes are deliberately removed from saved
sessions. See the [System Architecture Overview](system_architecture_overview.md)
and [Artifact and Preview Runtime](artifact_preview_runtime.md) for those
boundaries.

## Runtime Graph Vs Internal Capability Graph

The runtime graph stays small. Dense internal dependencies are documented in:

- [creative_intelligence_graph.md](creative_intelligence_graph.md) and
  [creative_intelligence_graph.mmd](creative_intelligence_graph.mmd)
- [generative_design_graph.md](generative_design_graph.md) and
  [generative_design_graph.mmd](generative_design_graph.mmd)
- [artifact_intelligence_graph.md](artifact_intelligence_graph.md) and
  [artifact_intelligence_graph.mmd](artifact_intelligence_graph.mmd)
- [workstation_surface_graph.md](workstation_surface_graph.md) and
  [workstation_surface_graph.mmd](workstation_surface_graph.mmd)

Those helpers and views produce metadata, design guidance, artifact
intelligence, evaluation summaries, and contract summaries, plus workstation surface contracts.
They are not code generation execution and are not separate
LangGraph nodes. Their data may be derived inside an existing handler and, only
where the prompt renderer explicitly includes it, reach generation. Passive
registry metadata that is not selected by the prompt contract does not enter
provider prompts.

## Repository-only non-runtime contract inventory

The following sections exist to prevent an important architecture mistake: the
repository exposes many importable metadata contracts, but they do not enlarge
the real graph shown above. They do not imply autonomous routing, live
telemetry, local-model execution, human-intervention requests, or a true
multi-agent runtime.

<details>
<summary>Show typed contract families and their execution boundaries</summary>

## V4.1 Multi-Agent Core Contract Boundary

The Agent Contract Registry, Agent Memory Contract Registry, Agent Metadata Registry,
agent identity/role/boundary registries, and capability metadata
describe possible responsibilities. They do not instantiate or invoke agents,
change prompts, add graph nodes, or change output.

## V4.2 Agent Orchestration Metadata Boundary

The Agent Routing Registry, Blackboard Memory Registry, shared-context,
dependency, scheduling, coordination, debate, consensus, capability-alignment,
escalation-signal, lifecycle, state-synchronization, Workflow Agent Handoff
Registry, and Orchestration Contract Integration Registry are passive orchestration metadata.
They do not execute orchestration, mutate blackboard
state, add retries, or change LangGraph node ordering. Their registry records do
not enter provider prompts; in other words, they do not enter provider prompts.

Exact source label: `Workflow Agent Handoff Registry`.

## V4.3 Hybrid Agentic Workflow Metadata Boundary

This passive hybrid workflow metadata includes the V3 Backbone Mode Registry,
Conditional Multi-Agent Escalation Registry, Specialist Agent Loop Registry,
Escalation Gate Registry, Creative Escalation Policy Registry, Reflection
Escalation Registry, Hybrid Agent Debate Loop Registry, Hybrid Agent Voting
Registry, Agent Confidence Fusion Registry, Decision Provenance Registry,
Escalation Trace Registry, Creative Exploration Budget Registry, Result
Normalization Registry, Return-to-Workflow Handoff Registry, HITL Escalation
Gate Registry, Confidence Threshold Routing Registry, Cost Threshold Routing
Registry, Latency Threshold Routing Registry, Ambiguity Escalation Registry,
Risk Escalation Registry, Quality Escalation Registry, and Adaptive Multi-Agent
Escalation Registry. Hybrid Workflow Integration source coverage is descriptive
only. These contracts do not execute escalation, route providers, request human
input, or bypass failure normalization.

## V4.4 Hybrid Studio Metadata Boundary

This passive hybrid studio metadata includes the Local Model Registry, Cloud
Model Registry, Hybrid Execution Registry, Auto Mode Registry, Studio Mode
Registry, HITL Decision Registry, Provider Selection Registry, Execution
Simulator Registry, Model Profile Registry, Cost Profile Registry, Quality
Profile Registry, Local/Cloud Comparison Registry, Agent Workspace Registry,
Agent Conversation View Registry, Workspace Snapshot Registry, Session Replay
Registry, Execution Replay Registry, and Hybrid Studio Integration Registry.
Hybrid Studio Integration source coverage is metadata only. It does not
activate providers, select a local model, write replay storage, or request human
input; specifically, it does not activate Studio runtime.

Exact source labels: `Cloud Model Registry`; `Studio Mode Registry`;
`Execution Simulator Registry`; `Quality Profile Registry`; `Session Replay Registry`.

## V4.5 Multimodal Studio Metadata Boundary

This passive multimodal studio metadata includes the Live Preview Registry,
Multi Preview Registry, Interactive Canvas Registry, Visual Workspace Registry,
Runtime Collaboration Registry, Artifact Collaboration Registry, Artifact
Provenance Registry, Artifact Lineage Registry, Cross-Agent Workspace Registry,
Shared Artifact Board Registry, Workspace History Registry, Branching Timeline
Registry, Creative Evolution Timeline Registry, Real-Time Workflow Visualization
Registry, and Multimodal Studio Integration Registry. Multimodal Studio Integration source coverage
is descriptive only. It does not execute rendering,
open network connections, persist collaboration state, or mutate artifacts.

Exact source labels: `Artifact Provenance Registry`; `Branching Timeline Registry`;
`Real-Time Workflow Visualization Registry`.

## V4.6 Agentic Studio Hardening Metadata Boundary

This passive hardening and audit metadata includes the Agent Contract Audit
Registry, Escalation Policy Audit Registry, Hybrid Workflow Audit Registry,
Agent Registry Audit Registry, Blackboard Audit Registry, Shared Context Audit
Registry, Agent Collaboration Audit Registry, Creative Diversity Audit Registry,
Agent Explainability Audit Registry, Agent Reliability Audit Registry, Agent
Determinism Audit Registry, Agent Telemetry Foundation Registry, Agent Cost
Tracking Foundation Registry, Agent Performance Tracking Foundation Registry,
Architecture Consistency Pass Registry, Final V4 Hardening Registry, and
LangGraph Error Path Audit. These contracts do not execute hardening checks,
emit telemetry, activate passive registries, or bypass failure normalization.

Exact source labels: `Agent Contract Audit Registry`; `Shared Context Audit Registry`;
`Agent Determinism Audit Registry`; `Agent Cost Tracking Foundation Registry`.

## V5.2 Intelligent Model Routing Metadata Boundary

The advisory model-routing metadata covers Model Router, Local vs Cloud Routing,
Hybrid Routing, Quality/Cost Optimizer, Cost Estimator, Budget Policies, HITL
Budget Gate, Runtime Recommendation Engine, Execution Policy Engine, Model
Recommendation Engine, Model Capability Matrix, Provider Capability Matrix,
Quality Prediction Engine, Cost Prediction Engine, Creative Quality Predictor,
Creative Diversity Predictor, Creative Consistency Predictor, and Routing
Explainability. These surfaces do not apply routing, enforce budgets, execute
providers, or switch the configured adapter: there is no provider/model
switching.

The operational boundary is **no provider/model switching**. Exact source labels:
`HITL Budget Gate`; `Model Recommendation Engine`; `Routing Explainability`.

## V5.4 Production Observability Metadata Boundary

The read-only production observability metadata covers Token Dashboard, Cost
Dashboard, Quality Dashboard, Performance Dashboard, Production Telemetry,
Workflow Diagnostics, Agent Diagnostics, Routing Diagnostics, Escalation
Diagnostics, Failure Analysis, Error Intelligence, Workflow Health Monitoring,
System Health Monitoring, Creative Analytics, Confidence Analytics, Creative
Diversity Analytics, Runtime Timeline, Workflow Explainability Dashboard,
Production Observability Architecture Consistency, and Production Observability
Failure Path Audit. These surfaces do not collect or emit live telemetry,
control workflows, remediate failures, or persist monitoring data.

Exact source labels: `Cost Dashboard`; `Escalation Diagnostics`;
`Creative Diversity Analytics`; `Production Observability Failure Path Audit`.

## V5.5 Adaptive Execution Intelligence Metadata Boundary

The controlled adaptive execution policy/simulation surfaces cover Adaptive
Hybrid Workflow Optimizer, Adaptive Escalation Optimizer, Agent Activation
Optimizer, Adaptive Cost/Quality Optimizer, Adaptive Latency Optimizer, Adaptive
Execution Strategy Selection, Adaptive Execution Policy Engine, Dynamic Agent
Allocation, Dynamic Resource Allocation, Workflow Self-Tuning Policies,
Execution Confidence Engine, Workflow Risk Engine, Creative Exploration
Optimizer, Emergence Optimizer, Agent Diversity Optimizer, Reflection Budget
Optimizer, Adaptive Policy Explainability, Adaptive Execution Architecture
Consistency, and Adaptive Execution Failure Path Audit. These contracts do not
execute providers, allocate resources, apply routing, mutate the graph, or
trigger retries beyond the one compiled review loop.

Exact source labels: `Adaptive Hybrid Workflow Optimizer`; `Agent Activation Optimizer`;
`Adaptive Execution Strategy Selection`; `Dynamic Agent Allocation`;
`Creative Exploration Optimizer`; `Reflection Budget Optimizer`;
`Adaptive Execution Architecture Consistency`.

</details>

## Reviewer verification

Run the focused alignment test and inspect the Mermaid source:

```bash
.venv/bin/python -m pytest -q tests/test_workflow_documentation_alignment.py
.venv/bin/python scripts/v7_quality_gates.py docs-mermaid
```

Then submit one Single and one Multi request and compare the streamed node lists.
For Auto, verify the published resolved mode rather than predicting it in the
client. See the [Architecture Walkthrough](../docs/ARCHITECTURE_WALKTHROUGH.md)
for the full UI-to-provider request path.
