# Published Multi-Agent Role Flows

The Multi Agent route publishes five role names: Planner, Researcher, Generator,
Critic, and Reviewer. They are sequential responsibilities inside one compiled
LangGraph, not five autonomous workers or a parallel swarm. Planning, direction,
reasoning, critique, and review are deterministic application stages. Only
`generation` calls the configured OpenAI generation provider.

The diagrams below show executable stages and handoffs. A role label describes
ownership; green is a real runtime node or stage, yellow is bounded evidence,
orange is an external provider, and gray is a non-executing boundary.

## Planner

Purpose: turn the routed request and assembled context into bounded creative
guidance before prompt rendering.

```mermaid
flowchart TB
    classDef input fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef stage fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef evidence fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px
    classDef output fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:6 4
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    responsibility["Responsibility<br/>plan, direct, synthesize"]:::boundary
    inputs["Inputs<br/>request + route<br/>prompt input + context"]:::input
    helpers["Typed planning helpers<br/>no LLM call"]:::evidence
    planning["planning<br/>creative plan"]:::stage
    director["director<br/>deterministic brief"]:::stage
    reasoning["reasoning<br/>deterministic synthesis"]:::stage
    rendering["prompt_rendering<br/>provider-neutral prompt"]:::stage
    handoff["Handoff<br/>Generator"]:::output
    single["Single Agent<br/>bypasses these stages"]:::boundary
    refinement["Later refinement<br/>does not replan"]:::boundary
    failure["failure<br/>typed terminal event"]:::failure

    responsibility --> planning
    inputs --> planning --> director --> reasoning --> rendering --> handoff
    helpers -. "derives metadata" .-> planning
    helpers -. "derives metadata" .-> director
    helpers -. "derives metadata" .-> reasoning
    single -. "direct path" .-> rendering
    refinement -. "direct to generation" .-> handoff
    planning -. "node exception" .-> failure
    director -. "node exception" .-> failure
    reasoning -. "node exception" .-> failure
```

What to notice:

- Director and Reasoning are deterministic handoffs within the Planner
  responsibility; neither is another model call.
- Single Agent goes from `prompt_input` directly to `prompt_rendering`. Multi
  Agent follows `planning → director → reasoning → prompt_rendering`.
- A later review retry does not re-enter Planner; refinement returns directly to
  Generator.

Truth boundary: the many typed planning and advisory registries do not become
extra runtime agents merely because the planning node derives their metadata.

Deeper links: [planning node](../src/creative_coding_assistant/orchestration/runtime/nodes/planning_node.py),
[Director node](../src/creative_coding_assistant/orchestration/runtime/nodes/director.py),
[Reasoning node](../src/creative_coding_assistant/orchestration/runtime/nodes/reasoning.py), and
[route transitions](../src/creative_coding_assistant/orchestration/runtime/nodes/transitions.py).

## Researcher

Purpose: retrieve bounded official-document evidence for the Multi Agent route
before context assembly.

```mermaid
flowchart TB
    classDef input fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef stage fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef evidence fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px
    classDef external fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px
    classDef output fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:6 4
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    responsibility["Responsibility<br/>ground the request"]:::boundary
    inputs["Inputs<br/>query + route + domains"]:::input
    retrieval["retrieval<br/>bounded KB search"]:::stage
    embeddings["Query embeddings<br/>OpenAI adapter"]:::external
    chroma["Local Chroma<br/>official-doc chunks"]:::evidence
    local_policy["Approved local corpus<br/>sync-controlled freshness"]:::boundary
    result["RetrievalContext<br/>ranked official evidence"]:::output
    empty["Recoverable empty context"]:::boundary
    skip["Explicit skip<br/>Single or unavailable"]:::boundary
    assembly["context_assembly<br/>then prompt_input"]:::stage
    failure["failure<br/>uncaught node error"]:::failure

    responsibility --> retrieval
    inputs --> retrieval
    embeddings -. "query vector" .-> retrieval
    retrieval <--> chroma
    local_policy -. "request-time boundary" .-> retrieval
    retrieval --> result --> assembly
    retrieval -. "gateway error" .-> empty --> assembly
    skip --> assembly
    retrieval -. "uncaught exception" .-> failure
```

What to notice:

- The default researcher uses query embeddings and schema-versioned records in
  the local official-document Chroma collection. Excluding request-time
  open-web fetches is intended to improve reproducibility, provenance, latency,
  evaluation stability, and resistance to changing or untrusted content. Index
  freshness is bounded by explicit per-source synchronization.
- A retrieval-gateway error is normalized to an empty, recoverable context and
  generation continues. Missing retrieval infrastructure and Single Agent are
  explicit skips.
- Review-driven refinement does not rerun retrieval; it returns directly to
  generation.

Truth boundary: Researcher is the retrieval responsibility, not an independent
LLM worker. Memory retrieval is a separate preceding graph node and Chroma
collection.

Deeper links: [retrieval node](../src/creative_coding_assistant/orchestration/runtime/nodes/retrieval.py),
[retrieval adapter](../src/creative_coding_assistant/orchestration/runtime/retrieval.py),
[recoverable retrieval boundary](../src/creative_coding_assistant/orchestration/runtime/service.py), and
[service composition](../src/creative_coding_assistant/app/bootstrap.py).

## Generator

Purpose: turn the rendered provider-neutral prompt into the assistant answer and
optional code artifacts.

```mermaid
flowchart TB
    classDef input fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef stage fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef evidence fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px
    classDef external fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px
    classDef output fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:6 4
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    responsibility["Responsibility<br/>produce the response"]:::boundary
    inputs["Inputs<br/>rendered prompt<br/>context + plan + images"]:::input
    generation["generation<br/>stream provider events"]:::stage
    openai["OpenAI Responses<br/>only generation LLM call"]:::external
    retry_policy["Provider SDK retries<br/>disabled"]:::boundary
    answer["Generation result<br/>answer + telemetry"]:::output
    route{"Explain route?"}:::boundary
    finalization["finalization"]:::stage
    extraction["artifact_extraction"]:::stage
    refinement["refinement<br/>review-owned guidance"]:::stage
    shell["Boundary absent<br/>generation skip<br/>shell answer downstream"]:::boundary
    failure["failure<br/>provider or node error"]:::failure

    responsibility --> generation
    inputs --> generation
    generation <--> openai
    retry_policy -. "max_retries = 0" .-> openai
    generation --> answer --> route
    route -->|"yes"| finalization
    route -->|"no"| extraction
    refinement -->|"direct loop"| generation
    generation -. "gateway/provider absent" .-> shell --> route
    generation -. "provider failure" .-> failure
    generation -. "node exception" .-> failure
```

What to notice:

- This is the only role that calls the configured OpenAI generation provider.
  Validated image data may cross this boundary with the rendered prompt.
- Explain goes directly from generation to finalization, so it skips artifact
  extraction, preview, critique, and review.
- A Reviewer-approved refinement is a new generation call, but it loops directly
  from `refinement` to `generation`; it does not rerun research or planning.

Truth boundary: provider retries are disabled. Workflow refinement owns any
additional generation call, and no dynamic provider/model router is active.

Deeper links: [generation node](../src/creative_coding_assistant/orchestration/runtime/nodes/generation.py),
[provider-neutral contracts](../src/creative_coding_assistant/llm/generation.py),
[OpenAI adapter](../src/creative_coding_assistant/llm/openai_adapter.py), and
[generation transition](../src/creative_coding_assistant/orchestration/runtime/nodes/transitions.py).

## Critic

Purpose: score inspectable generated artifacts and recommend what the Reviewer
should accept or refine.

```mermaid
flowchart TB
    classDef input fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef stage fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef evidence fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px
    classDef output fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:6 4
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    responsibility["Responsibility<br/>inspect and recommend"]:::boundary
    extraction["artifact_extraction<br/>source artifacts"]:::stage
    preparation["preview_preparation<br/>metadata only"]:::stage
    rules["Deterministic rules<br/>request + route + source"]:::evidence
    critique["artifact_critique<br/>score + compare"]:::stage
    output["Critique summary<br/>scores + recommendation<br/>refinement signal"]:::output
    review["Handoff<br/>Reviewer"]:::stage
    mode{"Resolved mode?"}:::boundary
    explain["Explain route<br/>skips artifact path"]:::boundary
    single["Single after preparation<br/>skips Critic + Reviewer"]:::boundary
    retry["Refined generation<br/>repeats extraction"]:::boundary
    final_event["final event"]:::output
    browser["Browser sandbox telemetry<br/>later, client-side"]:::boundary
    failure["failure<br/>node exception"]:::failure

    responsibility --> critique
    extraction --> preparation --> mode
    mode -->|"Multi"| critique
    mode -. "Single" .-> single --> final_event
    rules --> critique --> output --> review
    explain -. "from generation" .-> final_event
    retry --> extraction
    final_event --> browser
    critique -. "node exception" .-> failure
```

What to notice:

- Backend `preview_preparation` creates renderer/runtime metadata. It does not
  execute the artifact or observe a browser frame.
- Critique uses artifact source, route/request data, deterministic rules, and
  the prepared preview result. Browser sandbox telemetry exists only after the
  final event and feeds client Runtime, Preview, Dashboard, and Inspector views.
- A Reviewer-approved retry produces a new generation result, then repeats
  extraction, preview preparation, and critique. A browser error alone does not
  trigger that loop.

Truth boundary: browser telemetry is not an input to backend critique, and a
backend “preview prepared” result is not proof of visible runtime success.

Deeper links: [artifact nodes](../src/creative_coding_assistant/orchestration/runtime/nodes/artifacts.py),
[artifact and preview preparation](../src/creative_coding_assistant/orchestration/runtime/artifacts.py),
[browser runtime stage](../clients/nextjs/src/components/preview-runtime-stage.tsx), and
[sandbox runtime](../clients/nextjs/src/lib/preview-sandbox-runtime.ts).

## Reviewer

Purpose: apply the deterministic quality and deliverable gate, then finalize,
request bounded refinement, or fail a missing required deliverable.

```mermaid
flowchart TB
    classDef input fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef stage fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef evidence fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px
    classDef output fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray:6 4
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    responsibility["Responsibility<br/>gate the deliverable"]:::boundary
    inputs["Inputs<br/>answer + artifacts<br/>critique summary"]:::input
    checks["Deterministic checks<br/>quality + required output"]:::evidence
    review["review"]:::stage
    outcome{"Review outcome"}:::boundary
    retry_gate{"Retry allowed?<br/>count under 2<br/>artifact gate permits"}:::boundary
    runtime_limit["Executable runtime<br/>up to 2 attempts"]:::evidence
    refinement["refinement<br/>append guidance"]:::stage
    generation["generation<br/>direct re-entry"]:::stage
    finalization["finalization<br/>pass or stop"]:::output
    failure["failure<br/>typed terminal event"]:::failure
    bypass["Single and Explain<br/>skip review"]:::boundary

    responsibility --> review
    inputs --> review
    checks --> review --> outcome
    outcome -->|"pass"| finalization
    outcome -->|"needs refinement"| retry_gate
    runtime_limit --> retry_gate
    retry_gate -->|"yes"| refinement --> generation
    retry_gate -->|"no, other quality stop"| finalization
    retry_gate -->|"no, required output absent after 2"| failure
    bypass -. "direct path" .-> finalization
    review -. "node exception" .-> failure
```

What to notice:

- The executable gate permits up to two refinement attempts. Artifact-backed
  passes can stop earlier after sufficient improvement, preview-safety failure,
  or no useful opportunity.
- `refinement` appends guidance to the existing rendered prompt and returns
  directly to `generation`. Successful retry then repeats extraction, preview
  preparation, critique, and review.

Truth boundary: a required deliverable still missing after two attempts becomes
a typed terminal failure. Other non-retriable or exhausted quality findings may
finalize with the review evidence instead of causing that deliverable failure.
Single Agent and Explain do not execute this Reviewer path.

Deeper links: [review node](../src/creative_coding_assistant/orchestration/runtime/nodes/review.py),
[review transition logic](../src/creative_coding_assistant/orchestration/runtime/nodes/review_logic.py),
[refinement node](../src/creative_coding_assistant/orchestration/runtime/nodes/refinement.py),
and [runtime refinement limit](../src/creative_coding_assistant/orchestration/runtime/workflow_review.py).

## Shared execution boundary

These five views are slices of the same sequential graph. They should be read
with the [end-to-end product workflow](end_to_end_product_workflow.md), not
connected into a parallel-agent topology. Routing publishes the selected
responsibilities; LangGraph node events and final payloads are the execution
evidence.
