# Workflow Graph

This document describes the real LangGraph workflow currently executed by the backend. It is documentation for the implementation in:

- `src/creative_coding_assistant/orchestration/workflow_graph.py`
- `src/creative_coding_assistant/orchestration/workflow.py`
- `src/creative_coding_assistant/orchestration/service.py`
- `src/creative_coding_assistant/orchestration/events.py`
- `tests/test_langgraph_workflow_integration.py`

## Current Implemented Flow

The graph is compiled once in `AssistantService.__init__()` and executed through `graph.stream(..., stream_mode="custom")`. Control flow is linear until `review`, where the graph now applies a bounded quality gate. Passing outputs continue to `finalization`; failing outputs enter one `refinement` attempt and loop back to `generation`.

In the diagrams below:

- solid green nodes are implemented runtime nodes
- the blue diamond is the implemented conditional quality gate
- purple dashed nodes and edges are future-only extension points

```mermaid
flowchart TB
    classDef boundary fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px;
    classDef implemented fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef gate fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px;
    classDef terminal fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,stroke-width:1.5px;

    start([START])
    finish([END])

    subgraph phase_1["Phase 1: Intake and routing"]
        direction TB
        intake["Intake<br/>emit status: request_received"]
        routing["Routing<br/>select route<br/>store route_decision + route_payload<br/>emit status: route_selected"]
    end

    subgraph phase_2["Phase 2: Context preparation"]
        direction TB
        memory["Memory<br/>retrieve memory context<br/>complete or skip"]
        retrieval["Retrieval<br/>retrieve KB context<br/>complete or skip"]
        context_assembly["Context assembly<br/>combine memory + retrieval<br/>complete or skip"]
    end

    subgraph phase_3["Phase 3: Prompt preparation"]
        direction TB
        prompt_input["Prompt input<br/>build prompt inputs<br/>complete or skip"]
        prompt_rendering["Prompt rendering<br/>render provider prompt<br/>complete or skip"]
    end

    subgraph phase_4["Phase 4: Answer production"]
        direction TB
        generation["Generation<br/>prepare provider input<br/>stream tokens<br/>store generation_result"]
        review{{"Review quality gate<br/>deterministic checks<br/>records review_result"}}
        refinement["Refinement<br/>append guidance<br/>increment refinement_count<br/>max one attempt"]
        finalization["Finalization<br/>resolve final answer<br/>emit final<br/>finish_workflow()"]
    end

    start --> intake --> routing --> memory --> retrieval --> context_assembly --> prompt_input --> prompt_rendering --> generation --> review
    review -->|"pass or max retry"| finalization --> finish
    review -->|"needs refinement and count < 1"| refinement --> generation

    class start boundary
    class finish terminal
    class intake,routing,memory,retrieval,context_assembly,prompt_input,prompt_rendering,generation,refinement,finalization implemented
    class review gate
```

The raw Mermaid source for the implemented graph is also available in [workflow_graph.mmd](/Users/k/Desktop/CC/the_turing_college/extra_projects/creative_coding_assistant/docs/workflow_graph.mmd).

## Nodes And Transitions

`ASSISTANT_WORKFLOW_NODE_ORDER` is the source of truth for node ordering:

1. `intake`
2. `routing`
3. `memory`
4. `retrieval`
5. `context_assembly`
6. `prompt_input`
7. `prompt_rendering`
8. `generation`
9. `review`
10. `refinement`
11. `finalization`

Current transition rules:

- `START -> intake`
- Nodes point linearly from `intake` through `review`
- `review -> finalization` when the review passes or the refinement limit is reached
- `review -> refinement` when the review fails and `refinement_count < 1`
- `refinement -> generation`
- `finalization -> END`
- The only graph loop is the bounded `refinement -> generation -> review` loop
- Failures currently propagate as exceptions instead of traversing an explicit failure edge

Node responsibilities:

- `intake`: marks `WorkflowStep.INTAKE` active, emits `status/request_received`, then completes the step
- `routing`: computes `RouteDecision`, emits `status/route_selected`, stores `route_decision` in workflow state and `route_payload` in graph state
- `memory`: calls the memory step generator and either stores `memory_context` or skips the step
- `retrieval`: calls the retrieval step generator and either stores `retrieval_context` or skips the step
- `context_assembly`: combines memory and retrieval context when a context assembler is configured
- `prompt_input`: builds prompt inputs when a prompt input builder is configured
- `prompt_rendering`: renders the final provider prompt when prompt inputs exist
- `generation`: prepares provider input, forwards generation stream events, and stores the transient `generation_result`
- `review`: runs deterministic quality checks, stores `review_result`, and selects the next graph edge
- `refinement`: appends refinement guidance to the rendered prompt, increments `refinement_count`, and sends control back to `generation`
- `finalization`: resolves the final answer from `generation_result.answer` or the shell fallback, emits the `final` event, and marks the workflow completed

## Workflow State Lifecycle

There are two layers of runtime state.

`AssistantWorkflowState` is the durable typed workflow state:

- Created by `begin_assistant_workflow(request)`
- Starts as `status=running`, `current_step=None`
- Moves one step at a time through `start_workflow_step()`
- Resolves each step through `complete_workflow_step()` or `skip_workflow_step()`
- Stores durable outputs such as `route_decision`, `memory_context`, `retrieval_context`, `assembled_context`, `prompt_input`, `rendered_prompt`, and `final_answer`
- Stores review metadata through `review_result` and `refinement_count`
- Reaches terminal completion only through `finish_workflow()` while `FINALIZATION` is active
- Supports `fail_workflow()`, but that path is not yet wired into the LangGraph runtime

`AssistantWorkflowGraphState` is the LangGraph transport state:

- Always carries `workflow_state`
- Also carries `route_payload` for final event rendering
- Also carries `generation_result` as an ephemeral object needed by `finalization`
- Keeps the graph runtime small without forcing all transient objects into the Pydantic workflow model

Important current behavior:

- Optional steps skip when their gateway or input is missing
- `review` always runs and records a deterministic review result
- `refinement` runs at most once and only after a failed review
- `generation_result` is not persisted into `AssistantWorkflowState`; only `final_answer` is
- `WorkflowEventMetadata` exists on the state model but is not yet attached to emitted stream events

## Stream Event Flow

The graph preserves the existing event protocol by reusing one `StreamEventBuilder` instance for the entire request.

```mermaid
sequenceDiagram
    participant Client
    participant Service as AssistantService.stream
    participant Graph as compiled LangGraph
    participant Node as workflow node
    participant Step as stream_* helper
    participant Builder as StreamEventBuilder

    Client->>Service: stream(request)
    Service->>Graph: graph.stream(initial_state, context, stream_mode="custom")
    loop per node
        Graph->>Node: execute(state, runtime)
        Node->>Step: call step generator
        Step->>Builder: build StreamEvent
        Step-->>Node: yield StreamEvent
        Node->>Graph: get_stream_writer()(event)
        Graph-->>Service: custom stream item
    end
    Node->>Graph: emit final in finalization
    Graph-->>Service: final StreamEvent
    Service-->>Client: yielded StreamEvent iterator
```

What actually flows through the stream:

- `intake` emits `status`
- `routing` emits `status`
- `memory` emits `memory`
- `retrieval` emits `retrieval`
- `context_assembly` emits `context`
- `prompt_input` emits `prompt_input`
- `prompt_rendering` emits `prompt_rendered`
- `generation` emits `generation_input`, `token_delta`, and possibly `error`
- `review` and `refinement` currently update graph state without emitting stream events
- `finalization` emits `final`

Important stream guarantees:

- Sequence numbers remain monotonic because the same `StreamEventBuilder` instance is shared across all nodes
- Only `StreamEvent` instances are surfaced from the graph stream; helper return values become state updates instead
- The final event is still emitted exactly once by `finalization`
- The current tests verify backward-compatible event ordering for both shell and generation-backed paths

## Current Implemented Flow Vs Future Extension Points

Current implemented flow:

- Linear path until `review`
- Conditional review edge
- Bounded one-attempt refinement loop
- No tool nodes
- No preview pipeline
- No HITL checkpoints
- No explicit failure node

Future extension points can be added incrementally without replacing the current graph shape.

```mermaid
flowchart TB
    classDef implemented fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef gate fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px;
    classDef future fill:#F3E8FF,stroke:#7E22CE,color:#4C1D95,stroke-width:1.5px,stroke-dasharray: 6 4;

    subgraph current_path["Current implemented path"]
        direction TB
        routing["Routing"]
        memory["Memory"]
        retrieval["Retrieval"]
        context_assembly["Context assembly"]
        prompt_input["Prompt input"]
        prompt_rendering["Prompt rendering"]
        generation["Generation"]
        review{{"Review quality gate"}}
        refinement["Refinement<br/>max one attempt"]
        finalization["Finalization"]
        routing --> memory --> retrieval --> context_assembly --> prompt_input --> prompt_rendering --> generation --> review
        review -->|"pass or max retry"| finalization
        review -->|"needs refinement"| refinement --> generation
    end

    subgraph future_tools["Future tool insertion points"]
        direction TB
        tool_gate{{Tool gate<br/>after routing}}
        tool_loop{{Tool planning / execution loop<br/>before generation}}
    end

    subgraph future_preview["Future preview pipeline"]
        direction TB
        preview["Preview pipeline<br/>artifact build / render / capture"]
    end

    subgraph future_refinement["Future refinement and approval"]
        direction TB
        retry{{Review / refinement loop}}
        hitl{{HITL checkpoint}}
    end

    routing -. capability-based branch .-> tool_gate
    tool_gate -. rejoin .-> memory
    prompt_rendering -. optional tool loop .-> tool_loop
    tool_loop -. rejoin .-> generation
    generation -. preview mode branch .-> preview
    preview -. rejoin .-> review
    review -. refinement loop .-> retry
    retry -. back to prompt preparation .-> prompt_input
    review -. human approval .-> hitl
    hitl -. rejoin .-> finalization

    class routing,memory,retrieval,context_assembly,prompt_input,prompt_rendering,generation,refinement,finalization implemented
    class review gate
    class tool_gate,tool_loop,preview,retry,hitl future
```

Conservative insertion points:

- Tools: the least disruptive gate is immediately after `routing`, because route capabilities already exist there; a richer tool loop can also sit between `prompt_rendering` and `generation`
- Review loops: the current `review` gate is the natural anchor for richer future retry loops back to `prompt_input` or `generation`
- Preview pipeline: a preview branch can sit after `generation` and before `review` so preview artifacts can be inspected without changing the request/response contract
- HITL checkpoints: the safest first checkpoint is between `review` and `finalization`, where a human can approve, edit, or reject a nearly complete result

## Known Limits In The Current Runtime

- Failure handling is still exception-based at the graph boundary; there is no explicit `FAILED` graph path yet
- `WorkflowEventMetadata` is modeled but not emitted with stream events
- Route selection does not currently alter graph control flow
- `review` is deterministic and intentionally lightweight; it is not an LLM evaluator
- Stream event types such as `tool_start`, `tool_result`, `preview_artifact`, and `eval_update` exist in contracts but are not emitted by the current graph

## Validation Pointers

The current behavior described here is covered directly by:

- `tests/test_workflow_foundation.py`
- `tests/test_langgraph_workflow_integration.py`

Those tests currently verify:

- explicit step ordering
- state completion and skipped-step behavior
- compiled graph execution
- failure propagation
- stream ordering and event-shape compatibility
