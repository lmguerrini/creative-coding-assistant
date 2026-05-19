"""LangGraph runtime for the assistant workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowStep,
    begin_assistant_workflow,
    complete_workflow_step,
    finish_workflow,
    skip_workflow_step,
    start_workflow_step,
)


class GenerationResultLike(Protocol):
    answer: str


class AssistantWorkflowGraphState(TypedDict, total=False):
    workflow_state: AssistantWorkflowState
    route_payload: dict[str, object]
    generation_result: GenerationResultLike


class AssistantWorkflowGraphContext(TypedDict):
    runtime: AssistantWorkflowRuntime


@dataclass(frozen=True)
class AssistantWorkflowRuntime:
    """Runtime services needed by graph nodes for one assistant turn."""

    event_builder: StreamEventBuilder
    route_fn: Callable[[AssistantRequest], RouteDecision]
    stream_request_received: Callable[..., Iterator[object]]
    stream_route_selected: Callable[..., Iterator[object]]
    stream_memory_context: Callable[..., Iterator[object]]
    stream_retrieval_context: Callable[..., Iterator[object]]
    stream_assembled_context: Callable[..., Iterator[object]]
    stream_prompt_inputs: Callable[..., Iterator[object]]
    stream_rendered_prompt: Callable[..., Iterator[object]]
    stream_generation: Callable[..., Iterator[object]]
    build_shell_answer: Callable[[RouteDecision], str]


ASSISTANT_WORKFLOW_NODE_ORDER: tuple[str, ...] = (
    "intake",
    "routing",
    "memory",
    "retrieval",
    "context_assembly",
    "prompt_input",
    "prompt_rendering",
    "generation",
    "review",
    "finalization",
)


def build_initial_workflow_graph_state(
    request: AssistantRequest,
) -> AssistantWorkflowGraphState:
    return {"workflow_state": begin_assistant_workflow(request)}


def build_assistant_workflow_graph() -> Any:
    graph = StateGraph(
        AssistantWorkflowGraphState,
        context_schema=AssistantWorkflowGraphContext,
    )
    graph.add_node("intake", _intake_node)
    graph.add_node("routing", _routing_node)
    graph.add_node("memory", _memory_node)
    graph.add_node("retrieval", _retrieval_node)
    graph.add_node("context_assembly", _context_assembly_node)
    graph.add_node("prompt_input", _prompt_input_node)
    graph.add_node("prompt_rendering", _prompt_rendering_node)
    graph.add_node("generation", _generation_node)
    graph.add_node("review", _review_node)
    graph.add_node("finalization", _finalization_node)

    graph.add_edge(START, "intake")
    for index in range(len(ASSISTANT_WORKFLOW_NODE_ORDER) - 1):
        current_node = ASSISTANT_WORKFLOW_NODE_ORDER[index]
        next_node = ASSISTANT_WORKFLOW_NODE_ORDER[index + 1]
        graph.add_edge(current_node, next_node)
    graph.add_edge("finalization", END)
    return graph.compile()


def stream_assistant_workflow_events(
    *,
    graph: Any,
    request: AssistantRequest,
    runtime: AssistantWorkflowRuntime,
) -> Iterator[StreamEvent]:
    initial_state = build_initial_workflow_graph_state(request)
    for item in graph.stream(
        initial_state,
        context={"runtime": runtime},
        stream_mode="custom",
    ):
        if isinstance(item, StreamEvent):
            yield item


def _intake_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.INTAKE,
    )
    runtime_context = _runtime(runtime)
    _emit_streaming_step(
        runtime_context.stream_request_received(runtime_context.event_builder)
    )
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.INTAKE,
        )
    }


def _routing_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.ROUTING,
    )
    runtime_context = _runtime(runtime)
    decision = runtime_context.route_fn(workflow_state.request)
    route_payload = decision.model_dump(mode="json")
    _emit_streaming_step(
        runtime_context.stream_route_selected(
            builder=runtime_context.event_builder,
            decision=decision,
            route_payload=route_payload,
        )
    )
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.ROUTING,
            route_decision=decision,
        ),
        "route_payload": route_payload,
    }


def _memory_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.MEMORY,
    )
    runtime_context = _runtime(runtime)
    memory_context = _emit_streaming_step(
        runtime_context.stream_memory_context(
            builder=runtime_context.event_builder,
            request=workflow_state.request,
            decision=_route_decision(workflow_state),
        )
    )
    if memory_context is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.MEMORY,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.MEMORY,
            memory_context=memory_context,
        )
    }


def _retrieval_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.RETRIEVAL,
    )
    runtime_context = _runtime(runtime)
    retrieval_context = _emit_streaming_step(
        runtime_context.stream_retrieval_context(
            builder=runtime_context.event_builder,
            request=workflow_state.request,
            decision=_route_decision(workflow_state),
        )
    )
    if retrieval_context is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.RETRIEVAL,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.RETRIEVAL,
            retrieval_context=retrieval_context,
        )
    }


def _context_assembly_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.CONTEXT_ASSEMBLY,
    )
    runtime_context = _runtime(runtime)
    assembled_context = _emit_streaming_step(
        runtime_context.stream_assembled_context(
            builder=runtime_context.event_builder,
            decision=_route_decision(workflow_state),
            memory_context=workflow_state.memory_context,
            retrieval_context=workflow_state.retrieval_context,
        )
    )
    if assembled_context is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.CONTEXT_ASSEMBLY,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.CONTEXT_ASSEMBLY,
            assembled_context=assembled_context,
        )
    }


def _prompt_input_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.PROMPT_INPUT,
    )
    runtime_context = _runtime(runtime)
    prompt_input = _emit_streaming_step(
        runtime_context.stream_prompt_inputs(
            builder=runtime_context.event_builder,
            request=workflow_state.request,
            decision=_route_decision(workflow_state),
            assembled_context=workflow_state.assembled_context,
        )
    )
    if prompt_input is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.PROMPT_INPUT,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.PROMPT_INPUT,
            prompt_input=prompt_input,
        )
    }


def _prompt_rendering_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.PROMPT_RENDERING,
    )
    runtime_context = _runtime(runtime)
    rendered_prompt = _emit_streaming_step(
        runtime_context.stream_rendered_prompt(
            builder=runtime_context.event_builder,
            decision=_route_decision(workflow_state),
            prompt_inputs=workflow_state.prompt_input,
        )
    )
    if rendered_prompt is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.PROMPT_RENDERING,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.PROMPT_RENDERING,
            rendered_prompt=rendered_prompt,
        )
    }


def _generation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.GENERATION,
    )
    runtime_context = _runtime(runtime)
    generation_result = _emit_streaming_step(
        runtime_context.stream_generation(
            builder=runtime_context.event_builder,
            decision=_route_decision(workflow_state),
            rendered_prompt=workflow_state.rendered_prompt,
        )
    )
    if generation_result is None:
        return {
            "workflow_state": skip_workflow_step(
                workflow_state,
                WorkflowStep.GENERATION,
            )
        }
    return {
        "workflow_state": complete_workflow_step(
            workflow_state,
            WorkflowStep.GENERATION,
        ),
        "generation_result": generation_result,
    }


def _review_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    del runtime
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.REVIEW,
    )
    return {
        "workflow_state": skip_workflow_step(
            workflow_state,
            WorkflowStep.REVIEW,
        )
    }


def _finalization_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.FINALIZATION,
    )
    runtime_context = _runtime(runtime)
    generation_result = state.get("generation_result")
    if generation_result is not None:
        answer = generation_result.answer
    else:
        answer = runtime_context.build_shell_answer(_route_decision(workflow_state))

    final_state = finish_workflow(workflow_state, final_answer=answer)
    _emit(
        runtime_context.event_builder.final(
            answer=answer,
            route=state["route_payload"],
        )
    )
    return {"workflow_state": final_state}


def _emit_streaming_step(step: Iterator[object]) -> object:
    while True:
        try:
            item = next(step)
        except StopIteration as exc:
            return exc.value
        if isinstance(item, StreamEvent):
            _emit(item)


def _emit(event: StreamEvent) -> None:
    writer = get_stream_writer()
    writer(event)


def _runtime(
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowRuntime:
    return runtime.context["runtime"]


def _workflow_state(
    state: AssistantWorkflowGraphState,
) -> AssistantWorkflowState:
    return state["workflow_state"]


def _route_decision(state: AssistantWorkflowState) -> RouteDecision:
    if state.route_decision is None:
        raise ValueError("Workflow route decision is not available.")
    return state.route_decision
