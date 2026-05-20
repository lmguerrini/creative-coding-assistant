"""LangGraph runtime for the assistant workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from loguru import logger

from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
    RenderedPromptRole,
    RenderedPromptSection,
    RenderedPromptSectionName,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
    WorkflowStep,
    begin_assistant_workflow,
    complete_workflow_step,
    fail_workflow,
    finish_workflow,
    restart_workflow_step,
    skip_workflow_step,
    start_workflow_step,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
    WorkflowReviewOutcome,
    WorkflowReviewResult,
    review_assistant_answer,
)


class GenerationResultLike(Protocol):
    answer: str


class AssistantWorkflowGraphState(TypedDict, total=False):
    workflow_state: AssistantWorkflowState
    route_payload: dict[str, object]
    generation_result: GenerationResultLike | None
    pending_failure: WorkflowFailureInfo | None
    failure_event_emitted: bool


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
    "refinement",
    "finalization",
    "failure",
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
    graph.add_node("refinement", _refinement_node)
    graph.add_node("finalization", _finalization_node)
    graph.add_node("failure", _failure_node)

    graph.add_edge(START, "intake")
    review_index = ASSISTANT_WORKFLOW_NODE_ORDER.index("review")
    for index in range(review_index - 1):
        current_node = ASSISTANT_WORKFLOW_NODE_ORDER[index]
        next_node = ASSISTANT_WORKFLOW_NODE_ORDER[index + 1]
        graph.add_conditional_edges(
            current_node,
            lambda state, next_node=next_node: _next_node_or_failure(state, next_node),
            {
                next_node: next_node,
                "failure": "failure",
            },
        )
    graph.add_conditional_edges(
        "generation",
        lambda state: _next_node_or_failure(state, "review"),
        {
            "review": "review",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "review",
        _next_node_after_review,
        {
            "finalization": "finalization",
            "refinement": "refinement",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "refinement",
        lambda state: _next_node_or_failure(state, "generation"),
        {
            "generation": "generation",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "finalization",
        _next_node_after_finalization,
        {
            "end": END,
            "failure": "failure",
        },
    )
    graph.add_edge("failure", END)
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
    try:
        _emit_streaming_step(
            runtime_context.stream_request_received(runtime_context.event_builder)
        )
        return {
            "workflow_state": complete_workflow_step(
                workflow_state,
                WorkflowStep.INTAKE,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.INTAKE,
            exc=exc,
        )


def _routing_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.ROUTING,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ROUTING,
            exc=exc,
        )


def _memory_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.MEMORY,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.MEMORY,
            exc=exc,
        )


def _retrieval_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.RETRIEVAL,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.RETRIEVAL,
            exc=exc,
        )


def _context_assembly_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.CONTEXT_ASSEMBLY,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.CONTEXT_ASSEMBLY,
            exc=exc,
        )


def _prompt_input_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.PROMPT_INPUT,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PROMPT_INPUT,
            exc=exc,
        )


def _prompt_rendering_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.PROMPT_RENDERING,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PROMPT_RENDERING,
            exc=exc,
        )


def _generation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = _start_graph_workflow_step(
        _workflow_state(state),
        WorkflowStep.GENERATION,
        allow_reentry=True,
    )
    runtime_context = _runtime(runtime)
    try:
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
        generation_failure = _failure_info_from_generation_result(generation_result)
        if generation_failure is not None:
            return {
                "workflow_state": complete_workflow_step(
                    workflow_state,
                    WorkflowStep.GENERATION,
                    error_message=generation_failure.message,
                    failure_info=generation_failure,
                ),
                "pending_failure": generation_failure,
                "failure_event_emitted": True,
                "generation_result": None,
            }
        return {
            "workflow_state": complete_workflow_step(
                workflow_state,
                WorkflowStep.GENERATION,
            ),
            "generation_result": generation_result,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.GENERATION,
            exc=exc,
            clear_generation_result=True,
        )


def _review_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = _start_graph_workflow_step(
        _workflow_state(state),
        WorkflowStep.REVIEW,
        allow_reentry=True,
    )
    runtime_context = _runtime(runtime)
    try:
        review_result = review_assistant_answer(
            request=workflow_state.request,
            answer=_answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            refinement_count=workflow_state.refinement_count,
        )
        return {
            "workflow_state": complete_workflow_step(
                workflow_state,
                WorkflowStep.REVIEW,
                review_result=review_result,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REVIEW,
            exc=exc,
        )


def _refinement_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.REFINEMENT,
    )
    runtime_context = _runtime(runtime)
    review_result = workflow_state.review_result
    try:
        if review_result is None:
            raise ValueError("Workflow review result is not available for refinement.")

        refined_prompt = _append_refinement_guidance(
            rendered_prompt=workflow_state.rendered_prompt,
            review_result=review_result,
        )
        return {
            "workflow_state": complete_workflow_step(
                workflow_state,
                WorkflowStep.REFINEMENT,
                rendered_prompt=refined_prompt,
                refinement_count=workflow_state.refinement_count + 1,
            ),
            "generation_result": None,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REFINEMENT,
            exc=exc,
            clear_generation_result=True,
        )


def _finalization_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.FINALIZATION,
    )
    runtime_context = _runtime(runtime)
    try:
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
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.FINALIZATION,
            exc=exc,
            clear_generation_result=True,
        )


def _failure_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    workflow_state = start_workflow_step(
        _workflow_state(state),
        WorkflowStep.FAILURE,
    )
    runtime_context = _runtime(runtime)
    failure_info = _pending_failure_info(state, workflow_state)
    if not state.get("failure_event_emitted", False):
        _emit(
            runtime_context.event_builder.error(
                code=failure_info.code,
                message=failure_info.message,
            )
        )
    answer = _failure_answer(failure_info)
    final_state = fail_workflow(
        workflow_state,
        error_message=failure_info.message,
        failure_info=failure_info,
        final_answer=answer,
    )
    _emit(
        runtime_context.event_builder.final(
            answer=answer,
            route=state.get("route_payload"),
        )
    )
    return {
        "workflow_state": final_state,
        "pending_failure": None,
        "failure_event_emitted": True,
        "generation_result": None,
    }


def _next_node_after_review(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    workflow_state = _workflow_state(state)
    review_result = workflow_state.review_result
    if review_result is None:
        raise ValueError("Workflow review result is not available.")
    if (
        review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
        and workflow_state.refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
    ):
        return "refinement"
    return "finalization"


def _next_node_after_finalization(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    return "end"


def _next_node_or_failure(
    state: AssistantWorkflowGraphState,
    next_node: str,
) -> str:
    if _has_pending_failure(state):
        return "failure"
    return next_node


def _start_graph_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    if allow_reentry and (step in state.completed_steps or step in state.skipped_steps):
        return restart_workflow_step(state, step)
    return start_workflow_step(state, step)


def _answer_for_review(
    *,
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
) -> str:
    generation_result = state.get("generation_result")
    if generation_result is not None:
        return generation_result.answer
    return runtime.build_shell_answer(_route_decision(workflow_state))


def _failure_info_from_generation_result(
    generation_result: object,
) -> WorkflowFailureInfo | None:
    error_code = getattr(generation_result, "error_code", None)
    error_message = getattr(generation_result, "error_message", None)
    if not error_code or not error_message:
        return None
    return WorkflowFailureInfo(
        step=WorkflowStep.GENERATION,
        code=str(error_code),
        message=str(error_message),
    )


def _append_refinement_guidance(
    *,
    rendered_prompt: RenderedPromptResponse | None,
    review_result: WorkflowReviewResult,
) -> RenderedPromptResponse | None:
    if rendered_prompt is None:
        return None
    reasons = ", ".join(review_result.reasons) or "quality gate did not pass"
    refinement_section = RenderedPromptSection(
        role=RenderedPromptRole.SYSTEM,
        name=RenderedPromptSectionName.SYSTEM,
        content=(
            "Refinement guidance:\n"
            "- Revise the previous answer before finalization.\n"
            f"- Address review issue(s): {reasons}.\n"
            "- Preserve the original user request and existing context."
        ),
    )
    return rendered_prompt.model_copy(
        update={"sections": (*rendered_prompt.sections, refinement_section)}
    )


def _handle_workflow_exception(
    *,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    exc: Exception,
    clear_generation_result: bool = False,
) -> AssistantWorkflowGraphState:
    failure_info = WorkflowFailureInfo(
        step=step,
        code=f"workflow_{step.value}_failed",
        message=str(exc) or f"{step.value} failed.",
    )
    logger.bind(
        step=step.value,
        error_code=failure_info.code,
        error_type=type(exc).__name__,
    ).exception(
        "assistant_workflow_step_failed: {}: {}",
        type(exc).__name__,
        exc,
    )
    _emit(
        runtime.event_builder.error(
            code=failure_info.code,
            message=failure_info.message,
        )
    )
    update: AssistantWorkflowGraphState = {
        "workflow_state": workflow_state.model_copy(
            update={
                "current_step": None,
                "error_message": failure_info.message,
                "failure_info": failure_info,
            }
        ),
        "pending_failure": failure_info,
        "failure_event_emitted": True,
    }
    if clear_generation_result:
        update["generation_result"] = None
    return update


def _pending_failure_info(
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
) -> WorkflowFailureInfo:
    pending_failure = state.get("pending_failure")
    if pending_failure is not None:
        return pending_failure
    if workflow_state.failure_info is not None:
        return workflow_state.failure_info
    raise ValueError("Workflow failure info is not available.")


def _failure_answer(failure_info: WorkflowFailureInfo) -> str:
    if failure_info.step is WorkflowStep.GENERATION:
        return f"Generation failed ({failure_info.code}): {failure_info.message}"
    return (
        "Workflow failed during "
        f"{failure_info.step.value} ({failure_info.code}): "
        f"{failure_info.message}"
    )


def _has_pending_failure(state: AssistantWorkflowGraphState) -> bool:
    return state.get("pending_failure") is not None


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
