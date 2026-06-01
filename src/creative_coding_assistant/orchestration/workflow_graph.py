"""LangGraph runtime for the assistant workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from loguru import logger

from creative_coding_assistant.analytics import (
    LangSmithObservability,
    LangSmithRunMetadata,
)
from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.artifacts import (
    extract_workflow_artifacts,
    prepare_workflow_preview_results,
)
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
    observability: LangSmithObservability
    observability_run: LangSmithRunMetadata
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
    "artifact_extraction",
    "preview_preparation",
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
    graph.add_node("artifact_extraction", _artifact_extraction_node)
    graph.add_node("preview_preparation", _preview_preparation_node)
    graph.add_node("review", _review_node)
    graph.add_node("refinement", _refinement_node)
    graph.add_node("finalization", _finalization_node)
    graph.add_node("failure", _failure_node)

    graph.add_edge(START, "intake")
    review_index = ASSISTANT_WORKFLOW_NODE_ORDER.index("review")
    for index in range(review_index):
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.INTAKE,
    )
    try:
        _emit_streaming_step(
            runtime_context.stream_request_received(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                observability=runtime_context.observability,
                observability_run=runtime_context.observability_run,
            ),
            workflow_state=workflow_state,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.INTAKE,
                decision_reason="request_received",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ROUTING,
    )
    try:
        decision = runtime_context.route_fn(workflow_state.request)
        route_payload = decision.model_dump(mode="json")
        _emit_streaming_step(
            runtime_context.stream_route_selected(
                builder=runtime_context.event_builder,
                decision=decision,
                route_payload=route_payload,
            ),
            workflow_state=workflow_state,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ROUTING,
                decision_reason=f"route_selected:{decision.route.value}",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.MEMORY,
    )
    try:
        memory_context = _emit_streaming_step(
            runtime_context.stream_memory_context(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
            ),
            workflow_state=workflow_state,
        )
        if memory_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.MEMORY,
                    decision_reason="memory_context_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.MEMORY,
                decision_reason="memory_context_available",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.RETRIEVAL,
    )
    try:
        retrieval_context = _emit_streaming_step(
            runtime_context.stream_retrieval_context(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
                observability_run=runtime_context.observability_run,
            ),
            workflow_state=workflow_state,
        )
        if retrieval_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.RETRIEVAL,
                    decision_reason="retrieval_context_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.RETRIEVAL,
                decision_reason="retrieval_context_available",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.CONTEXT_ASSEMBLY,
    )
    try:
        assembled_context = _emit_streaming_step(
            runtime_context.stream_assembled_context(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                memory_context=workflow_state.memory_context,
                retrieval_context=workflow_state.retrieval_context,
            ),
            workflow_state=workflow_state,
        )
        if assembled_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.CONTEXT_ASSEMBLY,
                    decision_reason="context_assembly_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.CONTEXT_ASSEMBLY,
                decision_reason="context_assembled",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PROMPT_INPUT,
    )
    try:
        prompt_input = _emit_streaming_step(
            runtime_context.stream_prompt_inputs(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
                assembled_context=workflow_state.assembled_context,
            ),
            workflow_state=workflow_state,
        )
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_INPUT,
                    decision_reason="prompt_input_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PROMPT_INPUT,
                decision_reason="prompt_input_prepared",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PROMPT_RENDERING,
    )
    try:
        rendered_prompt = _emit_streaming_step(
            runtime_context.stream_rendered_prompt(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                prompt_inputs=workflow_state.prompt_input,
            ),
            workflow_state=workflow_state,
        )
        if rendered_prompt is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_RENDERING,
                    decision_reason="prompt_rendering_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PROMPT_RENDERING,
                decision_reason="prompt_rendered",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.GENERATION,
        allow_reentry=True,
    )
    try:
        generation_result = _emit_streaming_step(
            runtime_context.stream_generation(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                rendered_prompt=workflow_state.rendered_prompt,
            ),
            workflow_state=workflow_state,
        )
        if generation_result is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.GENERATION,
                    decision_reason="generation_unavailable",
                )
            }
        generation_failure = _failure_info_from_generation_result(generation_result)
        if generation_failure is not None:
            _emit_node_failed(
                runtime_context,
                workflow_state,
                WorkflowStep.GENERATION,
                generation_failure,
                decision_reason="generation_provider_failed",
            )
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
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.GENERATION,
                decision_reason="generation_completed",
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


def _artifact_extraction_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ARTIFACT_EXTRACTION,
        allow_reentry=True,
    )
    try:
        artifacts = extract_workflow_artifacts(
            _answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
        )
        if not artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(
                        update={"artifacts": (), "preview_results": ()}
                    ),
                    runtime_context,
                    WorkflowStep.ARTIFACT_EXTRACTION,
                    decision_reason="no_generated_artifacts",
                )
            }

        _emit(
            runtime_context.event_builder.artifact_extracted(
                artifacts=artifacts,
                code="artifact_extracted",
                message=(
                    f"Extracted {len(artifacts)} generated artifact"
                    f"{'s' if len(artifacts) != 1 else ''} from the answer."
                ),
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ARTIFACT_EXTRACTION,
                decision_reason="artifacts_extracted",
                artifacts=artifacts,
                preview_results=(),
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
            exc=exc,
        )


def _preview_preparation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PREVIEW_PREPARATION,
        allow_reentry=True,
    )
    try:
        if not workflow_state.artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_artifacts_for_preview",
                )
            }

        preview_results = prepare_workflow_preview_results(
            workflow_state.artifacts,
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
        )
        if not preview_results:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_previewable_artifacts",
                )
            }

        for result in preview_results:
            _emit(
                runtime_context.event_builder.preview_artifact(
                    result,
                    code="preview_artifact_prepared",
                    message=result.summary
                    or "Preview runtime metadata prepared for the artifact.",
                ),
                workflow_state=workflow_state,
                step=WorkflowStep.PREVIEW_PREPARATION,
            )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PREVIEW_PREPARATION,
                decision_reason="preview_metadata_prepared",
                preview_results=preview_results,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PREVIEW_PREPARATION,
            exc=exc,
        )


def _review_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REVIEW,
        allow_reentry=True,
    )
    try:
        previous_review_result = workflow_state.review_result
        review_result = review_assistant_answer(
            request=workflow_state.request,
            answer=_answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            refinement_count=workflow_state.refinement_count,
        )
        transition_target, decision_reason = _review_transition(
            review_result,
            workflow_state.refinement_count,
        )
        _emit_review_outcome(
            runtime_context,
            workflow_state,
            review_result,
            transition_target=transition_target,
            decision_reason=decision_reason,
        )
        if _review_requests_retry(review_result, workflow_state.refinement_count):
            _emit_refinement_requested(
                runtime_context,
                workflow_state,
                review_result,
            )
            _emit_retry_started(
                runtime_context,
                workflow_state,
                review_result,
            )
        elif workflow_state.refinement_count > 0:
            _emit_retry_completed(
                runtime_context,
                workflow_state,
                review_result,
                previous_review_result,
                transition_target=transition_target,
                decision_reason=decision_reason,
            )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.REVIEW,
                transition_target=transition_target,
                decision_reason=decision_reason,
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REFINEMENT,
    )
    review_result = workflow_state.review_result
    try:
        if review_result is None:
            raise ValueError("Workflow review result is not available for refinement.")

        refined_prompt = _append_refinement_guidance(
            rendered_prompt=workflow_state.rendered_prompt,
            review_result=review_result,
        )
        retry_count = workflow_state.refinement_count + 1
        _emit_refinement_completed(
            runtime_context,
            workflow_state,
            review_result,
            retry_count=retry_count,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.REFINEMENT,
                transition_target=WorkflowStep.GENERATION.value,
                decision_reason="refinement_completed",
                rendered_prompt=refined_prompt,
                refinement_count=retry_count,
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FINALIZATION,
    )
    try:
        generation_result = state.get("generation_result")
        if generation_result is not None:
            answer = generation_result.answer
        else:
            answer = runtime_context.build_shell_answer(_route_decision(workflow_state))
        telemetry = (
            getattr(generation_result, "telemetry", None)
            if generation_result is not None
            else None
        )

        final_state = finish_workflow(workflow_state, final_answer=answer)
        _emit_node_completed(
            runtime_context,
            final_state,
            WorkflowStep.FINALIZATION,
            transition_target="end",
            decision_reason="final_answer_emitted",
            resolution="completed",
        )
        _emit(
            runtime_context.event_builder.final(
                answer=answer,
                route=state["route_payload"],
                artifacts=[
                    artifact.model_dump(mode="json")
                    for artifact in final_state.artifacts
                ],
                preview_results=[
                    result.model_dump(mode="json")
                    for result in final_state.preview_results
                ],
                **_optional_event_payload(
                    "observability",
                    runtime_context.observability.event_payload(
                        runtime_context.observability_run,
                        lineage={"stage": WorkflowStep.FINALIZATION.value},
                    ),
                ),
                **({"telemetry": telemetry} if telemetry is not None else {}),
            ),
            workflow_state=final_state,
            step=WorkflowStep.FINALIZATION,
            phase="completed",
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
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FAILURE,
    )
    failure_info = _pending_failure_info(state, workflow_state)
    if not state.get("failure_event_emitted", False):
        _emit(
            runtime_context.event_builder.error(
                code=failure_info.code,
                message=failure_info.message,
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.FAILURE,
            phase="failed",
        )
    answer = _failure_answer(failure_info)
    final_state = fail_workflow(
        workflow_state,
        error_message=failure_info.message,
        failure_info=failure_info,
        final_answer=answer,
    )
    _emit_node_completed(
        runtime_context,
        final_state,
        WorkflowStep.FAILURE,
        transition_target="end",
        decision_reason="terminal_failure_answer_emitted",
        resolution="completed",
    )
    _emit(
        runtime_context.event_builder.final(
            answer=answer,
            route=state.get("route_payload"),
            **_optional_event_payload(
                "observability",
                runtime_context.observability.event_payload(
                    runtime_context.observability_run,
                    lineage={"stage": WorkflowStep.FAILURE.value},
                ),
            ),
        ),
        workflow_state=final_state,
        step=WorkflowStep.FAILURE,
        phase="failed",
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


def _review_transition(
    review_result: WorkflowReviewResult,
    refinement_count: int,
) -> tuple[str, str]:
    if _review_requests_retry(review_result, refinement_count):
        return WorkflowStep.REFINEMENT.value, "review_failed_retry_available"
    if review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT:
        return WorkflowStep.FINALIZATION.value, "review_failed_retry_limit_reached"
    return WorkflowStep.FINALIZATION.value, "review_passed"


def _review_requests_retry(
    review_result: WorkflowReviewResult,
    refinement_count: int,
) -> bool:
    return (
        review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
        and refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
    )


def _start_node(
    state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    workflow_state = _start_graph_workflow_step(
        state,
        step,
        allow_reentry=allow_reentry,
    )
    _emit_node_started(runtime, workflow_state, step)
    return workflow_state


def _complete_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_completed",
    resolution: str = "completed",
    **updates: object,
) -> AssistantWorkflowState:
    completed_state = complete_workflow_step(workflow_state, step, **updates)
    _emit_node_completed(
        runtime,
        completed_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution=resolution,
    )
    return completed_state


def _skip_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_skipped",
) -> AssistantWorkflowState:
    skipped_state = skip_workflow_step(workflow_state, step)
    _emit_node_completed(
        runtime,
        skipped_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution="skipped",
    )
    return skipped_state


def _emit_node_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
) -> None:
    _emit(
        runtime.event_builder.node_started(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} started.",
            attempt_count=_node_attempt_count(workflow_state, step),
        ),
        workflow_state=workflow_state,
        step=step,
    )


def _emit_node_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
    *,
    transition_target: str | None,
    decision_reason: str,
    resolution: str,
) -> None:
    target = transition_target or _default_transition_target(step)
    _emit(
        runtime.event_builder.node_completed(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} {resolution}.",
            resolution=resolution,
            **_transition_payload(step.value, target, decision_reason),
        ),
        workflow_state=workflow_state,
        step=step,
        phase="completed",
    )


def _emit_node_failed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
    failure_info: WorkflowFailureInfo,
    *,
    transition_target: str = "failure",
    decision_reason: str = "node_failed",
) -> None:
    _emit(
        runtime.event_builder.node_failed(
            node=step.value,
            node_label=_step_label(step),
            message=f"{_step_label(step)} failed: {failure_info.message}",
            error_code=failure_info.code,
            error_message=failure_info.message,
            **_transition_payload(step.value, transition_target, decision_reason),
        ),
        workflow_state=workflow_state,
        step=step,
        phase="failed",
    )


def _emit_review_outcome(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    *,
    transition_target: str,
    decision_reason: str,
) -> None:
    payload = {
        "score": review_result.score,
        "rationale": review_result.rationale,
        "review": review_result.model_dump(mode="json"),
        "review_outcome": review_result.outcome.value,
        "review_reasons": list(review_result.reasons),
        "refinement_count": review_result.refinement_count,
        **_transition_payload(
            WorkflowStep.REVIEW.value,
            transition_target,
            decision_reason,
        ),
    }
    if review_result.passed:
        event = runtime.event_builder.review_passed(
            message=review_result.rationale,
            **payload,
        )
    else:
        event = runtime.event_builder.review_failed(
            message=review_result.rationale,
            **payload,
        )
    _emit(event, workflow_state=workflow_state, step=WorkflowStep.REVIEW)


def _emit_refinement_requested(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
) -> None:
    retry_count = workflow_state.refinement_count + 1
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.refinement_requested(
            message=f"Refinement requested for retry {retry_count}: {reason}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                WorkflowStep.REFINEMENT.value,
                "review_failed_retry_available",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


def _emit_refinement_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    *,
    retry_count: int,
) -> None:
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.refinement_completed(
            message=f"Refinement guidance prepared for retry {retry_count}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REFINEMENT.value,
                WorkflowStep.GENERATION.value,
                "refinement_completed",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REFINEMENT,
    )


def _emit_retry_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
) -> None:
    retry_count = workflow_state.refinement_count + 1
    reason = _review_reason_text(review_result)
    _emit(
        runtime.event_builder.retry_started(
            message=f"Retry {retry_count} started: {reason}.",
            retry_count=retry_count,
            retry_reason=reason,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                WorkflowStep.REFINEMENT.value,
                "review_failed_retry_available",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


def _emit_retry_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    review_result: WorkflowReviewResult,
    previous_review_result: WorkflowReviewResult | None,
    *,
    transition_target: str,
    decision_reason: str,
) -> None:
    retry_count = workflow_state.refinement_count
    reason = _review_reason_text(previous_review_result or review_result)
    status = "passed" if review_result.passed else "exhausted"
    _emit(
        runtime.event_builder.retry_completed(
            message=f"Retry {retry_count} {status}: {review_result.rationale}",
            retry_count=retry_count,
            retry_reason=reason,
            retry_status=status,
            review=review_result.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.REVIEW.value,
                transition_target,
                decision_reason,
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.REVIEW,
    )


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
    _emit_node_failed(
        runtime,
        workflow_state,
        step,
        failure_info,
        decision_reason="node_exception",
    )
    _emit(
        runtime.event_builder.error(
            code=failure_info.code,
            message=failure_info.message,
        ),
        workflow_state=workflow_state,
        step=step,
        phase="failed",
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


def _emit_streaming_step(
    step: Iterator[object],
    *,
    workflow_state: AssistantWorkflowState,
) -> object:
    while True:
        try:
            item = next(step)
        except StopIteration as exc:
            return exc.value
        if isinstance(item, StreamEvent):
            _emit(item, workflow_state=workflow_state)


def _emit(
    event: StreamEvent,
    *,
    workflow_state: AssistantWorkflowState | None = None,
    step: WorkflowStep | None = None,
    phase: str = "running",
) -> None:
    writer = get_stream_writer()
    if workflow_state is None:
        writer(event)
        return
    writer(
        event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "workflow": _serialize_workflow_runtime(
                        workflow_state=workflow_state,
                        step=step,
                        phase=phase,
                    ),
                }
            }
        )
    )


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


def _serialize_workflow_runtime(
    *,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep | None,
    phase: str,
) -> dict[str, object]:
    runtime_step = step or workflow_state.current_step
    review_result = workflow_state.review_result

    return {
        "step": runtime_step.value if runtime_step is not None else None,
        "phase": phase,
        "status": workflow_state.status.value,
        "current_step": (
            workflow_state.current_step.value
            if workflow_state.current_step is not None
            else None
        ),
        "completed_steps": [item.value for item in workflow_state.completed_steps],
        "skipped_steps": [item.value for item in workflow_state.skipped_steps],
        "refinement_count": workflow_state.refinement_count,
        "review_outcome": (
            review_result.outcome.value if review_result is not None else None
        ),
        "review_reasons": list(review_result.reasons) if review_result else [],
        "artifact_count": len(workflow_state.artifacts),
        "preview_artifact_count": len(workflow_state.preview_results),
        "image_reference_count": len(workflow_state.request.attachments),
        "image_references": [
            {
                "id": image.id,
                "name": image.name,
                "mime_type": image.mime_type,
                "size_bytes": image.size_bytes,
            }
            for image in workflow_state.request.attachments
        ],
    }


def _optional_event_payload(
    key: str,
    value: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    return {key: value} if value is not None else {}


def _transition_payload(
    source: str,
    target: str,
    decision_reason: str,
) -> dict[str, object]:
    return {
        "transition_source": source,
        "transition_target": target,
        "decision_reason": decision_reason,
        "edge": {
            "source": source,
            "target": target,
            "decision_reason": decision_reason,
        },
    }


def _default_transition_target(step: WorkflowStep) -> str:
    if step is WorkflowStep.REFINEMENT:
        return WorkflowStep.GENERATION.value
    if step in {WorkflowStep.FINALIZATION, WorkflowStep.FAILURE}:
        return "end"

    try:
        next_index = ASSISTANT_WORKFLOW_NODE_ORDER.index(step.value) + 1
    except ValueError:
        return "end"

    if next_index >= len(ASSISTANT_WORKFLOW_NODE_ORDER):
        return "end"
    return ASSISTANT_WORKFLOW_NODE_ORDER[next_index]


def _step_label(step: WorkflowStep) -> str:
    return step.value.replace("_", " ").title()


def _node_attempt_count(
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep,
) -> int:
    if step is WorkflowStep.GENERATION:
        return workflow_state.refinement_count + 1
    if step is WorkflowStep.REFINEMENT:
        return workflow_state.refinement_count + 1
    return 1


def _review_reason_text(review_result: WorkflowReviewResult) -> str:
    return ", ".join(review_result.reasons) or "quality gate passed"
