"""Prompt rendering and generation node handlers."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit_node_failed, _emit_streaming_step
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _complete_node,
    _failure_info_from_generation_result,
    _handle_workflow_exception,
    _route_decision,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import (
    WorkflowStep,
    complete_workflow_step,
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
