"""Intake workflow node handler."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit_streaming_step
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _complete_node,
    _handle_workflow_exception,
    _runtime,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep


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
