"""Memory workflow node handler."""

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
    _route_decision,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep


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
