"""Routing workflow node handler."""

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
