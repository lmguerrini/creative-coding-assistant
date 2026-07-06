"""Planning workflow node handler."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit
from creative_coding_assistant.orchestration.runtime.nodes.planning_derivation import (
    _derive_planning_runtime_artifacts,
)
from creative_coding_assistant.orchestration.runtime.nodes.planning_state import (
    _planning_event_payload,
    _planning_runtime_updates,
)
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _complete_node,
    _handle_workflow_exception,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep


def _planning_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PLANNING,
    )
    try:
        prompt_input = workflow_state.prompt_input
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PLANNING,
                    decision_reason="prompt_input_unavailable_for_planning",
                )
            }

        artifacts = _derive_planning_runtime_artifacts(workflow_state, prompt_input)
        planning_updates = _planning_runtime_updates(artifacts)
        planned_prompt_input = prompt_input.model_copy(update=planning_updates)
        planned_state = workflow_state.model_copy(
            update={
                **planning_updates,
                "prompt_input": planned_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_plan_prepared",
                message="Creative execution plan prepared.",
                **_planning_event_payload(artifacts),
            ),
            workflow_state=planned_state,
            step=WorkflowStep.PLANNING,
        )
        return {
            "workflow_state": _complete_node(
                planned_state,
                runtime_context,
                WorkflowStep.PLANNING,
                decision_reason="creative_plan_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PLANNING,
            exc=exc,
        )
