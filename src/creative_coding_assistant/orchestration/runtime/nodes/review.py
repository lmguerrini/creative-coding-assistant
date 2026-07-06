"""Review workflow node handler."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import (
    _emit_refinement_requested,
    _emit_retry_completed,
    _emit_retry_started,
    _emit_review_outcome,
)
from creative_coding_assistant.orchestration.runtime.nodes.review_logic import (
    _review_requests_retry,
    _review_transition,
)
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _answer_for_review,
    _complete_node,
    _handle_workflow_exception,
    _runtime,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep
from creative_coding_assistant.orchestration.workflow_review import review_assistant_answer


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
            artifact_critique_summary=workflow_state.artifact_critique_summary,
        )
        transition_target, decision_reason = _review_transition(
            review_result,
            workflow_state,
        )
        _emit_review_outcome(
            runtime_context,
            workflow_state,
            review_result,
            transition_target=transition_target,
            decision_reason=decision_reason,
        )
        if _review_requests_retry(review_result, workflow_state):
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
