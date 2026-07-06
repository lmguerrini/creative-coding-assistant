"""State transition selectors for the assistant workflow runtime graph."""
from __future__ import annotations

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphState,
    _GraphTransitionSelector,
)
from creative_coding_assistant.orchestration.runtime.nodes.review_logic import (
    _review_requests_retry,
)
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _has_pending_failure,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep


def next_node_after_review(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    workflow_state = _workflow_state(state)
    review_result = workflow_state.review_result
    if review_result is None:
        raise ValueError("Workflow review result is not available.")
    if _review_requests_retry(review_result, workflow_state):
        return "refinement"
    return "finalization"


def next_node_after_finalization(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    return "end"


def next_node_or_failure(
    state: AssistantWorkflowGraphState,
    next_node: str,
) -> str:
    if _has_pending_failure(state):
        return "failure"
    return next_node


def next_node_after_prompt_input(state: AssistantWorkflowGraphState) -> str:
    if _has_pending_failure(state):
        return "failure"
    if _workflow_state(state).clarification is not None:
        return WorkflowStep.FINALIZATION.value
    return WorkflowStep.PLANNING.value


def next_node_selector(next_node: str) -> _GraphTransitionSelector:
    return lambda state: next_node_or_failure(state, next_node)
