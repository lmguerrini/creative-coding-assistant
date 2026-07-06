"""Shared workflow review transition logic."""

from __future__ import annotations

from creative_coding_assistant.orchestration.refinement_passes import (
    plan_next_refinement_pass,
    select_refinement_source,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowStep,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
    WorkflowReviewOutcome,
    WorkflowReviewResult,
)


def _review_transition(
    review_result: WorkflowReviewResult,
    workflow_state: AssistantWorkflowState,
) -> tuple[str, str]:
    if _review_requests_retry(review_result, workflow_state):
        return WorkflowStep.REFINEMENT.value, "review_failed_retry_available"
    if review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT:
        return WorkflowStep.FINALIZATION.value, "review_failed_retry_limit_reached"
    return WorkflowStep.FINALIZATION.value, "review_passed"

def _review_requests_retry(
    review_result: WorkflowReviewResult,
    workflow_state: AssistantWorkflowState,
) -> bool:
    source_artifact = select_refinement_source(workflow_state.artifacts)
    if source_artifact is None:
        return (
            review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
            and workflow_state.refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
        )
    decision = plan_next_refinement_pass(
        source_artifact=source_artifact,
        pass_history=workflow_state.refinement_passes,
        max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
    )
    return (
        review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
        and workflow_state.refinement_count < MAX_WORKFLOW_REFINEMENT_COUNT
        and decision.should_continue
    )
