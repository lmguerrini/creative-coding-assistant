"""Semantic product-outcome contract for workflow stream events.

Workflow finalization records that orchestration reached its terminal node.  It
does not, by itself, prove that a requested artifact was delivered or that a
browser runtime is healthy.  This module keeps those facts distinct so every
consumer can present an honest product result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from creative_coding_assistant.orchestration.runtime.workflow_review import (
    request_requires_deliverable,
    request_requests_preview,
)

if TYPE_CHECKING:
    from creative_coding_assistant.orchestration.runtime.workflow import (
        AssistantWorkflowState,
    )


def derive_product_outcome(
    workflow_state: AssistantWorkflowState,
) -> dict[str, str]:
    """Return the canonical, transport-safe semantic outcome payload.

    Browser execution happens in the client, so a prepared preview is reported
    as awaiting browser health rather than as a claim that a canvas ran.  The
    client may downgrade this payload when it observes a runtime failure.
    """

    workflow_status = workflow_state.status.value
    completed_steps = {step.value for step in workflow_state.completed_steps}
    failure_code = (
        workflow_state.failure_info.code
        if workflow_state.failure_info is not None
        else ""
    )
    requires_deliverable = request_requires_deliverable(
        workflow_state.request,
        workflow_state.route_decision,
    )
    requires_preview = request_requests_preview(workflow_state.request)
    has_artifacts = bool(workflow_state.artifacts)
    has_previewable_artifact = any(
        artifact.preview_eligible for artifact in workflow_state.artifacts
    )
    has_prepared_preview = bool(workflow_state.preview_results)
    has_failed_preview = any(
        result.status.value == "failed" for result in workflow_state.preview_results
    )
    generation_completed = "generation" in completed_steps
    generation_failed = workflow_status == "failed" and (
        "generation" in failure_code or "provider" in failure_code
    )

    if workflow_state.clarification is not None:
        return {
            "orchestration_status": "AWAITING_CLARIFICATION",
            "provider_status": "NOT_STARTED",
            "generation_status": "PENDING",
            "deliverable_status": "PENDING",
            "artifact_extraction_status": "PENDING",
            "artifact_runnability": "PENDING",
            "preview_status": "PENDING",
            "runtime_health": "PENDING",
            "product_outcome": "IN_PROGRESS",
            "summary": "A clarification is required before generation can continue.",
            "recovery_action": "Choose a clarification option to continue generation.",
        }

    orchestration_status = {
        "running": "RUNNING",
        "completed": "COMPLETED",
        "failed": "FAILED",
    }.get(workflow_status, "PENDING")
    provider_status = (
        "FAILED"
        if "provider" in failure_code
        else "COMPLETED"
        if generation_completed
        else "NOT_STARTED"
    )
    generation_status = (
        "FAILED"
        if generation_failed
        else "COMPLETED"
        if generation_completed
        else "PENDING"
        if workflow_status == "running"
        else "NOT_STARTED"
    )
    artifact_extraction_status = (
        "EXTRACTED"
        if has_artifacts
        else "NOT_PRODUCED"
        if requires_deliverable and workflow_status != "running"
        else "NOT_REQUIRED"
        if not requires_deliverable
        else "PENDING"
    )
    deliverable_status = (
        "USABLE"
        if has_artifacts
        else "NOT_PRODUCED"
        if requires_deliverable and workflow_status != "running"
        else "NOT_REQUIRED"
        if not requires_deliverable
        else "PENDING"
    )
    artifact_runnability = (
        "RUNNABLE"
        if has_previewable_artifact
        else "UNSUPPORTED"
        if has_artifacts and requires_preview
        else "NOT_REQUIRED"
        if has_artifacts or not requires_deliverable
        else "NOT_PRODUCED"
        if requires_deliverable and workflow_status != "running"
        else "PENDING"
    )
    preview_status = (
        "FAILED"
        if has_failed_preview
        else "PREPARED"
        if has_prepared_preview
        else "UNAVAILABLE"
        if has_artifacts and requires_preview
        else "NOT_REQUIRED"
        if not requires_preview
        else "PENDING"
    )
    runtime_health = (
        "PENDING_BROWSER_VALIDATION"
        if has_prepared_preview
        else "NOT_AVAILABLE"
        if has_artifacts and requires_preview
        else "NOT_REQUIRED"
        if not requires_preview
        else "PENDING"
    )

    if workflow_status == "running":
        product_outcome = "IN_PROGRESS"
        summary = "Generation and product validation are still in progress."
        recovery_action = "Wait for the active workflow to finish."
    elif workflow_status == "failed" or (
        requires_deliverable and not has_artifacts
    ):
        product_outcome = "FAILURE"
        summary = (
            "The requested deliverable was not produced."
            if requires_deliverable
            else "The workflow ended in failure before a usable result was produced."
        )
        recovery_action = "Review the failure details, then retry the requested artifact."
    elif requires_preview and (not has_prepared_preview or has_failed_preview):
        product_outcome = "PARTIAL"
        summary = (
            "A usable artifact was produced, but live preview is unavailable."
            if not has_failed_preview
            else "A usable artifact was produced, but the live preview failed."
        )
        recovery_action = "Open Code to use the artifact, then reload or regenerate the preview."
    else:
        product_outcome = "SUCCESS"
        summary = (
            "The requested deliverable is ready."
            if requires_deliverable
            else "The response is ready."
        )
        recovery_action = ""

    return {
        "orchestration_status": orchestration_status,
        "provider_status": provider_status,
        "generation_status": generation_status,
        "deliverable_status": deliverable_status,
        "artifact_extraction_status": artifact_extraction_status,
        "artifact_runnability": artifact_runnability,
        "preview_status": preview_status,
        "runtime_health": runtime_health,
        "product_outcome": product_outcome,
        "summary": summary,
        "recovery_action": recovery_action,
    }
