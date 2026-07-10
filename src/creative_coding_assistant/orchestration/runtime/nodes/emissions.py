"""Streaming event emission helpers for workflow nodes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.config import get_stream_writer

from creative_coding_assistant.contracts import StreamEvent, StreamEventType
from creative_coding_assistant.orchestration.artifact_critique import ArtifactCritiqueSummary
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifactCritique
from creative_coding_assistant.orchestration.runtime.nodes.constants import (
    _FINAL_EVENT_MODEL_PAYLOAD_KEYS,
    _WORKFLOW_RUNTIME_MODEL_PAYLOAD_SPECS,
    ASSISTANT_WORKFLOW_NODE_ORDER,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import AssistantWorkflowRuntime
from creative_coding_assistant.orchestration.runtime.product_outcome import (
    derive_product_outcome,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
    WorkflowStep,
)
from creative_coding_assistant.orchestration.workflow_review import WorkflowReviewResult


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

def _emit_artifact_critique_started(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="critique_started",
            message=(
                f"Critiquing {len(workflow_state.artifacts)} generated artifact"
                f"{'s' if len(workflow_state.artifacts) != 1 else ''}."
            ),
            artifact_count=len(workflow_state.artifacts),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )

def _emit_artifact_scored(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique: WorkflowArtifactCritique,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_scored",
            message=(
                f"Scored {critique.artifact_title} at {critique.overall_score:.2f}."
            ),
            artifact_id=critique.artifact_id,
            artifact_title=critique.artifact_title,
            score=critique.overall_score,
            rank=critique.rank,
            passed=critique.passed,
            critique=critique.model_dump(mode="json"),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )

def _emit_artifact_recommendation(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    if not critique_summary.recommended_artifact_id:
        return
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_selected_recommended",
            message=(
                f"Recommended {critique_summary.recommended_artifact_title} "
                "as the strongest artifact candidate."
            ),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            recommended_artifact_title=critique_summary.recommended_artifact_title,
            average_score=critique_summary.average_score,
            failed_artifact_count=critique_summary.failed_artifact_count,
            critique_summary=critique_summary.model_dump(mode="json"),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )

def _emit_artifact_refinement_requested(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="artifact_refinement_requested",
            message=(
                critique_summary.refinement_guidance
                or "Artifact critique requested refinement."
            ),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            refinement_reasons=list(critique_summary.refinement_reasons),
            refinement_guidance=critique_summary.refinement_guidance,
            critique_summary=critique_summary.model_dump(mode="json"),
            **_transition_payload(
                WorkflowStep.ARTIFACT_CRITIQUE.value,
                WorkflowStep.REVIEW.value,
                "artifact_critique_requested_refinement",
            ),
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )

def _emit_artifact_critique_completed(
    runtime: AssistantWorkflowRuntime,
    workflow_state: AssistantWorkflowState,
    critique_summary: ArtifactCritiqueSummary,
) -> None:
    _emit(
        runtime.event_builder.artifact_critique(
            code="critique_completed",
            message=(
                "Artifact critique completed; "
                f"recommended {critique_summary.recommended_artifact_title}."
            ),
            critique_summary=critique_summary.model_dump(mode="json"),
            recommended_artifact_id=critique_summary.recommended_artifact_id,
            average_score=critique_summary.average_score,
            refinement_required=critique_summary.refinement_required,
        ),
        workflow_state=workflow_state,
        step=WorkflowStep.ARTIFACT_CRITIQUE,
    )

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
                        include_model_payloads=(
                            event.event_type is not StreamEventType.TOKEN_DELTA
                        ),
                    ),
                }
            }
        )
    )

def _serialize_workflow_runtime(
    *,
    workflow_state: AssistantWorkflowState,
    step: WorkflowStep | None,
    phase: str,
    include_model_payloads: bool = True,
) -> dict[str, object]:
    runtime_step = step or workflow_state.current_step
    review_result = workflow_state.review_result
    clarification = workflow_state.clarification
    artifact_critique_summary = workflow_state.artifact_critique_summary

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
        "review_reasons": (
            list(review_result.reasons) if review_result is not None else []
        ),
        "artifact_count": len(workflow_state.artifacts),
        "artifact_critique_count": (
            len(artifact_critique_summary.critiques)
            if artifact_critique_summary is not None
            else 0
        ),
        "recommended_artifact_id": (
            artifact_critique_summary.recommended_artifact_id
            if artifact_critique_summary is not None
            else None
        ),
        "preview_artifact_count": len(workflow_state.preview_results),
        "product_outcome": derive_product_outcome(workflow_state),
        "clarification_required": clarification is not None,
        "clarification_reason": (
            clarification.reason.value if clarification is not None else None
        ),
        "clarification_question_count": (
            len(clarification.questions) if clarification is not None else 0
        ),
        "clarification": _model_json_payload(clarification)
        if include_model_payloads
        else None,
        **(
            _workflow_runtime_model_payloads(workflow_state)
            if include_model_payloads
            else {}
        ),
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

def _workflow_runtime_model_payloads(
    workflow_state: AssistantWorkflowState,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for spec in _WORKFLOW_RUNTIME_MODEL_PAYLOAD_SPECS:
        model_payload = _model_json_payload(
            getattr(workflow_state, spec.state_attribute)
        )
        payload[spec.payload_key] = model_payload
        if spec.availability_key is not None:
            payload[spec.availability_key] = model_payload is not None
    return payload

def _final_event_model_payloads(
    workflow_state: AssistantWorkflowState,
) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for payload_key in _FINAL_EVENT_MODEL_PAYLOAD_KEYS:
        model_payload = _model_json_payload(getattr(workflow_state, payload_key))
        if model_payload is not None:
            payloads[payload_key] = model_payload
    return payloads

def _model_json_payload(value: Any | None) -> dict[str, object] | None:
    if value is None:
        return None
    return value.model_dump(mode="json")

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
