"""Artifact extraction, preview, and critique node handlers."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.artifact_critique import (
    critique_workflow_artifacts,
)
from creative_coding_assistant.orchestration.artifacts import (
    extract_workflow_artifacts,
    prepare_workflow_preview_results,
)
from creative_coding_assistant.orchestration.refinement_passes import (
    attach_refinement_history,
    complete_latest_refinement_pass,
    select_refinement_source,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import (
    _emit,
    _emit_artifact_critique_completed,
    _emit_artifact_critique_started,
    _emit_artifact_recommendation,
    _emit_artifact_refinement_requested,
    _emit_artifact_scored,
)
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _answer_for_review,
    _complete_node,
    _handle_workflow_exception,
    _route_decision,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)


def _artifact_extraction_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ARTIFACT_EXTRACTION,
        allow_reentry=True,
    )
    try:
        artifacts = extract_workflow_artifacts(
            _answer_for_review(
                state=state,
                workflow_state=workflow_state,
                runtime=runtime_context,
            ),
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
            creative_translation=(
                workflow_state.prompt_input.creative_translation
                if workflow_state.prompt_input is not None
                else None
            ),
            creative_plan=workflow_state.creative_plan,
        )
        if not artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(
                        update={"artifacts": (), "preview_results": ()}
                    ),
                    runtime_context,
                    WorkflowStep.ARTIFACT_EXTRACTION,
                    decision_reason="no_generated_artifacts",
                )
            }

        _emit(
            runtime_context.event_builder.artifact_extracted(
                artifacts=artifacts,
                code="artifact_extracted",
                message=(
                    f"Extracted {len(artifacts)} generated artifact"
                    f"{'s' if len(artifacts) != 1 else ''} from the answer."
                ),
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ARTIFACT_EXTRACTION,
                decision_reason="artifacts_extracted",
                artifacts=artifacts,
                preview_results=(),
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ARTIFACT_EXTRACTION,
            exc=exc,
        )

def _preview_preparation_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PREVIEW_PREPARATION,
        allow_reentry=True,
    )
    try:
        if not workflow_state.artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_artifacts_for_preview",
                )
            }

        preview_results = prepare_workflow_preview_results(
            workflow_state.artifacts,
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
        )
        if not preview_results:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(update={"preview_results": ()}),
                    runtime_context,
                    WorkflowStep.PREVIEW_PREPARATION,
                    decision_reason="no_previewable_artifacts",
                )
            }

        for result in preview_results:
            _emit(
                runtime_context.event_builder.preview_artifact(
                    result,
                    code="preview_artifact_prepared",
                    message=result.summary
                    or "Preview runtime metadata prepared for the artifact.",
                ),
                workflow_state=workflow_state,
                step=WorkflowStep.PREVIEW_PREPARATION,
            )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PREVIEW_PREPARATION,
                decision_reason="preview_metadata_prepared",
                preview_results=preview_results,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PREVIEW_PREPARATION,
            exc=exc,
        )

def _artifact_critique_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.ARTIFACT_CRITIQUE,
        allow_reentry=True,
    )
    try:
        if not workflow_state.artifacts:
            return {
                "workflow_state": _skip_node(
                    workflow_state.model_copy(
                        update={"artifact_critique_summary": None}
                    ),
                    runtime_context,
                    WorkflowStep.ARTIFACT_CRITIQUE,
                    decision_reason="no_artifacts_for_critique",
                )
            }

        _emit_artifact_critique_started(runtime_context, workflow_state)
        artifacts, critique_summary = critique_workflow_artifacts(
            workflow_state.artifacts,
            request=workflow_state.request,
            route_decision=_route_decision(workflow_state),
            preview_results=workflow_state.preview_results,
        )
        refinement_passes = workflow_state.refinement_passes
        if workflow_state.refinement_count > 0 and refinement_passes:
            refinement_passes = complete_latest_refinement_pass(
                pass_history=refinement_passes,
                result_artifact=select_refinement_source(artifacts),
                max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
            )
            artifacts = attach_refinement_history(artifacts, refinement_passes)
        for critique in critique_summary.critiques:
            _emit_artifact_scored(runtime_context, workflow_state, critique)
        _emit_artifact_recommendation(runtime_context, workflow_state, critique_summary)
        if critique_summary.refinement_required:
            _emit_artifact_refinement_requested(
                runtime_context,
                workflow_state,
                critique_summary,
            )
        _emit_artifact_critique_completed(
            runtime_context,
            workflow_state,
            critique_summary,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.ARTIFACT_CRITIQUE,
                decision_reason=(
                    "artifact_critique_requested_refinement"
                    if critique_summary.refinement_required
                    else "artifact_critique_completed"
                ),
                artifacts=artifacts,
                artifact_critique_summary=critique_summary,
                refinement_passes=refinement_passes,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.ARTIFACT_CRITIQUE,
            exc=exc,
        )
