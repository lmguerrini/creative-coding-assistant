"""Workflow graph state helpers for runtime nodes."""

from __future__ import annotations

from langgraph.runtime import Runtime
from loguru import logger

from creative_coding_assistant.orchestration.artifact_critique import ArtifactCritiqueSummary
from creative_coding_assistant.orchestration.artifacts import RefinementPassRecord
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
    RenderedPromptRole,
    RenderedPromptSection,
    RenderedPromptSectionName,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
    AssistantWorkflowRuntime,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import (
    _emit,
    _emit_node_completed,
    _emit_node_failed,
    _emit_node_started,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
    WorkflowStep,
    complete_workflow_step,
    restart_workflow_step,
    skip_workflow_step,
    start_workflow_step,
)
from creative_coding_assistant.orchestration.workflow_review import WorkflowReviewResult


def _start_node(
    state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    workflow_state = _start_graph_workflow_step(
        state,
        step,
        allow_reentry=allow_reentry,
    )
    _emit_node_started(runtime, workflow_state, step)
    return workflow_state

def _complete_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_completed",
    resolution: str = "completed",
    **updates: object,
) -> AssistantWorkflowState:
    completed_state = complete_workflow_step(workflow_state, step, **updates)
    _emit_node_completed(
        runtime,
        completed_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution=resolution,
    )
    return completed_state

def _skip_node(
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    *,
    transition_target: str | None = None,
    decision_reason: str = "node_skipped",
) -> AssistantWorkflowState:
    skipped_state = skip_workflow_step(workflow_state, step)
    _emit_node_completed(
        runtime,
        skipped_state,
        step,
        transition_target=transition_target,
        decision_reason=decision_reason,
        resolution="skipped",
    )
    return skipped_state

def _start_graph_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
    *,
    allow_reentry: bool = False,
) -> AssistantWorkflowState:
    if allow_reentry and (step in state.completed_steps or step in state.skipped_steps):
        return restart_workflow_step(state, step)
    return start_workflow_step(state, step)

def _answer_for_review(
    *,
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
) -> str:
    generation_result = state.get("generation_result")
    if generation_result is not None:
        return generation_result.answer
    return runtime.build_shell_answer(_route_decision(workflow_state))

def _failure_info_from_generation_result(
    generation_result: object,
) -> WorkflowFailureInfo | None:
    error_code = getattr(generation_result, "error_code", None)
    error_message = getattr(generation_result, "error_message", None)
    if not error_code or not error_message:
        return None
    return WorkflowFailureInfo(
        step=WorkflowStep.GENERATION,
        code=str(error_code),
        message=str(error_message),
    )

def _append_refinement_guidance(
    *,
    rendered_prompt: RenderedPromptResponse | None,
    review_result: WorkflowReviewResult,
    artifact_critique_summary: ArtifactCritiqueSummary | None = None,
    refinement_pass: RefinementPassRecord | None = None,
) -> RenderedPromptResponse | None:
    if rendered_prompt is None:
        return None
    reasons = ", ".join(review_result.reasons) or "quality gate did not pass"
    artifact_guidance = (
        "\n- Artifact critique guidance: "
        f"{artifact_critique_summary.refinement_guidance}"
        if artifact_critique_summary and artifact_critique_summary.refinement_guidance
        else ""
    )
    pass_source = (
        refinement_pass.source_artifact_title or refinement_pass.source_artifact_id
        if refinement_pass is not None
        else None
    )
    pass_guidance = (
        "\n"
        f"- Refinement pass: {refinement_pass.pass_number}.\n"
        f"- Source artifact: {pass_source}.\n"
        f"- Pass objective: {refinement_pass.refinement_objective}"
        if refinement_pass is not None
        else ""
    )
    refinement_section = RenderedPromptSection(
        role=RenderedPromptRole.SYSTEM,
        name=RenderedPromptSectionName.SYSTEM,
        content=(
            "Refinement guidance:\n"
            "- Revise the previous answer before finalization.\n"
            f"- Address review issue(s): {reasons}.\n"
            "- Preserve the original user request and existing context."
            f"{artifact_guidance}"
            f"{pass_guidance}"
        ),
    )
    return rendered_prompt.model_copy(
        update={"sections": (*rendered_prompt.sections, refinement_section)}
    )

def _handle_workflow_exception(
    *,
    workflow_state: AssistantWorkflowState,
    runtime: AssistantWorkflowRuntime,
    step: WorkflowStep,
    exc: Exception,
    clear_generation_result: bool = False,
) -> AssistantWorkflowGraphState:
    failure_info = WorkflowFailureInfo(
        step=step,
        code=f"workflow_{step.value}_failed",
        message=str(exc) or f"{step.value} failed.",
    )
    logger.bind(
        step=step.value,
        error_code=failure_info.code,
        error_type=type(exc).__name__,
    ).exception(
        "assistant_workflow_step_failed: {}: {}",
        type(exc).__name__,
        exc,
    )
    _emit_node_failed(
        runtime,
        workflow_state,
        step,
        failure_info,
        decision_reason="node_exception",
    )
    _emit(
        runtime.event_builder.error(
            code=failure_info.code,
            message=failure_info.message,
        ),
        workflow_state=workflow_state,
        step=step,
        phase="failed",
    )
    update: AssistantWorkflowGraphState = {
        "workflow_state": workflow_state.model_copy(
            update={
                "current_step": None,
                "error_message": failure_info.message,
                "failure_info": failure_info,
            }
        ),
        "pending_failure": failure_info,
        "failure_event_emitted": True,
    }
    if clear_generation_result:
        update["generation_result"] = None
    return update

def _pending_failure_info(
    state: AssistantWorkflowGraphState,
    workflow_state: AssistantWorkflowState,
) -> WorkflowFailureInfo:
    pending_failure = state.get("pending_failure")
    if pending_failure is not None:
        return pending_failure
    if workflow_state.failure_info is not None:
        return workflow_state.failure_info
    raise ValueError("Workflow failure info is not available.")

def _failure_answer(failure_info: WorkflowFailureInfo) -> str:
    if failure_info.step is WorkflowStep.GENERATION:
        return f"Generation failed ({failure_info.code}): {failure_info.message}"
    return (
        "Workflow failed during "
        f"{failure_info.step.value} ({failure_info.code}): "
        f"{failure_info.message}"
    )

def _has_pending_failure(state: AssistantWorkflowGraphState) -> bool:
    return state.get("pending_failure") is not None

def _runtime(
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowRuntime:
    return runtime.context["runtime"]

def _workflow_state(
    state: AssistantWorkflowGraphState,
) -> AssistantWorkflowState:
    return state["workflow_state"]

def _route_decision(state: AssistantWorkflowState) -> RouteDecision:
    if state.route_decision is None:
        raise ValueError("Workflow route decision is not available.")
    return state.route_decision

def _format_clarification_answer(clarification: ClarificationRequest) -> str:
    lines = [
        "I need one quick clarification before generating.",
        "",
        clarification.summary,
        "",
    ]
    for index, question in enumerate(clarification.questions, start=1):
        lines.append(f"{index}. {question.prompt}")
        for option in question.suggested_options:
            lines.append(f"- {option}")
        if question.default_recommendation:
            lines.append(f"Default recommendation: {question.default_recommendation}")
        lines.append("")
    lines.append("Reply with your choice and I will continue generation.")
    return "\n".join(lines).strip()
