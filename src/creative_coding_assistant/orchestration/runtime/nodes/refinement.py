"""Refinement workflow node handler."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.artifacts import RefinementPassRecord
from creative_coding_assistant.orchestration.refinement_passes import (
    plan_next_refinement_pass,
    select_refinement_source,
    start_refinement_pass_record,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit_refinement_completed
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _append_refinement_guidance,
    _complete_node,
    _handle_workflow_exception,
    _runtime,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)


def _refinement_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REFINEMENT,
        allow_reentry=True,
    )
    review_result = workflow_state.review_result
    try:
        if review_result is None:
            raise ValueError("Workflow review result is not available for refinement.")

        pass_record: RefinementPassRecord | None = None
        source_artifact = select_refinement_source(workflow_state.artifacts)
        if source_artifact is not None:
            decision = plan_next_refinement_pass(
                source_artifact=source_artifact,
                pass_history=workflow_state.refinement_passes,
                max_passes=MAX_WORKFLOW_REFINEMENT_COUNT,
            )
            if not decision.should_continue:
                raise ValueError(
                    "Workflow refinement node entered after stop condition."
                )
            pass_record = start_refinement_pass_record(
                source_artifact=source_artifact,
                decision=decision,
            )
        refined_prompt = _append_refinement_guidance(
            rendered_prompt=workflow_state.rendered_prompt,
            review_result=review_result,
            artifact_critique_summary=workflow_state.artifact_critique_summary,
            refinement_pass=pass_record,
        )
        retry_count = workflow_state.refinement_count + 1
        _emit_refinement_completed(
            runtime_context,
            workflow_state,
            review_result,
            retry_count=retry_count,
        )
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.REFINEMENT,
                transition_target=WorkflowStep.GENERATION.value,
                decision_reason="refinement_completed",
                rendered_prompt=refined_prompt,
                refinement_count=retry_count,
                refinement_passes=(
                    (*workflow_state.refinement_passes, pass_record)
                    if pass_record is not None
                    else workflow_state.refinement_passes
                ),
            ),
            "generation_result": None,
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REFINEMENT,
            exc=exc,
            clear_generation_result=True,
        )
