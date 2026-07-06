"""Creative Assistant Director workflow node handler."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
    derive_creative_assistant_director_brief,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _complete_node,
    _handle_workflow_exception,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowStep,
)


def _director_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.DIRECTOR,
    )
    try:
        if workflow_state.prompt_input is None or workflow_state.creative_plan is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.DIRECTOR,
                    decision_reason="director_inputs_unavailable",
                )
            }

        director = _derive_director_brief(workflow_state)
        directed_prompt_input = workflow_state.prompt_input.model_copy(
            update={"creative_director": director}
        )
        directed_state = workflow_state.model_copy(
            update={
                "creative_director": director,
                "prompt_input": directed_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_director_prepared",
                message="Creative Assistant Director guidance prepared.",
                creative_director=director.model_dump(mode="json"),
            ),
            workflow_state=directed_state,
            step=WorkflowStep.DIRECTOR,
        )
        return {
            "workflow_state": _complete_node(
                directed_state,
                runtime_context,
                WorkflowStep.DIRECTOR,
                decision_reason="creative_director_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.DIRECTOR,
            exc=exc,
        )


def _derive_director_brief(
    workflow_state: AssistantWorkflowState,
) -> CreativeAssistantDirectorBrief:
    prompt_input = workflow_state.prompt_input
    return derive_creative_assistant_director_brief(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_strategy=workflow_state.creative_strategy,
        creative_techniques=workflow_state.creative_techniques,
        creative_plan=workflow_state.creative_plan,
        creative_constraints=workflow_state.creative_constraints,
        creative_constraint_priorities=workflow_state.creative_constraint_priorities,
        runtime_capabilities=workflow_state.runtime_capabilities,
        creative_tradeoffs=workflow_state.creative_tradeoffs,
        creative_quality_prediction=workflow_state.creative_quality_prediction,
        symbolic_narrative=workflow_state.symbolic_narrative,
        creative_composition=workflow_state.creative_composition,
        procedural_structure=workflow_state.procedural_structure,
        generative_structure=workflow_state.generative_structure,
        semantic_motif=workflow_state.semantic_motif,
        emotional_consistency=workflow_state.emotional_consistency,
        cross_modality=workflow_state.cross_modality,
        audio_visual_scene=workflow_state.audio_visual_scene,
        artifact_plan=workflow_state.artifact_plan,
        artifact_dependency_graph=workflow_state.artifact_dependency_graph,
        runtime_compatibility=workflow_state.runtime_compatibility,
        artifact_capability_matrix=workflow_state.artifact_capability_matrix,
        multi_artifact_strategy=workflow_state.multi_artifact_strategy,
        artifact_critic=workflow_state.artifact_critic,
        artifact_refiner=workflow_state.artifact_refiner,
        artifact_intelligence_synthesis=(
            workflow_state.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=workflow_state.artifact_merge_planner,
        artifact_export_intelligence=workflow_state.artifact_export_intelligence,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=(workflow_state.creative_improvement_planner),
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        creative_score=workflow_state.creative_score,
        consistency_validation=workflow_state.consistency_validation,
        evaluation_report=workflow_state.evaluation_report,
        evaluation_engine_contracts=workflow_state.evaluation_engine_contracts,
        clarification=workflow_state.clarification,
        retrieval_chunk_count=(
            len(prompt_input.retrieval_input.chunks)
            if prompt_input is not None and prompt_input.retrieval_input is not None
            else 0
        ),
        artifact_critique_summary=workflow_state.artifact_critique_summary,
        review_result=workflow_state.review_result,
        refinement_count=workflow_state.refinement_count,
    )
