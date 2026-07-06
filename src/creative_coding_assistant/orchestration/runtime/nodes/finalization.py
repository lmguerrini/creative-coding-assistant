"""Finalization and terminal failure node handlers."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.consistency_validation_engine import (
    derive_consistency_validation_profile,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    derive_creative_confidence_profile,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    derive_creative_improvement_planner_profile,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    derive_creative_score_profile,
)
from creative_coding_assistant.orchestration.evaluation_reports import (
    derive_evaluation_report_profile,
)
from creative_coding_assistant.orchestration.events import optional_event_payload
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    derive_reflection_loop_profile,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import (
    _emit,
    _emit_node_completed,
    _final_event_model_payloads,
    _model_json_payload,
)
from creative_coding_assistant.orchestration.runtime.nodes.planning import (
    _derive_director_brief,
    _derive_reasoning_result,
)
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _failure_answer,
    _format_clarification_answer,
    _handle_workflow_exception,
    _pending_failure_info,
    _route_decision,
    _runtime,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    derive_self_evaluation_profile,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowStep,
    fail_workflow,
    finish_workflow,
)


def _finalization_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FINALIZATION,
    )
    try:
        generation_result = state.get("generation_result")
        if generation_result is not None:
            answer = generation_result.answer
        elif workflow_state.clarification is not None:
            answer = _format_clarification_answer(workflow_state.clarification)
        else:
            answer = runtime_context.build_shell_answer(_route_decision(workflow_state))
        telemetry = (
            getattr(generation_result, "telemetry", None)
            if generation_result is not None
            else None
        )

        self_evaluation = _derive_self_evaluation_result(
            workflow_state,
            generated_response=answer,
        )
        evaluation_state = workflow_state.model_copy(
            update={"self_evaluation": self_evaluation}
        )
        creative_improvement_planner = _derive_creative_improvement_planner_result(
            evaluation_state,
            generated_response=answer,
        )
        reflection_loop = _derive_reflection_loop_result(
            evaluation_state.model_copy(
                update={"creative_improvement_planner": (creative_improvement_planner)}
            )
        )
        creative_confidence = _derive_creative_confidence_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (creative_improvement_planner),
                    "reflection_loop": reflection_loop,
                }
            )
        )
        creative_score = _derive_creative_score_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (creative_improvement_planner),
                    "reflection_loop": reflection_loop,
                    "creative_confidence": creative_confidence,
                }
            )
        )
        consistency_validation = _derive_consistency_validation_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (creative_improvement_planner),
                    "reflection_loop": reflection_loop,
                    "creative_confidence": creative_confidence,
                    "creative_score": creative_score,
                }
            )
        )
        evaluation_report = _derive_evaluation_report_result(
            evaluation_state.model_copy(
                update={
                    "creative_improvement_planner": (creative_improvement_planner),
                    "reflection_loop": reflection_loop,
                    "creative_confidence": creative_confidence,
                    "creative_score": creative_score,
                    "consistency_validation": consistency_validation,
                }
            )
        )
        evaluated_prompt_input = (
            workflow_state.prompt_input.model_copy(
                update={
                    "self_evaluation": self_evaluation,
                    "creative_improvement_planner": (creative_improvement_planner),
                    "reflection_loop": reflection_loop,
                    "creative_confidence": creative_confidence,
                    "creative_score": creative_score,
                    "consistency_validation": consistency_validation,
                    "evaluation_report": evaluation_report,
                }
            )
            if workflow_state.prompt_input is not None
            else None
        )
        evaluated_state = workflow_state.model_copy(
            update={
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": (creative_improvement_planner),
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
                "creative_score": creative_score,
                "consistency_validation": consistency_validation,
                "evaluation_report": evaluation_report,
                "prompt_input": evaluated_prompt_input,
            }
        )
        updated_director = _derive_director_brief(evaluated_state)
        directed_prompt_input = (
            evaluated_prompt_input.model_copy(
                update={"creative_director": updated_director}
            )
            if evaluated_prompt_input is not None
            else None
        )
        directed_state = evaluated_state.model_copy(
            update={
                "creative_director": updated_director,
                "prompt_input": directed_prompt_input,
            }
        )
        if workflow_state.creative_reasoning is not None:
            updated_reasoning = _derive_reasoning_result(directed_state)
            reasoned_prompt_input = (
                directed_prompt_input.model_copy(
                    update={"creative_reasoning": updated_reasoning}
                )
                if directed_prompt_input is not None
                else None
            )
            directed_state = directed_state.model_copy(
                update={
                    "creative_reasoning": updated_reasoning,
                    "prompt_input": reasoned_prompt_input,
                }
            )
        final_state = finish_workflow(directed_state, final_answer=answer)
        _emit_node_completed(
            runtime_context,
            final_state,
            WorkflowStep.FINALIZATION,
            transition_target="end",
            decision_reason="final_answer_emitted",
            resolution="completed",
        )
        _emit(
            runtime_context.event_builder.final(
                answer=answer,
                route=state["route_payload"],
                artifacts=[
                    artifact.model_dump(mode="json")
                    for artifact in final_state.artifacts
                ],
                artifact_critiques=[
                    critique.model_dump(mode="json")
                    for critique in (
                        final_state.artifact_critique_summary.critiques
                        if final_state.artifact_critique_summary is not None
                        else ()
                    )
                ],
                artifact_critique_summary=_model_json_payload(
                    final_state.artifact_critique_summary
                ),
                preview_results=[
                    result.model_dump(mode="json")
                    for result in final_state.preview_results
                ],
                **_final_event_model_payloads(final_state),
                **optional_event_payload(
                    "observability",
                    runtime_context.observability.event_payload(
                        runtime_context.observability_run,
                        lineage={"stage": WorkflowStep.FINALIZATION.value},
                    ),
                ),
                **({"telemetry": telemetry} if telemetry is not None else {}),
            ),
            workflow_state=final_state,
            step=WorkflowStep.FINALIZATION,
            phase="completed",
        )
        return {"workflow_state": final_state}
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.FINALIZATION,
            exc=exc,
            clear_generation_result=True,
        )

def _failure_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.FAILURE,
    )
    failure_info = _pending_failure_info(state, workflow_state)
    if not state.get("failure_event_emitted", False):
        _emit(
            runtime_context.event_builder.error(
                code=failure_info.code,
                message=failure_info.message,
            ),
            workflow_state=workflow_state,
            step=WorkflowStep.FAILURE,
            phase="failed",
        )
    answer = _failure_answer(failure_info)
    final_state = fail_workflow(
        workflow_state,
        error_message=failure_info.message,
        failure_info=failure_info,
        final_answer=answer,
    )
    _emit_node_completed(
        runtime_context,
        final_state,
        WorkflowStep.FAILURE,
        transition_target="end",
        decision_reason="terminal_failure_answer_emitted",
        resolution="completed",
    )
    _emit(
        runtime_context.event_builder.final(
            answer=answer,
            route=state.get("route_payload"),
            **optional_event_payload(
                "observability",
                runtime_context.observability.event_payload(
                    runtime_context.observability_run,
                    lineage={"stage": WorkflowStep.FAILURE.value},
                ),
            ),
        ),
        workflow_state=final_state,
        step=WorkflowStep.FAILURE,
        phase="failed",
    )
    return {
        "workflow_state": final_state,
        "pending_failure": None,
        "failure_event_emitted": True,
        "generation_result": None,
    }

def _derive_self_evaluation_result(
    workflow_state: AssistantWorkflowState,
    *,
    generated_response: str | None = None,
):
    prompt_input = workflow_state.prompt_input
    return derive_self_evaluation_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_plan=workflow_state.creative_plan,
        creative_constraints=workflow_state.creative_constraints,
        creative_constraint_priorities=workflow_state.creative_constraint_priorities,
        creative_strategy=workflow_state.creative_strategy,
        creative_techniques=workflow_state.creative_techniques,
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
        generated_response=generated_response,
        artifacts=workflow_state.artifacts,
    )

def _derive_creative_improvement_planner_result(
    workflow_state: AssistantWorkflowState,
    *,
    generated_response: str | None = None,
):
    return derive_creative_improvement_planner_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        generated_response=generated_response,
        artifacts=workflow_state.artifacts,
    )

def _derive_reflection_loop_result(workflow_state: AssistantWorkflowState):
    return derive_reflection_loop_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )

def _derive_creative_confidence_result(workflow_state: AssistantWorkflowState):
    return derive_creative_confidence_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        reflection_loop=workflow_state.reflection_loop,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )

def _derive_creative_score_result(workflow_state: AssistantWorkflowState):
    return derive_creative_score_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )

def _derive_consistency_validation_result(workflow_state: AssistantWorkflowState):
    return derive_consistency_validation_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        creative_score=workflow_state.creative_score,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )

def _derive_evaluation_report_result(workflow_state: AssistantWorkflowState):
    return derive_evaluation_report_profile(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_critic=workflow_state.creative_critic,
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=workflow_state.creative_improvement_planner,
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        creative_score=workflow_state.creative_score,
        consistency_validation=workflow_state.consistency_validation,
        planning_metadata=_reflection_planning_metadata(workflow_state),
    )

def _reflection_planning_metadata(
    workflow_state: AssistantWorkflowState,
) -> tuple[object, ...]:
    return tuple(
        item
        for item in (
            workflow_state.creative_intent,
            workflow_state.creative_hierarchy,
            workflow_state.creative_strategy,
            workflow_state.creative_techniques,
            workflow_state.creative_plan,
            workflow_state.creative_constraints,
            workflow_state.creative_constraint_priorities,
            workflow_state.runtime_capabilities,
            workflow_state.creative_tradeoffs,
            workflow_state.creative_quality_prediction,
            workflow_state.symbolic_narrative,
            workflow_state.creative_composition,
            workflow_state.procedural_structure,
            workflow_state.generative_structure,
            workflow_state.semantic_motif,
            workflow_state.emotional_consistency,
            workflow_state.cross_modality,
            workflow_state.audio_visual_scene,
            workflow_state.artifact_plan,
            workflow_state.artifact_dependency_graph,
            workflow_state.runtime_compatibility,
            workflow_state.artifact_capability_matrix,
            workflow_state.multi_artifact_strategy,
            workflow_state.artifact_critic,
            workflow_state.artifact_refiner,
            workflow_state.artifact_intelligence_synthesis,
            workflow_state.artifact_merge_planner,
            workflow_state.artifact_export_intelligence,
        )
        if item is not None
    )
