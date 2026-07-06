"""Planning, director, and reasoning node handlers."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration._metadata_utils import (
    PlanningMetadata,
    PlanningMetadataItem,
)
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    derive_artifact_capability_matrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    derive_artifact_critic_profile,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    derive_artifact_dependency_graph,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    artifact_intelligence_engine_contracts,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    derive_artifact_export_intelligence_profile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    derive_artifact_intelligence_synthesis_profile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    derive_artifact_merge_planner_profile,
)
from creative_coding_assistant.orchestration.artifact_planner import derive_artifact_plan
from creative_coding_assistant.orchestration.artifact_refiner import (
    derive_artifact_refiner_profile,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    derive_audio_visual_scene_profile,
)
from creative_coding_assistant.orchestration.consistency_validation_engine import (
    derive_consistency_validation_profile,
)
from creative_coding_assistant.orchestration.creative_composition import (
    derive_creative_composition_plan,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    derive_creative_confidence_profile,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    derive_creative_constraint_priorities,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    derive_creative_constraint_solution,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    derive_creative_critic_profile,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
    derive_creative_assistant_director_brief,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    derive_creative_hierarchy_plan,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    derive_creative_improvement_planner_profile,
)
from creative_coding_assistant.orchestration.creative_intent import (
    derive_creative_intent_decomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    derive_creative_execution_plan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    derive_creative_quality_prediction,
)
from creative_coding_assistant.orchestration.creative_reasoning import (
    CreativeReasoningResult,
    derive_creative_reasoning_result,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    derive_creative_score_profile,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    derive_creative_strategy_profile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    derive_creative_technique_profile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    derive_creative_tradeoff_profile,
)
from creative_coding_assistant.orchestration.cross_modality import (
    derive_cross_modality_composition_profile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    derive_emotional_consistency_profile,
)
from creative_coding_assistant.orchestration.evaluation_engine_contracts import (
    evaluation_engine_contracts,
)
from creative_coding_assistant.orchestration.evaluation_reports import (
    derive_evaluation_report_profile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    derive_generative_structure_blueprint,
)
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    derive_multi_artifact_strategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    derive_procedural_structure_plan,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    derive_reflection_loop_profile,
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
from creative_coding_assistant.orchestration.runtime_capabilities import (
    derive_runtime_capability_profile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    derive_runtime_compatibility_profile,
)
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    derive_self_evaluation_profile,
)
from creative_coding_assistant.orchestration.semantic_motif import (
    derive_semantic_motif_system,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    derive_symbolic_narrative_plan,
)
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowStep,
)


def _evaluation_planning_metadata(
    *items: PlanningMetadataItem,
) -> PlanningMetadata:
    return items

def _planning_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PLANNING,
    )
    try:
        prompt_input = workflow_state.prompt_input
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PLANNING,
                    decision_reason="prompt_input_unavailable_for_planning",
                )
            }

        retrieval_chunk_count = (
            len(prompt_input.retrieval_input.chunks)
            if prompt_input.retrieval_input is not None
            else 0
        )
        creative_intent = derive_creative_intent_decomposition(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
        )
        creative_hierarchy = derive_creative_hierarchy_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_translation=prompt_input.creative_translation,
        )
        strategy = derive_creative_strategy_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
        )
        techniques = derive_creative_technique_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )
        plan = derive_creative_execution_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            retrieval_chunk_count=retrieval_chunk_count,
        )
        constraints = derive_creative_constraint_solution(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
            creative_techniques=techniques,
            clarification=workflow_state.clarification,
            retrieval_chunk_count=retrieval_chunk_count,
        )
        runtime_capabilities = derive_runtime_capability_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_plan=plan,
            creative_constraints=constraints,
        )
        tradeoffs = derive_creative_tradeoff_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_plan=plan,
            creative_constraints=constraints,
            runtime_capabilities=runtime_capabilities,
        )
        constraint_priorities = derive_creative_constraint_priorities(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
        )
        quality_prediction = derive_creative_quality_prediction(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
        )
        symbolic_narrative = derive_symbolic_narrative_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
        )
        creative_composition = derive_creative_composition_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
        )
        procedural_structure = derive_procedural_structure_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
        )
        generative_structure = derive_generative_structure_blueprint(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
        )
        semantic_motif = derive_semantic_motif_system(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
        )
        emotional_consistency = derive_emotional_consistency_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
        )
        cross_modality = derive_cross_modality_composition_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
        )
        audio_visual_scene = derive_audio_visual_scene_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
        )
        artifact_plan = derive_artifact_plan(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
        )
        artifact_dependency_graph = derive_artifact_dependency_graph(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
        )
        runtime_compatibility = derive_runtime_compatibility_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_tradeoffs=tradeoffs,
        )
        artifact_capability_matrix = derive_artifact_capability_matrix(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_tradeoffs=tradeoffs,
        )
        multi_artifact_strategy = derive_multi_artifact_strategy(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_tradeoffs=tradeoffs,
        )
        artifact_critic = derive_artifact_critic_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
        )
        artifact_refiner = derive_artifact_refiner_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
        )
        artifact_intelligence_synthesis = (
            derive_artifact_intelligence_synthesis_profile(
                request=workflow_state.request,
                route_decision=workflow_state.route_decision,
                artifact_plan=artifact_plan,
                artifact_dependency_graph=artifact_dependency_graph,
                runtime_compatibility=runtime_compatibility,
                artifact_capability_matrix=artifact_capability_matrix,
                multi_artifact_strategy=multi_artifact_strategy,
                artifact_critic=artifact_critic,
                artifact_refiner=artifact_refiner,
            )
        )
        artifact_merge_planner = derive_artifact_merge_planner_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        )
        artifact_export_intelligence = derive_artifact_export_intelligence_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
        )
        artifact_engine_contracts = artifact_intelligence_engine_contracts()
        evaluation_engine_contracts_registry = evaluation_engine_contracts()
        creative_critic = derive_creative_critic_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            artifact_export_intelligence=artifact_export_intelligence,
        )
        self_evaluation = derive_self_evaluation_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_translation=prompt_input.creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=plan,
            creative_constraints=constraints,
            creative_constraint_priorities=constraint_priorities,
            creative_strategy=strategy,
            creative_techniques=techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=tradeoffs,
            creative_quality_prediction=quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            artifact_export_intelligence=artifact_export_intelligence,
            creative_critic=creative_critic,
        )
        creative_improvement_planner = derive_creative_improvement_planner_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
        )
        planning_metadata = _evaluation_planning_metadata(
            creative_intent,
            creative_hierarchy,
            strategy,
            techniques,
            plan,
            constraints,
            constraint_priorities,
            runtime_capabilities,
            tradeoffs,
            quality_prediction,
            symbolic_narrative,
            creative_composition,
            procedural_structure,
            generative_structure,
            semantic_motif,
            emotional_consistency,
            cross_modality,
            audio_visual_scene,
            artifact_plan,
            artifact_dependency_graph,
            runtime_compatibility,
            artifact_capability_matrix,
            multi_artifact_strategy,
            artifact_critic,
            artifact_refiner,
            artifact_intelligence_synthesis,
            artifact_merge_planner,
            artifact_export_intelligence,
        )
        reflection_loop = derive_reflection_loop_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            planning_metadata=planning_metadata,
        )
        creative_confidence = derive_creative_confidence_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            planning_metadata=planning_metadata,
        )
        creative_score = derive_creative_score_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            planning_metadata=planning_metadata,
        )
        consistency_validation = derive_consistency_validation_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
            planning_metadata=planning_metadata,
        )
        evaluation_report = derive_evaluation_report_profile(
            request=workflow_state.request,
            route_decision=workflow_state.route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
            consistency_validation=consistency_validation,
            planning_metadata=planning_metadata,
        )
        planned_prompt_input = prompt_input.model_copy(
            update={
                "creative_strategy": strategy,
                "creative_intent": creative_intent,
                "creative_hierarchy": creative_hierarchy,
                "creative_techniques": techniques,
                "creative_plan": plan,
                "creative_constraints": constraints,
                "creative_constraint_priorities": constraint_priorities,
                "runtime_capabilities": runtime_capabilities,
                "creative_tradeoffs": tradeoffs,
                "creative_quality_prediction": quality_prediction,
                "symbolic_narrative": symbolic_narrative,
                "creative_composition": creative_composition,
                "procedural_structure": procedural_structure,
                "generative_structure": generative_structure,
                "semantic_motif": semantic_motif,
                "emotional_consistency": emotional_consistency,
                "cross_modality": cross_modality,
                "audio_visual_scene": audio_visual_scene,
                "artifact_plan": artifact_plan,
                "artifact_dependency_graph": artifact_dependency_graph,
                "runtime_compatibility": runtime_compatibility,
                "artifact_capability_matrix": artifact_capability_matrix,
                "multi_artifact_strategy": multi_artifact_strategy,
                "artifact_critic": artifact_critic,
                "artifact_refiner": artifact_refiner,
                "artifact_intelligence_synthesis": (artifact_intelligence_synthesis),
                "artifact_merge_planner": artifact_merge_planner,
                "artifact_export_intelligence": artifact_export_intelligence,
                "artifact_engine_contracts": artifact_engine_contracts,
                "evaluation_engine_contracts": (evaluation_engine_contracts_registry),
                "creative_critic": creative_critic,
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": creative_improvement_planner,
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
                "creative_score": creative_score,
                "consistency_validation": consistency_validation,
                "evaluation_report": evaluation_report,
            }
        )
        planned_state = workflow_state.model_copy(
            update={
                "creative_strategy": strategy,
                "creative_intent": creative_intent,
                "creative_hierarchy": creative_hierarchy,
                "creative_techniques": techniques,
                "creative_plan": plan,
                "creative_constraints": constraints,
                "creative_constraint_priorities": constraint_priorities,
                "runtime_capabilities": runtime_capabilities,
                "creative_tradeoffs": tradeoffs,
                "creative_quality_prediction": quality_prediction,
                "symbolic_narrative": symbolic_narrative,
                "creative_composition": creative_composition,
                "procedural_structure": procedural_structure,
                "generative_structure": generative_structure,
                "semantic_motif": semantic_motif,
                "emotional_consistency": emotional_consistency,
                "cross_modality": cross_modality,
                "audio_visual_scene": audio_visual_scene,
                "artifact_plan": artifact_plan,
                "artifact_dependency_graph": artifact_dependency_graph,
                "runtime_compatibility": runtime_compatibility,
                "artifact_capability_matrix": artifact_capability_matrix,
                "multi_artifact_strategy": multi_artifact_strategy,
                "artifact_critic": artifact_critic,
                "artifact_refiner": artifact_refiner,
                "artifact_intelligence_synthesis": (artifact_intelligence_synthesis),
                "artifact_merge_planner": artifact_merge_planner,
                "artifact_export_intelligence": artifact_export_intelligence,
                "artifact_engine_contracts": artifact_engine_contracts,
                "evaluation_engine_contracts": (evaluation_engine_contracts_registry),
                "creative_critic": creative_critic,
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": (creative_improvement_planner),
                "reflection_loop": reflection_loop,
                "creative_confidence": creative_confidence,
                "creative_score": creative_score,
                "consistency_validation": consistency_validation,
                "evaluation_report": evaluation_report,
                "prompt_input": planned_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_plan_prepared",
                message="Creative execution plan prepared.",
                creative_intent=creative_intent.model_dump(mode="json"),
                creative_hierarchy=creative_hierarchy.model_dump(mode="json"),
                creative_strategy=strategy.model_dump(mode="json"),
                creative_techniques=techniques.model_dump(mode="json"),
                creative_plan=plan.model_dump(mode="json"),
                creative_constraints=constraints.model_dump(mode="json"),
                creative_constraint_priorities=constraint_priorities.model_dump(
                    mode="json"
                ),
                runtime_capabilities=runtime_capabilities.model_dump(mode="json"),
                creative_tradeoffs=tradeoffs.model_dump(mode="json"),
                creative_quality_prediction=quality_prediction.model_dump(mode="json"),
                symbolic_narrative=symbolic_narrative.model_dump(mode="json"),
                creative_composition=creative_composition.model_dump(mode="json"),
                procedural_structure=procedural_structure.model_dump(mode="json"),
                generative_structure=generative_structure.model_dump(mode="json"),
                semantic_motif=semantic_motif.model_dump(mode="json"),
                emotional_consistency=emotional_consistency.model_dump(mode="json"),
                cross_modality=cross_modality.model_dump(mode="json"),
                audio_visual_scene=audio_visual_scene.model_dump(mode="json"),
                artifact_plan=artifact_plan.model_dump(mode="json"),
                artifact_dependency_graph=artifact_dependency_graph.model_dump(
                    mode="json"
                ),
                runtime_compatibility=runtime_compatibility.model_dump(mode="json"),
                artifact_capability_matrix=artifact_capability_matrix.model_dump(
                    mode="json"
                ),
                multi_artifact_strategy=multi_artifact_strategy.model_dump(mode="json"),
                artifact_critic=artifact_critic.model_dump(mode="json"),
                artifact_refiner=artifact_refiner.model_dump(mode="json"),
                artifact_intelligence_synthesis=(
                    artifact_intelligence_synthesis.model_dump(mode="json")
                ),
                artifact_merge_planner=artifact_merge_planner.model_dump(mode="json"),
                artifact_export_intelligence=(
                    artifact_export_intelligence.model_dump(mode="json")
                ),
                artifact_engine_contracts=artifact_engine_contracts.model_dump(
                    mode="json"
                ),
                evaluation_engine_contracts=(
                    evaluation_engine_contracts_registry.model_dump(mode="json")
                ),
                creative_critic=creative_critic.model_dump(mode="json"),
                self_evaluation=self_evaluation.model_dump(mode="json"),
                creative_improvement_planner=(
                    creative_improvement_planner.model_dump(mode="json")
                ),
                reflection_loop=reflection_loop.model_dump(mode="json"),
                creative_confidence=creative_confidence.model_dump(mode="json"),
                creative_score=creative_score.model_dump(mode="json"),
                consistency_validation=consistency_validation.model_dump(mode="json"),
                evaluation_report=evaluation_report.model_dump(mode="json"),
            ),
            workflow_state=planned_state,
            step=WorkflowStep.PLANNING,
        )
        return {
            "workflow_state": _complete_node(
                planned_state,
                runtime_context,
                WorkflowStep.PLANNING,
                decision_reason="creative_plan_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PLANNING,
            exc=exc,
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

def _reasoning_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.REASONING,
    )
    try:
        if (
            workflow_state.prompt_input is None
            or workflow_state.creative_director is None
        ):
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.REASONING,
                    decision_reason="reasoning_inputs_unavailable",
                )
            }

        reasoning = _derive_reasoning_result(workflow_state)
        reasoned_prompt_input = workflow_state.prompt_input.model_copy(
            update={"creative_reasoning": reasoning}
        )
        reasoned_state = workflow_state.model_copy(
            update={
                "creative_reasoning": reasoning,
                "prompt_input": reasoned_prompt_input,
            }
        )
        _emit(
            runtime_context.event_builder.planning(
                code="creative_reasoning_prepared",
                message="Creative Reasoning Engine synthesis prepared.",
                creative_reasoning=reasoning.model_dump(mode="json"),
            ),
            workflow_state=reasoned_state,
            step=WorkflowStep.REASONING,
        )
        return {
            "workflow_state": _complete_node(
                reasoned_state,
                runtime_context,
                WorkflowStep.REASONING,
                decision_reason="creative_reasoning_prepared",
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.REASONING,
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

def _derive_reasoning_result(
    workflow_state: AssistantWorkflowState,
) -> CreativeReasoningResult:
    prompt_input = workflow_state.prompt_input
    return derive_creative_reasoning_result(
        request=workflow_state.request,
        route_decision=workflow_state.route_decision,
        creative_translation=(
            prompt_input.creative_translation if prompt_input is not None else None
        ),
        creative_intent=workflow_state.creative_intent,
        creative_hierarchy=workflow_state.creative_hierarchy,
        creative_plan=workflow_state.creative_plan,
        creative_director=workflow_state.creative_director,
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
        self_evaluation=workflow_state.self_evaluation,
        creative_improvement_planner=(workflow_state.creative_improvement_planner),
        reflection_loop=workflow_state.reflection_loop,
        creative_confidence=workflow_state.creative_confidence,
        creative_score=workflow_state.creative_score,
        consistency_validation=workflow_state.consistency_validation,
        evaluation_report=workflow_state.evaluation_report,
        evaluation_engine_contracts=workflow_state.evaluation_engine_contracts,
    )
