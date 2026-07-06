"""Planning metadata derivation pipeline for the planning workflow node."""

from __future__ import annotations

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
from creative_coding_assistant.orchestration.runtime.nodes.planning_contracts import (
    PlanningRuntimeArtifacts,
    _evaluation_planning_metadata,
)
from creative_coding_assistant.orchestration.runtime.prompt_inputs import (
    PromptInputResponse,
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
from creative_coding_assistant.orchestration.workflow import AssistantWorkflowState


def _derive_planning_runtime_artifacts(
    workflow_state: AssistantWorkflowState,
    prompt_input: PromptInputResponse,
) -> PlanningRuntimeArtifacts:
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
    return PlanningRuntimeArtifacts(
        creative_strategy=strategy,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_techniques=techniques,
        creative_plan=plan,
        creative_constraints=constraints,
        creative_constraint_priorities=constraint_priorities,
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
        artifact_engine_contracts=artifact_engine_contracts,
        evaluation_engine_contracts=evaluation_engine_contracts_registry,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
        evaluation_report=evaluation_report,
    )
