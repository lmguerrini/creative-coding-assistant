"""Signal composition helpers for Creative Assistant Director metadata."""

from __future__ import annotations

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
)
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    ArtifactExportIntelligenceProfile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    ArtifactIntelligenceSynthesisProfile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    ArtifactMergePlannerProfile,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.artifact_refiner import (
    ArtifactRefinerProfile,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)
from creative_coding_assistant.orchestration.workflow_review import (
    WorkflowReviewOutcome,
    WorkflowReviewResult,
)


def build_director_brief_payload(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    clarification: ClarificationRequest | None,
    retrieval_chunk_count: int,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> dict[str, object]:
    retrieval_posture = _retrieval_posture(route_decision, retrieval_chunk_count)
    ambiguity_signals = _ambiguity_signals(
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_quality_prediction=creative_quality_prediction,
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
        creative_plan=creative_plan,
        clarification=clarification,
    )

    return {
        "creative_brief": _creative_brief(
            request,
            creative_intent,
            creative_translation,
        ),
        "ambiguity_level": _ambiguity_level(clarification, ambiguity_signals),
        "ambiguity_signals": ambiguity_signals,
        "retrieval_posture": retrieval_posture,
        "modality_direction": (
            creative_plan.output_modality.value if creative_plan is not None else None
        ),
        "runtime_direction": _runtime_direction(creative_plan),
        "planning_focus": _planning_focus(
            creative_plan,
            creative_intent,
            creative_hierarchy,
            creative_strategy,
            creative_techniques,
            creative_constraints,
            creative_constraint_priorities,
            runtime_capabilities,
            creative_tradeoffs,
            creative_quality_prediction,
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
        ),
        "critique_focus": _critique_focus(
            creative_plan=creative_plan,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
            creative_quality_prediction=creative_quality_prediction,
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
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
        ),
        "refinement_focus": _refinement_focus(
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
            refinement_count=refinement_count,
        ),
        "next_actions": _next_actions(
            clarification=clarification,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            creative_quality_prediction=creative_quality_prediction,
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
            review_result=review_result,
            retrieval_posture=retrieval_posture,
        ),
        "hitl_required": clarification is not None,
        "hitl_reason": clarification.summary if clarification is not None else None,
        "evidence": _evidence(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
            creative_quality_prediction=creative_quality_prediction,
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
            retrieval_chunk_count=retrieval_chunk_count,
            clarification=clarification,
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
            refinement_count=refinement_count,
        ),
    }


def _creative_brief(
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
) -> str:
    if creative_intent is not None:
        return creative_intent.primary_expression
    if creative_translation is not None:
        return creative_translation.creative_intent
    return " ".join(request.query.split())[:360]


def _ambiguity_level(
    clarification: ClarificationRequest | None,
    ambiguity_signals: tuple[str, ...],
) -> str:
    if clarification is not None:
        return "high"
    if len(ambiguity_signals) >= 2:
        return "medium"
    return "low"


def _ambiguity_signals(
    *,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    clarification: ClarificationRequest | None,
) -> tuple[str, ...]:
    signals: list[str] = []
    if clarification is not None:
        signals.append(f"Clarification required: {clarification.reason.value}.")
    if creative_intent is not None:
        signals.extend(creative_intent.unresolved_intent_gaps[:2])
    if creative_hierarchy is not None:
        signals.extend(creative_hierarchy.priority_conflicts[:2])
    if creative_constraint_priorities is not None:
        signals.extend(
            item.summary
            for item in creative_constraint_priorities.conflict_relationships[:2]
        )
    if creative_quality_prediction is not None:
        if creative_quality_prediction.predicted_quality_level in {
            "ambiguous",
            "risky",
            "blocked",
        }:
            signals.append(
                "Creative quality readiness is "
                f"{creative_quality_prediction.predicted_quality_level} "
                f"({creative_quality_prediction.readiness_score}/100)."
            )
        signals.extend(creative_quality_prediction.missing_information[:2])
    if symbolic_narrative is not None:
        signals.extend(symbolic_narrative.unresolved_narrative_gaps[:2])
    if creative_composition is not None:
        signals.extend(creative_composition.unresolved_composition_gaps[:2])
    if procedural_structure is not None:
        signals.extend(procedural_structure.unresolved_procedural_gaps[:2])
    if generative_structure is not None:
        signals.extend(generative_structure.unresolved_implementation_gaps[:2])
    if semantic_motif is not None:
        signals.extend(semantic_motif.unresolved_motif_gaps[:2])
    if emotional_consistency is not None:
        signals.extend(emotional_consistency.unresolved_emotional_gaps[:2])
    if cross_modality is not None:
        signals.extend(cross_modality.unresolved_modality_gaps[:2])
    if audio_visual_scene is not None:
        signals.extend(audio_visual_scene.unresolved_scene_gaps[:2])
    if artifact_plan is not None:
        signals.extend(artifact_plan.missing_information[:2])
    if artifact_dependency_graph is not None:
        signals.extend(artifact_dependency_graph.missing_dependency_risks[:2])
        signals.extend(artifact_dependency_graph.blocking_dependencies[:1])
    if runtime_compatibility is not None:
        signals.extend(runtime_compatibility.missing_runtime_information[:2])
        signals.extend(runtime_compatibility.implementation_risks[:1])
    if artifact_capability_matrix is not None:
        signals.extend(artifact_capability_matrix.missing_capability_information[:2])
        signals.extend(artifact_capability_matrix.capability_risks[:1])
    if multi_artifact_strategy is not None:
        signals.extend(multi_artifact_strategy.missing_information[:2])
        signals.extend(multi_artifact_strategy.risk_areas[:1])
    if artifact_critic is not None:
        signals.extend(artifact_critic.missing_information[:2])
        signals.extend(artifact_critic.weaknesses[:1])
    if artifact_refiner is not None:
        signals.extend(artifact_refiner.hitl_questions[:1])
        signals.extend(artifact_refiner.priority_improvements[:1])
    if artifact_intelligence_synthesis is not None:
        signals.extend(artifact_intelligence_synthesis.hitl_questions[:1])
        signals.extend(artifact_intelligence_synthesis.major_risks[:1])
    if artifact_merge_planner is not None:
        signals.extend(artifact_merge_planner.hitl_questions[:1])
        signals.extend(artifact_merge_planner.composition_risks[:1])
    if artifact_export_intelligence is not None:
        signals.extend(artifact_export_intelligence.hitl_questions[:1])
        signals.extend(artifact_export_intelligence.export_risks[:1])
    if route_decision is not None and len(route_decision.domains) > 1:
        signals.append("Multiple effective domains require explicit bridging.")
    if route_decision is not None and not route_decision.domains:
        signals.append("Runtime/domain direction is inferred rather than selected.")
    if creative_plan is not None and not creative_plan.runtime_available:
        signals.append("No live preview runtime is available for the selected scope.")
    return tuple(signals[:8])


def _retrieval_posture(
    route_decision: RouteDecision | None,
    retrieval_chunk_count: int,
) -> str:
    if retrieval_chunk_count > 0:
        return "available"
    if (
        route_decision is not None
        and RouteCapability.OFFICIAL_DOCS in route_decision.capabilities
    ):
        return "useful"
    return "not_requested"


def _runtime_direction(plan: CreativeExecutionPlan | None) -> str | None:
    if plan is None:
        return None
    if plan.recommended_runtime is not None:
        return plan.recommended_runtime
    return plan.runtime_support_summary


def _artifact_intelligence_synthesis_focus(
    profile: ArtifactIntelligenceSynthesisProfile,
) -> str:
    return (
        " Artifact intelligence synthesis: "
        f"{profile.implementation_readiness}; "
        f"{profile.implementation_priority} priority; metadata only."
    )


def _artifact_merge_planner_focus(
    profile: ArtifactMergePlannerProfile,
) -> str:
    return (
        " Artifact merge planner: "
        f"{profile.merge_strategy}; "
        f"{len(profile.artifact_join_points)} join points; metadata only."
    )


def _artifact_export_intelligence_focus(
    profile: ArtifactExportIntelligenceProfile,
) -> str:
    return (
        " Artifact export intelligence: "
        f"{profile.export_readiness}; "
        f"{profile.preferred_export_target}; metadata only."
    )


def _planning_focus(
    plan: CreativeExecutionPlan | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
) -> tuple[str, ...]:
    focus: list[str] = []
    if creative_intent is not None:
        focus.append(f"Intent substrate: {creative_intent.primary_expression}.")
    if artifact_plan is not None:
        focus.append(
            "Artifact plan: "
            f"{artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
        focus.extend(artifact_plan.prompt_guidance[:1])
    if artifact_dependency_graph is not None:
        focus.append(
            "Artifact dependencies: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.dependency_edges)} edges."
        )
        focus.extend(artifact_dependency_graph.prompt_guidance[:1])
    if runtime_compatibility is not None:
        runtime_focus = (
            "Runtime compatibility: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + " preferred; metadata only."
        )
        if artifact_capability_matrix is not None:
            runtime_focus += (
                " Artifact capability matrix: "
                + ", ".join(artifact_capability_matrix.strongest_targets)
                + " strongest targets; metadata only."
            )
        if multi_artifact_strategy is not None:
            runtime_focus += (
                " Multi-artifact strategy: "
                f"{len(multi_artifact_strategy.supporting_artifacts)} "
                "supporting artifacts; metadata only."
            )
        if artifact_critic is not None:
            runtime_focus += (
                " Artifact critic: "
                f"{artifact_critic.risk_assessment} risk; metadata only."
            )
        if artifact_refiner is not None:
            runtime_focus += (
                " Artifact refiner: "
                f"{len(artifact_refiner.priority_improvements)} priority "
                "improvements; metadata only."
            )
        if artifact_intelligence_synthesis is not None:
            runtime_focus += _artifact_intelligence_synthesis_focus(
                artifact_intelligence_synthesis
            )
        if artifact_merge_planner is not None:
            runtime_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            runtime_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(runtime_focus)
        if (
            artifact_capability_matrix is None
            and multi_artifact_strategy is None
            and artifact_critic is None
            and artifact_refiner is None
            and artifact_intelligence_synthesis is None
            and artifact_merge_planner is None
            and artifact_export_intelligence is None
        ):
            focus.extend(runtime_compatibility.prompt_guidance[:1])
    elif artifact_capability_matrix is not None:
        capability_focus = (
            "Artifact capability matrix: "
            + ", ".join(artifact_capability_matrix.strongest_targets)
            + " strongest targets; metadata only."
        )
        if multi_artifact_strategy is not None:
            capability_focus += (
                " Multi-artifact strategy: "
                f"{len(multi_artifact_strategy.supporting_artifacts)} "
                "supporting artifacts; metadata only."
            )
        if artifact_critic is not None:
            capability_focus += (
                " Artifact critic: "
                f"{artifact_critic.risk_assessment} risk; metadata only."
            )
        if artifact_refiner is not None:
            capability_focus += (
                " Artifact refiner: "
                f"{len(artifact_refiner.priority_improvements)} priority "
                "improvements; metadata only."
            )
        if artifact_intelligence_synthesis is not None:
            capability_focus += _artifact_intelligence_synthesis_focus(
                artifact_intelligence_synthesis
            )
        if artifact_merge_planner is not None:
            capability_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            capability_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(capability_focus)
        if (
            multi_artifact_strategy is None
            and artifact_critic is None
            and artifact_refiner is None
            and artifact_intelligence_synthesis is None
            and artifact_merge_planner is None
            and artifact_export_intelligence is None
        ):
            focus.extend(artifact_capability_matrix.prompt_guidance[:1])
    elif multi_artifact_strategy is not None:
        strategy_focus = (
            "Multi-artifact strategy: "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting "
            f"artifacts; {multi_artifact_strategy.combination_mode}; metadata only."
        )
        if artifact_critic is not None:
            strategy_focus += (
                " Artifact critic: "
                f"{artifact_critic.risk_assessment} risk; metadata only."
            )
        if artifact_refiner is not None:
            strategy_focus += (
                " Artifact refiner: "
                f"{len(artifact_refiner.priority_improvements)} priority "
                "improvements; metadata only."
            )
        if artifact_intelligence_synthesis is not None:
            strategy_focus += _artifact_intelligence_synthesis_focus(
                artifact_intelligence_synthesis
            )
        if artifact_merge_planner is not None:
            strategy_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            strategy_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(strategy_focus)
        if (
            artifact_critic is None
            and artifact_refiner is None
            and artifact_intelligence_synthesis is None
            and artifact_merge_planner is None
            and artifact_export_intelligence is None
        ):
            focus.extend(multi_artifact_strategy.prompt_guidance[:1])
    elif artifact_critic is not None:
        critic_focus = (
            "Artifact critic: "
            f"{artifact_critic.risk_assessment} risk; "
            f"{len(artifact_critic.weaknesses)} weakness signals; metadata only."
        )
        if artifact_refiner is not None:
            critic_focus += (
                " Artifact refiner: "
                f"{len(artifact_refiner.priority_improvements)} priority "
                "improvements; metadata only."
            )
        if artifact_intelligence_synthesis is not None:
            critic_focus += _artifact_intelligence_synthesis_focus(
                artifact_intelligence_synthesis
            )
        if artifact_merge_planner is not None:
            critic_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            critic_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(critic_focus)
        if (
            artifact_refiner is None
            and artifact_intelligence_synthesis is None
            and artifact_merge_planner is None
            and artifact_export_intelligence is None
        ):
            focus.extend(artifact_critic.prompt_guidance[:1])
    elif artifact_refiner is not None:
        refiner_focus = (
            "Artifact refiner: "
            f"{len(artifact_refiner.priority_improvements)} priority "
            f"improvements; {len(artifact_refiner.refinement_candidates)} "
            "candidates; metadata only."
        )
        if artifact_intelligence_synthesis is not None:
            refiner_focus += _artifact_intelligence_synthesis_focus(
                artifact_intelligence_synthesis
            )
        if artifact_merge_planner is not None:
            refiner_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            refiner_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(refiner_focus)
        if (
            artifact_intelligence_synthesis is None
            and artifact_merge_planner is None
            and artifact_export_intelligence is None
        ):
            focus.extend(artifact_refiner.prompt_guidance[:1])
    elif artifact_intelligence_synthesis is not None:
        synthesis_focus = (
            "Artifact intelligence synthesis: "
            f"{artifact_intelligence_synthesis.implementation_readiness}; "
            f"{artifact_intelligence_synthesis.implementation_priority} "
            "priority; metadata only."
        )
        if artifact_merge_planner is not None:
            synthesis_focus += _artifact_merge_planner_focus(artifact_merge_planner)
        if artifact_export_intelligence is not None:
            synthesis_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(synthesis_focus)
        if artifact_merge_planner is None and artifact_export_intelligence is None:
            focus.extend(artifact_intelligence_synthesis.prompt_guidance[:1])
    elif artifact_merge_planner is not None:
        merge_focus = (
            "Artifact merge planner: "
            f"{artifact_merge_planner.merge_strategy}; "
            f"{len(artifact_merge_planner.artifact_join_points)} join points; "
            "metadata only."
        )
        if artifact_export_intelligence is not None:
            merge_focus += _artifact_export_intelligence_focus(
                artifact_export_intelligence
            )
        focus.append(merge_focus)
        if artifact_export_intelligence is None:
            focus.extend(artifact_merge_planner.prompt_guidance[:1])
    elif artifact_export_intelligence is not None:
        focus.append(
            "Artifact export intelligence: "
            f"{artifact_export_intelligence.export_readiness}; "
            f"{artifact_export_intelligence.preferred_export_target}; "
            "metadata only."
        )
        focus.extend(artifact_export_intelligence.prompt_guidance[:1])
    if procedural_structure is not None:
        focus.append(
            "Procedural structure: "
            f"{procedural_structure.primary_structure.family}; "
            f"{procedural_structure.combination_strategy}"
        )
        if emotional_consistency is None:
            focus.extend(procedural_structure.prompt_guidance[:1])
    if generative_structure is not None:
        focus.append(
            "Generative blueprint: "
            f"{generative_structure.blueprint_name}; "
            f"{generative_structure.generative_architecture}"
        )
        if emotional_consistency is None:
            focus.extend(generative_structure.prompt_guidance[:1])
    if semantic_motif is not None:
        focus.append(
            "Semantic motifs: "
            + ", ".join(motif.motif_id for motif in semantic_motif.primary_motifs)
            + "."
        )
        if emotional_consistency is None:
            focus.extend(semantic_motif.prompt_guidance[:1])
    if emotional_consistency is not None:
        focus.append(
            "Emotional consistency: "
            f"{emotional_consistency.primary_emotional_tone}; "
            f"{emotional_consistency.emotional_coherence_score}/100."
        )
    if audio_visual_scene is not None:
        focus.append(
            "Audio-visual scene: "
            f"{audio_visual_scene.scene_pattern}; "
            f"{audio_visual_scene.climax_scene.title}."
        )
    if cross_modality is not None:
        focus.append(
            "Cross-modality: "
            f"{cross_modality.primary_modality}; "
            f"{cross_modality.modality_pattern}."
        )
    if creative_quality_prediction is not None:
        focus.append(
            "Quality readiness: "
            f"{creative_quality_prediction.predicted_quality_level} "
            f"({creative_quality_prediction.readiness_score}/100)."
        )
        focus.extend(creative_quality_prediction.prompt_guidance[:1])
    if symbolic_narrative is not None:
        focus.append(
            "Narrative arc: "
            f"{symbolic_narrative.narrative_archetype}; "
            f"{symbolic_narrative.symbolic_arc}"
        )
        focus.extend(symbolic_narrative.prompt_guidance[:1])
    if creative_composition is not None:
        focus.append(
            "Composition pattern: "
            f"{creative_composition.composition_pattern}; "
            f"{creative_composition.primary_focal_point}"
        )
        focus.extend(creative_composition.prompt_guidance[:1])
    if creative_intent is not None:
        focus.extend(creative_intent.prompt_guidance[:2])
    if creative_hierarchy is not None:
        focus.append(
            "Hierarchy priorities: "
            + ", ".join(
                item.dimension
                for item in creative_hierarchy.primary_creative_priorities[:3]
            )
            + "."
        )
        focus.extend(creative_hierarchy.prompt_guidance[:2])
    if creative_strategy is not None:
        focus.append(f"High-level strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        focus.append(f"Primary technique: {creative_techniques.primary_technique}.")
    if runtime_capabilities is not None:
        focus.append(
            "Runtime capability candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
        focus.extend(runtime_capabilities.prompt_guidance[:2])
    if creative_tradeoffs is not None:
        focus.append(
            "Trade-off discussion: " + creative_tradeoffs.director_discussion_points[0]
        )
    if creative_strategy is not None:
        focus.extend(creative_strategy.strategy_directives[:2])
    if creative_techniques is not None:
        focus.extend(creative_techniques.implementation_notes[:2])
    if creative_constraints is not None:
        focus.extend(creative_constraints.prompt_guidance[:2])
    if creative_constraint_priorities is not None:
        focus.extend(creative_constraint_priorities.prompt_guidance[:2])
    if plan is None:
        focus.extend(
            (
                "Preserve the user's creative brief as the source of truth.",
                "Keep guidance bounded to the selected route and domains.",
            )
        )
        return _dedupe_text(focus)[:6]
    focus.extend([plan.generation_strategy, *plan.plan_steps[:3]])
    if plan.constraints:
        focus.append(f"Primary constraint: {plan.constraints[0]}")
    return _dedupe_text(focus)[:6]


def _critique_focus(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
) -> tuple[str, ...]:
    focus: list[str] = []
    if creative_intent is not None:
        focus.append(
            "Check output against decomposed symbolic, emotional, formal, "
            "motion, audio, and interaction intent."
        )
    if creative_hierarchy is not None:
        focus.append(
            "Verify output protects hierarchy primary priorities before "
            "secondary dimensions."
        )
        focus.extend(creative_hierarchy.priority_conflicts[:2])
    if creative_plan is not None:
        focus.append(
            "Check output against runtime support, domain scope, and plan constraints."
        )
    if creative_strategy is not None:
        focus.append(f"Strategy rationale: {creative_strategy.rationale}")
    if creative_techniques is not None:
        focus.append(f"Technique rationale: {creative_techniques.rationale}")
    if creative_constraints is not None:
        focus.extend(creative_constraints.conflicts[:2])
        focus.extend(
            tradeoff.summary for tradeoff in creative_constraints.tradeoffs[:2]
        )
    if creative_constraint_priorities is not None:
        focus.append(
            "Verify output protects non-negotiable constraint priorities before "
            "relaxable or sacrificial constraints."
        )
        focus.extend(
            item.negotiation_note
            for item in creative_constraint_priorities.conflict_relationships[:2]
        )
    if runtime_capabilities is not None:
        focus.append(
            "Runtime capability reasoner is non-binding; verify output stays "
            "inside selected route/runtime contract."
        )
        focus.extend(runtime_capabilities.candidate_runtimes[0].risks[:2])
    if creative_tradeoffs is not None:
        focus.append(
            "Trade-off explorer is non-binding; verify output reflects "
            "declared consequences."
        )
        focus.extend(
            tradeoff.summary for tradeoff in creative_tradeoffs.primary_tradeoffs[:2]
        )
    if creative_quality_prediction is not None:
        focus.append(
            "Quality predictor is pre-generation only; compare output against "
            "predicted weak signals during normal review."
        )
        focus.extend(
            item.summary
            for item in creative_quality_prediction.weakest_quality_signals[:2]
        )
    if symbolic_narrative is not None:
        focus.append(
            "Symbolic narrative planner is pre-generation only; compare output "
            "against the declared phase arc."
        )
        focus.extend(
            f"{phase.phase} phase: {phase.title}"
            for phase in symbolic_narrative.phases[:2]
        )
    if creative_composition is not None:
        focus.append(
            "Composition planner is pre-generation only; compare output "
            "against focal structure, hierarchy, density, and rhythm."
        )
        focus.extend(creative_composition.composition_risks[:2])
    if procedural_structure is not None:
        focus.append(
            "Procedural Structure Planner is pre-generation only; compare "
            "output against primary/secondary procedural families and fallbacks."
        )
        focus.extend(procedural_structure.performance_risks[:1])
        focus.extend(procedural_structure.implementation_risks[:1])
    if generative_structure is not None:
        focus.append(
            "Generative Structure Engine is pre-generation only; compare output "
            "against blueprint modules, parameters, evolution, and safeguards."
        )
        focus.extend(generative_structure.performance_safeguards[:2])
    if semantic_motif is not None:
        focus.append(
            "Semantic Motif Engine is pre-generation only; compare output "
            "against primary motifs, recurrence, mappings, and symbolic risks."
        )
        focus.extend(semantic_motif.coherence_risks[:1])
        focus.extend(semantic_motif.overuse_risks[:1])
    if emotional_consistency is not None:
        focus.append(
            "Emotional Consistency Engine is pre-generation only; compare "
            "output against tone hierarchy, emotional arc, mappings, and "
            "mismatch risks."
        )
        focus.extend(emotional_consistency.mismatch_risks[:1])
        focus.extend(emotional_consistency.flattening_risks[:1])
    if cross_modality is not None:
        focus.append(
            "Cross-Modality Composer is pre-generation only; compare output "
            "against modality hierarchy, synchronization, conflicts, and "
            "overload risks."
        )
        focus.extend(cross_modality.modality_conflicts[:1])
        focus.extend(cross_modality.overload_risks[:1])
    if audio_visual_scene is not None:
        focus.append(
            "Audio-Visual Scene System is pre-generation only; compare output "
            "against scene phases, cues, transitions, climax, resolution, "
            "timing, and pacing risks."
        )
        focus.extend(audio_visual_scene.scene_risks[:1])
        focus.extend(audio_visual_scene.pacing_risks[:1])
    if artifact_plan is not None:
        focus.append(
            "Artifact Planner is pre-generation only; compare output against "
            "declared artifact type, family, components, runtime-facing "
            "requirements, output structure, and artifact risks."
        )
        focus.extend(artifact_plan.implementation_risks[:2])
    if artifact_dependency_graph is not None:
        focus.append(
            "Artifact Dependency Graph is pre-generation only; compare output "
            "against required edges, runtime-facing dependencies, prompt-facing "
            "dependencies, blocking risks, and downstream consumer assumptions."
        )
        focus.extend(artifact_dependency_graph.dependency_conflicts[:2])
    if runtime_compatibility is not None:
        focus.append(
            "Runtime Compatibility Engine is pre-generation only; compare "
            "output against compatible, partial, and unsupported runtime "
            "metadata without changing runtime execution."
        )
        focus.extend(runtime_compatibility.implementation_risks[:2])
    if artifact_capability_matrix is not None:
        focus.append(
            "Artifact Capability Matrix is pre-generation only; compare "
            "output against target strengths, weaknesses, fit dimensions, "
            "unsupported capabilities, and capability risks without changing "
            "runtime execution."
        )
        focus.extend(artifact_capability_matrix.capability_risks[:2])
    if multi_artifact_strategy is not None:
        focus.append(
            "Multi-Artifact Strategy is pre-generation only; compare output "
            "against primary/supporting artifact order, grouping, separation, "
            "combination, dependency order, and handoffs without generating "
            "extra artifacts."
        )
        focus.extend(multi_artifact_strategy.risk_areas[:2])
    if artifact_critic is not None:
        focus.append(
            "Artifact Critic is pre-generation metadata critique only; compare "
            "the output against critic strengths, weaknesses, gaps, concerns, "
            "risk assessment, unsupported assumptions, and open questions "
            "without modifying, rejecting, refining, merging, or executing "
            "artifacts."
        )
        focus.extend(artifact_critic.weaknesses[:2])
    if artifact_refiner is not None:
        focus.append(
            "Artifact Refiner is pre-generation metadata refinement "
            "intelligence only; compare output against recommended "
            "improvements, priority improvements, candidates, suggestions, "
            "and alternative paths without modifying, executing, merging, "
            "exporting, selecting runtimes, routing, previewing, or retrying."
        )
        focus.extend(artifact_refiner.priority_improvements[:2])
    if artifact_intelligence_synthesis is not None:
        focus.append(
            "Artifact Intelligence Synthesis is pre-generation metadata "
            "synthesis only; compare output against recommended artifact "
            "path, strategy, runtime direction, readiness, priority, "
            "strengths, weaknesses, risks, and HITL questions without "
            "executing decisions, selecting runtimes, modifying artifacts, "
            "routing, previewing, escalating, triggering workflows, retrying, "
            "merging, or exporting."
        )
        focus.extend(artifact_intelligence_synthesis.major_risks[:1])
        focus.extend(artifact_intelligence_synthesis.major_weaknesses[:1])
    if artifact_merge_planner is not None:
        focus.append(
            "Artifact Merge Planner is pre-generation metadata merge "
            "planning only; compare output against boundaries, join points, "
            "separation points, integration order, alternatives, rejected "
            "paths, risks, and HITL questions without merging, modifying, "
            "executing, exporting, selecting runtimes, routing, previewing, "
            "triggering workflows, retrying, or escalating."
        )
        focus.extend(artifact_merge_planner.composition_risks[:1])
        focus.extend(artifact_merge_planner.rejected_merge_paths[:1])
    if artifact_export_intelligence is not None:
        focus.append(
            "Artifact Export Intelligence is pre-generation metadata export "
            "planning only; compare output against targets, preferred target, "
            "format recommendations, readiness, requirements, constraints, "
            "risks, runtime notes, package notes, portability, "
            "interoperability, documentation requirements, downstream "
            "handoffs, rejected paths, and HITL questions without exporting, "
            "writing files, packaging, modifying, merging, executing, "
            "selecting runtimes, deploying, routing, previewing, triggering "
            "workflows, retrying, or escalating."
        )
        focus.extend(artifact_export_intelligence.export_risks[:1])
        focus.extend(artifact_export_intelligence.rejected_export_paths[:1])
    if artifact_critique_summary is not None:
        focus.append(
            "Recommended artifact: "
            f"{artifact_critique_summary.recommended_artifact_title or 'none'}."
        )
        focus.append(
            f"Artifact average score: {artifact_critique_summary.average_score:.2f}."
        )
    if review_result is not None:
        focus.append(f"Workflow review outcome: {review_result.outcome.value}.")
        focus.append(review_result.rationale)
    if not focus:
        focus.append("Review generated output before finalization.")
    return _dedupe_text(focus)[:6]


def _refinement_focus(
    *,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> tuple[str, ...]:
    focus: list[str] = []
    if artifact_critique_summary and artifact_critique_summary.refinement_required:
        focus.extend(artifact_critique_summary.refinement_reasons)
        if artifact_critique_summary.refinement_guidance:
            focus.append(artifact_critique_summary.refinement_guidance)
    if (
        review_result is not None
        and review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
    ):
        focus.extend(review_result.reasons)
    if refinement_count > 0:
        focus.append(f"Completed refinement pass count: {refinement_count}.")
    if not focus:
        focus.append("Use bounded refinement only when review signals a concrete gap.")
    return _dedupe_text(focus)[:6]


def _next_actions(
    *,
    clarification: ClarificationRequest | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    review_result: WorkflowReviewResult | None,
    retrieval_posture: str,
) -> tuple[str, ...]:
    if clarification is not None:
        return ("Ask the listed HITL clarification before generation.",)
    if creative_constraints is not None and creative_constraints.hitl_advisable:
        return (
            creative_constraints.hitl_reason
            or "Surface the unresolved constraint trade-off before generation.",
        )
    if (
        creative_quality_prediction is not None
        and creative_quality_prediction.hitl_questions
    ):
        return (creative_quality_prediction.hitl_questions[0],)
    if symbolic_narrative is not None and symbolic_narrative.hitl_questions:
        return (symbolic_narrative.hitl_questions[0],)
    if creative_composition is not None and creative_composition.hitl_questions:
        return (creative_composition.hitl_questions[0],)
    if procedural_structure is not None and procedural_structure.hitl_questions:
        return (procedural_structure.hitl_questions[0],)
    if generative_structure is not None and generative_structure.hitl_questions:
        return (generative_structure.hitl_questions[0],)
    if semantic_motif is not None and semantic_motif.hitl_questions:
        return (semantic_motif.hitl_questions[0],)
    if emotional_consistency is not None and emotional_consistency.hitl_questions:
        return (emotional_consistency.hitl_questions[0],)
    if cross_modality is not None and cross_modality.hitl_questions:
        return (cross_modality.hitl_questions[0],)
    if audio_visual_scene is not None and audio_visual_scene.hitl_questions:
        return (audio_visual_scene.hitl_questions[0],)
    if artifact_plan is not None and artifact_plan.hitl_questions:
        return (artifact_plan.hitl_questions[0],)
    if (
        artifact_dependency_graph is not None
        and artifact_dependency_graph.hitl_questions
    ):
        return (artifact_dependency_graph.hitl_questions[0],)
    if runtime_compatibility is not None and runtime_compatibility.hitl_questions:
        return (runtime_compatibility.hitl_questions[0],)
    if (
        artifact_capability_matrix is not None
        and artifact_capability_matrix.hitl_questions
    ):
        return (artifact_capability_matrix.hitl_questions[0],)
    if multi_artifact_strategy is not None and multi_artifact_strategy.hitl_questions:
        return (multi_artifact_strategy.hitl_questions[0],)
    if artifact_critic is not None and artifact_critic.hitl_questions:
        return (artifact_critic.hitl_questions[0],)
    if artifact_refiner is not None and artifact_refiner.hitl_questions:
        return (artifact_refiner.hitl_questions[0],)
    if (
        artifact_intelligence_synthesis is not None
        and artifact_intelligence_synthesis.hitl_questions
    ):
        return (artifact_intelligence_synthesis.hitl_questions[0],)
    if artifact_merge_planner is not None and artifact_merge_planner.hitl_questions:
        return (artifact_merge_planner.hitl_questions[0],)
    if (
        artifact_export_intelligence is not None
        and artifact_export_intelligence.hitl_questions
    ):
        return (artifact_export_intelligence.hitl_questions[0],)
    if (
        review_result is not None
        and review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
    ):
        return ("Prepare bounded refinement guidance for the next generation pass.",)
    actions = [
        "Render the prompt and continue through the deterministic workflow."
        if creative_plan is not None
        else "Continue with the available workflow context.",
    ]
    if retrieval_posture in {"available", "useful"}:
        actions.append("Use official KB context when it is available and relevant.")
    actions.append("Keep final creative choices visible to the user.")
    return tuple(actions[:6])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    retrieval_chunk_count: int,
    clarification: ClarificationRequest | None,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
        domains = route_decision.domains or _request_domains(request)
        if domains:
            evidence.append(
                "Domains: " + ", ".join(domain.value for domain in domains) + "."
            )
    if creative_translation is not None:
        evidence.append(f"Creative intent: {creative_translation.creative_intent}.")
    if creative_intent is not None:
        evidence.append(f"Intent gaps: {len(creative_intent.unresolved_intent_gaps)}.")
    if creative_hierarchy is not None:
        evidence.append(
            f"Hierarchy confidence: {creative_hierarchy.hierarchy_confidence:.2f}."
        )
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
        evidence.append(f"Strategy confidence: {creative_strategy.confidence:.2f}.")
    if creative_techniques is not None:
        evidence.append(f"Creative technique: {creative_techniques.primary_technique}.")
        evidence.append(f"Technique confidence: {creative_techniques.confidence:.2f}.")
    if creative_plan is not None:
        evidence.append(f"Plan complexity: {creative_plan.expected_complexity}.")
        evidence.append(f"Export readiness: {creative_plan.export_readiness}.")
    if creative_constraints is not None:
        evidence.append(
            "Constraint solver: "
            f"{len(creative_constraints.active_constraints)} active constraint(s)."
        )
        evidence.append(f"Runtime fit: {creative_constraints.runtime_fit}.")
    if creative_constraint_priorities is not None:
        evidence.append(
            "Constraint prioritizer: "
            f"{len(creative_constraint_priorities.non_negotiable_constraints)} "
            "non-negotiable constraint(s)."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Runtime capability candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
        evidence.append(
            f"Runtime capability HITL: {runtime_capabilities.hitl_advisable}."
        )
    if creative_tradeoffs is not None:
        evidence.append(
            f"Trade-offs: {len(creative_tradeoffs.primary_tradeoffs)} primary."
        )
        evidence.append(f"Trade-off HITL: {creative_tradeoffs.hitl_advisable}.")
    if creative_quality_prediction is not None:
        evidence.append(
            "Quality prediction: "
            f"{creative_quality_prediction.predicted_quality_level} "
            f"({creative_quality_prediction.readiness_score}/100)."
        )
    if symbolic_narrative is not None:
        evidence.append(
            "Symbolic narrative: "
            f"{symbolic_narrative.narrative_archetype}."
        )
    if creative_composition is not None:
        evidence.append(
            "Creative composition: "
            f"{creative_composition.composition_pattern}."
        )
    if procedural_structure is not None:
        evidence.append(
            "Procedural structure: "
            f"{procedural_structure.primary_structure.family}."
        )
    if generative_structure is not None:
        evidence.append(
            "Generative structure: "
            f"{generative_structure.generative_architecture}."
        )
    if semantic_motif is not None:
        evidence.append(
            "Semantic motifs: "
            + ", ".join(motif.motif_id for motif in semantic_motif.primary_motifs)
            + "."
        )
    if emotional_consistency is not None:
        evidence.append(
            "Emotional consistency: "
            f"{emotional_consistency.primary_emotional_tone} "
            f"({emotional_consistency.emotional_coherence_score}/100)."
        )
    if cross_modality is not None:
        evidence.append(
            "Cross-modality: "
            f"{cross_modality.primary_modality}; "
            f"{cross_modality.modality_pattern}."
        )
    if audio_visual_scene is not None:
        evidence.append(
            "Audio-visual scene: "
            f"{audio_visual_scene.scene_pattern}; "
            f"{len(audio_visual_scene.scene_phases)} phases."
        )
    if artifact_plan is not None:
        evidence.append(
            "Artifact plan: "
            f"{artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if artifact_dependency_graph is not None:
        evidence.append(
            "Artifact dependency graph: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.dependency_edges)} edges."
        )
    if runtime_compatibility is not None:
        evidence.append(
            "Runtime compatibility: "
            f"{len(runtime_compatibility.compatible_runtimes)} compatible; "
            f"{len(runtime_compatibility.unsupported_runtimes)} unsupported."
        )
    if artifact_capability_matrix is not None:
        evidence.append(
            "Artifact capability matrix: "
            f"{len(artifact_capability_matrix.capability_profiles)} profiles; "
            f"{len(artifact_capability_matrix.unsupported_or_risky_capabilities)} "
            "unsupported/risky capabilities."
        )
    if multi_artifact_strategy is not None:
        evidence.append(
            "Multi-artifact strategy: "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting; "
            f"{len(multi_artifact_strategy.artifact_sequence)} sequence steps."
        )
    if artifact_critic is not None:
        evidence.append(
            "Artifact critic: "
            f"{artifact_critic.risk_assessment} risk; "
            f"{artifact_critic.critique_confidence:.2f} confidence."
        )
    if artifact_refiner is not None:
        evidence.append(
            "Artifact refiner: "
            f"{artifact_refiner.refinement_confidence:.2f} confidence; "
            f"{len(artifact_refiner.priority_improvements)} priority."
        )
    if artifact_intelligence_synthesis is not None:
        evidence.append(
            "Artifact intelligence synthesis: "
            f"{artifact_intelligence_synthesis.implementation_readiness} "
            "readiness; "
            f"{artifact_intelligence_synthesis.implementation_risk} risk; "
            f"{artifact_intelligence_synthesis.synthesis_confidence:.2f} "
            "confidence."
        )
    if artifact_merge_planner is not None:
        evidence.append(
            "Artifact merge planner: "
            f"{artifact_merge_planner.merge_strategy}; "
            f"{artifact_merge_planner.merge_confidence:.2f} confidence; "
            f"{len(artifact_merge_planner.artifact_join_points)} join points."
        )
    if artifact_export_intelligence is not None:
        evidence.append(
            "Artifact export intelligence: "
            f"{artifact_export_intelligence.export_readiness}; "
            f"{artifact_export_intelligence.export_confidence:.2f} confidence; "
            f"{artifact_export_intelligence.preferred_export_target} preferred."
        )
    if retrieval_chunk_count:
        evidence.append(f"Retrieval chunks: {retrieval_chunk_count}.")
    if clarification is not None:
        evidence.append(f"HITL reason: {clarification.reason.value}.")
    if artifact_critique_summary is not None:
        evidence.append(
            f"Artifact critique average: {artifact_critique_summary.average_score:.2f}."
        )
    if review_result is not None:
        evidence.append(f"Review outcome: {review_result.outcome.value}.")
    if refinement_count:
        evidence.append(f"Refinement count: {refinement_count}.")
    return _dedupe_text(evidence)[:10]


def _request_domains(
    request: AssistantRequest,
) -> tuple[CreativeCodingDomain, ...]:
    if request.domains:
        return request.domains
    if request.domain is not None:
        return (request.domain,)
    return ()


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)
