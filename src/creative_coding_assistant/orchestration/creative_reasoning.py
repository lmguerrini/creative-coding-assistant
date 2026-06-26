"""Bounded Creative Reasoning Engine public API."""

from __future__ import annotations

from collections.abc import Mapping

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
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
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    CreativeConfidenceProfile,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
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
from creative_coding_assistant.orchestration.creative_score_engine import (
    CreativeScoreProfile,
)
from creative_coding_assistant.orchestration.creative_reasoning_contracts import (
    CREATIVE_REASONING_AUTHORITY_BOUNDARY,
    CreativeReasoningEvidence,
    CreativeReasoningResult,
    CreativeReasoningStep,
    CreativeRejectedAlternative,
)
from creative_coding_assistant.orchestration.creative_reasoning_evidence import (
    build_evidence_chain,
)
from creative_coding_assistant.orchestration.creative_reasoning_signals import (
    build_reasoning_path,
    build_recommended_direction,
)
from creative_coding_assistant.orchestration.creative_reasoning_support import (
    build_hitl_questions,
    build_implementation_guidance,
    build_prompt_guidance,
    build_rejected_alternatives,
    build_strongest_signals,
    build_unresolved_decisions,
    normalize_future_knowledge_context,
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
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)


def derive_creative_reasoning_result(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_director: CreativeAssistantDirectorBrief | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    emotional_consistency: EmotionalConsistencyProfile | None = None,
    cross_modality: CrossModalityCompositionProfile | None = None,
    audio_visual_scene: AudioVisualSceneProfile | None = None,
    artifact_plan: ArtifactPlan | None = None,
    artifact_dependency_graph: ArtifactDependencyGraph | None = None,
    runtime_compatibility: RuntimeCompatibilityProfile | None = None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None,
    multi_artifact_strategy: MultiArtifactStrategy | None = None,
    artifact_critic: ArtifactCriticProfile | None = None,
    artifact_refiner: ArtifactRefinerProfile | None = None,
    artifact_intelligence_synthesis: (
        ArtifactIntelligenceSynthesisProfile | None
    ) = None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None = None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None = None,
    creative_critic: CreativeCriticProfile | None = None,
    self_evaluation: SelfEvaluationProfile | None = None,
    creative_improvement_planner: (
        CreativeImprovementPlannerProfile | None
    ) = None,
    reflection_loop: ReflectionLoopProfile | None = None,
    creative_confidence: CreativeConfidenceProfile | None = None,
    creative_score: CreativeScoreProfile | None = None,
    future_knowledge_context: Mapping[str, object] | None = None,
) -> CreativeReasoningResult:
    """Synthesize prior Creative Intelligence outputs into one decision brief."""

    direction = build_recommended_direction(
        request=request,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        creative_plan=creative_plan,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
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
        creative_critic=creative_critic,
    )
    unresolved = build_unresolved_decisions(
        creative_director=creative_director,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
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
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
    )
    return CreativeReasoningResult(
        recommended_creative_direction=direction,
        reasoning_path=build_reasoning_path(
            direction=direction,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
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
            creative_critic=creative_critic,
        ),
        evidence_chain=build_evidence_chain(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=creative_plan,
            creative_director=creative_director,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
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
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
        ),
        strongest_supporting_signals=build_strongest_signals(
            creative_director=creative_director,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
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
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
        ),
        rejected_alternatives=build_rejected_alternatives(
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        unresolved_decisions=unresolved,
        implementation_guidance=build_implementation_guidance(
            creative_plan=creative_plan,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            creative_techniques=creative_techniques,
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
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
        ),
        prompt_guidance=build_prompt_guidance(unresolved),
        hitl_questions=build_hitl_questions(unresolved),
        future_knowledge_context=normalize_future_knowledge_context(
            future_knowledge_context
        ),
    )


def creative_reasoning_prompt_lines(
    result: CreativeReasoningResult,
) -> tuple[str, ...]:
    """Render the reasoning brief as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {result.authority_boundary}",
        f"Reasoning recommendation: {result.recommended_creative_direction}",
    ]
    for step in result.reasoning_path:
        lines.append(
            f"Reasoning path ({step.stage}): {step.claim} Because {step.because}"
        )
        lines.extend(f"Reasoning implication: {item}" for item in step.implications[:2])
    lines.extend(
        f"Supporting signal: {item}"
        for item in result.strongest_supporting_signals[:5]
    )
    lines.extend(
        f"Rejected alternative: {item.alternative}; {item.reason}"
        for item in result.rejected_alternatives[:3]
    )
    lines.extend(
        f"Unresolved decision: {item}" for item in result.unresolved_decisions[:4]
    )
    lines.extend(
        f"Implementation guidance: {item}"
        for item in result.implementation_guidance[:5]
    )
    lines.extend(f"Prompt guidance: {item}" for item in result.prompt_guidance[:5])
    lines.extend(f"HITL question: {item}" for item in result.hitl_questions[:3])
    return tuple(lines[:32])


__all__ = [
    "CREATIVE_REASONING_AUTHORITY_BOUNDARY",
    "CreativeReasoningEvidence",
    "CreativeReasoningResult",
    "CreativeReasoningStep",
    "CreativeRejectedAlternative",
    "creative_reasoning_prompt_lines",
    "derive_creative_reasoning_result",
]
