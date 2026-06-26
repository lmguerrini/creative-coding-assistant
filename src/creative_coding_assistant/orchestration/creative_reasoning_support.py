"""Support builders for Creative Reasoning Engine synthesis."""

from __future__ import annotations

from collections.abc import Mapping

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
    CreativeRejectedAlternative,
)
from creative_coding_assistant.orchestration.creative_reasoning_signals import (
    _top_runtime,
    _tradeoff_summary,
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


def build_strongest_signals(
    *,
    creative_director: CreativeAssistantDirectorBrief | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
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
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> tuple[str, ...]:
    signals: list[str] = []
    if creative_intent is not None:
        signals.append(f"Intent substrate: {creative_intent.primary_expression}.")
    if creative_hierarchy is not None:
        signals.append(
            "Hierarchy priorities: "
            + ", ".join(
                item.dimension
                for item in creative_hierarchy.primary_creative_priorities
            )
            + "."
        )
    if creative_strategy is not None:
        signals.append(
            f"Strategy {creative_strategy.primary_strategy} confidence "
            f"{creative_strategy.confidence:.2f}: {creative_strategy.rationale}"
        )
    if creative_techniques is not None:
        signals.append(
            f"Technique {creative_techniques.primary_technique} has "
            f"{creative_techniques.compatibility} strategy compatibility."
        )
    top = _top_runtime(runtime_capabilities)
    if top is not None:
        signals.append(
            f"Runtime capability {top.label} has {top.suitability} suitability "
            f"with confidence {top.confidence:.2f}."
        )
    if creative_tradeoffs is not None:
        signals.append(f"Primary trade-off: {_tradeoff_summary(creative_tradeoffs)}")
    if creative_quality_prediction is not None:
        signals.append(
            "Quality readiness: "
            f"{creative_quality_prediction.predicted_quality_level} "
            f"({creative_quality_prediction.readiness_score}/100)."
        )
        signals.extend(
            f"Quality signal {item.dimension}: {item.summary}"
            for item in creative_quality_prediction.strongest_quality_signals[:2]
        )
    if symbolic_narrative is not None:
        signals.append(
            "Symbolic narrative: "
            f"{symbolic_narrative.narrative_archetype}; "
            f"{symbolic_narrative.opening_phase.title} to "
            f"{symbolic_narrative.resolution_phase.title}."
        )
    if creative_composition is not None:
        signals.append(
            "Composition pattern: "
            f"{creative_composition.composition_pattern}; "
            f"{creative_composition.primary_focal_point}."
        )
    if procedural_structure is not None:
        signals.append(
            "Procedural structure: "
            f"{procedural_structure.primary_structure.family}; "
            f"{procedural_structure.combination_strategy}"
        )
    if generative_structure is not None:
        module_kinds = ", ".join(
            module.kind for module in generative_structure.procedural_modules[:4]
        )
        signals.append(
            "Generative blueprint: "
            f"{generative_structure.generative_architecture}; {module_kinds}."
        )
    if semantic_motif is not None:
        signals.append(
            "Semantic motifs: "
            + ", ".join(motif.motif_id for motif in semantic_motif.primary_motifs)
            + "."
        )
    if emotional_consistency is not None:
        signals.append(
            "Emotional consistency: "
            f"{emotional_consistency.primary_emotional_tone} "
            f"({emotional_consistency.emotional_coherence_score}/100)."
        )
    if cross_modality is not None:
        signals.append(
            "Cross-modality: "
            f"{cross_modality.primary_modality}; "
            f"{cross_modality.modality_pattern}."
        )
    if audio_visual_scene is not None:
        signals.append(
            "Audio-visual scene: "
            f"{audio_visual_scene.scene_pattern}; "
            f"{audio_visual_scene.climax_scene.title} -> "
            f"{audio_visual_scene.resolution_scene.title}."
        )
    if artifact_plan is not None:
        signals.append(
            "Artifact plan: "
            f"{artifact_plan.artifact_type}; {artifact_plan.artifact_family}; "
            f"{len(artifact_plan.required_components)} required components."
        )
    if artifact_dependency_graph is not None:
        signals.append(
            "Artifact dependency graph: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.dependency_edges)} edges; "
            f"{len(artifact_dependency_graph.blocking_dependencies)} blocking."
        )
    if runtime_compatibility is not None:
        signals.append(
            "Runtime compatibility: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + " preferred; "
            f"{len(runtime_compatibility.compatible_runtimes)} compatible; "
            f"{len(runtime_compatibility.unsupported_runtimes)} unsupported."
        )
    if artifact_capability_matrix is not None:
        signals.append(
            "Artifact capability matrix: "
            + ", ".join(artifact_capability_matrix.strongest_targets)
            + " strongest; "
            f"{len(artifact_capability_matrix.capability_profiles)} profiles; "
            f"{len(artifact_capability_matrix.unsupported_or_risky_capabilities)} "
            "unsupported/risky."
        )
    if multi_artifact_strategy is not None:
        signals.append(
            "Multi-artifact strategy: "
            f"{multi_artifact_strategy.primary_artifact.artifact_id} primary; "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting; "
            f"{multi_artifact_strategy.combination_mode}."
        )
    if artifact_critic is not None:
        signals.append(
            "Artifact critic: "
            f"{artifact_critic.risk_assessment} risk; "
            f"{artifact_critic.critique_confidence:.2f} confidence; "
            f"{len(artifact_critic.weaknesses)} weaknesses."
        )
    if artifact_refiner is not None:
        signals.append(
            "Artifact refiner: "
            f"{artifact_refiner.refinement_confidence:.2f} confidence; "
            f"{len(artifact_refiner.priority_improvements)} priority; "
            f"{len(artifact_refiner.refinement_candidates)} candidates."
        )
    if artifact_intelligence_synthesis is not None:
        signals.append(
            "Artifact intelligence synthesis: "
            f"{artifact_intelligence_synthesis.implementation_readiness} "
            "readiness; "
            f"{artifact_intelligence_synthesis.implementation_risk} risk; "
            f"{artifact_intelligence_synthesis.implementation_priority} "
            "priority."
        )
    if artifact_merge_planner is not None:
        signals.append(
            "Artifact merge planner: "
            f"{artifact_merge_planner.merge_strategy}; "
            f"{artifact_merge_planner.merge_confidence:.2f} confidence; "
            f"{len(artifact_merge_planner.artifact_join_points)} join points."
        )
    if artifact_export_intelligence is not None:
        signals.append(
            "Artifact export intelligence: "
            f"{artifact_export_intelligence.export_readiness}; "
            f"{artifact_export_intelligence.export_confidence:.2f} confidence; "
            f"{artifact_export_intelligence.preferred_export_target} preferred."
        )
    if creative_critic is not None:
        signals.append(
            "Creative critic: "
            f"{creative_critic.risk_assessment} risk; "
            f"{creative_critic.critic_confidence:.2f} confidence; "
            f"{len(creative_critic.creative_weaknesses)} weakness signals."
        )
    if self_evaluation is not None:
        signals.append(
            "Self evaluation: "
            f"{self_evaluation.completeness_assessment}; "
            f"{self_evaluation.self_evaluation_confidence:.2f} confidence; "
            f"{len(self_evaluation.quality_gaps)} quality gaps."
        )
    if creative_improvement_planner is not None:
        signals.append(
            "Creative improvement planner: "
            f"{len(creative_improvement_planner.improvement_priorities)} priorities; "
            f"{creative_improvement_planner.confidence:.2f} confidence."
        )
    if reflection_loop is not None:
        signals.append(
            "Reflection loop: "
            f"{reflection_loop.reflection_priority} priority; "
            f"{reflection_loop.reflection_depth} depth; "
            f"{reflection_loop.reflection_confidence:.2f} confidence."
        )
    if creative_confidence is not None:
        signals.append(
            "Creative confidence: "
            f"{creative_confidence.confidence_level} level; "
            f"{creative_confidence.confidence_score:.2f} score; "
            f"{creative_confidence.hitl_recommendation} HITL."
        )
    if creative_score is not None:
        signals.append(
            "Creative score: "
            f"{creative_score.score_band} band; "
            f"{creative_score.overall_creative_score:.1f}/100; "
            f"{creative_score.hitl_recommendation} HITL."
        )
    if creative_constraints is not None:
        signals.append(
            f"Constraints: complexity {creative_constraints.complexity_pressure}, "
            f"performance {creative_constraints.performance_pressure}, "
            f"safety {creative_constraints.safety_pressure}."
        )
    if creative_constraint_priorities is not None:
        signals.append(
            "Constraint priorities: "
            + ", ".join(
                item.category
                for item in (
                    creative_constraint_priorities.non_negotiable_constraints
                    or creative_constraint_priorities.high_priority_constraints
                )
            )
            + "."
        )
    if creative_director is not None:
        signals.append(
            f"Director ambiguity posture is {creative_director.ambiguity_level}."
        )
    return tuple(signals[:8]) or ("Use the user request as the strongest signal.",)


def build_rejected_alternatives(
    *,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[CreativeRejectedAlternative, ...]:
    rejected: list[CreativeRejectedAlternative] = []
    if creative_strategy is not None:
        for alternative in creative_strategy.alternative_strategies[:2]:
            rejected.append(
                CreativeRejectedAlternative(
                    alternative=f"Strategy: {alternative.strategy}",
                    reason=(
                        f"Kept secondary because {creative_strategy.primary_strategy} "
                        f"has stronger support; {alternative.rationale}"
                    ),
                    evidence=(f"alternative confidence {alternative.confidence:.2f}",),
                )
            )
    if creative_techniques is not None:
        for alternative in creative_techniques.alternative_techniques[:2]:
            rejected.append(
                CreativeRejectedAlternative(
                    alternative=f"Technique: {alternative.technique}",
                    reason=(
                        f"Deferred because {creative_techniques.primary_technique} "
                        "more directly carries the selected strategy."
                    ),
                    evidence=(alternative.rationale,),
                )
            )
    top = _top_runtime(runtime_capabilities)
    if top is not None and creative_tradeoffs is not None:
        rejected.append(
            CreativeRejectedAlternative(
                alternative="Unbounded feature expansion",
                reason=(
                    "Rejected because inspected capability and the primary "
                    "trade-off favor a bounded execution path."
                ),
                evidence=(top.risks[0], _tradeoff_summary(creative_tradeoffs)),
            )
        )
    return tuple(rejected[:6])


def build_unresolved_decisions(
    *,
    creative_director: CreativeAssistantDirectorBrief | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
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
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> tuple[str, ...]:
    unresolved: list[str] = []
    if creative_intent is not None:
        unresolved.extend(creative_intent.unresolved_intent_gaps[:3])
    if creative_hierarchy is not None:
        unresolved.extend(creative_hierarchy.priority_conflicts[:3])
    if creative_constraint_priorities is not None:
        unresolved.extend(creative_constraint_priorities.hitl_questions[:3])
        unresolved.extend(
            item.summary
            for item in creative_constraint_priorities.conflict_relationships[:2]
        )
    _append_hitl(unresolved, creative_director)
    _append_hitl(unresolved, creative_constraints)
    _append_hitl(unresolved, runtime_capabilities)
    _append_hitl(unresolved, creative_tradeoffs)
    if creative_quality_prediction is not None:
        unresolved.extend(creative_quality_prediction.hitl_questions[:3])
        unresolved.extend(creative_quality_prediction.missing_information[:2])
    if symbolic_narrative is not None:
        unresolved.extend(symbolic_narrative.hitl_questions[:3])
        unresolved.extend(symbolic_narrative.unresolved_narrative_gaps[:2])
    if creative_composition is not None:
        unresolved.extend(creative_composition.hitl_questions[:3])
        unresolved.extend(creative_composition.unresolved_composition_gaps[:2])
    if procedural_structure is not None:
        unresolved.extend(procedural_structure.hitl_questions[:3])
        unresolved.extend(procedural_structure.unresolved_procedural_gaps[:2])
    if generative_structure is not None:
        unresolved.extend(generative_structure.hitl_questions[:3])
        unresolved.extend(generative_structure.unresolved_implementation_gaps[:2])
    if semantic_motif is not None:
        unresolved.extend(semantic_motif.hitl_questions[:3])
        unresolved.extend(semantic_motif.unresolved_motif_gaps[:2])
    if emotional_consistency is not None:
        unresolved.extend(emotional_consistency.hitl_questions[:3])
        unresolved.extend(emotional_consistency.unresolved_emotional_gaps[:2])
    if cross_modality is not None:
        unresolved.extend(cross_modality.hitl_questions[:3])
        unresolved.extend(cross_modality.unresolved_modality_gaps[:2])
    if audio_visual_scene is not None:
        unresolved.extend(audio_visual_scene.hitl_questions[:3])
        unresolved.extend(audio_visual_scene.unresolved_scene_gaps[:2])
    if artifact_plan is not None:
        unresolved.extend(artifact_plan.hitl_questions[:3])
        unresolved.extend(artifact_plan.missing_information[:2])
    if artifact_dependency_graph is not None:
        unresolved.extend(artifact_dependency_graph.hitl_questions[:3])
        unresolved.extend(artifact_dependency_graph.missing_dependency_risks[:2])
        unresolved.extend(artifact_dependency_graph.dependency_conflicts[:2])
    if runtime_compatibility is not None:
        unresolved.extend(runtime_compatibility.hitl_questions[:3])
        unresolved.extend(runtime_compatibility.missing_runtime_information[:2])
        unresolved.extend(runtime_compatibility.implementation_risks[:2])
    if artifact_capability_matrix is not None:
        unresolved.extend(artifact_capability_matrix.hitl_questions[:3])
        unresolved.extend(artifact_capability_matrix.missing_capability_information[:2])
        unresolved.extend(artifact_capability_matrix.capability_risks[:2])
    if multi_artifact_strategy is not None:
        unresolved.extend(multi_artifact_strategy.hitl_questions[:3])
        unresolved.extend(multi_artifact_strategy.missing_information[:2])
        unresolved.extend(multi_artifact_strategy.risk_areas[:2])
    if artifact_critic is not None:
        unresolved.extend(artifact_critic.hitl_questions[:3])
        unresolved.extend(artifact_critic.missing_information[:2])
        unresolved.extend(artifact_critic.open_questions[:2])
    if artifact_refiner is not None:
        unresolved.extend(artifact_refiner.hitl_questions[:3])
        unresolved.extend(artifact_refiner.priority_improvements[:2])
        unresolved.extend(artifact_refiner.alternative_refinement_paths[:1])
    if artifact_intelligence_synthesis is not None:
        unresolved.extend(artifact_intelligence_synthesis.hitl_questions[:3])
        unresolved.extend(artifact_intelligence_synthesis.major_risks[:2])
        unresolved.extend(artifact_intelligence_synthesis.major_weaknesses[:1])
    if artifact_merge_planner is not None:
        unresolved.extend(artifact_merge_planner.hitl_questions[:3])
        unresolved.extend(artifact_merge_planner.composition_risks[:2])
        unresolved.extend(artifact_merge_planner.dependency_merge_risks[:1])
    if artifact_export_intelligence is not None:
        unresolved.extend(artifact_export_intelligence.hitl_questions[:3])
        unresolved.extend(artifact_export_intelligence.export_risks[:2])
        unresolved.extend(artifact_export_intelligence.export_constraints[:1])
    if creative_critic is not None:
        unresolved.extend(creative_critic.hitl_questions[:3])
        unresolved.extend(creative_critic.missing_information[:2])
        unresolved.extend(creative_critic.unsupported_assumptions[:1])
    if self_evaluation is not None:
        unresolved.extend(self_evaluation.hitl_questions[:3])
        unresolved.extend(self_evaluation.missing_information[:2])
        unresolved.extend(self_evaluation.unsupported_assumptions[:1])
    if creative_improvement_planner is not None:
        unresolved.extend(creative_improvement_planner.hitl_questions[:3])
        unresolved.extend(
            creative_improvement_planner.future_refinement_candidates[:2]
        )
    if reflection_loop is not None:
        unresolved.extend(reflection_loop.unresolved_questions[:3])
        if reflection_loop.reflection_required:
            unresolved.extend(reflection_loop.refinement_candidates[:2])
    if creative_confidence is not None:
        unresolved.extend(creative_confidence.confidence_uncertainties[:3])
        if creative_confidence.hitl_recommendation in {"recommended", "required"}:
            unresolved.append(
                "Creative Confidence recommends human review before treating confidence as settled."
            )
    if creative_score is not None:
        if creative_score.hitl_recommendation in {"recommended", "required"}:
            unresolved.append(
                "Creative Score recommends human review before treating score as settled."
            )
        unresolved.extend(creative_score.weaknesses[:2])
    if creative_strategy is not None and creative_strategy.confidence < 0.55:
        unresolved.append("Creative strategy confidence is low; confirm direction.")
    if creative_techniques is not None and creative_techniques.compatibility == "weak":
        unresolved.append("Technique compatibility is weak; confirm technique.")
    return _dedupe(unresolved)[:6] or (
        "No blocking creative decision remains unresolved in current metadata.",
    )


def build_implementation_guidance(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_techniques: CreativeTechniqueProfile | None,
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
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> tuple[str, ...]:
    guidance: list[str] = []
    if creative_intent is not None:
        guidance.extend(creative_intent.prompt_guidance[:2])
    if creative_hierarchy is not None:
        guidance.extend(creative_hierarchy.prompt_guidance[:2])
    if creative_techniques is not None:
        guidance.append(
            f"Make {creative_techniques.primary_technique} visibly serve the "
            "selected strategy before adding secondary effects."
        )
        guidance.extend(creative_techniques.implementation_notes[:2])
    if creative_plan is not None:
        guidance.extend(creative_plan.plan_steps[:2])
    if creative_constraints is not None:
        guidance.extend(creative_constraints.prompt_guidance[:2])
    if creative_constraint_priorities is not None:
        guidance.extend(creative_constraint_priorities.prompt_guidance[:2])
    top = _top_runtime(runtime_capabilities)
    if top is not None:
        guidance.append(top.prompt_guidance[0])
    if creative_tradeoffs is not None:
        guidance.append(creative_tradeoffs.primary_tradeoffs[0].mitigation)
    if creative_quality_prediction is not None:
        guidance.extend(creative_quality_prediction.prompt_guidance[:2])
    if symbolic_narrative is not None:
        guidance.extend(symbolic_narrative.prompt_guidance[:2])
        guidance.append(
            "Preserve the symbolic phase order from opening through resolution."
        )
    if creative_composition is not None:
        guidance.extend(creative_composition.prompt_guidance[:2])
        guidance.append(
            "Preserve the primary focal point and visual hierarchy before effects."
        )
    if procedural_structure is not None:
        guidance.extend(procedural_structure.prompt_guidance[:2])
        guidance.append(
            "Preserve the primary procedural family before adding secondary systems."
        )
    if generative_structure is not None:
        guidance.extend(generative_structure.prompt_guidance[:2])
        guidance.extend(generative_structure.runtime_implementation_guidance[:2])
        guidance.extend(generative_structure.performance_safeguards[:1])
        guidance.append(
            "Preserve named generative modules and parameters as metadata guidance."
        )
    if semantic_motif is not None:
        guidance.extend(semantic_motif.prompt_guidance[:2])
        guidance.extend(semantic_motif.motif_recurrence_plan[:1])
        guidance.extend(semantic_motif.motif_transformation_plan[:1])
        guidance.append(
            "Preserve primary motifs as design metaphors, not factual claims."
        )
    if emotional_consistency is not None:
        guidance.extend(emotional_consistency.prompt_guidance[:2])
        guidance.extend(emotional_consistency.color_light_guidance[:1])
        guidance.extend(emotional_consistency.motion_rhythm_guidance[:1])
        guidance.append(
            "Preserve emotional tone hierarchy as guidance, not objective truth."
        )
    if cross_modality is not None:
        guidance.extend(cross_modality.prompt_guidance[:2])
        guidance.extend(cross_modality.modality_synchronization_plan[:1])
        guidance.extend(cross_modality.contrast_balance_plan[:1])
        guidance.append(
            "Preserve cross-modality mappings as design guidance, not runtime behavior."
        )
    if audio_visual_scene is not None:
        guidance.extend(audio_visual_scene.prompt_guidance[:2])
        guidance.extend(audio_visual_scene.synchronization_checkpoints[:1])
        guidance.extend(audio_visual_scene.scene_continuity_plan[:1])
        guidance.append(
            "Preserve scene timing as guidance, not generated audio or "
            "runtime behavior."
        )
    if artifact_plan is not None:
        guidance.extend(artifact_plan.prompt_guidance[:2])
        guidance.extend(artifact_plan.expected_output_structure[:2])
        guidance.append(
            "Preserve artifact planning as shape guidance, not artifact "
            "selection or critique."
        )
    if artifact_dependency_graph is not None:
        guidance.extend(artifact_dependency_graph.prompt_guidance[:2])
        guidance.extend(artifact_dependency_graph.prompt_facing_dependencies[:2])
        guidance.extend(artifact_dependency_graph.runtime_facing_dependencies[:1])
        guidance.append(
            "Preserve artifact dependency graph metadata as dependency "
            "guidance, not runtime compatibility selection or execution."
        )
    if runtime_compatibility is not None:
        guidance.extend(runtime_compatibility.prompt_guidance[:2])
        guidance.extend(runtime_compatibility.runtime_requirements[:2])
        guidance.extend(runtime_compatibility.runtime_limitations[:1])
        guidance.append(
            "Preserve runtime compatibility as metadata only, not runtime "
            "auto-selection, provider routing, preview behavior, or execution."
        )
    if artifact_capability_matrix is not None:
        guidance.extend(artifact_capability_matrix.prompt_guidance[:2])
        guidance.extend(artifact_capability_matrix.target_strengths[:1])
        guidance.extend(artifact_capability_matrix.target_weaknesses[:1])
        guidance.append(
            "Preserve artifact capability matrix metadata as target capability "
            "guidance, not runtime auto-selection, export intelligence, "
            "provider routing, preview behavior, or execution."
        )
    if multi_artifact_strategy is not None:
        guidance.extend(multi_artifact_strategy.prompt_guidance[:2])
        guidance.extend(multi_artifact_strategy.artifact_separation_strategy[:1])
        guidance.extend(multi_artifact_strategy.artifact_combination_strategy[:1])
        guidance.append(
            "Preserve multi-artifact strategy metadata as ordering, grouping, "
            "separation, combination, dependency, and handoff guidance, not "
            "artifact generation, merge planning, export intelligence, runtime "
            "auto-selection, provider routing, preview behavior, or execution."
        )
    if artifact_critic is not None:
        guidance.extend(artifact_critic.prompt_guidance[:2])
        guidance.extend(artifact_critic.improvement_opportunities[:2])
        guidance.append(
            "Preserve Artifact Critic metadata as advisory critique only, not "
            "artifact modification, strategy rejection, refinement, runtime "
            "selection, provider routing, preview behavior, or retry behavior."
        )
    if artifact_refiner is not None:
        guidance.extend(artifact_refiner.prompt_guidance[:2])
        guidance.extend(artifact_refiner.priority_improvements[:2])
        guidance.extend(artifact_refiner.implementation_suggestions[:1])
        guidance.append(
            "Preserve Artifact Refiner metadata as advisory refinement "
            "intelligence only, not artifact modification, automatic "
            "refinement, final implementation choice, execution, merge, "
            "export, runtime selection, provider routing, preview behavior, "
            "workflow triggering, or retry behavior."
        )
    if artifact_intelligence_synthesis is not None:
        guidance.extend(artifact_intelligence_synthesis.prompt_guidance[:2])
        guidance.append(artifact_intelligence_synthesis.recommended_artifact_path)
        guidance.append(artifact_intelligence_synthesis.recommended_strategy_summary)
        guidance.append(
            "Preserve Artifact Intelligence Synthesis metadata as advisory "
            "summary, ranking, and recommendation only, not artifact "
            "modification, execution, runtime auto-selection, provider "
            "routing, preview behavior, escalation behavior, workflow "
            "triggering, retries, merging, or export."
        )
    if artifact_merge_planner is not None:
        guidance.extend(artifact_merge_planner.prompt_guidance[:2])
        guidance.append(artifact_merge_planner.recommended_merge_path)
        guidance.extend(artifact_merge_planner.artifact_separation_points[:1])
        guidance.append(
            "Preserve Artifact Merge Planner metadata as advisory "
            "composition guidance only, not artifact merging, artifact "
            "modification, final implementation choice, execution, export, "
            "runtime selection, provider routing, preview behavior, workflow "
            "triggering, retry behavior, or escalation."
        )
    if artifact_export_intelligence is not None:
        guidance.extend(artifact_export_intelligence.prompt_guidance[:2])
        guidance.append(artifact_export_intelligence.export_summary)
        guidance.extend(artifact_export_intelligence.documentation_requirements[:1])
        guidance.append(
            "Preserve Artifact Export Intelligence metadata as advisory "
            "export guidance only, not file export, file writing, package "
            "generation, artifact modification, artifact merging, final "
            "runtime choice, execution, deployment, provider routing, preview "
            "behavior, workflow triggering, retry behavior, or escalation."
        )
    if creative_critic is not None:
        guidance.extend(creative_critic.prompt_guidance[:2])
        guidance.extend(creative_critic.improvement_opportunities[:2])
        guidance.append(
            "Preserve Creative Critic metadata as advisory evaluation only, "
            "not artifact modification, output rejection, runtime selection, "
            "provider routing, preview behavior, retry behavior, refinement, "
            "runtime repair, Studio Mode, or HoloMind."
        )
    if self_evaluation is not None:
        guidance.extend(self_evaluation.prompt_guidance[:2])
        guidance.extend(self_evaluation.improvement_opportunities[:2])
        guidance.append(
            "Preserve Self Evaluation metadata as advisory assessment only, "
            "not output modification, answer rejection, runtime selection, "
            "provider routing, preview behavior, retry behavior, refinement, "
            "reflection loops, runtime repair, Studio Mode, or HoloMind."
        )
    if creative_improvement_planner is not None:
        guidance.extend(creative_improvement_planner.prompt_guidance[:2])
        guidance.extend(
            creative_improvement_planner.highest_impact_opportunities[:2]
        )
        guidance.append(
            "Preserve Creative Improvement Planner metadata as advisory "
            "improvement guidance only, not artifact edits, retries, provider "
            "routing, runtime selection, preview changes, workflow loops, or "
            "V4 agent behavior."
        )
    if reflection_loop is not None:
        guidance.extend(reflection_loop.prompt_guidance[:2])
        guidance.extend(reflection_loop.refinement_candidates[:2])
        guidance.append(
            "Preserve Reflection Loop metadata as advisory reflection value "
            "only, not automatic refinement, retries, provider calls, runtime "
            "selection, routing, preview changes, workflow loops, artifact "
            "edits, or V4 agent behavior."
        )
    if creative_confidence is not None:
        guidance.extend(creative_confidence.prompt_guidance[:2])
        guidance.extend(creative_confidence.confidence_limitations[:1])
        guidance.extend(creative_confidence.confidence_uncertainties[:1])
        guidance.append(
            "Preserve Creative Confidence metadata as advisory confidence and "
            "uncertainty context only, not output changes, artifact edits, "
            "refinement, retries, routing, runtime selection, provider calls, "
            "preview changes, or V4 agent behavior."
        )
    if creative_score is not None:
        guidance.extend(creative_score.prompt_guidance[:2])
        guidance.extend(creative_score.weaknesses[:1])
        guidance.append(
            "Preserve Creative Score metadata as advisory score context only, "
            "not output changes, artifact edits, refinement, retries, routing, "
            "runtime selection, provider calls, preview changes, V4 agents, or "
            "V5 optimization."
        )
    return _dedupe(guidance)[:8] or (
        "Implement the smallest coherent version that preserves direction.",
    )


def build_prompt_guidance(unresolved: tuple[str, ...]) -> tuple[str, ...]:
    guidance = [
        "Use the Creative Reasoning Engine recommendation as the creative spine.",
        (
            "Explain why via strategy -> technique -> runtime -> trade-off "
            "-> recommendation."
        ),
        (
            "Do not treat reasoning as artifact selection, routing, or runtime "
            "auto-selection."
        ),
    ]
    if any("No blocking" not in item for item in unresolved):
        guidance.append("Ask HITL questions before expanding unresolved scope.")
    return tuple(guidance)


def build_hitl_questions(unresolved: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"Should we resolve this before generation: {item}"
        for item in unresolved
        if "No blocking" not in item
    )[:6]


def normalize_future_knowledge_context(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    if value is not None:
        return dict(value)
    return {
        "status": "not_attached",
        "purpose": "Schema-neutral slot for future HoloMind knowledge reasoning.",
    }


def _append_hitl(unresolved: list[str], profile: object | None) -> None:
    if profile is None:
        return
    hitl_required = getattr(profile, "hitl_required", False)
    hitl_advisable = getattr(profile, "hitl_advisable", False)
    if hitl_required or hitl_advisable:
        reason = getattr(profile, "hitl_reason", None)
        unresolved.append(str(reason or "HITL input is advisable."))


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return tuple(deduped)
