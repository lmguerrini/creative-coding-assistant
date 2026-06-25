"""Evidence-chain construction for Creative Reasoning Engine."""

from __future__ import annotations

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
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_reasoning_contracts import (
    CreativeReasoningEvidence,
)
from creative_coding_assistant.orchestration.creative_reasoning_signals import (
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
from creative_coding_assistant.orchestration.routing import RouteDecision
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


def build_evidence_chain(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_director: CreativeAssistantDirectorBrief | None,
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
) -> tuple[CreativeReasoningEvidence, ...]:
    evidence = [
        CreativeReasoningEvidence(
            source="request",
            signal=request.query[:240],
            interpretation="The recommendation must preserve stated intent.",
        )
    ]
    if route_decision is not None:
        domains = ", ".join(item.value for item in route_decision.domains) or "none"
        evidence.append(
            CreativeReasoningEvidence(
                source="planning",
                signal=f"Route {route_decision.route.value}; domains {domains}.",
                interpretation="Reasoning must stay inside route and domain scope.",
            )
        )
    if creative_translation is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="translation",
                signal=creative_translation.creative_intent,
                interpretation="Creative translation supplies intent to protect.",
            )
        )
    if creative_intent is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="creative_intent",
                signal=creative_intent.primary_expression,
                interpretation=(
                    "Intent decomposition separates expressive dimensions "
                    "before strategy or technique decisions."
                ),
            )
        )
    if creative_hierarchy is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="creative_hierarchy",
                signal=", ".join(
                    item.dimension
                    for item in creative_hierarchy.primary_creative_priorities
                ),
                interpretation=(
                    "Hierarchy planning determines which intent dimensions "
                    "should dominate before trade-offs are explained."
                ),
            )
        )
    _append_strategy_evidence(evidence, creative_strategy)
    _append_technique_evidence(evidence, creative_techniques)
    if creative_plan is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="planning",
                signal=_clip(creative_plan.generation_strategy, 240),
                interpretation="The output goal constrains executable scope.",
            )
        )
    _append_constraint_evidence(evidence, creative_constraints)
    if creative_constraint_priorities is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="constraint_prioritizer",
                signal=", ".join(
                    item.category
                    for item in (
                        creative_constraint_priorities.non_negotiable_constraints
                        or creative_constraint_priorities.high_priority_constraints
                    )
                ),
                interpretation=(
                    "Constraint prioritization explains what to protect, "
                    "relax, or sacrifice when trade-offs tighten."
                ),
            )
        )
    if runtime_capabilities is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="runtime_capability",
                signal=", ".join(runtime_capabilities.likely_candidates),
                interpretation=(
                    "Runtime evidence informs feasibility without selecting runtime."
                ),
            )
        )
    if creative_tradeoffs is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="tradeoff_explorer",
                signal=_tradeoff_summary(creative_tradeoffs),
                interpretation="Trade-off evidence explains the bounded stance.",
            )
        )
    if creative_quality_prediction is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="quality_predictor",
                signal=(
                    f"{creative_quality_prediction.predicted_quality_level} "
                    f"readiness {creative_quality_prediction.readiness_score}/100."
                ),
                interpretation=(
                    "Quality prediction estimates pre-generation readiness "
                    "without critiquing generated artifacts."
                ),
            )
        )
    if symbolic_narrative is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="symbolic_narrative",
                signal=_clip(
                    (
                        f"{symbolic_narrative.narrative_archetype}: "
                        f"{symbolic_narrative.symbolic_arc}"
                    ),
                    240,
                ),
                interpretation=(
                    "Symbolic narrative evidence orders the experiential arc "
                    "without claiming doctrine or changing execution."
                ),
            )
        )
    if creative_composition is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="creative_composition",
                signal=_clip(
                    (
                        f"{creative_composition.composition_pattern}: "
                        f"{creative_composition.primary_focal_point}"
                    ),
                    240,
                ),
                interpretation=(
                    "Composition evidence defines focal structure, hierarchy, "
                    "density, rhythm, and spatial organization before generation."
                ),
            )
        )
    if procedural_structure is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="procedural_structure",
                signal=_clip(
                    (
                        f"{procedural_structure.primary_structure.family}: "
                        f"{procedural_structure.combination_strategy}"
                    ),
                    240,
                ),
                interpretation=(
                    "Procedural structure evidence maps intent, composition, "
                    "runtime suitability, risks, and fallbacks before generation."
                ),
            )
        )
    if generative_structure is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="generative_structure",
                signal=_clip(
                    (
                        f"{generative_structure.blueprint_name}: "
                        f"{generative_structure.generative_architecture}"
                    ),
                    240,
                ),
                interpretation=(
                    "Generative structure evidence translates procedural "
                    "metadata into modules, parameters, evolution rules, "
                    "fallbacks, and bounded prompt guidance."
                ),
            )
        )
    if semantic_motif is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="semantic_motif",
                signal=_clip(
                    (
                        f"{semantic_motif.motif_system_name}: "
                        + ", ".join(
                            motif.motif_id
                            for motif in semantic_motif.primary_motifs
                        )
                    ),
                    240,
                ),
                interpretation=(
                    "Semantic motif evidence binds narrative, composition, "
                    "structure, blueprint parameters, recurrence, and symbolic "
                    "risk guidance without asserting doctrine."
                ),
            )
        )
    if emotional_consistency is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="emotional_consistency",
                signal=_clip(
                    (
                        f"{emotional_consistency.primary_emotional_tone} "
                        f"({emotional_consistency.emotional_coherence_score}/100): "
                        + ", ".join(
                            emotional_consistency.secondary_emotional_tones[:4]
                        )
                    ),
                    240,
                ),
                interpretation=(
                    "Emotional consistency evidence aligns tone hierarchy, "
                    "narrative phases, motifs, composition, structure, "
                    "parameters, light, rhythm, and mismatch guidance as "
                    "design metadata."
                ),
            )
        )
    if cross_modality is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="cross_modality",
                signal=_clip(
                    (
                        f"{cross_modality.modality_pattern}: "
                        f"{cross_modality.primary_modality} -> "
                        + ", ".join(cross_modality.supporting_modalities[:4])
                    ),
                    240,
                ),
                interpretation=(
                    "Cross-modality evidence coordinates visual, motion, audio, "
                    "rhythm, camera, structure, motif, and emotion as design "
                    "metadata without selecting runtime or generating media."
                ),
            )
        )
    if audio_visual_scene is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="audio_visual_scene",
                signal=_clip(
                    (
                        f"{audio_visual_scene.scene_pattern}: "
                        f"{audio_visual_scene.opening_scene.title} -> "
                        f"{audio_visual_scene.climax_scene.title} -> "
                        f"{audio_visual_scene.resolution_scene.title}"
                    ),
                    240,
                ),
                interpretation=(
                    "Audio-visual scene evidence orders phases, cues, "
                    "transitions, synchronization, climax, and resolution as "
                    "design metadata without generating media."
                ),
            )
        )
    if artifact_plan is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_plan",
                signal=_clip(
                    (
                        f"{artifact_plan.artifact_type}: "
                        f"{artifact_plan.artifact_family}; "
                        f"{artifact_plan.primary_artifact_intent}"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact planning evidence defines artifact intent, "
                    "type, family, components, runtime-facing requirements, "
                    "dependencies, risks, and output structure without "
                    "selecting, critiquing, refining, or executing artifacts."
                ),
            )
        )
    if artifact_dependency_graph is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_dependency_graph",
                signal=_clip(
                    (
                        f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
                        f"{len(artifact_dependency_graph.dependency_edges)} edges; "
                        "required upstream "
                        + ", ".join(
                            artifact_dependency_graph.required_upstream_metadata
                        )
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact dependency graph evidence maps planned artifact "
                    "dependencies, runtime-facing dependencies, prompt-facing "
                    "dependencies, downstream consumers, conflicts, and missing "
                    "risks as metadata without compatibility selection or "
                    "runtime execution."
                ),
            )
        )
    if runtime_compatibility is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="runtime_compatibility",
                signal=_clip(
                    (
                        "Preferred "
                        + ", ".join(runtime_compatibility.preferred_runtimes)
                        + "; compatible "
                        + ", ".join(runtime_compatibility.compatible_runtimes)
                        + "; unsupported "
                        + ", ".join(runtime_compatibility.unsupported_runtimes)
                    ),
                    240,
                ),
                interpretation=(
                    "Runtime compatibility evidence evaluates supported "
                    "runtimes against artifact and dependency metadata as "
                    "planning guidance only, without runtime execution, "
                    "auto-selection, routing, or preview changes."
                ),
            )
        )
    if artifact_capability_matrix is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_capability_matrix",
                signal=_clip(
                    (
                        "Strongest "
                        + ", ".join(artifact_capability_matrix.strongest_targets)
                        + "; profiles "
                        + str(len(artifact_capability_matrix.capability_profiles))
                        + "; unsupported/risky "
                        + str(
                            len(
                                artifact_capability_matrix.unsupported_or_risky_capabilities
                            )
                        )
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact capability matrix evidence describes target "
                    "strengths, weaknesses, fit dimensions, unsupported "
                    "capabilities, and risks as planning guidance only, "
                    "without runtime selection, execution, routing, or preview "
                    "changes."
                ),
            )
        )
    if multi_artifact_strategy is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="multi_artifact_strategy",
                signal=_clip(
                    (
                        f"Primary "
                        f"{multi_artifact_strategy.primary_artifact.artifact_id}; "
                        f"supporting "
                        f"{len(multi_artifact_strategy.supporting_artifacts)}; "
                        f"mode {multi_artifact_strategy.combination_mode}"
                    ),
                    240,
                ),
                interpretation=(
                    "Multi-artifact strategy evidence orders, separates, "
                    "groups, combines, and hands off artifact roles as "
                    "planning guidance only, without artifact generation, "
                    "merge, refinement, export intelligence, runtime "
                    "selection, provider routing, or preview changes."
                ),
            )
        )
    if artifact_critic is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_critic",
                signal=_clip(
                    (
                        f"{artifact_critic.risk_assessment} risk; "
                        f"{artifact_critic.critique_confidence:.2f} confidence; "
                        f"{len(artifact_critic.weaknesses)} weaknesses"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact critic evidence identifies metadata strengths, "
                    "weaknesses, gaps, concerns, unsupported assumptions, and "
                    "open questions as advisory guidance only, without "
                    "modifying artifacts, rejecting strategy, refining "
                    "strategy, selecting runtime, routing providers, changing "
                    "preview behavior, or triggering retries."
                ),
            )
        )
    if artifact_refiner is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_refiner",
                signal=_clip(
                    (
                        f"{artifact_refiner.refinement_confidence:.2f} confidence; "
                        f"{len(artifact_refiner.priority_improvements)} priority; "
                        f"{len(artifact_refiner.refinement_candidates)} candidates"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact refiner evidence recommends and prioritizes "
                    "metadata-only improvement paths without modifying "
                    "artifacts, choosing final implementation, executing, "
                    "merging, exporting, selecting runtimes, routing providers, "
                    "changing previews, triggering workflows, or retrying."
                ),
            )
        )
    if artifact_intelligence_synthesis is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_intelligence_synthesis",
                signal=_clip(
                    (
                        f"{artifact_intelligence_synthesis.implementation_readiness} "
                        "readiness; "
                        f"{artifact_intelligence_synthesis.implementation_risk} "
                        "risk; "
                        f"{artifact_intelligence_synthesis.implementation_priority} "
                        "priority; "
                        f"{artifact_intelligence_synthesis.synthesis_confidence:.2f} "
                        "confidence"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact intelligence synthesis evidence summarizes, "
                    "ranks, and recommends across existing planning metadata "
                    "as advisory guidance only, without executing decisions, "
                    "selecting runtimes, modifying artifacts, routing "
                    "providers, changing previews, triggering escalation, "
                    "triggering workflows, retrying, merging, or exporting."
                ),
            )
        )
    if artifact_merge_planner is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_merge_planner",
                signal=_clip(
                    (
                        f"{artifact_merge_planner.merge_strategy}; "
                        f"{artifact_merge_planner.merge_confidence:.2f} "
                        "confidence; "
                        f"{len(artifact_merge_planner.artifact_join_points)} "
                        "join points; "
                        f"{len(artifact_merge_planner.composition_risks)} "
                        "composition risks"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact merge planner evidence recommends merge and "
                    "composition strategy, boundaries, join points, "
                    "separation points, integration order, alternatives, "
                    "rejected paths, and risks as advisory metadata only, "
                    "without merging, modifying, executing, exporting, "
                    "selecting runtime, routing providers, changing previews, "
                    "triggering workflows, retrying, or escalating."
                ),
            )
        )
    if artifact_export_intelligence is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="artifact_export_intelligence",
                signal=_clip(
                    (
                        f"{artifact_export_intelligence.export_readiness} "
                        "readiness; "
                        f"{artifact_export_intelligence.export_confidence:.2f} "
                        "confidence; "
                        f"{artifact_export_intelligence.preferred_export_target} "
                        "preferred; "
                        f"{len(artifact_export_intelligence.export_risks)} "
                        "export risks"
                    ),
                    240,
                ),
                interpretation=(
                    "Artifact export intelligence evidence recommends export "
                    "targets, requirements, constraints, risks, package notes, "
                    "documentation needs, and downstream handoffs as advisory "
                    "metadata only, without exporting, writing files, "
                    "packaging, modifying, merging, executing, selecting "
                    "runtimes, deploying, routing, previewing, triggering "
                    "workflows, retrying, or escalating."
                ),
            )
        )
    if creative_critic is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="creative_critic",
                signal=_clip(
                    (
                        f"{creative_critic.risk_assessment} risk; "
                        f"{creative_critic.critic_confidence:.2f} confidence; "
                        f"{len(creative_critic.creative_weaknesses)} weaknesses"
                    ),
                    240,
                ),
                interpretation=(
                    "Creative critic evidence evaluates strengths, "
                    "weaknesses, quality scores, risks, missing information, "
                    "assumptions, and HITL questions as advisory metadata "
                    "only, without modifying artifacts, rejecting outputs, "
                    "selecting runtimes, routing providers, changing previews, "
                    "triggering retries or refinement, runtime repair, "
                    "Studio Mode, or HoloMind."
                ),
            )
        )
    if creative_director is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="director",
                signal=creative_director.creative_brief,
                interpretation="Director guidance frames brief and HITL posture.",
            )
        )
    if artifact_export_intelligence is not None and len(evidence) > 30:
        evidence = [
            item for item in evidence if item.source != "quality_predictor"
        ]
    return tuple(evidence[:30])


def _append_strategy_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeStrategyProfile | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="creative_strategy",
            signal=f"{profile.primary_strategy} confidence {profile.confidence:.2f}.",
            interpretation=profile.rationale,
        )
    )


def _append_technique_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeTechniqueProfile | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="creative_technique",
            signal=(
                f"{profile.primary_technique} "
                f"compatibility {profile.compatibility}."
            ),
            interpretation="Technique shows how strategy becomes behavior.",
        )
    )


def _append_constraint_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeConstraintSolution | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="constraint_solver",
            signal=(
                f"complexity {profile.complexity_pressure}; "
                f"performance {profile.performance_pressure}; "
                f"safety {profile.safety_pressure}."
            ),
            interpretation="Constraint pressures bound the recommendation.",
        )
    )


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."
