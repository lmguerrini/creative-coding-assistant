"""Signal synthesis helpers for Creative Reasoning Engine."""

from __future__ import annotations

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
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
    CreativeReasoningStep,
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
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)


def build_recommended_direction(
    *,
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
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
) -> str:
    intent = (
        creative_intent.primary_expression
        if creative_intent is not None
        else _translation_or_request_intent(creative_translation, request)
    )
    output_goal = (
        creative_plan.generation_strategy
        if creative_plan is not None
        else "Produce a bounded creative-coding response."
    )
    motif_clause = (
        f"Motifs as {_clip(_motif_label(semantic_motif), 90)}. "
        if semantic_motif is not None
        else ""
    )
    emotion_clause = (
        f"Emotion as {_clip(_emotional_label(emotional_consistency), 90)}. "
        if emotional_consistency is not None
        else ""
    )
    modality_clause = (
        f"Modalities as {_clip(_modality_label(cross_modality), 90)}. "
        if cross_modality is not None
        else ""
    )
    scene_clause = (
        f"Scenes as {_clip(_scene_label(audio_visual_scene), 90)}. "
        if audio_visual_scene is not None
        else ""
    )
    artifact_clause = (
        f"Artifact as {_clip(_artifact_label(artifact_plan), 90)}. "
        if artifact_plan is not None
        else ""
    )
    dependency_clause = (
        "Dependencies as "
        f"{_clip(_artifact_dependency_label(artifact_dependency_graph), 90)}. "
        if artifact_dependency_graph is not None
        else ""
    )
    direction = (
        f"Recommend {_strategy_label(creative_strategy)} via "
        f"{_technique_label(creative_techniques)} because it protects "
        f"'{_clip(intent, 70)}'. Prioritize "
        f"{_clip(_hierarchy_label(creative_hierarchy), 70)}. "
        f"Shape symbolic arc: {_clip(_narrative_label(symbolic_narrative), 90)}. "
        f"Compose as {_clip(_composition_label(creative_composition), 90)}. "
        f"{scene_clause}"
        f"{artifact_clause}"
        f"{dependency_clause}"
        f"{motif_clause}"
        f"{emotion_clause}"
        f"{modality_clause}"
        f"Structure procedurally as "
        f"{_clip(_procedural_label(procedural_structure), 90)}. "
        f"Blueprint as {_clip(_generative_label(generative_structure), 90)}. "
        f"Protect constraints: "
        f"{_clip(_constraint_priority_label(creative_constraint_priorities), 70)}. "
        f"Fit the output goal: {_clip(output_goal, 90)} "
        f"Use inspected runtime guidance: "
        f"{_clip(_runtime_label(runtime_capabilities, creative_plan), 70)}. "
        f"Bound the trade-off: {_clip(_tradeoff_summary(creative_tradeoffs), 100)}. "
        f"Quality readiness: "
        f"{_clip(_quality_label(creative_quality_prediction), 70)}"
    )
    return _clip(direction, 500)


def build_reasoning_path(
    *,
    direction: str,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
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
) -> tuple[CreativeReasoningStep, ...]:
    strategy = _strategy_label(creative_strategy)
    technique = _technique_label(creative_techniques)
    return (
        CreativeReasoningStep(
            stage="strategy",
            claim=f"Use {strategy} as the conceptual spine.",
            because=_clip(
                (
                    _strategy_reason(
                        creative_strategy,
                        creative_intent,
                        creative_hierarchy,
                    )
                    if creative_strategy is not None
                    else "No strategy profile is available, so user intent leads."
                ),
                360,
            ),
            implications=("Keep one visible creative idea in focus.",),
        ),
        CreativeReasoningStep(
            stage="technique",
            claim=f"Translate that strategy through {technique}.",
            because=_clip(_technique_reason(creative_techniques, strategy), 360),
            implications=("Technique choices should serve the strategy.",),
        ),
        CreativeReasoningStep(
            stage="runtime",
            claim=(
                "Shape implementation around inspected capability: "
                f"{_runtime_label(runtime_capabilities, creative_plan)}."
            ),
            because=_clip(_runtime_reason(runtime_capabilities, creative_plan), 360),
            implications=("Runtime guidance remains non-binding.",),
        ),
        CreativeReasoningStep(
            stage="tradeoff",
            claim=(
                "Manage the main consequence: "
                f"{_tradeoff_summary(creative_tradeoffs)}"
            ),
            because=_clip(
                _tradeoff_reason(
                    creative_tradeoffs,
                    creative_constraint_priorities,
                ),
                360,
            ),
            implications=(
                "Prefer bounded implementation over feature growth.",
                _constraint_priority_implication(creative_constraint_priorities),
            ),
        ),
        CreativeReasoningStep(
            stage="recommendation",
            claim=direction,
            because=_clip(
                (
                    "Strategy, technique, runtime capability, and trade-off "
                    "signals converge on the same bounded direction, shaped by "
                    f"{_narrative_label(symbolic_narrative)} and "
                    f"{_composition_label(creative_composition)}, structured as "
                    f"{_procedural_label(procedural_structure)}, blueprinted as "
                    f"{_generative_label(generative_structure)}, motif-bound as "
                    f"{_motif_label(semantic_motif)}, emotionally framed as "
                    f"{_emotional_label(emotional_consistency)}, composed "
                    f"cross-modally as {_modality_label(cross_modality)}, "
                    f"scene-timed as {_scene_label(audio_visual_scene)}, with "
                    f"artifact shape {_artifact_label(artifact_plan)}, "
                    "artifact dependencies "
                    f"{_artifact_dependency_label(artifact_dependency_graph)}, and "
                    f"{_quality_label(creative_quality_prediction)}."
                ),
                360,
            ),
            implications=(
                "Use this as the prompt spine before generation.",
                (
                    "Treat procedural and generative metadata as guidance, "
                    "not code or runtime selection."
                ),
                (
                    "Treat motifs and emotion as design guidance, not doctrine "
                    "or objective truth."
                ),
                (
                    "Treat cross-modality, scene timing, and artifact planning "
                    "plus dependency graph metadata as guidance, not runtime "
                    "behavior, compatibility selection, artifact selection, "
                    "or critique."
                ),
            ),
        ),
    )


def _top_runtime(
    profile: RuntimeCapabilityProfile | None,
) -> RuntimeCapabilityCandidate | None:
    if profile is None or not profile.candidate_runtimes:
        return None
    return profile.candidate_runtimes[0]


def _strategy_label(profile: CreativeStrategyProfile | None) -> str:
    return profile.primary_strategy if profile is not None else "bounded creative"


def _translation_or_request_intent(
    creative_translation: CreativeTranslation | None,
    request: AssistantRequest,
) -> str:
    if creative_translation is not None:
        return creative_translation.creative_intent
    return request.query


def _hierarchy_label(profile: CreativeHierarchyPlan | None) -> str:
    if profile is None:
        return "the current creative hierarchy"
    return ", ".join(item.dimension for item in profile.primary_creative_priorities)


def _constraint_priority_label(
    profile: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None or not profile.non_negotiable_constraints:
        return "the current constraint priority order"
    return ", ".join(item.category for item in profile.non_negotiable_constraints)


def _constraint_priority_implication(
    profile: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None or not profile.sacrificial_constraints:
        return "Constraint priority remains advisory."
    sacrificed = ", ".join(
        item.category for item in profile.sacrificial_constraints[:2]
    )
    return f"Relax {sacrificed} before protected constraints."


def _technique_label(profile: CreativeTechniqueProfile | None) -> str:
    return profile.primary_technique if profile is not None else "minimal viable"


def _runtime_label(
    profile: RuntimeCapabilityProfile | None,
    plan: CreativeExecutionPlan | None,
) -> str:
    top = _top_runtime(profile)
    if top is not None:
        return f"{top.label} ({top.suitability} inspected fit)"
    if plan is not None and plan.recommended_runtime is not None:
        return f"{plan.recommended_runtime} from the existing execution plan"
    return "the inspected runtime capability context"


def _tradeoff_summary(profile: CreativeTradeoffProfile | None) -> str:
    if profile is None:
        return "preserve intent while keeping implementation scope bounded."
    tradeoff = profile.primary_tradeoffs[0]
    return f"{tradeoff.source_axis} vs {tradeoff.target_axis}: {tradeoff.summary}"


def _quality_label(profile: CreativeQualityPrediction | None) -> str:
    if profile is None:
        return "quality readiness not predicted"
    return (
        f"{profile.predicted_quality_level} "
        f"({profile.readiness_score}/100 readiness)"
    )


def _narrative_label(profile: SymbolicNarrativePlan | None) -> str:
    if profile is None:
        return "no symbolic narrative plan"
    return f"{profile.narrative_archetype} arc"


def _composition_label(profile: CreativeCompositionPlan | None) -> str:
    if profile is None:
        return "no composition plan"
    return f"{profile.composition_pattern} around {profile.primary_focal_point}"


def _procedural_label(profile: ProceduralStructurePlan | None) -> str:
    if profile is None:
        return "no procedural structure plan"
    return (
        f"{profile.primary_structure.family} with "
        f"{', '.join(profile.recommended_families[:3])}"
    )


def _generative_label(profile: GenerativeStructureBlueprint | None) -> str:
    if profile is None:
        return "no generative structure blueprint"
    return (
        f"{profile.generative_architecture} using "
        f"{len(profile.procedural_modules)} modules"
    )


def _motif_label(profile: SemanticMotifSystem | None) -> str:
    if profile is None:
        return "no semantic motif system"
    return ", ".join(motif.motif_id for motif in profile.primary_motifs)


def _emotional_label(profile: EmotionalConsistencyProfile | None) -> str:
    if profile is None:
        return "no emotional consistency profile"
    return (
        f"{profile.primary_emotional_tone} "
        f"({profile.emotional_coherence_score}/100)"
    )


def _modality_label(profile: CrossModalityCompositionProfile | None) -> str:
    if profile is None:
        return "no cross-modality composition profile"
    supporting = ", ".join(profile.supporting_modalities[:3])
    return (
        f"{profile.primary_modality} leading {profile.modality_pattern}; "
        f"supports {supporting}"
    )


def _scene_label(profile: AudioVisualSceneProfile | None) -> str:
    if profile is None:
        return "no audio-visual scene profile"
    return (
        f"{profile.scene_pattern} with {profile.climax_scene.title} climax "
        f"and {profile.resolution_scene.title} resolution"
    )


def _artifact_label(profile: ArtifactPlan | None) -> str:
    if profile is None:
        return "no artifact plan"
    return (
        f"{profile.artifact_type}/{profile.artifact_family} with "
        f"{len(profile.required_components)} required components"
    )


def _artifact_dependency_label(profile: ArtifactDependencyGraph | None) -> str:
    if profile is None:
        return "no artifact dependency graph"
    return (
        f"{len(profile.artifact_nodes)} nodes, "
        f"{len(profile.dependency_edges)} edges, "
        f"{len(profile.blocking_dependencies)} blocking"
    )


def _technique_reason(
    profile: CreativeTechniqueProfile | None,
    strategy: str,
) -> str:
    if profile is None:
        return "No technique profile is available, so stay minimal."
    return (
        f"{profile.rationale} This connects the technique to {strategy} "
        f"with {profile.compatibility} compatibility."
    )


def _strategy_reason(
    profile: CreativeStrategyProfile,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
) -> str:
    details: list[str] = [profile.rationale]
    if creative_intent is not None:
        details.append(
            "Decomposed intent substrate: "
            f"{_clip(creative_intent.primary_expression, 120)}."
        )
    if creative_hierarchy is not None:
        details.append(
            "Hierarchy priorities: "
            f"{_clip(_hierarchy_label(creative_hierarchy), 120)}."
        )
    return (
        " ".join(details)
    )


def _runtime_reason(
    profile: RuntimeCapabilityProfile | None,
    plan: CreativeExecutionPlan | None,
) -> str:
    top = _top_runtime(profile)
    if top is not None:
        return (
            f"{top.label} shows {top.suitability} suitability, "
            f"{top.technique_compatibility} technique compatibility, and "
            f"{top.preview_support} preview support."
        )
    if plan is not None:
        return plan.runtime_support_summary
    return "No runtime capability profile is available, so avoid runtime claims."


def _tradeoff_reason(
    profile: CreativeTradeoffProfile | None,
    priorities: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None:
        return "No trade-off profile is available; stay conservative."
    tradeoff = profile.primary_tradeoffs[0]
    reason = (
        f"The creative benefit is '{tradeoff.creative_benefit}' while the "
        f"technical cost is '{tradeoff.technical_cost}'."
    )
    if priorities is not None and priorities.negotiation_notes:
        reason += f" Constraint priority says: {priorities.negotiation_notes[0]}"
    return reason


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."
