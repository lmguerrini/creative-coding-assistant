"""Evidence-chain construction for Creative Reasoning Engine."""

from __future__ import annotations

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
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
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
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
    if creative_director is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="director",
                signal=creative_director.creative_brief,
                interpretation="Director guidance frames brief and HITL posture.",
            )
        )
    return tuple(evidence[:20])


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
