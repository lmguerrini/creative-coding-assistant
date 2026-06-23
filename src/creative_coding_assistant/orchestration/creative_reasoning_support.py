"""Support builders for Creative Reasoning Engine synthesis."""

from __future__ import annotations

from collections.abc import Mapping

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
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
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
