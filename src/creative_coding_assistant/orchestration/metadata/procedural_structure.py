"""Bounded Procedural Structure Planner for V3.2 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

ProceduralFamily = Literal[
    "fractals",
    "recursive_geometry",
    "l_systems",
    "particle_systems",
    "boids",
    "cellular_automata",
    "reaction_diffusion",
    "voronoi_systems",
    "noise_fields",
    "flow_fields",
    "signed_distance_fields",
    "polar_radial_systems",
    "grid_systems",
    "graph_network_systems",
    "swarm_systems",
    "wave_systems",
    "harmonic_oscillators",
    "modular_tiling",
    "sacred_geometry_pattern_systems",
]
ProceduralComplexityLevel = Literal["low", "medium", "high"]

PROCEDURAL_STRUCTURE_AUTHORITY_BOUNDARY = (
    "The Procedural Structure Planner recommends inspectable procedural "
    "families and structural guidance only; it does not generate code, "
    "implement the Generative Structure Engine, auto-select runtimes, route "
    "providers or models, change preview behavior, run repair loops, or "
    "implement HoloMind."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_AUDIO_TOKENS = frozenset(
    {
        "audio",
        "audiovisual",
        "beat",
        "music",
        "pulse",
        "rhythm",
        "sound",
        "tempo",
        "tone",
    }
)
_INTERACTION_TOKENS = frozenset(
    {
        "click",
        "drag",
        "gesture",
        "interactive",
        "mouse",
        "scroll",
        "touch",
        "user",
    }
)
_AMBIGUITY_TOKENS = frozenset(
    {
        "cool",
        "deep",
        "maybe",
        "mood",
        "profound",
        "something",
        "unspecified",
        "vibe",
    }
)
_HIGH_DENSITY_FAMILIES = frozenset(
    {
        "particle_systems",
        "boids",
        "reaction_diffusion",
        "flow_fields",
        "signed_distance_fields",
        "swarm_systems",
    }
)


class ProceduralStructureChoice(BaseModel):
    """One recommended procedural structure family."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    family: ProceduralFamily
    label: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class ProceduralStructurePlan(BaseModel):
    """Inspectable pre-generation procedural structure metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["procedural_structure_planner"] = "procedural_structure_planner"
    recommended_families: tuple[ProceduralFamily, ...] = Field(
        min_length=1,
        max_length=5,
    )
    primary_structure: ProceduralStructureChoice
    secondary_structures: tuple[ProceduralStructureChoice, ...] = Field(
        min_length=1,
        max_length=4,
    )
    combination_strategy: str = Field(min_length=1, max_length=360)
    spatial_structure_plan: str = Field(min_length=1, max_length=360)
    temporal_structure_plan: str = Field(min_length=1, max_length=360)
    interaction_structure_plan: str | None = Field(default=None, max_length=320)
    audiovisual_structure_plan: str | None = Field(default=None, max_length=320)
    complexity_level: ProceduralComplexityLevel
    runtime_suitability_notes: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    performance_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    implementation_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    fallback_structure_options: tuple[ProceduralStructureChoice, ...] = Field(
        min_length=1,
        max_length=4,
    )
    unresolved_procedural_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=PROCEDURAL_STRUCTURE_AUTHORITY_BOUNDARY,
        max_length=560,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _ProceduralContext:
    request: AssistantRequest
    route_decision: RouteDecision | None
    creative_translation: CreativeTranslation | None
    creative_intent: CreativeIntentDecomposition | None
    creative_hierarchy: CreativeHierarchyPlan | None
    creative_plan: CreativeExecutionPlan | None
    creative_constraints: CreativeConstraintSolution | None
    creative_constraint_priorities: CreativeConstraintPrioritization | None
    creative_strategy: CreativeStrategyProfile | None
    creative_techniques: CreativeTechniqueProfile | None
    runtime_capabilities: RuntimeCapabilityProfile | None
    creative_tradeoffs: CreativeTradeoffProfile | None
    creative_quality_prediction: CreativeQualityPrediction | None
    symbolic_narrative: SymbolicNarrativePlan | None
    creative_composition: CreativeCompositionPlan | None
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class _FamilySpec:
    family: ProceduralFamily
    label: str
    keywords: tuple[str, ...]
    spatial_plan: str
    temporal_plan: str
    complexity: ProceduralComplexityLevel
    performance_risks: tuple[str, ...]
    implementation_risks: tuple[str, ...]
    fallback_families: tuple[ProceduralFamily, ...]


@dataclass(frozen=True)
class _ScoredFamily:
    spec: _FamilySpec
    score: int
    direct_matches: tuple[str, ...]
    evidence: tuple[str, ...]


def derive_procedural_structure_plan(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
) -> ProceduralStructurePlan:
    """Plan procedural structure without generating code or selecting runtime."""

    context = _context(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
    )
    scored = _score_families(context)
    primary = scored[0]
    secondary = tuple(
        _choice_from_scored(item, context, secondary=True)
        for item in scored[1:5]
        if item.score > 0 or item.direct_matches
    )
    if not secondary:
        secondary = (_choice_from_spec(_SPEC_BY_FAMILY["modular_tiling"], context),)
    unresolved = _unresolved_gaps(context, primary=primary)
    return ProceduralStructurePlan(
        recommended_families=tuple(item.spec.family for item in scored[:5]),
        primary_structure=_choice_from_scored(primary, context, secondary=False),
        secondary_structures=secondary,
        combination_strategy=_combination_strategy(primary.spec, secondary, context),
        spatial_structure_plan=_spatial_structure_plan(primary.spec, context),
        temporal_structure_plan=_temporal_structure_plan(primary.spec, context),
        interaction_structure_plan=_interaction_structure_plan(context, primary.spec),
        audiovisual_structure_plan=_audiovisual_structure_plan(context, primary.spec),
        complexity_level=_complexity_level(primary.spec, context),
        runtime_suitability_notes=_runtime_suitability_notes(
            context,
            primary=primary.spec,
        ),
        performance_risks=_performance_risks(context, primary.spec, secondary),
        implementation_risks=_implementation_risks(context, primary.spec, secondary),
        fallback_structure_options=_fallback_options(primary.spec, context),
        unresolved_procedural_gaps=unresolved,
        hitl_questions=_hitl_questions(unresolved),
        prompt_guidance=_prompt_guidance(primary.spec, unresolved),
        evidence=_evidence(context, primary=primary, scored=scored),
    )


def procedural_structure_prompt_lines(
    plan: ProceduralStructurePlan,
) -> tuple[str, ...]:
    """Render procedural metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {plan.authority_boundary}",
        "Recommended procedural families: "
        + ", ".join(plan.recommended_families)
        + ".",
        (
            "Primary procedural structure: "
            f"{plan.primary_structure.label} ({plan.primary_structure.family}). "
            f"{plan.primary_structure.rationale}"
        ),
        f"Combination strategy: {plan.combination_strategy}",
        f"Spatial structure plan: {plan.spatial_structure_plan}",
        f"Temporal structure plan: {plan.temporal_structure_plan}",
        f"Procedural complexity: {plan.complexity_level}.",
    ]
    if plan.interaction_structure_plan is not None:
        lines.append(f"Interaction structure plan: {plan.interaction_structure_plan}")
    if plan.audiovisual_structure_plan is not None:
        lines.append(f"Audiovisual structure plan: {plan.audiovisual_structure_plan}")
    lines.extend(
        "Secondary procedural structure: "
        f"{item.label} ({item.family}). {item.rationale}"
        for item in plan.secondary_structures
    )
    lines.extend(
        f"Runtime suitability note: {item}" for item in plan.runtime_suitability_notes
    )
    lines.extend(f"Performance risk: {item}" for item in plan.performance_risks)
    lines.extend(f"Implementation risk: {item}" for item in plan.implementation_risks)
    lines.extend(
        f"Fallback procedural structure: {item.label} ({item.family}). {item.rationale}"
        for item in plan.fallback_structure_options
    )
    lines.extend(
        f"Unresolved procedural gap: {item}" for item in plan.unresolved_procedural_gaps
    )
    lines.extend(f"HITL procedural question: {item}" for item in plan.hitl_questions)
    lines.extend(f"Procedural guidance: {item}" for item in plan.prompt_guidance)
    return tuple(lines[:40])


def _context(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
) -> _ProceduralContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
        symbolic_narrative.narrative_archetype if symbolic_narrative else "",
        symbolic_narrative.symbolic_arc if symbolic_narrative else "",
        creative_composition.composition_pattern if creative_composition else "",
        (
            creative_composition.primary_focal_point
            if creative_composition is not None
            else ""
        ),
    ]
    if creative_translation is not None:
        parts.extend(creative_translation.geometric_references)
        parts.extend(creative_translation.movement_language)
        parts.extend(creative_translation.structure_direction)
        parts.extend(creative_translation.runtime_recommendations)
    if creative_hierarchy is not None:
        parts.extend(
            item.dimension for item in creative_hierarchy.primary_creative_priorities
        )
    if runtime_capabilities is not None:
        parts.extend(runtime_capabilities.likely_candidates)
        parts.extend(runtime_capabilities.prompt_guidance)
    if creative_quality_prediction is not None:
        parts.extend(creative_quality_prediction.missing_information)
    if creative_composition is not None:
        parts.extend(creative_composition.transition_guidance)
    text = _normalize(" ".join(parts))
    return _ProceduralContext(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        text=text,
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _score_families(context: _ProceduralContext) -> tuple[_ScoredFamily, ...]:
    scored = [
        _score_family(spec, context=context, order=order)
        for order, spec in enumerate(_FAMILY_SPECS)
    ]
    scored.sort(
        key=lambda item: (
            item.score,
            len(item.direct_matches),
            -_SPEC_ORDER[item.spec.family],
        ),
        reverse=True,
    )
    return tuple(scored)


def _score_family(
    spec: _FamilySpec,
    *,
    context: _ProceduralContext,
    order: int,
) -> _ScoredFamily:
    del order
    direct_matches = tuple(
        keyword for keyword in spec.keywords if _matches(context, keyword)
    )
    evidence: list[str] = []
    score = len(direct_matches) * 3
    if direct_matches:
        evidence.append("Matched request terms: " + ", ".join(direct_matches[:5]) + ".")
    score += _hint_bonus(
        spec.family,
        context.creative_strategy.primary_strategy
        if context.creative_strategy is not None
        else None,
        _STRATEGY_FAMILY_HINTS,
        "strategy",
        evidence,
    )
    score += _hint_bonus(
        spec.family,
        context.creative_techniques.primary_technique
        if context.creative_techniques is not None
        else None,
        _TECHNIQUE_FAMILY_HINTS,
        "technique",
        evidence,
    )
    score += _hint_bonus(
        spec.family,
        context.symbolic_narrative.narrative_archetype
        if context.symbolic_narrative is not None
        else None,
        _NARRATIVE_FAMILY_HINTS,
        "narrative",
        evidence,
    )
    score += _hint_bonus(
        spec.family,
        context.creative_composition.composition_pattern
        if context.creative_composition is not None
        else None,
        _COMPOSITION_FAMILY_HINTS,
        "composition",
        evidence,
    )
    if _audio_relevant(context) and spec.family in {
        "wave_systems",
        "harmonic_oscillators",
        "particle_systems",
        "flow_fields",
    }:
        score += 2
        evidence.append("Audio/rhythm signals support this family.")
    if _interaction_relevant(context) and spec.family in {
        "particle_systems",
        "boids",
        "graph_network_systems",
        "flow_fields",
    }:
        score += 2
        evidence.append("Interaction signals support this family.")
    if score == 0 and spec.family == "grid_systems":
        score = 1
        evidence.append("Default bounded fallback when structure is underspecified.")
    return _ScoredFamily(
        spec=spec,
        score=score,
        direct_matches=direct_matches,
        evidence=tuple(evidence[:8]),
    )


def _hint_bonus(
    family: ProceduralFamily,
    key: str | None,
    hints: dict[str, tuple[ProceduralFamily, ...]],
    source: str,
    evidence: list[str],
) -> int:
    if key is None:
        return 0
    preferred = hints.get(key, ())
    if family not in preferred:
        return 0
    evidence.append(f"{source} signal favors {family}.")
    return 4 if preferred and preferred[0] == family else 2


def _choice_from_scored(
    scored: _ScoredFamily,
    context: _ProceduralContext,
    *,
    secondary: bool,
) -> ProceduralStructureChoice:
    rationale = (
        _secondary_rationale(scored, context)
        if secondary
        else _primary_rationale(scored, context)
    )
    return ProceduralStructureChoice(
        family=scored.spec.family,
        label=scored.spec.label,
        rationale=rationale,
        evidence=scored.evidence or (f"Selected score {scored.score}.",),
    )


def _choice_from_spec(
    spec: _FamilySpec,
    context: _ProceduralContext,
) -> ProceduralStructureChoice:
    return ProceduralStructureChoice(
        family=spec.family,
        label=spec.label,
        rationale=(
            f"Use {spec.label.lower()} as a lower-risk procedural fallback "
            f"that still supports {_subject_label(context)}."
        ),
        evidence=("Fallback from procedural structure planner.",),
    )


def _primary_rationale(
    scored: _ScoredFamily,
    context: _ProceduralContext,
) -> str:
    return (
        f"{scored.spec.label} best matches {_subject_label(context)} with "
        f"score {scored.score}; it aligns spatial organization, temporal "
        "behavior, and inspected creative constraints before generation."
    )


def _secondary_rationale(
    scored: _ScoredFamily,
    context: _ProceduralContext,
) -> str:
    return (
        f"Use {scored.spec.label.lower()} as a supporting layer when it "
        f"strengthens {_subject_label(context)} without overtaking the primary "
        "procedural spine."
    )


def _combination_strategy(
    primary: _FamilySpec,
    secondary: tuple[ProceduralStructureChoice, ...],
    context: _ProceduralContext,
) -> str:
    supporting = ", ".join(item.label for item in secondary[:2])
    base = (
        f"Lead with {primary.label.lower()} as the structural spine and use "
        f"{supporting} as bounded secondary systems."
    )
    if context.creative_constraint_priorities is not None:
        protected = context.creative_constraint_priorities.non_negotiable_constraints
        if protected:
            return (
                f"{base} Do not trade away {protected[0].category} to add "
                "more procedural layers."
            )
    return base


def _spatial_structure_plan(
    primary: _FamilySpec,
    context: _ProceduralContext,
) -> str:
    if context.creative_composition is not None:
        return (
            f"{primary.spatial_plan} Fit it inside "
            f"{context.creative_composition.composition_pattern} around "
            f"{context.creative_composition.primary_focal_point}"
        )
    return primary.spatial_plan


def _temporal_structure_plan(
    primary: _FamilySpec,
    context: _ProceduralContext,
) -> str:
    if context.symbolic_narrative is not None:
        return (
            f"{primary.temporal_plan} Progress through "
            f"{context.symbolic_narrative.narrative_archetype} phases without "
            "adding autonomous loops."
        )
    return primary.temporal_plan


def _interaction_structure_plan(
    context: _ProceduralContext,
    primary: _FamilySpec,
) -> str | None:
    if not _interaction_relevant(context):
        return None
    if _has_any(context, {"mouse", "drag", "touch", "gesture", "click"}):
        return (
            f"Map direct interaction to parameters of {primary.label.lower()} "
            "such as density, origin, threshold, or local force."
        )
    return (
        f"Interaction is relevant but underspecified; expose only one bounded "
        f"control over {primary.label.lower()} until HITL resolves the gesture."
    )


def _audiovisual_structure_plan(
    context: _ProceduralContext,
    primary: _FamilySpec,
) -> str | None:
    if not _audio_relevant(context):
        return None
    return (
        f"Map audio or rhythm to procedural parameters of {primary.label.lower()} "
        "such as phase, amplitude, density, radius, or damping, not to a new "
        "runtime feature."
    )


def _complexity_level(
    primary: _FamilySpec,
    context: _ProceduralContext,
) -> ProceduralComplexityLevel:
    if (
        context.creative_constraints is not None
        and context.creative_constraints.complexity_pressure == "high"
    ):
        return "high"
    if (
        context.creative_techniques is not None
        and context.creative_techniques.complexity_pressure == "high"
    ):
        return "high"
    if primary.complexity == "high":
        return "high"
    if (
        context.creative_tradeoffs is not None
        and context.creative_tradeoffs.complexity_risks
    ):
        return "medium"
    return primary.complexity


def _runtime_suitability_notes(
    context: _ProceduralContext,
    *,
    primary: _FamilySpec,
) -> tuple[str, ...]:
    notes: list[str] = []
    if context.runtime_capabilities is not None:
        labels = ", ".join(context.runtime_capabilities.likely_candidates)
        notes.append(
            "Use inspected runtime candidates as non-binding feasibility notes: "
            f"{labels}."
        )
        top = context.runtime_capabilities.candidate_runtimes[0]
        notes.append(
            f"{top.label} has {top.suitability} suitability and "
            f"{top.preview_support} preview support for bounded generation."
        )
    elif (
        context.creative_plan is not None and context.creative_plan.recommended_runtime
    ):
        notes.append(
            "Existing execution plan names "
            f"{context.creative_plan.recommended_runtime}; treat that as "
            "context, not runtime auto-selection."
        )
    else:
        notes.append(
            "No runtime capability profile is attached; avoid runtime-specific "
            "procedural claims."
        )
    if primary.family in _HIGH_DENSITY_FAMILIES:
        notes.append(
            f"{primary.label} can increase draw/update cost; cap agent, field, "
            "or iteration counts for browser runtimes."
        )
    if _has_any(context, {"p5", "p5.js", "canvas", "svg"}):
        notes.append(
            "p5.js, canvas, or SVG contexts favor clear 2D procedural systems "
            "over heavyweight volumetric simulation."
        )
    return _dedupe(notes)[:8]


def _performance_risks(
    context: _ProceduralContext,
    primary: _FamilySpec,
    secondary: tuple[ProceduralStructureChoice, ...],
) -> tuple[str, ...]:
    risks = list(primary.performance_risks)
    if any(item.family in _HIGH_DENSITY_FAMILIES for item in secondary):
        risks.append(
            "Secondary high-density structures can multiply update cost if "
            "combined naively."
        )
    if context.creative_tradeoffs is not None:
        risks.extend(context.creative_tradeoffs.performance_concerns[:3])
    if (
        context.creative_constraints is not None
        and context.creative_constraints.performance_pressure == "high"
    ):
        risks.append("Constraint solver marks performance pressure high.")
    return _dedupe(risks)[:8]


def _implementation_risks(
    context: _ProceduralContext,
    primary: _FamilySpec,
    secondary: tuple[ProceduralStructureChoice, ...],
) -> tuple[str, ...]:
    risks = list(primary.implementation_risks)
    if len(secondary) >= 3:
        risks.append("Too many procedural layers can obscure the primary structure.")
    if context.creative_tradeoffs is not None:
        risks.extend(context.creative_tradeoffs.complexity_risks[:3])
        risks.extend(context.creative_tradeoffs.maintainability_concerns[:2])
    if context.creative_quality_prediction is not None:
        risks.extend(context.creative_quality_prediction.likely_failure_modes[:2])
    return _dedupe(risks)[:8]


def _fallback_options(
    primary: _FamilySpec,
    context: _ProceduralContext,
) -> tuple[ProceduralStructureChoice, ...]:
    fallbacks = [
        _choice_from_spec(_SPEC_BY_FAMILY[family], context)
        for family in primary.fallback_families
    ]
    if not fallbacks:
        fallbacks.append(_choice_from_spec(_SPEC_BY_FAMILY["grid_systems"], context))
    return tuple(fallbacks[:4])


def _unresolved_gaps(
    context: _ProceduralContext,
    *,
    primary: _ScoredFamily,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if not primary.direct_matches:
        gaps.append("Procedural family is inferred rather than explicit.")
    if _has_any(context, _AMBIGUITY_TOKENS):
        gaps.append("Request contains abstract intent but few procedural cues.")
    if _interaction_relevant(context) and not _has_any(
        context,
        {"click", "drag", "gesture", "mouse", "touch"},
    ):
        gaps.append("Interaction is relevant but the controlling gesture is unclear.")
    if _audio_relevant(context) and not _has_any(
        context,
        {"beat", "pulse", "rhythm", "tempo"},
    ):
        gaps.append("Audio-reactive structure is relevant but timing is unclear.")
    if context.creative_quality_prediction is not None:
        gaps.extend(
            item
            for item in context.creative_quality_prediction.missing_information[:3]
            if "runtime" in item.lower()
            or "interaction" in item.lower()
            or "visual" in item.lower()
            or "motion" in item.lower()
        )
    if context.creative_composition is not None:
        gaps.extend(context.creative_composition.unresolved_composition_gaps[:2])
    return _dedupe(gaps)[:8]


def _hitl_questions(gaps: tuple[str, ...]) -> tuple[str, ...]:
    questions: list[str] = []
    for gap in gaps:
        lowered = gap.lower()
        if "family" in lowered or "procedural" in lowered:
            questions.append(
                "Should the structure lean recursive, particle-based, field-based, "
                "grid/tiling-based, radial, or network-based?"
            )
        elif "gesture" in lowered or "interaction" in lowered:
            questions.append("What user gesture should control the structure?")
        elif "audio" in lowered or "timing" in lowered:
            questions.append("What beat, pulse, or tempo should drive the structure?")
        elif "visual" in lowered or "motif" in lowered:
            questions.append("What visible motif should the procedure preserve?")
        elif "motion" in lowered:
            questions.append("Should motion emerge, loop, pulse, flow, or stabilize?")
    return _dedupe(questions)[:6]


def _prompt_guidance(
    primary: _FamilySpec,
    unresolved: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        (
            "Use procedural structure guidance as an implementation spine, "
            "not as code generation."
        ),
        (
            f"Make {primary.label.lower()} visible before adding secondary "
            "procedural systems."
        ),
        (
            "Keep procedural parameters inspectable: count, scale, phase, "
            "radius, force, or threshold."
        ),
        "Prefer bounded loops and caps over autonomous repair or open-ended evolution.",
    ]
    if unresolved:
        guidance.append(
            "Ask HITL questions before over-specifying procedural structure."
        )
    return tuple(guidance[:8])


def _evidence(
    context: _ProceduralContext,
    *,
    primary: _ScoredFamily,
    scored: tuple[_ScoredFamily, ...],
) -> tuple[str, ...]:
    evidence = [
        f"Primary procedural family: {primary.spec.family}.",
        f"Primary score: {primary.score}.",
        "Recommended families: "
        + ", ".join(item.spec.family for item in scored[:5])
        + ".",
    ]
    if primary.direct_matches:
        evidence.append(
            "Direct matches: " + ", ".join(primary.direct_matches[:6]) + "."
        )
    if context.creative_strategy is not None:
        evidence.append(
            f"Strategy source: {context.creative_strategy.primary_strategy}."
        )
    if context.creative_techniques is not None:
        evidence.append(
            f"Technique source: {context.creative_techniques.primary_technique}."
        )
    if context.runtime_capabilities is not None:
        evidence.append(
            "Runtime capability candidates: "
            + ", ".join(context.runtime_capabilities.likely_candidates)
            + "."
        )
    if context.creative_composition is not None:
        evidence.append(
            f"Composition source: {context.creative_composition.composition_pattern}."
        )
    return tuple(evidence[:12])


def _audio_relevant(context: _ProceduralContext) -> bool:
    if _has_any(context, _AUDIO_TOKENS):
        return True
    if context.creative_plan is not None:
        if context.creative_plan.output_modality.value in {"audio", "audiovisual"}:
            return True
    if context.creative_intent is not None:
        return (
            context.creative_intent.audio_intent.explicitness != "absent"
            or context.creative_intent.rhythm_intent.explicitness != "absent"
        )
    return False


def _interaction_relevant(context: _ProceduralContext) -> bool:
    if _has_any(context, _INTERACTION_TOKENS):
        return True
    if context.creative_plan is not None:
        if context.creative_plan.output_modality.value == "interactive":
            return True
    if context.creative_intent is not None:
        return context.creative_intent.interaction_intent.explicitness != "absent"
    return False


def _subject_label(context: _ProceduralContext) -> str:
    if context.creative_intent is not None:
        return context.creative_intent.primary_expression
    if context.creative_translation is not None:
        return context.creative_translation.creative_intent
    return "the user's stated creative goal"


def _matches(context: _ProceduralContext, keyword: str) -> bool:
    normalized = _normalize(keyword)
    if " " in normalized:
        return normalized in context.text
    return normalized in context.tokens


def _has_any(context: _ProceduralContext, values: frozenset[str] | set[str]) -> bool:
    return bool(context.tokens.intersection(values))


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


_FAMILY_SPECS: tuple[_FamilySpec, ...] = (
    _FamilySpec(
        "fractals",
        "Fractals",
        ("fractal", "self similar", "branching", "fern", "tree", "growth"),
        (
            "Use repeated self-similar forms across scale levels with visible "
            "depth limits."
        ),
        (
            "Reveal recursion gradually through controlled iteration depth or "
            "scale changes."
        ),
        "high",
        ("High recursion depth can exceed frame budgets in browser sketches.",),
        ("Without depth caps, fractal systems can become visually dense and brittle.",),
        ("recursive_geometry", "modular_tiling", "grid_systems"),
    ),
    _FamilySpec(
        "recursive_geometry",
        "Recursive Geometry",
        ("recursive", "recursion", "nested", "spiral", "branch", "subdivision"),
        (
            "Build geometry by repeatedly transforming a simple form with "
            "explicit depth caps."
        ),
        "Animate recursive depth, rotation, scale, or reveal order over time.",
        "medium",
        ("Nested drawing can become expensive if each level adds many children.",),
        ("Recursive transforms need clear stopping rules and readable parameters.",),
        ("polar_radial_systems", "fractals", "modular_tiling"),
    ),
    _FamilySpec(
        "l_systems",
        "L-systems",
        ("l-system", "lsystem", "grammar", "turtle", "plant", "branching"),
        (
            "Use grammar-like rewrite rules to grow branching structures from "
            "a seed string."
        ),
        "Step through rewrite generations rather than generating all depth at once.",
        "high",
        ("Grammar expansion can grow exponentially without generation limits.",),
        ("Rule readability can suffer if symbolic growth is overcomplicated.",),
        ("recursive_geometry", "fractals", "grid_systems"),
    ),
    _FamilySpec(
        "particle_systems",
        "Particle Systems",
        (
            "particle",
            "particles",
            "spark",
            "sparks",
            "dust",
            "ember",
            "embers",
            "dissolve",
            "dissolves",
            "reform",
            "reforms",
            "phoenix",
        ),
        (
            "Represent the concept as many bounded agents with position, "
            "velocity, and lifespan."
        ),
        "Use birth, drift, attraction, decay, and reassembly phases.",
        "medium",
        ("Particle counts, trails, and blending can pressure frame rate.",),
        ("Particles need explicit lifecycle rules or they become decorative noise.",),
        ("noise_fields", "flow_fields", "grid_systems"),
    ),
    _FamilySpec(
        "boids",
        "Boids",
        ("boid", "flock", "flocking", "alignment", "separation", "cohesion"),
        "Structure motion around local alignment, cohesion, and separation rules.",
        "Let flock behavior shift between gathering, dispersal, and stable orbit.",
        "high",
        ("Neighbor checks can become expensive without spatial partitioning or caps.",),
        ("Emergent flocking can obscure symbolic intent if forces are not bounded.",),
        ("swarm_systems", "particle_systems", "flow_fields"),
    ),
    _FamilySpec(
        "cellular_automata",
        "Cellular Automata",
        ("cellular", "automata", "cells", "grid", "life", "evolve"),
        "Use a visible grid of local state rules and neighbor transitions.",
        "Advance states in discrete generations with clear rule changes or thresholds.",
        "medium",
        ("Large grids can pressure CPU updates and pixel writes.",),
        ("Rules can feel arbitrary unless state meanings are named.",),
        ("grid_systems", "modular_tiling", "reaction_diffusion"),
    ),
    _FamilySpec(
        "reaction_diffusion",
        "Reaction Diffusion",
        ("reaction", "diffusion", "turing", "chemical", "morphogenesis"),
        "Use interacting fields to form organic spots, waves, or veined surfaces.",
        "Let pattern formation emerge through iterative diffusion-like updates.",
        "high",
        ("Reaction-diffusion simulation is update-heavy at high resolutions.",),
        (
            "Parameter tuning is sensitive and can collapse into mush or "
            "static texture.",
        ),
        ("noise_fields", "cellular_automata", "modular_tiling"),
    ),
    _FamilySpec(
        "voronoi_systems",
        "Voronoi Systems",
        ("voronoi", "cells", "sites", "crystal", "territory", "partition"),
        "Partition space around seed sites to create cellular regions and edges.",
        "Move or reveal sites over time while preserving region legibility.",
        "medium",
        ("Many dynamic sites can make region recomputation costly.",),
        ("Voronoi cells need visual hierarchy or they become a generic texture.",),
        ("grid_systems", "graph_network_systems", "modular_tiling"),
    ),
    _FamilySpec(
        "noise_fields",
        "Noise Fields",
        ("noise", "perlin", "simplex", "turbulence", "organic", "field"),
        (
            "Use coherent noise to drive density, displacement, color, or "
            "contour variation."
        ),
        "Animate noise through time offsets, phase shifts, or slow domain warping.",
        "medium",
        ("Layered noise can become expensive and visually muddy.",),
        ("Noise needs a clear mapping to intent rather than generic texture.",),
        ("grid_systems", "flow_fields", "modular_tiling"),
    ),
    _FamilySpec(
        "flow_fields",
        "Flow Fields",
        ("flow", "vector", "field", "stream", "current", "wind"),
        (
            "Use vector directions to guide particles, strokes, or geometry "
            "through space."
        ),
        "Evolve the field slowly so motion reads as continuous flow.",
        "medium",
        ("Dense advected particles or trails can pressure frame rate.",),
        ("Field direction must remain interpretable or motion becomes noise.",),
        ("noise_fields", "particle_systems", "grid_systems"),
    ),
    _FamilySpec(
        "signed_distance_fields",
        "Signed Distance Fields",
        ("sdf", "signed distance", "raymarch", "metaball", "implicit"),
        "Define shapes by distance functions, blends, and thresholded contours.",
        "Animate distance parameters, blends, or smooth minimum transitions.",
        "high",
        (
            "Raymarching or complex SDF blending can exceed lightweight "
            "runtime budgets.",
        ),
        ("Distance operations need strong simplification for non-shader outputs.",),
        ("polar_radial_systems", "recursive_geometry", "modular_tiling"),
    ),
    _FamilySpec(
        "polar_radial_systems",
        "Polar/Radial Systems",
        ("polar", "radial", "mandala", "concentric", "orbit", "spiral"),
        (
            "Use radius, angle, rings, spokes, and rotational symmetry as "
            "structural coordinates."
        ),
        "Animate angle, radius, phase, orbit, or ring activation over time.",
        "low",
        ("High segment counts or nested rings can still pressure 2D draw cost.",),
        ("Radial order can become static unless phase or variation is purposeful.",),
        ("recursive_geometry", "modular_tiling", "grid_systems"),
    ),
    _FamilySpec(
        "grid_systems",
        "Grid Systems",
        ("grid", "matrix", "lattice", "rows", "columns", "tiles"),
        (
            "Organize the structure into explicit cells, coordinates, or "
            "lattice positions."
        ),
        "Use cell state, reveal order, or scanline rhythm as temporal structure.",
        "low",
        ("Very large grids can still pressure draw calls or per-cell updates.",),
        ("A grid can feel mechanical if cell state does not carry intent.",),
        ("modular_tiling", "polar_radial_systems", "noise_fields"),
    ),
    _FamilySpec(
        "graph_network_systems",
        "Graph/Network Systems",
        ("graph", "network", "nodes", "edges", "connection", "constellation"),
        "Represent relations as nodes, edges, clusters, and paths.",
        "Animate links through activation, propagation, clustering, or pruning.",
        "medium",
        ("Dense all-to-all connections can overwhelm rendering and readability.",),
        ("Networks need explicit semantics or they become decorative lines.",),
        ("grid_systems", "particle_systems", "modular_tiling"),
    ),
    _FamilySpec(
        "swarm_systems",
        "Swarm Systems",
        ("swarm", "agents", "collective", "emergent", "scatter", "gather"),
        "Use many simple agents coordinated by attraction, repulsion, and local goals.",
        "Shift the swarm between dispersal, pursuit, clustering, and rest phases.",
        "high",
        ("Swarm simulations can become expensive with many pairwise forces.",),
        ("Emergent motion needs bounds so it does not override the creative plan.",),
        ("particle_systems", "flow_fields", "grid_systems"),
    ),
    _FamilySpec(
        "wave_systems",
        "Wave Systems",
        ("wave", "sine", "ripple", "oscillation", "undulate", "pulse"),
        (
            "Use waves, ripples, interference, and phase offsets to organize "
            "motion or form."
        ),
        "Let phase, amplitude, frequency, and damping carry temporal progression.",
        "low",
        ("Many layered waves can create visual clutter or aliasing.",),
        ("Wave parameters need semantic labels to avoid arbitrary motion.",),
        ("harmonic_oscillators", "polar_radial_systems", "grid_systems"),
    ),
    _FamilySpec(
        "harmonic_oscillators",
        "Harmonic Oscillators",
        ("harmonic", "oscillator", "pendulum", "resonance", "frequency"),
        "Use coupled periodic values to coordinate scale, color, radius, or motion.",
        "Compose phases and frequencies into stable rhythmic loops.",
        "low",
        ("Stacked oscillators can become visually busy or hard to debug.",),
        ("Frequency choices need clear hierarchy or motion feels arbitrary.",),
        ("wave_systems", "polar_radial_systems", "modular_tiling"),
    ),
    _FamilySpec(
        "modular_tiling",
        "Modular Tiling",
        ("tile", "tiling", "module", "mosaic", "repeat", "pattern"),
        "Build the visual field from reusable modules with controlled variation.",
        "Animate module reveal, rotation, substitution, or state transitions.",
        "low",
        ("Excessive modules can increase draw cost and reduce hierarchy.",),
        ("Tiling can feel repetitive without purposeful variation.",),
        ("grid_systems", "polar_radial_systems", "noise_fields"),
    ),
    _FamilySpec(
        "sacred_geometry_pattern_systems",
        "Sacred Geometry Pattern Systems",
        ("sacred", "geometry", "mandala", "flower of life", "vesica", "yantra"),
        (
            "Use named geometric pattern rules, symmetry, ratios, and radial "
            "relationships."
        ),
        "Reveal pattern construction in phases so symbolic order remains legible.",
        "medium",
        ("High-detail geometric overlays can become costly and visually crowded.",),
        (
            "Symbolic pattern claims should remain visual, not doctrinal or "
            "unsupported.",
        ),
        ("polar_radial_systems", "modular_tiling", "recursive_geometry"),
    ),
)

_SPEC_BY_FAMILY = {item.family: item for item in _FAMILY_SPECS}
_SPEC_ORDER = {item.family: index for index, item in enumerate(_FAMILY_SPECS)}

_STRATEGY_FAMILY_HINTS: dict[str, tuple[ProceduralFamily, ...]] = {
    "recursive_emergence": (
        "recursive_geometry",
        "fractals",
        "particle_systems",
    ),
    "fractal_growth": ("fractals", "recursive_geometry", "l_systems"),
    "particle_cosmology": (
        "particle_systems",
        "swarm_systems",
        "flow_fields",
        "noise_fields",
    ),
    "cellular_evolution": (
        "cellular_automata",
        "reaction_diffusion",
        "voronoi_systems",
    ),
    "sacred_geometry": (
        "sacred_geometry_pattern_systems",
        "polar_radial_systems",
        "modular_tiling",
        "recursive_geometry",
    ),
    "field_dynamics": ("flow_fields", "noise_fields", "wave_systems"),
    "minimal_generative_systems": (
        "grid_systems",
        "modular_tiling",
        "harmonic_oscillators",
    ),
}

_TECHNIQUE_FAMILY_HINTS: dict[str, tuple[ProceduralFamily, ...]] = {
    "fractal_recursion": ("fractals", "recursive_geometry", "l_systems"),
    "particle_systems": (
        "particle_systems",
        "swarm_systems",
        "flow_fields",
    ),
    "reaction_diffusion": ("reaction_diffusion", "noise_fields"),
    "boids": ("boids", "swarm_systems", "particle_systems"),
    "cellular_automata": ("cellular_automata", "grid_systems"),
    "voronoi": ("voronoi_systems", "graph_network_systems"),
    "noise_fields": ("noise_fields", "flow_fields"),
    "recursive_geometry": (
        "recursive_geometry",
        "polar_radial_systems",
        "fractals",
    ),
    "sdf": ("signed_distance_fields", "polar_radial_systems"),
    "signed_distance_composition": (
        "signed_distance_fields",
        "polar_radial_systems",
    ),
    "feedback_systems": ("wave_systems", "flow_fields", "noise_fields"),
    "audio_reactive_mappings": (
        "wave_systems",
        "harmonic_oscillators",
        "particle_systems",
    ),
}

_NARRATIVE_FAMILY_HINTS: dict[str, tuple[ProceduralFamily, ...]] = {
    "death_and_rebirth": (
        "particle_systems",
        "recursive_geometry",
        "polar_radial_systems",
    ),
    "descent_and_return": ("flow_fields", "wave_systems", "recursive_geometry"),
    "emergence_from_chaos": ("noise_fields", "particle_systems", "flow_fields"),
    "initiation": ("polar_radial_systems", "sacred_geometry_pattern_systems"),
    "ascent": ("polar_radial_systems", "wave_systems"),
    "dissolution_and_reintegration": (
        "particle_systems",
        "noise_fields",
        "flow_fields",
    ),
    "expansion_from_seed_to_cosmos": (
        "fractals",
        "polar_radial_systems",
        "particle_systems",
    ),
    "fragmentation_and_recomposition": (
        "particle_systems",
        "graph_network_systems",
    ),
    "threshold_crossing": ("polar_radial_systems", "grid_systems"),
    "spiral_transformation": (
        "recursive_geometry",
        "polar_radial_systems",
        "fractals",
    ),
    "mirror_reflection_journey": (
        "sacred_geometry_pattern_systems",
        "polar_radial_systems",
    ),
    "dark_to_light_transformation": ("wave_systems", "particle_systems"),
    "symbolic_vignette": ("modular_tiling", "grid_systems"),
}

_COMPOSITION_FAMILY_HINTS: dict[str, tuple[ProceduralFamily, ...]] = {
    "central_emergence": ("polar_radial_systems", "particle_systems"),
    "radial_expansion": (
        "polar_radial_systems",
        "sacred_geometry_pattern_systems",
        "recursive_geometry",
    ),
    "spiral_composition": ("recursive_geometry", "polar_radial_systems"),
    "layered_depth": ("noise_fields", "flow_fields", "signed_distance_fields"),
    "field_composition": ("noise_fields", "flow_fields", "particle_systems"),
    "threshold_composition": ("grid_systems", "polar_radial_systems"),
    "descent_ascent_composition": ("wave_systems", "flow_fields"),
    "fragmented_recomposition": (
        "particle_systems",
        "graph_network_systems",
        "noise_fields",
    ),
    "mirrored_composition": (
        "sacred_geometry_pattern_systems",
        "polar_radial_systems",
    ),
    "orbiting_focal_structure": (
        "polar_radial_systems",
        "harmonic_oscillators",
        "particle_systems",
    ),
    "distributed_constellation": (
        "graph_network_systems",
        "particle_systems",
    ),
    "minimal_void_and_form_composition": (
        "modular_tiling",
        "grid_systems",
        "harmonic_oscillators",
    ),
}

__all__ = [
    "PROCEDURAL_STRUCTURE_AUTHORITY_BOUNDARY",
    "ProceduralComplexityLevel",
    "ProceduralFamily",
    "ProceduralStructureChoice",
    "ProceduralStructurePlan",
    "derive_procedural_structure_plan",
    "procedural_structure_prompt_lines",
]
