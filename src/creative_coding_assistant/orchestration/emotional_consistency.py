"""Bounded Emotional Consistency Engine for V3.2 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_composition import (
    CompositionPattern,
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
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeModuleKind,
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralFamily,
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import (
    SemanticMotifId,
    SemanticMotifSystem,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    NarrativePhaseName,
    SymbolicNarrativePlan,
)

if TYPE_CHECKING:
    from creative_coding_assistant.orchestration.creative_reasoning import (
        CreativeReasoningResult,
    )

EmotionalTone = Literal[
    "awe",
    "wonder",
    "mystery",
    "serenity",
    "tension",
    "rupture",
    "grief",
    "dissolution",
    "suspension",
    "emergence",
    "ecstasy",
    "clarity",
    "intimacy",
    "vastness",
    "ritual solemnity",
    "playful curiosity",
    "dread",
    "release",
    "transformation",
    "integration",
]
EmotionalIntensity = Literal["low", "medium", "high", "variable"]

EMOTIONAL_CONSISTENCY_AUTHORITY_BOUNDARY = (
    "The Emotional Consistency Engine organizes emotional direction as "
    "inspectable design metadata only; it does not generate code, claim "
    "objective emotional truth, auto-select runtimes, change preview behavior, "
    "route providers or models, run autonomous loops, implement V4 multi-agent "
    "runtime, or implement HoloMind."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_AMBIGUOUS_EMOTIONAL_TOKENS = frozenset(
    {
        "ambiguous",
        "atmosphere",
        "deep",
        "emotional",
        "evocative",
        "feeling",
        "maybe",
        "mood",
        "subtle",
        "vibe",
    }
)
_AUDIO_TOKENS = frozenset(
    {
        "audio",
        "audiovisual",
        "beat",
        "drone",
        "music",
        "pulse",
        "rhythm",
        "sound",
        "tone",
    }
)
_HIGH_INTENSITY_TOKENS = frozenset(
    {
        "aggressive",
        "chaotic",
        "explosive",
        "flashing",
        "intense",
        "strobe",
        "violent",
    }
)


class EmotionalPhaseMapping(BaseModel):
    """One phase of the emotional arc."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    phase: NarrativePhaseName
    tone: EmotionalTone
    intensity: EmotionalIntensity
    guidance: str = Field(min_length=1, max_length=300)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalNarrativeMapping(BaseModel):
    """Mapping from emotional tone to symbolic narrative."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tone: EmotionalTone
    narrative_phase: NarrativePhaseName
    narrative_function: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalMotifMapping(BaseModel):
    """Mapping from emotional tone to semantic motif."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tone: EmotionalTone
    motif_id: SemanticMotifId | None = None
    emotional_function: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalCompositionMapping(BaseModel):
    """Mapping from emotional tone to composition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tone: EmotionalTone
    composition_pattern: CompositionPattern | None = None
    composition_guidance: str = Field(min_length=1, max_length=320)
    spatial_or_density_guidance: str = Field(min_length=1, max_length=300)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalStructureMapping(BaseModel):
    """Mapping from emotional tone to procedural/generative structure."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tone: EmotionalTone
    procedural_families: tuple[ProceduralFamily, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generative_module_kinds: tuple[GenerativeModuleKind, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    structural_guidance: str = Field(min_length=1, max_length=340)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalParameterMapping(BaseModel):
    """Mapping from emotional tone to named generative parameters."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tone: EmotionalTone
    parameter_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    parameter_guidance: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class EmotionalFallbackStrategy(BaseModel):
    """Lower-risk emotional direction when scope is broad or ambiguous."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_primary_tone: EmotionalTone
    fallback_secondary_tones: tuple[EmotionalTone, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    simplification_strategy: str = Field(min_length=1, max_length=320)
    preserved_feeling: str = Field(min_length=1, max_length=280)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)


class EmotionalConsistencyProfile(BaseModel):
    """Inspectable emotional consistency metadata derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["emotional_consistency_engine"] = "emotional_consistency_engine"
    primary_emotional_tone: EmotionalTone
    secondary_emotional_tones: tuple[EmotionalTone, ...] = Field(
        min_length=1,
        max_length=6,
    )
    emotional_arc: tuple[str, ...] = Field(min_length=1, max_length=8)
    emotional_phase_mapping: tuple[EmotionalPhaseMapping, ...] = Field(
        min_length=1,
        max_length=5,
    )
    emotional_to_narrative_mapping: tuple[EmotionalNarrativeMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    emotional_to_motif_mapping: tuple[EmotionalMotifMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    emotional_to_composition_mapping: tuple[EmotionalCompositionMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    emotional_to_structure_mapping: tuple[EmotionalStructureMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    emotional_to_parameter_mapping: tuple[EmotionalParameterMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    color_light_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    motion_rhythm_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    audiovisual_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    emotional_coherence_score: int = Field(ge=0, le=100)
    emotional_tensions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    mismatch_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    flattening_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    over_intensity_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    under_intensity_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    fallback_emotional_strategy: EmotionalFallbackStrategy
    unresolved_emotional_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=EMOTIONAL_CONSISTENCY_AUTHORITY_BOUNDARY,
        max_length=640,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _EmotionalContext:
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
    procedural_structure: ProceduralStructurePlan | None
    generative_structure: GenerativeStructureBlueprint | None
    semantic_motif: SemanticMotifSystem | None
    creative_reasoning: CreativeReasoningResult | None
    text: str
    request_tokens: frozenset[str]
    tokens: frozenset[str]


def derive_emotional_consistency_profile(
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
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    creative_reasoning: CreativeReasoningResult | None = None,
) -> EmotionalConsistencyProfile:
    """Derive emotional consistency metadata without changing generation."""

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
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        creative_reasoning=creative_reasoning,
    )
    scored = _score_tones(context)
    selected = _selected_tones(scored)
    primary = _primary_tone(selected, context)
    secondary = _secondary_tones(primary, selected)
    tones = (primary, *secondary)
    tensions = _emotional_tensions(primary, secondary, context)
    mismatches = _mismatch_risks(primary, secondary, context)
    flattening = _flattening_risks(primary, secondary, context)
    over_intensity = _over_intensity_risks(primary, secondary, context)
    under_intensity = _under_intensity_risks(primary, secondary, context)
    unresolved = _unresolved_gaps(context, primary, secondary)
    return EmotionalConsistencyProfile(
        primary_emotional_tone=primary,
        secondary_emotional_tones=secondary,
        emotional_arc=_emotional_arc(primary, secondary, context),
        emotional_phase_mapping=_phase_mappings(primary, secondary, context),
        emotional_to_narrative_mapping=_narrative_mappings(tones, context),
        emotional_to_motif_mapping=_motif_mappings(tones, context),
        emotional_to_composition_mapping=_composition_mappings(tones, context),
        emotional_to_structure_mapping=_structure_mappings(tones, context),
        emotional_to_parameter_mapping=_parameter_mappings(tones, context),
        color_light_guidance=_color_light_guidance(primary, secondary, context),
        motion_rhythm_guidance=_motion_rhythm_guidance(primary, secondary, context),
        audiovisual_guidance=_audiovisual_guidance(primary, secondary, context),
        emotional_coherence_score=_coherence_score(
            context,
            tensions=tensions,
            mismatches=mismatches,
            unresolved=unresolved,
        ),
        emotional_tensions=tensions,
        mismatch_risks=mismatches,
        flattening_risks=flattening,
        over_intensity_risks=over_intensity,
        under_intensity_risks=under_intensity,
        fallback_emotional_strategy=_fallback_strategy(primary, secondary, context),
        unresolved_emotional_gaps=unresolved,
        hitl_questions=_hitl_questions(unresolved, mismatches),
        prompt_guidance=_prompt_guidance(primary, secondary, mismatches),
        evidence=_evidence(context, primary, secondary, scored),
    )


def emotional_consistency_prompt_lines(
    profile: EmotionalConsistencyProfile,
) -> tuple[str, ...]:
    """Render emotional consistency metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Primary emotional tone: {profile.primary_emotional_tone}.",
        "Secondary emotional tones: "
        + ", ".join(profile.secondary_emotional_tones)
        + ".",
        f"Emotional coherence score: {profile.emotional_coherence_score}/100.",
    ]
    lines.extend(f"Emotional arc: {item}" for item in profile.emotional_arc[:6])
    lines.extend(
        f"Emotional phase: {item.phase}; {item.tone}; {item.guidance}"
        for item in profile.emotional_phase_mapping[:5]
    )
    lines.extend(
        "Emotional narrative mapping: "
        f"{item.tone}; {item.narrative_phase}; {item.narrative_function}"
        for item in profile.emotional_to_narrative_mapping[:5]
    )
    lines.extend(
        "Emotional motif mapping: "
        f"{item.tone}; {item.motif_id or 'no motif'}; {item.emotional_function}"
        for item in profile.emotional_to_motif_mapping[:5]
    )
    lines.extend(
        f"Emotional composition mapping: {item.tone}; {item.composition_guidance}"
        for item in profile.emotional_to_composition_mapping[:5]
    )
    lines.extend(
        f"Emotional structure mapping: {item.tone}; {item.structural_guidance}"
        for item in profile.emotional_to_structure_mapping[:5]
    )
    lines.extend(
        f"Emotional parameter mapping: {item.tone}; {', '.join(item.parameter_names)}"
        for item in profile.emotional_to_parameter_mapping[:5]
    )
    lines.extend(
        f"Emotional color/light guidance: {item}"
        for item in profile.color_light_guidance
    )
    lines.extend(
        f"Emotional motion/rhythm guidance: {item}"
        for item in profile.motion_rhythm_guidance
    )
    lines.extend(
        f"Emotional audiovisual guidance: {item}"
        for item in profile.audiovisual_guidance
    )
    lines.extend(f"Emotional tension: {item}" for item in profile.emotional_tensions)
    lines.extend(f"Emotional mismatch risk: {item}" for item in profile.mismatch_risks)
    lines.extend(
        f"Emotional flattening risk: {item}" for item in profile.flattening_risks
    )
    lines.extend(
        f"Emotional over-intensity risk: {item}"
        for item in profile.over_intensity_risks
    )
    lines.extend(
        f"Emotional under-intensity risk: {item}"
        for item in profile.under_intensity_risks
    )
    lines.append(
        "Emotional fallback strategy: "
        f"{profile.fallback_emotional_strategy.fallback_primary_tone}; "
        f"{profile.fallback_emotional_strategy.simplification_strategy}"
    )
    lines.extend(
        f"Unresolved emotional gap: {item}"
        for item in profile.unresolved_emotional_gaps
    )
    lines.extend(f"HITL emotional question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Emotional prompt guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:56])


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
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    creative_reasoning: CreativeReasoningResult | None,
) -> _EmotionalContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_intent.experiential_goal if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
    ]
    if creative_translation is not None:
        parts.extend(creative_translation.mood_atmosphere)
        parts.extend(creative_translation.movement_language)
        parts.extend(creative_translation.color_material_direction)
        parts.extend(creative_translation.musical_references)
    if creative_intent is not None:
        parts.append(creative_intent.emotional_intent.summary)
        parts.extend(creative_intent.emotional_intent.signals)
        parts.append(creative_intent.motion_intent.summary)
        parts.append(creative_intent.rhythm_intent.summary)
        parts.append(creative_intent.light_color_intent.summary)
        parts.append(creative_intent.audio_intent.summary)
    if creative_hierarchy is not None:
        parts.extend(
            item.dimension for item in creative_hierarchy.primary_creative_priorities
        )
    if symbolic_narrative is not None:
        parts.extend(
            (
                symbolic_narrative.narrative_archetype,
                symbolic_narrative.symbolic_arc,
                symbolic_narrative.experiential_goal,
            )
        )
        parts.extend(symbolic_narrative.emotional_progression)
        parts.extend(symbolic_narrative.motion_progression)
        parts.extend(symbolic_narrative.audio_progression)
        for phase in symbolic_narrative.phases:
            parts.extend(
                (
                    phase.phase,
                    phase.title,
                    phase.emotional_state,
                    phase.visual_state,
                    phase.motion_state,
                    phase.audio_state or "",
                )
            )
    if creative_composition is not None:
        parts.extend(
            (
                creative_composition.composition_pattern,
                creative_composition.primary_focal_point,
                creative_composition.spatial_organization,
                creative_composition.density_plan,
                creative_composition.rhythm_plan,
                creative_composition.balance_plan,
                creative_composition.depth_layering_guidance,
            )
        )
        parts.extend(creative_composition.audiovisual_composition_notes)
    if procedural_structure is not None:
        parts.extend(procedural_structure.recommended_families)
        parts.append(procedural_structure.spatial_structure_plan)
        parts.append(procedural_structure.temporal_structure_plan)
        parts.append(procedural_structure.audiovisual_structure_plan or "")
    if generative_structure is not None:
        parts.append(generative_structure.blueprint_name)
        parts.append(generative_structure.generative_architecture)
        parts.append(generative_structure.spatial_evolution)
        parts.append(generative_structure.temporal_evolution)
        parts.extend(module.kind for module in generative_structure.procedural_modules)
        parts.extend(rule.phase for rule in generative_structure.evolution_rules)
        parts.extend(
            parameter.name for parameter in generative_structure.parameter_schema
        )
    if semantic_motif is not None:
        parts.append(semantic_motif.motif_system_name)
        parts.extend(motif.motif_id for motif in semantic_motif.primary_motifs)
        parts.extend(motif.motif_id for motif in semantic_motif.secondary_motifs)
        parts.extend(semantic_motif.motif_recurrence_plan)
        parts.extend(semantic_motif.motif_transformation_plan)
    if creative_reasoning is not None:
        parts.append(creative_reasoning.recommended_creative_direction)
    text = _normalize(" ".join(parts))
    return _EmotionalContext(
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
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        creative_reasoning=creative_reasoning,
        text=text,
        request_tokens=frozenset(_TOKEN_PATTERN.findall(_normalize(request.query))),
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _score_tones(
    context: _EmotionalContext,
) -> dict[EmotionalTone, tuple[int, tuple[str, ...]]]:
    scores: dict[EmotionalTone, int] = {tone: 0 for tone in _TONE_ORDER}
    evidence: dict[EmotionalTone, list[str]] = {tone: [] for tone in _TONE_ORDER}
    for tone in _TONE_ORDER:
        if _tone_in_text(tone, _normalize(context.request.query)):
            _add_score(scores, evidence, tone, 7, f"Explicit request tone: {tone}.")
        elif _tone_in_text(tone, context.text):
            _add_score(scores, evidence, tone, 2, f"Upstream tone signal: {tone}.")
    for alias, tones in _TONE_ALIASES.items():
        if _alias_in_text(alias, context.text):
            boost = 4 if _alias_in_text(alias, _normalize(context.request.query)) else 2
            for tone in tones:
                _add_score(scores, evidence, tone, boost, f"Emotional alias: {alias}.")
    if context.symbolic_narrative is not None:
        for tone in _NARRATIVE_TONES.get(
            context.symbolic_narrative.narrative_archetype,
            (),
        ):
            _add_score(
                scores,
                evidence,
                tone,
                5,
                f"Narrative archetype: {context.symbolic_narrative.narrative_archetype}.",
            )
    if context.creative_composition is not None:
        for tone in _COMPOSITION_TONES.get(
            context.creative_composition.composition_pattern,
            (),
        ):
            _add_score(
                scores,
                evidence,
                tone,
                3,
                f"Composition pattern: {context.creative_composition.composition_pattern}.",
            )
    if context.procedural_structure is not None:
        for family in context.procedural_structure.recommended_families:
            for tone in _PROCEDURAL_TONES.get(family, ()):
                _add_score(scores, evidence, tone, 2, f"Procedural family: {family}.")
    if context.generative_structure is not None:
        for module in context.generative_structure.procedural_modules:
            for tone in _MODULE_TONES.get(module.kind, ()):
                _add_score(
                    scores, evidence, tone, 2, f"Generative module: {module.kind}."
                )
        for parameter in context.generative_structure.parameter_schema:
            for tone in _PARAMETER_TONES.get(parameter.name, ()):
                _add_score(
                    scores,
                    evidence,
                    tone,
                    1,
                    f"Generative parameter: {parameter.name}.",
                )
    if context.semantic_motif is not None:
        for motif in context.semantic_motif.primary_motifs:
            for tone in _MOTIF_TONES.get(motif.motif_id, ()):
                _add_score(
                    scores, evidence, tone, 4, f"Primary motif: {motif.motif_id}."
                )
        for motif in context.semantic_motif.secondary_motifs:
            for tone in _MOTIF_TONES.get(motif.motif_id, ()):
                _add_score(
                    scores, evidence, tone, 2, f"Secondary motif: {motif.motif_id}."
                )
    return {tone: (scores[tone], tuple(evidence[tone][:8])) for tone in _TONE_ORDER}


def _selected_tones(
    scored: dict[EmotionalTone, tuple[int, tuple[str, ...]]],
) -> tuple[EmotionalTone, ...]:
    selected = sorted(
        (tone for tone, (score, _) in scored.items() if score > 0),
        key=lambda tone: (-scored[tone][0], _TONE_ORDER.index(tone)),
    )
    if selected:
        return tuple(selected[:8])
    return ("wonder", "clarity", "serenity", "mystery")


def _primary_tone(
    selected: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> EmotionalTone:
    if "transformation" in selected and (
        context.request_tokens.intersection(
            {"phoenix", "rebirth", "transform", "transformation"}
        )
        or (
            context.symbolic_narrative is not None
            and context.symbolic_narrative.narrative_archetype
            in {
                "death_and_rebirth",
                "dissolution_and_reintegration",
                "spiral_transformation",
            }
        )
    ):
        return "transformation"
    return selected[0]


def _secondary_tones(
    primary: EmotionalTone,
    selected: tuple[EmotionalTone, ...],
) -> tuple[EmotionalTone, ...]:
    secondary = [tone for tone in selected if tone != primary]
    for tone in _DEFAULT_SECONDARY_BY_PRIMARY.get(primary, ()):
        if tone not in secondary and tone != primary:
            secondary.append(tone)
    return tuple(secondary[:6])


def _emotional_arc(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    if primary in _ARC_BY_PRIMARY:
        arc = list(_ARC_BY_PRIMARY[primary])
    elif context.symbolic_narrative is not None:
        arc = [
            f"{phase.phase}: {phase.emotional_state}"
            for phase in context.symbolic_narrative.phases
        ]
    else:
        arc = [
            f"establish {primary}",
            f"develop {secondary[0] if secondary else primary}",
            f"resolve toward {_resolution_tone(primary, secondary)}",
        ]
    if context.generative_structure is not None:
        phases = ", ".join(
            rule.phase for rule in context.generative_structure.evolution_rules[:4]
        )
        arc.append(f"Align emotional turns with generative phases: {phases}.")
    return tuple(_dedupe(arc))[:8]


def _phase_mappings(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalPhaseMapping, ...]:
    phases = (
        context.symbolic_narrative.phases
        if context.symbolic_narrative is not None
        else ()
    )
    if not phases:
        return tuple(
            EmotionalPhaseMapping(
                phase=phase,
                tone=_phase_tone(phase, primary, secondary),
                intensity=_phase_intensity(phase, primary),
                guidance=_phase_guidance(phase, _phase_tone(phase, primary, secondary)),
                evidence=("No Symbolic Narrative Planner metadata attached.",),
            )
            for phase in _PHASE_ORDER
        )
    mappings: list[EmotionalPhaseMapping] = []
    for phase in phases:
        tone = _tone_from_text(phase.emotional_state, primary, secondary)
        mappings.append(
            EmotionalPhaseMapping(
                phase=phase.phase,
                tone=tone,
                intensity=_phase_intensity(phase.phase, primary),
                guidance=_clip(
                    f"Let {tone} carry {phase.title}: {phase.emotional_state}",
                    300,
                ),
                evidence=(f"Narrative phase: {phase.phase}.", phase.emotional_state),
            )
        )
    return tuple(mappings[:5])


def _narrative_mappings(
    tones: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalNarrativeMapping, ...]:
    mappings: list[EmotionalNarrativeMapping] = []
    for index, tone in enumerate(tones[:5]):
        phase = _phase_for_tone(tone, index)
        if context.symbolic_narrative is not None:
            function = (
                f"Use {tone} to support "
                f"{context.symbolic_narrative.narrative_archetype} without "
                "claiming universal emotional meaning."
            )
            evidence = (
                f"Narrative archetype: {context.symbolic_narrative.narrative_archetype}.",
            )
        else:
            function = f"Use {tone} as an inferred emotional beat."
            evidence = ("No Symbolic Narrative Planner metadata attached.",)
        mappings.append(
            EmotionalNarrativeMapping(
                tone=tone,
                narrative_phase=phase,
                narrative_function=_clip(function, 320),
                evidence=evidence,
            )
        )
    return tuple(mappings)


def _motif_mappings(
    tones: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalMotifMapping, ...]:
    mappings: list[EmotionalMotifMapping] = []
    available = (
        (
            *context.semantic_motif.primary_motifs,
            *context.semantic_motif.secondary_motifs,
        )
        if context.semantic_motif is not None
        else ()
    )
    for tone in tones[:6]:
        motif_id = next(
            (
                motif.motif_id
                for motif in available
                if tone in _MOTIF_TONES.get(motif.motif_id, ())
            ),
            available[0].motif_id if available else None,
        )
        mappings.append(
            EmotionalMotifMapping(
                tone=tone,
                motif_id=motif_id,
                emotional_function=_clip(
                    (
                        f"Let {motif_id} carry {tone} as a recurring design cue."
                        if motif_id is not None
                        else f"Carry {tone} without adding unsupported motif claims."
                    ),
                    320,
                ),
                evidence=(
                    (
                        (
                            "Semantic Motif Engine primary motifs: "
                            + ", ".join(
                                motif.motif_id
                                for motif in context.semantic_motif.primary_motifs
                            )
                            + "."
                        ),
                    )
                    if context.semantic_motif is not None
                    else ("No Semantic Motif Engine metadata attached.",)
                ),
            )
        )
    return tuple(mappings)


def _composition_mappings(
    tones: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalCompositionMapping, ...]:
    mappings: list[EmotionalCompositionMapping] = []
    for tone in tones[:5]:
        if context.creative_composition is not None:
            pattern = context.creative_composition.composition_pattern
            guidance = (
                f"Use {tone} inside {pattern} while preserving the primary "
                f"focal point: {context.creative_composition.primary_focal_point}"
            )
            density = context.creative_composition.density_plan
            evidence = (f"Composition pattern: {pattern}.",)
        else:
            pattern = None
            guidance = f"Use {tone} as a composition-level atmosphere cue."
            density = "Keep density changes readable enough to preserve emotion."
            evidence = ("No Creative Composition Planner metadata attached.",)
        mappings.append(
            EmotionalCompositionMapping(
                tone=tone,
                composition_pattern=pattern,
                composition_guidance=_clip(guidance, 320),
                spatial_or_density_guidance=_clip(density, 300),
                evidence=evidence,
            )
        )
    return tuple(mappings)


def _structure_mappings(
    tones: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalStructureMapping, ...]:
    return tuple(_structure_mapping(tone, context) for tone in tones[:6])


def _structure_mapping(
    tone: EmotionalTone,
    context: _EmotionalContext,
) -> EmotionalStructureMapping:
    families: list[ProceduralFamily] = []
    modules: list[GenerativeModuleKind] = []
    if context.procedural_structure is not None:
        for family in context.procedural_structure.recommended_families:
            if tone in _PROCEDURAL_TONES.get(family, ()):
                families.append(family)
        if not families:
            families.append(context.procedural_structure.primary_structure.family)
    if context.generative_structure is not None:
        for module in context.generative_structure.procedural_modules:
            if tone in _MODULE_TONES.get(module.kind, ()):
                modules.append(module.kind)
        if not modules:
            modules.append(context.generative_structure.procedural_modules[0].kind)
    return EmotionalStructureMapping(
        tone=tone,
        procedural_families=tuple(_dedupe(families))[:5],
        generative_module_kinds=tuple(_dedupe(modules))[:8],
        structural_guidance=_clip(
            f"Preserve {tone} through existing procedural and generative structure.",
            340,
        ),
        evidence=_structure_evidence(families, modules),
    )


def _parameter_mappings(
    tones: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[EmotionalParameterMapping, ...]:
    return tuple(_parameter_mapping(tone, context) for tone in tones[:6])


def _parameter_mapping(
    tone: EmotionalTone,
    context: _EmotionalContext,
) -> EmotionalParameterMapping:
    names: list[str] = []
    if context.generative_structure is not None:
        available = {
            parameter.name
            for parameter in context.generative_structure.parameter_schema
        }
        for name, tones in _PARAMETER_TONES.items():
            if tone in tones and name in available:
                names.append(name)
        if not names:
            names.extend(
                name
                for name in context.generative_structure.control_parameters[:3]
                if name in available
            )
    if not names:
        names = ["time_phase", "global_scale"]
    return EmotionalParameterMapping(
        tone=tone,
        parameter_names=tuple(_dedupe(names))[:8],
        parameter_guidance=_clip(
            f"Use {', '.join(tuple(_dedupe(names))[:3])} to tune {tone} without "
            "adding new runtime behavior.",
            320,
        ),
        evidence=(f"Emotional-to-parameter mapping for {tone}.",),
    )


def _color_light_guidance(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    guidance = [
        _COLOR_LIGHT_BY_TONE.get(primary, _COLOR_LIGHT_DEFAULT).format(tone=primary)
    ]
    guidance.extend(
        _COLOR_LIGHT_BY_TONE.get(tone, _COLOR_LIGHT_DEFAULT).format(tone=tone)
        for tone in secondary[:2]
    )
    if (
        context.creative_translation is not None
        and context.creative_translation.color_material_direction
    ):
        guidance.append(
            "Preserve requested color/material direction: "
            + ", ".join(context.creative_translation.color_material_direction[:3])
            + "."
        )
    return tuple(_dedupe(guidance))[:8]


def _motion_rhythm_guidance(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    guidance = [_MOTION_BY_TONE.get(primary, _MOTION_DEFAULT).format(tone=primary)]
    guidance.extend(
        _MOTION_BY_TONE.get(tone, _MOTION_DEFAULT).format(tone=tone)
        for tone in secondary[:2]
    )
    if context.creative_composition is not None:
        guidance.append(
            "Tie motion/rhythm to composition rhythm: "
            f"{context.creative_composition.rhythm_plan}"
        )
    return tuple(_dedupe(guidance))[:8]


def _audiovisual_guidance(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    if not _audio_relevant(context):
        return ()
    guidance = [
        f"Use audio or audiovisual changes to reinforce {primary}, not to add a separate emotional arc."
    ]
    if (
        context.symbolic_narrative is not None
        and context.symbolic_narrative.audio_progression
    ):
        guidance.append(
            "Align audio with narrative audio progression: "
            + " -> ".join(context.symbolic_narrative.audio_progression[:4])
            + "."
        )
    if (
        context.generative_structure is not None
        and context.generative_structure.audiovisual_hooks
    ):
        guidance.append("Keep audiovisual hooks bounded to existing blueprint hooks.")
    if secondary:
        guidance.append(f"Use {secondary[0]} as the secondary audio/rhythm color.")
    return tuple(_dedupe(guidance))[:8]


def _coherence_score(
    context: _EmotionalContext,
    *,
    tensions: tuple[str, ...],
    mismatches: tuple[str, ...],
    unresolved: tuple[str, ...],
) -> int:
    score = 58
    if context.creative_intent is not None:
        score += 8
    if context.symbolic_narrative is not None:
        score += 10
    if context.creative_composition is not None:
        score += 7
    if context.procedural_structure is not None:
        score += 5
    if context.generative_structure is not None:
        score += 6
    if context.semantic_motif is not None:
        score += 6
    score -= len(tensions) * 2
    score -= len(mismatches) * 5
    score -= len(unresolved) * 3
    return max(0, min(100, score))


def _emotional_tensions(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    tones = {primary, *secondary}
    tensions: list[str] = []
    if tones.intersection({"serenity", "release", "clarity"}) and tones.intersection(
        {"dread", "rupture", "tension", "grief"}
    ):
        tensions.append(
            "Calm or resolving tones must be staged separately from rupture, dread, or grief."
        )
    if "playful curiosity" in tones and tones.intersection(
        {"ritual solemnity", "grief", "dread"}
    ):
        tensions.append(
            "Playful curiosity and solemn or dark tones need explicit phase separation."
        )
    if (
        context.creative_tradeoffs is not None
        and context.creative_tradeoffs.hitl_advisable
    ):
        tensions.append("Creative trade-offs may alter the intended emotional weight.")
    return tuple(_dedupe(tensions))[:8]


def _mismatch_risks(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    tones = {primary, *secondary}
    risks: list[str] = []
    if "playful" in context.tokens and tones.intersection(
        {"ritual solemnity", "grief", "dread", "transformation"}
    ):
        risks.append(
            "Playful motion may weaken solemn transformation or ritual weight."
        )
    if context.tokens.intersection(
        {"bouncy", "cartoon", "cute"}
    ) and tones.intersection({"dread", "grief", "ritual solemnity"}):
        risks.append("Cute or bouncy styling may contradict the darker emotional tone.")
    if "serenity" in tones and context.tokens.intersection(_HIGH_INTENSITY_TOKENS):
        risks.append("High-intensity visual behavior may contradict serenity.")
    if context.symbolic_narrative is None:
        risks.append("Narrative emotional arc is inferred rather than planned.")
    return tuple(_dedupe(risks))[:8]


def _flattening_risks(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    risks = [
        f"Treating {primary} as a single constant mood may flatten the emotional arc."
    ]
    if len(secondary) > 4:
        risks.append("Too many secondary tones may blur the emotional hierarchy.")
    if (
        context.semantic_motif is not None
        and len(context.semantic_motif.primary_motifs) > 1
    ):
        risks.append(
            "Motif recurrence should evolve emotionally, not repeat unchanged."
        )
    return tuple(_dedupe(risks))[:8]


def _over_intensity_risks(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    tones = {primary, *secondary}
    risks: list[str] = []
    intense = tones.intersection({"dread", "rupture", "ecstasy", "tension"})
    if intense:
        risks.append(
            "Over-intensifying "
            + ", ".join(sorted(intense))
            + " may exhaust the viewer before resolution."
        )
    if context.tokens.intersection(_HIGH_INTENSITY_TOKENS):
        risks.append("High-intensity request language needs bounded contrast and rest.")
    return tuple(_dedupe(risks))[:8]


def _under_intensity_risks(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> tuple[str, ...]:
    risks: list[str] = []
    if primary in {"transformation", "release", "emergence"}:
        risks.append(f"Underplaying {primary} may make the arc feel unresolved.")
    if "subtle" in context.tokens and primary in {"rupture", "dread", "ecstasy"}:
        risks.append("Subtle styling may undercut the selected high-intensity tone.")
    if secondary:
        risks.append(
            f"Underusing {secondary[0]} may remove the emotional bridge into {primary}."
        )
    return tuple(_dedupe(risks))[:8]


def _fallback_strategy(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    context: _EmotionalContext,
) -> EmotionalFallbackStrategy:
    fallback = "clarity" if primary in {"dread", "rupture", "ecstasy"} else primary
    fallback_secondary = tuple(
        tone for tone in (secondary or ("serenity", "release")) if tone != fallback
    )[:3]
    return EmotionalFallbackStrategy(
        fallback_primary_tone=fallback,
        fallback_secondary_tones=fallback_secondary or ("serenity",),
        simplification_strategy=(
            f"Preserve {fallback} and reduce secondary emotional cues before "
            "changing structure, runtime, or motif hierarchy."
        ),
        preserved_feeling=_fallback_feeling(primary, context),
        prompt_guidance=(
            "Use fewer emotional tones before adding new visual systems.",
            "Treat emotional mapping as design guidance, not universal truth.",
        ),
    )


def _unresolved_gaps(
    context: _EmotionalContext,
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if context.tokens.intersection(_AMBIGUOUS_EMOTIONAL_TOKENS):
        gaps.append("Emotional language is abstract; confirm the dominant tone.")
    if "maybe" in context.tokens and len(secondary) > 2:
        gaps.append("Multiple possible emotional tones compete for priority.")
    if context.symbolic_narrative is None:
        gaps.append("No Symbolic Narrative Planner metadata is attached.")
    if context.semantic_motif is None:
        gaps.append("No Semantic Motif Engine metadata is attached.")
    if _audio_relevant(context) and not _audiovisual_guidance(
        primary, secondary, context
    ):
        gaps.append("Audio is relevant but emotional audio guidance is underspecified.")
    return tuple(_dedupe(gaps))[:8]


def _hitl_questions(
    unresolved: tuple[str, ...],
    mismatches: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    for gap in unresolved:
        lowered = gap.lower()
        if "dominant tone" in lowered or "compete" in lowered:
            questions.append("Which emotional tone should remain dominant?")
        elif "audio" in lowered:
            questions.append("Should audio reinforce the same emotional arc?")
        elif "narrative" in lowered:
            questions.append("Which narrative phase should carry the emotional turn?")
        elif "motif" in lowered:
            questions.append("Which motif should carry the emotional tone?")
    if mismatches:
        questions.append("Should the mismatch be resolved before generation?")
    return tuple(_dedupe(questions))[:6]


def _prompt_guidance(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    mismatches: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use emotional consistency as design guidance, not executable code.",
        f"Keep {primary} as the clearest emotional anchor.",
        "Make the emotional arc visible through phase, rhythm, light, and density.",
        "Map emotional tones to existing narrative, motifs, structure, and parameters.",
    ]
    if secondary:
        guidance.append("Let secondary emotional tones bridge rather than compete.")
    if mismatches:
        guidance.append("Resolve emotional mismatches before adding effects.")
    return tuple(_dedupe(guidance))[:8]


def _evidence(
    context: _EmotionalContext,
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
    scored: dict[EmotionalTone, tuple[int, tuple[str, ...]]],
) -> tuple[str, ...]:
    evidence = [
        f"Primary emotional tone: {primary}.",
        "Secondary emotional tones: " + ", ".join(secondary[:6]) + ".",
        f"Primary tone score: {scored[primary][0]}.",
    ]
    if context.symbolic_narrative is not None:
        evidence.append(
            f"Narrative source: {context.symbolic_narrative.narrative_archetype}."
        )
    if context.creative_composition is not None:
        evidence.append(
            f"Composition source: {context.creative_composition.composition_pattern}."
        )
    if context.procedural_structure is not None:
        evidence.append(
            f"Procedural source: {context.procedural_structure.primary_structure.family}."
        )
    if context.generative_structure is not None:
        evidence.append(
            f"Generative source: {context.generative_structure.generative_architecture}."
        )
    if context.semantic_motif is not None:
        evidence.append(
            "Motif source: "
            + ", ".join(
                motif.motif_id for motif in context.semantic_motif.primary_motifs
            )
            + "."
        )
    evidence.extend(scored[primary][1][:3])
    return tuple(_dedupe(evidence))[:12]


def _add_score(
    scores: dict[EmotionalTone, int],
    evidence: dict[EmotionalTone, list[str]],
    tone: EmotionalTone,
    score: int,
    reason: str,
) -> None:
    scores[tone] += score
    evidence[tone].append(reason)


def _tone_in_text(tone: EmotionalTone, text: str) -> bool:
    return _alias_in_text(tone, text)


def _alias_in_text(alias: str, text: str) -> bool:
    normalized = _normalize(alias)
    if " " in normalized:
        return normalized in text
    return normalized in _TOKEN_PATTERN.findall(text)


def _tone_from_text(
    text: str,
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
) -> EmotionalTone:
    normalized = _normalize(text)
    for tone in (primary, *secondary, *_TONE_ORDER):
        if _tone_in_text(tone, normalized):
            return tone
    for alias, tones in _TONE_ALIASES.items():
        if _alias_in_text(alias, normalized):
            return tones[0]
    return primary


def _phase_tone(
    phase: NarrativePhaseName,
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
) -> EmotionalTone:
    phase_tones = _PHASE_TONES_BY_PRIMARY.get(primary, {})
    tone = phase_tones.get(phase)
    if tone is not None:
        return tone
    fallback = {
        "opening": secondary[0] if secondary else primary,
        "development": primary,
        "threshold": "suspension",
        "climax": primary,
        "resolution": _resolution_tone(primary, secondary),
    }
    return fallback[phase]


def _phase_intensity(
    phase: NarrativePhaseName,
    primary: EmotionalTone,
) -> EmotionalIntensity:
    if phase in {"climax", "threshold"}:
        return (
            "high"
            if primary in {"rupture", "ecstasy", "dread", "transformation"}
            else "medium"
        )
    if phase == "resolution":
        return "low" if primary in {"serenity", "clarity"} else "medium"
    return "medium"


def _phase_guidance(phase: NarrativePhaseName, tone: EmotionalTone) -> str:
    return f"Use {tone} as the {phase} emotional state without treating it as objective truth."


def _phase_for_tone(tone: EmotionalTone, index: int) -> NarrativePhaseName:
    if tone in {"serenity", "mystery", "awe", "wonder", "vastness"}:
        return "opening"
    if tone in {"tension", "rupture", "dissolution", "grief", "dread"}:
        return "development"
    if tone == "suspension":
        return "threshold"
    if tone in {"emergence", "ecstasy", "transformation"}:
        return "climax"
    if tone in {"release", "integration", "clarity", "intimacy"}:
        return "resolution"
    return _PHASE_ORDER[min(index, len(_PHASE_ORDER) - 1)]


def _resolution_tone(
    primary: EmotionalTone,
    secondary: tuple[EmotionalTone, ...],
) -> EmotionalTone:
    if "integration" in secondary:
        return "integration"
    if "release" in secondary:
        return "release"
    if primary in {"transformation", "emergence", "dissolution"}:
        return "integration"
    if primary in {"tension", "dread", "rupture", "grief"}:
        return "release"
    return "clarity"


def _structure_evidence(
    families: list[ProceduralFamily],
    modules: list[GenerativeModuleKind],
) -> tuple[str, ...]:
    evidence: list[str] = []
    if families:
        evidence.append("Procedural families: " + ", ".join(families) + ".")
    if modules:
        evidence.append("Generative modules: " + ", ".join(modules) + ".")
    return tuple(evidence[:6])


def _audio_relevant(context: _EmotionalContext) -> bool:
    return bool(
        context.tokens.intersection(_AUDIO_TOKENS)
        or (
            context.creative_translation is not None
            and context.creative_translation.output_modality is not None
            and context.creative_translation.output_modality.value
            in {"audio", "audiovisual"}
        )
        or (
            context.symbolic_narrative is not None
            and context.symbolic_narrative.audio_progression
        )
        or (
            context.generative_structure is not None
            and context.generative_structure.audiovisual_hooks
        )
    )


def _fallback_feeling(
    primary: EmotionalTone,
    context: _EmotionalContext,
) -> str:
    if context.symbolic_narrative is not None:
        return (
            f"Preserve the {context.symbolic_narrative.narrative_archetype} "
            f"arc through {primary}."
        )
    return f"Preserve the user's intended feeling through {primary}."


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _dedupe(values: list | tuple) -> tuple:
    deduped: list[object] = []
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        candidate = value if not isinstance(value, str) else cleaned
        if cleaned and candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


_TONE_ORDER: tuple[EmotionalTone, ...] = (
    "awe",
    "wonder",
    "mystery",
    "serenity",
    "tension",
    "rupture",
    "grief",
    "dissolution",
    "suspension",
    "emergence",
    "ecstasy",
    "clarity",
    "intimacy",
    "vastness",
    "ritual solemnity",
    "playful curiosity",
    "dread",
    "release",
    "transformation",
    "integration",
)
_PHASE_ORDER: tuple[NarrativePhaseName, ...] = (
    "opening",
    "development",
    "threshold",
    "climax",
    "resolution",
)

_TONE_ALIASES: dict[str, tuple[EmotionalTone, ...]] = {
    "anxious": ("tension",),
    "bloom": ("emergence", "wonder"),
    "calm": ("serenity", "release"),
    "ceremony": ("ritual solemnity",),
    "chaos": ("tension", "rupture"),
    "chaotic": ("rupture", "tension"),
    "cosmic": ("awe", "vastness"),
    "curious": ("playful curiosity", "wonder"),
    "dark": ("dread", "mystery"),
    "dissolve": ("dissolution", "rupture"),
    "dissolves": ("dissolution", "rupture"),
    "dissolving": ("dissolution", "rupture"),
    "emerge": ("emergence", "transformation"),
    "emerging": ("emergence",),
    "gentle": ("serenity", "intimacy"),
    "joyful": ("ecstasy", "playful curiosity"),
    "luminous": ("awe", "clarity", "release"),
    "melancholy": ("grief", "mystery"),
    "meditative": ("serenity", "ritual solemnity"),
    "ominous": ("dread", "tension"),
    "peaceful": ("serenity",),
    "phoenix": ("transformation", "rupture", "release", "integration"),
    "playful": ("playful curiosity",),
    "reassemble": ("integration", "release"),
    "reassembling": ("integration", "release"),
    "rebirth": ("transformation", "emergence", "release", "integration"),
    "ritual": ("ritual solemnity", "mystery"),
    "sacred": ("ritual solemnity", "awe"),
    "scatter": ("rupture", "dissolution"),
    "serene": ("serenity",),
    "solemn": ("ritual solemnity", "mystery"),
    "sublime": ("awe", "vastness"),
    "suspended": ("suspension",),
    "threshold": ("suspension", "tension"),
    "transform": ("transformation",),
    "transformation": ("transformation",),
    "wonder": ("wonder", "awe"),
}

_NARRATIVE_TONES: dict[str, tuple[EmotionalTone, ...]] = {
    "descent_and_return": ("tension", "grief", "release", "integration"),
    "death_and_rebirth": (
        "transformation",
        "rupture",
        "suspension",
        "release",
        "integration",
    ),
    "emergence_from_chaos": ("emergence", "tension", "wonder", "clarity"),
    "initiation": ("suspension", "awe", "release"),
    "ascent": ("awe", "emergence", "release", "clarity"),
    "dissolution_and_reintegration": (
        "dissolution",
        "transformation",
        "suspension",
        "integration",
    ),
    "expansion_from_seed_to_cosmos": ("wonder", "awe", "vastness", "emergence"),
    "fragmentation_and_recomposition": (
        "rupture",
        "dissolution",
        "transformation",
        "integration",
    ),
    "threshold_crossing": ("tension", "suspension", "release", "clarity"),
    "spiral_transformation": ("transformation", "wonder", "integration"),
    "mirror_reflection_journey": ("mystery", "intimacy", "clarity"),
    "dark_to_light_transformation": ("dread", "emergence", "release", "clarity"),
    "symbolic_vignette": ("mystery", "wonder", "clarity"),
}

_MOTIF_TONES: dict[SemanticMotifId, tuple[EmotionalTone, ...]] = {
    "seed": ("emergence", "wonder"),
    "spiral": ("transformation", "wonder"),
    "threshold": ("suspension", "tension"),
    "mirror": ("mystery", "intimacy"),
    "void": ("mystery", "dread", "suspension"),
    "center": ("clarity", "serenity"),
    "circumference": ("awe", "vastness"),
    "axis": ("clarity", "ritual solemnity"),
    "descent": ("grief", "tension"),
    "ascent": ("emergence", "release"),
    "fragmentation": ("rupture", "dissolution", "tension"),
    "reintegration": ("integration", "release", "transformation"),
    "wave": ("serenity", "suspension"),
    "lattice": ("clarity", "tension"),
    "network": ("wonder", "clarity"),
    "pearl": ("intimacy", "wonder"),
    "flame": ("ecstasy", "emergence"),
    "root": ("intimacy", "serenity"),
    "tree": ("emergence", "integration"),
    "vessel": ("intimacy", "mystery"),
    "mandala": ("ritual solemnity", "serenity", "awe"),
    "grid": ("clarity", "tension"),
    "swarm": ("tension", "wonder"),
    "orbit": ("awe", "suspension"),
    "pulse": ("tension", "ecstasy"),
    "breath": ("serenity", "release"),
    "gate": ("suspension", "release"),
    "eye": ("mystery", "intimacy"),
    "river": ("serenity", "release"),
    "constellation": ("awe", "vastness", "wonder"),
}

_COMPOSITION_TONES: dict[CompositionPattern, tuple[EmotionalTone, ...]] = {
    "central_emergence": ("emergence", "clarity"),
    "radial_expansion": ("awe", "vastness", "wonder"),
    "spiral_composition": ("transformation", "wonder"),
    "layered_depth": ("mystery", "tension"),
    "field_composition": ("vastness", "wonder"),
    "threshold_composition": ("suspension", "tension", "release"),
    "descent_ascent_composition": ("grief", "release", "integration"),
    "fragmented_recomposition": ("rupture", "integration", "transformation"),
    "mirrored_composition": ("mystery", "intimacy"),
    "orbiting_focal_structure": ("awe", "suspension"),
    "distributed_constellation": ("vastness", "wonder"),
    "minimal_void_and_form_composition": ("mystery", "serenity"),
}

_PROCEDURAL_TONES: dict[ProceduralFamily, tuple[EmotionalTone, ...]] = {
    "fractals": ("wonder", "awe"),
    "recursive_geometry": ("transformation", "wonder"),
    "l_systems": ("emergence", "serenity"),
    "particle_systems": ("rupture", "dissolution", "integration"),
    "boids": ("wonder", "tension"),
    "cellular_automata": ("tension", "clarity"),
    "reaction_diffusion": ("emergence", "mystery"),
    "voronoi_systems": ("clarity", "tension"),
    "noise_fields": ("mystery", "dissolution"),
    "flow_fields": ("serenity", "release"),
    "signed_distance_fields": ("mystery", "clarity"),
    "polar_radial_systems": ("awe", "vastness"),
    "grid_systems": ("clarity",),
    "graph_network_systems": ("wonder", "clarity"),
    "swarm_systems": ("tension", "wonder"),
    "wave_systems": ("serenity", "suspension"),
    "harmonic_oscillators": ("serenity", "ecstasy"),
    "modular_tiling": ("clarity", "ritual solemnity"),
    "sacred_geometry_pattern_systems": ("ritual solemnity", "awe", "serenity"),
}

_MODULE_TONES: dict[GenerativeModuleKind, tuple[EmotionalTone, ...]] = {
    "seed_system": ("emergence",),
    "recursive_module": ("transformation", "wonder"),
    "particle_emitter": ("rupture", "dissolution"),
    "force_field": ("tension",),
    "attractor_field": ("integration", "suspension"),
    "noise_modulation_layer": ("mystery", "dissolution"),
    "symmetry_transform": ("clarity", "ritual solemnity"),
    "tiling_layer": ("clarity",),
    "graph_network_layer": ("wonder", "clarity"),
    "cellular_grid_layer": ("tension", "clarity"),
    "wave_oscillator": ("serenity", "ecstasy"),
    "geometry_reassembly_layer": ("integration", "release"),
    "color_modulation_layer": ("wonder", "awe"),
    "audio_reactive_modulation_layer": ("ecstasy", "tension"),
    "camera_motion_path_hook": ("awe", "suspension"),
}

_PARAMETER_TONES: dict[str, tuple[EmotionalTone, ...]] = {
    "time_phase": ("suspension", "transformation"),
    "recursion_depth": ("wonder", "transformation"),
    "spiral_tightness": ("transformation", "tension"),
    "particle_count": ("rupture", "dissolution"),
    "max_particle_count": ("rupture",),
    "reassembly_speed": ("integration", "release"),
    "fragmentation_amount": ("rupture", "dissolution"),
    "force_strength": ("tension",),
    "attractor_strength": ("integration",),
    "noise_strength": ("mystery", "dissolution"),
    "radial_symmetry": ("ritual solemnity", "awe"),
    "ring_count": ("awe", "vastness"),
    "orbit_speed": ("suspension", "awe"),
    "amplitude": ("ecstasy", "serenity"),
    "frequency": ("tension", "ecstasy"),
    "palette_shift": ("wonder", "release"),
    "audio_gain": ("ecstasy", "tension"),
}

_DEFAULT_SECONDARY_BY_PRIMARY: dict[EmotionalTone, tuple[EmotionalTone, ...]] = {
    "transformation": ("rupture", "suspension", "emergence", "integration", "release"),
    "dissolution": ("rupture", "suspension", "integration"),
    "emergence": ("wonder", "release", "clarity"),
    "ritual solemnity": ("awe", "mystery", "serenity"),
    "serenity": ("release", "clarity", "intimacy"),
    "dread": ("tension", "suspension", "release"),
    "awe": ("wonder", "vastness", "clarity"),
}

_ARC_BY_PRIMARY: dict[EmotionalTone, tuple[str, ...]] = {
    "transformation": (
        "contraction",
        "fragmentation or destabilization",
        "threshold stillness",
        "emergence",
        "integration",
    ),
    "dissolution": (
        "initial form",
        "soft breakdown",
        "suspension",
        "partial return",
        "clearer integration",
    ),
    "serenity": ("quiet arrival", "slow attention", "soft expansion", "calm release"),
    "awe": ("small opening", "scale reveal", "vast expansion", "clear resolution"),
    "dread": ("unease", "pressure", "threshold", "controlled release"),
    "ritual solemnity": (
        "formal opening",
        "measured repetition",
        "threshold pause",
        "luminous resolution",
    ),
}

_PHASE_TONES_BY_PRIMARY: dict[
    EmotionalTone, dict[NarrativePhaseName, EmotionalTone]
] = {
    "transformation": {
        "opening": "tension",
        "development": "rupture",
        "threshold": "suspension",
        "climax": "emergence",
        "resolution": "integration",
    },
    "dissolution": {
        "opening": "mystery",
        "development": "dissolution",
        "threshold": "suspension",
        "climax": "release",
        "resolution": "integration",
    },
    "ritual solemnity": {
        "opening": "mystery",
        "development": "ritual solemnity",
        "threshold": "suspension",
        "climax": "awe",
        "resolution": "serenity",
    },
}

_COLOR_LIGHT_DEFAULT = (
    "Use color and light to make {tone} legible without overexplaining it."
)
_COLOR_LIGHT_BY_TONE: dict[EmotionalTone, str] = {
    "transformation": "Begin muted, pass through low-contrast threshold light, then use luminous reintegration for transformation.",  # noqa: E501
    "rupture": "Use sharp contrast, broken highlights, and fractured color only at rupture beats.",
    "dissolution": "Use softened edges, fading contrast, and dissolving palette shifts for dissolution.",
    "suspension": "Use low-contrast still light and restrained saturation for suspension.",
    "emergence": "Let light rise from local glow to clearer visibility for emergence.",
    "integration": "Use balanced, coherent light and palette convergence for integration.",
    "release": "Use widening brightness and calmer contrast for release.",
    "serenity": "Use low contrast, soft gradients, and stable luminance for serenity.",
    "ritual solemnity": "Use restrained, ceremonial contrast with deliberate luminous accents.",
    "dread": "Use limited light, compressed contrast, and careful shadow density for dread.",
    "awe": "Use scale-revealing light, spacious contrast, and luminous depth for awe.",
}
_MOTION_DEFAULT = (
    "Use motion and rhythm to support {tone} without creating a separate mood."
)
_MOTION_BY_TONE: dict[EmotionalTone, str] = {
    "transformation": "Stage motion as contraction, scatter, pause, acceleration, then calm expansion.",
    "rupture": "Use short chaotic bursts for rupture, separated by readable rests.",
    "dissolution": "Use drift, decay, and dispersal before reassembly cues.",
    "suspension": "Slow motion near threshold moments; reduce rhythm density.",
    "emergence": "Increase motion coherence and acceleration as emergence becomes visible.",
    "integration": "Converge motion paths and reduce jitter for integration.",
    "release": "Open rhythm spacing and ease motion into release.",
    "serenity": "Use slow cadence, gentle drift, and stable loop timing.",
    "ritual solemnity": "Use measured repetition, pauses, and symmetrical timing.",
    "dread": "Use restrained pressure and delayed movement rather than constant chaos.",
    "ecstasy": "Use high energy rhythm with explicit rests to avoid fatigue.",
}

__all__ = [
    "EMOTIONAL_CONSISTENCY_AUTHORITY_BOUNDARY",
    "EmotionalCompositionMapping",
    "EmotionalConsistencyProfile",
    "EmotionalFallbackStrategy",
    "EmotionalIntensity",
    "EmotionalMotifMapping",
    "EmotionalNarrativeMapping",
    "EmotionalParameterMapping",
    "EmotionalPhaseMapping",
    "EmotionalStructureMapping",
    "EmotionalTone",
    "derive_emotional_consistency_profile",
    "emotional_consistency_prompt_lines",
]
