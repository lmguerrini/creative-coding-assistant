"""Bounded Symbolic Narrative Planner for V3 creative workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
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

NarrativeArchetype = Literal[
    "descent_and_return",
    "death_and_rebirth",
    "emergence_from_chaos",
    "initiation",
    "ascent",
    "dissolution_and_reintegration",
    "expansion_from_seed_to_cosmos",
    "fragmentation_and_recomposition",
    "threshold_crossing",
    "spiral_transformation",
    "mirror_reflection_journey",
    "dark_to_light_transformation",
    "symbolic_vignette",
]
NarrativePhaseName = Literal[
    "opening",
    "development",
    "threshold",
    "climax",
    "resolution",
]

SYMBOLIC_NARRATIVE_AUTHORITY_BOUNDARY = (
    "The Symbolic Narrative Planner structures a symbolic artwork journey for "
    "inspection only; it does not generate code, select artifacts, invent "
    "doctrine, claim hidden knowledge, implement HoloMind, choose runtimes, "
    "route providers or models, create variants, or change preview behavior."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_AMBIGUOUS_SYMBOLIC_TOKENS = frozenset(
    {
        "profound",
        "mystical",
        "symbolic",
        "meaningful",
        "deep",
        "evocative",
        "mystery",
        "archetypal",
    }
)
_TRANSFORMATION_TOKENS = frozenset(
    {
        "become",
        "change",
        "emerge",
        "evolve",
        "morph",
        "rebirth",
        "transform",
        "transformation",
        "transition",
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
        "voice",
    }
)


class SymbolicNarrativePhase(BaseModel):
    """One inspectable phase of the planned symbolic journey."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    phase: NarrativePhaseName
    title: str = Field(min_length=1, max_length=120)
    symbolic_function: str = Field(min_length=1, max_length=280)
    emotional_state: str = Field(min_length=1, max_length=220)
    visual_state: str = Field(min_length=1, max_length=260)
    motion_state: str = Field(min_length=1, max_length=240)
    audio_state: str | None = Field(default=None, max_length=220)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=5)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class SymbolicNarrativePlan(BaseModel):
    """Structured pre-generation symbolic/narrative arc."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["symbolic_narrative_planner"] = "symbolic_narrative_planner"
    narrative_archetype: NarrativeArchetype
    symbolic_arc: str = Field(min_length=1, max_length=420)
    opening_phase: SymbolicNarrativePhase
    development_phase: SymbolicNarrativePhase
    threshold_phase: SymbolicNarrativePhase
    climax_phase: SymbolicNarrativePhase
    resolution_phase: SymbolicNarrativePhase
    symbolic_transitions: tuple[str, ...] = Field(min_length=1, max_length=8)
    emotional_progression: tuple[str, ...] = Field(min_length=1, max_length=8)
    visual_progression: tuple[str, ...] = Field(min_length=1, max_length=8)
    motion_progression: tuple[str, ...] = Field(min_length=1, max_length=8)
    audio_progression: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    experiential_goal: str = Field(min_length=1, max_length=420)
    unresolved_narrative_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=SYMBOLIC_NARRATIVE_AUTHORITY_BOUNDARY,
        max_length=560,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)

    @property
    def phases(self) -> tuple[SymbolicNarrativePhase, ...]:
        return (
            self.opening_phase,
            self.development_phase,
            self.threshold_phase,
            self.climax_phase,
            self.resolution_phase,
        )


@dataclass(frozen=True)
class _NarrativeContext:
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
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class _PhaseTemplate:
    title: str
    symbolic_function: str
    emotional_state: str
    visual_state: str
    motion_state: str
    audio_state: str


@dataclass(frozen=True)
class _ArchetypeSpec:
    archetype: NarrativeArchetype
    label: str
    keywords: tuple[str, ...]
    arc: str
    phases: tuple[
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
    ]


def derive_symbolic_narrative_plan(
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
) -> SymbolicNarrativePlan:
    """Plan a symbolic/narrative journey without generating artifacts."""

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
    )
    spec, matched_signals, confidence_score = _select_archetype(context)
    audio_relevant = _audio_relevant(context)
    phases = tuple(
        _phase(
            name,
            template,
            context=context,
            audio_relevant=audio_relevant,
            matched_signals=matched_signals,
        )
        for name, template in zip(
            ("opening", "development", "threshold", "climax", "resolution"),
            spec.phases,
            strict=True,
        )
    )
    unresolved = _unresolved_gaps(
        context,
        archetype=spec.archetype,
        confidence_score=confidence_score,
        audio_relevant=audio_relevant,
    )
    return SymbolicNarrativePlan(
        narrative_archetype=spec.archetype,
        symbolic_arc=_symbolic_arc(spec, context),
        opening_phase=phases[0],
        development_phase=phases[1],
        threshold_phase=phases[2],
        climax_phase=phases[3],
        resolution_phase=phases[4],
        symbolic_transitions=_transitions(phases),
        emotional_progression=_progression(phases, "emotional_state"),
        visual_progression=_progression(phases, "visual_state"),
        motion_progression=_progression(phases, "motion_state"),
        audio_progression=(
            _progression(phases, "audio_state") if audio_relevant else ()
        ),
        experiential_goal=_experiential_goal(context, spec),
        unresolved_narrative_gaps=unresolved,
        hitl_questions=_hitl_questions(unresolved),
        prompt_guidance=_prompt_guidance(
            spec=spec,
            phases=phases,
            unresolved=unresolved,
            audio_relevant=audio_relevant,
        ),
        evidence=_evidence(
            context,
            spec=spec,
            matched_signals=matched_signals,
            confidence_score=confidence_score,
            audio_relevant=audio_relevant,
        ),
    )


def symbolic_narrative_prompt_lines(
    plan: SymbolicNarrativePlan,
) -> tuple[str, ...]:
    """Render symbolic narrative metadata as compact provider guidance."""

    lines = [
        f"Authority boundary: {plan.authority_boundary}",
        f"Narrative archetype: {plan.narrative_archetype}.",
        f"Symbolic arc: {plan.symbolic_arc}",
        f"Experiential goal: {plan.experiential_goal}",
    ]
    for phase in plan.phases:
        lines.append(
            f"{phase.phase} phase: {phase.title}; {phase.symbolic_function}"
        )
        lines.append(f"{phase.phase} visual: {phase.visual_state}")
        lines.append(f"{phase.phase} motion: {phase.motion_state}")
        if phase.audio_state:
            lines.append(f"{phase.phase} audio: {phase.audio_state}")
    lines.extend(f"Symbolic transition: {item}" for item in plan.symbolic_transitions)
    lines.extend(
        f"Unresolved narrative gap: {item}"
        for item in plan.unresolved_narrative_gaps
    )
    lines.extend(f"HITL narrative question: {item}" for item in plan.hitl_questions)
    lines.extend(f"Narrative guidance: {item}" for item in plan.prompt_guidance)
    return tuple(lines[:36])


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
) -> _NarrativeContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_intent.experiential_goal if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
    ]
    if creative_hierarchy is not None:
        parts.extend(
            item.dimension for item in creative_hierarchy.primary_creative_priorities
        )
    if creative_quality_prediction is not None:
        parts.extend(creative_quality_prediction.missing_information)
        parts.extend(
            item.dimension
            for item in creative_quality_prediction.weakest_quality_signals
        )
    text = _normalize(" ".join(parts))
    return _NarrativeContext(
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
        text=text,
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _select_archetype(
    context: _NarrativeContext,
) -> tuple[_ArchetypeSpec, tuple[str, ...], int]:
    scored: list[tuple[int, _ArchetypeSpec, tuple[str, ...]]] = []
    for spec in _ARCHETYPES:
        matches = tuple(
            keyword for keyword in spec.keywords if _matches(context, keyword)
        )
        score = len(matches) * 3
        score += _intent_bonus(context, spec)
        score += _strategy_bonus(context, spec)
        score += _hierarchy_bonus(context, spec)
        scored.append((score, spec, matches))
    scored.sort(key=lambda item: (item[0], item[1].archetype), reverse=True)
    score, spec, matches = scored[0]
    if score <= 1:
        return _SYMBOLIC_VIGNETTE, ("fallback symbolic vignette",), score
    return spec, matches or (f"inferred:{spec.archetype}",), score


def _phase(
    name: NarrativePhaseName,
    template: _PhaseTemplate,
    *,
    context: _NarrativeContext,
    audio_relevant: bool,
    matched_signals: tuple[str, ...],
) -> SymbolicNarrativePhase:
    return SymbolicNarrativePhase(
        phase=name,
        title=template.title,
        symbolic_function=template.symbolic_function,
        emotional_state=_adapt_emotion(template.emotional_state, context),
        visual_state=_adapt_visual(template.visual_state, context),
        motion_state=_adapt_motion(template.motion_state, context),
        audio_state=(
            _adapt_audio(template.audio_state, context) if audio_relevant else None
        ),
        guidance=_phase_guidance(name, template, audio_relevant),
        evidence=matched_signals[:4],
    )


def _symbolic_arc(spec: _ArchetypeSpec, context: _NarrativeContext) -> str:
    subject = _subject_label(context)
    return f"{spec.arc} Anchor the arc in {subject} without unsupported doctrine."


def _experiential_goal(context: _NarrativeContext, spec: _ArchetypeSpec) -> str:
    if context.creative_intent is not None:
        return (
            f"{context.creative_intent.experiential_goal} Shape it as "
            f"{spec.label.lower()}."
        )[:420]
    if context.creative_translation is not None:
        return (
            f"Guide the audience through {spec.label.lower()} while preserving "
            f"{context.creative_translation.creative_intent}."
        )[:420]
    return f"Guide the audience through {spec.label.lower()} with a bounded arc."


def _unresolved_gaps(
    context: _NarrativeContext,
    *,
    archetype: NarrativeArchetype,
    confidence_score: int,
    audio_relevant: bool,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if confidence_score < 5 and _symbolic_intent_active(context):
        gaps.append("Symbolic request lacks a clear narrative journey or archetype.")
    if confidence_score < 3:
        gaps.append("Narrative arc is inferred rather than explicit.")
    if _symbolic_intent_active(context) and not _has_transformation(context):
        gaps.append("Symbolic direction is present, but transformation is unclear.")
    if context.creative_intent is not None:
        gaps.extend(
            item
            for item in context.creative_intent.unresolved_intent_gaps[:3]
            if "narrative" in item.lower()
            or "interaction" in item.lower()
            or "audio" in item.lower()
        )
    if audio_relevant and not _has_any(context, {"rhythm", "tempo", "pulse", "beat"}):
        gaps.append("Audio progression is relevant but rhythm or timing is unclear.")
    if context.creative_quality_prediction is not None:
        gaps.extend(
            item
            for item in context.creative_quality_prediction.missing_information[:3]
            if "symbolic" in item.lower()
            or "narrative" in item.lower()
            or "interaction" in item.lower()
        )
    if archetype == "symbolic_vignette" and _has_any(
        context,
        _AMBIGUOUS_SYMBOLIC_TOKENS,
    ):
        gaps.append("Symbolic vocabulary is broad and needs a chosen journey.")
    return _dedupe(gaps)[:8]


def _hitl_questions(gaps: tuple[str, ...]) -> tuple[str, ...]:
    questions: list[str] = []
    for gap in gaps:
        lowered = gap.lower()
        if "archetype" in lowered or "journey" in lowered:
            questions.append(
                "Which symbolic journey should lead: descent, rebirth, ascent, "
                "threshold crossing, or spiral transformation?"
            )
        elif "transformation" in lowered:
            questions.append("What should visibly transform by the climax?")
        elif "audio" in lowered or "rhythm" in lowered:
            questions.append(
                "What rhythm, silence, pulse, or sonic shift marks the arc?"
            )
        elif "interaction" in lowered:
            questions.append("What narrative state should interaction change?")
        elif "narrative" in lowered:
            questions.append("Should the piece follow a literal story or symbolic arc?")
        elif "symbolic" in lowered:
            questions.append("What visible form should carry the symbolic meaning?")
    return _dedupe(questions)[:6]


def _prompt_guidance(
    *,
    spec: _ArchetypeSpec,
    phases: tuple[SymbolicNarrativePhase, ...],
    unresolved: tuple[str, ...],
    audio_relevant: bool,
) -> tuple[str, ...]:
    guidance = [
        "Use the symbolic narrative as an ordering spine, not as doctrine.",
        f"Preserve the {spec.label.lower()} arc across all generated structure.",
        (
            "Keep opening, development, threshold, climax, and resolution "
            "visibly distinguishable."
        ),
    ]
    guidance.append(
        "Tie motion changes to narrative phase changes before adding extra effects."
    )
    if audio_relevant:
        guidance.append("Let audio progression support the same phase order.")
    if unresolved:
        guidance.append("Ask HITL questions before over-specifying the symbolic arc.")
    guidance.append(f"Resolve toward: {phases[-1].title}.")
    return tuple(guidance[:8])


def _transitions(phases: tuple[SymbolicNarrativePhase, ...]) -> tuple[str, ...]:
    transitions: list[str] = []
    for left, right in zip(phases, phases[1:], strict=False):
        transitions.append(f"{left.title} -> {right.title}: {right.symbolic_function}")
    return tuple(transitions[:8])


def _progression(
    phases: tuple[SymbolicNarrativePhase, ...],
    field_name: str,
) -> tuple[str, ...]:
    values = []
    for phase in phases:
        value = getattr(phase, field_name)
        if value:
            values.append(f"{phase.phase}: {value}")
    return tuple(values[:8])


def _evidence(
    context: _NarrativeContext,
    *,
    spec: _ArchetypeSpec,
    matched_signals: tuple[str, ...],
    confidence_score: int,
    audio_relevant: bool,
) -> tuple[str, ...]:
    evidence = [
        f"Narrative archetype: {spec.archetype}.",
        f"Archetype score: {confidence_score}.",
        "Matched narrative signals: " + ", ".join(matched_signals[:6]) + ".",
        f"Audio progression relevant: {audio_relevant}.",
    ]
    if context.creative_intent is not None:
        evidence.append(
            f"Intent expression: {context.creative_intent.primary_expression}."
        )
    if context.creative_hierarchy is not None:
        evidence.append(
            "Hierarchy priorities: "
            + ", ".join(
                item.dimension
                for item in context.creative_hierarchy.primary_creative_priorities[:3]
            )
            + "."
        )
    if context.creative_quality_prediction is not None:
        evidence.append(
            "Quality readiness: "
            f"{context.creative_quality_prediction.predicted_quality_level} "
            f"{context.creative_quality_prediction.readiness_score}/100."
        )
    return tuple(evidence[:12])


def _intent_bonus(context: _NarrativeContext, spec: _ArchetypeSpec) -> int:
    bonus = 0
    if context.creative_intent is None:
        return bonus
    active = {
        item.name
        for item in context.creative_intent.atomic_dimensions
        if item.explicitness != "absent"
    }
    if "symbolic" in active:
        bonus += 1
    if "narrative" in active and spec.archetype in _LITERAL_ARC_ARCHETYPES:
        bonus += 2
    if "climax_transformation" in active and spec.archetype in _TRANSFORM_ARCHETYPES:
        bonus += 2
    if "emotional" in active and spec.archetype in _EMOTION_ARCHETYPES:
        bonus += 1
    return bonus


def _strategy_bonus(context: _NarrativeContext, spec: _ArchetypeSpec) -> int:
    strategy = (
        context.creative_strategy.primary_strategy
        if context.creative_strategy
        else ""
    )
    if strategy == "sacred_geometry" and spec.archetype in {
        "initiation",
        "spiral_transformation",
        "threshold_crossing",
    }:
        return 2
    if strategy == "particle_cosmology" and spec.archetype in {
        "emergence_from_chaos",
        "expansion_from_seed_to_cosmos",
    }:
        return 2
    if strategy == "fractal_growth" and spec.archetype in {
        "spiral_transformation",
        "expansion_from_seed_to_cosmos",
    }:
        return 2
    return 0


def _hierarchy_bonus(context: _NarrativeContext, spec: _ArchetypeSpec) -> int:
    if context.creative_hierarchy is None:
        return 0
    dimensions = {
        item.dimension
        for item in context.creative_hierarchy.primary_creative_priorities
    }
    bonus = 0
    if "symbolism" in dimensions:
        bonus += 1
    if "narrative" in dimensions and spec.archetype in _LITERAL_ARC_ARCHETYPES:
        bonus += 2
    if "experiential_depth" in dimensions:
        bonus += 1
    return bonus


def _phase_guidance(
    name: NarrativePhaseName,
    template: _PhaseTemplate,
    audio_relevant: bool,
) -> tuple[str, ...]:
    guidance = [
        f"Make the {name} phase visibly distinct: {template.symbolic_function}",
        f"Use visual state as the phase anchor: {template.visual_state}",
        f"Use motion state as the temporal cue: {template.motion_state}",
    ]
    if audio_relevant:
        guidance.append(f"Use audio state as support only: {template.audio_state}")
    return tuple(guidance[:5])


def _adapt_emotion(value: str, context: _NarrativeContext) -> str:
    if context.creative_intent is None:
        return value
    emotional = context.creative_intent.emotional_intent
    if emotional.explicitness == "absent":
        return value
    signals = ", ".join(emotional.signals[:3])
    return f"{value}; preserve explicit emotional cues: {signals}."


def _adapt_visual(value: str, context: _NarrativeContext) -> str:
    if context.creative_translation is None:
        return value
    cues = [
        *context.creative_translation.geometric_references[:2],
        *context.creative_translation.color_material_direction[:2],
    ]
    if not cues:
        return value
    return f"{value}; visual cues: {', '.join(cues)}."


def _adapt_motion(value: str, context: _NarrativeContext) -> str:
    if context.creative_translation is None:
        return value
    cues = context.creative_translation.movement_language[:3]
    if not cues:
        return value
    return f"{value}; movement cues: {', '.join(cues)}."


def _adapt_audio(value: str, context: _NarrativeContext) -> str:
    if context.creative_intent is None:
        return value
    rhythm = context.creative_intent.rhythm_intent
    if rhythm.explicitness == "absent":
        return value
    return f"{value}; rhythm cues: {', '.join(rhythm.signals[:3])}."


def _audio_relevant(context: _NarrativeContext) -> bool:
    if _has_any(context, _AUDIO_TOKENS):
        return True
    if context.creative_plan is not None:
        if context.creative_plan.output_modality.value in {
            "audio",
            "audiovisual",
        }:
            return True
    if context.creative_intent is not None:
        return context.creative_intent.audio_intent.explicitness != "absent"
    return False


def _symbolic_intent_active(context: _NarrativeContext) -> bool:
    if context.creative_intent is None:
        return _has_any(context, _AMBIGUOUS_SYMBOLIC_TOKENS)
    return context.creative_intent.symbolic_intent.explicitness != "absent"


def _has_transformation(context: _NarrativeContext) -> bool:
    if _has_any(context, _TRANSFORMATION_TOKENS):
        return True
    if context.creative_intent is None:
        return False
    return context.creative_intent.climax_transformation_intent.explicitness != "absent"


def _subject_label(context: _NarrativeContext) -> str:
    if context.creative_intent is not None:
        return context.creative_intent.primary_expression
    if context.creative_translation is not None:
        return context.creative_translation.creative_intent
    return "the user's stated motif"


def _matches(context: _NarrativeContext, keyword: str) -> bool:
    normalized = _normalize(keyword)
    if " " in normalized:
        return normalized in context.text
    return normalized in context.tokens


def _has_any(context: _NarrativeContext, values: frozenset[str] | set[str]) -> bool:
    return bool(context.tokens.intersection(values))


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _spec(
    archetype: NarrativeArchetype,
    label: str,
    keywords: tuple[str, ...],
    arc: str,
    phases: tuple[
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
        _PhaseTemplate,
    ],
) -> _ArchetypeSpec:
    return _ArchetypeSpec(
        archetype=archetype,
        label=label,
        keywords=keywords,
        arc=arc,
        phases=phases,
    )


def _phases(
    opening: tuple[str, str, str, str, str, str],
    development: tuple[str, str, str, str, str, str],
    threshold: tuple[str, str, str, str, str, str],
    climax: tuple[str, str, str, str, str, str],
    resolution: tuple[str, str, str, str, str, str],
) -> tuple[
    _PhaseTemplate,
    _PhaseTemplate,
    _PhaseTemplate,
    _PhaseTemplate,
    _PhaseTemplate,
]:
    return tuple(
        _PhaseTemplate(*item)
        for item in (opening, development, threshold, climax, resolution)
    )


_ARCHETYPES: tuple[_ArchetypeSpec, ...] = (
    _spec(
        "death_and_rebirth",
        "Death And Rebirth",
        ("death", "rebirth", "born", "renewal", "resurrection", "phoenix"),
        "Move from symbolic ending through dissolution into renewed form.",
        _phases(
            (
                "Descent Into Darkness",
                "Introduce loss, contraction, or the ending of an old form.",
                "solemn contraction and focused gravity",
                "low light, compressed geometry, and dark negative space",
                "slow inward pull or sinking motion",
                "low drone, sparse pulse, or near silence",
            ),
            (
                "Fragmentation Of Form",
                "Break the initial order into shards, particles, or unstable signs.",
                "uncertainty, grief, or charged suspension",
                "fractured symmetry, dispersed marks, and broken outlines",
                "erratic drift, shedding, or dissolving motion",
                "grain, fracture, filtered noise, or thinning rhythm",
            ),
            (
                "Silent Threshold",
                "Hold a liminal pause before a new structure appears.",
                "stillness, vulnerability, and expectancy",
                "minimal threshold line, dim core, or suspended center",
                "near-still pause, hover, or breath-like hold",
                "rest, silence, or a single sustained tone",
            ),
            (
                "Emergence Of New Order",
                "Let light, symmetry, or a new motif reorganize the field.",
                "release, awe, and sudden recognition",
                "bright reordering, coherent geometry, and rising contrast",
                "outward bloom, rotation, or constructive recomposition",
                "clear pulse, harmonic opening, or rising resonance",
            ),
            (
                "Reintegrated Form",
                "Resolve as transformed wholeness rather than a return to baseline.",
                "calm recognition and integrated strength",
                "stable symmetry with traces of the earlier fracture",
                "slow expansion into a balanced final rhythm",
                "settled tone, soft cadence, or warm residual pulse",
            ),
        ),
    ),
    _spec(
        "descent_and_return",
        "Descent And Return",
        ("descent", "underworld", "return", "below", "depth", "submerge"),
        "Travel downward into depth, confront a threshold, then return changed.",
        _phases(
            (
                "Surface Departure",
                "Begin from a recognizable surface before entering depth.",
                "curiosity and unease",
                "clear surface plane with a visible point of departure",
                "gradual downward drift",
                "steady pulse descending in register",
            ),
            (
                "Deepening Passage",
                "Move through layered darkness or density.",
                "pressure, uncertainty, and concentration",
                "stacked layers, dim gradients, and narrowing space",
                "sinking, looping, or slowed traversal",
                "muffled texture or reduced high frequencies",
            ),
            (
                "Depth Threshold",
                "Suspend the journey at the deepest boundary.",
                "austere stillness and attention",
                "thin boundary, central void, or compressed horizon",
                "hovering pause at lowest intensity",
                "near silence or held bass tone",
            ),
            (
                "Return Current",
                "Let the work rise with evidence of what changed below.",
                "relief and renewed force",
                "rising light, widened space, and reappearing motif",
                "upward current or spiraling ascent",
                "pulse returns with clearer overtones",
            ),
            (
                "Changed Surface",
                "End at the surface with transformed marks or symmetry.",
                "quiet integration",
                "open field with transformed residue",
                "slow settling into balanced motion",
                "resolved cadence with a faint depth trace",
            ),
        ),
    ),
    _spec(
        "emergence_from_chaos",
        "Emergence From Chaos",
        ("chaos", "emerge", "emergence", "order", "turbulence", "noise"),
        "Begin in unordered field behavior and reveal coherent structure.",
        _phases(
            (
                "Unordered Field",
                "Open with unstable, unformed material.",
                "restless possibility",
                "noisy marks, scattered points, and low hierarchy",
                "random drift or jitter",
                "textured noise or irregular pulse",
            ),
            (
                "Pattern Pressure",
                "Let repeated tendencies start to appear inside chaos.",
                "curiosity and gathering focus",
                "partial alignments and early repeating forms",
                "clusters begin to orbit or align",
                "irregular rhythm starts to imply meter",
            ),
            (
                "Ordering Threshold",
                "Cross from accident into visible rule.",
                "suspense and recognition",
                "one clear axis, center, or rule emerges",
                "motion synchronizes briefly",
                "pulse locks into a simple pattern",
            ),
            (
                "Coherent Emergence",
                "Reveal the governing structure at full strength.",
                "awe and clarity",
                "organized field, constellation, or lattice",
                "flow becomes coordinated",
                "rhythm opens into fuller resonance",
            ),
            (
                "Living Order",
                "Resolve with order that still carries generative variation.",
                "calm vitality",
                "stable system with subtle residual noise",
                "gentle self-renewing motion",
                "steady bed with small variations",
            ),
        ),
    ),
    _spec(
        "initiation",
        "Initiation",
        ("initiation", "initiatic", "rite", "ritual", "ordeal", "mystery"),
        (
            "Guide the viewer through invitation, trial, threshold, "
            "revelation, and integration."
        ),
        _phases(
            (
                "Invitation",
                "Present the symbolic gate or call into the work.",
                "anticipation and reverence",
                "threshold frame, central sign, or ritual boundary",
                "slow approach toward a focal center",
                "low invocation tone or sparse pulse",
            ),
            (
                "Trial",
                "Complicate the motif through pressure or obscuration.",
                "tension and commitment",
                "veiled geometry, obstructed path, or layered signs",
                "testing oscillation or repeated approach",
                "tightened rhythm or darker timbre",
            ),
            (
                "Crossing",
                "Mark the liminal crossing from old state to new state.",
                "focus, suspension, and risk",
                "gate line, aperture, or high-contrast boundary",
                "single decisive pass through a threshold",
                "silence, held tone, or one clear accent",
            ),
            (
                "Revelation",
                "Reveal the reorganized symbolic order.",
                "wonder and recognition",
                "clear symbol, bright geometry, or opened pattern",
                "expansion from center with stable rhythm",
                "harmonic brightening or widened pulse",
            ),
            (
                "Integration",
                "Stabilize the revealed order into inhabitable form.",
                "settled meaning and presence",
                "balanced field containing the threshold trace",
                "slow cyclical motion",
                "soft cadence or grounded drone",
            ),
        ),
    ),
    _spec(
        "ascent",
        "Ascent",
        ("ascent", "ascend", "rise", "rising", "elevate", "sky"),
        "Move from grounded density toward elevation, lightness, and expanded view.",
        _phases(
            (
                "Grounded Base",
                "Start in weight, horizon, or low field density.",
                "grounded focus",
                "low composition, dense base, and restrained light",
                "slow lift begins from below",
                "low register pulse",
            ),
            (
                "Rising Path",
                "Build vertical or radial lift through repeated motion.",
                "effort and aspiration",
                "ascending lines, widening intervals, and brighter traces",
                "upward flow or climbing oscillation",
                "gradual pitch or intensity rise",
            ),
            (
                "High Threshold",
                "Pause at the edge of expansion.",
                "breathless suspension",
                "thin high boundary or open aperture",
                "hover near maximum height",
                "held tone or suspended beat",
            ),
            (
                "Open Height",
                "Reveal expanded scale or luminous view.",
                "release and spaciousness",
                "wide field, bright upper space, and reduced density",
                "floating or outward drift",
                "open resonance or widened stereo image",
            ),
            (
                "Sustained Elevation",
                "Resolve in stable lightness without losing structure.",
                "quiet uplift",
                "balanced high composition with clear focal hierarchy",
                "slow buoyant cycle",
                "soft sustained cadence",
            ),
        ),
    ),
    _spec(
        "dissolution_and_reintegration",
        "Dissolution And Reintegration",
        ("dissolve", "dissolution", "reintegration", "melt", "reintegrate"),
        "Dissolve a coherent form and return it as a changed integrated whole.",
        _phases(
            (
                "Coherent Form",
                "Begin with a legible symbolic structure.",
                "stability and recognition",
                "clear geometry or motif with defined edges",
                "slow stable cycle",
                "simple grounded pulse",
            ),
            (
                "Dissolving Edges",
                "Let boundaries soften and identity diffuse.",
                "uncertainty and release",
                "blurred outlines, diffusion, and fading contrast",
                "melting, dispersal, or soft turbulence",
                "washed texture or smeared rhythm",
            ),
            (
                "Formless Threshold",
                "Hold the work in near-formless suspension.",
                "openness and vulnerability",
                "mist, field, or near-abstract residue",
                "slow suspended drift",
                "ambient bed or near silence",
            ),
            (
                "Rebinding Pattern",
                "Reassemble traces into a new relation.",
                "renewed attention",
                "returning edges and fresh alignments",
                "magnetized gathering or constructive orbit",
                "pulse re-enters with clearer contour",
            ),
            (
                "Integrated Whole",
                "Resolve as changed unity with visible memory of dissolution.",
                "acceptance and coherence",
                "stable form carrying softened traces",
                "balanced cyclical motion",
                "warm settled resonance",
            ),
        ),
    ),
    _spec(
        "expansion_from_seed_to_cosmos",
        "Expansion From Seed To Cosmos",
        ("seed", "cosmos", "cosmic", "universe", "expand", "expansion"),
        "Grow from a compact seed into a large-scale field or cosmos.",
        _phases(
            (
                "Seed Point",
                "Begin with one compressed generative origin.",
                "quiet potential",
                "single point, small core, or minimal seed geometry",
                "subtle pulse inside a tight radius",
                "soft click, pulse, or quiet seed tone",
            ),
            (
                "First Growth",
                "Let the seed branch, replicate, or radiate.",
                "curiosity and growth",
                "rings, branches, or first particles around the origin",
                "radial expansion or branching drift",
                "pulse gains layers",
            ),
            (
                "Scale Threshold",
                "Shift from local motif to expansive system.",
                "wonder and threshold awareness",
                "visible jump in scale, density, or field depth",
                "accelerated expansion or camera pullback",
                "frequency range opens",
            ),
            (
                "Cosmic Bloom",
                "Reveal the full field as cosmos or living system.",
                "awe and spaciousness",
                "constellation, galaxy, or large generative field",
                "orbital flow or broad expansion",
                "wide resonance and layered pulse",
            ),
            (
                "Living Cosmos",
                "Resolve with a stable field still connected to the seed.",
                "belonging and continuity",
                "large field with visible origin echo",
                "slow orbit around the originating pattern",
                "settled cosmic bed",
            ),
        ),
    ),
    _spec(
        "fragmentation_and_recomposition",
        "Fragmentation And Recomposition",
        ("fragment", "fragmentation", "shatter", "shard", "recompose", "recomposition"),
        "Break the image into fragments and recompose them into new order.",
        _phases(
            (
                "Whole Before Fracture",
                "Show an initial form before disruption.",
                "calm before rupture",
                "clear silhouette, symbol, or grid",
                "stable low-amplitude motion",
                "simple pulse",
            ),
            (
                "Fracture",
                "Split the form into fragments or shards.",
                "shock and instability",
                "broken outlines, shards, and displaced pieces",
                "sharp scattering or angular displacement",
                "stutter, crackle, or broken rhythm",
            ),
            (
                "Suspended Pieces",
                "Hold fragments before recomposition.",
                "suspended uncertainty",
                "floating pieces with visible distance between them",
                "slow hover or micro-jitter",
                "thin suspended texture",
            ),
            (
                "Recomposition",
                "Draw fragments into a new relation.",
                "focus and reconstruction",
                "aligning shards and renewed symmetry",
                "magnetic gathering or snapping into place",
                "rhythm coheres",
            ),
            (
                "New Composite",
                "Resolve as a composite form that preserves fracture memory.",
                "integrated resilience",
                "whole form with visible seams or traces",
                "settled oscillation around the new structure",
                "resolved pulse with residual texture",
            ),
        ),
    ),
    _spec(
        "threshold_crossing",
        "Threshold Crossing",
        ("threshold", "crossing", "gate", "portal", "door", "liminal"),
        "Move from one symbolic state through a boundary into another state.",
        _phases(
            (
                "Before The Gate",
                "Establish the state before crossing.",
                "anticipation",
                "foreground boundary, gate, or distant aperture",
                "approach motion",
                "measured pulse",
            ),
            (
                "Approach",
                "Increase focus and pressure near the threshold.",
                "tension and concentration",
                "tight framing and increased contrast at the boundary",
                "accelerating approach or repeated near-crossing",
                "tightened rhythmic pattern",
            ),
            (
                "Crossing Moment",
                "Mark the exact transition between states.",
                "suspension and decision",
                "bright seam, cut, aperture, or inverted edge",
                "single decisive crossing gesture",
                "accent, silence, or tonal flip",
            ),
            (
                "Other Side",
                "Reveal changed rules beyond the boundary.",
                "surprise and recognition",
                "altered palette, geometry, or spatial law",
                "new motion rule appears",
                "new timbre or rhythm enters",
            ),
            (
                "New Orientation",
                "Settle into the transformed state.",
                "orientation and integration",
                "stable composition in the new rules",
                "slow exploratory cycle",
                "settled version of the new motif",
            ),
        ),
    ),
    _spec(
        "spiral_transformation",
        "Spiral Transformation",
        ("spiral", "coil", "helix", "orbit", "recursive", "mandala"),
        "Transform through recurring returns that change scale, color, or form.",
        _phases(
            (
                "Central Seed",
                "Begin from a compact center or first turn.",
                "quiet concentration",
                "small center, seed ring, or initial spiral turn",
                "slow rotation around a tight center",
                "small repeated pulse",
            ),
            (
                "Widening Turn",
                "Expand the spiral while changing visible state.",
                "curiosity and momentum",
                "wider rings, added arms, and growing contrast",
                "orbit widens with each loop",
                "pulse layers accumulate",
            ),
            (
                "Inversion Point",
                "Mark the turn where the spiral changes logic.",
                "threshold tension",
                "color inversion, direction shift, or compressed center flash",
                "rotation pauses, reverses, or changes radius",
                "accent or rhythmic break",
            ),
            (
                "Transformed Spiral",
                "Reveal the spiral at higher order or intensity.",
                "recognition and awe",
                "expanded spiral, mandala, or recursive geometry",
                "confident rotation and radial breathing",
                "fuller rhythmic cycle",
            ),
            (
                "Stabilized Orbit",
                "Resolve as a stable cycle that remembers the path.",
                "calm continuity",
                "balanced radial field with visible phase echoes",
                "slow orbit at equilibrium",
                "settled pulse",
            ),
        ),
    ),
    _spec(
        "mirror_reflection_journey",
        "Mirror Reflection Journey",
        ("mirror", "reflection", "reflect", "symmetry", "double", "twin"),
        "Move through doubling, recognition, distortion, and integrated reflection.",
        _phases(
            (
                "First Image",
                "Introduce the initial self or motif.",
                "recognition",
                "single motif with clean orientation",
                "small stable motion",
                "simple tone",
            ),
            (
                "Mirrored Double",
                "Introduce reflection or doubling.",
                "curiosity and tension",
                "symmetrical double, waterline, or reflective split",
                "counter-motion or mirrored orbit",
                "echo or call-response",
            ),
            (
                "Distorted Reflection",
                "Disrupt the mirror so difference becomes visible.",
                "disorientation",
                "warped reflection, offset symmetry, or fragmented double",
                "phase offset or ripple",
                "detuned echo",
            ),
            (
                "Recognition",
                "Let the motif see or absorb its reflected difference.",
                "insight",
                "aligned halves or bright reflected core",
                "motions synchronize",
                "echo resolves into harmony",
            ),
            (
                "Integrated Reflection",
                "Resolve with unity that preserves difference.",
                "balanced self-awareness",
                "stable bilateral or reflective composition",
                "gentle mirrored cycle",
                "soft call-response cadence",
            ),
        ),
    ),
    _spec(
        "dark_to_light_transformation",
        "Dark-To-Light Transformation",
        ("dark", "darkness", "light", "dawn", "illumination", "glow"),
        "Move from obscurity through transition into revealed luminosity.",
        _phases(
            (
                "Obscured Beginning",
                "Begin in darkness or low visibility.",
                "mystery and restraint",
                "dim field, hidden motif, or low-contrast silhouette",
                "slow minimal motion",
                "dark drone or quiet texture",
            ),
            (
                "First Glimmer",
                "Introduce small light signals or emerging edges.",
                "hope and curiosity",
                "small highlights, rim light, or faint geometry",
                "gentle flicker or pulse",
                "soft bright tone enters",
            ),
            (
                "Luminous Threshold",
                "Hold the moment where light starts to govern the field.",
                "expectancy",
                "sharp light boundary or glowing aperture",
                "pulse synchronizes with the light",
                "held harmonic or suspended shimmer",
            ),
            (
                "Revelation In Light",
                "Reveal form, symbol, or space through illumination.",
                "awe and clarity",
                "bright structure, clear palette, and revealed depth",
                "expanding glow or radiant motion",
                "open resonance",
            ),
            (
                "Settled Illumination",
                "Resolve with durable luminosity rather than glare.",
                "peace and presence",
                "stable glow, readable hierarchy, and balanced contrast",
                "slow breathing shimmer",
                "soft cadence",
            ),
        ),
    ),
)

_SYMBOLIC_VIGNETTE = _spec(
    "symbolic_vignette",
    "Symbolic Vignette",
    ("symbolic", "profound", "evocative", "mystery"),
    "Hold a symbolic atmosphere in phases without claiming a specific doctrine.",
    _phases(
        (
            "Initial Motif",
            "Introduce the symbolic material without overexplaining it.",
            "open curiosity",
            "one clear motif or atmospheric field",
            "slow reveal",
            "quiet tonal bed",
        ),
        (
            "Amplified Motif",
            "Repeat or vary the motif so its behavior becomes legible.",
            "gathering attention",
            "stronger contrast and repeated motif cues",
            "gentle variation or orbit",
            "subtle pulse",
        ),
        (
            "Ambiguous Threshold",
            "Hold the motif at a possible turning point.",
            "uncertainty",
            "boundary, pause, or suspended center",
            "brief stillness",
            "held tone or silence",
        ),
        (
            "Symbolic Intensification",
            "Make the motif most visible without inventing unsupported meaning.",
            "heightened presence",
            "brighter motif, denser field, or focused geometry",
            "expanded motion",
            "fuller texture",
        ),
        (
            "Open Resolution",
            "Resolve as an interpretable but not doctrinal image.",
            "quiet openness",
            "stable final composition",
            "slow settling",
            "soft cadence",
        ),
    ),
)

_ARCHETYPES = (*_ARCHETYPES, _SYMBOLIC_VIGNETTE)

_LITERAL_ARC_ARCHETYPES = {
    "death_and_rebirth",
    "descent_and_return",
    "initiation",
    "threshold_crossing",
}
_TRANSFORM_ARCHETYPES = {
    "death_and_rebirth",
    "dissolution_and_reintegration",
    "fragmentation_and_recomposition",
    "spiral_transformation",
    "dark_to_light_transformation",
}
_EMOTION_ARCHETYPES = {
    "death_and_rebirth",
    "descent_and_return",
    "initiation",
    "dark_to_light_transformation",
}
