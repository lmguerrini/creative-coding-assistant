"""Bounded Creative Composition Planner for V3 workflows."""

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
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

CompositionPattern = Literal[
    "central_emergence",
    "radial_expansion",
    "spiral_composition",
    "layered_depth",
    "field_composition",
    "threshold_composition",
    "descent_ascent_composition",
    "fragmented_recomposition",
    "mirrored_composition",
    "orbiting_focal_structure",
    "distributed_constellation",
    "minimal_void_and_form_composition",
]

CREATIVE_COMPOSITION_AUTHORITY_BOUNDARY = (
    "The Creative Composition Planner structures artwork organization for "
    "inspection only; it does not generate code, implement a Composition "
    "Engine, select artifacts, auto-select runtimes, route providers or "
    "models, change preview behavior, run repair loops, or implement HoloMind."
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
        "tone",
    }
)
_VIEWPOINT_TOKENS = frozenset(
    {"3d", "camera", "depth", "perspective", "scene", "spatial", "viewpoint"}
)


class CreativeCompositionPlan(BaseModel):
    """Structured pre-generation composition plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_composition_planner"] = "creative_composition_planner"
    composition_pattern: CompositionPattern
    primary_focal_point: str = Field(min_length=1, max_length=260)
    secondary_focal_elements: tuple[str, ...] = Field(min_length=1, max_length=8)
    spatial_organization: str = Field(min_length=1, max_length=320)
    foreground_background_relationship: str = Field(min_length=1, max_length=320)
    visual_hierarchy: tuple[str, ...] = Field(min_length=1, max_length=8)
    density_plan: str = Field(min_length=1, max_length=320)
    rhythm_plan: str = Field(min_length=1, max_length=320)
    balance_plan: str = Field(min_length=1, max_length=320)
    symmetry_asymmetry_guidance: str = Field(min_length=1, max_length=320)
    depth_layering_guidance: str = Field(min_length=1, max_length=320)
    transition_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    camera_viewpoint_guidance: str | None = Field(default=None, max_length=260)
    audiovisual_composition_notes: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    composition_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    unresolved_composition_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CREATIVE_COMPOSITION_AUTHORITY_BOUNDARY,
        max_length=560,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _CompositionContext:
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
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class _PatternSpec:
    pattern: CompositionPattern
    label: str
    keywords: tuple[str, ...]
    primary_focal_point: str
    spatial_organization: str
    foreground_background_relationship: str
    density_plan: str
    rhythm_plan: str
    balance_plan: str
    symmetry_guidance: str
    depth_guidance: str
    camera_guidance: str | None


def derive_creative_composition_plan(
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
) -> CreativeCompositionPlan:
    """Plan artwork organization without generating compositional structure."""

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
    )
    spec, matches, score = _select_pattern(context)
    audio_relevant = _audio_relevant(context)
    unresolved = _unresolved_gaps(
        context,
        score=score,
        audio_relevant=audio_relevant,
    )
    return CreativeCompositionPlan(
        composition_pattern=spec.pattern,
        primary_focal_point=_primary_focal_point(spec, context),
        secondary_focal_elements=_secondary_focal_elements(context),
        spatial_organization=_adapt_with_visual_cues(
            spec.spatial_organization,
            context,
        ),
        foreground_background_relationship=(
            _foreground_background_relationship(spec, context)
        ),
        visual_hierarchy=_visual_hierarchy(context, spec),
        density_plan=_density_plan(spec, context),
        rhythm_plan=_rhythm_plan(spec, context, audio_relevant=audio_relevant),
        balance_plan=_balance_plan(spec, context),
        symmetry_asymmetry_guidance=_symmetry_guidance(spec, context),
        depth_layering_guidance=_depth_guidance(spec, context),
        transition_guidance=_transition_guidance(spec, context),
        camera_viewpoint_guidance=_camera_guidance(spec, context),
        audiovisual_composition_notes=_audiovisual_notes(
            context,
            audio_relevant=audio_relevant,
        ),
        composition_risks=_composition_risks(context, spec),
        unresolved_composition_gaps=unresolved,
        hitl_questions=_hitl_questions(unresolved),
        prompt_guidance=_prompt_guidance(spec, unresolved),
        evidence=_evidence(
            context,
            spec=spec,
            matches=matches,
            score=score,
            audio_relevant=audio_relevant,
        ),
    )


def creative_composition_prompt_lines(
    plan: CreativeCompositionPlan,
) -> tuple[str, ...]:
    """Render composition metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {plan.authority_boundary}",
        f"Composition pattern: {plan.composition_pattern}.",
        f"Primary focal point: {plan.primary_focal_point}",
        f"Spatial organization: {plan.spatial_organization}",
        "Foreground/background: " + plan.foreground_background_relationship,
        f"Density plan: {plan.density_plan}",
        f"Rhythm plan: {plan.rhythm_plan}",
        f"Balance plan: {plan.balance_plan}",
        "Symmetry/asymmetry: " + plan.symmetry_asymmetry_guidance,
        "Depth/layering: " + plan.depth_layering_guidance,
    ]
    if plan.camera_viewpoint_guidance:
        lines.append(f"Camera/viewpoint: {plan.camera_viewpoint_guidance}")
    lines.extend(
        f"Secondary focal element: {item}" for item in plan.secondary_focal_elements
    )
    lines.extend(f"Visual hierarchy: {item}" for item in plan.visual_hierarchy)
    lines.extend(f"Composition transition: {item}" for item in plan.transition_guidance)
    lines.extend(
        f"Audiovisual composition note: {item}"
        for item in plan.audiovisual_composition_notes
    )
    lines.extend(f"Composition risk: {item}" for item in plan.composition_risks)
    lines.extend(
        f"Unresolved composition gap: {item}"
        for item in plan.unresolved_composition_gaps
    )
    lines.extend(f"HITL composition question: {item}" for item in plan.hitl_questions)
    lines.extend(f"Composition guidance: {item}" for item in plan.prompt_guidance)
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
    symbolic_narrative: SymbolicNarrativePlan | None,
) -> _CompositionContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
        symbolic_narrative.narrative_archetype if symbolic_narrative else "",
        symbolic_narrative.symbolic_arc if symbolic_narrative else "",
    ]
    if creative_translation is not None:
        parts.extend(creative_translation.geometric_references)
        parts.extend(creative_translation.movement_language)
        parts.extend(creative_translation.color_material_direction)
    if creative_hierarchy is not None:
        parts.extend(
            item.dimension for item in creative_hierarchy.primary_creative_priorities
        )
    if creative_quality_prediction is not None:
        parts.extend(creative_quality_prediction.missing_information)
    text = _normalize(" ".join(parts))
    return _CompositionContext(
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
        text=text,
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _select_pattern(
    context: _CompositionContext,
) -> tuple[_PatternSpec, tuple[str, ...], int]:
    scored: list[tuple[int, _PatternSpec, tuple[str, ...]]] = []
    for spec in _PATTERNS:
        matches = tuple(
            keyword for keyword in spec.keywords if _matches(context, keyword)
        )
        score = len(matches) * 3
        score += _narrative_bonus(context, spec)
        score += _strategy_bonus(context, spec)
        score += _technique_bonus(context, spec)
        scored.append((score, spec, matches))
    scored.sort(key=lambda item: (item[0], item[1].pattern), reverse=True)
    score, spec, matches = scored[0]
    if score <= 1:
        return _PATTERN_BY_ID["minimal_void_and_form_composition"], (
            "fallback minimal composition",
        ), score
    return spec, matches or (f"inferred:{spec.pattern}",), score


def _primary_focal_point(
    spec: _PatternSpec,
    context: _CompositionContext,
) -> str:
    subject = _subject_label(context)
    return f"{spec.primary_focal_point} Anchor it around {subject}."


def _secondary_focal_elements(context: _CompositionContext) -> tuple[str, ...]:
    elements: list[str] = []
    if context.creative_translation is not None:
        elements.extend(context.creative_translation.geometric_references[:2])
        elements.extend(context.creative_translation.color_material_direction[:2])
        elements.extend(context.creative_translation.movement_language[:2])
    if context.symbolic_narrative is not None:
        elements.extend(
            (
                f"threshold cue: {context.symbolic_narrative.threshold_phase.title}",
                f"resolution cue: {context.symbolic_narrative.resolution_phase.title}",
            )
        )
    if not elements and context.creative_intent is not None:
        elements.append(context.creative_intent.primary_expression)
    return _dedupe(elements)[:8] or ("supporting negative space",)


def _foreground_background_relationship(
    spec: _PatternSpec,
    context: _CompositionContext,
) -> str:
    value = spec.foreground_background_relationship
    if context.symbolic_narrative is not None:
        return (
            f"{value} Let foreground/background changes track "
            f"{context.symbolic_narrative.narrative_archetype}."
        )
    return value


def _visual_hierarchy(
    context: _CompositionContext,
    spec: _PatternSpec,
) -> tuple[str, ...]:
    hierarchy = [f"Lead with {spec.label.lower()} as the layout spine."]
    if context.creative_hierarchy is not None:
        hierarchy.extend(
            f"Protect {item.dimension} before secondary composition details."
            for item in context.creative_hierarchy.primary_creative_priorities[:4]
        )
    if context.creative_strategy is not None:
        hierarchy.append(
            f"Let {context.creative_strategy.primary_strategy} remain visible."
        )
    return _dedupe(hierarchy)[:8]


def _density_plan(spec: _PatternSpec, context: _CompositionContext) -> str:
    value = spec.density_plan
    if (
        context.creative_tradeoffs is not None
        and context.creative_tradeoffs.performance_concerns
    ):
        return f"{value} Cap density where performance concerns appear."
    return value


def _rhythm_plan(
    spec: _PatternSpec,
    context: _CompositionContext,
    *,
    audio_relevant: bool,
) -> str:
    value = spec.rhythm_plan
    if context.symbolic_narrative is not None:
        value = f"{value} Follow the symbolic phase order for rhythm changes."
    if audio_relevant:
        value += " Align visual rhythm with audio pulse or silence cues."
    return value


def _balance_plan(spec: _PatternSpec, context: _CompositionContext) -> str:
    if context.creative_constraint_priorities is not None:
        protected = context.creative_constraint_priorities.non_negotiable_constraints
        if protected:
            return (
                f"{spec.balance_plan} Do not rebalance by weakening "
                f"{protected[0].category}."
            )
    return spec.balance_plan


def _symmetry_guidance(spec: _PatternSpec, context: _CompositionContext) -> str:
    if context.creative_strategy is not None and (
        context.creative_strategy.primary_strategy == "sacred_geometry"
    ):
        return f"{spec.symmetry_guidance} Preserve geometric legibility."
    return spec.symmetry_guidance


def _depth_guidance(spec: _PatternSpec, context: _CompositionContext) -> str:
    if context.creative_plan is not None:
        if context.creative_plan.output_modality.value in {
            "audiovisual",
            "interactive",
        }:
            return f"{spec.depth_guidance} Keep interactive layers readable."
    return spec.depth_guidance


def _transition_guidance(
    spec: _PatternSpec,
    context: _CompositionContext,
) -> tuple[str, ...]:
    if context.symbolic_narrative is not None:
        return tuple(
            f"Compose transition through narrative phase: {item}"
            for item in context.symbolic_narrative.symbolic_transitions[:4]
        )
    return (
        f"Start with {spec.primary_focal_point}",
        f"Develop through {spec.spatial_organization}",
        "Resolve with a stable final hierarchy.",
    )


def _camera_guidance(
    spec: _PatternSpec,
    context: _CompositionContext,
) -> str | None:
    if spec.camera_guidance is not None:
        return spec.camera_guidance
    if _has_any(context, _VIEWPOINT_TOKENS):
        return "Use a stable viewpoint that preserves focal hierarchy."
    return None


def _audiovisual_notes(
    context: _CompositionContext,
    *,
    audio_relevant: bool,
) -> tuple[str, ...]:
    if not audio_relevant:
        return ()
    notes = [
        "Use audio as a compositional timing cue, not as a new feature layer.",
        "Let density, spacing, or focal intensity respond to pulse changes.",
    ]
    if context.symbolic_narrative is not None:
        notes.append(
            "Map sonic changes onto opening, threshold, climax, and resolution."
        )
    return tuple(notes[:6])


def _composition_risks(
    context: _CompositionContext,
    spec: _PatternSpec,
) -> tuple[str, ...]:
    risks: list[str] = []
    if context.creative_quality_prediction is not None:
        risks.extend(
            item.summary
            for item in context.creative_quality_prediction.weakest_quality_signals[:2]
        )
        risks.extend(context.creative_quality_prediction.likely_failure_modes[:2])
    if context.creative_tradeoffs is not None:
        risks.extend(context.creative_tradeoffs.performance_concerns[:2])
        risks.extend(context.creative_tradeoffs.fidelity_risks[:2])
    if spec.pattern in {"field_composition", "distributed_constellation"}:
        risks.append("Distributed layouts can lose a readable focal hierarchy.")
    return _dedupe(risks)[:8]


def _unresolved_gaps(
    context: _CompositionContext,
    *,
    score: int,
    audio_relevant: bool,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if score < 5:
        gaps.append("Composition pattern is inferred rather than explicit.")
    if context.creative_translation is None or not (
        context.creative_translation.geometric_references
        or context.creative_translation.color_material_direction
    ):
        gaps.append("Primary visual motif, palette, or layout cue is unclear.")
    if context.creative_quality_prediction is not None:
        gaps.extend(
            item
            for item in context.creative_quality_prediction.missing_information[:3]
            if "palette" in item.lower()
            or "subject" in item.lower()
            or "visual" in item.lower()
            or "interaction" in item.lower()
        )
    if context.symbolic_narrative is not None:
        gaps.extend(context.symbolic_narrative.unresolved_narrative_gaps[:2])
    if audio_relevant and not _has_any(context, {"rhythm", "pulse", "beat", "tempo"}):
        gaps.append("Audio-reactive composition is relevant but timing is unclear.")
    return _dedupe(gaps)[:8]


def _hitl_questions(gaps: tuple[str, ...]) -> tuple[str, ...]:
    questions: list[str] = []
    for gap in gaps:
        lowered = gap.lower()
        if "pattern" in lowered:
            questions.append(
                "Should the composition be central, spiral, layered, field-like, "
                "threshold-based, or minimal?"
            )
        elif "motif" in lowered or "palette" in lowered or "visual" in lowered:
            questions.append("What should be the primary visible focal motif?")
        elif "audio" in lowered or "timing" in lowered:
            questions.append("What pulse, silence, or beat should organize the layout?")
        elif "interaction" in lowered:
            questions.append("What composition layer should interaction affect?")
        elif "transformation" in lowered:
            questions.append("What visual structure should transform at the climax?")
    return _dedupe(questions)[:6]


def _prompt_guidance(
    spec: _PatternSpec,
    unresolved: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use the composition plan as layout guidance, not code structure.",
        f"Preserve the {spec.label.lower()} pattern before adding effects.",
        "Keep the primary focal point readable at generation time.",
        "Treat density, rhythm, balance, and depth as coordinated choices.",
    ]
    if unresolved:
        guidance.append("Ask HITL questions before over-specifying composition.")
    return tuple(guidance[:8])


def _evidence(
    context: _CompositionContext,
    *,
    spec: _PatternSpec,
    matches: tuple[str, ...],
    score: int,
    audio_relevant: bool,
) -> tuple[str, ...]:
    evidence = [
        f"Composition pattern: {spec.pattern}.",
        f"Pattern score: {score}.",
        "Matched composition signals: " + ", ".join(matches[:6]) + ".",
        f"Audio composition relevant: {audio_relevant}.",
    ]
    if context.symbolic_narrative is not None:
        evidence.append(
            f"Narrative source: {context.symbolic_narrative.narrative_archetype}."
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
    return tuple(evidence[:12])


def _narrative_bonus(context: _CompositionContext, spec: _PatternSpec) -> int:
    if context.symbolic_narrative is None:
        return 0
    archetype = context.symbolic_narrative.narrative_archetype
    preferred = _NARRATIVE_PATTERN_HINTS.get(archetype, ())
    return 4 if spec.pattern in preferred[:1] else 2 if spec.pattern in preferred else 0


def _strategy_bonus(context: _CompositionContext, spec: _PatternSpec) -> int:
    if context.creative_strategy is None:
        return 0
    preferred = _STRATEGY_PATTERN_HINTS.get(
        context.creative_strategy.primary_strategy,
        (),
    )
    return 2 if spec.pattern in preferred else 0


def _technique_bonus(context: _CompositionContext, spec: _PatternSpec) -> int:
    if context.creative_techniques is None:
        return 0
    preferred = _TECHNIQUE_PATTERN_HINTS.get(
        context.creative_techniques.primary_technique,
        (),
    )
    return 2 if spec.pattern in preferred else 0


def _audio_relevant(context: _CompositionContext) -> bool:
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


def _subject_label(context: _CompositionContext) -> str:
    if context.creative_intent is not None:
        return context.creative_intent.primary_expression
    if context.creative_translation is not None:
        return context.creative_translation.creative_intent
    return "the user's stated subject"


def _adapt_with_visual_cues(value: str, context: _CompositionContext) -> str:
    if context.creative_translation is None:
        return value
    cues = [
        *context.creative_translation.geometric_references[:2],
        *context.creative_translation.color_material_direction[:2],
    ]
    if not cues:
        return value
    return f"{value} Use visual cues: {', '.join(cues)}."


def _matches(context: _CompositionContext, keyword: str) -> bool:
    normalized = _normalize(keyword)
    if " " in normalized:
        return normalized in context.text
    return normalized in context.tokens


def _has_any(context: _CompositionContext, values: frozenset[str] | set[str]) -> bool:
    return bool(context.tokens.intersection(values))


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


_PATTERNS: tuple[_PatternSpec, ...] = (
    _PatternSpec(
        "central_emergence",
        "Central Emergence",
        ("center", "central", "core", "seed", "emerge", "bloom", "birth"),
        "A compact center that becomes the primary anchor.",
        "Organize forms from a center outward with clear radial breathing room.",
        "Keep the focal center in front while background layers reveal context.",
        "Begin dense at the focal core and taper detail toward the edges.",
        "Use slow expansion, pulse, or reveal around the center.",
        "Balance around the focal center with controlled peripheral support.",
        "Prefer radial or near-symmetric order unless the brief asks otherwise.",
        "Use subtle layer separation from center to outer field.",
        "Use a frontal or top-down view that preserves the central anchor.",
    ),
    _PatternSpec(
        "radial_expansion",
        "Radial Expansion",
        ("radial", "radiate", "mandala", "concentric", "sun", "expansion"),
        "A central source with visible rings, spokes, or expanding geometry.",
        "Arrange elements in concentric or radial bands.",
        "Foreground rings carry detail; background bands support atmosphere.",
        "Increase density near symbolic rings and simplify interstitial space.",
        "Use repeated radial beats or outward phase changes.",
        "Balance symmetry with small variations that prevent stiffness.",
        "Keep rotational symmetry legible while allowing controlled asymmetry.",
        "Layer rings from inner detail to outer atmosphere.",
        "Use a stable top-down or orthographic viewpoint.",
    ),
    _PatternSpec(
        "spiral_composition",
        "Spiral Composition",
        ("spiral", "vortex", "whirl", "recursive", "turn", "orbit"),
        "A spiral path that carries attention from origin to transformation.",
        "Organize forms along a readable spiral trajectory.",
        "Foreground follows the spiral path; background clarifies its envelope.",
        "Vary density along turns so the eye can follow the path.",
        "Use rotational rhythm, widening loops, or radius modulation.",
        "Balance directional movement with a stable origin or endpoint.",
        "Use rotational asymmetry while preserving the spiral's readability.",
        "Layer inner, middle, and outer turns with clear spacing.",
        "Use a fixed view that keeps the spiral trajectory visible.",
    ),
    _PatternSpec(
        "layered_depth",
        "Layered Depth",
        ("layer", "layers", "depth", "3d", "parallax", "foreground", "background"),
        "A clear foreground anchor inside stacked depth layers.",
        "Organize the work as foreground, midground, and background bands.",
        "Foreground carries the main symbol; background supplies atmosphere.",
        "Keep foreground dense enough to read and background comparatively calm.",
        "Use parallax, offset cycles, or staggered reveals.",
        "Balance near/far contrast without hiding the focal layer.",
        "Use asymmetry if needed to clarify depth and occlusion.",
        "Separate layers by scale, blur, opacity, or motion speed.",
        "Use a stable perspective with restrained camera movement.",
    ),
    _PatternSpec(
        "field_composition",
        "Field Composition",
        ("field", "noise", "swarm", "particles", "flow", "turbulence"),
        "A distributed field with one readable local focus.",
        "Organize many elements through flow, clusters, or gradients.",
        "Foreground clusters define attention; background field shows scale.",
        "Use variable density to prevent a flat all-over texture.",
        "Use wave, flow, or particle rhythm across the field.",
        "Balance field richness with a clear path for attention.",
        "Prefer organic asymmetry unless the brief requires symmetry.",
        "Layer density, scale, and opacity to avoid visual noise.",
        None,
    ),
    _PatternSpec(
        "threshold_composition",
        "Threshold Composition",
        ("threshold", "gate", "portal", "boundary", "door", "crossing"),
        "A gate, edge, or boundary as the main compositional hinge.",
        "Organize space around before/after zones separated by a threshold.",
        "Foreground marks the crossing; background reveals what lies beyond.",
        "Keep density lower at the threshold so the boundary reads clearly.",
        "Use approach, pause, crossing, and release as rhythm.",
        "Balance the two sides while making the crossing decisive.",
        "Use bilateral tension or asymmetry to show transition.",
        "Layer near-side, threshold plane, and far-side space distinctly.",
        "Use a stable viewpoint looking into or across the threshold.",
    ),
    _PatternSpec(
        "descent_ascent_composition",
        "Descent/Ascent Composition",
        ("descent", "ascent", "rise", "fall", "above", "below", "depth"),
        "A vertical or depth-axis journey with clear directional pull.",
        "Organize forms along a downward/upward or near/far axis.",
        "Foreground marks the current state; background shows destination depth.",
        "Compress density at the turning point and open space near resolution.",
        "Use directional motion or value shifts along the axis.",
        "Balance gravity with a visible return or release path.",
        "Use asymmetry if it clarifies directional journey.",
        "Layer space as surface, passage, threshold, and return.",
        "Use a viewpoint that preserves the journey axis.",
    ),
    _PatternSpec(
        "fragmented_recomposition",
        "Fragmented Recomposition",
        ("fragment", "fragments", "shatter", "shards", "recompose", "rebirth"),
        "A broken focal structure that visibly reorganizes.",
        "Organize pieces around a missing or reforming center.",
        "Foreground fragments carry sharp detail; background shows the void.",
        "Start with scattered density and resolve into a cleaner structure.",
        "Use rupture, suspension, and recomposition as rhythm.",
        "Balance chaos with a visible reintegration path.",
        "Use broken symmetry that resolves toward stable order.",
        "Layer fragments by size and distance from the focal center.",
        None,
    ),
    _PatternSpec(
        "mirrored_composition",
        "Mirrored Composition",
        ("mirror", "mirrored", "reflection", "symmetry", "bilateral"),
        "A mirrored focal axis or reflective pair.",
        "Organize the composition around an explicit reflection line.",
        "Foreground carries the mirrored pair; background emphasizes axis clarity.",
        "Keep density balanced across the mirror with small variations.",
        "Use call-and-response rhythm across the reflective axis.",
        "Balance both sides while preserving meaningful differences.",
        "Use symmetry intentionally and break it only for emphasis.",
        "Layer reflection, source, and boundary separately.",
        "Use a direct viewpoint perpendicular to the mirror axis.",
    ),
    _PatternSpec(
        "orbiting_focal_structure",
        "Orbiting Focal Structure",
        ("orbit", "orbital", "satellite", "ring", "around", "planetary"),
        "A stable focal body with orbiting secondary elements.",
        "Organize satellites, rings, or paths around a focal anchor.",
        "Foreground orbiters cue motion; background preserves spatial scale.",
        "Keep the anchor readable and distribute orbiters by radius.",
        "Use orbital timing, looping paths, or phase offsets.",
        "Balance central stability with moving peripheral emphasis.",
        "Use radial structure with asymmetric timing variation.",
        "Layer orbit paths by radius, scale, and speed.",
        "Use a stable view that keeps orbit paths legible.",
    ),
    _PatternSpec(
        "distributed_constellation",
        "Distributed Constellation",
        ("constellation", "stars", "network", "nodes", "galaxy", "distributed"),
        "A network of focal nodes connected by implied paths.",
        "Organize points as clusters, constellations, or relational nodes.",
        "Foreground nodes carry meaning; background field gives atmosphere.",
        "Use local density clusters separated by readable negative space.",
        "Use blinking, linking, or sequential activation rhythm.",
        "Balance the whole field through cluster weight and spacing.",
        "Prefer asymmetric distribution with clear relational structure.",
        "Layer near nodes, distant nodes, and connective traces.",
        None,
    ),
    _PatternSpec(
        "minimal_void_and_form_composition",
        "Minimal Void-And-Form Composition",
        ("minimal", "void", "sparse", "simple", "empty", "negative space"),
        "A single restrained form set against deliberate negative space.",
        "Organize one primary form with large quiet space around it.",
        "Foreground form stays minimal; background void carries tension.",
        "Keep density low and reserve detail for the focal form.",
        "Use slow reveal, stillness, or sparse pulse.",
        "Balance form and void through placement, scale, and silence.",
        "Use asymmetry when it strengthens the void/form tension.",
        "Use minimal layers with clear separation.",
        None,
    ),
)

_PATTERN_BY_ID = {item.pattern: item for item in _PATTERNS}

_NARRATIVE_PATTERN_HINTS: dict[str, tuple[CompositionPattern, ...]] = {
    "death_and_rebirth": (
        "fragmented_recomposition",
        "central_emergence",
        "radial_expansion",
    ),
    "descent_and_return": ("descent_ascent_composition", "layered_depth"),
    "emergence_from_chaos": ("field_composition", "central_emergence"),
    "initiation": ("threshold_composition", "radial_expansion"),
    "ascent": ("descent_ascent_composition", "radial_expansion"),
    "dissolution_and_reintegration": ("fragmented_recomposition", "field_composition"),
    "expansion_from_seed_to_cosmos": (
        "central_emergence",
        "radial_expansion",
        "distributed_constellation",
    ),
    "fragmentation_and_recomposition": ("fragmented_recomposition",),
    "threshold_crossing": ("threshold_composition",),
    "spiral_transformation": ("spiral_composition", "orbiting_focal_structure"),
    "mirror_reflection_journey": ("mirrored_composition",),
    "dark_to_light_transformation": ("central_emergence", "layered_depth"),
    "symbolic_vignette": ("minimal_void_and_form_composition",),
}

_STRATEGY_PATTERN_HINTS: dict[str, tuple[CompositionPattern, ...]] = {
    "sacred_geometry": (
        "radial_expansion",
        "mirrored_composition",
        "spiral_composition",
        "central_emergence",
    ),
    "particle_cosmology": (
        "distributed_constellation",
        "field_composition",
        "radial_expansion",
    ),
    "fractal_growth": ("spiral_composition", "radial_expansion"),
    "field_dynamics": ("field_composition", "layered_depth"),
}

_TECHNIQUE_PATTERN_HINTS: dict[str, tuple[CompositionPattern, ...]] = {
    "recursive_geometry": (
        "radial_expansion",
        "spiral_composition",
        "mirrored_composition",
    ),
    "particle_systems": ("field_composition", "distributed_constellation"),
    "noise_fields": ("field_composition", "layered_depth"),
}
