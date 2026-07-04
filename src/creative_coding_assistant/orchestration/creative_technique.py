"""Deterministic Creative Technique Selector for V3 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyId,
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

CreativeTechniqueId = Literal[
    "fractal_recursion",
    "particle_systems",
    "reaction_diffusion",
    "boids",
    "cellular_automata",
    "voronoi",
    "noise_fields",
    "recursive_geometry",
    "sdf",
    "signed_distance_composition",
    "feedback_systems",
    "audio_reactive_mappings",
]
TechniqueCompatibility = Literal["strong", "moderate", "weak"]
TechniquePressure = Literal["low", "medium", "high"]

TECHNIQUE_AUTHORITY_BOUNDARY = (
    "The Creative Technique Selector recommends creative implementation "
    "techniques only; it does not choose runtime, renderer, provider, model, "
    "execution profile, or preview behavior."
)


class CreativeTechniqueAlternative(BaseModel):
    """A secondary technique compatible with the selected strategy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    technique: CreativeTechniqueId
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=260)


class CreativeTechniqueProfile(BaseModel):
    """Inspectable technique metadata derived before provider generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_technique_selector"] = "creative_technique_selector"
    primary_technique: CreativeTechniqueId
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=360)
    strategy_alignment: CreativeStrategyId | None = None
    compatibility: TechniqueCompatibility
    complexity_pressure: TechniquePressure
    performance_pressure: TechniquePressure
    artistic_suitability: tuple[str, ...] = Field(min_length=1, max_length=6)
    implementation_notes: tuple[str, ...] = Field(min_length=1, max_length=6)
    alternative_techniques: tuple[CreativeTechniqueAlternative, ...] = Field(
        default_factory=tuple,
        max_length=3,
    )
    technique_constraints: tuple[str, ...] = Field(min_length=1, max_length=6)
    selection_boundary: str = Field(
        default=TECHNIQUE_AUTHORITY_BOUNDARY,
        max_length=360,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_technique_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_translation: CreativeTranslation | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
) -> CreativeTechniqueProfile:
    """Select implementation technique guidance without selecting runtime."""

    normalized = _technique_text(
        request,
        creative_intent,
        creative_hierarchy,
        creative_translation,
        creative_strategy,
    )
    scored = sorted(
        (
            _score_technique(
                signal,
                normalized=normalized,
                request=request,
                creative_intent=creative_intent,
                creative_hierarchy=creative_hierarchy,
                creative_translation=creative_translation,
                creative_strategy=creative_strategy,
            )
            for signal in _TECHNIQUES
        ),
        key=lambda item: (item.score, item.technique),
        reverse=True,
    )
    primary = scored[0]
    alternatives = tuple(
        CreativeTechniqueAlternative(
            technique=item.technique,
            confidence=_confidence(item.score),
            rationale=_alternative_rationale(item),
        )
        for item in scored[1:4]
        if item.score > 0
    )
    return CreativeTechniqueProfile(
        primary_technique=primary.technique,
        confidence=_confidence(primary.score),
        rationale=_primary_rationale(primary),
        strategy_alignment=(
            creative_strategy.primary_strategy
            if creative_strategy is not None
            else None
        ),
        compatibility=_compatibility(primary, creative_strategy),
        complexity_pressure=primary.complexity_pressure,
        performance_pressure=primary.performance_pressure,
        artistic_suitability=_artistic_suitability(primary, creative_strategy),
        implementation_notes=primary.notes,
        alternative_techniques=alternatives,
        technique_constraints=_technique_constraints(primary),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=creative_translation,
            creative_strategy=creative_strategy,
            scored=scored,
        ),
    )


def creative_technique_prompt_lines(
    profile: CreativeTechniqueProfile,
) -> tuple[str, ...]:
    """Render technique metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {profile.selection_boundary}",
        f"Primary technique: {profile.primary_technique}.",
        f"Technique confidence: {profile.confidence:.2f}.",
        f"Technique rationale: {profile.rationale}",
        f"Strategy compatibility: {profile.compatibility}.",
        f"Complexity pressure: {profile.complexity_pressure}.",
        f"Performance pressure: {profile.performance_pressure}.",
    ]
    lines.extend(
        f"Artistic suitability: {item}" for item in profile.artistic_suitability
    )
    lines.extend(
        f"Implementation note: {item}" for item in profile.implementation_notes
    )
    lines.extend(
        f"Technique constraint: {item}" for item in profile.technique_constraints
    )
    return tuple(lines[:18])


@dataclass(frozen=True)
class _TechniqueSignal:
    technique: CreativeTechniqueId
    label: str
    keywords: tuple[str, ...]
    strategy_affinities: tuple[CreativeStrategyId, ...]
    complexity_pressure: TechniquePressure
    performance_pressure: TechniquePressure
    rationale: str
    suitability: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _ScoredTechnique:
    technique: CreativeTechniqueId
    label: str
    score: int
    matched_signals: tuple[str, ...]
    strategy_affinities: tuple[CreativeStrategyId, ...]
    complexity_pressure: TechniquePressure
    performance_pressure: TechniquePressure
    rationale: str
    suitability: tuple[str, ...]
    notes: tuple[str, ...]


def _score_technique(
    signal: _TechniqueSignal,
    *,
    normalized: str,
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
) -> _ScoredTechnique:
    matched: list[str] = []
    score = 0
    for keyword in signal.keywords:
        if _keyword_matches(normalized, keyword):
            matched.append(keyword)
            score += 2
    if (
        creative_strategy is not None
        and creative_strategy.primary_strategy in signal.strategy_affinities
    ):
        score += 4
        matched.append(f"strategy:{creative_strategy.primary_strategy}")
    if creative_translation is not None:
        score += _translation_score(signal, creative_translation, matched)
    if creative_intent is not None:
        score += _intent_score(signal, creative_intent, matched)
    if creative_hierarchy is not None:
        score += _hierarchy_score(signal, creative_hierarchy, matched)
    if request.attachments and signal.technique in _REFERENCE_FRIENDLY_TECHNIQUES:
        score += 1
        matched.append("reference input")
    if score == 0 and signal.technique == "noise_fields":
        score = 1
        matched.append("default bounded technique")
    return _ScoredTechnique(
        technique=signal.technique,
        label=signal.label,
        score=score,
        matched_signals=tuple(_dedupe_text(matched))[:8],
        strategy_affinities=signal.strategy_affinities,
        complexity_pressure=signal.complexity_pressure,
        performance_pressure=signal.performance_pressure,
        rationale=signal.rationale,
        suitability=signal.suitability,
        notes=signal.notes,
    )


def _translation_score(
    signal: _TechniqueSignal,
    translation: CreativeTranslation,
    matched: list[str],
) -> int:
    score = 0
    if signal.technique == "audio_reactive_mappings" and (
        translation.audio_reactive is not None or translation.musical_references
    ):
        score += 5
        matched.append("audio-reactive or musical signal")
    if signal.technique in {"recursive_geometry", "fractal_recursion"} and (
        translation.sacred_geometry is not None or translation.geometric_references
    ):
        score += 3
        matched.append("geometry guidance")
    if signal.technique == "particle_systems" and any(
        value in {"drift", "orbit", "swarm", "pulse"}
        for value in translation.movement_language
    ):
        score += 2
        matched.append("particle-friendly motion")
    for value in (
        *translation.symbolic_references,
        *translation.geometric_references,
        *translation.musical_references,
        *translation.mood_atmosphere,
        *translation.movement_language,
        *translation.structure_direction,
    ):
        if any(keyword in value.lower() for keyword in signal.keywords):
            score += 1
            matched.append(value)
    return score


def _intent_score(
    signal: _TechniqueSignal,
    intent: CreativeIntentDecomposition,
    matched: list[str],
) -> int:
    score = 0
    active_names = {
        item.name for item in intent.atomic_dimensions if item.explicitness != "absent"
    }
    if signal.technique == "audio_reactive_mappings" and (
        "audio" in active_names or "rhythm" in active_names
    ):
        score += 4
        matched.append("decomposed audio/rhythm intent")
    if signal.technique in {"recursive_geometry", "fractal_recursion", "sdf"} and (
        "geometric" in active_names or "symbolic" in active_names
    ):
        score += 2
        matched.append("decomposed geometric/symbolic intent")
    if signal.technique == "particle_systems" and "motion" in active_names:
        score += 2
        matched.append("decomposed motion intent")
    for dimension in intent.atomic_dimensions:
        for value in dimension.signals:
            if any(keyword in value.lower() for keyword in signal.keywords):
                score += 1
                matched.append(value)
    return score


def _hierarchy_score(
    signal: _TechniqueSignal,
    hierarchy: CreativeHierarchyPlan,
    matched: list[str],
) -> int:
    score = 0
    primary = {priority.dimension for priority in hierarchy.primary_creative_priorities}
    if signal.technique == "audio_reactive_mappings" and primary & {
        "audio",
        "rhythm",
    }:
        score += 4
        matched.append("hierarchy:audio/rhythm")
    if signal.technique in {"recursive_geometry", "fractal_recursion", "sdf"} and (
        primary & {"geometry", "symbolism", "experiential_depth"}
    ):
        score += 3
        matched.append("hierarchy:geometry/symbolism")
    if signal.technique == "particle_systems" and primary & {
        "motion",
        "visual_impact",
        "light_color",
    }:
        score += 3
        matched.append("hierarchy:motion/visual impact")
    if signal.technique == "noise_fields" and primary & {"simplicity", "performance"}:
        score += 2
        matched.append("hierarchy:simplicity/performance")
    return score


def _confidence(score: int) -> float:
    return min(0.95, max(0.35, round(0.35 + score * 0.08, 2)))


def _compatibility(
    scored: _ScoredTechnique,
    creative_strategy: CreativeStrategyProfile | None,
) -> TechniqueCompatibility:
    if creative_strategy is None:
        return "moderate"
    if creative_strategy.primary_strategy in scored.strategy_affinities:
        return "strong"
    if scored.score >= 4:
        return "moderate"
    return "weak"


def _primary_rationale(scored: _ScoredTechnique) -> str:
    if scored.matched_signals:
        joined = ", ".join(scored.matched_signals[:4])
        return f"{scored.label} best matches detected signals: {joined}."
    return f"{scored.label} provides bounded technique guidance."


def _alternative_rationale(scored: _ScoredTechnique) -> str:
    if scored.matched_signals:
        return f"Also relevant because of {', '.join(scored.matched_signals[:3])}."
    return scored.rationale


def _artistic_suitability(
    scored: _ScoredTechnique,
    creative_strategy: CreativeStrategyProfile | None,
) -> tuple[str, ...]:
    suitability = list(scored.suitability)
    if creative_strategy is not None:
        suitability.insert(
            0,
            f"Supports strategy: {creative_strategy.primary_strategy}.",
        )
    return tuple(_dedupe_text(suitability))[:6]


def _technique_constraints(scored: _ScoredTechnique) -> tuple[str, ...]:
    constraints = [
        "Do not treat technique selection as runtime or renderer selection.",
        "Use technique guidance only where it supports the creative brief.",
    ]
    if scored.complexity_pressure == "high":
        constraints.append("Keep technique scope bounded before adding variations.")
    if scored.performance_pressure == "high":
        constraints.append("Surface performance-sensitive technique choices clearly.")
    return tuple(constraints[:6])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    scored: list[_ScoredTechnique],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if creative_translation is not None:
        evidence.append(f"Creative intent: {creative_translation.creative_intent}.")
    if creative_intent is not None:
        evidence.append(f"Intent substrate: {creative_intent.primary_expression}.")
    if creative_hierarchy is not None:
        evidence.append(
            "Hierarchy priorities: "
            + ", ".join(
                item.dimension
                for item in creative_hierarchy.primary_creative_priorities
            )
            + "."
        )
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
    if scored and scored[0].matched_signals:
        evidence.append(
            "Primary technique signals: "
            + ", ".join(scored[0].matched_signals[:4])
            + "."
        )
    evidence.append(
        "Technique scores: "
        + ", ".join(f"{item.technique}={item.score}" for item in scored[:4])
        + "."
    )
    return tuple(_dedupe_text(evidence))[:10]


def _technique_text(
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
) -> str:
    parts = [request.query]
    if creative_hierarchy is not None:
        parts.extend(
            (
                " ".join(
                    item.dimension
                    for item in creative_hierarchy.primary_creative_priorities
                ),
                " ".join(creative_hierarchy.non_negotiable_dimensions),
            )
        )
    if creative_intent is not None:
        parts.extend(
            (
                creative_intent.primary_expression,
                creative_intent.experiential_goal,
                " ".join(
                    signal
                    for dimension in creative_intent.atomic_dimensions
                    for signal in dimension.signals
                ),
            )
        )
    if creative_translation is not None:
        parts.extend(
            (
                creative_translation.creative_intent,
                " ".join(creative_translation.symbolic_references),
                " ".join(creative_translation.geometric_references),
                " ".join(creative_translation.musical_references),
                " ".join(creative_translation.mood_atmosphere),
                " ".join(creative_translation.movement_language),
                " ".join(creative_translation.structure_direction),
            )
        )
    if creative_strategy is not None:
        parts.extend(
            (
                creative_strategy.primary_strategy,
                " ".join(creative_strategy.creative_goals),
                " ".join(creative_strategy.strategy_directives),
                " ".join(creative_strategy.symbolic_alignment),
            )
        )
    return _compact(" ".join(parts)).lower()


def _keyword_matches(normalized: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", normalized) is not None


def _compact(value: str) -> str:
    return " ".join(value.strip().split())


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = _compact(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


_TECHNIQUES: tuple[_TechniqueSignal, ...] = (
    _TechniqueSignal(
        technique="fractal_recursion",
        label="Fractal Recursion",
        keywords=("fractal", "recursion", "recursive", "branch", "growth", "tree"),
        strategy_affinities=("fractal_growth", "recursive_emergence"),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Use self-similar repetition across scale.",
        suitability=(
            "Strong for nested forms, growth, and repeated structural motifs.",
            "Works when variation should emerge from a compact rule set.",
        ),
        notes=(
            "Keep recursion depth bounded.",
            "Expose scale and variation as creative controls.",
        ),
    ),
    _TechniqueSignal(
        technique="particle_systems",
        label="Particle Systems",
        keywords=(
            "particle",
            "particles",
            "swarm",
            "nebula",
            "orbit",
            "stars",
            "dust",
        ),
        strategy_affinities=("particle_cosmology", "field_dynamics"),
        complexity_pressure="medium",
        performance_pressure="high",
        rationale="Use many simple elements to create collective visual behavior.",
        suitability=(
            "Strong for cosmic, atmospheric, swarm, and density-based briefs.",
            "Supports expressive motion through many small agents.",
        ),
        notes=(
            "Keep counts and lifetimes bounded.",
            "Favor clear behavior rules over many unrelated effects.",
        ),
    ),
    _TechniqueSignal(
        technique="reaction_diffusion",
        label="Reaction Diffusion",
        keywords=("reaction diffusion", "reaction-diffusion", "diffusion", "organic"),
        strategy_affinities=("cellular_evolution", "field_dynamics"),
        complexity_pressure="high",
        performance_pressure="high",
        rationale="Use spreading local interactions to evoke organic patterning.",
        suitability=(
            "Strong for biological textures, morphogenesis, and living surfaces.",
            "Useful when visual growth should feel chemical or cellular.",
        ),
        notes=(
            "Keep iteration counts conservative.",
            "Use as technique guidance, not as a required simulation runtime.",
        ),
    ),
    _TechniqueSignal(
        technique="boids",
        label="Boids",
        keywords=("boids", "flock", "flocking", "school", "swarm", "collective"),
        strategy_affinities=("recursive_emergence", "particle_cosmology"),
        complexity_pressure="medium",
        performance_pressure="high",
        rationale="Use local steering rules to create collective movement.",
        suitability=(
            "Strong for flocking, schooling, and emergent agent behavior.",
            "Works when the artwork needs visible collective intelligence.",
        ),
        notes=(
            "Keep steering forces legible.",
            "Bound neighbor checks or describe them conceptually.",
        ),
    ),
    _TechniqueSignal(
        technique="cellular_automata",
        label="Cellular Automata",
        keywords=("cellular automata", "automata", "cell", "cells", "grid", "life"),
        strategy_affinities=("cellular_evolution", "recursive_emergence"),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Use local grid-state rules to create evolving structure.",
        suitability=(
            "Strong for life-like systems, mutation, and emergent grids.",
            "Fits briefs that need visible rule evolution.",
        ),
        notes=(
            "Keep state rules explainable.",
            "Avoid overloading the grid with unrelated behaviors.",
        ),
    ),
    _TechniqueSignal(
        technique="voronoi",
        label="Voronoi",
        keywords=("voronoi", "cellular", "cells", "tessellation", "crystal"),
        strategy_affinities=(
            "cellular_evolution",
            "sacred_geometry",
            "minimal_generative_systems",
        ),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Use cell partitions and nearest-point structure.",
        suitability=(
            "Strong for crystalline, cellular, cartographic, and minimal forms.",
            "Supports clear structural rhythm without requiring heavy detail.",
        ),
        notes=(
            "Keep site count bounded.",
            "Use cell boundaries as compositional structure.",
        ),
    ),
    _TechniqueSignal(
        technique="noise_fields",
        label="Noise Fields",
        keywords=("noise", "field", "flow", "wind", "drift", "terrain", "organic"),
        strategy_affinities=("field_dynamics", "minimal_generative_systems"),
        complexity_pressure="low",
        performance_pressure="medium",
        rationale="Use continuous variation to guide form, motion, or density.",
        suitability=(
            "Strong for flow, atmosphere, terrain, and subtle organic motion.",
            "Provides a bounded default for flexible generative systems.",
        ),
        notes=(
            "Use noise as a coherent field, not random decoration.",
            "Keep scale, speed, and contrast aligned with mood.",
        ),
    ),
    _TechniqueSignal(
        technique="recursive_geometry",
        label="Recursive Geometry",
        keywords=("geometry", "recursive geometry", "mandala", "symmetry", "yantra"),
        strategy_affinities=("sacred_geometry", "fractal_growth"),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Use repeated geometric construction as the compositional method.",
        suitability=(
            "Strong for mandalas, yantras, symbolic symmetry, and nested forms.",
            "Keeps structure aligned with symbolic geometric briefs.",
        ),
        notes=(
            "Preserve the symbolic hierarchy of shapes.",
            "Bound repeated geometry so the composition stays legible.",
        ),
    ),
    _TechniqueSignal(
        technique="sdf",
        label="SDF",
        keywords=("sdf", "signed distance", "distance field", "raymarch"),
        strategy_affinities=("field_dynamics", "sacred_geometry"),
        complexity_pressure="high",
        performance_pressure="high",
        rationale="Use distance relationships as the shaping technique.",
        suitability=(
            "Strong for sculptural fields and mathematically precise forms.",
            "Useful when surfaces should emerge from distance logic.",
        ),
        notes=(
            "Treat SDF as technique guidance only, not a shader/runtime choice.",
            "Keep shape composition bounded and explainable.",
        ),
    ),
    _TechniqueSignal(
        technique="signed_distance_composition",
        label="Signed Distance Composition",
        keywords=("signed distance composition", "distance composition", "boolean"),
        strategy_affinities=("sacred_geometry", "field_dynamics"),
        complexity_pressure="high",
        performance_pressure="high",
        rationale="Use constructive distance relationships between forms.",
        suitability=(
            "Strong for precise symbolic forms and compositional shape logic.",
            "Supports blending, subtraction, and layered spatial relationships.",
        ),
        notes=(
            "Use composition rules sparingly.",
            "Do not imply a required rendering technology.",
        ),
    ),
    _TechniqueSignal(
        technique="feedback_systems",
        label="Feedback Systems",
        keywords=("feedback", "loop", "recursive feedback", "echo", "trail"),
        strategy_affinities=("recursive_emergence", "field_dynamics"),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Use prior state to shape evolving output over time.",
        suitability=(
            "Strong for echoes, trails, temporal memory, and evolving fields.",
            "Fits briefs that need visible history or recursive response.",
        ),
        notes=(
            "Keep feedback bounded to avoid visual collapse.",
            "Use temporal memory only where it supports the concept.",
        ),
    ),
    _TechniqueSignal(
        technique="audio_reactive_mappings",
        label="Audio-Reactive Mappings",
        keywords=(
            "audio reactive",
            "audio-reactive",
            "sound reactive",
            "beat",
            "music",
        ),
        strategy_affinities=("field_dynamics", "particle_cosmology"),
        complexity_pressure="medium",
        performance_pressure="medium",
        rationale="Map musical or signal features into visual behavior.",
        suitability=(
            "Strong for audiovisual briefs and sound-driven modulation.",
            "Useful when motion or form should respond to rhythm or amplitude.",
        ),
        notes=(
            "Map only meaningful signal features.",
            "Do not assume audio input or runtime support unless provided elsewhere.",
        ),
    ),
)
_REFERENCE_FRIENDLY_TECHNIQUES = frozenset(
    {"recursive_geometry", "voronoi", "noise_fields", "particle_systems"}
)
