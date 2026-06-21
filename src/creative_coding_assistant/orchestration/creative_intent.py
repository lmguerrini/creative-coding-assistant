"""Deterministic Creative Intent Decomposer for V3 workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

IntentDimensionName = Literal[
    "narrative",
    "symbolic",
    "emotional",
    "geometric",
    "motion",
    "rhythm",
    "light_color",
    "audio",
    "interaction",
    "climax_transformation",
]
IntentExplicitness = Literal["explicit", "inferred", "absent", "ambiguous"]
AbstractionLevel = Literal[
    "literal",
    "stylized",
    "symbolic",
    "abstract",
    "mixed",
    "unspecified",
]

INTENT_DECOMPOSER_AUTHORITY_BOUNDARY = (
    "The Creative Intent Decomposer structures user intent for inspection only; "
    "it does not choose strategy, technique, runtime, renderer, provider, model, "
    "artifact, preview behavior, execution profile, or autonomous repair path."
)


class CreativeIntentDimension(BaseModel):
    """One atomic dimension of a creative request."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: IntentDimensionName
    explicitness: IntentExplicitness
    summary: str = Field(min_length=1, max_length=300)
    signals: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=4)


class CreativeIntentDecomposition(BaseModel):
    """Inspectable intent substrate used before later creative decisions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_intent_decomposer"] = "creative_intent_decomposer"
    normalized_intent: str = Field(min_length=1, max_length=360)
    primary_expression: str = Field(min_length=1, max_length=360)
    narrative_intent: CreativeIntentDimension
    symbolic_intent: CreativeIntentDimension
    emotional_intent: CreativeIntentDimension
    geometric_intent: CreativeIntentDimension
    motion_intent: CreativeIntentDimension
    rhythm_intent: CreativeIntentDimension
    light_color_intent: CreativeIntentDimension
    audio_intent: CreativeIntentDimension
    interaction_intent: CreativeIntentDimension
    climax_transformation_intent: CreativeIntentDimension
    abstraction_level: AbstractionLevel
    experiential_goal: str = Field(min_length=1, max_length=420)
    unresolved_intent_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    atomic_dimensions: tuple[CreativeIntentDimension, ...] = Field(
        min_length=10,
        max_length=10,
    )
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=INTENT_DECOMPOSER_AUTHORITY_BOUNDARY,
        max_length=420,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_intent_decomposition(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
) -> CreativeIntentDecomposition:
    """Break request-visible creative intent into bounded atomic dimensions."""

    text = _normalized_text(request, creative_translation)
    dimensions = {
        "narrative": _dimension(
            "narrative",
            _signals(text, _NARRATIVE_PATTERNS),
            translation_signals=(),
        ),
        "symbolic": _dimension(
            "symbolic",
            _signals(text, _SYMBOLIC_PATTERNS),
            translation_signals=(
                creative_translation.symbolic_references
                if creative_translation is not None
                else ()
            ),
        ),
        "emotional": _dimension(
            "emotional",
            _signals(text, _EMOTIONAL_PATTERNS),
            translation_signals=(
                creative_translation.mood_atmosphere
                if creative_translation is not None
                else ()
            ),
        ),
        "geometric": _dimension(
            "geometric",
            _signals(text, _GEOMETRIC_PATTERNS),
            translation_signals=(
                creative_translation.geometric_references
                if creative_translation is not None
                else ()
            ),
        ),
        "motion": _dimension(
            "motion",
            _signals(text, _MOTION_PATTERNS),
            translation_signals=(
                creative_translation.movement_language
                if creative_translation is not None
                else ()
            ),
        ),
        "rhythm": _dimension(
            "rhythm",
            _signals(text, _RHYTHM_PATTERNS),
            translation_signals=(
                creative_translation.musical_references
                if creative_translation is not None
                else ()
            ),
        ),
        "light_color": _dimension(
            "light_color",
            _signals(text, _LIGHT_COLOR_PATTERNS),
            translation_signals=(
                creative_translation.color_material_direction
                if creative_translation is not None
                else ()
            ),
        ),
        "audio": _audio_dimension(text, creative_translation),
        "interaction": _dimension(
            "interaction",
            _signals(text, _INTERACTION_PATTERNS),
            translation_signals=(),
        ),
        "climax_transformation": _dimension(
            "climax_transformation",
            _signals(text, _TRANSFORMATION_PATTERNS),
            translation_signals=(),
        ),
    }
    atomic = tuple(dimensions[name] for name in _DIMENSION_ORDER)
    gaps = _unresolved_gaps(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        dimensions=atomic,
    )
    abstraction = _abstraction_level(text, dimensions)
    return CreativeIntentDecomposition(
        normalized_intent=_primary_intent(request, creative_translation),
        primary_expression=_primary_expression(atomic, request),
        narrative_intent=dimensions["narrative"],
        symbolic_intent=dimensions["symbolic"],
        emotional_intent=dimensions["emotional"],
        geometric_intent=dimensions["geometric"],
        motion_intent=dimensions["motion"],
        rhythm_intent=dimensions["rhythm"],
        light_color_intent=dimensions["light_color"],
        audio_intent=dimensions["audio"],
        interaction_intent=dimensions["interaction"],
        climax_transformation_intent=dimensions["climax_transformation"],
        abstraction_level=abstraction,
        experiential_goal=_experiential_goal(
            atomic,
            abstraction_level=abstraction,
            creative_translation=creative_translation,
        ),
        unresolved_intent_gaps=gaps,
        hitl_questions=_hitl_questions(gaps),
        atomic_dimensions=atomic,
        prompt_guidance=_prompt_guidance(atomic, gaps),
        evidence=_evidence(request, route_decision, creative_translation, atomic),
    )


def creative_intent_decomposition_prompt_lines(
    decomposition: CreativeIntentDecomposition,
) -> tuple[str, ...]:
    """Render intent decomposition as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {decomposition.authority_boundary}",
        f"Primary expression: {decomposition.primary_expression}",
        f"Abstraction level: {decomposition.abstraction_level}.",
        f"Experiential goal: {decomposition.experiential_goal}",
    ]
    for dimension in decomposition.atomic_dimensions:
        if dimension.explicitness == "absent":
            continue
        lines.append(
            f"{dimension.name} intent ({dimension.explicitness}): "
            f"{dimension.summary}"
        )
    lines.extend(
        f"Unresolved intent gap: {item}"
        for item in decomposition.unresolved_intent_gaps[:4]
    )
    lines.extend(
        f"HITL intent question: {item}"
        for item in decomposition.hitl_questions
    )
    lines.extend(f"Intent guidance: {item}" for item in decomposition.prompt_guidance)
    return tuple(lines[:28])


def active_intent_dimension_names(
    decomposition: CreativeIntentDecomposition | None,
) -> tuple[IntentDimensionName, ...]:
    if decomposition is None:
        return ()
    return tuple(
        item.name
        for item in decomposition.atomic_dimensions
        if item.explicitness in {"explicit", "inferred", "ambiguous"}
    )


def _dimension(
    name: IntentDimensionName,
    request_signals: Sequence[str],
    *,
    translation_signals: Sequence[str],
) -> CreativeIntentDimension:
    signals = _dedupe([*request_signals, *translation_signals])
    explicitness: IntentExplicitness = "absent"
    if request_signals:
        explicitness = "explicit"
    elif translation_signals:
        explicitness = "inferred"
    return CreativeIntentDimension(
        name=name,
        explicitness=explicitness,
        summary=_dimension_summary(name, explicitness, signals),
        signals=signals,
        guidance=_dimension_guidance(name, explicitness, signals),
    )


def _audio_dimension(
    text: str,
    creative_translation: CreativeTranslation | None,
) -> CreativeIntentDimension:
    translation_signals: tuple[str, ...] = ()
    if creative_translation is not None:
        translation_signals = creative_translation.musical_references
        if creative_translation.output_modality in {
            CreativeOutputModality.AUDIO,
            CreativeOutputModality.AUDIOVISUAL,
        }:
            translation_signals = _dedupe(
                [*translation_signals, creative_translation.output_modality.value]
            )
    return _dimension(
        "audio",
        _signals(text, _AUDIO_PATTERNS),
        translation_signals=translation_signals,
    )


def _dimension_summary(
    name: IntentDimensionName,
    explicitness: IntentExplicitness,
    signals: tuple[str, ...],
) -> str:
    label = name.replace("_", " ")
    if explicitness == "absent":
        return f"No explicit {label} intent was detected."
    joined = ", ".join(signals[:4])
    return f"Use {label} cues around {joined} as an atomic design dimension."


def _dimension_guidance(
    name: IntentDimensionName,
    explicitness: IntentExplicitness,
    signals: tuple[str, ...],
) -> tuple[str, ...]:
    label = name.replace("_", " ")
    if explicitness == "absent":
        return (f"Do not invent {label} behavior unless needed for coherence.",)
    return (
        f"Preserve {label} intent before adding secondary effects.",
        f"Keep {label} cues inspectable: {', '.join(signals[:3])}.",
    )


def _unresolved_gaps(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    dimensions: tuple[CreativeIntentDimension, ...],
) -> tuple[str, ...]:
    gaps: list[str] = []
    active = {item.name for item in dimensions if item.explicitness != "absent"}
    if len(_compact(request.query).split()) < 5:
        gaps.append("Core creative subject or motif is underspecified.")
    if creative_translation is None or creative_translation.output_modality is None:
        gaps.append("Output modality is inferred rather than explicit.")
    if route_decision is not None and len(route_decision.domains) > 1:
        gaps.append("Relationship between active creative domains is ambiguous.")
    if "emotional" not in active:
        gaps.append("Emotional tone is not explicit.")
    if _visual_intent_likely(creative_translation) and "light_color" not in active:
        gaps.append("Light, color, or material direction is not explicit.")
    if "interaction" in active and "climax_transformation" not in active:
        gaps.append("Interactive state change or transformation arc is unclear.")
    if "audio" in active and "rhythm" not in active:
        gaps.append("Audio intent is present but rhythm or timing is underspecified.")
    return tuple(_dedupe(gaps))[:8]


def _hitl_questions(gaps: tuple[str, ...]) -> tuple[str, ...]:
    questions = []
    for gap in gaps:
        if "subject" in gap:
            questions.append("What subject, motif, or scene should anchor the piece?")
        elif "modality" in gap:
            questions.append("Should the output be visual, audio, or audiovisual?")
        elif "domains" in gap:
            questions.append("Which selected creative domain should lead the result?")
        elif "Emotional tone" in gap:
            questions.append("What emotional tone should the piece prioritize?")
        elif "Light" in gap:
            questions.append("What palette, lighting, or material quality should lead?")
        elif "Interactive" in gap:
            questions.append("What should change when the user interacts?")
        elif "Audio" in gap:
            questions.append("What rhythm, tempo, or sonic behavior should drive it?")
    return tuple(_dedupe(questions))[:6]


def _prompt_guidance(
    dimensions: tuple[CreativeIntentDimension, ...],
    gaps: tuple[str, ...],
) -> tuple[str, ...]:
    active = [item.name.replace("_", " ") for item in dimensions if item.signals]
    guidance = [
        "Use decomposed intent dimensions as design constraints, not new features.",
        "Preserve symbolic, emotional, formal, motion, and audio cues separately.",
    ]
    if active:
        guidance.append(
            "Prioritize active intent dimensions: " + ", ".join(active[:6]) + "."
        )
    if gaps:
        guidance.append("Ask targeted HITL only for gaps that affect core direction.")
    return tuple(guidance[:8])


def _primary_expression(
    dimensions: tuple[CreativeIntentDimension, ...],
    request: AssistantRequest,
) -> str:
    active = [
        f"{item.name.replace('_', ' ')}={', '.join(item.signals[:2])}"
        for item in dimensions
        if item.signals
    ]
    if active:
        return _clip("; ".join(active), 360)
    return _clip(_compact(request.query), 360)


def _experiential_goal(
    dimensions: tuple[CreativeIntentDimension, ...],
    *,
    abstraction_level: AbstractionLevel,
    creative_translation: CreativeTranslation | None,
) -> str:
    active = [item for item in dimensions if item.signals]
    if not active:
        base = (
            creative_translation.creative_intent
            if creative_translation is not None
            else "the user request"
        )
        return f"Create a bounded {abstraction_level} experience around {base}."
    highlights = ", ".join(item.summary for item in active[:3])
    return _clip(
        f"Create a {abstraction_level} experience that balances {highlights}.",
        420,
    )


def _abstraction_level(
    text: str,
    dimensions: dict[str, CreativeIntentDimension],
) -> AbstractionLevel:
    literal = bool(_ABSTRACTION_PATTERNS["literal"].search(text))
    stylized = bool(_ABSTRACTION_PATTERNS["stylized"].search(text))
    abstract = bool(_ABSTRACTION_PATTERNS["abstract"].search(text))
    symbolic = bool(_ABSTRACTION_PATTERNS["symbolic"].search(text)) or bool(
        dimensions["symbolic"].signals
    )
    active = sum((literal, stylized, abstract, symbolic))
    if active > 1:
        return "mixed"
    if literal:
        return "literal"
    if stylized:
        return "stylized"
    if symbolic:
        return "symbolic"
    if abstract:
        return "abstract"
    return "unspecified"


def _evidence(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    dimensions: tuple[CreativeIntentDimension, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
        if route_decision.domains:
            evidence.append(
                "Domains: "
                + ", ".join(item.value for item in route_decision.domains)
                + "."
            )
    if creative_translation is not None:
        evidence.append(
            f"Creative translation: {creative_translation.creative_intent}."
        )
    active = [item.name for item in dimensions if item.signals]
    evidence.append("Active intent dimensions: " + ", ".join(active or ["none"]) + ".")
    if request.attachments:
        evidence.append(f"Image references: {len(request.attachments)}.")
    return tuple(_dedupe(evidence))[:10]


def _primary_intent(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> str:
    if creative_translation is not None:
        return creative_translation.creative_intent
    return _clip(_compact(request.query), 360)


def _visual_intent_likely(
    creative_translation: CreativeTranslation | None,
) -> bool:
    if creative_translation is None:
        return True
    return creative_translation.output_modality in {
        None,
        CreativeOutputModality.VISUAL,
        CreativeOutputModality.AUDIOVISUAL,
    }


def _normalized_text(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> str:
    parts = [request.query]
    if creative_translation is not None:
        parts.extend(
            [
                creative_translation.creative_intent,
                " ".join(creative_translation.symbolic_references),
                " ".join(creative_translation.geometric_references),
                " ".join(creative_translation.musical_references),
                " ".join(creative_translation.mood_atmosphere),
                " ".join(creative_translation.movement_language),
                " ".join(creative_translation.color_material_direction),
                " ".join(creative_translation.structure_direction),
            ]
        )
    return _compact(" ".join(parts)).lower()


def _signals(
    text: str,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[str, ...]:
    return tuple(label for label, pattern in patterns if pattern.search(text))[:8]


def _compact(value: str) -> str:
    return " ".join(value.strip().split())


def _clip(value: str, limit: int) -> str:
    normalized = _compact(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        cleaned = _compact(str(value))
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen)


def _pattern(label: str, pattern: str) -> tuple[str, re.Pattern[str]]:
    return (label, re.compile(pattern))


_DIMENSION_ORDER: tuple[IntentDimensionName, ...] = (
    "narrative",
    "symbolic",
    "emotional",
    "geometric",
    "motion",
    "rhythm",
    "light_color",
    "audio",
    "interaction",
    "climax_transformation",
)
_NARRATIVE_PATTERNS = (
    _pattern("journey", r"\b(?:journey|story|narrative|arc|scene|world)\b"),
    _pattern("ritual", r"\b(?:ritual|ceremony|myth|dream|memory)\b"),
)
_SYMBOLIC_PATTERNS = (
    _pattern("symbolic", r"\b(?:symbolic|symboli[sz]e|metaphor|archetype)\b"),
    _pattern("sacred motif", r"\b(?:alchemy|mandala|sigil|tarot|ouroboros|lotus)\b"),
    _pattern("threshold", r"\b(?:portal|temple|labyrinth|threshold)\b"),
)
_EMOTIONAL_PATTERNS = (
    _pattern("calm", r"\b(?:calm|serene|meditative|peaceful|gentle)\b"),
    _pattern("awe", r"\b(?:awe|wonder|sublime|cosmic|transcendent)\b"),
    _pattern("tension", r"\b(?:tense|anxious|eerie|dark|melancholy)\b"),
    _pattern("energy", r"\b(?:playful|energetic|joyful|ecstatic|hypnotic)\b"),
)
_GEOMETRIC_PATTERNS = (
    _pattern("radial geometry", r"\b(?:radial|concentric|mandala|yantra)\b"),
    _pattern("recursive geometry", r"\b(?:fractal|recursive|spiral|fibonacci)\b"),
    _pattern("grid structure", r"\b(?:grid|tessellation|voronoi|hexagon)\b"),
    _pattern("3d form", r"\b(?:sphere|torus|polyhedron|sculpture)\b"),
)
_MOTION_PATTERNS = (
    _pattern("pulse", r"\b(?:pulse|pulsing|breathe|breathing)\b"),
    _pattern("flow", r"\b(?:flow|drift|ripple|wave|cascade)\b"),
    _pattern("orbit", r"\b(?:orbit|rotate|swirl|spiral)\b"),
    _pattern("swarm", r"\b(?:swarm|flock|scatter|emerge)\b"),
)
_RHYTHM_PATTERNS = (
    _pattern("beat", r"\b(?:beat|tempo|bpm|meter|groove)\b"),
    _pattern("rhythm", r"\b(?:rhythm|polyrhythm|syncopation|loop)\b"),
    _pattern("sequence", r"\b(?:sequence|arpeggio|chord|drone)\b"),
)
_LIGHT_COLOR_PATTERNS = (
    _pattern("glow", r"\b(?:light|glow|luminous|radiant|shadow|bloom)\b"),
    _pattern("neon", r"\b(?:neon|iridescent|chrome|metallic|glass)\b"),
    _pattern("palette", r"\b(?:palette|color|gradient|monochrome|pastel)\b"),
    _pattern("hue", r"\b(?:red|orange|amber|gold|green|cyan|blue|violet)\b"),
)
_AUDIO_PATTERNS = (
    _pattern("audio", r"\b(?:audio|sound|music|synth|sonic|tone\.js)\b"),
    _pattern("audio reactive", r"\b(?:audio[\s-]?reactive|sound[\s-]?reactive)\b"),
)
_INTERACTION_PATTERNS = (
    _pattern("pointer", r"\b(?:interactive|mouse|cursor|click|drag|touch)\b"),
    _pattern("input", r"\b(?:keyboard|scroll|webcam|microphone|live input)\b"),
)
_TRANSFORMATION_PATTERNS = (
    _pattern("morph", r"\b(?:morph|transform|transformation|metamorphosis)\b"),
    _pattern("reveal", r"\b(?:reveal|climax|culminate|evolve|emergence)\b"),
    _pattern("phase change", r"\b(?:from .+ to|become|dissolve|bloom)\b"),
)
_ABSTRACTION_PATTERNS = {
    "literal": re.compile(r"\b(?:literal|realistic|representational|portrait)\b"),
    "stylized": re.compile(r"\b(?:stylized|minimal|low-poly|cinematic|cartoon)\b"),
    "symbolic": re.compile(r"\b(?:symbolic|metaphor|archetype|ritual|myth)\b"),
    "abstract": re.compile(r"\b(?:abstract|nonrepresentational|generative|field)\b"),
}
