"""Deterministic translation of creative intent into generation guidance."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.sacred_geometry import (
    SacredGeometryGuidance,
    derive_sacred_geometry_guidance,
    sacred_geometry_prompt_lines,
)
from creative_coding_assistant.orchestration.shader_presets import (
    ShaderPresetGuidance,
    derive_shader_preset_guidance,
    shader_preset_prompt_lines,
)
from creative_coding_assistant.orchestration.visual_styles import (
    VisualStyleGuidance,
    derive_visual_style_guidance,
    visual_style_prompt_lines,
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class CreativeOutputModality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"
    AUDIOVISUAL = "audiovisual"


class CreativeTranslation(BaseModel):
    """Bounded creative guidance derived only from request-visible signals."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    output_modality: CreativeOutputModality | None = None
    creative_intent: str = Field(min_length=1, max_length=280)
    symbolic_references: tuple[str, ...] = Field(default_factory=tuple)
    geometric_references: tuple[str, ...] = Field(default_factory=tuple)
    musical_references: tuple[str, ...] = Field(default_factory=tuple)
    mood_atmosphere: tuple[str, ...] = Field(default_factory=tuple)
    movement_language: tuple[str, ...] = Field(default_factory=tuple)
    color_material_direction: tuple[str, ...] = Field(default_factory=tuple)
    runtime_recommendations: tuple[str, ...] = Field(default_factory=tuple)
    structure_direction: tuple[str, ...] = Field(default_factory=tuple)
    generation_constraints: tuple[str, ...] = Field(default_factory=tuple)
    refinement_targets: tuple[str, ...] = Field(default_factory=tuple)
    sacred_geometry: SacredGeometryGuidance | None = None
    shader_presets: ShaderPresetGuidance | None = None
    visual_style: VisualStyleGuidance | None = None


_AUDIO_DOMAINS = frozenset(
    {
        CreativeCodingDomain.ABLETON_LIVE,
        CreativeCodingDomain.MAX_MSP,
        CreativeCodingDomain.P5_SOUND,
        CreativeCodingDomain.PURE_DATA,
        CreativeCodingDomain.SONIC_PI,
        CreativeCodingDomain.SUPERCOLLIDER,
        CreativeCodingDomain.TIDALCYCLES,
        CreativeCodingDomain.TONE_JS,
        CreativeCodingDomain.VCV_RACK,
        CreativeCodingDomain.WEB_AUDIO_API,
    }
)
_VISUAL_DOMAINS = frozenset(set(CreativeCodingDomain) - _AUDIO_DOMAINS)

_SYMBOLIC_REFERENCES = (
    "alchemy",
    "labyrinth",
    "lotus",
    "mandala",
    "ouroboros",
    "sacred geometry",
    "sigil",
    "tarot",
    "tree of life",
    "yin yang",
)
_GEOMETRIC_REFERENCES = (
    "cathedral geometry",
    "circle",
    "concentric",
    "fibonacci",
    "fractal",
    "fractal symmetry",
    "flower of life",
    "golden ratio",
    "grid",
    "hexagon",
    "lissajous",
    "merkaba",
    "metatron's cube",
    "polyhedron",
    "radial symmetry",
    "sacred geometry",
    "spiral",
    "sri yantra",
    "symmetry",
    "temple geometry",
    "tessellation",
    "torus",
    "vesica piscis",
    "voronoi",
    "yantra",
)
_MUSICAL_REFERENCES = (
    "arpeggio",
    "beat",
    "bpm",
    "chord",
    "drone",
    "harmony",
    "melody",
    "polyrhythm",
    "rhythm",
    "scale",
    "sequence",
    "syncopation",
    "tempo",
)
_MOOD_REFERENCES = (
    "calm",
    "cinematic",
    "dark",
    "dreamlike",
    "energetic",
    "ethereal",
    "hypnotic",
    "meditative",
    "minimal",
    "organic",
    "playful",
    "ritual",
    "serene",
    "tense",
)
_MOVEMENT_PATTERNS = (
    ("breathe", re.compile(r"\b(?:breathe|breathing)\b")),
    ("cascade", re.compile(r"\bcascad(?:e|es|ing)\b")),
    ("drift", re.compile(r"\bdrift(?:s|ed|ing)?\b")),
    ("flow", re.compile(r"\bflow(?:s|ed|ing)?\b")),
    ("morph", re.compile(r"\bmorph(?:s|ed|ing)?\b")),
    ("orbit", re.compile(r"\borbit(?:s|ed|ing)?\b")),
    ("oscillate", re.compile(r"\boscillat(?:e|es|ed|ing|ion)\b")),
    ("pulse", re.compile(r"\bpuls(?:e|es|ed|ing)\b")),
    ("ripple", re.compile(r"\brippl(?:e|es|ed|ing)\b")),
    ("rotate", re.compile(r"\brotat(?:e|es|ed|ing|ion)\b")),
    ("swarm", re.compile(r"\bswarm(?:s|ed|ing)?\b")),
)
_COLOR_MATERIAL_REFERENCES = (
    "amber",
    "black and white",
    "blue",
    "chrome",
    "cyan",
    "glass",
    "gold",
    "gradient",
    "green",
    "iridescent",
    "metallic",
    "monochrome",
    "neon",
    "orange",
    "pastel",
    "red",
    "translucent",
    "violet",
    "white",
)
_CONSTRAINT_REFERENCES = (
    "60 fps",
    "accessible",
    "low-poly",
    "minimal",
    "mobile",
    "monochrome",
    "no autoplay",
    "performance",
    "responsive",
    "seamless loop",
)

_AUDIO_SIGNAL = re.compile(
    r"\b(?:audio|beat|chord|drone|melody|music|musical|rhythm|sound|synth|tempo)\b"
)
_VISUAL_SIGNAL = re.compile(
    r"\b(?:animation|canvas|color|geometry|image|light|particles?|scene|shader|"
    r"sketch|sculpture|visual)\b"
)
_AUDIOVISUAL_SIGNAL = re.compile(
    r"\b(?:audio[\s-]?reactive|audiovisual|music visuali[sz]er|"
    r"sound[\s-]?reactive)\b"
)
_RUNTIME_SIGNAL_ORDER = (
    (re.compile(r"\b(?:tone\.js|tonejs)\b"), "Tone.js"),
    (re.compile(r"\b(?:web audio|audiocontext)\b"), "Web Audio API"),
    (re.compile(r"\bhydra\b"), "Hydra"),
    (re.compile(r"\b(?:glsl|fragment shader|shader)\b"), "GLSL"),
    (re.compile(r"\b(?:react three fiber|r3f)\b"), "React Three Fiber"),
    (re.compile(r"\b(?:three\.js|threejs|3d scene)\b"), "Three.js"),
    (re.compile(r"\b(?:p5\.js|p5js|creative coding sketch)\b"), "p5.js"),
)

_RUNTIME_LABELS = {
    CreativeCodingDomain.ABLETON_LIVE: "Ableton Live",
    CreativeCodingDomain.CANVAS_2D: "Canvas 2D",
    CreativeCodingDomain.GLSL: "GLSL",
    CreativeCodingDomain.HYDRA: "Hydra",
    CreativeCodingDomain.MAX_MSP: "Max/MSP",
    CreativeCodingDomain.P5_JS: "p5.js",
    CreativeCodingDomain.P5_SOUND: "p5.sound",
    CreativeCodingDomain.PURE_DATA: "Pure Data",
    CreativeCodingDomain.REACT_THREE_FIBER: "React Three Fiber",
    CreativeCodingDomain.SHADERTOY: "Shadertoy",
    CreativeCodingDomain.SONIC_PI: "Sonic Pi",
    CreativeCodingDomain.SUPERCOLLIDER: "SuperCollider",
    CreativeCodingDomain.THREE_JS: "Three.js",
    CreativeCodingDomain.TIDALCYCLES: "TidalCycles",
    CreativeCodingDomain.TONE_JS: "Tone.js",
    CreativeCodingDomain.VCV_RACK: "VCV Rack",
    CreativeCodingDomain.WEB_AUDIO_API: "Web Audio API",
}


def derive_creative_translation(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    selected_runtime: str | None = None,
    has_image_references: bool = False,
    base_translation: CreativeTranslation | None = None,
) -> CreativeTranslation:
    """Translate explicit request cues into bounded generation guidance."""

    normalized = _normalize_text(query)
    domain_tuple = _dedupe_domains(domains)
    symbolic = _matched_phrases(normalized, _SYMBOLIC_REFERENCES)
    geometric = _matched_phrases(normalized, _GEOMETRIC_REFERENCES)
    musical = _matched_phrases(normalized, _MUSICAL_REFERENCES)
    mood = _matched_phrases(normalized, _MOOD_REFERENCES)
    movement = _matched_patterns(normalized, _MOVEMENT_PATTERNS)
    color_material = _matched_phrases(normalized, _COLOR_MATERIAL_REFERENCES)
    modality = _derive_modality(
        normalized,
        domains=domain_tuple,
        geometric=geometric,
        musical=musical,
    )
    runtimes = _runtime_recommendations(
        normalized,
        domains=domain_tuple,
        modality=modality,
        selected_runtime=selected_runtime,
    )
    constraints = list(_matched_phrases(normalized, _CONSTRAINT_REFERENCES))
    if modality in {
        CreativeOutputModality.AUDIO,
        CreativeOutputModality.AUDIOVISUAL,
    }:
        constraints.append("Require explicit user interaction before audio playback")
    if has_image_references:
        constraints.append("Use supplied image references for visual direction")

    sacred_geometry = derive_sacred_geometry_guidance(
        query,
        output_modality=modality.value if modality is not None else None,
        base_guidance=(
            base_translation.sacred_geometry
            if base_translation is not None
            else None
        ),
    )
    shader_presets = derive_shader_preset_guidance(
        query,
        output_modality=modality.value if modality is not None else None,
        mood_atmosphere=mood,
        color_material_direction=color_material,
        runtime_recommendations=runtimes,
        selected_runtime=selected_runtime,
        sacred_geometry=sacred_geometry,
        base_guidance=(
            base_translation.shader_presets
            if base_translation is not None
            else None
        ),
    )
    visual_style = derive_visual_style_guidance(
        query,
        output_modality=modality.value if modality is not None else None,
        mood_atmosphere=mood,
        color_material_direction=color_material,
        runtime_recommendations=runtimes,
        selected_runtime=selected_runtime,
        sacred_geometry=sacred_geometry,
        shader_presets=shader_presets,
        base_guidance=(
            base_translation.visual_style
            if base_translation is not None
            else None
        ),
    )
    current = CreativeTranslation(
        output_modality=modality,
        creative_intent=_truncate_text(query, 280),
        symbolic_references=symbolic,
        geometric_references=geometric,
        musical_references=musical,
        mood_atmosphere=mood,
        movement_language=movement,
        color_material_direction=color_material,
        runtime_recommendations=runtimes,
        structure_direction=_structure_direction(
            modality=modality,
            geometric=geometric,
            musical=musical,
        ),
        generation_constraints=_dedupe_text(constraints),
        refinement_targets=_refinement_targets(
            mood=mood,
            movement=movement,
            color_material=color_material,
        ),
        sacred_geometry=sacred_geometry,
        shader_presets=shader_presets,
        visual_style=visual_style,
    )
    if base_translation is None:
        return current
    return _merge_refinement_translation(
        base_translation,
        current,
        refinement_query=query,
    )


def creative_translation_prompt_lines(
    translation: CreativeTranslation,
) -> tuple[str, ...]:
    """Render compact provider-independent guidance from one translation."""

    lines = [
        (
            "Intended modality: "
            + (
                translation.output_modality.value
                if translation.output_modality is not None
                else "not explicitly specified"
            )
        ),
        f"Creative intent: {translation.creative_intent}",
    ]
    _append_list_line(lines, "Symbolic references", translation.symbolic_references)
    _append_list_line(lines, "Geometric references", translation.geometric_references)
    _append_list_line(lines, "Musical references", translation.musical_references)
    _append_list_line(lines, "Mood and atmosphere", translation.mood_atmosphere)
    _append_list_line(lines, "Movement language", translation.movement_language)
    _append_list_line(
        lines,
        "Color and material direction",
        translation.color_material_direction,
    )
    _append_list_line(
        lines,
        "Recommended runtime families",
        translation.runtime_recommendations,
    )
    lines.extend(translation.structure_direction)
    _append_list_line(
        lines,
        "Generation constraints",
        translation.generation_constraints,
    )
    _append_list_line(lines, "Refinement targets", translation.refinement_targets)
    if translation.sacred_geometry is not None:
        lines.extend(sacred_geometry_prompt_lines(translation.sacred_geometry))
    if translation.shader_presets is not None:
        lines.extend(shader_preset_prompt_lines(translation.shader_presets))
    if translation.visual_style is not None:
        lines.extend(visual_style_prompt_lines(translation.visual_style))
    if translation.symbolic_references:
        lines.append(
            "Use symbolic references as requested motifs only; do not invent "
            "unsupported symbolic meaning."
        )
    return tuple(lines)


def _derive_modality(
    query: str,
    *,
    domains: tuple[CreativeCodingDomain, ...],
    geometric: tuple[str, ...],
    musical: tuple[str, ...],
) -> CreativeOutputModality | None:
    has_audio = bool(set(domains) & _AUDIO_DOMAINS) or bool(
        musical or _AUDIO_SIGNAL.search(query)
    )
    has_visual = bool(set(domains) & _VISUAL_DOMAINS) or bool(
        geometric or _VISUAL_SIGNAL.search(query)
    )
    if _AUDIOVISUAL_SIGNAL.search(query) or (has_audio and has_visual):
        return CreativeOutputModality.AUDIOVISUAL
    if has_audio:
        return CreativeOutputModality.AUDIO
    if has_visual:
        return CreativeOutputModality.VISUAL
    return None


def _merge_refinement_translation(
    base: CreativeTranslation,
    current: CreativeTranslation,
    *,
    refinement_query: str,
) -> CreativeTranslation:
    refinement_target = f"Current refinement: {_truncate_text(refinement_query, 180)}"
    return CreativeTranslation(
        output_modality=base.output_modality or current.output_modality,
        creative_intent=base.creative_intent,
        symbolic_references=_merge_text(
            base.symbolic_references,
            current.symbolic_references,
        ),
        geometric_references=_merge_text(
            base.geometric_references,
            current.geometric_references,
        ),
        musical_references=_merge_text(
            base.musical_references,
            current.musical_references,
        ),
        mood_atmosphere=_merge_text(
            base.mood_atmosphere,
            current.mood_atmosphere,
        ),
        movement_language=_merge_text(
            base.movement_language,
            current.movement_language,
        ),
        color_material_direction=_merge_text(
            base.color_material_direction,
            current.color_material_direction,
        ),
        runtime_recommendations=_merge_text(
            current.runtime_recommendations,
            base.runtime_recommendations,
        ),
        structure_direction=_merge_text(
            base.structure_direction,
            current.structure_direction,
        ),
        generation_constraints=_merge_text(
            base.generation_constraints,
            current.generation_constraints,
        ),
        refinement_targets=_merge_text(
            base.refinement_targets,
            current.refinement_targets,
            (refinement_target,),
        ),
        sacred_geometry=current.sacred_geometry or base.sacred_geometry,
        shader_presets=current.shader_presets or base.shader_presets,
        visual_style=current.visual_style or base.visual_style,
    )


def _runtime_recommendations(
    query: str,
    *,
    domains: tuple[CreativeCodingDomain, ...],
    modality: CreativeOutputModality | None,
    selected_runtime: str | None,
) -> tuple[str, ...]:
    recommendations: list[str] = []
    if selected_runtime:
        recommendations.append(_format_runtime_label(selected_runtime))
    for domain in domains:
        recommendations.append(
            _RUNTIME_LABELS.get(domain, domain.value.replace("_", " ").title())
        )
    for pattern, label in _RUNTIME_SIGNAL_ORDER:
        if pattern.search(query):
            recommendations.append(label)

    if not recommendations and modality is CreativeOutputModality.AUDIO:
        recommendations.append("Tone.js")
    elif not recommendations and modality is CreativeOutputModality.AUDIOVISUAL:
        recommendations.extend(("p5.js", "Tone.js"))
    elif not recommendations and modality is CreativeOutputModality.VISUAL:
        recommendations.append("p5.js")

    return _dedupe_text(recommendations)


def _structure_direction(
    *,
    modality: CreativeOutputModality | None,
    geometric: tuple[str, ...],
    musical: tuple[str, ...],
) -> tuple[str, ...]:
    direction: list[str] = []
    if geometric:
        direction.append(
            "Build visual structure from the requested geometry: "
            + ", ".join(geometric)
            + "."
        )
    if musical:
        direction.append(
            "Build audio or timing structure from: " + ", ".join(musical) + "."
        )
    if (
        modality is CreativeOutputModality.AUDIOVISUAL
        and geometric
        and musical
    ):
        direction.append(
            "Coordinate visual changes with the requested musical structure."
        )
    return tuple(direction)


def _refinement_targets(
    *,
    mood: tuple[str, ...],
    movement: tuple[str, ...],
    color_material: tuple[str, ...],
) -> tuple[str, ...]:
    targets: list[str] = []
    if mood:
        targets.append("Preserve atmosphere: " + ", ".join(mood))
    if movement:
        targets.append("Tune motion character: " + ", ".join(movement))
    if color_material:
        targets.append(
            "Preserve color and material direction: " + ", ".join(color_material)
        )
    return tuple(targets)


def _matched_phrases(query: str, phrases: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        phrase
        for phrase in phrases
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query)
    )[:8]


def _matched_patterns(
    query: str,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[str, ...]:
    return tuple(label for label, pattern in patterns if pattern.search(query))[:8]


def _dedupe_domains(
    domains: Sequence[CreativeCodingDomain],
) -> tuple[CreativeCodingDomain, ...]:
    return tuple(dict.fromkeys(domains))


def _dedupe_text(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))[:8]


def _merge_text(*values: Sequence[str]) -> tuple[str, ...]:
    return _dedupe_text([item for group in values for item in group])


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _truncate_text(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _format_runtime_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    labels = {
        "glsl": "GLSL",
        "hydra": "Hydra",
        "p5": "p5.js",
        "three": "Three.js",
        "tone": "Tone.js",
    }
    return labels.get(normalized, value.strip())


def _append_list_line(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:
    if values:
        lines.append(f"{label}: {', '.join(values)}")
