"""Bounded image-reference fusion guidance for creative generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class ReferenceImageMetadata(Protocol):
    id: str
    name: str
    mime_type: str
    size_bytes: int


class ReferenceFusionGuidance(BaseModel):
    """Deterministic, non-identifying guidance from uploaded references."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    source_count: int = Field(ge=1, le=4)
    source_names: tuple[str, ...] = Field(min_length=1, max_length=4)
    palette_direction: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    composition: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    lighting_contrast: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    texture_material_cues: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    geometric_structure: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mood_atmosphere: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    motion_implications: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    runtime_style_implications: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    safety_constraints: tuple[str, ...] = Field(min_length=2, max_length=5)
    summary: str = Field(min_length=1, max_length=360)


_REFERENCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_PALETTE_CUES = {
    "amber": "amber highlights",
    "black": "deep dark negative space",
    "blue": "cool blue palette",
    "cool": "cool blue palette",
    "cyan": "cyan accent palette",
    "gold": "warm gold accents",
    "green": "green organic palette",
    "mono": "monochrome value structure",
    "monochrome": "monochrome value structure",
    "neon": "neon accent contrast",
    "pastel": "soft pastel palette",
    "red": "warm red accents",
    "teal": "teal accent palette",
    "violet": "violet spectral accents",
    "warm": "warm palette bias",
    "white": "bright high-value highlights",
}
_COMPOSITION_CUES = {
    "asymmetric": "asymmetric balance",
    "center": "center-weighted focal composition",
    "central": "center-weighted focal composition",
    "collage": "layered collage composition",
    "grid": "grid-based spatial layout",
    "landscape": "wide landscape framing",
    "minimal": "minimal open-space composition",
    "portrait": "central subject framing without identity assumptions",
    "symmetry": "symmetrical composition",
    "triptych": "three-panel composition",
}
_LIGHTING_CUES = {
    "contrast": "high-contrast lighting",
    "dark": "low-key lighting",
    "glow": "soft emissive glow",
    "haze": "diffuse hazy lighting",
    "light": "bright diffuse lighting",
    "shadow": "strong shadow separation",
    "soft": "soft low-contrast lighting",
}
_TEXTURE_CUES = {
    "cloth": "woven textile texture",
    "fabric": "woven textile texture",
    "glass": "glasslike refraction cues",
    "grain": "subtle film-grain texture",
    "liquid": "fluid material behavior",
    "metal": "metallic material cues",
    "paper": "paper grain texture",
    "stone": "stone-like surface weight",
}
_GEOMETRY_CUES = {
    "circle": "circular structure",
    "fractal": "fractal repetition",
    "grid": "rectilinear grid",
    "hex": "hexagonal structure",
    "lattice": "lattice structure",
    "mandala": "radial mandala structure",
    "radial": "radial organization",
    "spiral": "spiral structure",
    "symmetry": "symmetry structure",
    "voronoi": "voronoi cellular structure",
}
_MOOD_CUES = {
    "calm": "calm atmosphere",
    "dream": "dreamlike atmosphere",
    "dreamy": "dreamlike atmosphere",
    "energetic": "energetic atmosphere",
    "ethereal": "ethereal atmosphere",
    "moody": "moody atmosphere",
    "nocturne": "nocturne mood",
    "ritual": "ritual atmosphere without symbolic overclaiming",
    "soft": "soft atmosphere",
}
_MOTION_CUES = {
    "drift": "slow drifting motion",
    "flow": "fluid flowing motion",
    "glitch": "controlled glitch motion",
    "pulse": "pulsing temporal behavior",
    "ripple": "ripple propagation",
    "swirl": "swirling motion",
}
_RUNTIME_CUES = {
    "3d": "Three.js or React Three Fiber can support spatial depth.",
    "bloom": "Shader bloom or postprocessing may suit the reference.",
    "glass": "Shader refraction presets may suit the material direction.",
    "glow": "GLSL or p5.js glow treatment may suit the reference.",
    "grid": "p5.js or GLSL can support procedural grid structure.",
    "liquid": "GLSL shader treatment may suit fluid material cues.",
    "particle": "p5.js or Three.js can support particle structure.",
    "shader": "GLSL shader treatment may suit the reference.",
}
_IDENTITY_TOKENS = {"face", "human", "person", "portrait", "selfie"}
_SAFETY_CONSTRAINTS = (
    "Use references for aesthetic, palette, composition, and material guidance only.",
    "Do not identify people, infer identity, or describe facial/person attributes.",
    "Do not claim exact copying or replication of any uploaded reference.",
)


def derive_reference_fusion_guidance(
    image_references: Sequence[ReferenceImageMetadata],
) -> ReferenceFusionGuidance | None:
    """Extract bounded non-identifying guidance from image reference metadata."""

    if not image_references:
        return None

    names = tuple(reference.name for reference in image_references[:4])
    tokens = _reference_tokens(names)
    palette = _matched_guidance(tokens, _PALETTE_CUES) or (
        "sample broad palette relationships from the references",
    )
    composition = _matched_guidance(tokens, _COMPOSITION_CUES) or (
        "use reference framing as directional layout guidance",
    )
    lighting = _matched_guidance(tokens, _LIGHTING_CUES) or (
        "use reference lighting and contrast only as broad tonal guidance",
    )
    texture = _matched_guidance(tokens, _TEXTURE_CUES) or (
        "translate material cues into procedural texture direction",
    )
    geometry = _matched_guidance(tokens, _GEOMETRY_CUES) or (
        "derive only broad spatial structure from the references",
    )
    mood = _matched_guidance(tokens, _MOOD_CUES) or (
        "preserve the broad mood implied by the references",
    )
    motion = _matched_guidance(tokens, _MOTION_CUES) or (
        "derive motion only when it supports the prompt and runtime",
    )
    runtime = _matched_guidance(tokens, _RUNTIME_CUES) or (
        "choose runtime/style treatments that support the fused reference cues",
    )
    safety = _SAFETY_CONSTRAINTS
    if tokens.intersection(_IDENTITY_TOKENS):
        safety = (
            *_SAFETY_CONSTRAINTS,
            "If a reference appears person-like, use only non-identifying layout cues.",
        )

    return ReferenceFusionGuidance(
        source_count=len(names),
        source_names=names,
        palette_direction=palette,
        composition=composition,
        lighting_contrast=lighting,
        texture_material_cues=texture,
        geometric_structure=geometry,
        mood_atmosphere=mood,
        motion_implications=motion,
        runtime_style_implications=runtime,
        safety_constraints=safety,
        summary=_summary(names, palette, composition, mood),
    )


def reference_fusion_prompt_lines(
    guidance: ReferenceFusionGuidance,
) -> tuple[str, ...]:
    """Render compact prompt lines for reference fusion metadata."""

    lines = [
        f"Reference fusion sources: {guidance.source_count}",
        f"Reference fusion summary: {guidance.summary}",
    ]
    _append_list_line(lines, "Reference palette direction", guidance.palette_direction)
    _append_list_line(lines, "Reference composition", guidance.composition)
    _append_list_line(lines, "Reference lighting", guidance.lighting_contrast)
    _append_list_line(
        lines,
        "Reference texture/material",
        guidance.texture_material_cues,
    )
    _append_list_line(lines, "Reference geometry", guidance.geometric_structure)
    _append_list_line(lines, "Reference mood", guidance.mood_atmosphere)
    _append_list_line(lines, "Reference motion", guidance.motion_implications)
    _append_list_line(
        lines,
        "Reference runtime/style implications",
        guidance.runtime_style_implications,
    )
    _append_list_line(lines, "Reference safety", guidance.safety_constraints)
    return tuple(lines)


def _reference_tokens(names: tuple[str, ...]) -> set[str]:
    normalized = " ".join(names).lower().replace("_", " ").replace("-", " ")
    return set(_REFERENCE_TOKEN_PATTERN.findall(normalized))


def _matched_guidance(
    tokens: set[str],
    cue_map: dict[str, str],
) -> tuple[str, ...]:
    matches = tuple(
        dict.fromkeys(cue for token, cue in cue_map.items() if token in tokens)
    )
    return matches[:5]


def _summary(
    names: tuple[str, ...],
    palette: tuple[str, ...],
    composition: tuple[str, ...],
    mood: tuple[str, ...],
) -> str:
    source_label = names[0] if len(names) == 1 else f"{len(names)} references"
    return (
        f"Fused {source_label} into non-identifying guidance for "
        f"{palette[0]}, {composition[0]}, and {mood[0]}."
    )


def _append_list_line(lines: list[str], label: str, values: tuple[str, ...]) -> None:
    if values:
        lines.append(f"{label}: " + " / ".join(values))
