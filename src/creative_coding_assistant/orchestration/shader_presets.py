"""Deterministic shader-style preset detection and generation guidance."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.orchestration.sacred_geometry import (
    SacredGeometryGuidance,
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class ShaderPresetId(StrEnum):
    GLOW = "glow"
    AURA = "aura"
    PLASMA = "plasma"
    BLOOM_EMISSION = "bloom-like emission"
    REFRACTION = "refraction"
    GLASS_CRYSTAL = "glass / crystal"
    VOLUMETRIC_ATMOSPHERE = "volumetric atmosphere"
    FRACTAL_FIELD = "fractal field"
    KALEIDOSCOPIC_SYMMETRY = "kaleidoscopic symmetry"
    SACRED_LIGHT = "sacred light / ritual ambience"


class ShaderPresetGuidance(BaseModel):
    """Bounded implementation guidance for selected visual style presets."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    presets: tuple[ShaderPresetId, ...] = Field(min_length=1)
    color_behavior: tuple[str, ...] = Field(default_factory=tuple)
    light_material_behavior: tuple[str, ...] = Field(default_factory=tuple)
    motion_behavior: tuple[str, ...] = Field(default_factory=tuple)
    shader_structure: tuple[str, ...] = Field(default_factory=tuple)
    runtime_suitability: tuple[str, ...] = Field(default_factory=tuple)
    performance_constraints: tuple[str, ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class _PresetProfile:
    color: tuple[str, ...]
    light_material: tuple[str, ...]
    motion: tuple[str, ...]
    structure: tuple[str, ...]
    runtimes: tuple[str, ...]
    performance: tuple[str, ...]


_PRESET_PATTERNS = (
    (
        ShaderPresetId.BLOOM_EMISSION,
        re.compile(r"\b(?:bloom(?:-like)?|emissive|emission)\b"),
    ),
    (
        ShaderPresetId.GLASS_CRYSTAL,
        re.compile(r"\b(?:glass|crystal|crystalline)\b"),
    ),
    (
        ShaderPresetId.VOLUMETRIC_ATMOSPHERE,
        re.compile(
            r"\b(?:volumetric(?:\s+(?:fog|light|atmosphere))?|"
            r"atmospheric\s+fog|god\s*rays?|misty\s+atmosphere)\b"
        ),
    ),
    (
        ShaderPresetId.KALEIDOSCOPIC_SYMMETRY,
        re.compile(r"\b(?:kaleidoscop(?:e|ic)|mirrored\s+symmetry)\b"),
    ),
    (
        ShaderPresetId.SACRED_LIGHT,
        re.compile(r"\b(?:sacred\s+light|ritual\s+ambience|ritual\s+light)\b"),
    ),
    (
        ShaderPresetId.FRACTAL_FIELD,
        re.compile(r"\b(?:fractal(?:\s+field)?|recursive\s+field)\b"),
    ),
    (
        ShaderPresetId.REFRACTION,
        re.compile(r"\b(?:refract(?:ion|ive|ed)?|chromatic\s+dispersion)\b"),
    ),
    (
        ShaderPresetId.PLASMA,
        re.compile(r"\bplasma\b"),
    ),
    (
        ShaderPresetId.AURA,
        re.compile(r"\b(?:aura|halo)\b"),
    ),
    (
        ShaderPresetId.GLOW,
        re.compile(r"\b(?:glow|glowing|luminous|luminescent)\b"),
    ),
)

_PROFILES = {
    ShaderPresetId.GLOW: _PresetProfile(
        color=("Use a bright core color with a restrained falloff into darker tones.",),
        light_material=(
            "Build glow from layered emission or distance-based intensity, not "
            "unbounded blur.",
        ),
        motion=("Modulate intensity slowly and avoid full-frame flashing.",),
        structure=("Separate the luminous mask from the final color composition.",),
        runtimes=("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js"),
        performance=(
            "Use a bounded number of glow layers or samples and clamp intensity.",
        ),
    ),
    ShaderPresetId.AURA: _PresetProfile(
        color=("Use soft hue transitions around a stable central silhouette.",),
        light_material=(
            "Treat the aura as a stylized rim or field rather than physical light.",
        ),
        motion=("Use low-frequency breathing, drift, or phase modulation.",),
        structure=(
            "Derive a soft outer field from distance, noise, or layered contours.",
        ),
        runtimes=("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js"),
        performance=("Keep noise octaves and layered contours bounded.",),
    ),
    ShaderPresetId.PLASMA: _PresetProfile(
        color=("Use a compact palette with smooth phase-based color mixing.",),
        light_material=(
            "Use procedural intensity bands instead of simulated physical material.",
        ),
        motion=("Animate a small set of phase-shifted waves at different speeds.",),
        structure=(
            "Combine bounded sine fields, coordinates, and optional low-cost noise.",
        ),
        runtimes=("GLSL", "Hydra", "p5.js", "Three.js", "React Three Fiber"),
        performance=(
            "Avoid excessive trigonometric layers and normalize coordinates.",
        ),
    ),
    ShaderPresetId.BLOOM_EMISSION: _PresetProfile(
        color=("Reserve the brightest values for a small emissive region.",),
        light_material=(
            "Use bloom-like layering or emissive contrast without claiming a "
            "post-processing bloom pass unless one is implemented.",
        ),
        motion=("Pulse emission within a narrow intensity range.",),
        structure=(
            "Keep an explicit emission mask that can feed a supported bloom or "
            "additive approximation.",
        ),
        runtimes=("Three.js", "React Three Fiber", "GLSL", "Hydra", "p5.js"),
        performance=(
            "Prefer one bounded post-process pass or a small additive layer stack.",
        ),
    ),
    ShaderPresetId.REFRACTION: _PresetProfile(
        color=(
            "Use subtle channel offsets or environment color shifts at strong angles.",
        ),
        light_material=(
            "Treat refraction as a stylized screen-space or environment distortion "
            "unless a physical transmission model is implemented.",
        ),
        motion=(
            "Animate normals or distortion fields slowly enough to remain legible.",
        ),
        structure=(
            "Separate the distortion field, sampled source, and edge treatment.",
        ),
        runtimes=("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js"),
        performance=(
            "Limit texture reads and use a low-sample approximation on 2D runtimes.",
        ),
    ),
    ShaderPresetId.GLASS_CRYSTAL: _PresetProfile(
        color=("Use restrained tinting with controlled highlights and darker edges.",),
        light_material=(
            "Combine transparency, reflection, and stylized transmission only where "
            "the selected runtime supports them.",
        ),
        motion=("Use slow rotation or normal modulation to reveal facets.",),
        structure=(
            "Separate surface shape, normals, edge response, and environment color.",
        ),
        runtimes=("Three.js", "React Three Fiber", "GLSL", "p5.js"),
        performance=(
            "Use a bounded reflection/refraction approximation and avoid deep "
            "ray loops.",
        ),
    ),
    ShaderPresetId.VOLUMETRIC_ATMOSPHERE: _PresetProfile(
        color=("Use low-contrast depth gradients with one controlled light color.",),
        light_material=(
            "Treat fog and light shafts as stylized atmosphere unless volumetric "
            "integration is explicitly implemented.",
        ),
        motion=("Move density fields slowly to preserve depth cues.",),
        structure=("Use depth fog, layered noise, or a short bounded ray-march.",),
        runtimes=("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js"),
        performance=("Cap ray-march steps, noise octaves, and transparent layers.",),
    ),
    ShaderPresetId.FRACTAL_FIELD: _PresetProfile(
        color=("Tie color bands to bounded iteration or distance values.",),
        light_material=(
            "Use distance or iteration contrast to reveal structure without "
            "overexposing fine detail.",
        ),
        motion=("Animate parameters gradually and keep the focal region stable.",),
        structure=(
            "Use bounded recursive transforms, domain folding, or iterative distance.",
        ),
        runtimes=("GLSL", "Hydra", "p5.js", "Three.js", "React Three Fiber"),
        performance=(
            "Set a strict iteration limit and provide an early-exit condition.",
        ),
    ),
    ShaderPresetId.KALEIDOSCOPIC_SYMMETRY: _PresetProfile(
        color=("Use segment-aware color variation without obscuring the symmetry.",),
        light_material=(
            "Keep highlights aligned with the repeated angular structure.",
        ),
        motion=("Rotate or phase alternating segments with restrained speed.",),
        structure=("Fold polar coordinates into an explicit, bounded segment count.",),
        runtimes=("GLSL", "Hydra", "p5.js", "Three.js", "React Three Fiber"),
        performance=("Keep the segment count explicit and avoid recursive mirroring.",),
    ),
    ShaderPresetId.SACRED_LIGHT: _PresetProfile(
        color=(
            "Use a restrained gold, amber, white, or requested palette with deep "
            "negative space.",
        ),
        light_material=(
            "Use luminous edges, measured halos, or soft shafts as visual motifs "
            "without assigning spiritual authority.",
        ),
        motion=("Use slow reveals, breathing light, or measured radial pulses.",),
        structure=("Anchor light behavior to the requested geometric hierarchy.",),
        runtimes=("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js"),
        performance=("Limit layered halos, bloom approximations, and fog samples.",),
    ),
}

_RUNTIME_ALIASES = {
    "glsl": "GLSL",
    "shader": "GLSL",
    "three": "Three.js",
    "three.js": "Three.js",
    "threejs": "Three.js",
    "react three fiber": "React Three Fiber",
    "r3f": "React Three Fiber",
    "hydra": "Hydra",
    "p5": "p5.js",
    "p5.js": "p5.js",
    "p5js": "p5.js",
}
_VISUAL_RUNTIMES = frozenset(_RUNTIME_ALIASES.values())
_SACRED_KALEIDOSCOPIC_CONCEPTS = frozenset(
    {
        "mandala",
        "yantra",
        "Sri Yantra",
        "Flower of Life",
        "radial symmetry",
    }
)
_SACRED_LIGHT_CONCEPTS = frozenset(
    {
        "temple geometry",
        "cathedral geometry",
    }
)
_GLOBAL_CONSTRAINTS = (
    "Treat shader presets as implementation guidance, not proof of physical accuracy.",
    "Do not imply unsupported post-processing, transmission, or volumetric features.",
)


def derive_shader_preset_guidance(
    query: str,
    *,
    output_modality: str | None = None,
    mood_atmosphere: Sequence[str] = (),
    color_material_direction: Sequence[str] = (),
    runtime_recommendations: Sequence[str] = (),
    selected_runtime: str | None = None,
    sacred_geometry: SacredGeometryGuidance | None = None,
    base_guidance: ShaderPresetGuidance | None = None,
) -> ShaderPresetGuidance | None:
    """Derive bounded visual preset guidance from request-visible signals."""

    if output_modality == "audio":
        return base_guidance

    presets = detect_shader_presets(
        query,
        mood_atmosphere=mood_atmosphere,
        color_material_direction=color_material_direction,
        sacred_geometry=sacred_geometry,
    )
    if not presets:
        return base_guidance

    profiles = tuple(_PROFILES[preset] for preset in presets)
    current = ShaderPresetGuidance(
        presets=presets,
        color_behavior=_collect(profiles, "color"),
        light_material_behavior=_collect(profiles, "light_material"),
        motion_behavior=_collect(profiles, "motion"),
        shader_structure=_collect(profiles, "structure"),
        runtime_suitability=_runtime_suitability(
            profiles,
            runtime_recommendations=runtime_recommendations,
            selected_runtime=selected_runtime,
        ),
        performance_constraints=_merge(
            _collect(profiles, "performance"),
            _GLOBAL_CONSTRAINTS,
        ),
    )
    if base_guidance is None:
        return current
    return _merge_guidance(base_guidance, current)


def detect_shader_presets(
    query: str,
    *,
    mood_atmosphere: Sequence[str] = (),
    color_material_direction: Sequence[str] = (),
    sacred_geometry: SacredGeometryGuidance | None = None,
) -> tuple[ShaderPresetId, ...]:
    """Detect a bounded preset set from explicit and structured metadata cues."""

    normalized = " ".join(query.strip().lower().split())
    presets = [
        preset for preset, pattern in _PRESET_PATTERNS if pattern.search(normalized)
    ]
    color_values = {value.lower() for value in color_material_direction}
    mood_values = {value.lower() for value in mood_atmosphere}

    if any("glass" in value for value in color_values):
        presets.append(ShaderPresetId.GLASS_CRYSTAL)
    if any("neon" in value for value in color_values):
        presets.append(ShaderPresetId.GLOW)
    if any("ethereal" in value for value in mood_values):
        presets.append(ShaderPresetId.AURA)
    if any("ritual" in value for value in mood_values):
        presets.append(ShaderPresetId.SACRED_LIGHT)

    if sacred_geometry is not None:
        concepts = set(sacred_geometry.concepts)
        if concepts & _SACRED_KALEIDOSCOPIC_CONCEPTS:
            presets.append(ShaderPresetId.KALEIDOSCOPIC_SYMMETRY)
        if "fractal symmetry" in concepts:
            presets.append(ShaderPresetId.FRACTAL_FIELD)
        if concepts & _SACRED_LIGHT_CONCEPTS:
            presets.append(ShaderPresetId.SACRED_LIGHT)

    return _dedupe_presets(presets)


def shader_preset_prompt_lines(
    guidance: ShaderPresetGuidance,
) -> tuple[str, ...]:
    """Render compact provider-independent shader implementation guidance."""

    lines = [
        "Shader/style presets: "
        + ", ".join(preset.value for preset in guidance.presets)
    ]
    _append_line(lines, "Preset color behavior", guidance.color_behavior)
    _append_line(
        lines,
        "Preset light and material behavior",
        guidance.light_material_behavior,
    )
    _append_line(lines, "Preset motion behavior", guidance.motion_behavior)
    _append_line(lines, "Preset shader structure", guidance.shader_structure)
    _append_line(lines, "Preset runtime suitability", guidance.runtime_suitability)
    _append_line(
        lines,
        "Preset performance constraints",
        guidance.performance_constraints,
    )
    return tuple(lines)


def _runtime_suitability(
    profiles: Sequence[_PresetProfile],
    *,
    runtime_recommendations: Sequence[str],
    selected_runtime: str | None,
) -> tuple[str, ...]:
    supported = set.intersection(*(set(profile.runtimes) for profile in profiles))
    requested = _normalize_runtimes(
        (selected_runtime,) if selected_runtime else runtime_recommendations
    )
    compatible = tuple(runtime for runtime in requested if runtime in supported)
    if compatible:
        return ("Use the selected compatible runtime: " + ", ".join(compatible) + ".",)
    if requested:
        return (
            "The selected runtime does not provide the full preset combination; "
            "use a bounded stylized approximation.",
        )
    ordered_supported = tuple(
        runtime
        for runtime in ("GLSL", "Three.js", "React Three Fiber", "Hydra", "p5.js")
        if runtime in supported
    )
    return ("Suitable visual runtimes: " + ", ".join(ordered_supported) + ".",)


def _normalize_runtimes(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        runtime = _RUNTIME_ALIASES.get(value.strip().lower())
        if runtime in _VISUAL_RUNTIMES and runtime not in normalized:
            normalized.append(runtime)
    return tuple(normalized)


def _merge_guidance(
    base: ShaderPresetGuidance,
    current: ShaderPresetGuidance,
) -> ShaderPresetGuidance:
    return ShaderPresetGuidance(
        presets=_dedupe_presets((*base.presets, *current.presets)),
        color_behavior=_merge(base.color_behavior, current.color_behavior),
        light_material_behavior=_merge(
            base.light_material_behavior,
            current.light_material_behavior,
        ),
        motion_behavior=_merge(base.motion_behavior, current.motion_behavior),
        shader_structure=_merge(base.shader_structure, current.shader_structure),
        runtime_suitability=_merge(
            current.runtime_suitability,
            base.runtime_suitability,
        ),
        performance_constraints=_merge(
            base.performance_constraints,
            current.performance_constraints,
        ),
    )


def _collect(
    profiles: Sequence[_PresetProfile],
    field: str,
) -> tuple[str, ...]:
    return _dedupe(item for profile in profiles for item in getattr(profile, field))


def _merge(*groups: Sequence[str]) -> tuple[str, ...]:
    return _dedupe(item for group in groups for item in group)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _dedupe_presets(
    values: Iterable[ShaderPresetId],
) -> tuple[ShaderPresetId, ...]:
    return tuple(dict.fromkeys(values))[:6]


def _append_line(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:
    if values:
        lines.append(f"{label}: {' '.join(values)}")
