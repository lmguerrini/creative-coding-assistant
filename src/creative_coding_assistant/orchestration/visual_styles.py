"""Deterministic visual-style detection and practical generation guidance."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.orchestration.sacred_geometry import (
    SacredGeometryGuidance,
)
from creative_coding_assistant.orchestration.shader_presets import (
    ShaderPresetGuidance,
    ShaderPresetId,
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class VisualStyleId(StrEnum):
    MINIMAL = "minimal"
    CYBERPUNK = "cyberpunk"
    ORGANIC = "organic"
    RITUAL = "ritual"
    SACRED_GEOMETRY = "sacred geometry"
    GENERATIVE_MODERNISM = "generative modernism"
    RETRO_COMPUTATIONAL = "retro computational"
    ETHEREAL = "ethereal"
    PSYCHEDELIC = "psychedelic"
    ARCHITECTURAL = "architectural"
    MONOCHROME = "monochrome"
    MAXIMALIST = "maximalist"


class VisualStyleGuidance(BaseModel):
    """Bounded implementation guidance for selected artistic identities."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    styles: tuple[VisualStyleId, ...] = Field(min_length=1)
    palette_behavior: tuple[str, ...] = Field(default_factory=tuple)
    contrast_behavior: tuple[str, ...] = Field(default_factory=tuple)
    composition_tendencies: tuple[str, ...] = Field(default_factory=tuple)
    motion_tendencies: tuple[str, ...] = Field(default_factory=tuple)
    texture_tendencies: tuple[str, ...] = Field(default_factory=tuple)
    spatial_organization: tuple[str, ...] = Field(default_factory=tuple)
    runtime_suitability: tuple[str, ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class _StyleProfile:
    palette: tuple[str, ...]
    contrast: tuple[str, ...]
    composition: tuple[str, ...]
    motion: tuple[str, ...]
    texture: tuple[str, ...]
    spatial: tuple[str, ...]
    runtimes: tuple[str, ...]


_STYLE_PATTERNS = (
    (
        VisualStyleId.SACRED_GEOMETRY,
        re.compile(r"\b(?:sacred[\s-]+geometry|geometric[\s-]+ritual)\b"),
    ),
    (
        VisualStyleId.GENERATIVE_MODERNISM,
        re.compile(
            r"\b(?:generative[\s-]+modernis(?:m|t)|"
            r"modernist[\s-]+generative|bauhaus)\b"
        ),
    ),
    (
        VisualStyleId.RETRO_COMPUTATIONAL,
        re.compile(
            r"\b(?:retro[\s-]+computational|retro[\s-]+computer|demoscene|"
            r"early[\s-]+computer[\s-]+graphics|crt[\s-]+aesthetic)\b"
        ),
    ),
    (
        VisualStyleId.CYBERPUNK,
        re.compile(r"\b(?:cyberpunk|cyber[\s-]+punk)\b"),
    ),
    (
        VisualStyleId.PSYCHEDELIC,
        re.compile(r"\b(?:psychedelic|psychedelia|trippy)\b"),
    ),
    (
        VisualStyleId.ARCHITECTURAL,
        re.compile(r"\b(?:architectural|architecture|spatial[\s-]+installation)\b"),
    ),
    (
        VisualStyleId.MONOCHROME,
        re.compile(
            r"\b(?:monochrome|monochromatic|grayscale|black[\s-]+and[\s-]+white)\b"
        ),
    ),
    (
        VisualStyleId.MAXIMALIST,
        re.compile(r"\b(?:maximalist|maximalism)\b"),
    ),
    (
        VisualStyleId.MINIMAL,
        re.compile(r"\b(?:minimal|minimalist|minimalism)\b"),
    ),
    (
        VisualStyleId.ORGANIC,
        re.compile(r"\b(?:organic|biomorphic)\b"),
    ),
    (
        VisualStyleId.RITUAL,
        re.compile(r"\b(?:ritual|ceremonial)\b"),
    ),
    (
        VisualStyleId.ETHEREAL,
        re.compile(r"\b(?:ethereal|otherworldly|weightless)\b"),
    ),
)

_PROFILES = {
    VisualStyleId.MINIMAL: _StyleProfile(
        palette=("Use one dominant tone plus one restrained accent.",),
        contrast=("Create hierarchy through spacing and value, not visual noise.",),
        composition=(
            "Use few elements, clear alignment, and deliberate negative space.",
        ),
        motion=("Use one primary motion system with slow, readable transitions.",),
        texture=("Keep surfaces clean with little or no procedural grain.",),
        spatial=("Favor a stable focal point and generous separation.",),
        runtimes=("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra"),
    ),
    VisualStyleId.CYBERPUNK: _StyleProfile(
        palette=(
            "Use dark fields with bounded neon cyan, magenta, or requested accents.",
        ),
        contrast=("Pair deep shadows with narrow high-luminance edges.",),
        composition=(
            "Layer interfaces, structures, or light traces without hiding "
            "the focal form.",
        ),
        motion=(
            "Use scanning, flicker, data flow, or glitch in controlled intervals.",
        ),
        texture=("Add sparse scanlines, digital noise, or worn synthetic surfaces.",),
        spatial=("Use layered depth, perspective corridors, or dense urban framing.",),
        runtimes=("Three.js", "React Three Fiber", "GLSL", "Hydra", "p5.js"),
    ),
    VisualStyleId.ORGANIC: _StyleProfile(
        palette=("Use related natural hues with gradual local variation.",),
        contrast=("Prefer soft transitions with a few structurally important edges.",),
        composition=(
            "Grow forms through branching, clustering, or biomorphic repetition.",
        ),
        motion=("Use drift, flow, breathing, and non-uniform oscillation.",),
        texture=(
            "Use bounded noise, cellular variation, or layered translucent marks.",
        ),
        spatial=(
            "Organize elements as fields, colonies, or connected growth systems.",
        ),
        runtimes=("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra"),
    ),
    VisualStyleId.RITUAL: _StyleProfile(
        palette=("Use a restrained symbolic palette with deep negative space.",),
        contrast=("Concentrate luminosity around the primary motif or sequence.",),
        composition=(
            "Use repetition, thresholds, and centered or processional structure.",
        ),
        motion=("Use measured pulses, reveals, rotations, or cyclical transitions.",),
        texture=("Use subtle smoke, grain, stone, metal, or luminous line cues.",),
        spatial=(
            "Create a clear center, boundary, and ordered path through "
            "the composition.",
        ),
        runtimes=("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra"),
    ),
    VisualStyleId.SACRED_GEOMETRY: _StyleProfile(
        palette=(
            "Use controlled luminous accents that preserve geometric legibility.",
        ),
        contrast=("Separate primary geometry from supporting construction lines.",),
        composition=(
            "Build from explicit symmetry, proportion, and nested geometric hierarchy.",
        ),
        motion=(
            "Animate layers without breaking alignment or proportional relationships.",
        ),
        texture=("Keep texture subordinate to line, intersection, and repeated form.",),
        spatial=("Maintain a clear center, axis, or bounded geometric field.",),
        runtimes=("p5.js", "GLSL", "Three.js", "React Three Fiber", "Hydra"),
    ),
    VisualStyleId.GENERATIVE_MODERNISM: _StyleProfile(
        palette=("Use a limited graphic palette with deliberate color relationships.",),
        contrast=("Use crisp value blocks and clear figure-ground separation.",),
        composition=(
            "Use grids, modular systems, serial variation, and asymmetric balance.",
        ),
        motion=(
            "Animate rule changes, permutations, or measured modular transitions.",
        ),
        texture=(
            "Prefer flat fields, clean marks, and restrained print-like variation.",
        ),
        spatial=(
            "Organize the frame through grids, margins, and proportional modules.",
        ),
        runtimes=("p5.js", "GLSL", "Three.js", "React Three Fiber", "Hydra"),
    ),
    VisualStyleId.RETRO_COMPUTATIONAL: _StyleProfile(
        palette=("Use a small indexed or phosphor-inspired palette.",),
        contrast=(
            "Favor hard edges, stepped values, and readable pixel-scale contrast.",
        ),
        composition=(
            "Use raster grids, vector traces, terminal fields, or demo-like layers.",
        ),
        motion=(
            "Use discrete stepping, looping trajectories, or low-rate modulation.",
        ),
        texture=(
            "Use dithering, scanlines, pixel structure, or bounded signal noise.",
        ),
        spatial=("Expose the coordinate grid or screen-space construction.",),
        runtimes=("p5.js", "GLSL", "Hydra", "Three.js", "React Three Fiber"),
    ),
    VisualStyleId.ETHEREAL: _StyleProfile(
        palette=("Use low-saturation gradients with restrained luminous highlights.",),
        contrast=(
            "Keep broad transitions soft while preserving one readable focal edge.",
        ),
        composition=(
            "Use suspended forms, open space, and overlapping translucent layers.",
        ),
        motion=("Use slow drift, dissolve, breathing, and gentle parallax.",),
        texture=("Use soft noise, mist, fine particles, or translucent interference.",),
        spatial=("Create depth through scale, haze, and widely spaced layers.",),
        runtimes=("Three.js", "React Three Fiber", "GLSL", "Hydra", "p5.js"),
    ),
    VisualStyleId.PSYCHEDELIC: _StyleProfile(
        palette=("Use controlled high-chroma relationships with deliberate cycling.",),
        contrast=(
            "Balance intense color transitions with stable dark or neutral anchors.",
        ),
        composition=("Use repetition, folding, feedback, and transformed symmetry.",),
        motion=(
            "Use phase shifts, morphing, feedback, and rhythmic color modulation.",
        ),
        texture=(
            "Use interference, warped fields, liquid bands, or recursive patterning.",
        ),
        spatial=(
            "Build nested, mirrored, or tunnel-like depth with a stable "
            "orientation cue.",
        ),
        runtimes=("Hydra", "GLSL", "p5.js", "Three.js", "React Three Fiber"),
    ),
    VisualStyleId.ARCHITECTURAL: _StyleProfile(
        palette=(
            "Use material-led neutrals with one controlled light or accent color.",
        ),
        contrast=("Use light and shadow to clarify planes, openings, and depth.",),
        composition=(
            "Use axes, modules, bays, frames, and strong perspective relationships.",
        ),
        motion=("Use camera movement, light travel, or measured structural assembly.",),
        texture=("Use bounded concrete, glass, metal, stone, or line-work cues.",),
        spatial=(
            "Prioritize scale, depth, circulation, and legible structural hierarchy.",
        ),
        runtimes=("Three.js", "React Three Fiber", "p5.js", "GLSL"),
    ),
    VisualStyleId.MONOCHROME: _StyleProfile(
        palette=("Use one hue family or grayscale with explicit value steps.",),
        contrast=("Build hierarchy through luminance, opacity, and density.",),
        composition=("Let silhouette, spacing, and rhythm carry the visual identity.",),
        motion=("Use value shifts and shape movement instead of hue cycling.",),
        texture=("Use grain, line density, or material variation sparingly.",),
        spatial=("Separate layers through value and scale rather than color.",),
        runtimes=("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra"),
    ),
    VisualStyleId.MAXIMALIST: _StyleProfile(
        palette=(
            "Use a broad palette with repeated anchor colors to prevent randomness.",
        ),
        contrast=(
            "Alternate dense high-contrast regions with intentional visual rests.",
        ),
        composition=("Layer multiple systems while preserving a dominant hierarchy.",),
        motion=(
            "Coordinate several motion bands with shared timing or phase "
            "relationships.",
        ),
        texture=("Combine patterns and materials, but cap each texture family.",),
        spatial=(
            "Use foreground, middle, and background layers with explicit "
            "focal priority.",
        ),
        runtimes=("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra"),
    ),
}

_RUNTIME_ALIASES = {
    "p5": "p5.js",
    "p5.js": "p5.js",
    "p5js": "p5.js",
    "three": "Three.js",
    "three.js": "Three.js",
    "threejs": "Three.js",
    "react three fiber": "React Three Fiber",
    "r3f": "React Three Fiber",
    "glsl": "GLSL",
    "shader": "GLSL",
    "hydra": "Hydra",
}
_RUNTIME_DIRECTIONS = {
    "p5.js": "Express the style through bounded 2D marks, geometry, and canvas layers.",
    "Three.js": (
        "Express the style through scene composition, materials, lighting, and camera."
    ),
    "React Three Fiber": (
        "Express the style through declarative scene composition, materials, "
        "lighting, and camera."
    ),
    "GLSL": (
        "Express the style through coordinates, fields, color functions, "
        "and bounded iteration."
    ),
    "Hydra": (
        "Express the style through source transforms, compositing, feedback, "
        "and color operations."
    ),
}
_MOOD_STYLE_MAP = {
    "minimal": VisualStyleId.MINIMAL,
    "organic": VisualStyleId.ORGANIC,
    "ritual": VisualStyleId.RITUAL,
    "ethereal": VisualStyleId.ETHEREAL,
}
_COLOR_STYLE_MAP = {
    "monochrome": VisualStyleId.MONOCHROME,
    "black and white": VisualStyleId.MONOCHROME,
}
_SACRED_ARCHITECTURAL_CONCEPTS = frozenset({"temple geometry", "cathedral geometry"})


def derive_visual_style_guidance(
    query: str,
    *,
    output_modality: str | None = None,
    mood_atmosphere: Sequence[str] = (),
    color_material_direction: Sequence[str] = (),
    runtime_recommendations: Sequence[str] = (),
    selected_runtime: str | None = None,
    sacred_geometry: SacredGeometryGuidance | None = None,
    shader_presets: ShaderPresetGuidance | None = None,
    base_guidance: VisualStyleGuidance | None = None,
) -> VisualStyleGuidance | None:
    """Derive bounded visual identity guidance from request-visible signals."""

    if output_modality == "audio":
        return base_guidance

    styles = detect_visual_styles(
        query,
        mood_atmosphere=mood_atmosphere,
        color_material_direction=color_material_direction,
        sacred_geometry=sacred_geometry,
        shader_presets=shader_presets,
    )
    if not styles:
        return base_guidance

    profiles = tuple(_PROFILES[style] for style in styles)
    current = VisualStyleGuidance(
        styles=styles,
        palette_behavior=_collect(profiles, "palette"),
        contrast_behavior=_collect(profiles, "contrast"),
        composition_tendencies=_collect(profiles, "composition"),
        motion_tendencies=_collect(profiles, "motion"),
        texture_tendencies=_collect(profiles, "texture"),
        spatial_organization=_collect(profiles, "spatial"),
        runtime_suitability=_runtime_suitability(
            profiles,
            runtime_recommendations=runtime_recommendations,
            selected_runtime=selected_runtime,
        ),
    )
    if base_guidance is None:
        return current
    return _merge_guidance(base_guidance, current)


def detect_visual_styles(
    query: str,
    *,
    mood_atmosphere: Sequence[str] = (),
    color_material_direction: Sequence[str] = (),
    sacred_geometry: SacredGeometryGuidance | None = None,
    shader_presets: ShaderPresetGuidance | None = None,
) -> tuple[VisualStyleId, ...]:
    """Detect only bounded styles supported by explicit or structured cues."""

    normalized = " ".join(query.strip().lower().split())
    styles = [style for style, pattern in _STYLE_PATTERNS if pattern.search(normalized)]

    for mood in mood_atmosphere:
        style = _MOOD_STYLE_MAP.get(mood.lower())
        if style is not None:
            styles.append(style)
    for color in color_material_direction:
        style = _COLOR_STYLE_MAP.get(color.lower())
        if style is not None:
            styles.append(style)

    if sacred_geometry is not None:
        styles.append(VisualStyleId.SACRED_GEOMETRY)
        if set(sacred_geometry.concepts) & _SACRED_ARCHITECTURAL_CONCEPTS:
            styles.append(VisualStyleId.ARCHITECTURAL)

    if shader_presets is not None:
        presets = set(shader_presets.presets)
        if presets & {
            ShaderPresetId.PLASMA,
            ShaderPresetId.KALEIDOSCOPIC_SYMMETRY,
        }:
            styles.append(VisualStyleId.PSYCHEDELIC)
        if ShaderPresetId.SACRED_LIGHT in presets:
            styles.append(VisualStyleId.RITUAL)

    return _dedupe_styles(styles)


def visual_style_prompt_lines(
    guidance: VisualStyleGuidance,
) -> tuple[str, ...]:
    """Render compact provider-independent style implementation guidance."""

    lines = [
        "Visual style identities: "
        + ", ".join(style.value for style in guidance.styles)
    ]
    _append_line(lines, "Style palette behavior", guidance.palette_behavior)
    _append_line(lines, "Style contrast behavior", guidance.contrast_behavior)
    _append_line(
        lines,
        "Style composition tendencies",
        guidance.composition_tendencies,
    )
    _append_line(lines, "Style motion tendencies", guidance.motion_tendencies)
    _append_line(lines, "Style texture tendencies", guidance.texture_tendencies)
    _append_line(lines, "Style spatial organization", guidance.spatial_organization)
    _append_line(lines, "Style runtime guidance", guidance.runtime_suitability)
    return tuple(lines)


def _runtime_suitability(
    profiles: Sequence[_StyleProfile],
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
        runtime = compatible[0]
        return (
            f"Use the selected compatible runtime: {runtime}.",
            _RUNTIME_DIRECTIONS[runtime],
        )
    if requested:
        runtime = requested[0]
        return (
            f"Use a bounded stylized approximation in {runtime}; the full style "
            "combination is not natively suited to that runtime.",
            _RUNTIME_DIRECTIONS[runtime],
        )
    ordered_supported = tuple(
        runtime
        for runtime in ("p5.js", "Three.js", "React Three Fiber", "GLSL", "Hydra")
        if runtime in supported
    )
    return ("Suitable visual runtimes: " + ", ".join(ordered_supported) + ".",)


def _normalize_runtimes(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        runtime = _RUNTIME_ALIASES.get(value.strip().lower())
        if runtime is not None and runtime not in normalized:
            normalized.append(runtime)
    return tuple(normalized)


def _merge_guidance(
    base: VisualStyleGuidance,
    current: VisualStyleGuidance,
) -> VisualStyleGuidance:
    return VisualStyleGuidance(
        styles=_dedupe_styles((*base.styles, *current.styles)),
        palette_behavior=_merge(base.palette_behavior, current.palette_behavior),
        contrast_behavior=_merge(base.contrast_behavior, current.contrast_behavior),
        composition_tendencies=_merge(
            base.composition_tendencies,
            current.composition_tendencies,
        ),
        motion_tendencies=_merge(
            base.motion_tendencies,
            current.motion_tendencies,
        ),
        texture_tendencies=_merge(
            base.texture_tendencies,
            current.texture_tendencies,
        ),
        spatial_organization=_merge(
            base.spatial_organization,
            current.spatial_organization,
        ),
        runtime_suitability=_merge(
            current.runtime_suitability,
            base.runtime_suitability,
        ),
    )


def _collect(
    profiles: Sequence[_StyleProfile],
    field: str,
) -> tuple[str, ...]:
    return _dedupe(item for profile in profiles for item in getattr(profile, field))


def _merge(*groups: Sequence[str]) -> tuple[str, ...]:
    return _dedupe(item for group in groups for item in group)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _dedupe_styles(
    values: Iterable[VisualStyleId],
) -> tuple[VisualStyleId, ...]:
    return tuple(dict.fromkeys(values))[:6]


def _append_line(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:
    if values:
        lines.append(f"{label}: {' '.join(values)}")
