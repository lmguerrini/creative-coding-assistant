"""Bounded sacred-geometry detection and practical generation guidance."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class SacredGeometryGuidance(BaseModel):
    """Practical creative guidance derived from explicit geometric concepts."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    concepts: tuple[str, ...] = Field(min_length=1)
    geometric_structure: tuple[str, ...] = Field(default_factory=tuple)
    symmetry_type: tuple[str, ...] = Field(default_factory=tuple)
    movement_behavior: tuple[str, ...] = Field(default_factory=tuple)
    visual_composition: tuple[str, ...] = Field(default_factory=tuple)
    color_material_direction: tuple[str, ...] = Field(default_factory=tuple)
    runtime_recommendations: tuple[str, ...] = Field(default_factory=tuple)
    audio_implications: tuple[str, ...] = Field(default_factory=tuple)
    generation_constraints: tuple[str, ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class _GuidanceProfile:
    structure: tuple[str, ...]
    symmetry: tuple[str, ...]
    movement: tuple[str, ...]
    composition: tuple[str, ...]
    color_material: tuple[str, ...]
    runtimes: tuple[str, ...]
    audio: tuple[str, ...] = ()


_CONCEPT_PATTERNS = (
    ("Sri Yantra", re.compile(r"(?<!\w)sri[\s-]+yantra(?!\w)")),
    ("Flower of Life", re.compile(r"(?<!\w)flower[\s-]+of[\s-]+life(?!\w)")),
    (
        "Metatron's Cube",
        re.compile(r"(?<!\w)metatron(?:['’]s)?[\s-]+cube(?!\w)"),
    ),
    ("vesica piscis", re.compile(r"(?<!\w)vesica[\s-]+piscis(?!\w)")),
    ("golden ratio", re.compile(r"(?<!\w)golden[\s-]+ratio(?!\w)")),
    ("fractal symmetry", re.compile(r"(?<!\w)fractal[\s-]+symmetry(?!\w)")),
    ("radial symmetry", re.compile(r"(?<!\w)radial[\s-]+symmetry(?!\w)")),
    ("cathedral geometry", re.compile(r"(?<!\w)cathedral[\s-]+geometry(?!\w)")),
    ("temple geometry", re.compile(r"(?<!\w)temple[\s-]+geometry(?!\w)")),
    ("mandala", re.compile(r"(?<!\w)mandala(?!\w)")),
    ("yantra", re.compile(r"(?<!\w)yantra(?!\w)")),
    ("Merkaba", re.compile(r"(?<!\w)merkaba(?!\w)")),
    ("torus", re.compile(r"(?<!\w)torus(?!\w)")),
    ("spiral", re.compile(r"(?<!\w)spiral(?!\w)")),
    ("Fibonacci", re.compile(r"(?<!\w)fibonacci(?!\w)")),
)

_PROFILES = {
    "mandala": _GuidanceProfile(
        structure=("Build nested rings or cells around a clear center.",),
        symmetry=("Use radial symmetry with a deliberately limited segment count.",),
        movement=("Animate layers with slow phase offsets or counter-rotation.",),
        composition=("Keep a strong center and readable concentric hierarchy.",),
        color_material=(
            "Use a restrained layered palette with clear tonal separation.",
        ),
        runtimes=("p5.js", "GLSL"),
        audio=("Map concentric layers to distinct rhythmic or frequency bands.",),
    ),
    "yantra": _GuidanceProfile(
        structure=(
            "Construct a precise layered diagram from triangles, circles, and frames.",
        ),
        symmetry=("Keep bilateral and radial alignments exact.",),
        movement=(
            "Reveal or pulse structural layers without distorting their alignment.",
        ),
        composition=("Use centered framing and clear negative space between layers.",),
        color_material=("Favor crisp line work, flat fills, and controlled contrast.",),
        runtimes=("p5.js", "GLSL"),
    ),
    "Sri Yantra": _GuidanceProfile(
        structure=(
            "Use interlocking upward and downward triangles as the primary scaffold.",
        ),
        symmetry=("Preserve a centered, balanced triangular construction.",),
        movement=("Animate triangle groups by restrained scale or opacity phases.",),
        composition=("Build outward from a precise central focal point.",),
        color_material=("Use fine luminous lines over a calm, low-noise field.",),
        runtimes=("p5.js", "GLSL"),
        audio=(
            "Use converging rhythmic layers rather than assigning symbolic meanings.",
        ),
    ),
    "Flower of Life": _GuidanceProfile(
        structure=("Construct an overlapping equal-radius circle lattice.",),
        symmetry=("Use hexagonal and radial repetition with consistent spacing.",),
        movement=("Grow the lattice ring by ring or modulate circle intersections.",),
        composition=(
            "Let the repeated circle field remain legible at the frame edges.",
        ),
        color_material=("Use translucent strokes or fills to expose intersections.",),
        runtimes=("p5.js", "GLSL"),
    ),
    "Metatron's Cube": _GuidanceProfile(
        structure=(
            "Connect a bounded set of circle centers into a line-and-node network.",
        ),
        symmetry=("Use radial balance and consistent node spacing.",),
        movement=(
            "Rotate or depth-shift selected line groups while preserving connectivity.",
        ),
        composition=("Separate primary nodes from secondary connection lines.",),
        color_material=("Use fine emissive lines with restrained depth cues.",),
        runtimes=("Three.js", "React Three Fiber", "p5.js"),
    ),
    "Merkaba": _GuidanceProfile(
        structure=("Build two intersecting tetrahedral forms as a geometric study.",),
        symmetry=("Use balanced rotational symmetry around the shared center.",),
        movement=("Apply slow counter-rotation to the paired forms.",),
        composition=("Keep the silhouette and intersections readable in depth.",),
        color_material=(
            "Use translucent or wireframe materials with controlled highlights.",
        ),
        runtimes=("Three.js", "React Three Fiber"),
    ),
    "torus": _GuidanceProfile(
        structure=(
            "Use a ring surface, toroidal particle field, or wrapped parametric grid.",
        ),
        symmetry=("Preserve rotational symmetry around the torus axis.",),
        movement=("Flow particles or waves around both toroidal directions.",),
        composition=("Choose a camera angle that keeps the central opening visible.",),
        color_material=(
            "Use directional gradients or reflective bands to reveal curvature.",
        ),
        runtimes=("Three.js", "React Three Fiber", "GLSL"),
        audio=("Map phase or frequency bands around the toroidal path.",),
    ),
    "spiral": _GuidanceProfile(
        structure=("Build from a continuous polar or parametric spiral path.",),
        symmetry=("Use rotational progression rather than forced mirror symmetry.",),
        movement=("Move points along the curve or modulate its radius over time.",),
        composition=(
            "Protect the spiral origin and maintain readable spacing between turns.",
        ),
        color_material=("Use a directional value or hue progression along the path.",),
        runtimes=("p5.js", "GLSL", "Three.js"),
    ),
    "golden ratio": _GuidanceProfile(
        structure=("Use golden-ratio proportions as a compositional spacing rule.",),
        symmetry=("Prefer proportional balance over perfect bilateral symmetry.",),
        movement=("Vary growth or spacing by a bounded proportional progression.",),
        composition=("Place focal regions using a clear proportional hierarchy.",),
        color_material=(
            "Keep material changes subordinate to the proportional structure.",
        ),
        runtimes=("p5.js", "GLSL"),
    ),
    "Fibonacci": _GuidanceProfile(
        structure=("Use a bounded Fibonacci sequence for counts, radii, or spacing.",),
        symmetry=("Use progressive repetition with controlled density.",),
        movement=("Grow or reveal elements in discrete sequence-based stages.",),
        composition=("Cap the sequence so the result remains legible and performant.",),
        color_material=("Use gradual palette changes that track the sequence.",),
        runtimes=("p5.js", "GLSL"),
        audio=("Map bounded sequence values to rhythm, intervals, or event density.",),
    ),
    "vesica piscis": _GuidanceProfile(
        structure=(
            "Use two equal overlapping circles and emphasize their shared lens.",
        ),
        symmetry=("Maintain bilateral symmetry across the circle centers.",),
        movement=("Animate overlap distance while keeping radii equal.",),
        composition=("Keep the shared lens as the primary focal region.",),
        color_material=(
            "Use transparent layers or contrasting outlines at the overlap.",
        ),
        runtimes=("p5.js", "GLSL"),
    ),
    "fractal symmetry": _GuidanceProfile(
        structure=("Repeat a base form across a bounded number of scales.",),
        symmetry=("Use self-similar repetition without unbounded recursion.",),
        movement=("Animate scale transitions or recursive depth gradually.",),
        composition=(
            "Reserve quiet regions so repeated detail does not become uniform noise.",
        ),
        color_material=("Reduce palette complexity as geometric detail increases.",),
        runtimes=("GLSL", "p5.js", "Three.js"),
        audio=("Use nested rhythmic subdivisions with a strict depth limit.",),
    ),
    "radial symmetry": _GuidanceProfile(
        structure=("Repeat a clear motif around a shared center.",),
        symmetry=("Use an explicit radial segment count and stable angular spacing.",),
        movement=("Rotate, pulse, or offset alternating radial segments.",),
        composition=("Balance center detail against the outer radial boundary.",),
        color_material=(
            "Use controlled segment variation without breaking the radial read.",
        ),
        runtimes=("p5.js", "GLSL"),
    ),
    "temple geometry": _GuidanceProfile(
        structure=(
            "Use axes, bays, grids, or proportional frames as an "
            "architectural scaffold.",
        ),
        symmetry=("Use strong axial or bilateral organization.",),
        movement=(
            "Reveal structural layers through measured traversal or light changes.",
        ),
        composition=(
            "Emphasize thresholds, depth, and repeated structural intervals.",
        ),
        color_material=(
            "Use stone, shadow, and warm light as optional material cues.",
        ),
        runtimes=("Three.js", "React Three Fiber", "p5.js"),
    ),
    "cathedral geometry": _GuidanceProfile(
        structure=(
            "Use arches, ribs, rose-window grids, or vertical bays as form studies.",
        ),
        symmetry=("Use axial balance with repeated vertical and radial divisions.",),
        movement=("Animate light, camera drift, or gradual structural reveals.",),
        composition=("Emphasize vertical scale, depth, and a legible central axis.",),
        color_material=(
            "Use glass-like color fields, stone tones, and bounded bloom.",
        ),
        runtimes=("Three.js", "React Three Fiber", "GLSL"),
    ),
}

_SAFETY_CONSTRAINTS = (
    "Treat sacred-geometry terms as practical design motifs, not "
    "authoritative spiritual claims.",
    "Do not add religious, historical, or symbolic claims not present in the request.",
)


def derive_sacred_geometry_guidance(
    query: str,
    *,
    output_modality: str | None = None,
    base_guidance: SacredGeometryGuidance | None = None,
) -> SacredGeometryGuidance | None:
    """Return deterministic guidance only when supported concepts are explicit."""

    concepts = detect_sacred_geometry_concepts(query)
    if not concepts:
        return base_guidance

    profiles = tuple(_PROFILES[concept] for concept in concepts)
    current = SacredGeometryGuidance(
        concepts=concepts,
        geometric_structure=_collect(profiles, "structure"),
        symmetry_type=_collect(profiles, "symmetry"),
        movement_behavior=_collect(profiles, "movement"),
        visual_composition=_collect(profiles, "composition"),
        color_material_direction=_collect(profiles, "color_material"),
        runtime_recommendations=_runtime_guidance(
            profiles,
            output_modality=output_modality,
        ),
        audio_implications=_audio_guidance(
            profiles,
            output_modality=output_modality,
        ),
        generation_constraints=_SAFETY_CONSTRAINTS,
    )
    if base_guidance is None:
        return current
    return _merge_guidance(base_guidance, current)


def detect_sacred_geometry_concepts(query: str) -> tuple[str, ...]:
    """Detect only the bounded vocabulary explicitly present in the request."""

    normalized = " ".join(query.strip().lower().split())
    matches = [
        label for label, pattern in _CONCEPT_PATTERNS if pattern.search(normalized)
    ]
    if "Sri Yantra" in matches:
        matches = [label for label in matches if label != "yantra"]
    return tuple(matches)


def sacred_geometry_prompt_lines(
    guidance: SacredGeometryGuidance,
) -> tuple[str, ...]:
    """Render compact, provider-independent sacred-geometry instructions."""

    lines = [f"Sacred geometry concepts: {', '.join(guidance.concepts)}"]
    _append_line(lines, "Geometric structure", guidance.geometric_structure)
    _append_line(lines, "Symmetry", guidance.symmetry_type)
    _append_line(lines, "Movement behavior", guidance.movement_behavior)
    _append_line(lines, "Visual composition", guidance.visual_composition)
    _append_line(
        lines,
        "Color and material guidance",
        guidance.color_material_direction,
    )
    _append_line(lines, "Sacred geometry runtimes", guidance.runtime_recommendations)
    _append_line(lines, "Audio implications", guidance.audio_implications)
    _append_line(
        lines,
        "Sacred geometry constraints",
        guidance.generation_constraints,
    )
    return tuple(lines)


def _collect(
    profiles: Sequence[_GuidanceProfile],
    field: str,
) -> tuple[str, ...]:
    return _dedupe(item for profile in profiles for item in getattr(profile, field))


def _runtime_guidance(
    profiles: Sequence[_GuidanceProfile],
    *,
    output_modality: str | None,
) -> tuple[str, ...]:
    if output_modality == "audio":
        return ("Tone.js",)

    runtimes = list(_collect(profiles, "runtimes"))
    if output_modality == "audiovisual":
        runtimes.append("Tone.js")
    return _dedupe(runtimes)


def _audio_guidance(
    profiles: Sequence[_GuidanceProfile],
    *,
    output_modality: str | None,
) -> tuple[str, ...]:
    if output_modality not in {"audio", "audiovisual"}:
        return ()
    implications = list(_collect(profiles, "audio"))
    if not implications:
        implications.append(
            "Map visible geometric layers to bounded musical layers or modulation."
        )
    implications.append("Keep audio playback behind an explicit user interaction.")
    return _dedupe(implications)


def _merge_guidance(
    base: SacredGeometryGuidance,
    current: SacredGeometryGuidance,
) -> SacredGeometryGuidance:
    return SacredGeometryGuidance(
        concepts=_merge(base.concepts, current.concepts),
        geometric_structure=_merge(
            base.geometric_structure,
            current.geometric_structure,
        ),
        symmetry_type=_merge(base.symmetry_type, current.symmetry_type),
        movement_behavior=_merge(
            base.movement_behavior,
            current.movement_behavior,
        ),
        visual_composition=_merge(
            base.visual_composition,
            current.visual_composition,
        ),
        color_material_direction=_merge(
            base.color_material_direction,
            current.color_material_direction,
        ),
        runtime_recommendations=_merge(
            current.runtime_recommendations,
            base.runtime_recommendations,
        ),
        audio_implications=_merge(
            base.audio_implications,
            current.audio_implications,
        ),
        generation_constraints=_merge(
            base.generation_constraints,
            current.generation_constraints,
        ),
    )


def _merge(*groups: Sequence[str]) -> tuple[str, ...]:
    return _dedupe(item for group in groups for item in group)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _append_line(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:
    if values:
        lines.append(f"{label}: {' '.join(values)}")
