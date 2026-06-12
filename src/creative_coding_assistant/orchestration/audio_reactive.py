"""Bounded audio-to-visual mapping guidance for audiovisual generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.orchestration.sacred_geometry import (
    SacredGeometryGuidance,
)
from creative_coding_assistant.orchestration.shader_presets import (
    ShaderPresetGuidance,
)
from creative_coding_assistant.orchestration.visual_styles import (
    VisualStyleGuidance,
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class AudioReactiveSource(StrEnum):
    AMPLITUDE = "amplitude"
    BASS = "bass"
    MIDS = "mids"
    HIGHS = "highs"
    RHYTHM = "rhythm"
    ENVELOPE = "envelope"
    DRONE_INTENSITY = "drone_intensity"


class AudioReactiveVisualTarget(StrEnum):
    SCALE = "scale"
    GLOW = "glow"
    BRIGHTNESS = "brightness"
    PULSE = "pulse"
    EXPANSION = "expansion"
    CAMERA_MOVEMENT = "camera_movement"
    COLOR_SHIFT = "color_shift"
    TEXTURE_MODULATION = "texture_modulation"
    SPARKLE = "sparkle"
    PARTICLES = "particles"
    DETAIL = "detail"
    ROTATION = "rotation"
    PATTERN_PHASE = "pattern_phase"
    SCENE_TRANSITIONS = "scene_transitions"
    OPACITY = "opacity"
    BLOOM = "bloom"
    GEOMETRY_EMERGENCE = "geometry_emergence"
    FOG = "fog"
    AURA = "aura"
    FIELD_DENSITY = "field_density"


class AudioReactiveIntensity(StrEnum):
    SUBTLE = "subtle"
    BALANCED = "balanced"
    STRONG = "strong"


class AudioReactiveMapping(BaseModel):
    """One deterministic relationship between an audio feature and visual targets."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    source: AudioReactiveSource
    targets: tuple[AudioReactiveVisualTarget, ...] = Field(
        min_length=1,
        max_length=3,
    )
    intensity: AudioReactiveIntensity = AudioReactiveIntensity.BALANCED
    behavior: str = Field(min_length=1, max_length=240)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=5)


class AudioReactiveGuidance(BaseModel):
    """Optional audiovisual mapping metadata carried through prompts and artifacts."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    mappings: tuple[AudioReactiveMapping, ...] = Field(
        min_length=1,
        max_length=6,
    )
    audio_runtime: str | None = None
    visual_runtime: str | None = None
    activation: Literal["explicit_user_gesture"] = "explicit_user_gesture"
    summary: str = Field(min_length=1, max_length=320)


_AUDIO_REACTIVE_SIGNAL = re.compile(
    r"\b(?:audio[\s-]?reactive|audiovisual|music visuali[sz]er|"
    r"sound[\s-]?reactive|reacts? to (?:audio|music|sound))\b"
)
_AUDIO_REACTIVE_DISABLED = re.compile(
    r"(?:audio reactivity:\s*disabled|disable audio[\s-]?reactiv(?:e|ity)|"
    r"without audio[\s-]?reactiv(?:e|ity))"
)
_SOURCE_PATTERNS: tuple[tuple[AudioReactiveSource, re.Pattern[str]], ...] = (
    (
        AudioReactiveSource.AMPLITUDE,
        re.compile(r"\b(?:amplitude|loudness|volume|audio level|meter)\b"),
    ),
    (
        AudioReactiveSource.BASS,
        re.compile(r"\b(?:bass|low[\s-]?frequency|low end|kick)\b"),
    ),
    (
        AudioReactiveSource.MIDS,
        re.compile(r"\b(?:mids?|midrange|middle frequenc(?:y|ies))\b"),
    ),
    (
        AudioReactiveSource.HIGHS,
        re.compile(r"\b(?:highs?|treble|high[\s-]?frequency|sparkle)\b"),
    ),
    (
        AudioReactiveSource.RHYTHM,
        re.compile(
            r"\b(?:rhythm|beat|bpm|tempo|transport|pattern phase|"
            r"rotation speed|rhythm density)\b"
        ),
    ),
    (
        AudioReactiveSource.ENVELOPE,
        re.compile(
            r"\b(?:envelope|attack|decay|release|opacity|bloom intensity|"
            r"geometry emergence)\b"
        ),
    ),
    (
        AudioReactiveSource.DRONE_INTENSITY,
        re.compile(
            r"\b(?:drone|sustain(?:ed)?|drone intensity|fog density|aura|"
            r"field density)\b"
        ),
    ),
)
_TONE_HINT_PATTERNS: tuple[tuple[AudioReactiveSource, re.Pattern[str]], ...] = (
    (
        AudioReactiveSource.AMPLITUDE,
        re.compile(r"\bTone\.(?:Meter|AmplitudeEnvelope)\b", re.I),
    ),
    (
        AudioReactiveSource.RHYTHM,
        re.compile(r"\bTone\.(?:Transport|Loop|Pattern|Sequence)\b", re.I),
    ),
    (
        AudioReactiveSource.ENVELOPE,
        re.compile(
            r"\bTone\.(?:AmplitudeEnvelope|Envelope|FrequencyEnvelope)\b",
            re.I,
        ),
    ),
    (
        AudioReactiveSource.BASS,
        re.compile(r"\bTone\.(?:MembraneSynth|FFT)\b", re.I),
    ),
    (
        AudioReactiveSource.HIGHS,
        re.compile(r"\bTone\.(?:MetalSynth|FFT)\b", re.I),
    ),
)
_AUDIO_RUNTIME_ALIASES = {
    "tone": "Tone.js",
    "tone.js": "Tone.js",
    "tone_js": "Tone.js",
    "web audio": "Web Audio API",
    "web audio api": "Web Audio API",
    "web_audio_api": "Web Audio API",
    "p5.sound": "p5.sound",
    "p5_sound": "p5.sound",
}
_VISUAL_RUNTIME_ALIASES = {
    "p5": "p5.js",
    "p5.js": "p5.js",
    "p5_js": "p5.js",
    "three": "Three.js",
    "three.js": "Three.js",
    "three_js": "Three.js",
    "react three fiber": "React Three Fiber",
    "react_three_fiber": "React Three Fiber",
    "r3f": "React Three Fiber",
    "glsl": "GLSL",
    "shader": "GLSL",
    "hydra": "Hydra",
}


def derive_audio_reactive_guidance(
    query: str,
    *,
    output_modality: object,
    musical_references: Sequence[str] = (),
    movement_language: Sequence[str] = (),
    runtime_recommendations: Sequence[str] = (),
    selected_runtime: str | None = None,
    sacred_geometry: SacredGeometryGuidance | None = None,
    shader_presets: ShaderPresetGuidance | None = None,
    visual_style: VisualStyleGuidance | None = None,
    tone_metadata: str | None = None,
    dynamic_parameter_guidance: str | None = None,
    base_guidance: AudioReactiveGuidance | None = None,
) -> AudioReactiveGuidance | None:
    """Derive bounded mappings only for requests with audiovisual intent."""

    modality = getattr(output_modality, "value", output_modality)
    dynamic_text = _normalize_text(dynamic_parameter_guidance or "")
    if modality != "audiovisual" or _AUDIO_REACTIVE_DISABLED.search(dynamic_text):
        return None

    evidence_text = _normalize_text(
        " ".join(
            (
                query,
                *musical_references,
                *movement_language,
                *(sacred_geometry.audio_implications if sacred_geometry else ()),
                *(
                    tuple(preset.value for preset in shader_presets.presets)
                    if shader_presets
                    else ()
                ),
                *(
                    tuple(style.value for style in visual_style.styles)
                    if visual_style
                    else ()
                ),
                dynamic_parameter_guidance or "",
            )
        )
    )
    sources = _detect_sources(
        evidence_text,
        tone_metadata=tone_metadata,
    )
    if not sources and base_guidance is not None:
        return base_guidance
    if not sources:
        sources = (
            AudioReactiveSource.AMPLITUDE,
            AudioReactiveSource.RHYTHM,
        )

    audio_runtime, visual_runtime = _resolve_runtimes(
        runtime_recommendations,
        selected_runtime=selected_runtime,
        base_guidance=base_guidance,
    )
    evidence = _mapping_evidence(
        query=query,
        tone_metadata=tone_metadata,
        sacred_geometry=sacred_geometry,
        shader_presets=shader_presets,
        visual_style=visual_style,
        dynamic_parameter_guidance=dynamic_parameter_guidance,
    )
    intensity = _derive_intensity(evidence_text)
    mappings = tuple(
        AudioReactiveMapping(
            source=source,
            targets=_targets_for_source(
                source,
                visual_runtime=visual_runtime,
                evidence_text=evidence_text,
                shader_presets=shader_presets,
            ),
            intensity=intensity,
            behavior=_behavior_for_source(source),
            evidence=evidence,
        )
        for source in sources[:6]
    )

    return AudioReactiveGuidance(
        mappings=mappings,
        audio_runtime=audio_runtime,
        visual_runtime=visual_runtime,
        summary=_summary_for_mappings(mappings),
    )


def audio_reactive_prompt_lines(
    guidance: AudioReactiveGuidance,
) -> tuple[str, ...]:
    """Render compact mapping guidance without implying runtime execution."""

    lines = ["Audio-reactive mapping plan: " + guidance.summary]
    for mapping in guidance.mappings:
        targets = ", ".join(
            target.value.replace("_", " ") for target in mapping.targets
        )
        lines.append(
            f"Map {mapping.source.value.replace('_', ' ')} to {targets} "
            f"with a {mapping.intensity.value} response. {mapping.behavior}"
        )
    lines.append(
        "Keep audio silent until explicit user activation; mapping analysis must "
        "not start playback or mutate source invisibly."
    )
    return tuple(lines)


def audio_reactivity_is_explicitly_disabled(value: str) -> bool:
    return bool(_AUDIO_REACTIVE_DISABLED.search(_normalize_text(value)))


def _detect_sources(
    evidence_text: str,
    *,
    tone_metadata: str | None,
) -> tuple[AudioReactiveSource, ...]:
    sources = [
        source
        for source, pattern in _SOURCE_PATTERNS
        if pattern.search(evidence_text)
    ]
    for source, pattern in _TONE_HINT_PATTERNS:
        if tone_metadata and pattern.search(tone_metadata) and source not in sources:
            sources.append(source)
    if _AUDIO_REACTIVE_SIGNAL.search(evidence_text) and not sources:
        sources.extend(
            (
                AudioReactiveSource.AMPLITUDE,
                AudioReactiveSource.RHYTHM,
            )
        )
    return tuple(sources)


def _targets_for_source(
    source: AudioReactiveSource,
    *,
    visual_runtime: str | None,
    evidence_text: str,
    shader_presets: ShaderPresetGuidance | None,
) -> tuple[AudioReactiveVisualTarget, ...]:
    targets: list[AudioReactiveVisualTarget]
    if source is AudioReactiveSource.AMPLITUDE:
        targets = [
            AudioReactiveVisualTarget.SCALE,
            AudioReactiveVisualTarget.BRIGHTNESS,
        ]
        if _has_luminous_preset(shader_presets):
            targets.append(AudioReactiveVisualTarget.GLOW)
    elif source is AudioReactiveSource.BASS:
        targets = [
            AudioReactiveVisualTarget.PULSE,
            AudioReactiveVisualTarget.EXPANSION,
        ]
        if visual_runtime in {"Three.js", "React Three Fiber"}:
            targets.append(AudioReactiveVisualTarget.CAMERA_MOVEMENT)
    elif source is AudioReactiveSource.MIDS:
        targets = [
            AudioReactiveVisualTarget.COLOR_SHIFT,
            AudioReactiveVisualTarget.TEXTURE_MODULATION,
        ]
    elif source is AudioReactiveSource.HIGHS:
        targets = [
            AudioReactiveVisualTarget.SPARKLE,
            AudioReactiveVisualTarget.PARTICLES,
            AudioReactiveVisualTarget.DETAIL,
        ]
    elif source is AudioReactiveSource.RHYTHM:
        targets = [
            AudioReactiveVisualTarget.ROTATION,
            AudioReactiveVisualTarget.PATTERN_PHASE,
        ]
        if "scene transition" in evidence_text:
            targets.append(AudioReactiveVisualTarget.SCENE_TRANSITIONS)
    elif source is AudioReactiveSource.ENVELOPE:
        targets = [
            AudioReactiveVisualTarget.OPACITY,
            AudioReactiveVisualTarget.GEOMETRY_EMERGENCE,
        ]
        if _has_luminous_preset(shader_presets) or "bloom intensity" in evidence_text:
            targets.append(AudioReactiveVisualTarget.BLOOM)
    else:
        targets = [
            AudioReactiveVisualTarget.FOG,
            AudioReactiveVisualTarget.AURA,
            AudioReactiveVisualTarget.FIELD_DENSITY,
        ]

    if "accent color" in evidence_text and source is AudioReactiveSource.MIDS:
        targets.insert(0, AudioReactiveVisualTarget.COLOR_SHIFT)
    if "rotation speed" in evidence_text and source is AudioReactiveSource.RHYTHM:
        targets.insert(0, AudioReactiveVisualTarget.ROTATION)
    if "fog density" in evidence_text and source is AudioReactiveSource.DRONE_INTENSITY:
        targets.insert(0, AudioReactiveVisualTarget.FOG)
    return _dedupe_targets(targets)[:3]


def _behavior_for_source(source: AudioReactiveSource) -> str:
    return {
        AudioReactiveSource.AMPLITUDE: (
            "Smooth short peaks so scale and light remain readable."
        ),
        AudioReactiveSource.BASS: (
            "Use low-frequency energy for weighted expansion rather than jitter."
        ),
        AudioReactiveSource.MIDS: (
            "Use mid-band energy for gradual palette and surface variation."
        ),
        AudioReactiveSource.HIGHS: (
            "Use high-band transients for bounded detail accents."
        ),
        AudioReactiveSource.RHYTHM: (
            "Quantize structural changes to the requested pulse or BPM."
        ),
        AudioReactiveSource.ENVELOPE: (
            "Follow attack and release contours with eased visual emergence."
        ),
        AudioReactiveSource.DRONE_INTENSITY: (
            "Use sustained energy for slow atmospheric density changes."
        ),
    }[source]


def _resolve_runtimes(
    runtime_recommendations: Sequence[str],
    *,
    selected_runtime: str | None,
    base_guidance: AudioReactiveGuidance | None,
) -> tuple[str | None, str | None]:
    values = tuple(
        value
        for value in (selected_runtime, *runtime_recommendations)
        if value and value.strip()
    )
    audio_runtime = next(
        (
            _AUDIO_RUNTIME_ALIASES[normalized]
            for value in values
            if (normalized := value.strip().lower()) in _AUDIO_RUNTIME_ALIASES
        ),
        base_guidance.audio_runtime if base_guidance else None,
    )
    visual_runtime = next(
        (
            _VISUAL_RUNTIME_ALIASES[normalized]
            for value in values
            if (normalized := value.strip().lower()) in _VISUAL_RUNTIME_ALIASES
        ),
        base_guidance.visual_runtime if base_guidance else None,
    )
    return audio_runtime or "Tone.js", visual_runtime


def _mapping_evidence(
    *,
    query: str,
    tone_metadata: str | None,
    sacred_geometry: SacredGeometryGuidance | None,
    shader_presets: ShaderPresetGuidance | None,
    visual_style: VisualStyleGuidance | None,
    dynamic_parameter_guidance: str | None,
) -> tuple[str, ...]:
    evidence = ["user prompt"]
    if tone_metadata and any(
        pattern.search(tone_metadata) for _, pattern in _TONE_HINT_PATTERNS
    ):
        evidence.append("Tone.js metadata")
    if sacred_geometry and sacred_geometry.audio_implications:
        evidence.append("sacred geometry")
    if shader_presets is not None:
        evidence.append("shader presets")
    if visual_style is not None:
        evidence.append("visual style")
    if dynamic_parameter_guidance and ":" in dynamic_parameter_guidance:
        evidence.append("dynamic parameters")
    if not query.strip():
        evidence.remove("user prompt")
    return tuple(evidence)


def _derive_intensity(evidence_text: str) -> AudioReactiveIntensity:
    if re.search(r"\b(?:strong|intense|dramatic|energetic|maximal)\b", evidence_text):
        return AudioReactiveIntensity.STRONG
    if re.search(r"\b(?:subtle|calm|minimal|restrained|gentle)\b", evidence_text):
        return AudioReactiveIntensity.SUBTLE
    return AudioReactiveIntensity.BALANCED


def _has_luminous_preset(guidance: ShaderPresetGuidance | None) -> bool:
    if guidance is None:
        return False
    return any(
        preset.value in {"glow", "aura", "bloom-like emission", "plasma"}
        for preset in guidance.presets
    )


def _summary_for_mappings(
    mappings: tuple[AudioReactiveMapping, ...],
) -> str:
    relationships = [
        (
            mapping.source.value.replace("_", " ")
            + " -> "
            + " / ".join(target.value.replace("_", " ") for target in mapping.targets)
        )
        for mapping in mappings
    ]
    return "; ".join(relationships)


def _dedupe_targets(
    values: Sequence[AudioReactiveVisualTarget],
) -> tuple[AudioReactiveVisualTarget, ...]:
    return tuple(dict.fromkeys(values))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
