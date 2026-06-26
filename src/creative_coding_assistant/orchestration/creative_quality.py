"""Deterministic creative-quality observations for generated artifacts."""

from __future__ import annotations

import re
from collections.abc import Callable
from statistics import mean

from creative_coding_assistant.orchestration.artifacts import (
    CreativeQualityEvaluation,
    CreativeQualityObservation,
    WorkflowArtifact,
)
from creative_coding_assistant.orchestration._metadata_utils import _token_set
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)

_NUMBER_PATTERN = re.compile(r"(?<![a-z])[-+]?(?:\d+\.\d+|\d+)(?![a-z])")

_COMPOSITION_MARKERS = frozenset(
    {
        "angle",
        "camera",
        "center",
        "circle",
        "grid",
        "height",
        "orbit",
        "position",
        "radius",
        "scale",
        "translate",
        "width",
    }
)
_ORIGINALITY_MARKERS = frozenset(
    {
        "angle",
        "cos",
        "field",
        "flow",
        "fractal",
        "framecount",
        "modulate",
        "noise",
        "orbit",
        "particles",
        "radius",
        "random",
        "shader",
        "sin",
    }
)
_COHERENCE_MARKERS = frozenset(
    {
        "const",
        "draw",
        "export",
        "for",
        "function",
        "import",
        "let",
        "main",
        "return",
        "setup",
        "uniform",
        "void",
    }
)
_AESTHETIC_MARKERS = frozenset(
    {
        "background",
        "bloom",
        "color",
        "colormode",
        "fill",
        "glow",
        "gradient",
        "hsl",
        "light",
        "material",
        "opacity",
        "palette",
        "stroke",
    }
)
_EXPRESSIVE_MARKERS = frozenset(
    {
        "animate",
        "cos",
        "framecount",
        "lerp",
        "modulate",
        "noise",
        "orbit",
        "particles",
        "pulse",
        "rotate",
        "rotation",
        "sin",
    }
)

_DIMENSION_LABELS = {
    "composition": "Composition",
    "originality": "Originality",
    "coherence": "Coherence",
    "aesthetic_consistency": "Aesthetic consistency",
    "expressiveness": "Expressiveness",
}
_REFINEMENT_OPPORTUNITIES = {
    "composition": (
        "Clarify focal hierarchy and spatial balance with explicit placement "
        "or framing rules."
    ),
    "originality": (
        "Add a distinctive generative rule or transformation beyond baseline "
        "runtime scaffolding."
    ),
    "coherence": (
        "Resolve incomplete structure and make visual, motion, and runtime "
        "decisions reinforce one concept."
    ),
    "aesthetic_consistency": (
        "Define a consistent palette, material, or contrast system across the "
        "artifact."
    ),
    "expressiveness": (
        "Strengthen motion, variation, or interaction so the concept develops "
        "over time."
    ),
}


def evaluate_artifact_creative_quality(
    artifact: WorkflowArtifact,
) -> CreativeQualityEvaluation:
    """Analyze an artifact without executing or mutating its content."""

    text = " ".join(
        (
            artifact.title,
            artifact.summary,
            artifact.language,
            artifact.content,
        )
    )
    tokens = _tokens(text)
    line_count = len(artifact.content.splitlines())

    observations = {
        "composition": _score_composition(
            artifact,
            tokens=tokens,
            line_count=line_count,
        ),
        "originality": _score_originality(
            artifact,
            tokens=tokens,
        ),
        "coherence": _score_coherence(
            artifact,
            tokens=tokens,
            line_count=line_count,
        ),
        "aesthetic_consistency": _score_aesthetic_consistency(
            artifact,
            tokens=tokens,
            line_count=line_count,
        ),
        "expressiveness": _score_expressiveness(
            artifact,
            tokens=tokens,
            line_count=line_count,
        ),
    }
    overall = round(mean(item.score for item in observations.values()), 3)
    strengths = tuple(
        (
            f"{_DIMENSION_LABELS[name]}: "
            f"{observation.observation.rstrip('.')}."
        )
        for name, observation in observations.items()
        if observation.score >= 0.72
    )[:3]
    refinement_opportunities = tuple(
        _REFINEMENT_OPPORTUNITIES[name]
        for name, observation in observations.items()
        if observation.score < 0.72
    )
    strong_count = sum(
        observation.level == "strong" for observation in observations.values()
    )
    refinement_count = sum(
        observation.score < 0.72 for observation in observations.values()
    )

    return CreativeQualityEvaluation(
        overall_score=overall,
        composition=observations["composition"],
        originality=observations["originality"],
        coherence=observations["coherence"],
        aesthetic_consistency=observations["aesthetic_consistency"],
        expressiveness=observations["expressiveness"],
        strengths=strengths,
        refinement_opportunities=refinement_opportunities,
        summary=(
            f"{strong_count} of 5 creative dimensions are strong; "
            f"{refinement_count} require focused refinement. "
            f"Deterministic static-analysis score: {overall:.2f}."
        ),
    )


def _score_composition(
    artifact: WorkflowArtifact,
    *,
    tokens: set[str],
    line_count: int,
) -> CreativeQualityObservation:
    markers = _matched(tokens, _COMPOSITION_MARKERS)
    metadata_count = _translation_signal_count(
        artifact,
        lambda translation: (
            *translation.structure_direction,
            *(
                translation.visual_style.composition_tendencies
                if translation.visual_style
                else ()
            ),
            *(
                translation.sacred_geometry.visual_composition
                if translation.sacred_geometry
                else ()
            ),
        ),
    )
    score = (
        0.3
        + (0.12 if artifact.is_creative else 0.0)
        + min(len(markers), 4) * 0.1
        + min(metadata_count, 2) * 0.06
        + (0.06 if line_count >= 8 else 0.0)
    )
    return _observation(
        score,
        (
            f"Detected {len(markers)} spatial hierarchy signal(s) and "
            f"{metadata_count} composition metadata cue(s)"
        ),
        _evidence(markers, metadata_count, "composition metadata"),
    )


def _score_originality(
    artifact: WorkflowArtifact,
    *,
    tokens: set[str],
) -> CreativeQualityObservation:
    markers = _matched(tokens, _ORIGINALITY_MARKERS)
    numeric_variation = len(set(_NUMBER_PATTERN.findall(artifact.content)))
    metadata_count = _translation_signal_count(
        artifact,
        lambda translation: (
            *translation.symbolic_references,
            *translation.geometric_references,
            *translation.musical_references,
        ),
    )
    score = (
        0.26
        + (0.12 if artifact.is_creative else 0.0)
        + min(len(markers), 4) * 0.09
        + (0.14 if numeric_variation >= 6 else 0.0)
        + min(metadata_count, 2) * 0.06
    )
    return _observation(
        score,
        (
            f"Detected {len(markers)} generative variation signal(s), "
            f"{numeric_variation} distinct numeric choice(s), and "
            f"{metadata_count} concept cue(s)"
        ),
        _evidence(
            markers,
            metadata_count,
            "concept metadata",
            extra=(
                f"{numeric_variation} numeric choices"
                if numeric_variation
                else None
            ),
        ),
    )


def _score_coherence(
    artifact: WorkflowArtifact,
    *,
    tokens: set[str],
    line_count: int,
) -> CreativeQualityObservation:
    markers = _matched(tokens, _COHERENCE_MARKERS)
    opening = artifact.content.count("{")
    closing = artifact.content.count("}")
    balanced_structure = opening > 0 and opening == closing
    incomplete = bool(
        re.search(
            r"\b(?:todo|fixme|placeholder|implement later)\b",
            artifact.content,
            re.I,
        )
    )
    score = (
        0.25
        + min(len(markers), 4) * 0.1
        + (0.15 if balanced_structure else 0.0)
        + (0.1 if artifact.runtime and artifact.renderer_id else 0.0)
        + (0.07 if line_count >= 8 else 0.0)
        + (0.08 if not incomplete else -0.18)
    )
    evidence = [
        f"{len(markers)} structural markers",
        "balanced blocks" if balanced_structure else "no balanced block evidence",
    ]
    if artifact.runtime and artifact.renderer_id:
        evidence.append("matched runtime metadata")
    if incomplete:
        evidence.append("incomplete marker")
    structure_state = "balanced" if balanced_structure else "not evidenced"
    return _observation(
        score,
        (
            f"Detected {len(markers)} code structure signal(s); "
            f"block structure is {structure_state}"
        ),
        tuple(evidence[:5]),
    )


def _score_aesthetic_consistency(
    artifact: WorkflowArtifact,
    *,
    tokens: set[str],
    line_count: int,
) -> CreativeQualityObservation:
    markers = _matched(tokens, _AESTHETIC_MARKERS)
    metadata_count = _translation_signal_count(
        artifact,
        lambda translation: (
            *translation.color_material_direction,
            *(
                translation.visual_style.palette_behavior
                if translation.visual_style
                else ()
            ),
            *(
                translation.visual_style.contrast_behavior
                if translation.visual_style
                else ()
            ),
        ),
    )
    color_literal_count = len(
        set(re.findall(r"#[0-9a-f]{3,8}\b", artifact.content, re.I))
    )
    score = (
        0.28
        + (0.12 if artifact.is_creative else 0.0)
        + min(len(markers), 4) * 0.1
        + min(metadata_count, 2) * 0.06
        + (0.08 if color_literal_count >= 2 else 0.0)
        + (0.04 if line_count >= 8 else 0.0)
    )
    return _observation(
        score,
        (
            f"Detected {len(markers)} palette/material signal(s), "
            f"{color_literal_count} explicit color choice(s), and "
            f"{metadata_count} aesthetic metadata cue(s)"
        ),
        _evidence(
            markers,
            metadata_count,
            "aesthetic metadata",
            extra=(
                f"{color_literal_count} explicit colors"
                if color_literal_count
                else None
            ),
        ),
    )


def _score_expressiveness(
    artifact: WorkflowArtifact,
    *,
    tokens: set[str],
    line_count: int,
) -> CreativeQualityObservation:
    markers = _matched(tokens, _EXPRESSIVE_MARKERS)
    metadata_count = _translation_signal_count(
        artifact,
        lambda translation: (
            *translation.movement_language,
            *(
                (
                    mapping.behavior
                    for mapping in translation.audio_reactive.mappings
                )
                if translation.audio_reactive
                else ()
            ),
        ),
    )
    score = (
        0.25
        + (0.12 if artifact.is_creative else 0.0)
        + min(len(markers), 4) * 0.1
        + min(metadata_count, 2) * 0.07
        + (0.08 if line_count >= 8 else 0.0)
    )
    return _observation(
        score,
        (
            f"Detected {len(markers)} motion/variation signal(s) and "
            f"{metadata_count} expressive metadata cue(s)"
        ),
        _evidence(markers, metadata_count, "expressive metadata"),
    )


def _translation_signal_count(
    artifact: WorkflowArtifact,
    selector: Callable[[CreativeTranslation], tuple[str, ...]],
) -> int:
    translation = artifact.creative_translation
    if translation is None:
        return 0
    return len(tuple(value for value in selector(translation) if value.strip()))


def _observation(
    score: float,
    observation: str,
    evidence: tuple[str, ...],
) -> CreativeQualityObservation:
    bounded = round(min(max(score, 0.0), 1.0), 3)
    level = (
        "strong"
        if bounded >= 0.78
        else "developing"
        if bounded >= 0.55
        else "weak"
    )
    return CreativeQualityObservation(
        score=bounded,
        level=level,
        observation=observation,
        evidence=evidence,
    )


def _matched(tokens: set[str], markers: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(tokens.intersection(markers)))


def _evidence(
    markers: tuple[str, ...],
    metadata_count: int,
    metadata_label: str,
    *,
    extra: str | None = None,
) -> tuple[str, ...]:
    evidence = [f"marker: {marker}" for marker in markers[:3]]
    if metadata_count:
        evidence.append(f"{metadata_count} {metadata_label} cue(s)")
    if extra:
        evidence.append(extra)
    return tuple(evidence[:5])


def _tokens(value: str) -> set[str]:
    return _token_set(value)
