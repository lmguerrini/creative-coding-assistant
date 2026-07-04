"""Deterministic symbolic and sacred-geometry consistency checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import mean

from creative_coding_assistant.orchestration.artifacts import (
    SacredConsistencyEvaluation,
    SacredConsistencyObservation,
    WorkflowArtifact,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#'-]+")
_OVERCLAIM_PATTERNS = (
    re.compile(
        r"\b(?:heal|heals|healing|cure|cures|curing)\b"
        r".{0,32}\b(?:frequency|energy|geometry|symbol|chakra)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:activate|activates|activating|open|opens|opening)\b"
        r".{0,32}\b(?:chakra|energy field|spiritual|consciousness)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:guaranteed|proven|authoritative|divine truth|cosmic truth|"
        r"ancient secret|mystical power|spiritual authority)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:manifest|manifests|manifesting|enlighten|enlightenment|"
        r"raises? vibration)\b",
        re.I,
    ),
)
_GEOMETRY_MARKERS = frozenset(
    {
        "angle",
        "arc",
        "axis",
        "center",
        "circle",
        "concentric",
        "cos",
        "curve",
        "grid",
        "hex",
        "hexagon",
        "line",
        "node",
        "orbit",
        "phi",
        "polygon",
        "radius",
        "radial",
        "ring",
        "rotate",
        "segment",
        "sin",
        "spiral",
        "symmetry",
        "triangle",
        "vertex",
    }
)
_VISUAL_MARKERS = frozenset(
    {
        "background",
        "canvas",
        "color",
        "draw",
        "fill",
        "fragment",
        "glow",
        "light",
        "material",
        "palette",
        "scene",
        "shader",
        "stroke",
        "visual",
    }
)
_AUDIO_MARKERS = frozenset(
    {
        "amplitude",
        "audio",
        "band",
        "bass",
        "beat",
        "frequency",
        "mids",
        "oscillator",
        "rhythm",
        "sound",
        "spectrum",
        "tempo",
    }
)
_MOTION_MARKERS = frozenset(
    {
        "animate",
        "breathe",
        "drift",
        "flow",
        "framecount",
        "modulate",
        "orbit",
        "phase",
        "pulse",
        "rotate",
        "wave",
    }
)
_CONCEPT_ALIASES = {
    "cathedral geometry": frozenset(
        {"arch", "arches", "axis", "bay", "cathedral", "grid", "rose", "window"}
    ),
    "fibonacci": frozenset({"fibonacci", "sequence", "spiral", "radius"}),
    "flower of life": frozenset(
        {"circle", "circles", "flower", "hexagonal", "lattice", "overlap"}
    ),
    "fractal": frozenset({"fractal", "recursive", "scale", "self-similar"}),
    "fractal symmetry": frozenset(
        {"fractal", "recursive", "scale", "self-similar", "symmetry"}
    ),
    "golden ratio": frozenset({"golden", "phi", "proportion", "ratio"}),
    "mandala": frozenset(
        {"center", "circle", "concentric", "mandala", "radial", "ring"}
    ),
    "merkaba": frozenset({"merkaba", "tetra", "tetrahedron", "wireframe"}),
    "metatron's cube": frozenset(
        {"connect", "cube", "line", "metatron", "network", "node"}
    ),
    "radial symmetry": frozenset(
        {"angle", "center", "radial", "rotate", "segment", "symmetry"}
    ),
    "sacred geometry": frozenset(
        {"circle", "geometry", "grid", "radial", "sacred", "symmetry"}
    ),
    "spiral": frozenset({"angle", "orbit", "radius", "spiral"}),
    "sri yantra": frozenset(
        {"circle", "frame", "sri", "triangle", "triangles", "yantra"}
    ),
    "temple geometry": frozenset(
        {"axis", "bay", "frame", "grid", "proportion", "temple", "threshold"}
    ),
    "torus": frozenset({"donut", "ring", "toroidal", "torus"}),
    "vesica piscis": frozenset({"circle", "lens", "overlap", "piscis", "vesica"}),
    "yantra": frozenset({"circle", "frame", "triangle", "triangles", "yantra"}),
}
_DIMENSION_LABELS = {
    "alignment": "Guidance alignment",
    "motif_consistency": "Geometric motifs",
    "modality_coherence": "Modality coherence",
    "claim_safety": "Claim safety",
}
_REFINEMENT_OPPORTUNITIES = {
    "alignment": (
        "Restate the requested symbolic or sacred-geometry motif through explicit "
        "artifact structure, naming, or comments."
    ),
    "motif_consistency": (
        "Add concrete geometric construction signals such as center, radius, "
        "segment count, symmetry, or proportional spacing."
    ),
    "modality_coherence": (
        "Tie symbolic or geometric cues to the selected visual, audio, or "
        "audiovisual modality without adding new claims."
    ),
    "claim_safety": (
        "Replace symbolic authority claims with bounded visual or interaction "
        "design language."
    ),
}


def evaluate_artifact_sacred_consistency(
    artifact: WorkflowArtifact,
) -> SacredConsistencyEvaluation | None:
    """Evaluate symbolic/geometric consistency without mutating generated content."""

    translation = artifact.creative_translation
    if not _has_sacred_consistency_metadata(translation):
        return None

    text = " ".join(
        (
            artifact.title,
            artifact.summary,
            artifact.language,
            artifact.content,
        )
    )
    tokens = _tokens(text)
    metadata_terms = _metadata_terms(translation)

    observations = {
        "alignment": _score_alignment(
            translation,
            tokens=tokens,
            metadata_terms=metadata_terms,
        ),
        "motif_consistency": _score_motif_consistency(
            translation,
            tokens=tokens,
            metadata_terms=metadata_terms,
        ),
        "modality_coherence": _score_modality_coherence(
            translation,
            tokens=tokens,
        ),
        "claim_safety": _score_claim_safety(text),
    }
    overall = round(mean(observation.score for observation in observations.values()), 3)
    strengths = tuple(
        (f"{_DIMENSION_LABELS[name]}: {observation.observation.rstrip('.')}.")
        for name, observation in observations.items()
        if observation.score >= 0.78
    )[:3]
    refinement_opportunities = tuple(
        _REFINEMENT_OPPORTUNITIES[name]
        for name, observation in observations.items()
        if observation.score < 0.72
    )[:5]
    refinement_count = len(refinement_opportunities)
    aligned_count = sum(
        observation.level == "aligned" for observation in observations.values()
    )

    return SacredConsistencyEvaluation(
        overall_score=overall,
        alignment=observations["alignment"],
        motif_consistency=observations["motif_consistency"],
        modality_coherence=observations["modality_coherence"],
        claim_safety=observations["claim_safety"],
        strengths=strengths,
        refinement_opportunities=refinement_opportunities,
        summary=(
            f"Checked {len(metadata_terms)} symbolic/geometric metadata cue(s); "
            f"{aligned_count} of 4 consistency dimensions are aligned and "
            f"{refinement_count} need refinement. "
            f"Bounded motif-analysis score: {overall:.2f}."
        ),
    )


def _score_alignment(
    translation: CreativeTranslation,
    *,
    tokens: set[str],
    metadata_terms: tuple[str, ...],
) -> SacredConsistencyObservation:
    matched_terms = _matched_metadata_terms(metadata_terms, tokens)
    guidance_count = _guidance_signal_count(translation)
    match_ratio = len(matched_terms) / max(len(metadata_terms), 1)
    score = (
        0.36
        + match_ratio * 0.38
        + min(guidance_count, 4) * 0.04
        + (0.08 if translation.sacred_geometry is not None else 0.0)
    )
    return _observation(
        score,
        (
            f"Matched {len(matched_terms)} of {len(metadata_terms)} "
            f"symbolic/geometric metadata cue(s) in artifact signals"
        ),
        _evidence(matched_terms, extra=f"{guidance_count} guidance cue(s)"),
    )


def _score_motif_consistency(
    translation: CreativeTranslation,
    *,
    tokens: set[str],
    metadata_terms: tuple[str, ...],
) -> SacredConsistencyObservation:
    markers = tuple(sorted(tokens.intersection(_GEOMETRY_MARKERS)))
    matched_terms = _matched_metadata_terms(metadata_terms, tokens)
    sacred = translation.sacred_geometry
    structural_cues = (
        len(sacred.geometric_structure)
        + len(sacred.symmetry_type)
        + len(sacred.visual_composition)
        if sacred is not None
        else 0
    )
    score = (
        0.28
        + min(len(markers), 5) * 0.08
        + min(structural_cues, 4) * 0.04
        + min(len(matched_terms), 3) * 0.06
    )
    return _observation(
        score,
        (
            f"Detected {len(markers)} geometric construction signal(s) and "
            f"{structural_cues} structural guidance cue(s)"
        ),
        _evidence(markers, extra=f"{len(matched_terms)} matched motif cue(s)"),
    )


def _score_modality_coherence(
    translation: CreativeTranslation,
    *,
    tokens: set[str],
) -> SacredConsistencyObservation:
    visual_markers = tuple(sorted(tokens.intersection(_VISUAL_MARKERS)))
    audio_markers = tuple(sorted(tokens.intersection(_AUDIO_MARKERS)))
    motion_markers = tuple(sorted(tokens.intersection(_MOTION_MARKERS)))
    sacred = translation.sacred_geometry
    audio_guidance_count = len(sacred.audio_implications) if sacred is not None else 0
    modality = translation.output_modality

    if modality == CreativeOutputModality.AUDIO:
        score = 0.42 + min(len(audio_markers) + audio_guidance_count, 4) * 0.11
        observation = (
            f"Detected {len(audio_markers)} audio signal(s) for an audio "
            "symbolic/geometric request"
        )
        evidence = _evidence(
            audio_markers,
            extra=f"{audio_guidance_count} audio cue(s)",
        )
    elif modality == CreativeOutputModality.AUDIOVISUAL:
        score = (
            0.34
            + min(len(visual_markers), 3) * 0.08
            + min(len(audio_markers) + audio_guidance_count, 3) * 0.08
            + min(len(motion_markers), 2) * 0.06
        )
        observation = (
            f"Detected {len(visual_markers)} visual, {len(audio_markers)} audio, "
            f"and {len(motion_markers)} motion signal(s)"
        )
        evidence = _evidence(
            (*visual_markers[:2], *audio_markers[:2], *motion_markers[:1]),
            extra=f"{audio_guidance_count} audio guidance cue(s)",
        )
    else:
        score = (
            0.42
            + min(len(visual_markers), 4) * 0.08
            + min(len(motion_markers), 2) * 0.06
        )
        observation = (
            f"Detected {len(visual_markers)} visual signal(s) and "
            f"{len(motion_markers)} motion signal(s)"
        )
        evidence = _evidence((*visual_markers[:3], *motion_markers[:2]))

    return _observation(score, observation, evidence)


def _score_claim_safety(text: str) -> SacredConsistencyObservation:
    matches = _overclaim_matches(text)
    if matches:
        return _observation(
            0.24,
            (
                "Detected unsupported symbolic authority language in generated "
                "artifact signals"
            ),
            tuple(f"overclaim: {match}" for match in matches[:5]),
        )
    return _observation(
        0.9,
        "No unsupported symbolic authority markers were detected",
        ("bounded design-motif language",),
    )


def _has_sacred_consistency_metadata(
    translation: CreativeTranslation | None,
) -> bool:
    if translation is None:
        return False
    return bool(
        translation.sacred_geometry is not None
        or translation.symbolic_references
        or translation.geometric_references
    )


def _metadata_terms(translation: CreativeTranslation) -> tuple[str, ...]:
    terms = [
        *translation.symbolic_references,
        *translation.geometric_references,
        *(
            translation.sacred_geometry.concepts
            if translation.sacred_geometry is not None
            else ()
        ),
    ]
    return _dedupe(value.lower() for value in terms if value.strip())


def _matched_metadata_terms(
    metadata_terms: tuple[str, ...],
    tokens: set[str],
) -> tuple[str, ...]:
    matched: list[str] = []
    for term in metadata_terms:
        aliases = _aliases_for_term(term)
        if aliases.intersection(tokens):
            matched.append(term)
    return tuple(matched)


def _aliases_for_term(term: str) -> frozenset[str]:
    normalized = term.lower().replace("’", "'")
    aliases = _CONCEPT_ALIASES.get(normalized)
    if aliases is not None:
        return aliases
    return frozenset(_tokens(normalized))


def _guidance_signal_count(translation: CreativeTranslation) -> int:
    sacred = translation.sacred_geometry
    if sacred is None:
        return 0
    return sum(
        len(values)
        for values in (
            sacred.geometric_structure,
            sacred.symmetry_type,
            sacred.movement_behavior,
            sacred.visual_composition,
            sacred.audio_implications,
            sacred.generation_constraints,
        )
    )


def _overclaim_matches(text: str) -> tuple[str, ...]:
    matches: list[str] = []
    for pattern in _OVERCLAIM_PATTERNS:
        matches.extend(match.group(0).strip() for match in pattern.finditer(text))
    return tuple(matches[:5])


def _observation(
    score: float,
    observation: str,
    evidence: tuple[str, ...],
) -> SacredConsistencyObservation:
    bounded = round(min(max(score, 0.0), 1.0), 3)
    level = (
        "aligned"
        if bounded >= 0.78
        else "partial"
        if bounded >= 0.55
        else "unsupported"
    )
    return SacredConsistencyObservation(
        score=bounded,
        level=level,
        observation=observation,
        evidence=evidence,
    )


def _evidence(
    values: Iterable[str],
    *,
    extra: str | None = None,
) -> tuple[str, ...]:
    evidence = [f"marker: {value}" for value in tuple(values)[:4]]
    if extra:
        evidence.append(extra)
    return tuple(evidence[:5])


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(value.lower().replace("’", "'")))
