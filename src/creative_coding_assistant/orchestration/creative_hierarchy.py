"""Deterministic Creative Hierarchy Planner for V3 workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
    IntentDimensionName,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

HierarchyDimension = Literal[
    "symbolism",
    "narrative",
    "emotion",
    "geometry",
    "motion",
    "rhythm",
    "light_color",
    "audio",
    "interaction",
    "visual_impact",
    "performance",
    "simplicity",
    "complexity",
    "runtime_safety",
    "experiential_depth",
]
HierarchyTier = Literal["primary", "secondary", "flexible"]
HierarchySource = Literal["explicit", "implied", "coherence", "constraint"]

HIERARCHY_AUTHORITY_BOUNDARY = (
    "The Creative Hierarchy Planner ranks creative priorities for inspection "
    "only; it does not select runtimes, providers, models, execution profiles, "
    "artifacts, preview behavior, repair loops, or final creative choices."
)


class CreativeHierarchyPriority(BaseModel):
    """One ranked creative dimension in the current hierarchy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    dimension: HierarchyDimension
    tier: HierarchyTier
    rank: int = Field(ge=1, le=15)
    priority_score: int = Field(ge=0, le=12)
    source: HierarchySource
    rationale: str = Field(min_length=1, max_length=280)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    sacrifice_guidance: str = Field(min_length=1, max_length=240)


class CreativeHierarchyPlan(BaseModel):
    """Inspectable priority hierarchy derived after intent decomposition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_hierarchy_planner"] = "creative_hierarchy_planner"
    primary_creative_priorities: tuple[CreativeHierarchyPriority, ...] = Field(
        min_length=1,
        max_length=5,
    )
    secondary_creative_priorities: tuple[CreativeHierarchyPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    non_negotiable_dimensions: tuple[HierarchyDimension, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    flexible_dimensions: tuple[HierarchyDimension, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    priority_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    priority_conflicts: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hierarchy_confidence: float = Field(ge=0, le=1)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=HIERARCHY_AUTHORITY_BOUNDARY,
        max_length=440,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_hierarchy_plan(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
) -> CreativeHierarchyPlan:
    """Rank creative dimensions without changing runtime or generation behavior."""

    text = _hierarchy_text(request, creative_intent, creative_translation)
    scored = _ranked_dimensions(
        text=text,
        creative_intent=creative_intent,
        creative_translation=creative_translation,
    )
    primary = tuple(
        _priority(item, tier="primary", rank=index + 1)
        for index, item in enumerate(scored[:5])
        if item.score >= 4
    ) or (_priority(scored[0], tier="primary", rank=1),)
    secondary = tuple(
        _priority(item, tier="secondary", rank=index + 1)
        for index, item in enumerate(scored[len(primary) : len(primary) + 5])
        if item.score >= 2
    )
    conflicts = _priority_conflicts(scored, creative_intent, creative_translation)
    hitl_questions = _hitl_questions(scored, conflicts, creative_intent)
    return CreativeHierarchyPlan(
        primary_creative_priorities=primary,
        secondary_creative_priorities=secondary,
        non_negotiable_dimensions=_non_negotiable_dimensions(scored, primary, text),
        flexible_dimensions=_flexible_dimensions(scored, primary, secondary),
        priority_rationale=_priority_rationale(primary, secondary),
        priority_conflicts=conflicts,
        hierarchy_confidence=_hierarchy_confidence(scored, conflicts, creative_intent),
        hitl_questions=hitl_questions,
        prompt_guidance=_prompt_guidance(primary, secondary, conflicts),
        evidence=_evidence(request, route_decision, creative_intent, scored),
    )


def creative_hierarchy_plan_prompt_lines(
    plan: CreativeHierarchyPlan,
) -> tuple[str, ...]:
    """Render hierarchy metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {plan.authority_boundary}",
        f"Hierarchy confidence: {plan.hierarchy_confidence:.2f}.",
    ]
    for item in plan.primary_creative_priorities:
        lines.append(
            f"Primary priority #{item.rank}: {item.dimension}; {item.rationale}"
        )
    for item in plan.secondary_creative_priorities[:4]:
        lines.append(
            f"Secondary priority #{item.rank}: {item.dimension}; {item.rationale}"
        )
    if plan.non_negotiable_dimensions:
        lines.append(
            "Non-negotiable dimensions: "
            + ", ".join(plan.non_negotiable_dimensions)
            + "."
        )
    if plan.flexible_dimensions:
        lines.append(
            "Flexible dimensions: " + ", ".join(plan.flexible_dimensions[:6]) + "."
        )
    lines.extend(f"Priority conflict: {item}" for item in plan.priority_conflicts)
    lines.extend(f"HITL hierarchy question: {item}" for item in plan.hitl_questions)
    lines.extend(f"Hierarchy guidance: {item}" for item in plan.prompt_guidance)
    return tuple(lines[:28])


@dataclass(frozen=True)
class _ScoredDimension:
    dimension: HierarchyDimension
    score: int
    source: HierarchySource
    evidence: tuple[str, ...]


def _ranked_dimensions(
    *,
    text: str,
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
) -> tuple[_ScoredDimension, ...]:
    scores: dict[HierarchyDimension, int] = {item: 0 for item in _ALL_DIMENSIONS}
    evidence: dict[HierarchyDimension, list[str]] = {
        item: [] for item in _ALL_DIMENSIONS
    }
    sources: dict[HierarchyDimension, HierarchySource] = {
        item: "coherence" for item in _ALL_DIMENSIONS
    }
    _score_intent_dimensions(scores, evidence, sources, creative_intent)
    _score_translation(scores, evidence, sources, creative_translation)
    _score_text_emphasis(scores, evidence, sources, text)
    _score_coherence(scores, evidence, sources)
    ranked = sorted(
        (
            _ScoredDimension(
                dimension=dimension,
                score=min(score, 12),
                source=sources[dimension],
                evidence=tuple(_dedupe(evidence[dimension]))[:5],
            )
            for dimension, score in scores.items()
        ),
        key=lambda item: (item.score, -_DIMENSION_ORDER[item.dimension]),
        reverse=True,
    )
    return ranked


def _score_intent_dimensions(
    scores: dict[HierarchyDimension, int],
    evidence: dict[HierarchyDimension, list[str]],
    sources: dict[HierarchyDimension, HierarchySource],
    creative_intent: CreativeIntentDecomposition | None,
) -> None:
    if creative_intent is None:
        return
    for dimension in creative_intent.atomic_dimensions:
        mapped = _INTENT_TO_HIERARCHY[dimension.name]
        if dimension.explicitness == "absent":
            continue
        scores[mapped] += {"explicit": 5, "inferred": 3, "ambiguous": 2}.get(
            dimension.explicitness,
            1,
        )
        scores[mapped] += min(len(dimension.signals), 2)
        sources[mapped] = (
            "explicit" if dimension.explicitness == "explicit" else "implied"
        )
        evidence[mapped].extend(dimension.signals or (dimension.summary,))
    if creative_intent.abstraction_level in {"abstract", "symbolic", "mixed"}:
        scores["experiential_depth"] += 2
        evidence["experiential_depth"].append(
            f"abstraction:{creative_intent.abstraction_level}"
        )


def _score_translation(
    scores: dict[HierarchyDimension, int],
    evidence: dict[HierarchyDimension, list[str]],
    sources: dict[HierarchyDimension, HierarchySource],
    translation: CreativeTranslation | None,
) -> None:
    if translation is None:
        return
    if translation.output_modality is CreativeOutputModality.AUDIO:
        _bump(scores, evidence, sources, "audio", "audio modality", 3)
    elif translation.output_modality is CreativeOutputModality.AUDIOVISUAL:
        _bump(scores, evidence, sources, "audio", "audiovisual modality", 2)
        _bump(scores, evidence, sources, "visual_impact", "audiovisual modality", 2)
    if translation.shader_presets or translation.visual_style:
        _bump(scores, evidence, sources, "visual_impact", "visual style guidance", 2)
    if translation.generation_constraints:
        _bump(scores, evidence, sources, "runtime_safety", "generation constraints", 2)


def _score_text_emphasis(
    scores: dict[HierarchyDimension, int],
    evidence: dict[HierarchyDimension, list[str]],
    sources: dict[HierarchyDimension, HierarchySource],
    text: str,
) -> None:
    for dimension, pattern in _DIMENSION_PATTERNS.items():
        if pattern.search(text):
            _bump(scores, evidence, sources, dimension, f"keyword:{dimension}", 2)
            if _NON_NEGOTIABLE_PATTERN.search(text):
                _bump(
                    scores,
                    evidence,
                    sources,
                    dimension,
                    f"emphasis:{dimension}",
                    2,
                )
    for dimension, pattern in _CONSTRAINT_PATTERNS.items():
        if pattern.search(text):
            _bump(scores, evidence, sources, dimension, f"constraint:{dimension}", 4)


def _score_coherence(
    scores: dict[HierarchyDimension, int],
    evidence: dict[HierarchyDimension, list[str]],
    sources: dict[HierarchyDimension, HierarchySource],
) -> None:
    if scores["symbolism"] >= 5 or scores["narrative"] >= 5:
        _bump(scores, evidence, sources, "experiential_depth", "coherence depth", 2)
    if scores["motion"] >= 5 and scores["light_color"] >= 4:
        _bump(scores, evidence, sources, "visual_impact", "motion plus light", 2)
    if scores["performance"] >= 4:
        _bump(
            scores,
            evidence,
            sources,
            "simplicity",
            "performance needs simplicity",
            2,
        )


def _priority(
    scored: _ScoredDimension,
    *,
    tier: HierarchyTier,
    rank: int,
) -> CreativeHierarchyPriority:
    return CreativeHierarchyPriority(
        dimension=scored.dimension,
        tier=tier,
        rank=rank,
        priority_score=scored.score,
        source=scored.source,
        rationale=_rationale(scored, tier),
        evidence=scored.evidence,
        sacrifice_guidance=_sacrifice_guidance(scored.dimension, tier),
    )


def _rationale(scored: _ScoredDimension, tier: HierarchyTier) -> str:
    label = scored.dimension.replace("_", " ")
    if tier == "primary":
        return f"{label} should dominate because score {scored.score} is strongest."
    if tier == "secondary":
        return f"{label} supports coherence but should not override primary intent."
    return f"{label} may flex if constraints require compromise."


def _sacrifice_guidance(dimension: HierarchyDimension, tier: HierarchyTier) -> str:
    if tier == "primary":
        return f"Do not sacrifice {dimension} unless the user explicitly redirects."
    if dimension in {"performance", "runtime_safety", "simplicity"}:
        return f"Keep {dimension} visible when reducing creative scope."
    return f"{dimension} can be simplified before primary priorities are weakened."


def _non_negotiable_dimensions(
    scored: tuple[_ScoredDimension, ...],
    primary: tuple[CreativeHierarchyPriority, ...],
    text: str,
) -> tuple[HierarchyDimension, ...]:
    emphasized = _NON_NEGOTIABLE_PATTERN.search(text) is not None
    values = [
        item.dimension
        for item in primary
        if item.priority_score >= 7 or emphasized
    ]
    if not values and scored[0].score >= 6:
        values.append(scored[0].dimension)
    return tuple(_dedupe(values))[:6]


def _flexible_dimensions(
    scored: tuple[_ScoredDimension, ...],
    primary: tuple[CreativeHierarchyPriority, ...],
    secondary: tuple[CreativeHierarchyPriority, ...],
) -> tuple[HierarchyDimension, ...]:
    fixed = {item.dimension for item in (*primary, *secondary)}
    flexible = [item.dimension for item in scored if item.dimension not in fixed]
    return tuple(flexible[:8])


def _priority_rationale(
    primary: tuple[CreativeHierarchyPriority, ...],
    secondary: tuple[CreativeHierarchyPriority, ...],
) -> tuple[str, ...]:
    rationale = [
        f"Primary hierarchy: {', '.join(item.dimension for item in primary)}."
    ]
    if secondary:
        rationale.append(
            f"Secondary support: {', '.join(item.dimension for item in secondary[:4])}."
        )
    rationale.extend(item.rationale for item in primary)
    return tuple(rationale[:8])


def _priority_conflicts(
    scored: tuple[_ScoredDimension, ...],
    creative_intent: CreativeIntentDecomposition | None,
    translation: CreativeTranslation | None,
) -> tuple[str, ...]:
    score = {item.dimension: item.score for item in scored}
    conflicts: list[str] = []
    if score["complexity"] >= 4 and score["simplicity"] >= 4:
        conflicts.append("Complexity and simplicity are both emphasized.")
    if score["visual_impact"] >= 4 and score["performance"] >= 4:
        conflicts.append("Visual impact may compete with performance priority.")
    if score["audio"] >= 5 and score["visual_impact"] >= 5:
        conflicts.append("Audio and visual impact may need an explicit lead.")
    if score["interaction"] >= 5 and score["runtime_safety"] >= 4:
        conflicts.append("Interaction priority must stay inside runtime safety.")
    if creative_intent is not None:
        conflicts.extend(creative_intent.unresolved_intent_gaps[:2])
    if translation is not None and translation.output_modality is None:
        conflicts.append("Output modality priority is inferred.")
    return tuple(_dedupe(conflicts))[:8]


def _hitl_questions(
    scored: tuple[_ScoredDimension, ...],
    conflicts: tuple[str, ...],
    creative_intent: CreativeIntentDecomposition | None,
) -> tuple[str, ...]:
    questions: list[str] = []
    if len(scored) > 1 and scored[0].score == scored[1].score:
        questions.append(
            f"Should {scored[0].dimension} or {scored[1].dimension} lead?"
        )
    if creative_intent is not None:
        questions.extend(creative_intent.hitl_questions[:2])
    for conflict in conflicts:
        if "Audio and visual" in conflict:
            questions.append("Should audio or visual impact dominate the result?")
        elif "Visual impact" in conflict:
            questions.append("Should visual richness or performance win first?")
        elif "Complexity" in conflict:
            questions.append("Should the piece favor richness or simplicity?")
    return tuple(_dedupe(questions))[:6]


def _hierarchy_confidence(
    scored: tuple[_ScoredDimension, ...],
    conflicts: tuple[str, ...],
    creative_intent: CreativeIntentDecomposition | None,
) -> float:
    active_count = sum(1 for item in scored if item.score > 0)
    confidence = 0.42 + min(active_count, 8) * 0.04 + scored[0].score * 0.025
    confidence -= min(len(conflicts), 4) * 0.05
    if creative_intent is not None:
        confidence -= min(len(creative_intent.unresolved_intent_gaps), 3) * 0.04
    return round(min(0.94, max(0.32, confidence)), 2)


def _prompt_guidance(
    primary: tuple[CreativeHierarchyPriority, ...],
    secondary: tuple[CreativeHierarchyPriority, ...],
    conflicts: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use hierarchy priorities as ordering guidance, not as new features.",
        "Protect primary priorities before optimizing secondary dimensions.",
    ]
    guidance.append(
        "Primary dimensions: " + ", ".join(item.dimension for item in primary) + "."
    )
    if secondary:
        guidance.append(
            "Secondary dimensions may flex first: "
            + ", ".join(item.dimension for item in secondary[:4])
            + "."
        )
    if conflicts:
        guidance.append("Make hierarchy conflicts explicit before expanding scope.")
    return tuple(guidance[:8])


def _evidence(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    scored: tuple[_ScoredDimension, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if creative_intent is not None:
        evidence.append(f"Intent substrate: {creative_intent.primary_expression}.")
    evidence.append(
        "Top hierarchy scores: "
        + ", ".join(f"{item.dimension}={item.score}" for item in scored[:5])
        + "."
    )
    return tuple(_dedupe(evidence))[:10]


def _hierarchy_text(
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
) -> str:
    parts = [request.query]
    if creative_intent is not None:
        parts.extend(
            [
                creative_intent.primary_expression,
                creative_intent.experiential_goal,
                " ".join(
                    signal
                    for dimension in creative_intent.atomic_dimensions
                    for signal in dimension.signals
                ),
            ]
        )
    if creative_translation is not None:
        parts.extend(
            [
                " ".join(creative_translation.generation_constraints),
                " ".join(creative_translation.runtime_recommendations),
            ]
        )
    return " ".join(" ".join(parts).lower().split())


def _bump(
    scores: dict[HierarchyDimension, int],
    evidence: dict[HierarchyDimension, list[str]],
    sources: dict[HierarchyDimension, HierarchySource],
    dimension: HierarchyDimension,
    note: str,
    amount: int,
) -> None:
    scores[dimension] += amount
    evidence[dimension].append(note)
    if sources[dimension] != "explicit":
        sources[dimension] = (
            "constraint" if note.startswith("constraint") else "implied"
        )


def _dedupe(values: Sequence[object]) -> tuple:
    deduped: list[object] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return tuple(deduped)


_ALL_DIMENSIONS: tuple[HierarchyDimension, ...] = (
    "symbolism",
    "narrative",
    "emotion",
    "geometry",
    "motion",
    "rhythm",
    "light_color",
    "audio",
    "interaction",
    "visual_impact",
    "performance",
    "simplicity",
    "complexity",
    "runtime_safety",
    "experiential_depth",
)
_DIMENSION_ORDER = {dimension: index for index, dimension in enumerate(_ALL_DIMENSIONS)}
_INTENT_TO_HIERARCHY: dict[IntentDimensionName, HierarchyDimension] = {
    "narrative": "narrative",
    "symbolic": "symbolism",
    "emotional": "emotion",
    "geometric": "geometry",
    "motion": "motion",
    "rhythm": "rhythm",
    "light_color": "light_color",
    "audio": "audio",
    "interaction": "interaction",
    "climax_transformation": "experiential_depth",
}
_DIMENSION_PATTERNS: dict[HierarchyDimension, re.Pattern[str]] = {
    "symbolism": re.compile(r"\b(?:symbol|symbolic|myth|ritual|archetype)\b"),
    "narrative": re.compile(r"\b(?:story|narrative|journey|arc)\b"),
    "emotion": re.compile(
        r"\b(?:emotion|emotional|feeling|mood|awe|calm|tense|descent|emergence)\b"
    ),
    "geometry": re.compile(r"\b(?:geometry|mandala|spiral|grid|fractal)\b"),
    "motion": re.compile(r"\b(?:motion|movement|morph|pulse|flow|drift)\b"),
    "rhythm": re.compile(r"\b(?:rhythm|tempo|beat|loop|polyrhythm)\b"),
    "light_color": re.compile(r"\b(?:light|color|palette|glow|luminous)\b"),
    "audio": re.compile(r"\b(?:audio|sound|music|sonic|synth)\b"),
    "interaction": re.compile(r"\b(?:interaction|interactive|mouse|click|touch)\b"),
    "visual_impact": re.compile(r"\b(?:visual impact|spectacle|immersive|cinematic)\b"),
    "experiential_depth": re.compile(r"\b(?:experience|depth|transformation)\b"),
}
_CONSTRAINT_PATTERNS: dict[HierarchyDimension, re.Pattern[str]] = {
    "performance": re.compile(r"\b(?:performance|60\s?fps|fast|realtime|mobile)\b"),
    "simplicity": re.compile(r"\b(?:simple|minimal|lightweight|small)\b"),
    "complexity": re.compile(r"\b(?:complex|dense|elaborate|many layers)\b"),
    "runtime_safety": re.compile(r"\b(?:safe|no autoplay|browser-safe|accessible)\b"),
}
_NON_NEGOTIABLE_PATTERN = re.compile(
    r"\b(?:must|non-negotiable|highest priority|most important|prioriti[sz]e)\b"
)
