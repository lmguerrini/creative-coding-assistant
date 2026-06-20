"""Deterministic Creative Strategy Engine for V3 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

CreativeStrategyId = Literal[
    "recursive_emergence",
    "fractal_growth",
    "particle_cosmology",
    "cellular_evolution",
    "sacred_geometry",
    "field_dynamics",
    "minimal_generative_systems",
]

STRATEGY_AUTHORITY_BOUNDARY = (
    "The Creative Strategy Engine selects high-level artistic strategy only; "
    "it does not choose runtimes, rendering technology, implementation "
    "techniques, models, or providers."
)


class CreativeStrategyAlternative(BaseModel):
    """A secondary high-level strategy worth considering."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    strategy: CreativeStrategyId
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=260)


class CreativeStrategyProfile(BaseModel):
    """Inspectable creative strategy metadata for one assistant run."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_strategy_engine"] = "creative_strategy_engine"
    primary_strategy: CreativeStrategyId
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=360)
    creative_goals: tuple[str, ...] = Field(min_length=1, max_length=6)
    symbolic_alignment: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    alternative_strategies: tuple[CreativeStrategyAlternative, ...] = Field(
        default_factory=tuple,
        max_length=3,
    )
    strategy_directives: tuple[str, ...] = Field(min_length=1, max_length=6)
    implementation_boundary: str = Field(
        default=STRATEGY_AUTHORITY_BOUNDARY,
        max_length=360,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_strategy_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
) -> CreativeStrategyProfile:
    """Select high-level creative strategy using deterministic request signals."""

    normalized = _strategy_text(request, creative_translation)
    domains = _effective_domains(request, route_decision)
    scored = sorted(
        (
            _score_strategy(
                strategy,
                normalized=normalized,
                request=request,
                route_decision=route_decision,
                creative_translation=creative_translation,
            )
            for strategy in _STRATEGIES
        ),
        key=lambda item: (item.score, item.strategy),
        reverse=True,
    )
    primary = scored[0]
    alternatives = tuple(
        CreativeStrategyAlternative(
            strategy=item.strategy,
            confidence=_confidence(item.score),
            rationale=_alternative_rationale(item),
        )
        for item in scored[1:4]
        if item.score > 0
    )
    return CreativeStrategyProfile(
        primary_strategy=primary.strategy,
        confidence=_confidence(primary.score),
        rationale=_primary_rationale(primary),
        creative_goals=_creative_goals(
            strategy=primary.strategy,
            request=request,
            creative_translation=creative_translation,
        ),
        symbolic_alignment=_symbolic_alignment(
            strategy=primary.strategy,
            creative_translation=creative_translation,
        ),
        alternative_strategies=alternatives,
        strategy_directives=_strategy_directives(primary.strategy),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            domains=domains,
            scored=scored,
        ),
    )


def creative_strategy_prompt_lines(
    profile: CreativeStrategyProfile,
) -> tuple[str, ...]:
    """Render strategy metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {profile.implementation_boundary}",
        f"Primary strategy: {profile.primary_strategy}.",
        f"Strategy confidence: {profile.confidence:.2f}.",
        f"Strategy rationale: {profile.rationale}",
    ]
    lines.extend(f"Creative goal: {item}" for item in profile.creative_goals)
    lines.extend(
        f"Symbolic alignment: {item}" for item in profile.symbolic_alignment[:4]
    )
    lines.extend(f"Strategy directive: {item}" for item in profile.strategy_directives)
    return tuple(lines[:18])


@dataclass(frozen=True)
class _StrategySignal:
    strategy: CreativeStrategyId
    label: str
    keywords: tuple[str, ...]
    rationale: str
    directives: tuple[str, ...]
    goals: tuple[str, ...]


@dataclass(frozen=True)
class _ScoredStrategy:
    strategy: CreativeStrategyId
    label: str
    score: int
    matched_signals: tuple[str, ...]
    rationale: str


def _score_strategy(
    signal: _StrategySignal,
    *,
    normalized: str,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
) -> _ScoredStrategy:
    matched: list[str] = []
    score = 0
    for keyword in signal.keywords:
        if _keyword_matches(normalized, keyword):
            matched.append(keyword)
            score += 2
    if creative_translation is not None:
        score += _translation_score(signal, creative_translation, matched)
    if route_decision is not None and len(route_decision.domains) > 1:
        score += int(signal.strategy in _MULTI_DOMAIN_STRATEGIES)
    if request.attachments:
        score += int(signal.strategy in _REFERENCE_FRIENDLY_STRATEGIES)
    if score == 0 and signal.strategy == "minimal_generative_systems":
        score = 1
        matched.append("default bounded strategy")
    return _ScoredStrategy(
        strategy=signal.strategy,
        label=signal.label,
        score=score,
        matched_signals=tuple(_dedupe_text(matched))[:8],
        rationale=signal.rationale,
    )


def _translation_score(
    signal: _StrategySignal,
    translation: CreativeTranslation,
    matched: list[str],
) -> int:
    score = 0
    if signal.strategy == "sacred_geometry" and translation.sacred_geometry is not None:
        score += 5
        matched.append("sacred geometry guidance")
    if signal.strategy == "field_dynamics" and translation.audio_reactive is not None:
        score += 2
        matched.append("audio-reactive mapping")
    if signal.strategy == "minimal_generative_systems" and (
        "minimal" in translation.mood_atmosphere
        or "calm" in translation.mood_atmosphere
    ):
        score += 3
        matched.append("minimal mood")
    for value in (
        *translation.symbolic_references,
        *translation.geometric_references,
        *translation.mood_atmosphere,
        *translation.movement_language,
        *translation.structure_direction,
    ):
        if any(keyword in value.lower() for keyword in signal.keywords):
            score += 1
            matched.append(value)
    return score


def _confidence(score: int) -> float:
    return min(0.95, max(0.35, round(0.35 + score * 0.08, 2)))


def _primary_rationale(scored: _ScoredStrategy) -> str:
    if scored.matched_signals:
        joined = ", ".join(scored.matched_signals[:4])
        return f"{scored.label} best matches detected signals: {joined}."
    return f"{scored.label} provides a bounded default creative strategy."


def _alternative_rationale(scored: _ScoredStrategy) -> str:
    if scored.matched_signals:
        return (
            f"Also relevant because of {', '.join(scored.matched_signals[:3])}."
        )
    return scored.rationale


def _creative_goals(
    *,
    strategy: CreativeStrategyId,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> tuple[str, ...]:
    signal = _SIGNAL_BY_STRATEGY[strategy]
    goals = list(signal.goals)
    if creative_translation is not None:
        goals.insert(0, f"Preserve intent: {creative_translation.creative_intent}.")
        if creative_translation.mood_atmosphere:
            goals.append(
                "Carry mood: "
                + ", ".join(creative_translation.mood_atmosphere[:3])
                + "."
            )
    else:
        goals.insert(0, f"Preserve request: {_compact(request.query)[:180]}.")
    return tuple(_dedupe_text(goals))[:6]


def _symbolic_alignment(
    *,
    strategy: CreativeStrategyId,
    creative_translation: CreativeTranslation | None,
) -> tuple[str, ...]:
    alignment: list[str] = [_SIGNAL_BY_STRATEGY[strategy].label]
    if creative_translation is not None:
        alignment.extend(creative_translation.symbolic_references)
        alignment.extend(creative_translation.geometric_references)
        alignment.extend(creative_translation.musical_references[:2])
    return tuple(_dedupe_text(alignment))[:8]


def _strategy_directives(strategy: CreativeStrategyId) -> tuple[str, ...]:
    return _SIGNAL_BY_STRATEGY[strategy].directives


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    domains: tuple[CreativeCodingDomain, ...],
    scored: list[_ScoredStrategy],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if domains:
        evidence.append(
            "Domains: " + ", ".join(domain.value for domain in domains) + "."
        )
    if creative_translation is not None:
        evidence.append(f"Creative intent: {creative_translation.creative_intent}.")
    if scored and scored[0].matched_signals:
        evidence.append(
            "Primary signals: " + ", ".join(scored[0].matched_signals[:4]) + "."
        )
    evidence.append(
        "Strategy scores: "
        + ", ".join(f"{item.strategy}={item.score}" for item in scored[:4])
        + "."
    )
    return tuple(_dedupe_text(evidence))[:10]


def _strategy_text(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> str:
    parts = [request.query]
    if creative_translation is not None:
        parts.extend(
            (
                creative_translation.creative_intent,
                " ".join(creative_translation.symbolic_references),
                " ".join(creative_translation.geometric_references),
                " ".join(creative_translation.musical_references),
                " ".join(creative_translation.mood_atmosphere),
                " ".join(creative_translation.movement_language),
                " ".join(creative_translation.structure_direction),
            )
        )
    return _compact(" ".join(parts)).lower()


def _keyword_matches(normalized: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", normalized) is not None


def _effective_domains(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> tuple[CreativeCodingDomain, ...]:
    if request.domains:
        return request.domains
    if request.domain is not None:
        return (request.domain,)
    if route_decision is not None and route_decision.domains:
        return route_decision.domains
    return ()


def _compact(value: str) -> str:
    return " ".join(value.strip().split())


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = _compact(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


_STRATEGIES: tuple[_StrategySignal, ...] = (
    _StrategySignal(
        strategy="recursive_emergence",
        label="Recursive Emergence",
        keywords=(
            "emergence",
            "emergent",
            "recursive",
            "self organizing",
            "self-organizing",
            "swarm",
            "feedback",
            "iteration",
        ),
        rationale="Use layered repetition and feedback as the artistic logic.",
        directives=(
            "Use emergence as the conceptual spine, not as an implementation mandate.",
            "Let simple rules imply complex behavior in the final creative direction.",
        ),
        goals=(
            "Express complexity emerging from simple repeated relationships.",
            "Keep the system legible enough for the user to direct.",
        ),
    ),
    _StrategySignal(
        strategy="fractal_growth",
        label="Fractal Growth",
        keywords=(
            "fractal",
            "branch",
            "branching",
            "growth",
            "recursive growth",
            "root",
            "tree",
            "vine",
        ),
        rationale="Use self-similar growth and scale relationships.",
        directives=(
            "Prioritize scale, branching, and progressive structure.",
            "Avoid locking the strategy to any specific algorithm or renderer.",
        ),
        goals=(
            "Create a sense of organic expansion across nested scales.",
            "Balance repetition with visible variation.",
        ),
    ),
    _StrategySignal(
        strategy="particle_cosmology",
        label="Particle Cosmology",
        keywords=(
            "particle",
            "particles",
            "cosmic",
            "cosmos",
            "galaxy",
            "nebula",
            "stars",
            "orbit",
            "constellation",
        ),
        rationale="Use many small agents to evoke cosmic or atmospheric structure.",
        directives=(
            "Frame the concept around collective motion and spatial density.",
            "Keep particle density subordinate to the requested mood and clarity.",
        ),
        goals=(
            "Evoke a coherent world through many small moving elements.",
            "Use density, drift, and clustering as expressive levers.",
        ),
    ),
    _StrategySignal(
        strategy="cellular_evolution",
        label="Cellular Evolution",
        keywords=(
            "cell",
            "cellular",
            "organism",
            "biological",
            "evolution",
            "ecosystem",
            "life",
            "mutation",
        ),
        rationale="Use life-like adaptation, growth, and local interaction.",
        directives=(
            "Shape the concept around local relationships and gradual change.",
            "Keep biological metaphor high-level unless the user asks for detail.",
        ),
        goals=(
            "Suggest living systems through adaptation and local behavior.",
            "Preserve a clear relationship between individual forms and the whole.",
        ),
    ),
    _StrategySignal(
        strategy="sacred_geometry",
        label="Sacred Geometry",
        keywords=(
            "sacred geometry",
            "mandala",
            "yantra",
            "flower of life",
            "metatron",
            "golden ratio",
            "symmetry",
            "ritual",
        ),
        rationale="Use symbolic geometry as the main compositional logic.",
        directives=(
            "Preserve symbolic structure as high-level composition guidance.",
            "Keep spiritual symbolism respectful and user-directed.",
        ),
        goals=(
            "Align form, rhythm, and symmetry with the symbolic brief.",
            "Use geometry to organize attention and meaning.",
        ),
    ),
    _StrategySignal(
        strategy="field_dynamics",
        label="Field Dynamics",
        keywords=(
            "field",
            "flow",
            "fluid",
            "magnetic",
            "wind",
            "wave",
            "ripple",
            "current",
            "drift",
        ),
        rationale="Use invisible forces, gradients, or flows as the creative logic.",
        directives=(
            "Prioritize directional forces and spatial relationships.",
            "Describe motion language without selecting a concrete simulation method.",
        ),
        goals=(
            "Make invisible forces feel perceptible through motion and structure.",
            "Keep the experience coherent across local and global movement.",
        ),
    ),
    _StrategySignal(
        strategy="minimal_generative_systems",
        label="Minimal Generative Systems",
        keywords=(
            "minimal",
            "simple",
            "quiet",
            "restrained",
            "calm",
            "monochrome",
            "subtle",
            "sparse",
        ),
        rationale="Use restraint, limited rules, and focused variation.",
        directives=(
            "Prefer a small number of strong creative rules.",
            "Do not add extra complexity unless it supports the user's intent.",
        ),
        goals=(
            "Create impact through restraint and precise variation.",
            "Keep the system understandable and easy to refine.",
        ),
    ),
)
_SIGNAL_BY_STRATEGY = {item.strategy: item for item in _STRATEGIES}
_MULTI_DOMAIN_STRATEGIES = frozenset(
    {"recursive_emergence", "field_dynamics", "minimal_generative_systems"}
)
_REFERENCE_FRIENDLY_STRATEGIES = frozenset(
    {"sacred_geometry", "minimal_generative_systems", "field_dynamics"}
)
