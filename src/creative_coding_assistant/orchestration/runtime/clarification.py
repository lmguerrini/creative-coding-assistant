"""Bounded human-in-the-loop clarification for ambiguous generation requests."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import AssistantArtifactRefinement
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class ClarificationReason(StrEnum):
    AMBIGUOUS_MODALITY = "ambiguous_modality"
    UNCLEAR_OUTPUT_TARGET = "unclear_output_target"
    CONFLICTING_STYLE_RUNTIME = "conflicting_style_runtime"
    MISSING_PERFORMANCE_EXPORT_TARGET = "missing_performance_export_target"
    HIGH_COST_MULTI_CANDIDATE = "high_cost_multi_candidate"


class ClarificationQuestion(BaseModel):
    """One focused clarification question with bounded suggested answers."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=240)
    kind: Literal["single_choice", "short_answer"] = "single_choice"
    suggested_options: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    default_recommendation: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_options(self) -> ClarificationQuestion:
        if self.kind == "single_choice" and not self.suggested_options:
            raise ValueError("Single-choice clarification needs suggested options.")
        if (
            self.default_recommendation is not None
            and self.suggested_options
            and self.default_recommendation not in self.suggested_options
        ):
            raise ValueError("Default recommendation must match a suggested option.")
        return self


class ClarificationRequest(BaseModel):
    """Persisted clarification metadata emitted before expensive generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reason: ClarificationReason
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1, max_length=280)
    original_query: str = Field(min_length=1, max_length=800)
    questions: tuple[ClarificationQuestion, ...] = Field(min_length=1, max_length=3)
    suggested_options: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    default_recommendation: str | None = Field(default=None, max_length=160)
    signal_summary: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


_CREATIVE_ACTION_SIGNAL = re.compile(
    r"\b(?:build|create|design|generate|make|compose|prototype|sketch)\b"
)
_EXPLICIT_RUNTIME_SIGNAL = re.compile(
    r"\b(?:p5(?:\.js)?|three(?:\.js)?|react\s+three\s+fiber|r3f|glsl|"
    r"fragment\s+shader|shader|webgpu|wgsl|hydra|tone(?:\.js)?|web\s+audio|"
    r"processing|canvas)\b"
)
_STYLE_RUNTIME_CONFLICT_SIGNAL = (
    (
        re.compile(r"\b(?:tone(?:\.js|\s+js)|web\s+audio|synth|audio)\b"),
        re.compile(r"\b(?:fragment\s+shader|glsl|shader|shadertoy)\b"),
    ),
    (
        re.compile(r"\bhydra\b"),
        re.compile(r"\b(?:three(?:\.js)?|react\s+three\s+fiber|r3f)\b"),
    ),
)
_MAX_CLARIFICATION_ORIGINAL_QUERY_CHARS = 800
_AUDIOVISUAL_BRIDGE_SIGNAL = re.compile(
    r"\b(?:audio[\s-]?reactive|audiovisual|sound[\s-]?reactive|visuali[sz]er)\b"
)
_COMPLEX_OUTPUT_SIGNAL = re.compile(
    r"\b(?:installation|projection|gallery|webgpu|wgsl|raymarch|3d|particles?|"
    r"multi[\s-]?screen|kiosk|dense|thousands|interactive wall)\b"
)
_PERFORMANCE_EXPORT_SIGNAL = re.compile(
    r"\b(?:60\s*fps|fps|mobile|desktop|browser|projection|wall|export|record|"
    r"loop|resolution|4k|1080p|performance|responsive)\b"
)
_MULTI_CANDIDATE_SIGNAL = re.compile(
    r"\b(?:multiple|several|many|compare|comparison|alternatives?|options?|"
    r"variations?|candidates?)\b"
)
_DIRECTION_SIGNAL = re.compile(
    r"\b(?:minimal|maximal|calm|energetic|dark|bright|monochrome|neon|organic|"
    r"geometric|sacred|mandala|shader|p5|three|glsl|audio[\s-]?reactive)\b"
)


def derive_hitl_clarification(
    *,
    query: str,
    route_decision: RouteDecision,
    creative_translation: CreativeTranslation,
    clarification_response: str | None = None,
    artifact_refinement: AssistantArtifactRefinement | None = None,
    cost_complexity_estimate: float | None = None,
) -> ClarificationRequest | None:
    """Return targeted clarification only when generation intent is underspecified."""

    if route_decision.route is not RouteName.GENERATE:
        return None
    if artifact_refinement is not None:
        return None
    if clarification_response and clarification_response.strip():
        return None

    normalized = _normalize_text(query)
    signals = _signal_summary(
        route_decision=route_decision,
        creative_translation=creative_translation,
        cost_complexity_estimate=cost_complexity_estimate,
    )

    if _has_style_runtime_conflict(normalized):
        return _clarification(
            reason=ClarificationReason.CONFLICTING_STYLE_RUNTIME,
            confidence=0.52,
            summary=(
                "The request mixes runtime/style cues that can lead to different "
                "implementations."
            ),
            query=query,
            questions=(
                ClarificationQuestion(
                    id="runtime_priority",
                    prompt="Which direction should drive the first generated artifact?",
                    suggested_options=(
                        "Prioritize the visual shader/runtime",
                        "Prioritize the audio engine",
                        "Build an audiovisual bridge",
                    ),
                    default_recommendation="Build an audiovisual bridge",
                ),
            ),
            signal_summary=signals,
        )

    if _has_high_cost_multi_candidate_gap(
        normalized,
        route_decision=route_decision,
        creative_translation=creative_translation,
        cost_complexity_estimate=cost_complexity_estimate,
    ):
        return _clarification(
            reason=ClarificationReason.HIGH_COST_MULTI_CANDIDATE,
            confidence=0.61,
            summary=(
                "The request asks for multiple candidates without enough direction "
                "to spend generation budget efficiently."
            ),
            query=query,
            questions=(
                ClarificationQuestion(
                    id="candidate_priority",
                    prompt="Which candidate direction should receive the first pass?",
                    suggested_options=(
                        "Prioritize one refined candidate",
                        "Generate three compact variations",
                        "Compare runtime approaches",
                    ),
                    default_recommendation="Prioritize one refined candidate",
                ),
                ClarificationQuestion(
                    id="selection_axis",
                    prompt="What should distinguish the candidates?",
                    suggested_options=(
                        "Visual style",
                        "Runtime technique",
                        "Performance profile",
                    ),
                    default_recommendation="Visual style",
                ),
            ),
            signal_summary=signals,
        )

    if _has_ambiguous_modality(
        normalized,
        route_decision=route_decision,
        creative_translation=creative_translation,
    ):
        return _clarification(
            reason=ClarificationReason.AMBIGUOUS_MODALITY,
            confidence=0.44,
            summary=(
                "The request has creative intent but does not make the output "
                "modality explicit."
            ),
            query=query,
            questions=(
                ClarificationQuestion(
                    id="output_modality",
                    prompt="What should the assistant generate first?",
                    suggested_options=(
                        "Visual sketch",
                        "Audio piece",
                        "Audiovisual piece",
                    ),
                    default_recommendation="Visual sketch",
                ),
            ),
            signal_summary=signals,
        )

    if _has_complex_target_gap(normalized):
        return _clarification(
            reason=ClarificationReason.MISSING_PERFORMANCE_EXPORT_TARGET,
            confidence=0.64,
            summary=(
                "The request appears complex, but the performance or export target "
                "is missing."
            ),
            query=query,
            questions=(
                ClarificationQuestion(
                    id="performance_export_target",
                    prompt="What should the implementation optimize for?",
                    suggested_options=(
                        "Browser preview at 60 fps",
                        "Projection-ready fullscreen output",
                        "Exportable looping recording",
                    ),
                    default_recommendation="Browser preview at 60 fps",
                ),
            ),
            signal_summary=signals,
        )

    if _has_unclear_output_target(
        normalized,
        route_decision=route_decision,
        creative_translation=creative_translation,
    ):
        return _clarification(
            reason=ClarificationReason.UNCLEAR_OUTPUT_TARGET,
            confidence=0.58,
            summary=(
                "The creative brief is understandable, but the runtime/output target "
                "is still unclear."
            ),
            query=query,
            questions=(
                ClarificationQuestion(
                    id="runtime_target",
                    prompt="Which output target should I use?",
                    suggested_options=(
                        "p5.js browser sketch",
                        "Three.js scene",
                        "GLSL fragment shader",
                    ),
                    default_recommendation="p5.js browser sketch",
                ),
            ),
            signal_summary=signals,
        )

    return None


def _clarification(
    *,
    reason: ClarificationReason,
    confidence: float,
    summary: str,
    query: str,
    questions: tuple[ClarificationQuestion, ...],
    signal_summary: tuple[str, ...],
) -> ClarificationRequest:
    first_question = questions[0]
    return ClarificationRequest(
        reason=reason,
        confidence=confidence,
        summary=summary,
        original_query=_bounded_original_query(query),
        questions=questions[:3],
        suggested_options=first_question.suggested_options,
        default_recommendation=first_question.default_recommendation,
        signal_summary=signal_summary,
    )


def _bounded_original_query(query: str) -> str:
    normalized = query.strip()
    if len(normalized) <= _MAX_CLARIFICATION_ORIGINAL_QUERY_CHARS:
        return normalized
    return (
        normalized[: _MAX_CLARIFICATION_ORIGINAL_QUERY_CHARS - 1].rstrip()
        + "…"
    )


def _has_ambiguous_modality(
    query: str,
    *,
    route_decision: RouteDecision,
    creative_translation: CreativeTranslation,
) -> bool:
    return (
        bool(_CREATIVE_ACTION_SIGNAL.search(query))
        and creative_translation.output_modality is None
        and not route_decision.domains
        and not creative_translation.runtime_recommendations
    )


def _has_unclear_output_target(
    query: str,
    *,
    route_decision: RouteDecision,
    creative_translation: CreativeTranslation,
) -> bool:
    return (
        bool(_CREATIVE_ACTION_SIGNAL.search(query))
        and bool(creative_translation.output_modality)
        and not route_decision.domains
        and not _EXPLICIT_RUNTIME_SIGNAL.search(query)
    )


def _has_style_runtime_conflict(query: str) -> bool:
    if _AUDIOVISUAL_BRIDGE_SIGNAL.search(query):
        return False
    return any(
        left.search(query) and right.search(query)
        for left, right in _STYLE_RUNTIME_CONFLICT_SIGNAL
    )


def _has_complex_target_gap(query: str) -> bool:
    return bool(_COMPLEX_OUTPUT_SIGNAL.search(query)) and not bool(
        _PERFORMANCE_EXPORT_SIGNAL.search(query)
    )


def _has_high_cost_multi_candidate_gap(
    query: str,
    *,
    route_decision: RouteDecision,
    creative_translation: CreativeTranslation,
    cost_complexity_estimate: float | None,
) -> bool:
    estimated_costly = (
        cost_complexity_estimate is not None and cost_complexity_estimate >= 0.75
    )
    lacks_direction = not (
        _DIRECTION_SIGNAL.search(query)
        or creative_translation.visual_style is not None
        or creative_translation.reference_fusion is not None
        or creative_translation.sacred_geometry is not None
    )
    multi_runtime = len(route_decision.domains) > 1
    return bool(_MULTI_CANDIDATE_SIGNAL.search(query)) and (
        lacks_direction or multi_runtime or estimated_costly
    )


def _signal_summary(
    *,
    route_decision: RouteDecision,
    creative_translation: CreativeTranslation,
    cost_complexity_estimate: float | None,
) -> tuple[str, ...]:
    signals: list[str] = [
        f"route={route_decision.route.value}",
        "domains="
        + (
            ",".join(domain.value for domain in route_decision.domains)
            if route_decision.domains
            else "none"
        ),
        "modality="
        + (
            creative_translation.output_modality.value
            if creative_translation.output_modality is not None
            else "unspecified"
        ),
    ]
    if creative_translation.runtime_recommendations:
        signals.append(
            "runtime=" + ",".join(creative_translation.runtime_recommendations[:3])
        )
    if creative_translation.visual_style is not None:
        signals.append("visual_style=present")
    if creative_translation.sacred_geometry is not None:
        signals.append("sacred_geometry=present")
    if creative_translation.audio_reactive is not None:
        signals.append("audio_reactive=present")
    if creative_translation.reference_fusion is not None:
        signals.append("reference_fusion=present")
    if cost_complexity_estimate is not None:
        signals.append(f"cost_complexity={cost_complexity_estimate:.2f}")
    return tuple(signals[:8])


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())
