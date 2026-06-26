"""Metadata-only Creative Score Engine for V3.4 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    CreativeConfidenceProfile,
    ExpectedHumanReviewNeed,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)

ScoreDimension = Literal[
    "creativity",
    "technical",
    "coherence",
    "feasibility",
    "artifact",
    "runtime",
]
ScoreBand = Literal["excellent", "strong", "solid", "weak", "critical"]
ScoreSignalSource = Literal[
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "planning_metadata",
]

CREATIVE_SCORE_ENGINE_AUTHORITY_BOUNDARY = (
    "The Creative Score Engine scores, ranks, and summarizes existing "
    "evaluation metadata only; it does not modify outputs, regenerate "
    "responses, execute artifacts, trigger refinement, trigger retries, change "
    "routing, select runtimes, alter previews, invoke future V4 agents, or "
    "perform V5 execution optimization."
)


class CreativeScoreBreakdownItem(BaseModel):
    """One weighted score dimension used in the overall score."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    dimension: ScoreDimension
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    rationale: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class CreativeScoreComponent(BaseModel):
    """One source-level contribution to score calibration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: ScoreSignalSource
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    weighted_contribution: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class CreativeScoreProfile(BaseModel):
    """Inspectable aggregate score derived from evaluation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_score_engine"] = "creative_score_engine"
    serialization_version: Literal["v1"] = "v1"
    overall_creative_score: float = Field(ge=0, le=100)
    score_band: ScoreBand
    score_summary: str = Field(min_length=1, max_length=720)
    score_breakdown: tuple[CreativeScoreBreakdownItem, ...] = Field(
        min_length=6,
        max_length=6,
    )
    score_components: tuple[CreativeScoreComponent, ...] = Field(
        min_length=1,
        max_length=8,
    )
    creativity_score: float = Field(ge=0, le=100)
    technical_score: float = Field(ge=0, le=100)
    coherence_score: float = Field(ge=0, le=100)
    feasibility_score: float = Field(ge=0, le=100)
    artifact_score: float = Field(ge=0, le=100)
    runtime_score: float = Field(ge=0, le=100)
    confidence_weight: float = Field(ge=0, le=1)
    reflection_weight: float = Field(ge=0, le=1)
    consistency_weight: float = Field(ge=0, le=1)
    artifact_weight: float = Field(ge=0, le=1)
    runtime_weight: float = Field(ge=0, le=1)
    uncertainty_penalty: float = Field(ge=0, le=30)
    risk_penalty: float = Field(ge=0, le=30)
    positive_contributions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    negative_contributions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    strengths: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    score_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    score_calibration_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    score_explainability: str = Field(min_length=1, max_length=720)
    score_evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    hitl_recommendation: ExpectedHumanReviewNeed
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CREATIVE_SCORE_ENGINE_AUTHORITY_BOUNDARY,
        max_length=920,
    )


def derive_creative_score_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object] = (),
) -> CreativeScoreProfile:
    """Aggregate evaluation metadata into deterministic score metadata."""

    raw_scores = _dimension_scores(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        planning_metadata=planning_metadata,
    )
    breakdown = _score_breakdown(
        raw_scores=raw_scores,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        planning_metadata=planning_metadata,
    )
    confidence_weight = _confidence_weight(creative_confidence)
    uncertainty_penalty = _uncertainty_penalty(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        planning_metadata=planning_metadata,
    )
    risk_penalty = _risk_penalty(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
    )
    weighted_dimension_score = sum(
        item.score * item.weight for item in breakdown
    ) / sum(item.weight for item in breakdown)
    score_components = _score_components(
        weighted_dimension_score=weighted_dimension_score,
        planning_metadata=planning_metadata,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
    )
    overall = _bounded_percent(
        weighted_dimension_score * confidence_weight
        - uncertainty_penalty
        - risk_penalty
    )
    band = _score_band(overall)
    hitl = _hitl_recommendation(
        overall_score=overall,
        uncertainty_penalty=uncertainty_penalty,
        risk_penalty=risk_penalty,
        creative_confidence=creative_confidence,
        self_evaluation=self_evaluation,
    )

    return CreativeScoreProfile(
        overall_creative_score=overall,
        score_band=band,
        score_summary=_score_summary(
            overall_score=overall,
            score_band=band,
            confidence_weight=confidence_weight,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
            hitl=hitl,
        ),
        score_breakdown=breakdown,
        score_components=score_components,
        creativity_score=raw_scores["creativity"],
        technical_score=raw_scores["technical"],
        coherence_score=raw_scores["coherence"],
        feasibility_score=raw_scores["feasibility"],
        artifact_score=raw_scores["artifact"],
        runtime_score=raw_scores["runtime"],
        confidence_weight=confidence_weight,
        reflection_weight=_reflection_weight(reflection_loop),
        consistency_weight=_dimension_weight(breakdown, "coherence"),
        artifact_weight=_dimension_weight(breakdown, "artifact"),
        runtime_weight=_dimension_weight(breakdown, "runtime"),
        uncertainty_penalty=uncertainty_penalty,
        risk_penalty=risk_penalty,
        positive_contributions=_positive_contributions(
            breakdown=breakdown,
            score_components=score_components,
            confidence_weight=confidence_weight,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
        ),
        negative_contributions=_negative_contributions(
            breakdown=breakdown,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
            creative_confidence=creative_confidence,
        ),
        strengths=_score_strengths(
            breakdown=breakdown,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_confidence=creative_confidence,
        ),
        weaknesses=_score_weaknesses(
            breakdown=breakdown,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
        ),
        score_rationale=_score_rationale(
            breakdown=breakdown,
            confidence_weight=confidence_weight,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
        ),
        score_calibration_notes=_score_calibration_notes(
            weighted_dimension_score=weighted_dimension_score,
            confidence_weight=confidence_weight,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
            overall_score=overall,
        ),
        score_explainability=_score_explainability(
            weighted_dimension_score=weighted_dimension_score,
            confidence_weight=confidence_weight,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
            overall_score=overall,
        ),
        score_evidence=_score_evidence(
            request=request,
            route_decision=route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            planning_metadata=planning_metadata,
        ),
        hitl_recommendation=hitl,
        prompt_guidance=_prompt_guidance(
            score_band=band,
            hitl=hitl,
            uncertainty_penalty=uncertainty_penalty,
            risk_penalty=risk_penalty,
        ),
    )


def creative_score_prompt_lines(profile: CreativeScoreProfile) -> tuple[str, ...]:
    """Render score metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Overall creative score: {profile.overall_creative_score:.1f}/100.",
        f"Score band: {profile.score_band}.",
        f"Score summary: {profile.score_summary}",
        f"Confidence weight: {profile.confidence_weight:.2f}.",
        f"Reflection weight: {profile.reflection_weight:.2f}.",
        f"Consistency weight: {profile.consistency_weight:.2f}.",
        f"Artifact weight: {profile.artifact_weight:.2f}.",
        f"Runtime weight: {profile.runtime_weight:.2f}.",
        f"Uncertainty penalty: {profile.uncertainty_penalty:.1f}.",
        f"Risk penalty: {profile.risk_penalty:.1f}.",
        f"Score explainability: {profile.score_explainability}",
        f"HITL recommendation: {profile.hitl_recommendation}.",
    ]
    lines.extend(
        (
            "Score breakdown: "
            f"{item.dimension}; {item.score:.1f}/100; "
            f"{item.weight:.2f} weight; {item.rationale}"
        )
        for item in profile.score_breakdown
    )
    lines.extend(
        (
            "Score component: "
            f"{item.source}; {item.score:.1f}/100; "
            f"{item.weight:.2f} weight; "
            f"{item.weighted_contribution:.1f} contribution; {item.rationale}"
        )
        for item in profile.score_components
    )
    lines.extend(
        f"Positive score contribution: {item}"
        for item in profile.positive_contributions
    )
    lines.extend(
        f"Negative score contribution: {item}"
        for item in profile.negative_contributions
    )
    lines.extend(
        f"Score calibration note: {item}"
        for item in profile.score_calibration_notes
    )
    lines.extend(f"Score strength: {item}" for item in profile.strengths)
    lines.extend(f"Score weakness: {item}" for item in profile.weaknesses)
    lines.extend(f"Score rationale: {item}" for item in profile.score_rationale)
    lines.extend(f"Score prompt guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:64])


def _dimension_scores(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object],
) -> dict[ScoreDimension, float]:
    planning_score = _planning_metadata_score(planning_metadata)
    confidence_score = (
        creative_confidence.confidence_score * 100
        if creative_confidence is not None
        else planning_score
    )
    scores: dict[ScoreDimension, float] = {
        "creativity": _weighted_average(
            (
                _critic_average(
                    creative_critic,
                    "concept_quality",
                    "originality_quality",
                    "coherence_quality",
                ),
                _metadata_average(planning_metadata, "confidence"),
                confidence_score,
            ),
            fallback=planning_score,
        ),
        "technical": _weighted_average(
            (
                _critic_average(
                    creative_critic,
                    "execution_quality",
                    "feasibility_quality",
                    "runtime_fit_quality",
                ),
                _self_average(
                    self_evaluation,
                    "technical_coherence",
                    "runtime_alignment",
                    "constraint_alignment",
                ),
                planning_score,
            ),
            fallback=planning_score,
        ),
        "coherence": _weighted_average(
            (
                _critic_average(creative_critic, "coherence_quality", "clarity_quality"),
                _self_average(
                    self_evaluation,
                    "request_alignment",
                    "intent_alignment",
                    "creative_coherence",
                ),
                confidence_score,
            ),
            fallback=planning_score,
        ),
        "feasibility": _weighted_average(
            (
                _critic_average(
                    creative_critic,
                    "feasibility_quality",
                    "execution_quality",
                    "runtime_fit_quality",
                ),
                _self_average(
                    self_evaluation,
                    "constraint_alignment",
                    "runtime_alignment",
                    "technical_coherence",
                ),
                _improvement_readiness_score(creative_improvement_planner),
                _reflection_readiness_score(reflection_loop),
            ),
            fallback=planning_score,
        ),
        "artifact": _weighted_average(
            (
                _critic_average(
                    creative_critic,
                    "artifact_quality",
                    "concept_quality",
                    "execution_quality",
                ),
                _self_average(
                    self_evaluation,
                    "artifact_alignment",
                    "request_alignment",
                    "creative_coherence",
                ),
                _metadata_average(
                    planning_metadata,
                    "critique_confidence",
                    "refinement_confidence",
                    "synthesis_confidence",
                    "merge_confidence",
                    "export_confidence",
                ),
            ),
            fallback=planning_score,
        ),
        "runtime": _weighted_average(
            (
                _critic_average(
                    creative_critic,
                    "runtime_fit_quality",
                    "feasibility_quality",
                    "execution_quality",
                ),
                _self_average(
                    self_evaluation,
                    "runtime_alignment",
                    "technical_coherence",
                ),
                _metadata_average(
                    planning_metadata,
                    "readiness_score",
                    "capability_confidence",
                    "confidence",
                ),
            ),
            fallback=planning_score,
        ),
    }
    return {
        dimension: _bounded_percent(score)
        for dimension, score in scores.items()
    }


def _score_breakdown(
    *,
    raw_scores: dict[ScoreDimension, float],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object],
) -> tuple[CreativeScoreBreakdownItem, ...]:
    return (
        CreativeScoreBreakdownItem(
            dimension="creativity",
            score=raw_scores["creativity"],
            weight=0.18,
            rationale="Scores concept quality, originality, coherence, and creative confidence.",
            evidence=_dimension_evidence(
                "creative",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
            ),
        ),
        CreativeScoreBreakdownItem(
            dimension="technical",
            score=raw_scores["technical"],
            weight=0.17,
            rationale="Scores execution quality, technical coherence, constraints, and planning readiness.",
            evidence=_dimension_evidence(
                "technical",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
            ),
        ),
        CreativeScoreBreakdownItem(
            dimension="coherence",
            score=raw_scores["coherence"],
            weight=0.18,
            rationale="Scores request alignment, intent alignment, clarity, and creative coherence.",
            evidence=_dimension_evidence(
                "coherence",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
            ),
        ),
        CreativeScoreBreakdownItem(
            dimension="feasibility",
            score=raw_scores["feasibility"],
            weight=0.17,
            rationale="Scores feasibility, execution readiness, reflection pressure, and improvement pressure.",
            evidence=_dimension_evidence(
                "feasibility",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
                creative_improvement_planner,
                reflection_loop,
            ),
        ),
        CreativeScoreBreakdownItem(
            dimension="artifact",
            score=raw_scores["artifact"],
            weight=0.16,
            rationale="Scores artifact alignment, artifact quality, and artifact-planning metadata confidence.",
            evidence=_dimension_evidence(
                "artifact",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
            ),
        ),
        CreativeScoreBreakdownItem(
            dimension="runtime",
            score=raw_scores["runtime"],
            weight=0.14,
            rationale="Scores runtime fit, runtime alignment, and runtime-facing planning readiness.",
            evidence=_dimension_evidence(
                "runtime",
                creative_critic,
                self_evaluation,
                creative_confidence,
                planning_metadata,
            ),
        ),
    )


def _score_components(
    *,
    weighted_dimension_score: float,
    planning_metadata: Sequence[object],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
) -> tuple[CreativeScoreComponent, ...]:
    raw_components: list[
        tuple[ScoreSignalSource, float, float, str, tuple[str, ...]]
    ] = []
    if creative_critic is not None:
        raw_components.append(
            (
                "creative_critic",
                _critic_average(
                    creative_critic,
                    "concept_quality",
                    "execution_quality",
                    "artifact_quality",
                    "coherence_quality",
                    "runtime_fit_quality",
                    "originality_quality",
                    "clarity_quality",
                    "feasibility_quality",
                )
                or weighted_dimension_score,
                0.28,
                "Creative Critic contributes quality, risk, and feasibility signals.",
                (creative_critic.critique_summary,),
            )
        )
    if self_evaluation is not None:
        raw_components.append(
            (
                "self_evaluation",
                _self_average(
                    self_evaluation,
                    "request_alignment",
                    "intent_alignment",
                    "constraint_alignment",
                    "artifact_alignment",
                    "runtime_alignment",
                    "creative_coherence",
                    "technical_coherence",
                )
                or weighted_dimension_score,
                0.26,
                "Self Evaluation contributes alignment, coherence, and completeness signals.",
                (self_evaluation.evaluation_summary,),
            )
        )
    if creative_confidence is not None:
        raw_components.append(
            (
                "creative_confidence",
                creative_confidence.confidence_score * 100,
                0.16,
                "Creative Confidence calibrates score reliability and uncertainty.",
                (creative_confidence.confidence_summary,),
            )
        )
    if reflection_loop is not None:
        raw_components.append(
            (
                "reflection_loop",
                _reflection_readiness_score(reflection_loop)
                or weighted_dimension_score,
                0.10,
                "Reflection Loop contributes advisory reflection pressure.",
                (reflection_loop.reflection_summary,),
            )
        )
    if creative_improvement_planner is not None:
        raw_components.append(
            (
                "creative_improvement_planner",
                _improvement_readiness_score(creative_improvement_planner)
                or weighted_dimension_score,
                0.10,
                "Creative Improvement Planner contributes improvement pressure.",
                (creative_improvement_planner.improvement_summary,),
            )
        )
    if planning_metadata:
        raw_components.append(
            (
                "planning_metadata",
                _planning_metadata_score(planning_metadata),
                0.10,
                "Planning metadata contributes upstream readiness and confidence signals.",
                (f"{len(planning_metadata)} planning metadata object(s) included.",),
            )
        )
    if not raw_components:
        raw_components.append(
            (
                "planning_metadata",
                weighted_dimension_score,
                1.0,
                "Fallback composition uses weighted dimension score.",
                ("No source-specific evaluation metadata was available.",),
            )
        )

    total_weight = sum(item[2] for item in raw_components)
    return tuple(
        CreativeScoreComponent(
            source=source,
            score=_bounded_percent(score),
            weight=round(weight / total_weight, 3),
            weighted_contribution=round(
                _bounded_percent(score) * (weight / total_weight),
                1,
            ),
            rationale=rationale,
            evidence=evidence,
        )
        for source, score, weight, rationale, evidence in raw_components
    )


def _reflection_weight(profile: ReflectionLoopProfile | None) -> float:
    return 0.10 if profile is not None else 0.0


def _dimension_weight(
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    dimension: ScoreDimension,
) -> float:
    for item in breakdown:
        if item.dimension == dimension:
            return item.weight
    return 0.0


def _positive_contributions(
    *,
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    score_components: tuple[CreativeScoreComponent, ...],
    confidence_weight: float,
    uncertainty_penalty: float,
    risk_penalty: float,
) -> tuple[str, ...]:
    contributions: list[str] = [
        (
            f"{item.source} contributes {item.weighted_contribution:.1f} "
            f"weighted points from a {item.score:.1f}/100 source score."
        )
        for item in sorted(
            score_components,
            key=lambda value: value.weighted_contribution,
            reverse=True,
        )[:3]
    ]
    contributions.extend(
        (
            f"{item.dimension} dimension supports the score at "
            f"{item.score:.1f}/100 with {item.weight:.2f} weight."
        )
        for item in sorted(breakdown, key=lambda value: value.score, reverse=True)[:2]
        if item.score >= 75
    )
    if confidence_weight >= 0.95:
        contributions.append(
            f"Confidence weight preserves most of the base score at {confidence_weight:.2f}."
        )
    if uncertainty_penalty == 0:
        contributions.append("No uncertainty penalty was applied.")
    if risk_penalty == 0:
        contributions.append("No risk penalty was applied.")
    return tuple(_dedupe(contributions)[:8])


def _negative_contributions(
    *,
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    uncertainty_penalty: float,
    risk_penalty: float,
    creative_confidence: CreativeConfidenceProfile | None,
) -> tuple[str, ...]:
    contributions: list[str] = []
    contributions.extend(
        (
            f"{item.dimension} dimension constrains the score at "
            f"{item.score:.1f}/100."
        )
        for item in sorted(breakdown, key=lambda value: value.score)[:2]
        if item.score < 75
    )
    if uncertainty_penalty > 0:
        contributions.append(
            f"Uncertainty penalty subtracts {uncertainty_penalty:.1f} points."
        )
    if risk_penalty > 0:
        contributions.append(f"Risk penalty subtracts {risk_penalty:.1f} points.")
    if creative_confidence is not None and creative_confidence.confidence_uncertainties:
        contributions.extend(creative_confidence.confidence_uncertainties[:2])
    return tuple(_dedupe(contributions)[:8])


def _score_calibration_notes(
    *,
    weighted_dimension_score: float,
    confidence_weight: float,
    uncertainty_penalty: float,
    risk_penalty: float,
    overall_score: float,
) -> tuple[str, ...]:
    return (
        f"Weighted dimension base score: {weighted_dimension_score:.1f}/100.",
        f"Confidence weight multiplies base score by {confidence_weight:.2f}.",
        f"Uncertainty penalty subtracts {uncertainty_penalty:.1f} points.",
        f"Risk penalty subtracts {risk_penalty:.1f} points.",
        f"Final bounded score after calibration: {overall_score:.1f}/100.",
        "Calibration is metadata-only and cannot trigger execution changes.",
    )


def _score_explainability(
    *,
    weighted_dimension_score: float,
    confidence_weight: float,
    uncertainty_penalty: float,
    risk_penalty: float,
    overall_score: float,
) -> str:
    return _clip(
        (
            "Final score = bounded("
            f"{weighted_dimension_score:.1f} weighted dimension base * "
            f"{confidence_weight:.2f} confidence weight - "
            f"{uncertainty_penalty:.1f} uncertainty penalty - "
            f"{risk_penalty:.1f} risk penalty) = "
            f"{overall_score:.1f}/100."
        ),
        720,
    )


def _confidence_weight(profile: CreativeConfidenceProfile | None) -> float:
    if profile is None:
        return 0.9
    return round(0.82 + (profile.confidence_score * 0.18), 3)


def _uncertainty_penalty(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object],
) -> float:
    count = 0
    if creative_critic is not None:
        count += len(creative_critic.missing_information)
        count += len(creative_critic.unsupported_assumptions)
    if self_evaluation is not None:
        count += len(self_evaluation.missing_information)
        count += len(self_evaluation.unsupported_assumptions)
        if self_evaluation.ambiguity_assessment == "medium":
            count += 1
        elif self_evaluation.ambiguity_assessment == "high":
            count += 3
    if creative_improvement_planner is not None:
        count += len(creative_improvement_planner.hitl_questions)
    if reflection_loop is not None:
        count += len(reflection_loop.unresolved_questions)
    if creative_confidence is not None:
        count += len(creative_confidence.confidence_uncertainties)
        if creative_confidence.confidence_trend in {"conflicting", "unknown"}:
            count += 2
    for item in planning_metadata:
        count += len(_metadata_values(item, "hitl_questions")[:2])
        count += len(_metadata_values(item, "missing_information")[:2])
    return round(min(30.0, count * 1.2), 1)


def _risk_penalty(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
) -> float:
    penalty = 0.0
    if creative_critic is not None:
        penalty += {
            "low": 0.0,
            "medium": 3.0,
            "high": 7.0,
            "blocked": 14.0,
        }[creative_critic.risk_assessment]
    if self_evaluation is not None:
        penalty += {
            "complete": 0.0,
            "mostly_complete": 1.5,
            "partial": 5.0,
            "blocked": 12.0,
        }[self_evaluation.completeness_assessment]
        penalty += {"low": 0.0, "medium": 3.0, "high": 7.0}[
            self_evaluation.hallucination_risk
        ]
        penalty += {"low": 0.0, "medium": 2.0, "high": 5.0}[
            self_evaluation.underdelivery_risk
        ]
    if creative_improvement_planner is not None:
        for item in creative_improvement_planner.improvement_priorities[:4]:
            penalty += {
                "critical": 2.5,
                "high": 1.5,
                "medium": 0.7,
                "low": 0.2,
            }[item.priority]
            if item.risk == "high":
                penalty += 1.0
    if reflection_loop is not None:
        penalty += {
            "none": 0.0,
            "low": 0.5,
            "medium": 1.5,
            "high": 4.0,
            "critical": 7.0,
        }[reflection_loop.reflection_priority]
    if creative_confidence is not None:
        penalty += {
            "very_high": 0.0,
            "high": 0.5,
            "medium": 2.0,
            "low": 5.0,
            "critical": 9.0,
        }[creative_confidence.confidence_level]
    return round(min(30.0, penalty), 1)


def _hitl_recommendation(
    *,
    overall_score: float,
    uncertainty_penalty: float,
    risk_penalty: float,
    creative_confidence: CreativeConfidenceProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> ExpectedHumanReviewNeed:
    if (
        overall_score < 45
        or risk_penalty >= 16
        or (
            self_evaluation is not None
            and self_evaluation.completeness_assessment == "blocked"
        )
    ):
        return "required"
    if (
        overall_score < 62
        or risk_penalty >= 10
        or uncertainty_penalty >= 12
        or (
            creative_confidence is not None
            and creative_confidence.hitl_recommendation == "required"
        )
    ):
        return "recommended"
    if (
        overall_score < 78
        or uncertainty_penalty >= 6
        or (
            creative_confidence is not None
            and creative_confidence.hitl_recommendation == "recommended"
        )
    ):
        return "optional"
    return "not_needed"


def _score_band(score: float) -> ScoreBand:
    if score >= 85:
        return "excellent"
    if score >= 75:
        return "strong"
    if score >= 60:
        return "solid"
    if score >= 40:
        return "weak"
    return "critical"


def _score_summary(
    *,
    overall_score: float,
    score_band: ScoreBand,
    confidence_weight: float,
    uncertainty_penalty: float,
    risk_penalty: float,
    hitl: ExpectedHumanReviewNeed,
) -> str:
    return _clip(
        (
            "Creative Score Engine synthesized evaluation and planning "
            f"metadata into a {score_band} score ({overall_score:.1f}/100) "
            f"with confidence weight {confidence_weight:.2f}, uncertainty "
            f"penalty {uncertainty_penalty:.1f}, risk penalty {risk_penalty:.1f}, "
            "and explicit source composition. "
            f"HITL recommendation is {hitl}. This score is advisory metadata "
            "only and does not change outputs, retries, refinement, routing, "
            "runtime selection, previews, or future agent behavior."
        ),
        720,
    )


def _score_strengths(
    *,
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
) -> tuple[str, ...]:
    strengths: list[str] = [
        f"{item.dimension} score is {item.score:.1f}/100."
        for item in sorted(breakdown, key=lambda value: value.score, reverse=True)[:2]
        if item.score >= 70
    ]
    if creative_critic is not None:
        strengths.extend(creative_critic.creative_strengths[:2])
    if self_evaluation is not None and self_evaluation.completeness_assessment in {
        "complete",
        "mostly_complete",
    }:
        strengths.append(
            f"Self evaluation reports {self_evaluation.completeness_assessment} completeness."
        )
    if creative_confidence is not None and creative_confidence.confidence_strengths:
        strengths.extend(creative_confidence.confidence_strengths[:2])
    return tuple(_dedupe(strengths)[:8])


def _score_weaknesses(
    *,
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
) -> tuple[str, ...]:
    weaknesses: list[str] = [
        f"{item.dimension} score is {item.score:.1f}/100."
        for item in sorted(breakdown, key=lambda value: value.score)[:2]
        if item.score < 72
    ]
    if creative_critic is not None:
        weaknesses.extend(creative_critic.creative_weaknesses[:2])
        weaknesses.extend(creative_critic.unsupported_assumptions[:1])
    if self_evaluation is not None:
        weaknesses.extend(self_evaluation.quality_gaps[:2])
        weaknesses.extend(self_evaluation.missing_information[:1])
    if creative_improvement_planner is not None:
        weaknesses.extend(
            item.title for item in creative_improvement_planner.improvement_priorities[:2]
        )
    if reflection_loop is not None and reflection_loop.reflection_required:
        weaknesses.extend(reflection_loop.refinement_candidates[:2])
    if creative_confidence is not None:
        weaknesses.extend(creative_confidence.confidence_weaknesses[:2])
    return tuple(_dedupe(weaknesses)[:8])


def _score_rationale(
    *,
    breakdown: tuple[CreativeScoreBreakdownItem, ...],
    confidence_weight: float,
    uncertainty_penalty: float,
    risk_penalty: float,
) -> tuple[str, ...]:
    top = max(breakdown, key=lambda item: item.score)
    low = min(breakdown, key=lambda item: item.score)
    return (
        (
            f"{top.dimension} is the strongest dimension at "
            f"{top.score:.1f}/100."
        ),
        (
            f"{low.dimension} is the lowest dimension at "
            f"{low.score:.1f}/100."
        ),
        f"Confidence weight applied as {confidence_weight:.2f}.",
        f"Uncertainty penalty applied as {uncertainty_penalty:.1f}.",
        f"Risk penalty applied as {risk_penalty:.1f}.",
        "Score remains advisory metadata and cannot trigger execution changes.",
    )


def _score_evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object],
) -> tuple[str, ...]:
    evidence = [
        f"Request scored: {request.query[:160]}",
        (
            "Route scope: "
            f"{route_decision.route.value if route_decision is not None else 'unknown'}."
        ),
    ]
    if creative_critic is not None:
        evidence.append(
            "Creative Critic: "
            f"{creative_critic.risk_assessment} risk; "
            f"{creative_critic.critic_confidence:.2f} confidence."
        )
    if self_evaluation is not None:
        evidence.append(
            "Self Evaluation: "
            f"{self_evaluation.completeness_assessment}; "
            f"{self_evaluation.self_evaluation_confidence:.2f} confidence."
        )
    if creative_improvement_planner is not None:
        evidence.append(
            "Creative Improvement Planner: "
            f"{len(creative_improvement_planner.improvement_priorities)} priorities; "
            f"{creative_improvement_planner.confidence:.2f} confidence."
        )
    if reflection_loop is not None:
        evidence.append(
            "Reflection Loop: "
            f"{reflection_loop.reflection_priority} priority; "
            f"{reflection_loop.reflection_depth} depth."
        )
    if creative_confidence is not None:
        evidence.append(
            "Creative Confidence: "
            f"{creative_confidence.confidence_level}; "
            f"{creative_confidence.confidence_score:.2f} score."
        )
    if planning_metadata:
        evidence.append(
            f"Planning metadata components scored: {len(planning_metadata)}."
        )
    evidence.append("Authority boundary verified: score is metadata-only.")
    return tuple(_dedupe(evidence)[:16])


def _prompt_guidance(
    *,
    score_band: ScoreBand,
    hitl: ExpectedHumanReviewNeed,
    uncertainty_penalty: float,
    risk_penalty: float,
) -> tuple[str, ...]:
    guidance = [
        (
            "Use Creative Score metadata as advisory scoring context only; do "
            "not modify outputs, execute artifacts, trigger refinement, "
            "trigger retries, route providers, select runtimes, alter previews, "
            "invoke V4 agents, or perform V5 optimization."
        ),
        f"Treat the score band as {score_band} and preserve score rationale when explaining evaluation confidence.",
    ]
    if hitl in {"recommended", "required"}:
        guidance.append(
            f"Surface {hitl} human review need before treating the score as settled."
        )
    if uncertainty_penalty >= 8:
        guidance.append(
            "Call out uncertainty penalty when presenting score limitations."
        )
    if risk_penalty >= 8:
        guidance.append("Call out risk penalty when presenting score limitations.")
    return tuple(guidance[:8])


def _dimension_evidence(
    label: str,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    planning_metadata: Sequence[object],
    creative_improvement_planner: CreativeImprovementPlannerProfile | None = None,
    reflection_loop: ReflectionLoopProfile | None = None,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if creative_critic is not None:
        evidence.append(f"Creative Critic supports {label} scoring.")
    if self_evaluation is not None:
        evidence.append(f"Self Evaluation supports {label} scoring.")
    if creative_confidence is not None:
        evidence.append(f"Creative Confidence supports {label} scoring.")
    if creative_improvement_planner is not None:
        evidence.append("Creative Improvement Planner pressure is included.")
    if reflection_loop is not None:
        evidence.append("Reflection Loop advisory pressure is included.")
    if planning_metadata:
        evidence.append(f"{len(planning_metadata)} planning metadata object(s) included.")
    return tuple(_dedupe(evidence)[:4])


def _critic_average(profile: CreativeCriticProfile | None, *fields: str) -> float | None:
    if profile is None:
        return None
    values = [
        _normalize_percent(getattr(profile, field, None))
        for field in fields
        if _normalize_percent(getattr(profile, field, None)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _self_average(profile: SelfEvaluationProfile | None, *fields: str) -> float | None:
    if profile is None:
        return None
    values = [
        _normalize_percent(getattr(profile, field, None))
        for field in fields
        if _normalize_percent(getattr(profile, field, None)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _metadata_average(metadata: Sequence[object], *fields: str) -> float | None:
    values: list[float] = []
    for item in metadata:
        for field in fields:
            value = _normalize_percent(getattr(item, field, None))
            if value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _planning_metadata_score(planning_metadata: Sequence[object]) -> float:
    return _metadata_average(
        planning_metadata,
        "confidence",
        "hierarchy_confidence",
        "readiness_score",
        "synthesis_confidence",
        "merge_confidence",
        "export_confidence",
        "refinement_confidence",
        "critique_confidence",
        "capability_confidence",
    ) or 58.0


def _improvement_readiness_score(
    profile: CreativeImprovementPlannerProfile | None,
) -> float | None:
    if profile is None:
        return None
    penalty = 0.0
    for item in profile.improvement_priorities[:4]:
        penalty += {
            "critical": 7.0,
            "high": 4.0,
            "medium": 2.0,
            "low": 0.5,
        }[item.priority]
    if profile.hitl_questions:
        penalty += 4.0
    return _bounded_percent(profile.confidence * 100 - penalty)


def _reflection_readiness_score(profile: ReflectionLoopProfile | None) -> float | None:
    if profile is None:
        return None
    penalty = {
        "none": 0.0,
        "low": 2.0,
        "medium": 5.0,
        "high": 10.0,
        "critical": 16.0,
    }[profile.reflection_priority]
    if profile.reflection_required:
        penalty += 4.0
    return _bounded_percent(profile.confidence_after_reflection * 100 - penalty)


def _weighted_average(
    values: Sequence[float | None],
    *,
    fallback: float,
) -> float:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return fallback
    return sum(clean_values) / len(clean_values)


def _normalize_percent(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    normalized = float(value) * 100 if value <= 1 else float(value)
    return _bounded_percent(normalized)


def _bounded_percent(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _metadata_values(item: object, field: str) -> tuple[str, ...]:
    value = getattr(item, field, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(entry) for entry in value if str(entry))
    return ()
