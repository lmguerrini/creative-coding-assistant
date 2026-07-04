"""Metadata-only Consistency Validation Engine for V3.4 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import (
    PlanningMetadata,
    _clip,
    _dedupe,
)
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
from creative_coding_assistant.orchestration.creative_score_engine import (
    CreativeScoreProfile,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)

ConsistencyStatus = Literal[
    "consistent",
    "needs_attention",
    "inconsistent",
    "insufficient_evidence",
]
ConsistencyCheckStatus = Literal["aligned", "watch", "conflict", "missing"]
ContradictionLevel = Literal["none", "low", "medium", "high"]
AmbiguityLevel = Literal["low", "medium", "high"]
EvaluationIntegrity = Literal["strong", "adequate", "fragile", "compromised"]

CONSISTENCY_VALIDATION_ENGINE_AUTHORITY_BOUNDARY = (
    "The Consistency Validation Engine validates internal agreement across "
    "existing V3.4 evaluation metadata only; it does not modify outputs, "
    "execute artifacts, trigger refinement, trigger retries, change routing, "
    "select runtimes, alter previews, invoke future V4 agents, or perform "
    "future reporting, escalation, optimization, or learning behavior."
)


class ConsistencyValidationCheck(BaseModel):
    """One bounded consistency check across prior evaluation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    status: ConsistencyCheckStatus
    summary: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    conflict_signals: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class ConsistencyValidationProfile(BaseModel):
    """Inspectable validation of V3.4 evaluation metadata consistency."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["consistency_validation_engine"] = "consistency_validation_engine"
    serialization_version: Literal["v1"] = "v1"
    consistency_status: ConsistencyStatus
    consistency_summary: str = Field(min_length=1, max_length=720)
    detected_conflicts: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    score_consistency: ConsistencyValidationCheck
    confidence_consistency: ConsistencyValidationCheck
    reflection_consistency: ConsistencyValidationCheck
    critic_consistency: ConsistencyValidationCheck
    planner_consistency: ConsistencyValidationCheck
    reasoning_consistency: ConsistencyValidationCheck
    contradiction_level: ContradictionLevel
    ambiguity_level: AmbiguityLevel
    unsupported_conclusions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    evaluation_integrity: EvaluationIntegrity
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    hitl_recommendation: ExpectedHumanReviewNeed
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CONSISTENCY_VALIDATION_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1080,
    )


def derive_consistency_validation_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    planning_metadata: PlanningMetadata = (),
) -> ConsistencyValidationProfile:
    """Validate internal consistency across existing V3.4 evaluation metadata."""

    score_consistency = _score_consistency(
        creative_score=creative_score,
        creative_confidence=creative_confidence,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    confidence_consistency = _confidence_consistency(
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        reflection_loop=reflection_loop,
    )
    reflection_consistency = _reflection_consistency(
        reflection_loop=reflection_loop,
        creative_score=creative_score,
        creative_confidence=creative_confidence,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    critic_consistency = _critic_consistency(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    planner_consistency = _planner_consistency(
        creative_improvement_planner=creative_improvement_planner,
        creative_score=creative_score,
        creative_confidence=creative_confidence,
    )
    reasoning_consistency = _reasoning_consistency(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
    )
    checks = (
        score_consistency,
        confidence_consistency,
        reflection_consistency,
        critic_consistency,
        planner_consistency,
        reasoning_consistency,
    )
    detected_conflicts = _detected_conflicts(checks)
    unsupported = _unsupported_conclusions(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        creative_score=creative_score,
    )
    ambiguity = _ambiguity_level(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        detected_conflicts=detected_conflicts,
    )
    contradiction = _contradiction_level(detected_conflicts, checks)
    integrity = _evaluation_integrity(
        checks=checks,
        contradiction_level=contradiction,
        ambiguity_level=ambiguity,
        unsupported_conclusions=unsupported,
    )
    status = _consistency_status(
        checks=checks,
        contradiction_level=contradiction,
        evaluation_integrity=integrity,
    )
    hitl = _hitl_recommendation(
        status=status,
        contradiction_level=contradiction,
        ambiguity_level=ambiguity,
        evaluation_integrity=integrity,
        unsupported_conclusions=unsupported,
    )

    return ConsistencyValidationProfile(
        consistency_status=status,
        consistency_summary=_consistency_summary(
            status=status,
            contradiction_level=contradiction,
            ambiguity_level=ambiguity,
            evaluation_integrity=integrity,
            conflict_count=len(detected_conflicts),
            hitl=hitl,
        ),
        detected_conflicts=detected_conflicts,
        score_consistency=score_consistency,
        confidence_consistency=confidence_consistency,
        reflection_consistency=reflection_consistency,
        critic_consistency=critic_consistency,
        planner_consistency=planner_consistency,
        reasoning_consistency=reasoning_consistency,
        contradiction_level=contradiction,
        ambiguity_level=ambiguity,
        unsupported_conclusions=unsupported,
        evaluation_integrity=integrity,
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            creative_confidence=creative_confidence,
            creative_score=creative_score,
            planning_metadata=planning_metadata,
            checks=checks,
        ),
        hitl_recommendation=hitl,
        prompt_guidance=_prompt_guidance(
            status=status,
            contradiction_level=contradiction,
            ambiguity_level=ambiguity,
            hitl=hitl,
        ),
    )


def consistency_validation_prompt_lines(
    profile: ConsistencyValidationProfile,
) -> tuple[str, ...]:
    """Render consistency validation metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Consistency status: {profile.consistency_status}.",
        f"Consistency summary: {profile.consistency_summary}",
        f"Contradiction level: {profile.contradiction_level}.",
        f"Ambiguity level: {profile.ambiguity_level}.",
        f"Evaluation integrity: {profile.evaluation_integrity}.",
        f"HITL recommendation: {profile.hitl_recommendation}.",
        _check_line("Score consistency", profile.score_consistency),
        _check_line("Confidence consistency", profile.confidence_consistency),
        _check_line("Reflection consistency", profile.reflection_consistency),
        _check_line("Critic consistency", profile.critic_consistency),
        _check_line("Planner consistency", profile.planner_consistency),
        _check_line("Reasoning consistency", profile.reasoning_consistency),
    ]
    lines.extend(
        f"Detected consistency conflict: {item}" for item in profile.detected_conflicts
    )
    lines.extend(
        f"Unsupported consistency conclusion: {item}"
        for item in profile.unsupported_conclusions
    )
    lines.extend(f"Consistency evidence: {item}" for item in profile.evidence[:6])
    lines.extend(
        f"Consistency prompt guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:52])


def _score_consistency(
    *,
    creative_score: CreativeScoreProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> ConsistencyValidationCheck:
    if creative_score is None:
        return _missing_check("Creative Score metadata is not available.")

    conflicts: list[str] = []
    evidence = [
        (
            f"Score {creative_score.overall_creative_score:.1f}/100 "
            f"({creative_score.score_band})."
        )
    ]
    if creative_confidence is not None:
        confidence_score = creative_confidence.confidence_score * 100
        evidence.append(
            f"Confidence {confidence_score:.1f}/100 ({creative_confidence.confidence_level})."
        )
        if (
            creative_score.overall_creative_score >= 75
            and creative_confidence.confidence_score < 0.55
        ):
            conflicts.append("High score conflicts with low confidence.")
        if (
            creative_score.overall_creative_score < 55
            and creative_confidence.confidence_score >= 0.8
        ):
            conflicts.append("Low score conflicts with high confidence.")
        if abs(creative_score.overall_creative_score - confidence_score) > 35:
            conflicts.append("Score and confidence differ by more than 35 points.")
    if creative_critic is not None:
        evidence.append(f"Critic risk is {creative_critic.risk_assessment}.")
        if (
            creative_score.overall_creative_score >= 75
            and creative_critic.risk_assessment in {"high", "blocked"}
        ):
            conflicts.append("Strong score conflicts with high critic risk.")
    if self_evaluation is not None:
        evidence.append(
            f"Self evaluation is {self_evaluation.completeness_assessment}."
        )
        if (
            creative_score.overall_creative_score >= 75
            and self_evaluation.completeness_assessment in {"partial", "missing"}
        ):
            conflicts.append("Strong score conflicts with incomplete self evaluation.")

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=creative_score.hitl_recommendation in {"optional", "recommended"},
        aligned_summary="Creative Score is directionally aligned with confidence and risk metadata.",
        watch_summary="Creative Score is usable but has review-sensitive calibration signals.",
        conflict_summary="Creative Score conflicts with one or more evaluation signals.",
        evidence=evidence,
    )


def _confidence_consistency(
    *,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> ConsistencyValidationCheck:
    if creative_confidence is None:
        return _missing_check("Creative Confidence metadata is not available.")

    conflicts: list[str] = []
    evidence = [
        (
            f"Confidence {creative_confidence.confidence_score:.2f} "
            f"({creative_confidence.confidence_level})."
        )
    ]
    if creative_confidence.confidence_trend == "conflicting":
        conflicts.append("Creative Confidence reports conflicting trend.")
    if creative_score is not None:
        evidence.append(
            f"Score band {creative_score.score_band}; {creative_score.overall_creative_score:.1f}/100."
        )
        if creative_confidence.confidence_level in {
            "very_high",
            "high",
        } and creative_score.score_band in {"weak", "critical"}:
            conflicts.append("High confidence conflicts with weak score band.")
    if reflection_loop is not None:
        evidence.append(
            f"Reflection priority {reflection_loop.reflection_priority}; required {reflection_loop.reflection_required}."  # noqa: E501
        )
        if creative_confidence.confidence_level in {
            "very_high",
            "high",
        } and reflection_loop.reflection_priority in {"critical", "high"}:
            conflicts.append("High confidence conflicts with high reflection pressure.")

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=bool(creative_confidence.confidence_uncertainties),
        aligned_summary="Creative Confidence agrees with score and reflection metadata.",
        watch_summary="Creative Confidence is plausible but carries explicit uncertainties.",
        conflict_summary="Creative Confidence conflicts with score or reflection metadata.",
        evidence=evidence,
    )


def _reflection_consistency(
    *,
    reflection_loop: ReflectionLoopProfile | None,
    creative_score: CreativeScoreProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> ConsistencyValidationCheck:
    if reflection_loop is None:
        return _missing_check("Reflection Loop metadata is not available.")

    conflicts: list[str] = []
    evidence = [
        (
            f"Reflection {reflection_loop.reflection_priority}; "
            f"required {reflection_loop.reflection_required}."
        )
    ]
    if creative_score is not None:
        evidence.append(f"Score band {creative_score.score_band}.")
        if (
            reflection_loop.reflection_required
            and creative_score.score_band in {"excellent", "strong"}
            and reflection_loop.reflection_priority in {"critical", "high"}
        ):
            conflicts.append("High reflection requirement conflicts with strong score.")
        if not reflection_loop.reflection_required and creative_score.score_band in {
            "weak",
            "critical",
        }:
            conflicts.append("No reflection requirement conflicts with weak score.")
    if creative_confidence is not None:
        evidence.append(f"Confidence level {creative_confidence.confidence_level}.")
        if (
            reflection_loop.reflection_priority in {"critical", "high"}
            and creative_confidence.hitl_recommendation == "not_needed"
        ):
            conflicts.append(
                "High reflection pressure conflicts with no HITL confidence posture."
            )
    if creative_critic is not None and creative_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        evidence.append(f"Critic risk {creative_critic.risk_assessment}.")
        if not reflection_loop.reflection_required:
            conflicts.append(
                "High critic risk conflicts with no reflection requirement."
            )
    if self_evaluation is not None and self_evaluation.underdelivery_risk == "high":
        evidence.append("Self evaluation underdelivery risk is high.")
        if not reflection_loop.reflection_required:
            conflicts.append(
                "High underdelivery risk conflicts with no reflection requirement."
            )

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=bool(reflection_loop.unresolved_questions),
        aligned_summary="Reflection Loop agrees with risk, score, and confidence metadata.",
        watch_summary="Reflection Loop is plausible but includes unresolved questions.",
        conflict_summary="Reflection Loop conflicts with risk, score, or confidence metadata.",
        evidence=evidence,
    )


def _critic_consistency(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> ConsistencyValidationCheck:
    if creative_critic is None:
        return _missing_check("Creative Critic metadata is not available.")

    conflicts: list[str] = []
    critic_average = _critic_quality_average(creative_critic)
    evidence = [
        (
            f"Critic average {critic_average:.2f}; "
            f"risk {creative_critic.risk_assessment}."
        )
    ]
    if critic_average >= 0.78 and creative_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        conflicts.append("High critic quality conflicts with high critic risk.")
    if critic_average < 0.55 and creative_critic.risk_assessment == "low":
        conflicts.append("Low critic quality conflicts with low critic risk.")
    if self_evaluation is not None:
        self_average = _self_evaluation_average(self_evaluation)
        evidence.append(f"Self-evaluation average {self_average:.2f}.")
        if abs(critic_average - self_average) > 0.32:
            conflicts.append(
                "Critic and self-evaluation quality averages diverge sharply."
            )

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=bool(
            creative_critic.unsupported_assumptions
            or creative_critic.missing_information
        ),
        aligned_summary="Creative Critic agrees with self-evaluation and its own risk posture.",
        watch_summary="Creative Critic is plausible but includes missing or unsupported signals.",
        conflict_summary="Creative Critic conflicts with quality or self-evaluation metadata.",
        evidence=evidence,
    )


def _planner_consistency(
    *,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    creative_score: CreativeScoreProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
) -> ConsistencyValidationCheck:
    if creative_improvement_planner is None:
        return _missing_check("Creative Improvement Planner metadata is not available.")

    priority_count = len(creative_improvement_planner.improvement_priorities)
    opportunity_count = len(creative_improvement_planner.highest_impact_opportunities)
    conflicts: list[str] = []
    evidence = [
        (
            f"Planner has {priority_count} priorities and "
            f"{opportunity_count} high-impact opportunities."
        )
    ]
    if creative_score is not None:
        evidence.append(f"Score band {creative_score.score_band}.")
        if creative_score.score_band in {"excellent", "strong"} and priority_count >= 5:
            conflicts.append("Many improvement priorities conflict with strong score.")
        if creative_score.score_band in {"weak", "critical"} and priority_count == 0:
            conflicts.append("Weak score conflicts with absent improvement priorities.")
    if creative_confidence is not None:
        evidence.append(f"Confidence level {creative_confidence.confidence_level}.")
        if (
            creative_confidence.confidence_level in {"very_high", "high"}
            and opportunity_count >= 4
        ):
            conflicts.append(
                "Many high-impact opportunities conflict with high confidence."
            )

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=bool(creative_improvement_planner.hitl_questions),
        aligned_summary="Improvement Planner pressure agrees with score and confidence metadata.",
        watch_summary="Improvement Planner is plausible but includes HITL-sensitive opportunities.",
        conflict_summary="Improvement Planner pressure conflicts with score or confidence metadata.",
        evidence=evidence,
    )


def _reasoning_consistency(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> ConsistencyValidationCheck:
    unsupported = _unsupported_conclusions(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        creative_score=creative_score,
    )
    missing_evidence_count = sum(
        item is None
        for item in (
            creative_critic,
            self_evaluation,
            creative_improvement_planner,
            reflection_loop,
            creative_confidence,
            creative_score,
        )
    )
    evidence = [
        f"Unsupported conclusions: {len(unsupported)}.",
        f"Missing evaluation inputs: {missing_evidence_count}.",
    ]
    conflicts = []
    if (
        unsupported
        and creative_score is not None
        and creative_score.score_band
        in {
            "excellent",
            "strong",
        }
    ):
        conflicts.append("Strong score is paired with unsupported conclusions.")
    if missing_evidence_count >= 3:
        conflicts.append("Reasoning support is missing multiple evaluation inputs.")

    return _check_from_conflicts(
        conflicts=conflicts,
        watch=bool(unsupported) or missing_evidence_count > 0,
        aligned_summary="Evaluation conclusions are supported by available metadata.",
        watch_summary="Evaluation conclusions are usable but have unsupported or missing evidence.",
        conflict_summary="Evaluation conclusions are not fully supported by available metadata.",
        evidence=evidence,
    )


def _missing_check(summary: str) -> ConsistencyValidationCheck:
    return ConsistencyValidationCheck(
        status="missing",
        summary=summary,
        evidence=("Required source metadata was absent.",),
        conflict_signals=(),
    )


def _check_from_conflicts(
    *,
    conflicts: Sequence[str],
    watch: bool,
    aligned_summary: str,
    watch_summary: str,
    conflict_summary: str,
    evidence: Sequence[str],
) -> ConsistencyValidationCheck:
    if conflicts:
        return ConsistencyValidationCheck(
            status="conflict",
            summary=conflict_summary,
            evidence=_dedupe(evidence)[:4],
            conflict_signals=_dedupe(conflicts)[:4],
        )
    if watch:
        return ConsistencyValidationCheck(
            status="watch",
            summary=watch_summary,
            evidence=_dedupe(evidence)[:4],
            conflict_signals=(),
        )
    return ConsistencyValidationCheck(
        status="aligned",
        summary=aligned_summary,
        evidence=_dedupe(evidence)[:4],
        conflict_signals=(),
    )


def _detected_conflicts(
    checks: tuple[ConsistencyValidationCheck, ...],
) -> tuple[str, ...]:
    conflicts: list[str] = []
    for check in checks:
        conflicts.extend(check.conflict_signals)
    return _dedupe(conflicts, clip_limit=320)[:10]


def _contradiction_level(
    detected_conflicts: tuple[str, ...],
    checks: tuple[ConsistencyValidationCheck, ...],
) -> ContradictionLevel:
    conflict_count = len(detected_conflicts)
    missing_count = sum(1 for check in checks if check.status == "missing")
    if conflict_count >= 4:
        return "high"
    if conflict_count >= 2:
        return "medium"
    if conflict_count == 1 or missing_count >= 2:
        return "low"
    return "none"


def _ambiguity_level(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    detected_conflicts: tuple[str, ...],
) -> AmbiguityLevel:
    signal_count = len(detected_conflicts)
    if creative_critic is not None:
        signal_count += len(creative_critic.missing_information)
        signal_count += len(creative_critic.hitl_questions)
    if self_evaluation is not None:
        signal_count += len(self_evaluation.missing_information)
        signal_count += len(self_evaluation.hitl_questions)
        if self_evaluation.ambiguity_assessment == "high":
            signal_count += 3
        elif self_evaluation.ambiguity_assessment == "medium":
            signal_count += 1
    if creative_improvement_planner is not None:
        signal_count += len(creative_improvement_planner.hitl_questions)
    if reflection_loop is not None:
        signal_count += len(reflection_loop.unresolved_questions)
    if creative_confidence is not None:
        signal_count += len(creative_confidence.confidence_uncertainties)
    if creative_score is not None:
        signal_count += len(creative_score.negative_contributions)
    if signal_count >= 7:
        return "high"
    if signal_count >= 3:
        return "medium"
    return "low"


def _evaluation_integrity(
    *,
    checks: tuple[ConsistencyValidationCheck, ...],
    contradiction_level: ContradictionLevel,
    ambiguity_level: AmbiguityLevel,
    unsupported_conclusions: tuple[str, ...],
) -> EvaluationIntegrity:
    conflict_count = sum(1 for check in checks if check.status == "conflict")
    missing_count = sum(1 for check in checks if check.status == "missing")
    if contradiction_level == "high" or conflict_count >= 4:
        return "compromised"
    if (
        contradiction_level == "medium"
        or ambiguity_level == "high"
        or len(unsupported_conclusions) >= 4
        or missing_count >= 2
    ):
        return "fragile"
    if conflict_count or ambiguity_level == "medium" or unsupported_conclusions:
        return "adequate"
    return "strong"


def _consistency_status(
    *,
    checks: tuple[ConsistencyValidationCheck, ...],
    contradiction_level: ContradictionLevel,
    evaluation_integrity: EvaluationIntegrity,
) -> ConsistencyStatus:
    if all(check.status == "missing" for check in checks):
        return "insufficient_evidence"
    if contradiction_level == "high" or evaluation_integrity == "compromised":
        return "inconsistent"
    if (
        contradiction_level in {"low", "medium"}
        or evaluation_integrity in {"adequate", "fragile"}
        or any(check.status in {"watch", "missing"} for check in checks)
    ):
        return "needs_attention"
    return "consistent"


def _hitl_recommendation(
    *,
    status: ConsistencyStatus,
    contradiction_level: ContradictionLevel,
    ambiguity_level: AmbiguityLevel,
    evaluation_integrity: EvaluationIntegrity,
    unsupported_conclusions: tuple[str, ...],
) -> ExpectedHumanReviewNeed:
    if (
        status == "inconsistent"
        or contradiction_level == "high"
        or evaluation_integrity == "compromised"
    ):
        return "required"
    if (
        contradiction_level == "medium"
        or ambiguity_level == "high"
        or evaluation_integrity == "fragile"
        or len(unsupported_conclusions) >= 3
    ):
        return "recommended"
    if status == "needs_attention" or ambiguity_level == "medium":
        return "optional"
    return "not_needed"


def _unsupported_conclusions(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> tuple[str, ...]:
    unsupported: list[str] = []
    if creative_critic is not None:
        unsupported.extend(creative_critic.unsupported_assumptions)
    if self_evaluation is not None:
        unsupported.extend(self_evaluation.unsupported_assumptions)
    if creative_score is not None and creative_score.score_band in {
        "excellent",
        "strong",
    }:
        unsupported.extend(
            item
            for item in creative_score.negative_contributions
            if "unsupported" in item.lower()
        )
    return _dedupe(unsupported, clip_limit=320)[:8]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    planning_metadata: PlanningMetadata,
    checks: tuple[ConsistencyValidationCheck, ...],
) -> tuple[str, ...]:
    evidence = [f"Request inspected for consistency: {_clip(request.query, 120)}"]
    if route_decision is not None:
        evidence.append(
            f"Route {route_decision.route.value}; {len(route_decision.domains)} domain(s)."
        )
    if creative_critic is not None:
        evidence.append(
            f"Creative Critic risk {creative_critic.risk_assessment}; confidence {creative_critic.critic_confidence:.2f}."  # noqa: E501
        )
    if self_evaluation is not None:
        evidence.append(
            f"Self Evaluation {self_evaluation.completeness_assessment}; ambiguity {self_evaluation.ambiguity_assessment}."  # noqa: E501
        )
    if creative_improvement_planner is not None:
        evidence.append(
            f"Improvement Planner {len(creative_improvement_planner.improvement_priorities)} priority signal(s)."
        )
    if reflection_loop is not None:
        evidence.append(
            f"Reflection Loop {reflection_loop.reflection_priority}; required {reflection_loop.reflection_required}."
        )
    if creative_confidence is not None:
        evidence.append(
            f"Creative Confidence {creative_confidence.confidence_level}; {creative_confidence.confidence_score:.2f}."
        )
    if creative_score is not None:
        evidence.append(
            f"Creative Score {creative_score.score_band}; {creative_score.overall_creative_score:.1f}/100."
        )
    evidence.append(f"Planning metadata objects: {len(planning_metadata)}.")
    evidence.append(
        "Validation checks: "
        + ", ".join(f"{index + 1}:{check.status}" for index, check in enumerate(checks))
        + "."
    )
    evidence.append(
        "Authority boundary verified: consistency validation is metadata-only."
    )
    return _dedupe(evidence, clip_limit=320)[:16]


def _prompt_guidance(
    *,
    status: ConsistencyStatus,
    contradiction_level: ContradictionLevel,
    ambiguity_level: AmbiguityLevel,
    hitl: ExpectedHumanReviewNeed,
) -> tuple[str, ...]:
    guidance = [
        (
            "Use Consistency Validation metadata as advisory evaluation "
            "integrity context only."
        ),
        (
            "Do not modify outputs, execute artifacts, retry, refine, route, "
            "select runtime, alter previews, or invoke future agents from "
            "consistency validation metadata."
        ),
        (
            f"Treat consistency status {status}, contradiction level "
            f"{contradiction_level}, and ambiguity level {ambiguity_level} as "
            "review guidance."
        ),
    ]
    if hitl in {"optional", "recommended", "required"}:
        guidance.append(
            f"Surface {hitl} human review when consistency metadata affects trust."
        )
    return tuple(guidance[:8])


def _consistency_summary(
    *,
    status: ConsistencyStatus,
    contradiction_level: ContradictionLevel,
    ambiguity_level: AmbiguityLevel,
    evaluation_integrity: EvaluationIntegrity,
    conflict_count: int,
    hitl: ExpectedHumanReviewNeed,
) -> str:
    return _clip(
        (
            "Consistency Validation Engine found "
            f"{status} metadata with {conflict_count} conflict signal(s), "
            f"{contradiction_level} contradiction level, {ambiguity_level} "
            f"ambiguity, {evaluation_integrity} evaluation integrity, and "
            f"{hitl} HITL recommendation. This validation is advisory "
            "metadata only and cannot change execution behavior."
        ),
        720,
    )


def _check_line(label: str, check: ConsistencyValidationCheck) -> str:
    return f"{label}: {check.status}; {check.summary}"


def _critic_quality_average(profile: CreativeCriticProfile) -> float:
    values = (
        profile.concept_quality,
        profile.execution_quality,
        profile.artifact_quality,
        profile.coherence_quality,
        profile.runtime_fit_quality,
        profile.originality_quality,
        profile.clarity_quality,
        profile.feasibility_quality,
    )
    return round(sum(values) / len(values), 2)


def _self_evaluation_average(profile: SelfEvaluationProfile) -> float:
    values = (
        profile.request_alignment,
        profile.intent_alignment,
        profile.constraint_alignment,
        profile.artifact_alignment,
        profile.runtime_alignment,
        profile.creative_coherence,
        profile.technical_coherence,
    )
    return round(sum(values) / len(values), 2)
