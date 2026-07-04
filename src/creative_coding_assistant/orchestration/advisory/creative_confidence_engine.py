"""Metadata-only Creative Confidence Engine for V3.4 evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import (
    PlanningMetadata,
    _clip,
    _dedupe,
    _metadata_label,
    _metadata_values,
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

ConfidenceLevel = Literal["very_high", "high", "medium", "low", "critical"]
ConfidenceComponentSource = Literal[
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "planning_metadata",
]
ExpectedOutputReliability = Literal[
    "very_high",
    "high",
    "medium",
    "low",
    "blocked",
]
ExpectedExecutionReadiness = Literal[
    "ready",
    "needs_caveats",
    "needs_hitl",
    "blocked",
]
ExpectedHumanReviewNeed = Literal[
    "not_needed",
    "optional",
    "recommended",
    "required",
]
EscalationRecommendation = Literal[
    "none",
    "monitor",
    "hitl_review",
    "future_escalation",
]
ConfidenceTrend = Literal[
    "improving",
    "stable",
    "declining",
    "conflicting",
    "unknown",
]

CREATIVE_CONFIDENCE_ENGINE_AUTHORITY_BOUNDARY = (
    "The Creative Confidence Engine estimates confidence and uncertainty from "
    "existing evaluation metadata only; it does not change outputs, modify "
    "artifacts, trigger refinement, trigger retries, change routing, select "
    "runtimes, call providers, alter previews, invoke future V4 agents, or "
    "perform future scoring, reporting, runtime routing, cost optimization, "
    "or V5 execution behavior."
)


class CreativeConfidenceComponent(BaseModel):
    """One weighted confidence component with source evidence."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: ConfidenceComponentSource
    score: float = Field(ge=0, le=1)
    weight: float = Field(gt=0, le=1)
    rationale: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class CreativeConfidenceProfile(BaseModel):
    """Inspectable confidence assessment across evaluation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_confidence_engine"] = "creative_confidence_engine"
    serialization_version: Literal["v1"] = "v1"
    confidence_score: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    confidence_summary: str = Field(min_length=1, max_length=720)
    confidence_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    confidence_components: tuple[CreativeConfidenceComponent, ...] = Field(
        min_length=1,
        max_length=8,
    )
    confidence_limitations: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    confidence_uncertainties: tuple[str, ...] = Field(
        default_factory=tuple, max_length=8
    )
    confidence_strengths: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    confidence_weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    expected_output_reliability: ExpectedOutputReliability
    expected_execution_readiness: ExpectedExecutionReadiness
    expected_human_review_need: ExpectedHumanReviewNeed
    escalation_recommendation: EscalationRecommendation
    confidence_trend: ConfidenceTrend
    confidence_evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    hitl_recommendation: ExpectedHumanReviewNeed
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CREATIVE_CONFIDENCE_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1080,
    )


def derive_creative_confidence_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    planning_metadata: PlanningMetadata = (),
) -> CreativeConfidenceProfile:
    """Aggregate evaluation metadata into an advisory confidence profile."""

    components = _confidence_components(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        planning_metadata=planning_metadata,
    )
    score = _weighted_score(components)
    level = _confidence_level(
        confidence_score=score,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        reflection_loop=reflection_loop,
    )
    uncertainties = _confidence_uncertainties(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        planning_metadata=planning_metadata,
    )
    weaknesses = _confidence_weaknesses(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
    )
    hitl = _human_review_need(
        level=level,
        uncertainties=uncertainties,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        reflection_loop=reflection_loop,
    )

    return CreativeConfidenceProfile(
        confidence_score=score,
        confidence_level=level,
        confidence_summary=_confidence_summary(
            confidence_score=score,
            level=level,
            hitl=hitl,
            component_count=len(components),
        ),
        confidence_rationale=_confidence_rationale(
            components=components,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
        ),
        confidence_components=components,
        confidence_limitations=_confidence_limitations(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            reflection_loop=reflection_loop,
            planning_metadata=planning_metadata,
        ),
        confidence_uncertainties=uncertainties,
        confidence_strengths=_confidence_strengths(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
        ),
        confidence_weaknesses=weaknesses,
        expected_output_reliability=_expected_output_reliability(level),
        expected_execution_readiness=_expected_execution_readiness(
            level=level,
            hitl=hitl,
            reflection_loop=reflection_loop,
        ),
        expected_human_review_need=hitl,
        escalation_recommendation=_escalation_recommendation(
            level=level,
            hitl=hitl,
            reflection_loop=reflection_loop,
        ),
        confidence_trend=_confidence_trend(
            components=components,
            reflection_loop=reflection_loop,
        ),
        confidence_evidence=_confidence_evidence(
            request=request,
            route_decision=route_decision,
            components=components,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            reflection_loop=reflection_loop,
            planning_metadata=planning_metadata,
        ),
        hitl_recommendation=hitl,
        prompt_guidance=_prompt_guidance(level=level, hitl=hitl),
    )


def creative_confidence_prompt_lines(
    profile: CreativeConfidenceProfile,
) -> tuple[str, ...]:
    """Render confidence metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Confidence score: {profile.confidence_score:.2f}.",
        f"Confidence level: {profile.confidence_level}.",
        f"Confidence summary: {profile.confidence_summary}",
        f"Expected output reliability: {profile.expected_output_reliability}.",
        f"Expected execution readiness: {profile.expected_execution_readiness}.",
        f"Expected human review need: {profile.expected_human_review_need}.",
        f"Escalation recommendation: {profile.escalation_recommendation}.",
        f"Confidence trend: {profile.confidence_trend}.",
        f"HITL recommendation: {profile.hitl_recommendation}.",
    ]
    lines.extend(
        (
            "Confidence component: "
            f"{item.source}; {item.score:.2f} score; {item.weight:.2f} weight; "
            f"{item.rationale}"
        )
        for item in profile.confidence_components
    )
    lines.extend(
        f"Confidence rationale: {item}" for item in profile.confidence_rationale
    )
    lines.extend(
        f"Confidence limitation: {item}" for item in profile.confidence_limitations
    )
    lines.extend(
        f"Confidence uncertainty: {item}" for item in profile.confidence_uncertainties
    )
    lines.extend(
        f"Confidence strength: {item}" for item in profile.confidence_strengths
    )
    lines.extend(
        f"Confidence weakness: {item}" for item in profile.confidence_weaknesses
    )
    lines.extend(
        f"Confidence prompt guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:64])


def _confidence_components(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    planning_metadata: PlanningMetadata,
) -> tuple[CreativeConfidenceComponent, ...]:
    components: list[CreativeConfidenceComponent] = []
    if creative_critic is not None:
        components.append(
            CreativeConfidenceComponent(
                source="creative_critic",
                score=_critic_score(creative_critic),
                weight=0.26,
                rationale=(
                    "Creative Critic contributes quality, feasibility, runtime "
                    "fit, and risk assessment."
                ),
                evidence=(creative_critic.critique_summary,),
            )
        )
    if self_evaluation is not None:
        components.append(
            CreativeConfidenceComponent(
                source="self_evaluation",
                score=_self_evaluation_score(self_evaluation),
                weight=0.3,
                rationale=(
                    "Self Evaluation contributes request alignment, coherence, "
                    "completeness, and risk signals."
                ),
                evidence=(self_evaluation.evaluation_summary,),
            )
        )
    if creative_improvement_planner is not None:
        components.append(
            CreativeConfidenceComponent(
                source="creative_improvement_planner",
                score=_improvement_score(creative_improvement_planner),
                weight=0.18,
                rationale=(
                    "Creative Improvement Planner contributes priority, "
                    "opportunity, and HITL pressure."
                ),
                evidence=(creative_improvement_planner.improvement_summary,),
            )
        )
    if reflection_loop is not None:
        components.append(
            CreativeConfidenceComponent(
                source="reflection_loop",
                score=_reflection_score(reflection_loop),
                weight=0.18,
                rationale=(
                    "Reflection Loop contributes theoretical improvement need, "
                    "risk reduction, and post-reflection confidence."
                ),
                evidence=(reflection_loop.reflection_summary,),
            )
        )
    if planning_metadata:
        components.append(
            CreativeConfidenceComponent(
                source="planning_metadata",
                score=_planning_metadata_score(planning_metadata),
                weight=0.08,
                rationale=(
                    "Planning metadata contributes supporting confidence and "
                    "unresolved-signal density."
                ),
                evidence=(_planning_metadata_evidence(planning_metadata),),
            )
        )
    return tuple(components) or (
        CreativeConfidenceComponent(
            source="planning_metadata",
            score=0.35,
            weight=1.0,
            rationale="Only request-level context is available for confidence.",
            evidence=("No evaluation metadata was available.",),
        ),
    )


def _critic_score(profile: CreativeCriticProfile) -> float:
    quality_values = (
        profile.concept_quality,
        profile.execution_quality,
        profile.artifact_quality,
        profile.coherence_quality,
        profile.runtime_fit_quality,
        profile.originality_quality,
        profile.clarity_quality,
        profile.feasibility_quality,
    )
    score = sum(quality_values) / len(quality_values)
    score -= {"low": 0.0, "medium": 0.08, "high": 0.18, "blocked": 0.35}[
        profile.risk_assessment
    ]
    score -= min(0.12, len(profile.unsupported_assumptions) * 0.03)
    return _bounded_score(score)


def _self_evaluation_score(profile: SelfEvaluationProfile) -> float:
    alignment_values = (
        profile.request_alignment,
        profile.intent_alignment,
        profile.constraint_alignment,
        profile.artifact_alignment,
        profile.runtime_alignment,
        profile.creative_coherence,
        profile.technical_coherence,
    )
    score = sum(alignment_values) / len(alignment_values)
    score -= {
        "complete": 0.0,
        "mostly_complete": 0.04,
        "partial": 0.16,
        "blocked": 0.34,
    }[profile.completeness_assessment]
    score -= {"low": 0.0, "medium": 0.05, "high": 0.12}[profile.ambiguity_assessment]
    score -= {"low": 0.0, "medium": 0.06, "high": 0.14}[profile.hallucination_risk]
    score -= {"low": 0.0, "medium": 0.04, "high": 0.1}[profile.underdelivery_risk]
    return _bounded_score(score)


def _improvement_score(profile: CreativeImprovementPlannerProfile) -> float:
    penalty = 0.0
    for item in profile.improvement_priorities[:4]:
        penalty += {
            "critical": 0.08,
            "high": 0.05,
            "medium": 0.03,
            "low": 0.01,
        }[item.priority]
        if item.risk == "high":
            penalty += 0.04
    if profile.hitl_questions:
        penalty += 0.05
    return _bounded_score(profile.confidence - min(0.28, penalty))


def _reflection_score(profile: ReflectionLoopProfile) -> float:
    score = profile.confidence_after_reflection
    score -= {
        "none": 0.0,
        "low": 0.03,
        "medium": 0.08,
        "high": 0.14,
        "critical": 0.22,
    }[profile.reflection_priority]
    if profile.reflection_required:
        score -= 0.06
    if profile.hitl_recommendation in {"recommended", "required"}:
        score -= 0.05
    return _bounded_score(score)


def _planning_metadata_score(planning_metadata: PlanningMetadata) -> float:
    values: list[float] = []
    unresolved_count = 0
    for item in planning_metadata:
        for attribute in (
            "confidence",
            "hierarchy_confidence",
            "readiness_score",
            "synthesis_confidence",
            "merge_confidence",
            "export_confidence",
            "refinement_confidence",
            "critique_confidence",
        ):
            value = getattr(item, attribute, None)
            if isinstance(value, int | float):
                normalized = float(value) / 100 if value > 1 else float(value)
                values.append(max(0.0, min(1.0, normalized)))
        unresolved_count += len(_metadata_values(item, "hitl_questions")[:2])
        unresolved_count += len(_metadata_values(item, "missing_information")[:2])
    if not values:
        base = 0.58
    else:
        base = sum(values) / len(values)
    return _bounded_score(base - min(0.18, unresolved_count * 0.015))


def _weighted_score(components: tuple[CreativeConfidenceComponent, ...]) -> float:
    total_weight = sum(item.weight for item in components)
    if total_weight <= 0:
        return 0.0
    return round(sum(item.score * item.weight for item in components) / total_weight, 2)


def _confidence_level(
    *,
    confidence_score: float,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> ConfidenceLevel:
    if (
        confidence_score < 0.35
        or (
            creative_critic is not None and creative_critic.risk_assessment == "blocked"
        )
        or (
            self_evaluation is not None
            and self_evaluation.completeness_assessment == "blocked"
        )
    ):
        return "critical"
    if confidence_score < 0.55:
        return "low"
    if confidence_score < 0.72 or (
        reflection_loop is not None
        and reflection_loop.reflection_priority in {"high", "critical"}
    ):
        return "medium"
    if confidence_score < 0.86:
        return "high"
    return "very_high"


def _confidence_summary(
    *,
    confidence_score: float,
    level: ConfidenceLevel,
    hitl: ExpectedHumanReviewNeed,
    component_count: int,
) -> str:
    return _clip(
        (
            "Creative Confidence Engine aggregated "
            f"{component_count} metadata component(s) into {level} confidence "
            f"({confidence_score:.2f}). Expected human review need is {hitl}. "
            "This confidence assessment is advisory metadata only and does not "
            "change outputs, routing, retries, or refinement behavior."
        ),
        720,
    )


def _confidence_rationale(
    *,
    components: tuple[CreativeConfidenceComponent, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> tuple[str, ...]:
    rationale = [
        f"{len(components)} confidence component(s) were weighted deterministically.",
    ]
    if creative_critic is not None:
        rationale.append(
            f"Creative Critic risk is {creative_critic.risk_assessment} with {creative_critic.critic_confidence:.2f} confidence."  # noqa: E501
        )
    if self_evaluation is not None:
        rationale.append(
            f"Self Evaluation completeness is {self_evaluation.completeness_assessment} with {self_evaluation.self_evaluation_confidence:.2f} confidence."  # noqa: E501
        )
    if creative_improvement_planner is not None:
        rationale.append(
            "Creative Improvement Planner contributes "
            f"{len(creative_improvement_planner.improvement_priorities)} priority signal(s)."
        )
    if reflection_loop is not None:
        rationale.append(
            "Reflection Loop reports "
            f"{reflection_loop.reflection_priority} priority and "
            f"{reflection_loop.expected_quality_gain} expected quality gain."
        )
    rationale.append("Confidence remains metadata-only advisory guidance.")
    return _dedupe(rationale)[:8]


def _confidence_limitations(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    planning_metadata: PlanningMetadata,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if creative_critic is None:
        limitations.append("Creative Critic metadata is unavailable.")
    if self_evaluation is None:
        limitations.append("Self Evaluation metadata is unavailable.")
    if reflection_loop is None:
        limitations.append("Reflection Loop metadata is unavailable.")
    if not planning_metadata:
        limitations.append("Planning metadata context is limited.")
    limitations.append(
        "Confidence is deterministic metadata and not an executed quality proof."
    )
    return _dedupe(limitations)[:8]


def _confidence_uncertainties(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    planning_metadata: PlanningMetadata,
) -> tuple[str, ...]:
    uncertainties: list[str] = []
    if creative_critic is not None:
        uncertainties.extend(creative_critic.missing_information[:2])
        uncertainties.extend(creative_critic.unsupported_assumptions[:2])
        uncertainties.extend(creative_critic.hitl_questions[:2])
    if self_evaluation is not None:
        uncertainties.extend(self_evaluation.missing_information[:2])
        uncertainties.extend(self_evaluation.unsupported_assumptions[:2])
        uncertainties.extend(self_evaluation.hitl_questions[:2])
    if creative_improvement_planner is not None:
        uncertainties.extend(creative_improvement_planner.hitl_questions[:2])
    if reflection_loop is not None:
        uncertainties.extend(reflection_loop.unresolved_questions[:3])
    for item in planning_metadata:
        uncertainties.extend(_metadata_values(item, "hitl_questions")[:1])
        uncertainties.extend(_metadata_values(item, "missing_information")[:1])
    return _dedupe(uncertainties)[:8]


def _confidence_strengths(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> tuple[str, ...]:
    strengths: list[str] = []
    if creative_critic is not None:
        strengths.extend(creative_critic.creative_strengths[:3])
    if self_evaluation is not None:
        strengths.append(
            f"Request alignment {self_evaluation.request_alignment:.2f}; creative coherence {self_evaluation.creative_coherence:.2f}."  # noqa: E501
        )
    if creative_improvement_planner is not None:
        strengths.extend(creative_improvement_planner.low_risk_improvements[:2])
    if reflection_loop is not None and not reflection_loop.reflection_required:
        strengths.append("Reflection Loop does not require additional refinement.")
    return _dedupe(strengths)[:8]


def _confidence_weaknesses(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    if creative_critic is not None:
        weaknesses.extend(creative_critic.creative_weaknesses[:3])
    if self_evaluation is not None:
        weaknesses.extend(self_evaluation.quality_gaps[:3])
    if creative_improvement_planner is not None:
        weaknesses.extend(
            item.title
            for item in creative_improvement_planner.improvement_priorities[:3]
            if item.priority in {"critical", "high"}
        )
    if reflection_loop is not None and reflection_loop.reflection_required:
        weaknesses.extend(reflection_loop.refinement_candidates[:2])
    return _dedupe(weaknesses)[:8]


def _expected_output_reliability(level: ConfidenceLevel) -> ExpectedOutputReliability:
    return {
        "very_high": "very_high",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "critical": "blocked",
    }[level]


def _expected_execution_readiness(
    *,
    level: ConfidenceLevel,
    hitl: ExpectedHumanReviewNeed,
    reflection_loop: ReflectionLoopProfile | None,
) -> ExpectedExecutionReadiness:
    if level == "critical":
        return "blocked"
    if hitl == "required":
        return "needs_hitl"
    if level in {"low", "medium"} or (
        reflection_loop is not None and reflection_loop.reflection_required
    ):
        return "needs_caveats"
    return "ready"


def _human_review_need(
    *,
    level: ConfidenceLevel,
    uncertainties: tuple[str, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> ExpectedHumanReviewNeed:
    if level == "critical":
        return "required"
    if (
        level == "low"
        or (
            creative_critic is not None
            and creative_critic.risk_assessment in {"high", "blocked"}
        )
        or (
            self_evaluation is not None and self_evaluation.hallucination_risk == "high"
        )
        or (
            reflection_loop is not None
            and reflection_loop.hitl_recommendation == "required"
        )
    ):
        return "required"
    if (
        level == "medium"
        or len(uncertainties) >= 3
        or (
            reflection_loop is not None
            and reflection_loop.hitl_recommendation == "recommended"
        )
    ):
        return "recommended"
    if uncertainties:
        return "optional"
    return "not_needed"


def _escalation_recommendation(
    *,
    level: ConfidenceLevel,
    hitl: ExpectedHumanReviewNeed,
    reflection_loop: ReflectionLoopProfile | None,
) -> EscalationRecommendation:
    if level == "critical" or hitl == "required":
        return "hitl_review"
    if level == "low" or (
        reflection_loop is not None
        and reflection_loop.reflection_priority in {"high", "critical"}
    ):
        return "future_escalation"
    if hitl in {"optional", "recommended"}:
        return "monitor"
    return "none"


def _confidence_trend(
    *,
    components: tuple[CreativeConfidenceComponent, ...],
    reflection_loop: ReflectionLoopProfile | None,
) -> ConfidenceTrend:
    if len(components) < 2:
        return "unknown"
    scores = [item.score for item in components]
    if max(scores) - min(scores) >= 0.32:
        return "conflicting"
    if reflection_loop is not None and reflection_loop.reflection_priority in {
        "high",
        "critical",
    }:
        return "declining"
    if reflection_loop is not None and not reflection_loop.reflection_required:
        return "stable"
    return "improving" if scores[-1] >= scores[0] else "stable"


def _confidence_evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    components: tuple[CreativeConfidenceComponent, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    planning_metadata: PlanningMetadata,
) -> tuple[str, ...]:
    evidence = [f"Request: {_clip(request.query, 220)}"]
    if route_decision is not None:
        domains = ", ".join(domain.value for domain in route_decision.domains) or "none"
        evidence.append(f"Route: {route_decision.route.value}; domains {domains}.")
    evidence.append(
        "Components: "
        + ", ".join(f"{item.source}:{item.score:.2f}" for item in components)
        + "."
    )
    if creative_critic is not None:
        evidence.append(
            f"Creative critic: {creative_critic.risk_assessment} risk; {creative_critic.critic_confidence:.2f} confidence."  # noqa: E501
        )
    if self_evaluation is not None:
        evidence.append(
            f"Self evaluation: {self_evaluation.completeness_assessment}; {self_evaluation.self_evaluation_confidence:.2f} confidence."  # noqa: E501
        )
    if creative_improvement_planner is not None:
        evidence.append(
            "Creative improvement planner: "
            f"{len(creative_improvement_planner.improvement_priorities)} priorities; "
            f"{creative_improvement_planner.confidence:.2f} confidence."
        )
    if reflection_loop is not None:
        evidence.append(
            "Reflection loop: "
            f"{reflection_loop.reflection_priority} priority; "
            f"{reflection_loop.reflection_confidence:.2f} confidence."
        )
    if planning_metadata:
        evidence.append(
            f"Planning metadata considered: {len(planning_metadata)} profile(s)."
        )
    evidence.append("Authority boundary verified: metadata-only confidence assessment.")
    return _dedupe(evidence)[:16]


def _prompt_guidance(
    *,
    level: ConfidenceLevel,
    hitl: ExpectedHumanReviewNeed,
) -> tuple[str, ...]:
    guidance = [
        "Use Creative Confidence metadata as advisory confidence and uncertainty context only.",
        "Do not change outputs, modify artifacts, trigger refinement, trigger retries, change routing, select runtimes, call providers, alter previews, or invoke V4 agents.",  # noqa: E501
    ]
    if hitl in {"recommended", "required"}:
        guidance.append(
            f"Surface {hitl} human review need before presenting confidence as settled."
        )
    if level in {"low", "critical"}:
        guidance.append(
            "Preserve caveats around low confidence and unresolved evaluation signals."
        )
    return _dedupe(guidance)[:8]


def _planning_metadata_evidence(planning_metadata: PlanningMetadata) -> str:
    labels = ", ".join(_metadata_label(item) for item in planning_metadata[:8])
    return f"{len(planning_metadata)} planning profile(s): {labels}."


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)
