"""Metadata-only Reflection Loop Engine for V3.4 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)

ReflectionPriority = Literal["critical", "high", "medium", "low", "none"]
ReflectionDepth = Literal["none", "light", "moderate", "deep"]
ReflectionEstimate = Literal["none", "low", "medium", "high"]
HitlRecommendation = Literal["not_needed", "optional", "recommended", "required"]

REFLECTION_LOOP_ENGINE_AUTHORITY_BOUNDARY = (
    "The Reflection Loop Engine evaluates whether an additional refinement "
    "pass would theoretically improve quality, but remains advisory metadata "
    "only; it does not regenerate responses, trigger refinement, retry "
    "providers, modify artifacts, change runtime selection, change routing, "
    "modify previews, invoke V4 agents, execute artifacts, or create workflow "
    "loops."
)


class ReflectionLoopProfile(BaseModel):
    """Inspectable metadata describing theoretical reflection value."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["reflection_loop_engine"] = "reflection_loop_engine"
    serialization_version: Literal["v1"] = "v1"
    reflection_confidence: float = Field(ge=0, le=1)
    reflection_summary: str = Field(min_length=1, max_length=680)
    reflection_required: bool
    reflection_priority: ReflectionPriority
    reflection_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    reflection_depth: ReflectionDepth
    expected_quality_gain: ReflectionEstimate
    expected_risk_reduction: ReflectionEstimate
    expected_cost: ReflectionEstimate
    expected_latency: ReflectionEstimate
    confidence_after_reflection: float = Field(ge=0, le=1)
    unresolved_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    refinement_candidates: tuple[str, ...] = Field(min_length=1, max_length=8)
    stop_conditions: tuple[str, ...] = Field(min_length=1, max_length=8)
    hitl_recommendation: HitlRecommendation
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    authority_boundary: str = Field(
        default=REFLECTION_LOOP_ENGINE_AUTHORITY_BOUNDARY,
        max_length=980,
    )


def derive_reflection_loop_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object] = (),
) -> ReflectionLoopProfile:
    """Estimate theoretical reflection value without changing workflow control."""

    quality_score = _quality_score(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    risk_score = _risk_score(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        planning_metadata=planning_metadata,
    )
    priority = _reflection_priority(
        quality_score=quality_score,
        risk_score=risk_score,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
    )
    required = priority in {"critical", "high", "medium"}
    depth = _reflection_depth(priority)
    quality_gain = _expected_quality_gain(
        quality_score=quality_score,
        risk_score=risk_score,
        priority=priority,
    )
    risk_reduction = _expected_risk_reduction(risk_score=risk_score, priority=priority)
    unresolved = _unresolved_questions(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        planning_metadata=planning_metadata,
    )
    candidates = _refinement_candidates(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        planning_metadata=planning_metadata,
        required=required,
    )

    return ReflectionLoopProfile(
        reflection_confidence=_reflection_confidence(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            planning_metadata=planning_metadata,
        ),
        reflection_summary=_reflection_summary(
            priority=priority,
            depth=depth,
            quality_gain=quality_gain,
            risk_reduction=risk_reduction,
            required=required,
            top_candidate=candidates[0],
        ),
        reflection_required=required,
        reflection_priority=priority,
        reflection_rationale=_reflection_rationale(
            quality_score=quality_score,
            risk_score=risk_score,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
        ),
        reflection_depth=depth,
        expected_quality_gain=quality_gain,
        expected_risk_reduction=risk_reduction,
        expected_cost=_expected_cost(depth),
        expected_latency=_expected_latency(depth),
        confidence_after_reflection=_confidence_after_reflection(
            quality_score=quality_score,
            quality_gain=quality_gain,
            risk_reduction=risk_reduction,
        ),
        unresolved_questions=unresolved,
        refinement_candidates=candidates,
        stop_conditions=_stop_conditions(required=required),
        hitl_recommendation=_hitl_recommendation(
            priority=priority,
            unresolved_questions=unresolved,
        ),
        prompt_guidance=_prompt_guidance(
            required=required,
            priority=priority,
            depth=depth,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            creative_improvement_planner=creative_improvement_planner,
            planning_metadata=planning_metadata,
            quality_score=quality_score,
            risk_score=risk_score,
        ),
    )


def reflection_loop_prompt_lines(profile: ReflectionLoopProfile) -> tuple[str, ...]:
    """Render Reflection Loop metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Reflection confidence: {profile.reflection_confidence:.2f}.",
        f"Reflection required: {profile.reflection_required}.",
        f"Reflection priority: {profile.reflection_priority}.",
        f"Reflection depth: {profile.reflection_depth}.",
        f"Expected quality gain: {profile.expected_quality_gain}.",
        f"Expected risk reduction: {profile.expected_risk_reduction}.",
        f"Expected cost: {profile.expected_cost}.",
        f"Expected latency: {profile.expected_latency}.",
        f"Confidence after reflection: {profile.confidence_after_reflection:.2f}.",
        f"Reflection summary: {profile.reflection_summary}",
    ]
    lines.extend(f"Reflection rationale: {item}" for item in profile.reflection_rationale)
    lines.extend(
        f"Reflection unresolved question: {item}"
        for item in profile.unresolved_questions
    )
    lines.extend(
        f"Reflection refinement candidate: {item}"
        for item in profile.refinement_candidates
    )
    lines.extend(f"Reflection stop condition: {item}" for item in profile.stop_conditions)
    lines.append(f"Reflection HITL recommendation: {profile.hitl_recommendation}.")
    lines.extend(f"Reflection prompt guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:48])


def _quality_score(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> float:
    values: list[float] = []
    if creative_critic is not None:
        values.extend(
            (
                creative_critic.concept_quality,
                creative_critic.execution_quality,
                creative_critic.artifact_quality,
                creative_critic.coherence_quality,
                creative_critic.runtime_fit_quality,
                creative_critic.clarity_quality,
                creative_critic.feasibility_quality,
            )
        )
    if self_evaluation is not None:
        values.extend(
            (
                self_evaluation.request_alignment,
                self_evaluation.intent_alignment,
                self_evaluation.constraint_alignment,
                self_evaluation.artifact_alignment,
                self_evaluation.runtime_alignment,
                self_evaluation.creative_coherence,
                self_evaluation.technical_coherence,
            )
        )
    if not values:
        return 0.5
    return round(sum(values) / len(values), 2)


def _risk_score(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object],
) -> int:
    score = 0
    if creative_critic is not None:
        score += {"low": 0, "medium": 1, "high": 3, "blocked": 5}[
            creative_critic.risk_assessment
        ]
        score += min(2, len(creative_critic.unsupported_assumptions))
        score += min(2, len(creative_critic.missing_information))
    if self_evaluation is not None:
        score += {
            "complete": 0,
            "mostly_complete": 0,
            "partial": 2,
            "blocked": 5,
        }[self_evaluation.completeness_assessment]
        score += {"low": 0, "medium": 1, "high": 3}[
            self_evaluation.ambiguity_assessment
        ]
        score += {"low": 0, "medium": 1, "high": 3}[
            self_evaluation.hallucination_risk
        ]
        score += {"low": 0, "medium": 1, "high": 2}[
            self_evaluation.overreach_risk
        ]
        score += {"low": 0, "medium": 1, "high": 2}[
            self_evaluation.underdelivery_risk
        ]
        score += min(3, len(self_evaluation.quality_gaps))
    if creative_improvement_planner is not None:
        priority_score = {
            "critical": 4,
            "high": 2,
            "medium": 1,
            "low": 0,
        }
        score += sum(
            priority_score[item.priority]
            for item in creative_improvement_planner.improvement_priorities[:3]
        )
    for item in planning_metadata:
        score += min(1, len(_metadata_values(item, "hitl_questions")[:1]))
        score += min(1, len(_metadata_values(item, "missing_information")[:1]))
    return min(score, 18)


def _reflection_priority(
    *,
    quality_score: float,
    risk_score: int,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
) -> ReflectionPriority:
    has_critical_priority = (
        creative_improvement_planner is not None
        and any(
            item.priority == "critical"
            for item in creative_improvement_planner.improvement_priorities
        )
    )
    if (
        has_critical_priority
        or risk_score >= 10
        or quality_score < 0.55
        or (
            creative_critic is not None
            and creative_critic.risk_assessment == "blocked"
        )
        or (
            self_evaluation is not None
            and self_evaluation.completeness_assessment == "blocked"
        )
    ):
        return "critical"
    if risk_score >= 6 or quality_score < 0.68:
        return "high"
    if risk_score >= 3 or quality_score < 0.78:
        return "medium"
    if risk_score > 0 or quality_score < 0.86:
        return "low"
    return "none"


def _reflection_depth(priority: ReflectionPriority) -> ReflectionDepth:
    return {
        "critical": "deep",
        "high": "moderate",
        "medium": "light",
        "low": "light",
        "none": "none",
    }[priority]


def _expected_quality_gain(
    *,
    quality_score: float,
    risk_score: int,
    priority: ReflectionPriority,
) -> ReflectionEstimate:
    if priority == "none":
        return "none"
    if priority == "critical" or quality_score < 0.6:
        return "high"
    if priority == "high" or quality_score < 0.75 or risk_score >= 5:
        return "medium"
    return "low"


def _expected_risk_reduction(
    *,
    risk_score: int,
    priority: ReflectionPriority,
) -> ReflectionEstimate:
    if priority == "none" or risk_score == 0:
        return "none"
    if priority == "critical" or risk_score >= 8:
        return "high"
    if priority == "high" or risk_score >= 4:
        return "medium"
    return "low"


def _expected_cost(depth: ReflectionDepth) -> ReflectionEstimate:
    return {
        "none": "none",
        "light": "low",
        "moderate": "medium",
        "deep": "high",
    }[depth]


def _expected_latency(depth: ReflectionDepth) -> ReflectionEstimate:
    return {
        "none": "none",
        "light": "low",
        "moderate": "medium",
        "deep": "high",
    }[depth]


def _confidence_after_reflection(
    *,
    quality_score: float,
    quality_gain: ReflectionEstimate,
    risk_reduction: ReflectionEstimate,
) -> float:
    gain = {"none": 0.0, "low": 0.04, "medium": 0.09, "high": 0.14}[quality_gain]
    risk = {"none": 0.0, "low": 0.02, "medium": 0.05, "high": 0.08}[
        risk_reduction
    ]
    return round(min(0.98, max(0.05, quality_score + gain + risk)), 2)


def _reflection_confidence(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object],
) -> float:
    values: list[float] = []
    if creative_critic is not None:
        values.append(creative_critic.critic_confidence)
    if self_evaluation is not None:
        values.append(self_evaluation.self_evaluation_confidence)
    if creative_improvement_planner is not None:
        values.append(creative_improvement_planner.confidence)
    if planning_metadata:
        values.append(min(0.78, 0.5 + len(planning_metadata) * 0.02))
    if not values:
        return 0.25
    return round(sum(values) / len(values), 2)


def _reflection_summary(
    *,
    priority: ReflectionPriority,
    depth: ReflectionDepth,
    quality_gain: ReflectionEstimate,
    risk_reduction: ReflectionEstimate,
    required: bool,
    top_candidate: str,
) -> str:
    return _clip(
        (
            "Reflection Loop Engine estimates that additional refinement is "
            f"{'recommended' if required else 'not required'} with {priority} "
            f"priority and {depth} theoretical depth. Expected quality gain is "
            f"{quality_gain}; expected risk reduction is {risk_reduction}. Top "
            f"candidate: {top_candidate}. This does not trigger workflow loops "
            "or modify outputs."
        ),
        680,
    )


def _reflection_rationale(
    *,
    quality_score: float,
    risk_score: int,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
) -> tuple[str, ...]:
    rationale = [
        f"Aggregate quality score is {quality_score:.2f}.",
        f"Aggregate reflection risk score is {risk_score}.",
    ]
    if creative_critic is not None:
        rationale.append(
            f"Creative Critic reports {creative_critic.risk_assessment} risk."
        )
    if self_evaluation is not None:
        rationale.append(
            "Self Evaluation reports "
            f"{self_evaluation.completeness_assessment} completeness."
        )
    if creative_improvement_planner is not None:
        top = creative_improvement_planner.improvement_priorities[0]
        rationale.append(
            f"Creative Improvement Planner top priority is {top.priority}: {top.title}"
        )
    rationale.append("Reflection remains advisory and cannot execute refinement.")
    return _dedupe(rationale)[:8]


def _unresolved_questions(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object],
) -> tuple[str, ...]:
    questions: list[str] = []
    if creative_improvement_planner is not None:
        questions.extend(creative_improvement_planner.hitl_questions[:3])
    if self_evaluation is not None:
        questions.extend(self_evaluation.hitl_questions[:3])
        questions.extend(self_evaluation.missing_information[:2])
    if creative_critic is not None:
        questions.extend(creative_critic.hitl_questions[:3])
        questions.extend(creative_critic.missing_information[:2])
    for item in planning_metadata:
        questions.extend(_metadata_values(item, "hitl_questions")[:1])
        questions.extend(_metadata_values(item, "missing_information")[:1])
    return _dedupe(questions)[:8]


def _refinement_candidates(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object],
    required: bool,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if creative_improvement_planner is not None:
        candidates.extend(creative_improvement_planner.future_refinement_candidates[:3])
        candidates.extend(creative_improvement_planner.highest_impact_opportunities[:3])
    if self_evaluation is not None:
        candidates.extend(self_evaluation.improvement_opportunities[:3])
        candidates.extend(self_evaluation.quality_gaps[:2])
    if creative_critic is not None:
        candidates.extend(creative_critic.improvement_opportunities[:3])
        candidates.extend(creative_critic.creative_weaknesses[:2])
    for item in planning_metadata:
        candidates.extend(_metadata_values(item, "unresolved_intent_gaps")[:1])
        candidates.extend(_metadata_values(item, "unresolved_composition_gaps")[:1])
        candidates.extend(_metadata_values(item, "unresolved_implementation_gaps")[:1])
        candidates.extend(_metadata_values(item, "prompt_guidance")[:1])
    if not candidates and not required:
        candidates.append(
            "Preserve the current response; no additional refinement candidate is recommended."
        )
    return _dedupe(candidates)[:8]


def _stop_conditions(*, required: bool) -> tuple[str, ...]:
    conditions = [
        "Do not trigger an automatic reflection loop from this metadata.",
        "Do not call providers, retry generation, modify artifacts, or change previews.",
        "Do not change routing, runtime selection, workflow control, or V4 agent behavior.",
    ]
    if not required:
        conditions.append(
            "Stop because current metadata does not justify additional refinement."
        )
    else:
        conditions.append(
            "Stop at advisory recommendation unless a future explicit workflow owns refinement."
        )
    return tuple(conditions)


def _hitl_recommendation(
    *,
    priority: ReflectionPriority,
    unresolved_questions: tuple[str, ...],
) -> HitlRecommendation:
    if priority == "critical":
        return "required"
    if priority in {"high", "medium"}:
        return "recommended" if unresolved_questions else "optional"
    if unresolved_questions:
        return "optional"
    return "not_needed"


def _prompt_guidance(
    *,
    required: bool,
    priority: ReflectionPriority,
    depth: ReflectionDepth,
) -> tuple[str, ...]:
    guidance = [
        "Use Reflection Loop metadata only to explain theoretical improvement value.",
        "Do not perform refinement, retry generation, call providers, change runtime, route providers, modify previews, or invoke V4 agents.",
    ]
    if required:
        guidance.append(
            f"Surface {priority} priority / {depth} depth as advisory guidance before claiming the result is final."
        )
    else:
        guidance.append(
            "Treat reflection as not required and preserve the current output direction."
        )
    return _dedupe(guidance)[:8]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    planning_metadata: Sequence[object],
    quality_score: float,
    risk_score: int,
) -> tuple[str, ...]:
    evidence = [f"Request: {_clip(request.query, 220)}"]
    if route_decision is not None:
        domains = ", ".join(domain.value for domain in route_decision.domains) or "none"
        evidence.append(f"Route: {route_decision.route.value}; domains {domains}.")
    evidence.append(f"Aggregate quality score: {quality_score:.2f}.")
    evidence.append(f"Aggregate reflection risk score: {risk_score}.")
    if creative_critic is not None:
        evidence.append(
            f"Creative critic: {creative_critic.risk_assessment} risk; {creative_critic.critic_confidence:.2f} confidence."
        )
    if self_evaluation is not None:
        evidence.append(
            f"Self evaluation: {self_evaluation.completeness_assessment}; {self_evaluation.self_evaluation_confidence:.2f} confidence."
        )
    if creative_improvement_planner is not None:
        evidence.append(
            "Creative improvement planner: "
            f"{len(creative_improvement_planner.improvement_priorities)} priorities; "
            f"{creative_improvement_planner.confidence:.2f} confidence."
        )
    if planning_metadata:
        labels = ", ".join(_metadata_label(item) for item in planning_metadata[:8])
        evidence.append(
            f"Planning metadata considered: {len(planning_metadata)} profile(s): {labels}."
        )
    evidence.append("Authority boundary verified: metadata-only reflection planning.")
    return _dedupe(evidence)[:16]


def _metadata_values(item: object, attribute: str) -> tuple[str, ...]:
    value = getattr(item, attribute, ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(entry) for entry in value if entry)
    return ()


def _metadata_label(item: object) -> str:
    role = getattr(item, "role", None)
    if isinstance(role, str) and role:
        return role
    return item.__class__.__name__
