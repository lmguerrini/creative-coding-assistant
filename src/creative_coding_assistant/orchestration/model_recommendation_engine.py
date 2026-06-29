"""V5.2 advisory model recommendation metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_policy_engine import (
    ExecutionPolicyDecision,
    ExecutionPolicyPlan,
    ExecutionPolicyPosture,
    evaluate_execution_policies,
)
from creative_coding_assistant.orchestration.hitl_budget_gate import (
    HitlBudgetGateStatus,
)
from creative_coding_assistant.orchestration.hybrid_studio import ModelProfileKind
from creative_coding_assistant.orchestration.model_router import (
    ModelRouteCandidate,
    ModelRouteCandidateStatus,
    ModelRouteFitBand,
    ModelRoutingPlan,
    route_model_request,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

ModelRecommendationStatus = ModelRouteCandidateStatus

MODEL_RECOMMENDATION_DECISION_SERIALIZATION_VERSION = (
    "model_recommendation_decision.v1"
)
MODEL_RECOMMENDATION_PLAN_SERIALIZATION_VERSION = "model_recommendation_plan.v1"
MODEL_RECOMMENDATION_AUTHORITY_BOUNDARY = (
    "The V5.2 Model Recommendation Engine combines advisory model route "
    "candidates with execution policy posture only; it does not select or "
    "switch the configured provider or model, apply execution policies, apply "
    "runtime recommendations, emit human input requests, enforce budgets, "
    "execute providers, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "configured_model_switching",
    "configured_provider_switching",
    "automatic_model_selection",
    "automatic_provider_selection",
    "provider_or_model_routing",
    "execution_policy_application",
    "runtime_recommendation_application",
    "human_input_request_emission",
    "budget_enforcement",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ModelRecommendationDecision(BaseModel):
    """One advisory model recommendation decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    recommendation_id: str = Field(min_length=1, max_length=180)
    rank: int = Field(ge=1, le=12)
    source_model_route_candidate_id: str = Field(min_length=1, max_length=180)
    source_execution_policy_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    model_profile_kind: ModelProfileKind
    profile_name: str = Field(min_length=1, max_length=160)
    route_name: RouteName
    route_fit_score: int = Field(ge=0, le=250)
    route_fit_band: ModelRouteFitBand
    execution_policy_posture: ExecutionPolicyPosture
    gate_status: HitlBudgetGateStatus
    recommendation_summary: str = Field(min_length=1, max_length=300)
    status: ModelRecommendationStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    model_recommendation_engine_implemented: Literal[True] = True
    model_recommendation_implemented: Literal[True] = True
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["model_recommendation_decision.v1"] = (
        MODEL_RECOMMENDATION_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_rank_and_status(self) -> Self:
        if self.status == "recommended" and self.rank != 1:
            raise ValueError("recommended model recommendation must be rank 1")
        if self.status == "fallback" and self.rank == 1:
            raise ValueError("rank 1 model recommendation must be recommended")
        if self.recommendation_summary != _recommendation_summary(
            self.profile_name,
            self.execution_policy_posture,
        ):
            raise ValueError("recommendation_summary must match policy posture")
        return self


class ModelRecommendationPlan(BaseModel):
    """Bounded V5.2 advisory model recommendation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_recommendation_engine"] = "model_recommendation_engine"
    serialization_version: Literal["model_recommendation_plan.v1"] = (
        MODEL_RECOMMENDATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MODEL_RECOMMENDATION_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_model_routing_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_execution_policy_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_model_route_candidate_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    source_execution_policy_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendations: tuple[ModelRecommendationDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_recommendation_id: str = Field(min_length=1, max_length=180)
    recommended_model_profile_id: str = Field(min_length=1, max_length=120)
    recommended_execution_policy_posture: ExecutionPolicyPosture
    fallback_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    recommendation_count: int = Field(ge=1, le=12)
    guarded_recommendation_count: int = Field(ge=0, le=12)
    manual_review_recommendation_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    model_recommendation_engine_implemented: Literal[True] = True
    model_recommendation_implemented: Literal[True] = True
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_recommendations(self) -> Self:
        derived_recommendation_ids = tuple(
            recommendation.recommendation_id
            for recommendation in self.recommendations
        )
        if len(set(derived_recommendation_ids)) != len(derived_recommendation_ids):
            raise ValueError("recommendation_ids must be unique")
        if self.recommendation_ids != derived_recommendation_ids:
            raise ValueError("recommendation_ids must match recommendations")
        if self.recommendation_count != len(self.recommendations):
            raise ValueError("recommendation_count must match recommendations")
        if self.source_model_route_candidate_ids != tuple(
            recommendation.source_model_route_candidate_id
            for recommendation in self.recommendations
        ):
            raise ValueError("source_model_route_candidate_ids must match")
        if self.source_execution_policy_ids != tuple(
            recommendation.source_execution_policy_id
            for recommendation in self.recommendations
        ):
            raise ValueError("source_execution_policy_ids must match")

        recommended = tuple(
            recommendation
            for recommendation in self.recommendations
            if recommendation.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended model recommendation is required")
        recommended_decision = recommended[0]
        if (
            self.recommended_recommendation_id
            != recommended_decision.recommendation_id
        ):
            raise ValueError("recommended_recommendation_id must match decision")
        if (
            self.recommended_model_profile_id
            != recommended_decision.source_model_profile_id
        ):
            raise ValueError("recommended_model_profile_id must match decision")
        if (
            self.recommended_execution_policy_posture
            != recommended_decision.execution_policy_posture
        ):
            raise ValueError("recommended_execution_policy_posture must match decision")
        if self.fallback_recommendation_ids != tuple(
            recommendation.recommendation_id
            for recommendation in self.recommendations
            if recommendation.status == "fallback"
        ):
            raise ValueError("fallback_recommendation_ids must match recommendations")
        if self.guarded_recommendation_count != sum(
            1
            for recommendation in self.recommendations
            if recommendation.execution_policy_posture == "guarded_execution_policy"
        ):
            raise ValueError("guarded_recommendation_count must match recommendations")
        if self.manual_review_recommendation_count != sum(
            1
            for recommendation in self.recommendations
            if recommendation.execution_policy_posture
            == "manual_review_execution_policy"
        ):
            raise ValueError(
                "manual_review_recommendation_count must match recommendations"
            )
        for recommendation in self.recommendations:
            if recommendation.route_name != self.route_name:
                raise ValueError("recommendation route_name must match plan route_name")
        return self


def recommend_model_profile(
    *,
    model_routing: ModelRoutingPlan | None = None,
    execution_policies: ExecutionPolicyPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> ModelRecommendationPlan:
    """Return advisory model recommendations without selecting a model."""

    model_plan = model_routing or route_model_request(
        route_decision=route_decision,
        route=route,
    )
    policy_plan = execution_policies or evaluate_execution_policies(
        route_decision=route_decision,
        route=model_plan.route_name,
    )
    if policy_plan.route_name != model_plan.route_name:
        raise ValueError("execution policy route must match model routing route")

    policies_by_profile = {
        policy.source_model_profile_id: policy for policy in policy_plan.policies
    }
    recommendations = tuple(
        _recommendation_from_sources(
            candidate,
            _policy_for_candidate(candidate, policies_by_profile),
        )
        for candidate in model_plan.candidates
    )
    recommended = _recommended_recommendation(recommendations)

    return ModelRecommendationPlan(
        source_model_routing_serialization_version=model_plan.serialization_version,
        source_execution_policy_serialization_version=policy_plan.serialization_version,
        route_name=model_plan.route_name,
        source_model_route_candidate_ids=tuple(
            recommendation.source_model_route_candidate_id
            for recommendation in recommendations
        ),
        source_execution_policy_ids=tuple(
            recommendation.source_execution_policy_id
            for recommendation in recommendations
        ),
        recommendations=recommendations,
        recommendation_ids=tuple(
            recommendation.recommendation_id
            for recommendation in recommendations
        ),
        recommended_recommendation_id=recommended.recommendation_id,
        recommended_model_profile_id=recommended.source_model_profile_id,
        recommended_execution_policy_posture=recommended.execution_policy_posture,
        fallback_recommendation_ids=tuple(
            recommendation.recommendation_id
            for recommendation in recommendations
            if recommendation.status == "fallback"
        ),
        recommendation_count=len(recommendations),
        guarded_recommendation_count=sum(
            1
            for recommendation in recommendations
            if recommendation.execution_policy_posture == "guarded_execution_policy"
        ),
        manual_review_recommendation_count=sum(
            1
            for recommendation in recommendations
            if recommendation.execution_policy_posture
            == "manual_review_execution_policy"
        ),
        advisory_actions=_plan_actions(model_plan.route_name, recommended),
    )


def model_recommendation_by_id(
    recommendation_id: str,
    plan: ModelRecommendationPlan | None = None,
) -> ModelRecommendationDecision | None:
    """Return one advisory model recommendation without selecting it."""

    source_plan = plan or recommend_model_profile()
    for recommendation in source_plan.recommendations:
        if recommendation.recommendation_id == recommendation_id:
            return recommendation
    return None


def model_recommendations_for_policy_posture(
    posture: ExecutionPolicyPosture,
    plan: ModelRecommendationPlan | None = None,
) -> tuple[ModelRecommendationDecision, ...]:
    """Return advisory model recommendations for one policy posture."""

    source_plan = plan or recommend_model_profile()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.execution_policy_posture == posture
    )


def _recommendation_from_sources(
    candidate: ModelRouteCandidate,
    policy: ExecutionPolicyDecision,
) -> ModelRecommendationDecision:
    return ModelRecommendationDecision(
        recommendation_id=f"model_recommendation::{candidate.source_model_profile_id}",
        rank=candidate.rank,
        source_model_route_candidate_id=candidate.candidate_id,
        source_execution_policy_id=policy.policy_id,
        source_model_profile_id=candidate.source_model_profile_id,
        model_profile_kind=candidate.model_profile_kind,
        profile_name=candidate.profile_name,
        route_name=candidate.route_name,
        route_fit_score=candidate.fit_score,
        route_fit_band=candidate.fit_band,
        execution_policy_posture=policy.execution_policy_posture,
        gate_status=policy.gate_status,
        recommendation_summary=_recommendation_summary(
            candidate.profile_name,
            policy.execution_policy_posture,
        ),
        status=candidate.status,
        evidence=(
            f"Derived from {candidate.candidate_id}.",
            f"Execution policy source: {policy.policy_id}.",
            f"Route fit band: {candidate.fit_band}.",
            "Model recommendation is advisory metadata only.",
        ),
        advisory_actions=_decision_actions(policy.execution_policy_posture),
    )


def _policy_for_candidate(
    candidate: ModelRouteCandidate,
    policies_by_profile: dict[str, ExecutionPolicyDecision],
) -> ExecutionPolicyDecision:
    try:
        return policies_by_profile[candidate.source_model_profile_id]
    except KeyError as exc:
        raise ValueError("execution policy is missing for model candidate") from exc


def _recommendation_summary(
    profile_name: str,
    posture: ExecutionPolicyPosture,
) -> str:
    if posture == "manual_review_execution_policy":
        return f"Recommend {profile_name} as manual-review-gated advisory metadata."
    if posture == "guarded_execution_policy":
        return f"Recommend {profile_name} with guarded policy review metadata."
    return f"Recommend {profile_name} with standard policy metadata."


def _recommended_recommendation(
    recommendations: tuple[ModelRecommendationDecision, ...],
) -> ModelRecommendationDecision:
    for recommendation in recommendations:
        if recommendation.status == "recommended":
            return recommendation
    raise ValueError("model recommendation engine requires a recommended decision")


def _decision_actions(
    posture: ExecutionPolicyPosture,
) -> tuple[str, ...]:
    return (
        f"Present advisory model recommendation with {posture}.",
        "Keep configured model switching disabled.",
        "Leave provider/model routing to explicit future runtime integration.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: ModelRecommendationDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.source_model_profile_id} recommendation for "
            f"{route_name.value}."
        ),
        "Do not select or switch the configured model automatically.",
        "Keep provider execution and execution policy application disabled.",
    )
