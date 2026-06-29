"""V5.2 advisory runtime recommendation metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.budget_policies import (
    BudgetPolicyPosture,
)
from creative_coding_assistant.orchestration.hitl_budget_gate import (
    HitlBudgetGateDecision,
    HitlBudgetGatePlan,
    HitlBudgetGateStatus,
    evaluate_hitl_budget_gate,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

RuntimeRecommendationPosture = Literal[
    "standard_runtime",
    "guarded_runtime_review",
    "operator_review_required",
]
RuntimeRecommendationStatus = Literal["recommended", "fallback"]

RUNTIME_RECOMMENDATION_DECISION_SERIALIZATION_VERSION = (
    "runtime_recommendation_decision.v1"
)
RUNTIME_RECOMMENDATION_PLAN_SERIALIZATION_VERSION = (
    "runtime_recommendation_plan.v1"
)
RUNTIME_RECOMMENDATION_AUTHORITY_BOUNDARY = (
    "The V5.2 Runtime Recommendation Engine summarizes advisory runtime "
    "posture from HITL budget gate metadata only; it does not apply runtime "
    "recommendations, emit human input requests, enforce budgets, apply "
    "execution policies, block execution, select or switch providers or "
    "models, execute providers, control workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_recommendation_application",
    "human_input_request_emission",
    "execution_policy_application",
    "execution_blocking",
    "budget_enforcement",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class RuntimeRecommendationDecision(BaseModel):
    """One advisory runtime recommendation decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    recommendation_id: str = Field(min_length=1, max_length=180)
    source_hitl_budget_gate_id: str = Field(min_length=1, max_length=180)
    source_budget_policy_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    gate_status: HitlBudgetGateStatus
    budget_posture: BudgetPolicyPosture
    runtime_posture: RuntimeRecommendationPosture
    recommendation_summary: str = Field(min_length=1, max_length=260)
    status: RuntimeRecommendationStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    runtime_recommendation_engine_implemented: Literal[True] = True
    runtime_recommendation_implemented: Literal[True] = True
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["runtime_recommendation_decision.v1"] = (
        RUNTIME_RECOMMENDATION_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_source_status(self) -> Self:
        if self.runtime_posture != _runtime_posture(self.gate_status):
            raise ValueError("runtime_posture must match gate_status")
        if self.recommendation_summary != _recommendation_summary(
            self.runtime_posture,
        ):
            raise ValueError("recommendation_summary must match runtime_posture")
        return self


class RuntimeRecommendationPlan(BaseModel):
    """Bounded V5.2 advisory runtime recommendation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_recommendation_engine"] = "runtime_recommendation_engine"
    serialization_version: Literal["runtime_recommendation_plan.v1"] = (
        RUNTIME_RECOMMENDATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RUNTIME_RECOMMENDATION_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_hitl_budget_gate_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_hitl_budget_gate_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendations: tuple[RuntimeRecommendationDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_recommendation_id: str = Field(min_length=1, max_length=180)
    recommended_runtime_posture: RuntimeRecommendationPosture
    recommended_gate_status: HitlBudgetGateStatus
    fallback_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    recommendation_count: int = Field(ge=1, le=12)
    guarded_recommendation_count: int = Field(ge=0, le=12)
    operator_review_required_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    runtime_recommendation_engine_implemented: Literal[True] = True
    runtime_recommendation_implemented: Literal[True] = True
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
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
        if self.source_hitl_budget_gate_ids != tuple(
            recommendation.source_hitl_budget_gate_id
            for recommendation in self.recommendations
        ):
            raise ValueError("source_hitl_budget_gate_ids must match recommendations")

        recommended = tuple(
            recommendation
            for recommendation in self.recommendations
            if recommendation.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended runtime recommendation is required")
        recommended_decision = recommended[0]
        if (
            self.recommended_recommendation_id
            != recommended_decision.recommendation_id
        ):
            raise ValueError("recommended_recommendation_id must match decision")
        if self.recommended_runtime_posture != recommended_decision.runtime_posture:
            raise ValueError("recommended_runtime_posture must match decision")
        if self.recommended_gate_status != recommended_decision.gate_status:
            raise ValueError("recommended_gate_status must match decision")
        if self.fallback_recommendation_ids != tuple(
            recommendation.recommendation_id
            for recommendation in self.recommendations
            if recommendation.status == "fallback"
        ):
            raise ValueError("fallback_recommendation_ids must match recommendations")
        if self.guarded_recommendation_count != sum(
            1
            for recommendation in self.recommendations
            if recommendation.runtime_posture == "guarded_runtime_review"
        ):
            raise ValueError("guarded_recommendation_count must match recommendations")
        if self.operator_review_required_count != sum(
            1
            for recommendation in self.recommendations
            if recommendation.runtime_posture == "operator_review_required"
        ):
            raise ValueError(
                "operator_review_required_count must match recommendations"
            )
        for recommendation in self.recommendations:
            if recommendation.route_name != self.route_name:
                raise ValueError("recommendation route_name must match plan route_name")
        return self


def recommend_runtime_execution(
    *,
    hitl_budget_gate: HitlBudgetGatePlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> RuntimeRecommendationPlan:
    """Return advisory runtime recommendations without applying runtime behavior."""

    gate_plan = hitl_budget_gate or evaluate_hitl_budget_gate(
        route_decision=route_decision,
        route=route,
    )
    recommendations = tuple(
        _recommendation_from_gate(decision) for decision in gate_plan.decisions
    )
    recommended = _recommended_recommendation(recommendations)

    return RuntimeRecommendationPlan(
        source_hitl_budget_gate_serialization_version=gate_plan.serialization_version,
        route_name=gate_plan.route_name,
        source_hitl_budget_gate_ids=tuple(
            recommendation.source_hitl_budget_gate_id
            for recommendation in recommendations
        ),
        recommendations=recommendations,
        recommendation_ids=tuple(
            recommendation.recommendation_id
            for recommendation in recommendations
        ),
        recommended_recommendation_id=recommended.recommendation_id,
        recommended_runtime_posture=recommended.runtime_posture,
        recommended_gate_status=recommended.gate_status,
        fallback_recommendation_ids=tuple(
            recommendation.recommendation_id
            for recommendation in recommendations
            if recommendation.status == "fallback"
        ),
        recommendation_count=len(recommendations),
        guarded_recommendation_count=sum(
            1
            for recommendation in recommendations
            if recommendation.runtime_posture == "guarded_runtime_review"
        ),
        operator_review_required_count=sum(
            1
            for recommendation in recommendations
            if recommendation.runtime_posture == "operator_review_required"
        ),
        advisory_actions=_plan_actions(gate_plan.route_name, recommended),
    )


def runtime_recommendation_by_id(
    recommendation_id: str,
    plan: RuntimeRecommendationPlan | None = None,
) -> RuntimeRecommendationDecision | None:
    """Return one advisory runtime recommendation without applying it."""

    source_plan = plan or recommend_runtime_execution()
    for recommendation in source_plan.recommendations:
        if recommendation.recommendation_id == recommendation_id:
            return recommendation
    return None


def runtime_recommendations_for_posture(
    posture: RuntimeRecommendationPosture,
    plan: RuntimeRecommendationPlan | None = None,
) -> tuple[RuntimeRecommendationDecision, ...]:
    """Return advisory runtime recommendations for one posture."""

    source_plan = plan or recommend_runtime_execution()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.runtime_posture == posture
    )


def _recommendation_from_gate(
    gate: HitlBudgetGateDecision,
) -> RuntimeRecommendationDecision:
    posture = _runtime_posture(gate.gate_status)
    return RuntimeRecommendationDecision(
        recommendation_id=f"runtime_recommendation::{gate.source_model_profile_id}",
        source_hitl_budget_gate_id=gate.gate_id,
        source_budget_policy_id=gate.source_budget_policy_id,
        source_model_profile_id=gate.source_model_profile_id,
        route_name=gate.route_name,
        gate_status=gate.gate_status,
        budget_posture=gate.budget_posture,
        runtime_posture=posture,
        recommendation_summary=_recommendation_summary(posture),
        status=gate.status,
        evidence=(
            f"Derived from {gate.gate_id}.",
            f"HITL budget gate status: {gate.gate_status}.",
            f"Budget posture: {gate.budget_posture}.",
            "Runtime recommendation is advisory metadata only.",
        ),
        advisory_actions=_decision_actions(posture),
    )


def _runtime_posture(
    gate_status: HitlBudgetGateStatus,
) -> RuntimeRecommendationPosture:
    if gate_status == "required":
        return "operator_review_required"
    if gate_status == "review_recommended":
        return "guarded_runtime_review"
    return "standard_runtime"


def _recommendation_summary(posture: RuntimeRecommendationPosture) -> str:
    if posture == "operator_review_required":
        return "Surface operator review as required before any future runtime application."
    if posture == "guarded_runtime_review":
        return "Surface guarded runtime review before any future runtime application."
    return "Surface standard runtime posture with no budget review escalation."


def _recommended_recommendation(
    recommendations: tuple[RuntimeRecommendationDecision, ...],
) -> RuntimeRecommendationDecision:
    for recommendation in recommendations:
        if recommendation.status == "recommended":
            return recommendation
    raise ValueError("runtime recommendation engine requires a recommended decision")


def _decision_actions(
    posture: RuntimeRecommendationPosture,
) -> tuple[str, ...]:
    return (
        f"Present {posture} as advisory runtime metadata.",
        "Keep runtime recommendation application disabled.",
        "Leave execution policy application to explicit future integration.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: RuntimeRecommendationDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.runtime_posture} recommendation for "
            f"{route_name.value}."
        ),
        "Do not apply runtime recommendations automatically.",
        "Keep provider/model routing and execution policy application disabled.",
    )
