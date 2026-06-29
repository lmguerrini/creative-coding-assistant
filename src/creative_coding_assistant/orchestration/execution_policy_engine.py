"""V5.2 advisory execution policy metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hitl_budget_gate import (
    HitlBudgetGateStatus,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName
from creative_coding_assistant.orchestration.runtime_recommendation_engine import (
    RuntimeRecommendationDecision,
    RuntimeRecommendationPlan,
    RuntimeRecommendationPosture,
    recommend_runtime_execution,
)

ExecutionPolicyPosture = Literal[
    "standard_execution_policy",
    "guarded_execution_policy",
    "manual_review_execution_policy",
]
ExecutionPolicyStatus = Literal["recommended", "fallback"]

EXECUTION_POLICY_DECISION_SERIALIZATION_VERSION = "execution_policy_decision.v1"
EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION = "execution_policy_plan.v1"
EXECUTION_POLICY_AUTHORITY_BOUNDARY = (
    "The V5.2 Execution Policy Engine derives advisory execution policy "
    "posture from runtime recommendation metadata only; it does not apply "
    "execution policies, apply runtime recommendations, emit human input "
    "requests, enforce budgets, block execution, select or switch providers "
    "or models, execute providers, control workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "execution_policy_application",
    "runtime_recommendation_application",
    "human_input_request_emission",
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


class ExecutionPolicyDecision(BaseModel):
    """One advisory execution policy decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=180)
    source_runtime_recommendation_id: str = Field(min_length=1, max_length=180)
    source_hitl_budget_gate_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    runtime_posture: RuntimeRecommendationPosture
    gate_status: HitlBudgetGateStatus
    execution_policy_posture: ExecutionPolicyPosture
    policy_summary: str = Field(min_length=1, max_length=260)
    status: ExecutionPolicyStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    execution_policy_engine_implemented: Literal[True] = True
    execution_policy_recommendation_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[False] = False
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_policy_decision.v1"] = (
        EXECUTION_POLICY_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_runtime_posture(self) -> Self:
        if self.execution_policy_posture != _policy_posture(self.runtime_posture):
            raise ValueError("execution_policy_posture must match runtime_posture")
        if self.policy_summary != _policy_summary(self.execution_policy_posture):
            raise ValueError("policy_summary must match execution_policy_posture")
        return self


class ExecutionPolicyPlan(BaseModel):
    """Bounded V5.2 advisory execution policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_policy_engine"] = "execution_policy_engine"
    serialization_version: Literal["execution_policy_plan.v1"] = (
        EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_POLICY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_runtime_recommendation_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_runtime_recommendation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    policies: tuple[ExecutionPolicyDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    policy_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_policy_id: str = Field(min_length=1, max_length=180)
    recommended_execution_policy_posture: ExecutionPolicyPosture
    recommended_gate_status: HitlBudgetGateStatus
    fallback_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    policy_count: int = Field(ge=1, le=12)
    guarded_policy_count: int = Field(ge=0, le=12)
    manual_review_policy_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    execution_policy_engine_implemented: Literal[True] = True
    execution_policy_recommendation_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[False] = False
    runtime_recommendation_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
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
    def _plan_matches_policies(self) -> Self:
        derived_policy_ids = tuple(policy.policy_id for policy in self.policies)
        if len(set(derived_policy_ids)) != len(derived_policy_ids):
            raise ValueError("policy_ids must be unique")
        if self.policy_ids != derived_policy_ids:
            raise ValueError("policy_ids must match policies")
        if self.policy_count != len(self.policies):
            raise ValueError("policy_count must match policies")
        if self.source_runtime_recommendation_ids != tuple(
            policy.source_runtime_recommendation_id for policy in self.policies
        ):
            raise ValueError("source_runtime_recommendation_ids must match policies")

        recommended = tuple(
            policy for policy in self.policies if policy.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended execution policy is required")
        recommended_policy = recommended[0]
        if self.recommended_policy_id != recommended_policy.policy_id:
            raise ValueError("recommended_policy_id must match policy")
        if (
            self.recommended_execution_policy_posture
            != recommended_policy.execution_policy_posture
        ):
            raise ValueError("recommended_execution_policy_posture must match policy")
        if self.recommended_gate_status != recommended_policy.gate_status:
            raise ValueError("recommended_gate_status must match policy")
        if self.fallback_policy_ids != tuple(
            policy.policy_id for policy in self.policies if policy.status == "fallback"
        ):
            raise ValueError("fallback_policy_ids must match policies")
        if self.guarded_policy_count != sum(
            1
            for policy in self.policies
            if policy.execution_policy_posture == "guarded_execution_policy"
        ):
            raise ValueError("guarded_policy_count must match policies")
        if self.manual_review_policy_count != sum(
            1
            for policy in self.policies
            if policy.execution_policy_posture == "manual_review_execution_policy"
        ):
            raise ValueError("manual_review_policy_count must match policies")
        for policy in self.policies:
            if policy.route_name != self.route_name:
                raise ValueError("policy route_name must match plan route_name")
        return self


def evaluate_execution_policies(
    *,
    runtime_recommendations: RuntimeRecommendationPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> ExecutionPolicyPlan:
    """Return advisory execution policy metadata without applying it."""

    runtime_plan = runtime_recommendations or recommend_runtime_execution(
        route_decision=route_decision,
        route=route,
    )
    policies = tuple(
        _policy_from_runtime_recommendation(recommendation)
        for recommendation in runtime_plan.recommendations
    )
    recommended = _recommended_policy(policies)

    return ExecutionPolicyPlan(
        source_runtime_recommendation_serialization_version=(
            runtime_plan.serialization_version
        ),
        route_name=runtime_plan.route_name,
        source_runtime_recommendation_ids=tuple(
            policy.source_runtime_recommendation_id for policy in policies
        ),
        policies=policies,
        policy_ids=tuple(policy.policy_id for policy in policies),
        recommended_policy_id=recommended.policy_id,
        recommended_execution_policy_posture=recommended.execution_policy_posture,
        recommended_gate_status=recommended.gate_status,
        fallback_policy_ids=tuple(
            policy.policy_id for policy in policies if policy.status == "fallback"
        ),
        policy_count=len(policies),
        guarded_policy_count=sum(
            1
            for policy in policies
            if policy.execution_policy_posture == "guarded_execution_policy"
        ),
        manual_review_policy_count=sum(
            1
            for policy in policies
            if policy.execution_policy_posture == "manual_review_execution_policy"
        ),
        advisory_actions=_plan_actions(runtime_plan.route_name, recommended),
    )


def execution_policy_by_id(
    policy_id: str,
    plan: ExecutionPolicyPlan | None = None,
) -> ExecutionPolicyDecision | None:
    """Return one advisory execution policy without applying it."""

    source_plan = plan or evaluate_execution_policies()
    for policy in source_plan.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


def execution_policies_for_posture(
    posture: ExecutionPolicyPosture,
    plan: ExecutionPolicyPlan | None = None,
) -> tuple[ExecutionPolicyDecision, ...]:
    """Return advisory execution policies for one posture."""

    source_plan = plan or evaluate_execution_policies()
    return tuple(
        policy
        for policy in source_plan.policies
        if policy.execution_policy_posture == posture
    )


def _policy_from_runtime_recommendation(
    recommendation: RuntimeRecommendationDecision,
) -> ExecutionPolicyDecision:
    posture = _policy_posture(recommendation.runtime_posture)
    return ExecutionPolicyDecision(
        policy_id=f"execution_policy::{recommendation.source_model_profile_id}",
        source_runtime_recommendation_id=recommendation.recommendation_id,
        source_hitl_budget_gate_id=recommendation.source_hitl_budget_gate_id,
        source_model_profile_id=recommendation.source_model_profile_id,
        route_name=recommendation.route_name,
        runtime_posture=recommendation.runtime_posture,
        gate_status=recommendation.gate_status,
        execution_policy_posture=posture,
        policy_summary=_policy_summary(posture),
        status=recommendation.status,
        evidence=(
            f"Derived from {recommendation.recommendation_id}.",
            f"Runtime posture: {recommendation.runtime_posture}.",
            f"HITL budget gate status: {recommendation.gate_status}.",
            "Execution policy remains advisory metadata only.",
        ),
        advisory_actions=_decision_actions(posture),
    )


def _policy_posture(
    runtime_posture: RuntimeRecommendationPosture,
) -> ExecutionPolicyPosture:
    if runtime_posture == "operator_review_required":
        return "manual_review_execution_policy"
    if runtime_posture == "guarded_runtime_review":
        return "guarded_execution_policy"
    return "standard_execution_policy"


def _policy_summary(posture: ExecutionPolicyPosture) -> str:
    if posture == "manual_review_execution_policy":
        return "Surface manual review as required before any future policy application."
    if posture == "guarded_execution_policy":
        return "Surface guarded execution policy posture for future review."
    return "Surface standard execution policy posture with no review escalation."


def _recommended_policy(
    policies: tuple[ExecutionPolicyDecision, ...],
) -> ExecutionPolicyDecision:
    for policy in policies:
        if policy.status == "recommended":
            return policy
    raise ValueError("execution policy engine requires a recommended policy")


def _decision_actions(
    posture: ExecutionPolicyPosture,
) -> tuple[str, ...]:
    return (
        f"Present {posture} as advisory execution policy metadata.",
        "Keep execution policy application disabled.",
        "Leave workflow control to explicit future runtime integration.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: ExecutionPolicyDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.execution_policy_posture} policy for "
            f"{route_name.value}."
        ),
        "Do not apply execution policies automatically.",
        "Keep workflow control, provider routing, and execution blocking disabled.",
    )
