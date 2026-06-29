"""V5.2 advisory budget policy metadata for routing cost estimates."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cost_estimator import (
    CostEstimateScenario,
    CostEstimationPlan,
    estimate_routing_cost,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

BudgetPolicyPosture = Literal["within_budget", "review_recommended", "over_budget"]
BudgetPolicyStatus = Literal["recommended", "fallback"]

BUDGET_POLICY_DECISION_SERIALIZATION_VERSION = "budget_policy_decision.v1"
BUDGET_POLICY_PLAN_SERIALIZATION_VERSION = "budget_policy_plan.v1"
BUDGET_POLICY_AUTHORITY_BOUNDARY = (
    "The V5.2 Budget Policies surface evaluates advisory budget posture from "
    "bounded relative cost estimates only; it does not enforce budgets, "
    "request HITL, block execution, select or switch providers or models, "
    "execute providers, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "budget_enforcement",
    "hitl_request",
    "execution_blocking",
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
_SOFT_LIMIT_UNITS = 5
_HARD_LIMIT_UNITS = 7


class BudgetPolicyDecision(BaseModel):
    """One advisory budget policy decision for a cost estimate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=180)
    source_cost_scenario_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    soft_limit_units: int = Field(ge=0, le=100)
    hard_limit_units: int = Field(ge=0, le=100)
    estimated_max_cost_units: int = Field(ge=0, le=100)
    budget_margin_units: int = Field(ge=-100, le=100)
    budget_posture: BudgetPolicyPosture
    status: BudgetPolicyStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    budget_policy_implemented: Literal[True] = True
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["budget_policy_decision.v1"] = (
        BUDGET_POLICY_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_policy_thresholds(self) -> Self:
        if self.soft_limit_units > self.hard_limit_units:
            raise ValueError("soft_limit_units must not exceed hard_limit_units")
        if self.budget_margin_units != (
            self.hard_limit_units - self.estimated_max_cost_units
        ):
            raise ValueError("budget_margin_units must match hard limit")
        if self.budget_posture != _posture(
            max_cost_units=self.estimated_max_cost_units,
            soft_limit_units=self.soft_limit_units,
            hard_limit_units=self.hard_limit_units,
        ):
            raise ValueError("budget_posture must match thresholds")
        return self


class BudgetPolicyPlan(BaseModel):
    """Bounded V5.2 advisory budget policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["budget_policy_evaluator"] = "budget_policy_evaluator"
    serialization_version: Literal["budget_policy_plan.v1"] = (
        BUDGET_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=BUDGET_POLICY_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_cost_estimation_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_cost_scenario_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    decisions: tuple[BudgetPolicyDecision, ...] = Field(min_length=1, max_length=12)
    policy_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_policy_id: str = Field(min_length=1, max_length=180)
    recommended_budget_posture: BudgetPolicyPosture
    fallback_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    policy_count: int = Field(ge=1, le=12)
    review_recommended_count: int = Field(ge=0, le=12)
    over_budget_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    budget_policy_implemented: Literal[True] = True
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
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
    def _plan_matches_decisions(self) -> Self:
        derived_policy_ids = tuple(decision.policy_id for decision in self.decisions)
        if len(set(derived_policy_ids)) != len(derived_policy_ids):
            raise ValueError("policy_ids must be unique")
        if self.policy_ids != derived_policy_ids:
            raise ValueError("policy_ids must match decisions")
        if self.policy_count != len(self.decisions):
            raise ValueError("policy_count must match decisions")
        if self.source_cost_scenario_ids != tuple(
            decision.source_cost_scenario_id for decision in self.decisions
        ):
            raise ValueError("source_cost_scenario_ids must match decisions")

        recommended = tuple(
            decision for decision in self.decisions if decision.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended policy is required")
        recommended_policy = recommended[0]
        if self.recommended_policy_id != recommended_policy.policy_id:
            raise ValueError("recommended_policy_id must match decision")
        if self.recommended_budget_posture != recommended_policy.budget_posture:
            raise ValueError("recommended_budget_posture must match decision")
        if self.fallback_policy_ids != tuple(
            decision.policy_id
            for decision in self.decisions
            if decision.status == "fallback"
        ):
            raise ValueError("fallback_policy_ids must match decisions")
        if self.review_recommended_count != sum(
            1
            for decision in self.decisions
            if decision.budget_posture == "review_recommended"
        ):
            raise ValueError("review_recommended_count must match decisions")
        if self.over_budget_count != sum(
            1 for decision in self.decisions if decision.budget_posture == "over_budget"
        ):
            raise ValueError("over_budget_count must match decisions")
        for decision in self.decisions:
            if decision.route_name != self.route_name:
                raise ValueError("decision route_name must match plan route_name")
        return self


def evaluate_budget_policies(
    *,
    cost_estimation: CostEstimationPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> BudgetPolicyPlan:
    """Return advisory budget policy posture without enforcing it."""

    cost_plan = cost_estimation or estimate_routing_cost(
        route_decision=route_decision,
        route=route,
    )
    decisions = tuple(
        _decision_from_scenario(scenario) for scenario in cost_plan.scenarios
    )
    recommended = _recommended_policy(decisions)

    return BudgetPolicyPlan(
        source_cost_estimation_serialization_version=cost_plan.serialization_version,
        route_name=cost_plan.route_name,
        source_cost_scenario_ids=tuple(
            decision.source_cost_scenario_id for decision in decisions
        ),
        decisions=decisions,
        policy_ids=tuple(decision.policy_id for decision in decisions),
        recommended_policy_id=recommended.policy_id,
        recommended_budget_posture=recommended.budget_posture,
        fallback_policy_ids=tuple(
            decision.policy_id for decision in decisions if decision.status == "fallback"
        ),
        policy_count=len(decisions),
        review_recommended_count=sum(
            1 for decision in decisions if decision.budget_posture == "review_recommended"
        ),
        over_budget_count=sum(
            1 for decision in decisions if decision.budget_posture == "over_budget"
        ),
        advisory_actions=_plan_actions(cost_plan.route_name, recommended),
    )


def budget_policy_by_id(
    policy_id: str,
    plan: BudgetPolicyPlan | None = None,
) -> BudgetPolicyDecision | None:
    """Return one advisory budget policy decision without enforcing it."""

    source_plan = plan or evaluate_budget_policies()
    for decision in source_plan.decisions:
        if decision.policy_id == policy_id:
            return decision
    return None


def budget_policies_for_posture(
    posture: BudgetPolicyPosture,
    plan: BudgetPolicyPlan | None = None,
) -> tuple[BudgetPolicyDecision, ...]:
    """Return advisory budget policy decisions for a posture."""

    source_plan = plan or evaluate_budget_policies()
    return tuple(
        decision
        for decision in source_plan.decisions
        if decision.budget_posture == posture
    )


def _decision_from_scenario(scenario: CostEstimateScenario) -> BudgetPolicyDecision:
    return BudgetPolicyDecision(
        policy_id=f"budget_policy::{scenario.source_model_profile_id}",
        source_cost_scenario_id=scenario.scenario_id,
        source_model_profile_id=scenario.source_model_profile_id,
        route_name=scenario.route_name,
        soft_limit_units=_SOFT_LIMIT_UNITS,
        hard_limit_units=_HARD_LIMIT_UNITS,
        estimated_max_cost_units=scenario.estimated_max_cost_units,
        budget_margin_units=_HARD_LIMIT_UNITS - scenario.estimated_max_cost_units,
        budget_posture=_posture(
            max_cost_units=scenario.estimated_max_cost_units,
            soft_limit_units=_SOFT_LIMIT_UNITS,
            hard_limit_units=_HARD_LIMIT_UNITS,
        ),
        status=scenario.status,
        evidence=(
            f"Derived from {scenario.scenario_id}.",
            f"Estimated max relative cost units: {scenario.estimated_max_cost_units}.",
            "Budget posture is advisory only.",
        ),
        advisory_actions=(
            "Surface budget posture for review.",
            "Do not enforce budgets or block execution.",
        ),
    )


def _posture(
    *,
    max_cost_units: int,
    soft_limit_units: int,
    hard_limit_units: int,
) -> BudgetPolicyPosture:
    if max_cost_units > hard_limit_units:
        return "over_budget"
    if max_cost_units > soft_limit_units:
        return "review_recommended"
    return "within_budget"


def _recommended_policy(
    decisions: tuple[BudgetPolicyDecision, ...],
) -> BudgetPolicyDecision:
    for decision in decisions:
        if decision.status == "recommended":
            return decision
    raise ValueError("budget policies require a recommended decision")


def _plan_actions(
    route_name: RouteName,
    recommended: BudgetPolicyDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.budget_posture} budget posture for "
            f"{route_name.value}."
        ),
        "Keep budget enforcement and execution blocking disabled.",
        "Defer HITL budget gate behavior to the scoped HITL task.",
    )
