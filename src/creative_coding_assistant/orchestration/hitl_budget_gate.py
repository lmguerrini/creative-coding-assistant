"""V5.2 advisory HITL budget gate metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.budget_policies import (
    BudgetPolicyDecision,
    BudgetPolicyPlan,
    BudgetPolicyPosture,
    evaluate_budget_policies,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

HitlBudgetGateStatus = Literal["not_required", "review_recommended", "required"]
HitlBudgetGateDecisionStatus = Literal["recommended", "fallback"]

HITL_BUDGET_GATE_DECISION_SERIALIZATION_VERSION = "hitl_budget_gate_decision.v1"
HITL_BUDGET_GATE_PLAN_SERIALIZATION_VERSION = "hitl_budget_gate_plan.v1"
HITL_BUDGET_GATE_AUTHORITY_BOUNDARY = (
    "The V5.2 HITL Budget Gate surface derives advisory HITL budget review "
    "posture from budget policy metadata only; it does not emit human input "
    "requests, block execution, enforce budgets, select or switch providers "
    "or models, execute providers, control workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
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


class HitlBudgetGateDecision(BaseModel):
    """One advisory HITL budget gate decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    gate_id: str = Field(min_length=1, max_length=180)
    source_budget_policy_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    budget_posture: BudgetPolicyPosture
    gate_status: HitlBudgetGateStatus
    operator_review_reason: str = Field(min_length=1, max_length=260)
    status: HitlBudgetGateDecisionStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hitl_budget_gate_implemented: Literal[True] = True
    hitl_request_emitted: Literal[False] = False
    human_input_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hitl_budget_gate_decision.v1"] = (
        HITL_BUDGET_GATE_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_budget_posture(self) -> Self:
        if self.gate_status != _gate_status(self.budget_posture):
            raise ValueError("gate_status must match budget_posture")
        return self


class HitlBudgetGatePlan(BaseModel):
    """Bounded V5.2 advisory HITL budget gate plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hitl_budget_gate"] = "hitl_budget_gate"
    serialization_version: Literal["hitl_budget_gate_plan.v1"] = (
        HITL_BUDGET_GATE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HITL_BUDGET_GATE_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_budget_policy_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_budget_policy_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    decisions: tuple[HitlBudgetGateDecision, ...] = Field(min_length=1, max_length=12)
    gate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_gate_id: str = Field(min_length=1, max_length=180)
    recommended_gate_status: HitlBudgetGateStatus
    fallback_gate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    gate_count: int = Field(ge=1, le=12)
    review_recommended_count: int = Field(ge=0, le=12)
    required_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    hitl_budget_gate_implemented: Literal[True] = True
    hitl_request_emitted: Literal[False] = False
    human_input_blocking_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
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
        derived_gate_ids = tuple(decision.gate_id for decision in self.decisions)
        if len(set(derived_gate_ids)) != len(derived_gate_ids):
            raise ValueError("gate_ids must be unique")
        if self.gate_ids != derived_gate_ids:
            raise ValueError("gate_ids must match decisions")
        if self.gate_count != len(self.decisions):
            raise ValueError("gate_count must match decisions")
        if self.source_budget_policy_ids != tuple(
            decision.source_budget_policy_id for decision in self.decisions
        ):
            raise ValueError("source_budget_policy_ids must match decisions")

        recommended = tuple(
            decision for decision in self.decisions if decision.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended gate is required")
        recommended_gate = recommended[0]
        if self.recommended_gate_id != recommended_gate.gate_id:
            raise ValueError("recommended_gate_id must match decision")
        if self.recommended_gate_status != recommended_gate.gate_status:
            raise ValueError("recommended_gate_status must match decision")
        if self.fallback_gate_ids != tuple(
            decision.gate_id
            for decision in self.decisions
            if decision.status == "fallback"
        ):
            raise ValueError("fallback_gate_ids must match decisions")
        if self.review_recommended_count != sum(
            1
            for decision in self.decisions
            if decision.gate_status == "review_recommended"
        ):
            raise ValueError("review_recommended_count must match decisions")
        if self.required_count != sum(
            1 for decision in self.decisions if decision.gate_status == "required"
        ):
            raise ValueError("required_count must match decisions")
        for decision in self.decisions:
            if decision.route_name != self.route_name:
                raise ValueError("decision route_name must match plan route_name")
        return self


def evaluate_hitl_budget_gate(
    *,
    budget_policies: BudgetPolicyPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> HitlBudgetGatePlan:
    """Return advisory HITL budget gate metadata without emitting HITL requests."""

    policy_plan = budget_policies or evaluate_budget_policies(
        route_decision=route_decision,
        route=route,
    )
    decisions = tuple(
        _decision_from_budget_policy(decision) for decision in policy_plan.decisions
    )
    recommended = _recommended_gate(decisions)

    return HitlBudgetGatePlan(
        source_budget_policy_serialization_version=policy_plan.serialization_version,
        route_name=policy_plan.route_name,
        source_budget_policy_ids=tuple(
            decision.source_budget_policy_id for decision in decisions
        ),
        decisions=decisions,
        gate_ids=tuple(decision.gate_id for decision in decisions),
        recommended_gate_id=recommended.gate_id,
        recommended_gate_status=recommended.gate_status,
        fallback_gate_ids=tuple(
            decision.gate_id for decision in decisions if decision.status == "fallback"
        ),
        gate_count=len(decisions),
        review_recommended_count=sum(
            1 for decision in decisions if decision.gate_status == "review_recommended"
        ),
        required_count=sum(
            1 for decision in decisions if decision.gate_status == "required"
        ),
        advisory_actions=_plan_actions(policy_plan.route_name, recommended),
    )


def hitl_budget_gate_by_id(
    gate_id: str,
    plan: HitlBudgetGatePlan | None = None,
) -> HitlBudgetGateDecision | None:
    """Return one advisory HITL budget gate decision without emitting HITL."""

    source_plan = plan or evaluate_hitl_budget_gate()
    for decision in source_plan.decisions:
        if decision.gate_id == gate_id:
            return decision
    return None


def hitl_budget_gates_for_status(
    gate_status: HitlBudgetGateStatus,
    plan: HitlBudgetGatePlan | None = None,
) -> tuple[HitlBudgetGateDecision, ...]:
    """Return advisory HITL budget gates for one status."""

    source_plan = plan or evaluate_hitl_budget_gate()
    return tuple(
        decision
        for decision in source_plan.decisions
        if decision.gate_status == gate_status
    )


def _decision_from_budget_policy(
    policy: BudgetPolicyDecision,
) -> HitlBudgetGateDecision:
    gate_status = _gate_status(policy.budget_posture)
    return HitlBudgetGateDecision(
        gate_id=f"hitl_budget_gate::{policy.source_model_profile_id}",
        source_budget_policy_id=policy.policy_id,
        source_model_profile_id=policy.source_model_profile_id,
        route_name=policy.route_name,
        budget_posture=policy.budget_posture,
        gate_status=gate_status,
        operator_review_reason=_review_reason(gate_status),
        status=policy.status,
        evidence=(
            f"Derived from {policy.policy_id}.",
            f"Budget posture: {policy.budget_posture}.",
            "No human request is emitted by this metadata surface.",
        ),
        advisory_actions=(
            "Surface HITL budget posture for review.",
            "Do not block execution or emit a human request.",
        ),
    )


def _gate_status(posture: BudgetPolicyPosture) -> HitlBudgetGateStatus:
    if posture == "over_budget":
        return "required"
    if posture == "review_recommended":
        return "review_recommended"
    return "not_required"


def _review_reason(status: HitlBudgetGateStatus) -> str:
    if status == "required":
        return "Estimated relative cost exceeds the hard budget threshold."
    if status == "review_recommended":
        return "Estimated relative cost exceeds the soft budget threshold."
    return "Estimated relative cost is inside advisory budget thresholds."


def _recommended_gate(
    decisions: tuple[HitlBudgetGateDecision, ...],
) -> HitlBudgetGateDecision:
    for decision in decisions:
        if decision.status == "recommended":
            return decision
    raise ValueError("HITL budget gate requires a recommended decision")


def _plan_actions(
    route_name: RouteName,
    recommended: HitlBudgetGateDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.gate_status} HITL budget posture for "
            f"{route_name.value}."
        ),
        "Keep human request emission and execution blocking disabled.",
        "Leave actual HITL interaction to explicit runtime integration.",
    )
