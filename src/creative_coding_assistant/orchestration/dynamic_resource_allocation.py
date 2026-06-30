"""V5.5 advisory dynamic resource allocation intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_cost_quality_optimizer import (
    AdaptiveCostQualityPlan,
    AdaptiveCostQualityPosture,
    optimize_adaptive_cost_quality,
)
from creative_coding_assistant.orchestration.adaptive_latency_optimizer import (
    AdaptiveLatencyPlan,
    AdaptiveLatencyPosture,
    optimize_adaptive_latency,
)
from creative_coding_assistant.orchestration.budget_policies import (
    BudgetPolicyPlan,
    BudgetPolicyPosture,
    evaluate_budget_policies,
)
from creative_coding_assistant.orchestration.dynamic_agent_allocation import (
    DynamicAgentAllocationPlan,
    allocate_dynamic_agents,
)
from creative_coding_assistant.orchestration.resource_utilization_optimizer import (
    ResourceUtilizationOptimizationPlan,
    ResourceUtilizationPressure,
    ResourceUtilizationRecommendation,
    ResourceUtilizationStatus,
    optimize_resource_utilization,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

DynamicResourceAllocationStatus = Literal[
    "recommended",
    "capacity_guardrail",
    "review_required",
    "boundary_guardrail",
]

DYNAMIC_RESOURCE_ALLOCATION_CANDIDATE_SERIALIZATION_VERSION = (
    "dynamic_resource_allocation_candidate.v1"
)
DYNAMIC_RESOURCE_ALLOCATION_PLAN_SERIALIZATION_VERSION = (
    "dynamic_resource_allocation_plan.v1"
)
DYNAMIC_RESOURCE_ALLOCATION_AUTHORITY_BOUNDARY = (
    "V5.5 dynamic resource allocation combines advisory resource utilization, "
    "dynamic agent allocation, adaptive cost/quality, adaptive latency, and "
    "budget policy metadata into inspectable resource allocation posture only; "
    "it does not allocate resources, measure CPU or memory, change concurrency "
    "limits, manage queues, autoscale capacity, enforce capacity or budgets, "
    "route providers or models, execute providers, control workflows, mutate "
    "workflow graphs, execute workflows, invoke agents, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "resource_allocation",
    "runtime_resource_measurement",
    "cpu_memory_measurement",
    "concurrency_limit_change",
    "queue_management_runtime",
    "autoscaling",
    "capacity_enforcement",
    "budget_enforcement",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class DynamicResourceAllocationCandidate(BaseModel):
    """One advisory V5.5 dynamic resource allocation candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    allocation_id: str = Field(min_length=1, max_length=180)
    source_resource_recommendation_id: str = Field(min_length=1, max_length=180)
    source_resource_utilization_id: str = Field(min_length=1, max_length=120)
    source_resource_status: ResourceUtilizationStatus
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_dynamic_agent_allocation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    source_dynamic_strategy_id: str = Field(min_length=1, max_length=180)
    source_cost_quality_candidate_id: str = Field(min_length=1, max_length=180)
    source_latency_candidate_id: str = Field(min_length=1, max_length=180)
    source_budget_policy_id: str = Field(min_length=1, max_length=180)
    allocation_status: DynamicResourceAllocationStatus
    resource_utilization_pressure: ResourceUtilizationPressure
    adaptive_cost_quality_posture: AdaptiveCostQualityPosture
    adaptive_latency_posture: AdaptiveLatencyPosture
    budget_posture: BudgetPolicyPosture
    advisory_resource_units: int = Field(ge=0, le=500_000)
    advisory_reserve_units: int = Field(ge=0, le=250_000)
    advisory_pressure_units: int = Field(ge=0, le=60_000)
    applied_resource_units: int = Field(ge=0, le=0)
    applied_reserve_units: int = Field(ge=0, le=0)
    resource_utilization_score: int = Field(ge=0, le=3_000)
    agent_allocation_score: int = Field(ge=0, le=320)
    cost_quality_score: int = Field(ge=0, le=240)
    latency_score: int = Field(ge=0, le=240)
    budget_posture_weight: int = Field(ge=0, le=80)
    guardrail_penalty: int = Field(ge=0, le=160)
    dynamic_resource_score: int = Field(ge=0, le=500)
    agent_allocation_count: int = Field(ge=1, le=12)
    hitl_required_agent_allocation_count: int = Field(ge=0, le=12)
    hitl_required: bool
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    dynamic_resource_allocation_implemented: Literal[True] = True
    resource_allocation_recommendation_implemented: Literal[True] = True
    resource_utilization_metadata_used: Literal[True] = True
    dynamic_agent_allocation_metadata_used: Literal[True] = True
    adaptive_cost_quality_metadata_used: Literal[True] = True
    adaptive_latency_metadata_used: Literal[True] = True
    budget_policy_metadata_used: Literal[True] = True
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    cpu_memory_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    autoscaling_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["dynamic_resource_allocation_candidate.v1"] = (
        DYNAMIC_RESOURCE_ALLOCATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.allocation_id != (
            f"dynamic_resource_allocation::{self.source_resource_utilization_id}"
        ):
            raise ValueError("allocation_id must match source_resource_utilization_id")
        if self.source_resource_recommendation_id != (
            f"resource_utilization::{self.source_resource_utilization_id}"
        ):
            raise ValueError("source_resource_recommendation_id must match utilization")
        if self.dynamic_resource_score != _dynamic_resource_score(
            resource_utilization_score=self.resource_utilization_score,
            agent_allocation_score=self.agent_allocation_score,
            cost_quality_score=self.cost_quality_score,
            latency_score=self.latency_score,
            budget_posture_weight=self.budget_posture_weight,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("dynamic_resource_score must combine source scores")
        if self.allocation_status != _allocation_status(self.source_resource_status):
            raise ValueError("allocation_status must match source resource status")
        if self.applied_resource_units or self.applied_reserve_units:
            raise ValueError("applied resource units must remain zero")
        if self.hitl_required_agent_allocation_count and not self.hitl_required:
            raise ValueError("HITL agent allocations require resource HITL posture")
        return self


class DynamicResourceAllocationPlan(BaseModel):
    """Bounded V5.5 advisory dynamic resource allocation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["dynamic_resource_allocator"] = "dynamic_resource_allocator"
    serialization_version: Literal["dynamic_resource_allocation_plan.v1"] = (
        DYNAMIC_RESOURCE_ALLOCATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=DYNAMIC_RESOURCE_ALLOCATION_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_resource_utilization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_agent_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_cost_quality_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_latency_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_budget_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    allocations: tuple[DynamicResourceAllocationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    allocation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    capacity_guardrail_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_required_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    boundary_guardrail_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    hitl_required_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    applied_resource_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    allocation_count: int = Field(ge=1, le=12)
    recommended_allocation_count: int = Field(ge=0, le=12)
    hitl_required_allocation_count: int = Field(ge=0, le=12)
    total_advisory_resource_units: int = Field(ge=0, le=1_000_000)
    total_advisory_reserve_units: int = Field(ge=0, le=500_000)
    total_advisory_pressure_units: int = Field(ge=0, le=60_000)
    total_applied_resource_units: int = Field(ge=0, le=0)
    highest_dynamic_resource_score: int = Field(ge=0, le=500)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    dynamic_resource_allocation_implemented: Literal[True] = True
    resource_allocation_recommendation_implemented: Literal[True] = True
    resource_utilization_metadata_used: Literal[True] = True
    dynamic_agent_allocation_metadata_used: Literal[True] = True
    adaptive_cost_quality_metadata_used: Literal[True] = True
    adaptive_latency_metadata_used: Literal[True] = True
    budget_policy_metadata_used: Literal[True] = True
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    cpu_memory_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    autoscaling_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_allocations(self) -> Self:
        derived_allocation_ids = tuple(
            allocation.allocation_id for allocation in self.allocations
        )
        if len(set(derived_allocation_ids)) != len(derived_allocation_ids):
            raise ValueError("allocation_ids must be unique")
        if self.allocation_ids != derived_allocation_ids:
            raise ValueError("allocation_ids must match allocations")
        if self.allocation_count != len(self.allocations):
            raise ValueError("allocation_count must match allocations")
        if self.recommended_allocation_ids != _allocation_ids_for_status(
            self.allocations,
            "recommended",
        ):
            raise ValueError("recommended_allocation_ids must match allocations")
        if self.capacity_guardrail_allocation_ids != _allocation_ids_for_status(
            self.allocations,
            "capacity_guardrail",
        ):
            raise ValueError("capacity_guardrail_allocation_ids must match allocations")
        if self.review_required_allocation_ids != _allocation_ids_for_status(
            self.allocations,
            "review_required",
        ):
            raise ValueError("review_required_allocation_ids must match allocations")
        if self.boundary_guardrail_allocation_ids != _allocation_ids_for_status(
            self.allocations,
            "boundary_guardrail",
        ):
            raise ValueError("boundary_guardrail_allocation_ids must match allocations")
        if self.hitl_required_allocation_ids != tuple(
            allocation.allocation_id
            for allocation in self.allocations
            if allocation.hitl_required
        ):
            raise ValueError("hitl_required_allocation_ids must match allocations")
        if self.applied_resource_allocation_ids:
            raise ValueError("applied_resource_allocation_ids must remain empty")
        if self.recommended_allocation_count != len(self.recommended_allocation_ids):
            raise ValueError("recommended_allocation_count must match allocations")
        if self.hitl_required_allocation_count != len(self.hitl_required_allocation_ids):
            raise ValueError("hitl_required_allocation_count must match allocations")
        if self.total_advisory_resource_units != sum(
            allocation.advisory_resource_units for allocation in self.allocations
        ):
            raise ValueError("total_advisory_resource_units must match allocations")
        if self.total_advisory_reserve_units != sum(
            allocation.advisory_reserve_units for allocation in self.allocations
        ):
            raise ValueError("total_advisory_reserve_units must match allocations")
        if self.total_advisory_pressure_units != sum(
            allocation.advisory_pressure_units for allocation in self.allocations
        ):
            raise ValueError("total_advisory_pressure_units must match allocations")
        if self.total_applied_resource_units != 0:
            raise ValueError("total_applied_resource_units must remain zero")
        if self.highest_dynamic_resource_score != max(
            allocation.dynamic_resource_score for allocation in self.allocations
        ):
            raise ValueError("highest_dynamic_resource_score must match allocations")
        for allocation in self.allocations:
            if allocation.route_name != self.route_name:
                raise ValueError("allocation route_name must match plan")
        return self


def allocate_dynamic_resources(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    resource_utilization: ResourceUtilizationOptimizationPlan | None = None,
    agent_allocation: DynamicAgentAllocationPlan | None = None,
    cost_quality: AdaptiveCostQualityPlan | None = None,
    latency: AdaptiveLatencyPlan | None = None,
    budget_policy: BudgetPolicyPlan | None = None,
) -> DynamicResourceAllocationPlan:
    """Recommend dynamic resource allocation metadata without applying it."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    agent_plan = agent_allocation or allocate_dynamic_agents(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or agent_plan.allocations[0].execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    resource_plan = resource_utilization or optimize_resource_utilization()
    cost_quality_plan = cost_quality or optimize_adaptive_cost_quality(
        route=route_name,
        task_type=agent_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    latency_plan = latency or optimize_adaptive_latency(
        route=route_name,
        task_type=agent_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    budget_plan = budget_policy or evaluate_budget_policies(route=route_name)
    recommended_budget = next(
        decision
        for decision in budget_plan.decisions
        if decision.policy_id == budget_plan.recommended_policy_id
    )
    allocations = tuple(
        _allocation_candidate(
            source=recommendation,
            route_name=route_name,
            task_type=agent_plan.task_type,
            execution_mode_id=normalized_mode,  # type: ignore[arg-type]
            agent_allocation=agent_plan,
            cost_quality=cost_quality_plan,
            latency=latency_plan,
            budget_policy=budget_plan,
            recommended_budget_policy_id=recommended_budget.policy_id,
        )
        for recommendation in resource_plan.recommendations
    )
    return DynamicResourceAllocationPlan(
        route_name=route_name,
        task_type=agent_plan.task_type,
        source_resource_utilization_serialization_version=resource_plan.serialization_version,
        source_dynamic_agent_allocation_serialization_version=(
            agent_plan.serialization_version
        ),
        source_adaptive_cost_quality_serialization_version=(
            cost_quality_plan.serialization_version
        ),
        source_adaptive_latency_serialization_version=latency_plan.serialization_version,
        source_budget_policy_serialization_version=budget_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        allocations=allocations,
        allocation_ids=tuple(allocation.allocation_id for allocation in allocations),
        recommended_allocation_ids=_allocation_ids_for_status(
            allocations,
            "recommended",
        ),
        capacity_guardrail_allocation_ids=_allocation_ids_for_status(
            allocations,
            "capacity_guardrail",
        ),
        review_required_allocation_ids=_allocation_ids_for_status(
            allocations,
            "review_required",
        ),
        boundary_guardrail_allocation_ids=_allocation_ids_for_status(
            allocations,
            "boundary_guardrail",
        ),
        hitl_required_allocation_ids=tuple(
            allocation.allocation_id for allocation in allocations if allocation.hitl_required
        ),
        applied_resource_allocation_ids=(),
        allocation_count=len(allocations),
        recommended_allocation_count=len(
            _allocation_ids_for_status(allocations, "recommended")
        ),
        hitl_required_allocation_count=sum(
            1 for allocation in allocations if allocation.hitl_required
        ),
        total_advisory_resource_units=sum(
            allocation.advisory_resource_units for allocation in allocations
        ),
        total_advisory_reserve_units=sum(
            allocation.advisory_reserve_units for allocation in allocations
        ),
        total_advisory_pressure_units=sum(
            allocation.advisory_pressure_units for allocation in allocations
        ),
        total_applied_resource_units=0,
        highest_dynamic_resource_score=max(
            allocation.dynamic_resource_score for allocation in allocations
        ),
        advisory_actions=_plan_actions(budget_plan),
    )


def dynamic_resource_allocation_by_id(
    allocation_id: str,
    plan: DynamicResourceAllocationPlan | None = None,
) -> DynamicResourceAllocationCandidate | None:
    """Return one dynamic resource allocation without applying it."""

    source_plan = plan or allocate_dynamic_resources()
    for allocation in source_plan.allocations:
        if allocation.allocation_id == allocation_id:
            return allocation
    return None


def dynamic_resource_allocations_for_status(
    status: DynamicResourceAllocationStatus,
    plan: DynamicResourceAllocationPlan | None = None,
) -> tuple[DynamicResourceAllocationCandidate, ...]:
    """Return dynamic resource allocations by advisory status."""

    source_plan = plan or allocate_dynamic_resources()
    return tuple(
        allocation for allocation in source_plan.allocations if allocation.allocation_status == status
    )


def _allocation_candidate(
    *,
    source: ResourceUtilizationRecommendation,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    agent_allocation: DynamicAgentAllocationPlan,
    cost_quality: AdaptiveCostQualityPlan,
    latency: AdaptiveLatencyPlan,
    budget_policy: BudgetPolicyPlan,
    recommended_budget_policy_id: str,
) -> DynamicResourceAllocationCandidate:
    source_agent_ids = _source_agent_allocation_ids(source, agent_allocation)
    guardrail_penalty = _guardrail_penalty(source.status)
    budget_weight = _budget_weight(budget_policy.recommended_budget_posture)
    hitl_count = len(agent_allocation.hitl_required_allocation_ids)
    hitl_required = (
        hitl_count > 0
        or source.status != "optimization_candidate"
        or budget_policy.recommended_budget_posture != "within_budget"
        or cost_quality.hitl_required_candidate_count > 0
        or latency.hitl_required_candidate_count > 0
    )
    return DynamicResourceAllocationCandidate(
        allocation_id=f"dynamic_resource_allocation::{source.utilization_id}",
        source_resource_recommendation_id=source.recommendation_id,
        source_resource_utilization_id=source.utilization_id,
        source_resource_status=source.status,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_dynamic_agent_allocation_ids=source_agent_ids,
        source_dynamic_strategy_id=agent_allocation.selected_dynamic_strategy_id,
        source_cost_quality_candidate_id=cost_quality.recommended_candidate_id,
        source_latency_candidate_id=latency.recommended_candidate_id,
        source_budget_policy_id=recommended_budget_policy_id,
        allocation_status=_allocation_status(source.status),
        resource_utilization_pressure=source.utilization_pressure,
        adaptive_cost_quality_posture=cost_quality.recommended_adaptive_posture,
        adaptive_latency_posture=latency.recommended_adaptive_latency_posture,
        budget_posture=budget_policy.recommended_budget_posture,
        advisory_resource_units=source.advisory_resource_units,
        advisory_reserve_units=source.advisory_reserve_units,
        advisory_pressure_units=source.advisory_pressure_units,
        applied_resource_units=0,
        applied_reserve_units=0,
        resource_utilization_score=source.advisory_utilization_score,
        agent_allocation_score=agent_allocation.highest_allocation_score,
        cost_quality_score=cost_quality.recommended_adaptive_score,
        latency_score=latency.recommended_adaptive_latency_score,
        budget_posture_weight=budget_weight,
        guardrail_penalty=guardrail_penalty,
        dynamic_resource_score=_dynamic_resource_score(
            resource_utilization_score=source.advisory_utilization_score,
            agent_allocation_score=agent_allocation.highest_allocation_score,
            cost_quality_score=cost_quality.recommended_adaptive_score,
            latency_score=latency.recommended_adaptive_latency_score,
            budget_posture_weight=budget_weight,
            guardrail_penalty=guardrail_penalty,
        ),
        agent_allocation_count=agent_allocation.allocation_count,
        hitl_required_agent_allocation_count=hitl_count,
        hitl_required=hitl_required,
        fallback_summary=_fallback_summary(source.status),
        advisory_actions=_candidate_actions(source.status),
        evidence=(
            f"resource_recommendation:{source.recommendation_id}",
            f"agent_allocations:{len(source_agent_ids)}",
            f"strategy:{agent_allocation.selected_dynamic_strategy_id}",
            f"cost_quality:{cost_quality.recommended_candidate_id}",
            f"latency:{latency.recommended_candidate_id}",
            f"budget_posture:{budget_policy.recommended_budget_posture}",
        ),
    )


def _dynamic_resource_score(
    *,
    resource_utilization_score: int,
    agent_allocation_score: int,
    cost_quality_score: int,
    latency_score: int,
    budget_posture_weight: int,
    guardrail_penalty: int,
) -> int:
    return min(
        500,
        max(
            0,
            resource_utilization_score // 8
            + agent_allocation_score // 3
            + cost_quality_score // 2
            + latency_score // 2
            + budget_posture_weight
            - guardrail_penalty,
        ),
    )


def _source_agent_allocation_ids(
    source: ResourceUtilizationRecommendation,
    agent_allocation: DynamicAgentAllocationPlan,
) -> tuple[str, ...]:
    if source.status == "optimization_candidate":
        return agent_allocation.primary_allocation_ids or agent_allocation.allocation_ids[:3]
    if source.status == "boundary_guardrail":
        return agent_allocation.primary_allocation_ids or agent_allocation.allocation_ids[:1]
    return agent_allocation.standby_allocation_ids[:3] or agent_allocation.allocation_ids[:3]


def _allocation_status(
    source_status: ResourceUtilizationStatus,
) -> DynamicResourceAllocationStatus:
    if source_status == "optimization_candidate":
        return "recommended"
    if source_status == "capacity_guardrail":
        return "capacity_guardrail"
    if source_status == "review_guardrail":
        return "review_required"
    return "boundary_guardrail"


def _allocation_ids_for_status(
    allocations: tuple[DynamicResourceAllocationCandidate, ...],
    status: DynamicResourceAllocationStatus,
) -> tuple[str, ...]:
    return tuple(
        allocation.allocation_id
        for allocation in allocations
        if allocation.allocation_status == status
    )


def _budget_weight(
    posture: BudgetPolicyPosture,
) -> int:
    if posture == "within_budget":
        return 56
    if posture == "review_recommended":
        return 28
    return 0


def _guardrail_penalty(
    source_status: ResourceUtilizationStatus,
) -> int:
    if source_status == "optimization_candidate":
        return 0
    if source_status == "capacity_guardrail":
        return 48
    if source_status == "review_guardrail":
        return 72
    return 140


def _fallback_summary(
    source_status: ResourceUtilizationStatus,
) -> str:
    if source_status == "optimization_candidate":
        return "Keep resource allocation metadata advisory until runtime authority exists."
    if source_status == "capacity_guardrail":
        return "Keep capacity-sensitive resources detached from enforcement."
    if source_status == "review_guardrail":
        return "Keep resource pressure in review-only posture."
    return "Preserve runtime resource boundary without applying allocation."


def _candidate_actions(
    source_status: ResourceUtilizationStatus,
) -> tuple[str, ...]:
    return (
        f"Surface {source_status} resource posture as advisory allocation metadata.",
        "Keep resource allocation, measurement, capacity enforcement, workflow, provider, agent, storage, and output behavior disabled.",
    )


def _plan_actions(
    budget_policy: BudgetPolicyPlan,
) -> tuple[str, ...]:
    actions = [
        "Expose dynamic resource allocation posture as advisory metadata only.",
        "Keep applied resource allocation ids empty and applied units at zero.",
        "Preserve measurement, capacity, budget, provider, workflow, agent, storage, and output boundaries.",
    ]
    if budget_policy.recommended_budget_posture != "within_budget":
        actions.append("Require budget review before any future allocation behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
