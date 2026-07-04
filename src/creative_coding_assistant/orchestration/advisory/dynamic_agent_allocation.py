"""V5.5 advisory dynamic agent allocation intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategySelectionPlan,
    DynamicExecutionStrategyKind,
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.agent_activation_optimizer import (
    AgentActivationOptimizationCandidate,
    AgentActivationOptimizationPlan,
    AgentActivationStatus,
    optimize_agent_activation,
)
from creative_coding_assistant.orchestration.agent_dependency_graph import (
    AgentDependencyGraphRegistry,
    agent_dependency_graph_registry,
    agent_dependency_node_by_id,
)
from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    ParallelSchedulingRegistry,
    SchedulingHint,
    parallel_scheduling_group_for_agent,
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.agent_routing import (
    AgentRoutingPriorityBand,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

DynamicAgentAllocationLane = Literal[
    "strategy_primary",
    "strategy_support",
    "standby_pool",
]
DynamicAgentAllocationStatus = Literal[
    "allocated_metadata",
    "requires_hitl",
    "standby",
]

DYNAMIC_AGENT_ALLOCATION_CANDIDATE_SERIALIZATION_VERSION = (
    "dynamic_agent_allocation_candidate.v1"
)
DYNAMIC_AGENT_ALLOCATION_PLAN_SERIALIZATION_VERSION = "dynamic_agent_allocation_plan.v1"
DYNAMIC_AGENT_ALLOCATION_AUTHORITY_BOUNDARY = (
    "V5.5 dynamic agent allocation combines advisory agent activation, "
    "dynamic execution strategy selection, passive parallel scheduling, and "
    "dependency graph metadata into inspectable agent allocation metadata "
    "only; it does not allocate agents at runtime, instantiate or invoke "
    "agents, change lifecycle state, run schedulers, execute parallel tasks, "
    "change workflow routing or timing, route providers or models, select "
    "runtimes, emit HITL requests, enforce budgets, mutate memory, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_agent_allocation",
    "agent_instantiation",
    "agent_invocation",
    "agent_activation",
    "lifecycle_transition_execution",
    "graph_scheduler_execution",
    "parallel_task_execution",
    "async_behavior_change",
    "workflow_routing_change",
    "workflow_timing_change",
    "workflow_control",
    "provider_or_model_routing",
    "runtime_selection",
    "human_review_request",
    "hitl_request_emission",
    "budget_enforcement",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class DynamicAgentAllocationCandidate(BaseModel):
    """One advisory dynamic agent allocation candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    allocation_id: str = Field(min_length=1, max_length=180)
    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_activation_candidate_id: str = Field(min_length=1, max_length=160)
    source_activation_status: AgentActivationStatus
    source_dynamic_strategy_id: str = Field(min_length=1, max_length=180)
    source_dynamic_strategy_kind: DynamicExecutionStrategyKind
    source_scheduling_group_id: str = Field(min_length=1, max_length=120)
    source_dependency_node_id: str = Field(min_length=1, max_length=160)
    priority_band: AgentRoutingPriorityBand
    allocation_lane: DynamicAgentAllocationLane
    allocation_status: DynamicAgentAllocationStatus
    activation_order: int = Field(ge=1, le=20)
    allocation_order: int = Field(ge=1, le=20)
    activation_score: int = Field(ge=0, le=240)
    strategy_score: int = Field(ge=0, le=400)
    scheduling_weight: int = Field(ge=0, le=40)
    capability_weight: int = Field(ge=0, le=40)
    hitl_penalty: int = Field(ge=0, le=80)
    allocation_score: int = Field(ge=0, le=320)
    scheduling_hint: SchedulingHint
    parallelizable: bool
    blocking_group_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    downstream_group_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    source_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    required_metadata_input_count: int = Field(ge=1, le=40)
    produced_metadata_output_count: int = Field(ge=1, le=40)
    upstream_dependency_count: int = Field(ge=0, le=32)
    downstream_dependency_count: int = Field(ge=0, le=32)
    hitl_required: bool
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    dynamic_agent_allocation_implemented: Literal[True] = True
    agent_allocation_recommendation_implemented: Literal[True] = True
    dynamic_execution_strategy_metadata_used: Literal[True] = True
    activation_metadata_used: Literal[True] = True
    scheduling_metadata_used: Literal[True] = True
    dependency_metadata_used: Literal[True] = True
    runtime_agent_allocation_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    lifecycle_transition_execution_implemented: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_behavior_changed: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_timing_changed: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["dynamic_agent_allocation_candidate.v1"] = (
        DYNAMIC_AGENT_ALLOCATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.allocation_id != f"dynamic_agent_allocation::{self.agent_id}":
            raise ValueError("allocation_id must match agent_id")
        if self.source_activation_candidate_id != f"agent_activation::{self.agent_id}":
            raise ValueError("source_activation_candidate_id must match agent_id")
        if self.source_dependency_node_id != f"agent::{self.agent_id}":
            raise ValueError("source_dependency_node_id must match agent_id")
        if self.allocation_order != self.activation_order:
            raise ValueError("allocation_order must follow activation_order")
        if self.allocation_score != _allocation_score(
            activation_score=self.activation_score,
            strategy_score=self.strategy_score,
            scheduling_weight=self.scheduling_weight,
            capability_weight=self.capability_weight,
            hitl_penalty=self.hitl_penalty,
        ):
            raise ValueError("allocation_score must combine source weights")
        if self.allocation_status == "requires_hitl" and not self.hitl_required:
            raise ValueError("requires_hitl allocations must set hitl_required")
        if self.source_activation_status == "standby" and (
            self.allocation_lane != "standby_pool"
        ):
            raise ValueError("standby activation candidates must remain standby")
        return self


class DynamicAgentAllocationPlan(BaseModel):
    """Bounded V5.5 advisory dynamic agent allocation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["dynamic_agent_allocator"] = "dynamic_agent_allocator"
    serialization_version: Literal["dynamic_agent_allocation_plan.v1"] = (
        DYNAMIC_AGENT_ALLOCATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=DYNAMIC_AGENT_ALLOCATION_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_agent_activation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_execution_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_parallel_scheduling_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dependency_graph_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    selected_dynamic_strategy_id: str = Field(min_length=1, max_length=180)
    selected_dynamic_strategy_kind: DynamicExecutionStrategyKind
    allocations: tuple[DynamicAgentAllocationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    allocation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    primary_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    support_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    standby_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    hitl_required_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    applied_allocation_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    allocation_count: int = Field(ge=1, le=12)
    primary_allocation_count: int = Field(ge=0, le=12)
    hitl_required_allocation_count: int = Field(ge=0, le=12)
    highest_allocation_score: int = Field(ge=0, le=320)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    dynamic_agent_allocation_implemented: Literal[True] = True
    agent_allocation_recommendation_implemented: Literal[True] = True
    dynamic_execution_strategy_metadata_used: Literal[True] = True
    activation_metadata_used: Literal[True] = True
    scheduling_metadata_used: Literal[True] = True
    dependency_metadata_used: Literal[True] = True
    runtime_agent_allocation_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    lifecycle_transition_execution_implemented: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_behavior_changed: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    workflow_timing_changed: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
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
        if self.primary_allocation_ids != _allocation_ids_for_lane(
            self.allocations,
            "strategy_primary",
        ):
            raise ValueError("primary_allocation_ids must match allocations")
        if self.support_allocation_ids != _allocation_ids_for_lane(
            self.allocations,
            "strategy_support",
        ):
            raise ValueError("support_allocation_ids must match allocations")
        if self.standby_allocation_ids != _allocation_ids_for_lane(
            self.allocations,
            "standby_pool",
        ):
            raise ValueError("standby_allocation_ids must match allocations")
        if self.hitl_required_allocation_ids != tuple(
            allocation.allocation_id
            for allocation in self.allocations
            if allocation.hitl_required
        ):
            raise ValueError("hitl_required_allocation_ids must match allocations")
        if self.applied_allocation_ids:
            raise ValueError("applied_allocation_ids must remain empty")
        if self.primary_allocation_count != len(self.primary_allocation_ids):
            raise ValueError("primary_allocation_count must match allocations")
        if self.hitl_required_allocation_count != len(
            self.hitl_required_allocation_ids
        ):
            raise ValueError("hitl_required_allocation_count must match allocations")
        if self.highest_allocation_score != max(
            allocation.allocation_score for allocation in self.allocations
        ):
            raise ValueError("highest_allocation_score must match allocations")
        for allocation in self.allocations:
            if allocation.route_name != self.route_name:
                raise ValueError("allocation route_name must match plan")
            if (
                allocation.source_dynamic_strategy_id
                != self.selected_dynamic_strategy_id
            ):
                raise ValueError("allocation strategy id must match plan")
        return self


def allocate_dynamic_agents(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    activation: AgentActivationOptimizationPlan | None = None,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan | None = None,
    scheduling: ParallelSchedulingRegistry | None = None,
    dependency_graph: AgentDependencyGraphRegistry | None = None,
) -> DynamicAgentAllocationPlan:
    """Recommend dynamic agent allocation metadata without allocating agents."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    strategy_plan = dynamic_strategy or select_dynamic_execution_strategy(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or strategy_plan.selected_execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    activation_plan = activation or optimize_agent_activation(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    scheduling_registry = scheduling or parallel_scheduling_registry()
    dependency_registry = dependency_graph or agent_dependency_graph_registry()
    allocations = tuple(
        _allocation_candidate(
            activation_candidate=candidate,
            route_name=route_name,
            task_type=strategy_plan.task_type,
            execution_mode_id=normalized_mode,  # type: ignore[arg-type]
            strategy_plan=strategy_plan,
            scheduling=scheduling_registry,
            dependency_graph=dependency_registry,
        )
        for candidate in activation_plan.candidates
    )
    return DynamicAgentAllocationPlan(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        source_agent_activation_serialization_version=activation_plan.serialization_version,
        source_dynamic_execution_strategy_serialization_version=(
            strategy_plan.serialization_version
        ),
        source_parallel_scheduling_serialization_version=(
            scheduling_registry.serialization_version
        ),
        source_dependency_graph_serialization_version=dependency_registry.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        selected_dynamic_strategy_id=strategy_plan.selected_strategy_id,
        selected_dynamic_strategy_kind=strategy_plan.selected_strategy_kind,
        allocations=allocations,
        allocation_ids=tuple(allocation.allocation_id for allocation in allocations),
        primary_allocation_ids=_allocation_ids_for_lane(
            allocations, "strategy_primary"
        ),
        support_allocation_ids=_allocation_ids_for_lane(
            allocations, "strategy_support"
        ),
        standby_allocation_ids=_allocation_ids_for_lane(allocations, "standby_pool"),
        hitl_required_allocation_ids=tuple(
            allocation.allocation_id
            for allocation in allocations
            if allocation.hitl_required
        ),
        applied_allocation_ids=(),
        allocation_count=len(allocations),
        primary_allocation_count=len(
            _allocation_ids_for_lane(allocations, "strategy_primary")
        ),
        hitl_required_allocation_count=sum(
            1 for allocation in allocations if allocation.hitl_required
        ),
        highest_allocation_score=max(
            allocation.allocation_score for allocation in allocations
        ),
        advisory_actions=_plan_actions(strategy_plan),
    )


def dynamic_agent_allocation_by_agent_id(
    agent_id: str,
    plan: DynamicAgentAllocationPlan | None = None,
) -> DynamicAgentAllocationCandidate | None:
    """Return one dynamic agent allocation without applying it."""

    source_plan = plan or allocate_dynamic_agents()
    for allocation in source_plan.allocations:
        if allocation.agent_id == agent_id:
            return allocation
    return None


def dynamic_agent_allocations_for_lane(
    lane: DynamicAgentAllocationLane,
    plan: DynamicAgentAllocationPlan | None = None,
) -> tuple[DynamicAgentAllocationCandidate, ...]:
    """Return dynamic agent allocations for one advisory lane."""

    source_plan = plan or allocate_dynamic_agents()
    return tuple(
        allocation
        for allocation in source_plan.allocations
        if allocation.allocation_lane == lane
    )


def _allocation_candidate(
    *,
    activation_candidate: AgentActivationOptimizationCandidate,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
    scheduling: ParallelSchedulingRegistry,
    dependency_graph: AgentDependencyGraphRegistry,
) -> DynamicAgentAllocationCandidate:
    group = parallel_scheduling_group_for_agent(
        activation_candidate.agent_id, scheduling
    )
    node = agent_dependency_node_by_id(
        f"agent::{activation_candidate.agent_id}",
        dependency_graph,
    )
    if group is None or node is None:
        raise ValueError("required agent scheduling or dependency metadata is missing")

    lane = _lane_for_activation(activation_candidate)
    status = _status_for_activation(activation_candidate)
    hitl_required = (
        activation_candidate.hitl_required
        or strategy_plan.selected_strategy_hitl_required
        and lane == "strategy_primary"
    )
    scheduling_weight = _scheduling_weight(group.scheduling_hint)
    capability_weight = min(40, len(activation_candidate.source_capability_ids) * 8)
    hitl_penalty = 24 if hitl_required else 0
    return DynamicAgentAllocationCandidate(
        allocation_id=f"dynamic_agent_allocation::{activation_candidate.agent_id}",
        agent_id=activation_candidate.agent_id,
        role_id=activation_candidate.role_id,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_activation_candidate_id=activation_candidate.candidate_id,
        source_activation_status=activation_candidate.status,
        source_dynamic_strategy_id=strategy_plan.selected_strategy_id,
        source_dynamic_strategy_kind=strategy_plan.selected_strategy_kind,
        source_scheduling_group_id=group.group_id,
        source_dependency_node_id=node.node_id,
        priority_band=activation_candidate.priority_band,
        allocation_lane=lane,
        allocation_status=status,
        activation_order=activation_candidate.activation_order,
        allocation_order=activation_candidate.activation_order,
        activation_score=activation_candidate.activation_score,
        strategy_score=strategy_plan.selected_strategy_score,
        scheduling_weight=scheduling_weight,
        capability_weight=capability_weight,
        hitl_penalty=hitl_penalty,
        allocation_score=_allocation_score(
            activation_score=activation_candidate.activation_score,
            strategy_score=strategy_plan.selected_strategy_score,
            scheduling_weight=scheduling_weight,
            capability_weight=capability_weight,
            hitl_penalty=hitl_penalty,
        ),
        scheduling_hint=group.scheduling_hint,
        parallelizable=group.scheduling_hint == "parallel_after_upstream_dependencies",
        blocking_group_ids=group.blocking_group_ids,
        downstream_group_ids=group.downstream_group_ids,
        source_capability_ids=activation_candidate.source_capability_ids,
        required_metadata_input_count=(
            activation_candidate.required_metadata_input_count
        ),
        produced_metadata_output_count=(
            activation_candidate.produced_metadata_output_count
        ),
        upstream_dependency_count=len(node.upstream_node_ids),
        downstream_dependency_count=len(node.downstream_node_ids),
        hitl_required=hitl_required,
        fallback_summary=_fallback_summary(lane, activation_candidate),
        advisory_actions=_candidate_actions(lane),
        evidence=(
            f"activation_candidate:{activation_candidate.candidate_id}",
            f"dynamic_strategy:{strategy_plan.selected_strategy_id}",
            f"scheduling_group:{group.group_id}",
            f"dependency_node:{node.node_id}",
            f"capabilities:{len(activation_candidate.source_capability_ids)}",
        ),
    )


def _allocation_score(
    *,
    activation_score: int,
    strategy_score: int,
    scheduling_weight: int,
    capability_weight: int,
    hitl_penalty: int,
) -> int:
    return min(
        320,
        max(
            0,
            activation_score
            + strategy_score // 5
            + scheduling_weight
            + capability_weight
            - hitl_penalty,
        ),
    )


def _lane_for_activation(
    activation_candidate: AgentActivationOptimizationCandidate,
) -> DynamicAgentAllocationLane:
    if activation_candidate.status == "standby":
        return "standby_pool"
    if activation_candidate.activation_order <= 3:
        return "strategy_primary"
    return "strategy_support"


def _status_for_activation(
    activation_candidate: AgentActivationOptimizationCandidate,
) -> DynamicAgentAllocationStatus:
    if activation_candidate.status == "requires_hitl":
        return "requires_hitl"
    if activation_candidate.status == "standby":
        return "standby"
    return "allocated_metadata"


def _scheduling_weight(
    scheduling_hint: SchedulingHint,
) -> int:
    if scheduling_hint == "parallel_after_upstream_dependencies":
        return 28
    return 16


def _allocation_ids_for_lane(
    allocations: tuple[DynamicAgentAllocationCandidate, ...],
    lane: DynamicAgentAllocationLane,
) -> tuple[str, ...]:
    return tuple(
        allocation.allocation_id
        for allocation in allocations
        if allocation.allocation_lane == lane
    )


def _fallback_summary(
    lane: DynamicAgentAllocationLane,
    activation_candidate: AgentActivationOptimizationCandidate,
) -> str:
    if lane == "standby_pool":
        return activation_candidate.fallback_summary
    return (
        "Keep this agent allocation as metadata until HITL, scheduling, "
        "lifecycle, workflow, and runtime boundaries are explicitly approved."
    )


def _candidate_actions(
    lane: DynamicAgentAllocationLane,
) -> tuple[str, ...]:
    return (
        f"Surface {lane} agent allocation as advisory metadata only.",
        "Keep runtime allocation, activation, invocation, scheduling, workflow, provider, memory, storage, and output behavior disabled.",  # noqa: E501
    )


def _plan_actions(
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
) -> tuple[str, ...]:
    actions = [
        "Expose dynamic agent allocation lanes as advisory metadata only.",
        "Keep applied_allocation_ids empty until an explicit runtime contract exists.",
        "Preserve agent activation, invocation, scheduling, workflow, provider, memory, storage, and output boundaries.",  # noqa: E501
    ]
    if strategy_plan.selected_strategy_hitl_required:
        actions.append("Require HITL before any future runtime allocation behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
