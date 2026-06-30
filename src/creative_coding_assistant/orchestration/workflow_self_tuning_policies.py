"""V5.5 advisory workflow self-tuning policy intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategySelectionPlan,
    DynamicExecutionStrategyKind,
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.dynamic_resource_allocation import (
    DynamicResourceAllocationPlan,
    allocate_dynamic_resources,
    dynamic_resource_allocation_by_id,
)
from creative_coding_assistant.orchestration.load_balancer import (
    LoadBalancePressure,
    LoadBalancerPlan,
    load_balance_candidate_by_id,
    plan_load_balancer,
)
from creative_coding_assistant.orchestration.retry_policies import (
    RetryPolicyPlan,
    RetryPolicyPressure,
    retry_policy_candidate_by_id,
    plan_retry_policies,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

WorkflowSelfTuningPolicyKind = Literal[
    "retry_self_tuning_policy",
    "load_self_tuning_policy",
    "resource_self_tuning_policy",
    "strategy_guardrail_policy",
]
WorkflowSelfTuningPolicyStatus = Literal[
    "recommended",
    "review_required",
    "guardrail",
]

WORKFLOW_SELF_TUNING_POLICY_SERIALIZATION_VERSION = (
    "workflow_self_tuning_policy.v1"
)
WORKFLOW_SELF_TUNING_POLICY_PLAN_SERIALIZATION_VERSION = (
    "workflow_self_tuning_policy_plan.v1"
)
WORKFLOW_SELF_TUNING_POLICY_AUTHORITY_BOUNDARY = (
    "V5.5 workflow self-tuning policies combine advisory retry policy, load "
    "balancing, dynamic resource allocation, and dynamic execution strategy "
    "metadata into inspectable self-tuning policy recommendations only; they "
    "do not apply tuning policies, control workflows, mutate workflow graphs "
    "or order, compile or execute graphs, trigger retries or refinements, "
    "distribute requests, allocate resources, measure runtime resources, "
    "enforce capacity or budgets, route providers or models, execute "
    "providers, invoke agents or node handlers, mutate prompts, write storage, "
    "or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_self_tuning_application",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_order_change",
    "langgraph_compilation",
    "workflow_execution",
    "retry_triggering",
    "refinement_triggering",
    "request_distribution",
    "load_balancing_runtime",
    "resource_allocation",
    "runtime_resource_measurement",
    "capacity_enforcement",
    "budget_enforcement",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "node_handler_invocation",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class WorkflowSelfTuningPolicy(BaseModel):
    """One advisory workflow self-tuning policy recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=180)
    policy_kind: WorkflowSelfTuningPolicyKind
    status: WorkflowSelfTuningPolicyStatus
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_retry_policy_id: str = Field(min_length=1, max_length=180)
    source_load_balance_candidate_id: str = Field(min_length=1, max_length=180)
    source_dynamic_resource_allocation_id: str = Field(min_length=1, max_length=180)
    source_dynamic_strategy_id: str = Field(min_length=1, max_length=180)
    source_dynamic_strategy_kind: DynamicExecutionStrategyKind
    retry_policy_pressure: RetryPolicyPressure
    load_balancing_pressure: LoadBalancePressure
    dynamic_resource_score: int = Field(ge=0, le=500)
    dynamic_strategy_score: int = Field(ge=0, le=400)
    retry_policy_score: int = Field(ge=0, le=800)
    load_balance_score: int = Field(ge=0, le=2_000)
    guardrail_penalty: int = Field(ge=0, le=200)
    self_tuning_score: int = Field(ge=0, le=600)
    hitl_required: bool
    applied_policy: Literal[False] = False
    policy_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    workflow_self_tuning_policy_implemented: Literal[True] = True
    self_tuning_recommendation_implemented: Literal[True] = True
    retry_policy_metadata_used: Literal[True] = True
    load_balance_metadata_used: Literal[True] = True
    dynamic_resource_allocation_metadata_used: Literal[True] = True
    dynamic_execution_strategy_metadata_used: Literal[True] = True
    workflow_self_tuning_application_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_self_tuning_policy.v1"] = (
        WORKFLOW_SELF_TUNING_POLICY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _policy_matches_contract(self) -> Self:
        if self.policy_id != f"workflow_self_tuning::{self.policy_kind}":
            raise ValueError("policy_id must match policy_kind")
        if self.self_tuning_score != _self_tuning_score(
            retry_policy_score=self.retry_policy_score,
            load_balance_score=self.load_balance_score,
            dynamic_resource_score=self.dynamic_resource_score,
            dynamic_strategy_score=self.dynamic_strategy_score,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("self_tuning_score must combine source scores")
        if self.status == "guardrail" and not self.hitl_required:
            raise ValueError("guardrail policies require HITL posture")
        if self.applied_policy:
            raise ValueError("applied_policy must remain false")
        return self


class WorkflowSelfTuningPolicyPlan(BaseModel):
    """Bounded V5.5 advisory workflow self-tuning policy plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_self_tuning_policy_planner"] = (
        "workflow_self_tuning_policy_planner"
    )
    serialization_version: Literal["workflow_self_tuning_policy_plan.v1"] = (
        WORKFLOW_SELF_TUNING_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_SELF_TUNING_POLICY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_retry_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_load_balancer_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_resource_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_execution_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    policies: tuple[WorkflowSelfTuningPolicy, ...] = Field(
        min_length=4,
        max_length=4,
    )
    policy_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    recommended_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    review_required_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    guardrail_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    hitl_required_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    applied_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    policy_count: int = Field(ge=4, le=4)
    recommended_policy_count: int = Field(ge=0, le=4)
    hitl_required_policy_count: int = Field(ge=0, le=4)
    highest_self_tuning_score: int = Field(ge=0, le=600)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    workflow_self_tuning_policy_implemented: Literal[True] = True
    self_tuning_recommendation_implemented: Literal[True] = True
    retry_policy_metadata_used: Literal[True] = True
    load_balance_metadata_used: Literal[True] = True
    dynamic_resource_allocation_metadata_used: Literal[True] = True
    dynamic_execution_strategy_metadata_used: Literal[True] = True
    workflow_self_tuning_application_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    load_balancing_runtime_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
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
        if self.recommended_policy_ids != _policy_ids_for_status(
            self.policies,
            "recommended",
        ):
            raise ValueError("recommended_policy_ids must match policies")
        if self.review_required_policy_ids != _policy_ids_for_status(
            self.policies,
            "review_required",
        ):
            raise ValueError("review_required_policy_ids must match policies")
        if self.guardrail_policy_ids != _policy_ids_for_status(
            self.policies,
            "guardrail",
        ):
            raise ValueError("guardrail_policy_ids must match policies")
        if self.hitl_required_policy_ids != tuple(
            policy.policy_id for policy in self.policies if policy.hitl_required
        ):
            raise ValueError("hitl_required_policy_ids must match policies")
        if self.applied_policy_ids:
            raise ValueError("applied_policy_ids must remain empty")
        if self.recommended_policy_count != len(self.recommended_policy_ids):
            raise ValueError("recommended_policy_count must match policies")
        if self.hitl_required_policy_count != len(self.hitl_required_policy_ids):
            raise ValueError("hitl_required_policy_count must match policies")
        if self.highest_self_tuning_score != max(
            policy.self_tuning_score for policy in self.policies
        ):
            raise ValueError("highest_self_tuning_score must match policies")
        for policy in self.policies:
            if policy.route_name != self.route_name:
                raise ValueError("policy route_name must match plan")
        return self


def plan_workflow_self_tuning_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    retry_policy: RetryPolicyPlan | None = None,
    load_balancer: LoadBalancerPlan | None = None,
    dynamic_resource_allocation: DynamicResourceAllocationPlan | None = None,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan | None = None,
) -> WorkflowSelfTuningPolicyPlan:
    """Recommend workflow self-tuning policies without applying tuning."""

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

    retry_plan = retry_policy or plan_retry_policies()
    load_plan = load_balancer or plan_load_balancer(retry_policy=retry_plan)
    resource_plan = dynamic_resource_allocation or allocate_dynamic_resources(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    policies = _policies(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        retry_policy=retry_plan,
        load_balancer=load_plan,
        dynamic_resource_allocation=resource_plan,
        dynamic_strategy=strategy_plan,
    )
    return WorkflowSelfTuningPolicyPlan(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        source_retry_policy_serialization_version=retry_plan.serialization_version,
        source_load_balancer_serialization_version=load_plan.serialization_version,
        source_dynamic_resource_allocation_serialization_version=(
            resource_plan.serialization_version
        ),
        source_dynamic_execution_strategy_serialization_version=(
            strategy_plan.serialization_version
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
        policies=policies,
        policy_ids=tuple(policy.policy_id for policy in policies),
        recommended_policy_ids=_policy_ids_for_status(policies, "recommended"),
        review_required_policy_ids=_policy_ids_for_status(policies, "review_required"),
        guardrail_policy_ids=_policy_ids_for_status(policies, "guardrail"),
        hitl_required_policy_ids=tuple(
            policy.policy_id for policy in policies if policy.hitl_required
        ),
        applied_policy_ids=(),
        policy_count=len(policies),
        recommended_policy_count=len(_policy_ids_for_status(policies, "recommended")),
        hitl_required_policy_count=sum(1 for policy in policies if policy.hitl_required),
        highest_self_tuning_score=max(policy.self_tuning_score for policy in policies),
        advisory_actions=_plan_actions(strategy_plan),
    )


def workflow_self_tuning_policy_by_id(
    policy_id: str,
    plan: WorkflowSelfTuningPolicyPlan | None = None,
) -> WorkflowSelfTuningPolicy | None:
    """Return one workflow self-tuning policy without applying it."""

    source_plan = plan or plan_workflow_self_tuning_policies()
    for policy in source_plan.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


def workflow_self_tuning_policies_for_status(
    status: WorkflowSelfTuningPolicyStatus,
    plan: WorkflowSelfTuningPolicyPlan | None = None,
) -> tuple[WorkflowSelfTuningPolicy, ...]:
    """Return workflow self-tuning policies by advisory status."""

    source_plan = plan or plan_workflow_self_tuning_policies()
    return tuple(policy for policy in source_plan.policies if policy.status == status)


def _policies(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    retry_policy: RetryPolicyPlan,
    load_balancer: LoadBalancerPlan,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan,
) -> tuple[WorkflowSelfTuningPolicy, ...]:
    return (
        _policy(
            kind="retry_self_tuning_policy",
            status="review_required",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retry_policy=retry_policy,
            retry_policy_id="retry_policy::review_refinement",
            load_balancer=load_balancer,
            load_candidate_id="load_balancer::retry_capacity_reserve",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::reasoning_budget_utilization"
            ),
            dynamic_strategy=dynamic_strategy,
            guardrail_penalty=40,
        ),
        _policy(
            kind="load_self_tuning_policy",
            status="recommended",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retry_policy=retry_policy,
            retry_policy_id="retry_policy::review_refinement",
            load_balancer=load_balancer,
            load_candidate_id="load_balancer::async_slot_distribution",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::throughput_capacity_utilization"
            ),
            dynamic_strategy=dynamic_strategy,
            guardrail_penalty=0,
        ),
        _policy(
            kind="resource_self_tuning_policy",
            status="recommended",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retry_policy=retry_policy,
            retry_policy_id="retry_policy::cost_retry_reserve",
            load_balancer=load_balancer,
            load_candidate_id="load_balancer::latency_pressure_distribution",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::benchmark_workload_utilization"
            ),
            dynamic_strategy=dynamic_strategy,
            guardrail_penalty=20,
        ),
        _policy(
            kind="strategy_guardrail_policy",
            status="guardrail",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            retry_policy=retry_policy,
            retry_policy_id="retry_policy::generation_failure",
            load_balancer=load_balancer,
            load_candidate_id="load_balancer::provider_capacity_visibility",
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id="dynamic_resource_allocation::runtime_resource_boundary",
            dynamic_strategy=dynamic_strategy,
            guardrail_penalty=120,
        ),
    )


def _policy(
    *,
    kind: WorkflowSelfTuningPolicyKind,
    status: WorkflowSelfTuningPolicyStatus,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    retry_policy: RetryPolicyPlan,
    retry_policy_id: str,
    load_balancer: LoadBalancerPlan,
    load_candidate_id: str,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    resource_allocation_id: str,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan,
    guardrail_penalty: int,
) -> WorkflowSelfTuningPolicy:
    retry_candidate = retry_policy_candidate_by_id(retry_policy_id, retry_policy)
    load_candidate = load_balance_candidate_by_id(load_candidate_id, load_balancer)
    resource_allocation = dynamic_resource_allocation_by_id(
        resource_allocation_id,
        dynamic_resource_allocation,
    )
    if retry_candidate is None or load_candidate is None or resource_allocation is None:
        raise ValueError("required self-tuning source metadata is missing")
    hitl_required = (
        status == "guardrail"
        or dynamic_strategy.selected_strategy_hitl_required
        or resource_allocation.hitl_required
        or retry_policy.retry_policy_pressure == "high"
        or load_balancer.load_balancing_pressure == "guarded"
    )
    return WorkflowSelfTuningPolicy(
        policy_id=f"workflow_self_tuning::{kind}",
        policy_kind=kind,
        status=status,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_retry_policy_id=retry_candidate.candidate_id,
        source_load_balance_candidate_id=load_candidate.candidate_id,
        source_dynamic_resource_allocation_id=resource_allocation.allocation_id,
        source_dynamic_strategy_id=dynamic_strategy.selected_strategy_id,
        source_dynamic_strategy_kind=dynamic_strategy.selected_strategy_kind,
        retry_policy_pressure=retry_policy.retry_policy_pressure,
        load_balancing_pressure=load_balancer.load_balancing_pressure,
        dynamic_resource_score=resource_allocation.dynamic_resource_score,
        dynamic_strategy_score=dynamic_strategy.selected_strategy_score,
        retry_policy_score=retry_candidate.advisory_retry_score,
        load_balance_score=load_candidate.advisory_load_score,
        guardrail_penalty=guardrail_penalty,
        self_tuning_score=_self_tuning_score(
            retry_policy_score=retry_candidate.advisory_retry_score,
            load_balance_score=load_candidate.advisory_load_score,
            dynamic_resource_score=resource_allocation.dynamic_resource_score,
            dynamic_strategy_score=dynamic_strategy.selected_strategy_score,
            guardrail_penalty=guardrail_penalty,
        ),
        hitl_required=hitl_required,
        policy_summary=_policy_summary(kind),
        fallback_summary=_fallback_summary(status),
        advisory_actions=_policy_actions(kind),
        evidence=(
            f"retry_policy:{retry_candidate.candidate_id}",
            f"load_candidate:{load_candidate.candidate_id}",
            f"resource_allocation:{resource_allocation.allocation_id}",
            f"dynamic_strategy:{dynamic_strategy.selected_strategy_id}",
            f"hitl_required:{hitl_required}",
        ),
    )


def _self_tuning_score(
    *,
    retry_policy_score: int,
    load_balance_score: int,
    dynamic_resource_score: int,
    dynamic_strategy_score: int,
    guardrail_penalty: int,
) -> int:
    return min(
        600,
        max(
            0,
            retry_policy_score // 4
            + load_balance_score // 10
            + dynamic_resource_score // 3
            + dynamic_strategy_score // 8
            - guardrail_penalty,
        ),
    )


def _policy_ids_for_status(
    policies: tuple[WorkflowSelfTuningPolicy, ...],
    status: WorkflowSelfTuningPolicyStatus,
) -> tuple[str, ...]:
    return tuple(policy.policy_id for policy in policies if policy.status == status)


def _policy_summary(kind: WorkflowSelfTuningPolicyKind) -> str:
    return {
        "retry_self_tuning_policy": (
            "Review retry and refinement posture as self-tuning metadata only."
        ),
        "load_self_tuning_policy": (
            "Review async and load distribution posture as self-tuning metadata only."
        ),
        "resource_self_tuning_policy": (
            "Review resource pressure and capacity posture as self-tuning metadata only."
        ),
        "strategy_guardrail_policy": (
            "Keep dynamic strategy boundaries visible as self-tuning guardrails."
        ),
    }[kind]


def _fallback_summary(status: WorkflowSelfTuningPolicyStatus) -> str:
    if status == "recommended":
        return "Keep recommended self-tuning policy advisory until runtime authority exists."
    if status == "review_required":
        return "Require review before any future self-tuning behavior can be applied."
    return "Preserve guardrail policy without workflow mutation or execution."


def _policy_actions(kind: WorkflowSelfTuningPolicyKind) -> tuple[str, ...]:
    return (
        f"Surface {kind} as advisory workflow self-tuning policy metadata.",
        "Keep tuning application, workflow control, retries, load balancing, resources, providers, agents, storage, and output behavior disabled.",
    )


def _plan_actions(
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
) -> tuple[str, ...]:
    actions = [
        "Expose workflow self-tuning policies as advisory metadata only.",
        "Keep applied policy ids empty until explicit runtime authority exists.",
        "Preserve workflow, retry, load, resource, provider, agent, storage, and output boundaries.",
    ]
    if strategy_plan.selected_strategy_hitl_required:
        actions.append("Require HITL before any future workflow self-tuning behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
