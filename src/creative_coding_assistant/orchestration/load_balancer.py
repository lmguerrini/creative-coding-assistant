"""V5.3 advisory load balancer planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .async_execution import AsyncExecutionPlan, plan_async_execution
from .latency_optimizer import LatencyOptimizationPlan, optimize_latency
from .provider_capability_matrix import (
    ProviderCapabilityMatrix,
    build_provider_capability_matrix,
)
from .retry_policies import RetryPolicyPlan, plan_retry_policies

LoadBalanceKind = Literal[
    "async_slot_distribution",
    "latency_pressure_distribution",
    "retry_capacity_reserve",
    "provider_capacity_visibility",
]
LoadBalanceStatus = Literal[
    "balancing_candidate",
    "capacity_guardrail",
    "routing_guardrail",
]
LoadBalancePressure = Literal["low", "medium", "high", "guarded"]

LOAD_BALANCE_CANDIDATE_SERIALIZATION_VERSION = "load_balance_candidate.v1"
LOAD_BALANCER_PLAN_SERIALIZATION_VERSION = "load_balancer_plan.v1"
LOAD_BALANCER_AUTHORITY_BOUNDARY = (
    "Load balancer planning derives advisory load distribution candidates "
    "from async readiness, latency optimization, retry policy, and passive "
    "provider capability metadata only; it does not distribute requests, "
    "shape traffic, select providers or models, route providers or models, "
    "run async or parallel execution, alter workflow timing, mutate graph "
    "order, compile or execute workflow graphs, invoke agents or node "
    "handlers, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "request_distribution",
    "load_balancing_runtime",
    "traffic_shaping",
    "capacity_enforcement",
    "provider_selection_execution",
    "automatic_model_selection",
    "provider_or_model_routing",
    "parallel_task_execution",
    "async_runtime_execution",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class LoadBalanceCandidate(BaseModel):
    """One advisory V5.3 load balancing candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    balance_id: str = Field(min_length=1, max_length=120)
    balance_kind: LoadBalanceKind
    status: LoadBalanceStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    advisory_load_units: int = Field(ge=0, le=10_000)
    advisory_capacity_slots: int = Field(ge=0, le=120)
    advisory_load_score: int = Field(ge=0, le=2_000)
    load_pressure: LoadBalancePressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    load_balancer_planning_implemented: Literal[True] = True
    load_balancing_runtime_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    traffic_shaping_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_runtime_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["load_balance_candidate.v1"] = (
        LOAD_BALANCE_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_policy(self) -> Self:
        if self.candidate_id != f"load_balancer::{self.balance_id}":
            raise ValueError("candidate_id must match balance_id")
        expected_score = _load_score(
            status=self.status,
            load_units=self.advisory_load_units,
            capacity_slots=self.advisory_capacity_slots,
        )
        if self.advisory_load_score != expected_score:
            raise ValueError("advisory_load_score must match candidate inputs")
        if self.load_pressure != _load_pressure(
            status=self.status,
            load_score=self.advisory_load_score,
            capacity_slots=self.advisory_capacity_slots,
        ):
            raise ValueError("load_pressure must match candidate inputs")
        if self.status == "balancing_candidate" and self.advisory_load_units <= 0:
            raise ValueError("balancing candidates require advisory load units")
        if self.status == "routing_guardrail" and self.advisory_capacity_slots != 0:
            raise ValueError("routing guardrails must not declare capacity slots")
        return self


class LoadBalancerPlan(BaseModel):
    """Bounded V5.3 advisory load balancer plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["load_balancer"] = "load_balancer"
    serialization_version: Literal["load_balancer_plan.v1"] = (
        LOAD_BALANCER_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LOAD_BALANCER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_async_execution_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_retry_policy_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_provider_capability_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    provider_candidate_count: int = Field(ge=1, le=12)
    route_count: int = Field(ge=1, le=6)
    candidates: tuple[LoadBalanceCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    balancing_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    capacity_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    routing_guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    candidate_count: int = Field(ge=1, le=12)
    balancing_candidate_count: int = Field(ge=0, le=12)
    capacity_guardrail_count: int = Field(ge=0, le=12)
    routing_guardrail_count: int = Field(ge=0, le=12)
    total_advisory_load_units: int = Field(ge=0, le=20_000)
    max_advisory_capacity_slots: int = Field(ge=0, le=120)
    highest_advisory_load_score: int = Field(ge=0, le=2_000)
    total_advisory_load_score: int = Field(ge=0, le=20_000)
    load_balancing_pressure: LoadBalancePressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    load_balancer_planning_implemented: Literal[True] = True
    load_balancing_runtime_implemented: Literal[False] = False
    request_distribution_implemented: Literal[False] = False
    traffic_shaping_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_runtime_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.balancing_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "balancing_candidate",
        ):
            raise ValueError("balancing_candidate_ids must match candidates")
        if self.capacity_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "capacity_guardrail",
        ):
            raise ValueError("capacity_guardrail_candidate_ids must match candidates")
        if self.routing_guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "routing_guardrail",
        ):
            raise ValueError("routing_guardrail_candidate_ids must match candidates")
        if self.balancing_candidate_count != len(self.balancing_candidate_ids):
            raise ValueError("balancing_candidate_count must match candidates")
        if self.capacity_guardrail_count != len(
            self.capacity_guardrail_candidate_ids
        ):
            raise ValueError("capacity_guardrail_count must match candidates")
        if self.routing_guardrail_count != len(self.routing_guardrail_candidate_ids):
            raise ValueError("routing_guardrail_count must match candidates")

        expected_units = sum(
            candidate.advisory_load_units for candidate in self.candidates
        )
        if self.total_advisory_load_units != expected_units:
            raise ValueError("total_advisory_load_units must match candidates")
        expected_slots = max(
            candidate.advisory_capacity_slots for candidate in self.candidates
        )
        if self.max_advisory_capacity_slots != expected_slots:
            raise ValueError("max_advisory_capacity_slots must match candidates")
        expected_highest_score = max(
            candidate.advisory_load_score for candidate in self.candidates
        )
        if self.highest_advisory_load_score != expected_highest_score:
            raise ValueError("highest_advisory_load_score must match candidates")
        expected_total_score = sum(
            candidate.advisory_load_score for candidate in self.candidates
        )
        if self.total_advisory_load_score != expected_total_score:
            raise ValueError("total_advisory_load_score must match candidates")
        if self.load_balancing_pressure != _plan_pressure(
            candidates=self.candidates,
            highest_score=self.highest_advisory_load_score,
        ):
            raise ValueError("load_balancing_pressure must match candidates")
        return self


def plan_load_balancer(
    *,
    async_execution: AsyncExecutionPlan | None = None,
    latency_optimization: LatencyOptimizationPlan | None = None,
    retry_policy: RetryPolicyPlan | None = None,
    provider_capability: ProviderCapabilityMatrix | None = None,
) -> LoadBalancerPlan:
    """Plan advisory load balancing without distributing requests."""

    async_plan = async_execution or plan_async_execution()
    latency_plan = latency_optimization or optimize_latency()
    retry_plan = retry_policy or plan_retry_policies()
    provider_matrix = provider_capability or build_provider_capability_matrix()
    candidates = _candidates(
        async_plan=async_plan,
        latency_plan=latency_plan,
        retry_plan=retry_plan,
        provider_matrix=provider_matrix,
    )
    highest_score = max(candidate.advisory_load_score for candidate in candidates)

    return LoadBalancerPlan(
        source_async_execution_serialization_version=async_plan.serialization_version,
        source_latency_optimization_serialization_version=(
            latency_plan.serialization_version
        ),
        source_retry_policy_serialization_version=retry_plan.serialization_version,
        source_provider_capability_serialization_version=(
            provider_matrix.serialization_version
        ),
        provider_candidate_count=provider_matrix.provider_candidate_count,
        route_count=provider_matrix.route_count,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        balancing_candidate_ids=_candidate_ids_for_status(
            candidates,
            "balancing_candidate",
        ),
        capacity_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "capacity_guardrail",
        ),
        routing_guardrail_candidate_ids=_candidate_ids_for_status(
            candidates,
            "routing_guardrail",
        ),
        candidate_count=len(candidates),
        balancing_candidate_count=len(
            _candidate_ids_for_status(candidates, "balancing_candidate")
        ),
        capacity_guardrail_count=len(
            _candidate_ids_for_status(candidates, "capacity_guardrail")
        ),
        routing_guardrail_count=len(
            _candidate_ids_for_status(candidates, "routing_guardrail")
        ),
        total_advisory_load_units=sum(
            candidate.advisory_load_units for candidate in candidates
        ),
        max_advisory_capacity_slots=max(
            candidate.advisory_capacity_slots for candidate in candidates
        ),
        highest_advisory_load_score=highest_score,
        total_advisory_load_score=sum(
            candidate.advisory_load_score for candidate in candidates
        ),
        load_balancing_pressure=_plan_pressure(
            candidates=candidates,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(candidates),
    )


def load_balance_candidate_by_id(
    candidate_id: str,
    plan: LoadBalancerPlan | None = None,
) -> LoadBalanceCandidate | None:
    """Return one advisory load balancing candidate without routing."""

    source_plan = plan or plan_load_balancer()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def load_balance_candidates_for_status(
    status: LoadBalanceStatus,
    plan: LoadBalancerPlan | None = None,
) -> tuple[LoadBalanceCandidate, ...]:
    """Return load balancing candidates by status without distribution."""

    source_plan = plan or plan_load_balancer()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.status == status
    )


def _candidates(
    *,
    async_plan: AsyncExecutionPlan,
    latency_plan: LatencyOptimizationPlan,
    retry_plan: RetryPolicyPlan,
    provider_matrix: ProviderCapabilityMatrix,
) -> tuple[LoadBalanceCandidate, ...]:
    return (
        _candidate(
            balance_id="async_slot_distribution",
            kind="async_slot_distribution",
            status="balancing_candidate",
            source_id="async_execution_plan",
            source_serialization_version=async_plan.serialization_version,
            source_candidate_ids=async_plan.async_ready_candidate_ids,
            load_units=async_plan.total_advisory_async_slots,
            capacity_slots=async_plan.max_async_width,
            evidence=(
                f"async_ready_candidates:{async_plan.async_ready_candidate_count}",
                f"async_pressure:{async_plan.async_execution_pressure}",
            ),
        ),
        _candidate(
            balance_id="latency_pressure_distribution",
            kind="latency_pressure_distribution",
            status="balancing_candidate",
            source_id="latency_optimization_plan",
            source_serialization_version=latency_plan.serialization_version,
            source_candidate_ids=latency_plan.optimization_candidate_ids,
            load_units=latency_plan.total_blocking_input_count,
            capacity_slots=latency_plan.optimization_candidate_count,
            evidence=(
                f"latency_pressure:{latency_plan.latency_optimization_pressure}",
                f"blocking_inputs:{latency_plan.total_blocking_input_count}",
            ),
        ),
        _candidate(
            balance_id="retry_capacity_reserve",
            kind="retry_capacity_reserve",
            status="capacity_guardrail",
            source_id="retry_policy_plan",
            source_serialization_version=retry_plan.serialization_version,
            source_candidate_ids=retry_plan.bounded_retry_candidate_ids,
            load_units=retry_plan.max_retry_attempts,
            capacity_slots=retry_plan.max_retry_attempts,
            evidence=(
                f"retry_pressure:{retry_plan.retry_policy_pressure}",
                f"retry_budget_tokens:{retry_plan.total_retry_budget_tokens}",
            ),
        ),
        _candidate(
            balance_id="provider_capacity_visibility",
            kind="provider_capacity_visibility",
            status="routing_guardrail",
            source_id="provider_capability_matrix",
            source_serialization_version=provider_matrix.serialization_version,
            source_candidate_ids=provider_matrix.provider_candidate_ids,
            load_units=provider_matrix.provider_candidate_count,
            capacity_slots=0,
            evidence=(
                f"provider_candidates:{provider_matrix.provider_candidate_count}",
                f"route_count:{provider_matrix.route_count}",
            ),
        ),
    )


def _candidate(
    *,
    balance_id: str,
    kind: LoadBalanceKind,
    status: LoadBalanceStatus,
    source_id: str,
    source_serialization_version: str,
    source_candidate_ids: tuple[str, ...],
    load_units: int,
    capacity_slots: int,
    evidence: tuple[str, ...],
) -> LoadBalanceCandidate:
    return LoadBalanceCandidate(
        candidate_id=f"load_balancer::{balance_id}",
        balance_id=balance_id,
        balance_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_candidate_ids=source_candidate_ids,
        advisory_load_units=load_units,
        advisory_capacity_slots=capacity_slots,
        advisory_load_score=_load_score(
            status=status,
            load_units=load_units,
            capacity_slots=capacity_slots,
        ),
        load_pressure=_load_pressure(
            status=status,
            load_score=_load_score(
                status=status,
                load_units=load_units,
                capacity_slots=capacity_slots,
            ),
            capacity_slots=capacity_slots,
        ),
        evidence=evidence,
        advisory_actions=_candidate_actions(status),
    )


def _candidate_ids_for_status(
    candidates: tuple[LoadBalanceCandidate, ...],
    status: LoadBalanceStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if candidate.status == status
    )


def _load_score(
    *,
    status: LoadBalanceStatus,
    load_units: int,
    capacity_slots: int,
) -> int:
    if status == "routing_guardrail":
        return 0
    slot_score = capacity_slots * 100
    return min(2_000, load_units * 25 + slot_score)


def _load_pressure(
    *,
    status: LoadBalanceStatus,
    load_score: int,
    capacity_slots: int,
) -> LoadBalancePressure:
    if status == "routing_guardrail":
        return "guarded"
    if load_score >= 600 or capacity_slots >= 3:
        return "high"
    if load_score >= 250 or capacity_slots >= 2:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    candidates: tuple[LoadBalanceCandidate, ...],
    highest_score: int,
) -> LoadBalancePressure:
    if any(candidate.status == "routing_guardrail" for candidate in candidates):
        return "guarded"
    if highest_score >= 600:
        return "high"
    if highest_score >= 250:
        return "medium"
    return "low"


def _candidate_actions(status: LoadBalanceStatus) -> tuple[str, ...]:
    if status == "balancing_candidate":
        return (
            "Expose load distribution metadata for inspection only.",
            "Require explicit runtime authority before distributing requests.",
        )
    if status == "capacity_guardrail":
        return (
            "Expose capacity reserve metadata without enforcing capacity.",
        )
    return (
        "Preserve provider visibility without routing or selection.",
    )


def _plan_actions(
    candidates: tuple[LoadBalanceCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose load balancing posture as advisory metadata only.",
        "Preserve provider, model, execution, retry, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "routing_guardrail"):
        actions.append("Keep provider capacity visibility detached from routing.")
    return tuple(actions)
