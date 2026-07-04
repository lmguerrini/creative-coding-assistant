"""V5.3 advisory resource utilization optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from .performance_benchmarking import (
    PerformanceBenchmarkingPlan,
    plan_performance_benchmarking,
)
from .performance_regression_detection import (
    PerformanceRegressionDetectionPlan,
    detect_performance_regressions,
)
from .reasoning_budget_optimizer import (
    ReasoningBudgetOptimizationPlan,
    optimize_reasoning_budget,
)
from .throughput_optimizer import ThroughputOptimizationPlan, optimize_throughput

ResourceUtilizationKind = Literal[
    "throughput_capacity_utilization",
    "profiling_scope_utilization",
    "benchmark_workload_utilization",
    "reasoning_budget_utilization",
    "regression_baseline_utilization",
    "runtime_resource_boundary",
]
ResourceUtilizationStatus = Literal[
    "optimization_candidate",
    "capacity_guardrail",
    "review_guardrail",
    "boundary_guardrail",
]
ResourceUtilizationPressure = Literal["low", "medium", "high", "guarded"]

RESOURCE_UTILIZATION_RECOMMENDATION_SERIALIZATION_VERSION = (
    "resource_utilization_recommendation.v1"
)
RESOURCE_UTILIZATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "resource_utilization_optimization_plan.v1"
)
RESOURCE_UTILIZATION_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "Resource utilization optimization planning derives advisory utilization "
    "recommendations from throughput optimization, execution profiling, "
    "performance benchmarking, reasoning budget, and performance regression "
    "metadata only; it does not allocate resources, measure CPU or memory, "
    "change concurrency limits, manage queues, autoscale capacity, enforce "
    "capacity, execute benchmarks, install profilers, enforce budgets, route "
    "providers or models, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "resource_allocation",
    "runtime_resource_measurement",
    "cpu_memory_measurement",
    "concurrency_limit_change",
    "queue_management_runtime",
    "autoscaling",
    "capacity_enforcement",
    "benchmark_execution",
    "runtime_profiling",
    "timer_collection",
    "runtime_trace_collection",
    "budget_enforcement",
    "provider_or_model_routing",
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


class ResourceUtilizationRecommendation(BaseModel):
    """One advisory V5.3 resource utilization recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    recommendation_id: str = Field(min_length=1, max_length=180)
    utilization_id: str = Field(min_length=1, max_length=120)
    utilization_kind: ResourceUtilizationKind
    status: ResourceUtilizationStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    advisory_resource_units: int = Field(ge=0, le=500_000)
    advisory_reserve_units: int = Field(ge=0, le=250_000)
    advisory_pressure_units: int = Field(ge=0, le=20_000)
    advisory_utilization_score: int = Field(ge=0, le=3_000)
    utilization_pressure: ResourceUtilizationPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    resource_utilization_optimizer_planning_implemented: Literal[True] = True
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    cpu_memory_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    autoscaling_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    runtime_profiling_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
    serialization_version: Literal["resource_utilization_recommendation.v1"] = (
        RESOURCE_UTILIZATION_RECOMMENDATION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _recommendation_matches_inputs(self) -> Self:
        if self.recommendation_id != f"resource_utilization::{self.utilization_id}":
            raise ValueError("recommendation_id must match utilization_id")
        expected_score = _utilization_score(
            status=self.status,
            resource_units=self.advisory_resource_units,
            reserve_units=self.advisory_reserve_units,
            pressure_units=self.advisory_pressure_units,
        )
        if self.advisory_utilization_score != expected_score:
            raise ValueError(
                "advisory_utilization_score must match recommendation inputs"
            )
        if self.utilization_pressure != _pressure(
            status=self.status,
            utilization_score=self.advisory_utilization_score,
        ):
            raise ValueError("utilization_pressure must match recommendation")
        if (
            self.status == "optimization_candidate"
            and self.advisory_resource_units <= 0
        ):
            raise ValueError("optimization candidates require resource units")
        if self.status == "capacity_guardrail" and self.advisory_reserve_units <= 0:
            raise ValueError("capacity guardrails require reserve units")
        if self.status == "boundary_guardrail" and (
            self.advisory_resource_units
            or self.advisory_reserve_units
            or self.advisory_pressure_units
        ):
            raise ValueError("boundary guardrails must not declare utilization units")
        return self


class ResourceUtilizationOptimizationPlan(BaseModel):
    """Bounded V5.3 advisory resource utilization optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["resource_utilization_optimizer"] = "resource_utilization_optimizer"
    serialization_version: Literal["resource_utilization_optimization_plan.v1"] = (
        RESOURCE_UTILIZATION_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESOURCE_UTILIZATION_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    source_throughput_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_benchmarking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_reasoning_budget_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_regression_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    recommendations: tuple[ResourceUtilizationRecommendation, ...] = Field(
        min_length=1,
        max_length=12,
    )
    recommendation_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    optimization_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    capacity_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    review_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    boundary_guardrail_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    recommendation_count: int = Field(ge=1, le=12)
    optimization_candidate_count: int = Field(ge=0, le=12)
    capacity_guardrail_count: int = Field(ge=0, le=12)
    review_guardrail_count: int = Field(ge=0, le=12)
    boundary_guardrail_count: int = Field(ge=0, le=12)
    total_advisory_resource_units: int = Field(ge=0, le=1_000_000)
    total_advisory_reserve_units: int = Field(ge=0, le=500_000)
    total_advisory_pressure_units: int = Field(ge=0, le=60_000)
    highest_advisory_utilization_score: int = Field(ge=0, le=3_000)
    total_advisory_utilization_score: int = Field(ge=0, le=20_000)
    resource_utilization_pressure: ResourceUtilizationPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    resource_utilization_optimizer_planning_implemented: Literal[True] = True
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    cpu_memory_measurement_implemented: Literal[False] = False
    concurrency_limit_change_implemented: Literal[False] = False
    queue_management_runtime_implemented: Literal[False] = False
    autoscaling_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    runtime_profiling_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
    def _plan_matches_recommendations(self) -> Self:
        derived_ids = tuple(
            recommendation.recommendation_id for recommendation in self.recommendations
        )
        if len(set(derived_ids)) != len(derived_ids):
            raise ValueError("recommendation_ids must be unique")
        if self.recommendation_ids != derived_ids:
            raise ValueError("recommendation_ids must match recommendations")
        if self.recommendation_count != len(self.recommendations):
            raise ValueError("recommendation_count must match recommendations")
        if self.optimization_candidate_ids != _recommendation_ids_for_status(
            self.recommendations,
            "optimization_candidate",
        ):
            raise ValueError("optimization_candidate_ids must match recommendations")
        if self.capacity_guardrail_ids != _recommendation_ids_for_status(
            self.recommendations,
            "capacity_guardrail",
        ):
            raise ValueError("capacity_guardrail_ids must match recommendations")
        if self.review_guardrail_ids != _recommendation_ids_for_status(
            self.recommendations,
            "review_guardrail",
        ):
            raise ValueError("review_guardrail_ids must match recommendations")
        if self.boundary_guardrail_ids != _recommendation_ids_for_status(
            self.recommendations,
            "boundary_guardrail",
        ):
            raise ValueError("boundary_guardrail_ids must match recommendations")
        if self.optimization_candidate_count != len(self.optimization_candidate_ids):
            raise ValueError("optimization_candidate_count must match recommendations")
        if self.capacity_guardrail_count != len(self.capacity_guardrail_ids):
            raise ValueError("capacity_guardrail_count must match recommendations")
        if self.review_guardrail_count != len(self.review_guardrail_ids):
            raise ValueError("review_guardrail_count must match recommendations")
        if self.boundary_guardrail_count != len(self.boundary_guardrail_ids):
            raise ValueError("boundary_guardrail_count must match recommendations")

        expected_resource_units = sum(
            recommendation.advisory_resource_units
            for recommendation in self.recommendations
        )
        if self.total_advisory_resource_units != expected_resource_units:
            raise ValueError("total_advisory_resource_units must match")
        expected_reserve_units = sum(
            recommendation.advisory_reserve_units
            for recommendation in self.recommendations
        )
        if self.total_advisory_reserve_units != expected_reserve_units:
            raise ValueError("total_advisory_reserve_units must match")
        expected_pressure_units = sum(
            recommendation.advisory_pressure_units
            for recommendation in self.recommendations
        )
        if self.total_advisory_pressure_units != expected_pressure_units:
            raise ValueError("total_advisory_pressure_units must match")
        expected_highest = max(
            recommendation.advisory_utilization_score
            for recommendation in self.recommendations
        )
        if self.highest_advisory_utilization_score != expected_highest:
            raise ValueError("highest_advisory_utilization_score must match")
        expected_total = sum(
            recommendation.advisory_utilization_score
            for recommendation in self.recommendations
        )
        if self.total_advisory_utilization_score != expected_total:
            raise ValueError("total_advisory_utilization_score must match")
        if self.resource_utilization_pressure != _plan_pressure(
            recommendations=self.recommendations,
            highest_score=self.highest_advisory_utilization_score,
        ):
            raise ValueError("resource_utilization_pressure must match recommendations")
        return self


def optimize_resource_utilization(
    *,
    throughput_optimization: ThroughputOptimizationPlan | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
    performance_benchmarking: PerformanceBenchmarkingPlan | None = None,
    reasoning_budget: ReasoningBudgetOptimizationPlan | None = None,
    performance_regression: PerformanceRegressionDetectionPlan | None = None,
) -> ResourceUtilizationOptimizationPlan:
    """Plan advisory resource utilization optimization without allocation."""

    throughput = throughput_optimization or optimize_throughput()
    profiling = execution_profiling or plan_execution_profiling()
    benchmarking = performance_benchmarking or plan_performance_benchmarking(
        throughput_optimization=throughput,
        execution_profiling=profiling,
    )
    reasoning = reasoning_budget or optimize_reasoning_budget(
        performance_benchmarking=benchmarking,
    )
    regression = performance_regression or detect_performance_regressions(
        performance_benchmarking=benchmarking,
        reasoning_budget=reasoning,
    )
    recommendations = _recommendations(
        throughput=throughput,
        profiling=profiling,
        benchmarking=benchmarking,
        reasoning=reasoning,
        regression=regression,
    )
    highest_score = max(
        recommendation.advisory_utilization_score for recommendation in recommendations
    )

    return ResourceUtilizationOptimizationPlan(
        source_throughput_optimization_serialization_version=(
            throughput.serialization_version
        ),
        source_execution_profiling_serialization_version=profiling.serialization_version,
        source_performance_benchmarking_serialization_version=(
            benchmarking.serialization_version
        ),
        source_reasoning_budget_serialization_version=reasoning.serialization_version,
        source_performance_regression_serialization_version=(
            regression.serialization_version
        ),
        recommendations=recommendations,
        recommendation_ids=tuple(
            recommendation.recommendation_id for recommendation in recommendations
        ),
        optimization_candidate_ids=_recommendation_ids_for_status(
            recommendations,
            "optimization_candidate",
        ),
        capacity_guardrail_ids=_recommendation_ids_for_status(
            recommendations,
            "capacity_guardrail",
        ),
        review_guardrail_ids=_recommendation_ids_for_status(
            recommendations,
            "review_guardrail",
        ),
        boundary_guardrail_ids=_recommendation_ids_for_status(
            recommendations,
            "boundary_guardrail",
        ),
        recommendation_count=len(recommendations),
        optimization_candidate_count=len(
            _recommendation_ids_for_status(recommendations, "optimization_candidate")
        ),
        capacity_guardrail_count=len(
            _recommendation_ids_for_status(recommendations, "capacity_guardrail")
        ),
        review_guardrail_count=len(
            _recommendation_ids_for_status(recommendations, "review_guardrail")
        ),
        boundary_guardrail_count=len(
            _recommendation_ids_for_status(recommendations, "boundary_guardrail")
        ),
        total_advisory_resource_units=sum(
            recommendation.advisory_resource_units for recommendation in recommendations
        ),
        total_advisory_reserve_units=sum(
            recommendation.advisory_reserve_units for recommendation in recommendations
        ),
        total_advisory_pressure_units=sum(
            recommendation.advisory_pressure_units for recommendation in recommendations
        ),
        highest_advisory_utilization_score=highest_score,
        total_advisory_utilization_score=sum(
            recommendation.advisory_utilization_score
            for recommendation in recommendations
        ),
        resource_utilization_pressure=_plan_pressure(
            recommendations=recommendations,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(recommendations),
    )


def resource_utilization_recommendation_by_id(
    recommendation_id: str,
    plan: ResourceUtilizationOptimizationPlan | None = None,
) -> ResourceUtilizationRecommendation | None:
    """Return one advisory utilization recommendation without allocation."""

    source_plan = plan or optimize_resource_utilization()
    for recommendation in source_plan.recommendations:
        if recommendation.recommendation_id == recommendation_id:
            return recommendation
    return None


def resource_utilization_recommendations_for_status(
    status: ResourceUtilizationStatus,
    plan: ResourceUtilizationOptimizationPlan | None = None,
) -> tuple[ResourceUtilizationRecommendation, ...]:
    """Return advisory utilization recommendations by status."""

    source_plan = plan or optimize_resource_utilization()
    return tuple(
        recommendation
        for recommendation in source_plan.recommendations
        if recommendation.status == status
    )


def _recommendations(
    *,
    throughput: ThroughputOptimizationPlan,
    profiling: ExecutionProfilingPlan,
    benchmarking: PerformanceBenchmarkingPlan,
    reasoning: ReasoningBudgetOptimizationPlan,
    regression: PerformanceRegressionDetectionPlan,
) -> tuple[ResourceUtilizationRecommendation, ...]:
    return (
        _recommendation(
            utilization_id="throughput_capacity_utilization",
            kind="throughput_capacity_utilization",
            status="optimization_candidate",
            source_id="throughput_optimization_plan",
            source_serialization_version=throughput.serialization_version,
            source_item_ids=throughput.candidate_ids,
            resource_units=(
                throughput.total_advisory_throughput_units
                + throughput.total_advisory_capacity_units
            ),
            reserve_units=throughput.total_advisory_backpressure_units,
            pressure_units=throughput.total_advisory_throughput_score,
            evidence=(
                f"throughput_pressure:{throughput.throughput_optimization_pressure}",
                f"throughput_units:{throughput.total_advisory_throughput_units}",
            ),
        ),
        _recommendation(
            utilization_id="profiling_scope_utilization",
            kind="profiling_scope_utilization",
            status="review_guardrail",
            source_id="execution_profiling_plan",
            source_serialization_version=profiling.serialization_version,
            source_item_ids=profiling.candidate_ids,
            resource_units=(
                profiling.total_profiled_node_count
                + profiling.total_profiled_agent_count
            ),
            reserve_units=profiling.total_blocking_input_count,
            pressure_units=profiling.total_advisory_profile_score,
            evidence=(
                f"profile_pressure:{profiling.execution_profile_pressure}",
                f"profiled_nodes:{profiling.total_profiled_node_count}",
            ),
        ),
        _recommendation(
            utilization_id="benchmark_workload_utilization",
            kind="benchmark_workload_utilization",
            status="capacity_guardrail",
            source_id="performance_benchmarking_plan",
            source_serialization_version=benchmarking.serialization_version,
            source_item_ids=benchmarking.scenario_ids,
            resource_units=benchmarking.total_advisory_benchmark_units,
            reserve_units=benchmarking.total_advisory_sample_count,
            pressure_units=benchmarking.total_benchmark_priority_score,
            evidence=(
                f"benchmarking_readiness:{benchmarking.benchmarking_readiness}",
                f"benchmark_units:{benchmarking.total_advisory_benchmark_units}",
            ),
        ),
        _recommendation(
            utilization_id="reasoning_budget_utilization",
            kind="reasoning_budget_utilization",
            status="optimization_candidate",
            source_id="reasoning_budget_optimization_plan",
            source_serialization_version=reasoning.serialization_version,
            source_item_ids=reasoning.recommendation_ids,
            resource_units=reasoning.total_advisory_reasoning_tokens,
            reserve_units=reasoning.total_advisory_reserve_tokens,
            pressure_units=reasoning.total_advisory_pressure_score,
            evidence=(
                f"reasoning_pressure:{reasoning.reasoning_budget_pressure}",
                f"reasoning_tokens:{reasoning.total_advisory_reasoning_tokens}",
            ),
        ),
        _recommendation(
            utilization_id="regression_baseline_utilization",
            kind="regression_baseline_utilization",
            status="review_guardrail",
            source_id="performance_regression_detection_plan",
            source_serialization_version=regression.serialization_version,
            source_item_ids=regression.signal_ids,
            resource_units=regression.total_baseline_reference_count,
            reserve_units=regression.regression_candidate_count,
            pressure_units=regression.total_advisory_regression_score,
            evidence=(
                f"regression_severity:{regression.regression_detection_severity}",
                f"baseline_references:{regression.total_baseline_reference_count}",
            ),
        ),
        _recommendation(
            utilization_id="runtime_resource_boundary",
            kind="runtime_resource_boundary",
            status="boundary_guardrail",
            source_id="performance_regression_detection_plan",
            source_serialization_version=regression.serialization_version,
            source_item_ids=regression.baseline_guardrail_ids,
            resource_units=0,
            reserve_units=0,
            pressure_units=0,
            evidence=(
                "runtime_resource_measurement:blocked",
                "resource_allocation:blocked",
            ),
        ),
    )


def _recommendation(
    *,
    utilization_id: str,
    kind: ResourceUtilizationKind,
    status: ResourceUtilizationStatus,
    source_id: str,
    source_serialization_version: str,
    source_item_ids: tuple[str, ...],
    resource_units: int,
    reserve_units: int,
    pressure_units: int,
    evidence: tuple[str, ...],
) -> ResourceUtilizationRecommendation:
    score = _utilization_score(
        status=status,
        resource_units=resource_units,
        reserve_units=reserve_units,
        pressure_units=pressure_units,
    )
    return ResourceUtilizationRecommendation(
        recommendation_id=f"resource_utilization::{utilization_id}",
        utilization_id=utilization_id,
        utilization_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_item_ids=source_item_ids,
        advisory_resource_units=resource_units,
        advisory_reserve_units=reserve_units,
        advisory_pressure_units=pressure_units,
        advisory_utilization_score=score,
        utilization_pressure=_pressure(
            status=status,
            utilization_score=score,
        ),
        evidence=evidence,
        advisory_actions=_recommendation_actions(status),
    )


def _recommendation_ids_for_status(
    recommendations: tuple[ResourceUtilizationRecommendation, ...],
    status: ResourceUtilizationStatus,
) -> tuple[str, ...]:
    return tuple(
        recommendation.recommendation_id
        for recommendation in recommendations
        if recommendation.status == status
    )


def _utilization_score(
    *,
    status: ResourceUtilizationStatus,
    resource_units: int,
    reserve_units: int,
    pressure_units: int,
) -> int:
    if status == "boundary_guardrail":
        return 0
    score = resource_units // 10 + reserve_units // 5 + pressure_units // 2
    if status == "capacity_guardrail":
        score += 150
    if status == "review_guardrail":
        score += 75
    return min(3_000, score)


def _pressure(
    *,
    status: ResourceUtilizationStatus,
    utilization_score: int,
) -> ResourceUtilizationPressure:
    if status != "optimization_candidate":
        return "guarded"
    if utilization_score >= 1_200:
        return "high"
    if utilization_score >= 400:
        return "medium"
    return "low"


def _plan_pressure(
    *,
    recommendations: tuple[ResourceUtilizationRecommendation, ...],
    highest_score: int,
) -> ResourceUtilizationPressure:
    if any(
        recommendation.status != "optimization_candidate"
        for recommendation in recommendations
    ):
        return "guarded"
    if highest_score >= 1_200:
        return "high"
    if highest_score >= 400:
        return "medium"
    return "low"


def _recommendation_actions(
    status: ResourceUtilizationStatus,
) -> tuple[str, ...]:
    if status == "optimization_candidate":
        return (
            "Expose resource utilization recommendation as advisory metadata only.",
            "Require explicit runtime authority before resource allocation changes.",
        )
    if status == "capacity_guardrail":
        return (
            "Keep benchmark workload utilization detached from execution.",
            "Preserve capacity, measurement, workflow, and output boundaries.",
        )
    if status == "review_guardrail":
        return (
            "Keep utilization pressure review-only until runtime authority exists.",
        )
    return (
        "Preserve resource allocation, measurement, scaling, and output boundaries.",
    )


def _plan_actions(
    recommendations: tuple[ResourceUtilizationRecommendation, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose resource utilization posture as advisory metadata only.",
        "Preserve resource allocation, measurement, scaling, capacity, routing, "
        "workflow, and output boundaries.",
    ]
    if _recommendation_ids_for_status(recommendations, "boundary_guardrail"):
        actions.append("Keep runtime resource boundaries detached from enforcement.")
    return tuple(actions)
