"""V5.3 advisory performance benchmarking planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_profiling import ExecutionProfilingPlan, plan_execution_profiling
from .latency_optimizer import LatencyOptimizationPlan, optimize_latency
from .performance_prediction import PerformancePredictionPlan, predict_performance
from .throughput_optimizer import ThroughputOptimizationPlan, optimize_throughput

PerformanceBenchmarkKind = Literal[
    "prediction_baseline",
    "throughput_benchmark",
    "latency_benchmark",
    "profiling_boundary",
]
PerformanceBenchmarkStatus = Literal[
    "baseline_candidate",
    "benchmark_candidate",
    "guardrail",
]
PerformanceBenchmarkReadiness = Literal["low", "medium", "high", "guarded"]

PERFORMANCE_BENCHMARK_SCENARIO_SERIALIZATION_VERSION = (
    "performance_benchmark_scenario.v1"
)
PERFORMANCE_BENCHMARKING_PLAN_SERIALIZATION_VERSION = "performance_benchmarking_plan.v1"
PERFORMANCE_BENCHMARKING_AUTHORITY_BOUNDARY = (
    "Performance benchmarking planning derives advisory benchmark scenarios "
    "from performance prediction, throughput optimization, latency "
    "optimization, and execution profiling metadata only; it does not execute "
    "benchmarks, measure runtime performance, collect timers, install "
    "profilers, collect traces, run workloads, replay workflows, route "
    "providers or models, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "benchmark_execution",
    "runtime_performance_measurement",
    "timer_collection",
    "profiler_hook_installation",
    "runtime_trace_collection",
    "workload_execution",
    "workflow_replay_execution",
    "workflow_execution",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class PerformanceBenchmarkScenario(BaseModel):
    """One advisory V5.3 performance benchmarking scenario."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1, max_length=180)
    benchmark_id: str = Field(min_length=1, max_length=120)
    benchmark_kind: PerformanceBenchmarkKind
    status: PerformanceBenchmarkStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    advisory_sample_count: int = Field(ge=0, le=200)
    advisory_benchmark_units: int = Field(ge=0, le=50_000)
    baseline_reference_count: int = Field(ge=0, le=200)
    benchmark_priority_score: int = Field(ge=0, le=3_000)
    benchmark_readiness: PerformanceBenchmarkReadiness
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_benchmarking_planning_implemented: Literal[True] = True
    benchmark_execution_implemented: Literal[False] = False
    runtime_performance_measurement_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    workload_execution_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["performance_benchmark_scenario.v1"] = (
        PERFORMANCE_BENCHMARK_SCENARIO_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _scenario_matches_inputs(self) -> Self:
        if self.scenario_id != f"performance_benchmark::{self.benchmark_id}":
            raise ValueError("scenario_id must match benchmark_id")
        expected_score = _priority_score(
            status=self.status,
            sample_count=self.advisory_sample_count,
            benchmark_units=self.advisory_benchmark_units,
            baseline_reference_count=self.baseline_reference_count,
        )
        if self.benchmark_priority_score != expected_score:
            raise ValueError("benchmark_priority_score must match scenario inputs")
        if self.benchmark_readiness != _readiness(
            status=self.status,
            priority_score=self.benchmark_priority_score,
        ):
            raise ValueError("benchmark_readiness must match scenario inputs")
        if self.status == "benchmark_candidate" and self.advisory_sample_count <= 0:
            raise ValueError("benchmark candidates require advisory samples")
        if self.status == "baseline_candidate" and self.baseline_reference_count <= 0:
            raise ValueError("baseline candidates require baseline references")
        if self.status == "guardrail" and (
            self.advisory_sample_count or self.advisory_benchmark_units
        ):
            raise ValueError("guardrails must not declare benchmark samples")
        return self


class PerformanceBenchmarkingPlan(BaseModel):
    """Bounded V5.3 advisory performance benchmarking plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_benchmark_planner"] = "performance_benchmark_planner"
    serialization_version: Literal["performance_benchmarking_plan.v1"] = (
        PERFORMANCE_BENCHMARKING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERFORMANCE_BENCHMARKING_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_performance_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_throughput_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_profiling_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    scenarios: tuple[PerformanceBenchmarkScenario, ...] = Field(
        min_length=1,
        max_length=12,
    )
    scenario_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    baseline_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    benchmark_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    guardrail_scenario_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    scenario_count: int = Field(ge=1, le=12)
    baseline_candidate_count: int = Field(ge=0, le=12)
    benchmark_candidate_count: int = Field(ge=0, le=12)
    guardrail_count: int = Field(ge=0, le=12)
    total_advisory_sample_count: int = Field(ge=0, le=1_000)
    total_advisory_benchmark_units: int = Field(ge=0, le=100_000)
    highest_benchmark_priority_score: int = Field(ge=0, le=3_000)
    total_benchmark_priority_score: int = Field(ge=0, le=20_000)
    benchmarking_readiness: PerformanceBenchmarkReadiness
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_benchmarking_planning_implemented: Literal[True] = True
    benchmark_execution_implemented: Literal[False] = False
    runtime_performance_measurement_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    workload_execution_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_scenarios(self) -> Self:
        derived_scenario_ids = tuple(
            scenario.scenario_id for scenario in self.scenarios
        )
        if len(set(derived_scenario_ids)) != len(derived_scenario_ids):
            raise ValueError("scenario_ids must be unique")
        if self.scenario_ids != derived_scenario_ids:
            raise ValueError("scenario_ids must match scenarios")
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario_count must match scenarios")
        if self.baseline_candidate_ids != _scenario_ids_for_status(
            self.scenarios,
            "baseline_candidate",
        ):
            raise ValueError("baseline_candidate_ids must match scenarios")
        if self.benchmark_candidate_ids != _scenario_ids_for_status(
            self.scenarios,
            "benchmark_candidate",
        ):
            raise ValueError("benchmark_candidate_ids must match scenarios")
        if self.guardrail_scenario_ids != _scenario_ids_for_status(
            self.scenarios,
            "guardrail",
        ):
            raise ValueError("guardrail_scenario_ids must match scenarios")
        if self.baseline_candidate_count != len(self.baseline_candidate_ids):
            raise ValueError("baseline_candidate_count must match scenarios")
        if self.benchmark_candidate_count != len(self.benchmark_candidate_ids):
            raise ValueError("benchmark_candidate_count must match scenarios")
        if self.guardrail_count != len(self.guardrail_scenario_ids):
            raise ValueError("guardrail_count must match scenarios")

        expected_samples = sum(
            scenario.advisory_sample_count for scenario in self.scenarios
        )
        if self.total_advisory_sample_count != expected_samples:
            raise ValueError("total_advisory_sample_count must match scenarios")
        expected_units = sum(
            scenario.advisory_benchmark_units for scenario in self.scenarios
        )
        if self.total_advisory_benchmark_units != expected_units:
            raise ValueError("total_advisory_benchmark_units must match scenarios")
        expected_highest = max(
            scenario.benchmark_priority_score for scenario in self.scenarios
        )
        if self.highest_benchmark_priority_score != expected_highest:
            raise ValueError("highest_benchmark_priority_score must match scenarios")
        expected_total = sum(
            scenario.benchmark_priority_score for scenario in self.scenarios
        )
        if self.total_benchmark_priority_score != expected_total:
            raise ValueError("total_benchmark_priority_score must match scenarios")
        if self.benchmarking_readiness != _plan_readiness(
            scenarios=self.scenarios,
            highest_score=self.highest_benchmark_priority_score,
        ):
            raise ValueError("benchmarking_readiness must match scenarios")
        return self


def plan_performance_benchmarking(
    *,
    performance_prediction: PerformancePredictionPlan | None = None,
    throughput_optimization: ThroughputOptimizationPlan | None = None,
    latency_optimization: LatencyOptimizationPlan | None = None,
    execution_profiling: ExecutionProfilingPlan | None = None,
) -> PerformanceBenchmarkingPlan:
    """Plan advisory benchmark scenarios without executing benchmarks."""

    latency = latency_optimization or optimize_latency()
    profiling = execution_profiling or plan_execution_profiling(
        latency_optimization=latency
    )
    throughput = throughput_optimization or optimize_throughput()
    prediction = performance_prediction or predict_performance(
        throughput_optimization=throughput,
        latency_optimization=latency,
        execution_profiling=profiling,
    )
    scenarios = _scenarios(
        prediction=prediction,
        throughput=throughput,
        latency=latency,
        profiling=profiling,
    )
    highest_score = max(scenario.benchmark_priority_score for scenario in scenarios)

    return PerformanceBenchmarkingPlan(
        source_performance_prediction_serialization_version=(
            prediction.serialization_version
        ),
        source_throughput_optimization_serialization_version=(
            throughput.serialization_version
        ),
        source_latency_optimization_serialization_version=latency.serialization_version,
        source_execution_profiling_serialization_version=(
            profiling.serialization_version
        ),
        scenarios=scenarios,
        scenario_ids=tuple(scenario.scenario_id for scenario in scenarios),
        baseline_candidate_ids=_scenario_ids_for_status(
            scenarios,
            "baseline_candidate",
        ),
        benchmark_candidate_ids=_scenario_ids_for_status(
            scenarios,
            "benchmark_candidate",
        ),
        guardrail_scenario_ids=_scenario_ids_for_status(scenarios, "guardrail"),
        scenario_count=len(scenarios),
        baseline_candidate_count=len(
            _scenario_ids_for_status(scenarios, "baseline_candidate")
        ),
        benchmark_candidate_count=len(
            _scenario_ids_for_status(scenarios, "benchmark_candidate")
        ),
        guardrail_count=len(_scenario_ids_for_status(scenarios, "guardrail")),
        total_advisory_sample_count=sum(
            scenario.advisory_sample_count for scenario in scenarios
        ),
        total_advisory_benchmark_units=sum(
            scenario.advisory_benchmark_units for scenario in scenarios
        ),
        highest_benchmark_priority_score=highest_score,
        total_benchmark_priority_score=sum(
            scenario.benchmark_priority_score for scenario in scenarios
        ),
        benchmarking_readiness=_plan_readiness(
            scenarios=scenarios,
            highest_score=highest_score,
        ),
        advisory_actions=_plan_actions(scenarios),
    )


def performance_benchmark_scenario_by_id(
    scenario_id: str,
    plan: PerformanceBenchmarkingPlan | None = None,
) -> PerformanceBenchmarkScenario | None:
    """Return one advisory benchmark scenario without executing benchmarks."""

    source_plan = plan or plan_performance_benchmarking()
    for scenario in source_plan.scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


def performance_benchmark_scenarios_for_status(
    status: PerformanceBenchmarkStatus,
    plan: PerformanceBenchmarkingPlan | None = None,
) -> tuple[PerformanceBenchmarkScenario, ...]:
    """Return advisory benchmark scenarios by status without measurement."""

    source_plan = plan or plan_performance_benchmarking()
    return tuple(
        scenario for scenario in source_plan.scenarios if scenario.status == status
    )


def _scenarios(
    *,
    prediction: PerformancePredictionPlan,
    throughput: ThroughputOptimizationPlan,
    latency: LatencyOptimizationPlan,
    profiling: ExecutionProfilingPlan,
) -> tuple[PerformanceBenchmarkScenario, ...]:
    return (
        _scenario(
            benchmark_id="prediction_baseline",
            kind="prediction_baseline",
            status="baseline_candidate",
            source_id="performance_prediction_plan",
            source_serialization_version=prediction.serialization_version,
            source_candidate_ids=prediction.prediction_ids,
            sample_count=prediction.prediction_count,
            benchmark_units=prediction.highest_predicted_performance_midpoint,
            baseline_reference_count=prediction.prediction_count,
            evidence=(
                f"recommended_band:{prediction.recommended_performance_band}",
                f"prediction_count:{prediction.prediction_count}",
            ),
        ),
        _scenario(
            benchmark_id="throughput_benchmark",
            kind="throughput_benchmark",
            status="benchmark_candidate",
            source_id="throughput_optimization_plan",
            source_serialization_version=throughput.serialization_version,
            source_candidate_ids=throughput.throughput_candidate_ids,
            sample_count=throughput.throughput_candidate_count,
            benchmark_units=throughput.total_advisory_throughput_units,
            baseline_reference_count=throughput.candidate_count,
            evidence=(
                f"throughput_pressure:{throughput.throughput_optimization_pressure}",
                f"throughput_units:{throughput.total_advisory_throughput_units}",
            ),
        ),
        _scenario(
            benchmark_id="latency_benchmark",
            kind="latency_benchmark",
            status="benchmark_candidate",
            source_id="latency_optimization_plan",
            source_serialization_version=latency.serialization_version,
            source_candidate_ids=latency.optimization_candidate_ids,
            sample_count=latency.optimization_candidate_count,
            benchmark_units=latency.total_advisory_latency_savings_score,
            baseline_reference_count=latency.candidate_count,
            evidence=(
                f"latency_pressure:{latency.latency_optimization_pressure}",
                f"latency_savings:{latency.total_advisory_latency_savings_score}",
            ),
        ),
        _scenario(
            benchmark_id="profiling_boundary",
            kind="profiling_boundary",
            status="guardrail",
            source_id="execution_profiling_plan",
            source_serialization_version=profiling.serialization_version,
            source_candidate_ids=profiling.candidate_ids,
            sample_count=0,
            benchmark_units=0,
            baseline_reference_count=(
                profiling.measurement_guardrail_count
                + profiling.failure_guardrail_count
            ),
            evidence=(
                "profiler_hook_installation:blocked",
                f"profile_pressure:{profiling.execution_profile_pressure}",
            ),
        ),
    )


def _scenario(
    *,
    benchmark_id: str,
    kind: PerformanceBenchmarkKind,
    status: PerformanceBenchmarkStatus,
    source_id: str,
    source_serialization_version: str,
    source_candidate_ids: tuple[str, ...],
    sample_count: int,
    benchmark_units: int,
    baseline_reference_count: int,
    evidence: tuple[str, ...],
) -> PerformanceBenchmarkScenario:
    score = _priority_score(
        status=status,
        sample_count=sample_count,
        benchmark_units=benchmark_units,
        baseline_reference_count=baseline_reference_count,
    )
    return PerformanceBenchmarkScenario(
        scenario_id=f"performance_benchmark::{benchmark_id}",
        benchmark_id=benchmark_id,
        benchmark_kind=kind,
        status=status,
        source_id=source_id,
        source_serialization_version=source_serialization_version,
        source_candidate_ids=source_candidate_ids,
        advisory_sample_count=sample_count,
        advisory_benchmark_units=benchmark_units,
        baseline_reference_count=baseline_reference_count,
        benchmark_priority_score=score,
        benchmark_readiness=_readiness(status=status, priority_score=score),
        evidence=evidence,
        advisory_actions=_scenario_actions(status),
    )


def _scenario_ids_for_status(
    scenarios: tuple[PerformanceBenchmarkScenario, ...],
    status: PerformanceBenchmarkStatus,
) -> tuple[str, ...]:
    return tuple(
        scenario.scenario_id for scenario in scenarios if scenario.status == status
    )


def _priority_score(
    *,
    status: PerformanceBenchmarkStatus,
    sample_count: int,
    benchmark_units: int,
    baseline_reference_count: int,
) -> int:
    if status == "guardrail":
        return 0
    score = sample_count * 120 + baseline_reference_count * 60 + benchmark_units
    if status == "baseline_candidate":
        score += 100
    return min(3_000, score)


def _readiness(
    *,
    status: PerformanceBenchmarkStatus,
    priority_score: int,
) -> PerformanceBenchmarkReadiness:
    if status == "guardrail":
        return "guarded"
    if priority_score >= 1_200:
        return "high"
    if priority_score >= 400:
        return "medium"
    return "low"


def _plan_readiness(
    *,
    scenarios: tuple[PerformanceBenchmarkScenario, ...],
    highest_score: int,
) -> PerformanceBenchmarkReadiness:
    if any(scenario.status == "guardrail" for scenario in scenarios):
        return "guarded"
    if highest_score >= 1_200:
        return "high"
    if highest_score >= 400:
        return "medium"
    return "low"


def _scenario_actions(
    status: PerformanceBenchmarkStatus,
) -> tuple[str, ...]:
    if status == "benchmark_candidate":
        return (
            "Expose benchmark candidate as advisory metadata only.",
            "Require explicit runtime authority before executing benchmark work.",
        )
    if status == "baseline_candidate":
        return ("Keep prediction baseline available for benchmark planning only.",)
    return ("Preserve profiler, timer, trace, workflow, and output boundaries.",)


def _plan_actions(
    scenarios: tuple[PerformanceBenchmarkScenario, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose performance benchmark scenarios as advisory metadata only.",
        "Preserve benchmark execution, measurement, profiling, trace, workflow, "
        "routing, and output boundaries.",
    ]
    if _scenario_ids_for_status(scenarios, "guardrail"):
        actions.append("Keep profiling-sensitive benchmarks behind review.")
    return tuple(actions)
