"""Passive V5.3 performance architecture consistency metadata."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.async_execution import (
    plan_async_execution,
)
from creative_coding_assistant.orchestration.bottleneck_detection import (
    detect_bottlenecks,
)
from creative_coding_assistant.orchestration.execution_profiling import (
    plan_execution_profiling,
)
from creative_coding_assistant.orchestration.execution_replay_engine import (
    plan_execution_replay,
)
from creative_coding_assistant.orchestration.latency_optimizer import (
    optimize_latency,
)
from creative_coding_assistant.orchestration.load_balancer import (
    plan_load_balancer,
)
from creative_coding_assistant.orchestration.parallel_scheduler import (
    plan_parallel_scheduler,
)
from creative_coding_assistant.orchestration.performance_benchmarking import (
    plan_performance_benchmarking,
)
from creative_coding_assistant.orchestration.performance_prediction import (
    predict_performance,
)
from creative_coding_assistant.orchestration.performance_regression_detection import (
    detect_performance_regressions,
)
from creative_coding_assistant.orchestration.reasoning_budget_optimizer import (
    optimize_reasoning_budget,
)
from creative_coding_assistant.orchestration.resource_utilization_optimizer import (
    optimize_resource_utilization,
)
from creative_coding_assistant.orchestration.retry_policies import (
    plan_retry_policies,
)
from creative_coding_assistant.orchestration.streaming_optimizer import (
    optimize_streaming,
)
from creative_coding_assistant.orchestration.throughput_optimizer import (
    optimize_throughput,
)
from creative_coding_assistant.orchestration.workflow_replay_engine import (
    plan_workflow_replay,
)

PerformanceArchitectureLayer = Literal[
    "scheduling_runtime_boundary",
    "throughput_latency_boundary",
    "profiling_replay_boundary",
    "prediction_benchmark_boundary",
    "budget_resource_boundary",
]
PerformanceArchitectureStage = Literal["v5_3_architecture_consistency_pass"]
PerformanceArchitectureStatus = Literal["pass"]

PERFORMANCE_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION = (
    "performance_architecture_consistency_record.v1"
)
PERFORMANCE_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION = (
    "performance_architecture_consistency_registry.v1"
)
PERFORMANCE_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY = (
    "V5.3 performance architecture consistency metadata checks advisory "
    "surface coverage, serialization, passive runtime boundaries, V4 "
    "compatibility, and version-level HITL/runtime evolution rules only; it "
    "does not measure performance, execute workflows, execute benchmarks, "
    "allocate resources, enforce capacity or budgets, route providers or "
    "models, control workflows, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_ARCHITECTURE_LAYERS: tuple[PerformanceArchitectureLayer, ...] = (
    "scheduling_runtime_boundary",
    "throughput_latency_boundary",
    "profiling_replay_boundary",
    "prediction_benchmark_boundary",
    "budget_resource_boundary",
)
_VALIDATED_VERSION_RULES = (
    "v5_3_surface_role_declared",
    "serialization_version_declared",
    "advisory_metadata_only",
    "runtime_measurement_not_applied",
    "workflow_execution_not_applied",
    "provider_model_routing_not_applied",
    "generated_output_mutation_blocked",
    "workflow_control_blocked",
    "runtime_evolution_not_applied",
    "v4_boundary_compatibility_confirmed",
    "human_gate_not_emitted",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "runtime_measurement_blocked",
    "workflow_execution_blocked",
    "workflow_graph_mutation_blocked",
    "provider_model_routing_blocked",
    "resource_allocation_blocked",
    "capacity_enforcement_blocked",
    "budget_enforcement_blocked",
    "retry_triggering_blocked",
    "prompt_mutation_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
    "runtime_evolution_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_performance_measurement",
    "runtime_profiling",
    "benchmark_execution",
    "resource_allocation",
    "capacity_enforcement",
    "budget_enforcement",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)
_ACTIVE_RUNTIME_FLAGS = (
    "runtime_performance_measurement_implemented",
    "runtime_resource_measurement_implemented",
    "runtime_profiling_implemented",
    "benchmark_execution_implemented",
    "resource_allocation_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "workflow_state_mutation_implemented",
    "graph_compilation_implemented",
    "provider_model_routing_implemented",
    "provider_selection_implemented",
    "automatic_model_selection_implemented",
    "runtime_selection_implemented",
    "latency_based_routing_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
)
_COUNT_FIELD_CANDIDATES = (
    "candidate_count",
    "scenario_count",
    "prediction_count",
    "recommendation_count",
    "signal_count",
)
_SURFACE_LAYERS: tuple[tuple[str, PerformanceArchitectureLayer], ...] = (
    ("parallel_scheduler", "scheduling_runtime_boundary"),
    ("latency_optimizer", "throughput_latency_boundary"),
    ("async_execution", "scheduling_runtime_boundary"),
    ("streaming_optimizer", "scheduling_runtime_boundary"),
    ("retry_policies", "scheduling_runtime_boundary"),
    ("load_balancer", "scheduling_runtime_boundary"),
    ("execution_profiling", "profiling_replay_boundary"),
    ("workflow_replay_engine", "profiling_replay_boundary"),
    ("execution_replay_engine", "profiling_replay_boundary"),
    ("bottleneck_detection", "throughput_latency_boundary"),
    ("throughput_optimizer", "throughput_latency_boundary"),
    ("performance_prediction", "prediction_benchmark_boundary"),
    ("performance_benchmarking", "prediction_benchmark_boundary"),
    ("reasoning_budget_optimizer", "budget_resource_boundary"),
    ("performance_regression_detection", "prediction_benchmark_boundary"),
    ("resource_utilization_optimizer", "budget_resource_boundary"),
)


class PerformanceArchitectureConsistencyRecord(BaseModel):
    """One passive V5.3 architecture consistency record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=120)
    architecture_layer: PerformanceArchitectureLayer
    architecture_stage: PerformanceArchitectureStage = (
        "v5_3_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=120)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=120)
    validated_version_rules: tuple[str, ...] = Field(min_length=11, max_length=11)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=12, max_length=12)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=40,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    source_advisory_only_declared: Literal[True] = True
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundary_compatibility_confirmed: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    architecture_consistency_status: PerformanceArchitectureStatus = "pass"
    runtime_measurement_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "performance_architecture_consistency_record.v1"
    ] = PERFORMANCE_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION
    advisory_only: Literal[True] = True


class PerformanceArchitectureConsistencyRegistry(BaseModel):
    """Passive V5.3 performance architecture consistency pass registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_architecture_consistency_registry"] = (
        "performance_architecture_consistency_registry"
    )
    serialization_version: Literal[
        "performance_architecture_consistency_registry.v1"
    ] = PERFORMANCE_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=PERFORMANCE_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    architecture_stage: PerformanceArchitectureStage = (
        "v5_3_architecture_consistency_pass"
    )
    records: tuple[PerformanceArchitectureConsistencyRecord, ...] = Field(
        min_length=16,
        max_length=16,
    )
    surface_ids: tuple[str, ...] = Field(min_length=16, max_length=16)
    record_count: int = Field(ge=16, le=16)
    architecture_layers: tuple[PerformanceArchitectureLayer, ...] = Field(
        min_length=5,
        max_length=5,
    )
    validated_version_rules: tuple[str, ...] = Field(min_length=11, max_length=11)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=12, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    all_surfaces_covered: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    v4_boundaries_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    runtime_measurement_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        derived_surface_ids = tuple(record.surface_id for record in self.records)
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _surface_ids():
            raise ValueError("surface_ids must match V5.3 surface order")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.architecture_layers != _ARCHITECTURE_LAYERS:
            raise ValueError("architecture_layers must match V5.3 layer order")
        if self.validated_version_rules != _VALIDATED_VERSION_RULES:
            raise ValueError("validated_version_rules must match registry")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match registry")
        for record in self.records:
            if record.architecture_stage != self.architecture_stage:
                raise ValueError("architecture_stage must match registry")
            if record.validated_version_rules != self.validated_version_rules:
                raise ValueError("validated_version_rules must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_active_runtime_flags:
                raise ValueError("records must not contain active runtime flags")
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
        return self


def performance_architecture_consistency_registry(
) -> PerformanceArchitectureConsistencyRegistry:
    """Return passive V5.3 performance architecture consistency metadata."""

    records = tuple(
        _record_from_source(surface_id, layer, source)
        for surface_id, layer, source in _source_specs()
    )
    return PerformanceArchitectureConsistencyRegistry(
        records=records,
        surface_ids=tuple(record.surface_id for record in records),
        record_count=len(records),
        architecture_layers=_ARCHITECTURE_LAYERS,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )


def performance_architecture_consistency_by_surface(
    surface_id: str,
    registry: PerformanceArchitectureConsistencyRegistry | None = None,
) -> PerformanceArchitectureConsistencyRecord | None:
    """Return one V5.3 architecture record by surface id."""

    source_registry = registry or performance_architecture_consistency_registry()
    normalized_surface_id = str(surface_id).strip()
    for record in source_registry.records:
        if record.surface_id == normalized_surface_id:
            return record
    return None


def performance_architecture_consistency_records_for_layer(
    architecture_layer: str,
    registry: PerformanceArchitectureConsistencyRegistry | None = None,
) -> tuple[PerformanceArchitectureConsistencyRecord, ...]:
    """Return passive V5.3 architecture records for one layer."""

    source_registry = registry or performance_architecture_consistency_registry()
    normalized_layer = str(architecture_layer).strip()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized_layer
    )


def _source_specs(
) -> tuple[tuple[str, PerformanceArchitectureLayer, Any], ...]:
    sources = _source_objects()
    return tuple(
        (surface_id, layer, sources[surface_id])
        for surface_id, layer in _SURFACE_LAYERS
    )


def _source_objects() -> dict[str, Any]:
    parallel = plan_parallel_scheduler()
    latency = optimize_latency(parallel_scheduler=parallel)
    async_plan = plan_async_execution(
        parallel_scheduler=parallel,
        latency_optimization=latency,
    )
    streaming = optimize_streaming(async_execution=async_plan)
    retry = plan_retry_policies(streaming_optimization=streaming)
    load = plan_load_balancer(
        async_execution=async_plan,
        latency_optimization=latency,
        retry_policy=retry,
    )
    profiling = plan_execution_profiling(
        latency_optimization=latency,
        load_balancer=load,
    )
    workflow_replay = plan_workflow_replay(execution_profiling=profiling)
    execution_replay = plan_execution_replay(
        workflow_replay=workflow_replay,
        execution_profiling=profiling,
    )
    bottlenecks = detect_bottlenecks(
        latency_optimization=latency,
        load_balancer=load,
        execution_profiling=profiling,
        execution_replay=execution_replay,
    )
    throughput = optimize_throughput(
        async_execution=async_plan,
        streaming_optimization=streaming,
        load_balancer=load,
        bottleneck_detection=bottlenecks,
    )
    prediction = predict_performance(
        throughput_optimization=throughput,
        latency_optimization=latency,
        bottleneck_detection=bottlenecks,
        execution_profiling=profiling,
    )
    benchmarking = plan_performance_benchmarking(
        performance_prediction=prediction,
        throughput_optimization=throughput,
        latency_optimization=latency,
        execution_profiling=profiling,
    )
    reasoning = optimize_reasoning_budget(
        performance_prediction=prediction,
        performance_benchmarking=benchmarking,
    )
    regression = detect_performance_regressions(
        performance_prediction=prediction,
        performance_benchmarking=benchmarking,
        reasoning_budget=reasoning,
    )
    resource = optimize_resource_utilization(
        throughput_optimization=throughput,
        execution_profiling=profiling,
        performance_benchmarking=benchmarking,
        reasoning_budget=reasoning,
        performance_regression=regression,
    )

    return {
        "parallel_scheduler": parallel,
        "latency_optimizer": latency,
        "async_execution": async_plan,
        "streaming_optimizer": streaming,
        "retry_policies": retry,
        "load_balancer": load,
        "execution_profiling": profiling,
        "workflow_replay_engine": workflow_replay,
        "execution_replay_engine": execution_replay,
        "bottleneck_detection": bottlenecks,
        "throughput_optimizer": throughput,
        "performance_prediction": prediction,
        "performance_benchmarking": benchmarking,
        "reasoning_budget_optimizer": reasoning,
        "performance_regression_detection": regression,
        "resource_utilization_optimizer": resource,
    }


def _record_from_source(
    surface_id: str,
    layer: PerformanceArchitectureLayer,
    source: Any,
) -> PerformanceArchitectureConsistencyRecord:
    count_field, source_count = _source_count(source)
    serialization_version = str(getattr(source, "serialization_version", ""))
    active_flags = _active_runtime_flags(source)
    missing = _missing_coverage(
        source=source,
        serialization_version=serialization_version,
        source_count=source_count,
        active_flags=active_flags,
    )
    return PerformanceArchitectureConsistencyRecord(
        surface_id=surface_id,
        architecture_layer=layer,
        source_role=str(getattr(source, "role", surface_id)),
        source_serialization_version=serialization_version,
        source_count_field=count_field,
        source_count=source_count,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        source_blocked_runtime_behaviors=_source_blocked_runtime_behaviors(source),
        source_active_runtime_flags=active_flags,
        missing_coverage_items=missing,
        source_advisory_only_declared=_source_advisory_only_declared(source),
    )


def _source_count(source: Any) -> tuple[str, int]:
    for count_field in _COUNT_FIELD_CANDIDATES:
        if hasattr(source, count_field):
            return count_field, int(getattr(source, count_field))
    raise ValueError("V5.3 architecture source must expose a count field")


def _source_blocked_runtime_behaviors(source: Any) -> tuple[str, ...]:
    blocked = getattr(source, "blocked_runtime_behaviors", None)
    if blocked:
        return tuple(str(item) for item in blocked)
    return _BLOCKED_RUNTIME_BEHAVIORS


def _source_advisory_only_declared(source: Any) -> bool:
    return bool(
        getattr(
            source,
            "advisory_only",
            getattr(source, "metadata_only", True),
        )
    )


def _active_runtime_flags(source: Any) -> tuple[str, ...]:
    return tuple(
        flag for flag in _ACTIVE_RUNTIME_FLAGS if bool(getattr(source, flag, False))
    )


def _missing_coverage(
    *,
    source: Any,
    serialization_version: str,
    source_count: int,
    active_flags: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not getattr(source, "role", ""):
        missing.append("role_missing")
    if not serialization_version.endswith(".v1"):
        missing.append("serialization_version_missing")
    if source_count < 1:
        missing.append("source_count_missing")
    if not _source_blocked_runtime_behaviors(source):
        missing.append("blocked_runtime_behaviors_missing")
    if active_flags:
        missing.append("active_runtime_flags_present")
    if not _source_advisory_only_declared(source):
        missing.append("advisory_only_declaration_missing")
    return tuple(missing)


def _surface_ids() -> tuple[str, ...]:
    return tuple(surface_id for surface_id, _layer in _SURFACE_LAYERS)
