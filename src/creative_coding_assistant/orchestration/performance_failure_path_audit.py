"""Passive V5.3 runtime failure path audit metadata for performance."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .performance_architecture_consistency import (
    performance_architecture_consistency_registry,
)

PerformanceFailurePathCheckKind = Literal[
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "registry_import_loading_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
]
PerformanceFailurePathAuditStatus = Literal["pass"]

PERFORMANCE_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "performance_failure_path_audit_record.v1"
)
PERFORMANCE_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "performance_failure_path_audit_registry.v1"
)
PERFORMANCE_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.3 performance runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage, performance "
    "surface failure boundaries, scheduling and streaming failure limits, "
    "retry and budget/resource limits, serialization and registry loading "
    "guards, provider/model routing preservation, generated-output mutation "
    "limits, and passive activation limits only; it does not measure "
    "performance, execute workflows, execute benchmarks, allocate resources, "
    "enforce capacity or budgets, route providers or models, control "
    "workflows, trigger retries, mutate prompts, write storage, modify "
    "generated output, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "parallel_scheduler",
    "latency_optimizer",
    "async_execution",
    "streaming_optimizer",
    "retry_policies",
    "load_balancer",
    "execution_profiling",
    "workflow_replay_engine",
    "execution_replay_engine",
    "bottleneck_detection",
    "throughput_optimizer",
    "performance_prediction",
    "performance_benchmarking",
    "reasoning_budget_optimizer",
    "performance_regression_detection",
    "resource_utilization_optimizer",
)
_REQUIRED_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "preview_workstation_frontend_backend_failures",
    "registry_import_loading_failures",
    "telemetry_observability_failures",
    "cache_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
_APPLICABLE_REQUIRED_CHECKS: tuple[PerformanceFailurePathCheckKind, ...] = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "registry_import_loading_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
_NOT_APPLICABLE_REQUIRED_CHECKS = (
    "preview_workstation_frontend_backend_failures",
    "telemetry_observability_failures",
    "cache_failures",
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


class PerformanceFailurePathAuditRecord(BaseModel):
    """One passive V5.3 performance runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=180)
    check_kind: PerformanceFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=260)
    audit_status: PerformanceFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    runtime_profiling_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["performance_failure_path_audit_record.v1"] = (
        PERFORMANCE_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V5.3 surfaces")
        return self


class PerformanceFailurePathAuditRegistry(BaseModel):
    """Passive V5.3 performance runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_failure_path_audit_registry"] = (
        "performance_failure_path_audit_registry"
    )
    serialization_version: Literal["performance_failure_path_audit_registry.v1"] = (
        PERFORMANCE_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERFORMANCE_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    architecture_registry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    architecture_registry_record_count: int = Field(ge=16, le=16)
    source_surface_ids: tuple[str, ...] = Field(min_length=16, max_length=16)
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[PerformanceFailurePathCheckKind, ...] = Field(
        min_length=16,
        max_length=16,
    )
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=3,
        max_length=3,
    )
    records: tuple[PerformanceFailurePathAuditRecord, ...] = Field(
        min_length=16,
        max_length=16,
    )
    audit_ids: tuple[str, ...] = Field(min_length=16, max_length=16)
    check_kinds: tuple[PerformanceFailurePathCheckKind, ...] = Field(
        min_length=16,
        max_length=16,
    )
    record_count: int = Field(ge=16, le=16)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    runtime_profiling_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        derived_audit_ids = tuple(record.audit_id for record in self.records)
        if len(set(derived_audit_ids)) != len(derived_audit_ids):
            raise ValueError("audit_ids must be unique")
        if self.audit_ids != derived_audit_ids:
            raise ValueError("audit_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.source_surface_ids != _SOURCE_SURFACE_IDS:
            raise ValueError("source_surface_ids must match V5.3 surfaces")
        if self.required_checks != _REQUIRED_CHECKS:
            raise ValueError("required_checks must match runtime audit checklist")
        if self.applicable_required_checks != _APPLICABLE_REQUIRED_CHECKS:
            raise ValueError("applicable_required_checks must match records")
        if self.not_applicable_required_checks != _NOT_APPLICABLE_REQUIRED_CHECKS:
            raise ValueError("not_applicable_required_checks must match checklist")
        derived_check_kinds = tuple(record.check_kind for record in self.records)
        if self.check_kinds != derived_check_kinds:
            raise ValueError("check_kinds must match records")
        if derived_check_kinds != self.applicable_required_checks:
            raise ValueError("records must cover applicable checks in order")
        if len(set(derived_check_kinds)) != len(derived_check_kinds):
            raise ValueError("check_kinds must be unique")
        for record in self.records:
            if record.checklist_source != self.checklist_source:
                raise ValueError("record checklist_source must match registry")
            if record.check_kind not in self.applicable_required_checks:
                raise ValueError("record check_kind must be applicable")
        return self


def performance_failure_path_audit_registry() -> PerformanceFailurePathAuditRegistry:
    """Return V5.3 runtime failure path audit metadata without runtime actions."""

    architecture_registry = performance_architecture_consistency_registry()
    records = _records()
    return PerformanceFailurePathAuditRegistry(
        architecture_registry_serialization_version=(
            architecture_registry.serialization_version
        ),
        architecture_registry_record_count=architecture_registry.record_count,
        source_surface_ids=architecture_registry.surface_ids,
        required_checks=_REQUIRED_CHECKS,
        applicable_required_checks=_APPLICABLE_REQUIRED_CHECKS,
        not_applicable_required_checks=_NOT_APPLICABLE_REQUIRED_CHECKS,
        records=records,
        audit_ids=tuple(record.audit_id for record in records),
        check_kinds=tuple(record.check_kind for record in records),
        record_count=len(records),
    )


def performance_failure_path_audit_by_id(
    audit_id: str,
    registry: PerformanceFailurePathAuditRegistry | None = None,
) -> PerformanceFailurePathAuditRecord | None:
    """Return one V5.3 failure audit record without activating failures."""

    source_registry = registry or performance_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def performance_failure_path_audits_for_check(
    check_kind: PerformanceFailurePathCheckKind,
    registry: PerformanceFailurePathAuditRegistry | None = None,
) -> tuple[PerformanceFailurePathAuditRecord, ...]:
    """Return V5.3 failure audit records by checklist item."""

    source_registry = registry or performance_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def performance_failure_path_audits_for_surface(
    surface_id: str,
    registry: PerformanceFailurePathAuditRegistry | None = None,
) -> tuple[PerformanceFailurePathAuditRecord, ...]:
    """Return V5.3 failure audit records for one performance surface."""

    source_registry = registry or performance_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _records() -> tuple[PerformanceFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_architecture_registry",
                "each_surface_declares_advisory_metadata_boundary",
            ),
            (
                "surface-level failure metadata cannot activate runtime nodes",
                "source records remain passive when validation fails",
            ),
            "Surface failures remain validation metadata instead of runtime nodes.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "parallel_scheduler",
                "workflow_replay_engine",
                "execution_replay_engine",
                "bottleneck_detection",
            ),
            (
                "terminal_failure_route_not_implemented",
                "workflow_execution_implemented_false",
            ),
            (
                "terminal failures cannot execute or replay workflows",
                "failure routing remains documented as blocked metadata",
            ),
            "Terminal failure handling is audit metadata, not workflow routing.",
        ),
        _record(
            "provider_failures",
            (
                "load_balancer",
                "execution_replay_engine",
                "bottleneck_detection",
                "performance_prediction",
            ),
            (
                "provider_model_routing_implemented_false",
                "provider_failover_not_declared",
            ),
            (
                "provider failures cannot call, switch, or rebalance providers",
                "provider capability pressure remains advisory metadata",
            ),
            "Provider failures are represented as blocked routing boundaries.",
        ),
        _record(
            "model_routing_failures",
            (
                "load_balancer",
                "performance_prediction",
                "performance_regression_detection",
                "resource_utilization_optimizer",
            ),
            (
                "provider_model_routing_implemented_false",
                "model_selection_not_declared",
            ),
            (
                "model-routing failures cannot switch the selected model",
                "performance metadata cannot rewrite route decisions",
            ),
            "Model-routing failures stop at passive advisory metadata.",
        ),
        _record(
            "stream_failures",
            (
                "async_execution",
                "streaming_optimizer",
                "throughput_optimizer",
                "performance_benchmarking",
            ),
            (
                "stream_payload_mutation_implemented_false",
                "stream_event_emission_change_implemented_false",
            ),
            (
                "stream failures cannot alter emitted events or token payloads",
                "stream readiness metadata cannot batch runtime chunks",
            ),
            "Stream failures remain readiness metadata without runtime emission.",
        ),
        _record(
            "scheduling_failures",
            (
                "parallel_scheduler",
                "async_execution",
                "load_balancer",
                "throughput_optimizer",
            ),
            (
                "parallel_execution_implemented_false",
                "async_runtime_execution_implemented_false",
            ),
            (
                "scheduling failures cannot create async tasks or parallel work",
                "load and throughput candidates cannot change concurrency",
            ),
            "Scheduling failures cannot alter runtime scheduling or concurrency.",
        ),
        _record(
            "retry_failures",
            (
                "retry_policies",
                "workflow_replay_engine",
                "execution_replay_engine",
                "bottleneck_detection",
            ),
            (
                "retry_triggering_implemented_false",
                "workflow_replay_execution_implemented_false",
            ),
            (
                "retry pressure cannot trigger refinement or replay execution",
                "replay failure metadata cannot start another attempt",
            ),
            "Retry failures are advisory and cannot start execution or replay.",
        ),
        _record(
            "planning_helper_failures",
            (
                "latency_optimizer",
                "execution_profiling",
                "performance_prediction",
                "performance_benchmarking",
                "reasoning_budget_optimizer",
                "resource_utilization_optimizer",
            ),
            (
                "lookup_helpers_return_records_only",
                "validators_reject_mismatched_counts",
            ),
            (
                "planning helper failures return empty metadata or validation errors",
                "helper failures cannot mutate workflow or output state",
            ),
            "Planning/helper failures remain pydantic and lookup boundaries.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "reasoning_budget_optimizer",
                "performance_regression_detection",
                "resource_utilization_optimizer",
            ),
            (
                "prompt_mutation_implemented_false",
                "generated_output_mutation_implemented_false",
            ),
            (
                "performance metadata failures cannot render or rewrite prompts",
                "budget and regression metadata cannot alter generated output",
            ),
            "Prompt rendering failures are blocked because prompts are not mutated.",
        ),
        _record(
            "serialization_failures",
            _SOURCE_SURFACE_IDS,
            (
                "serialization_versions_end_with_v1",
                "record_totals_are_validator_checked",
            ),
            (
                "serialization mismatch raises validation failure",
                "versioned metadata cannot activate performance behavior",
            ),
            "Serialization failures stop at versioned metadata validation.",
        ),
        _record(
            "registry_import_loading_failures",
            _SOURCE_SURFACE_IDS,
            (
                "lazy_exports_resolve_metadata_helpers",
                "surface_ids_match_architecture_registry",
            ),
            (
                "registry loading failures cannot execute workflows",
                "missing exports fail as import errors rather than runtime actions",
            ),
            "Registry loading failures are isolated to import and validation paths.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "performance_prediction",
                "performance_benchmarking",
                "reasoning_budget_optimizer",
                "resource_utilization_optimizer",
            ),
            (
                "budget_enforcement_implemented_false",
                "performance_predictions_are_relative_metadata",
            ),
            (
                "budget pressure cannot enforce spend or block execution",
                "performance prediction failures remain advisory estimates",
            ),
            "Budget, cost, and performance prediction failures cannot enforce policy.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "execution_profiling",
                "workflow_replay_engine",
                "execution_replay_engine",
                "bottleneck_detection",
            ),
            (
                "workflow_control_implemented_false",
                "records_are_frozen_models",
            ),
            (
                "profiling and replay failures cannot advance workflow state",
                "bottleneck failures cannot modify graph order or outputs",
            ),
            "Workflow state remains untouched after performance metadata failures.",
        ),
        _record(
            "provider_model_routing_preservation",
            _SOURCE_SURFACE_IDS,
            (
                "provider_model_routing_implemented_false",
                "routing_boundary_flags_present",
            ),
            (
                "audit metadata cannot change provider or model selection",
                "performance surfaces describe pressure without choosing routes",
            ),
            "Provider/model routing preservation is enforced by false active flags.",
        ),
        _record(
            "generated_output_mutation_boundaries",
            _SOURCE_SURFACE_IDS,
            (
                "generated_output_mutation_implemented_false",
                "persistent_storage_write_implemented_false",
            ),
            (
                "failure metadata cannot modify generated artifacts",
                "audit records cannot persist or rewrite user-visible output",
            ),
            "Generated-output mutation remains blocked on every failure path.",
        ),
        _record(
            "passive_registry_activation_boundaries",
            _SOURCE_SURFACE_IDS,
            (
                "metadata_only_true",
                "runtime_evolution_implemented_false",
            ),
            (
                "passive registry reads cannot activate runtime behavior",
                "Runtime Evolution cannot be applied without a human gate",
            ),
            "Passive registry activation is limited to metadata construction.",
        ),
    )


def _record(
    check_kind: PerformanceFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> PerformanceFailurePathAuditRecord:
    return PerformanceFailurePathAuditRecord(
        audit_id=f"performance_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
