"""V5.1 runtime failure path audit metadata for execution optimization."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

FailureAuditCheckKind = Literal[
    "terminal_failure_routing",
    "provider_model_routing_preservation",
    "retry_failure_boundary",
    "planning_helper_validation",
    "serialization_guard",
    "cache_failure_mode",
    "budget_cost_prediction_boundary",
    "workflow_state_output_boundary",
    "metadata_activation_boundary",
]
FailureAuditStatus = Literal["pass"]

EXECUTION_OPTIMIZATION_FAILURE_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "execution_optimization_failure_audit_record.v1"
)
EXECUTION_OPTIMIZATION_FAILURE_AUDIT_SERIALIZATION_VERSION = (
    "execution_optimization_failure_audit.v1"
)
EXECUTION_OPTIMIZATION_FAILURE_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.1 execution optimization failure audit metadata verifies failure-path "
    "coverage, runtime boundary flags, serialization guards, cache miss/stale "
    "behavior, budget and cost prediction boundaries, and metadata activation "
    "limits only; it does not execute workflows, trigger retries, route "
    "providers or models, enforce budgets, mutate prompts, write storage, or "
    "modify generated output."
)

_SOURCE_SURFACE_IDS = (
    "execution_graph_analysis",
    "workflow_cost_analysis",
    "workflow_complexity_analysis",
    "creative_complexity_analysis",
    "context_budget_plan",
    "exploration_budget_plan",
    "context_routing_plan",
    "prompt_compression_result",
    "retrieval_compression_result",
    "memory_summarization_result",
    "execution_cache_lookup",
    "context_reuse_plan",
    "workflow_pruning_plan",
    "execution_cost_forecast",
    "execution_path_optimization_plan",
    "execution_strategy_selection",
)
_APPLICABLE_REQUIRED_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "retry_failures",
    "planning_helper_failures",
    "serialization_failures",
    "cache_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
_NOT_APPLICABLE_REQUIRED_CHECKS = (
    "stream_failures",
    "scheduling_failures",
    "preview_workstation_frontend_backend_failures",
    "telemetry_observability_failures",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_execution",
    "retry_or_refinement_triggering",
    "provider_or_model_routing",
    "budget_enforcement",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionOptimizationFailureAuditRecord(BaseModel):
    """One V5.1 execution optimization runtime failure audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=180)
    check_kind: FailureAuditCheckKind
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    audit_status: FailureAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
    )
    runtime_failure_audit_implemented: Literal[True] = True
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_optimization_failure_audit_record.v1"] = (
        EXECUTION_OPTIMIZATION_FAILURE_AUDIT_RECORD_SERIALIZATION_VERSION
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
            raise ValueError("source_surface_ids must reference known V5.1 surfaces")
        return self


class ExecutionOptimizationFailureAuditRegistry(BaseModel):
    """V5.1 capability-scoped runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_optimization_failure_audit"] = (
        "execution_optimization_failure_audit"
    )
    serialization_version: Literal["execution_optimization_failure_audit.v1"] = (
        EXECUTION_OPTIMIZATION_FAILURE_AUDIT_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_OPTIMIZATION_FAILURE_AUDIT_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=16, max_length=16)
    applicable_required_checks: tuple[str, ...] = Field(min_length=13, max_length=13)
    not_applicable_required_checks: tuple[str, ...] = Field(min_length=4, max_length=4)
    records: tuple[ExecutionOptimizationFailureAuditRecord, ...] = Field(
        min_length=9,
        max_length=9,
    )
    audit_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    check_kinds: tuple[FailureAuditCheckKind, ...] = Field(min_length=9, max_length=9)
    record_count: int = Field(ge=9, le=9)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    runtime_failure_audit_implemented: Literal[True] = True
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=10,
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
            raise ValueError("source_surface_ids must match V5.1 surfaces")
        if self.applicable_required_checks != _APPLICABLE_REQUIRED_CHECKS:
            raise ValueError("applicable_required_checks must match audit checklist")
        if self.not_applicable_required_checks != _NOT_APPLICABLE_REQUIRED_CHECKS:
            raise ValueError("not_applicable_required_checks must match checklist")
        derived_check_kinds = tuple(record.check_kind for record in self.records)
        if self.check_kinds != derived_check_kinds:
            raise ValueError("check_kinds must match records")
        if len(set(derived_check_kinds)) != len(derived_check_kinds):
            raise ValueError("check_kinds must be unique")
        return self


def execution_optimization_failure_audit_registry() -> (
    ExecutionOptimizationFailureAuditRegistry
):
    """Return V5.1 runtime failure path audit metadata without runtime actions."""

    records = _records()
    return ExecutionOptimizationFailureAuditRegistry(
        source_surface_ids=_SOURCE_SURFACE_IDS,
        applicable_required_checks=_APPLICABLE_REQUIRED_CHECKS,
        not_applicable_required_checks=_NOT_APPLICABLE_REQUIRED_CHECKS,
        records=records,
        audit_ids=tuple(record.audit_id for record in records),
        check_kinds=tuple(record.check_kind for record in records),
        record_count=len(records),
    )


def execution_optimization_failure_audit_by_id(
    audit_id: str,
    registry: ExecutionOptimizationFailureAuditRegistry | None = None,
) -> ExecutionOptimizationFailureAuditRecord | None:
    """Return one failure audit record without executing failure behavior."""

    source_registry = registry or execution_optimization_failure_audit_registry()
    for record in source_registry.records:
        if record.audit_id == audit_id:
            return record
    return None


def execution_optimization_failure_audits_for_check(
    check_kind: FailureAuditCheckKind,
    registry: ExecutionOptimizationFailureAuditRegistry | None = None,
) -> tuple[ExecutionOptimizationFailureAuditRecord, ...]:
    """Return failure audit records by check kind without runtime activation."""

    source_registry = registry or execution_optimization_failure_audit_registry()
    return tuple(record for record in source_registry.records if record.check_kind == check_kind)


def _records() -> tuple[ExecutionOptimizationFailureAuditRecord, ...]:
    return (
        _record(
            "terminal_failure_routing",
            ("execution_graph_analysis", "workflow_complexity_analysis"),
            (
                "failure_terminal_path_present",
                "failure_path_reachable_true",
            ),
            (
                "terminal failure routing remains modeled and non-executing",
                "failure edges are observed without graph compilation",
            ),
        ),
        _record(
            "provider_model_routing_preservation",
            _SOURCE_SURFACE_IDS,
            (
                "provider_model_routing_implemented_false",
                "provider_or_model_routing_blocked",
            ),
            (
                "optimization metadata cannot change provider routing",
                "model routing remains outside V5.1 execution optimization",
            ),
        ),
        _record(
            "retry_failure_boundary",
            (
                "execution_graph_analysis",
                "workflow_cost_analysis",
                "workflow_pruning_plan",
                "execution_path_optimization_plan",
                "execution_strategy_selection",
            ),
            (
                "bounded_retry_cycle_detected",
                "retry_triggering_implemented_false",
            ),
            (
                "retry paths are budgeted and ranked without triggering retries",
                "strategy selection does not apply retry behavior",
            ),
        ),
        _record(
            "planning_helper_validation",
            (
                "context_budget_plan",
                "exploration_budget_plan",
                "context_routing_plan",
                "context_reuse_plan",
            ),
            (
                "pydantic_validators_cover_totals",
                "lookup_helpers_return_metadata_only",
            ),
            (
                "planning helper failures fail validation instead of mutating state",
                "context routing and reuse remain advisory metadata",
            ),
        ),
        _record(
            "serialization_guard",
            _SOURCE_SURFACE_IDS,
            (
                "serialization_versions_v1",
                "model_validators_reject_mismatched_totals",
            ),
            (
                "serialization mismatch produces validation failure",
                "versioned metadata does not activate runtime behavior",
            ),
        ),
        _record(
            "cache_failure_mode",
            ("execution_cache_lookup",),
            (
                "cache_miss_and_stale_states_modeled",
                "persistent_storage_write_implemented_false",
            ),
            (
                "cache failure remains in-memory hit miss stale metadata",
                "cache layer cannot write persistent storage or network caches",
            ),
        ),
        _record(
            "budget_cost_prediction_boundary",
            (
                "workflow_cost_analysis",
                "context_budget_plan",
                "exploration_budget_plan",
                "execution_cost_forecast",
            ),
            (
                "budget_enforcement_implemented_false",
                "cost_based_routing_implemented_false",
            ),
            (
                "cost prediction failures remain advisory estimates",
                "budget pressure cannot enforce budgets or route by cost",
            ),
        ),
        _record(
            "workflow_state_output_boundary",
            (
                "prompt_compression_result",
                "retrieval_compression_result",
                "memory_summarization_result",
                "execution_strategy_selection",
            ),
            (
                "source_artifacts_preserved",
                "generated_output_mutation_implemented_false",
            ),
            (
                "compression and summarization produce separate artifacts",
                "generated output mutation remains blocked after failures",
            ),
        ),
        _record(
            "metadata_activation_boundary",
            _SOURCE_SURFACE_IDS,
            (
                "metadata_only_or_planning_only_flags_true",
                "strategy_application_implemented_false",
            ),
            (
                "metadata-only records cannot trigger runtime behavior",
                "selected strategies remain advisory until explicit downstream use",
            ),
        ),
    )


def _record(
    check_kind: FailureAuditCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
) -> ExecutionOptimizationFailureAuditRecord:
    return ExecutionOptimizationFailureAuditRecord(
        audit_id=f"execution_optimization_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
    )
