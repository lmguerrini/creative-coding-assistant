"""Passive V5.4 runtime failure path audit metadata for observability."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .production_observability_architecture_consistency import (
    production_observability_architecture_registry,
)

ProductionObservabilityFailurePathCheckKind = Literal[
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
    "telemetry_observability_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
]
ProductionObservabilityFailurePathAuditStatus = Literal["pass"]

PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "production_observability_failure_path_audit_record.v1"
)
PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "production_observability_failure_path_audit_registry.v1"
)
PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.4 production observability runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage, observability "
    "dashboard and telemetry failure boundaries, diagnostics and health failure "
    "limits, analytics and explainability failure limits, serialization and "
    "registry loading guards, provider/model routing preservation, "
    "generated-output mutation limits, and passive activation limits only; it "
    "does not collect live metrics, emit telemetry or alerts, capture traces, "
    "execute workflows, route providers or models, invoke agents, classify live "
    "errors, remediate failures, reconstruct timelines, generate explanations, "
    "request human review, trigger retries, mutate prompts, write storage, "
    "modify generated output, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
    "production_telemetry",
    "workflow_diagnostics",
    "agent_diagnostics",
    "routing_diagnostics",
    "escalation_diagnostics",
    "failure_analysis",
    "error_intelligence",
    "workflow_health_monitoring",
    "system_health_monitoring",
    "creative_analytics",
    "confidence_analytics",
    "creative_diversity_analytics",
    "runtime_timeline",
    "workflow_explainability_dashboard",
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
_APPLICABLE_REQUIRED_CHECKS: tuple[
    ProductionObservabilityFailurePathCheckKind,
    ...,
] = (
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
    "telemetry_observability_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
_NOT_APPLICABLE_REQUIRED_CHECKS = (
    "preview_workstation_frontend_backend_failures",
    "cache_failures",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_metric_collection",
    "telemetry_emission",
    "trace_capture_or_export",
    "alert_emission",
    "workflow_execution",
    "workflow_control",
    "workflow_state_or_graph_mutation",
    "provider_or_model_routing",
    "provider_execution",
    "agent_or_node_invocation",
    "health_check_execution",
    "live_error_classification",
    "automated_remediation",
    "timeline_reconstruction",
    "decision_provenance_recording",
    "explanation_generation",
    "human_review_request",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "memory_or_persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ProductionObservabilityFailurePathAuditRecord(BaseModel):
    """One passive V5.4 observability runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=200)
    check_kind: ProductionObservabilityFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=18)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=300)
    audit_status: ProductionObservabilityFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    live_metric_collection_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "production_observability_failure_path_audit_record.v1"
    ] = PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V5.4 surfaces")
        return self


class ProductionObservabilityFailurePathAuditRegistry(BaseModel):
    """Passive V5.4 observability runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_observability_failure_path_audit_registry"] = (
        "production_observability_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "production_observability_failure_path_audit_registry.v1"
    ] = PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=PRODUCTION_OBSERVABILITY_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    architecture_registry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    architecture_registry_record_count: int = Field(ge=18, le=18)
    source_surface_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        ProductionObservabilityFailurePathCheckKind,
        ...,
    ] = Field(min_length=17, max_length=17)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[ProductionObservabilityFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[ProductionObservabilityFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    telemetry_observability_failure_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    live_metric_collection_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
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
            raise ValueError("source_surface_ids must match V5.4 surfaces")
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


def production_observability_failure_path_audit_registry(
) -> ProductionObservabilityFailurePathAuditRegistry:
    """Return V5.4 runtime failure path audit metadata without runtime actions."""

    architecture_registry = production_observability_architecture_registry()
    records = _records()
    return ProductionObservabilityFailurePathAuditRegistry(
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


def production_observability_failure_path_audit_by_id(
    audit_id: str,
    registry: ProductionObservabilityFailurePathAuditRegistry | None = None,
) -> ProductionObservabilityFailurePathAuditRecord | None:
    """Return one V5.4 failure audit record without activating failures."""

    source_registry = registry or production_observability_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def production_observability_failure_path_audits_for_check(
    check_kind: ProductionObservabilityFailurePathCheckKind,
    registry: ProductionObservabilityFailurePathAuditRegistry | None = None,
) -> tuple[ProductionObservabilityFailurePathAuditRecord, ...]:
    """Return V5.4 failure audit records by checklist item."""

    source_registry = registry or production_observability_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def production_observability_failure_path_audits_for_surface(
    surface_id: str,
    registry: ProductionObservabilityFailurePathAuditRegistry | None = None,
) -> tuple[ProductionObservabilityFailurePathAuditRecord, ...]:
    """Return V5.4 failure audit records for one observability surface."""

    source_registry = registry or production_observability_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _records() -> tuple[ProductionObservabilityFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_architecture_registry",
                "each_surface_declares_observability_metadata_boundary",
            ),
            (
                "surface-level failures cannot activate runtime nodes",
                "source records remain passive when validation fails",
            ),
            "Surface failures remain validation metadata instead of runtime nodes.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "workflow_diagnostics",
                "failure_analysis",
                "error_intelligence",
                "workflow_health_monitoring",
                "runtime_timeline",
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
                "routing_diagnostics",
                "cost_dashboard",
                "quality_dashboard",
                "performance_dashboard",
                "system_health_monitoring",
            ),
            (
                "provider_model_routing_implemented_false",
                "provider_execution_implemented_false",
            ),
            (
                "provider failures cannot call, switch, or fail over providers",
                "observability pressure remains read-only metadata",
            ),
            "Provider failures are represented as blocked routing boundaries.",
        ),
        _record(
            "model_routing_failures",
            (
                "routing_diagnostics",
                "quality_dashboard",
                "confidence_analytics",
                "workflow_explainability_dashboard",
            ),
            (
                "provider_model_routing_implemented_false",
                "automatic_model_selection_not_declared",
            ),
            (
                "model-routing failures cannot switch the selected model",
                "confidence and explainability metadata cannot rewrite routes",
            ),
            "Model-routing failures stop at passive observability metadata.",
        ),
        _record(
            "stream_failures",
            (
                "production_telemetry",
                "workflow_diagnostics",
                "runtime_timeline",
                "workflow_explainability_dashboard",
            ),
            (
                "telemetry_emission_implemented_false",
                "trace_capture_implemented_false",
            ),
            (
                "stream failures cannot emit telemetry or traces",
                "timeline metadata cannot replay or export runtime events",
            ),
            "Stream failures remain observability metadata without emission.",
        ),
        _record(
            "scheduling_failures",
            (
                "workflow_diagnostics",
                "workflow_health_monitoring",
                "system_health_monitoring",
                "performance_dashboard",
            ),
            (
                "workflow_control_implemented_false",
                "workflow_state_mutation_implemented_false",
            ),
            (
                "scheduling failures cannot create async tasks or change order",
                "health metadata cannot modify workflow state or concurrency",
            ),
            "Scheduling failures cannot alter runtime scheduling or workflow state.",
        ),
        _record(
            "retry_failures",
            (
                "failure_analysis",
                "error_intelligence",
                "workflow_health_monitoring",
                "creative_diversity_analytics",
                "runtime_timeline",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "retry pressure cannot trigger refinement or replay execution",
                "health and analytics failures cannot start another attempt",
            ),
            "Retry failures are passive and cannot start execution or replay.",
        ),
        _record(
            "planning_helper_failures",
            (
                "token_dashboard",
                "cost_dashboard",
                "quality_dashboard",
                "performance_dashboard",
                "creative_analytics",
                "confidence_analytics",
                "creative_diversity_analytics",
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
                "creative_analytics",
                "confidence_analytics",
                "creative_diversity_analytics",
                "workflow_explainability_dashboard",
            ),
            (
                "prompt_mutation_implemented_false",
                "generated_output_mutation_implemented_false",
            ),
            (
                "observability metadata failures cannot render or rewrite prompts",
                "analytics metadata cannot alter generated output",
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
                "versioned metadata cannot activate observability behavior",
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
            "telemetry_observability_failures",
            (
                "production_telemetry",
                "workflow_diagnostics",
                "agent_diagnostics",
                "routing_diagnostics",
                "escalation_diagnostics",
                "runtime_timeline",
                "workflow_explainability_dashboard",
            ),
            (
                "live_metric_collection_implemented_false",
                "telemetry_emission_implemented_false",
            ),
            (
                "observability failures cannot collect live metrics",
                "telemetry failures cannot emit alerts, traces, or events",
            ),
            "Telemetry and observability failures remain read-only metadata.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "token_dashboard",
                "cost_dashboard",
                "quality_dashboard",
                "performance_dashboard",
                "system_health_monitoring",
            ),
            (
                "budget_enforcement_not_declared",
                "cost_signals_are_relative_metadata",
            ),
            (
                "budget pressure cannot enforce spend or block execution",
                "cost prediction failures remain advisory estimates",
            ),
            "Budget and cost failures cannot enforce policy or route execution.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "workflow_diagnostics",
                "workflow_health_monitoring",
                "failure_analysis",
                "runtime_timeline",
                "workflow_explainability_dashboard",
            ),
            (
                "workflow_control_implemented_false",
                "records_are_frozen_models",
            ),
            (
                "diagnostic and timeline failures cannot advance workflow state",
                "explainability failures cannot modify graph order or outputs",
            ),
            "Workflow state remains untouched after observability metadata failures.",
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
                "observability surfaces describe signals without choosing routes",
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
    check_kind: ProductionObservabilityFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> ProductionObservabilityFailurePathAuditRecord:
    return ProductionObservabilityFailurePathAuditRecord(
        audit_id=f"production_observability_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
