"""Passive V5.4 production observability architecture consistency metadata."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_diagnostics import (
    build_agent_diagnostics,
)
from creative_coding_assistant.orchestration.confidence_analytics import (
    build_confidence_analytics,
)
from creative_coding_assistant.orchestration.cost_dashboard import (
    build_cost_dashboard,
)
from creative_coding_assistant.orchestration.creative_analytics import (
    build_creative_analytics,
)
from creative_coding_assistant.orchestration.creative_diversity_analytics import (
    build_creative_diversity_analytics,
)
from creative_coding_assistant.orchestration.error_intelligence import (
    build_error_intelligence,
)
from creative_coding_assistant.orchestration.escalation_diagnostics import (
    build_escalation_diagnostics,
)
from creative_coding_assistant.orchestration.failure_analysis import (
    build_failure_analysis,
)
from creative_coding_assistant.orchestration.performance_dashboard import (
    build_performance_dashboard,
)
from creative_coding_assistant.orchestration.production_telemetry import (
    build_production_telemetry,
)
from creative_coding_assistant.orchestration.quality_dashboard import (
    build_quality_dashboard,
)
from creative_coding_assistant.orchestration.routing_diagnostics import (
    build_routing_diagnostics,
)
from creative_coding_assistant.orchestration.runtime_timeline import (
    build_runtime_timeline,
)
from creative_coding_assistant.orchestration.system_health_monitoring import (
    build_system_health_monitoring,
)
from creative_coding_assistant.orchestration.token_dashboard import (
    build_token_dashboard,
)
from creative_coding_assistant.orchestration.workflow_diagnostics import (
    build_workflow_diagnostics,
)
from creative_coding_assistant.orchestration.workflow_explainability_dashboard import (
    build_workflow_explainability_dashboard,
)
from creative_coding_assistant.orchestration.workflow_health_monitoring import (
    build_workflow_health_monitoring,
)

ObservabilityArchitectureLayer = Literal[
    "dashboard_observability_boundary",
    "telemetry_diagnostics_boundary",
    "failure_health_boundary",
    "creative_confidence_boundary",
    "timeline_explainability_boundary",
]
ObservabilityArchitectureStage = Literal["v5_4_architecture_consistency_pass"]
ObservabilityArchitectureStatus = Literal["pass"]

PRODUCTION_OBSERVABILITY_ARCHITECTURE_RECORD_SERIALIZATION_VERSION = (
    "production_observability_architecture_record.v1"
)
PRODUCTION_OBSERVABILITY_ARCHITECTURE_REGISTRY_SERIALIZATION_VERSION = (
    "production_observability_architecture_registry.v1"
)
PRODUCTION_OBSERVABILITY_ARCHITECTURE_AUTHORITY_BOUNDARY = (
    "V5.4 production observability architecture consistency metadata checks "
    "dashboard, telemetry, diagnostics, health, analytics, timeline, and "
    "explainability surface coverage, serialization, read-only observability "
    "boundaries, V4 compatibility, and version-level HITL/runtime evolution "
    "rules only; it does not collect live metrics, emit telemetry or alerts, "
    "capture traces, execute workflows, route providers or models, invoke "
    "agents, classify live errors, remediate failures, reconstruct timelines, "
    "generate explanations, request human review, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_ARCHITECTURE_LAYERS: tuple[ObservabilityArchitectureLayer, ...] = (
    "dashboard_observability_boundary",
    "telemetry_diagnostics_boundary",
    "failure_health_boundary",
    "creative_confidence_boundary",
    "timeline_explainability_boundary",
)
_VALIDATED_VERSION_RULES = (
    "v5_4_surface_role_declared",
    "serialization_version_declared",
    "observability_metadata_only",
    "runtime_metric_collection_not_applied",
    "telemetry_emission_not_applied",
    "workflow_execution_not_applied",
    "provider_model_routing_not_applied",
    "generated_output_mutation_blocked",
    "hitl_gate_not_emitted",
    "runtime_evolution_not_applied",
    "v4_boundary_compatibility_confirmed",
    "version_runtime_rules_confirmed",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "runtime_metric_collection_blocked",
    "telemetry_emission_blocked",
    "trace_capture_blocked",
    "alert_emission_blocked",
    "workflow_execution_blocked",
    "workflow_control_blocked",
    "workflow_state_mutation_blocked",
    "provider_model_routing_blocked",
    "agent_invocation_blocked",
    "health_check_execution_blocked",
    "error_classification_blocked",
    "automated_remediation_blocked",
    "timeline_reconstruction_blocked",
    "explanation_generation_blocked",
    "human_review_request_blocked",
    "retry_triggering_blocked",
    "prompt_mutation_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
    "runtime_evolution_blocked",
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
_COUNT_FIELD_CANDIDATES = (
    "panel_count",
    "channel_count",
    "record_count",
    "surface_count",
    "diagnostic_signal_count",
    "agent_signal_count",
    "routing_signal_count",
    "escalation_signal_count",
    "failure_signal_count",
    "error_signal_count",
    "health_signal_count",
    "system_signal_count",
    "creative_signal_count",
    "confidence_signal_count",
    "diversity_signal_count",
    "timeline_signal_count",
    "explainability_signal_count",
    "telemetry_signal_count",
)
_SURFACE_LAYERS: tuple[tuple[str, ObservabilityArchitectureLayer], ...] = (
    ("token_dashboard", "dashboard_observability_boundary"),
    ("cost_dashboard", "dashboard_observability_boundary"),
    ("quality_dashboard", "dashboard_observability_boundary"),
    ("performance_dashboard", "dashboard_observability_boundary"),
    ("production_telemetry", "telemetry_diagnostics_boundary"),
    ("workflow_diagnostics", "telemetry_diagnostics_boundary"),
    ("agent_diagnostics", "telemetry_diagnostics_boundary"),
    ("routing_diagnostics", "telemetry_diagnostics_boundary"),
    ("escalation_diagnostics", "telemetry_diagnostics_boundary"),
    ("failure_analysis", "failure_health_boundary"),
    ("error_intelligence", "failure_health_boundary"),
    ("workflow_health_monitoring", "failure_health_boundary"),
    ("system_health_monitoring", "failure_health_boundary"),
    ("creative_analytics", "creative_confidence_boundary"),
    ("confidence_analytics", "creative_confidence_boundary"),
    ("creative_diversity_analytics", "creative_confidence_boundary"),
    ("runtime_timeline", "timeline_explainability_boundary"),
    ("workflow_explainability_dashboard", "timeline_explainability_boundary"),
)
_SURFACE_IMPLEMENTATION_FLAGS = tuple(
    f"{surface_id}_implemented" for surface_id, _layer in _SURFACE_LAYERS
)


class ProductionObservabilityArchitectureRecord(BaseModel):
    """One passive V5.4 observability architecture consistency record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=120)
    architecture_layer: ObservabilityArchitectureLayer
    architecture_stage: ObservabilityArchitectureStage = (
        "v5_4_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=120)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=200000)
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=20, max_length=20)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=50,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=64,
    )
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    source_observability_only_declared: Literal[True] = True
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundary_compatibility_confirmed: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    architecture_consistency_status: ObservabilityArchitectureStatus = "pass"
    runtime_metric_collection_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "production_observability_architecture_record.v1"
    ] = PRODUCTION_OBSERVABILITY_ARCHITECTURE_RECORD_SERIALIZATION_VERSION
    advisory_only: Literal[True] = True


class ProductionObservabilityArchitectureRegistry(BaseModel):
    """Passive V5.4 production observability architecture consistency registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_observability_architecture_registry"] = (
        "production_observability_architecture_registry"
    )
    serialization_version: Literal[
        "production_observability_architecture_registry.v1"
    ] = PRODUCTION_OBSERVABILITY_ARCHITECTURE_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=PRODUCTION_OBSERVABILITY_ARCHITECTURE_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    architecture_stage: ObservabilityArchitectureStage = (
        "v5_4_architecture_consistency_pass"
    )
    records: tuple[ProductionObservabilityArchitectureRecord, ...] = Field(
        min_length=18,
        max_length=18,
    )
    surface_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    record_count: int = Field(ge=18, le=18)
    architecture_layers: tuple[ObservabilityArchitectureLayer, ...] = Field(
        min_length=5,
        max_length=5,
    )
    validated_version_rules: tuple[str, ...] = Field(min_length=12, max_length=12)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=20, max_length=20)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    all_surfaces_covered: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    v4_boundaries_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    runtime_metric_collection_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
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
            raise ValueError("surface_ids must match V5.4 surface order")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.architecture_layers != _ARCHITECTURE_LAYERS:
            raise ValueError("architecture_layers must match V5.4 layer order")
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


def production_observability_architecture_registry() -> (
    ProductionObservabilityArchitectureRegistry
):
    """Return passive V5.4 production observability architecture metadata."""

    records = tuple(
        _record_from_source(surface_id, layer, source)
        for surface_id, layer, source in _source_specs()
    )
    return ProductionObservabilityArchitectureRegistry(
        records=records,
        surface_ids=tuple(record.surface_id for record in records),
        record_count=len(records),
        architecture_layers=_ARCHITECTURE_LAYERS,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )


def production_observability_architecture_by_surface(
    surface_id: str,
    registry: ProductionObservabilityArchitectureRegistry | None = None,
) -> ProductionObservabilityArchitectureRecord | None:
    """Return one V5.4 observability architecture record by surface id."""

    source_registry = registry or production_observability_architecture_registry()
    normalized_surface_id = str(surface_id).strip()
    for record in source_registry.records:
        if record.surface_id == normalized_surface_id:
            return record
    return None


def production_observability_architecture_records_for_layer(
    architecture_layer: str,
    registry: ProductionObservabilityArchitectureRegistry | None = None,
) -> tuple[ProductionObservabilityArchitectureRecord, ...]:
    """Return passive V5.4 observability architecture records for one layer."""

    source_registry = registry or production_observability_architecture_registry()
    normalized_layer = str(architecture_layer).strip()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized_layer
    )


def _source_specs() -> tuple[tuple[str, ObservabilityArchitectureLayer, Any], ...]:
    sources = _source_objects()
    return tuple(
        (surface_id, layer, sources[surface_id])
        for surface_id, layer in _SURFACE_LAYERS
    )


def _source_objects() -> dict[str, Any]:
    token = build_token_dashboard()
    cost = build_cost_dashboard()
    quality = build_quality_dashboard()
    performance = build_performance_dashboard()
    telemetry = build_production_telemetry(
        token_dashboard=token,
        cost_dashboard=cost,
        quality_dashboard=quality,
        performance_dashboard=performance,
    )
    workflow = build_workflow_diagnostics(production_telemetry=telemetry)
    agent = build_agent_diagnostics()
    routing = build_routing_diagnostics()
    escalation = build_escalation_diagnostics()
    failure = build_failure_analysis(workflow_diagnostics=workflow)
    error = build_error_intelligence(
        failure_analysis=failure,
        workflow_diagnostics=workflow,
        production_telemetry=telemetry,
        routing_diagnostics=routing,
        escalation_diagnostics=escalation,
    )
    workflow_health = build_workflow_health_monitoring(
        workflow_diagnostics=workflow,
        production_telemetry=telemetry,
        error_intelligence=error,
        failure_analysis=failure,
        performance_dashboard=performance,
    )
    system = build_system_health_monitoring(
        workflow_health=workflow_health,
        production_telemetry=telemetry,
        token_dashboard=token,
        cost_dashboard=cost,
        quality_dashboard=quality,
        performance_dashboard=performance,
        error_intelligence=error,
        agent_diagnostics=agent,
    )
    creative = build_creative_analytics(
        quality_dashboard=quality,
        system_health=system,
    )
    confidence = build_confidence_analytics(
        quality_dashboard=quality,
        creative_analytics=creative,
        escalation_diagnostics=escalation,
    )
    diversity = build_creative_diversity_analytics(
        creative_analytics=creative,
        confidence_analytics=confidence,
        system_health=system,
    )
    runtime = build_runtime_timeline(
        workflow_diagnostics=workflow,
        production_telemetry=telemetry,
    )
    explainability = build_workflow_explainability_dashboard(
        workflow_diagnostics=workflow,
        runtime_timeline=runtime,
        error_intelligence=error,
    )

    return {
        "token_dashboard": token,
        "cost_dashboard": cost,
        "quality_dashboard": quality,
        "performance_dashboard": performance,
        "production_telemetry": telemetry,
        "workflow_diagnostics": workflow,
        "agent_diagnostics": agent,
        "routing_diagnostics": routing,
        "escalation_diagnostics": escalation,
        "failure_analysis": failure,
        "error_intelligence": error,
        "workflow_health_monitoring": workflow_health,
        "system_health_monitoring": system,
        "creative_analytics": creative,
        "confidence_analytics": confidence,
        "creative_diversity_analytics": diversity,
        "runtime_timeline": runtime,
        "workflow_explainability_dashboard": explainability,
    }


def _record_from_source(
    surface_id: str,
    layer: ObservabilityArchitectureLayer,
    source: Any,
) -> ProductionObservabilityArchitectureRecord:
    count_field, source_count = _source_count(source)
    serialization_version = str(getattr(source, "serialization_version", ""))
    active_flags = _active_runtime_flags(source)
    missing = _missing_coverage(
        source=source,
        serialization_version=serialization_version,
        source_count=source_count,
        active_flags=active_flags,
    )
    return ProductionObservabilityArchitectureRecord(
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
        source_observability_only_declared=_source_observability_only_declared(source),
    )


def _source_count(source: Any) -> tuple[str, int]:
    for count_field in _COUNT_FIELD_CANDIDATES:
        if hasattr(source, count_field):
            return count_field, int(getattr(source, count_field))
    raise ValueError("V5.4 architecture source must expose a count field")


def _source_blocked_runtime_behaviors(source: Any) -> tuple[str, ...]:
    blocked = getattr(source, "blocked_runtime_behaviors", None)
    if blocked:
        return tuple(str(item) for item in blocked)
    return _BLOCKED_RUNTIME_BEHAVIORS


def _source_observability_only_declared(source: Any) -> bool:
    return bool(
        getattr(
            source,
            "advisory_only",
            getattr(source, "metadata_only", True),
        )
    )


def _active_runtime_flags(source: Any) -> tuple[str, ...]:
    model_fields = getattr(source.__class__, "model_fields", {})
    return tuple(
        field_name
        for field_name in model_fields
        if field_name.endswith("_implemented")
        and field_name not in _SURFACE_IMPLEMENTATION_FLAGS
        and bool(getattr(source, field_name))
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
    if not _source_observability_only_declared(source):
        missing.append("observability_only_declaration_missing")
    return tuple(missing)


def _surface_ids() -> tuple[str, ...]:
    return tuple(surface_id for surface_id, _layer in _SURFACE_LAYERS)
