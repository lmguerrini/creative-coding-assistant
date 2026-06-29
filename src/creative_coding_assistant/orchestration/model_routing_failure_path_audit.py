"""Passive V5.2 runtime failure path audit metadata for model routing."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.model_routing_architecture_consistency import (
    model_routing_architecture_consistency_registry,
)

ModelRoutingFailurePathCheckKind = Literal[
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
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
ModelRoutingFailurePathAuditStatus = Literal["pass"]

MODEL_ROUTING_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "model_routing_failure_path_audit_record.v1"
)
MODEL_ROUTING_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "model_routing_failure_path_audit_registry.v1"
)
MODEL_ROUTING_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.2 model-routing runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage, source-surface "
    "failure boundaries, routing preservation, budget and prediction failure "
    "limits, serialization and registry loading guards, generated-output "
    "mutation limits, and passive activation limits only; it does not apply "
    "routing, select or switch models, execute providers, enforce budgets, "
    "emit HITL requests, control workflows, trigger retries, mutate prompts, "
    "write storage, modify generated output, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "model_router",
    "local_cloud_routing",
    "hybrid_routing",
    "quality_cost_optimizer",
    "cost_estimator",
    "budget_policies",
    "hitl_budget_gate",
    "runtime_recommendation_engine",
    "execution_policy_engine",
    "model_recommendation_engine",
    "model_capability_matrix",
    "provider_capability_matrix",
    "quality_prediction_engine",
    "cost_prediction_engine",
    "creative_quality_predictor",
    "creative_diversity_predictor",
    "creative_consistency_predictor",
    "routing_explainability",
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
_APPLICABLE_REQUIRED_CHECKS: tuple[ModelRoutingFailurePathCheckKind, ...] = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
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
    "stream_failures",
    "scheduling_failures",
    "preview_workstation_frontend_backend_failures",
    "telemetry_observability_failures",
    "cache_failures",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_or_model_routing",
    "configured_model_switching",
    "provider_execution",
    "budget_enforcement",
    "human_input_request_emission",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ModelRoutingFailurePathAuditRecord(BaseModel):
    """One passive V5.2 model-routing runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=180)
    check_kind: ModelRoutingFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=18)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=240)
    audit_status: ModelRoutingFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "model_routing_failure_path_audit_record.v1"
    ] = MODEL_ROUTING_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V5.2 surfaces")
        return self


class ModelRoutingFailurePathAuditRegistry(BaseModel):
    """Passive V5.2 model-routing runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_routing_failure_path_audit_registry"] = (
        "model_routing_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "model_routing_failure_path_audit_registry.v1"
    ] = MODEL_ROUTING_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=MODEL_ROUTING_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=1600,
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
    applicable_required_checks: tuple[ModelRoutingFailurePathCheckKind, ...] = Field(
        min_length=14,
        max_length=14,
    )
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    records: tuple[ModelRoutingFailurePathAuditRecord, ...] = Field(
        min_length=14,
        max_length=14,
    )
    audit_ids: tuple[str, ...] = Field(min_length=14, max_length=14)
    check_kinds: tuple[ModelRoutingFailurePathCheckKind, ...] = Field(
        min_length=14,
        max_length=14,
    )
    record_count: int = Field(ge=14, le=14)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
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
            raise ValueError("source_surface_ids must match V5.2 surfaces")
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


def model_routing_failure_path_audit_registry(
) -> ModelRoutingFailurePathAuditRegistry:
    """Return V5.2 runtime failure path audit metadata without runtime actions."""

    architecture_registry = model_routing_architecture_consistency_registry()
    records = _records()
    return ModelRoutingFailurePathAuditRegistry(
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


def model_routing_failure_path_audit_by_id(
    audit_id: str,
    registry: ModelRoutingFailurePathAuditRegistry | None = None,
) -> ModelRoutingFailurePathAuditRecord | None:
    """Return one V5.2 failure audit record without activating failure logic."""

    source_registry = registry or model_routing_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def model_routing_failure_path_audits_for_check(
    check_kind: ModelRoutingFailurePathCheckKind,
    registry: ModelRoutingFailurePathAuditRegistry | None = None,
) -> tuple[ModelRoutingFailurePathAuditRecord, ...]:
    """Return V5.2 failure audit records by checklist item."""

    source_registry = registry or model_routing_failure_path_audit_registry()
    return tuple(record for record in source_registry.records if record.check_kind == check_kind)


def model_routing_failure_path_audits_for_surface(
    surface_id: str,
    registry: ModelRoutingFailurePathAuditRegistry | None = None,
) -> tuple[ModelRoutingFailurePathAuditRecord, ...]:
    """Return V5.2 failure audit records for one model-routing surface."""

    source_registry = registry or model_routing_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _records() -> tuple[ModelRoutingFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_architecture_registry",
                "each_surface_declares_metadata_only_boundary",
            ),
            (
                "surface-level failure metadata cannot activate runtime nodes",
                "source records remain passive even when validation fails",
            ),
            "Surface failures remain validation metadata instead of runtime nodes.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "model_router",
                "local_cloud_routing",
                "hybrid_routing",
                "execution_policy_engine",
            ),
            (
                "terminal_route_application_absent",
                "routing_application_flags_false",
            ),
            (
                "terminal failures cannot choose a fallback provider or model",
                "policy failures remain advisory records until explicit use",
            ),
            "Terminal failure handling is documented as blocked routing behavior.",
        ),
        _record(
            "provider_failures",
            (
                "local_cloud_routing",
                "hybrid_routing",
                "provider_capability_matrix",
                "model_recommendation_engine",
            ),
            (
                "provider_execution_implemented_false",
                "provider_capability_rows_are_metadata",
            ),
            (
                "provider failures cannot call or replace a provider",
                "provider capability gaps are recorded without failover",
            ),
            "Provider failures are represented as advisory capability gaps.",
        ),
        _record(
            "model_routing_failures",
            (
                "model_router",
                "local_cloud_routing",
                "hybrid_routing",
                "model_capability_matrix",
                "model_recommendation_engine",
                "routing_explainability",
            ),
            (
                "provider_model_routing_implemented_false",
                "route_name_preserved_generate",
            ),
            (
                "model-routing failures cannot switch the selected model",
                "explainability failures cannot rewrite route decisions",
            ),
            "Model-routing failures stop at metadata validation boundaries.",
        ),
        _record(
            "retry_failures",
            (
                "hitl_budget_gate",
                "runtime_recommendation_engine",
                "execution_policy_engine",
            ),
            (
                "retry_triggering_implemented_false",
                "runtime_recommendations_are_not_applied",
            ),
            (
                "retry pressure cannot trigger refinement or provider calls",
                "runtime recommendations remain non-executing",
            ),
            "Retry failures are advisory and cannot start another attempt.",
        ),
        _record(
            "planning_helper_failures",
            (
                "quality_cost_optimizer",
                "cost_estimator",
                "budget_policies",
                "hitl_budget_gate",
                "runtime_recommendation_engine",
                "execution_policy_engine",
                "model_recommendation_engine",
            ),
            (
                "lookup_helpers_return_records_only",
                "validators_reject_mismatched_counts",
            ),
            (
                "helper lookup failures return empty metadata or validation errors",
                "planning helpers cannot mutate workflow state",
            ),
            "Planning/helper failures remain pydantic and lookup boundaries.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "creative_quality_predictor",
                "creative_diversity_predictor",
                "creative_consistency_predictor",
                "routing_explainability",
            ),
            (
                "prompt_mutation_implemented_false",
                "generated_output_mutation_implemented_false",
            ),
            (
                "creative prediction failures cannot render or rewrite prompts",
                "explanation metadata cannot alter generated output",
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
                "versioned metadata cannot activate routing behavior",
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
                "registry loading failures cannot apply routing side effects",
                "missing exports fail as import errors rather than runtime actions",
            ),
            "Registry loading failures are isolated to import and validation paths.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "quality_cost_optimizer",
                "cost_estimator",
                "budget_policies",
                "hitl_budget_gate",
                "quality_prediction_engine",
                "cost_prediction_engine",
            ),
            (
                "budget_enforcement_implemented_false",
                "cost_predictions_are_relative_bands",
            ),
            (
                "budget pressure cannot enforce spend or block execution",
                "cost prediction failures remain advisory estimates",
            ),
            "Budget and cost failures cannot enforce policy or route by price.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "runtime_recommendation_engine",
                "execution_policy_engine",
                "model_recommendation_engine",
                "routing_explainability",
            ),
            (
                "workflow_control_implemented_false",
                "records_are_frozen_models",
            ),
            (
                "runtime policy failures cannot advance workflow state",
                "explainability failures cannot modify workflow outputs",
            ),
            "Workflow state remains untouched after model-routing metadata failures.",
        ),
        _record(
            "provider_model_routing_preservation",
            _SOURCE_SURFACE_IDS,
            (
                "provider_model_routing_implemented_false",
                "configured_model_switching_implemented_false",
            ),
            (
                "audit metadata cannot change provider or model selection",
                "capability matrices describe options without choosing them",
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
    check_kind: ModelRoutingFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> ModelRoutingFailurePathAuditRecord:
    return ModelRoutingFailurePathAuditRecord(
        audit_id=f"model_routing_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
