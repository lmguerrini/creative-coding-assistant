"""V5.6 production release runtime failure path audit metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.production_architecture_consistency import (
    production_architecture_consistency_registry,
)

ProductionReleaseFailurePathCheckKind = Literal[
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
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
]
ProductionReleaseFailurePathAuditStatus = Literal["pass"]

PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "production_release_failure_path_audit_record.v1"
)
PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "production_release_failure_path_audit_registry.v1"
)
PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.6 production release runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for release "
    "readiness, packaging, deployment, demo, readiness review, architecture "
    "freeze, release audit, hardening, and architecture consistency surfaces "
    "only; it does not create runtime failure handlers, change terminal "
    "routing, execute providers or workflows, mutate provider/model routing, "
    "install dependencies or runtimes, build packages, deploy services, "
    "generate assets, execute retrieval, render previews, write storage, "
    "modify generated output, emit HITL requests, merge, push, tag, or apply "
    "Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "production_release_final_optimization",
    "production_release_packaging",
    "production_release_candidate",
    "production_demo_assets",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
    "production_architecture_freeze",
    "production_release_audit",
    "production_final_hardening",
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
_APPLICABLE_REQUIRED_CHECKS: tuple[ProductionReleaseFailurePathCheckKind, ...] = (
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
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
_NOT_APPLICABLE_REQUIRED_CHECKS = ("cache_failures",)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_failure_handler_creation",
    "terminal_failure_routing_mutation",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "dependency_installation",
    "runtime_installation",
    "package_build_execution",
    "deployment_execution",
    "asset_generation",
    "retrieval_execution",
    "preview_rendering_execution",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "hitl_request_emission",
    "merge_push_tag_operation",
    "runtime_evolution_application",
)


class ProductionReleaseFailurePathAuditRecord(BaseModel):
    """One V5.6 production release runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: ProductionReleaseFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=360)
    audit_status: ProductionReleaseFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    runtime_failure_handler_creation_implemented: Literal[False] = False
    terminal_failure_routing_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "production_release_failure_path_audit_record.v1"
    ] = PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V5.6 surfaces")
        return self


class ProductionReleaseFailurePathAuditRegistry(BaseModel):
    """V5.6 production release runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_release_failure_path_audit_registry"] = (
        "production_release_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "production_release_failure_path_audit_registry.v1"
    ] = PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=PRODUCTION_RELEASE_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    architecture_registry_serialization_version: str = Field(
        min_length=1,
        max_length=140,
    )
    architecture_registry_record_count: int = Field(ge=10, le=10)
    source_surface_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        ProductionReleaseFailurePathCheckKind,
        ...,
    ] = Field(min_length=18, max_length=18)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=1,
        max_length=1,
    )
    records: tuple[ProductionReleaseFailurePathAuditRecord, ...] = Field(
        min_length=18,
        max_length=18,
    )
    audit_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    check_kinds: tuple[ProductionReleaseFailurePathCheckKind, ...] = Field(
        min_length=18,
        max_length=18,
    )
    record_count: int = Field(ge=18, le=18)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    deployment_failure_boundary_preserved: Literal[True] = True
    release_operation_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    runtime_failure_handler_creation_implemented: Literal[False] = False
    terminal_failure_routing_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    package_build_executed: Literal[False] = False
    deployment_execution_implemented: Literal[False] = False
    asset_generation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    preview_rendering_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
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
            raise ValueError("source_surface_ids must match V5.6 surfaces")
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


def production_release_failure_path_audit_registry(
    project_root: str | Path | None = None,
) -> ProductionReleaseFailurePathAuditRegistry:
    """Return V5.6 runtime failure audit metadata without runtime actions."""

    architecture_registry = production_architecture_consistency_registry(project_root)
    records = _records()
    return ProductionReleaseFailurePathAuditRegistry(
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


def production_release_failure_path_audit_by_id(
    audit_id: str,
    registry: ProductionReleaseFailurePathAuditRegistry | None = None,
) -> ProductionReleaseFailurePathAuditRecord | None:
    """Return one V5.6 failure audit record by id."""

    source_registry = registry or production_release_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def production_release_failure_path_audits_for_check(
    check_kind: ProductionReleaseFailurePathCheckKind,
    registry: ProductionReleaseFailurePathAuditRegistry | None = None,
) -> tuple[ProductionReleaseFailurePathAuditRecord, ...]:
    """Return V5.6 failure audit records by checklist item."""

    source_registry = registry or production_release_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def production_release_failure_path_audits_for_surface(
    surface_id: str,
    registry: ProductionReleaseFailurePathAuditRegistry | None = None,
) -> tuple[ProductionReleaseFailurePathAuditRecord, ...]:
    """Return V5.6 failure audit records for one production release surface."""

    source_registry = registry or production_release_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _records() -> tuple[ProductionReleaseFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_architecture_registry",
                "metadata_only_rule_satisfied",
            ),
            (
                "surface failures cannot instantiate runtime nodes",
                "release metadata cannot execute providers, workflows, or output mutation",
            ),
            "Node-level failures stop at metadata construction and validation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "production_release_candidate",
                "production_readiness_review",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "terminal_failure_route_not_implemented",
                "workflow_execution_implemented_false",
            ),
            (
                "terminal failures cannot mutate workflow routing",
                "pending gates remain guarded metadata rather than terminal routes",
            ),
            "Terminal failure routing cannot change runtime graph or workflow state.",
        ),
        _record(
            "provider_failures",
            (
                "production_release_final_optimization",
                "production_release_candidate",
                "production_readiness_review",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "provider_execution_implemented_false",
                "missing_provider_credentials_guarded",
            ),
            (
                "provider failures cannot call or switch providers",
                "missing credentials remain deterministic guarded metadata",
            ),
            "Provider failures are represented as release guardrails only.",
        ),
        _record(
            "model_routing_failures",
            (
                "production_release_final_optimization",
                "production_release_candidate",
                "production_architecture_freeze",
                "production_release_audit",
            ),
            (
                "provider_model_routing_implemented_false",
                "automatic_model_download_blocked",
            ),
            (
                "model routing failures cannot change configured routing",
                "local model download and provider provisioning remain manual/HITL-only",
            ),
            "Model-routing failures cannot mutate routing or model selection.",
        ),
        _record(
            "stream_failures",
            (
                "production_deployment",
                "production_readiness_review",
                "production_release_audit",
            ),
            (
                "server_start_implemented_false",
                "workflow_execution_implemented_false",
            ),
            (
                "stream failures cannot start servers or execute workflows",
                "deployment metadata records stream paths without subscribing",
            ),
            "Stream failures remain deployment/readiness metadata only.",
        ),
        _record(
            "scheduling_failures",
            (
                "production_architecture_freeze",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "workflow_control_implemented_false",
                "merge_push_tag_implemented_false",
            ),
            (
                "scheduling failures cannot alter task order or release gates",
                "hardening cannot schedule deployment or release operations",
            ),
            "Scheduling failures cannot control workflow or release sequencing.",
        ),
        _record(
            "retry_failures",
            (
                "production_readiness_review",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "retry_triggering_not_present",
                "workflow_control_implemented_false",
            ),
            (
                "retry failures cannot trigger retries or refinement",
                "pending gates remain metadata instead of automatic retry loops",
            ),
            "Retry failures cannot start execution, refinement, or workflow control.",
        ),
        _record(
            "planning_helper_failures",
            _SOURCE_SURFACE_IDS,
            (
                "lookup_helpers_return_records_only",
                "validators_reject_mismatched_counts",
            ),
            (
                "planning helper failures return None, tuples, or validation errors",
                "helper failures cannot mutate runtime state",
            ),
            "Planning/helper failures stay inside pure metadata helpers.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "production_demo_assets",
                "production_creative_readiness_review",
                "production_final_hardening",
            ),
            (
                "prompt_mutation_implemented_false",
                "generated_output_mutation_implemented_false",
            ),
            (
                "demo prompt failures cannot rewrite prompts",
                "creative readiness cannot mutate generated output",
            ),
            "Prompt rendering failures are blocked because prompts are not rendered or mutated.",
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
                "versioned release metadata cannot activate runtime behavior",
            ),
            "Serialization failures stop at pydantic validation boundaries.",
        ),
        _record(
            "preview_workstation_frontend_backend_failures",
            (
                "production_demo_assets",
                "production_deployment",
                "production_creative_readiness_review",
            ),
            (
                "preview_rendering_execution_implemented_false",
                "deployment_execution_implemented_false",
            ),
            (
                "preview failures cannot render or overwrite media",
                "frontend/backend deployment failures cannot start services",
            ),
            "Preview, workstation, frontend, and backend failures remain inventory metadata.",
        ),
        _record(
            "registry_import_loading_failures",
            (
                "production_architecture_freeze",
                "production_release_audit",
                "production_final_hardening",
            ),
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
                "production_readiness_review",
                "production_release_audit",
                "production_final_hardening",
                "production_architecture_freeze",
            ),
            (
                "telemetry_emission_not_present",
                "persistent_storage_write_implemented_false",
            ),
            (
                "observability failures cannot emit telemetry or alerts",
                "audit and hardening evidence cannot write monitoring storage",
            ),
            "Telemetry and observability failures remain passive release evidence.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "production_release_final_optimization",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "budget_enforcement_not_present",
                "package_build_executed_false",
            ),
            (
                "cost or budget failures cannot enforce budgets",
                "release hardening cannot start builds or provider execution",
            ),
            "Budget and cost prediction failures cannot trigger enforcement or execution.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "production_readiness_review",
                "production_architecture_freeze",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "workflow_graph_mutation_implemented_false",
                "workflow_control_implemented_false",
            ),
            (
                "workflow state remains unchanged after metadata failures",
                "release audit and hardening cannot control workflow state",
            ),
            "Workflow state integrity is preserved because V5.6 adds no workflow control.",
        ),
        _record(
            "provider_model_routing_preservation",
            (
                "production_release_final_optimization",
                "production_release_candidate",
                "production_architecture_freeze",
            ),
            (
                "provider_model_routing_implemented_false",
                "provider_execution_implemented_false",
            ),
            (
                "provider/model routing is not applied by V5.6 metadata",
                "guarded provider assumptions cannot switch providers or models",
            ),
            "Provider/model routing is preserved under every production release failure.",
        ),
        _record(
            "generated_output_mutation_boundaries",
            (
                "production_demo_assets",
                "production_creative_readiness_review",
                "production_release_audit",
                "production_final_hardening",
            ),
            (
                "generated_output_mutation_implemented_false",
                "asset_generation_implemented_false",
            ),
            (
                "output failures cannot modify generated artifacts",
                "demo asset failures cannot generate or overwrite media",
            ),
            "Generated-output mutation is blocked for all production release failures.",
        ),
        _record(
            "passive_registry_activation_boundaries",
            _SOURCE_SURFACE_IDS,
            (
                "metadata_only_declared_for_all_surfaces",
                "runtime_evolution_not_applied",
            ),
            (
                "passive registry failures cannot activate runtime behavior",
                "Runtime Evolution remains gated outside this audit",
            ),
            "Passive registry activation is blocked by metadata-only declarations.",
        ),
    )


def _record(
    check_kind: ProductionReleaseFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> ProductionReleaseFailurePathAuditRecord:
    return ProductionReleaseFailurePathAuditRecord(
        audit_id=f"production_release_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
