"""V6.6 Cognitive OS runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_ROADMAP_ITEMS,
    CognitiveOSCapability,
)
from creative_coding_assistant.orchestration.cognitive_os_governance_safety import (
    COGNITIVE_OS_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION,
    COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES,
    COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
    CognitiveOSGovernanceSafetyPlan,
    build_cognitive_os_governance_safety,
)
from creative_coding_assistant.orchestration.cognitive_os_secondary_surface import (
    COGNITIVE_OS_FOUNDATION_SYSTEMS,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

CognitiveOSFailurePathCheckKind = Literal[
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
CognitiveOSFailurePathAuditStatus = Literal["pass"]

COGNITIVE_OS_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "cognitive_os_failure_path_audit_record.v1"
)
COGNITIVE_OS_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "cognitive_os_failure_path_audit_registry.v1"
)
COGNITIVE_OS_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive OS runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for the core "
    "surface, secondary surface, governance/safety boundaries, all six "
    "Cognitive OS capability surfaces, and all 24 V6.6 roadmap items. It "
    "audits missing HITL, missing explainability, governance gaps, safety "
    "boundary violations, dependency ambiguity, no-automation violations, "
    "provider/model routing attempts, storage writes, report generation, "
    "prompt/workflow/routing/memory/retrieval mutation, generated-output "
    "mutation, and Runtime Evolution attempts as metadata only; it does not "
    "enforce audits, observe live failures, classify live errors, route "
    "terminal failures, handle or repair failures, execute workflows, apply "
    "routing, emit HITL requests, request human input, enforce governance or "
    "safety policies, activate automation, activate surfaces, write storage, "
    "execute providers, probe runtimes, invoke agents, trigger retries or "
    "refinements, mutate outputs or runtime state, or apply Runtime Evolution."
)

CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
REQUIRED_FAILURE_PATH_CHECKS = (
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
APPLICABLE_FAILURE_PATH_CHECKS: tuple[CognitiveOSFailurePathCheckKind, ...] = (
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
NOT_APPLICABLE_FAILURE_PATH_CHECKS = (
    "preview_workstation_frontend_backend_failures",
    "cache_failures",
)
COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS = (
    "audit_enforcement",
    "live_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling_or_repair",
    "automatic_remediation",
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "hitl_request_emission",
    "human_input_request",
    "hitl_decision_application",
    "automation_activation",
    "core_surface_activation",
    "secondary_surface_activation",
    "execution_graph_application",
    "routing_application",
    "storage_write",
    "prompt_rendering_or_mutation",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "routing_mutation",
    "provider_or_model_routing_mutation",
    "memory_mutation",
    "retrieval_mutation",
    "provider_execution",
    "agent_invocation",
    "telemetry_collection",
    "runtime_probe",
    "dependency_installation",
    "retry_or_refinement_triggering",
    "generated_output_evaluation",
    "generated_output_mutation",
    "runtime_evolution_application",
)


class CognitiveOSFailurePathAuditRecord(BaseModel):
    """One passive V6.6 Cognitive OS runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: CognitiveOSFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=21, max_length=21)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=21,
        max_length=21,
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(
        min_length=6,
        max_length=6,
    )
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    governance_boundary_count: int = Field(ge=6, le=6)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    evidence: tuple[str, ...] = Field(min_length=5, max_length=10)
    invariant_assertions: tuple[str, ...] = Field(min_length=5, max_length=10)
    failure_response_boundary: str = Field(min_length=1, max_length=640)
    audit_status: CognitiveOSFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=34, max_length=34)
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    cognitive_os_layer_verified: Literal[True] = True
    governance_boundary_verified: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_capability_surfaces_traceable: Literal[True] = True
    foundation_traceability_verified: Literal[True] = True
    metadata_only_rule_satisfied: Literal[True] = True
    no_automation_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    audit_enforcement_implemented: Literal[False] = False
    live_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    core_surface_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    applied_audit_fix_ids: tuple[str, ...] = Field(default_factory=tuple)
    handled_failure_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_terminal_failure_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    generated_report_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    serialization_version: Literal["cognitive_os_failure_path_audit_record.v1"] = (
        COGNITIVE_OS_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        expected_id = f"cognitive_os_failure_path_audit::{self.check_kind}"
        if self.audit_id != expected_id:
            raise ValueError("audit_id must match check_kind")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source versions must align with source surfaces")
        if self.covered_roadmap_items != COGNITIVE_OS_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.6 roadmap")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.governance_boundary_count != 6:
            raise ValueError("governance_boundary_count must be six")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if (
            self.blocked_runtime_behaviors
            != COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS
        ):
            raise ValueError("blocked_runtime_behaviors must match audit boundary")
        if any(
            (
                self.applied_audit_fix_ids,
                self.handled_failure_ids,
                self.routed_terminal_failure_ids,
                self.emitted_hitl_request_ids,
                self.activated_core_surface_ids,
                self.activated_secondary_surface_ids,
                self.generated_report_artifact_ids,
                self.written_storage_record_ids,
                self.provider_execution_ids,
                self.mutated_output_ids,
            )
        ):
            raise ValueError("failure path audit mutation ids must be empty")
        return self


class CognitiveOSFailurePathAuditRegistry(BaseModel):
    """Passive V6.6 Cognitive OS runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_os_failure_path_audit_registry"] = (
        "cognitive_os_failure_path_audit_registry"
    )
    serialization_version: Literal["cognitive_os_failure_path_audit_registry.v1"] = (
        COGNITIVE_OS_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_OS_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=3600,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        CHECKLIST_SOURCE
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    source_surface_ids: tuple[str, ...] = Field(min_length=21, max_length=21)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=21,
        max_length=21,
    )
    source_surface_id_count: int = Field(ge=21, le=21)
    source_surface_roles: tuple[str, ...] = Field(min_length=2, max_length=2)
    source_surface_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[CognitiveOSFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    not_applicable_required_checks: tuple[str, ...] = Field(min_length=2, max_length=2)
    check_kinds: tuple[CognitiveOSFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    records: tuple[CognitiveOSFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    record_count: int = Field(ge=17, le=17)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    covered_roadmap_item_count: int = Field(ge=24, le=24)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(
        min_length=6,
        max_length=6,
    )
    capability_count: int = Field(ge=6, le=6)
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    foundation_system_count: int = Field(ge=7, le=7)
    governance_boundary_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    governance_boundary_count: int = Field(ge=6, le=6)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=34, max_length=34)
    all_applicable_checks_covered: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_capability_surfaces_traceable: Literal[True] = True
    foundation_traceability_verified: Literal[True] = True
    governance_safety_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    audit_enforcement_implemented: Literal[False] = False
    live_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    core_surface_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    applied_audit_fix_ids: tuple[str, ...] = Field(default_factory=tuple)
    handled_failure_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_terminal_failure_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    generated_report_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records_and_contract(self) -> Self:
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source versions must align with source surfaces")
        if self.source_surface_id_count != len(self.source_surface_ids):
            raise ValueError("source_surface_id_count must match sources")
        if self.source_surface_roles != COGNITIVE_OS_GOVERNANCE_SOURCE_ROLES:
            raise ValueError("source_surface_roles must match V6.6 surfaces")
        if (
            self.source_surface_serialization_versions
            != COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS
        ):
            raise ValueError("source_surface_serialization_versions must match")
        if self.required_checks != REQUIRED_FAILURE_PATH_CHECKS:
            raise ValueError("required_checks must match runtime checklist")
        if self.applicable_required_checks != APPLICABLE_FAILURE_PATH_CHECKS:
            raise ValueError("applicable_required_checks must match audit contract")
        if self.not_applicable_required_checks != NOT_APPLICABLE_FAILURE_PATH_CHECKS:
            raise ValueError("not_applicable_required_checks must match audit contract")
        if self.check_kinds != tuple(record.check_kind for record in self.records):
            raise ValueError("check_kinds must match records")
        if self.record_ids != tuple(record.audit_id for record in self.records):
            raise ValueError("record_ids must match records")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.covered_roadmap_items != COGNITIVE_OS_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.6 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.foundation_system_count != len(self.foundation_systems):
            raise ValueError("foundation_system_count must match foundations")
        if self.governance_boundary_count != len(self.governance_boundary_ids):
            raise ValueError("governance_boundary_count must match boundaries")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if (
            self.blocked_runtime_behaviors
            != COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS
        ):
            raise ValueError("blocked_runtime_behaviors must match audit boundary")
        for record in self.records:
            if record.source_surface_ids != self.source_surface_ids:
                raise ValueError("record source_surface_ids must match registry")
            if record.source_serialization_versions != (
                self.source_serialization_versions
            ):
                raise ValueError("record source versions must match registry")
            if record.covered_roadmap_items != self.covered_roadmap_items:
                raise ValueError("record roadmap items must match registry")
            if record.capability_ids != self.capability_ids:
                raise ValueError("record capability ids must match registry")
        if any(
            (
                self.applied_audit_fix_ids,
                self.handled_failure_ids,
                self.routed_terminal_failure_ids,
                self.emitted_hitl_request_ids,
                self.activated_core_surface_ids,
                self.activated_secondary_surface_ids,
                self.generated_report_artifact_ids,
                self.written_storage_record_ids,
                self.provider_execution_ids,
                self.mutated_output_ids,
            )
        ):
            raise ValueError("failure path audit mutation ids must be empty")
        if not all(record.metadata_only for record in self.records):
            raise ValueError("all failure path audit records must be metadata only")
        return self


def cognitive_os_failure_path_audit_registry(
    governance_plan: CognitiveOSGovernanceSafetyPlan | None = None,
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CognitiveOSFailurePathAuditRegistry:
    """Build passive Cognitive OS runtime failure-path audit metadata."""

    plan = governance_plan or build_cognitive_os_governance_safety(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    source_ids = _source_surface_ids(plan)
    source_versions = _source_serialization_versions(plan)
    records = tuple(
        _audit_record(
            check_kind=check_kind,
            plan=plan,
            source_surface_ids=source_ids,
            source_serialization_versions=source_versions,
        )
        for check_kind in APPLICABLE_FAILURE_PATH_CHECKS
    )
    return CognitiveOSFailurePathAuditRegistry(
        route_name=plan.route_name,
        task_type=plan.task_type,
        execution_mode_ids=plan.execution_mode_ids,
        source_surface_ids=source_ids,
        source_serialization_versions=source_versions,
        source_surface_id_count=len(source_ids),
        source_surface_roles=plan.source_surface_roles,
        source_surface_serialization_versions=(
            plan.source_surface_serialization_versions
        ),
        required_checks=REQUIRED_FAILURE_PATH_CHECKS,
        applicable_required_checks=APPLICABLE_FAILURE_PATH_CHECKS,
        not_applicable_required_checks=NOT_APPLICABLE_FAILURE_PATH_CHECKS,
        check_kinds=tuple(record.check_kind for record in records),
        records=records,
        record_ids=tuple(record.audit_id for record in records),
        record_count=len(records),
        covered_roadmap_items=plan.governed_roadmap_items,
        covered_roadmap_item_count=plan.governed_roadmap_item_count,
        capability_ids=plan.capability_ids,
        capabilities=plan.capabilities,
        capability_count=plan.capability_count,
        foundation_systems=plan.foundation_systems,
        foundation_system_count=plan.foundation_system_count,
        governance_boundary_ids=plan.governance_boundary_ids,
        governance_boundary_count=plan.governance_boundary_count,
        cross_cutting_contracts=plan.cross_cutting_contracts,
        blocked_runtime_behaviors=COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
    )


def cognitive_os_failure_path_audit_by_id(
    audit_id: str,
    registry: CognitiveOSFailurePathAuditRegistry | None = None,
) -> CognitiveOSFailurePathAuditRecord | None:
    """Return one failure-path audit record without applying it."""

    source_registry = registry or cognitive_os_failure_path_audit_registry()
    for record in source_registry.records:
        if record.audit_id == audit_id:
            return record
    return None


def cognitive_os_failure_path_audits_for_check(
    check_kind: CognitiveOSFailurePathCheckKind,
    registry: CognitiveOSFailurePathAuditRegistry | None = None,
) -> tuple[CognitiveOSFailurePathAuditRecord, ...]:
    """Return failure-path audit records for one checklist kind."""

    source_registry = registry or cognitive_os_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def cognitive_os_failure_path_audits_for_surface(
    source_surface_id: str,
    registry: CognitiveOSFailurePathAuditRegistry | None = None,
) -> tuple[CognitiveOSFailurePathAuditRecord, ...]:
    """Return failure-path audit records that cite one source surface."""

    source_registry = registry or cognitive_os_failure_path_audit_registry()
    return tuple(
        record
        for record in source_registry.records
        if source_surface_id in record.source_surface_ids
    )


def _audit_record(
    *,
    check_kind: CognitiveOSFailurePathCheckKind,
    plan: CognitiveOSGovernanceSafetyPlan,
    source_surface_ids: tuple[str, ...],
    source_serialization_versions: tuple[str, ...],
) -> CognitiveOSFailurePathAuditRecord:
    return CognitiveOSFailurePathAuditRecord(
        audit_id=f"cognitive_os_failure_path_audit::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        source_serialization_versions=source_serialization_versions,
        covered_roadmap_items=plan.governed_roadmap_items,
        capability_ids=plan.capability_ids,
        capabilities=plan.capabilities,
        foundation_systems=plan.foundation_systems,
        governance_boundary_count=plan.governance_boundary_count,
        cross_cutting_contracts=plan.cross_cutting_contracts,
        evidence=(
            "cognitive_os_core_surface",
            "cognitive_os_secondary_surface",
            "cognitive_os_governance_safety",
            f"source_surface_id_count:{len(source_surface_ids)}",
            f"governance_boundary_count:{plan.governance_boundary_count}",
        ),
        invariant_assertions=(
            "metadata_only_rule_satisfied",
            "active_behavior_rule_satisfied",
            "all_roadmap_items_traceable",
            "provider_model_routing_preserved",
            "generated_output_mutation_boundary_preserved",
        ),
        failure_response_boundary=(
            f"{check_kind} audit records metadata coverage only; it does not "
            "observe, classify, route, repair, retry, execute, activate, "
            "enforce, request HITL, mutate runtime state, or apply Runtime "
            "Evolution."
        ),
        blocked_runtime_behaviors=(COGNITIVE_OS_FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS),
    )


def _source_surface_ids(
    plan: CognitiveOSGovernanceSafetyPlan,
) -> tuple[str, ...]:
    return (
        "cognitive_os_core_surface",
        "cognitive_os_secondary_surface",
        "cognitive_os_governance_safety",
        *plan.source_core_surface_ids,
        *plan.source_secondary_surface_ids,
        *plan.governance_boundary_ids,
    )


def _source_serialization_versions(
    plan: CognitiveOSGovernanceSafetyPlan,
) -> tuple[str, ...]:
    return (
        *COGNITIVE_OS_GOVERNANCE_SOURCE_SERIALIZATION_VERSIONS,
        plan.serialization_version,
        *(("cognitive_os_core_surface.v1",) * len(plan.source_core_surface_ids)),
        *(
            ("cognitive_os_secondary_surface.v1",)
            * len(plan.source_secondary_surface_ids)
        ),
        *(
            (COGNITIVE_OS_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION,)
            * len(plan.governance_boundary_ids)
        ),
    )
