"""V6.5 self-evolution runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.self_evolution_governance import (
    SelfEvolutionGovernancePlan,
    build_self_evolution_governance,
)

SelfEvolutionFailurePathCheckKind = Literal[
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
SelfEvolutionFailurePathAuditStatus = Literal["pass"]

SELF_EVOLUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "self_evolution_failure_path_audit_record.v1"
)
SELF_EVOLUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "self_evolution_failure_path_audit_registry.v1"
)
SELF_EVOLUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.5 Self Evolution runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for the 22 "
    "explicit roadmap plans plus core surface, secondary surface, and "
    "governance/safety metadata only. It audits failure-path coverage for "
    "missing HITL, missing explainability, missing rollback, ownership "
    "boundary violations, cross-capability governance gaps, downstream impact "
    "ambiguity, automation attempts, provider execution attempts, storage "
    "writes, report artifact generation, prompt/workflow/routing/memory/"
    "retrieval mutation, and generated-output mutation; it does not enforce "
    "audits, classify live errors, route terminal failures, handle or repair "
    "failures, apply proposals, apply governance, emit HITL requests, request "
    "human input, execute rollback, generate report artifacts, write storage, "
    "change provider/model routing, execute providers, probe runtimes, invoke "
    "agents, execute or control workflows, mutate workflow graphs, trigger "
    "retries or refinements, render or rewrite prompts, mutate memory or "
    "retrieval, mutate generated output, or apply Runtime Evolution."
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
APPLICABLE_FAILURE_PATH_CHECKS: tuple[SelfEvolutionFailurePathCheckKind, ...] = (
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
FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS = (
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
    "proposal_application",
    "rollback_execution",
    "report_artifact_generation",
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


class SelfEvolutionFailurePathAuditRecord(BaseModel):
    """One passive V6.5 self-evolution runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: SelfEvolutionFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=25, max_length=25)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=22, max_length=22)
    proposal_count: int = Field(ge=110, le=110)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=10)
    failure_response_boundary: str = Field(min_length=1, max_length=520)
    audit_status: SelfEvolutionFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=33,
        max_length=33,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    self_evolution_orchestration_layer_verified: Literal[True] = True
    v6_signal_sources_integrated: Literal[True] = True
    proposal_traceability_verified: Literal[True] = True
    governance_boundary_verified: Literal[True] = True
    metadata_only_rule_satisfied: Literal[True] = True
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
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
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
    applied_audit_fix_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=25)
    handled_failure_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=25)
    routed_terminal_failure_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=25,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=25,
    )
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    provider_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    serialization_version: Literal["self_evolution_failure_path_audit_record.v1"] = (
        SELF_EVOLUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_contract(self) -> Self:
        if self.audit_id != f"self_evolution_failure_path_audit::{self.check_kind}":
            raise ValueError("audit_id must match check_kind")
        if self.covered_roadmap_items != CORE_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.5 roadmap")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match audit boundary")
        empty_fields = (
            self.applied_audit_fix_ids,
            self.handled_failure_ids,
            self.routed_terminal_failure_ids,
            self.applied_evolution_proposal_ids,
            self.emitted_hitl_request_ids,
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("failure path audit mutation ids must be empty")
        return self


class SelfEvolutionFailurePathAuditRegistry(BaseModel):
    """Passive V6.5 self-evolution runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["self_evolution_failure_path_audit_registry"] = (
        "self_evolution_failure_path_audit_registry"
    )
    serialization_version: Literal["self_evolution_failure_path_audit_registry.v1"] = (
        SELF_EVOLUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SELF_EVOLUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=3200,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=25, max_length=25)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=25,
        max_length=25,
    )
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[SelfEvolutionFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    not_applicable_required_checks: tuple[str, ...] = Field(min_length=2, max_length=2)
    check_kinds: tuple[SelfEvolutionFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    records: tuple[SelfEvolutionFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    record_count: int = Field(ge=17, le=17)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=22, max_length=22)
    covered_roadmap_item_count: int = Field(ge=22, le=22)
    proposal_count: int = Field(ge=110, le=110)
    governance_boundary_count: int = Field(ge=22, le=22)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=33,
        max_length=33,
    )
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_proposals_traceable: Literal[True] = True
    upstream_signal_sources_traceable: Literal[True] = True
    governance_safety_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
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
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
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
    applied_audit_fix_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=25)
    handled_failure_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=25)
    routed_terminal_failure_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=25,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=25,
    )
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    provider_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        if self.required_checks != REQUIRED_FAILURE_PATH_CHECKS:
            raise ValueError("required_checks must match checklist")
        if self.applicable_required_checks != APPLICABLE_FAILURE_PATH_CHECKS:
            raise ValueError("applicable_required_checks must match checklist")
        if self.not_applicable_required_checks != NOT_APPLICABLE_FAILURE_PATH_CHECKS:
            raise ValueError("not_applicable_required_checks must match checklist")
        if self.check_kinds != tuple(record.check_kind for record in self.records):
            raise ValueError("check_kinds must match records")
        if self.record_ids != tuple(record.audit_id for record in self.records):
            raise ValueError("record_ids must match records")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.covered_roadmap_items != CORE_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.5 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match audit boundary")
        if any(
            record.source_surface_ids != self.source_surface_ids
            for record in self.records
        ):
            raise ValueError("record source_surface_ids must match registry")
        if any(not record.metadata_only for record in self.records):
            raise ValueError("all failure path records must be metadata only")
        empty_fields = (
            self.applied_audit_fix_ids,
            self.handled_failure_ids,
            self.routed_terminal_failure_ids,
            self.applied_evolution_proposal_ids,
            self.emitted_hitl_request_ids,
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("failure path registry mutation ids must be empty")
        return self


def self_evolution_failure_path_audit_registry(
    governance_plan: SelfEvolutionGovernancePlan | None = None,
) -> SelfEvolutionFailurePathAuditRegistry:
    """Build passive V6.5 failure path audit metadata."""

    plan = governance_plan or build_self_evolution_governance()
    source_surface_ids = _source_surface_ids(plan)
    source_versions = _source_serialization_versions(plan)
    records = tuple(
        _build_failure_path_record(
            check_kind=check_kind,
            source_surface_ids=source_surface_ids,
            plan=plan,
        )
        for check_kind in APPLICABLE_FAILURE_PATH_CHECKS
    )
    return SelfEvolutionFailurePathAuditRegistry(
        source_surface_ids=source_surface_ids,
        source_serialization_versions=source_versions,
        required_checks=REQUIRED_FAILURE_PATH_CHECKS,
        applicable_required_checks=APPLICABLE_FAILURE_PATH_CHECKS,
        not_applicable_required_checks=NOT_APPLICABLE_FAILURE_PATH_CHECKS,
        check_kinds=tuple(record.check_kind for record in records),
        records=records,
        record_ids=tuple(record.audit_id for record in records),
        record_count=len(records),
        covered_roadmap_items=plan.covered_roadmap_items,
        covered_roadmap_item_count=plan.covered_roadmap_item_count,
        proposal_count=plan.proposal_count,
        governance_boundary_count=plan.governance_boundary_count,
        upstream_capabilities=plan.upstream_capabilities,
        cross_cutting_contracts=plan.cross_cutting_contracts,
        blocked_runtime_behaviors=FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
    )


def self_evolution_failure_path_audit_by_id(
    audit_id: str,
    registry: SelfEvolutionFailurePathAuditRegistry | None = None,
) -> SelfEvolutionFailurePathAuditRecord | None:
    """Return one failure path audit record by id."""

    source_registry = registry or self_evolution_failure_path_audit_registry()
    for record in source_registry.records:
        if record.audit_id == audit_id:
            return record
    return None


def self_evolution_failure_path_audits_for_check(
    check_kind: SelfEvolutionFailurePathCheckKind,
    registry: SelfEvolutionFailurePathAuditRegistry | None = None,
) -> tuple[SelfEvolutionFailurePathAuditRecord, ...]:
    """Return failure path audit records for one checklist item."""

    source_registry = registry or self_evolution_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def self_evolution_failure_path_audits_for_surface(
    source_surface_id: str,
    registry: SelfEvolutionFailurePathAuditRegistry | None = None,
) -> tuple[SelfEvolutionFailurePathAuditRecord, ...]:
    """Return failure path audit records that cover one source surface."""

    source_registry = registry or self_evolution_failure_path_audit_registry()
    return tuple(
        record
        for record in source_registry.records
        if source_surface_id in record.source_surface_ids
    )


def _build_failure_path_record(
    *,
    check_kind: SelfEvolutionFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    plan: SelfEvolutionGovernancePlan,
) -> SelfEvolutionFailurePathAuditRecord:
    return SelfEvolutionFailurePathAuditRecord(
        audit_id=f"self_evolution_failure_path_audit::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        covered_roadmap_items=plan.covered_roadmap_items,
        proposal_count=plan.proposal_count,
        upstream_capabilities=plan.upstream_capabilities,
        cross_cutting_contracts=plan.cross_cutting_contracts,
        evidence=(
            f"check_kind:{check_kind}",
            f"roadmap_item_count:{plan.covered_roadmap_item_count}",
            f"proposal_count:{plan.proposal_count}",
            f"governance_boundary_count:{plan.governance_boundary_count}",
            "metadata_only:true",
        ),
        invariant_assertions=(
            "all_applicable_failure_path_checks_are_metadata_only",
            "all_22_v6_5_roadmap_items_remain_traceable",
            "all_110_proposals_remain_traceable",
            "v6_1_through_v6_4_signal_sources_remain_traceable",
            "governance_boundaries_are_not_enforced",
            "runtime_evolution_is_not_applied",
        ),
        failure_response_boundary=(
            f"{check_kind} is audited for V6.5 metadata coverage only; the "
            "audit cannot observe live failures, classify errors, route "
            "terminal failures, repair failures, apply proposals, mutate "
            "runtime state, or write storage."
        ),
        blocked_runtime_behaviors=FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
    )


def _source_surface_ids(plan: SelfEvolutionGovernancePlan) -> tuple[str, ...]:
    return (
        *(boundary.plan_role for boundary in plan.governance_boundaries),
        "self_evolution_core_surface",
        "self_evolution_secondary_surface",
        plan.role,
    )


def _source_serialization_versions(
    plan: SelfEvolutionGovernancePlan,
) -> tuple[str, ...]:
    return (
        *(f"{boundary.plan_role}_plan.v1" for boundary in plan.governance_boundaries),
        "self_evolution_core_surface.v1",
        "self_evolution_secondary_surface.v1",
        plan.serialization_version,
    )
