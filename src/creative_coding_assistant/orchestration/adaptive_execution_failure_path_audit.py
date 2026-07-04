"""V5.5 runtime failure path audit metadata for adaptive execution."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adaptive_execution_architecture_consistency import (
    adaptive_execution_architecture_consistency_registry,
)

AdaptiveExecutionFailurePathCheckKind = Literal[
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
AdaptiveExecutionFailurePathAuditStatus = Literal["pass"]

ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "adaptive_execution_failure_path_audit_record.v1"
)
ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "adaptive_execution_failure_path_audit_registry.v1"
)
ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V5.5 adaptive execution runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage, adaptive policy "
    "and strategy failure boundaries, agent and resource allocation failure "
    "limits, confidence/risk/explainability failure limits, serialization and "
    "registry loading guards, provider/model routing preservation, "
    "generated-output mutation limits, and passive activation limits only; it "
    "allows only controlled adaptive policy allow/confirm/block decisions and "
    "does not change provider or model routing, switch providers or models, "
    "execute providers, invoke or activate agents, allocate resources, measure "
    "runtime resources, enforce budgets, emit HITL requests, control or "
    "execute workflows, mutate workflow graphs, compile graphs, trigger "
    "retries or refinements, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "adaptive_hybrid_workflow_optimizer",
    "adaptive_escalation_optimizer",
    "agent_activation_optimizer",
    "adaptive_cost_quality_optimizer",
    "adaptive_latency_optimizer",
    "adaptive_execution_strategy_selection",
    "adaptive_execution_policy_engine",
    "dynamic_agent_allocation",
    "dynamic_resource_allocation",
    "workflow_self_tuning_policies",
    "execution_confidence_engine",
    "workflow_risk_engine",
    "creative_exploration_optimizer",
    "emergence_optimizer",
    "agent_diversity_optimizer",
    "reflection_budget_optimizer",
    "adaptive_policy_explainability",
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
_APPLICABLE_REQUIRED_CHECKS: tuple[AdaptiveExecutionFailurePathCheckKind, ...] = (
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
    "adaptive_policy_application",
    "execution_policy_application",
    "strategy_application",
    "routing_application",
    "risk_decision_application",
    "confidence_application",
    "self_tuning_application",
    "emergence_behavior_application",
    "agent_diversity_behavior_application",
    "reflection_budget_enforcement",
    "provider_or_model_routing",
    "provider_execution",
    "provider_switching",
    "model_switching",
    "agent_invocation",
    "agent_activation",
    "agent_instantiation",
    "runtime_agent_allocation",
    "resource_allocation",
    "runtime_resource_measurement",
    "budget_enforcement",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "graph_compilation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class AdaptiveExecutionFailurePathAuditRecord(BaseModel):
    """One passive V5.5 adaptive execution runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: AdaptiveExecutionFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=17)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=320)
    audit_status: AdaptiveExecutionFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=36,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    policy_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    strategy_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    risk_decision_application_implemented: Literal[False] = False
    confidence_application_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    emergence_behavior_application_implemented: Literal[False] = False
    agent_diversity_behavior_application_implemented: Literal[False] = False
    reflection_budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    runtime_agent_allocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "adaptive_execution_failure_path_audit_record.v1"
    ] = ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V5.5 surfaces")
        return self


class AdaptiveExecutionFailurePathAuditRegistry(BaseModel):
    """Passive V5.5 adaptive execution runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_execution_failure_path_audit_registry"] = (
        "adaptive_execution_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "adaptive_execution_failure_path_audit_registry.v1"
    ] = ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ADAPTIVE_EXECUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    architecture_registry_serialization_version: str = Field(
        min_length=1,
        max_length=140,
    )
    architecture_registry_record_count: int = Field(ge=17, le=17)
    source_surface_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        AdaptiveExecutionFailurePathCheckKind,
        ...,
    ] = Field(min_length=17, max_length=17)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[AdaptiveExecutionFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[AdaptiveExecutionFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    adaptive_policy_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    policy_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    strategy_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    risk_decision_application_implemented: Literal[False] = False
    confidence_application_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    emergence_behavior_application_implemented: Literal[False] = False
    agent_diversity_behavior_application_implemented: Literal[False] = False
    reflection_budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    agent_activation_implemented: Literal[False] = False
    agent_instantiation_implemented: Literal[False] = False
    runtime_agent_allocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    runtime_resource_measurement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=36,
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
            raise ValueError("source_surface_ids must match V5.5 surfaces")
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


def adaptive_execution_failure_path_audit_registry() -> (
    AdaptiveExecutionFailurePathAuditRegistry
):
    """Return V5.5 runtime failure path audit metadata without runtime actions."""

    architecture_registry = adaptive_execution_architecture_consistency_registry()
    records = _records()
    return AdaptiveExecutionFailurePathAuditRegistry(
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


def adaptive_execution_failure_path_audit_by_id(
    audit_id: str,
    registry: AdaptiveExecutionFailurePathAuditRegistry | None = None,
) -> AdaptiveExecutionFailurePathAuditRecord | None:
    """Return one V5.5 failure audit record without activating failures."""

    source_registry = registry or adaptive_execution_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def adaptive_execution_failure_path_audits_for_check(
    check_kind: AdaptiveExecutionFailurePathCheckKind,
    registry: AdaptiveExecutionFailurePathAuditRegistry | None = None,
) -> tuple[AdaptiveExecutionFailurePathAuditRecord, ...]:
    """Return V5.5 failure audit records by checklist item."""

    source_registry = registry or adaptive_execution_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def adaptive_execution_failure_path_audits_for_surface(
    surface_id: str,
    registry: AdaptiveExecutionFailurePathAuditRegistry | None = None,
) -> tuple[AdaptiveExecutionFailurePathAuditRecord, ...]:
    """Return V5.5 failure audit records for one adaptive execution surface."""

    source_registry = registry or adaptive_execution_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _records() -> tuple[AdaptiveExecutionFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_architecture_registry",
                "each_surface_declares_advisory_or_controlled_policy_boundary",
            ),
            (
                "surface-level failures cannot activate provider or workflow runtime nodes",
                "controlled policy failures stop at allow/confirm/block decisions",
            ),
            "Surface failures cannot execute providers, workflows, or output mutation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "adaptive_hybrid_workflow_optimizer",
                "adaptive_escalation_optimizer",
                "adaptive_execution_policy_engine",
                "workflow_self_tuning_policies",
                "workflow_risk_engine",
                "adaptive_policy_explainability",
            ),
            (
                "terminal_failure_route_not_implemented",
                "workflow_execution_implemented_false",
            ),
            (
                "terminal failures cannot execute workflows",
                "controlled policy decisions can only return blocked or HITL states",
            ),
            "Terminal failure handling cannot route or execute workflows.",
        ),
        _record(
            "provider_failures",
            (
                "adaptive_hybrid_workflow_optimizer",
                "adaptive_cost_quality_optimizer",
                "adaptive_latency_optimizer",
                "adaptive_execution_strategy_selection",
                "adaptive_execution_policy_engine",
                "execution_confidence_engine",
                "adaptive_policy_explainability",
            ),
            (
                "provider_model_routing_implemented_false",
                "provider_execution_implemented_false",
            ),
            (
                "provider failures cannot call, switch, or fail over providers",
                "controlled policy can mark execution blocked or fallback-only",
            ),
            "Provider failures are represented as blocked policy decisions.",
        ),
        _record(
            "model_routing_failures",
            (
                "adaptive_hybrid_workflow_optimizer",
                "adaptive_cost_quality_optimizer",
                "adaptive_latency_optimizer",
                "adaptive_execution_strategy_selection",
                "adaptive_execution_policy_engine",
                "execution_confidence_engine",
                "adaptive_policy_explainability",
            ),
            (
                "automatic_model_switching_implemented_false",
                "routing_application_implemented_false",
            ),
            (
                "model-routing failures cannot switch the selected model",
                "controlled policy path selection cannot mutate configured routing",
            ),
            "Model-routing failures stop at controlled policy decisions.",
        ),
        _record(
            "stream_failures",
            (
                "adaptive_latency_optimizer",
                "dynamic_resource_allocation",
                "workflow_self_tuning_policies",
                "execution_confidence_engine",
            ),
            (
                "workflow_execution_implemented_false",
                "runtime_resource_measurement_implemented_false",
            ),
            (
                "stream failures cannot adjust workflow timing or resources",
                "confidence and latency metadata cannot start async execution",
            ),
            "Stream failures remain adaptive metadata without runtime execution.",
        ),
        _record(
            "scheduling_failures",
            (
                "agent_activation_optimizer",
                "dynamic_agent_allocation",
                "dynamic_resource_allocation",
                "workflow_self_tuning_policies",
                "agent_diversity_optimizer",
            ),
            (
                "runtime_agent_allocation_implemented_false",
                "workflow_control_implemented_false",
            ),
            (
                "scheduling failures cannot create agent work or change order",
                "allocation metadata cannot modify workflow state or concurrency",
            ),
            "Scheduling failures cannot alter runtime scheduling or workflow state.",
        ),
        _record(
            "retry_failures",
            (
                "workflow_self_tuning_policies",
                "workflow_risk_engine",
                "creative_exploration_optimizer",
                "emergence_optimizer",
                "reflection_budget_optimizer",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "retry pressure cannot trigger refinement or replay execution",
                "risk and exploration failures cannot start another attempt",
            ),
            "Retry failures are passive and cannot start execution or refinement.",
        ),
        _record(
            "planning_helper_failures",
            _SOURCE_SURFACE_IDS,
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
                "creative_exploration_optimizer",
                "emergence_optimizer",
                "reflection_budget_optimizer",
                "adaptive_policy_explainability",
            ),
            (
                "prompt_mutation_implemented_false",
                "generated_output_mutation_implemented_false",
            ),
            (
                "adaptive metadata failures cannot render or rewrite prompts",
                "exploration and explanation metadata cannot alter generated output",
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
                "versioned metadata cannot activate adaptive behavior",
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
                "adaptive_cost_quality_optimizer",
                "adaptive_latency_optimizer",
                "dynamic_resource_allocation",
                "execution_confidence_engine",
                "workflow_risk_engine",
                "adaptive_policy_explainability",
            ),
            (
                "confidence_application_implemented_false",
                "risk_decision_application_implemented_false",
            ),
            (
                "observability failures cannot apply confidence or risk decisions",
                "adaptive telemetry pressure remains metadata-only evidence",
            ),
            "Telemetry and observability failures remain read-only metadata.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "adaptive_hybrid_workflow_optimizer",
                "adaptive_execution_policy_engine",
                "adaptive_cost_quality_optimizer",
                "dynamic_resource_allocation",
                "reflection_budget_optimizer",
                "adaptive_execution_strategy_selection",
            ),
            (
                "budget_enforcement_implemented_false",
                "reflection_budget_enforcement_implemented_false",
            ),
            (
                "budget pressure cannot enforce spend",
                "cost prediction failures can only require confirmation or block policy allowance",
            ),
            "Budget and cost failures cannot route or execute providers.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "adaptive_hybrid_workflow_optimizer",
                "adaptive_escalation_optimizer",
                "adaptive_execution_policy_engine",
                "workflow_self_tuning_policies",
                "workflow_risk_engine",
                "adaptive_policy_explainability",
            ),
            (
                "workflow_control_implemented_false",
                "workflow_graph_mutation_implemented_false",
            ),
            (
                "adaptive failures cannot advance workflow state",
                "controlled policy failures cannot modify graph order or outputs",
            ),
            "Workflow state remains untouched after adaptive metadata failures.",
        ),
        _record(
            "provider_model_routing_preservation",
            _SOURCE_SURFACE_IDS,
            (
                "provider_model_routing_implemented_false",
                "automatic_provider_switching_implemented_false",
            ),
            (
                "audit metadata cannot change provider or model selection",
                "controlled policy path selection cannot mutate configured routes",
            ),
            "Provider/model routing preservation is enforced by false routing flags.",
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
                "advisory_or_controlled_policy_boundary_declared",
                "runtime_evolution_implemented_false",
            ),
            (
                "registry reads cannot activate provider or workflow execution",
                "Runtime Evolution cannot be applied without a human gate",
            ),
            "Registry activation is limited to metadata and controlled policy construction.",
        ),
    )


def _record(
    check_kind: AdaptiveExecutionFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> AdaptiveExecutionFailurePathAuditRecord:
    return AdaptiveExecutionFailurePathAuditRecord(
        audit_id=f"adaptive_execution_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
