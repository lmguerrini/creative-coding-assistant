"""V6.2 creative memory runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifact_history import build_artifact_history
from .creative_dna import build_creative_dna
from .creative_lineage import build_creative_lineage
from .creative_memory_core_surface import build_creative_memory_core_surface
from .creative_memory_governance import build_creative_memory_governance
from .creative_memory_secondary_surface import build_creative_memory_secondary_surface
from .creative_ontology import build_creative_ontology
from .long_term_creative_memory import build_long_term_creative_memory
from .personalization_engine import build_personalization_engine
from .project_memory import build_project_memory
from .session_memory_evolution import build_session_memory_evolution
from .style_profiles import build_style_profiles
from .user_preferences import build_user_preferences

CreativeMemoryFailurePathCheckKind = Literal[
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
CreativeMemoryFailurePathAuditStatus = Literal["pass"]

CREATIVE_MEMORY_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "creative_memory_failure_path_audit_record.v1"
)
CREATIVE_MEMORY_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "creative_memory_failure_path_audit_registry.v1"
)
CREATIVE_MEMORY_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.2 creative memory runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for creative "
    "memory, preference, style, project, Creative DNA, personalization, "
    "session memory, artifact history, creative lineage, ontology, core "
    "surface, secondary surface, and governance/safety metadata only; it does "
    "not write memory, retrieve memory, consolidate memory, learn preferences, "
    "apply user models, apply personalization, persist artifact history, infer "
    "lineage or ontology, enforce governance or safety policies, emit HITL "
    "requests, activate automation, classify live errors, route terminal "
    "failures, handle or repair failures, change provider/model routing, "
    "execute providers, probe runtimes, install dependencies, execute or "
    "control workflows, mutate workflow graphs, trigger retries or refinements, "
    "mutate prompts, write storage, mutate generated output, or apply Runtime "
    "Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "long_term_creative_memory",
    "user_preferences",
    "style_profiles",
    "project_memory",
    "creative_dna",
    "personalization_engine",
    "session_memory_evolution",
    "artifact_history",
    "creative_lineage",
    "creative_ontology",
    "creative_memory_core_surface",
    "creative_memory_secondary_surface",
    "creative_memory_governance_safety",
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
    CreativeMemoryFailurePathCheckKind,
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
    "memory_storage_write",
    "memory_retrieval_execution",
    "memory_consolidation_execution",
    "preference_learning_execution",
    "user_model_application",
    "personalization_application",
    "creative_dna_application",
    "artifact_history_persistence",
    "creative_lineage_inference",
    "ontology_relationship_inference",
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "automation_activation",
    "hitl_request_emission",
    "human_input_request",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "runtime_probe",
    "local_model_download",
    "agent_invocation",
    "resource_allocation",
    "telemetry_collection",
    "live_memory_observation",
    "live_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling_or_repair",
    "generated_output_evaluation",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "retry_or_refinement_triggering",
    "replay_execution",
    "model_training",
    "runtime_mutation",
    "automatic_preference_mutation",
    "automatic_remediation",
    "artifact_execution",
    "dependency_installation",
    "graph_compilation",
    "prompt_rendering_or_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CreativeMemoryFailurePathAuditRecord(BaseModel):
    """One passive V6.2 creative memory runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: CreativeMemoryFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=13)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=360)
    audit_status: CreativeMemoryFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_execution_implemented: Literal[False] = False
    preference_learning_execution_implemented: Literal[False] = False
    user_model_application_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    artifact_history_persistence_implemented: Literal[False] = False
    creative_lineage_inference_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_memory_observation_implemented: Literal[False] = False
    live_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    replay_execution_implemented: Literal[False] = False
    model_training_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_memory_failure_path_audit_record.v1"] = (
        CREATIVE_MEMORY_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
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
            raise ValueError("source_surface_ids must reference V6.2 surfaces")
        if self.audit_id != f"creative_memory_failure::{self.check_kind}":
            raise ValueError("audit_id must match check_kind")
        return self


class CreativeMemoryFailurePathAuditRegistry(BaseModel):
    """Passive V6.2 creative memory runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_memory_failure_path_audit_registry"] = (
        "creative_memory_failure_path_audit_registry"
    )
    serialization_version: Literal["creative_memory_failure_path_audit_registry.v1"] = (
        CREATIVE_MEMORY_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_MEMORY_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=13, max_length=13)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=13,
        max_length=13,
    )
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        CreativeMemoryFailurePathCheckKind,
        ...,
    ] = Field(min_length=17, max_length=17)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[CreativeMemoryFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[CreativeMemoryFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    memory_storage_boundary_preserved: Literal[True] = True
    preference_learning_boundary_preserved: Literal[True] = True
    governance_safety_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    memory_storage_write_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_consolidation_execution_implemented: Literal[False] = False
    preference_learning_execution_implemented: Literal[False] = False
    user_model_application_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    artifact_history_persistence_implemented: Literal[False] = False
    creative_lineage_inference_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_memory_observation_implemented: Literal[False] = False
    live_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    failure_repair_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    replay_execution_implemented: Literal[False] = False
    model_training_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
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
            raise ValueError("source_surface_ids must match V6.2 surfaces")
        if self.source_serialization_versions != _source_serialization_versions():
            raise ValueError("source_serialization_versions must match sources")
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


def creative_memory_failure_path_audit_registry() -> (
    CreativeMemoryFailurePathAuditRegistry
):
    """Return V6.2 runtime failure path audit metadata without runtime actions."""

    records = _records()
    return CreativeMemoryFailurePathAuditRegistry(
        source_surface_ids=_SOURCE_SURFACE_IDS,
        source_serialization_versions=_source_serialization_versions(),
        required_checks=_REQUIRED_CHECKS,
        applicable_required_checks=_APPLICABLE_REQUIRED_CHECKS,
        not_applicable_required_checks=_NOT_APPLICABLE_REQUIRED_CHECKS,
        records=records,
        audit_ids=tuple(record.audit_id for record in records),
        check_kinds=tuple(record.check_kind for record in records),
        record_count=len(records),
    )


def creative_memory_failure_path_audit_by_id(
    audit_id: str,
    registry: CreativeMemoryFailurePathAuditRegistry | None = None,
) -> CreativeMemoryFailurePathAuditRecord | None:
    """Return one V6.2 failure audit record without activating failures."""

    source_registry = registry or creative_memory_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def creative_memory_failure_path_audits_for_check(
    check_kind: CreativeMemoryFailurePathCheckKind,
    registry: CreativeMemoryFailurePathAuditRegistry | None = None,
) -> tuple[CreativeMemoryFailurePathAuditRecord, ...]:
    """Return V6.2 failure audit records by checklist item."""

    source_registry = registry or creative_memory_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def creative_memory_failure_path_audits_for_surface(
    surface_id: str,
    registry: CreativeMemoryFailurePathAuditRegistry | None = None,
) -> tuple[CreativeMemoryFailurePathAuditRecord, ...]:
    """Return V6.2 failure audit records for one creative memory surface."""

    source_registry = registry or creative_memory_failure_path_audit_registry()
    normalized_surface_id = str(surface_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_surface_id in record.source_surface_ids
    )


def _source_serialization_versions() -> tuple[str, ...]:
    return tuple(
        plan.serialization_version
        for plan in (
            build_long_term_creative_memory(),
            build_user_preferences(),
            build_style_profiles(),
            build_project_memory(),
            build_creative_dna(),
            build_personalization_engine(),
            build_session_memory_evolution(),
            build_artifact_history(),
            build_creative_lineage(),
            build_creative_ontology(),
            build_creative_memory_core_surface(),
            build_creative_memory_secondary_surface(),
            build_creative_memory_governance(),
        )
    )


def _records() -> tuple[CreativeMemoryFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            ("source_surface_order_matches_v6_2_memory_registry", "metadata_only"),
            (
                "creative memory surfaces cannot instantiate runtime nodes",
                "memory metadata cannot execute providers or workflows",
            ),
            "Node-level failures stop at metadata construction and validation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "creative_memory_governance_safety",
                "creative_memory_secondary_surface",
                "artifact_history",
            ),
            (
                "terminal_failure_routing_implemented_false",
                "failure_handling_implemented_false",
            ),
            (
                "creative memory metadata cannot route terminal failures",
                "governance metadata cannot request or execute recovery",
            ),
            "Terminal failure routing remains outside V6.2 creative memory metadata.",
        ),
        _record(
            "provider_failures",
            (
                "personalization_engine",
                "creative_memory_core_surface",
                "creative_memory_governance_safety",
            ),
            (
                "provider_execution_implemented_false",
                "provider_model_routing_implemented_false",
            ),
            (
                "provider failures cannot call providers",
                "personalization metadata cannot switch providers or models",
            ),
            "Provider failures are represented only as passive guardrails.",
        ),
        _record(
            "model_routing_failures",
            (
                "creative_memory_secondary_surface",
                "creative_memory_governance_safety",
                "creative_ontology",
            ),
            (
                "provider_model_routing_implemented_false",
                "local_model_download_implemented_false",
            ),
            (
                "model-routing failures cannot alter configured routing",
                "ontology and governance metadata cannot download or select models",
            ),
            "Model-routing failures cannot mutate routing or model selection.",
        ),
        _record(
            "stream_failures",
            (
                "session_memory_evolution",
                "artifact_history",
                "creative_lineage",
            ),
            (
                "workflow_execution_implemented_false",
                "telemetry_collection_implemented_false",
            ),
            (
                "stream failures cannot subscribe to runtime output",
                "session and artifact metadata cannot observe live streams",
            ),
            "Stream failures remain passive memory metadata.",
        ),
        _record(
            "scheduling_failures",
            (
                "creative_memory_core_surface",
                "creative_memory_secondary_surface",
                "creative_memory_governance_safety",
            ),
            (
                "workflow_control_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "scheduling failures cannot reorder workflow execution",
                "creative memory metadata cannot allocate resources",
            ),
            "Scheduling failures cannot control workflow order or capacity.",
        ),
        _record(
            "retry_failures",
            (
                "project_memory",
                "session_memory_evolution",
                "creative_memory_governance_safety",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "memory signals cannot trigger retries",
                "governance metadata cannot start refinement loops",
            ),
            "Retry failures cannot start execution, retry, or refinement.",
        ),
        _record(
            "planning_helper_failures",
            _SOURCE_SURFACE_IDS,
            (
                "planning_helper_failures_are_metadata_only",
                "workflow_control_implemented_false",
            ),
            (
                "planning helper failures cannot mutate workflow state",
                "creative memory helpers cannot execute or repair workflows",
            ),
            "Planning helper failures remain validation-time metadata concerns.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "long_term_creative_memory",
                "user_preferences",
                "style_profiles",
                "creative_memory_core_surface",
                "creative_memory_secondary_surface",
            ),
            (
                "prompt_rendering_implemented_false",
                "prompt_mutation_implemented_false",
            ),
            (
                "memory metadata cannot render prompts",
                "preference and style metadata cannot mutate prompts",
            ),
            "Prompt rendering failures cannot create prompt mutation behavior.",
        ),
        _record(
            "serialization_failures",
            _SOURCE_SURFACE_IDS,
            ("pydantic_model_validation_only", "registry_imports_are_lazy"),
            (
                "serialization failures cannot write storage",
                "failed serialization cannot trigger Runtime Evolution",
            ),
            "Serialization failures stop at typed metadata validation.",
        ),
        _record(
            "registry_import_loading_failures",
            _SOURCE_SURFACE_IDS,
            (
                "lazy_exports_only",
                "passive_registry_activation_boundary_preserved",
            ),
            (
                "registry loading failures cannot activate passive registries",
                "import failures cannot execute providers or workflows",
            ),
            "Registry import failures remain import-time failures.",
        ),
        _record(
            "telemetry_observability_failures",
            (
                "session_memory_evolution",
                "artifact_history",
                "creative_lineage",
                "creative_memory_governance_safety",
            ),
            (
                "telemetry_collection_implemented_false",
                "live_memory_observation_implemented_false",
            ),
            (
                "observability failures cannot collect telemetry",
                "memory metadata cannot observe live outcomes",
            ),
            "Telemetry failures cannot introduce observation side effects.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "personalization_engine",
                "creative_memory_secondary_surface",
                "creative_memory_governance_safety",
            ),
            (
                "provider_model_routing_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "budget failures cannot enforce budgets",
                "creative memory metadata cannot allocate or route resources",
            ),
            "Budget and cost failures remain advisory metadata boundaries.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "session_memory_evolution",
                "artifact_history",
                "creative_lineage",
                "creative_memory_core_surface",
                "creative_memory_governance_safety",
            ),
            (
                "workflow_state_integrity_boundary_preserved",
                "workflow_graph_mutation_implemented_false",
            ),
            (
                "failure metadata cannot mutate workflow state",
                "lineage and artifact metadata cannot reconstruct workflow state",
            ),
            "Workflow state integrity is preserved by no mutation behavior.",
        ),
        _record(
            "provider_model_routing_preservation",
            (
                "creative_dna",
                "personalization_engine",
                "creative_memory_secondary_surface",
                "creative_memory_governance_safety",
            ),
            (
                "provider_model_routing_preserved",
                "provider_model_routing_implemented_false",
            ),
            (
                "Creative DNA cannot apply model routing preferences",
                "personalization metadata cannot select providers",
            ),
            "Provider and model routing remains owned by existing routing paths.",
        ),
        _record(
            "generated_output_mutation_boundaries",
            _SOURCE_SURFACE_IDS,
            (
                "generated_output_mutation_boundary_preserved",
                "generated_output_mutation_implemented_false",
            ),
            (
                "creative memory metadata cannot modify generated output",
                "failure audit metadata cannot repair or rewrite output",
            ),
            "Generated output mutation remains outside V6.2 creative memory.",
        ),
        _record(
            "passive_registry_activation_boundaries",
            _SOURCE_SURFACE_IDS,
            (
                "passive_registry_activation_boundary_preserved",
                "runtime_evolution_implemented_false",
            ),
            (
                "failure audits cannot activate passive registries",
                "Runtime Evolution remains blocked without HITL",
            ),
            "Passive registries remain inspectable metadata only.",
        ),
    )


def _record(
    check_kind: CreativeMemoryFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> CreativeMemoryFailurePathAuditRecord:
    return CreativeMemoryFailurePathAuditRecord(
        audit_id=f"creative_memory_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
