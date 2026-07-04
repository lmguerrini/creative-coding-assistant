"""V6.3 knowledge evolution runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.automatic_kb_updates import build_automatic_kb_updates
from creative_coding_assistant.orchestration.documentation_intelligence import build_documentation_intelligence
from creative_coding_assistant.orchestration.embedding_refresh import build_embedding_refresh
from creative_coding_assistant.orchestration.knowledge_conflict_resolver import build_knowledge_conflict_resolver
from creative_coding_assistant.orchestration.knowledge_consolidation import build_knowledge_consolidation
from creative_coding_assistant.orchestration.knowledge_drift_detection import build_knowledge_drift_detection
from creative_coding_assistant.orchestration.knowledge_evolution_core_surface import (
    build_knowledge_evolution_core_surface,
)
from creative_coding_assistant.orchestration.knowledge_evolution_governance import build_knowledge_evolution_governance
from creative_coding_assistant.orchestration.knowledge_evolution_secondary_surface import (
    build_knowledge_evolution_secondary_surface,
)
from creative_coding_assistant.orchestration.knowledge_freshness_tracking import build_knowledge_freshness_tracking
from creative_coding_assistant.orchestration.knowledge_gap_detection import build_knowledge_gap_detection
from creative_coding_assistant.orchestration.knowledge_health_monitoring import build_knowledge_health_monitoring
from creative_coding_assistant.orchestration.knowledge_lifecycle_management import build_knowledge_lifecycle_management
from creative_coding_assistant.orchestration.knowledge_provenance_evolution import build_knowledge_provenance_evolution
from creative_coding_assistant.orchestration.knowledge_quality_scoring import build_knowledge_quality_scoring
from creative_coding_assistant.orchestration.knowledge_rollback import build_knowledge_rollback
from creative_coding_assistant.orchestration.knowledge_snapshot_engine import build_knowledge_snapshot_engine
from creative_coding_assistant.orchestration.knowledge_trust_score import build_knowledge_trust_score
from creative_coding_assistant.orchestration.knowledge_versioning import build_knowledge_versioning
from creative_coding_assistant.orchestration.ranking_optimization import build_ranking_optimization
from creative_coding_assistant.orchestration.retrieval_evolution import build_retrieval_evolution
from creative_coding_assistant.orchestration.source_reliability_engine import build_source_reliability_engine

KnowledgeEvolutionFailurePathCheckKind = Literal[
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
KnowledgeEvolutionFailurePathAuditStatus = Literal["pass"]

KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "knowledge_evolution_failure_path_audit_record.v1"
)
KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "knowledge_evolution_failure_path_audit_registry.v1"
)
KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.3 knowledge evolution runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for the 19 "
    "explicit knowledge evolution roadmap surfaces plus core surface, "
    "secondary surface, and governance/safety metadata only; it does not "
    "execute automatic KB updates, fetch documentation, refresh embeddings, "
    "execute retrieval, mutate ranking, run health monitoring, compute quality "
    "or trust scores, detect gaps, resolve conflicts, detect drift, score "
    "source reliability, consolidate knowledge, manage lifecycle state, mutate "
    "provenance graphs, mutate version graphs, execute snapshots, execute "
    "rollback, run freshness scans, enforce governance or safety policies, "
    "emit HITL requests, activate automation, classify live errors, route "
    "terminal failures, handle or repair failures, change provider/model "
    "routing, execute providers, probe runtimes, install dependencies, execute "
    "or control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, write storage, mutate generated output, or "
    "apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "automatic_kb_updates",
    "documentation_intelligence",
    "embedding_refresh",
    "retrieval_evolution",
    "ranking_optimization",
    "knowledge_health_monitoring",
    "knowledge_quality_scoring",
    "knowledge_gap_detection",
    "knowledge_conflict_resolver",
    "knowledge_drift_detection",
    "source_reliability_engine",
    "knowledge_consolidation",
    "knowledge_lifecycle_management",
    "knowledge_provenance_evolution",
    "knowledge_versioning",
    "knowledge_snapshot_engine",
    "knowledge_rollback",
    "knowledge_freshness_tracking",
    "knowledge_trust_score",
    "knowledge_evolution_core_surface",
    "knowledge_evolution_secondary_surface",
    "knowledge_evolution_governance_safety",
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
    KnowledgeEvolutionFailurePathCheckKind,
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
    "automatic_kb_update_execution",
    "documentation_fetch_execution",
    "embedding_refresh_execution",
    "retrieval_execution",
    "ranking_mutation",
    "knowledge_health_monitoring_execution",
    "quality_score_computation",
    "knowledge_gap_detection_execution",
    "knowledge_conflict_resolution_execution",
    "knowledge_drift_detection_execution",
    "source_reliability_scoring_execution",
    "knowledge_consolidation_execution",
    "knowledge_lifecycle_management_execution",
    "provenance_graph_mutation",
    "version_graph_mutation",
    "knowledge_snapshot_engine_execution",
    "knowledge_rollback_execution",
    "freshness_scan_execution",
    "trust_score_computation",
    "kb_storage_write",
    "source_record_update",
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
    "live_knowledge_observation",
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
    "automatic_remediation",
    "dependency_installation",
    "graph_compilation",
    "prompt_rendering_or_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeEvolutionFailurePathAuditRecord(BaseModel):
    """One passive V6.3 knowledge evolution runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: KnowledgeEvolutionFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=22)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=420)
    audit_status: KnowledgeEvolutionFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=56,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    automatic_kb_update_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    knowledge_conflict_resolution_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
    live_knowledge_observation_implemented: Literal[False] = False
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
    automatic_remediation_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "knowledge_evolution_failure_path_audit_record.v1"
    ] = KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_sources_are_known(self) -> Self:
        unknown = tuple(
            source_id
            for source_id in self.source_surface_ids
            if source_id not in _SOURCE_SURFACE_IDS
        )
        if unknown:
            raise ValueError("source_surface_ids must reference V6.3 surfaces")
        if self.audit_id != f"knowledge_evolution_failure::{self.check_kind}":
            raise ValueError("audit_id must match check_kind")
        return self


class KnowledgeEvolutionFailurePathAuditRegistry(BaseModel):
    """Passive V6.3 knowledge evolution runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_evolution_failure_path_audit_registry"] = (
        "knowledge_evolution_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "knowledge_evolution_failure_path_audit_registry.v1"
    ] = KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=KNOWLEDGE_EVOLUTION_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=3200,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=22, max_length=22)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=22,
        max_length=22,
    )
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        KnowledgeEvolutionFailurePathCheckKind,
        ...,
    ] = Field(min_length=17, max_length=17)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[KnowledgeEvolutionFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[KnowledgeEvolutionFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    all_roadmap_surfaces_traceable: Literal[True] = True
    automatic_update_boundary_preserved: Literal[True] = True
    retrieval_boundary_preserved: Literal[True] = True
    knowledge_storage_boundary_preserved: Literal[True] = True
    governance_safety_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    automatic_kb_update_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    knowledge_conflict_resolution_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
    live_knowledge_observation_implemented: Literal[False] = False
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
    automatic_remediation_implemented: Literal[False] = False
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
        max_length=56,
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
            raise ValueError("source_surface_ids must match V6.3 surfaces")
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


def knowledge_evolution_failure_path_audit_registry() -> (
    KnowledgeEvolutionFailurePathAuditRegistry
):
    """Return V6.3 runtime failure path audit metadata without runtime actions."""

    records = _records()
    return KnowledgeEvolutionFailurePathAuditRegistry(
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


def knowledge_evolution_failure_path_audit_by_id(
    audit_id: str,
    registry: KnowledgeEvolutionFailurePathAuditRegistry | None = None,
) -> KnowledgeEvolutionFailurePathAuditRecord | None:
    """Return one V6.3 failure audit record without activating failures."""

    source_registry = registry or knowledge_evolution_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def knowledge_evolution_failure_path_audits_for_check(
    check_kind: KnowledgeEvolutionFailurePathCheckKind,
    registry: KnowledgeEvolutionFailurePathAuditRegistry | None = None,
) -> tuple[KnowledgeEvolutionFailurePathAuditRecord, ...]:
    """Return V6.3 failure audit records by checklist item."""

    source_registry = registry or knowledge_evolution_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def knowledge_evolution_failure_path_audits_for_surface(
    surface_id: str,
    registry: KnowledgeEvolutionFailurePathAuditRegistry | None = None,
) -> tuple[KnowledgeEvolutionFailurePathAuditRecord, ...]:
    """Return V6.3 failure audit records for one knowledge evolution surface."""

    source_registry = registry or knowledge_evolution_failure_path_audit_registry()
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
            build_automatic_kb_updates(),
            build_documentation_intelligence(),
            build_embedding_refresh(),
            build_retrieval_evolution(),
            build_ranking_optimization(),
            build_knowledge_health_monitoring(),
            build_knowledge_quality_scoring(),
            build_knowledge_gap_detection(),
            build_knowledge_conflict_resolver(),
            build_knowledge_drift_detection(),
            build_source_reliability_engine(),
            build_knowledge_consolidation(),
            build_knowledge_lifecycle_management(),
            build_knowledge_provenance_evolution(),
            build_knowledge_versioning(),
            build_knowledge_snapshot_engine(),
            build_knowledge_rollback(),
            build_knowledge_freshness_tracking(),
            build_knowledge_trust_score(),
            build_knowledge_evolution_core_surface(),
            build_knowledge_evolution_secondary_surface(),
            build_knowledge_evolution_governance(),
        )
    )


def _records() -> tuple[KnowledgeEvolutionFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            ("source_surface_order_matches_v6_3_knowledge_registry", "metadata_only"),
            (
                "knowledge evolution surfaces cannot instantiate runtime nodes",
                "knowledge metadata cannot execute providers or workflows",
            ),
            "Node-level failures stop at metadata construction and validation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "knowledge_evolution_governance_safety",
                "knowledge_evolution_secondary_surface",
                "knowledge_rollback",
            ),
            (
                "terminal_failure_routing_implemented_false",
                "failure_handling_implemented_false",
            ),
            (
                "knowledge metadata cannot route terminal failures",
                "governance metadata cannot request or execute recovery",
            ),
            "Terminal failure routing remains outside V6.3 knowledge metadata.",
        ),
        _record(
            "provider_failures",
            (
                "automatic_kb_updates",
                "knowledge_evolution_core_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "provider_execution_implemented_false",
                "provider_model_routing_implemented_false",
            ),
            (
                "provider failures cannot call providers",
                "knowledge update metadata cannot infer provider credentials",
            ),
            "Provider failures are represented only as passive guardrails.",
        ),
        _record(
            "model_routing_failures",
            (
                "source_reliability_engine",
                "knowledge_evolution_secondary_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "provider_model_routing_implemented_false",
                "local_model_download_implemented_false",
            ),
            (
                "model-routing failures cannot alter configured routing",
                "source reliability metadata cannot download or select models",
            ),
            "Model-routing failures cannot mutate routing or model selection.",
        ),
        _record(
            "stream_failures",
            (
                "documentation_intelligence",
                "knowledge_drift_detection",
                "knowledge_freshness_tracking",
            ),
            (
                "workflow_execution_implemented_false",
                "telemetry_collection_implemented_false",
            ),
            (
                "stream failures cannot subscribe to runtime output",
                "knowledge freshness metadata cannot observe live streams",
            ),
            "Stream failures remain passive knowledge metadata.",
        ),
        _record(
            "scheduling_failures",
            (
                "knowledge_evolution_core_surface",
                "knowledge_evolution_secondary_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "workflow_control_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "scheduling failures cannot reorder workflow execution",
                "knowledge evolution metadata cannot allocate resources",
            ),
            "Scheduling failures cannot control workflow order or capacity.",
        ),
        _record(
            "retry_failures",
            (
                "knowledge_snapshot_engine",
                "knowledge_rollback",
                "knowledge_evolution_governance_safety",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "knowledge recovery signals cannot trigger retries",
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
                "knowledge evolution helpers cannot execute or repair workflows",
            ),
            "Planning helper failures remain validation-time metadata concerns.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "automatic_kb_updates",
                "documentation_intelligence",
                "retrieval_evolution",
                "knowledge_evolution_core_surface",
                "knowledge_evolution_secondary_surface",
            ),
            (
                "prompt_rendering_implemented_false",
                "prompt_mutation_implemented_false",
            ),
            (
                "knowledge metadata cannot render prompts",
                "retrieval metadata cannot mutate prompts or generated output",
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
                "knowledge_health_monitoring",
                "knowledge_freshness_tracking",
                "knowledge_evolution_governance_safety",
            ),
            (
                "telemetry_collection_implemented_false",
                "live_knowledge_observation_implemented_false",
            ),
            (
                "observability failures cannot collect telemetry",
                "knowledge metadata cannot observe live outcomes",
            ),
            "Telemetry failures cannot introduce observation side effects.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "ranking_optimization",
                "knowledge_evolution_secondary_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "provider_model_routing_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "budget failures cannot enforce budgets",
                "knowledge metadata cannot allocate or route resources",
            ),
            "Budget and cost failures remain advisory metadata boundaries.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "knowledge_snapshot_engine",
                "knowledge_rollback",
                "knowledge_evolution_core_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "workflow_state_integrity_boundary_preserved",
                "workflow_graph_mutation_implemented_false",
            ),
            (
                "failure metadata cannot mutate workflow state",
                "snapshot and rollback metadata cannot restore runtime state",
            ),
            "Workflow state integrity is preserved by no mutation behavior.",
        ),
        _record(
            "provider_model_routing_preservation",
            (
                "source_reliability_engine",
                "knowledge_trust_score",
                "knowledge_evolution_secondary_surface",
                "knowledge_evolution_governance_safety",
            ),
            (
                "provider_model_routing_preserved",
                "provider_model_routing_implemented_false",
            ),
            (
                "source reliability cannot apply provider preferences",
                "trust metadata cannot select providers or models",
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
                "knowledge metadata cannot modify generated output",
                "failure audit metadata cannot repair or rewrite output",
            ),
            "Generated output mutation remains outside V6.3 knowledge evolution.",
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
    check_kind: KnowledgeEvolutionFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> KnowledgeEvolutionFailurePathAuditRecord:
    return KnowledgeEvolutionFailurePathAuditRecord(
        audit_id=f"knowledge_evolution_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
