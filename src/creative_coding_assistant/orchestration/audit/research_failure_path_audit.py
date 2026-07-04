"""V6.4 research runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.automatic_kb_enrichment import build_automatic_kb_enrichment
from creative_coding_assistant.orchestration.contradiction_detection import build_contradiction_detection
from creative_coding_assistant.orchestration.creative_research_engine import build_creative_research_engine
from creative_coding_assistant.orchestration.cross_domain_inspiration_discovery import (
    build_cross_domain_inspiration_discovery,
)
from creative_coding_assistant.orchestration.cross_source_comparison import build_cross_source_comparison
from creative_coding_assistant.orchestration.knowledge_distillation import build_knowledge_distillation
from creative_coding_assistant.orchestration.paper_research import build_paper_research
from creative_coding_assistant.orchestration.research_confidence_engine import build_research_confidence_engine
from creative_coding_assistant.orchestration.research_core_surface import build_research_core_surface
from creative_coding_assistant.orchestration.research_decomposer import build_research_decomposer
from creative_coding_assistant.orchestration.research_execution_policy import build_research_execution_policy
from creative_coding_assistant.orchestration.research_gap_discovery import build_research_gap_discovery
from creative_coding_assistant.orchestration.research_governance import build_research_governance
from creative_coding_assistant.orchestration.research_hitl_policies import build_research_hitl_policies
from creative_coding_assistant.orchestration.research_memory import build_research_memory
from creative_coding_assistant.orchestration.research_planner import build_research_planner
from creative_coding_assistant.orchestration.research_recommendation_engine import build_research_recommendation_engine
from creative_coding_assistant.orchestration.research_reports import build_research_reports
from creative_coding_assistant.orchestration.research_secondary_surface import build_research_secondary_surface
from creative_coding_assistant.orchestration.source_credibility_engine import build_source_credibility_engine
from creative_coding_assistant.orchestration.source_validation_engine import build_source_validation_engine
from creative_coding_assistant.orchestration.web_research import build_web_research

ResearchFailurePathCheckKind = Literal[
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
ResearchFailurePathAuditStatus = Literal["pass"]

RESEARCH_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "research_failure_path_audit_record.v1"
)
RESEARCH_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "research_failure_path_audit_registry.v1"
)
RESEARCH_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.4 research runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for the 19 "
    "explicit research roadmap surfaces plus core surface, secondary surface, "
    "and governance/safety metadata only; it does not execute research, mutate "
    "research plans, create research tasks, execute paper or web research, "
    "fetch external sources, browse the web, download papers, run "
    "cross-source comparison, execute knowledge distillation, enrich the KB, "
    "write KB storage, generate research reports, write research memory, "
    "execute source validation, score source credibility, execute "
    "contradiction detection, score research confidence, discover research "
    "gaps, generate recommendations, apply research execution policy, emit "
    "HITL requests, apply HITL decisions, generate creative output, execute "
    "inspiration discovery, perform live cross-domain search, enforce "
    "governance or safety policies, activate automation, classify live "
    "errors, route terminal failures, handle or repair failures, change "
    "provider/model routing, execute providers, probe runtimes, install "
    "dependencies, execute or control workflows, mutate workflow graphs, "
    "trigger retries or refinements, mutate prompts, write storage, mutate "
    "generated output, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "research_planner",
    "research_decomposer",
    "paper_research",
    "web_research",
    "cross_source_comparison",
    "knowledge_distillation",
    "automatic_kb_enrichment",
    "research_reports",
    "research_memory",
    "source_validation_engine",
    "source_credibility_engine",
    "contradiction_detection",
    "research_confidence_engine",
    "research_gap_discovery",
    "research_recommendation_engine",
    "research_execution_policy",
    "research_hitl_policies",
    "creative_research_engine",
    "cross_domain_inspiration_discovery",
    "research_core_surface",
    "research_secondary_surface",
    "research_governance_safety",
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
_APPLICABLE_REQUIRED_CHECKS: tuple[ResearchFailurePathCheckKind, ...] = (
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
    "research_execution",
    "research_plan_mutation",
    "research_task_creation",
    "paper_research_execution",
    "web_research_execution",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "cross_source_comparison_execution",
    "knowledge_distillation_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "research_report_generation",
    "research_memory_write",
    "source_validation_execution",
    "source_credibility_scoring_execution",
    "contradiction_detection_execution",
    "research_confidence_scoring_execution",
    "research_gap_discovery_execution",
    "research_recommendation_generation",
    "research_execution_policy_application",
    "hitl_request_emission",
    "hitl_decision_application",
    "creative_output_generation",
    "inspiration_discovery_execution",
    "live_cross_domain_search",
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "automation_activation",
    "human_input_request",
    "routing_application",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "runtime_probe",
    "local_model_download",
    "agent_invocation",
    "resource_allocation",
    "telemetry_collection",
    "live_research_observation",
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


class ResearchFailurePathAuditRecord(BaseModel):
    """One passive V6.4 research runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: ResearchFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=22)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=420)
    audit_status: ResearchFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=64,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    cross_source_comparison_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    research_memory_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    creative_output_generation_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_research_observation_implemented: Literal[False] = False
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
    serialization_version: Literal["research_failure_path_audit_record.v1"] = (
        RESEARCH_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
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
            raise ValueError("source_surface_ids must reference V6.4 surfaces")
        if self.audit_id != f"research_failure::{self.check_kind}":
            raise ValueError("audit_id must match check_kind")
        return self


class ResearchFailurePathAuditRegistry(BaseModel):
    """Passive V6.4 research runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_failure_path_audit_registry"] = (
        "research_failure_path_audit_registry"
    )
    serialization_version: Literal["research_failure_path_audit_registry.v1"] = (
        RESEARCH_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=3600,
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
    applicable_required_checks: tuple[ResearchFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[ResearchFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[ResearchFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    all_roadmap_surfaces_traceable: Literal[True] = True
    external_source_boundary_preserved: Literal[True] = True
    research_execution_boundary_preserved: Literal[True] = True
    knowledge_storage_boundary_preserved: Literal[True] = True
    governance_safety_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    cross_source_comparison_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    research_memory_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    creative_output_generation_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_research_observation_implemented: Literal[False] = False
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
        max_length=64,
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
            raise ValueError("source_surface_ids must match V6.4 surfaces")
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


def research_failure_path_audit_registry() -> ResearchFailurePathAuditRegistry:
    """Return V6.4 runtime failure path audit metadata without runtime actions."""

    records = _records()
    return ResearchFailurePathAuditRegistry(
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


def research_failure_path_audit_by_id(
    audit_id: str,
    registry: ResearchFailurePathAuditRegistry | None = None,
) -> ResearchFailurePathAuditRecord | None:
    """Return one V6.4 failure audit record without activating failures."""

    source_registry = registry or research_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def research_failure_path_audits_for_check(
    check_kind: ResearchFailurePathCheckKind,
    registry: ResearchFailurePathAuditRegistry | None = None,
) -> tuple[ResearchFailurePathAuditRecord, ...]:
    """Return V6.4 failure audit records by checklist item."""

    source_registry = registry or research_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def research_failure_path_audits_for_surface(
    surface_id: str,
    registry: ResearchFailurePathAuditRegistry | None = None,
) -> tuple[ResearchFailurePathAuditRecord, ...]:
    """Return V6.4 failure audit records for one research surface."""

    source_registry = registry or research_failure_path_audit_registry()
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
            build_research_planner(),
            build_research_decomposer(),
            build_paper_research(),
            build_web_research(),
            build_cross_source_comparison(),
            build_knowledge_distillation(),
            build_automatic_kb_enrichment(),
            build_research_reports(),
            build_research_memory(),
            build_source_validation_engine(),
            build_source_credibility_engine(),
            build_contradiction_detection(),
            build_research_confidence_engine(),
            build_research_gap_discovery(),
            build_research_recommendation_engine(),
            build_research_execution_policy(),
            build_research_hitl_policies(),
            build_creative_research_engine(),
            build_cross_domain_inspiration_discovery(),
            build_research_core_surface(),
            build_research_secondary_surface(),
            build_research_governance(),
        )
    )


def _records() -> tuple[ResearchFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            ("source_surface_order_matches_v6_4_research_registry", "metadata_only"),
            (
                "research surfaces cannot instantiate runtime nodes",
                "research metadata cannot execute providers or workflows",
            ),
            "Node-level failures stop at metadata construction and validation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "research_governance_safety",
                "research_secondary_surface",
                "research_execution_policy",
            ),
            (
                "terminal_failure_routing_implemented_false",
                "failure_handling_implemented_false",
            ),
            (
                "research metadata cannot route terminal failures",
                "governance metadata cannot request or execute recovery",
            ),
            "Terminal failure routing remains outside V6.4 research metadata.",
        ),
        _record(
            "provider_failures",
            (
                "paper_research",
                "web_research",
                "research_core_surface",
                "research_governance_safety",
            ),
            (
                "provider_execution_implemented_false",
                "provider_model_routing_implemented_false",
            ),
            (
                "provider failures cannot call providers",
                "research metadata cannot infer provider credentials",
            ),
            "Provider failures are represented only as passive guardrails.",
        ),
        _record(
            "model_routing_failures",
            (
                "source_credibility_engine",
                "research_secondary_surface",
                "research_governance_safety",
            ),
            (
                "provider_model_routing_implemented_false",
                "local_model_download_implemented_false",
            ),
            (
                "model-routing failures cannot alter configured routing",
                "source credibility metadata cannot download or select models",
            ),
            "Model-routing failures cannot mutate routing or model selection.",
        ),
        _record(
            "stream_failures",
            (
                "web_research",
                "research_reports",
                "research_memory",
            ),
            (
                "workflow_execution_implemented_false",
                "telemetry_collection_implemented_false",
            ),
            (
                "stream failures cannot subscribe to runtime output",
                "research memory metadata cannot observe live streams",
            ),
            "Stream failures remain passive research metadata.",
        ),
        _record(
            "scheduling_failures",
            (
                "research_core_surface",
                "research_secondary_surface",
                "research_governance_safety",
            ),
            (
                "workflow_control_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "scheduling failures cannot reorder workflow execution",
                "research metadata cannot allocate resources",
            ),
            "Scheduling failures cannot control workflow order or capacity.",
        ),
        _record(
            "retry_failures",
            (
                "research_execution_policy",
                "research_hitl_policies",
                "research_governance_safety",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "research policies cannot trigger retries",
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
                "research helpers cannot execute or repair workflows",
            ),
            "Planning helper failures remain validation-time metadata concerns.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "research_planner",
                "research_decomposer",
                "research_reports",
                "research_core_surface",
                "research_secondary_surface",
            ),
            (
                "prompt_rendering_implemented_false",
                "prompt_mutation_implemented_false",
            ),
            (
                "research metadata cannot render prompts",
                "report metadata cannot mutate prompts or generated output",
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
                "research_memory",
                "research_reports",
                "research_governance_safety",
            ),
            (
                "telemetry_collection_implemented_false",
                "live_research_observation_implemented_false",
            ),
            (
                "observability failures cannot collect telemetry",
                "research metadata cannot observe live outcomes",
            ),
            "Telemetry failures cannot introduce observation side effects.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "research_recommendation_engine",
                "research_secondary_surface",
                "research_governance_safety",
            ),
            (
                "provider_model_routing_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "budget failures cannot enforce budgets",
                "research metadata cannot allocate or route resources",
            ),
            "Budget and cost failures remain advisory metadata boundaries.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "research_execution_policy",
                "research_core_surface",
                "research_governance_safety",
            ),
            (
                "workflow_state_integrity_boundary_preserved",
                "workflow_graph_mutation_implemented_false",
            ),
            (
                "failure metadata cannot mutate workflow state",
                "execution policy metadata cannot restore runtime state",
            ),
            "Workflow state integrity is preserved by no mutation behavior.",
        ),
        _record(
            "provider_model_routing_preservation",
            (
                "source_credibility_engine",
                "research_confidence_engine",
                "research_secondary_surface",
                "research_governance_safety",
            ),
            (
                "provider_model_routing_preserved",
                "provider_model_routing_implemented_false",
            ),
            (
                "source credibility cannot apply provider preferences",
                "confidence metadata cannot select providers or models",
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
                "research metadata cannot modify generated output",
                "failure audit metadata cannot repair or rewrite output",
            ),
            "Generated output mutation remains outside V6.4 research.",
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
    check_kind: ResearchFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> ResearchFailurePathAuditRecord:
    return ResearchFailurePathAuditRecord(
        audit_id=f"research_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
