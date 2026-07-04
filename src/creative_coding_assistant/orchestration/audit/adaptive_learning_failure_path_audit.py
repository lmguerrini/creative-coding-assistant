"""V6.1 adaptive learning runtime failure path audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import evaluate_adaptive_learning_engine
from creative_coding_assistant.orchestration.artifact_learning import learn_artifacts
from creative_coding_assistant.orchestration.continuous_improvement_signals import (
    derive_continuous_improvement_signals,
)
from creative_coding_assistant.orchestration.creative_failure_learning import learn_creative_failures
from creative_coding_assistant.orchestration.creative_success_learning import learn_creative_success
from creative_coding_assistant.orchestration.evaluation_learning import learn_evaluations
from creative_coding_assistant.orchestration.failure_pattern_discovery import discover_failure_patterns
from creative_coding_assistant.orchestration.failure_tracking import track_failures
from creative_coding_assistant.orchestration.learning_confidence_calibration import calibrate_learning_confidence
from creative_coding_assistant.orchestration.learning_governance import build_learning_governance
from creative_coding_assistant.orchestration.learning_replay_engine import build_learning_replay_engine
from creative_coding_assistant.orchestration.routing_learning import learn_routing
from creative_coding_assistant.orchestration.runtime_learning import learn_runtimes
from creative_coding_assistant.orchestration.strategy_learning import learn_strategies
from creative_coding_assistant.orchestration.success_pattern_discovery import discover_success_patterns
from creative_coding_assistant.orchestration.technique_learning import learn_techniques
from creative_coding_assistant.orchestration.workflow_success_tracking import track_workflow_success

AdaptiveLearningFailurePathCheckKind = Literal[
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
AdaptiveLearningFailurePathAuditStatus = Literal["pass"]

ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "adaptive_learning_failure_path_audit_record.v1"
)
ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "adaptive_learning_failure_path_audit_registry.v1"
)
ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "V6.1 adaptive learning runtime failure path audit metadata verifies "
    "runtime/RUNTIME_FAILURE_PATH_AUDIT.md checklist coverage for adaptive "
    "learning, workflow success tracking, failure tracking, strategy, "
    "technique, runtime, routing, artifact, evaluation, continuous "
    "improvement, success pattern, failure pattern, learning governance, "
    "learning replay, confidence calibration, creative success learning, and "
    "creative failure learning surfaces only; it does not persist learning "
    "memory, apply feedback, update or enforce learning policies, execute "
    "replay, train models, observe live outcomes, classify live errors, route "
    "terminal failures, handle or repair failures, change provider/model "
    "routing, execute providers, probe runtimes, install dependencies, emit "
    "HITL requests, evaluate generated output, execute or control workflows, "
    "mutate workflow graphs, write storage, mutate generated output or "
    "preferences, perform remediation, or apply Runtime Evolution."
)

_CHECKLIST_SOURCE = "runtime/RUNTIME_FAILURE_PATH_AUDIT.md"
_SOURCE_SURFACE_IDS = (
    "adaptive_learning_engine",
    "workflow_success_tracking",
    "failure_tracking",
    "strategy_learning",
    "technique_learning",
    "runtime_learning",
    "routing_learning",
    "artifact_learning",
    "evaluation_learning",
    "continuous_improvement_signals",
    "success_pattern_discovery",
    "failure_pattern_discovery",
    "learning_governance",
    "learning_replay_engine",
    "learning_confidence_calibration",
    "creative_success_learning",
    "creative_failure_learning",
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
    AdaptiveLearningFailurePathCheckKind,
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
    "learning_memory_persistence",
    "learning_feedback_application",
    "learning_policy_update",
    "learning_policy_enforcement",
    "strategy_mutation",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "runtime_probe",
    "local_model_download",
    "agent_invocation",
    "resource_allocation",
    "hitl_request_emission",
    "budget_enforcement",
    "telemetry_collection",
    "live_success_observation",
    "live_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling_or_repair",
    "generated_output_evaluation",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "retry_or_refinement_triggering",
    "learning_replay_execution",
    "workflow_replay_execution",
    "model_training",
    "runtime_mutation",
    "automatic_preference_mutation",
    "automatic_remediation",
    "technique_application",
    "runtime_selection",
    "artifact_execution",
    "dependency_installation",
    "preview_behavior_change",
    "graph_compilation",
    "prompt_rendering_or_mutation",
    "persistent_storage_write",
    "success_metric_persistence",
    "generated_output_modification",
    "runtime_evolution_application",
)


class AdaptiveLearningFailurePathAuditRecord(BaseModel):
    """One passive V6.1 adaptive learning runtime failure path audit record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    audit_id: str = Field(min_length=1, max_length=220)
    check_kind: AdaptiveLearningFailurePathCheckKind
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=17)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    invariant_assertions: tuple[str, ...] = Field(min_length=1, max_length=8)
    failure_response_boundary: str = Field(min_length=1, max_length=360)
    audit_status: AdaptiveLearningFailurePathAuditStatus = "pass"
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    checklist_item_applicable: Literal[True] = True
    runtime_failure_path_audit_implemented: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    learning_policy_enforcement_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_success_observation_implemented: Literal[False] = False
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
    learning_replay_execution_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    model_training_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    technique_application_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_learning_failure_path_audit_record.v1"] = (
        ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
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
            raise ValueError("source_surface_ids must reference V6.1 surfaces")
        return self


class AdaptiveLearningFailurePathAuditRegistry(BaseModel):
    """Passive V6.1 adaptive learning runtime failure path audit registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_learning_failure_path_audit_registry"] = (
        "adaptive_learning_failure_path_audit_registry"
    )
    serialization_version: Literal[
        "adaptive_learning_failure_path_audit_registry.v1"
    ] = ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=ADAPTIVE_LEARNING_FAILURE_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    checklist_source: Literal["runtime/RUNTIME_FAILURE_PATH_AUDIT.md"] = (
        _CHECKLIST_SOURCE
    )
    source_surface_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=17,
        max_length=17,
    )
    required_checks: tuple[str, ...] = Field(min_length=19, max_length=19)
    applicable_required_checks: tuple[
        AdaptiveLearningFailurePathCheckKind,
        ...,
    ] = Field(min_length=17, max_length=17)
    not_applicable_required_checks: tuple[str, ...] = Field(
        min_length=2,
        max_length=2,
    )
    records: tuple[AdaptiveLearningFailurePathAuditRecord, ...] = Field(
        min_length=17,
        max_length=17,
    )
    audit_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    check_kinds: tuple[AdaptiveLearningFailurePathCheckKind, ...] = Field(
        min_length=17,
        max_length=17,
    )
    record_count: int = Field(ge=17, le=17)
    metadata_only_rule_satisfied: Literal[True] = True
    active_behavior_rule_satisfied: Literal[True] = True
    all_applicable_checks_covered: Literal[True] = True
    learning_memory_boundary_preserved: Literal[True] = True
    feedback_policy_boundary_preserved: Literal[True] = True
    runtime_failure_boundary_preserved: Literal[True] = True
    workflow_state_integrity_boundary_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    generated_output_mutation_boundary_preserved: Literal[True] = True
    passive_registry_activation_boundary_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_update_implemented: Literal[False] = False
    learning_policy_enforcement_implemented: Literal[False] = False
    strategy_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    runtime_probe_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    telemetry_collection_implemented: Literal[False] = False
    live_success_observation_implemented: Literal[False] = False
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
    learning_replay_execution_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    model_training_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    automatic_preference_mutation_implemented: Literal[False] = False
    automatic_remediation_implemented: Literal[False] = False
    technique_application_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    dependency_installation_implemented: Literal[False] = False
    preview_behavior_change_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    prompt_rendering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    success_metric_persistence_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
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
            raise ValueError("source_surface_ids must match V6.1 surfaces")
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


def adaptive_learning_failure_path_audit_registry() -> (
    AdaptiveLearningFailurePathAuditRegistry
):
    """Return V6.1 runtime failure path audit metadata without runtime actions."""

    records = _records()
    return AdaptiveLearningFailurePathAuditRegistry(
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


def adaptive_learning_failure_path_audit_by_id(
    audit_id: str,
    registry: AdaptiveLearningFailurePathAuditRegistry | None = None,
) -> AdaptiveLearningFailurePathAuditRecord | None:
    """Return one V6.1 failure audit record without activating failures."""

    source_registry = registry or adaptive_learning_failure_path_audit_registry()
    normalized_audit_id = str(audit_id).strip()
    for record in source_registry.records:
        if record.audit_id == normalized_audit_id:
            return record
    return None


def adaptive_learning_failure_path_audits_for_check(
    check_kind: AdaptiveLearningFailurePathCheckKind,
    registry: AdaptiveLearningFailurePathAuditRegistry | None = None,
) -> tuple[AdaptiveLearningFailurePathAuditRecord, ...]:
    """Return V6.1 failure audit records by checklist item."""

    source_registry = registry or adaptive_learning_failure_path_audit_registry()
    return tuple(
        record for record in source_registry.records if record.check_kind == check_kind
    )


def adaptive_learning_failure_path_audits_for_surface(
    surface_id: str,
    registry: AdaptiveLearningFailurePathAuditRegistry | None = None,
) -> tuple[AdaptiveLearningFailurePathAuditRecord, ...]:
    """Return V6.1 failure audit records for one adaptive learning surface."""

    source_registry = registry or adaptive_learning_failure_path_audit_registry()
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
            evaluate_adaptive_learning_engine(),
            track_workflow_success(),
            track_failures(),
            learn_strategies(),
            learn_techniques(),
            learn_runtimes(),
            learn_routing(),
            learn_artifacts(),
            learn_evaluations(),
            derive_continuous_improvement_signals(),
            discover_success_patterns(),
            discover_failure_patterns(),
            build_learning_governance(),
            build_learning_replay_engine(),
            calibrate_learning_confidence(),
            learn_creative_success(),
            learn_creative_failures(),
        )
    )


def _records() -> tuple[AdaptiveLearningFailurePathAuditRecord, ...]:
    return (
        _record(
            "node_level_failure_paths",
            _SOURCE_SURFACE_IDS,
            (
                "source_surface_order_matches_v6_1_learning_registry",
                "metadata_only_rule_satisfied",
            ),
            (
                "learning surface failures cannot instantiate runtime nodes",
                "learning metadata cannot execute providers or workflows",
            ),
            "Node-level failures stop at metadata construction and validation.",
        ),
        _record(
            "terminal_failure_routing",
            (
                "failure_tracking",
                "failure_pattern_discovery",
                "learning_governance",
            ),
            (
                "terminal_failure_routing_implemented_false",
                "failure_handling_implemented_false",
            ),
            (
                "failure metadata cannot route terminal failures",
                "guarded governance metadata cannot request or execute recovery",
            ),
            "Terminal failure routing remains outside V6.1 learning metadata.",
        ),
        _record(
            "provider_failures",
            (
                "adaptive_learning_engine",
                "routing_learning",
                "continuous_improvement_signals",
                "learning_governance",
                "learning_replay_engine",
                "learning_confidence_calibration",
            ),
            (
                "provider_execution_implemented_false",
                "provider_model_routing_implemented_false",
            ),
            (
                "provider failures cannot call providers",
                "routing learning cannot switch providers or models",
                "learning replay and confidence calibration cannot call providers",
            ),
            "Provider failures are recorded only as passive learning guardrails.",
        ),
        _record(
            "model_routing_failures",
            (
                "adaptive_learning_engine",
                "routing_learning",
                "failure_pattern_discovery",
                "learning_governance",
                "learning_confidence_calibration",
            ),
            (
                "provider_model_routing_implemented_false",
                "local_model_download_implemented_false",
            ),
            (
                "model-routing failures cannot change configured routing",
                "learning metadata cannot download or select models",
            ),
            "Model-routing failures cannot mutate routing or model selection.",
        ),
        _record(
            "stream_failures",
            (
                "runtime_learning",
                "workflow_success_tracking",
                "continuous_improvement_signals",
            ),
            (
                "workflow_execution_implemented_false",
                "telemetry_collection_implemented_false",
            ),
            (
                "stream failures cannot subscribe to runtime output",
                "success tracking cannot observe live streaming outcomes",
            ),
            "Stream failures remain read-only learning metadata.",
        ),
        _record(
            "scheduling_failures",
            (
                "strategy_learning",
                "runtime_learning",
                "artifact_learning",
                "learning_governance",
            ),
            (
                "workflow_control_implemented_false",
                "resource_allocation_implemented_false",
            ),
            (
                "scheduling failures cannot reorder workflow execution",
                "learning recommendations cannot allocate resources",
            ),
            "Scheduling failures cannot control workflow order or capacity.",
        ),
        _record(
            "retry_failures",
            (
                "failure_tracking",
                "failure_pattern_discovery",
                "continuous_improvement_signals",
                "creative_failure_learning",
            ),
            (
                "retry_triggering_implemented_false",
                "refinement_triggering_implemented_false",
            ),
            (
                "failure patterns cannot trigger retries",
                "continuous improvement metadata cannot start refinement loops",
            ),
            "Retry failures cannot start execution, retry, or refinement.",
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
                "helper failures cannot mutate learning or workflow state",
            ),
            "Planning/helper failures stay inside pure metadata helpers.",
        ),
        _record(
            "prompt_rendering_failures",
            (
                "technique_learning",
                "artifact_learning",
                "evaluation_learning",
                "creative_failure_learning",
            ),
            (
                "prompt_rendering_implemented_false",
                "prompt_mutation_implemented_false",
            ),
            (
                "technique learning cannot render or rewrite prompts",
                "evaluation learning cannot modify generated output",
            ),
            "Prompt rendering failures are blocked because prompts are not rendered.",
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
                "versioned learning metadata cannot activate runtime behavior",
            ),
            "Serialization failures stop at pydantic validation boundaries.",
        ),
        _record(
            "registry_import_loading_failures",
            _SOURCE_SURFACE_IDS,
            (
                "lazy_exports_resolve_metadata_helpers",
                "source_serialization_versions_match_sources",
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
                "workflow_success_tracking",
                "failure_tracking",
                "success_pattern_discovery",
                "failure_pattern_discovery",
            ),
            (
                "telemetry_collection_implemented_false",
                "live_success_observation_implemented_false",
            ),
            (
                "success tracking cannot collect live telemetry",
                "failure tracking cannot observe or classify live failures",
            ),
            "Telemetry and observability failures remain passive source metadata.",
        ),
        _record(
            "budget_cost_prediction_failures",
            (
                "adaptive_learning_engine",
                "routing_learning",
                "learning_governance",
            ),
            (
                "budget_enforcement_implemented_false",
                "provider_execution_implemented_false",
            ),
            (
                "cost or budget failures cannot enforce budgets",
                "routing learning cannot start provider execution",
            ),
            "Budget and cost prediction failures cannot trigger enforcement.",
        ),
        _record(
            "workflow_state_integrity_after_failure",
            (
                "workflow_success_tracking",
                "continuous_improvement_signals",
                "learning_governance",
                "learning_replay_engine",
                "creative_success_learning",
                "creative_failure_learning",
            ),
            (
                "workflow_graph_mutation_implemented_false",
                "workflow_control_implemented_false",
            ),
            (
                "workflow state remains unchanged after learning metadata failures",
                "continuous improvement signals cannot control workflows",
            ),
            "Workflow state integrity is preserved because V6.1 adds no control.",
        ),
        _record(
            "provider_model_routing_preservation",
            (
                "routing_learning",
                "adaptive_learning_engine",
                "learning_governance",
                "learning_confidence_calibration",
            ),
            (
                "provider_model_routing_implemented_false",
                "provider_execution_implemented_false",
            ),
            (
                "provider/model routing is not applied by V6.1 metadata",
                "learning governance cannot switch providers or models",
            ),
            "Provider/model routing is preserved under every learning failure.",
        ),
        _record(
            "generated_output_mutation_boundaries",
            (
                "artifact_learning",
                "evaluation_learning",
                "success_pattern_discovery",
                "failure_pattern_discovery",
                "creative_success_learning",
                "creative_failure_learning",
            ),
            (
                "generated_output_mutation_implemented_false",
                "generated_output_evaluation_implemented_false",
            ),
            (
                "artifact learning cannot mutate generated artifacts",
                "evaluation learning cannot score or modify generated output",
            ),
            "Generated-output mutation is blocked for all adaptive learning failures.",
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
    check_kind: AdaptiveLearningFailurePathCheckKind,
    source_surface_ids: tuple[str, ...],
    evidence: tuple[str, ...],
    invariant_assertions: tuple[str, ...],
    failure_response_boundary: str,
) -> AdaptiveLearningFailurePathAuditRecord:
    return AdaptiveLearningFailurePathAuditRecord(
        audit_id=f"adaptive_learning_failure::{check_kind}",
        check_kind=check_kind,
        source_surface_ids=source_surface_ids,
        evidence=evidence,
        invariant_assertions=invariant_assertions,
        failure_response_boundary=failure_response_boundary,
    )
