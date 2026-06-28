"""Passive V4.6 final V4 hardening closure metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.architecture_consistency_pass import (
    ArchitectureConsistencyPassRegistry,
    architecture_consistency_pass_registry,
    architecture_consistency_record_by_source_registry,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
)

FinalV4HardeningDomain = Literal[
    "contract_registry_boundary",
    "context_memory_boundary",
    "collaboration_workflow_boundary",
    "quality_reliability_determinism_boundary",
    "observability_foundation_boundary",
    "architecture_closure_boundary",
    "langgraph_error_path_boundary",
]
LangGraphErrorPathAuditSurface = Literal[
    "runtime_node_failure_paths",
    "provider_errors",
    "stream_errors",
    "planning_sub_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "workflow_state_consistency",
    "refinement_loop_failures",
    "review_failures",
    "workstation_hydration_failures",
    "preview_preparation_failures",
    "artifact_extraction_failures",
    "artifact_critique_failures",
    "registry_loading_failures",
    "passive_metadata_import_failures",
    "backend_frontend_boundary_failures",
]
FinalV4HardeningStage = Literal["v4_6_final_v4_hardening"]
FinalV4HardeningStatus = Literal["pass"]

LANGGRAPH_ERROR_PATH_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "langgraph_error_path_audit_record.v1"
)
LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "langgraph_error_path_audit.v1"
)
LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID = "langgraph_error_path_audit"
FINAL_V4_HARDENING_RECORD_SERIALIZATION_VERSION = "final_v4_hardening_record.v1"
FINAL_V4_HARDENING_REGISTRY_SERIALIZATION_VERSION = (
    "final_v4_hardening_registry.v1"
)
FINAL_V4_HARDENING_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 final V4 hardening metadata closes passive hardening coverage over "
    "contract, registry, context, collaboration, workflow, reliability, "
    "determinism, observability, cost, performance, architecture consistency, "
    "and LangGraph error-path surfaces only; it does not execute hardening "
    "checks at runtime, mutate architecture documents, change workflow graph "
    "order, route providers or models, select runtimes, invoke agents, "
    "trigger retries, write memory or storage, execute artifacts, or modify "
    "generated output."
)
LANGGRAPH_ERROR_PATH_AUDIT_AUTHORITY_BOUNDARY = (
    "Final V4 LangGraph error-path audit metadata documents existing failure "
    "normalization coverage for runtime nodes and adjacent backend/frontend "
    "boundaries only; it does not add LangGraph nodes, recover by mutating "
    "generated outputs, activate passive registries, alter provider/model "
    "routing, execute multi-agent behavior, or change workflow control."
)

_VALIDATED_HARDENING_SURFACES = (
    "source_registry_coverage",
    "architecture_consistency_coverage",
    "langgraph_error_path_coverage",
    "metadata_only_declarations",
    "blocked_runtime_behaviors",
    "passive_boundary_flags",
    "active_runtime_flag_absence",
    "architecture_doc_refs",
    "routing_preservation",
)
_TERMINAL_FAILURE_NODE = "failure"
_LANGGRAPH_ERROR_PATH_SOURCE_NODE_IDS = tuple(
    node for node in ASSISTANT_WORKFLOW_NODE_ORDER if node != _TERMINAL_FAILURE_NODE
)
_LANGGRAPH_ERROR_PATH_INVARIANTS = (
    "terminal_failure_node_reached",
    "failure_info_normalized",
    "workflow_state_consistent",
    "provider_model_routing_preserved",
    "passive_registries_not_activated",
    "generated_outputs_not_mutated",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "runtime_hardening_engine_blocked",
    "architecture_doc_mutation_blocked",
    "workflow_graph_mutation_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "agent_invocation_blocked",
    "workflow_control_blocked",
    "retry_triggering_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_hardening_engine",
    "architecture_doc_mutation",
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "agent_invocation",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_or_storage_write",
    "artifact_execution",
    "generated_output_modification",
)
_HARDENING_FINDINGS = (
    "source_registry_coverage_confirmed",
    "architecture_consistency_coverage_confirmed",
    "metadata_only_boundaries_confirmed",
    "runtime_behavior_blocks_confirmed",
    "passive_boundary_flags_confirmed",
    "active_runtime_flags_absent",
    "routing_preservation_confirmed",
    "langgraph_error_paths_confirmed",
)
_LOCAL_FINAL_HARDENING_SOURCE_REGISTRY_IDS = (
    LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID,
)
_LANGGRAPH_ERROR_PATH_SURFACE_SPECS: tuple[
    tuple[
        LangGraphErrorPathAuditSurface,
        str,
        tuple[str, ...],
        Literal["tested", "documented", "tested_and_documented"],
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "runtime_node_failure_paths",
        "Runtime Node Failure Paths",
        _LANGGRAPH_ERROR_PATH_SOURCE_NODE_IDS,
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_every_runtime_node_has_conditional_failure_target",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_assistant_workflow_conditional_edge_specs",
        ),
    ),
    (
        "provider_errors",
        "Provider Errors",
        ("generation",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_graph_routes_provider_failures_to_terminal_failure_path",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_failure_info_from_generation_result",
        ),
    ),
    (
        "stream_errors",
        "Stream Errors",
        (
            "intake",
            "routing",
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering",
            "generation",
        ),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_stream_exceptions_route_to_terminal_failure_without_state_corruption",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_emit_streaming_step",
        ),
    ),
    (
        "planning_sub_helper_failures",
        "Planning Sub Helper Failures",
        ("planning",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_planning_sub_helper_failure_routes_to_terminal_failure_without_partial_plan",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_planning_node",
        ),
    ),
    (
        "prompt_rendering_failures",
        "Prompt Rendering Failures",
        ("prompt_rendering",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_prompt_rendering_exception_routes_to_terminal_failure",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_prompt_rendering_node",
        ),
    ),
    (
        "serialization_failures",
        "Serialization Failures",
        (),
        "tested_and_documented",
        (
            "tests/test_nextjs_streaming_bridge.py::"
            "test_iter_stream_ndjson_normalizes_serialization_failures",
            "src/creative_coding_assistant/api/streaming.py::"
            "iter_assistant_stream_ndjson",
        ),
    ),
    (
        "workflow_state_consistency",
        "Workflow State Consistency After Failures",
        _LANGGRAPH_ERROR_PATH_SOURCE_NODE_IDS,
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_stream_exceptions_route_to_terminal_failure_without_state_corruption",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_handle_workflow_exception",
        ),
    ),
    (
        "refinement_loop_failures",
        "Refinement Loop Failures",
        ("review", "refinement", "generation"),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_review_and_refinement_failures_route_to_terminal_failure",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_refinement_node",
        ),
    ),
    (
        "review_failures",
        "Review Failures",
        ("review",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_review_and_refinement_failures_route_to_terminal_failure",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_review_node",
        ),
    ),
    (
        "workstation_hydration_failures",
        "Workstation Hydration Failures",
        (),
        "tested_and_documented",
        (
            "tests/test_workspace_session_persistence.py::"
            "test_wsgi_endpoint_returns_404_for_missing_session",
            "src/creative_coding_assistant/api/workspace_sessions.py::"
            "WorkspaceSessionApplication._handle_get",
        ),
    ),
    (
        "preview_preparation_failures",
        "Preview Preparation Failures",
        ("preview_preparation",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_artifact_pipeline_failures_preserve_generated_outputs",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_preview_preparation_node",
        ),
    ),
    (
        "artifact_extraction_failures",
        "Artifact Extraction Failures",
        ("artifact_extraction",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_artifact_pipeline_failures_preserve_generated_outputs",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_artifact_extraction_node",
        ),
    ),
    (
        "artifact_critique_failures",
        "Artifact Critique Failures",
        ("artifact_critique",),
        "tested_and_documented",
        (
            "tests/test_langgraph_workflow_integration.py::"
            "test_artifact_pipeline_failures_preserve_generated_outputs",
            "src/creative_coding_assistant/orchestration/workflow_graph.py::"
            "_artifact_critique_node",
        ),
    ),
    (
        "registry_loading_failures",
        "Registry Loading Failures",
        (),
        "documented",
        (
            "src/creative_coding_assistant/orchestration/__init__.py::"
            "__getattr__",
            "tests/test_final_v4_hardening.py::"
            "test_langgraph_error_path_audit_covers_required_surfaces",
        ),
    ),
    (
        "passive_metadata_import_failures",
        "Passive Metadata Import Failures",
        (),
        "documented",
        (
            "src/creative_coding_assistant/orchestration/__init__.py::"
            "__getattr__",
            "tests/test_final_v4_hardening.py::"
            "test_langgraph_error_path_audit_is_passive",
        ),
    ),
    (
        "backend_frontend_boundary_failures",
        "Backend Frontend Boundary Failures",
        (),
        "tested_and_documented",
        (
            "tests/test_nextjs_streaming_bridge.py::"
            "test_wsgi_endpoint_rejects_invalid_request",
            "tests/test_nextjs_streaming_bridge.py::"
            "test_iter_stream_ndjson_normalizes_serialization_failures",
        ),
    ),
)
_DOMAIN_SPECS: tuple[
    tuple[FinalV4HardeningDomain, str, tuple[str, ...]],
    ...,
] = (
    (
        "contract_registry_boundary",
        "Contract And Registry Boundary",
        (
            "agent_contract_audit_registry",
            "agent_registry_audit_registry",
        ),
    ),
    (
        "context_memory_boundary",
        "Context And Memory Boundary",
        (
            "blackboard_audit_registry",
            "shared_context_audit_registry",
        ),
    ),
    (
        "collaboration_workflow_boundary",
        "Collaboration And Workflow Boundary",
        (
            "escalation_policy_audit_registry",
            "hybrid_workflow_audit_registry",
            "agent_collaboration_audit_registry",
            "creative_diversity_audit_registry",
        ),
    ),
    (
        "quality_reliability_determinism_boundary",
        "Quality Reliability And Determinism Boundary",
        (
            "agent_explainability_audit_registry",
            "agent_reliability_audit_registry",
            "agent_determinism_audit_registry",
        ),
    ),
    (
        "observability_foundation_boundary",
        "Observability Cost And Performance Boundary",
        (
            "agent_telemetry_foundation_registry",
            "agent_cost_tracking_foundation_registry",
            "agent_performance_tracking_foundation_registry",
        ),
    ),
    (
        "architecture_closure_boundary",
        "Architecture Closure Boundary",
        (
            "architecture_consistency_pass_registry",
            "engine_contract_consistency_registry",
        ),
    ),
    (
        "langgraph_error_path_boundary",
        "LangGraph Error Path Boundary",
        (LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID,),
    ),
)


class LangGraphErrorPathAuditRecord(BaseModel):
    """One passive final V4 LangGraph error-path coverage record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: LangGraphErrorPathAuditSurface
    surface_name: str = Field(min_length=1, max_length=160)
    hardening_stage: FinalV4HardeningStage = "v4_6_final_v4_hardening"
    source_runtime_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    terminal_failure_node: Literal["failure"] = _TERMINAL_FAILURE_NODE
    coverage_mode: Literal["tested", "documented", "tested_and_documented"]
    coverage_refs: tuple[str, ...] = Field(min_length=1, max_length=6)
    failure_invariants: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple)
    terminal_failure_path_confirmed: Literal[True] = True
    failure_normalization_confirmed: Literal[True] = True
    workflow_state_consistency_confirmed: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    passive_registry_runtime_block_confirmed: Literal[True] = True
    generated_output_mutation_blocked: Literal[True] = True
    new_langgraph_nodes_implemented: Literal[False] = False
    active_multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    workflow_behavior_change_implemented: Literal[False] = False
    passive_registry_runtime_activation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["langgraph_error_path_audit_record.v1"] = (
        LANGGRAPH_ERROR_PATH_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class LangGraphErrorPathAuditRegistry(BaseModel):
    """Passive final V4 LangGraph error-path audit metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["langgraph_error_path_audit"] = (
        LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID
    )
    serialization_version: Literal["langgraph_error_path_audit.v1"] = (
        LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LANGGRAPH_ERROR_PATH_AUDIT_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    hardening_stage: FinalV4HardeningStage = "v4_6_final_v4_hardening"
    runtime_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    source_runtime_node_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    terminal_failure_node: Literal["failure"] = _TERMINAL_FAILURE_NODE
    records: tuple[LangGraphErrorPathAuditRecord, ...] = Field(
        min_length=16,
        max_length=16,
    )
    surface_ids: tuple[LangGraphErrorPathAuditSurface, ...] = Field(
        min_length=16,
        max_length=16,
    )
    record_count: int = Field(ge=16, le=16)
    failure_invariants: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple)
    all_runtime_nodes_have_failure_edges: Literal[True] = True
    all_required_error_surfaces_covered: Literal[True] = True
    failure_normalization_preserved: Literal[True] = True
    workflow_state_consistency_preserved: Literal[True] = True
    provider_model_routing_preserved: Literal[True] = True
    passive_registries_runtime_blocked: Literal[True] = True
    generated_output_mutation_blocked: Literal[True] = True
    new_langgraph_nodes_implemented: Literal[False] = False
    active_multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_change_implemented: Literal[False] = False
    workflow_behavior_change_implemented: Literal[False] = False
    passive_registry_runtime_activation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_error_path_records(self) -> Self:
        derived_surface_ids = tuple(record.surface_id for record in self.records)
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _langgraph_error_path_surface_ids():
            raise ValueError("surface_ids must match required error surfaces")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.runtime_node_ids != ASSISTANT_WORKFLOW_NODE_ORDER:
            raise ValueError("runtime_node_ids must match LangGraph node order")
        if self.source_runtime_node_ids != _LANGGRAPH_ERROR_PATH_SOURCE_NODE_IDS:
            raise ValueError("source_runtime_node_ids must match runtime nodes")
        if self.terminal_failure_node in self.source_runtime_node_ids:
            raise ValueError("terminal failure node cannot be a source node")
        if self.failure_invariants != _LANGGRAPH_ERROR_PATH_INVARIANTS:
            raise ValueError("failure_invariants must match audit invariants")
        if self.missing_coverage_items:
            raise ValueError("LangGraph error path audit must not miss coverage")
        if any(
            (
                self.new_langgraph_nodes_implemented,
                self.active_multi_agent_execution_implemented,
                self.provider_model_routing_change_implemented,
                self.workflow_behavior_change_implemented,
                self.passive_registry_runtime_activation_implemented,
                self.generated_output_mutation_implemented,
            )
        ):
            raise ValueError("LangGraph error path audit must remain passive")

        known_nodes = set(self.source_runtime_node_ids)
        for record in self.records:
            if record.hardening_stage != self.hardening_stage:
                raise ValueError("hardening_stage must match registry")
            if record.terminal_failure_node != self.terminal_failure_node:
                raise ValueError("terminal_failure_node must match registry")
            if not set(record.source_runtime_node_ids).issubset(known_nodes):
                raise ValueError("source_runtime_node_ids must be known")
            if record.failure_invariants != self.failure_invariants:
                raise ValueError("failure_invariants must match registry")
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
            if any(
                (
                    record.new_langgraph_nodes_implemented,
                    record.active_multi_agent_execution_implemented,
                    record.provider_model_routing_change_implemented,
                    record.workflow_behavior_change_implemented,
                    record.passive_registry_runtime_activation_implemented,
                    record.generated_output_mutation_implemented,
                )
            ):
                raise ValueError("LangGraph error path records must remain passive")
        return self


class FinalV4HardeningRecord(BaseModel):
    """One passive final V4 hardening closure record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    domain_id: FinalV4HardeningDomain
    domain_name: str = Field(min_length=1, max_length=160)
    hardening_stage: FinalV4HardeningStage = "v4_6_final_v4_hardening"
    source_registry_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_registry_count: int = Field(ge=1, le=4)
    architecture_consistency_record_ids: tuple[str, ...] = Field(
        min_length=0,
        max_length=4,
    )
    architecture_doc_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    validated_hardening_surfaces: tuple[str, ...] = Field(
        min_length=9,
        max_length=9,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    hardening_findings: tuple[str, ...] = Field(min_length=8, max_length=8)
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=24,
    )
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    source_metadata_only_declared: Literal[True] = True
    architecture_consistency_confirmed: Literal[True] = True
    final_hardening_status: FinalV4HardeningStatus = "pass"
    runtime_hardening_engine_implemented: Literal[False] = False
    architecture_doc_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["final_v4_hardening_record.v1"] = (
        FINAL_V4_HARDENING_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class FinalV4HardeningRegistry(BaseModel):
    """Stable passive V4.6 final V4 hardening closure registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["final_v4_hardening_registry"] = "final_v4_hardening_registry"
    serialization_version: Literal["final_v4_hardening_registry.v1"] = (
        FINAL_V4_HARDENING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=FINAL_V4_HARDENING_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    hardening_stage: FinalV4HardeningStage = "v4_6_final_v4_hardening"
    records: tuple[FinalV4HardeningRecord, ...] = Field(min_length=7, max_length=7)
    domain_ids: tuple[FinalV4HardeningDomain, ...] = Field(min_length=7, max_length=7)
    record_count: int = Field(ge=7, le=7)
    source_architecture_consistency_registry: Literal[
        "architecture_consistency_pass_registry"
    ] = "architecture_consistency_pass_registry"
    source_langgraph_error_path_audit_registry: Literal[
        "langgraph_error_path_audit"
    ] = LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID
    source_registry_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    architecture_doc_refs: tuple[str, ...] = Field(min_length=6, max_length=6)
    langgraph_error_path_surface_ids: tuple[
        LangGraphErrorPathAuditSurface,
        ...,
    ] = Field(min_length=16, max_length=16)
    validated_hardening_surfaces: tuple[str, ...] = Field(
        min_length=9,
        max_length=9,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_domains_covered: Literal[True] = True
    architecture_consistency_covered: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    runtime_hardening_engine_implemented: Literal[False] = False
    architecture_doc_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        derived_domain_ids = tuple(record.domain_id for record in self.records)
        if len(set(derived_domain_ids)) != len(derived_domain_ids):
            raise ValueError("domain_ids must be unique")
        if self.domain_ids != derived_domain_ids:
            raise ValueError("domain_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.domain_ids != _domain_ids():
            raise ValueError("domain_ids must match final hardening domains")
        if self.source_registry_ids != _final_source_registry_ids():
            raise ValueError("source_registry_ids must match final source coverage")
        if (
            self.source_langgraph_error_path_audit_registry
            not in self.source_registry_ids
        ):
            raise ValueError("LangGraph error path audit source must be covered")
        if self.langgraph_error_path_surface_ids != (
            langgraph_error_path_audit_registry().surface_ids
        ):
            raise ValueError(
                "langgraph_error_path_surface_ids must match audit registry"
            )
        if self.validated_hardening_surfaces != _VALIDATED_HARDENING_SURFACES:
            raise ValueError("validated_hardening_surfaces must match registry")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match registry")
        if any(
            (
                self.runtime_hardening_engine_implemented,
                self.architecture_doc_mutation_implemented,
                self.workflow_graph_mutation_implemented,
                self.provider_model_routing_implemented,
                self.runtime_selection_implemented,
                self.agent_invocation_implemented,
                self.workflow_control_implemented,
                self.retry_triggering_implemented,
                self.storage_write_implemented,
                self.artifact_execution_implemented,
                self.generated_output_mutation_implemented,
            )
        ):
            raise ValueError("final V4 hardening must remain passive")

        known_source_ids = set(self.source_registry_ids)
        for record in self.records:
            if record.hardening_stage != self.hardening_stage:
                raise ValueError("hardening_stage must match registry")
            if not set(record.source_registry_ids).issubset(known_source_ids):
                raise ValueError("source_registry_ids must be known")
            if record.source_registry_count != len(record.source_registry_ids):
                raise ValueError("source_registry_count must match source ids")
            if record.architecture_doc_refs != self.architecture_doc_refs:
                raise ValueError("architecture_doc_refs must match registry")
            if record.validated_hardening_surfaces != (
                self.validated_hardening_surfaces
            ):
                raise ValueError("validated_hardening_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_active_runtime_flags:
                raise ValueError("records must not contain active runtime flags")
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
        return self


def langgraph_error_path_audit_registry() -> LangGraphErrorPathAuditRegistry:
    """Return passive final V4 LangGraph error-path audit metadata."""

    return LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY


def langgraph_error_path_audit_record_by_surface_id(
    surface_id: str,
    registry: LangGraphErrorPathAuditRegistry | None = None,
) -> LangGraphErrorPathAuditRecord | None:
    """Return one passive LangGraph error-path audit record by surface id."""

    source_registry = registry or LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY
    normalized_surface_id = str(surface_id).strip()
    for record in source_registry.records:
        if record.surface_id == normalized_surface_id:
            return record
    return None


def langgraph_error_path_audit_records_for_node(
    node_id: str,
    registry: LangGraphErrorPathAuditRegistry | None = None,
) -> tuple[LangGraphErrorPathAuditRecord, ...]:
    """Return passive error-path audit records covering one runtime node."""

    source_registry = registry or LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY
    normalized_node_id = str(node_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_node_id in record.source_runtime_node_ids
    )


def final_v4_hardening_registry() -> FinalV4HardeningRegistry:
    """Return passive V4.6 final V4 hardening closure metadata."""

    return FINAL_V4_HARDENING_REGISTRY


def final_v4_hardening_record_by_domain_id(
    domain_id: str,
    registry: FinalV4HardeningRegistry | None = None,
) -> FinalV4HardeningRecord | None:
    """Return one passive final hardening record by domain id."""

    source_registry = registry or FINAL_V4_HARDENING_REGISTRY
    normalized_domain_id = str(domain_id).strip()
    for record in source_registry.records:
        if record.domain_id == normalized_domain_id:
            return record
    return None


def final_v4_hardening_records_for_source_registry(
    source_registry_id: str,
    registry: FinalV4HardeningRegistry | None = None,
) -> tuple[FinalV4HardeningRecord, ...]:
    """Return final V4 hardening records covering one source registry."""

    source_registry = registry or FINAL_V4_HARDENING_REGISTRY
    normalized_source_id = str(source_registry_id).strip()
    return tuple(
        record
        for record in source_registry.records
        if normalized_source_id in record.source_registry_ids
    )


def _domain_ids() -> tuple[FinalV4HardeningDomain, ...]:
    return tuple(domain_id for domain_id, _domain_name, _sources in _DOMAIN_SPECS)


def _langgraph_error_path_surface_ids() -> tuple[LangGraphErrorPathAuditSurface, ...]:
    return tuple(
        surface_id
        for surface_id, _surface_name, _nodes, _coverage_mode, _refs in (
            _LANGGRAPH_ERROR_PATH_SURFACE_SPECS
        )
    )


def _final_source_registry_ids() -> tuple[str, ...]:
    architecture = architecture_consistency_pass_registry()
    return (
        "architecture_consistency_pass_registry",
        *architecture.source_registry_ids,
        *_LOCAL_FINAL_HARDENING_SOURCE_REGISTRY_IDS,
    )


def _architecture_record_ids(
    source_registry_ids: tuple[str, ...],
    architecture: ArchitectureConsistencyPassRegistry,
) -> tuple[str, ...]:
    return tuple(
        source_registry_id
        for source_registry_id in source_registry_ids
        if source_registry_id in architecture.source_registry_ids
    )


def _source_active_runtime_flags(
    source_registry_ids: tuple[str, ...],
    architecture: ArchitectureConsistencyPassRegistry,
) -> tuple[str, ...]:
    active_flags: list[str] = []
    if (
        "architecture_consistency_pass_registry" in source_registry_ids
        and not architecture.no_active_runtime_flags
    ):
        active_flags.append("architecture_consistency_active_runtime_flags")
    for source_registry_id in source_registry_ids:
        record = architecture_consistency_record_by_source_registry(
            source_registry_id,
            architecture,
        )
        if record is None:
            continue
        active_flags.extend(record.source_active_runtime_flags)
    return tuple(active_flags)


def _missing_coverage_items(
    *,
    source_registry_ids: tuple[str, ...],
    architecture_record_ids: tuple[str, ...],
    active_flags: tuple[str, ...],
    architecture: ArchitectureConsistencyPassRegistry,
) -> tuple[str, ...]:
    missing: list[str] = []
    known_source_ids = set(_final_source_registry_ids())
    if not set(source_registry_ids).issubset(known_source_ids):
        missing.append("unknown_source_registry")
    non_architecture_source_ids = tuple(
        source_id
        for source_id in source_registry_ids
        if source_id != "architecture_consistency_pass_registry"
        and source_id not in _LOCAL_FINAL_HARDENING_SOURCE_REGISTRY_IDS
    )
    if set(architecture_record_ids) != set(non_architecture_source_ids):
        missing.append("architecture_consistency_record_missing")
    if not architecture.metadata_only or not architecture.no_missing_coverage:
        missing.append("architecture_consistency_coverage_missing")
    if not architecture.architecture_docs_referenced:
        missing.append("architecture_doc_refs_missing")
    if LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY_ID in source_registry_ids:
        error_path_audit = langgraph_error_path_audit_registry()
        if (
            not error_path_audit.metadata_only
            or not error_path_audit.all_required_error_surfaces_covered
            or error_path_audit.missing_coverage_items
        ):
            missing.append("langgraph_error_path_coverage_missing")
    if active_flags:
        missing.append("active_runtime_flags_present")
    return tuple(missing)


def _record(
    domain_id: FinalV4HardeningDomain,
    domain_name: str,
    source_registry_ids: tuple[str, ...],
) -> FinalV4HardeningRecord:
    architecture = architecture_consistency_pass_registry()
    architecture_record_ids = _architecture_record_ids(source_registry_ids, architecture)
    active_flags = _source_active_runtime_flags(source_registry_ids, architecture)
    return FinalV4HardeningRecord(
        domain_id=domain_id,
        domain_name=domain_name,
        source_registry_ids=source_registry_ids,
        source_registry_count=len(source_registry_ids),
        architecture_consistency_record_ids=architecture_record_ids,
        architecture_doc_refs=architecture.architecture_doc_refs,
        validated_hardening_surfaces=_VALIDATED_HARDENING_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        hardening_findings=_HARDENING_FINDINGS,
        source_active_runtime_flags=active_flags,
        missing_coverage_items=_missing_coverage_items(
            source_registry_ids=source_registry_ids,
            architecture_record_ids=architecture_record_ids,
            active_flags=active_flags,
            architecture=architecture,
        ),
        source_metadata_only_declared=architecture.metadata_only,
        architecture_consistency_confirmed=(
            architecture.metadata_only
            and architecture.no_missing_coverage
            and architecture.no_active_runtime_flags
        ),
    )


LANGGRAPH_ERROR_PATH_AUDIT_RECORDS = tuple(
    LangGraphErrorPathAuditRecord(
        surface_id=surface_id,
        surface_name=surface_name,
        source_runtime_node_ids=source_runtime_node_ids,
        coverage_mode=coverage_mode,
        coverage_refs=coverage_refs,
        failure_invariants=_LANGGRAPH_ERROR_PATH_INVARIANTS,
    )
    for (
        surface_id,
        surface_name,
        source_runtime_node_ids,
        coverage_mode,
        coverage_refs,
    ) in _LANGGRAPH_ERROR_PATH_SURFACE_SPECS
)
LANGGRAPH_ERROR_PATH_AUDIT_REGISTRY = LangGraphErrorPathAuditRegistry(
    runtime_node_ids=ASSISTANT_WORKFLOW_NODE_ORDER,
    source_runtime_node_ids=_LANGGRAPH_ERROR_PATH_SOURCE_NODE_IDS,
    records=LANGGRAPH_ERROR_PATH_AUDIT_RECORDS,
    surface_ids=tuple(record.surface_id for record in LANGGRAPH_ERROR_PATH_AUDIT_RECORDS),
    record_count=len(LANGGRAPH_ERROR_PATH_AUDIT_RECORDS),
    failure_invariants=_LANGGRAPH_ERROR_PATH_INVARIANTS,
)

FINAL_V4_HARDENING_RECORDS = tuple(
    _record(domain_id, domain_name, source_registry_ids)
    for domain_id, domain_name, source_registry_ids in _DOMAIN_SPECS
)
FINAL_V4_HARDENING_REGISTRY = FinalV4HardeningRegistry(
    records=FINAL_V4_HARDENING_RECORDS,
    domain_ids=tuple(record.domain_id for record in FINAL_V4_HARDENING_RECORDS),
    record_count=len(FINAL_V4_HARDENING_RECORDS),
    source_registry_ids=_final_source_registry_ids(),
    architecture_doc_refs=architecture_consistency_pass_registry().architecture_doc_refs,
    langgraph_error_path_surface_ids=langgraph_error_path_audit_registry().surface_ids,
    validated_hardening_surfaces=_VALIDATED_HARDENING_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
