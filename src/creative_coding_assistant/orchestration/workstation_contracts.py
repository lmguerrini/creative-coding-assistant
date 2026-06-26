"""Shared Creative Workstation contracts for V3.5 metadata surfaces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WorkstationContractCategory = Literal["creative_workstation"]
WorkstationContractCacheability = Literal[
    "client_snapshot_derived",
    "client_snapshot_and_stream_derived",
]
WorkstationContractHydrationMode = Literal[
    "local_client_projection",
    "stream_metadata_projection",
    "composite_metadata_projection",
]

WORKSTATION_ENGINE_CONTRACT_SERIALIZATION_VERSION = (
    "workstation_engine_contract.v1"
)
WORKSTATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION = (
    "workstation_engine_contract_registry.v1"
)
WORKSTATION_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "Creative Workstation contracts describe V3.5 metadata surfaces, "
    "dependencies, signals, missing-metadata behavior, and future hooks only; "
    "they do not implement V4 agents, V5 execution optimization, V6 learning, "
    "provider routing, runtime selection, retries, workflow control, artifact "
    "execution, preview execution, artifact modification, or generated output "
    "changes."
)


class WorkstationSurfaceCostMetadata(BaseModel):
    """Static estimated cost metadata for a workstation surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_cost: Literal["low"] = "low"
    external_provider_calls: bool = False
    cost_basis: str = Field(min_length=1, max_length=260)
    cache_sensitivity: str = Field(min_length=1, max_length=260)


class WorkstationSurfaceLatencyMetadata(BaseModel):
    """Static estimated latency metadata for a workstation surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_latency: Literal["low"] = "low"
    latency_basis: str = Field(min_length=1, max_length=260)
    blocking_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class WorkstationEngineContract(BaseModel):
    """Common metadata contract exposed by every V3.5 workstation surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=80)
    surface_name: str = Field(min_length=1, max_length=140)
    surface_version: str = Field(min_length=1, max_length=24)
    surface_category: WorkstationContractCategory = "creative_workstation"
    authority_boundary: str = Field(min_length=1, max_length=900)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    optional_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    exposed_metadata: tuple[str, ...] = Field(min_length=1, max_length=18)
    exposed_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    stability_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    missing_metadata_behavior: str = Field(min_length=1, max_length=360)
    downstream_consumers: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=18,
    )
    upstream_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=18)
    cacheability: WorkstationContractCacheability
    hydration_mode: WorkstationContractHydrationMode
    estimated_cost_metadata: WorkstationSurfaceCostMetadata
    estimated_latency_metadata: WorkstationSurfaceLatencyMetadata
    serialization_version: Literal["workstation_engine_contract.v1"] = (
        WORKSTATION_ENGINE_CONTRACT_SERIALIZATION_VERSION
    )
    future_agent_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    future_execution_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    future_evolution_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)


class WorkstationEngineContractRegistry(BaseModel):
    """Stable registry of Creative Workstation metadata surface contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workstation_engine_contract_registry"] = (
        "workstation_engine_contract_registry"
    )
    surface_category: WorkstationContractCategory = "creative_workstation"
    serialization_version: Literal[
        "workstation_engine_contract_registry.v1"
    ] = WORKSTATION_ENGINE_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=WORKSTATION_ENGINE_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    surface_contracts: tuple[WorkstationEngineContract, ...] = Field(
        min_length=7,
        max_length=7,
    )
    surface_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    contract_count: int = Field(ge=7, le=7)
    future_capability_consumers: tuple[str, ...] = Field(
        min_length=3,
        max_length=3,
    )


def workstation_engine_contracts() -> WorkstationEngineContractRegistry:
    """Return the static Creative Workstation contract registry."""

    return WORKSTATION_ENGINE_CONTRACT_REGISTRY


def workstation_engine_contract_by_id(
    surface_id: str,
) -> WorkstationEngineContract | None:
    """Return one workstation contract by id without changing behavior."""

    for contract in WORKSTATION_ENGINE_CONTRACTS:
        if contract.surface_id == surface_id:
            return contract
    return None


def _contract(
    *,
    surface_id: str,
    surface_name: str,
    authority_boundary: str,
    required_inputs: tuple[str, ...],
    optional_inputs: tuple[str, ...] = (),
    exposed_metadata: tuple[str, ...],
    exposed_signals: tuple[str, ...],
    stability_signals: tuple[str, ...],
    missing_metadata_behavior: str,
    downstream_consumers: tuple[str, ...],
    upstream_dependencies: tuple[str, ...],
    cacheability: WorkstationContractCacheability = (
        "client_snapshot_and_stream_derived"
    ),
    hydration_mode: WorkstationContractHydrationMode = (
        "composite_metadata_projection"
    ),
    future_agent_hooks: tuple[str, ...] = (
        "v4_agentic_studio_context_packet",
        "v4_operator_handoff_context",
        "v4_agentic_review_surface",
    ),
    future_execution_hooks: tuple[str, ...] = (
        "v5_adaptive_execution_context",
        "v5_operator_policy_signal",
        "v5_execution_readiness_summary",
    ),
    future_evolution_hooks: tuple[str, ...] = (
        "v6_creative_evolution_timeline_context",
        "v6_learning_signal_contract",
        "v6_lineage_review_context",
    ),
) -> WorkstationEngineContract:
    return WorkstationEngineContract(
        surface_id=surface_id,
        surface_name=surface_name,
        surface_version="v3.5",
        authority_boundary=authority_boundary,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
        exposed_metadata=exposed_metadata,
        exposed_signals=exposed_signals,
        stability_signals=stability_signals,
        missing_metadata_behavior=missing_metadata_behavior,
        downstream_consumers=downstream_consumers,
        upstream_dependencies=upstream_dependencies,
        cacheability=cacheability,
        hydration_mode=hydration_mode,
        estimated_cost_metadata=WorkstationSurfaceCostMetadata(
            cost_basis=(
                "Deterministic workstation metadata projection from existing "
                "client snapshot, stream, and workflow trace state."
            ),
            cache_sensitivity=(
                "Cache key must include the active workspace snapshot, stream "
                "sequence, selection state, and declared upstream surfaces."
            ),
        ),
        estimated_latency_metadata=WorkstationSurfaceLatencyMetadata(
            latency_basis=(
                "Bounded local metadata construction with no network, "
                "provider, runtime, preview, artifact execution, or export work."
            ),
            blocking_inputs=required_inputs,
        ),
        future_agent_hooks=future_agent_hooks,
        future_execution_hooks=future_execution_hooks,
        future_evolution_hooks=future_evolution_hooks,
    )


WORKSTATION_STATE_CONTRACT = _contract(
    surface_id="workstation_state",
    surface_name="Workstation State",
    authority_boundary=(
        "Projects session, run, selection, panel, readiness, and metadata "
        "status from existing workspace state only; it does not change "
        "workflow control, provider routing, runtime selection, artifacts, "
        "previews, retries, or generated output."
    ),
    required_inputs=("assistant_workspace_snapshot",),
    optional_inputs=(
        "workflow_runtime_trace",
        "stream_error",
        "selected_artifact",
        "selected_evaluation",
        "active_inspector_tab",
        "layout_state",
    ),
    exposed_metadata=(
        "WorkstationState",
        "WorkstationSessionState",
        "WorkstationCurrentRunState",
        "WorkstationSelectionState",
        "WorkstationReadinessSummary",
    ),
    exposed_signals=(
        "readiness_state",
        "current_run_state",
        "active_artifact_id",
        "active_workflow_node_id",
        "metadata_statuses",
    ),
    stability_signals=(
        "snapshot_schema_alignment",
        "bounded_missing_metadata_summary",
        "selection_resolution",
    ),
    missing_metadata_behavior=(
        "Reports unavailable metadata as missing or degraded readiness while "
        "preserving the active workstation shell state."
    ),
    downstream_consumers=(
        "session_intelligence",
        "workflow_explorer",
        "provenance_engine",
        "creative_timeline",
        "v3_inspector_panels",
        "workstation_dashboard",
    ),
    upstream_dependencies=("assistant_workspace_snapshot", "assistant_stream_events"),
    cacheability="client_snapshot_derived",
    hydration_mode="local_client_projection",
)

SESSION_INTELLIGENCE_CONTRACT = _contract(
    surface_id="session_intelligence",
    surface_name="Session Intelligence",
    authority_boundary=(
        "Summarizes session readiness, completion, warnings, and recommended "
        "operator actions from existing workstation metadata only; it does "
        "not create autonomous actions, agent plans, retries, or execution "
        "policy."
    ),
    required_inputs=("workstation_state", "assistant_workspace_snapshot"),
    optional_inputs=("session_intelligence_stream_metadata", "stream_error"),
    exposed_metadata=(
        "SessionIntelligenceModel",
        "session_readiness",
        "completion_status",
        "session_warnings",
        "recommended_actions",
    ),
    exposed_signals=(
        "metadata_completion_status",
        "operator_next_action",
        "run_context_summary",
        "session_warning_count",
    ),
    stability_signals=(
        "readiness_state_alignment",
        "warning_deduplication",
        "bounded_action_recommendations",
    ),
    missing_metadata_behavior=(
        "Falls back to workstation readiness and explicit missing metadata "
        "summaries without inventing session facts."
    ),
    downstream_consumers=("workstation_dashboard", "v4_agentic_studio_context"),
    upstream_dependencies=("workstation_state",),
)

WORKFLOW_EXPLORER_CONTRACT = _contract(
    surface_id="workflow_explorer",
    surface_name="Workflow Explorer",
    authority_boundary=(
        "Projects workflow nodes, edges, runtime status, active step, and "
        "progress from existing workflow metadata only; it does not alter "
        "LangGraph order, retry policy, provider routing, runtime selection, "
        "or execution semantics."
    ),
    required_inputs=(
        "workstation_state",
        "assistant_workspace_snapshot",
        "workflow_runtime_trace",
    ),
    optional_inputs=("assistant_stream_events", "workflow_runtime_model"),
    exposed_metadata=(
        "WorkflowExplorerModel",
        "workflow_nodes",
        "workflow_edges",
        "active_node",
        "progress_summary",
    ),
    exposed_signals=(
        "active_step_status",
        "node_reachability",
        "transition_count",
        "runtime_status",
        "workflow_progress_ratio",
    ),
    stability_signals=(
        "node_id_alignment",
        "transition_ordering",
        "active_step_resolution",
    ),
    missing_metadata_behavior=(
        "Keeps the workflow explorer inspectable with empty or pending node "
        "state when runtime trace metadata is incomplete."
    ),
    downstream_consumers=(
        "creative_timeline",
        "workstation_dashboard",
        "v4_agentic_studio_workflow_context",
    ),
    upstream_dependencies=("workstation_state", "workflow_runtime_trace"),
)

PROVENANCE_ENGINE_CONTRACT = _contract(
    surface_id="provenance_engine",
    surface_name="Provenance Engine",
    authority_boundary=(
        "Aggregates evidence, dependency, artifact, evaluation, final payload, "
        "and missing-source provenance from existing metadata only; it does "
        "not fetch sources, verify external claims, execute artifacts, or "
        "change generation behavior."
    ),
    required_inputs=("workstation_state", "assistant_workspace_snapshot"),
    optional_inputs=(
        "workflow_trace_events",
        "artifact_metadata",
        "evaluation_metadata",
        "final_payload_metadata",
    ),
    exposed_metadata=(
        "ProvenanceEngineModel",
        "evidence_sources",
        "dependency_sources",
        "artifact_sources",
        "evaluation_sources",
        "unsupported_or_missing_sources",
    ),
    exposed_signals=(
        "source_count",
        "dependency_count",
        "artifact_source_count",
        "evaluation_source_count",
        "missing_source_count",
    ),
    stability_signals=(
        "source_category_alignment",
        "missing_source_visibility",
        "non_invented_provenance",
    ),
    missing_metadata_behavior=(
        "Surfaces unavailable provenance as unsupported or missing source "
        "records instead of creating inferred evidence."
    ),
    downstream_consumers=(
        "creative_timeline",
        "v3_inspector_panels",
        "workstation_dashboard",
        "v6_lineage_review_context",
    ),
    upstream_dependencies=(
        "workstation_state",
        "artifact_intelligence_metadata",
        "creative_evaluation_metadata",
    ),
)

CREATIVE_TIMELINE_CONTRACT = _contract(
    surface_id="creative_timeline",
    surface_name="Creative Timeline",
    authority_boundary=(
        "Orders existing request, planning, retrieval, creative intelligence, "
        "generative design, artifact intelligence, evaluation, and final "
        "metadata into timeline stages only; it does not schedule work, "
        "modify outputs, or introduce learning behavior."
    ),
    required_inputs=("workstation_state", "workflow_explorer", "provenance_engine"),
    optional_inputs=(
        "creative_intelligence_metadata",
        "generative_design_metadata",
        "artifact_intelligence_metadata",
        "creative_evaluation_metadata",
        "assistant_stream_events",
    ),
    exposed_metadata=(
        "CreativeTimelineModel",
        "timeline_stages",
        "stage_statuses",
        "stage_provenance_counts",
        "timeline_warnings",
    ),
    exposed_signals=(
        "stage_completion_state",
        "metadata_hydration_state",
        "provenance_count",
        "warning_count",
        "latest_stage_event",
    ),
    stability_signals=(
        "stable_stage_order",
        "partial_stage_visibility",
        "bounded_warning_summary",
    ),
    missing_metadata_behavior=(
        "Keeps stages visible as pending or partial and attaches explicit "
        "warnings when expected V3 metadata is absent."
    ),
    downstream_consumers=(
        "v3_inspector_panels",
        "workstation_dashboard",
        "v6_creative_evolution_timeline_context",
    ),
    upstream_dependencies=(
        "workstation_state",
        "workflow_explorer",
        "provenance_engine",
    ),
    future_evolution_hooks=(
        "v6_creative_evolution_timeline_context",
        "v6_lineage_stage_signal",
        "v6_learning_signal_contract",
    ),
)

V3_INSPECTOR_PANELS_CONTRACT = _contract(
    surface_id="v3_inspector_panels",
    surface_name="V3 Inspector Panels",
    authority_boundary=(
        "Groups existing creative intelligence, generative design, artifact "
        "intelligence, creative evaluation, and provenance records into "
        "inspectable panels only; it does not score, critique, repair, retry, "
        "or invoke future agents."
    ),
    required_inputs=("workstation_state", "provenance_engine"),
    optional_inputs=(
        "creative_intelligence_metadata",
        "generative_design_metadata",
        "artifact_intelligence_metadata",
        "creative_evaluation_metadata",
        "workflow_explorer",
    ),
    exposed_metadata=(
        "V3InspectorPanelsModel",
        "creative_intelligence_panel",
        "artifact_intelligence_panel",
        "creative_evaluation_panel",
        "provenance_panel",
    ),
    exposed_signals=(
        "available_panel_count",
        "partial_panel_count",
        "missing_panel_count",
        "record_hydration_state",
        "metadata_group_status",
    ),
    stability_signals=(
        "known_panel_group_order",
        "partial_record_fallbacks",
        "source_status_alignment",
    ),
    missing_metadata_behavior=(
        "Renders missing panel records as explicit missing items while keeping "
        "available partial metadata inspectable."
    ),
    downstream_consumers=(
        "workstation_dashboard",
        "v4_agentic_studio_review_context",
        "v5_adaptive_execution_context",
    ),
    upstream_dependencies=(
        "workstation_state",
        "provenance_engine",
        "creative_intelligence_metadata",
        "artifact_intelligence_metadata",
        "creative_evaluation_metadata",
    ),
    future_agent_hooks=(
        "v4_agentic_studio_context_packet",
        "v4_agentic_review_surface",
        "v4_agent_handoff_readiness",
    ),
)

WORKSTATION_DASHBOARD_CONTRACT = _contract(
    surface_id="workstation_dashboard",
    surface_name="Workstation Dashboard",
    authority_boundary=(
        "Summarizes quality, confidence, consistency, artifact readiness, "
        "runtime fit, evaluation report, workflow health, and HITL signals "
        "from existing workstation surfaces only; it does not trigger "
        "execution optimization, retries, routing, or autonomous decisions."
    ),
    required_inputs=(
        "workstation_state",
        "workflow_runtime_model",
        "v3_inspector_panels",
    ),
    optional_inputs=("assistant_workspace_snapshot", "active_artifact"),
    exposed_metadata=(
        "WorkstationDashboardModel",
        "dashboard_cards",
        "dashboard_summary_counts",
        "hitl_recommendation_card",
        "workflow_health_card",
    ),
    exposed_signals=(
        "good_count",
        "watch_count",
        "missing_count",
        "error_count",
        "hitl_recommendation",
        "runtime_fit_status",
    ),
    stability_signals=(
        "fixed_card_order",
        "bounded_summary_counts",
        "explicit_missing_card_state",
    ),
    missing_metadata_behavior=(
        "Shows missing or watch cards for unavailable source metadata instead "
        "of deriving hidden execution policy."
    ),
    downstream_consumers=(
        "v4_agentic_studio_overview_context",
        "v5_operator_policy_signal",
        "v6_learning_signal_contract",
    ),
    upstream_dependencies=(
        "workstation_state",
        "workflow_explorer",
        "v3_inspector_panels",
    ),
    future_execution_hooks=(
        "v5_adaptive_execution_context",
        "v5_operator_policy_signal",
        "v5_execution_readiness_summary",
    ),
)

WORKSTATION_ENGINE_CONTRACTS = (
    WORKSTATION_STATE_CONTRACT,
    SESSION_INTELLIGENCE_CONTRACT,
    WORKFLOW_EXPLORER_CONTRACT,
    PROVENANCE_ENGINE_CONTRACT,
    CREATIVE_TIMELINE_CONTRACT,
    V3_INSPECTOR_PANELS_CONTRACT,
    WORKSTATION_DASHBOARD_CONTRACT,
)

WORKSTATION_ENGINE_CONTRACT_REGISTRY = WorkstationEngineContractRegistry(
    surface_contracts=WORKSTATION_ENGINE_CONTRACTS,
    surface_ids=tuple(contract.surface_id for contract in WORKSTATION_ENGINE_CONTRACTS),
    contract_count=len(WORKSTATION_ENGINE_CONTRACTS),
    future_capability_consumers=(
        "v4_agentic_studio",
        "v5_adaptive_creative_execution",
        "v6_creative_evolution",
    ),
)
