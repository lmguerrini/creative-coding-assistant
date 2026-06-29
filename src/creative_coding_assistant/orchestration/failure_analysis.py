"""V5.4 advisory failure analysis metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_optimization_failure_audit import (
    ExecutionOptimizationFailureAuditRegistry,
    execution_optimization_failure_audit_registry,
)
from .final_v4_hardening import (
    LangGraphErrorPathAuditRegistry,
    langgraph_error_path_audit_registry,
)
from .model_routing_failure_path_audit import (
    ModelRoutingFailurePathAuditRegistry,
    model_routing_failure_path_audit_registry,
)
from .performance_failure_path_audit import (
    PerformanceFailurePathAuditRegistry,
    performance_failure_path_audit_registry,
)
from .retry_policies import RetryPolicyPlan, plan_retry_policies
from .workflow_diagnostics import WorkflowDiagnostics, build_workflow_diagnostics

FailureAnalysisPanelKind = Literal[
    "langgraph_error_paths",
    "execution_failure_audit",
    "routing_failure_audit",
    "performance_failure_audit",
    "retry_failure_boundaries",
    "observability_failure_boundary",
]
FailureAnalysisStatus = Literal["ready", "guarded"]

FAILURE_ANALYSIS_PANEL_SERIALIZATION_VERSION = "failure_analysis_panel.v1"
FAILURE_ANALYSIS_SERIALIZATION_VERSION = "failure_analysis.v1"
FAILURE_ANALYSIS_AUTHORITY_BOUNDARY = (
    "The V5.4 Failure Analysis surface summarizes existing LangGraph "
    "error-path audit, execution optimization failure audit, model-routing "
    "failure audit, performance failure audit, retry policy, and workflow "
    "diagnostic metadata as read-only failure analysis only; it does not "
    "observe runtime failures, classify live errors, route terminal failures, "
    "handle or repair failures, trigger retries or refinement, execute "
    "workflows or replays, mutate workflow graphs, call providers or models, "
    "emit telemetry, alerts, or human review requests, write memory or "
    "storage, modify generated output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "langgraph_error_path_audit_registry",
    "execution_optimization_failure_audit_registry",
    "model_routing_failure_path_audit_registry",
    "performance_failure_path_audit_registry",
    "retry_policy_plan",
    "workflow_diagnostics",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_failure_observation",
    "live_error_classification",
    "terminal_failure_routing",
    "failure_handling_or_repair",
    "retry_or_refinement_triggering",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_replay_execution",
    "execution_replay_execution",
    "provider_or_model_routing",
    "provider_execution",
    "telemetry_emission",
    "alert_emission",
    "human_review_request",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)
_LANGGRAPH_GUARDRAIL_FLAGS = (
    "new_langgraph_nodes_implemented_false",
    "active_multi_agent_execution_implemented_false",
    "provider_model_routing_change_implemented_false",
    "workflow_behavior_change_implemented_false",
    "passive_registry_runtime_activation_implemented_false",
    "generated_output_mutation_implemented_false",
)


class FailureAnalysisPanel(BaseModel):
    """One read-only V5.4 failure analysis panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: FailureAnalysisPanelKind
    status: FailureAnalysisStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    failure_signal_count: int = Field(ge=0, le=3000)
    guardrail_signal_count: int = Field(ge=0, le=1000)
    observed_failure_count: None = None
    handled_failure_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    failure_analysis_panel_implemented: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["failure_analysis_panel.v1"] = (
        FAILURE_ANALYSIS_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"failure_analysis::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_failure_count is not None:
            raise ValueError("observed_failure_count must remain unset")
        if self.handled_failure_count is not None:
            raise ValueError("handled_failure_count must remain unset")
        if self.guardrail_signal_count > self.failure_signal_count:
            raise ValueError("guardrail_signal_count must fit failure_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class FailureAnalysis(BaseModel):
    """Read-only V5.4 failure analysis over passive failure metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["failure_analysis"] = "failure_analysis"
    serialization_version: Literal["failure_analysis.v1"] = (
        FAILURE_ANALYSIS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=FAILURE_ANALYSIS_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    source_langgraph_error_path_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_execution_failure_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_failure_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_performance_failure_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_retry_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[FailureAnalysisPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    failure_signal_count: int = Field(ge=0, le=6000)
    guardrail_signal_count: int = Field(ge=0, le=2000)
    observed_failure_count: None = None
    handled_failure_count: None = None
    failure_analysis_status: FailureAnalysisStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    failure_analysis_implemented: Literal[True] = True
    runtime_failure_observation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    failure_handling_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _analysis_matches_panels(self) -> Self:
        derived_panel_ids = tuple(panel.panel_id for panel in self.panels)
        if len(set(derived_panel_ids)) != len(derived_panel_ids):
            raise ValueError("panel_ids must be unique")
        if self.panel_ids != derived_panel_ids:
            raise ValueError("panel_ids must match panels")
        if self.panel_count != len(self.panels):
            raise ValueError("panel_count must match panels")
        if self.ready_panel_ids != _panel_ids_for_status(self.panels, "ready"):
            raise ValueError("ready_panel_ids must match panels")
        if self.guarded_panel_ids != _panel_ids_for_status(self.panels, "guarded"):
            raise ValueError("guarded_panel_ids must match panels")
        if self.failure_signal_count != sum(
            panel.failure_signal_count for panel in self.panels
        ):
            raise ValueError("failure_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_failure_count is not None:
            raise ValueError("observed_failure_count must remain unset")
        if self.handled_failure_count is not None:
            raise ValueError("handled_failure_count must remain unset")
        if self.failure_analysis_status != _analysis_status(self.panels):
            raise ValueError("failure_analysis_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match failure analysis sources")
        return self


def build_failure_analysis(
    *,
    langgraph_error_paths: LangGraphErrorPathAuditRegistry | None = None,
    execution_failure_audit: ExecutionOptimizationFailureAuditRegistry | None = None,
    routing_failure_audit: ModelRoutingFailurePathAuditRegistry | None = None,
    performance_failure_audit: PerformanceFailurePathAuditRegistry | None = None,
    retry_policy: RetryPolicyPlan | None = None,
    workflow_diagnostics: WorkflowDiagnostics | None = None,
) -> FailureAnalysis:
    """Build read-only failure analysis without observing runtime failures."""

    langgraph_source = langgraph_error_paths or langgraph_error_path_audit_registry()
    execution_source = (
        execution_failure_audit or execution_optimization_failure_audit_registry()
    )
    routing_source = (
        routing_failure_audit or model_routing_failure_path_audit_registry()
    )
    performance_source = (
        performance_failure_audit or performance_failure_path_audit_registry()
    )
    retry_source = retry_policy or plan_retry_policies()
    workflow_source = workflow_diagnostics or build_workflow_diagnostics()
    panels = (
        _langgraph_panel(langgraph_source),
        _execution_failure_panel(execution_source),
        _routing_failure_panel(routing_source),
        _performance_failure_panel(performance_source),
        _retry_policy_panel(retry_source),
        _workflow_diagnostics_panel(workflow_source),
    )

    return FailureAnalysis(
        source_langgraph_error_path_serialization_version=(
            langgraph_source.serialization_version
        ),
        source_execution_failure_audit_serialization_version=(
            execution_source.serialization_version
        ),
        source_routing_failure_audit_serialization_version=(
            routing_source.serialization_version
        ),
        source_performance_failure_audit_serialization_version=(
            performance_source.serialization_version
        ),
        source_retry_policy_serialization_version=retry_source.serialization_version,
        source_workflow_diagnostics_serialization_version=(
            workflow_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        failure_signal_count=sum(panel.failure_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        failure_analysis_status=_analysis_status(panels),
        advisory_actions=_analysis_actions(panels),
    )


def failure_analysis_panel_by_id(
    panel_id: str,
    analysis: FailureAnalysis | None = None,
) -> FailureAnalysisPanel | None:
    """Return one failure analysis panel without runtime behavior."""

    source_analysis = analysis or build_failure_analysis()
    for panel in source_analysis.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def failure_analysis_panels_for_status(
    status: FailureAnalysisStatus,
    analysis: FailureAnalysis | None = None,
) -> tuple[FailureAnalysisPanel, ...]:
    """Return failure analysis panels by status without live classification."""

    source_analysis = analysis or build_failure_analysis()
    return tuple(panel for panel in source_analysis.panels if panel.status == status)


def _langgraph_panel(
    registry: LangGraphErrorPathAuditRegistry,
) -> FailureAnalysisPanel:
    guardrails = len(_LANGGRAPH_GUARDRAIL_FLAGS)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::langgraph_error_paths",
        panel_kind="langgraph_error_paths",
        status=_status_for_guardrails(guardrails),
        source_id="langgraph_error_path_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.surface_ids,
        failure_signal_count=(
            registry.record_count
            + len(registry.surface_ids)
            + len(registry.failure_invariants)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"error_path_records:{registry.record_count}",
            f"runtime_nodes:{len(registry.runtime_node_ids)}",
            f"terminal_failure_node:{registry.terminal_failure_node}",
        ),
        advisory_actions=(
            "Display LangGraph error-path coverage without routing failures.",
            "Keep graph mutation, workflow behavior changes, routing, and output mutation disabled.",
        ),
    )


def _execution_failure_panel(
    registry: ExecutionOptimizationFailureAuditRegistry,
) -> FailureAnalysisPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::execution_failure_audit",
        panel_kind="execution_failure_audit",
        status=_status_for_guardrails(guardrails),
        source_id="execution_optimization_failure_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.audit_ids,
        failure_signal_count=(
            registry.record_count
            + len(registry.applicable_required_checks)
            + len(registry.source_surface_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"audit_records:{registry.record_count}",
            f"applicable_checks:{len(registry.applicable_required_checks)}",
            f"source_surfaces:{len(registry.source_surface_ids)}",
        ),
        advisory_actions=(
            "Display execution optimization failure audit coverage without executing workflows.",
            "Keep retries, provider/model routing, budgets, storage, and output mutation disabled.",
        ),
    )


def _routing_failure_panel(
    registry: ModelRoutingFailurePathAuditRegistry,
) -> FailureAnalysisPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::routing_failure_audit",
        panel_kind="routing_failure_audit",
        status=_status_for_guardrails(guardrails),
        source_id="model_routing_failure_path_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.audit_ids,
        failure_signal_count=(
            registry.record_count
            + len(registry.applicable_required_checks)
            + len(registry.source_surface_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"audit_records:{registry.record_count}",
            f"applicable_checks:{len(registry.applicable_required_checks)}",
            f"source_surfaces:{len(registry.source_surface_ids)}",
        ),
        advisory_actions=(
            "Display model-routing failure audit coverage without applying routing.",
            "Keep provider execution, model switching, HITL requests, workflow control, and retries disabled.",
        ),
    )


def _performance_failure_panel(
    registry: PerformanceFailurePathAuditRegistry,
) -> FailureAnalysisPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::performance_failure_audit",
        panel_kind="performance_failure_audit",
        status=_status_for_guardrails(guardrails),
        source_id="performance_failure_path_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.audit_ids,
        failure_signal_count=(
            registry.record_count
            + len(registry.applicable_required_checks)
            + len(registry.source_surface_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"audit_records:{registry.record_count}",
            f"applicable_checks:{len(registry.applicable_required_checks)}",
            f"source_surfaces:{len(registry.source_surface_ids)}",
        ),
        advisory_actions=(
            "Display performance failure audit coverage without measuring or replaying runtime behavior.",
            "Keep benchmarks, resource allocation, routing, workflow execution, and retries disabled.",
        ),
    )


def _retry_policy_panel(plan: RetryPolicyPlan) -> FailureAnalysisPanel:
    guardrails = len(plan.blocked_runtime_behaviors)
    failure_path_signal = int(plan.failure_path_reachable)
    bounded_cycle_signal = int(plan.bounded_retry_cycle_detected)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::retry_failure_boundaries",
        panel_kind="retry_failure_boundaries",
        status=_status_for_guardrails(guardrails),
        source_id="retry_policy_plan",
        source_serialization_version=plan.serialization_version,
        source_item_ids=plan.candidate_ids,
        failure_signal_count=(
            plan.candidate_count
            + plan.guardrail_candidate_count
            + plan.review_only_candidate_count
            + failure_path_signal
            + bounded_cycle_signal
            + guardrails
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"retry_candidates:{plan.candidate_count}",
            f"failure_path_reachable:{plan.failure_path_reachable}",
            f"max_retry_attempts:{plan.max_retry_attempts}",
        ),
        advisory_actions=(
            "Display retry failure boundaries without triggering retry or refinement.",
            "Keep workflow graph mutation, workflow execution, provider/model routing, and output mutation disabled.",
        ),
    )


def _workflow_diagnostics_panel(
    diagnostics: WorkflowDiagnostics,
) -> FailureAnalysisPanel:
    guardrails = len(diagnostics.blocked_runtime_behaviors)
    return FailureAnalysisPanel(
        panel_id="failure_analysis::observability_failure_boundary",
        panel_kind="observability_failure_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="workflow_diagnostics",
        source_serialization_version=diagnostics.serialization_version,
        source_item_ids=diagnostics.panel_ids,
        failure_signal_count=(
            diagnostics.panel_count
            + len(diagnostics.guarded_panel_ids)
            + len(diagnostics.ready_panel_ids)
            + guardrails
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"diagnostic_panels:{diagnostics.panel_count}",
            f"guarded_panels:{len(diagnostics.guarded_panel_ids)}",
            f"status:{diagnostics.workflow_diagnostics_status}",
        ),
        advisory_actions=(
            "Display workflow diagnostic failure boundaries without live failure analysis.",
            "Keep graph compilation, replay execution, telemetry emission, workflow control, and routing disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[FailureAnalysisPanel, ...],
    status: FailureAnalysisStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> FailureAnalysisStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _analysis_status(
    panels: tuple[FailureAnalysisPanel, ...],
) -> FailureAnalysisStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _analysis_actions(
    panels: tuple[FailureAnalysisPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose failure analysis panels as read-only observability metadata.",
        "Preserve runtime failure observation, terminal routing, retry, replay, "
        "workflow, provider, telemetry, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded failure analysis panels detached from runtime failure handling."
        )
    return tuple(actions)
