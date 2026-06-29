"""V5.4 advisory workflow health monitoring metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .error_intelligence import ErrorIntelligence, build_error_intelligence
from .failure_analysis import FailureAnalysis, build_failure_analysis
from .performance_dashboard import PerformanceDashboard, build_performance_dashboard
from .production_telemetry import ProductionTelemetrySurface, build_production_telemetry
from .retry_policies import RetryPolicyPlan, plan_retry_policies
from .workflow_diagnostics import WorkflowDiagnostics, build_workflow_diagnostics

WorkflowHealthPanelKind = Literal[
    "diagnostic_health",
    "telemetry_health",
    "error_health",
    "failure_health",
    "performance_health",
    "retry_health",
]
WorkflowHealthStatus = Literal["ready", "guarded"]

WORKFLOW_HEALTH_PANEL_SERIALIZATION_VERSION = "workflow_health_panel.v1"
WORKFLOW_HEALTH_MONITORING_SERIALIZATION_VERSION = "workflow_health_monitoring.v1"
WORKFLOW_HEALTH_MONITORING_AUTHORITY_BOUNDARY = (
    "The V5.4 Workflow Health Monitoring surface summarizes workflow "
    "diagnostics, production telemetry, error intelligence, failure analysis, "
    "performance dashboard, and retry policy metadata as read-only workflow "
    "health monitoring only; it does not monitor live workflows, capture "
    "runtime events, execute health checks, emit alerts or telemetry, execute "
    "or control workflows, mutate workflow state or graphs, execute replays, "
    "invoke nodes, route providers or models, trigger retries or refinement, "
    "trigger escalations, invoke agents, write memory or storage, modify "
    "generated output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "workflow_diagnostics",
    "production_telemetry",
    "error_intelligence",
    "failure_analysis",
    "performance_dashboard",
    "retry_policy_plan",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_workflow_monitoring",
    "runtime_event_capture",
    "health_check_execution",
    "alert_emission",
    "telemetry_emission",
    "workflow_execution",
    "workflow_control",
    "workflow_state_mutation",
    "workflow_graph_mutation",
    "workflow_replay_execution",
    "execution_replay_execution",
    "node_handler_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "escalation_triggering",
    "agent_invocation",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class WorkflowHealthPanel(BaseModel):
    """One read-only V5.4 workflow health panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: WorkflowHealthPanelKind
    status: WorkflowHealthStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    health_signal_count: int = Field(ge=0, le=10000)
    guardrail_signal_count: int = Field(ge=0, le=3000)
    observed_runtime_event_count: None = None
    emitted_health_alert_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    workflow_health_panel_implemented: Literal[True] = True
    live_workflow_monitoring_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["workflow_health_panel.v1"] = (
        WORKFLOW_HEALTH_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"workflow_health::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.emitted_health_alert_count is not None:
            raise ValueError("emitted_health_alert_count must remain unset")
        if self.guardrail_signal_count > self.health_signal_count:
            raise ValueError("guardrail_signal_count must fit health_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class WorkflowHealthMonitoring(BaseModel):
    """Read-only V5.4 workflow health monitoring over passive metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_health_monitoring"] = "workflow_health_monitoring"
    serialization_version: Literal["workflow_health_monitoring.v1"] = (
        WORKFLOW_HEALTH_MONITORING_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_HEALTH_MONITORING_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    source_workflow_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_production_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_error_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_failure_analysis_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_performance_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_retry_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[WorkflowHealthPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    health_signal_count: int = Field(ge=0, le=12000)
    guardrail_signal_count: int = Field(ge=0, le=4000)
    observed_runtime_event_count: None = None
    emitted_health_alert_count: None = None
    workflow_health_status: WorkflowHealthStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    workflow_health_monitoring_implemented: Literal[True] = True
    live_workflow_monitoring_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _monitoring_matches_panels(self) -> Self:
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
        if self.health_signal_count != sum(
            panel.health_signal_count for panel in self.panels
        ):
            raise ValueError("health_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.emitted_health_alert_count is not None:
            raise ValueError("emitted_health_alert_count must remain unset")
        if self.workflow_health_status != _monitoring_status(self.panels):
            raise ValueError("workflow_health_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match workflow health sources")
        return self


def build_workflow_health_monitoring(
    *,
    workflow_diagnostics: WorkflowDiagnostics | None = None,
    production_telemetry: ProductionTelemetrySurface | None = None,
    error_intelligence: ErrorIntelligence | None = None,
    failure_analysis: FailureAnalysis | None = None,
    performance_dashboard: PerformanceDashboard | None = None,
    retry_policy: RetryPolicyPlan | None = None,
) -> WorkflowHealthMonitoring:
    """Build read-only workflow health monitoring without live checks."""

    workflow_source = workflow_diagnostics or build_workflow_diagnostics()
    telemetry_source = production_telemetry or build_production_telemetry()
    error_source = error_intelligence or build_error_intelligence()
    failure_source = failure_analysis or build_failure_analysis()
    performance_source = performance_dashboard or build_performance_dashboard()
    retry_source = retry_policy or plan_retry_policies()
    panels = (
        _workflow_panel(workflow_source),
        _telemetry_panel(telemetry_source),
        _error_panel(error_source),
        _failure_panel(failure_source),
        _performance_panel(performance_source),
        _retry_panel(retry_source),
    )

    return WorkflowHealthMonitoring(
        source_workflow_diagnostics_serialization_version=(
            workflow_source.serialization_version
        ),
        source_production_telemetry_serialization_version=(
            telemetry_source.serialization_version
        ),
        source_error_intelligence_serialization_version=(
            error_source.serialization_version
        ),
        source_failure_analysis_serialization_version=(
            failure_source.serialization_version
        ),
        source_performance_dashboard_serialization_version=(
            performance_source.serialization_version
        ),
        source_retry_policy_serialization_version=retry_source.serialization_version,
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        health_signal_count=sum(panel.health_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        workflow_health_status=_monitoring_status(panels),
        advisory_actions=_monitoring_actions(panels),
    )


def workflow_health_panel_by_id(
    panel_id: str,
    monitoring: WorkflowHealthMonitoring | None = None,
) -> WorkflowHealthPanel | None:
    """Return one workflow health panel without runtime monitoring."""

    source_monitoring = monitoring or build_workflow_health_monitoring()
    for panel in source_monitoring.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def workflow_health_panels_for_status(
    status: WorkflowHealthStatus,
    monitoring: WorkflowHealthMonitoring | None = None,
) -> tuple[WorkflowHealthPanel, ...]:
    """Return workflow health panels by status without live checks."""

    source_monitoring = monitoring or build_workflow_health_monitoring()
    return tuple(panel for panel in source_monitoring.panels if panel.status == status)


def _workflow_panel(diagnostics: WorkflowDiagnostics) -> WorkflowHealthPanel:
    guardrails = len(diagnostics.blocked_runtime_behaviors)
    return WorkflowHealthPanel(
        panel_id="workflow_health::diagnostic_health",
        panel_kind="diagnostic_health",
        status=_status_for_guardrails(guardrails),
        source_id="workflow_diagnostics",
        source_serialization_version=diagnostics.serialization_version,
        source_item_ids=diagnostics.panel_ids,
        health_signal_count=(
            diagnostics.diagnostic_signal_count + diagnostics.panel_count
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"diagnostic_panels:{diagnostics.panel_count}",
            f"diagnostic_signals:{diagnostics.diagnostic_signal_count}",
            f"status:{diagnostics.workflow_diagnostics_status}",
        ),
        advisory_actions=(
            "Display workflow diagnostic health without capturing runtime events.",
            "Keep graph compilation, workflow execution, replay execution, routing, and retries disabled.",
        ),
    )


def _telemetry_panel(telemetry: ProductionTelemetrySurface) -> WorkflowHealthPanel:
    guardrails = len(telemetry.blocked_runtime_behaviors)
    return WorkflowHealthPanel(
        panel_id="workflow_health::telemetry_health",
        panel_kind="telemetry_health",
        status=_status_for_guardrails(guardrails),
        source_id="production_telemetry",
        source_serialization_version=telemetry.serialization_version,
        source_item_ids=telemetry.channel_ids,
        health_signal_count=(
            telemetry.telemetry_signal_count
            + telemetry.guarded_signal_count
            + telemetry.channel_count
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"telemetry_channels:{telemetry.channel_count}",
            f"guarded_signals:{telemetry.guarded_signal_count}",
            f"status:{telemetry.production_telemetry_status}",
        ),
        advisory_actions=(
            "Display telemetry health metadata without emitting telemetry.",
            "Keep trace capture, event export, alerts, HITL requests, and storage writes disabled.",
        ),
    )


def _error_panel(intelligence: ErrorIntelligence) -> WorkflowHealthPanel:
    guardrails = len(intelligence.blocked_runtime_behaviors)
    return WorkflowHealthPanel(
        panel_id="workflow_health::error_health",
        panel_kind="error_health",
        status=_status_for_guardrails(guardrails),
        source_id="error_intelligence",
        source_serialization_version=intelligence.serialization_version,
        source_item_ids=intelligence.panel_ids,
        health_signal_count=intelligence.error_signal_count + intelligence.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"error_panels:{intelligence.panel_count}",
            f"error_signals:{intelligence.error_signal_count}",
            f"status:{intelligence.error_intelligence_status}",
        ),
        advisory_actions=(
            "Display error intelligence health without classifying live errors.",
            "Keep exception interception, remediation, alerts, workflow control, and routing disabled.",
        ),
    )


def _failure_panel(analysis: FailureAnalysis) -> WorkflowHealthPanel:
    guardrails = len(analysis.blocked_runtime_behaviors)
    return WorkflowHealthPanel(
        panel_id="workflow_health::failure_health",
        panel_kind="failure_health",
        status=_status_for_guardrails(guardrails),
        source_id="failure_analysis",
        source_serialization_version=analysis.serialization_version,
        source_item_ids=analysis.panel_ids,
        health_signal_count=analysis.failure_signal_count + analysis.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"failure_panels:{analysis.panel_count}",
            f"failure_signals:{analysis.failure_signal_count}",
            f"status:{analysis.failure_analysis_status}",
        ),
        advisory_actions=(
            "Display failure analysis health without handling or repairing failures.",
            "Keep terminal routing, retries, replays, telemetry, storage, and output mutation disabled.",
        ),
    )


def _performance_panel(dashboard: PerformanceDashboard) -> WorkflowHealthPanel:
    guardrails = len(dashboard.blocked_runtime_behaviors)
    return WorkflowHealthPanel(
        panel_id="workflow_health::performance_health",
        panel_kind="performance_health",
        status=_status_for_guardrails(guardrails),
        source_id="performance_dashboard",
        source_serialization_version=dashboard.serialization_version,
        source_item_ids=dashboard.panel_ids,
        health_signal_count=dashboard.performance_signal_count + dashboard.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"performance_panels:{dashboard.panel_count}",
            f"performance_signals:{dashboard.performance_signal_count}",
            f"pressure:{dashboard.dashboard_pressure}",
        ),
        advisory_actions=(
            "Display performance health metadata without runtime measurement.",
            "Keep benchmarks, profiling, resource allocation, workflow control, and retries disabled.",
        ),
    )


def _retry_panel(plan: RetryPolicyPlan) -> WorkflowHealthPanel:
    guardrails = len(plan.blocked_runtime_behaviors)
    health_signal_count = (
        plan.candidate_count
        + plan.guardrail_candidate_count
        + plan.review_only_candidate_count
        + int(plan.failure_path_reachable)
        + int(plan.bounded_retry_cycle_detected)
        + guardrails
    )
    return WorkflowHealthPanel(
        panel_id="workflow_health::retry_health",
        panel_kind="retry_health",
        status=_status_for_guardrails(guardrails),
        source_id="retry_policy_plan",
        source_serialization_version=plan.serialization_version,
        source_item_ids=plan.candidate_ids,
        health_signal_count=health_signal_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"retry_candidates:{plan.candidate_count}",
            f"retry_pressure:{plan.retry_policy_pressure}",
            f"max_retry_attempts:{plan.max_retry_attempts}",
        ),
        advisory_actions=(
            "Display retry health metadata without triggering retry or refinement.",
            "Keep workflow graph mutation, workflow execution, node invocation, routing, and storage disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[WorkflowHealthPanel, ...],
    status: WorkflowHealthStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> WorkflowHealthStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _monitoring_status(
    panels: tuple[WorkflowHealthPanel, ...],
) -> WorkflowHealthStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _monitoring_actions(
    panels: tuple[WorkflowHealthPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow health panels as read-only observability metadata.",
        "Preserve live monitoring, runtime event capture, health checks, alerts, "
        "workflow control, routing, escalation, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded workflow health panels detached from runtime monitoring."
        )
    return tuple(actions)
