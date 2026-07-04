"""V5.4 advisory system health monitoring metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_diagnostics import AgentDiagnostics, build_agent_diagnostics
from creative_coding_assistant.orchestration.cost_dashboard import CostDashboard, build_cost_dashboard
from creative_coding_assistant.orchestration.error_intelligence import ErrorIntelligence, build_error_intelligence
from creative_coding_assistant.orchestration.performance_dashboard import (
    PerformanceDashboard,
    build_performance_dashboard,
)
from creative_coding_assistant.orchestration.production_telemetry import (
    ProductionTelemetrySurface,
    build_production_telemetry,
)
from creative_coding_assistant.orchestration.quality_dashboard import QualityDashboard, build_quality_dashboard
from creative_coding_assistant.orchestration.token_dashboard import TokenDashboard, build_token_dashboard
from creative_coding_assistant.orchestration.workflow_health_monitoring import (
    WorkflowHealthMonitoring,
    build_workflow_health_monitoring,
)

SystemHealthPanelKind = Literal[
    "workflow_system_health",
    "telemetry_system_health",
    "token_system_health",
    "cost_system_health",
    "quality_system_health",
    "performance_system_health",
    "error_system_health",
    "agent_system_health",
]
SystemHealthStatus = Literal["ready", "guarded"]

SYSTEM_HEALTH_PANEL_SERIALIZATION_VERSION = "system_health_panel.v1"
SYSTEM_HEALTH_MONITORING_SERIALIZATION_VERSION = "system_health_monitoring.v1"
SYSTEM_HEALTH_MONITORING_AUTHORITY_BOUNDARY = (
    "The V5.4 System Health Monitoring surface summarizes workflow health, "
    "production telemetry, token dashboard, cost dashboard, quality dashboard, "
    "performance dashboard, error intelligence, and agent diagnostics metadata "
    "as read-only system health monitoring only; it does not monitor live "
    "systems, collect runtime metrics, execute health checks, emit alerts or "
    "telemetry, allocate resources, enforce capacity or budgets, evaluate "
    "quality, execute or control workflows, route providers or models, invoke "
    "agents, trigger escalations, trigger retries or refinement, write memory "
    "or storage, modify generated output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "workflow_health_monitoring",
    "production_telemetry",
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
    "error_intelligence",
    "agent_diagnostics",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_system_monitoring",
    "runtime_metric_collection",
    "health_check_execution",
    "alert_emission",
    "telemetry_emission",
    "resource_allocation",
    "capacity_enforcement",
    "budget_enforcement",
    "quality_evaluation",
    "workflow_execution",
    "workflow_control",
    "provider_or_model_routing",
    "agent_invocation",
    "escalation_triggering",
    "retry_or_refinement_triggering",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class SystemHealthPanel(BaseModel):
    """One read-only V5.4 system health panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: SystemHealthPanelKind
    status: SystemHealthStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    system_signal_count: int = Field(ge=0, le=20000)
    guardrail_signal_count: int = Field(ge=0, le=5000)
    observed_system_event_count: None = None
    emitted_system_alert_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    system_health_panel_implemented: Literal[True] = True
    live_system_monitoring_implemented: Literal[False] = False
    runtime_metric_collection_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["system_health_panel.v1"] = (
        SYSTEM_HEALTH_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"system_health::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_system_event_count is not None:
            raise ValueError("observed_system_event_count must remain unset")
        if self.emitted_system_alert_count is not None:
            raise ValueError("emitted_system_alert_count must remain unset")
        if self.guardrail_signal_count > self.system_signal_count:
            raise ValueError("guardrail_signal_count must fit system_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class SystemHealthMonitoring(BaseModel):
    """Read-only V5.4 system health monitoring over passive metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["system_health_monitoring"] = "system_health_monitoring"
    serialization_version: Literal["system_health_monitoring.v1"] = (
        SYSTEM_HEALTH_MONITORING_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SYSTEM_HEALTH_MONITORING_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    source_workflow_health_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_production_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_token_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_cost_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_quality_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_performance_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_error_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=8, max_length=8)
    panels: tuple[SystemHealthPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    system_signal_count: int = Field(ge=0, le=24000)
    guardrail_signal_count: int = Field(ge=0, le=8000)
    observed_system_event_count: None = None
    emitted_system_alert_count: None = None
    system_health_status: SystemHealthStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    system_health_monitoring_implemented: Literal[True] = True
    live_system_monitoring_implemented: Literal[False] = False
    runtime_metric_collection_implemented: Literal[False] = False
    health_check_execution_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    quality_evaluation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
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
        if self.system_signal_count != sum(
            panel.system_signal_count for panel in self.panels
        ):
            raise ValueError("system_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_system_event_count is not None:
            raise ValueError("observed_system_event_count must remain unset")
        if self.emitted_system_alert_count is not None:
            raise ValueError("emitted_system_alert_count must remain unset")
        if self.system_health_status != _monitoring_status(self.panels):
            raise ValueError("system_health_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match system health sources")
        return self


def build_system_health_monitoring(
    *,
    workflow_health: WorkflowHealthMonitoring | None = None,
    production_telemetry: ProductionTelemetrySurface | None = None,
    token_dashboard: TokenDashboard | None = None,
    cost_dashboard: CostDashboard | None = None,
    quality_dashboard: QualityDashboard | None = None,
    performance_dashboard: PerformanceDashboard | None = None,
    error_intelligence: ErrorIntelligence | None = None,
    agent_diagnostics: AgentDiagnostics | None = None,
) -> SystemHealthMonitoring:
    """Build read-only system health monitoring without live metric collection."""

    workflow_source = workflow_health or build_workflow_health_monitoring()
    telemetry_source = production_telemetry or build_production_telemetry()
    token_source = token_dashboard or build_token_dashboard()
    cost_source = cost_dashboard or build_cost_dashboard()
    quality_source = quality_dashboard or build_quality_dashboard()
    performance_source = performance_dashboard or build_performance_dashboard()
    error_source = error_intelligence or build_error_intelligence()
    agent_source = agent_diagnostics or build_agent_diagnostics()
    panels = (
        _workflow_panel(workflow_source),
        _telemetry_panel(telemetry_source),
        _token_panel(token_source),
        _cost_panel(cost_source),
        _quality_panel(quality_source),
        _performance_panel(performance_source),
        _error_panel(error_source),
        _agent_panel(agent_source),
    )

    return SystemHealthMonitoring(
        source_workflow_health_serialization_version=(
            workflow_source.serialization_version
        ),
        source_production_telemetry_serialization_version=(
            telemetry_source.serialization_version
        ),
        source_token_dashboard_serialization_version=token_source.serialization_version,
        source_cost_dashboard_serialization_version=cost_source.serialization_version,
        source_quality_dashboard_serialization_version=(
            quality_source.serialization_version
        ),
        source_performance_dashboard_serialization_version=(
            performance_source.serialization_version
        ),
        source_error_intelligence_serialization_version=(
            error_source.serialization_version
        ),
        source_agent_diagnostics_serialization_version=(
            agent_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        system_signal_count=sum(panel.system_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        system_health_status=_monitoring_status(panels),
        advisory_actions=_monitoring_actions(panels),
    )


def system_health_panel_by_id(
    panel_id: str,
    monitoring: SystemHealthMonitoring | None = None,
) -> SystemHealthPanel | None:
    """Return one system health panel without runtime monitoring."""

    source_monitoring = monitoring or build_system_health_monitoring()
    for panel in source_monitoring.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def system_health_panels_for_status(
    status: SystemHealthStatus,
    monitoring: SystemHealthMonitoring | None = None,
) -> tuple[SystemHealthPanel, ...]:
    """Return system health panels by status without live checks."""

    source_monitoring = monitoring or build_system_health_monitoring()
    return tuple(panel for panel in source_monitoring.panels if panel.status == status)


def _workflow_panel(source: WorkflowHealthMonitoring) -> SystemHealthPanel:
    return _panel(
        "workflow_system_health",
        "workflow_health_monitoring",
        source.serialization_version,
        source.panel_ids,
        source.health_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"workflow_health_panels:{source.panel_count}",
            f"workflow_health_signals:{source.health_signal_count}",
            f"status:{source.workflow_health_status}",
        ),
        "Display workflow system health without monitoring live workflows.",
    )


def _telemetry_panel(source: ProductionTelemetrySurface) -> SystemHealthPanel:
    return _panel(
        "telemetry_system_health",
        "production_telemetry",
        source.serialization_version,
        source.channel_ids,
        source.telemetry_signal_count
        + source.guarded_signal_count
        + source.channel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"telemetry_channels:{source.channel_count}",
            f"guarded_signals:{source.guarded_signal_count}",
            f"status:{source.production_telemetry_status}",
        ),
        "Display telemetry system health without emitting telemetry or alerts.",
    )


def _token_panel(source: TokenDashboard) -> SystemHealthPanel:
    return _panel(
        "token_system_health",
        "token_dashboard",
        source.serialization_version,
        source.panel_ids,
        source.panel_count
        + len(source.guarded_panel_ids)
        + len(source.review_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"token_panels:{source.panel_count}",
            f"planned_tokens:{source.planned_token_total}",
            f"pressure:{source.dashboard_pressure}",
        ),
        "Display token system health without metering usage or enforcing budgets.",
    )


def _cost_panel(source: CostDashboard) -> SystemHealthPanel:
    return _panel(
        "cost_system_health",
        "cost_dashboard",
        source.serialization_version,
        source.panel_ids,
        source.cost_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"cost_panels:{source.panel_count}",
            f"cost_signals:{source.cost_signal_count}",
            f"pressure:{source.dashboard_pressure}",
        ),
        "Display cost system health without metering usage or enforcing spend.",
    )


def _quality_panel(source: QualityDashboard) -> SystemHealthPanel:
    return _panel(
        "quality_system_health",
        "quality_dashboard",
        source.serialization_version,
        source.panel_ids,
        source.quality_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"quality_panels:{source.panel_count}",
            f"quality_signals:{source.quality_signal_count}",
            f"pressure:{source.dashboard_pressure}",
        ),
        "Display quality system health without evaluating generated output.",
    )


def _performance_panel(source: PerformanceDashboard) -> SystemHealthPanel:
    return _panel(
        "performance_system_health",
        "performance_dashboard",
        source.serialization_version,
        source.panel_ids,
        source.performance_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"performance_panels:{source.panel_count}",
            f"performance_signals:{source.performance_signal_count}",
            f"pressure:{source.dashboard_pressure}",
        ),
        "Display performance system health without runtime measurement.",
    )


def _error_panel(source: ErrorIntelligence) -> SystemHealthPanel:
    return _panel(
        "error_system_health",
        "error_intelligence",
        source.serialization_version,
        source.panel_ids,
        source.error_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"error_panels:{source.panel_count}",
            f"error_signals:{source.error_signal_count}",
            f"status:{source.error_intelligence_status}",
        ),
        "Display error system health without live error classification.",
    )


def _agent_panel(source: AgentDiagnostics) -> SystemHealthPanel:
    return _panel(
        "agent_system_health",
        "agent_diagnostics",
        source.serialization_version,
        source.panel_ids,
        source.agent_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"agent_panels:{source.panel_count}",
            f"agent_signals:{source.agent_signal_count}",
            f"status:{source.agent_diagnostics_status}",
        ),
        "Display agent system health without invoking agents.",
    )


def _panel(
    panel_kind: SystemHealthPanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> SystemHealthPanel:
    return SystemHealthPanel(
        panel_id=f"system_health::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        system_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep live monitoring, health checks, alerts, telemetry, routing, storage, and output mutation disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[SystemHealthPanel, ...],
    status: SystemHealthStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> SystemHealthStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _monitoring_status(
    panels: tuple[SystemHealthPanel, ...],
) -> SystemHealthStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _monitoring_actions(
    panels: tuple[SystemHealthPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose system health panels as read-only observability metadata.",
        "Preserve live monitoring, metric collection, health checks, alerts, "
        "workflow, routing, agent, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded system health panels detached from runtime monitoring."
        )
    return tuple(actions)
