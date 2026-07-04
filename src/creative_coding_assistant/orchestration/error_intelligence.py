"""V5.4 advisory error intelligence metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .escalation_diagnostics import EscalationDiagnostics, build_escalation_diagnostics
from .failure_analysis import FailureAnalysis, build_failure_analysis
from .final_v4_hardening import (
    LangGraphErrorPathAuditRegistry,
    langgraph_error_path_audit_registry,
)
from .production_telemetry import ProductionTelemetrySurface, build_production_telemetry
from .routing_diagnostics import RoutingDiagnostics, build_routing_diagnostics
from .workflow_diagnostics import WorkflowDiagnostics, build_workflow_diagnostics

ErrorIntelligencePanelKind = Literal[
    "error_path_taxonomy",
    "failure_pattern_summary",
    "workflow_error_context",
    "telemetry_error_boundary",
    "routing_error_boundary",
    "escalation_error_boundary",
]
ErrorIntelligenceStatus = Literal["ready", "guarded"]

ERROR_INTELLIGENCE_PANEL_SERIALIZATION_VERSION = "error_intelligence_panel.v1"
ERROR_INTELLIGENCE_SERIALIZATION_VERSION = "error_intelligence.v1"
ERROR_INTELLIGENCE_AUTHORITY_BOUNDARY = (
    "The V5.4 Error Intelligence surface converts existing failure analysis, "
    "LangGraph error-path audit, workflow diagnostics, production telemetry, "
    "routing diagnostics, and escalation diagnostics metadata into read-only "
    "error intelligence only; it does not capture runtime errors, classify "
    "live errors, intercept exceptions, route terminal failures, remediate "
    "errors, trigger retries or refinement, emit alerts or telemetry, request "
    "human review, execute or control workflows, mutate workflow graphs, route "
    "providers or models, execute providers, trigger escalations, invoke "
    "agents, write memory or storage, modify generated output, or apply "
    "Runtime Evolution."
)

_SOURCE_SURFACES = (
    "failure_analysis",
    "langgraph_error_path_audit_registry",
    "workflow_diagnostics",
    "production_telemetry",
    "routing_diagnostics",
    "escalation_diagnostics",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_error_capture",
    "live_error_classification",
    "exception_interception",
    "terminal_failure_routing",
    "automated_remediation",
    "retry_or_refinement_triggering",
    "alert_emission",
    "telemetry_emission",
    "human_review_request",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "provider_execution",
    "escalation_triggering",
    "agent_invocation",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)
_LANGGRAPH_ERROR_GUARDRAILS = (
    "new_langgraph_nodes_implemented_false",
    "active_multi_agent_execution_implemented_false",
    "provider_model_routing_change_implemented_false",
    "workflow_behavior_change_implemented_false",
    "passive_registry_runtime_activation_implemented_false",
    "generated_output_mutation_implemented_false",
)


class ErrorIntelligencePanel(BaseModel):
    """One read-only V5.4 error intelligence panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: ErrorIntelligencePanelKind
    status: ErrorIntelligenceStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    error_signal_count: int = Field(ge=0, le=6000)
    guardrail_signal_count: int = Field(ge=0, le=2000)
    observed_error_count: None = None
    remediated_error_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    error_intelligence_panel_implemented: Literal[True] = True
    runtime_error_capture_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    exception_interception_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["error_intelligence_panel.v1"] = (
        ERROR_INTELLIGENCE_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"error_intelligence::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_error_count is not None:
            raise ValueError("observed_error_count must remain unset")
        if self.remediated_error_count is not None:
            raise ValueError("remediated_error_count must remain unset")
        if self.guardrail_signal_count > self.error_signal_count:
            raise ValueError("guardrail_signal_count must fit error_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class ErrorIntelligence(BaseModel):
    """Read-only V5.4 error intelligence over passive observability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["error_intelligence"] = "error_intelligence"
    serialization_version: Literal["error_intelligence.v1"] = (
        ERROR_INTELLIGENCE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ERROR_INTELLIGENCE_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    source_failure_analysis_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_langgraph_error_path_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_production_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[ErrorIntelligencePanel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    error_signal_count: int = Field(ge=0, le=10000)
    guardrail_signal_count: int = Field(ge=0, le=3000)
    observed_error_count: None = None
    remediated_error_count: None = None
    error_intelligence_status: ErrorIntelligenceStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    error_intelligence_implemented: Literal[True] = True
    runtime_error_capture_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    exception_interception_implemented: Literal[False] = False
    terminal_failure_routing_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _intelligence_matches_panels(self) -> Self:
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
        if self.error_signal_count != sum(
            panel.error_signal_count for panel in self.panels
        ):
            raise ValueError("error_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_error_count is not None:
            raise ValueError("observed_error_count must remain unset")
        if self.remediated_error_count is not None:
            raise ValueError("remediated_error_count must remain unset")
        if self.error_intelligence_status != _intelligence_status(self.panels):
            raise ValueError("error_intelligence_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match error intelligence sources")
        return self


def build_error_intelligence(
    *,
    failure_analysis: FailureAnalysis | None = None,
    langgraph_error_paths: LangGraphErrorPathAuditRegistry | None = None,
    workflow_diagnostics: WorkflowDiagnostics | None = None,
    production_telemetry: ProductionTelemetrySurface | None = None,
    routing_diagnostics: RoutingDiagnostics | None = None,
    escalation_diagnostics: EscalationDiagnostics | None = None,
) -> ErrorIntelligence:
    """Build read-only error intelligence without live error capture."""

    failure_source = failure_analysis or build_failure_analysis()
    langgraph_source = langgraph_error_paths or langgraph_error_path_audit_registry()
    workflow_source = workflow_diagnostics or build_workflow_diagnostics()
    telemetry_source = production_telemetry or build_production_telemetry()
    routing_source = routing_diagnostics or build_routing_diagnostics()
    escalation_source = escalation_diagnostics or build_escalation_diagnostics()
    panels = (
        _error_path_taxonomy_panel(langgraph_source),
        _failure_pattern_panel(failure_source),
        _workflow_error_panel(workflow_source),
        _telemetry_error_panel(telemetry_source),
        _routing_error_panel(routing_source),
        _escalation_error_panel(escalation_source),
    )

    return ErrorIntelligence(
        source_failure_analysis_serialization_version=(
            failure_source.serialization_version
        ),
        source_langgraph_error_path_serialization_version=(
            langgraph_source.serialization_version
        ),
        source_workflow_diagnostics_serialization_version=(
            workflow_source.serialization_version
        ),
        source_production_telemetry_serialization_version=(
            telemetry_source.serialization_version
        ),
        source_routing_diagnostics_serialization_version=(
            routing_source.serialization_version
        ),
        source_escalation_diagnostics_serialization_version=(
            escalation_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        error_signal_count=sum(panel.error_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        error_intelligence_status=_intelligence_status(panels),
        advisory_actions=_intelligence_actions(panels),
    )


def error_intelligence_panel_by_id(
    panel_id: str,
    intelligence: ErrorIntelligence | None = None,
) -> ErrorIntelligencePanel | None:
    """Return one error intelligence panel without runtime classification."""

    source_intelligence = intelligence or build_error_intelligence()
    for panel in source_intelligence.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def error_intelligence_panels_for_status(
    status: ErrorIntelligenceStatus,
    intelligence: ErrorIntelligence | None = None,
) -> tuple[ErrorIntelligencePanel, ...]:
    """Return error intelligence panels by status without live error analysis."""

    source_intelligence = intelligence or build_error_intelligence()
    return tuple(
        panel for panel in source_intelligence.panels if panel.status == status
    )


def _error_path_taxonomy_panel(
    registry: LangGraphErrorPathAuditRegistry,
) -> ErrorIntelligencePanel:
    guardrails = len(_LANGGRAPH_ERROR_GUARDRAILS)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::error_path_taxonomy",
        panel_kind="error_path_taxonomy",
        status=_status_for_guardrails(guardrails),
        source_id="langgraph_error_path_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.surface_ids,
        error_signal_count=(
            registry.record_count
            + len(registry.surface_ids)
            + len(registry.failure_invariants)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"error_surfaces:{len(registry.surface_ids)}",
            f"runtime_nodes:{len(registry.runtime_node_ids)}",
            f"terminal_failure_node:{registry.terminal_failure_node}",
        ),
        advisory_actions=(
            "Display static error-path taxonomy without classifying live errors.",
            "Keep exception interception, terminal routing, graph mutation, and remediation disabled.",
        ),
    )


def _failure_pattern_panel(analysis: FailureAnalysis) -> ErrorIntelligencePanel:
    guardrails = len(analysis.blocked_runtime_behaviors)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::failure_pattern_summary",
        panel_kind="failure_pattern_summary",
        status=_status_for_guardrails(guardrails),
        source_id="failure_analysis",
        source_serialization_version=analysis.serialization_version,
        source_item_ids=analysis.panel_ids,
        error_signal_count=analysis.failure_signal_count + analysis.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"failure_panels:{analysis.panel_count}",
            f"failure_signals:{analysis.failure_signal_count}",
            f"status:{analysis.failure_analysis_status}",
        ),
        advisory_actions=(
            "Display failure pattern metadata without observing or repairing failures.",
            "Keep retries, replays, provider routing, telemetry, storage, and output mutation disabled.",
        ),
    )


def _workflow_error_panel(diagnostics: WorkflowDiagnostics) -> ErrorIntelligencePanel:
    guardrails = len(diagnostics.blocked_runtime_behaviors)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::workflow_error_context",
        panel_kind="workflow_error_context",
        status=_status_for_guardrails(guardrails),
        source_id="workflow_diagnostics",
        source_serialization_version=diagnostics.serialization_version,
        source_item_ids=diagnostics.panel_ids,
        error_signal_count=diagnostics.diagnostic_signal_count
        + diagnostics.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"diagnostic_panels:{diagnostics.panel_count}",
            f"diagnostic_signals:{diagnostics.diagnostic_signal_count}",
            f"status:{diagnostics.workflow_diagnostics_status}",
        ),
        advisory_actions=(
            "Display workflow error context without compiling graphs or executing workflows.",
            "Keep node invocation, replay execution, telemetry emission, routing, and retries disabled.",
        ),
    )


def _telemetry_error_panel(
    telemetry: ProductionTelemetrySurface,
) -> ErrorIntelligencePanel:
    guardrails = len(telemetry.blocked_runtime_behaviors)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::telemetry_error_boundary",
        panel_kind="telemetry_error_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="production_telemetry",
        source_serialization_version=telemetry.serialization_version,
        source_item_ids=telemetry.channel_ids,
        error_signal_count=(
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
            "Display telemetry error boundaries without emitting telemetry or alerts.",
            "Keep live metrics, trace capture, event export, HITL requests, and storage writes disabled.",
        ),
    )


def _routing_error_panel(diagnostics: RoutingDiagnostics) -> ErrorIntelligencePanel:
    guardrails = len(diagnostics.blocked_runtime_behaviors)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::routing_error_boundary",
        panel_kind="routing_error_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="routing_diagnostics",
        source_serialization_version=diagnostics.serialization_version,
        source_item_ids=diagnostics.panel_ids,
        error_signal_count=diagnostics.routing_signal_count + diagnostics.panel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"routing_panels:{diagnostics.panel_count}",
            f"routing_signals:{diagnostics.routing_signal_count}",
            f"status:{diagnostics.routing_diagnostics_status}",
        ),
        advisory_actions=(
            "Display routing error boundaries without applying route decisions.",
            "Keep provider execution, model switching, API-key assumptions, HITL requests, and retries disabled.",
        ),
    )


def _escalation_error_panel(
    diagnostics: EscalationDiagnostics,
) -> ErrorIntelligencePanel:
    guardrails = len(diagnostics.blocked_runtime_behaviors)
    return ErrorIntelligencePanel(
        panel_id="error_intelligence::escalation_error_boundary",
        panel_kind="escalation_error_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="escalation_diagnostics",
        source_serialization_version=diagnostics.serialization_version,
        source_item_ids=diagnostics.panel_ids,
        error_signal_count=(
            diagnostics.escalation_signal_count + diagnostics.panel_count
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"escalation_panels:{diagnostics.panel_count}",
            f"escalation_signals:{diagnostics.escalation_signal_count}",
            f"status:{diagnostics.escalation_diagnostics_status}",
        ),
        advisory_actions=(
            "Display escalation error boundaries without triggering escalation.",
            "Keep policy evaluation, HITL triggering, trace emission, agent invocation, and routing disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[ErrorIntelligencePanel, ...],
    status: ErrorIntelligenceStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> ErrorIntelligenceStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _intelligence_status(
    panels: tuple[ErrorIntelligencePanel, ...],
) -> ErrorIntelligenceStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _intelligence_actions(
    panels: tuple[ErrorIntelligencePanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose error intelligence panels as read-only observability metadata.",
        "Preserve error capture, live classification, remediation, alerting, "
        "workflow, routing, escalation, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded error intelligence panels detached from runtime error handling."
        )
    return tuple(actions)
