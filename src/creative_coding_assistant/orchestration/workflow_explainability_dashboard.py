"""V5.4 advisory workflow explainability dashboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_explainability_audit import (
    AgentExplainabilityAuditRegistry,
    agent_explainability_audit_registry,
)
from .error_intelligence import ErrorIntelligence, build_error_intelligence
from .hybrid_agentic_workflow import (
    DecisionProvenanceRegistry,
    decision_provenance_registry,
)
from .routing_explainability import (
    RoutingExplainabilityPlan,
    explain_routing_decision,
)
from .runtime_timeline import RuntimeTimeline, build_runtime_timeline
from .workflow_diagnostics import WorkflowDiagnostics, build_workflow_diagnostics

WorkflowExplainabilityPanelKind = Literal[
    "workflow_reasoning_context",
    "routing_explanation_context",
    "agent_explainability_audit",
    "decision_provenance_context",
    "runtime_timeline_context",
    "error_explainability_context",
]
WorkflowExplainabilityStatus = Literal["ready", "guarded"]

WORKFLOW_EXPLAINABILITY_PANEL_SERIALIZATION_VERSION = "workflow_explainability_panel.v1"
WORKFLOW_EXPLAINABILITY_DASHBOARD_SERIALIZATION_VERSION = (
    "workflow_explainability_dashboard.v1"
)
WORKFLOW_EXPLAINABILITY_DASHBOARD_AUTHORITY_BOUNDARY = (
    "The V5.4 Workflow Explainability Dashboard summarizes workflow "
    "diagnostics, routing explainability, agent explainability audit, decision "
    "provenance, runtime timeline, and error intelligence metadata as "
    "read-only workflow explainability observability only; it does not "
    "generate live explanations, generate reasoning, record provenance, log "
    "decisions, capture or emit traces, capture runtime events, reconstruct "
    "timelines, apply routing, execute providers, invoke agents, execute or "
    "control workflows, mutate workflow state, classify live errors, remediate "
    "errors, request human review, trigger retries or refinement, write memory "
    "or storage, modify generated output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "workflow_diagnostics",
    "routing_explainability",
    "agent_explainability_audit_registry",
    "decision_provenance_registry",
    "runtime_timeline",
    "error_intelligence",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "explanation_generation",
    "live_reasoning_generation",
    "decision_provenance_recording",
    "decision_logging",
    "trace_capture",
    "trace_emission",
    "runtime_event_capture",
    "timeline_reconstruction",
    "routing_application",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_execution",
    "workflow_control",
    "workflow_state_mutation",
    "live_error_classification",
    "automated_remediation",
    "human_review_request",
    "retry_or_refinement_triggering",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class WorkflowExplainabilityPanel(BaseModel):
    """One read-only V5.4 workflow explainability dashboard panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=220)
    panel_kind: WorkflowExplainabilityPanelKind
    status: WorkflowExplainabilityStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=260)
    explainability_signal_count: int = Field(ge=0, le=80000)
    guardrail_signal_count: int = Field(ge=0, le=30000)
    generated_explanation_count: None = None
    recorded_provenance_count: None = None
    captured_trace_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=26,
    )
    workflow_explainability_panel_implemented: Literal[True] = True
    explanation_generation_implemented: Literal[False] = False
    live_reasoning_generation_implemented: Literal[False] = False
    decision_provenance_recording_implemented: Literal[False] = False
    decision_logging_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["workflow_explainability_panel.v1"] = (
        WORKFLOW_EXPLAINABILITY_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"workflow_explainability::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.generated_explanation_count is not None:
            raise ValueError("generated_explanation_count must remain unset")
        if self.recorded_provenance_count is not None:
            raise ValueError("recorded_provenance_count must remain unset")
        if self.captured_trace_count is not None:
            raise ValueError("captured_trace_count must remain unset")
        if self.guardrail_signal_count > self.explainability_signal_count:
            raise ValueError(
                "guardrail_signal_count must fit explainability_signal_count"
            )
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class WorkflowExplainabilityDashboard(BaseModel):
    """Read-only V5.4 workflow explainability dashboard over passive metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_explainability_dashboard"] = (
        "workflow_explainability_dashboard"
    )
    serialization_version: Literal["workflow_explainability_dashboard.v1"] = (
        WORKFLOW_EXPLAINABILITY_DASHBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_EXPLAINABILITY_DASHBOARD_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    source_workflow_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_explainability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_explainability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_decision_provenance_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_runtime_timeline_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_error_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[WorkflowExplainabilityPanel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    explainability_signal_count: int = Field(ge=0, le=160000)
    guardrail_signal_count: int = Field(ge=0, le=60000)
    generated_explanation_count: None = None
    recorded_provenance_count: None = None
    captured_trace_count: None = None
    workflow_explainability_status: WorkflowExplainabilityStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=26,
    )
    workflow_explainability_dashboard_implemented: Literal[True] = True
    explanation_generation_implemented: Literal[False] = False
    live_reasoning_generation_implemented: Literal[False] = False
    decision_provenance_recording_implemented: Literal[False] = False
    decision_logging_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    timeline_reconstruction_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    live_error_classification_implemented: Literal[False] = False
    automated_remediation_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _dashboard_matches_panels(self) -> Self:
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
        if self.explainability_signal_count != sum(
            panel.explainability_signal_count for panel in self.panels
        ):
            raise ValueError("explainability_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.generated_explanation_count is not None:
            raise ValueError("generated_explanation_count must remain unset")
        if self.recorded_provenance_count is not None:
            raise ValueError("recorded_provenance_count must remain unset")
        if self.captured_trace_count is not None:
            raise ValueError("captured_trace_count must remain unset")
        if self.workflow_explainability_status != _dashboard_status(self.panels):
            raise ValueError("workflow_explainability_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError(
                "source_surfaces must match workflow explainability sources"
            )
        return self


def build_workflow_explainability_dashboard(
    *,
    workflow_diagnostics: WorkflowDiagnostics | None = None,
    routing_explainability: RoutingExplainabilityPlan | None = None,
    agent_explainability: AgentExplainabilityAuditRegistry | None = None,
    decision_provenance: DecisionProvenanceRegistry | None = None,
    runtime_timeline: RuntimeTimeline | None = None,
    error_intelligence: ErrorIntelligence | None = None,
) -> WorkflowExplainabilityDashboard:
    """Build read-only workflow explainability without generating explanations."""

    workflow_source = workflow_diagnostics or build_workflow_diagnostics()
    routing_source = routing_explainability or explain_routing_decision()
    agent_source = agent_explainability or agent_explainability_audit_registry()
    provenance_source = decision_provenance or decision_provenance_registry()
    timeline_source = runtime_timeline or build_runtime_timeline(
        workflow_diagnostics=workflow_source,
    )
    error_source = error_intelligence or build_error_intelligence(
        workflow_diagnostics=workflow_source,
    )
    panels = (
        _workflow_panel(workflow_source),
        _routing_panel(routing_source),
        _agent_panel(agent_source),
        _provenance_panel(provenance_source),
        _timeline_panel(timeline_source),
        _error_panel(error_source),
    )

    return WorkflowExplainabilityDashboard(
        source_workflow_diagnostics_serialization_version=(
            workflow_source.serialization_version
        ),
        source_routing_explainability_serialization_version=(
            routing_source.serialization_version
        ),
        source_agent_explainability_serialization_version=(
            agent_source.serialization_version
        ),
        source_decision_provenance_serialization_version=(
            provenance_source.serialization_version
        ),
        source_runtime_timeline_serialization_version=(
            timeline_source.serialization_version
        ),
        source_error_intelligence_serialization_version=(
            error_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        explainability_signal_count=sum(
            panel.explainability_signal_count for panel in panels
        ),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        workflow_explainability_status=_dashboard_status(panels),
        advisory_actions=_dashboard_actions(panels),
    )


def workflow_explainability_panel_by_id(
    panel_id: str,
    dashboard: WorkflowExplainabilityDashboard | None = None,
) -> WorkflowExplainabilityPanel | None:
    """Return one workflow explainability panel without generating explanations."""

    source_dashboard = dashboard or build_workflow_explainability_dashboard()
    for panel in source_dashboard.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def workflow_explainability_panels_for_status(
    status: WorkflowExplainabilityStatus,
    dashboard: WorkflowExplainabilityDashboard | None = None,
) -> tuple[WorkflowExplainabilityPanel, ...]:
    """Return workflow explainability panels by status without trace capture."""

    source_dashboard = dashboard or build_workflow_explainability_dashboard()
    return tuple(panel for panel in source_dashboard.panels if panel.status == status)


def _workflow_panel(source: WorkflowDiagnostics) -> WorkflowExplainabilityPanel:
    return _panel(
        "workflow_reasoning_context",
        "workflow_diagnostics",
        source.serialization_version,
        source.panel_ids,
        source.diagnostic_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"workflow_panels:{source.panel_count}",
            f"diagnostic_signals:{source.diagnostic_signal_count}",
            f"status:{source.workflow_diagnostics_status}",
        ),
        "Display workflow diagnostic rationale without executing workflows.",
    )


def _routing_panel(source: RoutingExplainabilityPlan) -> WorkflowExplainabilityPanel:
    return _panel(
        "routing_explanation_context",
        "routing_explainability",
        source.serialization_version,
        source.explanation_ids,
        source.explanation_count
        + source.source_surface_count
        + len(source.advisory_actions),
        len(source.blocked_runtime_behaviors),
        (
            f"route:{source.route_name.value}",
            f"explanations:{source.explanation_count}",
            f"primary:{source.primary_explanation_id}",
        ),
        "Display routing explanation metadata without applying routing.",
    )


def _agent_panel(
    source: AgentExplainabilityAuditRegistry,
) -> WorkflowExplainabilityPanel:
    return _panel(
        "agent_explainability_audit",
        "agent_explainability_audit_registry",
        source.serialization_version,
        source.agent_ids,
        (
            source.audit_count
            + len(source.validated_explainability_surfaces)
            + len(source.source_registry_refs)
            + len(source.memory_reference_sources)
        ),
        len(source.blocked_runtime_behaviors) + len(source.passive_boundary_flags),
        (
            f"agent_audits:{source.audit_count}",
            f"validated_surfaces:{len(source.validated_explainability_surfaces)}",
            f"provenance_referenced:{source.all_records_provenance_referenced}",
        ),
        "Display agent explainability audit coverage without invoking agents.",
    )


def _provenance_panel(
    source: DecisionProvenanceRegistry,
) -> WorkflowExplainabilityPanel:
    profile_contexts = sum(
        len(profile.provenance_dimensions) + len(profile.advisory_outputs)
        for profile in source.provenance_profiles
    )
    return _panel(
        "decision_provenance_context",
        "decision_provenance_registry",
        source.serialization_version,
        source.provenance_profile_ids,
        (
            source.profile_count
            + len(source.topic_ids)
            + len(source.backbone_node_ids)
            + len(source.workstation_surface_ids)
            + profile_contexts
        ),
        len(source.blocked_runtime_behaviors),
        (
            f"provenance_profiles:{source.profile_count}",
            f"backbone_nodes:{len(source.backbone_node_ids)}",
            f"workstation_surfaces:{len(source.workstation_surface_ids)}",
        ),
        "Display decision provenance metadata without recording provenance.",
    )


def _timeline_panel(source: RuntimeTimeline) -> WorkflowExplainabilityPanel:
    return _panel(
        "runtime_timeline_context",
        "runtime_timeline",
        source.serialization_version,
        source.panel_ids,
        source.timeline_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"timeline_panels:{source.panel_count}",
            f"timeline_signals:{source.timeline_signal_count}",
            f"status:{source.runtime_timeline_status}",
        ),
        "Display runtime timeline context without reconstructing timelines.",
    )


def _error_panel(source: ErrorIntelligence) -> WorkflowExplainabilityPanel:
    return _panel(
        "error_explainability_context",
        "error_intelligence",
        source.serialization_version,
        source.panel_ids,
        source.error_signal_count + source.panel_count + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"error_panels:{source.panel_count}",
            f"error_signals:{source.error_signal_count}",
            f"status:{source.error_intelligence_status}",
        ),
        "Display error explainability context without classifying live errors.",
    )


def _panel(
    panel_kind: WorkflowExplainabilityPanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> WorkflowExplainabilityPanel:
    return WorkflowExplainabilityPanel(
        panel_id=f"workflow_explainability::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        explainability_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep explanation generation, provenance recording, trace capture, routing, workflow execution, storage, and output mutation disabled.",  # noqa: E501
        ),
    )


def _panel_ids_for_status(
    panels: tuple[WorkflowExplainabilityPanel, ...],
    status: WorkflowExplainabilityStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> WorkflowExplainabilityStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _dashboard_status(
    panels: tuple[WorkflowExplainabilityPanel, ...],
) -> WorkflowExplainabilityStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _dashboard_actions(
    panels: tuple[WorkflowExplainabilityPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow explainability panels as read-only observability metadata.",
        "Preserve explanation generation, provenance recording, trace, routing, "
        "workflow, memory, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded workflow explainability panels detached from active explanation generation."
        )
    return tuple(actions)
