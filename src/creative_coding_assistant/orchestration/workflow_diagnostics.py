"""V5.4 advisory workflow diagnostics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from .execution_replay_engine import ExecutionReplayPlan, plan_execution_replay
from .performance_failure_path_audit import (
    PerformanceFailurePathAuditRegistry,
    performance_failure_path_audit_registry,
)
from .production_telemetry import (
    ProductionTelemetrySurface,
    build_production_telemetry,
)
from .workflow import WORKFLOW_STEP_ORDER, WorkflowStatus
from .workflow_replay_engine import WorkflowReplayPlan, plan_workflow_replay
from .workflow_review import MAX_WORKFLOW_REFINEMENT_COUNT

WorkflowDiagnosticPanelKind = Literal[
    "graph_topology",
    "state_transition_contract",
    "workflow_replay",
    "execution_replay",
    "failure_audit",
    "telemetry_boundary",
]
WorkflowDiagnosticStatus = Literal["ready", "guarded"]

WORKFLOW_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION = "workflow_diagnostic_panel.v1"
WORKFLOW_DIAGNOSTICS_SERIALIZATION_VERSION = "workflow_diagnostics.v1"
WORKFLOW_STATE_CONTRACT_SERIALIZATION_VERSION = "workflow_state_contract.v1"
WORKFLOW_DIAGNOSTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Workflow Diagnostics surface converts existing execution graph, "
    "workflow state, workflow replay, execution replay, failure audit, and "
    "production telemetry metadata into read-only workflow diagnostics only; "
    "it does not compile graphs, execute workflows, invoke node handlers, "
    "collect runtime events, mutate workflow state or graph order, replay "
    "events, emit telemetry, route providers or models, trigger retries, "
    "mutate prompts, write storage, or modify generated output."
)

_SOURCE_SURFACES = (
    "execution_graph_analysis",
    "workflow_state_contract",
    "workflow_replay_plan",
    "execution_replay_plan",
    "performance_failure_path_audit_registry",
    "production_telemetry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "langgraph_compilation",
    "workflow_execution",
    "workflow_control",
    "workflow_state_mutation",
    "workflow_graph_mutation",
    "workflow_order_mutation",
    "node_handler_invocation",
    "runtime_event_capture",
    "workflow_replay_execution",
    "execution_replay_execution",
    "telemetry_emission",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class WorkflowDiagnosticPanel(BaseModel):
    """One read-only V5.4 workflow diagnostics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: WorkflowDiagnosticPanelKind
    status: WorkflowDiagnosticStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=160)
    diagnostic_signal_count: int = Field(ge=0, le=1000)
    guardrail_signal_count: int = Field(ge=0, le=240)
    observed_runtime_event_count: None = None
    compiled_graph_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    workflow_diagnostic_panel_implemented: Literal[True] = True
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_diagnostic_panel.v1"] = (
        WORKFLOW_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"workflow_diagnostics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.compiled_graph_count is not None:
            raise ValueError("compiled_graph_count must remain unset")
        if self.guardrail_signal_count > self.diagnostic_signal_count:
            raise ValueError("guardrail_signal_count must fit diagnostic_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class WorkflowDiagnostics(BaseModel):
    """Read-only V5.4 workflow diagnostics over existing workflow metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_diagnostics"] = "workflow_diagnostics"
    serialization_version: Literal["workflow_diagnostics.v1"] = (
        WORKFLOW_DIAGNOSTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_DIAGNOSTICS_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_execution_graph_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_workflow_state_serialization_version: Literal[
        "workflow_state_contract.v1"
    ] = WORKFLOW_STATE_CONTRACT_SERIALIZATION_VERSION
    source_workflow_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_replay_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_failure_audit_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_production_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[WorkflowDiagnosticPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    diagnostic_signal_count: int = Field(ge=0, le=2000)
    guardrail_signal_count: int = Field(ge=0, le=480)
    observed_runtime_event_count: None = None
    compiled_graph_count: None = None
    workflow_diagnostics_status: WorkflowDiagnosticStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    workflow_diagnostics_implemented: Literal[True] = True
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _diagnostics_matches_panels(self) -> Self:
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
        if self.diagnostic_signal_count != sum(
            panel.diagnostic_signal_count for panel in self.panels
        ):
            raise ValueError("diagnostic_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.compiled_graph_count is not None:
            raise ValueError("compiled_graph_count must remain unset")
        if self.workflow_diagnostics_status != _diagnostics_status(self.panels):
            raise ValueError("workflow_diagnostics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match workflow diagnostic sources")
        return self


def build_workflow_diagnostics(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    workflow_replay: WorkflowReplayPlan | None = None,
    execution_replay: ExecutionReplayPlan | None = None,
    failure_audit: PerformanceFailurePathAuditRegistry | None = None,
    production_telemetry: ProductionTelemetrySurface | None = None,
) -> WorkflowDiagnostics:
    """Build read-only workflow diagnostics without running the workflow."""

    graph = execution_graph or analyze_assistant_execution_graph()
    workflow = workflow_replay or plan_workflow_replay(execution_graph=graph)
    execution = execution_replay or plan_execution_replay(workflow_replay=workflow)
    audit = failure_audit or performance_failure_path_audit_registry()
    telemetry = production_telemetry or build_production_telemetry()
    panels = (
        _graph_topology_panel(graph),
        _state_transition_contract_panel(),
        _workflow_replay_panel(workflow),
        _execution_replay_panel(execution),
        _failure_audit_panel(audit),
        _telemetry_boundary_panel(telemetry),
    )

    return WorkflowDiagnostics(
        source_execution_graph_serialization_version=graph.serialization_version,
        source_workflow_replay_serialization_version=workflow.serialization_version,
        source_execution_replay_serialization_version=execution.serialization_version,
        source_failure_audit_serialization_version=audit.serialization_version,
        source_production_telemetry_serialization_version=(
            telemetry.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        diagnostic_signal_count=sum(
            panel.diagnostic_signal_count for panel in panels
        ),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        workflow_diagnostics_status=_diagnostics_status(panels),
        advisory_actions=_diagnostics_actions(panels),
    )


def workflow_diagnostic_panel_by_id(
    panel_id: str,
    diagnostics: WorkflowDiagnostics | None = None,
) -> WorkflowDiagnosticPanel | None:
    """Return one workflow diagnostic panel without executing workflows."""

    source_diagnostics = diagnostics or build_workflow_diagnostics()
    for panel in source_diagnostics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def workflow_diagnostic_panels_for_status(
    status: WorkflowDiagnosticStatus,
    diagnostics: WorkflowDiagnostics | None = None,
) -> tuple[WorkflowDiagnosticPanel, ...]:
    """Return workflow diagnostic panels by status without runtime collection."""

    source_diagnostics = diagnostics or build_workflow_diagnostics()
    return tuple(panel for panel in source_diagnostics.panels if panel.status == status)


def _graph_topology_panel(graph: ExecutionGraphAnalysis) -> WorkflowDiagnosticPanel:
    guardrails = graph.failure_edge_count + len(graph.retry_entry_node_ids)
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::graph_topology",
        panel_kind="graph_topology",
        status=_status_for_guardrails(guardrails),
        source_id="execution_graph_analysis",
        source_serialization_version=graph.serialization_version,
        source_item_ids=graph.node_order,
        diagnostic_signal_count=graph.node_count + graph.edge_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"workflow_nodes:{graph.node_count}",
            f"workflow_edges:{graph.edge_count}",
            f"failure_edges:{graph.failure_edge_count}",
        ),
        advisory_actions=(
            "Display static workflow topology without compiling LangGraph.",
            "Keep node handlers, transitions, retries, and routing disabled.",
        ),
    )


def _state_transition_contract_panel() -> WorkflowDiagnosticPanel:
    steps = tuple(step.value for step in WORKFLOW_STEP_ORDER)
    statuses = tuple(status.value for status in WorkflowStatus)
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::state_transition_contract",
        panel_kind="state_transition_contract",
        status="ready",
        source_id="workflow_state_contract",
        source_serialization_version=WORKFLOW_STATE_CONTRACT_SERIALIZATION_VERSION,
        source_item_ids=steps + statuses,
        diagnostic_signal_count=len(steps) + len(statuses),
        guardrail_signal_count=0,
        evidence=(
            f"workflow_steps:{len(steps)}",
            f"workflow_statuses:{len(statuses)}",
            f"max_refinement_count:{MAX_WORKFLOW_REFINEMENT_COUNT}",
        ),
        advisory_actions=(
            "Display workflow state contract without mutating active state.",
            "Keep workflow order, status, and refinement transitions read-only.",
        ),
    )


def _workflow_replay_panel(workflow: WorkflowReplayPlan) -> WorkflowDiagnosticPanel:
    guardrails = workflow.failure_guardrail_count + workflow.storage_guardrail_count
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::workflow_replay",
        panel_kind="workflow_replay",
        status=_status_for_guardrails(guardrails),
        source_id="workflow_replay_plan",
        source_serialization_version=workflow.serialization_version,
        source_item_ids=workflow.candidate_ids,
        diagnostic_signal_count=workflow.candidate_count
        + workflow.total_replay_context_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"workflow_replay_pressure:{workflow.workflow_replay_pressure}",
            f"replay_candidates:{workflow.replay_candidate_count}",
            f"guardrails:{guardrails}",
        ),
        advisory_actions=(
            "Display workflow replay diagnostics without replaying runtime events.",
            "Keep timeline reconstruction, replay persistence, and storage disabled.",
        ),
    )


def _execution_replay_panel(execution: ExecutionReplayPlan) -> WorkflowDiagnosticPanel:
    guardrails = (
        execution.provider_guardrail_count
        + execution.scoring_guardrail_count
        + execution.storage_guardrail_count
    )
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::execution_replay",
        panel_kind="execution_replay",
        status=_status_for_guardrails(guardrails),
        source_id="execution_replay_plan",
        source_serialization_version=execution.serialization_version,
        source_item_ids=execution.candidate_ids,
        diagnostic_signal_count=execution.candidate_count
        + execution.total_replay_context_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"execution_replay_pressure:{execution.execution_replay_pressure}",
            f"replay_candidates:{execution.replay_candidate_count}",
            f"guardrails:{guardrails}",
        ),
        advisory_actions=(
            "Display execution replay diagnostics without provider execution.",
            "Keep trace reconstruction, scoring, routing, and storage disabled.",
        ),
    )


def _failure_audit_panel(
    audit: PerformanceFailurePathAuditRegistry,
) -> WorkflowDiagnosticPanel:
    workflow_checks = tuple(
        check
        for check in audit.applicable_required_checks
        if "workflow" in check or "failure" in check
    )
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::failure_audit",
        panel_kind="failure_audit",
        status="ready",
        source_id="performance_failure_path_audit_registry",
        source_serialization_version=audit.serialization_version,
        source_item_ids=audit.audit_ids,
        diagnostic_signal_count=audit.record_count + len(workflow_checks),
        guardrail_signal_count=0,
        evidence=(
            f"audit_records:{audit.record_count}",
            f"workflow_failure_checks:{len(workflow_checks)}",
            f"all_applicable_checks_covered:{audit.all_applicable_checks_covered}",
        ),
        advisory_actions=(
            "Display workflow failure audit coverage without live failure analysis.",
            "Keep workflow execution, retries, routing, and output mutation disabled.",
        ),
    )


def _telemetry_boundary_panel(
    telemetry: ProductionTelemetrySurface,
) -> WorkflowDiagnosticPanel:
    guardrails = len(telemetry.guarded_channel_ids)
    return WorkflowDiagnosticPanel(
        panel_id="workflow_diagnostics::telemetry_boundary",
        panel_kind="telemetry_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="production_telemetry",
        source_serialization_version=telemetry.serialization_version,
        source_item_ids=telemetry.channel_ids,
        diagnostic_signal_count=telemetry.channel_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"telemetry_status:{telemetry.production_telemetry_status}",
            f"telemetry_channels:{telemetry.channel_count}",
            f"guarded_channels:{len(telemetry.guarded_channel_ids)}",
        ),
        advisory_actions=(
            "Display telemetry boundary diagnostics without emitting telemetry.",
            "Keep metrics collection, event export, alerts, and workflow control disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[WorkflowDiagnosticPanel, ...],
    status: WorkflowDiagnosticStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> WorkflowDiagnosticStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _diagnostics_status(
    panels: tuple[WorkflowDiagnosticPanel, ...],
) -> WorkflowDiagnosticStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _diagnostics_actions(
    panels: tuple[WorkflowDiagnosticPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose workflow diagnostics panels as read-only observability metadata.",
        "Preserve graph compilation, workflow execution, replay, telemetry, "
        "routing, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded workflow diagnostic panels detached from runtime execution."
        )
    return tuple(actions)
