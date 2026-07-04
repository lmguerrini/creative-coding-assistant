"""V5.4 advisory runtime timeline metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution_replay_engine import ExecutionReplayPlan, plan_execution_replay
from .multimodal_studio import (
    MultimodalBranchingTimelineRegistry,
    MultimodalCreativeEvolutionTimelineRegistry,
    multimodal_branching_timeline_registry,
    multimodal_creative_evolution_timeline_registry,
)
from .production_telemetry import ProductionTelemetrySurface, build_production_telemetry
from .workflow_diagnostics import WorkflowDiagnostics, build_workflow_diagnostics
from .workflow_replay_engine import WorkflowReplayPlan, plan_workflow_replay

RuntimeTimelinePanelKind = Literal[
    "workflow_diagnostic_timeline",
    "production_telemetry_timeline",
    "workflow_replay_timeline",
    "execution_replay_timeline",
    "branching_timeline_context",
    "creative_evolution_timeline_context",
]
RuntimeTimelineStatus = Literal["ready", "guarded"]

RUNTIME_TIMELINE_PANEL_SERIALIZATION_VERSION = "runtime_timeline_panel.v1"
RUNTIME_TIMELINE_SERIALIZATION_VERSION = "runtime_timeline.v1"
RUNTIME_TIMELINE_AUTHORITY_BOUNDARY = (
    "The V5.4 Runtime Timeline surface summarizes workflow diagnostics, "
    "production telemetry, workflow replay planning, execution replay "
    "planning, passive branching timeline metadata, and passive creative "
    "evolution timeline metadata as read-only runtime timeline observability "
    "only; it does not reconstruct timelines, capture runtime events, replay "
    "events, record sessions, capture snapshots, capture or emit traces, emit "
    "telemetry, export events, create branches, generate creative evolution, "
    "persist replay data, mutate workflow state, control or execute workflows, "
    "route providers or models, invoke agents or node handlers, trigger "
    "retries or refinement, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "workflow_diagnostics",
    "production_telemetry",
    "workflow_replay_plan",
    "execution_replay_plan",
    "multimodal_branching_timeline_registry",
    "multimodal_creative_evolution_timeline_registry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "timeline_reconstruction",
    "runtime_event_capture",
    "runtime_event_replay",
    "workflow_replay_execution",
    "execution_replay_execution",
    "session_recording",
    "snapshot_capture",
    "trace_capture",
    "trace_emission",
    "telemetry_emission",
    "event_export",
    "branch_creation",
    "evolution_generation",
    "replay_persistence",
    "persistent_storage_write",
    "workflow_state_mutation",
    "workflow_control",
    "workflow_execution",
    "provider_or_model_routing",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "generated_output_modification",
    "runtime_evolution_application",
)


class RuntimeTimelinePanel(BaseModel):
    """One read-only V5.4 runtime timeline panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=200)
    panel_kind: RuntimeTimelinePanelKind
    status: RuntimeTimelineStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=240)
    timeline_signal_count: int = Field(ge=0, le=60000)
    guardrail_signal_count: int = Field(ge=0, le=24000)
    observed_runtime_event_count: None = None
    reconstructed_timeline_count: None = None
    emitted_timeline_event_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    runtime_timeline_panel_implemented: Literal[True] = True
    timeline_reconstruction_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    event_export_implemented: Literal[False] = False
    branch_creation_implemented: Literal[False] = False
    evolution_generation_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["runtime_timeline_panel.v1"] = (
        RUNTIME_TIMELINE_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"runtime_timeline::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.reconstructed_timeline_count is not None:
            raise ValueError("reconstructed_timeline_count must remain unset")
        if self.emitted_timeline_event_count is not None:
            raise ValueError("emitted_timeline_event_count must remain unset")
        if self.guardrail_signal_count > self.timeline_signal_count:
            raise ValueError("guardrail_signal_count must fit timeline_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class RuntimeTimeline(BaseModel):
    """Read-only V5.4 runtime timeline over passive observability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_timeline"] = "runtime_timeline"
    serialization_version: Literal["runtime_timeline.v1"] = (
        RUNTIME_TIMELINE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RUNTIME_TIMELINE_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    source_workflow_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_production_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_replay_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_execution_replay_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_branching_timeline_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_evolution_timeline_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[RuntimeTimelinePanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    timeline_signal_count: int = Field(ge=0, le=120000)
    guardrail_signal_count: int = Field(ge=0, le=50000)
    observed_runtime_event_count: None = None
    reconstructed_timeline_count: None = None
    emitted_timeline_event_count: None = None
    runtime_timeline_status: RuntimeTimelineStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    runtime_timeline_implemented: Literal[True] = True
    timeline_reconstruction_implemented: Literal[False] = False
    runtime_event_capture_implemented: Literal[False] = False
    runtime_event_replay_implemented: Literal[False] = False
    workflow_replay_execution_implemented: Literal[False] = False
    execution_replay_execution_implemented: Literal[False] = False
    session_recording_implemented: Literal[False] = False
    snapshot_capture_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    event_export_implemented: Literal[False] = False
    branch_creation_implemented: Literal[False] = False
    evolution_generation_implemented: Literal[False] = False
    replay_persistence_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    workflow_state_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _timeline_matches_panels(self) -> Self:
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
        if self.timeline_signal_count != sum(
            panel.timeline_signal_count for panel in self.panels
        ):
            raise ValueError("timeline_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_runtime_event_count is not None:
            raise ValueError("observed_runtime_event_count must remain unset")
        if self.reconstructed_timeline_count is not None:
            raise ValueError("reconstructed_timeline_count must remain unset")
        if self.emitted_timeline_event_count is not None:
            raise ValueError("emitted_timeline_event_count must remain unset")
        if self.runtime_timeline_status != _timeline_status(self.panels):
            raise ValueError("runtime_timeline_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match runtime timeline sources")
        return self


def build_runtime_timeline(
    *,
    workflow_diagnostics: WorkflowDiagnostics | None = None,
    production_telemetry: ProductionTelemetrySurface | None = None,
    workflow_replay: WorkflowReplayPlan | None = None,
    execution_replay: ExecutionReplayPlan | None = None,
    branching_timeline: MultimodalBranchingTimelineRegistry | None = None,
    creative_evolution_timeline: (
        MultimodalCreativeEvolutionTimelineRegistry | None
    ) = None,
) -> RuntimeTimeline:
    """Build read-only runtime timeline metadata without reconstructing events."""

    workflow_source = workflow_replay or plan_workflow_replay()
    execution_source = execution_replay or plan_execution_replay(
        workflow_replay=workflow_source,
    )
    telemetry_source = production_telemetry or build_production_telemetry()
    diagnostics_source = workflow_diagnostics or build_workflow_diagnostics(
        workflow_replay=workflow_source,
        execution_replay=execution_source,
        production_telemetry=telemetry_source,
    )
    branching_source = branching_timeline or multimodal_branching_timeline_registry()
    evolution_source = (
        creative_evolution_timeline or multimodal_creative_evolution_timeline_registry()
    )
    panels = (
        _workflow_diagnostics_panel(diagnostics_source),
        _telemetry_panel(telemetry_source),
        _workflow_replay_panel(workflow_source),
        _execution_replay_panel(execution_source),
        _branching_panel(branching_source),
        _evolution_panel(evolution_source),
    )

    return RuntimeTimeline(
        source_workflow_diagnostics_serialization_version=(
            diagnostics_source.serialization_version
        ),
        source_production_telemetry_serialization_version=(
            telemetry_source.serialization_version
        ),
        source_workflow_replay_serialization_version=workflow_source.serialization_version,
        source_execution_replay_serialization_version=(
            execution_source.serialization_version
        ),
        source_branching_timeline_serialization_version=(
            branching_source.serialization_version
        ),
        source_creative_evolution_timeline_serialization_version=(
            evolution_source.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        timeline_signal_count=sum(panel.timeline_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        runtime_timeline_status=_timeline_status(panels),
        advisory_actions=_timeline_actions(panels),
    )


def runtime_timeline_panel_by_id(
    panel_id: str,
    timeline: RuntimeTimeline | None = None,
) -> RuntimeTimelinePanel | None:
    """Return one runtime timeline panel without replaying events."""

    source_timeline = timeline or build_runtime_timeline()
    for panel in source_timeline.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def runtime_timeline_panels_for_status(
    status: RuntimeTimelineStatus,
    timeline: RuntimeTimeline | None = None,
) -> tuple[RuntimeTimelinePanel, ...]:
    """Return runtime timeline panels by status without reconstruction."""

    source_timeline = timeline or build_runtime_timeline()
    return tuple(panel for panel in source_timeline.panels if panel.status == status)


def _workflow_diagnostics_panel(source: WorkflowDiagnostics) -> RuntimeTimelinePanel:
    return _panel(
        "workflow_diagnostic_timeline",
        "workflow_diagnostics",
        source.serialization_version,
        source.panel_ids,
        source.diagnostic_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"workflow_diagnostic_panels:{source.panel_count}",
            f"diagnostic_signals:{source.diagnostic_signal_count}",
            f"status:{source.workflow_diagnostics_status}",
        ),
        "Display workflow diagnostic timeline context without runtime event capture.",
    )


def _telemetry_panel(source: ProductionTelemetrySurface) -> RuntimeTimelinePanel:
    return _panel(
        "production_telemetry_timeline",
        "production_telemetry",
        source.serialization_version,
        source.channel_ids,
        source.telemetry_signal_count
        + source.guarded_signal_count
        + source.channel_count,
        len(source.blocked_runtime_behaviors) + len(source.guarded_channel_ids),
        (
            f"telemetry_channels:{source.channel_count}",
            f"telemetry_signals:{source.telemetry_signal_count}",
            f"status:{source.production_telemetry_status}",
        ),
        "Display production telemetry timeline context without emitting telemetry.",
    )


def _workflow_replay_panel(source: WorkflowReplayPlan) -> RuntimeTimelinePanel:
    return _panel(
        "workflow_replay_timeline",
        "workflow_replay_plan",
        source.serialization_version,
        source.candidate_ids,
        (
            source.total_replay_context_count
            + source.total_workflow_node_count
            + source.candidate_count
            + source.total_advisory_replay_score
        ),
        (
            len(source.blocked_runtime_behaviors)
            + source.failure_guardrail_count
            + source.storage_guardrail_count
        ),
        (
            f"workflow_replay_candidates:{source.candidate_count}",
            f"replay_contexts:{source.total_replay_context_count}",
            f"pressure:{source.workflow_replay_pressure}",
        ),
        "Display workflow replay timeline posture without replaying workflow events.",
    )


def _execution_replay_panel(source: ExecutionReplayPlan) -> RuntimeTimelinePanel:
    return _panel(
        "execution_replay_timeline",
        "execution_replay_plan",
        source.serialization_version,
        source.candidate_ids,
        (
            source.total_replay_context_count
            + source.total_execution_replay_profile_count
            + source.total_workflow_replay_candidate_count
            + source.candidate_count
            + source.total_advisory_replay_score
        ),
        (
            len(source.blocked_runtime_behaviors)
            + source.provider_guardrail_count
            + source.scoring_guardrail_count
            + source.storage_guardrail_count
        ),
        (
            f"execution_replay_candidates:{source.candidate_count}",
            f"replay_contexts:{source.total_replay_context_count}",
            f"pressure:{source.execution_replay_pressure}",
        ),
        "Display execution replay timeline posture without replaying runtime events.",
    )


def _branching_panel(
    source: MultimodalBranchingTimelineRegistry,
) -> RuntimeTimelinePanel:
    profile_contexts = sum(
        len(profile.branch_context_fields) + len(profile.advisory_outputs)
        for profile in source.branching_timeline_profiles
    )
    return _panel(
        "branching_timeline_context",
        "multimodal_branching_timeline_registry",
        source.serialization_version,
        source.profile_ids,
        (
            source.profile_count
            + len(source.branching_timeline_surface_refs)
            + len(source.source_reference_ids)
            + profile_contexts
        ),
        len(source.blocked_runtime_behaviors),
        (
            f"branching_profiles:{source.profile_count}",
            f"branch_surfaces:{len(source.branching_timeline_surface_refs)}",
            f"routes:{len(source.route_names)}",
        ),
        "Display branching timeline context without creating branches.",
    )


def _evolution_panel(
    source: MultimodalCreativeEvolutionTimelineRegistry,
) -> RuntimeTimelinePanel:
    profile_contexts = sum(
        len(profile.evolution_context_fields) + len(profile.advisory_outputs)
        for profile in source.creative_evolution_timeline_profiles
    )
    return _panel(
        "creative_evolution_timeline_context",
        "multimodal_creative_evolution_timeline_registry",
        source.serialization_version,
        source.profile_ids,
        (
            source.profile_count
            + len(source.creative_evolution_surface_refs)
            + len(source.source_reference_ids)
            + profile_contexts
        ),
        len(source.blocked_runtime_behaviors),
        (
            f"creative_evolution_profiles:{source.profile_count}",
            f"evolution_surfaces:{len(source.creative_evolution_surface_refs)}",
            f"routes:{len(source.route_names)}",
        ),
        "Display creative evolution timeline context without generating evolution.",
    )


def _panel(
    panel_kind: RuntimeTimelinePanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> RuntimeTimelinePanel:
    return RuntimeTimelinePanel(
        panel_id=f"runtime_timeline::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        timeline_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep reconstruction, runtime event capture, replay, telemetry emission, storage, routing, and workflow execution disabled.",  # noqa: E501
        ),
    )


def _panel_ids_for_status(
    panels: tuple[RuntimeTimelinePanel, ...],
    status: RuntimeTimelineStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> RuntimeTimelineStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _timeline_status(
    panels: tuple[RuntimeTimelinePanel, ...],
) -> RuntimeTimelineStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _timeline_actions(
    panels: tuple[RuntimeTimelinePanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose runtime timeline panels as read-only observability metadata.",
        "Preserve reconstruction, event capture, replay, telemetry, branch, "
        "workflow, routing, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded runtime timeline panels detached from runtime replay."
        )
    return tuple(actions)
