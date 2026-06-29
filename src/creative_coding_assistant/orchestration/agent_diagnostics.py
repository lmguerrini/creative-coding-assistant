"""V5.4 advisory agent diagnostics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_capability_alignment import (
    AgentCapabilityAlignmentRegistry,
    agent_capability_alignment_registry,
)
from .agent_determinism_audit import (
    AgentDeterminismAuditRegistry,
    agent_determinism_audit_registry,
)
from .agent_explainability_audit import (
    AgentExplainabilityAuditRegistry,
    agent_explainability_audit_registry,
)
from .agent_lifecycle import AgentLifecycleRegistry, agent_lifecycle_registry
from .agent_metadata import AgentMetadataRegistry, agent_metadata_registry
from .agent_reliability_audit import (
    AgentReliabilityAuditRegistry,
    agent_reliability_audit_registry,
)
from .agent_telemetry_foundation import (
    AgentTelemetryFoundationRegistry,
    agent_telemetry_foundation_registry,
)

AgentDiagnosticPanelKind = Literal[
    "metadata_coverage",
    "lifecycle_coverage",
    "capability_alignment",
    "telemetry_coverage",
    "reliability_audit",
    "determinism_audit",
    "explainability_audit",
]
AgentDiagnosticStatus = Literal["ready", "guarded"]

AGENT_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION = "agent_diagnostic_panel.v1"
AGENT_DIAGNOSTICS_SERIALIZATION_VERSION = "agent_diagnostics.v1"
AGENT_DIAGNOSTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Agent Diagnostics surface converts existing agent metadata, "
    "lifecycle, capability alignment, telemetry foundation, reliability "
    "audit, determinism audit, and explainability audit metadata into "
    "read-only agent diagnostics only; it does not invoke agents, execute "
    "agent capabilities, run lifecycle transitions, synchronize runtime "
    "state, generate explanations, capture traces, emit telemetry, select "
    "runtimes, route providers or models, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_SOURCE_SURFACES = (
    "agent_metadata_registry",
    "agent_lifecycle_registry",
    "agent_capability_alignment_registry",
    "agent_telemetry_foundation_registry",
    "agent_reliability_audit_registry",
    "agent_determinism_audit_registry",
    "agent_explainability_audit_registry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_invocation",
    "agent_capability_execution",
    "runtime_lifecycle_engine",
    "state_transition_execution",
    "runtime_state_synchronization",
    "explanation_generation",
    "trace_capture",
    "telemetry_emission",
    "runtime_selection",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AgentDiagnosticPanel(BaseModel):
    """One read-only V5.4 agent diagnostics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: AgentDiagnosticPanelKind
    status: AgentDiagnosticStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    agent_signal_count: int = Field(ge=0, le=1000)
    guardrail_signal_count: int = Field(ge=0, le=240)
    observed_agent_run_count: None = None
    runtime_failure_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    agent_diagnostic_panel_implemented: Literal[True] = True
    active_agent_execution_implemented: Literal[False] = False
    agent_capability_execution_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    state_transition_execution_implemented: Literal[False] = False
    runtime_state_synchronization_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_diagnostic_panel.v1"] = (
        AGENT_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"agent_diagnostics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_agent_run_count is not None:
            raise ValueError("observed_agent_run_count must remain unset")
        if self.runtime_failure_count is not None:
            raise ValueError("runtime_failure_count must remain unset")
        if self.guardrail_signal_count > self.agent_signal_count:
            raise ValueError("guardrail_signal_count must fit agent_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class AgentDiagnostics(BaseModel):
    """Read-only V5.4 agent diagnostics over existing passive agent metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_diagnostics"] = "agent_diagnostics"
    serialization_version: Literal["agent_diagnostics.v1"] = (
        AGENT_DIAGNOSTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_DIAGNOSTICS_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_agent_metadata_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_lifecycle_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_capability_alignment_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_reliability_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_determinism_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_explainability_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    panels: tuple[AgentDiagnosticPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    agent_signal_count: int = Field(ge=0, le=2000)
    guardrail_signal_count: int = Field(ge=0, le=480)
    observed_agent_run_count: None = None
    runtime_failure_count: None = None
    agent_diagnostics_status: AgentDiagnosticStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    agent_diagnostics_implemented: Literal[True] = True
    active_agent_execution_implemented: Literal[False] = False
    agent_capability_execution_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    state_transition_execution_implemented: Literal[False] = False
    runtime_state_synchronization_implemented: Literal[False] = False
    explanation_generation_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
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
        if self.agent_signal_count != sum(
            panel.agent_signal_count for panel in self.panels
        ):
            raise ValueError("agent_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_agent_run_count is not None:
            raise ValueError("observed_agent_run_count must remain unset")
        if self.runtime_failure_count is not None:
            raise ValueError("runtime_failure_count must remain unset")
        if self.agent_diagnostics_status != _diagnostics_status(self.panels):
            raise ValueError("agent_diagnostics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match agent diagnostic sources")
        return self


def build_agent_diagnostics(
    *,
    agent_metadata: AgentMetadataRegistry | None = None,
    agent_lifecycle: AgentLifecycleRegistry | None = None,
    capability_alignment: AgentCapabilityAlignmentRegistry | None = None,
    agent_telemetry: AgentTelemetryFoundationRegistry | None = None,
    reliability_audit: AgentReliabilityAuditRegistry | None = None,
    determinism_audit: AgentDeterminismAuditRegistry | None = None,
    explainability_audit: AgentExplainabilityAuditRegistry | None = None,
) -> AgentDiagnostics:
    """Build read-only agent diagnostics without invoking agents."""

    metadata = agent_metadata or agent_metadata_registry()
    lifecycle = agent_lifecycle or agent_lifecycle_registry()
    alignment = capability_alignment or agent_capability_alignment_registry()
    telemetry = agent_telemetry or agent_telemetry_foundation_registry()
    reliability = reliability_audit or agent_reliability_audit_registry()
    determinism = determinism_audit or agent_determinism_audit_registry()
    explainability = explainability_audit or agent_explainability_audit_registry()
    panels = (
        _metadata_panel(metadata),
        _lifecycle_panel(lifecycle),
        _capability_alignment_panel(alignment),
        _telemetry_panel(telemetry),
        _reliability_panel(reliability),
        _determinism_panel(determinism),
        _explainability_panel(explainability),
    )

    return AgentDiagnostics(
        source_agent_metadata_serialization_version=metadata.serialization_version,
        source_agent_lifecycle_serialization_version=lifecycle.serialization_version,
        source_capability_alignment_serialization_version=(
            alignment.serialization_version
        ),
        source_agent_telemetry_serialization_version=telemetry.serialization_version,
        source_reliability_audit_serialization_version=reliability.serialization_version,
        source_determinism_audit_serialization_version=determinism.serialization_version,
        source_explainability_audit_serialization_version=(
            explainability.serialization_version
        ),
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        agent_signal_count=sum(panel.agent_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        agent_diagnostics_status=_diagnostics_status(panels),
        advisory_actions=_diagnostics_actions(panels),
    )


def agent_diagnostic_panel_by_id(
    panel_id: str,
    diagnostics: AgentDiagnostics | None = None,
) -> AgentDiagnosticPanel | None:
    """Return one agent diagnostic panel without invoking agents."""

    source_diagnostics = diagnostics or build_agent_diagnostics()
    for panel in source_diagnostics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def agent_diagnostic_panels_for_status(
    status: AgentDiagnosticStatus,
    diagnostics: AgentDiagnostics | None = None,
) -> tuple[AgentDiagnosticPanel, ...]:
    """Return agent diagnostic panels by status without runtime collection."""

    source_diagnostics = diagnostics or build_agent_diagnostics()
    return tuple(panel for panel in source_diagnostics.panels if panel.status == status)


def _metadata_panel(registry: AgentMetadataRegistry) -> AgentDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::metadata_coverage",
        panel_kind="metadata_coverage",
        status=_status_for_guardrails(guardrails),
        source_id="agent_metadata_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        agent_signal_count=(
            registry.metadata_count
            + len(registry.observability_surfaces)
            + len(registry.auditability_surfaces)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"agent_metadata:{registry.metadata_count}",
            f"observability_surfaces:{len(registry.observability_surfaces)}",
            f"auditability_surfaces:{len(registry.auditability_surfaces)}",
        ),
        advisory_actions=(
            "Display agent metadata coverage without runtime selection.",
            "Keep caching, parallel execution, routing, and agent execution disabled.",
        ),
    )


def _lifecycle_panel(registry: AgentLifecycleRegistry) -> AgentDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::lifecycle_coverage",
        panel_kind="lifecycle_coverage",
        status=_status_for_guardrails(guardrails),
        source_id="agent_lifecycle_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids + registry.transition_ids,
        agent_signal_count=(
            registry.profile_count + len(registry.states) + len(registry.transitions)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"lifecycle_profiles:{registry.profile_count}",
            f"lifecycle_states:{len(registry.states)}",
            f"lifecycle_transitions:{len(registry.transitions)}",
        ),
        advisory_actions=(
            "Display lifecycle coverage without running state transitions.",
            "Keep lifecycle engines, workflow state changes, and agent invocation disabled.",
        ),
    )


def _capability_alignment_panel(
    registry: AgentCapabilityAlignmentRegistry,
) -> AgentDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::capability_alignment",
        panel_kind="capability_alignment",
        status=_status_for_guardrails(guardrails),
        source_id="agent_capability_alignment_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids + tuple(registry.capability_ids),
        agent_signal_count=(
            registry.alignment_count
            + len(registry.capability_ids)
            + len(registry.source_registries)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"alignment_profiles:{registry.alignment_count}",
            f"capabilities:{len(registry.capability_ids)}",
            f"source_registries:{len(registry.source_registries)}",
        ),
        advisory_actions=(
            "Display capability alignment without activating capabilities.",
            "Keep runtime work routing, prompts, coordination, and debates disabled.",
        ),
    )


def _telemetry_panel(
    registry: AgentTelemetryFoundationRegistry,
) -> AgentDiagnosticPanel:
    guardrails = len(registry.passive_boundary_flags)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::telemetry_coverage",
        panel_kind="telemetry_coverage",
        status=_status_for_guardrails(guardrails),
        source_id="agent_telemetry_foundation_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        agent_signal_count=(
            registry.profile_count
            + len(registry.telemetry_event_types)
            + len(registry.trace_profile_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"telemetry_profiles:{registry.profile_count}",
            f"event_types:{len(registry.telemetry_event_types)}",
            f"trace_profiles:{len(registry.trace_profile_ids)}",
        ),
        advisory_actions=(
            "Display agent telemetry coverage without emitting telemetry.",
            "Keep trace capture, monitoring, memory writes, and routing disabled.",
        ),
    )


def _reliability_panel(
    registry: AgentReliabilityAuditRegistry,
) -> AgentDiagnosticPanel:
    guardrails = len(registry.passive_boundary_flags)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::reliability_audit",
        panel_kind="reliability_audit",
        status=_status_for_guardrails(guardrails),
        source_id="agent_reliability_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        agent_signal_count=(
            registry.audit_count + len(registry.validated_reliability_surfaces)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"reliability_audits:{registry.audit_count}",
            f"validated_surfaces:{len(registry.validated_reliability_surfaces)}",
            f"no_missing_coverage:{registry.no_missing_coverage}",
        ),
        advisory_actions=(
            "Display reliability audit coverage without recovery execution.",
            "Keep state sync, conflict resolution, escalation, and retry execution disabled.",
        ),
    )


def _determinism_panel(
    registry: AgentDeterminismAuditRegistry,
) -> AgentDiagnosticPanel:
    guardrails = len(registry.passive_boundary_flags)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::determinism_audit",
        panel_kind="determinism_audit",
        status=_status_for_guardrails(guardrails),
        source_id="agent_determinism_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        agent_signal_count=(
            registry.audit_count + len(registry.validated_determinism_surfaces)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"determinism_audits:{registry.audit_count}",
            f"validated_surfaces:{len(registry.validated_determinism_surfaces)}",
            f"no_missing_coverage:{registry.no_missing_coverage}",
        ),
        advisory_actions=(
            "Display determinism audit coverage without sampling or execution.",
            "Keep workflow order, route selection, scheduling, and prompt changes disabled.",
        ),
    )


def _explainability_panel(
    registry: AgentExplainabilityAuditRegistry,
) -> AgentDiagnosticPanel:
    guardrails = len(registry.passive_boundary_flags)
    return AgentDiagnosticPanel(
        panel_id="agent_diagnostics::explainability_audit",
        panel_kind="explainability_audit",
        status=_status_for_guardrails(guardrails),
        source_id="agent_explainability_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        agent_signal_count=(
            registry.audit_count + len(registry.validated_explainability_surfaces)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"explainability_audits:{registry.audit_count}",
            f"validated_surfaces:{len(registry.validated_explainability_surfaces)}",
            f"no_missing_coverage:{registry.no_missing_coverage}",
        ),
        advisory_actions=(
            "Display explainability audit coverage without generating explanations.",
            "Keep trace capture, memory writes, runtime selection, and routing disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[AgentDiagnosticPanel, ...],
    status: AgentDiagnosticStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> AgentDiagnosticStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _diagnostics_status(
    panels: tuple[AgentDiagnosticPanel, ...],
) -> AgentDiagnosticStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _diagnostics_actions(
    panels: tuple[AgentDiagnosticPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose agent diagnostics panels as read-only observability metadata.",
        "Preserve agent invocation, lifecycle, state sync, explanation, trace, "
        "telemetry, routing, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded agent diagnostic panels detached from runtime agent behavior."
        )
    return tuple(actions)
