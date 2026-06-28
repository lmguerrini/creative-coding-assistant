"""Passive V4.6 agent telemetry foundation metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration.agent_lifecycle import (
    agent_lifecycle_profile_by_agent_id,
    agent_lifecycle_registry,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AgentOperationalMetadata,
    agent_metadata_by_agent_id,
    agent_metadata_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    decision_provenance_registry,
    escalation_trace_registry,
)

AgentTelemetryFoundationStage = Literal["v4_6_agent_telemetry_foundation"]
AgentTelemetryFoundationStatus = Literal["pass"]

AGENT_TELEMETRY_FOUNDATION_PROFILE_SERIALIZATION_VERSION = (
    "agent_telemetry_foundation_profile.v1"
)
AGENT_TELEMETRY_FOUNDATION_REGISTRY_SERIALIZATION_VERSION = (
    "agent_telemetry_foundation_registry.v1"
)
AGENT_TELEMETRY_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent telemetry foundation metadata describes passive agent "
    "observability surfaces, auditability surfaces, lifecycle references, "
    "stream event type references, decision provenance references, escalation "
    "trace references, and telemetry boundary flags only; it does not emit "
    "telemetry, capture traces, record provenance, mutate event streams, "
    "start external monitoring, write memory, invoke agents, control "
    "workflows, route providers or models, trigger retries, or modify "
    "generated output."
)

_SOURCE_TELEMETRY_REGISTRIES = (
    "agent_metadata_registry",
    "agent_lifecycle_registry",
    "stream_event_contracts",
    "decision_provenance_registry",
    "escalation_trace_registry",
)
_TELEMETRY_DIMENSIONS = (
    "agent_identity",
    "role_identity",
    "lifecycle_reference",
    "input_surface",
    "output_surface",
    "event_type_reference",
    "provenance_reference",
    "trace_reference",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "telemetry_emission_blocked",
    "trace_capture_blocked",
    "provenance_recording_blocked",
    "event_stream_mutation_blocked",
    "external_monitoring_blocked",
    "memory_write_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "telemetry_emission",
    "trace_capture",
    "provenance_recording",
    "event_stream_mutation",
    "external_monitoring",
    "memory_write",
    "agent_invocation",
    "workflow_control",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)
_FOUNDATION_FINDINGS = (
    "agent_observability_surfaces_confirmed",
    "agent_auditability_surfaces_confirmed",
    "lifecycle_reference_confirmed",
    "stream_event_type_reference_confirmed",
    "decision_provenance_reference_confirmed",
    "escalation_trace_reference_confirmed",
    "runtime_telemetry_blocks_confirmed",
)


class AgentTelemetryFoundationProfile(BaseModel):
    """One passive telemetry foundation profile for an agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    telemetry_stage: AgentTelemetryFoundationStage = (
        "v4_6_agent_telemetry_foundation"
    )
    metadata_serialization_version: str = Field(min_length=1, max_length=80)
    lifecycle_profile_id: str = Field(min_length=1, max_length=140)
    observability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    auditability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    telemetry_event_types: tuple[str, ...] = Field(min_length=20, max_length=32)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    telemetry_dimensions: tuple[str, ...] = Field(min_length=8, max_length=8)
    telemetry_source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    foundation_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    metadata_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    lifecycle_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    provenance_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    trace_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    foundation_status: AgentTelemetryFoundationStatus = "pass"
    metadata_only_declared: Literal[True] = True
    observability_surface_coverage_present: Literal[True] = True
    event_type_reference_present: Literal[True] = True
    provenance_trace_reference_present: Literal[True] = True
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    event_stream_mutation_implemented: Literal[False] = False
    external_monitoring_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_telemetry_foundation_profile.v1"] = (
        AGENT_TELEMETRY_FOUNDATION_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentTelemetryFoundationRegistry(BaseModel):
    """Stable passive V4.6 registry for agent telemetry foundation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_telemetry_foundation_registry"] = (
        "agent_telemetry_foundation_registry"
    )
    serialization_version: Literal["agent_telemetry_foundation_registry.v1"] = (
        AGENT_TELEMETRY_FOUNDATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_TELEMETRY_FOUNDATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    telemetry_stage: AgentTelemetryFoundationStage = (
        "v4_6_agent_telemetry_foundation"
    )
    profiles: tuple[AgentTelemetryFoundationProfile, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    profile_count: int = Field(ge=12, le=12)
    source_agent_metadata_registry: Literal["agent_metadata_registry"] = (
        "agent_metadata_registry"
    )
    source_agent_lifecycle_registry: Literal["agent_lifecycle_registry"] = (
        "agent_lifecycle_registry"
    )
    source_stream_event_contracts: Literal["stream_event_contracts"] = (
        "stream_event_contracts"
    )
    source_decision_provenance_registry: Literal["decision_provenance_registry"] = (
        "decision_provenance_registry"
    )
    source_escalation_trace_registry: Literal["escalation_trace_registry"] = (
        "escalation_trace_registry"
    )
    telemetry_source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    observability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    auditability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    telemetry_event_types: tuple[str, ...] = Field(min_length=20, max_length=32)
    lifecycle_profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    telemetry_dimensions: tuple[str, ...] = Field(min_length=8, max_length=8)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_agents_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    telemetry_emission_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    provenance_recording_implemented: Literal[False] = False
    event_stream_mutation_implemented: Literal[False] = False
    external_monitoring_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        derived_agent_ids = tuple(profile.agent_id for profile in self.profiles)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match profiles")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")
        if self.telemetry_source_registries != _SOURCE_TELEMETRY_REGISTRIES:
            raise ValueError("telemetry_source_registries must match sources")

        known_lifecycle_profiles = set(self.lifecycle_profile_ids)
        for profile in self.profiles:
            if profile.telemetry_stage != self.telemetry_stage:
                raise ValueError("telemetry_stage must match registry")
            if profile.lifecycle_profile_id not in known_lifecycle_profiles:
                raise ValueError("lifecycle_profile_id must be known")
            if profile.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if profile.auditability_surfaces != self.auditability_surfaces:
                raise ValueError("auditability_surfaces must match registry")
            if profile.telemetry_event_types != self.telemetry_event_types:
                raise ValueError("telemetry_event_types must match registry")
            if profile.provenance_profile_ids != self.provenance_profile_ids:
                raise ValueError("provenance_profile_ids must match registry")
            if profile.trace_profile_ids != self.trace_profile_ids:
                raise ValueError("trace_profile_ids must match registry")
            if profile.telemetry_dimensions != self.telemetry_dimensions:
                raise ValueError("telemetry_dimensions must match registry")
            if profile.telemetry_source_registries != self.telemetry_source_registries:
                raise ValueError("telemetry_source_registries must match registry")
            if profile.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if profile.missing_coverage_items:
                raise ValueError("profiles must not contain missing coverage")
        return self


def agent_telemetry_foundation_registry() -> AgentTelemetryFoundationRegistry:
    """Return passive V4.6 agent telemetry foundation metadata."""

    return AGENT_TELEMETRY_FOUNDATION_REGISTRY


def agent_telemetry_profile_by_agent_id(
    agent_id: str,
    registry: AgentTelemetryFoundationRegistry | None = None,
) -> AgentTelemetryFoundationProfile | None:
    """Return one telemetry foundation profile by agent id."""

    source_registry = registry or AGENT_TELEMETRY_FOUNDATION_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_telemetry_profiles_for_event_type(
    event_type: StreamEventType | str,
    registry: AgentTelemetryFoundationRegistry | None = None,
) -> tuple[AgentTelemetryFoundationProfile, ...]:
    """Return passive telemetry profiles referencing one stream event type."""

    source_registry = registry or AGENT_TELEMETRY_FOUNDATION_REGISTRY
    normalized_event_type = (
        event_type.value if isinstance(event_type, StreamEventType) else str(event_type)
    ).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if normalized_event_type in profile.telemetry_event_types
    )


def agent_telemetry_profiles_for_dimension(
    telemetry_dimension: str,
    registry: AgentTelemetryFoundationRegistry | None = None,
) -> tuple[AgentTelemetryFoundationProfile, ...]:
    """Return passive telemetry profiles referencing one telemetry dimension."""

    source_registry = registry or AGENT_TELEMETRY_FOUNDATION_REGISTRY
    normalized_dimension = str(telemetry_dimension).strip()
    return tuple(
        profile
        for profile in source_registry.profiles
        if normalized_dimension in profile.telemetry_dimensions
    )


def _missing_coverage_items(
    *,
    metadata: AgentOperationalMetadata,
    lifecycle_profile_id: str,
) -> tuple[str, ...]:
    metadata_registry = agent_metadata_registry()
    provenance = decision_provenance_registry()
    trace = escalation_trace_registry()
    missing: list[str] = []
    if metadata.observability_surfaces != metadata_registry.observability_surfaces:
        missing.append("observability_surface_coverage_missing")
    if metadata.auditability_surfaces != metadata_registry.auditability_surfaces:
        missing.append("auditability_surface_coverage_missing")
    if not lifecycle_profile_id:
        missing.append("lifecycle_profile_reference_missing")
    if not _stream_event_type_values():
        missing.append("stream_event_type_reference_missing")
    if not provenance.provenance_profile_ids:
        missing.append("provenance_profile_reference_missing")
    if not trace.trace_profile_ids:
        missing.append("trace_profile_reference_missing")
    if not metadata.metadata_only:
        missing.append("metadata_only_declaration_missing")
    if "agent_execution" not in metadata_registry.blocked_runtime_behaviors:
        missing.append("agent_execution_block_missing")
    if "provenance_recording" not in provenance.blocked_runtime_behaviors:
        missing.append("provenance_recording_block_missing")
    if "trace_capture" not in trace.blocked_runtime_behaviors:
        missing.append("trace_capture_block_missing")
    if "trace_emission" not in trace.blocked_runtime_behaviors:
        missing.append("trace_emission_block_missing")
    if "memory_write" not in trace.blocked_runtime_behaviors:
        missing.append("memory_write_block_missing")
    if "provider_or_model_routing" not in trace.blocked_runtime_behaviors:
        missing.append("provider_model_routing_block_missing")
    if "workflow_control" not in trace.blocked_runtime_behaviors:
        missing.append("workflow_control_block_missing")
    if "retry_triggering" not in trace.blocked_runtime_behaviors:
        missing.append("retry_triggering_block_missing")
    if "generated_output_modification" not in trace.blocked_runtime_behaviors:
        missing.append("generated_output_mutation_block_missing")
    if provenance.provenance_recording_implemented:
        missing.append("provenance_recording_enabled")
    if provenance.trace_emission_implemented:
        missing.append("trace_emission_enabled")
    if trace.trace_capture_implemented:
        missing.append("trace_capture_enabled")
    if trace.trace_emission_implemented:
        missing.append("trace_emission_enabled")
    return tuple(missing)


def _profile(agent_id: str) -> AgentTelemetryFoundationProfile:
    metadata = agent_metadata_by_agent_id(agent_id)
    lifecycle_profile = agent_lifecycle_profile_by_agent_id(agent_id)
    provenance = decision_provenance_registry()
    trace = escalation_trace_registry()
    if metadata is None:
        raise ValueError(f"missing agent metadata for {agent_id}")
    if lifecycle_profile is None:
        raise ValueError(f"missing lifecycle profile for {agent_id}")

    return AgentTelemetryFoundationProfile(
        agent_id=agent_id,
        role_id=metadata.role_id,
        metadata_serialization_version=metadata.serialization_version,
        lifecycle_profile_id=lifecycle_profile.lifecycle_profile_id,
        observability_surfaces=metadata.observability_surfaces,
        auditability_surfaces=metadata.auditability_surfaces,
        telemetry_event_types=_stream_event_type_values(),
        provenance_profile_ids=provenance.provenance_profile_ids,
        trace_profile_ids=trace.trace_profile_ids,
        telemetry_dimensions=_TELEMETRY_DIMENSIONS,
        telemetry_source_registries=_SOURCE_TELEMETRY_REGISTRIES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        foundation_findings=_FOUNDATION_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            metadata=metadata,
            lifecycle_profile_id=lifecycle_profile.lifecycle_profile_id,
        ),
        metadata_blocked_runtime_behaviors=(
            agent_metadata_registry().blocked_runtime_behaviors
        ),
        lifecycle_blocked_runtime_behaviors=(
            lifecycle_profile.blocked_runtime_behaviors
        ),
        provenance_blocked_runtime_behaviors=provenance.blocked_runtime_behaviors,
        trace_blocked_runtime_behaviors=trace.blocked_runtime_behaviors,
        metadata_only_declared=(
            metadata.metadata_only
            and lifecycle_profile.metadata_only
            and provenance.metadata_only
            and trace.metadata_only
        ),
    )


def _stream_event_type_values() -> tuple[str, ...]:
    return tuple(event_type.value for event_type in StreamEventType)


AGENT_TELEMETRY_FOUNDATION_PROFILES = tuple(
    _profile(agent_id) for agent_id in agent_metadata_registry().agent_ids
)
AGENT_TELEMETRY_FOUNDATION_REGISTRY = AgentTelemetryFoundationRegistry(
    profiles=AGENT_TELEMETRY_FOUNDATION_PROFILES,
    agent_ids=tuple(profile.agent_id for profile in AGENT_TELEMETRY_FOUNDATION_PROFILES),
    profile_count=len(AGENT_TELEMETRY_FOUNDATION_PROFILES),
    telemetry_source_registries=_SOURCE_TELEMETRY_REGISTRIES,
    observability_surfaces=agent_metadata_registry().observability_surfaces,
    auditability_surfaces=agent_metadata_registry().auditability_surfaces,
    telemetry_event_types=_stream_event_type_values(),
    lifecycle_profile_ids=tuple(
        profile.lifecycle_profile_id for profile in agent_lifecycle_registry().profiles
    ),
    provenance_profile_ids=decision_provenance_registry().provenance_profile_ids,
    trace_profile_ids=escalation_trace_registry().trace_profile_ids,
    telemetry_dimensions=_TELEMETRY_DIMENSIONS,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
