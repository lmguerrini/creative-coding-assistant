"""V5.4 advisory production telemetry metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_telemetry_foundation import (
    AgentTelemetryFoundationRegistry,
    agent_telemetry_foundation_registry,
)
from creative_coding_assistant.orchestration.cost_dashboard import CostDashboard, build_cost_dashboard
from creative_coding_assistant.orchestration.performance_dashboard import (
    PerformanceDashboard,
    build_performance_dashboard,
)
from creative_coding_assistant.orchestration.quality_dashboard import QualityDashboard, build_quality_dashboard
from creative_coding_assistant.orchestration.token_dashboard import TokenDashboard, build_token_dashboard

ProductionTelemetryChannelKind = Literal[
    "agent_foundation",
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
    "emission_boundary",
]
ProductionTelemetryStatus = Literal["ready", "guarded"]

PRODUCTION_TELEMETRY_CHANNEL_SERIALIZATION_VERSION = "production_telemetry_channel.v1"
PRODUCTION_TELEMETRY_SERIALIZATION_VERSION = "production_telemetry.v1"
TELEMETRY_EMISSION_BOUNDARY_SERIALIZATION_VERSION = "telemetry_emission_boundary.v1"
PRODUCTION_TELEMETRY_AUTHORITY_BOUNDARY = (
    "The V5.4 Production Telemetry surface links existing agent telemetry "
    "foundation metadata and V5.4 dashboard metadata into read-only "
    "production observability summaries only; it does not emit telemetry, "
    "collect live metrics, capture traces, export events, write persistent "
    "storage, send alerts, request HITL, select or route providers or models, "
    "control or execute workflows, invoke agents, trigger retries, mutate "
    "prompts, or modify generated output."
)

_SOURCE_REGISTRIES = (
    "agent_telemetry_foundation_registry",
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "telemetry_emission",
    "live_metrics_collection",
    "trace_capture",
    "event_export",
    "external_monitoring_sink",
    "persistent_storage_write",
    "alert_emission",
    "hitl_request",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "generated_output_modification",
)


class ProductionTelemetryChannel(BaseModel):
    """One read-only production telemetry channel summary."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    channel_id: str = Field(min_length=1, max_length=180)
    channel_kind: ProductionTelemetryChannelKind
    status: ProductionTelemetryStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=120)
    telemetry_signal_count: int = Field(ge=0, le=500)
    guarded_signal_count: int = Field(ge=0, le=120)
    emitted_event_count: None = None
    exported_event_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    production_telemetry_channel_implemented: Literal[True] = True
    telemetry_emission_implemented: Literal[False] = False
    live_metrics_collection_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    event_export_implemented: Literal[False] = False
    external_monitoring_sink_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["production_telemetry_channel.v1"] = (
        PRODUCTION_TELEMETRY_CHANNEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _channel_matches_contract(self) -> Self:
        if self.channel_id != f"production_telemetry::{self.channel_kind}":
            raise ValueError("channel_id must match channel_kind")
        if self.emitted_event_count is not None:
            raise ValueError("emitted_event_count must remain unset")
        if self.exported_event_count is not None:
            raise ValueError("exported_event_count must remain unset")
        if self.guarded_signal_count > self.telemetry_signal_count:
            raise ValueError("guarded_signal_count must fit telemetry_signal_count")
        if self.channel_kind == "emission_boundary" and (
            self.telemetry_signal_count or self.guarded_signal_count
        ):
            raise ValueError("emission boundary cannot declare telemetry signals")
        if self.channel_kind == "emission_boundary":
            if self.status != "guarded":
                raise ValueError("emission boundary status must be guarded")
        elif self.status != _status_for_guarded_count(self.guarded_signal_count):
            raise ValueError("status must match guarded_signal_count")
        return self


class ProductionTelemetrySurface(BaseModel):
    """Read-only V5.4 production telemetry over existing observability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_telemetry"] = "production_telemetry"
    serialization_version: Literal["production_telemetry.v1"] = (
        PRODUCTION_TELEMETRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_TELEMETRY_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_agent_telemetry_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_token_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_cost_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_quality_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    telemetry_source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    channels: tuple[ProductionTelemetryChannel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    channel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_channel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_channel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    channel_count: int = Field(ge=1, le=8)
    telemetry_signal_count: int = Field(ge=0, le=1000)
    guarded_signal_count: int = Field(ge=0, le=240)
    emitted_event_count: None = None
    exported_event_count: None = None
    production_telemetry_status: ProductionTelemetryStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    production_telemetry_implemented: Literal[True] = True
    telemetry_emission_implemented: Literal[False] = False
    live_metrics_collection_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    event_export_implemented: Literal[False] = False
    external_monitoring_sink_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    alert_emission_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _surface_matches_channels(self) -> Self:
        derived_channel_ids = tuple(channel.channel_id for channel in self.channels)
        if len(set(derived_channel_ids)) != len(derived_channel_ids):
            raise ValueError("channel_ids must be unique")
        if self.channel_ids != derived_channel_ids:
            raise ValueError("channel_ids must match channels")
        if self.channel_count != len(self.channels):
            raise ValueError("channel_count must match channels")
        if self.ready_channel_ids != _channel_ids_for_status(self.channels, "ready"):
            raise ValueError("ready_channel_ids must match channels")
        if self.guarded_channel_ids != _channel_ids_for_status(
            self.channels,
            "guarded",
        ):
            raise ValueError("guarded_channel_ids must match channels")
        if self.telemetry_signal_count != sum(
            channel.telemetry_signal_count for channel in self.channels
        ):
            raise ValueError("telemetry_signal_count must match channels")
        if self.guarded_signal_count != sum(
            channel.guarded_signal_count for channel in self.channels
        ):
            raise ValueError("guarded_signal_count must match channels")
        if self.emitted_event_count is not None:
            raise ValueError("emitted_event_count must remain unset")
        if self.exported_event_count is not None:
            raise ValueError("exported_event_count must remain unset")
        if self.production_telemetry_status != _surface_status(self.channels):
            raise ValueError("production_telemetry_status must match channels")
        if self.telemetry_source_registries != _SOURCE_REGISTRIES:
            raise ValueError("telemetry_source_registries must match sources")
        return self


def build_production_telemetry(
    *,
    agent_telemetry: AgentTelemetryFoundationRegistry | None = None,
    token_dashboard: TokenDashboard | None = None,
    cost_dashboard: CostDashboard | None = None,
    quality_dashboard: QualityDashboard | None = None,
    performance_dashboard: PerformanceDashboard | None = None,
) -> ProductionTelemetrySurface:
    """Build read-only production telemetry metadata without emitting telemetry."""

    agent_source = agent_telemetry or agent_telemetry_foundation_registry()
    token_source = token_dashboard or build_token_dashboard()
    cost_source = cost_dashboard or build_cost_dashboard()
    quality_source = quality_dashboard or build_quality_dashboard()
    performance_source = performance_dashboard or build_performance_dashboard()
    channels = (
        _agent_foundation_channel(agent_source),
        _token_dashboard_channel(token_source),
        _cost_dashboard_channel(cost_source),
        _quality_dashboard_channel(quality_source),
        _performance_dashboard_channel(performance_source),
        _emission_boundary_channel(),
    )

    return ProductionTelemetrySurface(
        source_agent_telemetry_serialization_version=(
            agent_source.serialization_version
        ),
        source_token_dashboard_serialization_version=token_source.serialization_version,
        source_cost_dashboard_serialization_version=cost_source.serialization_version,
        source_quality_dashboard_serialization_version=(
            quality_source.serialization_version
        ),
        source_performance_dashboard_serialization_version=(
            performance_source.serialization_version
        ),
        telemetry_source_registries=_SOURCE_REGISTRIES,
        channels=channels,
        channel_ids=tuple(channel.channel_id for channel in channels),
        ready_channel_ids=_channel_ids_for_status(channels, "ready"),
        guarded_channel_ids=_channel_ids_for_status(channels, "guarded"),
        channel_count=len(channels),
        telemetry_signal_count=sum(
            channel.telemetry_signal_count for channel in channels
        ),
        guarded_signal_count=sum(channel.guarded_signal_count for channel in channels),
        production_telemetry_status=_surface_status(channels),
        advisory_actions=_surface_actions(channels),
    )


def production_telemetry_channel_by_id(
    channel_id: str,
    telemetry: ProductionTelemetrySurface | None = None,
) -> ProductionTelemetryChannel | None:
    """Return one production telemetry channel without exporting events."""

    source_telemetry = telemetry or build_production_telemetry()
    for channel in source_telemetry.channels:
        if channel.channel_id == channel_id:
            return channel
    return None


def production_telemetry_channels_for_status(
    status: ProductionTelemetryStatus,
    telemetry: ProductionTelemetrySurface | None = None,
) -> tuple[ProductionTelemetryChannel, ...]:
    """Return production telemetry channels by status without collection."""

    source_telemetry = telemetry or build_production_telemetry()
    return tuple(
        channel for channel in source_telemetry.channels if channel.status == status
    )


def _agent_foundation_channel(
    registry: AgentTelemetryFoundationRegistry,
) -> ProductionTelemetryChannel:
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::agent_foundation",
        channel_kind="agent_foundation",
        status="ready",
        source_id="agent_telemetry_foundation_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.agent_ids,
        telemetry_signal_count=registry.profile_count,
        guarded_signal_count=0,
        evidence=(
            f"agent_profiles:{registry.profile_count}",
            f"event_types:{len(registry.telemetry_event_types)}",
            f"trace_profiles:{len(registry.trace_profile_ids)}",
        ),
        advisory_actions=(
            "Display passive agent telemetry coverage without emitting events.",
            "Keep trace capture, provenance recording, routing, and workflow control disabled.",
        ),
    )


def _token_dashboard_channel(dashboard: TokenDashboard) -> ProductionTelemetryChannel:
    guarded_count = len(dashboard.guarded_panel_ids)
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::token_dashboard",
        channel_kind="token_dashboard",
        status=_status_for_guarded_count(guarded_count),
        source_id="token_dashboard",
        source_serialization_version=dashboard.serialization_version,
        source_item_ids=dashboard.panel_ids,
        telemetry_signal_count=dashboard.panel_count,
        guarded_signal_count=guarded_count,
        evidence=(
            f"dashboard_pressure:{dashboard.dashboard_pressure}",
            f"planned_tokens:{dashboard.planned_token_total}",
            f"reported_tokens:{dashboard.reported_token_total}",
        ),
        advisory_actions=(
            "Expose token dashboard telemetry without live usage metering.",
            "Keep token collection, enforcement, routing, and workflow execution disabled.",
        ),
    )


def _cost_dashboard_channel(dashboard: CostDashboard) -> ProductionTelemetryChannel:
    guarded_count = len(dashboard.guarded_panel_ids)
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::cost_dashboard",
        channel_kind="cost_dashboard",
        status=_status_for_guarded_count(guarded_count),
        source_id="cost_dashboard",
        source_serialization_version=dashboard.serialization_version,
        source_item_ids=dashboard.panel_ids,
        telemetry_signal_count=dashboard.panel_count,
        guarded_signal_count=guarded_count,
        evidence=(
            f"dashboard_pressure:{dashboard.dashboard_pressure}",
            f"cost_signals:{dashboard.cost_signal_count}",
            f"reported_usd_cost:{dashboard.reported_usd_cost}",
        ),
        advisory_actions=(
            "Expose cost dashboard telemetry without provider pricing lookup.",
            "Keep metering, budget enforcement, routing, and execution blocking disabled.",
        ),
    )


def _quality_dashboard_channel(
    dashboard: QualityDashboard,
) -> ProductionTelemetryChannel:
    guarded_count = len(dashboard.guarded_panel_ids)
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::quality_dashboard",
        channel_kind="quality_dashboard",
        status=_status_for_guarded_count(guarded_count),
        source_id="quality_dashboard",
        source_serialization_version=dashboard.serialization_version,
        source_item_ids=dashboard.panel_ids,
        telemetry_signal_count=dashboard.panel_count,
        guarded_signal_count=guarded_count,
        evidence=(
            f"dashboard_pressure:{dashboard.dashboard_pressure}",
            f"quality_signals:{dashboard.quality_signal_count}",
            f"evaluated_output_score:{dashboard.evaluated_output_score}",
        ),
        advisory_actions=(
            "Expose quality dashboard telemetry without evaluating generated output.",
            "Keep quality scoring, refinement, routing, and workflow execution disabled.",
        ),
    )


def _performance_dashboard_channel(
    dashboard: PerformanceDashboard,
) -> ProductionTelemetryChannel:
    guarded_count = len(dashboard.guarded_panel_ids)
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::performance_dashboard",
        channel_kind="performance_dashboard",
        status=_status_for_guarded_count(guarded_count),
        source_id="performance_dashboard",
        source_serialization_version=dashboard.serialization_version,
        source_item_ids=dashboard.panel_ids,
        telemetry_signal_count=dashboard.panel_count,
        guarded_signal_count=guarded_count,
        evidence=(
            f"dashboard_pressure:{dashboard.dashboard_pressure}",
            f"performance_signals:{dashboard.performance_signal_count}",
            f"measured_latency_ms:{dashboard.measured_latency_ms}",
        ),
        advisory_actions=(
            "Expose performance dashboard telemetry without runtime measurement.",
            "Keep benchmarks, traces, profilers, routing, and workflow control disabled.",
        ),
    )


def _emission_boundary_channel() -> ProductionTelemetryChannel:
    return ProductionTelemetryChannel(
        channel_id="production_telemetry::emission_boundary",
        channel_kind="emission_boundary",
        status="guarded",
        source_id="telemetry_emission_boundary",
        source_serialization_version=TELEMETRY_EMISSION_BOUNDARY_SERIALIZATION_VERSION,
        source_item_ids=("telemetry_emission_disabled",),
        telemetry_signal_count=0,
        guarded_signal_count=0,
        evidence=(
            "emitted_event_count:unavailable",
            "exported_event_count:unavailable",
            "external_monitoring_sink:disabled",
        ),
        advisory_actions=(
            "Keep production telemetry emission empty until runtime sinks are scoped.",
            "Preserve export, alert, storage, routing, workflow, and output boundaries.",
        ),
    )


def _channel_ids_for_status(
    channels: tuple[ProductionTelemetryChannel, ...],
    status: ProductionTelemetryStatus,
) -> tuple[str, ...]:
    return tuple(channel.channel_id for channel in channels if channel.status == status)


def _status_for_guarded_count(guarded_count: int) -> ProductionTelemetryStatus:
    if guarded_count:
        return "guarded"
    return "ready"


def _surface_status(
    channels: tuple[ProductionTelemetryChannel, ...],
) -> ProductionTelemetryStatus:
    if _channel_ids_for_status(channels, "guarded"):
        return "guarded"
    return "ready"


def _surface_actions(
    channels: tuple[ProductionTelemetryChannel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose production telemetry channels as read-only observability metadata.",
        "Preserve telemetry emission, metrics, trace, export, alert, routing, "
        "workflow, storage, and output boundaries.",
    ]
    if _channel_ids_for_status(channels, "guarded"):
        actions.append(
            "Keep guarded telemetry channels non-emitting until explicitly scoped."
        )
    return tuple(actions)
