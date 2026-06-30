"""V5.4 advisory confidence analytics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import AssistantRequest

from .creative_analytics import CreativeAnalytics, build_creative_analytics
from .creative_confidence_engine import (
    CreativeConfidenceProfile,
    derive_creative_confidence_profile,
)
from .escalation_diagnostics import (
    EscalationDiagnostics,
    build_escalation_diagnostics,
)
from .hybrid_agentic_workflow import (
    AgentConfidenceFusionRegistry,
    ConfidenceThresholdRoutingRegistry,
    agent_confidence_fusion_registry,
    confidence_threshold_routing_registry,
)
from .quality_dashboard import QualityDashboard, build_quality_dashboard

ConfidenceAnalyticsPanelKind = Literal[
    "confidence_profile",
    "agent_confidence_fusion",
    "confidence_thresholds",
    "quality_confidence_context",
    "creative_confidence_context",
    "escalation_confidence_context",
]
ConfidenceAnalyticsStatus = Literal["ready", "guarded"]

CONFIDENCE_ANALYTICS_PANEL_SERIALIZATION_VERSION = "confidence_analytics_panel.v1"
CONFIDENCE_ANALYTICS_SERIALIZATION_VERSION = "confidence_analytics.v1"
CONFIDENCE_ANALYTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Confidence Analytics surface summarizes creative confidence "
    "profile metadata, passive agent confidence fusion metadata, passive "
    "confidence threshold routing metadata, quality dashboard metadata, "
    "creative analytics metadata, and escalation diagnostics metadata as "
    "read-only confidence analytics only; it does not calculate confidence "
    "scores, evaluate confidence thresholds, route by confidence, execute "
    "agent confidence fusion, evaluate generated output, score quality, "
    "collect creative metrics, request human review, trigger escalation, "
    "invoke agents, route providers or models, control workflows, trigger "
    "retries or refinement, write memory or storage, modify generated output, "
    "or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "creative_confidence_profile",
    "agent_confidence_fusion_registry",
    "confidence_threshold_routing_registry",
    "quality_dashboard",
    "creative_analytics",
    "escalation_diagnostics",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "confidence_score_calculation",
    "confidence_threshold_evaluation",
    "confidence_based_routing",
    "agent_confidence_fusion_execution",
    "generated_output_evaluation",
    "quality_scoring",
    "creative_metric_collection",
    "human_review_request",
    "escalation_triggering",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ConfidenceAnalyticsPanel(BaseModel):
    """One read-only V5.4 confidence analytics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: ConfidenceAnalyticsPanelKind
    status: ConfidenceAnalyticsStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=220)
    confidence_signal_count: int = Field(ge=0, le=30000)
    guardrail_signal_count: int = Field(ge=0, le=10000)
    calculated_confidence_score: None = None
    evaluated_threshold_count: None = None
    routed_confidence_decision_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    confidence_analytics_panel_implemented: Literal[True] = True
    confidence_score_calculation_implemented: Literal[False] = False
    confidence_threshold_evaluation_implemented: Literal[False] = False
    confidence_based_routing_implemented: Literal[False] = False
    agent_confidence_fusion_execution_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["confidence_analytics_panel.v1"] = (
        CONFIDENCE_ANALYTICS_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"confidence_analytics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.calculated_confidence_score is not None:
            raise ValueError("calculated_confidence_score must remain unset")
        if self.evaluated_threshold_count is not None:
            raise ValueError("evaluated_threshold_count must remain unset")
        if self.routed_confidence_decision_count is not None:
            raise ValueError("routed_confidence_decision_count must remain unset")
        if self.guardrail_signal_count > self.confidence_signal_count:
            raise ValueError("guardrail_signal_count must fit confidence_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class ConfidenceAnalytics(BaseModel):
    """Read-only V5.4 confidence analytics over passive confidence metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["confidence_analytics"] = "confidence_analytics"
    serialization_version: Literal["confidence_analytics.v1"] = (
        CONFIDENCE_ANALYTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONFIDENCE_ANALYTICS_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    source_confidence_profile_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_confidence_fusion_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_confidence_threshold_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_quality_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_analytics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_diagnostics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[ConfidenceAnalyticsPanel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    confidence_signal_count: int = Field(ge=0, le=60000)
    guardrail_signal_count: int = Field(ge=0, le=20000)
    calculated_confidence_score: None = None
    evaluated_threshold_count: None = None
    routed_confidence_decision_count: None = None
    confidence_analytics_status: ConfidenceAnalyticsStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    confidence_analytics_implemented: Literal[True] = True
    confidence_score_calculation_implemented: Literal[False] = False
    confidence_threshold_evaluation_implemented: Literal[False] = False
    confidence_based_routing_implemented: Literal[False] = False
    agent_confidence_fusion_execution_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _analytics_matches_panels(self) -> Self:
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
        if self.confidence_signal_count != sum(
            panel.confidence_signal_count for panel in self.panels
        ):
            raise ValueError("confidence_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.calculated_confidence_score is not None:
            raise ValueError("calculated_confidence_score must remain unset")
        if self.evaluated_threshold_count is not None:
            raise ValueError("evaluated_threshold_count must remain unset")
        if self.routed_confidence_decision_count is not None:
            raise ValueError("routed_confidence_decision_count must remain unset")
        if self.confidence_analytics_status != _analytics_status(self.panels):
            raise ValueError("confidence_analytics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match confidence analytics sources")
        return self


def build_confidence_analytics(
    *,
    confidence_profile: CreativeConfidenceProfile | None = None,
    agent_confidence_fusion: AgentConfidenceFusionRegistry | None = None,
    confidence_threshold_routing: ConfidenceThresholdRoutingRegistry | None = None,
    quality_dashboard: QualityDashboard | None = None,
    creative_analytics: CreativeAnalytics | None = None,
    escalation_diagnostics: EscalationDiagnostics | None = None,
) -> ConfidenceAnalytics:
    """Build read-only confidence analytics without evaluating confidence."""

    confidence_source = confidence_profile or _confidence_profile()
    fusion_source = agent_confidence_fusion or agent_confidence_fusion_registry()
    threshold_source = (
        confidence_threshold_routing or confidence_threshold_routing_registry()
    )
    quality_source = quality_dashboard or build_quality_dashboard()
    creative_source = creative_analytics or build_creative_analytics()
    escalation_source = escalation_diagnostics or build_escalation_diagnostics()
    panels = (
        _confidence_profile_panel(confidence_source),
        _fusion_panel(fusion_source),
        _threshold_panel(threshold_source),
        _quality_panel(quality_source),
        _creative_panel(creative_source),
        _escalation_panel(escalation_source),
    )

    return ConfidenceAnalytics(
        source_confidence_profile_serialization_version=(
            confidence_source.serialization_version
        ),
        source_agent_confidence_fusion_serialization_version=(
            fusion_source.serialization_version
        ),
        source_confidence_threshold_serialization_version=(
            threshold_source.serialization_version
        ),
        source_quality_dashboard_serialization_version=quality_source.serialization_version,
        source_creative_analytics_serialization_version=(
            creative_source.serialization_version
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
        confidence_signal_count=sum(
            panel.confidence_signal_count for panel in panels
        ),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        confidence_analytics_status=_analytics_status(panels),
        advisory_actions=_analytics_actions(panels),
    )


def confidence_analytics_panel_by_id(
    panel_id: str,
    analytics: ConfidenceAnalytics | None = None,
) -> ConfidenceAnalyticsPanel | None:
    """Return one confidence analytics panel without threshold evaluation."""

    source_analytics = analytics or build_confidence_analytics()
    for panel in source_analytics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def confidence_analytics_panels_for_status(
    status: ConfidenceAnalyticsStatus,
    analytics: ConfidenceAnalytics | None = None,
) -> tuple[ConfidenceAnalyticsPanel, ...]:
    """Return confidence analytics panels by status without runtime routing."""

    source_analytics = analytics or build_confidence_analytics()
    return tuple(panel for panel in source_analytics.panels if panel.status == status)


def _confidence_profile_panel(
    source: CreativeConfidenceProfile,
) -> ConfidenceAnalyticsPanel:
    guardrails = (
        len(source.confidence_limitations)
        + len(source.confidence_uncertainties)
        + int(source.hitl_recommendation in {"recommended", "required"})
        + 1
    )
    return _panel(
        "confidence_profile",
        "creative_confidence_profile",
        source.serialization_version,
        tuple(f"confidence_component::{item.source}" for item in source.confidence_components),
        (
            len(source.confidence_components)
            + len(source.confidence_rationale)
            + len(source.confidence_evidence)
            + len(source.confidence_strengths)
            + len(source.confidence_weaknesses)
            + len(source.prompt_guidance)
        ),
        guardrails,
        (
            f"confidence_level:{source.confidence_level}",
            f"confidence_score_band:{source.confidence_score:.2f}",
            f"hitl:{source.hitl_recommendation}",
        ),
        "Display confidence profile metadata without recalculating confidence.",
    )


def _fusion_panel(
    source: AgentConfidenceFusionRegistry,
) -> ConfidenceAnalyticsPanel:
    signal_count = source.profile_count + len(source.topic_ids)
    signal_count += sum(
        len(profile.confidence_signal_inputs)
        + len(profile.fusion_dimensions)
        + len(profile.advisory_outputs)
        for profile in source.fusion_profiles
    )
    return _panel(
        "agent_confidence_fusion",
        "agent_confidence_fusion_registry",
        source.serialization_version,
        source.fusion_profile_ids,
        signal_count,
        len(source.blocked_runtime_behaviors),
        (
            f"fusion_profiles:{source.profile_count}",
            f"topics:{len(source.topic_ids)}",
            f"confidence_surfaces:{len(source.confidence_surface_ids)}",
        ),
        "Display agent confidence fusion metadata without executing fusion.",
    )


def _threshold_panel(
    source: ConfidenceThresholdRoutingRegistry,
) -> ConfidenceAnalyticsPanel:
    signal_count = (
        source.profile_count
        + len(source.confidence_bands)
        + len(source.escalation_signal_ids)
    )
    signal_count += sum(
        len(profile.routing_dimensions) + len(profile.advisory_outputs)
        for profile in source.threshold_profiles
    )
    return _panel(
        "confidence_thresholds",
        "confidence_threshold_routing_registry",
        source.serialization_version,
        source.threshold_profile_ids,
        signal_count,
        len(source.blocked_runtime_behaviors),
        (
            f"threshold_profiles:{source.profile_count}",
            f"confidence_bands:{len(source.confidence_bands)}",
            f"escalation_signals:{len(source.escalation_signal_ids)}",
        ),
        "Display confidence threshold metadata without evaluating thresholds.",
    )


def _quality_panel(source: QualityDashboard) -> ConfidenceAnalyticsPanel:
    return _panel(
        "quality_confidence_context",
        "quality_dashboard",
        source.serialization_version,
        source.panel_ids,
        source.quality_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"quality_panels:{source.panel_count}",
            f"quality_signals:{source.quality_signal_count}",
            f"pressure:{source.dashboard_pressure}",
        ),
        "Display quality confidence context without quality scoring.",
    )


def _creative_panel(source: CreativeAnalytics) -> ConfidenceAnalyticsPanel:
    return _panel(
        "creative_confidence_context",
        "creative_analytics",
        source.serialization_version,
        source.panel_ids,
        source.creative_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"creative_panels:{source.panel_count}",
            f"creative_signals:{source.creative_signal_count}",
            f"status:{source.creative_analytics_status}",
        ),
        "Display creative confidence context without creative metric collection.",
    )


def _escalation_panel(source: EscalationDiagnostics) -> ConfidenceAnalyticsPanel:
    return _panel(
        "escalation_confidence_context",
        "escalation_diagnostics",
        source.serialization_version,
        source.panel_ids,
        source.escalation_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"escalation_panels:{source.panel_count}",
            f"escalation_signals:{source.escalation_signal_count}",
            f"status:{source.escalation_diagnostics_status}",
        ),
        "Display escalation confidence context without triggering escalation.",
    )


def _panel(
    panel_kind: ConfidenceAnalyticsPanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> ConfidenceAnalyticsPanel:
    return ConfidenceAnalyticsPanel(
        panel_id=f"confidence_analytics::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        confidence_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep confidence calculation, threshold evaluation, routing, HITL, storage, and output mutation disabled.",
        ),
    )


def _confidence_profile() -> CreativeConfidenceProfile:
    request = AssistantRequest(query="Design an audio reactive generative garden.")
    return derive_creative_confidence_profile(
        request=request,
        route_decision=None,
        creative_critic=None,
        self_evaluation=None,
        creative_improvement_planner=None,
        reflection_loop=None,
        planning_metadata=(),
    )


def _panel_ids_for_status(
    panels: tuple[ConfidenceAnalyticsPanel, ...],
    status: ConfidenceAnalyticsStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> ConfidenceAnalyticsStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _analytics_status(
    panels: tuple[ConfidenceAnalyticsPanel, ...],
) -> ConfidenceAnalyticsStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _analytics_actions(
    panels: tuple[ConfidenceAnalyticsPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose confidence analytics panels as read-only observability metadata.",
        "Preserve confidence calculation, threshold evaluation, confidence "
        "routing, HITL, escalation, agent, workflow, memory, storage, and "
        "output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded confidence analytics panels detached from runtime routing."
        )
    return tuple(actions)
