"""V5.4 advisory creative diversity analytics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .confidence_analytics import ConfidenceAnalytics, build_confidence_analytics
from .creative_analytics import CreativeAnalytics, build_creative_analytics
from .creative_diversity_audit import (
    CreativeDiversityAuditRegistry,
    creative_diversity_audit_registry,
)
from .creative_diversity_predictor import (
    CreativeDiversityPredictionPlan,
    predict_creative_diversity,
)
from .hybrid_agentic_workflow import (
    CreativeExplorationBudgetRegistry,
    creative_exploration_budget_registry,
)
from .system_health_monitoring import (
    SystemHealthMonitoring,
    build_system_health_monitoring,
)

CreativeDiversityAnalyticsPanelKind = Literal[
    "diversity_prediction",
    "diversity_audit_coverage",
    "exploration_budget_posture",
    "creative_diversity_context",
    "confidence_diversity_context",
    "system_diversity_context",
]
CreativeDiversityAnalyticsStatus = Literal["ready", "guarded"]

CREATIVE_DIVERSITY_ANALYTICS_PANEL_SERIALIZATION_VERSION = (
    "creative_diversity_analytics_panel.v1"
)
CREATIVE_DIVERSITY_ANALYTICS_SERIALIZATION_VERSION = "creative_diversity_analytics.v1"
CREATIVE_DIVERSITY_ANALYTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Creative Diversity Analytics surface summarizes creative "
    "diversity prediction metadata, creative diversity audit metadata, "
    "creative exploration budget metadata, creative analytics metadata, "
    "confidence analytics metadata, and system health metadata as read-only "
    "creative diversity analytics only; it does not collect diversity "
    "metrics, enforce budgets, generate variants, select variants or "
    "artifacts, collect creative metrics, evaluate generated output, trigger "
    "refinement, trigger retries, route by cost, request human review, "
    "trigger escalation, invoke agents, route providers or models, control "
    "workflows, mutate prompts, write memory or storage, modify generated "
    "output, or apply Runtime Evolution."
)

_SOURCE_SURFACES = (
    "creative_diversity_prediction_plan",
    "creative_diversity_audit_registry",
    "creative_exploration_budget_registry",
    "creative_analytics",
    "confidence_analytics",
    "system_health_monitoring",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "diversity_metric_collection",
    "budget_enforcement",
    "variant_generation",
    "variant_selection",
    "artifact_selection",
    "creative_metric_collection",
    "generated_output_evaluation",
    "refinement_triggering",
    "retry_triggering",
    "cost_routing",
    "human_review_request",
    "escalation_triggering",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "prompt_mutation",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CreativeDiversityAnalyticsPanel(BaseModel):
    """One read-only V5.4 creative diversity analytics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=200)
    panel_kind: CreativeDiversityAnalyticsPanelKind
    status: CreativeDiversityAnalyticsStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=240)
    diversity_signal_count: int = Field(ge=0, le=50000)
    guardrail_signal_count: int = Field(ge=0, le=20000)
    observed_diversity_event_count: None = None
    generated_variant_count: None = None
    enforced_budget_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    creative_diversity_analytics_panel_implemented: Literal[True] = True
    diversity_metric_collection_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_diversity_analytics_panel.v1"] = (
        CREATIVE_DIVERSITY_ANALYTICS_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"creative_diversity_analytics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_diversity_event_count is not None:
            raise ValueError("observed_diversity_event_count must remain unset")
        if self.generated_variant_count is not None:
            raise ValueError("generated_variant_count must remain unset")
        if self.enforced_budget_count is not None:
            raise ValueError("enforced_budget_count must remain unset")
        if self.guardrail_signal_count > self.diversity_signal_count:
            raise ValueError("guardrail_signal_count must fit diversity_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class CreativeDiversityAnalytics(BaseModel):
    """Read-only V5.4 diversity analytics over passive creative metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_diversity_analytics"] = "creative_diversity_analytics"
    serialization_version: Literal["creative_diversity_analytics.v1"] = (
        CREATIVE_DIVERSITY_ANALYTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_DIVERSITY_ANALYTICS_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    source_diversity_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_diversity_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_exploration_budget_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_analytics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_confidence_analytics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_system_health_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[CreativeDiversityAnalyticsPanel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    diversity_signal_count: int = Field(ge=0, le=100000)
    guardrail_signal_count: int = Field(ge=0, le=40000)
    observed_diversity_event_count: None = None
    generated_variant_count: None = None
    enforced_budget_count: None = None
    creative_diversity_analytics_status: CreativeDiversityAnalyticsStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    creative_diversity_analytics_implemented: Literal[True] = True
    diversity_metric_collection_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
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
        if self.diversity_signal_count != sum(
            panel.diversity_signal_count for panel in self.panels
        ):
            raise ValueError("diversity_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_diversity_event_count is not None:
            raise ValueError("observed_diversity_event_count must remain unset")
        if self.generated_variant_count is not None:
            raise ValueError("generated_variant_count must remain unset")
        if self.enforced_budget_count is not None:
            raise ValueError("enforced_budget_count must remain unset")
        if self.creative_diversity_analytics_status != _analytics_status(self.panels):
            raise ValueError("creative_diversity_analytics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError(
                "source_surfaces must match creative diversity analytics sources"
            )
        return self


def build_creative_diversity_analytics(
    *,
    diversity_prediction: CreativeDiversityPredictionPlan | None = None,
    diversity_audit: CreativeDiversityAuditRegistry | None = None,
    exploration_budget: CreativeExplorationBudgetRegistry | None = None,
    creative_analytics: CreativeAnalytics | None = None,
    confidence_analytics: ConfidenceAnalytics | None = None,
    system_health: SystemHealthMonitoring | None = None,
) -> CreativeDiversityAnalytics:
    """Build read-only creative diversity analytics without generating variants."""

    audit_source = diversity_audit or creative_diversity_audit_registry()
    prediction_source = diversity_prediction or predict_creative_diversity(
        diversity_audit=audit_source,
    )
    budget_source = exploration_budget or creative_exploration_budget_registry()
    creative_source = creative_analytics or build_creative_analytics(
        diversity_prediction=prediction_source,
    )
    confidence_source = confidence_analytics or build_confidence_analytics(
        creative_analytics=creative_source,
    )
    system_source = system_health or build_system_health_monitoring()
    panels = (
        _prediction_panel(prediction_source),
        _audit_panel(audit_source),
        _budget_panel(budget_source),
        _creative_panel(creative_source),
        _confidence_panel(confidence_source),
        _system_panel(system_source),
    )

    return CreativeDiversityAnalytics(
        source_diversity_prediction_serialization_version=(
            prediction_source.serialization_version
        ),
        source_diversity_audit_serialization_version=audit_source.serialization_version,
        source_exploration_budget_serialization_version=(
            budget_source.serialization_version
        ),
        source_creative_analytics_serialization_version=(
            creative_source.serialization_version
        ),
        source_confidence_analytics_serialization_version=(
            confidence_source.serialization_version
        ),
        source_system_health_serialization_version=system_source.serialization_version,
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        diversity_signal_count=sum(panel.diversity_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        creative_diversity_analytics_status=_analytics_status(panels),
        advisory_actions=_analytics_actions(panels),
    )


def creative_diversity_analytics_panel_by_id(
    panel_id: str,
    analytics: CreativeDiversityAnalytics | None = None,
) -> CreativeDiversityAnalyticsPanel | None:
    """Return one creative diversity analytics panel without metric collection."""

    source_analytics = analytics or build_creative_diversity_analytics()
    for panel in source_analytics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def creative_diversity_analytics_panels_for_status(
    status: CreativeDiversityAnalyticsStatus,
    analytics: CreativeDiversityAnalytics | None = None,
) -> tuple[CreativeDiversityAnalyticsPanel, ...]:
    """Return creative diversity analytics panels without generating variants."""

    source_analytics = analytics or build_creative_diversity_analytics()
    return tuple(panel for panel in source_analytics.panels if panel.status == status)


def _prediction_panel(
    source: CreativeDiversityPredictionPlan,
) -> CreativeDiversityAnalyticsPanel:
    return _panel(
        "diversity_prediction",
        "creative_diversity_prediction_plan",
        source.serialization_version,
        source.prediction_ids,
        (
            source.prediction_count
            + source.broad_prediction_count
            + source.guarded_prediction_count
            + len(source.fallback_prediction_ids)
            + source.recommended_diversity_readiness_score
        ),
        len(source.blocked_runtime_behaviors) + source.guarded_prediction_count,
        (
            f"predictions:{source.prediction_count}",
            f"recommended_band:{source.recommended_diversity_band}",
            f"readiness:{source.recommended_diversity_readiness_score}",
        ),
        "Display diversity prediction metadata without generating variants.",
    )


def _audit_panel(
    source: CreativeDiversityAuditRegistry,
) -> CreativeDiversityAnalyticsPanel:
    audit_findings = sum(len(record.audit_findings) for record in source.audit_records)
    return _panel(
        "diversity_audit_coverage",
        "creative_diversity_audit_registry",
        source.serialization_version,
        source.budget_profile_ids,
        (
            source.audit_count
            + len(source.validated_diversity_surfaces)
            + len(source.topic_ids)
            + audit_findings
        ),
        len(source.blocked_runtime_behaviors) + len(source.passive_boundary_flags),
        (
            f"audit_records:{source.audit_count}",
            f"validated_surfaces:{len(source.validated_diversity_surfaces)}",
            f"no_missing_coverage:{source.no_missing_coverage}",
        ),
        "Display diversity audit coverage without active audit behavior.",
    )


def _budget_panel(
    source: CreativeExplorationBudgetRegistry,
) -> CreativeDiversityAnalyticsPanel:
    advisory_bounds = sum(
        profile.max_advisory_variants + profile.max_advisory_refinement_passes
        for profile in source.budget_profiles
    )
    return _panel(
        "exploration_budget_posture",
        "creative_exploration_budget_registry",
        source.serialization_version,
        source.budget_profile_ids,
        (
            source.profile_count
            + len(source.topic_ids)
            + len(source.budget_postures)
            + advisory_bounds
        ),
        len(source.blocked_runtime_behaviors),
        (
            f"budget_profiles:{source.profile_count}",
            f"postures:{','.join(source.budget_postures)}",
            f"advisory_bounds:{advisory_bounds}",
        ),
        "Display exploration budget posture without enforcing budgets.",
    )


def _creative_panel(source: CreativeAnalytics) -> CreativeDiversityAnalyticsPanel:
    return _panel(
        "creative_diversity_context",
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
        "Display creative diversity context without collecting creative metrics.",
    )


def _confidence_panel(
    source: ConfidenceAnalytics,
) -> CreativeDiversityAnalyticsPanel:
    return _panel(
        "confidence_diversity_context",
        "confidence_analytics",
        source.serialization_version,
        source.panel_ids,
        source.confidence_signal_count
        + source.panel_count
        + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"confidence_panels:{source.panel_count}",
            f"confidence_signals:{source.confidence_signal_count}",
            f"status:{source.confidence_analytics_status}",
        ),
        "Display confidence context for diversity without threshold routing.",
    )


def _system_panel(source: SystemHealthMonitoring) -> CreativeDiversityAnalyticsPanel:
    return _panel(
        "system_diversity_context",
        "system_health_monitoring",
        source.serialization_version,
        source.panel_ids,
        source.system_signal_count + source.panel_count + len(source.guarded_panel_ids),
        len(source.blocked_runtime_behaviors),
        (
            f"system_panels:{source.panel_count}",
            f"system_signals:{source.system_signal_count}",
            f"status:{source.system_health_status}",
        ),
        "Display system context for diversity analytics without live monitoring.",
    )


def _panel(
    panel_kind: CreativeDiversityAnalyticsPanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> CreativeDiversityAnalyticsPanel:
    return CreativeDiversityAnalyticsPanel(
        panel_id=f"creative_diversity_analytics::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        diversity_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep diversity metric collection, budget enforcement, variant generation, routing, storage, and output mutation disabled.",  # noqa: E501
        ),
    )


def _panel_ids_for_status(
    panels: tuple[CreativeDiversityAnalyticsPanel, ...],
    status: CreativeDiversityAnalyticsStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> CreativeDiversityAnalyticsStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _analytics_status(
    panels: tuple[CreativeDiversityAnalyticsPanel, ...],
) -> CreativeDiversityAnalyticsStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _analytics_actions(
    panels: tuple[CreativeDiversityAnalyticsPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose creative diversity analytics panels as read-only observability metadata.",
        "Preserve diversity metric collection, budget enforcement, variant "
        "generation, refinement, cost routing, agent, workflow, memory, "
        "storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded creative diversity analytics panels detached from variant generation."
        )
    return tuple(actions)
