"""V5.4 advisory creative analytics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import AssistantRequest

from .creative_complexity_analyzer import (
    CreativeComplexityAnalysis,
    analyze_creative_complexity,
)
from .creative_consistency_predictor import (
    CreativeConsistencyPredictionPlan,
    predict_creative_consistency,
)
from .creative_diversity_predictor import (
    CreativeDiversityPredictionPlan,
    predict_creative_diversity,
)
from .creative_quality_prediction import (
    CreativeQualityPrediction,
    derive_creative_quality_prediction,
)
from .creative_score_engine import CreativeScoreProfile, derive_creative_score_profile
from .quality_dashboard import QualityDashboard, build_quality_dashboard
from .system_health_monitoring import (
    SystemHealthMonitoring,
    build_system_health_monitoring,
)

CreativeAnalyticsPanelKind = Literal[
    "quality_readiness",
    "diversity_readiness",
    "consistency_readiness",
    "complexity_profile",
    "score_profile",
    "system_context",
]
CreativeAnalyticsStatus = Literal["ready", "guarded"]

CREATIVE_ANALYTICS_PANEL_SERIALIZATION_VERSION = "creative_analytics_panel.v1"
CREATIVE_ANALYTICS_SERIALIZATION_VERSION = "creative_analytics.v1"
CREATIVE_ANALYTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Creative Analytics surface summarizes quality dashboard, "
    "creative diversity prediction, creative consistency prediction, creative "
    "complexity analysis, creative score profile, and system health metadata "
    "as read-only creative analytics only; it does not collect live creative "
    "metrics, evaluate generated output, execute creative scoring, generate "
    "variants, validate consistency, select artifacts, mutate prompts, control "
    "workflows, route providers or models, invoke agents, trigger retries or "
    "refinement, write memory or storage, modify generated output, or apply "
    "Runtime Evolution."
)

_SOURCE_SURFACES = (
    "quality_dashboard",
    "creative_diversity_prediction_plan",
    "creative_consistency_prediction_plan",
    "creative_complexity_analysis",
    "creative_score_profile",
    "system_health_monitoring",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_metric_collection",
    "generated_output_evaluation",
    "creative_scoring_execution",
    "variant_generation",
    "consistency_validation_execution",
    "artifact_selection",
    "prompt_mutation",
    "workflow_control",
    "provider_or_model_routing",
    "agent_invocation",
    "retry_or_refinement_triggering",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CreativeAnalyticsPanel(BaseModel):
    """One read-only V5.4 creative analytics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: CreativeAnalyticsPanelKind
    status: CreativeAnalyticsStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    creative_signal_count: int = Field(ge=0, le=12000)
    guardrail_signal_count: int = Field(ge=0, le=4000)
    observed_creative_event_count: None = None
    evaluated_output_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    creative_analytics_panel_implemented: Literal[True] = True
    creative_metric_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    creative_scoring_execution_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    consistency_validation_execution_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_analytics_panel.v1"] = (
        CREATIVE_ANALYTICS_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"creative_analytics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.observed_creative_event_count is not None:
            raise ValueError("observed_creative_event_count must remain unset")
        if self.evaluated_output_count is not None:
            raise ValueError("evaluated_output_count must remain unset")
        if self.guardrail_signal_count > self.creative_signal_count:
            raise ValueError("guardrail_signal_count must fit creative_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class CreativeAnalytics(BaseModel):
    """Read-only V5.4 creative analytics over passive creative metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_analytics"] = "creative_analytics"
    serialization_version: Literal["creative_analytics.v1"] = (
        CREATIVE_ANALYTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_ANALYTICS_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    source_quality_dashboard_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_diversity_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_consistency_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_complexity_analysis_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_score_profile_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_system_health_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=6, max_length=6)
    panels: tuple[CreativeAnalyticsPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    creative_signal_count: int = Field(ge=0, le=20000)
    guardrail_signal_count: int = Field(ge=0, le=8000)
    observed_creative_event_count: None = None
    evaluated_output_count: None = None
    creative_analytics_status: CreativeAnalyticsStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    creative_analytics_implemented: Literal[True] = True
    creative_metric_collection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    creative_scoring_execution_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    consistency_validation_execution_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
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
        if self.creative_signal_count != sum(
            panel.creative_signal_count for panel in self.panels
        ):
            raise ValueError("creative_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.observed_creative_event_count is not None:
            raise ValueError("observed_creative_event_count must remain unset")
        if self.evaluated_output_count is not None:
            raise ValueError("evaluated_output_count must remain unset")
        if self.creative_analytics_status != _analytics_status(self.panels):
            raise ValueError("creative_analytics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match creative analytics sources")
        return self


def build_creative_analytics(
    *,
    quality_dashboard: QualityDashboard | None = None,
    diversity_prediction: CreativeDiversityPredictionPlan | None = None,
    consistency_prediction: CreativeConsistencyPredictionPlan | None = None,
    complexity_analysis: CreativeComplexityAnalysis | None = None,
    score_profile: CreativeScoreProfile | None = None,
    system_health: SystemHealthMonitoring | None = None,
) -> CreativeAnalytics:
    """Build read-only creative analytics without evaluating generated output."""

    quality_source = quality_dashboard or build_quality_dashboard()
    diversity_source = diversity_prediction or predict_creative_diversity()
    quality_prediction = _quality_prediction()
    consistency_source = (
        consistency_prediction
        or predict_creative_consistency(creative_quality_prediction=quality_prediction)
    )
    complexity_source = complexity_analysis or analyze_creative_complexity()
    score_source = score_profile or _score_profile()
    system_source = system_health or build_system_health_monitoring()
    panels = (
        _quality_panel(quality_source),
        _diversity_panel(diversity_source),
        _consistency_panel(consistency_source),
        _complexity_panel(complexity_source),
        _score_panel(score_source),
        _system_panel(system_source),
    )

    return CreativeAnalytics(
        source_quality_dashboard_serialization_version=(
            quality_source.serialization_version
        ),
        source_diversity_prediction_serialization_version=(
            diversity_source.serialization_version
        ),
        source_consistency_prediction_serialization_version=(
            consistency_source.serialization_version
        ),
        source_complexity_analysis_serialization_version=(
            complexity_source.serialization_version
        ),
        source_score_profile_serialization_version=score_source.serialization_version,
        source_system_health_serialization_version=system_source.serialization_version,
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        creative_signal_count=sum(panel.creative_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        creative_analytics_status=_analytics_status(panels),
        advisory_actions=_analytics_actions(panels),
    )


def creative_analytics_panel_by_id(
    panel_id: str,
    analytics: CreativeAnalytics | None = None,
) -> CreativeAnalyticsPanel | None:
    """Return one creative analytics panel without runtime evaluation."""

    source_analytics = analytics or build_creative_analytics()
    for panel in source_analytics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def creative_analytics_panels_for_status(
    status: CreativeAnalyticsStatus,
    analytics: CreativeAnalytics | None = None,
) -> tuple[CreativeAnalyticsPanel, ...]:
    """Return creative analytics panels by status without metric collection."""

    source_analytics = analytics or build_creative_analytics()
    return tuple(panel for panel in source_analytics.panels if panel.status == status)


def _quality_panel(source: QualityDashboard) -> CreativeAnalyticsPanel:
    return _panel(
        "quality_readiness",
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
        "Display creative quality readiness without evaluating generated output.",
    )


def _diversity_panel(source: CreativeDiversityPredictionPlan) -> CreativeAnalyticsPanel:
    return _panel(
        "diversity_readiness",
        "creative_diversity_prediction_plan",
        source.serialization_version,
        source.prediction_ids,
        source.prediction_count
        + source.broad_prediction_count
        + source.guarded_prediction_count,
        len(source.blocked_runtime_behaviors),
        (
            f"diversity_predictions:{source.prediction_count}",
            f"recommended_band:{source.recommended_diversity_band}",
            f"readiness:{source.recommended_diversity_readiness_score}",
        ),
        "Display diversity readiness without generating variants.",
    )


def _consistency_panel(
    source: CreativeConsistencyPredictionPlan,
) -> CreativeAnalyticsPanel:
    return _panel(
        "consistency_readiness",
        "creative_consistency_prediction_plan",
        source.serialization_version,
        source.prediction_ids,
        source.prediction_count
        + source.strong_or_stable_prediction_count
        + source.watch_or_fragile_prediction_count
        + source.fragile_prediction_count,
        len(source.blocked_runtime_behaviors),
        (
            f"consistency_predictions:{source.prediction_count}",
            f"recommended_band:{source.recommended_consistency_band}",
            f"midpoint:{source.recommended_consistency_midpoint}",
        ),
        "Display consistency readiness without executing validation.",
    )


def _complexity_panel(source: CreativeComplexityAnalysis) -> CreativeAnalyticsPanel:
    return _panel(
        "complexity_profile",
        "creative_complexity_analysis",
        source.serialization_version,
        source.factor_ids,
        len(source.factor_ids)
        + source.active_intent_dimension_count
        + source.unresolved_gap_count
        + source.hierarchy_conflict_count
        + source.runtime_candidate_count
        + source.tradeoff_risk_count
        + int(source.hitl_advisable),
        len(source.blocked_runtime_behaviors),
        (
            f"complexity_factors:{len(source.factor_ids)}",
            f"complexity_level:{source.creative_complexity_level}",
            f"hitl_advisable:{source.hitl_advisable}",
        ),
        "Display creative complexity profile without mutating prompts or output.",
    )


def _score_panel(source: CreativeScoreProfile) -> CreativeAnalyticsPanel:
    return _panel(
        "score_profile",
        "creative_score_profile",
        source.serialization_version,
        tuple(item.dimension for item in source.score_breakdown),
        len(source.score_breakdown)
        + len(source.score_components)
        + len(source.strengths)
        + len(source.weaknesses)
        + len(source.score_evidence),
        len(_BLOCKED_RUNTIME_BEHAVIORS),
        (
            f"score_band:{source.score_band}",
            f"overall_score:{source.overall_creative_score:.1f}",
            f"hitl:{source.hitl_recommendation}",
        ),
        "Display creative score profile without executing creative scoring.",
    )


def _system_panel(source: SystemHealthMonitoring) -> CreativeAnalyticsPanel:
    return _panel(
        "system_context",
        "system_health_monitoring",
        source.serialization_version,
        source.panel_ids,
        source.system_signal_count + source.panel_count,
        len(source.blocked_runtime_behaviors),
        (
            f"system_panels:{source.panel_count}",
            f"system_signals:{source.system_signal_count}",
            f"status:{source.system_health_status}",
        ),
        "Display system context for creative analytics without live monitoring.",
    )


def _panel(
    panel_kind: CreativeAnalyticsPanelKind,
    source_id: str,
    serialization_version: str,
    item_ids: tuple[str, ...],
    signal_count: int,
    guardrail_count: int,
    evidence: tuple[str, str, str],
    primary_action: str,
) -> CreativeAnalyticsPanel:
    return CreativeAnalyticsPanel(
        panel_id=f"creative_analytics::{panel_kind}",
        panel_kind=panel_kind,
        status=_status_for_guardrails(guardrail_count),
        source_id=source_id,
        source_serialization_version=serialization_version,
        source_item_ids=item_ids,
        creative_signal_count=signal_count + guardrail_count,
        guardrail_signal_count=guardrail_count,
        evidence=evidence,
        advisory_actions=(
            primary_action,
            "Keep metric collection, output evaluation, variant generation, routing, storage, and output mutation disabled.",
        ),
    )


def _quality_prediction() -> CreativeQualityPrediction:
    request = AssistantRequest(query="Design an audio reactive generative garden.")
    return derive_creative_quality_prediction(
        request=request,
        route_decision=None,
        creative_translation=None,
        creative_intent=None,
        creative_hierarchy=None,
        creative_plan=None,
        creative_constraints=None,
        creative_constraint_priorities=None,
        creative_strategy=None,
        creative_techniques=None,
        runtime_capabilities=None,
        creative_tradeoffs=None,
    )


def _score_profile() -> CreativeScoreProfile:
    request = AssistantRequest(query="Design an audio reactive generative garden.")
    return derive_creative_score_profile(
        request=request,
        route_decision=None,
        creative_critic=None,
        self_evaluation=None,
        creative_improvement_planner=None,
        reflection_loop=None,
        creative_confidence=None,
        planning_metadata=(),
    )


def _panel_ids_for_status(
    panels: tuple[CreativeAnalyticsPanel, ...],
    status: CreativeAnalyticsStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> CreativeAnalyticsStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _analytics_status(
    panels: tuple[CreativeAnalyticsPanel, ...],
) -> CreativeAnalyticsStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _analytics_actions(
    panels: tuple[CreativeAnalyticsPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose creative analytics panels as read-only observability metadata.",
        "Preserve creative metric collection, generated-output evaluation, "
        "variant generation, validation, routing, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded creative analytics panels detached from runtime scoring."
        )
    return tuple(actions)
