"""V5.4 advisory quality dashboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hybrid_studio import QualityProfileRegistry, quality_profile_registry
from .quality_cost_optimizer import (
    QualityCostOptimizationPlan,
    optimize_quality_cost,
)
from .quality_prediction_engine import (
    QualityPredictionPlan,
    predict_quality_for_route,
)

QualityDashboardPanelKind = Literal[
    "quality_profile_coverage",
    "route_quality_prediction",
    "quality_cost_tradeoff",
    "evaluation_boundary",
]
QualityDashboardPressure = Literal["low", "medium", "high", "guarded"]
QualityDashboardStatus = Literal["ready", "review", "guarded"]

QUALITY_DASHBOARD_PANEL_SERIALIZATION_VERSION = "quality_dashboard_panel.v1"
QUALITY_DASHBOARD_SERIALIZATION_VERSION = "quality_dashboard.v1"
QUALITY_EVALUATION_BOUNDARY_SERIALIZATION_VERSION = (
    "generated_output_quality_boundary.v1"
)
QUALITY_DASHBOARD_AUTHORITY_BOUNDARY = (
    "The V5.4 Quality Dashboard converts existing quality profile, quality "
    "prediction, and quality/cost tradeoff metadata into read-only quality "
    "dashboard summaries only; it does not evaluate generated output, "
    "calculate live quality scores, execute quality escalation, trigger "
    "refinement, select or route providers or models, execute providers, "
    "request human input, control or execute workflows, trigger retries, "
    "mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_quality_evaluation",
    "quality_scoring",
    "quality_escalation_execution",
    "refinement_triggering",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "human_input_request_emission",
    "workflow_control",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_QUALITY_UNITS_BY_LEVEL = {
    "low": 35,
    "medium": 55,
    "high": 75,
    "critical": 88,
}


class QualityDashboardPanel(BaseModel):
    """One read-only V5.4 quality dashboard panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: QualityDashboardPanelKind
    status: QualityDashboardStatus
    pressure: QualityDashboardPressure
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    route_name: str | None = Field(default=None, max_length=80)
    recommended_quality_level: str | None = Field(default=None, max_length=80)
    relative_quality_units_total: int = Field(ge=0, le=10_000)
    recommended_quality_units: int = Field(ge=0, le=100)
    quality_signal_count: int = Field(ge=0, le=120)
    evaluated_output_score: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    quality_dashboard_panel_implemented: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["quality_dashboard_panel.v1"] = (
        QUALITY_DASHBOARD_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"quality_dashboard::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.evaluated_output_score is not None:
            raise ValueError("evaluated_output_score must remain unset")
        if self.status != _status_for_pressure(self.pressure):
            raise ValueError("status must match pressure")
        if self.recommended_quality_units > self.relative_quality_units_total:
            raise ValueError("recommended_quality_units must fit total units")
        if self.panel_kind == "evaluation_boundary" and (
            self.relative_quality_units_total
            or self.recommended_quality_units
            or self.quality_signal_count
        ):
            raise ValueError("evaluation boundary cannot declare quality units")
        return self


class QualityDashboard(BaseModel):
    """Read-only V5.4 quality dashboard over existing quality metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["quality_dashboard"] = "quality_dashboard"
    serialization_version: Literal["quality_dashboard.v1"] = (
        QUALITY_DASHBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=QUALITY_DASHBOARD_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_quality_profile_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_quality_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_quality_cost_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    panels: tuple[QualityDashboardPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    review_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    relative_quality_units_total: int = Field(ge=0, le=20_000)
    highest_relative_quality_units: int = Field(ge=0, le=100)
    quality_signal_count: int = Field(ge=0, le=240)
    evaluated_output_score: None = None
    dashboard_pressure: QualityDashboardPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    quality_dashboard_implemented: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
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
        if self.review_panel_ids != _panel_ids_for_status(self.panels, "review"):
            raise ValueError("review_panel_ids must match panels")
        if self.guarded_panel_ids != _panel_ids_for_status(self.panels, "guarded"):
            raise ValueError("guarded_panel_ids must match panels")
        if self.relative_quality_units_total != sum(
            panel.relative_quality_units_total for panel in self.panels
        ):
            raise ValueError("relative_quality_units_total must match panels")
        if self.highest_relative_quality_units != max(
            panel.recommended_quality_units for panel in self.panels
        ):
            raise ValueError("highest_relative_quality_units must match panels")
        if self.quality_signal_count != sum(
            panel.quality_signal_count for panel in self.panels
        ):
            raise ValueError("quality_signal_count must match panels")
        if self.evaluated_output_score is not None:
            raise ValueError("evaluated_output_score must remain unset")
        if self.dashboard_pressure != _dashboard_pressure(self.panels):
            raise ValueError("dashboard_pressure must match panels")
        return self


def build_quality_dashboard(
    *,
    quality_profiles: QualityProfileRegistry | None = None,
    quality_prediction: QualityPredictionPlan | None = None,
    quality_cost: QualityCostOptimizationPlan | None = None,
) -> QualityDashboard:
    """Build read-only quality dashboard metadata without quality evaluation."""

    profiles = quality_profiles or quality_profile_registry()
    prediction = quality_prediction or predict_quality_for_route()
    tradeoff = quality_cost or optimize_quality_cost(route=prediction.route_name)
    panels = (
        _quality_profile_panel(profiles),
        _quality_prediction_panel(prediction),
        _quality_cost_tradeoff_panel(tradeoff),
        _evaluation_boundary_panel(),
    )

    return QualityDashboard(
        source_quality_profile_serialization_version=profiles.serialization_version,
        source_quality_prediction_serialization_version=prediction.serialization_version,
        source_quality_cost_serialization_version=tradeoff.serialization_version,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        review_panel_ids=_panel_ids_for_status(panels, "review"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        relative_quality_units_total=sum(
            panel.relative_quality_units_total for panel in panels
        ),
        highest_relative_quality_units=max(
            panel.recommended_quality_units for panel in panels
        ),
        quality_signal_count=sum(panel.quality_signal_count for panel in panels),
        dashboard_pressure=_dashboard_pressure(panels),
        advisory_actions=_dashboard_actions(panels),
    )


def quality_dashboard_panel_by_id(
    panel_id: str,
    dashboard: QualityDashboard | None = None,
) -> QualityDashboardPanel | None:
    """Return one quality dashboard panel without evaluating output."""

    source_dashboard = dashboard or build_quality_dashboard()
    for panel in source_dashboard.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def quality_dashboard_panels_for_pressure(
    pressure: QualityDashboardPressure,
    dashboard: QualityDashboard | None = None,
) -> tuple[QualityDashboardPanel, ...]:
    """Return quality dashboard panels by pressure without refinement."""

    source_dashboard = dashboard or build_quality_dashboard()
    return tuple(
        panel for panel in source_dashboard.panels if panel.pressure == pressure
    )


def _quality_profile_panel(
    profiles: QualityProfileRegistry,
) -> QualityDashboardPanel:
    units = sum(_quality_units(level) for level in profiles.quality_levels)
    recommended_units = max(_quality_units(level) for level in profiles.quality_levels)
    critical_count = sum(1 for level in profiles.quality_levels if level == "critical")
    return QualityDashboardPanel(
        panel_id="quality_dashboard::quality_profile_coverage",
        panel_kind="quality_profile_coverage",
        status="ready",
        pressure="low",
        source_id="quality_profile_registry",
        source_serialization_version=profiles.serialization_version,
        source_item_ids=profiles.quality_profile_ids,
        route_name=None,
        recommended_quality_level="profile_coverage",
        relative_quality_units_total=units,
        recommended_quality_units=recommended_units,
        quality_signal_count=profiles.profile_count,
        evidence=(
            f"profile_count:{profiles.profile_count}",
            f"critical_profiles:{critical_count}",
            f"levels:{','.join(profiles.quality_levels)}",
        ),
        advisory_actions=(
            "Display passive quality profile coverage as read-only metadata.",
            "Keep quality profile execution disabled.",
        ),
    )


def _quality_prediction_panel(
    prediction: QualityPredictionPlan,
) -> QualityDashboardPanel:
    units = sum(
        decision.predicted_quality_midpoint for decision in prediction.predictions
    )
    pressure = _pressure_from_quality_level(prediction.recommended_quality_level)
    return QualityDashboardPanel(
        panel_id="quality_dashboard::route_quality_prediction",
        panel_kind="route_quality_prediction",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="quality_prediction_plan",
        source_serialization_version=prediction.serialization_version,
        source_item_ids=prediction.prediction_ids,
        route_name=prediction.route_name.value,
        recommended_quality_level=prediction.recommended_quality_level,
        relative_quality_units_total=units,
        recommended_quality_units=prediction.recommended_quality_midpoint,
        quality_signal_count=prediction.prediction_count,
        evidence=(
            f"route:{prediction.route_name.value}",
            f"recommended_quality_level:{prediction.recommended_quality_level}",
            f"recommended_quality_midpoint:{prediction.recommended_quality_midpoint}",
        ),
        advisory_actions=(
            "Display route quality predictions as relative units only.",
            "Keep generated-output evaluation and refinement triggering disabled.",
        ),
    )


def _quality_cost_tradeoff_panel(
    tradeoff: QualityCostOptimizationPlan,
) -> QualityDashboardPanel:
    units = sum(candidate.optimization_score for candidate in tradeoff.candidates)
    pressure = _pressure_from_tradeoff(tradeoff.recommended_tradeoff_posture)
    return QualityDashboardPanel(
        panel_id="quality_dashboard::quality_cost_tradeoff",
        panel_kind="quality_cost_tradeoff",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="quality_cost_optimization_plan",
        source_serialization_version=tradeoff.serialization_version,
        source_item_ids=tradeoff.candidate_ids,
        route_name=tradeoff.route_name.value,
        recommended_quality_level=tradeoff.recommended_tradeoff_posture,
        relative_quality_units_total=units,
        recommended_quality_units=min(100, tradeoff.recommended_optimization_score),
        quality_signal_count=tradeoff.candidate_count,
        evidence=(
            f"route:{tradeoff.route_name.value}",
            f"tradeoff_posture:{tradeoff.recommended_tradeoff_posture}",
            f"optimization_score:{tradeoff.recommended_optimization_score}",
        ),
        advisory_actions=(
            "Display quality/cost tradeoff posture as advisory metadata.",
            "Keep provider/model selection and workflow control disabled.",
        ),
    )


def _evaluation_boundary_panel() -> QualityDashboardPanel:
    return QualityDashboardPanel(
        panel_id="quality_dashboard::evaluation_boundary",
        panel_kind="evaluation_boundary",
        status="guarded",
        pressure="guarded",
        source_id="generated_output_quality_boundary",
        source_serialization_version=QUALITY_EVALUATION_BOUNDARY_SERIALIZATION_VERSION,
        source_item_ids=("generated_output_evaluation_disabled",),
        route_name=None,
        recommended_quality_level=None,
        relative_quality_units_total=0,
        recommended_quality_units=0,
        quality_signal_count=0,
        evidence=(
            "evaluated_output_score:unavailable",
            "generated_output_quality_evaluation:disabled",
            "refinement_triggering:disabled",
        ),
        advisory_actions=(
            "Keep generated-output scoring empty until evaluation is explicitly scoped.",
            "Preserve refinement, workflow, storage, and output mutation boundaries.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[QualityDashboardPanel, ...],
    status: QualityDashboardStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_pressure(pressure: QualityDashboardPressure) -> QualityDashboardStatus:
    if pressure == "guarded":
        return "guarded"
    if pressure == "high":
        return "review"
    return "ready"


def _dashboard_pressure(
    panels: tuple[QualityDashboardPanel, ...],
) -> QualityDashboardPressure:
    pressures = tuple(panel.pressure for panel in panels)
    if "guarded" in pressures:
        return "guarded"
    if "high" in pressures:
        return "high"
    if "medium" in pressures:
        return "medium"
    return "low"


def _quality_units(level: str) -> int:
    return _QUALITY_UNITS_BY_LEVEL.get(level, 0)


def _pressure_from_quality_level(level: str) -> QualityDashboardPressure:
    if level in {"critical", "high"}:
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def _pressure_from_tradeoff(posture: str) -> QualityDashboardPressure:
    if posture == "quality_favored":
        return "low"
    if posture == "balanced_tradeoff":
        return "medium"
    return "high"


def _dashboard_actions(
    panels: tuple[QualityDashboardPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose quality dashboard panels as read-only observability metadata.",
        "Preserve output evaluation, quality scoring, refinement, routing, "
        "workflow, storage, and output mutation boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded quality panels non-evaluating until explicitly scoped."
        )
    return tuple(actions)
