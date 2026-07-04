"""V5.4 advisory cost dashboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget_policies import BudgetPolicyPlan, evaluate_budget_policies
from .cost_estimator import CostEstimationPlan, estimate_routing_cost
from .cost_prediction_engine import CostPredictionPlan, predict_cost_for_route
from .workflow_cost_analyzer import WorkflowCostAnalysis, analyze_workflow_cost

CostDashboardPanelKind = Literal[
    "workflow_cost",
    "routing_cost_estimate",
    "route_cost_prediction",
    "budget_policy",
    "pricing_boundary",
]
CostDashboardPressure = Literal["low", "medium", "high", "guarded"]
CostDashboardStatus = Literal["ready", "review", "guarded"]

COST_DASHBOARD_PANEL_SERIALIZATION_VERSION = "cost_dashboard_panel.v1"
COST_DASHBOARD_SERIALIZATION_VERSION = "cost_dashboard.v1"
PRICING_BOUNDARY_SERIALIZATION_VERSION = "provider_pricing_boundary.v1"
COST_DASHBOARD_AUTHORITY_BOUNDARY = (
    "The V5.4 Cost Dashboard converts existing workflow cost, routing cost "
    "estimate, cost prediction, and budget policy metadata into read-only "
    "cost dashboard summaries only; it does not look up provider pricing, "
    "meter live usage, calculate cost scores, enforce budgets, route by cost, "
    "request HITL, block execution, select or route providers or models, "
    "execute providers, control or execute workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_usage_metering",
    "cost_scoring",
    "budget_enforcement",
    "cost_based_routing",
    "hitl_request",
    "execution_blocking",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class CostDashboardPanel(BaseModel):
    """One read-only V5.4 cost dashboard panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: CostDashboardPanelKind
    status: CostDashboardStatus
    pressure: CostDashboardPressure
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    route_name: str | None = Field(default=None, max_length=80)
    relative_cost_units_total: int = Field(ge=0, le=20_000)
    max_relative_cost_units: int = Field(ge=0, le=20_000)
    cost_signal_count: int = Field(ge=0, le=120)
    reported_usd_cost: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    cost_dashboard_panel_implemented: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["cost_dashboard_panel.v1"] = (
        COST_DASHBOARD_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"cost_dashboard::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.reported_usd_cost is not None:
            raise ValueError("reported_usd_cost must remain unset")
        if self.status != _status_for_pressure(self.pressure):
            raise ValueError("status must match pressure")
        if self.max_relative_cost_units > self.relative_cost_units_total:
            raise ValueError("max_relative_cost_units must fit total units")
        if self.panel_kind == "pricing_boundary" and (
            self.relative_cost_units_total
            or self.max_relative_cost_units
            or self.cost_signal_count
        ):
            raise ValueError("pricing boundary cannot declare cost units")
        return self


class CostDashboard(BaseModel):
    """Read-only V5.4 cost dashboard over existing advisory cost metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cost_dashboard"] = "cost_dashboard"
    serialization_version: Literal["cost_dashboard.v1"] = (
        COST_DASHBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COST_DASHBOARD_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_workflow_cost_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_cost_estimation_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_cost_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_budget_policy_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    panels: tuple[CostDashboardPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    review_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    relative_cost_units_total: int = Field(ge=0, le=40_000)
    highest_relative_cost_units: int = Field(ge=0, le=20_000)
    cost_signal_count: int = Field(ge=0, le=240)
    reported_usd_cost: None = None
    dashboard_pressure: CostDashboardPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    cost_dashboard_implemented: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    hitl_request_implemented: Literal[False] = False
    execution_blocking_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
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
        if self.relative_cost_units_total != sum(
            panel.relative_cost_units_total for panel in self.panels
        ):
            raise ValueError("relative_cost_units_total must match panels")
        if self.highest_relative_cost_units != max(
            panel.max_relative_cost_units for panel in self.panels
        ):
            raise ValueError("highest_relative_cost_units must match panels")
        if self.cost_signal_count != sum(
            panel.cost_signal_count for panel in self.panels
        ):
            raise ValueError("cost_signal_count must match panels")
        if self.reported_usd_cost is not None:
            raise ValueError("reported_usd_cost must remain unset")
        if self.dashboard_pressure != _dashboard_pressure(self.panels):
            raise ValueError("dashboard_pressure must match panels")
        return self


def build_cost_dashboard(
    *,
    workflow_cost: WorkflowCostAnalysis | None = None,
    cost_estimation: CostEstimationPlan | None = None,
    cost_prediction: CostPredictionPlan | None = None,
    budget_policies: BudgetPolicyPlan | None = None,
) -> CostDashboard:
    """Build read-only cost dashboard metadata without pricing lookup."""

    workflow = workflow_cost or analyze_workflow_cost()
    estimation = cost_estimation or estimate_routing_cost()
    prediction = cost_prediction or predict_cost_for_route(
        route=estimation.route_name,
    )
    policies = budget_policies or evaluate_budget_policies(
        cost_estimation=estimation,
    )
    panels = (
        _workflow_cost_panel(workflow),
        _routing_cost_estimate_panel(estimation),
        _route_cost_prediction_panel(prediction),
        _budget_policy_panel(policies),
        _pricing_boundary_panel(),
    )

    return CostDashboard(
        source_workflow_cost_serialization_version=workflow.serialization_version,
        source_cost_estimation_serialization_version=estimation.serialization_version,
        source_cost_prediction_serialization_version=prediction.serialization_version,
        source_budget_policy_serialization_version=policies.serialization_version,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        review_panel_ids=_panel_ids_for_status(panels, "review"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        relative_cost_units_total=sum(
            panel.relative_cost_units_total for panel in panels
        ),
        highest_relative_cost_units=max(
            panel.max_relative_cost_units for panel in panels
        ),
        cost_signal_count=sum(panel.cost_signal_count for panel in panels),
        dashboard_pressure=_dashboard_pressure(panels),
        advisory_actions=_dashboard_actions(panels),
    )


def cost_dashboard_panel_by_id(
    panel_id: str,
    dashboard: CostDashboard | None = None,
) -> CostDashboardPanel | None:
    """Return one cost dashboard panel without applying cost behavior."""

    source_dashboard = dashboard or build_cost_dashboard()
    for panel in source_dashboard.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def cost_dashboard_panels_for_pressure(
    pressure: CostDashboardPressure,
    dashboard: CostDashboard | None = None,
) -> tuple[CostDashboardPanel, ...]:
    """Return cost dashboard panels by pressure without enforcing budgets."""

    source_dashboard = dashboard or build_cost_dashboard()
    return tuple(
        panel for panel in source_dashboard.panels if panel.pressure == pressure
    )


def _workflow_cost_panel(workflow: WorkflowCostAnalysis) -> CostDashboardPanel:
    units = workflow.worst_case_token_estimate // 100
    return CostDashboardPanel(
        panel_id="cost_dashboard::workflow_cost",
        panel_kind="workflow_cost",
        status=_status_for_pressure(workflow.estimated_cost_pressure),
        pressure=workflow.estimated_cost_pressure,
        source_id="workflow_cost_analysis",
        source_serialization_version=workflow.serialization_version,
        source_item_ids=workflow.component_ids,
        route_name=None,
        relative_cost_units_total=units,
        max_relative_cost_units=units,
        cost_signal_count=workflow.component_count,
        evidence=(
            f"worst_case_token_estimate:{workflow.worst_case_token_estimate}",
            f"retry_token_reserve:{workflow.retry_token_reserve}",
            f"cost_sources:{workflow.cost_source_count}",
        ),
        advisory_actions=(
            "Display workflow cost pressure as read-only metadata.",
            "Keep workflow control and provider routing disabled.",
        ),
    )


def _routing_cost_estimate_panel(
    estimation: CostEstimationPlan,
) -> CostDashboardPanel:
    total_units = sum(
        scenario.estimated_midpoint_cost_units for scenario in estimation.scenarios
    )
    pressure = _pressure_from_units(
        estimation.recommended_max_cost_units,
        guarded=False,
    )
    return CostDashboardPanel(
        panel_id="cost_dashboard::routing_cost_estimate",
        panel_kind="routing_cost_estimate",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="cost_estimation_plan",
        source_serialization_version=estimation.serialization_version,
        source_item_ids=estimation.scenario_ids,
        route_name=estimation.route_name.value,
        relative_cost_units_total=total_units,
        max_relative_cost_units=estimation.recommended_max_cost_units,
        cost_signal_count=estimation.scenario_count,
        evidence=(
            f"route:{estimation.route_name.value}",
            f"recommended_max_cost_units:{estimation.recommended_max_cost_units}",
            f"high_cost_scenarios:{estimation.high_cost_scenario_count}",
        ),
        advisory_actions=(
            "Display routing cost estimates as bounded relative units.",
            "Keep provider pricing lookup and route switching disabled.",
        ),
    )


def _route_cost_prediction_panel(
    prediction: CostPredictionPlan,
) -> CostDashboardPanel:
    total_units = sum(
        decision.predicted_cost_midpoint for decision in prediction.predictions
    )
    pressure = _pressure_from_band(prediction.recommended_cost_band)
    return CostDashboardPanel(
        panel_id="cost_dashboard::route_cost_prediction",
        panel_kind="route_cost_prediction",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="cost_prediction_plan",
        source_serialization_version=prediction.serialization_version,
        source_item_ids=prediction.prediction_ids,
        route_name=prediction.route_name.value,
        relative_cost_units_total=total_units,
        max_relative_cost_units=max(
            decision.predicted_cost_midpoint for decision in prediction.predictions
        ),
        cost_signal_count=prediction.prediction_count,
        evidence=(
            f"route:{prediction.route_name.value}",
            f"recommended_cost_band:{prediction.recommended_cost_band}",
            f"recommended_cost_midpoint:{prediction.recommended_cost_midpoint}",
        ),
        advisory_actions=(
            "Display route cost predictions as relative units only.",
            "Keep cost scoring, pricing lookup, and cost routing disabled.",
        ),
    )


def _budget_policy_panel(policies: BudgetPolicyPlan) -> CostDashboardPanel:
    total_units = sum(
        decision.estimated_max_cost_units for decision in policies.decisions
    )
    pressure = _pressure_from_budget_posture(policies.recommended_budget_posture)
    return CostDashboardPanel(
        panel_id="cost_dashboard::budget_policy",
        panel_kind="budget_policy",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="budget_policy_plan",
        source_serialization_version=policies.serialization_version,
        source_item_ids=policies.policy_ids,
        route_name=policies.route_name.value,
        relative_cost_units_total=total_units,
        max_relative_cost_units=max(
            decision.estimated_max_cost_units for decision in policies.decisions
        ),
        cost_signal_count=policies.policy_count,
        evidence=(
            f"route:{policies.route_name.value}",
            f"budget_posture:{policies.recommended_budget_posture}",
            f"review_recommended:{policies.review_recommended_count}",
        ),
        advisory_actions=(
            "Display budget posture without enforcement or HITL emission.",
            "Keep execution blocking and provider selection disabled.",
        ),
    )


def _pricing_boundary_panel() -> CostDashboardPanel:
    return CostDashboardPanel(
        panel_id="cost_dashboard::pricing_boundary",
        panel_kind="pricing_boundary",
        status="guarded",
        pressure="guarded",
        source_id="provider_pricing_boundary",
        source_serialization_version=PRICING_BOUNDARY_SERIALIZATION_VERSION,
        source_item_ids=("provider_pricing_lookup_disabled",),
        route_name=None,
        relative_cost_units_total=0,
        max_relative_cost_units=0,
        cost_signal_count=0,
        evidence=(
            "reported_usd_cost:unavailable",
            "provider_pricing_lookup:disabled",
            "live_usage_metering:disabled",
        ),
        advisory_actions=(
            "Keep reported USD cost empty until provider pricing is explicitly scoped.",
            "Preserve pricing lookup, metering, and storage boundaries.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[CostDashboardPanel, ...],
    status: CostDashboardStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_pressure(pressure: CostDashboardPressure) -> CostDashboardStatus:
    if pressure == "guarded":
        return "guarded"
    if pressure == "high":
        return "review"
    return "ready"


def _dashboard_pressure(
    panels: tuple[CostDashboardPanel, ...],
) -> CostDashboardPressure:
    pressures = tuple(panel.pressure for panel in panels)
    if "guarded" in pressures:
        return "guarded"
    if "high" in pressures:
        return "high"
    if "medium" in pressures:
        return "medium"
    return "low"


def _pressure_from_units(
    units: int,
    *,
    guarded: bool,
) -> CostDashboardPressure:
    if guarded:
        return "guarded"
    if units >= 7:
        return "high"
    if units >= 4:
        return "medium"
    return "low"


def _pressure_from_band(band: str) -> CostDashboardPressure:
    if band == "guarded":
        return "guarded"
    if band == "high":
        return "high"
    if band == "medium":
        return "medium"
    return "low"


def _pressure_from_budget_posture(posture: str) -> CostDashboardPressure:
    if posture == "over_budget":
        return "guarded"
    if posture == "review_recommended":
        return "high"
    return "low"


def _dashboard_actions(
    panels: tuple[CostDashboardPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose cost dashboard panels as read-only observability metadata.",
        "Preserve pricing lookup, metering, budget enforcement, cost routing, "
        "workflow, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded cost panels non-enforcing until explicitly scoped."
        )
    return tuple(actions)
