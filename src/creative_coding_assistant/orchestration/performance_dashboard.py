"""V5.4 advisory performance dashboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .performance_benchmarking import (
    PerformanceBenchmarkingPlan,
    plan_performance_benchmarking,
)
from .performance_prediction import PerformancePredictionPlan, predict_performance
from .performance_regression_detection import (
    PerformanceRegressionDetectionPlan,
    detect_performance_regressions,
)
from .resource_utilization_optimizer import (
    ResourceUtilizationOptimizationPlan,
    optimize_resource_utilization,
)

PerformanceDashboardPanelKind = Literal[
    "performance_prediction",
    "benchmarking_readiness",
    "regression_detection",
    "resource_utilization",
    "measurement_boundary",
]
PerformanceDashboardPressure = Literal["low", "medium", "high", "guarded"]
PerformanceDashboardStatus = Literal["ready", "review", "guarded"]

PERFORMANCE_DASHBOARD_PANEL_SERIALIZATION_VERSION = "performance_dashboard_panel.v1"
PERFORMANCE_DASHBOARD_SERIALIZATION_VERSION = "performance_dashboard.v1"
PERFORMANCE_MEASUREMENT_BOUNDARY_SERIALIZATION_VERSION = (
    "runtime_performance_measurement_boundary.v1"
)
PERFORMANCE_DASHBOARD_AUTHORITY_BOUNDARY = (
    "The V5.4 Performance Dashboard converts existing performance prediction, "
    "benchmarking, regression, and resource utilization metadata into "
    "read-only performance dashboard summaries only; it does not measure "
    "runtime performance, execute benchmarks, collect timers, install "
    "profilers, collect traces, detect live regressions, allocate resources, "
    "enforce capacity, route providers or models, control workflows, trigger "
    "retries, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_performance_measurement",
    "benchmark_execution",
    "timer_collection",
    "profiler_hook_installation",
    "runtime_trace_collection",
    "runtime_regression_detection",
    "resource_allocation",
    "capacity_enforcement",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class PerformanceDashboardPanel(BaseModel):
    """One read-only V5.4 performance dashboard panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: PerformanceDashboardPanelKind
    status: PerformanceDashboardStatus
    pressure: PerformanceDashboardPressure
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    relative_performance_units_total: int = Field(ge=0, le=200_000)
    recommended_performance_units: int = Field(ge=0, le=100)
    performance_signal_count: int = Field(ge=0, le=120)
    measured_latency_ms: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_dashboard_panel_implemented: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    runtime_regression_detection_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["performance_dashboard_panel.v1"] = (
        PERFORMANCE_DASHBOARD_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"performance_dashboard::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.measured_latency_ms is not None:
            raise ValueError("measured_latency_ms must remain unset")
        if self.status != _status_for_pressure(self.pressure):
            raise ValueError("status must match pressure")
        if (
            self.recommended_performance_units > self.relative_performance_units_total
            and self.relative_performance_units_total > 0
        ):
            raise ValueError("recommended_performance_units must fit total units")
        if self.panel_kind == "measurement_boundary" and (
            self.relative_performance_units_total
            or self.recommended_performance_units
            or self.performance_signal_count
        ):
            raise ValueError("measurement boundary cannot declare performance units")
        return self


class PerformanceDashboard(BaseModel):
    """Read-only V5.4 dashboard over V5.3 performance metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["performance_dashboard"] = "performance_dashboard"
    serialization_version: Literal["performance_dashboard.v1"] = (
        PERFORMANCE_DASHBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PERFORMANCE_DASHBOARD_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    source_performance_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_benchmarking_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_performance_regression_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_resource_utilization_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    panels: tuple[PerformanceDashboardPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    review_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    relative_performance_units_total: int = Field(ge=0, le=400_000)
    highest_recommended_performance_units: int = Field(ge=0, le=100)
    performance_signal_count: int = Field(ge=0, le=240)
    measured_latency_ms: None = None
    dashboard_pressure: PerformanceDashboardPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=22,
    )
    performance_dashboard_implemented: Literal[True] = True
    runtime_performance_measurement_implemented: Literal[False] = False
    benchmark_execution_implemented: Literal[False] = False
    timer_collection_implemented: Literal[False] = False
    profiler_hook_installation_implemented: Literal[False] = False
    runtime_trace_collection_implemented: Literal[False] = False
    runtime_regression_detection_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    capacity_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
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
        if self.relative_performance_units_total != sum(
            panel.relative_performance_units_total for panel in self.panels
        ):
            raise ValueError("relative_performance_units_total must match panels")
        if self.highest_recommended_performance_units != max(
            panel.recommended_performance_units for panel in self.panels
        ):
            raise ValueError("highest_recommended_performance_units must match panels")
        if self.performance_signal_count != sum(
            panel.performance_signal_count for panel in self.panels
        ):
            raise ValueError("performance_signal_count must match panels")
        if self.measured_latency_ms is not None:
            raise ValueError("measured_latency_ms must remain unset")
        if self.dashboard_pressure != _dashboard_pressure(self.panels):
            raise ValueError("dashboard_pressure must match panels")
        return self


def build_performance_dashboard(
    *,
    performance_prediction: PerformancePredictionPlan | None = None,
    performance_benchmarking: PerformanceBenchmarkingPlan | None = None,
    performance_regression: PerformanceRegressionDetectionPlan | None = None,
    resource_utilization: ResourceUtilizationOptimizationPlan | None = None,
) -> PerformanceDashboard:
    """Build read-only performance dashboard metadata without measurement."""

    prediction = performance_prediction or predict_performance()
    benchmarking = performance_benchmarking or plan_performance_benchmarking(
        performance_prediction=prediction,
    )
    regression = performance_regression or detect_performance_regressions(
        performance_prediction=prediction,
        performance_benchmarking=benchmarking,
    )
    resources = resource_utilization or optimize_resource_utilization(
        performance_benchmarking=benchmarking,
        performance_regression=regression,
    )
    panels = (
        _performance_prediction_panel(prediction),
        _benchmarking_readiness_panel(benchmarking),
        _regression_detection_panel(regression),
        _resource_utilization_panel(resources),
        _measurement_boundary_panel(),
    )

    return PerformanceDashboard(
        source_performance_prediction_serialization_version=(
            prediction.serialization_version
        ),
        source_performance_benchmarking_serialization_version=(
            benchmarking.serialization_version
        ),
        source_performance_regression_serialization_version=(
            regression.serialization_version
        ),
        source_resource_utilization_serialization_version=resources.serialization_version,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        review_panel_ids=_panel_ids_for_status(panels, "review"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        relative_performance_units_total=sum(
            panel.relative_performance_units_total for panel in panels
        ),
        highest_recommended_performance_units=max(
            panel.recommended_performance_units for panel in panels
        ),
        performance_signal_count=sum(
            panel.performance_signal_count for panel in panels
        ),
        dashboard_pressure=_dashboard_pressure(panels),
        advisory_actions=_dashboard_actions(panels),
    )


def performance_dashboard_panel_by_id(
    panel_id: str,
    dashboard: PerformanceDashboard | None = None,
) -> PerformanceDashboardPanel | None:
    """Return one performance dashboard panel without measuring runtime."""

    source_dashboard = dashboard or build_performance_dashboard()
    for panel in source_dashboard.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def performance_dashboard_panels_for_pressure(
    pressure: PerformanceDashboardPressure,
    dashboard: PerformanceDashboard | None = None,
) -> tuple[PerformanceDashboardPanel, ...]:
    """Return performance dashboard panels by pressure without execution."""

    source_dashboard = dashboard or build_performance_dashboard()
    return tuple(
        panel for panel in source_dashboard.panels if panel.pressure == pressure
    )


def _performance_prediction_panel(
    prediction: PerformancePredictionPlan,
) -> PerformanceDashboardPanel:
    units = sum(
        item.predicted_performance_midpoint for item in prediction.predictions
    )
    pressure = _pressure_from_performance_band(
        prediction.recommended_performance_band,
    )
    return PerformanceDashboardPanel(
        panel_id="performance_dashboard::performance_prediction",
        panel_kind="performance_prediction",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="performance_prediction_plan",
        source_serialization_version=prediction.serialization_version,
        source_item_ids=prediction.prediction_ids,
        relative_performance_units_total=units,
        recommended_performance_units=prediction.recommended_performance_midpoint,
        performance_signal_count=prediction.prediction_count,
        evidence=(
            f"recommended_band:{prediction.recommended_performance_band}",
            f"recommended_midpoint:{prediction.recommended_performance_midpoint}",
            f"guarded_predictions:{prediction.guarded_prediction_count}",
        ),
        advisory_actions=(
            "Display relative performance prediction without measurement.",
            "Keep provider routing and workflow control disabled.",
        ),
    )


def _benchmarking_readiness_panel(
    benchmarking: PerformanceBenchmarkingPlan,
) -> PerformanceDashboardPanel:
    pressure = _pressure_from_readiness(benchmarking.benchmarking_readiness)
    return PerformanceDashboardPanel(
        panel_id="performance_dashboard::benchmarking_readiness",
        panel_kind="benchmarking_readiness",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="performance_benchmarking_plan",
        source_serialization_version=benchmarking.serialization_version,
        source_item_ids=benchmarking.scenario_ids,
        relative_performance_units_total=benchmarking.total_advisory_benchmark_units,
        recommended_performance_units=min(
            100,
            max(0, benchmarking.highest_benchmark_priority_score // 30),
        ),
        performance_signal_count=benchmarking.scenario_count,
        evidence=(
            f"readiness:{benchmarking.benchmarking_readiness}",
            f"benchmark_candidates:{benchmarking.benchmark_candidate_count}",
            f"guardrails:{benchmarking.guardrail_count}",
        ),
        advisory_actions=(
            "Display benchmark readiness without executing benchmarks.",
            "Keep timers, profilers, traces, workloads, and replay disabled.",
        ),
    )


def _regression_detection_panel(
    regression: PerformanceRegressionDetectionPlan,
) -> PerformanceDashboardPanel:
    pressure = regression.regression_detection_severity
    return PerformanceDashboardPanel(
        panel_id="performance_dashboard::regression_detection",
        panel_kind="regression_detection",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="performance_regression_detection_plan",
        source_serialization_version=regression.serialization_version,
        source_item_ids=regression.signal_ids,
        relative_performance_units_total=regression.total_advisory_regression_score,
        recommended_performance_units=min(
            100,
            max(0, regression.highest_advisory_regression_score // 30),
        ),
        performance_signal_count=regression.signal_count,
        evidence=(
            f"severity:{regression.regression_detection_severity}",
            f"regression_candidates:{regression.regression_candidate_count}",
            f"baseline_references:{regression.total_baseline_reference_count}",
        ),
        advisory_actions=(
            "Display regression posture without live detection or alerts.",
            "Keep threshold enforcement and workflow blocking disabled.",
        ),
    )


def _resource_utilization_panel(
    resources: ResourceUtilizationOptimizationPlan,
) -> PerformanceDashboardPanel:
    pressure = resources.resource_utilization_pressure
    return PerformanceDashboardPanel(
        panel_id="performance_dashboard::resource_utilization",
        panel_kind="resource_utilization",
        status=_status_for_pressure(pressure),
        pressure=pressure,
        source_id="resource_utilization_optimization_plan",
        source_serialization_version=resources.serialization_version,
        source_item_ids=resources.recommendation_ids,
        relative_performance_units_total=resources.total_advisory_utilization_score,
        recommended_performance_units=min(
            100,
            max(0, resources.highest_advisory_utilization_score // 30),
        ),
        performance_signal_count=resources.recommendation_count,
        evidence=(
            f"pressure:{resources.resource_utilization_pressure}",
            f"resource_units:{resources.total_advisory_resource_units}",
            f"reserve_units:{resources.total_advisory_reserve_units}",
        ),
        advisory_actions=(
            "Display resource utilization posture without allocation.",
            "Keep capacity enforcement, autoscaling, and queue changes disabled.",
        ),
    )


def _measurement_boundary_panel() -> PerformanceDashboardPanel:
    return PerformanceDashboardPanel(
        panel_id="performance_dashboard::measurement_boundary",
        panel_kind="measurement_boundary",
        status="guarded",
        pressure="guarded",
        source_id="runtime_performance_measurement_boundary",
        source_serialization_version=(
            PERFORMANCE_MEASUREMENT_BOUNDARY_SERIALIZATION_VERSION
        ),
        source_item_ids=("runtime_performance_measurement_disabled",),
        relative_performance_units_total=0,
        recommended_performance_units=0,
        performance_signal_count=0,
        evidence=(
            "measured_latency_ms:unavailable",
            "timer_collection:disabled",
            "runtime_trace_collection:disabled",
        ),
        advisory_actions=(
            "Keep measured latency empty until runtime measurement is scoped.",
            "Preserve benchmark, profiler, trace, storage, and workflow boundaries.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[PerformanceDashboardPanel, ...],
    status: PerformanceDashboardStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_pressure(
    pressure: PerformanceDashboardPressure,
) -> PerformanceDashboardStatus:
    if pressure == "guarded":
        return "guarded"
    if pressure == "high":
        return "review"
    return "ready"


def _dashboard_pressure(
    panels: tuple[PerformanceDashboardPanel, ...],
) -> PerformanceDashboardPressure:
    pressures = tuple(panel.pressure for panel in panels)
    if "guarded" in pressures:
        return "guarded"
    if "high" in pressures:
        return "high"
    if "medium" in pressures:
        return "medium"
    return "low"


def _pressure_from_performance_band(band: str) -> PerformanceDashboardPressure:
    if band == "guarded":
        return "guarded"
    if band == "high":
        return "low"
    if band == "medium":
        return "medium"
    return "high"


def _pressure_from_readiness(readiness: str) -> PerformanceDashboardPressure:
    if readiness == "guarded":
        return "guarded"
    if readiness == "high":
        return "low"
    if readiness == "medium":
        return "medium"
    return "high"


def _dashboard_actions(
    panels: tuple[PerformanceDashboardPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose performance dashboard panels as read-only observability metadata.",
        "Preserve measurement, benchmark, trace, regression, resource, routing, "
        "workflow, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded performance panels non-measuring until explicitly scoped."
        )
    return tuple(actions)
