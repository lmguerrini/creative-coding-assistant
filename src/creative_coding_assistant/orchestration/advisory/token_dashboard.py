"""V5.4 advisory token dashboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.context_budget_planner import ContextBudgetPlan, plan_context_budget
from creative_coding_assistant.orchestration.execution_cost_forecasting import (
    ExecutionCostForecast,
    forecast_execution_cost,
)
from creative_coding_assistant.orchestration.reasoning_budget_optimizer import (
    ReasoningBudgetOptimizationPlan,
    optimize_reasoning_budget,
)

TokenDashboardPanelKind = Literal[
    "context_budget",
    "execution_forecast",
    "reasoning_budget",
    "runtime_usage_boundary",
]
TokenDashboardPressure = Literal["low", "medium", "high", "guarded"]
TokenDashboardStatus = Literal["ready", "review", "guarded"]

TOKEN_DASHBOARD_PANEL_SERIALIZATION_VERSION = "token_dashboard_panel.v1"
TOKEN_DASHBOARD_SERIALIZATION_VERSION = "token_dashboard.v1"
TOKEN_USAGE_BOUNDARY_SERIALIZATION_VERSION = "runtime_token_usage_boundary.v1"
TOKEN_DASHBOARD_AUTHORITY_BOUNDARY = (
    "The V5.4 Token Dashboard converts existing context budget, execution "
    "cost forecast, and reasoning budget metadata into read-only token "
    "dashboard summaries only; it does not meter live usage, collect provider "
    "token telemetry, enforce token budgets, allocate tokens at runtime, trim "
    "context, compress prompts, summarize memory, emit HITL requests, select "
    "or route providers or models, control or execute workflows, invoke "
    "agents, trigger retries, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_usage_metering",
    "provider_token_telemetry_collection",
    "token_budget_enforcement",
    "runtime_token_allocation",
    "context_trimming",
    "prompt_compression",
    "memory_summarization",
    "human_input_request_emission",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "workflow_control",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "retry_or_refinement_triggering",
    "persistent_storage_write",
    "generated_output_modification",
)


class TokenDashboardPanel(BaseModel):
    """One read-only V5.4 token dashboard panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: TokenDashboardPanelKind
    status: TokenDashboardStatus
    pressure: TokenDashboardPressure
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=100)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    planned_token_total: int = Field(ge=0, le=480_000)
    reserve_token_total: int = Field(ge=0, le=240_000)
    reported_token_total: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    token_dashboard_panel_implemented: Literal[True] = True
    live_usage_metering_implemented: Literal[False] = False
    provider_token_collection_implemented: Literal[False] = False
    token_budget_enforcement_implemented: Literal[False] = False
    runtime_token_allocation_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["token_dashboard_panel.v1"] = (
        TOKEN_DASHBOARD_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"token_dashboard::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.reported_token_total is not None:
            raise ValueError("reported_token_total must remain unset")
        if self.status != _status_for_pressure(self.pressure):
            raise ValueError("status must match pressure")
        if self.panel_kind == "runtime_usage_boundary" and (
            self.planned_token_total or self.reserve_token_total
        ):
            raise ValueError("runtime usage boundary cannot declare token totals")
        return self


class TokenDashboard(BaseModel):
    """Read-only V5.4 token dashboard over existing token metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["token_dashboard"] = "token_dashboard"
    serialization_version: Literal["token_dashboard.v1"] = (
        TOKEN_DASHBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=TOKEN_DASHBOARD_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    source_context_budget_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_execution_cost_forecast_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    source_reasoning_budget_serialization_version: str = Field(
        min_length=1,
        max_length=100,
    )
    panels: tuple[TokenDashboardPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    review_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    planned_token_total: int = Field(ge=0, le=960_000)
    reserve_token_total: int = Field(ge=0, le=480_000)
    reported_token_total: None = None
    dashboard_pressure: TokenDashboardPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    token_dashboard_implemented: Literal[True] = True
    live_usage_metering_implemented: Literal[False] = False
    provider_token_collection_implemented: Literal[False] = False
    token_budget_enforcement_implemented: Literal[False] = False
    runtime_token_allocation_implemented: Literal[False] = False
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
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
        if self.planned_token_total != sum(
            panel.planned_token_total for panel in self.panels
        ):
            raise ValueError("planned_token_total must match panels")
        if self.reserve_token_total != sum(
            panel.reserve_token_total for panel in self.panels
        ):
            raise ValueError("reserve_token_total must match panels")
        if self.reported_token_total is not None:
            raise ValueError("reported_token_total must remain unset")
        if self.dashboard_pressure != _dashboard_pressure(self.panels):
            raise ValueError("dashboard_pressure must match panels")
        return self


def build_token_dashboard(
    *,
    context_budget: ContextBudgetPlan | None = None,
    execution_cost_forecast: ExecutionCostForecast | None = None,
    reasoning_budget: ReasoningBudgetOptimizationPlan | None = None,
) -> TokenDashboard:
    """Build read-only token dashboard metadata without live usage metering."""

    context = context_budget or plan_context_budget()
    forecast = execution_cost_forecast or forecast_execution_cost()
    reasoning = reasoning_budget or optimize_reasoning_budget(
        context_budget=context,
    )
    panels = (
        _context_budget_panel(context),
        _execution_forecast_panel(forecast),
        _reasoning_budget_panel(reasoning),
        _runtime_usage_boundary_panel(),
    )

    return TokenDashboard(
        source_context_budget_serialization_version=context.serialization_version,
        source_execution_cost_forecast_serialization_version=(
            forecast.serialization_version
        ),
        source_reasoning_budget_serialization_version=reasoning.serialization_version,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        review_panel_ids=_panel_ids_for_status(panels, "review"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        planned_token_total=sum(panel.planned_token_total for panel in panels),
        reserve_token_total=sum(panel.reserve_token_total for panel in panels),
        dashboard_pressure=_dashboard_pressure(panels),
        advisory_actions=_dashboard_actions(panels),
    )


def token_dashboard_panel_by_id(
    panel_id: str,
    dashboard: TokenDashboard | None = None,
) -> TokenDashboardPanel | None:
    """Return one token dashboard panel without collecting token usage."""

    source_dashboard = dashboard or build_token_dashboard()
    for panel in source_dashboard.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def token_dashboard_panels_for_pressure(
    pressure: TokenDashboardPressure,
    dashboard: TokenDashboard | None = None,
) -> tuple[TokenDashboardPanel, ...]:
    """Return token dashboard panels by pressure without applying budgets."""

    source_dashboard = dashboard or build_token_dashboard()
    return tuple(
        panel for panel in source_dashboard.panels if panel.pressure == pressure
    )


def _context_budget_panel(context: ContextBudgetPlan) -> TokenDashboardPanel:
    return TokenDashboardPanel(
        panel_id="token_dashboard::context_budget",
        panel_kind="context_budget",
        status=_status_for_pressure(context.budget_pressure),
        pressure=context.budget_pressure,
        source_id="context_budget_plan",
        source_serialization_version=context.serialization_version,
        source_item_ids=context.allocation_ids,
        planned_token_total=context.allocated_context_tokens,
        reserve_token_total=context.response_reserve_tokens,
        evidence=(
            f"requested_context_tokens:{context.requested_context_tokens}",
            f"allocated_context_tokens:{context.allocated_context_tokens}",
            f"response_reserve_tokens:{context.response_reserve_tokens}",
        ),
        advisory_actions=(
            "Display context allocation and response reserve as read-only metadata.",
            "Keep context trimming and prompt compression disabled.",
        ),
    )


def _execution_forecast_panel(forecast: ExecutionCostForecast) -> TokenDashboardPanel:
    return TokenDashboardPanel(
        panel_id="token_dashboard::execution_forecast",
        panel_kind="execution_forecast",
        status=_status_for_pressure(forecast.forecast_pressure),
        pressure=forecast.forecast_pressure,
        source_id="execution_cost_forecast",
        source_serialization_version=forecast.serialization_version,
        source_item_ids=forecast.scenario_ids,
        planned_token_total=forecast.worst_case_token_forecast,
        reserve_token_total=0,
        evidence=(
            f"minimum_token_forecast:{forecast.minimum_token_forecast}",
            f"worst_case_token_forecast:{forecast.worst_case_token_forecast}",
            f"pruning_adjusted_token_forecast:{forecast.pruning_adjusted_token_forecast}",
        ),
        advisory_actions=(
            "Display bounded execution token forecasts without live metering.",
            "Keep workflow control and retry triggering disabled.",
        ),
    )


def _reasoning_budget_panel(
    reasoning: ReasoningBudgetOptimizationPlan,
) -> TokenDashboardPanel:
    return TokenDashboardPanel(
        panel_id="token_dashboard::reasoning_budget",
        panel_kind="reasoning_budget",
        status=_status_for_pressure(reasoning.reasoning_budget_pressure),
        pressure=reasoning.reasoning_budget_pressure,
        source_id="reasoning_budget_optimization_plan",
        source_serialization_version=reasoning.serialization_version,
        source_item_ids=reasoning.recommendation_ids,
        planned_token_total=reasoning.total_advisory_reasoning_tokens,
        reserve_token_total=reasoning.total_advisory_reserve_tokens,
        evidence=(
            f"reasoning_tokens:{reasoning.total_advisory_reasoning_tokens}",
            f"reserve_tokens:{reasoning.total_advisory_reserve_tokens}",
            f"review_guardrails:{reasoning.review_guardrail_count}",
        ),
        advisory_actions=(
            "Display reasoning budget recommendations as advisory metadata only.",
            "Require explicit runtime authority before allocating tokens.",
        ),
    )


def _runtime_usage_boundary_panel() -> TokenDashboardPanel:
    return TokenDashboardPanel(
        panel_id="token_dashboard::runtime_usage_boundary",
        panel_kind="runtime_usage_boundary",
        status="guarded",
        pressure="guarded",
        source_id="runtime_token_usage_boundary",
        source_serialization_version=TOKEN_USAGE_BOUNDARY_SERIALIZATION_VERSION,
        source_item_ids=("live_usage_metering_disabled",),
        planned_token_total=0,
        reserve_token_total=0,
        evidence=(
            "reported_token_total:unavailable",
            "live_usage_metering:disabled",
            "provider_token_collection:disabled",
        ),
        advisory_actions=(
            "Keep reported token usage empty until live metering is explicitly scoped.",
            "Preserve provider telemetry collection and storage boundaries.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[TokenDashboardPanel, ...],
    status: TokenDashboardStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_pressure(pressure: TokenDashboardPressure) -> TokenDashboardStatus:
    if pressure == "guarded":
        return "guarded"
    if pressure == "high":
        return "review"
    return "ready"


def _dashboard_pressure(
    panels: tuple[TokenDashboardPanel, ...],
) -> TokenDashboardPressure:
    pressures = tuple(panel.pressure for panel in panels)
    if "guarded" in pressures:
        return "guarded"
    if "high" in pressures:
        return "high"
    if "medium" in pressures:
        return "medium"
    return "low"


def _dashboard_actions(
    panels: tuple[TokenDashboardPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose token dashboard panels as read-only observability metadata.",
        "Preserve live metering, provider collection, budget enforcement, "
        "routing, workflow, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded token panels non-enforcing until explicitly scoped."
        )
    return tuple(actions)
