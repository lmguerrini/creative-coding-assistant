"""V5.1 execution cost forecasting over bounded advisory cost metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
    analyze_workflow_cost,
)
from creative_coding_assistant.orchestration.workflow_pruning import (
    WorkflowPruningPlan,
    plan_workflow_pruning,
)

ExecutionCostForecastScenarioKind = Literal[
    "minimum_success_path",
    "single_retry_path",
    "worst_case_bound",
    "pruning_adjusted_bound",
]
ExecutionCostForecastRelativeClass = Literal["low", "medium", "high"]
ExecutionCostForecastConfidence = Literal["low", "medium", "high"]
ExecutionCostForecastPressure = Literal["low", "medium", "high"]

EXECUTION_COST_FORECAST_SCENARIO_SERIALIZATION_VERSION = (
    "execution_cost_forecast_scenario.v1"
)
EXECUTION_COST_FORECAST_SERIALIZATION_VERSION = "execution_cost_forecast.v1"
EXECUTION_COST_FORECAST_AUTHORITY_BOUNDARY = (
    "Execution cost forecasting derives bounded advisory token forecast "
    "scenarios from workflow cost and pruning metadata only; it does not look "
    "up provider pricing, meter live usage, enforce budgets, route by cost, "
    "apply pruning, select providers or models, control workflow execution, "
    "trigger retries, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_usage_metering",
    "budget_enforcement",
    "cost_based_routing",
    "workflow_pruning_application",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionCostForecastScenario(BaseModel):
    """One bounded advisory execution cost forecast scenario."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1, max_length=180)
    scenario_kind: ExecutionCostForecastScenarioKind
    source_id: str = Field(min_length=1, max_length=180)
    lower_bound_tokens: int = Field(ge=0, le=240_000)
    forecast_tokens: int = Field(ge=0, le=240_000)
    upper_bound_tokens: int = Field(ge=0, le=240_000)
    worst_case_token_estimate: int = Field(ge=1, le=240_000)
    token_delta_from_worst_case: int = Field(ge=-240_000, le=240_000)
    relative_cost: ExecutionCostForecastRelativeClass
    confidence: ExecutionCostForecastConfidence
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    execution_cost_forecasting_implemented: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    workflow_pruning_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_cost_forecast_scenario.v1"] = (
        EXECUTION_COST_FORECAST_SCENARIO_SERIALIZATION_VERSION
    )
    forecasting_only: Literal[True] = True

    @model_validator(mode="after")
    def _scenario_matches_bounds(self) -> Self:
        if self.lower_bound_tokens > self.forecast_tokens:
            raise ValueError("lower_bound_tokens must not exceed forecast_tokens")
        if self.forecast_tokens > self.upper_bound_tokens:
            raise ValueError("forecast_tokens must not exceed upper_bound_tokens")
        expected_delta = self.forecast_tokens - self.worst_case_token_estimate
        if self.token_delta_from_worst_case != expected_delta:
            raise ValueError("token_delta_from_worst_case must match forecast")
        if self.scenario_kind == "worst_case_bound" and expected_delta != 0:
            raise ValueError("worst_case_bound must match worst case tokens")
        return self


class ExecutionCostForecast(BaseModel):
    """Bounded V5.1 advisory execution cost forecast."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_cost_forecaster"] = "execution_cost_forecaster"
    serialization_version: Literal["execution_cost_forecast.v1"] = (
        EXECUTION_COST_FORECAST_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_COST_FORECAST_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_cost_serialization_version: str = Field(min_length=1, max_length=80)
    source_pruning_serialization_version: str = Field(min_length=1, max_length=80)
    scenarios: tuple[ExecutionCostForecastScenario, ...] = Field(
        min_length=1,
        max_length=12,
    )
    scenario_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    scenario_count: int = Field(ge=1, le=12)
    minimum_token_forecast: int = Field(ge=0, le=240_000)
    single_retry_token_forecast: int = Field(ge=0, le=240_000)
    worst_case_token_forecast: int = Field(ge=0, le=240_000)
    pruning_adjusted_token_forecast: int = Field(ge=0, le=240_000)
    forecast_token_spread: int = Field(ge=0, le=240_000)
    forecast_pressure: ExecutionCostForecastPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    execution_cost_forecasting_implemented: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    workflow_pruning_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    forecasting_only: Literal[True] = True

    @model_validator(mode="after")
    def _forecast_matches_scenarios(self) -> Self:
        derived_scenario_ids = tuple(
            scenario.scenario_id for scenario in self.scenarios
        )
        if len(set(derived_scenario_ids)) != len(derived_scenario_ids):
            raise ValueError("scenario_ids must be unique")
        if self.scenario_ids != derived_scenario_ids:
            raise ValueError("scenario_ids must match scenarios")
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario_count must match scenarios")

        minimum = _scenario_for_kind(self.scenarios, "minimum_success_path")
        single_retry = _scenario_for_kind(self.scenarios, "single_retry_path")
        worst_case = _scenario_for_kind(self.scenarios, "worst_case_bound")
        pruning_adjusted = _scenario_for_kind(
            self.scenarios,
            "pruning_adjusted_bound",
        )
        if self.minimum_token_forecast != minimum.forecast_tokens:
            raise ValueError("minimum_token_forecast must match scenario")
        if self.single_retry_token_forecast != single_retry.forecast_tokens:
            raise ValueError("single_retry_token_forecast must match scenario")
        if self.worst_case_token_forecast != worst_case.forecast_tokens:
            raise ValueError("worst_case_token_forecast must match scenario")
        if self.pruning_adjusted_token_forecast != pruning_adjusted.forecast_tokens:
            raise ValueError("pruning_adjusted_token_forecast must match scenario")

        forecasts = tuple(scenario.forecast_tokens for scenario in self.scenarios)
        expected_spread = max(forecasts) - min(forecasts)
        if self.forecast_token_spread != expected_spread:
            raise ValueError("forecast_token_spread must match scenarios")
        if self.forecast_pressure != _forecast_pressure(
            self.worst_case_token_forecast,
            self.pruning_adjusted_token_forecast,
        ):
            raise ValueError("forecast_pressure must match token forecasts")
        return self


def forecast_execution_cost(
    *,
    cost_analysis: WorkflowCostAnalysis | None = None,
    pruning_plan: WorkflowPruningPlan | None = None,
) -> ExecutionCostForecast:
    """Forecast bounded execution cost scenarios without runtime cost control."""

    costs = cost_analysis or analyze_workflow_cost()
    pruning = pruning_plan or plan_workflow_pruning(cost_analysis=costs)
    scenarios = _scenarios(costs=costs, pruning=pruning)
    forecasts = tuple(scenario.forecast_tokens for scenario in scenarios)

    return ExecutionCostForecast(
        source_cost_serialization_version=costs.serialization_version,
        source_pruning_serialization_version=pruning.serialization_version,
        scenarios=scenarios,
        scenario_ids=tuple(scenario.scenario_id for scenario in scenarios),
        scenario_count=len(scenarios),
        minimum_token_forecast=_scenario_for_kind(
            scenarios,
            "minimum_success_path",
        ).forecast_tokens,
        single_retry_token_forecast=_scenario_for_kind(
            scenarios,
            "single_retry_path",
        ).forecast_tokens,
        worst_case_token_forecast=_scenario_for_kind(
            scenarios,
            "worst_case_bound",
        ).forecast_tokens,
        pruning_adjusted_token_forecast=_scenario_for_kind(
            scenarios,
            "pruning_adjusted_bound",
        ).forecast_tokens,
        forecast_token_spread=max(forecasts) - min(forecasts),
        forecast_pressure=_forecast_pressure(
            costs.worst_case_token_estimate,
            _scenario_for_kind(scenarios, "pruning_adjusted_bound").forecast_tokens,
        ),
        advisory_actions=_forecast_actions(costs, pruning),
    )


def execution_cost_forecast_scenario_by_id(
    scenario_id: str,
    forecast: ExecutionCostForecast | None = None,
) -> ExecutionCostForecastScenario | None:
    """Return one forecast scenario without provider pricing lookup."""

    source_forecast = forecast or forecast_execution_cost()
    for scenario in source_forecast.scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


def execution_cost_forecast_scenarios_for_kind(
    scenario_kind: ExecutionCostForecastScenarioKind,
    forecast: ExecutionCostForecast | None = None,
) -> tuple[ExecutionCostForecastScenario, ...]:
    """Return forecast scenarios by kind without budget enforcement."""

    source_forecast = forecast or forecast_execution_cost()
    return tuple(
        scenario
        for scenario in source_forecast.scenarios
        if scenario.scenario_kind == scenario_kind
    )


def _scenarios(
    *,
    costs: WorkflowCostAnalysis,
    pruning: WorkflowPruningPlan,
) -> tuple[ExecutionCostForecastScenario, ...]:
    minimum = costs.critical_path_token_estimate
    single_retry = minimum + min(
        costs.retry_iteration_token_estimate,
        costs.retry_token_reserve,
    )
    worst_case = costs.worst_case_token_estimate
    pruning_adjusted = max(
        costs.critical_path_token_estimate + costs.failure_path_token_reserve,
        worst_case - pruning.estimated_token_savings,
    )

    return (
        _scenario(
            scenario_kind="minimum_success_path",
            source_id="critical_path",
            lower=minimum,
            forecast=minimum,
            upper=minimum,
            worst_case=worst_case,
            confidence="high",
            evidence=(
                f"critical_path_tokens:{costs.critical_path_token_estimate}",
                f"node_count:{costs.node_count}",
            ),
        ),
        _scenario(
            scenario_kind="single_retry_path",
            source_id="critical_path_plus_single_retry",
            lower=minimum,
            forecast=single_retry,
            upper=worst_case,
            worst_case=worst_case,
            confidence="medium",
            evidence=(
                f"retry_iteration_tokens:{costs.retry_iteration_token_estimate}",
                f"retry_iterations:{costs.retry_iteration_count}",
            ),
        ),
        _scenario(
            scenario_kind="worst_case_bound",
            source_id="workflow_cost_analysis",
            lower=worst_case,
            forecast=worst_case,
            upper=worst_case,
            worst_case=worst_case,
            confidence="high",
            evidence=(
                f"retry_reserve:{costs.retry_token_reserve}",
                f"failure_reserve:{costs.failure_path_token_reserve}",
            ),
        ),
        _scenario(
            scenario_kind="pruning_adjusted_bound",
            source_id="workflow_pruning_plan",
            lower=pruning_adjusted,
            forecast=pruning_adjusted,
            upper=worst_case,
            worst_case=worst_case,
            confidence="medium",
            evidence=(
                f"pruning_candidates:{pruning.prunable_candidate_count}",
                f"pruning_savings:{pruning.estimated_token_savings}",
            ),
        ),
    )


def _scenario(
    *,
    scenario_kind: ExecutionCostForecastScenarioKind,
    source_id: str,
    lower: int,
    forecast: int,
    upper: int,
    worst_case: int,
    confidence: ExecutionCostForecastConfidence,
    evidence: tuple[str, ...],
) -> ExecutionCostForecastScenario:
    return ExecutionCostForecastScenario(
        scenario_id=f"execution_cost_forecast::{scenario_kind}",
        scenario_kind=scenario_kind,
        source_id=source_id,
        lower_bound_tokens=lower,
        forecast_tokens=forecast,
        upper_bound_tokens=upper,
        worst_case_token_estimate=worst_case,
        token_delta_from_worst_case=forecast - worst_case,
        relative_cost=_relative_cost_for_tokens(forecast),
        confidence=confidence,
        evidence=evidence,
        advisory_actions=_scenario_actions(scenario_kind),
    )


def _scenario_for_kind(
    scenarios: tuple[ExecutionCostForecastScenario, ...],
    scenario_kind: ExecutionCostForecastScenarioKind,
) -> ExecutionCostForecastScenario:
    matches = tuple(
        scenario for scenario in scenarios if scenario.scenario_kind == scenario_kind
    )
    if len(matches) != 1:
        raise ValueError("forecast scenarios must include each required kind once")
    return matches[0]


def _relative_cost_for_tokens(tokens: int) -> ExecutionCostForecastRelativeClass:
    if tokens < 8_000:
        return "low"
    if tokens < 18_000:
        return "medium"
    return "high"


def _forecast_pressure(
    worst_case_tokens: int,
    pruning_adjusted_tokens: int,
) -> ExecutionCostForecastPressure:
    if worst_case_tokens >= 18_000 or pruning_adjusted_tokens >= 18_000:
        return "high"
    if worst_case_tokens >= 8_000 or pruning_adjusted_tokens >= 8_000:
        return "medium"
    return "low"


def _scenario_actions(
    scenario_kind: ExecutionCostForecastScenarioKind,
) -> tuple[str, ...]:
    if scenario_kind == "pruning_adjusted_bound":
        return ("Expose pruning-adjusted cost forecast without applying pruning.",)
    if scenario_kind == "worst_case_bound":
        return ("Preserve bounded worst-case cost forecast as advisory metadata.",)
    return ("Expose token forecast as advisory metadata only.",)


def _forecast_actions(
    costs: WorkflowCostAnalysis,
    pruning: WorkflowPruningPlan,
) -> tuple[str, ...]:
    actions = [
        "Expose execution cost forecasts as advisory metadata only.",
        "Preserve provider routing, budget, workflow, retry, and output boundaries.",
    ]
    if pruning.estimated_token_savings:
        actions.append("Compare pruning-adjusted bound without applying pruning.")
    if costs.estimated_cost_pressure == "high":
        actions.append("Keep high cost pressure visible for later strategy selection.")
    return tuple(actions)
