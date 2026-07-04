"""V5.2 bounded advisory cost estimator for routing candidates."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.quality_cost_optimizer import (
    QualityCostOptimizationCandidate,
    QualityCostOptimizationPlan,
    optimize_quality_cost,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

CostEstimateStatus = Literal["recommended", "fallback"]
CostEstimateConfidence = Literal["low", "medium", "high"]

COST_ESTIMATE_SCENARIO_SERIALIZATION_VERSION = "cost_estimate_scenario.v1"
COST_ESTIMATION_PLAN_SERIALIZATION_VERSION = "cost_estimation_plan.v1"
COST_ESTIMATOR_AUTHORITY_BOUNDARY = (
    "The V5.2 Cost Estimator converts existing advisory cost-profile ranges "
    "into bounded relative cost-unit estimates only; it does not look up "
    "provider pricing, meter live usage, predict provider cost, enforce "
    "budgets, select or switch providers or models, execute providers, "
    "control workflows, trigger retries, mutate prompts, write storage, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_usage_metering",
    "provider_cost_prediction",
    "budget_enforcement",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class CostEstimateScenario(BaseModel):
    """One bounded advisory relative-cost estimate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1, max_length=180)
    source_quality_cost_candidate_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    cost_band: str = Field(min_length=1, max_length=40)
    source_advisory_cost_range: tuple[int, int]
    estimated_min_cost_units: int = Field(ge=0, le=100)
    estimated_max_cost_units: int = Field(ge=0, le=100)
    estimated_midpoint_cost_units: int = Field(ge=0, le=100)
    estimate_confidence: CostEstimateConfidence
    status: CostEstimateStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    cost_estimation_implemented: Literal[True] = True
    relative_cost_units_only: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["cost_estimate_scenario.v1"] = (
        COST_ESTIMATE_SCENARIO_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _scenario_matches_cost_range(self) -> Self:
        low, high = self.source_advisory_cost_range
        if low > high:
            raise ValueError("source_advisory_cost_range must be ordered")
        if self.estimated_min_cost_units != low:
            raise ValueError("estimated_min_cost_units must match source range")
        if self.estimated_max_cost_units != high:
            raise ValueError("estimated_max_cost_units must match source range")
        if self.estimated_midpoint_cost_units != (low + high) // 2:
            raise ValueError("estimated_midpoint_cost_units must match source range")
        if self.estimate_confidence != _confidence(low, high):
            raise ValueError("estimate_confidence must match source range")
        return self


class CostEstimationPlan(BaseModel):
    """Bounded V5.2 advisory cost estimation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cost_estimator"] = "cost_estimator"
    serialization_version: Literal["cost_estimation_plan.v1"] = (
        COST_ESTIMATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COST_ESTIMATOR_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_quality_cost_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_quality_cost_candidate_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    scenarios: tuple[CostEstimateScenario, ...] = Field(min_length=1, max_length=12)
    scenario_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_scenario_id: str = Field(min_length=1, max_length=180)
    recommended_max_cost_units: int = Field(ge=0, le=100)
    fallback_scenario_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    scenario_count: int = Field(ge=1, le=12)
    high_cost_scenario_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    cost_estimation_implemented: Literal[True] = True
    relative_cost_units_only: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_scenarios(self) -> Self:
        derived_scenario_ids = tuple(
            scenario.scenario_id for scenario in self.scenarios
        )
        if len(set(derived_scenario_ids)) != len(derived_scenario_ids):
            raise ValueError("scenario_ids must be unique")
        if self.scenario_ids != derived_scenario_ids:
            raise ValueError("scenario_ids must match scenarios")
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario_count must match scenarios")
        if self.source_quality_cost_candidate_ids != tuple(
            scenario.source_quality_cost_candidate_id for scenario in self.scenarios
        ):
            raise ValueError("source_quality_cost_candidate_ids must match scenarios")

        recommended = tuple(
            scenario for scenario in self.scenarios if scenario.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended scenario is required")
        recommended_scenario = recommended[0]
        if self.recommended_scenario_id != recommended_scenario.scenario_id:
            raise ValueError("recommended_scenario_id must match scenario")
        if (
            self.recommended_max_cost_units
            != recommended_scenario.estimated_max_cost_units
        ):
            raise ValueError("recommended_max_cost_units must match scenario")
        if self.fallback_scenario_ids != tuple(
            scenario.scenario_id
            for scenario in self.scenarios
            if scenario.status == "fallback"
        ):
            raise ValueError("fallback_scenario_ids must match scenarios")
        if self.high_cost_scenario_count != sum(
            1 for scenario in self.scenarios if scenario.estimated_max_cost_units >= 6
        ):
            raise ValueError("high_cost_scenario_count must match scenarios")
        for scenario in self.scenarios:
            if scenario.route_name != self.route_name:
                raise ValueError("scenario route_name must match plan route_name")
        return self


def estimate_routing_cost(
    *,
    quality_cost_plan: QualityCostOptimizationPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> CostEstimationPlan:
    """Return bounded relative cost estimates without pricing lookup."""

    source_plan = quality_cost_plan or optimize_quality_cost(
        route_decision=route_decision,
        route=route,
    )
    scenarios = tuple(
        _scenario_from_candidate(candidate) for candidate in source_plan.candidates
    )
    recommended = _recommended_scenario(scenarios)

    return CostEstimationPlan(
        source_quality_cost_serialization_version=source_plan.serialization_version,
        route_name=source_plan.route_name,
        source_quality_cost_candidate_ids=tuple(
            scenario.source_quality_cost_candidate_id for scenario in scenarios
        ),
        scenarios=scenarios,
        scenario_ids=tuple(scenario.scenario_id for scenario in scenarios),
        recommended_scenario_id=recommended.scenario_id,
        recommended_max_cost_units=recommended.estimated_max_cost_units,
        fallback_scenario_ids=tuple(
            scenario.scenario_id
            for scenario in scenarios
            if scenario.status == "fallback"
        ),
        scenario_count=len(scenarios),
        high_cost_scenario_count=sum(
            1 for scenario in scenarios if scenario.estimated_max_cost_units >= 6
        ),
        advisory_actions=_plan_actions(source_plan.route_name, recommended),
    )


def cost_estimate_scenario_by_id(
    scenario_id: str,
    plan: CostEstimationPlan | None = None,
) -> CostEstimateScenario | None:
    """Return one advisory cost estimate scenario without applying it."""

    source_plan = plan or estimate_routing_cost()
    for scenario in source_plan.scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


def cost_estimate_scenarios_for_confidence(
    confidence: CostEstimateConfidence,
    plan: CostEstimationPlan | None = None,
) -> tuple[CostEstimateScenario, ...]:
    """Return advisory cost estimate scenarios by confidence."""

    source_plan = plan or estimate_routing_cost()
    return tuple(
        scenario
        for scenario in source_plan.scenarios
        if scenario.estimate_confidence == confidence
    )


def _scenario_from_candidate(
    candidate: QualityCostOptimizationCandidate,
) -> CostEstimateScenario:
    low, high = candidate.advisory_cost_range
    return CostEstimateScenario(
        scenario_id=f"cost_estimate::{candidate.source_model_profile_id}",
        source_quality_cost_candidate_id=candidate.candidate_id,
        source_model_profile_id=candidate.source_model_profile_id,
        route_name=candidate.route_name,
        cost_band=candidate.cost_band,
        source_advisory_cost_range=candidate.advisory_cost_range,
        estimated_min_cost_units=low,
        estimated_max_cost_units=high,
        estimated_midpoint_cost_units=(low + high) // 2,
        estimate_confidence=_confidence(low, high),
        status=candidate.status,
        evidence=(
            f"Derived from {candidate.candidate_id}.",
            f"Cost band: {candidate.cost_band}.",
            "Relative units come from passive cost-profile metadata.",
        ),
        advisory_actions=(
            "Use relative cost units for review only.",
            "Keep pricing lookup and budget enforcement disabled.",
        ),
    )


def _confidence(low: int, high: int) -> CostEstimateConfidence:
    width = high - low
    if width <= 2:
        return "high"
    if width <= 4:
        return "medium"
    return "low"


def _recommended_scenario(
    scenarios: tuple[CostEstimateScenario, ...],
) -> CostEstimateScenario:
    for scenario in scenarios:
        if scenario.status == "recommended":
            return scenario
    raise ValueError("cost estimator requires a recommended scenario")


def _plan_actions(
    route_name: RouteName,
    recommended: CostEstimateScenario,
) -> tuple[str, ...]:
    return (
        (
            f"Present relative cost range {recommended.estimated_min_cost_units}-"
            f"{recommended.estimated_max_cost_units} for {route_name.value}."
        ),
        "Use passive cost-profile ranges only.",
        "Defer budget policy, provider pricing, and execution behavior.",
    )
