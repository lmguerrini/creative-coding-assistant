"""V5.2 advisory cost prediction metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    CostProfile,
    CostProfileBand,
    CostProfileKind,
    CostProfileRegistry,
    cost_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

CostPredictionStatus = Literal["recommended", "fallback"]
CostPredictionConfidence = Literal["low", "medium", "high"]

COST_PREDICTION_DECISION_SERIALIZATION_VERSION = "cost_prediction_decision.v1"
COST_PREDICTION_PLAN_SERIALIZATION_VERSION = "cost_prediction_plan.v1"
COST_PREDICTION_AUTHORITY_BOUNDARY = (
    "The V5.2 Cost Prediction Engine converts existing passive cost profile "
    "ranges into bounded relative cost predictions only; it does not look up "
    "provider pricing, meter live usage, calculate cost scores, enforce "
    "budgets, route by cost, select or route providers or models, execute "
    "providers, control workflows, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_usage_metering",
    "cost_scoring",
    "budget_enforcement",
    "cost_based_routing",
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


class CostPredictionDecision(BaseModel):
    """One bounded advisory cost prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prediction_id: str = Field(min_length=1, max_length=180)
    source_cost_profile_id: str = Field(min_length=1, max_length=140)
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_name: RouteName
    cost_profile_kind: CostProfileKind
    predicted_cost_band: CostProfileBand
    predicted_cost_range: tuple[int, int]
    predicted_cost_midpoint: int = Field(ge=0, le=100)
    prediction_confidence: CostPredictionConfidence
    status: CostPredictionStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    cost_prediction_engine_implemented: Literal[True] = True
    advisory_cost_prediction_implemented: Literal[True] = True
    relative_cost_units_only: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["cost_prediction_decision.v1"] = (
        COST_PREDICTION_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_cost_range(self) -> Self:
        low, high = self.predicted_cost_range
        if low > high:
            raise ValueError("predicted_cost_range must be ordered")
        if self.predicted_cost_midpoint != (low + high) // 2:
            raise ValueError("predicted_cost_midpoint must match range")
        if self.prediction_confidence != _confidence(low, high):
            raise ValueError("prediction_confidence must match range")
        return self


class CostPredictionPlan(BaseModel):
    """Bounded V5.2 advisory cost prediction plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cost_prediction_engine"] = "cost_prediction_engine"
    serialization_version: Literal["cost_prediction_plan.v1"] = (
        COST_PREDICTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COST_PREDICTION_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_cost_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_cost_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    predictions: tuple[CostPredictionDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    prediction_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_prediction_id: str = Field(min_length=1, max_length=180)
    recommended_cost_band: CostProfileBand
    recommended_cost_midpoint: int = Field(ge=0, le=100)
    fallback_prediction_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    prediction_count: int = Field(ge=1, le=12)
    high_or_guarded_prediction_count: int = Field(ge=0, le=12)
    low_cost_prediction_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    cost_prediction_engine_implemented: Literal[True] = True
    advisory_cost_prediction_implemented: Literal[True] = True
    relative_cost_units_only: Literal[True] = True
    provider_pricing_lookup_implemented: Literal[False] = False
    live_usage_metering_implemented: Literal[False] = False
    cost_scoring_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    cost_based_routing_implemented: Literal[False] = False
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
    def _plan_matches_predictions(self) -> Self:
        derived_prediction_ids = tuple(
            prediction.prediction_id for prediction in self.predictions
        )
        if len(set(derived_prediction_ids)) != len(derived_prediction_ids):
            raise ValueError("prediction_ids must be unique")
        if self.prediction_ids != derived_prediction_ids:
            raise ValueError("prediction_ids must match predictions")
        if self.source_cost_profile_ids != tuple(
            prediction.source_cost_profile_id for prediction in self.predictions
        ):
            raise ValueError("source_cost_profile_ids must match predictions")
        if self.prediction_count != len(self.predictions):
            raise ValueError("prediction_count must match predictions")

        recommended = tuple(
            prediction
            for prediction in self.predictions
            if prediction.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended cost prediction is required")
        recommended_prediction = recommended[0]
        if self.recommended_prediction_id != recommended_prediction.prediction_id:
            raise ValueError("recommended_prediction_id must match prediction")
        if self.recommended_cost_band != recommended_prediction.predicted_cost_band:
            raise ValueError("recommended_cost_band must match prediction")
        if self.recommended_cost_midpoint != (
            recommended_prediction.predicted_cost_midpoint
        ):
            raise ValueError("recommended_cost_midpoint must match prediction")
        if self.fallback_prediction_ids != tuple(
            prediction.prediction_id
            for prediction in self.predictions
            if prediction.status == "fallback"
        ):
            raise ValueError("fallback_prediction_ids must match predictions")
        if self.high_or_guarded_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_cost_band in {"high", "guarded"}
        ):
            raise ValueError(
                "high_or_guarded_prediction_count must match predictions"
            )
        if self.low_cost_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_cost_band == "low"
        ):
            raise ValueError("low_cost_prediction_count must match predictions")
        for prediction in self.predictions:
            if prediction.route_name != self.route_name:
                raise ValueError("prediction route_name must match plan route_name")
        return self


def predict_cost_for_route(
    *,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    cost_profiles: CostProfileRegistry | None = None,
) -> CostPredictionPlan:
    """Return advisory cost predictions without pricing lookup or enforcement."""

    route_name = _resolve_route(route_decision=route_decision, route=route)
    registry = cost_profiles or cost_profile_registry()
    profiles = tuple(
        profile
        for profile in registry.cost_profiles
        if route_name in profile.route_applicability
    )
    if not profiles:
        raise ValueError("cost prediction requires an applicable cost profile")
    recommended_profile_id = _recommended_profile_id(profiles)
    predictions = tuple(
        _prediction_from_profile(
            route_name=route_name,
            profile=profile,
            status=(
                "recommended"
                if profile.cost_profile_id == recommended_profile_id
                else "fallback"
            ),
        )
        for profile in profiles
    )
    recommended = _recommended_prediction(predictions)

    return CostPredictionPlan(
        source_cost_profile_serialization_version=registry.serialization_version,
        route_name=route_name,
        source_cost_profile_ids=tuple(
            prediction.source_cost_profile_id for prediction in predictions
        ),
        predictions=predictions,
        prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
        recommended_prediction_id=recommended.prediction_id,
        recommended_cost_band=recommended.predicted_cost_band,
        recommended_cost_midpoint=recommended.predicted_cost_midpoint,
        fallback_prediction_ids=tuple(
            prediction.prediction_id
            for prediction in predictions
            if prediction.status == "fallback"
        ),
        prediction_count=len(predictions),
        high_or_guarded_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_cost_band in {"high", "guarded"}
        ),
        low_cost_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_cost_band == "low"
        ),
        advisory_actions=_plan_actions(route_name, recommended),
    )


def cost_prediction_by_id(
    prediction_id: str,
    plan: CostPredictionPlan | None = None,
) -> CostPredictionDecision | None:
    """Return one advisory cost prediction without applying it."""

    source_plan = plan or predict_cost_for_route()
    for prediction in source_plan.predictions:
        if prediction.prediction_id == prediction_id:
            return prediction
    return None


def cost_predictions_for_band(
    cost_band: CostProfileBand,
    plan: CostPredictionPlan | None = None,
) -> tuple[CostPredictionDecision, ...]:
    """Return advisory cost predictions for one cost band."""

    source_plan = plan or predict_cost_for_route()
    return tuple(
        prediction
        for prediction in source_plan.predictions
        if prediction.predicted_cost_band == cost_band
    )


def _resolve_route(
    *,
    route_decision: RouteDecision | None,
    route: RouteName | str | None,
) -> RouteName:
    explicit_route = None if route is None else (
        route if isinstance(route, RouteName) else RouteName(str(route))
    )
    if route_decision is None:
        return explicit_route or RouteName.GENERATE
    if explicit_route is not None and explicit_route != route_decision.route:
        raise ValueError("route must match route_decision")
    return route_decision.route


def _prediction_from_profile(
    *,
    route_name: RouteName,
    profile: CostProfile,
    status: CostPredictionStatus,
) -> CostPredictionDecision:
    low, high = profile.advisory_cost_range
    return CostPredictionDecision(
        prediction_id=f"cost_prediction::{profile.cost_profile_id}",
        source_cost_profile_id=profile.cost_profile_id,
        source_model_profile_ids=profile.source_model_profile_ids,
        source_provider_selection_profile_ids=(
            profile.source_provider_selection_profile_ids
        ),
        route_name=route_name,
        cost_profile_kind=profile.cost_profile_kind,
        predicted_cost_band=profile.cost_band,
        predicted_cost_range=profile.advisory_cost_range,
        predicted_cost_midpoint=(low + high) // 2,
        prediction_confidence=_confidence(low, high),
        status=status,
        evidence=(
            f"Derived from {profile.cost_profile_id}.",
            f"Passive cost band: {profile.cost_band}.",
            "Cost prediction uses bounded relative cost units only.",
        ),
        advisory_actions=(
            "Surface relative cost prediction for review.",
            "Keep pricing lookup, cost scoring, and budget enforcement disabled.",
        ),
    )


def _recommended_profile_id(profiles: tuple[CostProfile, ...]) -> str:
    ordered = sorted(
        profiles,
        key=lambda profile: (
            _midpoint(profile.advisory_cost_range),
            profile.cost_profile_id,
        ),
    )
    return ordered[0].cost_profile_id


def _recommended_prediction(
    predictions: tuple[CostPredictionDecision, ...],
) -> CostPredictionDecision:
    for prediction in predictions:
        if prediction.status == "recommended":
            return prediction
    raise ValueError("cost prediction requires a recommended prediction")


def _midpoint(cost_range: tuple[int, int]) -> int:
    low, high = cost_range
    return (low + high) // 2


def _confidence(low: int, high: int) -> CostPredictionConfidence:
    width = high - low
    if width <= 2:
        return "high"
    if width <= 4:
        return "medium"
    return "low"


def _plan_actions(
    route_name: RouteName,
    recommended: CostPredictionDecision,
) -> tuple[str, ...]:
    low, high = recommended.predicted_cost_range
    return (
        (
            f"Present {recommended.predicted_cost_band} cost prediction "
            f"range {low}-{high} for {route_name.value}."
        ),
        "Do not look up pricing, meter live usage, or enforce budgets.",
        "Keep cost-based routing, provider execution, and prompt mutation disabled.",
    )
