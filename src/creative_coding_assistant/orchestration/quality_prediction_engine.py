"""V5.2 advisory quality prediction metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    QualityProfile,
    QualityProfileKind,
    QualityProfileLevel,
    QualityProfileRegistry,
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

QualityPredictionStatus = Literal["recommended", "fallback"]
QualityPredictionConfidence = Literal["low", "medium", "high"]

QUALITY_PREDICTION_DECISION_SERIALIZATION_VERSION = "quality_prediction_decision.v1"
QUALITY_PREDICTION_PLAN_SERIALIZATION_VERSION = "quality_prediction_plan.v1"
QUALITY_PREDICTION_AUTHORITY_BOUNDARY = (
    "The V5.2 Quality Prediction Engine converts existing passive quality "
    "profile levels into bounded relative quality predictions only; it does "
    "not evaluate generated output, calculate quality scores, execute quality "
    "escalation, trigger refinement, select or route providers or models, "
    "execute providers, request human input, control workflows, trigger "
    "retries, mutate prompts, write storage, or modify generated output."
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
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_QUALITY_RANGE_BY_LEVEL: dict[QualityProfileLevel, tuple[int, int]] = {
    "low": (25, 45),
    "medium": (45, 65),
    "high": (65, 85),
    "critical": (80, 95),
}


class QualityPredictionDecision(BaseModel):
    """One bounded advisory quality prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prediction_id: str = Field(min_length=1, max_length=180)
    source_quality_profile_id: str = Field(min_length=1, max_length=140)
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_name: RouteName
    quality_profile_kind: QualityProfileKind
    predicted_quality_level: QualityProfileLevel
    predicted_quality_range: tuple[int, int]
    predicted_quality_midpoint: int = Field(ge=0, le=100)
    prediction_confidence: QualityPredictionConfidence
    status: QualityPredictionStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    quality_prediction_engine_implemented: Literal[True] = True
    advisory_quality_prediction_implemented: Literal[True] = True
    relative_quality_units_only: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["quality_prediction_decision.v1"] = (
        QUALITY_PREDICTION_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_quality_range(self) -> Self:
        expected_range = _QUALITY_RANGE_BY_LEVEL[self.predicted_quality_level]
        if self.predicted_quality_range != expected_range:
            raise ValueError("predicted_quality_range must match quality level")
        low, high = self.predicted_quality_range
        if self.predicted_quality_midpoint != (low + high) // 2:
            raise ValueError("predicted_quality_midpoint must match range")
        if self.prediction_confidence != _confidence(self.predicted_quality_level):
            raise ValueError("prediction_confidence must match quality level")
        return self


class QualityPredictionPlan(BaseModel):
    """Bounded V5.2 advisory quality prediction plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["quality_prediction_engine"] = "quality_prediction_engine"
    serialization_version: Literal["quality_prediction_plan.v1"] = (
        QUALITY_PREDICTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=QUALITY_PREDICTION_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_quality_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    predictions: tuple[QualityPredictionDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    prediction_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_prediction_id: str = Field(min_length=1, max_length=180)
    recommended_quality_level: QualityProfileLevel
    recommended_quality_midpoint: int = Field(ge=0, le=100)
    fallback_prediction_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    prediction_count: int = Field(ge=1, le=12)
    high_or_critical_prediction_count: int = Field(ge=0, le=12)
    critical_prediction_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    quality_prediction_engine_implemented: Literal[True] = True
    advisory_quality_prediction_implemented: Literal[True] = True
    relative_quality_units_only: Literal[True] = True
    generated_output_quality_evaluation_implemented: Literal[False] = False
    quality_scoring_implemented: Literal[False] = False
    quality_escalation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
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
        if self.source_quality_profile_ids != tuple(
            prediction.source_quality_profile_id for prediction in self.predictions
        ):
            raise ValueError("source_quality_profile_ids must match predictions")
        if self.prediction_count != len(self.predictions):
            raise ValueError("prediction_count must match predictions")

        recommended = tuple(
            prediction
            for prediction in self.predictions
            if prediction.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended quality prediction is required")
        recommended_prediction = recommended[0]
        if self.recommended_prediction_id != recommended_prediction.prediction_id:
            raise ValueError("recommended_prediction_id must match prediction")
        if self.recommended_quality_level != (
            recommended_prediction.predicted_quality_level
        ):
            raise ValueError("recommended_quality_level must match prediction")
        if self.recommended_quality_midpoint != (
            recommended_prediction.predicted_quality_midpoint
        ):
            raise ValueError("recommended_quality_midpoint must match prediction")
        if self.fallback_prediction_ids != tuple(
            prediction.prediction_id
            for prediction in self.predictions
            if prediction.status == "fallback"
        ):
            raise ValueError("fallback_prediction_ids must match predictions")
        if self.high_or_critical_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_quality_level in {"high", "critical"}
        ):
            raise ValueError("high_or_critical_prediction_count must match predictions")
        if self.critical_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_quality_level == "critical"
        ):
            raise ValueError("critical_prediction_count must match predictions")
        for prediction in self.predictions:
            if prediction.route_name != self.route_name:
                raise ValueError("prediction route_name must match plan route_name")
        return self


def predict_quality_for_route(
    *,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    quality_profiles: QualityProfileRegistry | None = None,
) -> QualityPredictionPlan:
    """Return advisory quality predictions without evaluating generated output."""

    route_name = _resolve_route(route_decision=route_decision, route=route)
    registry = quality_profiles or quality_profile_registry()
    profiles = tuple(
        profile
        for profile in registry.quality_profiles
        if route_name in profile.route_applicability
    )
    if not profiles:
        raise ValueError("quality prediction requires an applicable quality profile")
    recommended_profile_id = _recommended_profile_id(profiles)
    predictions = tuple(
        _prediction_from_profile(
            route_name=route_name,
            profile=profile,
            status=(
                "recommended"
                if profile.quality_profile_id == recommended_profile_id
                else "fallback"
            ),
        )
        for profile in profiles
    )
    recommended = _recommended_prediction(predictions)

    return QualityPredictionPlan(
        source_quality_profile_serialization_version=registry.serialization_version,
        route_name=route_name,
        source_quality_profile_ids=tuple(
            prediction.source_quality_profile_id for prediction in predictions
        ),
        predictions=predictions,
        prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
        recommended_prediction_id=recommended.prediction_id,
        recommended_quality_level=recommended.predicted_quality_level,
        recommended_quality_midpoint=recommended.predicted_quality_midpoint,
        fallback_prediction_ids=tuple(
            prediction.prediction_id
            for prediction in predictions
            if prediction.status == "fallback"
        ),
        prediction_count=len(predictions),
        high_or_critical_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_quality_level in {"high", "critical"}
        ),
        critical_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_quality_level == "critical"
        ),
        advisory_actions=_plan_actions(route_name, recommended),
    )


def quality_prediction_by_id(
    prediction_id: str,
    plan: QualityPredictionPlan | None = None,
) -> QualityPredictionDecision | None:
    """Return one advisory quality prediction without applying it."""

    source_plan = plan or predict_quality_for_route()
    for prediction in source_plan.predictions:
        if prediction.prediction_id == prediction_id:
            return prediction
    return None


def quality_predictions_for_level(
    quality_level: QualityProfileLevel,
    plan: QualityPredictionPlan | None = None,
) -> tuple[QualityPredictionDecision, ...]:
    """Return advisory quality predictions for one quality level."""

    source_plan = plan or predict_quality_for_route()
    return tuple(
        prediction
        for prediction in source_plan.predictions
        if prediction.predicted_quality_level == quality_level
    )


def _resolve_route(
    *,
    route_decision: RouteDecision | None,
    route: RouteName | str | None,
) -> RouteName:
    explicit_route = (
        None
        if route is None
        else (route if isinstance(route, RouteName) else RouteName(str(route)))
    )
    if route_decision is None:
        return explicit_route or RouteName.GENERATE
    if explicit_route is not None and explicit_route != route_decision.route:
        raise ValueError("route must match route_decision")
    return route_decision.route


def _prediction_from_profile(
    *,
    route_name: RouteName,
    profile: QualityProfile,
    status: QualityPredictionStatus,
) -> QualityPredictionDecision:
    low, high = _QUALITY_RANGE_BY_LEVEL[profile.quality_level]
    return QualityPredictionDecision(
        prediction_id=f"quality_prediction::{profile.quality_profile_id}",
        source_quality_profile_id=profile.quality_profile_id,
        source_model_profile_ids=profile.source_model_profile_ids,
        source_provider_selection_profile_ids=(
            profile.source_provider_selection_profile_ids
        ),
        route_name=route_name,
        quality_profile_kind=profile.quality_profile_kind,
        predicted_quality_level=profile.quality_level,
        predicted_quality_range=(low, high),
        predicted_quality_midpoint=(low + high) // 2,
        prediction_confidence=_confidence(profile.quality_level),
        status=status,
        evidence=(
            f"Derived from {profile.quality_profile_id}.",
            f"Passive quality level: {profile.quality_level}.",
            "Quality prediction uses bounded relative metadata only.",
        ),
        advisory_actions=(
            "Surface relative quality prediction for review.",
            "Keep quality evaluation, scoring, and refinement triggering disabled.",
        ),
    )


def _recommended_profile_id(profiles: tuple[QualityProfile, ...]) -> str:
    ordered = sorted(
        profiles,
        key=lambda profile: (
            -_midpoint(profile.quality_level),
            profile.quality_profile_id,
        ),
    )
    return ordered[0].quality_profile_id


def _recommended_prediction(
    predictions: tuple[QualityPredictionDecision, ...],
) -> QualityPredictionDecision:
    for prediction in predictions:
        if prediction.status == "recommended":
            return prediction
    raise ValueError("quality prediction requires a recommended prediction")


def _midpoint(level: QualityProfileLevel) -> int:
    low, high = _QUALITY_RANGE_BY_LEVEL[level]
    return (low + high) // 2


def _confidence(level: QualityProfileLevel) -> QualityPredictionConfidence:
    if level == "critical":
        return "medium"
    if level == "low":
        return "low"
    return "high"


def _plan_actions(
    route_name: RouteName,
    recommended: QualityPredictionDecision,
) -> tuple[str, ...]:
    low, high = recommended.predicted_quality_range
    return (
        (
            f"Present {recommended.predicted_quality_level} quality prediction "
            f"range {low}-{high} for {route_name.value}."
        ),
        "Do not evaluate output quality or trigger refinement automatically.",
        "Keep provider routing, provider execution, and prompt mutation disabled.",
    )
