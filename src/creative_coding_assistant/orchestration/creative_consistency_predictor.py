"""V5.2 advisory creative consistency predictor metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityLevel,
    CreativeQualityPrediction,
    CreativeQualitySignal,
)

CreativeConsistencyDimension = Literal[
    "symbolic_coherence",
    "narrative_coherence",
    "emotional_coherence",
    "aesthetic_coherence_potential",
    "constraint_alignment",
    "tradeoff_balance",
]
CreativeConsistencyBand = Literal["strong", "stable", "watch", "fragile"]
CreativeConsistencyPredictionStatus = Literal["recommended", "fallback"]
CreativeConsistencyPredictionConfidence = Literal["low", "medium", "high"]

CREATIVE_CONSISTENCY_PREDICTION_SERIALIZATION_VERSION = (
    "creative_consistency_prediction.v1"
)
CREATIVE_CONSISTENCY_PREDICTION_PLAN_SERIALIZATION_VERSION = (
    "creative_consistency_prediction_plan.v1"
)
CREATIVE_CONSISTENCY_PREDICTOR_AUTHORITY_BOUNDARY = (
    "The V5.2 Creative Consistency Predictor projects existing Creative "
    "Quality Predictor coherence signals into advisory consistency "
    "predictions only; it does not evaluate generated output, execute "
    "consistency validation, score artifacts, critique artifacts, select "
    "artifacts, trigger refinement, select or route providers or models, "
    "execute providers, request human input, control workflows, trigger "
    "retries, mutate prompts, write storage, or modify generated output."
)

_CONSISTENCY_DIMENSIONS: tuple[CreativeConsistencyDimension, ...] = (
    "symbolic_coherence",
    "narrative_coherence",
    "emotional_coherence",
    "aesthetic_coherence_potential",
    "constraint_alignment",
    "tradeoff_balance",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "generated_output_consistency_evaluation",
    "consistency_validation_execution",
    "artifact_scoring",
    "artifact_critique",
    "artifact_selection",
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


class CreativeConsistencyPrediction(BaseModel):
    """One bounded advisory creative consistency prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prediction_id: str = Field(min_length=1, max_length=180)
    source_quality_prediction_role: Literal["creative_quality_predictor"] = (
        "creative_quality_predictor"
    )
    source_predicted_quality_level: CreativeQualityLevel
    source_readiness_score: int = Field(ge=0, le=100)
    source_quality_signal_dimension: CreativeConsistencyDimension
    source_quality_signal_score: int = Field(ge=0, le=10)
    predicted_consistency_band: CreativeConsistencyBand
    predicted_consistency_range: tuple[int, int]
    predicted_consistency_midpoint: int = Field(ge=0, le=100)
    prediction_confidence: CreativeConsistencyPredictionConfidence
    status: CreativeConsistencyPredictionStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    creative_consistency_predictor_implemented: Literal[True] = True
    advisory_consistency_prediction_implemented: Literal[True] = True
    source_quality_metadata_only: Literal[True] = True
    generated_output_consistency_evaluation_implemented: Literal[False] = False
    consistency_validation_execution_implemented: Literal[False] = False
    artifact_scoring_implemented: Literal[False] = False
    artifact_critique_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_consistency_prediction.v1"] = (
        CREATIVE_CONSISTENCY_PREDICTION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _prediction_matches_signal(self) -> Self:
        if self.predicted_consistency_range != _consistency_range(
            self.source_quality_signal_score
        ):
            raise ValueError("predicted_consistency_range must match signal score")
        if self.predicted_consistency_midpoint != (
            self.source_quality_signal_score * 10
        ):
            raise ValueError("predicted_consistency_midpoint must match signal score")
        if self.predicted_consistency_band != _consistency_band(
            self.source_quality_signal_score
        ):
            raise ValueError("predicted_consistency_band must match signal score")
        if self.prediction_confidence != _confidence(
            self.source_quality_signal_score
        ):
            raise ValueError("prediction_confidence must match signal score")
        return self


class CreativeConsistencyPredictionPlan(BaseModel):
    """Bounded V5.2 advisory creative consistency prediction plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_consistency_predictor"] = (
        "creative_consistency_predictor"
    )
    serialization_version: Literal["creative_consistency_prediction_plan.v1"] = (
        CREATIVE_CONSISTENCY_PREDICTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_CONSISTENCY_PREDICTOR_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    source_creative_quality_prediction_role: Literal["creative_quality_predictor"] = (
        "creative_quality_predictor"
    )
    source_predicted_quality_level: CreativeQualityLevel
    source_readiness_score: int = Field(ge=0, le=100)
    source_consistency_dimensions: tuple[CreativeConsistencyDimension, ...] = Field(
        min_length=1,
        max_length=8,
    )
    predictions: tuple[CreativeConsistencyPrediction, ...] = Field(
        min_length=1,
        max_length=8,
    )
    prediction_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    recommended_prediction_id: str = Field(min_length=1, max_length=180)
    recommended_consistency_band: CreativeConsistencyBand
    recommended_consistency_midpoint: int = Field(ge=0, le=100)
    fallback_prediction_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prediction_count: int = Field(ge=1, le=8)
    strong_or_stable_prediction_count: int = Field(ge=0, le=8)
    watch_or_fragile_prediction_count: int = Field(ge=0, le=8)
    fragile_prediction_count: int = Field(ge=0, le=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    creative_consistency_predictor_implemented: Literal[True] = True
    advisory_consistency_prediction_implemented: Literal[True] = True
    source_quality_metadata_only: Literal[True] = True
    generated_output_consistency_evaluation_implemented: Literal[False] = False
    consistency_validation_execution_implemented: Literal[False] = False
    artifact_scoring_implemented: Literal[False] = False
    artifact_critique_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
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
        if self.source_consistency_dimensions != tuple(
            prediction.source_quality_signal_dimension
            for prediction in self.predictions
        ):
            raise ValueError("source_consistency_dimensions must match predictions")
        if self.prediction_count != len(self.predictions):
            raise ValueError("prediction_count must match predictions")

        recommended = tuple(
            prediction
            for prediction in self.predictions
            if prediction.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError(
                "exactly one recommended consistency prediction is required"
            )
        recommended_prediction = recommended[0]
        if self.recommended_prediction_id != recommended_prediction.prediction_id:
            raise ValueError("recommended_prediction_id must match prediction")
        if self.recommended_consistency_band != (
            recommended_prediction.predicted_consistency_band
        ):
            raise ValueError("recommended_consistency_band must match prediction")
        if self.recommended_consistency_midpoint != (
            recommended_prediction.predicted_consistency_midpoint
        ):
            raise ValueError("recommended_consistency_midpoint must match prediction")
        if self.fallback_prediction_ids != tuple(
            prediction.prediction_id
            for prediction in self.predictions
            if prediction.status == "fallback"
        ):
            raise ValueError("fallback_prediction_ids must match predictions")
        if self.strong_or_stable_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_consistency_band in {"strong", "stable"}
        ):
            raise ValueError(
                "strong_or_stable_prediction_count must match predictions"
            )
        if self.watch_or_fragile_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_consistency_band in {"watch", "fragile"}
        ):
            raise ValueError(
                "watch_or_fragile_prediction_count must match predictions"
            )
        if self.fragile_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_consistency_band == "fragile"
        ):
            raise ValueError("fragile_prediction_count must match predictions")
        for prediction in self.predictions:
            if (
                prediction.source_predicted_quality_level
                != self.source_predicted_quality_level
            ):
                raise ValueError(
                    "prediction source_predicted_quality_level must match plan"
                )
            if prediction.source_readiness_score != self.source_readiness_score:
                raise ValueError("prediction source_readiness_score must match plan")
        return self


def predict_creative_consistency(
    *,
    creative_quality_prediction: CreativeQualityPrediction,
) -> CreativeConsistencyPredictionPlan:
    """Return advisory consistency predictions from quality metadata only."""

    signals = _consistency_signals(creative_quality_prediction)
    recommended_dimension = _recommended_dimension(signals)
    predictions = tuple(
        _prediction_from_signal(
            source=creative_quality_prediction,
            signal=signal,
            status=(
                "recommended"
                if signal.dimension == recommended_dimension
                else "fallback"
            ),
        )
        for signal in signals
    )
    recommended = _recommended_prediction(predictions)

    return CreativeConsistencyPredictionPlan(
        source_predicted_quality_level=(
            creative_quality_prediction.predicted_quality_level
        ),
        source_readiness_score=creative_quality_prediction.readiness_score,
        source_consistency_dimensions=tuple(
            prediction.source_quality_signal_dimension for prediction in predictions
        ),
        predictions=predictions,
        prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
        recommended_prediction_id=recommended.prediction_id,
        recommended_consistency_band=recommended.predicted_consistency_band,
        recommended_consistency_midpoint=recommended.predicted_consistency_midpoint,
        fallback_prediction_ids=tuple(
            prediction.prediction_id
            for prediction in predictions
            if prediction.status == "fallback"
        ),
        prediction_count=len(predictions),
        strong_or_stable_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_consistency_band in {"strong", "stable"}
        ),
        watch_or_fragile_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_consistency_band in {"watch", "fragile"}
        ),
        fragile_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_consistency_band == "fragile"
        ),
        advisory_actions=(
            "Surface consistency prediction bands for review.",
            "Keep validation execution, artifact scoring, and routing disabled.",
        ),
    )


def creative_consistency_prediction_by_id(
    prediction_id: str,
    plan: CreativeConsistencyPredictionPlan,
) -> CreativeConsistencyPrediction | None:
    """Return one advisory consistency prediction without applying it."""

    for prediction in plan.predictions:
        if prediction.prediction_id == prediction_id:
            return prediction
    return None


def creative_consistency_predictions_for_band(
    consistency_band: CreativeConsistencyBand,
    plan: CreativeConsistencyPredictionPlan,
) -> tuple[CreativeConsistencyPrediction, ...]:
    """Return advisory consistency predictions for one consistency band."""

    return tuple(
        prediction
        for prediction in plan.predictions
        if prediction.predicted_consistency_band == consistency_band
    )


def _consistency_signals(
    source: CreativeQualityPrediction,
) -> tuple[CreativeQualitySignal, ...]:
    by_dimension: dict[str, CreativeQualitySignal] = {}
    for signal in (
        *source.strongest_quality_signals,
        *source.weakest_quality_signals,
    ):
        if signal.dimension not in _CONSISTENCY_DIMENSIONS:
            continue
        current = by_dimension.get(signal.dimension)
        if current is None or signal.score < current.score:
            by_dimension[signal.dimension] = signal
    if not by_dimension:
        fallback = _fallback_signal(source)
        by_dimension[fallback.dimension] = fallback
    return tuple(
        by_dimension[dimension]
        for dimension in _CONSISTENCY_DIMENSIONS
        if dimension in by_dimension
    )


def _prediction_from_signal(
    *,
    source: CreativeQualityPrediction,
    signal: CreativeQualitySignal,
    status: CreativeConsistencyPredictionStatus,
) -> CreativeConsistencyPrediction:
    dimension = signal.dimension
    if dimension not in _CONSISTENCY_DIMENSIONS:
        raise ValueError("creative consistency prediction requires consistency signals")
    return CreativeConsistencyPrediction(
        prediction_id=f"creative_consistency_prediction::{dimension}",
        source_predicted_quality_level=source.predicted_quality_level,
        source_readiness_score=source.readiness_score,
        source_quality_signal_dimension=dimension,
        source_quality_signal_score=signal.score,
        predicted_consistency_band=_consistency_band(signal.score),
        predicted_consistency_range=_consistency_range(signal.score),
        predicted_consistency_midpoint=signal.score * 10,
        prediction_confidence=_confidence(signal.score),
        status=status,
        evidence=(
            f"Derived from Creative Quality Predictor dimension {dimension}.",
            f"Source signal score: {signal.score}/10.",
            f"Source quality level: {source.predicted_quality_level}.",
            "Consistency prediction uses source metadata only.",
        ),
        advisory_actions=(
            "Present consistency posture without validating generated output.",
            "Keep artifact scoring, refinement, and provider routing disabled.",
        ),
    )


def _recommended_dimension(signals: tuple[CreativeQualitySignal, ...]) -> str:
    ordered = sorted(signals, key=lambda signal: (signal.score, signal.dimension))
    return ordered[0].dimension


def _recommended_prediction(
    predictions: tuple[CreativeConsistencyPrediction, ...],
) -> CreativeConsistencyPrediction:
    for prediction in predictions:
        if prediction.status == "recommended":
            return prediction
    raise ValueError("creative consistency prediction requires a recommendation")


def _fallback_signal(source: CreativeQualityPrediction) -> CreativeQualitySignal:
    return CreativeQualitySignal(
        dimension="aesthetic_coherence_potential",
        score=max(0, min(10, source.readiness_score // 10)),
        summary="Fallback consistency signal derived from source readiness.",
        evidence=("No explicit coherence signal was present in exposed summaries.",),
    )


def _consistency_band(score: int) -> CreativeConsistencyBand:
    if score >= 8:
        return "strong"
    if score >= 6:
        return "stable"
    if score >= 4:
        return "watch"
    return "fragile"


def _consistency_range(score: int) -> tuple[int, int]:
    midpoint = score * 10
    return (max(0, midpoint - 5), min(100, midpoint + 5))


def _confidence(score: int) -> CreativeConsistencyPredictionConfidence:
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"
