"""V5.2 advisory creative diversity predictor metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_diversity_audit import (
    CreativeDiversityAuditRecord,
    CreativeDiversityAuditRegistry,
    creative_diversity_audit_registry,
)

CreativeDiversityPredictionBand = Literal["narrow", "moderate", "broad", "guarded"]
CreativeDiversityPredictionStatus = Literal["recommended", "fallback"]

CREATIVE_DIVERSITY_PREDICTION_SERIALIZATION_VERSION = "creative_diversity_prediction.v1"
CREATIVE_DIVERSITY_PREDICTION_PLAN_SERIALIZATION_VERSION = (
    "creative_diversity_prediction_plan.v1"
)
CREATIVE_DIVERSITY_PREDICTOR_AUTHORITY_BOUNDARY = (
    "The V5.2 Creative Diversity Predictor converts passive creative "
    "diversity audit and exploration budget metadata into advisory diversity "
    "posture predictions only; it does not enforce budgets, generate "
    "variants, trigger refinement, route by cost, invoke agents, route "
    "providers or models, control workflows, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "budget_enforcement",
    "variant_generation",
    "refinement_triggering",
    "cost_routing",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class CreativeDiversityPrediction(BaseModel):
    """One advisory creative diversity posture prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    prediction_id: str = Field(min_length=1, max_length=220)
    source_budget_profile_id: str = Field(min_length=1, max_length=180)
    source_topic_id: str = Field(min_length=1, max_length=140)
    source_audit_status: Literal["pass"] = "pass"
    predicted_diversity_band: CreativeDiversityPredictionBand
    max_advisory_variants: int = Field(ge=0, le=3)
    max_advisory_refinement_passes: int = Field(ge=0, le=3)
    predicted_variant_range: tuple[int, int]
    diversity_readiness_score: int = Field(ge=0, le=100)
    source_trace_profile_id: str = Field(min_length=1, max_length=180)
    source_provenance_profile_id: str = Field(min_length=1, max_length=180)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    status: CreativeDiversityPredictionStatus
    creative_diversity_predictor_implemented: Literal[True] = True
    advisory_diversity_prediction_implemented: Literal[True] = True
    active_diversity_generation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_diversity_prediction.v1"] = (
        CREATIVE_DIVERSITY_PREDICTION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _prediction_matches_budget_bounds(self) -> Self:
        if self.predicted_variant_range != (0, self.max_advisory_variants):
            raise ValueError("predicted_variant_range must match advisory variants")
        if self.diversity_readiness_score != _readiness_score(
            variants=self.max_advisory_variants,
            refinements=self.max_advisory_refinement_passes,
        ):
            raise ValueError("diversity_readiness_score must match advisory bounds")
        return self


class CreativeDiversityPredictionPlan(BaseModel):
    """Bounded V5.2 advisory creative diversity prediction plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_diversity_predictor"] = "creative_diversity_predictor"
    serialization_version: Literal["creative_diversity_prediction_plan.v1"] = (
        CREATIVE_DIVERSITY_PREDICTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_DIVERSITY_PREDICTOR_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_creative_diversity_audit_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    predictions: tuple[CreativeDiversityPrediction, ...] = Field(
        min_length=1,
        max_length=12,
    )
    prediction_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    source_budget_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_prediction_id: str = Field(min_length=1, max_length=220)
    recommended_diversity_band: CreativeDiversityPredictionBand
    recommended_diversity_readiness_score: int = Field(ge=0, le=100)
    fallback_prediction_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    prediction_count: int = Field(ge=1, le=12)
    broad_prediction_count: int = Field(ge=0, le=12)
    guarded_prediction_count: int = Field(ge=0, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    creative_diversity_predictor_implemented: Literal[True] = True
    advisory_diversity_prediction_implemented: Literal[True] = True
    active_diversity_generation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
        if self.source_budget_profile_ids != tuple(
            prediction.source_budget_profile_id for prediction in self.predictions
        ):
            raise ValueError("source_budget_profile_ids must match predictions")
        if self.prediction_count != len(self.predictions):
            raise ValueError("prediction_count must match predictions")

        recommended = tuple(
            prediction
            for prediction in self.predictions
            if prediction.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended diversity prediction is required")
        recommended_prediction = recommended[0]
        if self.recommended_prediction_id != recommended_prediction.prediction_id:
            raise ValueError("recommended_prediction_id must match prediction")
        if self.recommended_diversity_band != (
            recommended_prediction.predicted_diversity_band
        ):
            raise ValueError("recommended_diversity_band must match prediction")
        if self.recommended_diversity_readiness_score != (
            recommended_prediction.diversity_readiness_score
        ):
            raise ValueError(
                "recommended_diversity_readiness_score must match prediction"
            )
        if self.fallback_prediction_ids != tuple(
            prediction.prediction_id
            for prediction in self.predictions
            if prediction.status == "fallback"
        ):
            raise ValueError("fallback_prediction_ids must match predictions")
        if self.broad_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_diversity_band == "broad"
        ):
            raise ValueError("broad_prediction_count must match predictions")
        if self.guarded_prediction_count != sum(
            1
            for prediction in self.predictions
            if prediction.predicted_diversity_band == "guarded"
        ):
            raise ValueError("guarded_prediction_count must match predictions")
        return self


def predict_creative_diversity(
    *,
    diversity_audit: CreativeDiversityAuditRegistry | None = None,
) -> CreativeDiversityPredictionPlan:
    """Return advisory creative diversity predictions without generating variants."""

    audit_registry = diversity_audit or creative_diversity_audit_registry()
    recommended_budget_id = _recommended_budget_profile_id(audit_registry.audit_records)
    predictions = tuple(
        _prediction_from_audit_record(
            record,
            status=(
                "recommended"
                if record.budget_profile_id == recommended_budget_id
                else "fallback"
            ),
        )
        for record in audit_registry.audit_records
    )
    recommended = _recommended_prediction(predictions)

    return CreativeDiversityPredictionPlan(
        source_creative_diversity_audit_serialization_version=(
            audit_registry.serialization_version
        ),
        predictions=predictions,
        prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
        source_budget_profile_ids=tuple(
            prediction.source_budget_profile_id for prediction in predictions
        ),
        recommended_prediction_id=recommended.prediction_id,
        recommended_diversity_band=recommended.predicted_diversity_band,
        recommended_diversity_readiness_score=recommended.diversity_readiness_score,
        fallback_prediction_ids=tuple(
            prediction.prediction_id
            for prediction in predictions
            if prediction.status == "fallback"
        ),
        prediction_count=len(predictions),
        broad_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_diversity_band == "broad"
        ),
        guarded_prediction_count=sum(
            1
            for prediction in predictions
            if prediction.predicted_diversity_band == "guarded"
        ),
        advisory_actions=(
            "Present diversity posture predictions for review.",
            "Keep variant generation and refinement triggering disabled.",
        ),
    )


def creative_diversity_prediction_by_id(
    prediction_id: str,
    plan: CreativeDiversityPredictionPlan | None = None,
) -> CreativeDiversityPrediction | None:
    """Return one advisory diversity prediction without generating variants."""

    source_plan = plan or predict_creative_diversity()
    for prediction in source_plan.predictions:
        if prediction.prediction_id == prediction_id:
            return prediction
    return None


def creative_diversity_predictions_for_band(
    diversity_band: CreativeDiversityPredictionBand,
    plan: CreativeDiversityPredictionPlan | None = None,
) -> tuple[CreativeDiversityPrediction, ...]:
    """Return advisory diversity predictions for one diversity band."""

    source_plan = plan or predict_creative_diversity()
    return tuple(
        prediction
        for prediction in source_plan.predictions
        if prediction.predicted_diversity_band == diversity_band
    )


def _prediction_from_audit_record(
    record: CreativeDiversityAuditRecord,
    *,
    status: CreativeDiversityPredictionStatus,
) -> CreativeDiversityPrediction:
    return CreativeDiversityPrediction(
        prediction_id=f"creative_diversity_prediction::{record.budget_profile_id}",
        source_budget_profile_id=record.budget_profile_id,
        source_topic_id=record.topic_id,
        predicted_diversity_band=_diversity_band(record.budget_posture),
        max_advisory_variants=record.max_advisory_variants,
        max_advisory_refinement_passes=record.max_advisory_refinement_passes,
        predicted_variant_range=(0, record.max_advisory_variants),
        diversity_readiness_score=_readiness_score(
            variants=record.max_advisory_variants,
            refinements=record.max_advisory_refinement_passes,
        ),
        source_trace_profile_id=record.source_trace_profile_id,
        source_provenance_profile_id=record.source_provenance_profile_id,
        evidence=(
            f"Derived from {record.budget_profile_id}.",
            f"Budget posture: {record.budget_posture}.",
            f"Audit status: {record.audit_status}.",
            "Diversity prediction is advisory metadata only.",
        ),
        advisory_actions=(
            "Surface diversity posture without generating variants.",
            "Keep budget enforcement and refinement triggering disabled.",
        ),
        status=status,
    )


def _diversity_band(budget_posture: str) -> CreativeDiversityPredictionBand:
    if budget_posture == "broad":
        return "broad"
    if budget_posture == "guarded":
        return "guarded"
    if budget_posture == "narrow":
        return "narrow"
    return "moderate"


def _recommended_budget_profile_id(
    records: tuple[CreativeDiversityAuditRecord, ...],
) -> str:
    ordered = sorted(
        records,
        key=lambda record: (
            -_readiness_score(
                variants=record.max_advisory_variants,
                refinements=record.max_advisory_refinement_passes,
            ),
            record.budget_profile_id,
        ),
    )
    return ordered[0].budget_profile_id


def _recommended_prediction(
    predictions: tuple[CreativeDiversityPrediction, ...],
) -> CreativeDiversityPrediction:
    for prediction in predictions:
        if prediction.status == "recommended":
            return prediction
    raise ValueError("creative diversity prediction requires a recommended prediction")


def _readiness_score(*, variants: int, refinements: int) -> int:
    return min(100, 30 + variants * 18 + refinements * 8)
