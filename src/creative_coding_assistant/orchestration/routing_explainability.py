"""V5.2 advisory routing explainability metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cost_prediction_engine import (
    CostPredictionPlan,
    predict_cost_for_route,
)
from creative_coding_assistant.orchestration.hybrid_routing import (
    HybridRoutingPlan,
    route_hybrid_model_request,
)
from creative_coding_assistant.orchestration.local_cloud_routing import (
    LocalCloudRoutingPlan,
    route_local_vs_cloud,
)
from creative_coding_assistant.orchestration.model_recommendation_engine import (
    ModelRecommendationPlan,
    recommend_model_profile,
)
from creative_coding_assistant.orchestration.model_router import (
    ModelRoutingPlan,
    route_model_request,
)
from creative_coding_assistant.orchestration.quality_prediction_engine import (
    QualityPredictionPlan,
    predict_quality_for_route,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

RoutingExplanationSource = Literal[
    "model_recommendation",
    "model_routing",
    "local_cloud_routing",
    "hybrid_routing",
    "quality_prediction",
    "cost_prediction",
]
RoutingExplanationStatus = Literal["primary", "supporting"]

ROUTING_EXPLANATION_RECORD_SERIALIZATION_VERSION = "routing_explanation_record.v1"
ROUTING_EXPLAINABILITY_PLAN_SERIALIZATION_VERSION = "routing_explainability_plan.v1"
ROUTING_EXPLAINABILITY_AUTHORITY_BOUNDARY = (
    "The V5.2 Routing Explainability surface converts existing advisory "
    "routing, recommendation, quality prediction, and cost prediction metadata "
    "into explanation records only; it does not select or switch providers or "
    "models, apply routing decisions, emit human input requests, enforce "
    "budgets, execute providers, control workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "configured_provider_switching",
    "configured_model_switching",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "routing_application",
    "human_input_request_emission",
    "budget_enforcement",
    "provider_execution",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class RoutingExplanationRecord(BaseModel):
    """One advisory explanation for a V5.2 routing source."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    explanation_id: str = Field(min_length=1, max_length=180)
    explanation_rank: int = Field(ge=1, le=8)
    source_surface: RoutingExplanationSource
    source_record_id: str = Field(min_length=1, max_length=180)
    route_name: RouteName
    explanation_summary: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    referenced_advisory_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    status: RoutingExplanationStatus
    routing_explainability_implemented: Literal[True] = True
    advisory_explanation_generation_implemented: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_explanation_record.v1"] = (
        ROUTING_EXPLANATION_RECORD_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_matches_rank_and_status(self) -> Self:
        if self.status == "primary" and self.explanation_rank != 1:
            raise ValueError("primary explanation must be rank 1")
        if self.status == "supporting" and self.explanation_rank == 1:
            raise ValueError("rank 1 explanation must be primary")
        if self.explanation_id != f"routing_explanation::{self.source_surface}":
            raise ValueError("explanation_id must match source_surface")
        return self


class RoutingExplainabilityPlan(BaseModel):
    """Bounded V5.2 advisory routing explainability plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_explainability"] = "routing_explainability"
    serialization_version: Literal["routing_explainability_plan.v1"] = (
        ROUTING_EXPLAINABILITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ROUTING_EXPLAINABILITY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    route_name: RouteName
    source_model_routing_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_local_cloud_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_hybrid_routing_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_quality_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_cost_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_model_recommendation_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    recommended_model_profile_id: str = Field(min_length=1, max_length=120)
    recommended_model_recommendation_id: str = Field(min_length=1, max_length=180)
    explanations: tuple[RoutingExplanationRecord, ...] = Field(
        min_length=1,
        max_length=8,
    )
    explanation_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_surfaces: tuple[RoutingExplanationSource, ...] = Field(
        min_length=1,
        max_length=8,
    )
    primary_explanation_id: str = Field(min_length=1, max_length=180)
    explanation_count: int = Field(ge=1, le=8)
    source_surface_count: int = Field(ge=1, le=8)
    route_consistency_confirmed: Literal[True] = True
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    routing_explainability_implemented: Literal[True] = True
    advisory_explanation_generation_implemented: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    configured_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_explanations(self) -> Self:
        derived_explanation_ids = tuple(
            explanation.explanation_id for explanation in self.explanations
        )
        if len(set(derived_explanation_ids)) != len(derived_explanation_ids):
            raise ValueError("explanation_ids must be unique")
        if self.explanation_ids != derived_explanation_ids:
            raise ValueError("explanation_ids must match explanations")
        if self.source_surfaces != tuple(
            explanation.source_surface for explanation in self.explanations
        ):
            raise ValueError("source_surfaces must match explanations")
        if self.explanation_count != len(self.explanations):
            raise ValueError("explanation_count must match explanations")
        if self.source_surface_count != len(set(self.source_surfaces)):
            raise ValueError("source_surface_count must match source_surfaces")

        primary = tuple(
            explanation
            for explanation in self.explanations
            if explanation.status == "primary"
        )
        if len(primary) != 1:
            raise ValueError("exactly one primary routing explanation is required")
        primary_explanation = primary[0]
        if self.primary_explanation_id != primary_explanation.explanation_id:
            raise ValueError("primary_explanation_id must match explanation")
        for explanation in self.explanations:
            if explanation.route_name != self.route_name:
                raise ValueError("explanation route_name must match plan route_name")
        return self


def explain_routing_decision(
    *,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    model_routing: ModelRoutingPlan | None = None,
    local_cloud_routing: LocalCloudRoutingPlan | None = None,
    hybrid_routing: HybridRoutingPlan | None = None,
    quality_prediction: QualityPredictionPlan | None = None,
    cost_prediction: CostPredictionPlan | None = None,
    model_recommendation: ModelRecommendationPlan | None = None,
) -> RoutingExplainabilityPlan:
    """Return advisory routing explanations without applying routing behavior."""

    model_plan = model_routing or route_model_request(
        route_decision=route_decision,
        route=route,
    )
    route_name = model_plan.route_name
    _assert_route(route_decision=route_decision, route=route, route_name=route_name)
    local_cloud_plan = local_cloud_routing or route_local_vs_cloud(
        model_routing=model_plan,
        route_decision=route_decision,
        route=route_name,
    )
    hybrid_plan = hybrid_routing or route_hybrid_model_request(
        local_cloud_routing=local_cloud_plan,
        model_routing=model_plan,
        route_decision=route_decision,
        route=route_name,
    )
    quality_plan = quality_prediction or predict_quality_for_route(route=route_name)
    cost_plan = cost_prediction or predict_cost_for_route(route=route_name)
    recommendation_plan = model_recommendation or recommend_model_profile(
        model_routing=model_plan,
        route_decision=route_decision,
        route=route_name,
    )
    _assert_source_routes(
        route_name,
        local_cloud_plan,
        hybrid_plan,
        quality_plan,
        cost_plan,
        recommendation_plan,
    )
    explanations = _explanations(
        model_routing=model_plan,
        local_cloud_routing=local_cloud_plan,
        hybrid_routing=hybrid_plan,
        quality_prediction=quality_plan,
        cost_prediction=cost_plan,
        model_recommendation=recommendation_plan,
    )
    primary = _primary_explanation(explanations)

    return RoutingExplainabilityPlan(
        route_name=route_name,
        source_model_routing_serialization_version=model_plan.serialization_version,
        source_local_cloud_serialization_version=local_cloud_plan.serialization_version,
        source_hybrid_routing_serialization_version=hybrid_plan.serialization_version,
        source_quality_prediction_serialization_version=(
            quality_plan.serialization_version
        ),
        source_cost_prediction_serialization_version=cost_plan.serialization_version,
        source_model_recommendation_serialization_version=(
            recommendation_plan.serialization_version
        ),
        recommended_model_profile_id=recommendation_plan.recommended_model_profile_id,
        recommended_model_recommendation_id=(
            recommendation_plan.recommended_recommendation_id
        ),
        explanations=explanations,
        explanation_ids=tuple(
            explanation.explanation_id for explanation in explanations
        ),
        source_surfaces=tuple(
            explanation.source_surface for explanation in explanations
        ),
        primary_explanation_id=primary.explanation_id,
        explanation_count=len(explanations),
        source_surface_count=len(
            {explanation.source_surface for explanation in explanations}
        ),
        advisory_actions=(
            "Surface advisory routing rationale for operator review.",
            "Keep provider/model routing, execution, budget enforcement, and storage disabled.",
        ),
    )


def routing_explanation_by_id(
    explanation_id: str,
    plan: RoutingExplainabilityPlan | None = None,
) -> RoutingExplanationRecord | None:
    """Return one routing explanation without applying routing behavior."""

    source_plan = plan or explain_routing_decision()
    for explanation in source_plan.explanations:
        if explanation.explanation_id == explanation_id:
            return explanation
    return None


def routing_explanations_for_source(
    source_surface: RoutingExplanationSource,
    plan: RoutingExplainabilityPlan | None = None,
) -> tuple[RoutingExplanationRecord, ...]:
    """Return routing explanations for one advisory source surface."""

    source_plan = plan or explain_routing_decision()
    return tuple(
        explanation
        for explanation in source_plan.explanations
        if explanation.source_surface == source_surface
    )


def _explanations(
    *,
    model_routing: ModelRoutingPlan,
    local_cloud_routing: LocalCloudRoutingPlan,
    hybrid_routing: HybridRoutingPlan,
    quality_prediction: QualityPredictionPlan,
    cost_prediction: CostPredictionPlan,
    model_recommendation: ModelRecommendationPlan,
) -> tuple[RoutingExplanationRecord, ...]:
    model_recommendation_record = _record(
        rank=1,
        source_surface="model_recommendation",
        source_record_id=model_recommendation.recommended_recommendation_id,
        route_name=model_recommendation.route_name,
        summary=(
            "Model recommendation is the primary explanation because it combines "
            f"model fit with {model_recommendation.recommended_execution_policy_posture}."
        ),
        evidence=(
            f"Recommended model profile: {model_recommendation.recommended_model_profile_id}.",
            f"Recommendation id: {model_recommendation.recommended_recommendation_id}.",
            f"Recommendation count: {model_recommendation.recommendation_count}.",
            "Recommendation remains advisory metadata only.",
        ),
        actions=model_recommendation.advisory_actions,
        status="primary",
    )
    return (
        model_recommendation_record,
        _record(
            rank=2,
            source_surface="model_routing",
            source_record_id=model_routing.recommended_candidate_id,
            route_name=model_routing.route_name,
            summary=(
                f"Model routing ranked {model_routing.recommended_model_profile_id} "
                f"for the {model_routing.route_name.value} route."
            ),
            evidence=(
                f"Recommended candidate: {model_routing.recommended_candidate_id}.",
                f"Recommendation confidence: {model_routing.recommendation_confidence}.",
                f"Candidate count: {model_routing.candidate_count}.",
            ),
            actions=model_routing.advisory_actions,
        ),
        _record(
            rank=3,
            source_surface="local_cloud_routing",
            source_record_id=local_cloud_routing.recommended_decision_id,
            route_name=local_cloud_routing.route_name,
            summary=(
                "Local/cloud comparison explains the advisory posture as "
                f"{local_cloud_routing.recommended_routing_posture}."
            ),
            evidence=(
                f"Recommended decision: {local_cloud_routing.recommended_decision_id}.",
                f"Routing lane: {local_cloud_routing.recommended_routing_lane}.",
                f"Routing confidence: {local_cloud_routing.routing_confidence}.",
            ),
            actions=local_cloud_routing.advisory_actions,
        ),
        _record(
            rank=4,
            source_surface="hybrid_routing",
            source_record_id=hybrid_routing.recommended_decision_id,
            route_name=hybrid_routing.route_name,
            summary=(
                "Hybrid routing explains the derived advisory mode as "
                f"{hybrid_routing.recommended_hybrid_mode}."
            ),
            evidence=(
                f"Recommended decision: {hybrid_routing.recommended_decision_id}.",
                f"Routing confidence: {hybrid_routing.routing_confidence}.",
                f"Decision count: {hybrid_routing.decision_count}.",
            ),
            actions=hybrid_routing.advisory_actions,
        ),
        _record(
            rank=5,
            source_surface="quality_prediction",
            source_record_id=quality_prediction.recommended_prediction_id,
            route_name=quality_prediction.route_name,
            summary=(
                "Quality prediction explains expected relative quality as "
                f"{quality_prediction.recommended_quality_level}."
            ),
            evidence=(
                f"Recommended prediction: {quality_prediction.recommended_prediction_id}.",
                f"Quality midpoint: {quality_prediction.recommended_quality_midpoint}.",
                f"Prediction count: {quality_prediction.prediction_count}.",
            ),
            actions=quality_prediction.advisory_actions,
        ),
        _record(
            rank=6,
            source_surface="cost_prediction",
            source_record_id=cost_prediction.recommended_prediction_id,
            route_name=cost_prediction.route_name,
            summary=(
                "Cost prediction explains expected relative cost as "
                f"{cost_prediction.recommended_cost_band}."
            ),
            evidence=(
                f"Recommended prediction: {cost_prediction.recommended_prediction_id}.",
                f"Cost midpoint: {cost_prediction.recommended_cost_midpoint}.",
                f"Prediction count: {cost_prediction.prediction_count}.",
            ),
            actions=cost_prediction.advisory_actions,
        ),
    )


def _record(
    *,
    rank: int,
    source_surface: RoutingExplanationSource,
    source_record_id: str,
    route_name: RouteName,
    summary: str,
    evidence: tuple[str, ...],
    actions: tuple[str, ...],
    status: RoutingExplanationStatus = "supporting",
) -> RoutingExplanationRecord:
    return RoutingExplanationRecord(
        explanation_id=f"routing_explanation::{source_surface}",
        explanation_rank=rank,
        source_surface=source_surface,
        source_record_id=source_record_id,
        route_name=route_name,
        explanation_summary=summary,
        evidence=evidence,
        referenced_advisory_actions=actions[:8],
        status=status,
    )


def _primary_explanation(
    explanations: tuple[RoutingExplanationRecord, ...],
) -> RoutingExplanationRecord:
    for explanation in explanations:
        if explanation.status == "primary":
            return explanation
    raise ValueError("routing explainability requires a primary explanation")


def _assert_route(
    *,
    route_decision: RouteDecision | None,
    route: RouteName | str | None,
    route_name: RouteName,
) -> None:
    explicit_route = (
        None
        if route is None
        else (route if isinstance(route, RouteName) else RouteName(str(route)))
    )
    if explicit_route is not None and explicit_route != route_name:
        raise ValueError("route must match routing explainability route")
    if route_decision is not None and route_decision.route != route_name:
        raise ValueError("route_decision must match routing explainability route")


def _assert_source_routes(
    route_name: RouteName,
    *plans: (
        LocalCloudRoutingPlan
        | HybridRoutingPlan
        | QualityPredictionPlan
        | CostPredictionPlan
        | ModelRecommendationPlan
    ),
) -> None:
    for plan in plans:
        if plan.route_name != route_name:
            raise ValueError("source route must match routing explainability route")
