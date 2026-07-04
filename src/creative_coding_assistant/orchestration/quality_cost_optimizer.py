"""V5.2 advisory quality/cost optimizer for routing metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_routing import (
    HybridRouteDecision,
    HybridRoutingMode,
    HybridRoutingPlan,
    route_hybrid_model_request,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    CostProfile,
    CostProfileBand,
    CostProfileRegistry,
    QualityProfile,
    QualityProfileLevel,
    QualityProfileRegistry,
    cost_profile_registry,
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

QualityCostTradeoffPosture = Literal[
    "quality_favored",
    "cost_favored",
    "balanced_tradeoff",
]
QualityCostOptimizationStatus = Literal["recommended", "fallback"]

QUALITY_COST_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "quality_cost_optimization_candidate.v1"
)
QUALITY_COST_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "quality_cost_optimization_plan.v1"
)
QUALITY_COST_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "The V5.2 Quality/Cost Optimizer derives advisory tradeoff scores from "
    "existing passive quality profiles, cost profiles, and hybrid routing "
    "metadata only; it does not estimate live provider cost, look up pricing, "
    "predict quality, predict cost, enforce budgets, select or switch "
    "providers or models, execute providers, control workflows, trigger "
    "retries, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_pricing_lookup",
    "live_cost_estimation",
    "quality_prediction",
    "cost_prediction",
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
_QUALITY_SCORE_BY_LEVEL: dict[QualityProfileLevel, int] = {
    "low": 25,
    "medium": 55,
    "high": 80,
    "critical": 90,
}
_COST_SCORE_BY_BAND: dict[CostProfileBand, int] = {
    "low": 85,
    "medium": 65,
    "guarded": 50,
    "high": 30,
}
_HYBRID_SCORE_BONUS: dict[HybridRoutingMode, int] = {
    "balanced_dual": 10,
    "local_primary": 8,
    "cloud_primary": 6,
}


class QualityCostOptimizationCandidate(BaseModel):
    """One advisory quality/cost tradeoff candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_hybrid_decision_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    hybrid_mode: HybridRoutingMode
    source_quality_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_cost_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    quality_level: QualityProfileLevel
    cost_band: CostProfileBand
    advisory_cost_range: tuple[int, int]
    quality_score: int = Field(ge=0, le=100)
    cost_score: int = Field(ge=0, le=100)
    hybrid_bonus: int = Field(ge=0, le=20)
    optimization_score: int = Field(ge=0, le=220)
    tradeoff_posture: QualityCostTradeoffPosture
    status: QualityCostOptimizationStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    quality_cost_optimization_implemented: Literal[True] = True
    quality_cost_scoring_implemented: Literal[True] = True
    quality_prediction_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    cost_estimation_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["quality_cost_optimization_candidate.v1"] = (
        QUALITY_COST_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_scores(self) -> Self:
        if self.quality_score != _QUALITY_SCORE_BY_LEVEL[self.quality_level]:
            raise ValueError("quality_score must match quality_level")
        if self.cost_score != _COST_SCORE_BY_BAND[self.cost_band]:
            raise ValueError("cost_score must match cost_band")
        if self.hybrid_bonus != _HYBRID_SCORE_BONUS[self.hybrid_mode]:
            raise ValueError("hybrid_bonus must match hybrid_mode")
        if self.optimization_score != (
            self.quality_score + self.cost_score + self.hybrid_bonus
        ):
            raise ValueError("optimization_score must combine scores")
        if self.tradeoff_posture != _tradeoff_posture(
            quality_score=self.quality_score,
            cost_score=self.cost_score,
        ):
            raise ValueError("tradeoff_posture must match quality/cost scores")
        if self.advisory_cost_range[0] > self.advisory_cost_range[1]:
            raise ValueError("advisory_cost_range must be ordered")
        return self


class QualityCostOptimizationPlan(BaseModel):
    """Bounded V5.2 advisory quality/cost optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["quality_cost_optimizer"] = "quality_cost_optimizer"
    serialization_version: Literal["quality_cost_optimization_plan.v1"] = (
        QUALITY_COST_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=QUALITY_COST_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_hybrid_routing_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_quality_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_cost_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_hybrid_decision_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    candidates: tuple[QualityCostOptimizationCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_candidate_id: str = Field(min_length=1, max_length=180)
    recommended_tradeoff_posture: QualityCostTradeoffPosture
    fallback_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=12
    )
    candidate_count: int = Field(ge=1, le=12)
    recommended_optimization_score: int = Field(ge=0, le=220)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    quality_cost_optimization_implemented: Literal[True] = True
    quality_cost_scoring_implemented: Literal[True] = True
    quality_prediction_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    cost_estimation_implemented: Literal[False] = False
    pricing_lookup_implemented: Literal[False] = False
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
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.source_hybrid_decision_ids != tuple(
            candidate.source_hybrid_decision_id for candidate in self.candidates
        ):
            raise ValueError("source_hybrid_decision_ids must match candidates")

        recommended = tuple(
            candidate
            for candidate in self.candidates
            if candidate.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended candidate is required")
        recommended_candidate = recommended[0]
        if self.recommended_candidate_id != recommended_candidate.candidate_id:
            raise ValueError("recommended_candidate_id must match candidate")
        if self.recommended_tradeoff_posture != recommended_candidate.tradeoff_posture:
            raise ValueError("recommended_tradeoff_posture must match candidate")
        if (
            self.recommended_optimization_score
            != recommended_candidate.optimization_score
        ):
            raise ValueError("recommended_optimization_score must match candidate")
        if self.fallback_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status == "fallback"
        ):
            raise ValueError("fallback_candidate_ids must match candidates")
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan route_name")
        return self


def optimize_quality_cost(
    *,
    hybrid_routing: HybridRoutingPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    quality_profiles: QualityProfileRegistry | None = None,
    cost_profiles: CostProfileRegistry | None = None,
) -> QualityCostOptimizationPlan:
    """Return advisory quality/cost tradeoff metadata without applying routing."""

    hybrid_plan = hybrid_routing or route_hybrid_model_request(
        route_decision=route_decision,
        route=route,
    )
    quality_registry = quality_profiles or quality_profile_registry()
    cost_registry = cost_profiles or cost_profile_registry()
    candidates = tuple(
        _candidate_from_hybrid_decision(
            decision=decision,
            quality_profiles=_matching_quality_profiles(
                decision,
                quality_registry,
            ),
            cost_profiles=_matching_cost_profiles(decision, cost_registry),
        )
        for decision in hybrid_plan.decisions
    )
    recommended = _recommended_candidate(candidates)

    return QualityCostOptimizationPlan(
        source_hybrid_routing_serialization_version=hybrid_plan.serialization_version,
        source_quality_profile_serialization_version=quality_registry.serialization_version,
        source_cost_profile_serialization_version=cost_registry.serialization_version,
        route_name=hybrid_plan.route_name,
        source_hybrid_decision_ids=tuple(
            candidate.source_hybrid_decision_id for candidate in candidates
        ),
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_id=recommended.candidate_id,
        recommended_tradeoff_posture=recommended.tradeoff_posture,
        fallback_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status == "fallback"
        ),
        candidate_count=len(candidates),
        recommended_optimization_score=recommended.optimization_score,
        advisory_actions=_plan_actions(hybrid_plan.route_name, recommended),
    )


def quality_cost_candidate_by_id(
    candidate_id: str,
    plan: QualityCostOptimizationPlan | None = None,
) -> QualityCostOptimizationCandidate | None:
    """Return one advisory quality/cost candidate without applying it."""

    source_plan = plan or optimize_quality_cost()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def quality_cost_candidates_for_posture(
    posture: QualityCostTradeoffPosture,
    plan: QualityCostOptimizationPlan | None = None,
) -> tuple[QualityCostOptimizationCandidate, ...]:
    """Return advisory quality/cost candidates for a tradeoff posture."""

    source_plan = plan or optimize_quality_cost()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.tradeoff_posture == posture
    )


def _candidate_from_hybrid_decision(
    *,
    decision: HybridRouteDecision,
    quality_profiles: tuple[QualityProfile, ...],
    cost_profiles: tuple[CostProfile, ...],
) -> QualityCostOptimizationCandidate:
    if not quality_profiles:
        raise ValueError("quality/cost optimizer requires quality profile metadata")
    if not cost_profiles:
        raise ValueError("quality/cost optimizer requires cost profile metadata")
    quality_profile = max(
        quality_profiles,
        key=lambda profile: _QUALITY_SCORE_BY_LEVEL[profile.quality_level],
    )
    cost_profile = max(
        cost_profiles,
        key=lambda profile: _COST_SCORE_BY_BAND[profile.cost_band],
    )
    quality_score = _QUALITY_SCORE_BY_LEVEL[quality_profile.quality_level]
    cost_score = _COST_SCORE_BY_BAND[cost_profile.cost_band]
    hybrid_bonus = _HYBRID_SCORE_BONUS[decision.hybrid_mode]
    advisory_ranges = tuple(profile.advisory_cost_range for profile in cost_profiles)
    return QualityCostOptimizationCandidate(
        candidate_id=f"quality_cost::{decision.source_model_profile_id}",
        source_hybrid_decision_id=decision.decision_id,
        source_model_profile_id=decision.source_model_profile_id,
        route_name=decision.route_name,
        hybrid_mode=decision.hybrid_mode,
        source_quality_profile_ids=tuple(
            profile.quality_profile_id for profile in quality_profiles
        ),
        source_cost_profile_ids=tuple(
            profile.cost_profile_id for profile in cost_profiles
        ),
        quality_level=quality_profile.quality_level,
        cost_band=cost_profile.cost_band,
        advisory_cost_range=(
            min(cost_range[0] for cost_range in advisory_ranges),
            max(cost_range[1] for cost_range in advisory_ranges),
        ),
        quality_score=quality_score,
        cost_score=cost_score,
        hybrid_bonus=hybrid_bonus,
        optimization_score=quality_score + cost_score + hybrid_bonus,
        tradeoff_posture=_tradeoff_posture(
            quality_score=quality_score,
            cost_score=cost_score,
        ),
        status=decision.status,
        evidence=(
            f"Quality profiles: {len(quality_profiles)}.",
            f"Cost profiles: {len(cost_profiles)}.",
            f"Hybrid mode: {decision.hybrid_mode}.",
        ),
        advisory_actions=_candidate_actions(quality_profile, cost_profile),
    )


def _matching_quality_profiles(
    decision: HybridRouteDecision,
    registry: QualityProfileRegistry,
) -> tuple[QualityProfile, ...]:
    route_matches = tuple(
        profile
        for profile in registry.quality_profiles
        if decision.source_model_profile_id in profile.source_model_profile_ids
        and decision.route_name in profile.route_applicability
    )
    if route_matches:
        return route_matches
    return tuple(
        profile
        for profile in registry.quality_profiles
        if decision.source_model_profile_id in profile.source_model_profile_ids
    )


def _matching_cost_profiles(
    decision: HybridRouteDecision,
    registry: CostProfileRegistry,
) -> tuple[CostProfile, ...]:
    route_matches = tuple(
        profile
        for profile in registry.cost_profiles
        if decision.source_model_profile_id in profile.source_model_profile_ids
        and decision.route_name in profile.route_applicability
    )
    if route_matches:
        return route_matches
    return tuple(
        profile
        for profile in registry.cost_profiles
        if decision.source_model_profile_id in profile.source_model_profile_ids
    )


def _tradeoff_posture(
    *,
    quality_score: int,
    cost_score: int,
) -> QualityCostTradeoffPosture:
    if quality_score - cost_score >= 30:
        return "quality_favored"
    if cost_score - quality_score >= 30:
        return "cost_favored"
    return "balanced_tradeoff"


def _recommended_candidate(
    candidates: tuple[QualityCostOptimizationCandidate, ...],
) -> QualityCostOptimizationCandidate:
    for candidate in candidates:
        if candidate.status == "recommended":
            return candidate
    raise ValueError("quality/cost optimizer requires a recommended candidate")


def _candidate_actions(
    quality_profile: QualityProfile,
    cost_profile: CostProfile,
) -> tuple[str, ...]:
    return (
        f"Compare {quality_profile.quality_profile_id} with {cost_profile.cost_profile_id}.",
        "Keep pricing lookup, budget enforcement, and provider routing disabled.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: QualityCostOptimizationCandidate,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.tradeoff_posture} posture for the "
            f"{route_name.value} route."
        ),
        "Use existing passive quality and cost metadata only.",
        "Defer cost estimation, prediction, budget, policy, and execution behavior.",
    )
