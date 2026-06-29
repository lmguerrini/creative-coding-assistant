"""V5.2 advisory model router over existing model profile metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    ModelProfile,
    ModelProfileKind,
    ModelProfileRegistry,
    ProviderSelectionRegistry,
    model_profile_registry,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

ModelRouteCandidateStatus = Literal["recommended", "fallback"]
ModelRouteFitBand = Literal["eligible", "moderate", "strong"]
ModelRouteConfidence = Literal["low", "medium", "high"]

MODEL_ROUTE_CANDIDATE_SERIALIZATION_VERSION = "model_route_candidate.v1"
MODEL_ROUTING_PLAN_SERIALIZATION_VERSION = "model_routing_plan.v1"
MODEL_ROUTER_AUTHORITY_BOUNDARY = (
    "The V5.2 Model Router ranks existing passive model profiles for route "
    "applicability and advisory capability fit only; it does not select or "
    "switch the configured provider or model, call providers, perform "
    "local/cloud routing, perform hybrid routing, optimize cost or quality, "
    "enforce budgets, control workflow execution, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "configured_provider_switching",
    "configured_model_switching",
    "automatic_model_selection",
    "provider_or_model_routing",
    "provider_execution",
    "local_cloud_routing",
    "hybrid_routing",
    "cost_optimization",
    "quality_optimization",
    "quality_prediction",
    "cost_prediction",
    "budget_enforcement",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)

_ROUTE_KIND_BONUS: dict[RouteName, dict[ModelProfileKind, int]] = {
    RouteName.GENERATE: {
        "creative_reasoning": 30,
        "code_assistance": 20,
        "fast_iteration": 10,
    },
    RouteName.EXPLAIN: {
        "creative_reasoning": 25,
        "evaluation_review": 20,
        "fast_iteration": 10,
    },
    RouteName.DEBUG: {
        "code_assistance": 30,
        "fast_iteration": 20,
    },
    RouteName.DESIGN: {
        "creative_reasoning": 30,
        "evaluation_review": 15,
    },
    RouteName.REVIEW: {
        "evaluation_review": 30,
        "code_assistance": 20,
        "creative_reasoning": 15,
    },
    RouteName.PREVIEW: {
        "fast_iteration": 25,
        "code_assistance": 20,
    },
}


class ModelRouteCandidate(BaseModel):
    """One advisory model-profile route candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    rank: int = Field(ge=1, le=12)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    model_profile_kind: ModelProfileKind
    profile_name: str = Field(min_length=1, max_length=160)
    route_name: RouteName
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    capability_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    fit_score: int = Field(ge=0, le=250)
    fit_band: ModelRouteFitBand
    status: ModelRouteCandidateStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    model_router_implemented: Literal[True] = True
    model_route_recommendation_implemented: Literal[True] = True
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_cloud_routing_implemented: Literal[False] = False
    hybrid_routing_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    quality_optimization_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["model_route_candidate.v1"] = (
        MODEL_ROUTE_CANDIDATE_SERIALIZATION_VERSION
    )
    recommendation_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_route_and_rank(self) -> Self:
        if self.route_name not in self.route_applicability:
            raise ValueError("route_name must be present in route_applicability")
        if self.status == "recommended" and self.rank != 1:
            raise ValueError("recommended candidate must be rank 1")
        if self.status == "fallback" and self.rank == 1:
            raise ValueError("rank 1 candidate must be recommended")
        if self.fit_band != _fit_band(self.fit_score):
            raise ValueError("fit_band must match fit_score")
        return self


class ModelRoutingPlan(BaseModel):
    """Bounded V5.2 advisory model routing plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_router"] = "model_router"
    serialization_version: Literal["model_routing_plan.v1"] = (
        MODEL_ROUTING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MODEL_ROUTER_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_model_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_provider_selection_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    source_provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    candidates: tuple[ModelRouteCandidate, ...] = Field(min_length=1, max_length=12)
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_candidate_id: str = Field(min_length=1, max_length=180)
    recommended_model_profile_id: str = Field(min_length=1, max_length=120)
    fallback_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    candidate_count: int = Field(ge=1, le=12)
    recommendation_confidence: ModelRouteConfidence
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    model_router_implemented: Literal[True] = True
    model_route_recommendation_implemented: Literal[True] = True
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_cloud_routing_implemented: Literal[False] = False
    hybrid_routing_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    quality_optimization_implemented: Literal[False] = False
    quality_prediction_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    recommendation_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.source_model_profile_ids != tuple(
            candidate.source_model_profile_id for candidate in self.candidates
        ):
            raise ValueError("source_model_profile_ids must match candidates")

        recommended = tuple(
            candidate for candidate in self.candidates if candidate.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended candidate is required")
        recommended_candidate = recommended[0]
        if self.recommended_candidate_id != recommended_candidate.candidate_id:
            raise ValueError("recommended_candidate_id must match recommended candidate")
        if (
            self.recommended_model_profile_id
            != recommended_candidate.source_model_profile_id
        ):
            raise ValueError("recommended_model_profile_id must match candidate")
        if self.fallback_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status == "fallback"
        ):
            raise ValueError("fallback_candidate_ids must match candidates")
        if self.recommendation_confidence != _confidence(recommended_candidate.fit_score):
            raise ValueError("recommendation_confidence must match candidate score")

        known_providers = set(self.source_provider_candidate_ids)
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan route_name")
            if not set(candidate.provider_candidate_ids).issubset(known_providers):
                raise ValueError("provider_candidate_ids must be known providers")
        return self


def route_model_request(
    *,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    model_profiles: ModelProfileRegistry | None = None,
    provider_selection: ProviderSelectionRegistry | None = None,
) -> ModelRoutingPlan:
    """Return advisory model route recommendations without applying them."""

    route_name = _resolve_route(route_decision=route_decision, route=route)
    model_registry = model_profiles or model_profile_registry()
    provider_registry = provider_selection or provider_selection_registry()
    candidates = _candidates_for_route(route_name, model_registry)
    if not candidates:
        raise ValueError("model router requires at least one applicable model profile")

    recommended = candidates[0]
    return ModelRoutingPlan(
        source_model_profile_serialization_version=model_registry.serialization_version,
        source_provider_selection_serialization_version=(
            provider_registry.serialization_version
        ),
        route_name=route_name,
        source_model_profile_ids=tuple(
            candidate.source_model_profile_id for candidate in candidates
        ),
        source_provider_candidate_ids=provider_registry.provider_candidate_ids,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_id=recommended.candidate_id,
        recommended_model_profile_id=recommended.source_model_profile_id,
        fallback_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status == "fallback"
        ),
        candidate_count=len(candidates),
        recommendation_confidence=_confidence(recommended.fit_score),
        advisory_actions=_plan_actions(route_name, recommended),
    )


def model_route_candidate_by_id(
    candidate_id: str,
    plan: ModelRoutingPlan | None = None,
) -> ModelRouteCandidate | None:
    """Return one advisory model route candidate without selecting it."""

    source_plan = plan or route_model_request()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def model_route_candidates_for_status(
    status: ModelRouteCandidateStatus,
    plan: ModelRoutingPlan | None = None,
) -> tuple[ModelRouteCandidate, ...]:
    """Return advisory model route candidates for a recommendation status."""

    source_plan = plan or route_model_request()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


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


def _candidates_for_route(
    route_name: RouteName,
    registry: ModelProfileRegistry,
) -> tuple[ModelRouteCandidate, ...]:
    profiles = tuple(
        profile
        for profile in registry.model_profiles
        if route_name in profile.route_applicability
    )
    ordered_profiles = tuple(
        sorted(
            profiles,
            key=lambda profile: (
                -_score_profile(route_name, profile),
                profile.model_profile_id,
            ),
        )
    )
    return tuple(
        _candidate_from_profile(
            route_name=route_name,
            profile=profile,
            rank=index + 1,
        )
        for index, profile in enumerate(ordered_profiles)
    )


def _candidate_from_profile(
    *,
    route_name: RouteName,
    profile: ModelProfile,
    rank: int,
) -> ModelRouteCandidate:
    score = _score_profile(route_name, profile)
    return ModelRouteCandidate(
        candidate_id=f"model_route::{profile.model_profile_id}",
        rank=rank,
        source_model_profile_id=profile.model_profile_id,
        model_profile_kind=profile.model_profile_kind,
        profile_name=profile.profile_name,
        route_name=route_name,
        route_applicability=profile.route_applicability,
        provider_candidate_ids=profile.provider_candidate_ids,
        capability_dimensions=profile.capability_dimensions,
        fit_score=score,
        fit_band=_fit_band(score),
        status="recommended" if rank == 1 else "fallback",
        evidence=(
            f"Profile is applicable to the {route_name.value} route.",
            f"Profile kind {profile.model_profile_kind} contributes route fit metadata.",
            "Provider candidates remain advisory metadata only.",
        ),
        advisory_actions=_candidate_actions(profile),
    )


def _score_profile(route_name: RouteName, profile: ModelProfile) -> int:
    route_bonus = _ROUTE_KIND_BONUS.get(route_name, {}).get(
        profile.model_profile_kind,
        0,
    )
    provider_coverage_bonus = min(len(profile.provider_candidate_ids) * 3, 12)
    capability_bonus = min(len(profile.capability_dimensions) * 2, 8)
    return 100 + route_bonus + provider_coverage_bonus + capability_bonus


def _fit_band(score: int) -> ModelRouteFitBand:
    if score >= 125:
        return "strong"
    if score >= 115:
        return "moderate"
    return "eligible"


def _confidence(score: int) -> ModelRouteConfidence:
    if score >= 125:
        return "high"
    if score >= 115:
        return "medium"
    return "low"


def _candidate_actions(profile: ModelProfile) -> tuple[str, ...]:
    return (
        f"Expose {profile.model_profile_id} as an advisory route candidate.",
        "Keep configured provider and model unchanged until explicit approval.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: ModelRouteCandidate,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.source_model_profile_id} as the advisory "
            f"model route for {route_name.value}."
        ),
        "Preserve current provider factory behavior and configured model.",
        "Defer local/cloud, hybrid, budget, cost, and quality routing to scoped tasks.",
    )
