"""V5.2 advisory local-vs-cloud routing comparison."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    CloudModelRegistry,
    LocalModelRegistry,
    ModelProfile,
    ModelProfileRegistry,
    cloud_model_registry,
    local_model_registry,
    model_profile_registry,
)
from creative_coding_assistant.orchestration.model_router import (
    ModelRouteCandidateStatus,
    ModelRoutingPlan,
    route_model_request,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

LocalCloudRoutingLane = Literal[
    "local_candidate",
    "cloud_candidate",
    "balanced_candidate",
]
LocalCloudRoutingPosture = Literal[
    "local_preferred",
    "cloud_preferred",
    "balanced",
]
LocalCloudRoutingConfidence = Literal["low", "medium", "high"]

LOCAL_CLOUD_ROUTE_DECISION_SERIALIZATION_VERSION = "local_cloud_route_decision.v1"
LOCAL_CLOUD_ROUTING_PLAN_SERIALIZATION_VERSION = "local_cloud_routing_plan.v1"
LOCAL_CLOUD_ROUTER_AUTHORITY_BOUNDARY = (
    "The V5.2 Local vs Cloud Routing surface compares existing passive local "
    "and cloud model surface metadata for advisory routing posture only; it "
    "does not discover installed local models, select providers or models, "
    "switch configured provider/model settings, call local or cloud providers, "
    "perform hybrid routing, optimize cost or quality, enforce budgets, "
    "control workflow execution, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "local_model_discovery",
    "configured_provider_switching",
    "configured_model_switching",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "local_provider_execution",
    "cloud_provider_execution",
    "hybrid_routing",
    "cost_optimization",
    "quality_optimization",
    "budget_enforcement",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_LOCAL_FIRST_ROUTES = (RouteName.DEBUG, RouteName.PREVIEW)
_CLOUD_FIRST_ROUTES = (RouteName.DESIGN, RouteName.REVIEW)


class LocalCloudRouteDecision(BaseModel):
    """One advisory local/cloud comparison for a model route candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    source_model_route_candidate_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    local_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    local_score: int = Field(ge=0, le=160)
    cloud_score: int = Field(ge=0, le=160)
    comparison_delta: int = Field(ge=0, le=160)
    routing_lane: LocalCloudRoutingLane
    routing_posture: LocalCloudRoutingPosture
    decision_status: ModelRouteCandidateStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    local_cloud_routing_implemented: Literal[True] = True
    local_model_discovery_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    hybrid_routing_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    quality_optimization_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["local_cloud_route_decision.v1"] = (
        LOCAL_CLOUD_ROUTE_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_scores(self) -> Self:
        if not self.local_surface_ids and not self.cloud_surface_ids:
            raise ValueError("local or cloud surface metadata is required")
        if self.comparison_delta != abs(self.local_score - self.cloud_score):
            raise ValueError("comparison_delta must match local/cloud scores")
        if self.routing_lane != _routing_lane(
            local_score=self.local_score,
            cloud_score=self.cloud_score,
        ):
            raise ValueError("routing_lane must match local/cloud scores")
        if self.routing_posture != _routing_posture(self.routing_lane):
            raise ValueError("routing_posture must match routing_lane")
        return self


class LocalCloudRoutingPlan(BaseModel):
    """Bounded V5.2 advisory local/cloud routing plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["local_cloud_router"] = "local_cloud_router"
    serialization_version: Literal["local_cloud_routing_plan.v1"] = (
        LOCAL_CLOUD_ROUTING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LOCAL_CLOUD_ROUTER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_model_routing_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_local_model_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_cloud_model_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_model_route_candidate_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    source_local_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_cloud_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    decisions: tuple[LocalCloudRouteDecision, ...] = Field(
        min_length=1,
        max_length=12,
    )
    decision_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_decision_id: str = Field(min_length=1, max_length=180)
    recommended_routing_lane: LocalCloudRoutingLane
    recommended_routing_posture: LocalCloudRoutingPosture
    fallback_decision_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    decision_count: int = Field(ge=1, le=12)
    local_candidate_count: int = Field(ge=0, le=12)
    cloud_candidate_count: int = Field(ge=0, le=12)
    balanced_candidate_count: int = Field(ge=0, le=12)
    routing_confidence: LocalCloudRoutingConfidence
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    local_cloud_routing_implemented: Literal[True] = True
    local_model_discovery_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    hybrid_routing_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    quality_optimization_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_decisions(self) -> Self:
        derived_decision_ids = tuple(decision.decision_id for decision in self.decisions)
        if len(set(derived_decision_ids)) != len(derived_decision_ids):
            raise ValueError("decision_ids must be unique")
        if self.decision_ids != derived_decision_ids:
            raise ValueError("decision_ids must match decisions")
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count must match decisions")
        if self.source_model_route_candidate_ids != tuple(
            decision.source_model_route_candidate_id for decision in self.decisions
        ):
            raise ValueError("source_model_route_candidate_ids must match decisions")

        recommended = tuple(
            decision
            for decision in self.decisions
            if decision.decision_status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended decision is required")
        recommended_decision = recommended[0]
        if self.recommended_decision_id != recommended_decision.decision_id:
            raise ValueError("recommended_decision_id must match decision")
        if self.recommended_routing_lane != recommended_decision.routing_lane:
            raise ValueError("recommended_routing_lane must match decision")
        if self.recommended_routing_posture != recommended_decision.routing_posture:
            raise ValueError("recommended_routing_posture must match decision")
        if self.fallback_decision_ids != tuple(
            decision.decision_id
            for decision in self.decisions
            if decision.decision_status == "fallback"
        ):
            raise ValueError("fallback_decision_ids must match decisions")
        if self.local_candidate_count != _count_lane(
            self.decisions,
            "local_candidate",
        ):
            raise ValueError("local_candidate_count must match decisions")
        if self.cloud_candidate_count != _count_lane(
            self.decisions,
            "cloud_candidate",
        ):
            raise ValueError("cloud_candidate_count must match decisions")
        if self.balanced_candidate_count != _count_lane(
            self.decisions,
            "balanced_candidate",
        ):
            raise ValueError("balanced_candidate_count must match decisions")
        if self.routing_confidence != _confidence(recommended_decision.comparison_delta):
            raise ValueError("routing_confidence must match decision delta")

        known_local_ids = set(self.source_local_surface_ids)
        known_cloud_ids = set(self.source_cloud_surface_ids)
        for decision in self.decisions:
            if decision.route_name != self.route_name:
                raise ValueError("decision route_name must match plan route_name")
            if not set(decision.local_surface_ids).issubset(known_local_ids):
                raise ValueError("local_surface_ids must be known local surfaces")
            if not set(decision.cloud_surface_ids).issubset(known_cloud_ids):
                raise ValueError("cloud_surface_ids must be known cloud surfaces")
        return self


def route_local_vs_cloud(
    *,
    model_routing: ModelRoutingPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
    model_profiles: ModelProfileRegistry | None = None,
    local_models: LocalModelRegistry | None = None,
    cloud_models: CloudModelRegistry | None = None,
) -> LocalCloudRoutingPlan:
    """Compare local/cloud metadata for model-route candidates without applying it."""

    route_name = _resolve_route(
        model_routing=model_routing,
        route_decision=route_decision,
        route=route,
    )
    model_plan = model_routing or route_model_request(
        route_decision=route_decision,
        route=route_name,
        model_profiles=model_profiles,
    )
    model_registry = model_profiles or model_profile_registry()
    local_registry = local_models or local_model_registry()
    cloud_registry = cloud_models or cloud_model_registry()
    profiles_by_id = {
        profile.model_profile_id: profile
        for profile in model_registry.model_profiles
    }
    decisions = tuple(
        _decision_from_candidate(
            route_name=route_name,
            profile=_profile_for_candidate(
                candidate.source_model_profile_id,
                profiles_by_id,
            ),
            candidate_id=candidate.candidate_id,
            status=candidate.status,
            provider_candidate_ids=candidate.provider_candidate_ids,
        )
        for candidate in model_plan.candidates
    )
    recommended = _recommended_decision(decisions)

    return LocalCloudRoutingPlan(
        source_model_routing_serialization_version=model_plan.serialization_version,
        source_local_model_serialization_version=local_registry.serialization_version,
        source_cloud_model_serialization_version=cloud_registry.serialization_version,
        route_name=route_name,
        source_model_route_candidate_ids=tuple(
            decision.source_model_route_candidate_id for decision in decisions
        ),
        source_local_surface_ids=local_registry.surface_ids,
        source_cloud_surface_ids=cloud_registry.surface_ids,
        decisions=decisions,
        decision_ids=tuple(decision.decision_id for decision in decisions),
        recommended_decision_id=recommended.decision_id,
        recommended_routing_lane=recommended.routing_lane,
        recommended_routing_posture=recommended.routing_posture,
        fallback_decision_ids=tuple(
            decision.decision_id
            for decision in decisions
            if decision.decision_status == "fallback"
        ),
        decision_count=len(decisions),
        local_candidate_count=_count_lane(decisions, "local_candidate"),
        cloud_candidate_count=_count_lane(decisions, "cloud_candidate"),
        balanced_candidate_count=_count_lane(decisions, "balanced_candidate"),
        routing_confidence=_confidence(recommended.comparison_delta),
        advisory_actions=_plan_actions(route_name, recommended),
    )


def local_cloud_route_decision_by_id(
    decision_id: str,
    plan: LocalCloudRoutingPlan | None = None,
) -> LocalCloudRouteDecision | None:
    """Return one advisory local/cloud route decision without applying it."""

    source_plan = plan or route_local_vs_cloud()
    for decision in source_plan.decisions:
        if decision.decision_id == decision_id:
            return decision
    return None


def local_cloud_route_decisions_for_lane(
    lane: LocalCloudRoutingLane,
    plan: LocalCloudRoutingPlan | None = None,
) -> tuple[LocalCloudRouteDecision, ...]:
    """Return advisory local/cloud decisions for one lane."""

    source_plan = plan or route_local_vs_cloud()
    return tuple(
        decision for decision in source_plan.decisions if decision.routing_lane == lane
    )


def _resolve_route(
    *,
    model_routing: ModelRoutingPlan | None,
    route_decision: RouteDecision | None,
    route: RouteName | str | None,
) -> RouteName:
    explicit_route = None if route is None else (
        route if isinstance(route, RouteName) else RouteName(str(route))
    )
    if model_routing is not None:
        if explicit_route is not None and explicit_route != model_routing.route_name:
            raise ValueError("route must match model_routing")
        if (
            route_decision is not None
            and route_decision.route != model_routing.route_name
        ):
            raise ValueError("route_decision must match model_routing")
        return model_routing.route_name
    if route_decision is not None:
        if explicit_route is not None and explicit_route != route_decision.route:
            raise ValueError("route must match route_decision")
        return route_decision.route
    return explicit_route or RouteName.GENERATE


def _decision_from_candidate(
    *,
    route_name: RouteName,
    profile: ModelProfile,
    candidate_id: str,
    status: ModelRouteCandidateStatus,
    provider_candidate_ids: tuple[str, ...],
) -> LocalCloudRouteDecision:
    local_score = _local_score(route_name, profile)
    cloud_score = _cloud_score(route_name, profile)
    lane = _routing_lane(local_score=local_score, cloud_score=cloud_score)
    return LocalCloudRouteDecision(
        decision_id=f"local_cloud_route::{profile.model_profile_id}",
        source_model_route_candidate_id=candidate_id,
        source_model_profile_id=profile.model_profile_id,
        route_name=route_name,
        local_surface_ids=profile.source_local_surface_ids,
        cloud_surface_ids=profile.source_cloud_surface_ids,
        provider_candidate_ids=provider_candidate_ids,
        local_score=local_score,
        cloud_score=cloud_score,
        comparison_delta=abs(local_score - cloud_score),
        routing_lane=lane,
        routing_posture=_routing_posture(lane),
        decision_status=status,
        evidence=(
            f"Compared local and cloud metadata for {profile.model_profile_id}.",
            f"Local surfaces: {len(profile.source_local_surface_ids)}.",
            f"Cloud surfaces: {len(profile.source_cloud_surface_ids)}.",
        ),
        advisory_actions=_decision_actions(lane),
    )


def _profile_for_candidate(
    model_profile_id: str,
    profiles_by_id: dict[str, ModelProfile],
) -> ModelProfile:
    profile = profiles_by_id.get(model_profile_id)
    if profile is None:
        raise ValueError("model route candidate must reference a known model profile")
    return profile


def _local_score(route_name: RouteName, profile: ModelProfile) -> int:
    score = len(profile.source_local_surface_ids) * 20
    if route_name in _LOCAL_FIRST_ROUTES:
        score += 30
    elif route_name not in _CLOUD_FIRST_ROUTES:
        score += 15
    return score


def _cloud_score(route_name: RouteName, profile: ModelProfile) -> int:
    score = len(profile.source_cloud_surface_ids) * 20
    if route_name in _CLOUD_FIRST_ROUTES:
        score += 30
    elif route_name not in _LOCAL_FIRST_ROUTES:
        score += 15
    return score


def _routing_lane(
    *,
    local_score: int,
    cloud_score: int,
) -> LocalCloudRoutingLane:
    if local_score > cloud_score:
        return "local_candidate"
    if cloud_score > local_score:
        return "cloud_candidate"
    return "balanced_candidate"


def _routing_posture(lane: LocalCloudRoutingLane) -> LocalCloudRoutingPosture:
    if lane == "local_candidate":
        return "local_preferred"
    if lane == "cloud_candidate":
        return "cloud_preferred"
    return "balanced"


def _confidence(delta: int) -> LocalCloudRoutingConfidence:
    if delta >= 30:
        return "high"
    if delta >= 15:
        return "medium"
    return "low"


def _recommended_decision(
    decisions: tuple[LocalCloudRouteDecision, ...],
) -> LocalCloudRouteDecision:
    for decision in decisions:
        if decision.decision_status == "recommended":
            return decision
    raise ValueError("local/cloud routing requires a recommended model route decision")


def _count_lane(
    decisions: tuple[LocalCloudRouteDecision, ...],
    lane: LocalCloudRoutingLane,
) -> int:
    return sum(1 for decision in decisions if decision.routing_lane == lane)


def _decision_actions(lane: LocalCloudRoutingLane) -> tuple[str, ...]:
    if lane == "local_candidate":
        return (
            "Expose local surfaces as advisory candidates for manual review.",
            "Keep provider/model configuration unchanged.",
        )
    if lane == "cloud_candidate":
        return (
            "Expose cloud surfaces as advisory candidates for manual review.",
            "Keep external provider calls disabled until explicit execution.",
        )
    return (
        "Expose local and cloud surfaces as balanced advisory candidates.",
        "Defer any concrete local/cloud choice to explicit routing policy tasks.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: LocalCloudRouteDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.routing_posture} posture for the "
            f"{route_name.value} route as advisory metadata."
        ),
        "Preserve current provider factory behavior and configured model.",
        "Defer hybrid, budget, cost, quality, and execution routing to scoped tasks.",
    )
