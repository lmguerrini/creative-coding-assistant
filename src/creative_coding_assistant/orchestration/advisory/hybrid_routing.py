"""V5.2 advisory hybrid routing over local/cloud routing metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.local_cloud_routing import (
    LocalCloudRouteDecision,
    LocalCloudRoutingLane,
    LocalCloudRoutingPlan,
    route_local_vs_cloud,
)
from creative_coding_assistant.orchestration.model_router import ModelRoutingPlan
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

HybridRoutingMode = Literal[
    "local_primary",
    "cloud_primary",
    "balanced_dual",
]
HybridRoutingStatus = Literal["recommended", "fallback"]
HybridRoutingConfidence = Literal["low", "medium", "high"]

HYBRID_ROUTE_DECISION_SERIALIZATION_VERSION = "hybrid_route_decision.v1"
HYBRID_ROUTING_PLAN_SERIALIZATION_VERSION = "hybrid_routing_plan.v1"
HYBRID_ROUTER_AUTHORITY_BOUNDARY = (
    "The V5.2 Hybrid Routing surface converts advisory local/cloud routing "
    "posture into inspectable hybrid route recommendations only; it does not "
    "execute hybrid workflows, select or switch providers or models, call "
    "local or cloud providers, merge provider outputs, optimize cost or "
    "quality, enforce budgets, control workflow execution, trigger retries, "
    "mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "hybrid_workflow_execution",
    "hybrid_routing_application",
    "configured_provider_switching",
    "configured_model_switching",
    "automatic_provider_selection",
    "automatic_model_selection",
    "provider_or_model_routing",
    "local_provider_execution",
    "cloud_provider_execution",
    "provider_output_merging",
    "cost_optimization",
    "quality_optimization",
    "budget_enforcement",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class HybridRouteDecision(BaseModel):
    """One advisory hybrid routing recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    source_local_cloud_decision_id: str = Field(min_length=1, max_length=180)
    source_model_route_candidate_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    route_name: RouteName
    hybrid_mode: HybridRoutingMode
    source_routing_lane: LocalCloudRoutingLane
    local_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    cloud_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    local_score: int = Field(ge=0, le=160)
    cloud_score: int = Field(ge=0, le=160)
    hybrid_score: int = Field(ge=0, le=320)
    comparison_delta: int = Field(ge=0, le=160)
    status: HybridRoutingStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    hybrid_routing_implemented: Literal[True] = True
    hybrid_routing_application_implemented: Literal[False] = False
    hybrid_workflow_execution_implemented: Literal[False] = False
    provider_output_merging_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    cost_optimization_implemented: Literal[False] = False
    quality_optimization_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_route_decision.v1"] = (
        HYBRID_ROUTE_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_source_scores(self) -> Self:
        if not self.local_surface_ids and not self.cloud_surface_ids:
            raise ValueError("hybrid routing requires local or cloud surface metadata")
        if self.hybrid_score != self.local_score + self.cloud_score:
            raise ValueError("hybrid_score must combine local and cloud scores")
        if self.comparison_delta != abs(self.local_score - self.cloud_score):
            raise ValueError("comparison_delta must match local/cloud scores")
        if self.hybrid_mode != _hybrid_mode(self.source_routing_lane):
            raise ValueError("hybrid_mode must match source routing lane")
        return self


class HybridRoutingPlan(BaseModel):
    """Bounded V5.2 advisory hybrid routing plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_router"] = "hybrid_router"
    serialization_version: Literal["hybrid_routing_plan.v1"] = (
        HYBRID_ROUTING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_ROUTER_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_local_cloud_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    route_name: RouteName
    source_local_cloud_decision_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    decisions: tuple[HybridRouteDecision, ...] = Field(min_length=1, max_length=12)
    decision_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    recommended_decision_id: str = Field(min_length=1, max_length=180)
    recommended_hybrid_mode: HybridRoutingMode
    fallback_decision_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    decision_count: int = Field(ge=1, le=12)
    local_primary_count: int = Field(ge=0, le=12)
    cloud_primary_count: int = Field(ge=0, le=12)
    balanced_dual_count: int = Field(ge=0, le=12)
    routing_confidence: HybridRoutingConfidence
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    hybrid_routing_implemented: Literal[True] = True
    hybrid_routing_application_implemented: Literal[False] = False
    hybrid_workflow_execution_implemented: Literal[False] = False
    provider_output_merging_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
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
        derived_decision_ids = tuple(
            decision.decision_id for decision in self.decisions
        )
        if len(set(derived_decision_ids)) != len(derived_decision_ids):
            raise ValueError("decision_ids must be unique")
        if self.decision_ids != derived_decision_ids:
            raise ValueError("decision_ids must match decisions")
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count must match decisions")
        if self.source_local_cloud_decision_ids != tuple(
            decision.source_local_cloud_decision_id for decision in self.decisions
        ):
            raise ValueError("source_local_cloud_decision_ids must match decisions")

        recommended = tuple(
            decision for decision in self.decisions if decision.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended decision is required")
        recommended_decision = recommended[0]
        if self.recommended_decision_id != recommended_decision.decision_id:
            raise ValueError("recommended_decision_id must match decision")
        if self.recommended_hybrid_mode != recommended_decision.hybrid_mode:
            raise ValueError("recommended_hybrid_mode must match decision")
        if self.fallback_decision_ids != tuple(
            decision.decision_id
            for decision in self.decisions
            if decision.status == "fallback"
        ):
            raise ValueError("fallback_decision_ids must match decisions")
        if self.local_primary_count != _count_mode(self.decisions, "local_primary"):
            raise ValueError("local_primary_count must match decisions")
        if self.cloud_primary_count != _count_mode(self.decisions, "cloud_primary"):
            raise ValueError("cloud_primary_count must match decisions")
        if self.balanced_dual_count != _count_mode(self.decisions, "balanced_dual"):
            raise ValueError("balanced_dual_count must match decisions")
        if self.routing_confidence != _confidence(recommended_decision):
            raise ValueError("routing_confidence must match recommended decision")
        for decision in self.decisions:
            if decision.route_name != self.route_name:
                raise ValueError("decision route_name must match plan route_name")
        return self


def route_hybrid_model_request(
    *,
    local_cloud_routing: LocalCloudRoutingPlan | None = None,
    model_routing: ModelRoutingPlan | None = None,
    route_decision: RouteDecision | None = None,
    route: RouteName | str | None = None,
) -> HybridRoutingPlan:
    """Return advisory hybrid routing metadata without applying route behavior."""

    local_cloud_plan = local_cloud_routing or route_local_vs_cloud(
        model_routing=model_routing,
        route_decision=route_decision,
        route=route,
    )
    decisions = tuple(
        _decision_from_local_cloud_decision(decision)
        for decision in local_cloud_plan.decisions
    )
    recommended = _recommended_decision(decisions)

    return HybridRoutingPlan(
        source_local_cloud_serialization_version=local_cloud_plan.serialization_version,
        route_name=local_cloud_plan.route_name,
        source_local_cloud_decision_ids=tuple(
            decision.source_local_cloud_decision_id for decision in decisions
        ),
        decisions=decisions,
        decision_ids=tuple(decision.decision_id for decision in decisions),
        recommended_decision_id=recommended.decision_id,
        recommended_hybrid_mode=recommended.hybrid_mode,
        fallback_decision_ids=tuple(
            decision.decision_id
            for decision in decisions
            if decision.status == "fallback"
        ),
        decision_count=len(decisions),
        local_primary_count=_count_mode(decisions, "local_primary"),
        cloud_primary_count=_count_mode(decisions, "cloud_primary"),
        balanced_dual_count=_count_mode(decisions, "balanced_dual"),
        routing_confidence=_confidence(recommended),
        advisory_actions=_plan_actions(local_cloud_plan.route_name, recommended),
    )


def hybrid_route_decision_by_id(
    decision_id: str,
    plan: HybridRoutingPlan | None = None,
) -> HybridRouteDecision | None:
    """Return one advisory hybrid route decision without applying it."""

    source_plan = plan or route_hybrid_model_request()
    for decision in source_plan.decisions:
        if decision.decision_id == decision_id:
            return decision
    return None


def hybrid_route_decisions_for_mode(
    mode: HybridRoutingMode,
    plan: HybridRoutingPlan | None = None,
) -> tuple[HybridRouteDecision, ...]:
    """Return advisory hybrid route decisions for one mode."""

    source_plan = plan or route_hybrid_model_request()
    return tuple(
        decision for decision in source_plan.decisions if decision.hybrid_mode == mode
    )


def _decision_from_local_cloud_decision(
    source: LocalCloudRouteDecision,
) -> HybridRouteDecision:
    mode = _hybrid_mode(source.routing_lane)
    return HybridRouteDecision(
        decision_id=f"hybrid_route::{source.source_model_profile_id}",
        source_local_cloud_decision_id=source.decision_id,
        source_model_route_candidate_id=source.source_model_route_candidate_id,
        source_model_profile_id=source.source_model_profile_id,
        route_name=source.route_name,
        hybrid_mode=mode,
        source_routing_lane=source.routing_lane,
        local_surface_ids=source.local_surface_ids,
        cloud_surface_ids=source.cloud_surface_ids,
        local_score=source.local_score,
        cloud_score=source.cloud_score,
        hybrid_score=source.local_score + source.cloud_score,
        comparison_delta=source.comparison_delta,
        status=source.decision_status,
        evidence=(
            f"Derived from {source.decision_id}.",
            f"Local/cloud posture is {source.routing_posture}.",
            "Hybrid route remains advisory metadata only.",
        ),
        advisory_actions=_decision_actions(mode),
    )


def _hybrid_mode(lane: LocalCloudRoutingLane) -> HybridRoutingMode:
    if lane == "local_candidate":
        return "local_primary"
    if lane == "cloud_candidate":
        return "cloud_primary"
    return "balanced_dual"


def _confidence(decision: HybridRouteDecision) -> HybridRoutingConfidence:
    if decision.hybrid_mode == "balanced_dual":
        return "medium"
    if decision.comparison_delta >= 30:
        return "high"
    if decision.comparison_delta >= 15:
        return "medium"
    return "low"


def _recommended_decision(
    decisions: tuple[HybridRouteDecision, ...],
) -> HybridRouteDecision:
    for decision in decisions:
        if decision.status == "recommended":
            return decision
    raise ValueError("hybrid routing requires a recommended local/cloud decision")


def _count_mode(
    decisions: tuple[HybridRouteDecision, ...],
    mode: HybridRoutingMode,
) -> int:
    return sum(1 for decision in decisions if decision.hybrid_mode == mode)


def _decision_actions(mode: HybridRoutingMode) -> tuple[str, ...]:
    if mode == "local_primary":
        return (
            "Expose local-primary hybrid posture for manual review.",
            "Keep cloud fallback metadata advisory only.",
        )
    if mode == "cloud_primary":
        return (
            "Expose cloud-primary hybrid posture for manual review.",
            "Keep local fallback metadata advisory only.",
        )
    return (
        "Expose balanced local/cloud hybrid posture for manual review.",
        "Do not merge provider outputs or execute multiple providers.",
    )


def _plan_actions(
    route_name: RouteName,
    recommended: HybridRouteDecision,
) -> tuple[str, ...]:
    return (
        (
            f"Present {recommended.hybrid_mode} as advisory hybrid posture "
            f"for the {route_name.value} route."
        ),
        "Preserve current provider factory behavior and configured model.",
        "Defer cost, quality, budget, policy, and execution behavior to scoped tasks.",
    )
