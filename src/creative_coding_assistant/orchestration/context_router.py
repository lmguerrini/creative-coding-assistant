"""V5.1 context router for advisory context lane planning."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.context_budget_planner import (
    ContextBudgetAllocation,
    ContextBudgetPlan,
    ContextBudgetPriority,
    ContextBudgetSourceKind,
    plan_context_budget,
)

ContextRouteLane = Literal[
    "primary_prompt",
    "conversation_memory",
    "project_memory",
    "retrieval_context",
    "planning_metadata",
    "response_budget",
]
ContextRouteDisposition = Literal["include", "defer", "reserve"]
ContextRoutePressure = Literal["low", "medium", "high"]

CONTEXT_ROUTE_DECISION_SERIALIZATION_VERSION = "context_route_decision.v1"
CONTEXT_ROUTING_PLAN_SERIALIZATION_VERSION = "context_routing_plan.v1"
CONTEXT_ROUTER_AUTHORITY_BOUNDARY = (
    "Context routing maps existing context budget allocations to advisory "
    "context lanes and deferred-token metadata only; it does not trim context, "
    "compress prompts, compress retrieval, summarize memory, select providers "
    "or models, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "context_trimming",
    "prompt_compression",
    "retrieval_compression",
    "memory_summarization",
    "provider_or_model_routing",
    "source_content_mutation",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_LANE_BY_SOURCE_KIND: dict[ContextBudgetSourceKind, ContextRouteLane] = {
    "user_request": "primary_prompt",
    "memory_recent_turns": "conversation_memory",
    "memory_summary": "conversation_memory",
    "project_memory": "project_memory",
    "retrieval_chunks": "retrieval_context",
    "creative_metadata": "planning_metadata",
    "workflow_overhead": "planning_metadata",
    "response_reserve": "response_budget",
}


class ContextRouteDecision(BaseModel):
    """One advisory routing decision for an existing context budget source."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    source_allocation_id: str = Field(min_length=1, max_length=180)
    source_kind: ContextBudgetSourceKind
    source_id: str = Field(min_length=1, max_length=160)
    lane: ContextRouteLane
    disposition: ContextRouteDisposition
    priority: ContextBudgetPriority
    requested_tokens: int = Field(ge=0, le=240_000)
    routed_tokens: int = Field(ge=0, le=240_000)
    deferred_tokens: int = Field(ge=0, le=240_000)
    pressure: ContextRoutePressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_routing_implemented: Literal[True] = True
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    source_content_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["context_route_decision.v1"] = (
        CONTEXT_ROUTE_DECISION_SERIALIZATION_VERSION
    )
    routing_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_preserves_source_budget(self) -> Self:
        if self.routed_tokens > self.requested_tokens:
            raise ValueError("routed_tokens must not exceed requested_tokens")
        if self.deferred_tokens != self.requested_tokens - self.routed_tokens:
            raise ValueError("deferred_tokens must match unrouted tokens")
        if self.source_kind == "response_reserve" and self.disposition != "reserve":
            raise ValueError("response reserve must use reserve disposition")
        if self.source_kind != "response_reserve" and self.disposition == "reserve":
            raise ValueError("reserve disposition is only for response reserve")
        if self.disposition == "defer" and self.routed_tokens > 0:
            raise ValueError("deferred context must not route tokens")
        return self


class ContextRoutingPlan(BaseModel):
    """Bounded V5.1 context routing plan over existing context allocations."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["context_router"] = "context_router"
    serialization_version: Literal["context_routing_plan.v1"] = (
        CONTEXT_ROUTING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONTEXT_ROUTER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_context_budget_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_allocation_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    decisions: tuple[ContextRouteDecision, ...] = Field(min_length=1, max_length=16)
    decision_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    routed_lanes: tuple[ContextRouteLane, ...] = Field(min_length=1, max_length=8)
    requested_context_tokens: int = Field(ge=0, le=240_000)
    routed_context_tokens: int = Field(ge=0, le=240_000)
    deferred_context_tokens: int = Field(ge=0, le=240_000)
    response_reserved_tokens: int = Field(ge=0, le=120_000)
    over_budget_tokens: int = Field(ge=0, le=240_000)
    context_budget_pressure: str = Field(min_length=1, max_length=40)
    routing_pressure: ContextRoutePressure
    routed_source_count: int = Field(ge=1, le=16)
    has_deferred_context: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_routing_implemented: Literal[True] = True
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    source_content_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    routing_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_decisions(self) -> Self:
        derived_decision_ids = tuple(decision.decision_id for decision in self.decisions)
        if len(set(derived_decision_ids)) != len(derived_decision_ids):
            raise ValueError("decision_ids must be unique")
        if self.decision_ids != derived_decision_ids:
            raise ValueError("decision_ids must match decisions")
        if self.source_allocation_ids != tuple(
            decision.source_allocation_id for decision in self.decisions
        ):
            raise ValueError("source_allocation_ids must match decisions")
        if self.routed_lanes != _unique_lanes(self.decisions):
            raise ValueError("routed_lanes must match decisions")
        if self.routed_source_count != len(self.decisions):
            raise ValueError("routed_source_count must match decisions")

        context_decisions = tuple(
            decision
            for decision in self.decisions
            if decision.source_kind != "response_reserve"
        )
        requested_context_tokens = sum(
            decision.requested_tokens for decision in context_decisions
        )
        routed_context_tokens = sum(
            decision.routed_tokens for decision in context_decisions
        )
        deferred_context_tokens = sum(
            decision.deferred_tokens for decision in context_decisions
        )
        response_reserved_tokens = sum(
            decision.routed_tokens
            for decision in self.decisions
            if decision.source_kind == "response_reserve"
        )
        if self.requested_context_tokens != requested_context_tokens:
            raise ValueError("requested_context_tokens must match decisions")
        if self.routed_context_tokens != routed_context_tokens:
            raise ValueError("routed_context_tokens must match decisions")
        if self.deferred_context_tokens != deferred_context_tokens:
            raise ValueError("deferred_context_tokens must match decisions")
        if self.response_reserved_tokens != response_reserved_tokens:
            raise ValueError("response_reserved_tokens must match decisions")
        if self.has_deferred_context != (deferred_context_tokens > 0):
            raise ValueError("has_deferred_context must match deferred tokens")
        return self


def route_context_sources(
    *,
    context_budget: ContextBudgetPlan | None = None,
) -> ContextRoutingPlan:
    """Route context allocation metadata to advisory lanes without mutation."""

    budget = context_budget or plan_context_budget()
    decisions = tuple(
        _decision_from_allocation(allocation, budget)
        for allocation in budget.allocations
    )
    context_decisions = tuple(
        decision
        for decision in decisions
        if decision.source_kind != "response_reserve"
    )
    deferred_context_tokens = sum(
        decision.deferred_tokens for decision in context_decisions
    )

    return ContextRoutingPlan(
        source_context_budget_serialization_version=budget.serialization_version,
        source_allocation_ids=tuple(
            decision.source_allocation_id for decision in decisions
        ),
        decisions=decisions,
        decision_ids=tuple(decision.decision_id for decision in decisions),
        routed_lanes=_unique_lanes(decisions),
        requested_context_tokens=budget.requested_context_tokens,
        routed_context_tokens=sum(
            decision.routed_tokens for decision in context_decisions
        ),
        deferred_context_tokens=deferred_context_tokens,
        response_reserved_tokens=sum(
            decision.routed_tokens
            for decision in decisions
            if decision.source_kind == "response_reserve"
        ),
        over_budget_tokens=budget.over_budget_tokens,
        context_budget_pressure=budget.budget_pressure,
        routing_pressure=_routing_pressure(
            budget_pressure=budget.budget_pressure,
            deferred_context_tokens=deferred_context_tokens,
        ),
        routed_source_count=len(decisions),
        has_deferred_context=deferred_context_tokens > 0,
        advisory_actions=_plan_actions(deferred_context_tokens),
    )


def context_route_decision_by_id(
    decision_id: str,
    plan: ContextRoutingPlan | None = None,
) -> ContextRouteDecision | None:
    """Return one route decision without changing context contents."""

    source_plan = plan or route_context_sources()
    for decision in source_plan.decisions:
        if decision.decision_id == decision_id:
            return decision
    return None


def context_route_decisions_for_lane(
    lane: ContextRouteLane,
    plan: ContextRoutingPlan | None = None,
) -> tuple[ContextRouteDecision, ...]:
    """Return route decisions by lane without provider or model routing."""

    source_plan = plan or route_context_sources()
    return tuple(decision for decision in source_plan.decisions if decision.lane == lane)


def _decision_from_allocation(
    allocation: ContextBudgetAllocation,
    budget: ContextBudgetPlan,
) -> ContextRouteDecision:
    lane = _LANE_BY_SOURCE_KIND[allocation.source_kind]
    disposition = _disposition(allocation)
    deferred_tokens = allocation.requested_tokens - allocation.allocated_tokens
    return ContextRouteDecision(
        decision_id=f"route::{allocation.source_kind}",
        source_allocation_id=allocation.allocation_id,
        source_kind=allocation.source_kind,
        source_id=allocation.source_id,
        lane=lane,
        disposition=disposition,
        priority=allocation.priority,
        requested_tokens=allocation.requested_tokens,
        routed_tokens=allocation.allocated_tokens,
        deferred_tokens=deferred_tokens,
        pressure=_decision_pressure(
            allocation_pressure=allocation.pressure,
            deferred_tokens=deferred_tokens,
        ),
        evidence=(
            f"allocation:{allocation.allocation_id}",
            f"allocation_pressure:{allocation.pressure}",
            f"context_budget_pressure:{budget.budget_pressure}",
            f"over_budget_tokens:{budget.over_budget_tokens}",
            f"lane:{lane}",
        ),
        advisory_actions=_decision_actions(allocation.source_kind, lane, disposition),
    )


def _disposition(allocation: ContextBudgetAllocation) -> ContextRouteDisposition:
    if allocation.source_kind == "response_reserve":
        return "reserve"
    if allocation.allocated_tokens == 0 and allocation.requested_tokens > 0:
        return "defer"
    return "include"


def _decision_pressure(
    *,
    allocation_pressure: str,
    deferred_tokens: int,
) -> ContextRoutePressure:
    if deferred_tokens > 0 or allocation_pressure == "high":
        return "high"
    if allocation_pressure == "medium":
        return "medium"
    return "low"


def _routing_pressure(
    *,
    budget_pressure: str,
    deferred_context_tokens: int,
) -> ContextRoutePressure:
    if budget_pressure == "high" or deferred_context_tokens > 0:
        return "high"
    if budget_pressure == "medium":
        return "medium"
    return "low"


def _unique_lanes(
    decisions: tuple[ContextRouteDecision, ...],
) -> tuple[ContextRouteLane, ...]:
    lanes: list[ContextRouteLane] = []
    for decision in decisions:
        if decision.lane not in lanes:
            lanes.append(decision.lane)
    return tuple(lanes)


def _decision_actions(
    source_kind: ContextBudgetSourceKind,
    lane: ContextRouteLane,
    disposition: ContextRouteDisposition,
) -> tuple[str, ...]:
    actions = [
        f"Route {source_kind} metadata to {lane} lane without editing content.",
    ]
    if disposition == "defer":
        actions.append("Record source as deferred context metadata only.")
    elif disposition == "reserve":
        actions.append("Reserve response budget without changing generated output.")
    else:
        actions.append("Keep source available to downstream prompt assembly.")
    return tuple(actions)


def _plan_actions(deferred_context_tokens: int) -> tuple[str, ...]:
    actions = [
        "Expose context lanes as routing metadata only.",
        "Preserve source contents, provider routing, and output boundaries.",
    ]
    if deferred_context_tokens:
        actions.append("Flag deferred context for later scoped optimization tasks.")
    else:
        actions.append("Route all requested context sources within the current budget.")
    return tuple(actions)
