"""V5.1 exploration budget planner for advisory exploration allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.context_budget_planner import (
    ContextBudgetPlan,
)
from creative_coding_assistant.orchestration.creative_complexity_analyzer import (
    CreativeComplexityAnalysis,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    CreativeExplorationBudgetProfile,
    CreativeExplorationBudgetRegistry,
    creative_exploration_budget_registry,
)
from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
)

ExplorationBudgetTopic = Literal[
    "planning_execution_fit",
    "style_aesthetic_alignment",
    "curation_refinement_need",
    "final_synthesis_readiness",
]
ExplorationBudgetPosture = Literal["narrow", "moderate", "broad", "guarded"]
ExplorationBudgetPriority = Literal["critical", "high", "medium", "low"]
ExplorationBudgetPressure = Literal["low", "medium", "high"]

EXPLORATION_BUDGET_ALLOCATION_SERIALIZATION_VERSION = (
    "exploration_budget_allocation.v1"
)
EXPLORATION_BUDGET_PLAN_SERIALIZATION_VERSION = "exploration_budget_plan.v1"
EXPLORATION_BUDGET_PLANNER_AUTHORITY_BOUNDARY = (
    "Exploration budget planning derives advisory variant and refinement-pass "
    "allocations from existing creative exploration budget metadata, optional "
    "creative complexity, workflow cost, and context budget signals only; it "
    "does not enforce budgets, generate variants, trigger refinement, route by "
    "cost, route context, select providers or models, invoke agents, control "
    "workflow execution, mutate prompts, write storage, or modify generated "
    "output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "budget_enforcement",
    "variant_generation",
    "refinement_triggering",
    "cost_based_routing",
    "context_routing",
    "provider_or_model_routing",
    "agent_invocation",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_TOPIC_PRIORITIES: dict[ExplorationBudgetTopic, ExplorationBudgetPriority] = {
    "planning_execution_fit": "medium",
    "style_aesthetic_alignment": "high",
    "curation_refinement_need": "high",
    "final_synthesis_readiness": "critical",
}
_PRIORITY_ORDER: tuple[ExplorationBudgetPriority, ...] = (
    "critical",
    "high",
    "medium",
    "low",
)


@dataclass(frozen=True)
class _DraftExplorationAllocation:
    allocation_id: str
    topic_id: ExplorationBudgetTopic
    source_budget_profile_id: str
    budget_posture: ExplorationBudgetPosture
    priority: ExplorationBudgetPriority
    requested_variants: int
    requested_refinement_passes: int
    max_advisory_variants: int
    max_advisory_refinement_passes: int
    evidence: tuple[str, ...]
    advisory_actions: tuple[str, ...]


class ExplorationBudgetAllocation(BaseModel):
    """One advisory exploration allocation for a creative decision topic."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    allocation_id: str = Field(min_length=1, max_length=180)
    topic_id: ExplorationBudgetTopic
    source_budget_profile_id: str = Field(min_length=1, max_length=180)
    budget_posture: ExplorationBudgetPosture
    priority: ExplorationBudgetPriority
    requested_variants: int = Field(ge=0, le=20)
    planned_variants: int = Field(ge=0, le=20)
    max_advisory_variants: int = Field(ge=0, le=20)
    requested_refinement_passes: int = Field(ge=0, le=20)
    planned_refinement_passes: int = Field(ge=0, le=20)
    max_advisory_refinement_passes: int = Field(ge=0, le=20)
    pressure: ExplorationBudgetPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["exploration_budget_allocation.v1"] = (
        EXPLORATION_BUDGET_ALLOCATION_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _allocation_is_advisory(self) -> Self:
        if self.planned_variants > self.requested_variants:
            raise ValueError("planned_variants must not exceed requested_variants")
        if self.planned_variants > self.max_advisory_variants:
            raise ValueError("planned_variants must not exceed max_advisory_variants")
        if self.planned_refinement_passes > self.requested_refinement_passes:
            raise ValueError(
                "planned_refinement_passes must not exceed requested_refinement_passes"
            )
        if (
            self.planned_refinement_passes
            > self.max_advisory_refinement_passes
        ):
            raise ValueError(
                "planned_refinement_passes must not exceed "
                "max_advisory_refinement_passes"
            )
        return self


class ExplorationBudgetPlan(BaseModel):
    """Bounded V5.1 advisory plan for creative exploration breadth/depth."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["exploration_budget_planner"] = "exploration_budget_planner"
    serialization_version: Literal["exploration_budget_plan.v1"] = (
        EXPLORATION_BUDGET_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXPLORATION_BUDGET_PLANNER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_registry_serialization_version: str = Field(min_length=1, max_length=80)
    source_budget_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    allocations: tuple[ExplorationBudgetAllocation, ...] = Field(
        min_length=1,
        max_length=8,
    )
    allocation_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    topic_ids: tuple[ExplorationBudgetTopic, ...] = Field(min_length=1, max_length=8)
    max_total_variants: int = Field(ge=0, le=40)
    total_requested_variants: int = Field(ge=0, le=40)
    total_planned_variants: int = Field(ge=0, le=40)
    max_total_refinement_passes: int = Field(ge=0, le=40)
    total_requested_refinement_passes: int = Field(ge=0, le=40)
    total_planned_refinement_passes: int = Field(ge=0, le=40)
    creative_complexity_level: str | None = Field(default=None, max_length=40)
    workflow_cost_pressure: str | None = Field(default=None, max_length=40)
    context_budget_pressure: str | None = Field(default=None, max_length=40)
    context_over_budget_tokens: int = Field(ge=0, le=240_000)
    exploration_pressure: ExplorationBudgetPressure
    budget_limited: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    exploration_budget_planning_implemented: Literal[True] = True
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_allocations(self) -> Self:
        derived_allocation_ids = tuple(
            allocation.allocation_id for allocation in self.allocations
        )
        if len(set(derived_allocation_ids)) != len(derived_allocation_ids):
            raise ValueError("allocation_ids must be unique")
        if self.allocation_ids != derived_allocation_ids:
            raise ValueError("allocation_ids must match allocations")
        if self.topic_ids != tuple(allocation.topic_id for allocation in self.allocations):
            raise ValueError("topic_ids must match allocations")
        if self.source_budget_profile_ids != tuple(
            allocation.source_budget_profile_id for allocation in self.allocations
        ):
            raise ValueError("source_budget_profile_ids must match allocations")

        requested_variants = sum(
            allocation.requested_variants for allocation in self.allocations
        )
        planned_variants = sum(
            allocation.planned_variants for allocation in self.allocations
        )
        requested_refinement = sum(
            allocation.requested_refinement_passes for allocation in self.allocations
        )
        planned_refinement = sum(
            allocation.planned_refinement_passes for allocation in self.allocations
        )
        if self.total_requested_variants != requested_variants:
            raise ValueError("total_requested_variants must match allocations")
        if self.total_planned_variants != planned_variants:
            raise ValueError("total_planned_variants must match allocations")
        if self.total_requested_refinement_passes != requested_refinement:
            raise ValueError(
                "total_requested_refinement_passes must match allocations"
            )
        if self.total_planned_refinement_passes != planned_refinement:
            raise ValueError("total_planned_refinement_passes must match allocations")
        if self.total_planned_variants > self.max_total_variants:
            raise ValueError("total_planned_variants must fit max_total_variants")
        if self.total_planned_refinement_passes > self.max_total_refinement_passes:
            raise ValueError(
                "total_planned_refinement_passes must fit "
                "max_total_refinement_passes"
            )
        limited = (
            planned_variants < requested_variants
            or planned_refinement < requested_refinement
        )
        if self.budget_limited != limited:
            raise ValueError("budget_limited must match planned exploration limits")
        return self


def plan_exploration_budget(
    *,
    creative_complexity: CreativeComplexityAnalysis | None = None,
    workflow_cost: WorkflowCostAnalysis | None = None,
    context_budget: ContextBudgetPlan | None = None,
    exploration_registry: CreativeExplorationBudgetRegistry | None = None,
    max_total_variants: int = 6,
    max_total_refinement_passes: int = 5,
) -> ExplorationBudgetPlan:
    """Plan advisory exploration breadth/depth without executing exploration."""

    if max_total_variants < 0:
        raise ValueError("max_total_variants must be non-negative")
    if max_total_refinement_passes < 0:
        raise ValueError("max_total_refinement_passes must be non-negative")

    registry = exploration_registry or creative_exploration_budget_registry()
    creative_level = (
        creative_complexity.creative_complexity_level
        if creative_complexity is not None
        else None
    )
    workflow_pressure = (
        workflow_cost.estimated_cost_pressure if workflow_cost is not None else None
    )
    context_pressure = context_budget.budget_pressure if context_budget is not None else None
    context_over_budget = (
        context_budget.over_budget_tokens if context_budget is not None else 0
    )
    variant_capacity = _variant_capacity(
        max_total_variants=max_total_variants,
        creative_level=creative_level,
        workflow_pressure=workflow_pressure,
        context_pressure=context_pressure,
        context_over_budget=context_over_budget,
    )
    refinement_capacity = _refinement_capacity(
        max_total_refinement_passes=max_total_refinement_passes,
        workflow_pressure=workflow_pressure,
        context_pressure=context_pressure,
        context_over_budget=context_over_budget,
    )
    drafts = tuple(
        _draft_from_profile(
            profile,
            creative_level=creative_level,
            workflow_pressure=workflow_pressure,
            context_pressure=context_pressure,
            context_over_budget=context_over_budget,
        )
        for profile in registry.budget_profiles
    )
    allocations = _allocate_exploration(
        drafts,
        variant_capacity=variant_capacity,
        refinement_capacity=refinement_capacity,
    )
    total_requested_variants = sum(
        allocation.requested_variants for allocation in allocations
    )
    total_planned_variants = sum(
        allocation.planned_variants for allocation in allocations
    )
    total_requested_refinement = sum(
        allocation.requested_refinement_passes for allocation in allocations
    )
    total_planned_refinement = sum(
        allocation.planned_refinement_passes for allocation in allocations
    )
    budget_limited = (
        total_planned_variants < total_requested_variants
        or total_planned_refinement < total_requested_refinement
    )

    return ExplorationBudgetPlan(
        source_registry_serialization_version=registry.serialization_version,
        source_budget_profile_ids=tuple(
            allocation.source_budget_profile_id for allocation in allocations
        ),
        allocations=allocations,
        allocation_ids=tuple(allocation.allocation_id for allocation in allocations),
        topic_ids=tuple(allocation.topic_id for allocation in allocations),
        max_total_variants=variant_capacity,
        total_requested_variants=total_requested_variants,
        total_planned_variants=total_planned_variants,
        max_total_refinement_passes=refinement_capacity,
        total_requested_refinement_passes=total_requested_refinement,
        total_planned_refinement_passes=total_planned_refinement,
        creative_complexity_level=creative_level,
        workflow_cost_pressure=workflow_pressure,
        context_budget_pressure=context_pressure,
        context_over_budget_tokens=context_over_budget,
        exploration_pressure=_exploration_pressure(
            budget_limited=budget_limited,
            creative_level=creative_level,
            workflow_pressure=workflow_pressure,
            context_pressure=context_pressure,
            context_over_budget=context_over_budget,
        ),
        budget_limited=budget_limited,
        advisory_actions=_plan_actions(budget_limited),
    )


def exploration_budget_allocation_by_id(
    allocation_id: str,
    plan: ExplorationBudgetPlan | None = None,
) -> ExplorationBudgetAllocation | None:
    """Return one exploration allocation without enforcing budgets."""

    source_plan = plan or plan_exploration_budget()
    for allocation in source_plan.allocations:
        if allocation.allocation_id == allocation_id:
            return allocation
    return None


def exploration_budget_allocations_for_topic(
    topic_id: ExplorationBudgetTopic,
    plan: ExplorationBudgetPlan | None = None,
) -> tuple[ExplorationBudgetAllocation, ...]:
    """Return exploration allocations by topic without generating variants."""

    source_plan = plan or plan_exploration_budget()
    return tuple(
        allocation
        for allocation in source_plan.allocations
        if allocation.topic_id == topic_id
    )


def _draft_from_profile(
    profile: CreativeExplorationBudgetProfile,
    *,
    creative_level: str | None,
    workflow_pressure: str | None,
    context_pressure: str | None,
    context_over_budget: int,
) -> _DraftExplorationAllocation:
    topic_id = profile.topic_id
    return _DraftExplorationAllocation(
        allocation_id=f"exploration::{topic_id}",
        topic_id=topic_id,
        source_budget_profile_id=profile.budget_profile_id,
        budget_posture=profile.budget_posture,
        priority=_TOPIC_PRIORITIES[topic_id],
        requested_variants=profile.max_advisory_variants,
        requested_refinement_passes=profile.max_advisory_refinement_passes,
        max_advisory_variants=profile.max_advisory_variants,
        max_advisory_refinement_passes=profile.max_advisory_refinement_passes,
        evidence=(
            f"profile_posture:{profile.budget_posture}",
            f"profile_variants:{profile.max_advisory_variants}",
            f"profile_refinement_passes:{profile.max_advisory_refinement_passes}",
            f"creative_complexity:{creative_level or 'unavailable'}",
            f"workflow_cost:{workflow_pressure or 'unavailable'}",
            f"context_budget:{context_pressure or 'unavailable'}",
            f"context_over_budget_tokens:{context_over_budget}",
        ),
        advisory_actions=_allocation_actions(profile),
    )


def _allocate_exploration(
    drafts: tuple[_DraftExplorationAllocation, ...],
    *,
    variant_capacity: int,
    refinement_capacity: int,
) -> tuple[ExplorationBudgetAllocation, ...]:
    planned_variants = _round_robin_capacity(
        drafts,
        requested_attr="requested_variants",
        capacity=variant_capacity,
    )
    planned_refinement = _round_robin_capacity(
        drafts,
        requested_attr="requested_refinement_passes",
        capacity=refinement_capacity,
    )

    return tuple(
        ExplorationBudgetAllocation(
            allocation_id=draft.allocation_id,
            topic_id=draft.topic_id,
            source_budget_profile_id=draft.source_budget_profile_id,
            budget_posture=draft.budget_posture,
            priority=draft.priority,
            requested_variants=draft.requested_variants,
            planned_variants=planned_variants[draft.allocation_id],
            max_advisory_variants=draft.max_advisory_variants,
            requested_refinement_passes=draft.requested_refinement_passes,
            planned_refinement_passes=planned_refinement[draft.allocation_id],
            max_advisory_refinement_passes=draft.max_advisory_refinement_passes,
            pressure=_allocation_pressure(
                requested_variants=draft.requested_variants,
                planned_variants=planned_variants[draft.allocation_id],
                requested_refinement=draft.requested_refinement_passes,
                planned_refinement=planned_refinement[draft.allocation_id],
            ),
            evidence=draft.evidence,
            advisory_actions=draft.advisory_actions,
        )
        for draft in drafts
    )


def _round_robin_capacity(
    drafts: tuple[_DraftExplorationAllocation, ...],
    *,
    requested_attr: Literal["requested_variants", "requested_refinement_passes"],
    capacity: int,
) -> dict[str, int]:
    remaining = max(0, capacity)
    planned = {draft.allocation_id: 0 for draft in drafts}

    for priority in _PRIORITY_ORDER:
        priority_drafts = tuple(draft for draft in drafts if draft.priority == priority)
        while remaining > 0 and any(
            planned[draft.allocation_id] < getattr(draft, requested_attr)
            for draft in priority_drafts
        ):
            for draft in priority_drafts:
                requested = getattr(draft, requested_attr)
                if remaining <= 0:
                    break
                if planned[draft.allocation_id] >= requested:
                    continue
                planned[draft.allocation_id] += 1
                remaining -= 1
    return planned


def _variant_capacity(
    *,
    max_total_variants: int,
    creative_level: str | None,
    workflow_pressure: str | None,
    context_pressure: str | None,
    context_over_budget: int,
) -> int:
    capacity = max_total_variants
    if creative_level == "low" and workflow_pressure is None and context_pressure is None:
        capacity = min(capacity, 4)
    if workflow_pressure == "high":
        capacity -= 2
    elif workflow_pressure == "medium":
        capacity -= 1
    if context_pressure == "high" or context_over_budget > 0:
        capacity -= 2
    elif context_pressure == "medium":
        capacity -= 1
    return max(0, min(max_total_variants, capacity))


def _refinement_capacity(
    *,
    max_total_refinement_passes: int,
    workflow_pressure: str | None,
    context_pressure: str | None,
    context_over_budget: int,
) -> int:
    capacity = max_total_refinement_passes
    if workflow_pressure == "high":
        capacity -= 1
    if context_pressure == "high" or context_over_budget > 0:
        capacity -= 1
    elif context_pressure == "medium":
        capacity -= 1
    return max(0, min(max_total_refinement_passes, capacity))


def _allocation_pressure(
    *,
    requested_variants: int,
    planned_variants: int,
    requested_refinement: int,
    planned_refinement: int,
) -> ExplorationBudgetPressure:
    if planned_variants < requested_variants or planned_refinement < requested_refinement:
        return "high"
    if requested_variants + requested_refinement >= 3:
        return "medium"
    return "low"


def _exploration_pressure(
    *,
    budget_limited: bool,
    creative_level: str | None,
    workflow_pressure: str | None,
    context_pressure: str | None,
    context_over_budget: int,
) -> ExplorationBudgetPressure:
    if (
        creative_level == "high"
        or workflow_pressure == "high"
        or context_pressure == "high"
        or context_over_budget > 0
    ):
        return "high"
    if (
        budget_limited
        or creative_level == "medium"
        or workflow_pressure == "medium"
        or context_pressure == "medium"
    ):
        return "medium"
    return "low"


def _allocation_actions(
    profile: CreativeExplorationBudgetProfile,
) -> tuple[str, ...]:
    return (
        "Plan variant breadth without generating variants.",
        "Plan refinement depth without triggering refinement.",
        f"Retain advisory posture from {profile.budget_profile_id}.",
    )


def _plan_actions(budget_limited: bool) -> tuple[str, ...]:
    actions = [
        "Expose exploration budgets as advisory metadata only.",
        "Preserve workflow, provider, context, and output mutation boundaries.",
    ]
    if budget_limited:
        actions.append("Flag reduced exploration capacity for later strategy selection.")
    else:
        actions.append("Keep registry exploration capacities available to planners.")
    return tuple(actions)
