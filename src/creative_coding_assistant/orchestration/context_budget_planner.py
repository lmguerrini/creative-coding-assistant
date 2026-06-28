"""V5.1 context budget planner for advisory context allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.context import AssembledContextResponse
from creative_coding_assistant.orchestration.creative_complexity_analyzer import (
    CreativeComplexityAnalysis,
)
from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
)

ContextBudgetSourceKind = Literal[
    "user_request",
    "memory_recent_turns",
    "memory_summary",
    "project_memory",
    "retrieval_chunks",
    "creative_metadata",
    "workflow_overhead",
    "response_reserve",
]
ContextBudgetPriority = Literal["critical", "high", "medium", "low", "reserve"]
ContextBudgetPressure = Literal["low", "medium", "high"]

CONTEXT_BUDGET_ALLOCATION_SERIALIZATION_VERSION = "context_budget_allocation.v1"
CONTEXT_BUDGET_PLAN_SERIALIZATION_VERSION = "context_budget_plan.v1"
CONTEXT_BUDGET_PLANNER_AUTHORITY_BOUNDARY = (
    "Context budget planning derives advisory token allocations from existing "
    "assembled context, creative complexity, and workflow cost metadata only; "
    "it does not trim context, compress prompts, compress retrieval, summarize "
    "memory, route context, select providers or models, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "context_trimming",
    "prompt_compression",
    "retrieval_compression",
    "memory_summarization",
    "context_routing",
    "provider_or_model_routing",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


@dataclass(frozen=True)
class _DraftAllocation:
    allocation_id: str
    source_kind: ContextBudgetSourceKind
    source_id: str
    priority: ContextBudgetPriority
    requested_tokens: int
    max_tokens: int
    min_tokens: int
    evidence: tuple[str, ...]
    advisory_actions: tuple[str, ...]


class ContextBudgetAllocation(BaseModel):
    """One advisory context token allocation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    allocation_id: str = Field(min_length=1, max_length=160)
    source_kind: ContextBudgetSourceKind
    source_id: str = Field(min_length=1, max_length=120)
    priority: ContextBudgetPriority
    requested_tokens: int = Field(ge=0, le=240_000)
    allocated_tokens: int = Field(ge=0, le=240_000)
    max_tokens: int = Field(ge=0, le=240_000)
    pressure: ContextBudgetPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["context_budget_allocation.v1"] = (
        CONTEXT_BUDGET_ALLOCATION_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _allocation_is_advisory(self) -> Self:
        if self.allocated_tokens > self.requested_tokens:
            raise ValueError("allocated_tokens must not exceed requested_tokens")
        if self.allocated_tokens > self.max_tokens:
            raise ValueError("allocated_tokens must not exceed max_tokens")
        return self


class ContextBudgetPlan(BaseModel):
    """Bounded V5.1 advisory plan for context token allocation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["context_budget_planner"] = "context_budget_planner"
    serialization_version: Literal["context_budget_plan.v1"] = (
        CONTEXT_BUDGET_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONTEXT_BUDGET_PLANNER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    total_budget_tokens: int = Field(ge=1, le=240_000)
    response_reserve_tokens: int = Field(ge=0, le=120_000)
    available_context_tokens: int = Field(ge=0, le=240_000)
    requested_context_tokens: int = Field(ge=0, le=240_000)
    allocated_context_tokens: int = Field(ge=0, le=240_000)
    over_budget_tokens: int = Field(ge=0, le=240_000)
    budget_pressure: ContextBudgetPressure
    allocations: tuple[ContextBudgetAllocation, ...] = Field(
        min_length=1,
        max_length=16,
    )
    allocation_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    memory_recent_turn_count: int = Field(ge=0, le=10_000)
    project_memory_count: int = Field(ge=0, le=10_000)
    retrieval_chunk_count: int = Field(ge=0, le=10_000)
    creative_complexity_level: str | None = Field(default=None, max_length=40)
    workflow_cost_pressure: str | None = Field(default=None, max_length=40)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_budget_planning_implemented: Literal[True] = True
    context_trimming_implemented: Literal[False] = False
    prompt_compression_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
        context_allocations = tuple(
            allocation
            for allocation in self.allocations
            if allocation.source_kind != "response_reserve"
        )
        requested = sum(allocation.requested_tokens for allocation in context_allocations)
        allocated = sum(allocation.allocated_tokens for allocation in context_allocations)
        if self.available_context_tokens != (
            self.total_budget_tokens - self.response_reserve_tokens
        ):
            raise ValueError("available_context_tokens must match total minus reserve")
        if self.requested_context_tokens != requested:
            raise ValueError("requested_context_tokens must match allocations")
        if self.allocated_context_tokens != allocated:
            raise ValueError("allocated_context_tokens must match allocations")
        if self.allocated_context_tokens > self.available_context_tokens:
            raise ValueError("allocated_context_tokens must fit available context")
        if self.over_budget_tokens != max(0, requested - self.available_context_tokens):
            raise ValueError("over_budget_tokens must match requested overflow")
        return self


def plan_context_budget(
    *,
    assembled_context: AssembledContextResponse | None = None,
    creative_complexity: CreativeComplexityAnalysis | None = None,
    workflow_cost: WorkflowCostAnalysis | None = None,
    user_query: str | None = None,
    total_budget_tokens: int = 16_000,
    response_reserve_tokens: int = 3_000,
) -> ContextBudgetPlan:
    """Plan advisory context allocation without trimming or mutation."""

    if response_reserve_tokens > total_budget_tokens:
        raise ValueError("response_reserve_tokens must not exceed total budget")

    drafts = _draft_allocations(
        assembled_context=assembled_context,
        creative_complexity=creative_complexity,
        workflow_cost=workflow_cost,
        user_query=user_query,
        response_reserve_tokens=response_reserve_tokens,
    )
    available_context_tokens = total_budget_tokens - response_reserve_tokens
    allocations = _allocate(drafts, available_context_tokens)
    context_allocations = tuple(
        allocation
        for allocation in allocations
        if allocation.source_kind != "response_reserve"
    )
    requested_context_tokens = sum(
        allocation.requested_tokens for allocation in context_allocations
    )
    allocated_context_tokens = sum(
        allocation.allocated_tokens for allocation in context_allocations
    )
    over_budget_tokens = max(0, requested_context_tokens - available_context_tokens)

    return ContextBudgetPlan(
        total_budget_tokens=total_budget_tokens,
        response_reserve_tokens=response_reserve_tokens,
        available_context_tokens=available_context_tokens,
        requested_context_tokens=requested_context_tokens,
        allocated_context_tokens=allocated_context_tokens,
        over_budget_tokens=over_budget_tokens,
        budget_pressure=_budget_pressure(
            requested_context_tokens,
            available_context_tokens,
        ),
        allocations=allocations,
        allocation_ids=tuple(allocation.allocation_id for allocation in allocations),
        memory_recent_turn_count=_recent_turn_count(assembled_context),
        project_memory_count=_project_memory_count(assembled_context),
        retrieval_chunk_count=_retrieval_chunk_count(assembled_context),
        creative_complexity_level=(
            creative_complexity.creative_complexity_level
            if creative_complexity is not None
            else None
        ),
        workflow_cost_pressure=(
            workflow_cost.estimated_cost_pressure if workflow_cost is not None else None
        ),
        advisory_actions=_plan_actions(over_budget_tokens),
    )


def context_budget_allocation_by_id(
    allocation_id: str,
    plan: ContextBudgetPlan | None = None,
) -> ContextBudgetAllocation | None:
    """Return one budget allocation without changing context."""

    source_plan = plan or plan_context_budget()
    for allocation in source_plan.allocations:
        if allocation.allocation_id == allocation_id:
            return allocation
    return None


def context_budget_allocations_for_kind(
    source_kind: ContextBudgetSourceKind,
    plan: ContextBudgetPlan | None = None,
) -> tuple[ContextBudgetAllocation, ...]:
    """Return budget allocations by source kind without routing context."""

    source_plan = plan or plan_context_budget()
    return tuple(
        allocation
        for allocation in source_plan.allocations
        if allocation.source_kind == source_kind
    )


def _draft_allocations(
    *,
    assembled_context: AssembledContextResponse | None,
    creative_complexity: CreativeComplexityAnalysis | None,
    workflow_cost: WorkflowCostAnalysis | None,
    user_query: str | None,
    response_reserve_tokens: int,
) -> tuple[_DraftAllocation, ...]:
    drafts = [
        _draft(
            "context::user_request",
            "user_request",
            "assistant_request",
            "critical",
            _user_query_tokens(assembled_context, user_query),
            1_200,
            120,
            ("request_query",),
            ("Preserve user request allocation before optional context.",),
        ),
        _draft(
            "context::memory_recent_turns",
            "memory_recent_turns",
            "memory_context.recent_turns",
            "high",
            _recent_turn_tokens(assembled_context),
            1_800,
            0,
            (f"recent_turns:{_recent_turn_count(assembled_context)}",),
            ("Plan recent-turn budget without summarizing memory.",),
        ),
        _draft(
            "context::memory_summary",
            "memory_summary",
            "memory_context.running_summary",
            "high",
            _summary_tokens(assembled_context),
            1_000,
            0,
            (f"has_summary:{_has_summary(assembled_context)}",),
            ("Plan summary budget without rewriting memory.",),
        ),
        _draft(
            "context::project_memory",
            "project_memory",
            "memory_context.project_memories",
            "medium",
            _project_memory_tokens(assembled_context),
            1_200,
            0,
            (f"project_memories:{_project_memory_count(assembled_context)}",),
            ("Plan project-memory budget without storage writes.",),
        ),
        _draft(
            "context::retrieval_chunks",
            "retrieval_chunks",
            "retrieval_context.chunks",
            "high",
            _retrieval_tokens(assembled_context),
            3_000,
            0,
            (f"retrieval_chunks:{_retrieval_chunk_count(assembled_context)}",),
            ("Plan retrieval budget without compressing chunks.",),
        ),
        _draft(
            "context::creative_metadata",
            "creative_metadata",
            "creative_complexity_analysis",
            "medium",
            _creative_metadata_tokens(creative_complexity),
            2_000,
            0,
            (
                (
                    "creative_complexity:"
                    f"{creative_complexity.creative_complexity_level}"
                    if creative_complexity is not None
                    else "creative_complexity:unavailable"
                ),
            ),
            ("Plan creative metadata budget without changing outputs.",),
        ),
        _draft(
            "context::workflow_overhead",
            "workflow_overhead",
            "workflow_cost_analysis",
            "low",
            _workflow_overhead_tokens(workflow_cost),
            800,
            0,
            (
                (
                    f"workflow_cost:{workflow_cost.estimated_cost_pressure}"
                    if workflow_cost is not None
                    else "workflow_cost:unavailable"
                ),
            ),
            ("Plan workflow overhead without changing execution.",),
        ),
        _draft(
            "context::response_reserve",
            "response_reserve",
            "assistant_response",
            "reserve",
            response_reserve_tokens,
            response_reserve_tokens,
            response_reserve_tokens,
            ("response_reserve",),
            ("Reserve output budget without modifying generated output.",),
        ),
    ]
    return tuple(drafts)


def _draft(
    allocation_id: str,
    source_kind: ContextBudgetSourceKind,
    source_id: str,
    priority: ContextBudgetPriority,
    requested_tokens: int,
    max_tokens: int,
    min_tokens: int,
    evidence: tuple[str, ...],
    advisory_actions: tuple[str, ...],
) -> _DraftAllocation:
    return _DraftAllocation(
        allocation_id=allocation_id,
        source_kind=source_kind,
        source_id=source_id,
        priority=priority,
        requested_tokens=max(0, requested_tokens),
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        evidence=evidence,
        advisory_actions=advisory_actions,
    )


def _allocate(
    drafts: tuple[_DraftAllocation, ...],
    available_context_tokens: int,
) -> tuple[ContextBudgetAllocation, ...]:
    allocation_values = {
        draft.allocation_id: min(draft.requested_tokens, draft.max_tokens)
        for draft in drafts
    }
    context_ids = tuple(
        draft.allocation_id
        for draft in drafts
        if draft.source_kind != "response_reserve"
    )
    excess = max(
        0,
        sum(allocation_values[allocation_id] for allocation_id in context_ids)
        - available_context_tokens,
    )
    if excess:
        for priority in ("low", "medium", "high"):
            for draft in drafts:
                if excess <= 0:
                    break
                if draft.priority != priority:
                    continue
                current = allocation_values[draft.allocation_id]
                floor = min(draft.min_tokens, current)
                reduction = min(excess, current - floor)
                allocation_values[draft.allocation_id] = current - reduction
                excess -= reduction
            if excess <= 0:
                break

    return tuple(
        ContextBudgetAllocation(
            allocation_id=draft.allocation_id,
            source_kind=draft.source_kind,
            source_id=draft.source_id,
            priority=draft.priority,
            requested_tokens=draft.requested_tokens,
            allocated_tokens=allocation_values[draft.allocation_id],
            max_tokens=draft.max_tokens,
            pressure=_allocation_pressure(
                draft.requested_tokens,
                allocation_values[draft.allocation_id],
            ),
            evidence=draft.evidence,
            advisory_actions=draft.advisory_actions,
        )
        for draft in drafts
    )


def _estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _user_query_tokens(
    assembled_context: AssembledContextResponse | None,
    user_query: str | None,
) -> int:
    if user_query is not None:
        return _estimate_text_tokens(user_query)
    retrieval_context = (
        assembled_context.retrieval_context if assembled_context is not None else None
    )
    if retrieval_context is not None:
        return _estimate_text_tokens(retrieval_context.request.query)
    return 120


def _recent_turn_tokens(assembled_context: AssembledContextResponse | None) -> int:
    memory_context = (
        assembled_context.memory_context if assembled_context is not None else None
    )
    if memory_context is None:
        return 0
    return sum(_estimate_text_tokens(turn.content) for turn in memory_context.recent_turns)


def _summary_tokens(assembled_context: AssembledContextResponse | None) -> int:
    memory_context = (
        assembled_context.memory_context if assembled_context is not None else None
    )
    if memory_context is None or memory_context.running_summary is None:
        return 0
    return _estimate_text_tokens(memory_context.running_summary.content)


def _project_memory_tokens(assembled_context: AssembledContextResponse | None) -> int:
    memory_context = (
        assembled_context.memory_context if assembled_context is not None else None
    )
    if memory_context is None:
        return 0
    return sum(
        _estimate_text_tokens(memory.content)
        for memory in memory_context.project_memories
    )


def _retrieval_tokens(assembled_context: AssembledContextResponse | None) -> int:
    retrieval_context = (
        assembled_context.retrieval_context if assembled_context is not None else None
    )
    if retrieval_context is None:
        return 0
    return sum(_estimate_text_tokens(chunk.excerpt) for chunk in retrieval_context.chunks)


def _creative_metadata_tokens(
    creative_complexity: CreativeComplexityAnalysis | None,
) -> int:
    if creative_complexity is None:
        return 200
    return min(2_000, max(240, creative_complexity.creative_complexity_score * 12))


def _workflow_overhead_tokens(workflow_cost: WorkflowCostAnalysis | None) -> int:
    if workflow_cost is None:
        return 240
    pressure_bonus = {"low": 0, "medium": 120, "high": 240}[
        workflow_cost.estimated_cost_pressure
    ]
    return min(800, 280 + pressure_bonus)


def _recent_turn_count(assembled_context: AssembledContextResponse | None) -> int:
    if assembled_context is None:
        return 0
    return assembled_context.summary.recent_turn_count


def _has_summary(assembled_context: AssembledContextResponse | None) -> bool:
    if assembled_context is None:
        return False
    return assembled_context.summary.has_running_summary


def _project_memory_count(assembled_context: AssembledContextResponse | None) -> int:
    if assembled_context is None:
        return 0
    return assembled_context.summary.project_memory_count


def _retrieval_chunk_count(assembled_context: AssembledContextResponse | None) -> int:
    if assembled_context is None:
        return 0
    return assembled_context.summary.retrieval_chunk_count


def _budget_pressure(requested: int, available: int) -> ContextBudgetPressure:
    if available <= 0 or requested > available:
        return "high"
    ratio = requested / available
    if ratio >= 0.82:
        return "medium"
    return "low"


def _allocation_pressure(
    requested: int,
    allocated: int,
) -> ContextBudgetPressure:
    if requested == 0:
        return "low"
    if allocated < requested:
        return "high"
    if requested >= 800:
        return "medium"
    return "low"


def _plan_actions(over_budget_tokens: int) -> tuple[str, ...]:
    actions = [
        "Expose context budget allocation as advisory metadata only.",
        "Preserve context source contents and provider routing boundaries.",
    ]
    if over_budget_tokens:
        actions.append("Flag overflow for later compression or routing tasks.")
    return tuple(actions)
