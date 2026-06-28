"""V5.1 context reuse planning over existing context budget metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.context_budget_planner import (
    ContextBudgetAllocation,
    ContextBudgetPlan,
    ContextBudgetSourceKind,
    plan_context_budget,
)

ContextReuseStatus = Literal["reusable", "not_reusable"]
ContextReuseConfidence = Literal["low", "medium", "high"]

CONTEXT_REUSE_CANDIDATE_SERIALIZATION_VERSION = "context_reuse_candidate.v1"
CONTEXT_REUSE_PLAN_SERIALIZATION_VERSION = "context_reuse_plan.v1"
CONTEXT_REUSE_AUTHORITY_BOUNDARY = (
    "Context reuse planning compares existing context budget metadata to identify "
    "advisory reusable context sources only; it does not materialize shared "
    "context, mutate source context, write cache or storage, route providers or "
    "models, control workflow execution, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "shared_context_materialization",
    "source_context_mutation",
    "cache_write",
    "persistent_storage_write",
    "provider_or_model_routing",
    "workflow_control",
    "generated_output_modification",
)


class ContextReuseCandidate(BaseModel):
    """One advisory context reuse candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_kind: ContextBudgetSourceKind
    previous_allocation_id: str = Field(min_length=1, max_length=180)
    current_allocation_id: str = Field(min_length=1, max_length=180)
    previous_source_id: str = Field(min_length=1, max_length=180)
    current_source_id: str = Field(min_length=1, max_length=180)
    reusable_tokens: int = Field(ge=0, le=240_000)
    requested_tokens: int = Field(ge=0, le=240_000)
    status: ContextReuseStatus
    confidence: ContextReuseConfidence
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_reuse_implemented: Literal[True] = True
    shared_context_materialization_implemented: Literal[False] = False
    source_context_mutation_implemented: Literal[False] = False
    cache_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["context_reuse_candidate.v1"] = (
        CONTEXT_REUSE_CANDIDATE_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_status(self) -> Self:
        if self.reusable_tokens > self.requested_tokens:
            raise ValueError("reusable_tokens must not exceed requested_tokens")
        expected_status = "reusable" if self.reusable_tokens > 0 else "not_reusable"
        if self.status != expected_status:
            raise ValueError("status must match reusable_tokens")
        if self.status == "reusable" and self.previous_source_id != self.current_source_id:
            raise ValueError("reusable candidates require matching source ids")
        return self


class ContextReusePlan(BaseModel):
    """Bounded V5.1 advisory context reuse plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["context_reuse_planner"] = "context_reuse_planner"
    serialization_version: Literal["context_reuse_plan.v1"] = (
        CONTEXT_REUSE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONTEXT_REUSE_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    previous_context_budget_serialization_version: str = Field(min_length=1, max_length=80)
    current_context_budget_serialization_version: str = Field(min_length=1, max_length=80)
    candidates: tuple[ContextReuseCandidate, ...] = Field(min_length=1, max_length=16)
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    reusable_source_kinds: tuple[ContextBudgetSourceKind, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    total_requested_tokens: int = Field(ge=0, le=240_000)
    total_reusable_tokens: int = Field(ge=0, le=240_000)
    reuse_ratio: float = Field(ge=0, le=1)
    reuse_confidence: ContextReuseConfidence
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    context_reuse_implemented: Literal[True] = True
    shared_context_materialization_implemented: Literal[False] = False
    source_context_mutation_implemented: Literal[False] = False
    cache_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        requested = sum(candidate.requested_tokens for candidate in self.candidates)
        reusable = sum(candidate.reusable_tokens for candidate in self.candidates)
        if self.total_requested_tokens != requested:
            raise ValueError("total_requested_tokens must match candidates")
        if self.total_reusable_tokens != reusable:
            raise ValueError("total_reusable_tokens must match candidates")
        expected_ratio = reusable / requested if requested else 0.0
        if abs(self.reuse_ratio - expected_ratio) > 0.0001:
            raise ValueError("reuse_ratio must match reusable tokens")
        if self.reusable_source_kinds != tuple(
            candidate.source_kind
            for candidate in self.candidates
            if candidate.status == "reusable"
        ):
            raise ValueError("reusable_source_kinds must match reusable candidates")
        return self


def plan_context_reuse(
    *,
    previous_context_budget: ContextBudgetPlan | None = None,
    current_context_budget: ContextBudgetPlan | None = None,
) -> ContextReusePlan:
    """Plan advisory reuse between two context budgets without materialization."""

    previous = previous_context_budget or plan_context_budget()
    current = current_context_budget or plan_context_budget()
    previous_by_kind = {
        allocation.source_kind: allocation for allocation in previous.allocations
    }
    candidates = tuple(
        _candidate(previous_by_kind.get(allocation.source_kind), allocation)
        for allocation in current.allocations
        if allocation.source_kind != "response_reserve"
    )
    requested = sum(candidate.requested_tokens for candidate in candidates)
    reusable = sum(candidate.reusable_tokens for candidate in candidates)
    ratio = reusable / requested if requested else 0.0

    return ContextReusePlan(
        previous_context_budget_serialization_version=previous.serialization_version,
        current_context_budget_serialization_version=current.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        reusable_source_kinds=tuple(
            candidate.source_kind for candidate in candidates if candidate.status == "reusable"
        ),
        total_requested_tokens=requested,
        total_reusable_tokens=reusable,
        reuse_ratio=ratio,
        reuse_confidence=_confidence(ratio),
        advisory_actions=_plan_actions(reusable),
    )


def context_reuse_candidate_by_id(
    candidate_id: str,
    plan: ContextReusePlan | None = None,
) -> ContextReuseCandidate | None:
    """Return one reuse candidate without materializing context."""

    source_plan = plan or plan_context_reuse()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def context_reuse_candidates_for_status(
    status: ContextReuseStatus,
    plan: ContextReusePlan | None = None,
) -> tuple[ContextReuseCandidate, ...]:
    """Return reuse candidates by status without cache writes."""

    source_plan = plan or plan_context_reuse()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _candidate(
    previous: ContextBudgetAllocation | None,
    current: ContextBudgetAllocation,
) -> ContextReuseCandidate:
    reusable_tokens = 0
    previous_allocation_id = "missing"
    previous_source_id = "missing"
    if previous is not None:
        previous_allocation_id = previous.allocation_id
        previous_source_id = previous.source_id
        if previous.source_id == current.source_id:
            reusable_tokens = min(previous.allocated_tokens, current.allocated_tokens)

    return ContextReuseCandidate(
        candidate_id=f"context_reuse::{current.source_kind}",
        source_kind=current.source_kind,
        previous_allocation_id=previous_allocation_id,
        current_allocation_id=current.allocation_id,
        previous_source_id=previous_source_id,
        current_source_id=current.source_id,
        reusable_tokens=reusable_tokens,
        requested_tokens=current.requested_tokens,
        status="reusable" if reusable_tokens else "not_reusable",
        confidence=_candidate_confidence(reusable_tokens, current.requested_tokens),
        evidence=(
            f"current_allocation:{current.allocation_id}",
            f"previous_allocation:{previous_allocation_id}",
            f"source_match:{previous_source_id == current.source_id}",
            f"reusable_tokens:{reusable_tokens}",
        ),
        advisory_actions=_candidate_actions(reusable_tokens),
    )


def _candidate_confidence(
    reusable_tokens: int,
    requested_tokens: int,
) -> ContextReuseConfidence:
    if requested_tokens <= 0 or reusable_tokens <= 0:
        return "low"
    ratio = reusable_tokens / requested_tokens
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def _confidence(ratio: float) -> ContextReuseConfidence:
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def _candidate_actions(reusable_tokens: int) -> tuple[str, ...]:
    if reusable_tokens:
        return ("Expose reusable context source as advisory metadata only.",)
    return ("Request fresh context source; no reuse metadata match was found.",)


def _plan_actions(reusable_tokens: int) -> tuple[str, ...]:
    actions = [
        "Expose context reuse opportunities as metadata only.",
        "Preserve source context, cache, routing, workflow, and output boundaries.",
    ]
    if reusable_tokens:
        actions.append("Use reuse plan only when explicitly selected downstream.")
    return tuple(actions)
