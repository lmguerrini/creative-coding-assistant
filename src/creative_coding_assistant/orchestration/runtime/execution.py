"""Bounded, observable workflow execution selection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import (
    AssistantRequest,
    WorkflowExecutionMode,
)
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)


class WorkflowExecutionPlan(BaseModel):
    """Truthful route record shared by graph execution and observability."""

    model_config = ConfigDict(frozen=True)

    requested_mode: WorkflowExecutionMode
    resolved_mode: WorkflowExecutionMode
    rationale: str = Field(min_length=1)
    agent_roles: tuple[str, ...] = Field(min_length=1)
    researcher_required: bool
    researcher_reason: str = Field(min_length=1)
    max_refinement_loops: int = Field(ge=0, le=1)

    @property
    def is_single_agent(self) -> bool:
        return self.resolved_mode is WorkflowExecutionMode.SINGLE_AGENT

    @property
    def is_multi_agent(self) -> bool:
        return self.resolved_mode is WorkflowExecutionMode.MULTI_AGENT


def resolve_workflow_execution_plan(
    request: AssistantRequest,
    decision: RouteDecision,
) -> WorkflowExecutionPlan:
    """Resolve one bounded route without hidden agents or duplicate calls."""

    requested = request.workflow_mode
    if requested is WorkflowExecutionMode.SINGLE_AGENT:
        return _single_agent_plan(requested, "Single Agent was selected for this run.")
    if requested is WorkflowExecutionMode.MULTI_AGENT:
        return _multi_agent_plan(
            requested,
            "Multi Agent was selected for this run.",
            researcher_reason=_researcher_reason(decision),
        )

    if _auto_prefers_single_agent(request, decision):
        return _single_agent_plan(
            requested,
            "Auto selected Single Agent for a lightweight explanation or debug request.",
        )
    return _multi_agent_plan(
        requested,
        (
            "Auto selected Multi Agent because the request benefits from planning, "
            "generation, critique, and bounded review."
        ),
        researcher_reason=_researcher_reason(decision),
    )


def _auto_prefers_single_agent(
    request: AssistantRequest,
    decision: RouteDecision,
) -> bool:
    return (
        decision.route in {RouteName.EXPLAIN, RouteName.DEBUG}
        and RouteCapability.OFFICIAL_DOCS not in decision.capabilities
        and not request.attachments
        and len(decision.domains) <= 1
    )


def _single_agent_plan(
    requested: WorkflowExecutionMode,
    rationale: str,
) -> WorkflowExecutionPlan:
    return WorkflowExecutionPlan(
        requested_mode=requested,
        resolved_mode=WorkflowExecutionMode.SINGLE_AGENT,
        rationale=rationale,
        agent_roles=("generator",),
        researcher_required=False,
        researcher_reason="Single Agent does not activate a separate researcher or reviewer route.",
        max_refinement_loops=0,
    )


def _multi_agent_plan(
    requested: WorkflowExecutionMode,
    rationale: str,
    *,
    researcher_reason: str,
) -> WorkflowExecutionPlan:
    return WorkflowExecutionPlan(
        requested_mode=requested,
        resolved_mode=WorkflowExecutionMode.MULTI_AGENT,
        rationale=rationale,
        agent_roles=("planner", "researcher", "generator", "critic", "reviewer"),
        researcher_required=True,
        researcher_reason=researcher_reason,
        max_refinement_loops=1,
    )


def _researcher_reason(decision: RouteDecision) -> str:
    if decision.domains:
        return (
            "Planner requested bounded retrieval so the multi-agent route can ground "
            "the selected creative domain before generation."
        )
    return (
        "Planner requested bounded retrieval so the multi-agent route can check "
        "available reference context before generation."
    )
