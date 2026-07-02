"""V6.5 advisory cost trend metadata."""

from __future__ import annotations

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    SelfEvolutionConfidence,
    SelfEvolutionPlan,
    SelfEvolutionProposal,
    SelfEvolutionStatus,
    build_self_evolution_plan,
)

CostTrendsPlan = SelfEvolutionPlan
CostTrendsProposal = SelfEvolutionProposal
CostTrendsStatus = SelfEvolutionStatus
CostTrendsConfidence = SelfEvolutionConfidence

_ROLE = "cost_trends"
_ROADMAP_ITEM = "Cost Trends"
_PROPOSAL_KINDS = (
    "cost_pressure_signal",
    "cost_quality_tradeoff",
    "cost_memory_storage_pressure",
    "cost_research_execution_pressure",
    "cost_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "cost_dashboard",
    "execution_cost_forecasting",
    "quality_cost_optimizer",
    "budget_policies",
)


def build_cost_trends(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CostTrendsPlan:
    """Build cost trend proposals without enforcing budgets or policy."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def cost_trends_proposal_by_id(
    proposal_id: str,
    plan: CostTrendsPlan | None = None,
) -> CostTrendsProposal | None:
    """Return one cost trend proposal without applying it."""

    source_plan = plan or build_cost_trends()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def cost_trends_proposals_for_status(
    status: CostTrendsStatus,
    plan: CostTrendsPlan | None = None,
) -> tuple[CostTrendsProposal, ...]:
    """Return cost trend proposals by advisory status."""

    source_plan = plan or build_cost_trends()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def cost_trends_proposals_for_confidence(
    confidence: CostTrendsConfidence,
    plan: CostTrendsPlan | None = None,
) -> tuple[CostTrendsProposal, ...]:
    """Return cost trend proposals by confidence band."""

    source_plan = plan or build_cost_trends()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
