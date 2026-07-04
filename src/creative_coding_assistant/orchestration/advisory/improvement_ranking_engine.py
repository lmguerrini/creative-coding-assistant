"""V6.5 advisory improvement ranking metadata."""

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

ImprovementRankingEnginePlan = SelfEvolutionPlan
ImprovementRankingEngineProposal = SelfEvolutionProposal
ImprovementRankingEngineStatus = SelfEvolutionStatus
ImprovementRankingEngineConfidence = SelfEvolutionConfidence

_ROLE = "improvement_ranking_engine"
_ROADMAP_ITEM = "Improvement Ranking Engine"
_PROPOSAL_KINDS = (
    "cross_capability_signal_ranking",
    "impact_confidence_rank_review",
    "cost_risk_rank_balancing",
    "dependency_rollback_tiebreak",
    "ranking_governance_report",
)
_DOWNSTREAM_SYSTEMS = (
    "self_evolution_common",
    "autonomous_optimization_suggestions",
    "quality_trends",
    "cost_trends",
    "benchmark_engine",
)


def build_improvement_ranking_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> ImprovementRankingEnginePlan:
    """Build improvement ranking proposals without applying rankings."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def improvement_ranking_engine_proposal_by_id(
    proposal_id: str,
    plan: ImprovementRankingEnginePlan | None = None,
) -> ImprovementRankingEngineProposal | None:
    """Return one improvement ranking proposal without applying it."""

    source_plan = plan or build_improvement_ranking_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def improvement_ranking_engine_proposals_for_status(
    status: ImprovementRankingEngineStatus,
    plan: ImprovementRankingEnginePlan | None = None,
) -> tuple[ImprovementRankingEngineProposal, ...]:
    """Return improvement ranking proposals by advisory status."""

    source_plan = plan or build_improvement_ranking_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def improvement_ranking_engine_proposals_for_confidence(
    confidence: ImprovementRankingEngineConfidence,
    plan: ImprovementRankingEnginePlan | None = None,
) -> tuple[ImprovementRankingEngineProposal, ...]:
    """Return improvement ranking proposals by confidence band."""

    source_plan = plan or build_improvement_ranking_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
