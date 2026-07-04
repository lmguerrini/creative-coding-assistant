"""V6.5 advisory strategy evolution metadata."""

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

StrategyEvolutionEnginePlan = SelfEvolutionPlan
StrategyEvolutionEngineProposal = SelfEvolutionProposal
StrategyEvolutionEngineStatus = SelfEvolutionStatus
StrategyEvolutionEngineConfidence = SelfEvolutionConfidence

_ROLE = "strategy_evolution_engine"
_ROADMAP_ITEM = "Strategy Evolution Engine"
_PROPOSAL_KINDS = (
    "strategy_signal_alignment",
    "strategy_selection_tradeoff",
    "strategy_cross_capability_dependency",
    "strategy_learning_feedback",
    "strategy_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "execution_strategy_selection",
    "creative_strategy",
    "strategy_learning",
    "runtime_recommendation_engine",
)


def build_strategy_evolution_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> StrategyEvolutionEnginePlan:
    """Build strategy evolution proposals without changing strategy selection."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def strategy_evolution_engine_proposal_by_id(
    proposal_id: str,
    plan: StrategyEvolutionEnginePlan | None = None,
) -> StrategyEvolutionEngineProposal | None:
    """Return one strategy evolution proposal without applying it."""

    source_plan = plan or build_strategy_evolution_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def strategy_evolution_engine_proposals_for_status(
    status: StrategyEvolutionEngineStatus,
    plan: StrategyEvolutionEnginePlan | None = None,
) -> tuple[StrategyEvolutionEngineProposal, ...]:
    """Return strategy evolution proposals by advisory status."""

    source_plan = plan or build_strategy_evolution_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def strategy_evolution_engine_proposals_for_confidence(
    confidence: StrategyEvolutionEngineConfidence,
    plan: StrategyEvolutionEnginePlan | None = None,
) -> tuple[StrategyEvolutionEngineProposal, ...]:
    """Return strategy evolution proposals by confidence band."""

    source_plan = plan or build_strategy_evolution_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
