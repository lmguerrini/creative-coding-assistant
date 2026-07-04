"""V6.5 advisory rollback strategy metadata."""

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

RollbackStrategyGeneratorPlan = SelfEvolutionPlan
RollbackStrategyGeneratorProposal = SelfEvolutionProposal
RollbackStrategyGeneratorStatus = SelfEvolutionStatus
RollbackStrategyGeneratorConfidence = SelfEvolutionConfidence

_ROLE = "rollback_strategy_generator"
_ROADMAP_ITEM = "Rollback Strategy Generator"
_PROPOSAL_KINDS = (
    "rollback_signal_alignment",
    "rollback_feasibility_review",
    "workflow_replay_dependency",
    "knowledge_rollback_guardrail",
    "rollback_handoff_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "knowledge_rollback",
    "workflow_replay_engine",
    "execution_replay_engine",
    "learning_replay_engine",
    "runtime_timeline",
)


def build_rollback_strategy_generator(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> RollbackStrategyGeneratorPlan:
    """Build rollback strategy proposals without executing rollback."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def rollback_strategy_generator_proposal_by_id(
    proposal_id: str,
    plan: RollbackStrategyGeneratorPlan | None = None,
) -> RollbackStrategyGeneratorProposal | None:
    """Return one rollback strategy proposal without applying it."""

    source_plan = plan or build_rollback_strategy_generator()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def rollback_strategy_generator_proposals_for_status(
    status: RollbackStrategyGeneratorStatus,
    plan: RollbackStrategyGeneratorPlan | None = None,
) -> tuple[RollbackStrategyGeneratorProposal, ...]:
    """Return rollback strategy proposals by advisory status."""

    source_plan = plan or build_rollback_strategy_generator()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def rollback_strategy_generator_proposals_for_confidence(
    confidence: RollbackStrategyGeneratorConfidence,
    plan: RollbackStrategyGeneratorPlan | None = None,
) -> tuple[RollbackStrategyGeneratorProposal, ...]:
    """Return rollback strategy proposals by confidence band."""

    source_plan = plan or build_rollback_strategy_generator()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
