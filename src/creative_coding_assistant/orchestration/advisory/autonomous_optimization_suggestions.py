"""V6.5 advisory autonomous optimization suggestion metadata."""

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

AutonomousOptimizationSuggestionsPlan = SelfEvolutionPlan
AutonomousOptimizationSuggestionsProposal = SelfEvolutionProposal
AutonomousOptimizationSuggestionsStatus = SelfEvolutionStatus
AutonomousOptimizationSuggestionsConfidence = SelfEvolutionConfidence

_ROLE = "autonomous_optimization_suggestions"
_ROADMAP_ITEM = "Autonomous Optimization Suggestions"
_PROPOSAL_KINDS = (
    "optimization_candidate_prioritization",
    "quality_cost_optimization_signal",
    "workflow_efficiency_suggestion",
    "research_memory_optimization_signal",
    "optimization_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "runtime_recommendation_engine",
    "quality_cost_optimizer",
    "workflow_review",
    "hitl_budget_gate",
)


def build_autonomous_optimization_suggestions(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> AutonomousOptimizationSuggestionsPlan:
    """Build optimization suggestions without executing optimizations."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def autonomous_optimization_suggestions_proposal_by_id(
    proposal_id: str,
    plan: AutonomousOptimizationSuggestionsPlan | None = None,
) -> AutonomousOptimizationSuggestionsProposal | None:
    """Return one autonomous optimization suggestion without applying it."""

    source_plan = plan or build_autonomous_optimization_suggestions()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def autonomous_optimization_suggestions_proposals_for_status(
    status: AutonomousOptimizationSuggestionsStatus,
    plan: AutonomousOptimizationSuggestionsPlan | None = None,
) -> tuple[AutonomousOptimizationSuggestionsProposal, ...]:
    """Return autonomous optimization suggestions by advisory status."""

    source_plan = plan or build_autonomous_optimization_suggestions()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def autonomous_optimization_suggestions_proposals_for_confidence(
    confidence: AutonomousOptimizationSuggestionsConfidence,
    plan: AutonomousOptimizationSuggestionsPlan | None = None,
) -> tuple[AutonomousOptimizationSuggestionsProposal, ...]:
    """Return autonomous optimization suggestions by confidence band."""

    source_plan = plan or build_autonomous_optimization_suggestions()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
