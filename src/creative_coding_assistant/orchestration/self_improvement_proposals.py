"""V6.5 advisory self-improvement proposal metadata."""

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

SelfImprovementProposalsPlan = SelfEvolutionPlan
SelfImprovementProposalsProposal = SelfEvolutionProposal
SelfImprovementProposalsStatus = SelfEvolutionStatus
SelfImprovementProposalsConfidence = SelfEvolutionConfidence

_ROLE = "self_improvement_proposals"
_ROADMAP_ITEM = "Self-Improvement Proposals"
_PROPOSAL_KINDS = (
    "self_improvement_signal_alignment",
    "self_improvement_quality_gap",
    "self_improvement_cost_risk_tradeoff",
    "self_improvement_dependency_review",
    "self_improvement_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "runtime_recommendation_engine",
    "system_health_monitoring",
    "quality_dashboard",
    "codex_engineering_audit",
)


def build_self_improvement_proposals(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> SelfImprovementProposalsPlan:
    """Build self-improvement proposals without applying them."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def self_improvement_proposals_proposal_by_id(
    proposal_id: str,
    plan: SelfImprovementProposalsPlan | None = None,
) -> SelfImprovementProposalsProposal | None:
    """Return one self-improvement proposal without applying it."""

    source_plan = plan or build_self_improvement_proposals()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def self_improvement_proposals_for_status(
    status: SelfImprovementProposalsStatus,
    plan: SelfImprovementProposalsPlan | None = None,
) -> tuple[SelfImprovementProposalsProposal, ...]:
    """Return self-improvement proposals by advisory status."""

    source_plan = plan or build_self_improvement_proposals()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def self_improvement_proposals_for_confidence(
    confidence: SelfImprovementProposalsConfidence,
    plan: SelfImprovementProposalsPlan | None = None,
) -> tuple[SelfImprovementProposalsProposal, ...]:
    """Return self-improvement proposals by confidence band."""

    source_plan = plan or build_self_improvement_proposals()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
