"""V6.5 advisory creative evolution policy metadata."""

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

CreativeEvolutionPoliciesPlan = SelfEvolutionPlan
CreativeEvolutionPoliciesProposal = SelfEvolutionProposal
CreativeEvolutionPoliciesStatus = SelfEvolutionStatus
CreativeEvolutionPoliciesConfidence = SelfEvolutionConfidence

_ROLE = "creative_evolution_policies"
_ROADMAP_ITEM = "Creative Evolution Policies"
_PROPOSAL_KINDS = (
    "creative_policy_signal_alignment",
    "creative_quality_policy_review",
    "creative_diversity_policy_guardrail",
    "creative_strategy_dependency",
    "creative_policy_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "creative_strategy",
    "creative_success_learning",
    "creative_quality_prediction",
    "creative_diversity_audit",
)


def build_creative_evolution_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CreativeEvolutionPoliciesPlan:
    """Build creative evolution policy proposals without mutating creativity."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def creative_evolution_policies_proposal_by_id(
    proposal_id: str,
    plan: CreativeEvolutionPoliciesPlan | None = None,
) -> CreativeEvolutionPoliciesProposal | None:
    """Return one creative evolution policy proposal without applying it."""

    source_plan = plan or build_creative_evolution_policies()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def creative_evolution_policies_proposals_for_status(
    status: CreativeEvolutionPoliciesStatus,
    plan: CreativeEvolutionPoliciesPlan | None = None,
) -> tuple[CreativeEvolutionPoliciesProposal, ...]:
    """Return creative evolution policy proposals by advisory status."""

    source_plan = plan or build_creative_evolution_policies()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def creative_evolution_policies_proposals_for_confidence(
    confidence: CreativeEvolutionPoliciesConfidence,
    plan: CreativeEvolutionPoliciesPlan | None = None,
) -> tuple[CreativeEvolutionPoliciesProposal, ...]:
    """Return creative evolution policy proposals by confidence band."""

    source_plan = plan or build_creative_evolution_policies()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
