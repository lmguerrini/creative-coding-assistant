"""V6.5 advisory routing evolution policy metadata."""

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

RoutingEvolutionPoliciesPlan = SelfEvolutionPlan
RoutingEvolutionPoliciesProposal = SelfEvolutionProposal
RoutingEvolutionPoliciesStatus = SelfEvolutionStatus
RoutingEvolutionPoliciesConfidence = SelfEvolutionConfidence

_ROLE = "routing_evolution_policies"
_ROADMAP_ITEM = "Routing Evolution Policies"
_PROPOSAL_KINDS = (
    "routing_policy_signal_alignment",
    "routing_quality_cost_tradeoff",
    "routing_handoff_boundary_policy",
    "routing_governance_guardrail",
    "routing_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "routing_intelligence",
    "model_router",
    "routing_explainability",
    "runtime_recommendation_engine",
)


def build_routing_evolution_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> RoutingEvolutionPoliciesPlan:
    """Build routing evolution policy proposals without mutating routing."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def routing_evolution_policies_proposal_by_id(
    proposal_id: str,
    plan: RoutingEvolutionPoliciesPlan | None = None,
) -> RoutingEvolutionPoliciesProposal | None:
    """Return one routing evolution policy proposal without applying it."""

    source_plan = plan or build_routing_evolution_policies()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def routing_evolution_policies_proposals_for_status(
    status: RoutingEvolutionPoliciesStatus,
    plan: RoutingEvolutionPoliciesPlan | None = None,
) -> tuple[RoutingEvolutionPoliciesProposal, ...]:
    """Return routing evolution policy proposals by advisory status."""

    source_plan = plan or build_routing_evolution_policies()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def routing_evolution_policies_proposals_for_confidence(
    confidence: RoutingEvolutionPoliciesConfidence,
    plan: RoutingEvolutionPoliciesPlan | None = None,
) -> tuple[RoutingEvolutionPoliciesProposal, ...]:
    """Return routing evolution policy proposals by confidence band."""

    source_plan = plan or build_routing_evolution_policies()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
