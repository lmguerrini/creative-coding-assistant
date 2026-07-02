"""V6.5 advisory retrieval evolution policy metadata."""

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

RetrievalEvolutionPoliciesPlan = SelfEvolutionPlan
RetrievalEvolutionPoliciesProposal = SelfEvolutionProposal
RetrievalEvolutionPoliciesStatus = SelfEvolutionStatus
RetrievalEvolutionPoliciesConfidence = SelfEvolutionConfidence

_ROLE = "retrieval_evolution_policies"
_ROADMAP_ITEM = "Retrieval Evolution Policies"
_PROPOSAL_KINDS = (
    "retrieval_policy_signal_alignment",
    "retrieval_index_refresh_policy",
    "retrieval_quality_boundary_policy",
    "retrieval_cross_capability_dependency",
    "retrieval_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "retrieval_foundation",
    "retrieval_integration_boundaries",
    "retrieval_compression",
    "kb_embedding_indexer_foundation",
)


def build_retrieval_evolution_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> RetrievalEvolutionPoliciesPlan:
    """Build retrieval evolution policy proposals without mutating retrieval."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def retrieval_evolution_policies_proposal_by_id(
    proposal_id: str,
    plan: RetrievalEvolutionPoliciesPlan | None = None,
) -> RetrievalEvolutionPoliciesProposal | None:
    """Return one retrieval evolution policy proposal without applying it."""

    source_plan = plan or build_retrieval_evolution_policies()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def retrieval_evolution_policies_proposals_for_status(
    status: RetrievalEvolutionPoliciesStatus,
    plan: RetrievalEvolutionPoliciesPlan | None = None,
) -> tuple[RetrievalEvolutionPoliciesProposal, ...]:
    """Return retrieval evolution policy proposals by advisory status."""

    source_plan = plan or build_retrieval_evolution_policies()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def retrieval_evolution_policies_proposals_for_confidence(
    confidence: RetrievalEvolutionPoliciesConfidence,
    plan: RetrievalEvolutionPoliciesPlan | None = None,
) -> tuple[RetrievalEvolutionPoliciesProposal, ...]:
    """Return retrieval evolution policy proposals by confidence band."""

    source_plan = plan or build_retrieval_evolution_policies()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
