"""V6.5 advisory memory evolution policy metadata."""

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

MemoryEvolutionPoliciesPlan = SelfEvolutionPlan
MemoryEvolutionPoliciesProposal = SelfEvolutionProposal
MemoryEvolutionPoliciesStatus = SelfEvolutionStatus
MemoryEvolutionPoliciesConfidence = SelfEvolutionConfidence

_ROLE = "memory_evolution_policies"
_ROADMAP_ITEM = "Memory Evolution Policies"
_PROPOSAL_KINDS = (
    "memory_policy_signal_alignment",
    "memory_retention_policy_review",
    "memory_quality_boundary_policy",
    "memory_cross_capability_dependency",
    "memory_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "creative_memory_core_surface",
    "memory_integration_boundaries",
    "long_term_creative_memory",
    "session_memory_evolution",
)


def build_memory_evolution_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> MemoryEvolutionPoliciesPlan:
    """Build memory evolution policy proposals without mutating memory."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def memory_evolution_policies_proposal_by_id(
    proposal_id: str,
    plan: MemoryEvolutionPoliciesPlan | None = None,
) -> MemoryEvolutionPoliciesProposal | None:
    """Return one memory evolution policy proposal without applying it."""

    source_plan = plan or build_memory_evolution_policies()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def memory_evolution_policies_proposals_for_status(
    status: MemoryEvolutionPoliciesStatus,
    plan: MemoryEvolutionPoliciesPlan | None = None,
) -> tuple[MemoryEvolutionPoliciesProposal, ...]:
    """Return memory evolution policy proposals by advisory status."""

    source_plan = plan or build_memory_evolution_policies()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def memory_evolution_policies_proposals_for_confidence(
    confidence: MemoryEvolutionPoliciesConfidence,
    plan: MemoryEvolutionPoliciesPlan | None = None,
) -> tuple[MemoryEvolutionPoliciesProposal, ...]:
    """Return memory evolution policy proposals by confidence band."""

    source_plan = plan or build_memory_evolution_policies()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
