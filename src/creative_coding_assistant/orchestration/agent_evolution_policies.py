"""V6.5 advisory agent evolution policy metadata."""

from __future__ import annotations

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    SelfEvolutionConfidence,
    SelfEvolutionPlan,
    SelfEvolutionProposal,
    SelfEvolutionStatus,
    build_self_evolution_plan,
)

AgentEvolutionPoliciesPlan = SelfEvolutionPlan
AgentEvolutionPoliciesProposal = SelfEvolutionProposal
AgentEvolutionPoliciesStatus = SelfEvolutionStatus
AgentEvolutionPoliciesConfidence = SelfEvolutionConfidence

_ROLE = "agent_evolution_policies"
_ROADMAP_ITEM = "Agent Evolution Policies"
_PROPOSAL_KINDS = (
    "agent_policy_signal_alignment",
    "agent_capability_gap_policy",
    "agent_collaboration_policy_review",
    "agent_activation_policy_guardrail",
    "agent_policy_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "agent_capability_registry",
    "agent_contract_audit",
    "agent_routing_metadata",
    "agent_lifecycle",
)


def build_agent_evolution_policies(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> AgentEvolutionPoliciesPlan:
    """Build agent evolution policy proposals without mutating agents."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def agent_evolution_policies_proposal_by_id(
    proposal_id: str,
    plan: AgentEvolutionPoliciesPlan | None = None,
) -> AgentEvolutionPoliciesProposal | None:
    """Return one agent evolution policy proposal without applying it."""

    source_plan = plan or build_agent_evolution_policies()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def agent_evolution_policies_proposals_for_status(
    status: AgentEvolutionPoliciesStatus,
    plan: AgentEvolutionPoliciesPlan | None = None,
) -> tuple[AgentEvolutionPoliciesProposal, ...]:
    """Return agent evolution policy proposals by advisory status."""

    source_plan = plan or build_agent_evolution_policies()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def agent_evolution_policies_proposals_for_confidence(
    confidence: AgentEvolutionPoliciesConfidence,
    plan: AgentEvolutionPoliciesPlan | None = None,
) -> tuple[AgentEvolutionPoliciesProposal, ...]:
    """Return agent evolution policy proposals by confidence band."""

    source_plan = plan or build_agent_evolution_policies()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )


def agent_invocation_is_blocked() -> bool:
    """Return whether the shared V6.5 boundary blocks agent invocation."""

    return "agent_invocation" in BLOCKED_RUNTIME_BEHAVIORS
