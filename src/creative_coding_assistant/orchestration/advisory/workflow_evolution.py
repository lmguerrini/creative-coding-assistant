"""V6.5 advisory workflow evolution metadata."""

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

WorkflowEvolutionPlan = SelfEvolutionPlan
WorkflowEvolutionProposal = SelfEvolutionProposal
WorkflowEvolutionStatus = SelfEvolutionStatus
WorkflowEvolutionConfidence = SelfEvolutionConfidence

_ROLE = "workflow_evolution"
_ROADMAP_ITEM = "Workflow Evolution"
_PROPOSAL_KINDS = (
    "workflow_success_pattern_gap",
    "workflow_failure_path_pressure",
    "workflow_dependency_alignment",
    "workflow_governance_gap",
    "workflow_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "workflow_graph",
    "workflow_review",
    "execution_replay",
    "routing_policy",
)


def build_workflow_evolution(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> WorkflowEvolutionPlan:
    """Build workflow evolution proposals without mutating workflows."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def workflow_evolution_proposal_by_id(
    proposal_id: str,
    plan: WorkflowEvolutionPlan | None = None,
) -> WorkflowEvolutionProposal | None:
    """Return one workflow evolution proposal without applying it."""

    source_plan = plan or build_workflow_evolution()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def workflow_evolution_proposals_for_status(
    status: WorkflowEvolutionStatus,
    plan: WorkflowEvolutionPlan | None = None,
) -> tuple[WorkflowEvolutionProposal, ...]:
    """Return workflow evolution proposals by advisory status."""

    source_plan = plan or build_workflow_evolution()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def workflow_evolution_proposals_for_confidence(
    confidence: WorkflowEvolutionConfidence,
    plan: WorkflowEvolutionPlan | None = None,
) -> tuple[WorkflowEvolutionProposal, ...]:
    """Return workflow evolution proposals by confidence band."""

    source_plan = plan or build_workflow_evolution()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
