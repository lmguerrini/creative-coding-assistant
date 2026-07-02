"""V6.5 advisory workflow mutation proposal metadata."""

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

WorkflowMutationEnginePlan = SelfEvolutionPlan
WorkflowMutationEngineProposal = SelfEvolutionProposal
WorkflowMutationEngineStatus = SelfEvolutionStatus
WorkflowMutationEngineConfidence = SelfEvolutionConfidence

_ROLE = "workflow_mutation_engine"
_ROADMAP_ITEM = "Workflow Mutation Engine"
_PROPOSAL_KINDS = (
    "workflow_mutation_candidate",
    "workflow_graph_refinement_signal",
    "workflow_execution_policy_alignment",
    "workflow_risk_guardrail_signal",
    "workflow_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "workflow_graph",
    "workflow_review",
    "workflow_risk_engine",
    "workflow_replay_engine",
)


def build_workflow_mutation_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> WorkflowMutationEnginePlan:
    """Build workflow mutation proposals without mutating workflows."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def workflow_mutation_engine_proposal_by_id(
    proposal_id: str,
    plan: WorkflowMutationEnginePlan | None = None,
) -> WorkflowMutationEngineProposal | None:
    """Return one workflow mutation proposal without applying it."""

    source_plan = plan or build_workflow_mutation_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def workflow_mutation_engine_proposals_for_status(
    status: WorkflowMutationEngineStatus,
    plan: WorkflowMutationEnginePlan | None = None,
) -> tuple[WorkflowMutationEngineProposal, ...]:
    """Return workflow mutation proposals by advisory status."""

    source_plan = plan or build_workflow_mutation_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def workflow_mutation_engine_proposals_for_confidence(
    confidence: WorkflowMutationEngineConfidence,
    plan: WorkflowMutationEnginePlan | None = None,
) -> tuple[WorkflowMutationEngineProposal, ...]:
    """Return workflow mutation proposals by confidence band."""

    source_plan = plan or build_workflow_mutation_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
