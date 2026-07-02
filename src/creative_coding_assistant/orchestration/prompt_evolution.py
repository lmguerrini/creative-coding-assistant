"""V6.5 advisory prompt evolution metadata."""

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

PromptEvolutionPlan = SelfEvolutionPlan
PromptEvolutionProposal = SelfEvolutionProposal
PromptEvolutionStatus = SelfEvolutionStatus
PromptEvolutionConfidence = SelfEvolutionConfidence

_ROLE = "prompt_evolution"
_ROADMAP_ITEM = "Prompt Evolution"
_PROPOSAL_KINDS = (
    "prompt_quality_drift",
    "prompt_context_reuse",
    "prompt_cost_pressure",
    "prompt_governance_gap",
    "prompt_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "prompt_templates",
    "provider_prompt_boundary",
    "workflow_planning",
    "evaluation_learning",
)


def build_prompt_evolution(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> PromptEvolutionPlan:
    """Build prompt evolution proposals without rewriting prompts."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def prompt_evolution_proposal_by_id(
    proposal_id: str,
    plan: PromptEvolutionPlan | None = None,
) -> PromptEvolutionProposal | None:
    """Return one prompt evolution proposal without applying it."""

    source_plan = plan or build_prompt_evolution()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def prompt_evolution_proposals_for_status(
    status: PromptEvolutionStatus,
    plan: PromptEvolutionPlan | None = None,
) -> tuple[PromptEvolutionProposal, ...]:
    """Return prompt evolution proposals by advisory status."""

    source_plan = plan or build_prompt_evolution()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def prompt_evolution_proposals_for_confidence(
    confidence: PromptEvolutionConfidence,
    plan: PromptEvolutionPlan | None = None,
) -> tuple[PromptEvolutionProposal, ...]:
    """Return prompt evolution proposals by confidence band."""

    source_plan = plan or build_prompt_evolution()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
