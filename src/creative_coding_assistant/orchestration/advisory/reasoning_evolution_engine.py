"""V6.5 advisory reasoning evolution metadata."""

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

ReasoningEvolutionEnginePlan = SelfEvolutionPlan
ReasoningEvolutionEngineProposal = SelfEvolutionProposal
ReasoningEvolutionEngineStatus = SelfEvolutionStatus
ReasoningEvolutionEngineConfidence = SelfEvolutionConfidence

_ROLE = "reasoning_evolution_engine"
_ROADMAP_ITEM = "Reasoning Evolution Engine"
_PROPOSAL_KINDS = (
    "reasoning_signal_alignment",
    "reasoning_path_quality_review",
    "reasoning_budget_dependency",
    "reasoning_model_selection_guardrail",
    "reasoning_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "creative_reasoning",
    "reasoning_budget_optimizer",
    "workflow_explainability_dashboard",
    "model_router",
)


def build_reasoning_evolution_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> ReasoningEvolutionEnginePlan:
    """Build reasoning evolution proposals without mutating reasoning policy."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def reasoning_evolution_engine_proposal_by_id(
    proposal_id: str,
    plan: ReasoningEvolutionEnginePlan | None = None,
) -> ReasoningEvolutionEngineProposal | None:
    """Return one reasoning evolution proposal without applying it."""

    source_plan = plan or build_reasoning_evolution_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def reasoning_evolution_engine_proposals_for_status(
    status: ReasoningEvolutionEngineStatus,
    plan: ReasoningEvolutionEnginePlan | None = None,
) -> tuple[ReasoningEvolutionEngineProposal, ...]:
    """Return reasoning evolution proposals by advisory status."""

    source_plan = plan or build_reasoning_evolution_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def reasoning_evolution_engine_proposals_for_confidence(
    confidence: ReasoningEvolutionEngineConfidence,
    plan: ReasoningEvolutionEnginePlan | None = None,
) -> tuple[ReasoningEvolutionEngineProposal, ...]:
    """Return reasoning evolution proposals by confidence band."""

    source_plan = plan or build_reasoning_evolution_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
