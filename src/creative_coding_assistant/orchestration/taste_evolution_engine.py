"""V6.5 advisory taste evolution metadata."""

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

TasteEvolutionEnginePlan = SelfEvolutionPlan
TasteEvolutionEngineProposal = SelfEvolutionProposal
TasteEvolutionEngineStatus = SelfEvolutionStatus
TasteEvolutionEngineConfidence = SelfEvolutionConfidence

_ROLE = "taste_evolution_engine"
_ROADMAP_ITEM = "Taste Evolution Engine"
_PROPOSAL_KINDS = (
    "taste_signal_alignment",
    "taste_preference_drift_review",
    "creative_memory_taste_dependency",
    "style_continuity_guardrail",
    "taste_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "creative_memory_secondary_surface",
    "long_term_creative_memory",
    "creative_strategy",
    "style_profiles",
)


def build_taste_evolution_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> TasteEvolutionEnginePlan:
    """Build taste evolution proposals without mutating taste models."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def taste_evolution_engine_proposal_by_id(
    proposal_id: str,
    plan: TasteEvolutionEnginePlan | None = None,
) -> TasteEvolutionEngineProposal | None:
    """Return one taste evolution proposal without applying it."""

    source_plan = plan or build_taste_evolution_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def taste_evolution_engine_proposals_for_status(
    status: TasteEvolutionEngineStatus,
    plan: TasteEvolutionEnginePlan | None = None,
) -> tuple[TasteEvolutionEngineProposal, ...]:
    """Return taste evolution proposals by advisory status."""

    source_plan = plan or build_taste_evolution_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def taste_evolution_engine_proposals_for_confidence(
    confidence: TasteEvolutionEngineConfidence,
    plan: TasteEvolutionEnginePlan | None = None,
) -> tuple[TasteEvolutionEngineProposal, ...]:
    """Return taste evolution proposals by confidence band."""

    source_plan = plan or build_taste_evolution_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
