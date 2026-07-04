"""V6.5 advisory quality trend metadata."""

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

QualityTrendsPlan = SelfEvolutionPlan
QualityTrendsProposal = SelfEvolutionProposal
QualityTrendsStatus = SelfEvolutionStatus
QualityTrendsConfidence = SelfEvolutionConfidence

_ROLE = "quality_trends"
_ROADMAP_ITEM = "Quality Trends"
_PROPOSAL_KINDS = (
    "quality_drift_signal",
    "quality_memory_alignment",
    "quality_knowledge_freshness_gap",
    "quality_research_confidence_gap",
    "quality_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "quality_dashboard",
    "evaluation_learning",
    "creative_quality_prediction",
    "system_health_monitoring",
)


def build_quality_trends(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> QualityTrendsPlan:
    """Build quality trend proposals without mutating evaluation records."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def quality_trends_proposal_by_id(
    proposal_id: str,
    plan: QualityTrendsPlan | None = None,
) -> QualityTrendsProposal | None:
    """Return one quality trend proposal without applying it."""

    source_plan = plan or build_quality_trends()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def quality_trends_proposals_for_status(
    status: QualityTrendsStatus,
    plan: QualityTrendsPlan | None = None,
) -> tuple[QualityTrendsProposal, ...]:
    """Return quality trend proposals by advisory status."""

    source_plan = plan or build_quality_trends()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def quality_trends_proposals_for_confidence(
    confidence: QualityTrendsConfidence,
    plan: QualityTrendsPlan | None = None,
) -> tuple[QualityTrendsProposal, ...]:
    """Return quality trend proposals by confidence band."""

    source_plan = plan or build_quality_trends()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
