"""V6.5 advisory cost/benefit analysis metadata."""

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

CostBenefitAnalysisPlan = SelfEvolutionPlan
CostBenefitAnalysisProposal = SelfEvolutionProposal
CostBenefitAnalysisStatus = SelfEvolutionStatus
CostBenefitAnalysisConfidence = SelfEvolutionConfidence

_ROLE = "cost_benefit_analysis"
_ROADMAP_ITEM = "Cost / Benefit Analysis"
_PROPOSAL_KINDS = (
    "cost_benefit_signal_alignment",
    "benefit_value_review",
    "cost_pressure_review",
    "cost_benefit_dependency_tradeoff",
    "cost_benefit_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "cost_trends",
    "cost_prediction_engine",
    "quality_cost_optimizer",
    "budget_policies",
    "improvement_ranking_engine",
)


def build_cost_benefit_analysis(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CostBenefitAnalysisPlan:
    """Build cost/benefit proposals without enforcing budgets or pricing."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def cost_benefit_analysis_proposal_by_id(
    proposal_id: str,
    plan: CostBenefitAnalysisPlan | None = None,
) -> CostBenefitAnalysisProposal | None:
    """Return one cost/benefit proposal without applying it."""

    source_plan = plan or build_cost_benefit_analysis()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def cost_benefit_analysis_proposals_for_status(
    status: CostBenefitAnalysisStatus,
    plan: CostBenefitAnalysisPlan | None = None,
) -> tuple[CostBenefitAnalysisProposal, ...]:
    """Return cost/benefit proposals by advisory status."""

    source_plan = plan or build_cost_benefit_analysis()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def cost_benefit_analysis_proposals_for_confidence(
    confidence: CostBenefitAnalysisConfidence,
    plan: CostBenefitAnalysisPlan | None = None,
) -> tuple[CostBenefitAnalysisProposal, ...]:
    """Return cost/benefit proposals by confidence band."""

    source_plan = plan or build_cost_benefit_analysis()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
