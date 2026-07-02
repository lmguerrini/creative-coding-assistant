"""V6.5 advisory risk analysis metadata."""

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

RiskAnalysisPlan = SelfEvolutionPlan
RiskAnalysisProposal = SelfEvolutionProposal
RiskAnalysisStatus = SelfEvolutionStatus
RiskAnalysisConfidence = SelfEvolutionConfidence

_ROLE = "risk_analysis"
_ROADMAP_ITEM = "Risk Analysis"
_PROPOSAL_KINDS = (
    "risk_signal_alignment",
    "mutation_risk_review",
    "workflow_risk_dependency",
    "governance_risk_guardrail",
    "risk_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "workflow_risk_engine",
    "failure_analysis",
    "failure_tracking",
    "knowledge_quality_scoring",
    "research_core_surface",
)


def build_risk_analysis(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> RiskAnalysisPlan:
    """Build risk analysis proposals without mutating risk policy."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def risk_analysis_proposal_by_id(
    proposal_id: str,
    plan: RiskAnalysisPlan | None = None,
) -> RiskAnalysisProposal | None:
    """Return one risk analysis proposal without applying it."""

    source_plan = plan or build_risk_analysis()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def risk_analysis_proposals_for_status(
    status: RiskAnalysisStatus,
    plan: RiskAnalysisPlan | None = None,
) -> tuple[RiskAnalysisProposal, ...]:
    """Return risk analysis proposals by advisory status."""

    source_plan = plan or build_risk_analysis()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def risk_analysis_proposals_for_confidence(
    confidence: RiskAnalysisConfidence,
    plan: RiskAnalysisPlan | None = None,
) -> tuple[RiskAnalysisProposal, ...]:
    """Return risk analysis proposals by confidence band."""

    source_plan = plan or build_risk_analysis()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
