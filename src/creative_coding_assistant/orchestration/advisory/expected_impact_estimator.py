"""V6.5 advisory expected impact estimator metadata."""

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

ExpectedImpactEstimatorPlan = SelfEvolutionPlan
ExpectedImpactEstimatorProposal = SelfEvolutionProposal
ExpectedImpactEstimatorStatus = SelfEvolutionStatus
ExpectedImpactEstimatorConfidence = SelfEvolutionConfidence

_ROLE = "expected_impact_estimator"
_ROADMAP_ITEM = "Expected Impact Estimator"
_PROPOSAL_KINDS = (
    "impact_signal_alignment",
    "quality_impact_estimate",
    "performance_impact_estimate",
    "cross_capability_impact_dependency",
    "impact_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "quality_prediction_engine",
    "performance_prediction",
    "improvement_ranking_engine",
    "cost_benefit_analysis",
    "risk_analysis",
)


def build_expected_impact_estimator(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> ExpectedImpactEstimatorPlan:
    """Build expected-impact proposals without mutating impact policy."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def expected_impact_estimator_proposal_by_id(
    proposal_id: str,
    plan: ExpectedImpactEstimatorPlan | None = None,
) -> ExpectedImpactEstimatorProposal | None:
    """Return one expected-impact proposal without applying it."""

    source_plan = plan or build_expected_impact_estimator()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def expected_impact_estimator_proposals_for_status(
    status: ExpectedImpactEstimatorStatus,
    plan: ExpectedImpactEstimatorPlan | None = None,
) -> tuple[ExpectedImpactEstimatorProposal, ...]:
    """Return expected-impact proposals by advisory status."""

    source_plan = plan or build_expected_impact_estimator()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def expected_impact_estimator_proposals_for_confidence(
    confidence: ExpectedImpactEstimatorConfidence,
    plan: ExpectedImpactEstimatorPlan | None = None,
) -> tuple[ExpectedImpactEstimatorProposal, ...]:
    """Return expected-impact proposals by confidence band."""

    source_plan = plan or build_expected_impact_estimator()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
