"""V6.5 advisory architecture evolution metadata."""

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

ArchitectureEvolutionEnginePlan = SelfEvolutionPlan
ArchitectureEvolutionEngineProposal = SelfEvolutionProposal
ArchitectureEvolutionEngineStatus = SelfEvolutionStatus
ArchitectureEvolutionEngineConfidence = SelfEvolutionConfidence

_ROLE = "architecture_evolution_engine"
_ROADMAP_ITEM = "Architecture Evolution Engine"
_PROPOSAL_KINDS = (
    "architecture_dependency_graph_signal",
    "architecture_boundary_alignment",
    "architecture_cross_capability_integration",
    "architecture_debt_prioritization",
    "architecture_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "engine_contract_consistency_registry",
    "architecture_consistency_pass",
    "system_integration_review",
    "technical_debt_ledger",
)


def build_architecture_evolution_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> ArchitectureEvolutionEnginePlan:
    """Build architecture evolution proposals without mutating architecture."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def architecture_evolution_engine_proposal_by_id(
    proposal_id: str,
    plan: ArchitectureEvolutionEnginePlan | None = None,
) -> ArchitectureEvolutionEngineProposal | None:
    """Return one architecture evolution proposal without applying it."""

    source_plan = plan or build_architecture_evolution_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def architecture_evolution_engine_proposals_for_status(
    status: ArchitectureEvolutionEngineStatus,
    plan: ArchitectureEvolutionEnginePlan | None = None,
) -> tuple[ArchitectureEvolutionEngineProposal, ...]:
    """Return architecture evolution proposals by advisory status."""

    source_plan = plan or build_architecture_evolution_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def architecture_evolution_engine_proposals_for_confidence(
    confidence: ArchitectureEvolutionEngineConfidence,
    plan: ArchitectureEvolutionEnginePlan | None = None,
) -> tuple[ArchitectureEvolutionEngineProposal, ...]:
    """Return architecture evolution proposals by confidence band."""

    source_plan = plan or build_architecture_evolution_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
