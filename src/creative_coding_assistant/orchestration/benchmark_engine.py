"""V6.5 advisory benchmark engine metadata."""

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

BenchmarkEnginePlan = SelfEvolutionPlan
BenchmarkEngineProposal = SelfEvolutionProposal
BenchmarkEngineStatus = SelfEvolutionStatus
BenchmarkEngineConfidence = SelfEvolutionConfidence

_ROLE = "benchmark_engine"
_ROADMAP_ITEM = "Benchmark Engine"
_PROPOSAL_KINDS = (
    "benchmark_signal_coverage",
    "benchmark_quality_regression_watch",
    "benchmark_cost_pressure_watch",
    "benchmark_cross_capability_comparison",
    "benchmark_rollback_readiness",
)
_DOWNSTREAM_SYSTEMS = (
    "evaluation_learning",
    "quality_dashboard",
    "cost_dashboard",
    "system_integration_review",
)


def build_benchmark_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> BenchmarkEnginePlan:
    """Build benchmark engine proposals without executing benchmarks."""

    return build_self_evolution_plan(
        role=_ROLE,
        roadmap_item=_ROADMAP_ITEM,
        proposal_kinds=_PROPOSAL_KINDS,
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        downstream_systems=_DOWNSTREAM_SYSTEMS,
    )


def benchmark_engine_proposal_by_id(
    proposal_id: str,
    plan: BenchmarkEnginePlan | None = None,
) -> BenchmarkEngineProposal | None:
    """Return one benchmark engine proposal without applying it."""

    source_plan = plan or build_benchmark_engine()
    for proposal in source_plan.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def benchmark_engine_proposals_for_status(
    status: BenchmarkEngineStatus,
    plan: BenchmarkEnginePlan | None = None,
) -> tuple[BenchmarkEngineProposal, ...]:
    """Return benchmark engine proposals by advisory status."""

    source_plan = plan or build_benchmark_engine()
    return tuple(
        proposal for proposal in source_plan.proposals if proposal.status == status
    )


def benchmark_engine_proposals_for_confidence(
    confidence: BenchmarkEngineConfidence,
    plan: BenchmarkEnginePlan | None = None,
) -> tuple[BenchmarkEngineProposal, ...]:
    """Return benchmark engine proposals by confidence band."""

    source_plan = plan or build_benchmark_engine()
    return tuple(
        proposal
        for proposal in source_plan.proposals
        if proposal.confidence == confidence
    )
