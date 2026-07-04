"""V6.5 advisory core surface for self-evolution proposals."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_evolution_policies import (
    build_agent_evolution_policies,
)
from creative_coding_assistant.orchestration.architecture_evolution_engine import (
    build_architecture_evolution_engine,
)
from creative_coding_assistant.orchestration.autonomous_optimization_suggestions import (  # noqa: E501
    build_autonomous_optimization_suggestions,
)
from creative_coding_assistant.orchestration.benchmark_engine import (
    build_benchmark_engine,
)
from creative_coding_assistant.orchestration.cost_benefit_analysis import (
    build_cost_benefit_analysis,
)
from creative_coding_assistant.orchestration.cost_trends import build_cost_trends
from creative_coding_assistant.orchestration.creative_evolution_policies import (
    build_creative_evolution_policies,
)
from creative_coding_assistant.orchestration.expected_impact_estimator import (
    build_expected_impact_estimator,
)
from creative_coding_assistant.orchestration.improvement_ranking_engine import (
    build_improvement_ranking_engine,
)
from creative_coding_assistant.orchestration.memory_evolution_policies import (
    build_memory_evolution_policies,
)
from creative_coding_assistant.orchestration.prompt_evolution import (
    build_prompt_evolution,
)
from creative_coding_assistant.orchestration.quality_trends import (
    build_quality_trends,
)
from creative_coding_assistant.orchestration.reasoning_evolution_engine import (
    build_reasoning_evolution_engine,
)
from creative_coding_assistant.orchestration.retrieval_evolution_policies import (
    build_retrieval_evolution_policies,
)
from creative_coding_assistant.orchestration.risk_analysis import build_risk_analysis
from creative_coding_assistant.orchestration.rollback_strategy_generator import (
    build_rollback_strategy_generator,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_evolution_policies import (
    build_routing_evolution_policies,
)
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
    SelfEvolutionConfidence,
    SelfEvolutionPlan,
    SelfEvolutionPosture,
    SelfEvolutionProposal,
    SelfEvolutionStatus,
    overall_evolution_posture,
    overall_proposal_rank_score,
    proposal_ids_for_status,
    proposals_for_confidence_ids,
)
from creative_coding_assistant.orchestration.self_improvement_proposals import (
    build_self_improvement_proposals,
)
from creative_coding_assistant.orchestration.strategy_evolution_engine import (
    build_strategy_evolution_engine,
)
from creative_coding_assistant.orchestration.taste_evolution_engine import (
    build_taste_evolution_engine,
)
from creative_coding_assistant.orchestration.workflow_evolution import (
    build_workflow_evolution,
)
from creative_coding_assistant.orchestration.workflow_mutation_engine import (
    build_workflow_mutation_engine,
)

CORE_ROADMAP_ITEMS = (
    "Prompt Evolution",
    "Workflow Evolution",
    "Benchmark Engine",
    "Quality Trends",
    "Cost Trends",
    "Autonomous Optimization Suggestions",
    "Architecture Evolution Engine",
    "Workflow Mutation Engine",
    "Strategy Evolution Engine",
    "Agent Evolution Policies",
    "Routing Evolution Policies",
    "Memory Evolution Policies",
    "Retrieval Evolution Policies",
    "Self-Improvement Proposals",
    "Creative Evolution Policies",
    "Taste Evolution Engine",
    "Reasoning Evolution Engine",
    "Improvement Ranking Engine",
    "Cost / Benefit Analysis",
    "Risk Analysis",
    "Expected Impact Estimator",
    "Rollback Strategy Generator",
)

_PLAN_BUILDERS = (
    build_prompt_evolution,
    build_workflow_evolution,
    build_benchmark_engine,
    build_quality_trends,
    build_cost_trends,
    build_autonomous_optimization_suggestions,
    build_architecture_evolution_engine,
    build_workflow_mutation_engine,
    build_strategy_evolution_engine,
    build_agent_evolution_policies,
    build_routing_evolution_policies,
    build_memory_evolution_policies,
    build_retrieval_evolution_policies,
    build_self_improvement_proposals,
    build_creative_evolution_policies,
    build_taste_evolution_engine,
    build_reasoning_evolution_engine,
    build_improvement_ranking_engine,
    build_cost_benefit_analysis,
    build_risk_analysis,
    build_expected_impact_estimator,
    build_rollback_strategy_generator,
)


class SelfEvolutionCoreSurfacePlan(BaseModel):
    """Aggregated advisory view over all explicit V6.5 roadmap item plans."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["self_evolution_core_surface"] = "self_evolution_core_surface"
    serialization_version: Literal["self_evolution_core_surface.v1"] = (
        "self_evolution_core_surface.v1"
    )
    plans: tuple[SelfEvolutionPlan, ...] = Field(min_length=22, max_length=22)
    plan_count: int = Field(ge=22, le=22)
    plan_roles: tuple[str, ...] = Field(min_length=22, max_length=22)
    plan_serialization_versions: tuple[str, ...] = Field(
        min_length=22,
        max_length=22,
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=22, max_length=22)
    covered_roadmap_item_count: int = Field(ge=22, le=22)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_count: int = Field(ge=4, le=4)
    upstream_signal_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    upstream_signal_id_count: int = Field(ge=20, le=20)
    proposals: tuple[SelfEvolutionProposal, ...] = Field(
        min_length=110,
        max_length=110,
    )
    proposal_ids: tuple[str, ...] = Field(min_length=110, max_length=110)
    proposal_count: int = Field(ge=110, le=110)
    candidate_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    review_required_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    guarded_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    high_confidence_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    hitl_required_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    generated_evolution_report_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_routing_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    mutated_retrieval_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    provider_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    candidate_proposal_count: int = Field(ge=0, le=110)
    review_required_proposal_count: int = Field(ge=0, le=110)
    guarded_proposal_count: int = Field(ge=0, le=110)
    high_confidence_proposal_count: int = Field(ge=0, le=110)
    hitl_required_proposal_count: int = Field(ge=0, le=110)
    highest_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_evolution_posture: SelfEvolutionPosture
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    all_v6_signal_sources_integrated: Literal[True] = True
    roadmap_traceability_implemented: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    advisory_evolution_report_implemented: Literal[True] = True
    autonomous_runtime_evolution_implemented: Literal[False] = False
    prompt_rewriting_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _core_surface_matches_plans(self) -> Self:
        if self.plan_count != len(self.plans):
            raise ValueError("plan_count must match plans")
        if self.plan_roles != tuple(plan.role for plan in self.plans):
            raise ValueError("plan_roles must match plans")
        if self.plan_serialization_versions != tuple(
            plan.serialization_version for plan in self.plans
        ):
            raise ValueError("plan_serialization_versions must match plans")
        roadmap_items = tuple(
            roadmap_item
            for plan in self.plans
            for roadmap_item in plan.covered_roadmap_items
        )
        if self.covered_roadmap_items != roadmap_items:
            raise ValueError("covered_roadmap_items must match plans")
        if self.covered_roadmap_items != CORE_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.5 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if any(
            plan.upstream_signal_source_ids != self.upstream_signal_source_ids
            for plan in self.plans
        ):
            raise ValueError("upstream_signal_source_ids must match plan sources")
        if self.upstream_signal_source_count != len(self.upstream_signal_source_ids):
            raise ValueError("upstream_signal_source_count must match sources")
        if any(
            plan.upstream_signal_ids != self.upstream_signal_ids for plan in self.plans
        ):
            raise ValueError("upstream_signal_ids must match plan signals")
        if self.upstream_signal_id_count != len(self.upstream_signal_ids):
            raise ValueError("upstream_signal_id_count must match signals")
        proposals = tuple(
            proposal for plan in self.plans for proposal in plan.proposals
        )
        if self.proposals != proposals:
            raise ValueError("proposals must match plans")
        proposal_ids = tuple(proposal.proposal_id for proposal in self.proposals)
        if self.proposal_ids != proposal_ids:
            raise ValueError("proposal_ids must match proposals")
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("proposal_ids must be unique")
        if self.proposal_count != len(self.proposals):
            raise ValueError("proposal_count must match proposals")
        if self.candidate_proposal_ids != proposal_ids_for_status(
            self.proposals,
            "candidate",
        ):
            raise ValueError("candidate_proposal_ids must match proposals")
        if self.review_required_proposal_ids != proposal_ids_for_status(
            self.proposals,
            "review_required",
        ):
            raise ValueError("review_required_proposal_ids must match proposals")
        if self.guarded_proposal_ids != proposal_ids_for_status(
            self.proposals,
            "guarded",
        ):
            raise ValueError("guarded_proposal_ids must match proposals")
        if self.high_confidence_proposal_ids != proposals_for_confidence_ids(
            self.proposals,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_proposal_ids must match proposals")
        if self.hitl_required_proposal_ids != tuple(
            proposal.proposal_id
            for proposal in self.proposals
            if proposal.hitl_required_before_application
        ):
            raise ValueError("hitl_required_proposal_ids must match proposals")
        empty_fields = (
            self.generated_evolution_report_ids,
            self.applied_evolution_proposal_ids,
            self.mutated_prompt_ids,
            self.mutated_workflow_ids,
            self.mutated_routing_ids,
            self.mutated_memory_ids,
            self.mutated_retrieval_ids,
            self.written_storage_record_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("core surface mutation ids must be empty")
        if self.review_required_proposal_count != len(
            self.review_required_proposal_ids
        ):
            raise ValueError("review_required_proposal_count must match proposals")
        if self.guarded_proposal_count != len(self.guarded_proposal_ids):
            raise ValueError("guarded_proposal_count must match proposals")
        if self.high_confidence_proposal_count != len(
            self.high_confidence_proposal_ids
        ):
            raise ValueError("high_confidence_proposal_count must match proposals")
        if self.hitl_required_proposal_count != len(self.hitl_required_proposal_ids):
            raise ValueError("hitl_required_proposal_count must match proposals")
        if self.highest_proposal_rank_score != max(
            proposal.proposal_rank_score for proposal in self.proposals
        ):
            raise ValueError("highest_proposal_rank_score must match proposals")
        if self.overall_proposal_rank_score != overall_proposal_rank_score(
            self.proposals
        ):
            raise ValueError("overall_proposal_rank_score must match proposals")
        if self.overall_evolution_posture != overall_evolution_posture(self.proposals):
            raise ValueError("overall_evolution_posture must match proposals")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.5 boundary")
        if any(not plan.advisory_only for plan in self.plans):
            raise ValueError("all plans must be advisory only")
        return self


def build_self_evolution_core_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> SelfEvolutionCoreSurfacePlan:
    """Aggregate all explicit V6.5 roadmap plans without applying proposals."""

    plans = tuple(
        builder(
            route=route,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        )
        for builder in _PLAN_BUILDERS
    )
    proposals = tuple(proposal for plan in plans for proposal in plan.proposals)
    return SelfEvolutionCoreSurfacePlan(
        plans=plans,
        plan_count=len(plans),
        plan_roles=tuple(plan.role for plan in plans),
        plan_serialization_versions=tuple(plan.serialization_version for plan in plans),
        covered_roadmap_items=tuple(
            roadmap_item
            for plan in plans
            for roadmap_item in plan.covered_roadmap_items
        ),
        covered_roadmap_item_count=sum(
            plan.covered_roadmap_item_count for plan in plans
        ),
        upstream_capabilities=UPSTREAM_CAPABILITIES,
        upstream_signal_source_ids=plans[0].upstream_signal_source_ids,
        upstream_signal_source_count=plans[0].upstream_signal_source_count,
        upstream_signal_ids=plans[0].upstream_signal_ids,
        upstream_signal_id_count=plans[0].upstream_signal_id_count,
        proposals=proposals,
        proposal_ids=tuple(proposal.proposal_id for proposal in proposals),
        proposal_count=len(proposals),
        candidate_proposal_ids=proposal_ids_for_status(proposals, "candidate"),
        review_required_proposal_ids=proposal_ids_for_status(
            proposals,
            "review_required",
        ),
        guarded_proposal_ids=proposal_ids_for_status(proposals, "guarded"),
        high_confidence_proposal_ids=proposals_for_confidence_ids(
            proposals,
            "high",
            "guarded",
        ),
        hitl_required_proposal_ids=tuple(
            proposal.proposal_id
            for proposal in proposals
            if proposal.hitl_required_before_application
        ),
        candidate_proposal_count=len(proposal_ids_for_status(proposals, "candidate")),
        review_required_proposal_count=len(
            proposal_ids_for_status(proposals, "review_required")
        ),
        guarded_proposal_count=len(proposal_ids_for_status(proposals, "guarded")),
        high_confidence_proposal_count=len(
            proposals_for_confidence_ids(proposals, "high", "guarded")
        ),
        hitl_required_proposal_count=sum(
            1 for proposal in proposals if proposal.hitl_required_before_application
        ),
        highest_proposal_rank_score=max(
            proposal.proposal_rank_score for proposal in proposals
        ),
        overall_proposal_rank_score=overall_proposal_rank_score(proposals),
        overall_evolution_posture=overall_evolution_posture(proposals),
        cross_cutting_contracts=CROSS_CUTTING_CONTRACTS,
        blocked_runtime_behaviors=BLOCKED_RUNTIME_BEHAVIORS,
    )


def self_evolution_core_surface_plan_by_roadmap_item(
    roadmap_item: str,
    surface: SelfEvolutionCoreSurfacePlan | None = None,
) -> SelfEvolutionPlan | None:
    """Return one roadmap plan from the core surface without applying it."""

    source_surface = surface or build_self_evolution_core_surface()
    for plan in source_surface.plans:
        if roadmap_item in plan.covered_roadmap_items:
            return plan
    return None


def self_evolution_core_surface_proposal_by_id(
    proposal_id: str,
    surface: SelfEvolutionCoreSurfacePlan | None = None,
) -> SelfEvolutionProposal | None:
    """Return one proposal from the core surface without applying it."""

    source_surface = surface or build_self_evolution_core_surface()
    for proposal in source_surface.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    return None


def self_evolution_core_surface_proposals_for_status(
    status: SelfEvolutionStatus,
    surface: SelfEvolutionCoreSurfacePlan | None = None,
) -> tuple[SelfEvolutionProposal, ...]:
    """Return core-surface proposals by advisory status."""

    source_surface = surface or build_self_evolution_core_surface()
    return tuple(
        proposal for proposal in source_surface.proposals if proposal.status == status
    )


def self_evolution_core_surface_proposals_for_confidence(
    confidence: SelfEvolutionConfidence,
    surface: SelfEvolutionCoreSurfacePlan | None = None,
) -> tuple[SelfEvolutionProposal, ...]:
    """Return core-surface proposals by confidence band."""

    source_surface = surface or build_self_evolution_core_surface()
    return tuple(
        proposal
        for proposal in source_surface.proposals
        if proposal.confidence == confidence
    )
