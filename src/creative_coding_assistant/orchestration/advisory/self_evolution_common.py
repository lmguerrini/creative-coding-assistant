"""Shared V6.5 advisory self-evolution proposal metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.creative_memory_core_surface import (
    build_creative_memory_core_surface,
)
from creative_coding_assistant.orchestration.knowledge_evolution_core_surface import (
    build_knowledge_evolution_core_surface,
)
from creative_coding_assistant.orchestration.research_core_surface import (
    build_research_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

SelfEvolutionStatus = Literal["candidate", "review_required", "guarded"]
SelfEvolutionConfidence = Literal["low", "medium", "high", "guarded"]
SelfEvolutionPosture = Literal["candidate", "review_required", "guarded"]

SELF_EVOLUTION_AUTHORITY_BOUNDARY = (
    "V6.5 Self Evolution Engine exposes cross-capability evolution proposals "
    "as inspectable advisory metadata only. It may read deterministic signals "
    "from V6.1 Adaptive Learning, V6.2 Creative Memory, V6.3 Knowledge "
    "Evolution, and V6.4 Autonomous Research, compare opportunities, rank "
    "proposals, explain upstream signals and downstream impacts, and prepare "
    "advisory evolution reports; it does not apply Runtime Evolution, rewrite "
    "prompts, mutate workflows, mutate routing, mutate memory, mutate "
    "retrieval, write storage, execute providers, invoke agents, modify "
    "generated output, or silently self-modify code."
)

CROSS_CUTTING_CONTRACTS = (
    "Cross-Capability Dependency Awareness",
    "Evolution Graph / Dependency Graph",
    "Evolution Proposal Contract",
    "Evolution Explainability Report",
    "Proposal Impact Model",
    "Cost / Benefit Model",
    "Risk Model",
    "Rollback Strategy Model",
    "Capability Ownership Boundary Check",
    "Cross-Capability Governance Check",
)

UPSTREAM_CAPABILITIES = (
    "V6.1 Adaptive Learning",
    "V6.2 Creative Memory",
    "V6.3 Knowledge Evolution",
    "V6.4 Autonomous Research",
)

BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_evolution_application",
    "autonomous_code_mutation",
    "prompt_rewriting",
    "workflow_mutation",
    "routing_mutation",
    "memory_mutation",
    "retrieval_mutation",
    "storage_write",
    "provider_execution",
    "agent_invocation",
    "generated_output_mutation",
    "hitl_decision_application",
)


class UpstreamEvolutionSignalSource(BaseModel):
    """One upstream V6 signal source read by V6.5 without applying behavior."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=120)
    capability: str = Field(min_length=1, max_length=120)
    source_role: str = Field(min_length=1, max_length=120)
    serialization_version: str = Field(min_length=1, max_length=120)
    signal_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    signal_count: int = Field(ge=1, le=40)
    signal_summary: str = Field(min_length=1, max_length=360)
    ownership_boundary: str = Field(min_length=1, max_length=360)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _source_matches_signals(self) -> Self:
        if self.signal_count != len(self.signal_ids):
            raise ValueError("signal_count must match signal_ids")
        return self


class SelfEvolutionProposal(BaseModel):
    """One advisory evolution proposal with scoring and explanation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1, max_length=180)
    surface_role: str = Field(min_length=1, max_length=120)
    proposal_kind: str = Field(min_length=1, max_length=120)
    roadmap_item: str = Field(min_length=1, max_length=120)
    status: SelfEvolutionStatus
    confidence: SelfEvolutionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    orchestration_axis: str = Field(min_length=1, max_length=120)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    downstream_systems: tuple[str, ...] = Field(min_length=1, max_length=8)
    impact_score: int = Field(ge=0, le=100)
    cost_score: int = Field(ge=0, le=100)
    risk_score: int = Field(ge=0, le=100)
    confidence_score: int = Field(ge=0, le=100)
    dependency_score: int = Field(ge=0, le=100)
    rollback_feasibility_score: int = Field(ge=0, le=100)
    proposal_rank_score: int = Field(ge=0, le=1_000)
    why_proposal_exists: str = Field(min_length=1, max_length=420)
    upstream_signal_explanation: str = Field(min_length=1, max_length=420)
    downstream_impact_explanation: str = Field(min_length=1, max_length=420)
    evolution_report_sections: tuple[str, ...] = Field(min_length=4, max_length=8)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    ownership_boundary_checks: tuple[str, ...] = Field(min_length=4, max_length=8)
    governance_checks: tuple[str, ...] = Field(min_length=4, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    hitl_required_before_application: Literal[True] = True
    self_evolution_capability_implemented: Literal[True] = True
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
    serialization_version: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def _proposal_matches_contract(self) -> Self:
        if self.proposal_id != f"{self.surface_role}::{self.proposal_kind}":
            raise ValueError("proposal_id must match surface_role and proposal_kind")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.5 boundary")
        if self.proposal_rank_score != proposal_rank_score(
            impact_score=self.impact_score,
            cost_score=self.cost_score,
            risk_score=self.risk_score,
            confidence_score=self.confidence_score,
            dependency_score=self.dependency_score,
            rollback_feasibility_score=self.rollback_feasibility_score,
        ):
            raise ValueError("proposal_rank_score must match score model")
        if self.status != proposal_status(self.proposal_rank_score):
            raise ValueError("status must match proposal_rank_score")
        if self.confidence != proposal_confidence(self.proposal_rank_score):
            raise ValueError("confidence must match proposal_rank_score")
        return self


class SelfEvolutionPlan(BaseModel):
    """One V6.5 advisory self-evolution plan for a roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: str = Field(min_length=1, max_length=120)
    serialization_version: str = Field(min_length=1, max_length=120)
    authority_boundary: str = Field(
        default=SELF_EVOLUTION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    upstream_signal_sources: tuple[UpstreamEvolutionSignalSource, ...] = Field(
        min_length=4,
        max_length=4,
    )
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_count: int = Field(ge=4, le=4)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_ids: tuple[str, ...] = Field(min_length=4, max_length=160)
    upstream_signal_id_count: int = Field(ge=4, le=160)
    source_plan_roles: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    proposals: tuple[SelfEvolutionProposal, ...] = Field(min_length=5, max_length=5)
    proposal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    proposal_count: int = Field(ge=5, le=5)
    candidate_proposal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_proposal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_evolution_report_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_routing_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_proposal_count: int = Field(ge=0, le=5)
    review_required_proposal_count: int = Field(ge=0, le=5)
    guarded_proposal_count: int = Field(ge=0, le=5)
    high_confidence_proposal_count: int = Field(ge=0, le=5)
    hitl_required_proposal_count: int = Field(ge=0, le=5)
    highest_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_evolution_posture: SelfEvolutionPosture
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    self_evolution_capability_implemented: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    all_v6_signal_sources_integrated: Literal[True] = True
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
    def _plan_matches_proposals(self) -> Self:
        proposal_ids = tuple(proposal.proposal_id for proposal in self.proposals)
        if self.proposal_ids != proposal_ids:
            raise ValueError("proposal_ids must match proposals")
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("proposal_ids must be unique")
        if self.proposal_count != len(self.proposals):
            raise ValueError("proposal_count must match proposals")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.upstream_signal_source_ids != tuple(
            source.source_id for source in self.upstream_signal_sources
        ):
            raise ValueError("upstream_signal_source_ids must match sources")
        if self.upstream_signal_source_count != len(self.upstream_signal_sources):
            raise ValueError("upstream_signal_source_count must match sources")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.upstream_signal_ids != tuple(
            signal_id
            for source in self.upstream_signal_sources
            for signal_id in source.signal_ids
        ):
            raise ValueError("upstream_signal_ids must match sources")
        if self.upstream_signal_id_count != len(self.upstream_signal_ids):
            raise ValueError("upstream_signal_id_count must match source signals")
        if self.source_plan_roles != tuple(
            source.source_role for source in self.upstream_signal_sources
        ):
            raise ValueError("source_plan_roles must match sources")
        if self.source_plan_serialization_versions != tuple(
            source.serialization_version for source in self.upstream_signal_sources
        ):
            raise ValueError("source_plan_serialization_versions must match sources")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.5 boundary")
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
            raise ValueError("runtime mutation and generated report ids must be empty")
        if self.candidate_proposal_count != len(self.candidate_proposal_ids):
            raise ValueError("candidate_proposal_count must match proposals")
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
        for proposal in self.proposals:
            if proposal.route_name != self.route_name:
                raise ValueError("proposal route_name must match plan")
            if proposal.task_type != self.task_type:
                raise ValueError("proposal task_type must match plan")
            if proposal.surface_role != self.role:
                raise ValueError("proposal surface_role must match plan")
            if proposal.roadmap_item not in self.covered_roadmap_items:
                raise ValueError("proposal roadmap_item must be covered")
        return self


def build_self_evolution_plan(
    *,
    role: str,
    roadmap_item: str,
    proposal_kinds: tuple[str, str, str, str, str],
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    downstream_systems: tuple[str, ...],
) -> SelfEvolutionPlan:
    """Build one V6.5 advisory evolution plan without applying proposals."""

    route_name = resolve_route(route)
    normalized_task_type = resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    sources = upstream_signal_sources(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    proposals = build_proposals(
        role=role,
        roadmap_item=roadmap_item,
        proposal_kinds=proposal_kinds,
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        upstream_signal_source_ids=tuple(source.source_id for source in sources),
        downstream_systems=downstream_systems,
    )
    return SelfEvolutionPlan(
        role=role,
        serialization_version=f"{role}_plan.v1",
        route_name=route_name,
        task_type=normalized_task_type,
        covered_roadmap_items=(roadmap_item,),
        covered_roadmap_item_count=1,
        upstream_signal_sources=sources,
        upstream_signal_source_ids=tuple(source.source_id for source in sources),
        upstream_signal_source_count=len(sources),
        upstream_capabilities=UPSTREAM_CAPABILITIES,
        upstream_signal_ids=tuple(
            signal_id for source in sources for signal_id in source.signal_ids
        ),
        upstream_signal_id_count=sum(source.signal_count for source in sources),
        source_plan_roles=tuple(source.source_role for source in sources),
        source_plan_serialization_versions=tuple(
            source.serialization_version for source in sources
        ),
        execution_mode_ids=execution_modes.execution_mode_ids,
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
        advisory_actions=plan_advisory_actions(role=role, roadmap_item=roadmap_item),
        blocked_runtime_behaviors=BLOCKED_RUNTIME_BEHAVIORS,
    )


def upstream_signal_sources(
    *,
    route: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> tuple[UpstreamEvolutionSignalSource, ...]:
    """Read V6 upstream metadata sources without activating their behavior."""

    adaptive = evaluate_adaptive_learning_engine(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    creative = build_creative_memory_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    knowledge = build_knowledge_evolution_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    research = build_research_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    return (
        UpstreamEvolutionSignalSource(
            source_id="v6_1_adaptive_learning",
            capability="V6.1 Adaptive Learning",
            source_role=str(adaptive.role),
            serialization_version=str(adaptive.serialization_version),
            signal_ids=tuple(str(signal_id) for signal_id in adaptive.signal_ids),
            signal_count=len(adaptive.signal_ids),
            signal_summary="Adaptive learning priority and guardrail signals.",
            ownership_boundary="V6.5 reads learning metadata only; V6.1 owns learning.",
        ),
        UpstreamEvolutionSignalSource(
            source_id="v6_2_creative_memory",
            capability="V6.2 Creative Memory",
            source_role=str(creative.role),
            serialization_version=str(creative.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in creative.entry_ids),
            signal_count=len(creative.entry_ids),
            signal_summary="Creative memory core surface traceability signals.",
            ownership_boundary="V6.5 reads memory metadata only; V6.2 owns memory.",
        ),
        UpstreamEvolutionSignalSource(
            source_id="v6_3_knowledge_evolution",
            capability="V6.3 Knowledge Evolution",
            source_role=str(knowledge.role),
            serialization_version=str(knowledge.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in knowledge.entry_ids),
            signal_count=len(knowledge.entry_ids),
            signal_summary="Knowledge evolution core surface health signals.",
            ownership_boundary="V6.5 reads knowledge metadata only; V6.3 owns KB.",
        ),
        UpstreamEvolutionSignalSource(
            source_id="v6_4_autonomous_research",
            capability="V6.4 Autonomous Research",
            source_role=str(research.role),
            serialization_version=str(research.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in research.entry_ids),
            signal_count=len(research.entry_ids),
            signal_summary="Autonomous research core surface governance signals.",
            ownership_boundary="V6.5 reads research metadata only; V6.4 owns research.",
        ),
    )


def build_proposals(
    *,
    role: str,
    roadmap_item: str,
    proposal_kinds: tuple[str, str, str, str, str],
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    upstream_signal_source_ids: tuple[str, ...],
    downstream_systems: tuple[str, ...],
) -> tuple[SelfEvolutionProposal, ...]:
    """Build five bounded advisory proposals for one roadmap item."""

    score_inputs = (
        (92, 42, 38, 88, 82, 84),
        (80, 36, 34, 84, 76, 80),
        (70, 50, 44, 78, 70, 72),
        (62, 58, 52, 72, 66, 68),
        (48, 64, 62, 66, 60, 74),
    )
    return tuple(
        build_proposal(
            role=role,
            roadmap_item=roadmap_item,
            proposal_kind=proposal_kind,
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            upstream_signal_source_ids=upstream_signal_source_ids,
            downstream_systems=downstream_systems,
            impact_score=impact_score,
            cost_score=cost_score,
            risk_score=risk_score,
            confidence_score=confidence_score,
            dependency_score=dependency_score,
            rollback_feasibility_score=rollback_score,
        )
        for proposal_kind, (
            impact_score,
            cost_score,
            risk_score,
            confidence_score,
            dependency_score,
            rollback_score,
        ) in zip(proposal_kinds, score_inputs, strict=True)
    )


def build_proposal(
    *,
    role: str,
    roadmap_item: str,
    proposal_kind: str,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    upstream_signal_source_ids: tuple[str, ...],
    downstream_systems: tuple[str, ...],
    impact_score: int,
    cost_score: int,
    risk_score: int,
    confidence_score: int,
    dependency_score: int,
    rollback_feasibility_score: int,
) -> SelfEvolutionProposal:
    """Build one advisory proposal with deterministic scoring."""

    score = proposal_rank_score(
        impact_score=impact_score,
        cost_score=cost_score,
        risk_score=risk_score,
        confidence_score=confidence_score,
        dependency_score=dependency_score,
        rollback_feasibility_score=rollback_feasibility_score,
    )
    return SelfEvolutionProposal(
        proposal_id=f"{role}::{proposal_kind}",
        surface_role=role,
        proposal_kind=proposal_kind,
        roadmap_item=roadmap_item,
        status=proposal_status(score),
        confidence=proposal_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        orchestration_axis=role,
        upstream_signal_source_ids=upstream_signal_source_ids,
        upstream_capabilities=UPSTREAM_CAPABILITIES,
        downstream_systems=downstream_systems,
        impact_score=impact_score,
        cost_score=cost_score,
        risk_score=risk_score,
        confidence_score=confidence_score,
        dependency_score=dependency_score,
        rollback_feasibility_score=rollback_feasibility_score,
        proposal_rank_score=score,
        why_proposal_exists=(
            f"{roadmap_item} proposal {proposal_kind} exists because V6.5 "
            "compares upstream V6 signals before suggesting evolution."
        ),
        upstream_signal_explanation=(
            "Signals are read from V6.1 learning, V6.2 memory, V6.3 knowledge, "
            "and V6.4 research metadata without applying their behavior."
        ),
        downstream_impact_explanation=(
            "Downstream impact is advisory and limited to the listed systems: "
            + ", ".join(downstream_systems)
            + "."
        ),
        evolution_report_sections=(
            "upstream_signals",
            "proposal_ranking",
            "impact_cost_risk_confidence",
            "rollback_feasibility",
            "ownership_and_governance",
        ),
        cross_cutting_contracts=CROSS_CUTTING_CONTRACTS,
        ownership_boundary_checks=ownership_boundary_checks(roadmap_item),
        governance_checks=governance_checks(roadmap_item),
        advisory_actions=(
            "prepare_advisory_evolution_report",
            "route_to_human_review_before_application",
            "preserve_capability_ownership_boundaries",
        ),
        evidence=(
            f"roadmap_item:{roadmap_item}",
            f"proposal_kind:{proposal_kind}",
            f"impact_score:{impact_score}",
            f"cost_score:{cost_score}",
            f"risk_score:{risk_score}",
            f"confidence_score:{confidence_score}",
            f"dependency_score:{dependency_score}",
            f"rollback_feasibility_score:{rollback_feasibility_score}",
            f"proposal_rank_score:{score}",
            "hitl_required_before_application:true",
        ),
        blocked_runtime_behaviors=BLOCKED_RUNTIME_BEHAVIORS,
        serialization_version=f"{role}_proposal.v1",
    )


def proposal_rank_score(
    *,
    impact_score: int,
    cost_score: int,
    risk_score: int,
    confidence_score: int,
    dependency_score: int,
    rollback_feasibility_score: int,
) -> int:
    """Rank advisory proposals without executing them."""

    return min(
        1_000,
        max(
            0,
            impact_score * 3
            + confidence_score * 2
            + rollback_feasibility_score * 2
            + dependency_score * 2
            + (100 - cost_score)
            + (100 - risk_score) * 2,
        ),
    )


def proposal_status(score: int) -> SelfEvolutionStatus:
    """Classify proposal review posture."""

    if score >= 820:
        return "guarded"
    if score >= 620:
        return "review_required"
    return "candidate"


def proposal_confidence(score: int) -> SelfEvolutionConfidence:
    """Classify proposal confidence band."""

    if score >= 820:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def proposal_ids_for_status(
    proposals: tuple[SelfEvolutionProposal, ...],
    status: SelfEvolutionStatus,
) -> tuple[str, ...]:
    """Return proposal ids for a status."""

    return tuple(
        proposal.proposal_id for proposal in proposals if proposal.status == status
    )


def proposals_for_confidence_ids(
    proposals: tuple[SelfEvolutionProposal, ...],
    *confidences: SelfEvolutionConfidence,
) -> tuple[str, ...]:
    """Return proposal ids for confidence bands."""

    return tuple(
        proposal.proposal_id
        for proposal in proposals
        if proposal.confidence in confidences
    )


def overall_proposal_rank_score(
    proposals: tuple[SelfEvolutionProposal, ...],
) -> int:
    """Calculate the overall advisory proposal score."""

    base = sum(proposal.proposal_rank_score for proposal in proposals) // len(proposals)
    guarded_count = len(proposal_ids_for_status(proposals, "guarded"))
    review_count = len(proposal_ids_for_status(proposals, "review_required"))
    return min(1_000, base + guarded_count * 15 + review_count * 8)


def overall_evolution_posture(
    proposals: tuple[SelfEvolutionProposal, ...],
) -> SelfEvolutionPosture:
    """Calculate the overall evolution posture."""

    if any(proposal.status == "guarded" for proposal in proposals):
        return "guarded"
    if any(proposal.status == "review_required" for proposal in proposals):
        return "review_required"
    return "candidate"


def plan_advisory_actions(*, role: str, roadmap_item: str) -> tuple[str, ...]:
    """Describe plan-level advisory actions."""

    return (
        f"compare_{role}_signals_across_v6_capabilities",
        f"rank_{role}_proposals_for_human_review",
        f"prepare_{roadmap_item}_advisory_evolution_report",
        "preserve_human_controlled_application_boundary",
    )


def ownership_boundary_checks(roadmap_item: str) -> tuple[str, ...]:
    """Return ownership checks applied to every V6.5 proposal."""

    return (
        f"{roadmap_item}: V6.1 learning remains source-owned",
        f"{roadmap_item}: V6.2 memory remains source-owned",
        f"{roadmap_item}: V6.3 knowledge remains source-owned",
        f"{roadmap_item}: V6.4 research remains source-owned",
    )


def governance_checks(roadmap_item: str) -> tuple[str, ...]:
    """Return governance checks applied to every V6.5 proposal."""

    return (
        f"{roadmap_item}: advisory-only proposal",
        f"{roadmap_item}: HITL required before application",
        f"{roadmap_item}: no autonomous mutation",
        f"{roadmap_item}: no provider execution",
    )


def resolve_route(route: RouteName | str) -> RouteName:
    """Resolve a route string to a known RouteName."""

    if isinstance(route, RouteName):
        return route
    try:
        return RouteName(str(route).strip())
    except ValueError as exc:
        raise ValueError("route must be a known RouteName") from exc


def resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    """Resolve a task type string to a known TaskRoutingType."""

    normalized = str(task_type).strip()
    allowed = set(get_args(TaskRoutingType))
    if normalized not in allowed:
        raise ValueError("task_type must be a known TaskRoutingType")
    return cast(TaskRoutingType, normalized)


def resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    """Resolve an execution mode string to a known execution mode."""

    normalized = str(execution_mode_id).strip()
    if normalized not in set(allowed_modes):
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
