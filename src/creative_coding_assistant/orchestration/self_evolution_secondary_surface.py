"""V6.5 advisory secondary surface for self-evolution report metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    BLOCKED_RUNTIME_BEHAVIORS,
    CROSS_CUTTING_CONTRACTS,
    SELF_EVOLUTION_AUTHORITY_BOUNDARY,
    UPSTREAM_CAPABILITIES,
    SelfEvolutionConfidence,
    SelfEvolutionPlan,
    SelfEvolutionPosture,
    SelfEvolutionStatus,
    overall_evolution_posture,
    overall_proposal_rank_score,
    proposal_ids_for_status,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
    SelfEvolutionCoreSurfacePlan,
    build_self_evolution_core_surface,
)

SECONDARY_REPORT_SECTIONS = (
    "roadmap_traceability",
    "upstream_signal_sources",
    "proposal_ranking",
    "impact_cost_risk_confidence",
    "rollback_feasibility",
    "ownership_and_governance",
)

SELF_EVOLUTION_SECONDARY_SURFACE_AUTHORITY_BOUNDARY = (
    SELF_EVOLUTION_AUTHORITY_BOUNDARY
    + " The secondary surface composes advisory report metadata from the Task "
    "24 core surface only; it does not generate report artifacts, write "
    "storage, apply proposals, execute rollback, or mutate any upstream or "
    "downstream capability surface."
)


class SelfEvolutionAdvisoryReportEntry(BaseModel):
    """One advisory report metadata entry derived from a core roadmap plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    report_id: str = Field(min_length=1, max_length=220)
    source_core_surface_role: Literal["self_evolution_core_surface"] = (
        "self_evolution_core_surface"
    )
    source_core_surface_serialization_version: Literal[
        "self_evolution_core_surface.v1"
    ] = "self_evolution_core_surface.v1"
    roadmap_item: str = Field(min_length=1, max_length=120)
    plan_role: str = Field(min_length=1, max_length=120)
    plan_serialization_version: str = Field(min_length=1, max_length=120)
    status: SelfEvolutionStatus
    confidence: SelfEvolutionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    proposal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    proposal_count: int = Field(ge=5, le=5)
    proposal_rank_scores: tuple[int, ...] = Field(min_length=5, max_length=5)
    guarded_proposal_ids: tuple[str, ...] = Field(max_length=5)
    guarded_proposal_count: int = Field(ge=0, le=5)
    review_required_proposal_ids: tuple[str, ...] = Field(max_length=5)
    review_required_proposal_count: int = Field(ge=0, le=5)
    hitl_required_proposal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    hitl_required_proposal_count: int = Field(ge=5, le=5)
    top_proposal_id: str = Field(min_length=1, max_length=180)
    top_proposal_rank_score: int = Field(ge=0, le=1_000)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    downstream_systems: tuple[str, ...] = Field(min_length=1, max_length=24)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    why_report_exists: str = Field(min_length=1, max_length=520)
    upstream_signal_explanation: str = Field(min_length=1, max_length=520)
    downstream_impact_explanation: str = Field(min_length=1, max_length=520)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    ownership_boundary_checks: tuple[str, ...] = Field(min_length=4, max_length=8)
    governance_checks: tuple[str, ...] = Field(min_length=4, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_rollback_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_prompt_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_routing_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_memory_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    provider_execution_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    secondary_surface_implemented: Literal[True] = True
    advisory_evolution_report_metadata_implemented: Literal[True] = True
    roadmap_traceability_implemented: Literal[True] = True
    all_v6_signal_sources_integrated: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    prompt_rewriting_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["self_evolution_advisory_report_entry.v1"] = (
        "self_evolution_advisory_report_entry.v1"
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_report_contract(self) -> Self:
        if self.report_id != f"self_evolution_advisory_report::{self.plan_role}":
            raise ValueError("report_id must match plan_role")
        if self.roadmap_item not in CORE_ROADMAP_ITEMS:
            raise ValueError("roadmap_item must be a V6.5 roadmap item")
        if self.proposal_count != len(self.proposal_ids):
            raise ValueError("proposal_count must match proposal_ids")
        if self.guarded_proposal_count != len(self.guarded_proposal_ids):
            raise ValueError("guarded_proposal_count must match proposals")
        if self.review_required_proposal_count != len(
            self.review_required_proposal_ids
        ):
            raise ValueError("review_required_proposal_count must match proposals")
        if self.hitl_required_proposal_count != len(self.hitl_required_proposal_ids):
            raise ValueError("hitl_required_proposal_count must match proposals")
        if self.top_proposal_id not in self.proposal_ids:
            raise ValueError("top_proposal_id must be one of proposal_ids")
        if self.top_proposal_rank_score != max(self.proposal_rank_scores):
            raise ValueError("top_proposal_rank_score must match proposal scores")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.report_sections != SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match V6.5 report contract")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.5 boundary")
        empty_fields = (
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.applied_evolution_proposal_ids,
            self.executed_rollback_ids,
            self.mutated_prompt_ids,
            self.mutated_workflow_ids,
            self.mutated_routing_ids,
            self.mutated_memory_ids,
            self.mutated_retrieval_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("secondary report mutation ids must be empty")
        return self


class SelfEvolutionSecondarySurfacePlan(BaseModel):
    """Secondary advisory report surface over the V6.5 core surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["self_evolution_secondary_surface"] = (
        "self_evolution_secondary_surface"
    )
    serialization_version: Literal["self_evolution_secondary_surface.v1"] = (
        "self_evolution_secondary_surface.v1"
    )
    authority_boundary: str = Field(
        default=SELF_EVOLUTION_SECONDARY_SURFACE_AUTHORITY_BOUNDARY,
        max_length=3000,
    )
    source_core_surface_role: Literal["self_evolution_core_surface"] = (
        "self_evolution_core_surface"
    )
    source_core_surface_serialization_version: Literal[
        "self_evolution_core_surface.v1"
    ] = "self_evolution_core_surface.v1"
    source_core_surface_plan_count: int = Field(ge=22, le=22)
    source_core_surface_proposal_count: int = Field(ge=110, le=110)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    covered_roadmap_items: tuple[str, ...] = Field(min_length=22, max_length=22)
    covered_roadmap_item_count: int = Field(ge=22, le=22)
    upstream_capabilities: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    upstream_signal_source_count: int = Field(ge=4, le=4)
    upstream_signal_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    upstream_signal_id_count: int = Field(ge=20, le=20)
    report_entries: tuple[SelfEvolutionAdvisoryReportEntry, ...] = Field(
        min_length=22,
        max_length=22,
    )
    report_entry_ids: tuple[str, ...] = Field(min_length=22, max_length=22)
    report_entry_count: int = Field(ge=22, le=22)
    proposal_ids: tuple[str, ...] = Field(min_length=110, max_length=110)
    proposal_count: int = Field(ge=110, le=110)
    guarded_proposal_ids: tuple[str, ...] = Field(max_length=110)
    guarded_proposal_count: int = Field(ge=0, le=110)
    review_required_proposal_ids: tuple[str, ...] = Field(max_length=110)
    review_required_proposal_count: int = Field(ge=0, le=110)
    hitl_required_proposal_ids: tuple[str, ...] = Field(min_length=110, max_length=110)
    hitl_required_proposal_count: int = Field(ge=110, le=110)
    high_confidence_proposal_ids: tuple[str, ...] = Field(max_length=110)
    high_confidence_proposal_count: int = Field(ge=0, le=110)
    top_report_entry_id: str = Field(min_length=1, max_length=220)
    top_proposal_id: str = Field(min_length=1, max_length=180)
    top_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_proposal_rank_score: int = Field(ge=0, le=1_000)
    overall_evolution_posture: SelfEvolutionPosture
    all_downstream_systems: tuple[str, ...] = Field(min_length=1, max_length=120)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    generated_report_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    applied_evolution_proposal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    executed_rollback_ids: tuple[str, ...] = Field(
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
    provider_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=110,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=110)
    secondary_surface_implemented: Literal[True] = True
    advisory_evolution_report_metadata_implemented: Literal[True] = True
    roadmap_traceability_implemented: Literal[True] = True
    all_v6_signal_sources_integrated: Literal[True] = True
    evolution_proposal_contract_implemented: Literal[True] = True
    evolution_graph_metadata_implemented: Literal[True] = True
    evolution_explainability_report_implemented: Literal[True] = True
    proposal_impact_model_implemented: Literal[True] = True
    cost_benefit_model_implemented: Literal[True] = True
    risk_model_implemented: Literal[True] = True
    rollback_strategy_model_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    cross_capability_governance_check_implemented: Literal[True] = True
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    proposal_application_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    prompt_rewriting_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _secondary_surface_matches_reports(self) -> Self:
        if self.report_entry_ids != tuple(
            entry.report_id for entry in self.report_entries
        ):
            raise ValueError("report_entry_ids must match report_entries")
        if len(set(self.report_entry_ids)) != len(self.report_entry_ids):
            raise ValueError("report_entry_ids must be unique")
        if self.report_entry_count != len(self.report_entries):
            raise ValueError("report_entry_count must match report_entries")
        if self.covered_roadmap_items != tuple(
            entry.roadmap_item for entry in self.report_entries
        ):
            raise ValueError("covered_roadmap_items must match reports")
        if self.covered_roadmap_items != CORE_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.5 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.proposal_ids != tuple(
            proposal_id
            for entry in self.report_entries
            for proposal_id in entry.proposal_ids
        ):
            raise ValueError("proposal_ids must match report_entries")
        if len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError("proposal_ids must be unique")
        if self.proposal_count != len(self.proposal_ids):
            raise ValueError("proposal_count must match proposal_ids")
        if self.guarded_proposal_count != len(self.guarded_proposal_ids):
            raise ValueError("guarded_proposal_count must match proposals")
        if self.review_required_proposal_count != len(
            self.review_required_proposal_ids
        ):
            raise ValueError("review_required_proposal_count must match proposals")
        if self.hitl_required_proposal_count != len(self.hitl_required_proposal_ids):
            raise ValueError("hitl_required_proposal_count must match proposals")
        if self.high_confidence_proposal_count != len(
            self.high_confidence_proposal_ids
        ):
            raise ValueError("high_confidence_proposal_count must match proposals")
        if self.top_report_entry_id not in self.report_entry_ids:
            raise ValueError("top_report_entry_id must match report_entries")
        if self.top_proposal_id not in self.proposal_ids:
            raise ValueError("top_proposal_id must match proposal_ids")
        if self.top_proposal_rank_score != max(
            entry.top_proposal_rank_score for entry in self.report_entries
        ):
            raise ValueError("top_proposal_rank_score must match reports")
        if self.upstream_capabilities != UPSTREAM_CAPABILITIES:
            raise ValueError("upstream_capabilities must include V6.1 through V6.4")
        if self.upstream_signal_source_count != len(self.upstream_signal_source_ids):
            raise ValueError("upstream_signal_source_count must match sources")
        if self.upstream_signal_id_count != len(self.upstream_signal_ids):
            raise ValueError("upstream_signal_id_count must match signals")
        if any(
            entry.upstream_signal_source_ids != self.upstream_signal_source_ids
            for entry in self.report_entries
        ):
            raise ValueError("report sources must match secondary surface")
        if any(entry.route_name != self.route_name for entry in self.report_entries):
            raise ValueError("report route_name must match secondary surface")
        if any(entry.task_type != self.task_type for entry in self.report_entries):
            raise ValueError("report task_type must match secondary surface")
        if any(
            entry.execution_mode_id != self.execution_mode_id
            for entry in self.report_entries
        ):
            raise ValueError("report execution_mode_id must match secondary surface")
        if self.all_downstream_systems != _unique_strings(
            tuple(
                downstream
                for entry in self.report_entries
                for downstream in entry.downstream_systems
            )
        ):
            raise ValueError("all_downstream_systems must match reports")
        if self.report_sections != SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match V6.5 report contract")
        if self.cross_cutting_contracts != CROSS_CUTTING_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.5 contracts")
        if self.blocked_runtime_behaviors != BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.5 boundary")
        empty_fields = (
            self.generated_report_artifact_ids,
            self.written_storage_record_ids,
            self.applied_evolution_proposal_ids,
            self.executed_rollback_ids,
            self.mutated_prompt_ids,
            self.mutated_workflow_ids,
            self.mutated_routing_ids,
            self.mutated_memory_ids,
            self.mutated_retrieval_ids,
            self.provider_execution_ids,
            self.mutated_output_ids,
        )
        if any(empty_fields):
            raise ValueError("secondary surface mutation ids must be empty")
        if any(not entry.advisory_only for entry in self.report_entries):
            raise ValueError("all report entries must be advisory only")
        return self


def build_self_evolution_secondary_surface(
    core_surface: SelfEvolutionCoreSurfacePlan | None = None,
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> SelfEvolutionSecondarySurfacePlan:
    """Build secondary advisory report metadata without generating artifacts."""

    surface = core_surface or build_self_evolution_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    report_entries = tuple(
        _build_report_entry(plan=plan, core_surface=surface) for plan in surface.plans
    )
    proposals = surface.proposals
    top_report = max(
        report_entries,
        key=lambda entry: entry.top_proposal_rank_score,
    )
    top_proposal = max(proposals, key=lambda proposal: proposal.proposal_rank_score)
    return SelfEvolutionSecondarySurfacePlan(
        source_core_surface_plan_count=surface.plan_count,
        source_core_surface_proposal_count=surface.proposal_count,
        route_name=surface.plans[0].route_name,
        task_type=surface.plans[0].task_type,
        execution_mode_id=surface.plans[0].proposals[0].execution_mode_id,
        covered_roadmap_items=surface.covered_roadmap_items,
        covered_roadmap_item_count=surface.covered_roadmap_item_count,
        upstream_capabilities=surface.upstream_capabilities,
        upstream_signal_source_ids=surface.upstream_signal_source_ids,
        upstream_signal_source_count=surface.upstream_signal_source_count,
        upstream_signal_ids=surface.upstream_signal_ids,
        upstream_signal_id_count=surface.upstream_signal_id_count,
        report_entries=report_entries,
        report_entry_ids=tuple(entry.report_id for entry in report_entries),
        report_entry_count=len(report_entries),
        proposal_ids=surface.proposal_ids,
        proposal_count=surface.proposal_count,
        guarded_proposal_ids=surface.guarded_proposal_ids,
        guarded_proposal_count=surface.guarded_proposal_count,
        review_required_proposal_ids=surface.review_required_proposal_ids,
        review_required_proposal_count=surface.review_required_proposal_count,
        hitl_required_proposal_ids=surface.hitl_required_proposal_ids,
        hitl_required_proposal_count=surface.hitl_required_proposal_count,
        high_confidence_proposal_ids=surface.high_confidence_proposal_ids,
        high_confidence_proposal_count=surface.high_confidence_proposal_count,
        top_report_entry_id=top_report.report_id,
        top_proposal_id=top_proposal.proposal_id,
        top_proposal_rank_score=top_proposal.proposal_rank_score,
        overall_proposal_rank_score=overall_proposal_rank_score(proposals),
        overall_evolution_posture=overall_evolution_posture(proposals),
        all_downstream_systems=_unique_strings(
            tuple(
                downstream
                for proposal in proposals
                for downstream in proposal.downstream_systems
            )
        ),
        report_sections=SECONDARY_REPORT_SECTIONS,
        cross_cutting_contracts=CROSS_CUTTING_CONTRACTS,
        blocked_runtime_behaviors=BLOCKED_RUNTIME_BEHAVIORS,
    )


def self_evolution_secondary_surface_report_by_id(
    report_id: str,
    surface: SelfEvolutionSecondarySurfacePlan | None = None,
) -> SelfEvolutionAdvisoryReportEntry | None:
    """Return one secondary report entry without applying it."""

    source_surface = surface or build_self_evolution_secondary_surface()
    for entry in source_surface.report_entries:
        if entry.report_id == report_id:
            return entry
    return None


def self_evolution_secondary_surface_report_by_roadmap_item(
    roadmap_item: str,
    surface: SelfEvolutionSecondarySurfacePlan | None = None,
) -> SelfEvolutionAdvisoryReportEntry | None:
    """Return one secondary report entry for a roadmap item."""

    source_surface = surface or build_self_evolution_secondary_surface()
    for entry in source_surface.report_entries:
        if entry.roadmap_item == roadmap_item:
            return entry
    return None


def self_evolution_secondary_surface_reports_for_status(
    status: SelfEvolutionStatus,
    surface: SelfEvolutionSecondarySurfacePlan | None = None,
) -> tuple[SelfEvolutionAdvisoryReportEntry, ...]:
    """Return secondary report entries by advisory status."""

    source_surface = surface or build_self_evolution_secondary_surface()
    return tuple(
        entry for entry in source_surface.report_entries if entry.status == status
    )


def self_evolution_secondary_surface_reports_for_confidence(
    confidence: SelfEvolutionConfidence,
    surface: SelfEvolutionSecondarySurfacePlan | None = None,
) -> tuple[SelfEvolutionAdvisoryReportEntry, ...]:
    """Return secondary report entries by confidence band."""

    source_surface = surface or build_self_evolution_secondary_surface()
    return tuple(
        entry
        for entry in source_surface.report_entries
        if entry.confidence == confidence
    )


def _build_report_entry(
    *,
    plan: SelfEvolutionPlan,
    core_surface: SelfEvolutionCoreSurfacePlan,
) -> SelfEvolutionAdvisoryReportEntry:
    top_proposal = max(
        plan.proposals,
        key=lambda proposal: proposal.proposal_rank_score,
    )
    downstream_systems = _unique_strings(
        tuple(
            downstream
            for proposal in plan.proposals
            for downstream in proposal.downstream_systems
        )
    )
    return SelfEvolutionAdvisoryReportEntry(
        report_id=f"self_evolution_advisory_report::{plan.role}",
        source_core_surface_role=core_surface.role,
        source_core_surface_serialization_version=core_surface.serialization_version,
        roadmap_item=plan.covered_roadmap_items[0],
        plan_role=plan.role,
        plan_serialization_version=plan.serialization_version,
        status=top_proposal.status,
        confidence=top_proposal.confidence,
        route_name=plan.route_name,
        task_type=plan.task_type,
        execution_mode_id=top_proposal.execution_mode_id,
        proposal_ids=plan.proposal_ids,
        proposal_count=plan.proposal_count,
        proposal_rank_scores=tuple(
            proposal.proposal_rank_score for proposal in plan.proposals
        ),
        guarded_proposal_ids=proposal_ids_for_status(plan.proposals, "guarded"),
        guarded_proposal_count=len(proposal_ids_for_status(plan.proposals, "guarded")),
        review_required_proposal_ids=proposal_ids_for_status(
            plan.proposals,
            "review_required",
        ),
        review_required_proposal_count=len(
            proposal_ids_for_status(plan.proposals, "review_required")
        ),
        hitl_required_proposal_ids=plan.hitl_required_proposal_ids,
        hitl_required_proposal_count=plan.hitl_required_proposal_count,
        top_proposal_id=top_proposal.proposal_id,
        top_proposal_rank_score=top_proposal.proposal_rank_score,
        upstream_capabilities=plan.upstream_capabilities,
        upstream_signal_source_ids=plan.upstream_signal_source_ids,
        upstream_signal_ids=plan.upstream_signal_ids,
        downstream_systems=downstream_systems,
        report_sections=SECONDARY_REPORT_SECTIONS,
        why_report_exists=(
            f"{plan.covered_roadmap_items[0]} advisory report metadata exists "
            "to explain the core V6.5 proposals before any human-approved "
            "application path can be considered."
        ),
        upstream_signal_explanation=(
            "The report preserves V6.1 learning, V6.2 memory, V6.3 knowledge, "
            "and V6.4 research signal provenance from the core surface."
        ),
        downstream_impact_explanation=(
            "The report names downstream advisory impact surfaces without "
            "mutating them: "
            + ", ".join(downstream_systems)
            + "."
        ),
        cross_cutting_contracts=plan.cross_cutting_contracts,
        ownership_boundary_checks=top_proposal.ownership_boundary_checks,
        governance_checks=top_proposal.governance_checks,
        advisory_actions=(
            "summarize_core_surface_proposals",
            "rank_reported_proposals_for_human_review",
            "preserve_human_controlled_application_boundary",
        ),
        evidence=(
            f"core_surface_role:{core_surface.role}",
            f"roadmap_item:{plan.covered_roadmap_items[0]}",
            f"plan_role:{plan.role}",
            f"proposal_count:{plan.proposal_count}",
            f"top_proposal_id:{top_proposal.proposal_id}",
            f"top_proposal_rank_score:{top_proposal.proposal_rank_score}",
            "report_artifact_generation:false",
            "storage_write:false",
            "proposal_application:false",
            "hitl_required_before_application:true",
        ),
        blocked_runtime_behaviors=plan.blocked_runtime_behaviors,
    )


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)
