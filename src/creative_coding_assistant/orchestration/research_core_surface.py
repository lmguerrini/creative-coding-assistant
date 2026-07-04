"""V6.4 advisory autonomous research core surface metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.automatic_kb_enrichment import (
    build_automatic_kb_enrichment,
)
from creative_coding_assistant.orchestration.contradiction_detection import (
    build_contradiction_detection,
)
from creative_coding_assistant.orchestration.creative_research_engine import (
    build_creative_research_engine,
)
from creative_coding_assistant.orchestration.cross_domain_inspiration_discovery import (
    build_cross_domain_inspiration_discovery,
)
from creative_coding_assistant.orchestration.cross_source_comparison import (
    build_cross_source_comparison,
)
from creative_coding_assistant.orchestration.knowledge_distillation import (
    build_knowledge_distillation,
)
from creative_coding_assistant.orchestration.paper_research import (
    build_paper_research,
)
from creative_coding_assistant.orchestration.research_confidence_engine import (
    build_research_confidence_engine,
)
from creative_coding_assistant.orchestration.research_decomposer import (
    build_research_decomposer,
)
from creative_coding_assistant.orchestration.research_execution_policy import (
    build_research_execution_policy,
)
from creative_coding_assistant.orchestration.research_gap_discovery import (
    build_research_gap_discovery,
)
from creative_coding_assistant.orchestration.research_hitl_policies import (
    build_research_hitl_policies,
)
from creative_coding_assistant.orchestration.research_memory import (
    build_research_memory,
)
from creative_coding_assistant.orchestration.research_planner import (
    build_research_planner,
)
from creative_coding_assistant.orchestration.research_recommendation_engine import (
    build_research_recommendation_engine,
)
from creative_coding_assistant.orchestration.research_reports import (
    build_research_reports,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.source_credibility_engine import (
    build_source_credibility_engine,
)
from creative_coding_assistant.orchestration.source_validation_engine import (
    build_source_validation_engine,
)
from creative_coding_assistant.orchestration.web_research import build_web_research

ResearchCoreSurfaceKind = Literal[
    "planning_acquisition_surface",
    "evidence_distillation_surface",
    "memory_source_integrity_surface",
    "confidence_execution_governance_surface",
    "creative_inspiration_surface",
]
ResearchCoreSurfaceStatus = Literal["candidate", "review_required", "guarded"]
ResearchCoreSurfaceConfidence = Literal["low", "medium", "high", "guarded"]
ResearchCoreSurfacePosture = Literal["candidate", "review_required", "guarded"]
ResearchCoreSurfaceAxis = Literal[
    "planning_acquisition",
    "evidence_distillation",
    "memory_source_integrity",
    "confidence_execution_governance",
    "creative_inspiration",
]

RESEARCH_CORE_ENTRY_SERIALIZATION_VERSION = "research_core_surface_entry.v1"
RESEARCH_CORE_PLAN_SERIALIZATION_VERSION = "research_core_surface_plan.v1"

RESEARCH_CORE_AUTHORITY_BOUNDARY = (
    "V6.4 Research Core Surface exposes the validated V6.4 Autonomous "
    "Research Engine roadmap surfaces as inspectable advisory metadata only; "
    "it does not activate core surfaces, execute research, mutate research "
    "plans, create research tasks, execute paper or web research, fetch "
    "external sources, browse the web, download papers, run cross-source "
    "comparison, execute knowledge distillation, enrich the KB, write KB "
    "storage, generate research reports, write research memory, execute "
    "source validation, score source credibility, execute contradiction "
    "detection, score research confidence, discover research gaps, generate "
    "recommendations, apply research execution policy, emit HITL requests, "
    "apply HITL decisions, execute inspiration discovery, perform live "
    "cross-domain search, change provider or model routing, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "modify generated output, or apply Runtime Evolution."
)

_SOURCE_PLAN_ROLES = (
    "research_planner",
    "research_decomposer",
    "paper_research",
    "web_research",
    "cross_source_comparison",
    "knowledge_distillation",
    "automatic_kb_enrichment",
    "research_reports",
    "research_memory",
    "source_validation_engine",
    "source_credibility_engine",
    "contradiction_detection",
    "research_confidence_engine",
    "research_gap_discovery",
    "research_recommendation_engine",
    "research_execution_policy",
    "research_hitl_policies",
    "creative_research_engine",
    "cross_domain_inspiration_discovery",
)

_ROADMAP_ITEMS = (
    "Research Planner",
    "Research Decomposer",
    "Paper Research",
    "Web Research",
    "Cross-source Comparison",
    "Knowledge Distillation",
    "Automatic KB Enrichment",
    "Research Reports",
    "Research Memory",
    "Source Validation Engine",
    "Source Credibility Engine",
    "Contradiction Detection",
    "Research Confidence Engine",
    "Research Gap Discovery",
    "Research Recommendation Engine",
    "Research Execution Policy",
    "Research HITL Policies",
    "Creative Research Engine",
    "Cross-domain Inspiration Discovery",
)

_SURFACE_ROLE_GROUPS: dict[ResearchCoreSurfaceKind, tuple[str, ...]] = {
    "planning_acquisition_surface": (
        "research_planner",
        "research_decomposer",
        "paper_research",
        "web_research",
    ),
    "evidence_distillation_surface": (
        "cross_source_comparison",
        "knowledge_distillation",
        "automatic_kb_enrichment",
        "research_reports",
    ),
    "memory_source_integrity_surface": (
        "research_memory",
        "source_validation_engine",
        "source_credibility_engine",
        "contradiction_detection",
    ),
    "confidence_execution_governance_surface": (
        "research_confidence_engine",
        "research_gap_discovery",
        "research_recommendation_engine",
        "research_execution_policy",
    ),
    "creative_inspiration_surface": (
        "research_hitl_policies",
        "creative_research_engine",
        "cross_domain_inspiration_discovery",
    ),
}

_BLOCKED_RUNTIME_BEHAVIORS = (
    "core_surface_activation",
    "research_execution",
    "research_plan_mutation",
    "research_task_creation",
    "paper_research_execution",
    "web_research_execution",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "cross_source_comparison_execution",
    "knowledge_distillation_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "research_report_generation",
    "research_memory_write",
    "source_validation_execution",
    "source_credibility_scoring_execution",
    "contradiction_detection_execution",
    "research_confidence_scoring_execution",
    "research_gap_discovery_execution",
    "research_recommendation_generation",
    "research_execution_policy_application",
    "hitl_request_emission",
    "hitl_decision_application",
    "inspiration_discovery_execution",
    "live_cross_domain_search",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchCoreSurfaceEntry(BaseModel):
    """One advisory entry in the V6.4 research core surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    core_surface_id: str = Field(min_length=1, max_length=200)
    surface_kind: ResearchCoreSurfaceKind
    status: ResearchCoreSurfaceStatus
    confidence: ResearchCoreSurfaceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    surface_axis: ResearchCoreSurfaceAxis
    roadmap_items: tuple[str, ...] = Field(min_length=3, max_length=4)
    roadmap_item_count: int = Field(ge=3, le=4)
    source_plan_roles: tuple[str, ...] = Field(min_length=3, max_length=4)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=3,
        max_length=4,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=15, max_length=20)
    source_item_count: int = Field(ge=15, le=20)
    surface_summary: str = Field(min_length=1, max_length=420)
    surface_coverage_score: int = Field(ge=0, le=100)
    source_traceability_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    activation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    core_surface_score: int = Field(ge=0, le=1_000)
    hitl_required_before_core_surface_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    cross_source_comparison_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    research_memory_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_core_surface_entry.v1"] = (
        RESEARCH_CORE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.core_surface_id != f"research_core::{self.surface_kind}":
            raise ValueError("core_surface_id must match surface_kind")
        if self.roadmap_item_count != len(self.roadmap_items):
            raise ValueError("roadmap_item_count must match roadmap_items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.core_surface_score != _core_surface_score(
            surface_coverage_score=self.surface_coverage_score,
            source_traceability_score=self.source_traceability_score,
            governance_alignment_score=self.governance_alignment_score,
            activation_risk_score=self.activation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("core_surface_score must combine source scores")
        if self.status != _core_surface_status(self.core_surface_score):
            raise ValueError("status must match core_surface_score")
        if self.confidence != _core_surface_confidence(self.core_surface_score):
            raise ValueError("confidence must match core_surface_score")
        if not self.hitl_required_before_core_surface_activation:
            raise ValueError("core surface activation requires HITL posture")
        return self


class ResearchCoreSurfacePlan(BaseModel):
    """Bounded V6.4 advisory autonomous research core surface plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_core_surface"] = "research_core_surface"
    serialization_version: Literal["research_core_surface_plan.v1"] = (
        RESEARCH_CORE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_CORE_AUTHORITY_BOUNDARY,
        max_length=3000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: object
    source_plan_roles: tuple[str, ...] = Field(min_length=19, max_length=19)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=19,
        max_length=19,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=95, max_length=95)
    source_item_count: int = Field(ge=95, le=95)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    covered_roadmap_item_count: int = Field(ge=19, le=19)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchCoreSurfaceEntry, ...] = Field(min_length=5, max_length=5)
    entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    entry_count: int = Field(ge=5, le=5)
    candidate_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    activated_core_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_research_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_workflow_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_posture: ResearchCoreSurfacePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    research_execution_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    cross_source_comparison_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    research_memory_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_execution_policy_application_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    inspiration_discovery_execution_implemented: Literal[False] = False
    live_cross_domain_search_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.core_surface_id for entry in self.entries)
        if len(set(derived_entry_ids)) != len(derived_entry_ids):
            raise ValueError("entry_ids must be unique")
        if self.entry_ids != derived_entry_ids:
            raise ValueError("entry_ids must match entries")
        if self.source_plan_roles != _SOURCE_PLAN_ROLES:
            raise ValueError("source_plan_roles must match V6.4 source roles")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match entries")
        if self.candidate_entry_ids != _entry_ids_for_status(
            self.entries,
            "candidate",
        ):
            raise ValueError("candidate_entry_ids must match entries")
        if self.review_required_entry_ids != _entry_ids_for_status(
            self.entries,
            "review_required",
        ):
            raise ValueError("review_required_entry_ids must match entries")
        if self.guarded_entry_ids != _entry_ids_for_status(self.entries, "guarded"):
            raise ValueError("guarded_entry_ids must match entries")
        if self.high_confidence_entry_ids != _entry_ids_for_confidence(
            self.entries,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_entry_ids must match entries")
        if self.hitl_required_entry_ids != tuple(
            entry.core_surface_id
            for entry in self.entries
            if entry.hitl_required_before_core_surface_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.activated_core_surface_ids:
            raise ValueError("activated_core_surface_ids must remain empty")
        if self.executed_research_ids:
            raise ValueError("executed_research_ids must remain empty")
        if self.fetched_source_ids:
            raise ValueError("fetched_source_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.mutated_workflow_ids:
            raise ValueError("mutated_workflow_ids must remain empty")
        if self.mutated_output_ids:
            raise ValueError("mutated_output_ids must remain empty")
        if self.candidate_entry_count != len(self.candidate_entry_ids):
            raise ValueError("candidate_entry_count must match entries")
        if self.review_required_entry_count != len(self.review_required_entry_ids):
            raise ValueError("review_required_entry_count must match entries")
        if self.guarded_entry_count != len(self.guarded_entry_ids):
            raise ValueError("guarded_entry_count must match entries")
        if self.high_confidence_entry_count != len(self.high_confidence_entry_ids):
            raise ValueError("high_confidence_entry_count must match entries")
        if self.hitl_required_entry_count != len(self.hitl_required_entry_ids):
            raise ValueError("hitl_required_entry_count must match entries")
        if self.highest_core_surface_score != max(
            entry.core_surface_score for entry in self.entries
        ):
            raise ValueError("highest_core_surface_score must match entries")
        if self.overall_core_surface_score != _overall_core_surface_score(self.entries):
            raise ValueError("overall_core_surface_score must match entries")
        if self.overall_core_surface_posture != _overall_core_surface_posture(
            self.entries
        ):
            raise ValueError("overall_core_surface_posture must match entries")
        if (
            tuple(item for entry in self.entries for item in entry.roadmap_items)
            != self.covered_roadmap_items
        ):
            raise ValueError("entry roadmap_items must preserve roadmap order")
        source_items = set(self.source_item_ids)
        source_roles = set(self.source_plan_roles)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if not set(entry.source_item_ids).issubset(source_items):
                raise ValueError("entry source_item_ids must be declared")
            if not set(entry.source_plan_roles).issubset(source_roles):
                raise ValueError("entry source_plan_roles must be declared")
        return self


def build_research_core_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> ResearchCoreSurfacePlan:
    """Build V6.4 Task 21 core surface metadata without activation."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    source_plans = _source_plans(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    plan_by_role = {str(plan.role): plan for plan in source_plans}
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        plan_by_role=plan_by_role,
    )
    source_item_ids = _source_item_ids_for_plans(source_plans)
    return ResearchCoreSurfacePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=source_plans[-1].model_dump()["checked_at"],
        source_plan_roles=tuple(str(plan.role) for plan in source_plans),
        source_plan_serialization_versions=tuple(
            str(plan.serialization_version) for plan in source_plans
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.core_surface_id for entry in entries),
        entry_count=len(entries),
        candidate_entry_ids=_entry_ids_for_status(entries, "candidate"),
        review_required_entry_ids=_entry_ids_for_status(entries, "review_required"),
        guarded_entry_ids=_entry_ids_for_status(entries, "guarded"),
        high_confidence_entry_ids=_entry_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_entry_ids=tuple(
            entry.core_surface_id
            for entry in entries
            if entry.hitl_required_before_core_surface_activation
        ),
        candidate_entry_count=len(_entry_ids_for_status(entries, "candidate")),
        review_required_entry_count=len(
            _entry_ids_for_status(entries, "review_required")
        ),
        guarded_entry_count=len(_entry_ids_for_status(entries, "guarded")),
        high_confidence_entry_count=len(
            _entry_ids_for_confidence(entries, "high", "guarded")
        ),
        hitl_required_entry_count=sum(
            1 for entry in entries if entry.hitl_required_before_core_surface_activation
        ),
        highest_core_surface_score=max(entry.core_surface_score for entry in entries),
        overall_core_surface_score=_overall_core_surface_score(entries),
        overall_core_surface_posture=_overall_core_surface_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_core_surface_entry_by_id(
    entry_id: str,
    plan: ResearchCoreSurfacePlan | None = None,
) -> ResearchCoreSurfaceEntry | None:
    """Return one core surface entry without activating the surface."""

    source_plan = plan or build_research_core_surface()
    for entry in source_plan.entries:
        if entry.core_surface_id == entry_id:
            return entry
    return None


def research_core_surface_entries_for_status(
    status: ResearchCoreSurfaceStatus,
    plan: ResearchCoreSurfacePlan | None = None,
) -> tuple[ResearchCoreSurfaceEntry, ...]:
    """Return research core surface entries by advisory status."""

    source_plan = plan or build_research_core_surface()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_core_surface_entries_for_confidence(
    confidence: ResearchCoreSurfaceConfidence,
    plan: ResearchCoreSurfacePlan | None = None,
) -> tuple[ResearchCoreSurfaceEntry, ...]:
    """Return research core surface entries by confidence band."""

    source_plan = plan or build_research_core_surface()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _source_plans(
    *,
    route: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> tuple[BaseModel, ...]:
    kwargs = {
        "route": route,
        "task_type": task_type,
        "execution_mode_id": execution_mode_id,
    }
    return (
        build_research_planner(**kwargs),
        build_research_decomposer(**kwargs),
        build_paper_research(**kwargs),
        build_web_research(**kwargs),
        build_cross_source_comparison(**kwargs),
        build_knowledge_distillation(**kwargs),
        build_automatic_kb_enrichment(**kwargs),
        build_research_reports(**kwargs),
        build_research_memory(**kwargs),
        build_source_validation_engine(**kwargs),
        build_source_credibility_engine(**kwargs),
        build_contradiction_detection(**kwargs),
        build_research_confidence_engine(**kwargs),
        build_research_gap_discovery(**kwargs),
        build_research_recommendation_engine(**kwargs),
        build_research_execution_policy(**kwargs),
        build_research_hitl_policies(**kwargs),
        build_creative_research_engine(**kwargs),
        build_cross_domain_inspiration_discovery(**kwargs),
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    plan_by_role: dict[str, BaseModel],
) -> tuple[ResearchCoreSurfaceEntry, ...]:
    return (
        _entry(
            kind="planning_acquisition_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="planning_acquisition",
            plan_by_role=plan_by_role,
            surface_coverage_score=90,
            source_traceability_score=90,
            governance_alignment_score=84,
            activation_risk_score=40,
            governance_weight=130,
        ),
        _entry(
            kind="evidence_distillation_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_distillation",
            plan_by_role=plan_by_role,
            surface_coverage_score=84,
            source_traceability_score=86,
            governance_alignment_score=80,
            activation_risk_score=38,
            governance_weight=110,
        ),
        _entry(
            kind="memory_source_integrity_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="memory_source_integrity",
            plan_by_role=plan_by_role,
            surface_coverage_score=88,
            source_traceability_score=88,
            governance_alignment_score=86,
            activation_risk_score=42,
            governance_weight=125,
        ),
        _entry(
            kind="confidence_execution_governance_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="confidence_execution_governance",
            plan_by_role=plan_by_role,
            surface_coverage_score=66,
            source_traceability_score=70,
            governance_alignment_score=78,
            activation_risk_score=28,
            governance_weight=75,
        ),
        _entry(
            kind="creative_inspiration_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="creative_inspiration",
            plan_by_role=plan_by_role,
            surface_coverage_score=54,
            source_traceability_score=58,
            governance_alignment_score=90,
            activation_risk_score=18,
            governance_weight=65,
        ),
    )


def _entry(
    *,
    kind: ResearchCoreSurfaceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchCoreSurfaceAxis,
    plan_by_role: dict[str, BaseModel],
    surface_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> ResearchCoreSurfaceEntry:
    source_roles = _SURFACE_ROLE_GROUPS[kind]
    source_plans = tuple(plan_by_role[role] for role in source_roles)
    source_item_ids = _source_item_ids_for_plans(source_plans)
    roadmap_items = tuple(
        item
        for plan in source_plans
        for item in tuple(plan.model_dump()["covered_roadmap_items"])
    )
    score = _core_surface_score(
        surface_coverage_score=surface_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchCoreSurfaceEntry(
        core_surface_id=f"research_core::{kind}",
        surface_kind=kind,
        status=_core_surface_status(score),
        confidence=_core_surface_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        surface_axis=axis,
        roadmap_items=roadmap_items,
        roadmap_item_count=len(roadmap_items),
        source_plan_roles=source_roles,
        source_serialization_versions=tuple(
            str(plan.serialization_version) for plan in source_plans
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        surface_summary=_surface_summary(kind),
        surface_coverage_score=surface_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
        core_surface_score=score,
        hitl_required_before_core_surface_activation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, source_roles),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"roadmap_item_count:{len(roadmap_items)}",
            f"source_plan_count:{len(source_plans)}",
            f"source_item_count:{len(source_item_ids)}",
            f"surface_axis:{axis}",
            f"surface_coverage_score:{surface_coverage_score}",
            f"source_traceability_score:{source_traceability_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"activation_risk_score:{activation_risk_score}",
            "hitl_required_before_core_surface_activation:true",
        ),
    )


def _source_item_ids_for_plans(plans: tuple[BaseModel, ...]) -> tuple[str, ...]:
    return tuple(item_id for plan in plans for item_id in _source_item_ids(plan))


def _source_item_ids(plan: BaseModel) -> tuple[str, ...]:
    payload = plan.model_dump()
    for key in ("entry_ids", "signal_ids", "candidate_ids"):
        if key in payload:
            return tuple(str(item_id) for item_id in payload[key])
    raise ValueError(f"Plan {plan.__class__.__name__} has no source ids")


def _core_surface_score(
    *,
    surface_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            surface_coverage_score * 3
            + source_traceability_score * 3
            + governance_alignment_score * 2
            + activation_risk_score * 2
            + governance_weight,
        ),
    )


def _core_surface_status(score: int) -> ResearchCoreSurfaceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _core_surface_confidence(score: int) -> ResearchCoreSurfaceConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_core_surface_score(
    entries: tuple[ResearchCoreSurfaceEntry, ...],
) -> int:
    base = sum(entry.core_surface_score for entry in entries) // len(entries)
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    review_count = len(_entry_ids_for_status(entries, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_core_surface_posture(
    entries: tuple[ResearchCoreSurfaceEntry, ...],
) -> ResearchCoreSurfacePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchCoreSurfaceEntry, ...],
    status: ResearchCoreSurfaceStatus,
) -> tuple[str, ...]:
    return tuple(entry.core_surface_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchCoreSurfaceEntry, ...],
    *confidences: ResearchCoreSurfaceConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.core_surface_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(entries: tuple[ResearchCoreSurfaceEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_core_surface_entries:{len(entries)}",
        "verify_all_v6_4_roadmap_items_remain_traceable",
        "confirm_no_core_surface_activation",
        "confirm_no_research_or_retrieval_execution",
        "request_hitl_before_core_surface_activation",
    )


def _entry_actions(kind: ResearchCoreSurfaceKind) -> tuple[str, ...]:
    actions: dict[ResearchCoreSurfaceKind, tuple[str, ...]] = {
        "planning_acquisition_surface": (
            "review_planning_acquisition_boundary",
            "confirm_no_paper_or_web_research_execution",
            "confirm_no_external_source_fetch",
        ),
        "evidence_distillation_surface": (
            "review_evidence_distillation_boundary",
            "confirm_no_kb_enrichment_execution",
            "confirm_no_report_generation",
        ),
        "memory_source_integrity_surface": (
            "review_memory_source_integrity_boundary",
            "confirm_no_source_validation_execution",
            "confirm_no_research_memory_write",
        ),
        "confidence_execution_governance_surface": (
            "review_confidence_execution_boundary",
            "confirm_no_recommendation_generation",
            "confirm_no_execution_policy_application",
        ),
        "creative_inspiration_surface": (
            "review_creative_inspiration_boundary",
            "confirm_no_inspiration_discovery_execution",
            "confirm_no_generated_output_mutation",
        ),
    }
    return actions[kind]


def _surface_summary(kind: ResearchCoreSurfaceKind) -> str:
    summaries: dict[ResearchCoreSurfaceKind, str] = {
        "planning_acquisition_surface": (
            "Advisory core surface grouping research planning, decomposition, "
            "paper research posture, and web research posture."
        ),
        "evidence_distillation_surface": (
            "Advisory core surface grouping comparison, distillation, KB "
            "enrichment posture, and report posture."
        ),
        "memory_source_integrity_surface": (
            "Advisory core surface grouping research memory, source validation, "
            "source credibility, and contradiction metadata."
        ),
        "confidence_execution_governance_surface": (
            "Advisory core surface grouping confidence, gap, recommendation, "
            "and execution policy metadata."
        ),
        "creative_inspiration_surface": (
            "Advisory core surface grouping research HITL, creative research, "
            "and cross-domain inspiration metadata."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchCoreSurfaceKind,
    axis: ResearchCoreSurfaceAxis,
) -> tuple[str, ...]:
    return (
        "research_core_surface",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "roadmap_traceable",
    )


def _explainability_notes(
    kind: ResearchCoreSurfaceKind,
    source_roles: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"surface_kind:{kind}",
        f"source_plan_count:{len(source_roles)}",
        "all_roadmap_items_traceable:true",
        "all_sources_metadata_only:true",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    candidate = str(task_type).strip()
    if candidate not in get_args(TaskRoutingType):
        raise ValueError(f"Unknown task routing type: {task_type!r}")
    return cast(TaskRoutingType, candidate)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    candidate = str(execution_mode_id).strip()
    if candidate not in allowed_modes:
        raise ValueError(f"Unknown execution mode id: {execution_mode_id!r}")
    return cast(ExecutionModeId, candidate)
