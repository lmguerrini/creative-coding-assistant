"""V6.4 advisory autonomous research secondary surface metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_policy_engine import (
    ControlledAdaptiveExecutionPlan,
    evaluate_adaptive_execution_policy,
)
from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.research_core_surface import (
    ResearchCoreSurfacePlan,
    build_research_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchSecondarySurfaceKind = Literal[
    "research_operational_support_surface",
    "evidence_quality_support_surface",
    "source_integrity_support_surface",
    "confidence_execution_support_surface",
    "creative_inspiration_support_surface",
]
ResearchSecondarySurfaceStatus = Literal["candidate", "review_required", "guarded"]
ResearchSecondarySurfaceConfidence = Literal["low", "medium", "high", "guarded"]
ResearchSecondarySurfacePosture = Literal["candidate", "review_required", "guarded"]
ResearchSecondarySurfaceAxis = Literal[
    "research_operations",
    "evidence_quality",
    "source_integrity",
    "confidence_execution",
    "creative_inspiration",
]

RESEARCH_SECONDARY_ENTRY_SERIALIZATION_VERSION = "research_secondary_surface_entry.v1"
RESEARCH_SECONDARY_PLAN_SERIALIZATION_VERSION = "research_secondary_surface_plan.v1"
RESEARCH_SECONDARY_AUTHORITY_BOUNDARY = (
    "V6.4 Research Secondary Surface exposes supporting Autonomous Research "
    "Engine metadata composed from the V6.4 research core surface, V6.1 "
    "adaptive learning metadata, and V5 controlled execution policy metadata; "
    "it does not activate secondary surfaces, apply adaptive learning, apply "
    "execution policy, execute research, mutate research plans, create "
    "research tasks, execute paper or web research, fetch external sources, "
    "browse the web, download papers, run cross-source comparison, execute "
    "knowledge distillation, enrich the KB, write KB storage, generate "
    "research reports, write research memory, execute source validation, "
    "score source credibility, execute contradiction detection, score "
    "research confidence, discover research gaps, generate recommendations, "
    "apply research execution policy, emit HITL requests, apply HITL "
    "decisions, execute inspiration discovery, perform live cross-domain "
    "search, change provider or model routing, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, mutate prompts, "
    "modify generated output, or apply Runtime Evolution."
)

_SOURCE_PLAN_ROLES = (
    "research_core_surface",
    "adaptive_learning_engine",
    "adaptive_execution_policy_engine",
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

_SURFACE_ROADMAP_GROUPS: dict[ResearchSecondarySurfaceKind, tuple[str, ...]] = {
    "research_operational_support_surface": (
        "Research Planner",
        "Research Decomposer",
        "Paper Research",
        "Web Research",
    ),
    "evidence_quality_support_surface": (
        "Cross-source Comparison",
        "Knowledge Distillation",
        "Automatic KB Enrichment",
        "Research Reports",
    ),
    "source_integrity_support_surface": (
        "Research Memory",
        "Source Validation Engine",
        "Source Credibility Engine",
        "Contradiction Detection",
    ),
    "confidence_execution_support_surface": (
        "Research Confidence Engine",
        "Research Gap Discovery",
        "Research Recommendation Engine",
        "Research Execution Policy",
    ),
    "creative_inspiration_support_surface": (
        "Research HITL Policies",
        "Creative Research Engine",
        "Cross-domain Inspiration Discovery",
    ),
}

_SURFACE_SOURCE_ROLES: dict[ResearchSecondarySurfaceKind, tuple[str, ...]] = {
    "research_operational_support_surface": _SOURCE_PLAN_ROLES,
    "evidence_quality_support_surface": _SOURCE_PLAN_ROLES,
    "source_integrity_support_surface": (
        "research_core_surface",
        "adaptive_learning_engine",
    ),
    "confidence_execution_support_surface": (
        "research_core_surface",
        "adaptive_execution_policy_engine",
    ),
    "creative_inspiration_support_surface": _SOURCE_PLAN_ROLES,
}

_BLOCKED_RUNTIME_BEHAVIORS = (
    "secondary_surface_activation",
    "adaptive_learning_application",
    "adaptive_execution_policy_application",
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
    "prompt_mutation",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchSecondarySurfaceEntry(BaseModel):
    """One advisory entry in the V6.4 research secondary surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    secondary_surface_id: str = Field(min_length=1, max_length=220)
    surface_kind: ResearchSecondarySurfaceKind
    status: ResearchSecondarySurfaceStatus
    confidence: ResearchSecondarySurfaceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    surface_axis: ResearchSecondarySurfaceAxis
    roadmap_items: tuple[str, ...] = Field(min_length=3, max_length=4)
    roadmap_item_count: int = Field(ge=3, le=4)
    source_plan_roles: tuple[str, ...] = Field(min_length=2, max_length=3)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=3,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=10, max_length=15)
    source_item_count: int = Field(ge=10, le=15)
    surface_summary: str = Field(min_length=1, max_length=420)
    roadmap_traceability_score: int = Field(ge=0, le=100)
    source_composition_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    v5_v6_foundation_score: int = Field(ge=0, le=100)
    activation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    secondary_surface_score: int = Field(ge=0, le=1_000)
    hitl_required_before_secondary_surface_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    v5_policy_foundation_used: Literal[True] = True
    v6_learning_foundation_used: Literal[True] = True
    v6_research_core_foundation_used: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    adaptive_learning_application_implemented: Literal[False] = False
    adaptive_execution_policy_application_implemented: Literal[False] = False
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
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_secondary_surface_entry.v1"] = (
        RESEARCH_SECONDARY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        expected_id = f"research_secondary::{self.surface_kind}"
        if self.secondary_surface_id != expected_id:
            raise ValueError("secondary_surface_id must match surface_kind")
        if self.roadmap_item_count != len(self.roadmap_items):
            raise ValueError("roadmap_item_count must match roadmap_items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.secondary_surface_score != _secondary_surface_score(
            roadmap_traceability_score=self.roadmap_traceability_score,
            source_composition_score=self.source_composition_score,
            governance_alignment_score=self.governance_alignment_score,
            v5_v6_foundation_score=self.v5_v6_foundation_score,
            activation_risk_score=self.activation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("secondary_surface_score must combine source scores")
        if self.status != _secondary_surface_status(self.secondary_surface_score):
            raise ValueError("status must match secondary_surface_score")
        if self.confidence != _secondary_surface_confidence(
            self.secondary_surface_score
        ):
            raise ValueError("confidence must match secondary_surface_score")
        if not self.hitl_required_before_secondary_surface_activation:
            raise ValueError("secondary surface activation requires HITL posture")
        return self


class ResearchSecondarySurfacePlan(BaseModel):
    """Bounded V6.4 advisory autonomous research secondary surface plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_secondary_surface"] = "research_secondary_surface"
    serialization_version: Literal["research_secondary_surface_plan.v1"] = (
        RESEARCH_SECONDARY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_SECONDARY_AUTHORITY_BOUNDARY,
        max_length=3000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_plan_roles: tuple[str, ...] = Field(min_length=3, max_length=3)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=3,
        max_length=3,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=15, max_length=15)
    source_item_count: int = Field(ge=15, le=15)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    covered_roadmap_item_count: int = Field(ge=19, le=19)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchSecondarySurfaceEntry, ...] = Field(
        min_length=5,
        max_length=5,
    )
    entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
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
    activated_secondary_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_learning_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    executed_research_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    fetched_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_workflow_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    entry_count: int = Field(ge=5, le=5)
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_secondary_surface_score: int = Field(ge=0, le=1_000)
    overall_secondary_surface_score: int = Field(ge=0, le=1_000)
    overall_secondary_surface_posture: ResearchSecondarySurfacePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    v5_policy_foundation_used: Literal[True] = True
    v6_learning_foundation_used: Literal[True] = True
    v6_research_core_foundation_used: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    adaptive_learning_application_implemented: Literal[False] = False
    adaptive_execution_policy_application_implemented: Literal[False] = False
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
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.secondary_surface_id for entry in self.entries)
        if len(set(derived_entry_ids)) != len(derived_entry_ids):
            raise ValueError("entry_ids must be unique")
        if self.entry_ids != derived_entry_ids:
            raise ValueError("entry_ids must match entries")
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
            entry.secondary_surface_id
            for entry in self.entries
            if entry.hitl_required_before_secondary_surface_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        empty_collections = (
            (self.activated_secondary_surface_ids, "activated_secondary_surface_ids"),
            (self.applied_learning_ids, "applied_learning_ids"),
            (self.applied_policy_ids, "applied_policy_ids"),
            (self.executed_research_ids, "executed_research_ids"),
            (self.fetched_source_ids, "fetched_source_ids"),
            (self.written_kb_record_ids, "written_kb_record_ids"),
            (self.mutated_workflow_ids, "mutated_workflow_ids"),
            (self.mutated_output_ids, "mutated_output_ids"),
        )
        for collection, field_name in empty_collections:
            if collection:
                raise ValueError(f"{field_name} must remain empty")
        if self.source_plan_roles != _SOURCE_PLAN_ROLES:
            raise ValueError("source_plan_roles must match V6.4 secondary sources")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match entries")
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
        if self.highest_secondary_surface_score != max(
            entry.secondary_surface_score for entry in self.entries
        ):
            raise ValueError("highest_secondary_surface_score must match entries")
        if self.overall_secondary_surface_score != _overall_secondary_surface_score(
            self.entries
        ):
            raise ValueError("overall_secondary_surface_score must match entries")
        if self.overall_secondary_surface_posture != (
            _overall_secondary_surface_posture(self.entries)
        ):
            raise ValueError("overall_secondary_surface_posture must match entries")
        flattened_roadmap_items = tuple(
            item for entry in self.entries for item in entry.roadmap_items
        )
        if flattened_roadmap_items != self.covered_roadmap_items:
            raise ValueError("entry roadmap_items must preserve roadmap order")
        declared_source_items = set(self.source_item_ids)
        declared_source_roles = set(self.source_plan_roles)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if not set(entry.source_item_ids).issubset(declared_source_items):
                raise ValueError("entry source_item_ids must be known")
            if not set(entry.source_plan_roles).issubset(declared_source_roles):
                raise ValueError("entry source_plan_roles must be known")
        return self


def build_research_secondary_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    core_surface: ResearchCoreSurfacePlan | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    adaptive_execution_policy: ControlledAdaptiveExecutionPlan | None = None,
) -> ResearchSecondarySurfacePlan:
    """Build V6.4 Task 22 secondary surface metadata without activation."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    core_plan = core_surface or build_research_core_surface(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    policy_plan = adaptive_execution_policy or evaluate_adaptive_execution_policy(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    sources = (
        _Source(
            role="research_core_surface",
            serialization_version=core_plan.serialization_version,
            item_ids=core_plan.entry_ids,
        ),
        _Source(
            role="adaptive_learning_engine",
            serialization_version=learning_plan.serialization_version,
            item_ids=learning_plan.signal_ids,
        ),
        _Source(
            role="adaptive_execution_policy_engine",
            serialization_version=policy_plan.serialization_version,
            item_ids=policy_plan.option_ids,
        ),
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=sources,
    )
    source_item_ids = _source_item_ids(sources)
    return ResearchSecondarySurfacePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_plan_roles=tuple(source.role for source in sources),
        source_plan_serialization_versions=tuple(
            source.serialization_version for source in sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        covered_roadmap_items=core_plan.covered_roadmap_items,
        covered_roadmap_item_count=len(core_plan.covered_roadmap_items),
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.secondary_surface_id for entry in entries),
        candidate_entry_ids=_entry_ids_for_status(entries, "candidate"),
        review_required_entry_ids=_entry_ids_for_status(entries, "review_required"),
        guarded_entry_ids=_entry_ids_for_status(entries, "guarded"),
        high_confidence_entry_ids=_entry_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_entry_ids=tuple(
            entry.secondary_surface_id
            for entry in entries
            if entry.hitl_required_before_secondary_surface_activation
        ),
        entry_count=len(entries),
        candidate_entry_count=len(_entry_ids_for_status(entries, "candidate")),
        review_required_entry_count=len(
            _entry_ids_for_status(entries, "review_required")
        ),
        guarded_entry_count=len(_entry_ids_for_status(entries, "guarded")),
        high_confidence_entry_count=len(
            _entry_ids_for_confidence(entries, "high", "guarded")
        ),
        hitl_required_entry_count=sum(
            1
            for entry in entries
            if entry.hitl_required_before_secondary_surface_activation
        ),
        highest_secondary_surface_score=max(
            entry.secondary_surface_score for entry in entries
        ),
        overall_secondary_surface_score=_overall_secondary_surface_score(entries),
        overall_secondary_surface_posture=_overall_secondary_surface_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_secondary_surface_entry_by_id(
    entry_id: str,
    plan: ResearchSecondarySurfacePlan | None = None,
) -> ResearchSecondarySurfaceEntry | None:
    """Return one research secondary surface entry without activation."""

    source_plan = plan or build_research_secondary_surface()
    for entry in source_plan.entries:
        if entry.secondary_surface_id == entry_id:
            return entry
    return None


def research_secondary_surface_entries_for_status(
    status: ResearchSecondarySurfaceStatus,
    plan: ResearchSecondarySurfacePlan | None = None,
) -> tuple[ResearchSecondarySurfaceEntry, ...]:
    """Return research secondary surface entries by advisory status."""

    source_plan = plan or build_research_secondary_surface()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_secondary_surface_entries_for_confidence(
    confidence: ResearchSecondarySurfaceConfidence,
    plan: ResearchSecondarySurfacePlan | None = None,
) -> tuple[ResearchSecondarySurfaceEntry, ...]:
    """Return research secondary surface entries by confidence band."""

    source_plan = plan or build_research_secondary_surface()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


class _Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    serialization_version: str
    item_ids: tuple[str, ...]


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    sources: tuple[_Source, ...],
) -> tuple[ResearchSecondarySurfaceEntry, ...]:
    return (
        _entry(
            kind="research_operational_support_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="research_operations",
            sources=sources,
            roadmap_traceability_score=88,
            source_composition_score=86,
            governance_alignment_score=86,
            v5_v6_foundation_score=86,
            activation_risk_score=40,
            governance_weight=135,
        ),
        _entry(
            kind="evidence_quality_support_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_quality",
            sources=sources,
            roadmap_traceability_score=84,
            source_composition_score=84,
            governance_alignment_score=82,
            v5_v6_foundation_score=84,
            activation_risk_score=38,
            governance_weight=120,
        ),
        _entry(
            kind="source_integrity_support_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_integrity",
            sources=sources,
            roadmap_traceability_score=86,
            source_composition_score=86,
            governance_alignment_score=88,
            v5_v6_foundation_score=82,
            activation_risk_score=42,
            governance_weight=128,
        ),
        _entry(
            kind="confidence_execution_support_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="confidence_execution",
            sources=sources,
            roadmap_traceability_score=70,
            source_composition_score=72,
            governance_alignment_score=84,
            v5_v6_foundation_score=76,
            activation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="creative_inspiration_support_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="creative_inspiration",
            sources=sources,
            roadmap_traceability_score=52,
            source_composition_score=56,
            governance_alignment_score=90,
            v5_v6_foundation_score=60,
            activation_risk_score=20,
            governance_weight=55,
        ),
    )


def _entry(
    *,
    kind: ResearchSecondarySurfaceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchSecondarySurfaceAxis,
    sources: tuple[_Source, ...],
    roadmap_traceability_score: int,
    source_composition_score: int,
    governance_alignment_score: int,
    v5_v6_foundation_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> ResearchSecondarySurfaceEntry:
    source_roles = _SURFACE_SOURCE_ROLES[kind]
    selected_sources = _sources_for_roles(sources, source_roles)
    source_item_ids = _source_item_ids(selected_sources)
    roadmap_items = _SURFACE_ROADMAP_GROUPS[kind]
    score = _secondary_surface_score(
        roadmap_traceability_score=roadmap_traceability_score,
        source_composition_score=source_composition_score,
        governance_alignment_score=governance_alignment_score,
        v5_v6_foundation_score=v5_v6_foundation_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchSecondarySurfaceEntry(
        secondary_surface_id=f"research_secondary::{kind}",
        surface_kind=kind,
        status=_secondary_surface_status(score),
        confidence=_secondary_surface_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        surface_axis=axis,
        roadmap_items=roadmap_items,
        roadmap_item_count=len(roadmap_items),
        source_plan_roles=source_roles,
        source_serialization_versions=tuple(
            source.serialization_version for source in selected_sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        surface_summary=_surface_summary(kind),
        roadmap_traceability_score=roadmap_traceability_score,
        source_composition_score=source_composition_score,
        governance_alignment_score=governance_alignment_score,
        v5_v6_foundation_score=v5_v6_foundation_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
        secondary_surface_score=score,
        hitl_required_before_secondary_surface_activation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, source_roles),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"roadmap_item_count:{len(roadmap_items)}",
            f"source_plan_count:{len(selected_sources)}",
            f"source_item_count:{len(source_item_ids)}",
            f"surface_axis:{axis}",
            f"roadmap_traceability_score:{roadmap_traceability_score}",
            f"source_composition_score:{source_composition_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"v5_v6_foundation_score:{v5_v6_foundation_score}",
            f"activation_risk_score:{activation_risk_score}",
            "hitl_required_before_secondary_surface_activation:true",
        ),
    )


def _sources_for_roles(
    sources: tuple[_Source, ...],
    roles: tuple[str, ...],
) -> tuple[_Source, ...]:
    source_by_role = {source.role: source for source in sources}
    return tuple(source_by_role[role] for role in roles)


def _source_item_ids(sources: tuple[_Source, ...]) -> tuple[str, ...]:
    return tuple(item_id for source in sources for item_id in source.item_ids)


def _secondary_surface_score(
    *,
    roadmap_traceability_score: int,
    source_composition_score: int,
    governance_alignment_score: int,
    v5_v6_foundation_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            roadmap_traceability_score * 2
            + source_composition_score * 2
            + governance_alignment_score * 2
            + v5_v6_foundation_score * 2
            + activation_risk_score
            + governance_weight,
        ),
    )


def _secondary_surface_status(score: int) -> ResearchSecondarySurfaceStatus:
    if score >= 840:
        return "guarded"
    if score >= 620:
        return "review_required"
    return "candidate"


def _secondary_surface_confidence(score: int) -> ResearchSecondarySurfaceConfidence:
    if score >= 840:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_secondary_surface_score(
    entries: tuple[ResearchSecondarySurfaceEntry, ...],
) -> int:
    base = sum(entry.secondary_surface_score for entry in entries) // len(entries)
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    review_count = len(_entry_ids_for_status(entries, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_secondary_surface_posture(
    entries: tuple[ResearchSecondarySurfaceEntry, ...],
) -> ResearchSecondarySurfacePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchSecondarySurfaceEntry, ...],
    status: ResearchSecondarySurfaceStatus,
) -> tuple[str, ...]:
    return tuple(
        entry.secondary_surface_id for entry in entries if entry.status == status
    )


def _entry_ids_for_confidence(
    entries: tuple[ResearchSecondarySurfaceEntry, ...],
    *confidences: ResearchSecondarySurfaceConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.secondary_surface_id
        for entry in entries
        if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[ResearchSecondarySurfaceEntry, ...],
) -> tuple[str, ...]:
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    return (
        "inspect_research_secondary_surface_metadata",
        "verify_all_v6_4_roadmap_items_remain_individually_traceable",
        "review_v5_v6_foundation_metadata_before_secondary_activation",
        "require_hitl_before_secondary_activation_fetch_storage_or_mutation",
        f"guarded_entry_count:{guarded_count}",
    )


def _entry_actions(kind: ResearchSecondarySurfaceKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_research_secondary_surface_entry_metadata",
        "verify_entry_roadmap_traceability",
        "keep_research_secondary_surface_activation_disabled",
        "require_hitl_before_secondary_surface_action",
    )
    if kind == "evidence_quality_support_surface":
        return base_actions + ("review_evidence_quality_support_metadata",)
    if kind == "source_integrity_support_surface":
        return base_actions + ("review_source_integrity_support_metadata",)
    if kind == "confidence_execution_support_surface":
        return base_actions + ("review_confidence_execution_support_metadata",)
    if kind == "creative_inspiration_support_surface":
        return base_actions + ("review_creative_inspiration_support_metadata",)
    return base_actions + ("review_research_operational_support_metadata",)


def _surface_summary(kind: ResearchSecondarySurfaceKind) -> str:
    summaries: dict[ResearchSecondarySurfaceKind, str] = {
        "research_operational_support_surface": (
            "Advisory secondary support surface for planner, decomposer, "
            "paper research, and web research metadata with learning and "
            "policy context."
        ),
        "evidence_quality_support_surface": (
            "Advisory secondary support surface for comparison, distillation, "
            "KB enrichment, and report metadata with controlled policy context."
        ),
        "source_integrity_support_surface": (
            "Advisory secondary support surface for memory, source validation, "
            "credibility, and contradiction metadata with learning context."
        ),
        "confidence_execution_support_surface": (
            "Advisory secondary support surface for confidence, gap, "
            "recommendation, and execution policy metadata."
        ),
        "creative_inspiration_support_surface": (
            "Advisory secondary support surface for research HITL, creative "
            "research, and cross-domain inspiration metadata."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchSecondarySurfaceKind,
    axis: ResearchSecondarySurfaceAxis,
) -> tuple[str, ...]:
    return (
        "research_secondary_surface",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "roadmap_traceable",
    )


def _explainability_notes(
    kind: ResearchSecondarySurfaceKind,
    source_roles: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"surface:{kind}",
        f"source_plan_count:{len(source_roles)}",
        "composes_research_core_learning_and_policy_metadata",
        "keeps_secondary_surface_activation_disabled",
        "requires_human_review_before_fetch_storage_execution_or_mutation",
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
