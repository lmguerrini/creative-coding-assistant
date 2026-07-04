"""V6.4 advisory research planner metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.rag.sources import (
    OfficialSource,
    approved_official_sources,
    official_source_domains,
)

ResearchPlanningKind = Literal[
    "research_scope_framing",
    "source_strategy_planning",
    "validation_strategy_planning",
    "distillation_report_planning",
    "governance_gate_planning",
]
ResearchPlanningStatus = Literal["candidate", "review_required", "guarded"]
ResearchPlanningConfidence = Literal["low", "medium", "high", "guarded"]
ResearchPlanningPosture = Literal["candidate", "review_required", "guarded"]
ResearchPlanningAxis = Literal[
    "scope_framing",
    "source_strategy",
    "validation_strategy",
    "distillation_reporting",
    "governance_gate",
]

RESEARCH_PLANNER_ENTRY_SERIALIZATION_VERSION = "research_planner_entry.v1"
RESEARCH_PLANNER_PLAN_SERIALIZATION_VERSION = "research_planner_plan.v1"
OFFICIAL_SOURCE_REGISTRY_SERIALIZATION_VERSION = "official_source_registry.v1"
_DEFAULT_CHECKED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

RESEARCH_PLANNER_AUTHORITY_BOUNDARY = (
    "V6.4 Research Planner exposes research scope, source strategy, validation "
    "strategy, distillation/reporting posture, and governance-gate readiness "
    "as inspectable advisory metadata only; it does not execute research "
    "plans, decompose research tasks, perform paper research, perform web "
    "research, fetch external sources, download papers, browse the web, "
    "validate sources live, compute source credibility, detect contradictions, "
    "score research confidence, discover gaps, generate recommendations, "
    "enrich the KB, write storage, mutate source registries, mutate retrieval "
    "configuration, execute retrieval, mutate ranking, provision providers, "
    "infer API keys, route providers or models, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, modify generated "
    "output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Planner",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_plan_execution",
    "research_task_decomposition",
    "paper_research_execution",
    "web_research_execution",
    "external_source_fetch",
    "paper_download",
    "web_browsing",
    "live_source_validation",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "research_confidence_scoring",
    "research_gap_discovery_execution",
    "research_recommendation_generation",
    "kb_enrichment_execution",
    "kb_storage_write",
    "source_registry_mutation",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "ranking_mutation",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchPlanningEntry(BaseModel):
    """One advisory research planning entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    planning_kind: ResearchPlanningKind
    status: ResearchPlanningStatus
    confidence: ResearchPlanningConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    planning_axis: ResearchPlanningAxis
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    source_type_count: int = Field(ge=1, le=4)
    planning_summary: str = Field(min_length=1, max_length=360)
    scope_clarity_score: int = Field(ge=0, le=100)
    source_strategy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    planning_score: int = Field(ge=0, le=1_000)
    hitl_required_before_research_execution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_planner_capability_implemented: Literal[True] = True
    research_planner_metadata_implemented: Literal[True] = True
    official_source_registry_used: Literal[True] = True
    research_plan_execution_implemented: Literal[False] = False
    research_task_decomposition_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    research_gap_discovery_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_planner_entry.v1"] = (
        RESEARCH_PLANNER_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_planner::{self.planning_kind}":
            raise ValueError("entry_id must match planning_kind")
        if self.source_count != len(self.source_ids):
            raise ValueError("source_count must match source_ids")
        if self.planning_score != _planning_score(
            scope_clarity_score=self.scope_clarity_score,
            source_strategy_score=self.source_strategy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("planning_score must combine source scores")
        if self.status != _planning_status(self.planning_score):
            raise ValueError("status must match planning_score")
        if self.confidence != _planning_confidence(self.planning_score):
            raise ValueError("confidence must match planning_score")
        if not self.hitl_required_before_research_execution:
            raise ValueError("research execution requires HITL posture")
        return self


class ResearchPlannerPlan(BaseModel):
    """Bounded V6.4 advisory research planner plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_planner"] = "research_planner"
    serialization_version: Literal["research_planner_plan.v1"] = (
        RESEARCH_PLANNER_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_PLANNER_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_registry_role: Literal["approved_official_sources"] = (
        "approved_official_sources"
    )
    source_registry_serialization_version: Literal["official_source_registry.v1"] = (
        OFFICIAL_SOURCE_REGISTRY_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    source_type_count: int = Field(ge=1, le=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchPlanningEntry, ...] = Field(min_length=5, max_length=5)
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
    planned_research_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    decomposed_research_task_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    paper_research_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    web_research_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_external_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    kb_enrichment_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_config_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_planning_score: int = Field(ge=0, le=1_000)
    overall_planning_score: int = Field(ge=0, le=1_000)
    overall_planning_posture: ResearchPlanningPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_planner_capability_implemented: Literal[True] = True
    research_planner_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    official_source_registry_used: Literal[True] = True
    research_plan_execution_implemented: Literal[False] = False
    research_task_decomposition_implemented: Literal[False] = False
    paper_research_execution_implemented: Literal[False] = False
    web_research_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    research_gap_discovery_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @field_validator("checked_at")
    @classmethod
    def _checked_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.entry_id for entry in self.entries)
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
            entry.entry_id
            for entry in self.entries
            if entry.hitl_required_before_research_execution
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.planned_research_execution_ids:
            raise ValueError("planned_research_execution_ids must remain empty")
        if self.decomposed_research_task_ids:
            raise ValueError("decomposed_research_task_ids must remain empty")
        if self.paper_research_execution_ids:
            raise ValueError("paper_research_execution_ids must remain empty")
        if self.web_research_execution_ids:
            raise ValueError("web_research_execution_ids must remain empty")
        if self.fetched_external_source_ids:
            raise ValueError("fetched_external_source_ids must remain empty")
        if self.kb_enrichment_execution_ids:
            raise ValueError("kb_enrichment_execution_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.mutated_retrieval_config_ids:
            raise ValueError("mutated_retrieval_config_ids must remain empty")
        if self.source_count != len(self.source_ids):
            raise ValueError("source_count must match source_ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 2 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
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
        if self.highest_planning_score != max(
            entry.planning_score for entry in self.entries
        ):
            raise ValueError("highest_planning_score must match entries")
        if self.overall_planning_score != _overall_planning_score(self.entries):
            raise ValueError("overall_planning_score must match entries")
        if self.overall_planning_posture != _overall_planning_posture(self.entries):
            raise ValueError("overall_planning_posture must match entries")
        declared_source_ids = set(self.source_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if not set(entry.source_ids).issubset(declared_source_ids):
                raise ValueError("entry source_ids must be declared")
        return self


def build_research_planner(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    sources: tuple[OfficialSource, ...] | None = None,
    checked_at: datetime = _DEFAULT_CHECKED_AT,
) -> ResearchPlannerPlan:
    """Build V6.4 Task 2 research planner metadata without executing research."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    source_registry = tuple(sources or approved_official_sources())
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=source_registry,
    )
    source_ids = tuple(source.source_id for source in source_registry)
    return ResearchPlannerPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=checked_at,
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_ids=source_ids,
        source_count=len(source_ids),
        domain_count=len({source.domain for source in source_registry}),
        source_type_count=len({source.source_type for source in source_registry}),
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.entry_id for entry in entries),
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
            entry.entry_id
            for entry in entries
            if entry.hitl_required_before_research_execution
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
            1 for entry in entries if entry.hitl_required_before_research_execution
        ),
        highest_planning_score=max(entry.planning_score for entry in entries),
        overall_planning_score=_overall_planning_score(entries),
        overall_planning_posture=_overall_planning_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_planning_entry_by_id(
    entry_id: str,
    plan: ResearchPlannerPlan | None = None,
) -> ResearchPlanningEntry | None:
    """Return one research planning entry without executing research."""

    source_plan = plan or build_research_planner()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_planning_entries_for_status(
    status: ResearchPlanningStatus,
    plan: ResearchPlannerPlan | None = None,
) -> tuple[ResearchPlanningEntry, ...]:
    """Return research planning entries by advisory status."""

    source_plan = plan or build_research_planner()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_planning_entries_for_confidence(
    confidence: ResearchPlanningConfidence,
    plan: ResearchPlannerPlan | None = None,
) -> tuple[ResearchPlanningEntry, ...]:
    """Return research planning entries by confidence band."""

    source_plan = plan or build_research_planner()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    sources: tuple[OfficialSource, ...],
) -> tuple[ResearchPlanningEntry, ...]:
    return (
        _entry(
            kind="research_scope_framing",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="scope_framing",
            sources=sources,
            scope_clarity_score=90,
            source_strategy_score=80,
            governance_alignment_score=88,
            mutation_risk_score=50,
            governance_weight=110,
        ),
        _entry(
            kind="source_strategy_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_strategy",
            sources=tuple(
                sorted(sources, key=lambda source: (source.priority, source.source_id))[
                    :12
                ]
            ),
            scope_clarity_score=82,
            source_strategy_score=76,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=100,
        ),
        _entry(
            kind="validation_strategy_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="validation_strategy",
            sources=_first_source_per_domain(sources),
            scope_clarity_score=76,
            source_strategy_score=74,
            governance_alignment_score=86,
            mutation_risk_score=42,
            governance_weight=95,
        ),
        _entry(
            kind="distillation_report_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="distillation_reporting",
            sources=tuple(
                sorted(sources, key=lambda source: (source.domain, source.source_id))[
                    :18
                ]
            ),
            scope_clarity_score=68,
            source_strategy_score=66,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="governance_gate_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            sources=sources,
            scope_clarity_score=44,
            source_strategy_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _entry(
    *,
    kind: ResearchPlanningKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchPlanningAxis,
    sources: tuple[OfficialSource, ...],
    scope_clarity_score: int,
    source_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchPlanningEntry:
    source_ids = tuple(source.source_id for source in sources)
    score = _planning_score(
        scope_clarity_score=scope_clarity_score,
        source_strategy_score=source_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchPlanningEntry(
        entry_id=f"research_planner::{kind}",
        planning_kind=kind,
        status=_planning_status(score),
        confidence=_planning_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        planning_axis=axis,
        source_ids=source_ids,
        source_count=len(source_ids),
        domain_count=len({source.domain for source in sources}),
        source_type_count=len({source.source_type for source in sources}),
        planning_summary=_planning_summary(kind),
        scope_clarity_score=scope_clarity_score,
        source_strategy_score=source_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        planning_score=score,
        hitl_required_before_research_execution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, source_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"source_count:{len(source_ids)}",
            f"domain_count:{len({source.domain for source in sources})}",
            f"source_type_count:{len({source.source_type for source in sources})}",
            f"planning_axis:{axis}",
            f"status:{_planning_status(score)}",
            f"confidence:{_planning_confidence(score)}",
            "hitl_required_before_research_execution:true",
        ),
    )


def _first_source_per_domain(
    sources: tuple[OfficialSource, ...],
) -> tuple[OfficialSource, ...]:
    source_by_domain = {
        domain: tuple(
            sorted(
                (source for source in sources if source.domain == domain),
                key=lambda source: (source.priority, source.source_id),
            )
        )
        for domain in official_source_domains()
    }
    return tuple(
        domain_sources[0]
        for domain_sources in source_by_domain.values()
        if domain_sources
    )


def _planning_score(
    *,
    scope_clarity_score: int,
    source_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            scope_clarity_score * 3
            + source_strategy_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _planning_status(score: int) -> ResearchPlanningStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _planning_confidence(score: int) -> ResearchPlanningConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_planning_score(entries: tuple[ResearchPlanningEntry, ...]) -> int:
    return round(sum(entry.planning_score for entry in entries) / len(entries))


def _overall_planning_posture(
    entries: tuple[ResearchPlanningEntry, ...],
) -> ResearchPlanningPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchPlanningEntry, ...],
    status: ResearchPlanningStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchPlanningEntry, ...],
    *confidences: ResearchPlanningConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[ResearchPlanningEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_planning_entries:{len(entries)}",
        "confirm_scope_before_execution",
        "confirm_allowed_source_strategy",
        "confirm_validation_and_provenance_requirements",
        "request_hitl_before_research_execution",
    )


def _entry_actions(kind: ResearchPlanningKind) -> tuple[str, ...]:
    actions: dict[ResearchPlanningKind, tuple[str, ...]] = {
        "research_scope_framing": (
            "review_research_question_scope",
            "identify_required_evidence_classes",
            "confirm_success_criteria",
        ),
        "source_strategy_planning": (
            "review_local_source_strategy",
            "identify_candidate_source_roles",
            "confirm_external_access_policy",
        ),
        "validation_strategy_planning": (
            "review_source_validation_plan",
            "confirm_provenance_requirements",
            "confirm_contradiction_review_policy",
        ),
        "distillation_report_planning": (
            "review_distillation_outline",
            "confirm_report_sections",
            "confirm_confidence_disclosure",
        ),
        "governance_gate_planning": (
            "review_hitl_gate",
            "confirm_no_runtime_mutation",
            "confirm_no_uncontrolled_storage_write",
        ),
    }
    return actions[kind]


def _planning_summary(kind: ResearchPlanningKind) -> str:
    summaries: dict[ResearchPlanningKind, str] = {
        "research_scope_framing": (
            "Frames the research objective, evidence classes, and acceptance "
            "criteria before any execution is allowed."
        ),
        "source_strategy_planning": (
            "Maps local approved-source metadata into an advisory source "
            "strategy without fetching or validating sources live."
        ),
        "validation_strategy_planning": (
            "Plans source validation, provenance capture, contradiction review, "
            "and confidence disclosure without performing those checks."
        ),
        "distillation_report_planning": (
            "Plans distillation and report structure while leaving generated "
            "output and artifact storage untouched."
        ),
        "governance_gate_planning": (
            "Models the HITL gate required before research execution, KB "
            "enrichment, external access, or runtime mutation."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchPlanningKind,
    axis: ResearchPlanningAxis,
) -> tuple[str, ...]:
    return (
        "research_planner",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchPlanningKind,
    source_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"planning_kind:{kind}",
        f"source_metadata_count:{len(source_ids)}",
        "official_source_registry_metadata_only",
        "no_research_execution_performed",
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
