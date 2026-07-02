"""V6.4 advisory web research metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_decomposer import (
    ResearchDecomposerPlan,
    build_research_decomposer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

WebResearchKind = Literal[
    "web_scope_planning",
    "web_source_strategy_review",
    "web_validation_strategy_review",
    "web_capture_policy_review",
    "web_research_governance_gate",
]
WebResearchStatus = Literal["candidate", "review_required", "guarded"]
WebResearchConfidence = Literal["low", "medium", "high", "guarded"]
WebResearchPosture = Literal["candidate", "review_required", "guarded"]
WebResearchAxis = Literal[
    "web_scope",
    "source_strategy",
    "validation_strategy",
    "capture_policy",
    "governance_gate",
]

WEB_RESEARCH_ENTRY_SERIALIZATION_VERSION = "web_research_entry.v1"
WEB_RESEARCH_PLAN_SERIALIZATION_VERSION = "web_research_plan.v1"
RESEARCH_DECOMPOSER_PLAN_SERIALIZATION_VERSION = "research_decomposer_plan.v1"

WEB_RESEARCH_AUTHORITY_BOUNDARY = (
    "V6.4 Web Research exposes web scope, source strategy, validation "
    "strategy, capture policy, and governance readiness as inspectable "
    "advisory metadata only; it does not execute web research, browse the "
    "web, crawl sites, fetch external sources, download content, scrape "
    "pages, validate sources live, compute source credibility, enrich the KB, "
    "write storage, mutate source registries, mutate retrieval configuration, "
    "execute retrieval, mutate ranking, provision providers, infer API keys, "
    "route providers or models, execute providers, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Web Research",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "web_research_execution",
    "web_browsing",
    "site_crawl_execution",
    "external_source_fetch",
    "content_download",
    "page_scrape_execution",
    "live_source_validation",
    "source_credibility_scoring",
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
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class WebResearchEntry(BaseModel):
    """One advisory web research entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    web_research_kind: WebResearchKind
    status: WebResearchStatus
    confidence: WebResearchConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    web_axis: WebResearchAxis
    decomposition_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    decomposition_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    web_research_summary: str = Field(min_length=1, max_length=360)
    web_scope_score: int = Field(ge=0, le=100)
    source_strategy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    web_research_score: int = Field(ge=0, le=1_000)
    hitl_required_before_web_research: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    web_research_capability_implemented: Literal[True] = True
    web_research_metadata_implemented: Literal[True] = True
    research_decomposer_metadata_used: Literal[True] = True
    web_research_execution_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    site_crawl_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    content_download_implemented: Literal[False] = False
    page_scrape_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
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
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["web_research_entry.v1"] = (
        WEB_RESEARCH_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"web_research::{self.web_research_kind}":
            raise ValueError("entry_id must match web_research_kind")
        if self.decomposition_entry_count != len(self.decomposition_entry_ids):
            raise ValueError("decomposition_entry_count must match decomposition ids")
        if self.web_research_score != _web_research_score(
            web_scope_score=self.web_scope_score,
            source_strategy_score=self.source_strategy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("web_research_score must combine source scores")
        if self.status != _web_research_status(self.web_research_score):
            raise ValueError("status must match web_research_score")
        if self.confidence != _web_research_confidence(self.web_research_score):
            raise ValueError("confidence must match web_research_score")
        if not self.hitl_required_before_web_research:
            raise ValueError("web research execution requires HITL posture")
        return self


class WebResearchPlan(BaseModel):
    """Bounded V6.4 advisory web research plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["web_research"] = "web_research"
    serialization_version: Literal["web_research_plan.v1"] = (
        WEB_RESEARCH_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WEB_RESEARCH_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_decomposer_role: Literal["research_decomposer"] = (
        "research_decomposer"
    )
    research_decomposer_serialization_version: Literal["research_decomposer_plan.v1"]
    decomposition_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    decomposition_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[WebResearchEntry, ...] = Field(min_length=5, max_length=5)
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
    executed_web_research_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    browsed_web_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    crawled_site_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    fetched_external_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    downloaded_content_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    scraped_page_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_web_research_score: int = Field(ge=0, le=1_000)
    overall_web_research_score: int = Field(ge=0, le=1_000)
    overall_web_research_posture: WebResearchPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    web_research_capability_implemented: Literal[True] = True
    web_research_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_decomposer_metadata_used: Literal[True] = True
    web_research_execution_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    site_crawl_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    content_download_implemented: Literal[False] = False
    page_scrape_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
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
            if entry.hitl_required_before_web_research
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_web_research_ids:
            raise ValueError("executed_web_research_ids must remain empty")
        if self.browsed_web_source_ids:
            raise ValueError("browsed_web_source_ids must remain empty")
        if self.crawled_site_ids:
            raise ValueError("crawled_site_ids must remain empty")
        if self.fetched_external_source_ids:
            raise ValueError("fetched_external_source_ids must remain empty")
        if self.downloaded_content_ids:
            raise ValueError("downloaded_content_ids must remain empty")
        if self.scraped_page_ids:
            raise ValueError("scraped_page_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.decomposition_entry_count != len(self.decomposition_entry_ids):
            raise ValueError("decomposition_entry_count must match decomposition ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 5 roadmap")
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
        if self.highest_web_research_score != max(
            entry.web_research_score for entry in self.entries
        ):
            raise ValueError("highest_web_research_score must match entries")
        if self.overall_web_research_score != _overall_web_research_score(
            self.entries
        ):
            raise ValueError("overall_web_research_score must match entries")
        if self.overall_web_research_posture != _overall_web_research_posture(
            self.entries
        ):
            raise ValueError("overall_web_research_posture must match entries")
        decomposer_entry_ids = set(self.decomposition_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.decomposition_entry_ids).issubset(decomposer_entry_ids):
                raise ValueError("entry decomposition ids must be declared")
        return self


def build_web_research(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    research_decomposer: ResearchDecomposerPlan | None = None,
) -> WebResearchPlan:
    """Build V6.4 Task 5 web research metadata without web access."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    decomposer = research_decomposer or build_research_decomposer(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        decomposer=decomposer,
    )
    return WebResearchPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=decomposer.checked_at,
        research_decomposer_serialization_version=decomposer.serialization_version,
        decomposition_entry_ids=decomposer.entry_ids,
        decomposition_entry_count=len(decomposer.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=decomposer.source_count,
        domain_count=decomposer.domain_count,
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
            if entry.hitl_required_before_web_research
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
            1 for entry in entries if entry.hitl_required_before_web_research
        ),
        highest_web_research_score=max(entry.web_research_score for entry in entries),
        overall_web_research_score=_overall_web_research_score(entries),
        overall_web_research_posture=_overall_web_research_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def web_research_entry_by_id(
    entry_id: str,
    plan: WebResearchPlan | None = None,
) -> WebResearchEntry | None:
    """Return one web research entry without executing web research."""

    source_plan = plan or build_web_research()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def web_research_entries_for_status(
    status: WebResearchStatus,
    plan: WebResearchPlan | None = None,
) -> tuple[WebResearchEntry, ...]:
    """Return web research entries by advisory status."""

    source_plan = plan or build_web_research()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def web_research_entries_for_confidence(
    confidence: WebResearchConfidence,
    plan: WebResearchPlan | None = None,
) -> tuple[WebResearchEntry, ...]:
    """Return web research entries by confidence band."""

    source_plan = plan or build_web_research()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    decomposer: ResearchDecomposerPlan,
) -> tuple[WebResearchEntry, ...]:
    return (
        _entry(
            kind="web_scope_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="web_scope",
            decomposition_entry_ids=decomposer.entry_ids,
            decomposer=decomposer,
            web_scope_score=90,
            source_strategy_score=80,
            governance_alignment_score=88,
            mutation_risk_score=50,
            governance_weight=110,
        ),
        _entry(
            kind="web_source_strategy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_strategy",
            decomposition_entry_ids=(
                "research_decomposer::evidence_thread_decomposition",
                "research_decomposer::source_validation_decomposition",
            ),
            decomposer=decomposer,
            web_scope_score=82,
            source_strategy_score=76,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=100,
        ),
        _entry(
            kind="web_validation_strategy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="validation_strategy",
            decomposition_entry_ids=(
                "research_decomposer::source_validation_decomposition",
                "research_decomposer::decomposition_governance_gate",
            ),
            decomposer=decomposer,
            web_scope_score=76,
            source_strategy_score=74,
            governance_alignment_score=86,
            mutation_risk_score=42,
            governance_weight=95,
        ),
        _entry(
            kind="web_capture_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="capture_policy",
            decomposition_entry_ids=(
                "research_decomposer::distillation_sequence_decomposition",
                "research_decomposer::evidence_thread_decomposition",
            ),
            decomposer=decomposer,
            web_scope_score=68,
            source_strategy_score=66,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="web_research_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            decomposition_entry_ids=decomposer.entry_ids,
            decomposer=decomposer,
            web_scope_score=44,
            source_strategy_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _entry(
    *,
    kind: WebResearchKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: WebResearchAxis,
    decomposition_entry_ids: tuple[str, ...],
    decomposer: ResearchDecomposerPlan,
    web_scope_score: int,
    source_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> WebResearchEntry:
    score = _web_research_score(
        web_scope_score=web_scope_score,
        source_strategy_score=source_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return WebResearchEntry(
        entry_id=f"web_research::{kind}",
        web_research_kind=kind,
        status=_web_research_status(score),
        confidence=_web_research_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        web_axis=axis,
        decomposition_entry_ids=decomposition_entry_ids,
        decomposition_entry_count=len(decomposition_entry_ids),
        source_count=decomposer.source_count,
        domain_count=decomposer.domain_count,
        web_research_summary=_web_research_summary(kind),
        web_scope_score=web_scope_score,
        source_strategy_score=source_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        web_research_score=score,
        hitl_required_before_web_research=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, decomposition_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"decomposition_entry_count:{len(decomposition_entry_ids)}",
            f"source_count:{decomposer.source_count}",
            f"domain_count:{decomposer.domain_count}",
            f"web_axis:{axis}",
            f"status:{_web_research_status(score)}",
            f"confidence:{_web_research_confidence(score)}",
            "hitl_required_before_web_research:true",
        ),
    )


def _web_research_score(
    *,
    web_scope_score: int,
    source_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            web_scope_score * 3
            + source_strategy_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _web_research_status(score: int) -> WebResearchStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _web_research_confidence(score: int) -> WebResearchConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_web_research_score(entries: tuple[WebResearchEntry, ...]) -> int:
    return round(sum(entry.web_research_score for entry in entries) / len(entries))


def _overall_web_research_posture(
    entries: tuple[WebResearchEntry, ...],
) -> WebResearchPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[WebResearchEntry, ...],
    status: WebResearchStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[WebResearchEntry, ...],
    *confidences: WebResearchConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(entries: tuple[WebResearchEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_web_research_entries:{len(entries)}",
        "confirm_web_scope_before_execution",
        "confirm_no_web_browsing",
        "confirm_no_external_fetch",
        "request_hitl_before_web_research",
    )


def _entry_actions(kind: WebResearchKind) -> tuple[str, ...]:
    actions: dict[WebResearchKind, tuple[str, ...]] = {
        "web_scope_planning": (
            "review_web_scope",
            "confirm_allowed_web_sources",
            "confirm_offline_metadata_only",
        ),
        "web_source_strategy_review": (
            "review_web_source_strategy",
            "confirm_source_selection_policy",
            "confirm_no_external_fetch",
        ),
        "web_validation_strategy_review": (
            "review_web_validation_plan",
            "confirm_provenance_requirements",
            "confirm_no_live_source_validation",
        ),
        "web_capture_policy_review": (
            "review_capture_policy",
            "confirm_no_scraping",
            "confirm_no_content_download",
        ),
        "web_research_governance_gate": (
            "review_web_research_hitl_gate",
            "confirm_no_web_browsing",
            "confirm_no_storage_write",
        ),
    }
    return actions[kind]


def _web_research_summary(kind: WebResearchKind) -> str:
    summaries: dict[WebResearchKind, str] = {
        "web_scope_planning": (
            "Frames web research scope and allowed-source criteria without "
            "browsing or fetching external content."
        ),
        "web_source_strategy_review": (
            "Models web source strategy review without crawling sites or "
            "mutating source registries."
        ),
        "web_validation_strategy_review": (
            "Describes web validation posture without live validation or "
            "credibility scoring."
        ),
        "web_capture_policy_review": (
            "Plans capture policy without scraping pages, downloading content, "
            "or writing records."
        ),
        "web_research_governance_gate": (
            "Models the HITL gate required before web research execution or "
            "external web access."
        ),
    }
    return summaries[kind]


def _context_tags(kind: WebResearchKind, axis: WebResearchAxis) -> tuple[str, ...]:
    return (
        "web_research",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: WebResearchKind,
    decomposition_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"web_research_kind:{kind}",
        f"decomposition_entry_count:{len(decomposition_entry_ids)}",
        "research_decomposer_metadata_only",
        "no_web_research_execution_performed",
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
