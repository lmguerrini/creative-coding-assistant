"""V6.4 advisory paper research metadata."""

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

PaperResearchKind = Literal[
    "literature_scope_planning",
    "citation_strategy_review",
    "paper_source_validation_review",
    "paper_evidence_extraction_plan",
    "paper_research_governance_gate",
]
PaperResearchStatus = Literal["candidate", "review_required", "guarded"]
PaperResearchConfidence = Literal["low", "medium", "high", "guarded"]
PaperResearchPosture = Literal["candidate", "review_required", "guarded"]
PaperResearchAxis = Literal[
    "literature_scope",
    "citation_strategy",
    "source_validation",
    "evidence_extraction",
    "governance_gate",
]

PAPER_RESEARCH_ENTRY_SERIALIZATION_VERSION = "paper_research_entry.v1"
PAPER_RESEARCH_PLAN_SERIALIZATION_VERSION = "paper_research_plan.v1"
RESEARCH_DECOMPOSER_PLAN_SERIALIZATION_VERSION = "research_decomposer_plan.v1"

PAPER_RESEARCH_AUTHORITY_BOUNDARY = (
    "V6.4 Paper Research exposes literature scope, citation strategy, paper "
    "source validation posture, evidence extraction planning, and governance "
    "readiness as inspectable advisory metadata only; it does not execute "
    "paper research, search external paper indexes, download papers, fetch "
    "external sources, parse PDFs, extract citations, validate sources live, "
    "score source credibility, enrich the KB, write storage, mutate retrieval "
    "configuration, execute retrieval, mutate ranking, provision providers, "
    "infer API keys, route providers or models, execute providers, control "
    "workflows, mutate workflow graphs, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Paper Research",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "paper_research_execution",
    "external_paper_search",
    "paper_download",
    "external_source_fetch",
    "pdf_parse_execution",
    "citation_extraction_execution",
    "live_source_validation",
    "source_credibility_scoring",
    "kb_enrichment_execution",
    "kb_storage_write",
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


class PaperResearchEntry(BaseModel):
    """One advisory paper research entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    paper_research_kind: PaperResearchKind
    status: PaperResearchStatus
    confidence: PaperResearchConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    paper_axis: PaperResearchAxis
    decomposition_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    decomposition_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    paper_research_summary: str = Field(min_length=1, max_length=360)
    literature_scope_score: int = Field(ge=0, le=100)
    citation_strategy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    paper_research_score: int = Field(ge=0, le=1_000)
    hitl_required_before_paper_research: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    paper_research_capability_implemented: Literal[True] = True
    paper_research_metadata_implemented: Literal[True] = True
    research_decomposer_metadata_used: Literal[True] = True
    paper_research_execution_implemented: Literal[False] = False
    external_paper_search_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    pdf_parse_execution_implemented: Literal[False] = False
    citation_extraction_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
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
    serialization_version: Literal["paper_research_entry.v1"] = (
        PAPER_RESEARCH_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"paper_research::{self.paper_research_kind}":
            raise ValueError("entry_id must match paper_research_kind")
        if self.decomposition_entry_count != len(self.decomposition_entry_ids):
            raise ValueError("decomposition_entry_count must match decomposition ids")
        if self.paper_research_score != _paper_research_score(
            literature_scope_score=self.literature_scope_score,
            citation_strategy_score=self.citation_strategy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("paper_research_score must combine source scores")
        if self.status != _paper_research_status(self.paper_research_score):
            raise ValueError("status must match paper_research_score")
        if self.confidence != _paper_research_confidence(self.paper_research_score):
            raise ValueError("confidence must match paper_research_score")
        if not self.hitl_required_before_paper_research:
            raise ValueError("paper research execution requires HITL posture")
        return self


class PaperResearchPlan(BaseModel):
    """Bounded V6.4 advisory paper research plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["paper_research"] = "paper_research"
    serialization_version: Literal["paper_research_plan.v1"] = (
        PAPER_RESEARCH_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PAPER_RESEARCH_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_decomposer_role: Literal["research_decomposer"] = "research_decomposer"
    research_decomposer_serialization_version: Literal["research_decomposer_plan.v1"]
    decomposition_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    decomposition_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[PaperResearchEntry, ...] = Field(min_length=5, max_length=5)
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
    executed_paper_research_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    searched_external_paper_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    downloaded_paper_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    parsed_pdf_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    extracted_citation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_paper_research_score: int = Field(ge=0, le=1_000)
    overall_paper_research_score: int = Field(ge=0, le=1_000)
    overall_paper_research_posture: PaperResearchPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    paper_research_capability_implemented: Literal[True] = True
    paper_research_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_decomposer_metadata_used: Literal[True] = True
    paper_research_execution_implemented: Literal[False] = False
    external_paper_search_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    pdf_parse_execution_implemented: Literal[False] = False
    citation_extraction_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
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
            if entry.hitl_required_before_paper_research
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_paper_research_ids:
            raise ValueError("executed_paper_research_ids must remain empty")
        if self.searched_external_paper_ids:
            raise ValueError("searched_external_paper_ids must remain empty")
        if self.downloaded_paper_ids:
            raise ValueError("downloaded_paper_ids must remain empty")
        if self.parsed_pdf_ids:
            raise ValueError("parsed_pdf_ids must remain empty")
        if self.extracted_citation_ids:
            raise ValueError("extracted_citation_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.decomposition_entry_count != len(self.decomposition_entry_ids):
            raise ValueError("decomposition_entry_count must match decomposition ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 4 roadmap")
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
        if self.highest_paper_research_score != max(
            entry.paper_research_score for entry in self.entries
        ):
            raise ValueError("highest_paper_research_score must match entries")
        if self.overall_paper_research_score != _overall_paper_research_score(
            self.entries
        ):
            raise ValueError("overall_paper_research_score must match entries")
        if self.overall_paper_research_posture != _overall_paper_research_posture(
            self.entries
        ):
            raise ValueError("overall_paper_research_posture must match entries")
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


def build_paper_research(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    research_decomposer: ResearchDecomposerPlan | None = None,
) -> PaperResearchPlan:
    """Build V6.4 Task 4 paper research metadata without paper access."""

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
    return PaperResearchPlan(
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
            if entry.hitl_required_before_paper_research
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
            1 for entry in entries if entry.hitl_required_before_paper_research
        ),
        highest_paper_research_score=max(
            entry.paper_research_score for entry in entries
        ),
        overall_paper_research_score=_overall_paper_research_score(entries),
        overall_paper_research_posture=_overall_paper_research_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def paper_research_entry_by_id(
    entry_id: str,
    plan: PaperResearchPlan | None = None,
) -> PaperResearchEntry | None:
    """Return one paper research entry without executing paper research."""

    source_plan = plan or build_paper_research()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def paper_research_entries_for_status(
    status: PaperResearchStatus,
    plan: PaperResearchPlan | None = None,
) -> tuple[PaperResearchEntry, ...]:
    """Return paper research entries by advisory status."""

    source_plan = plan or build_paper_research()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def paper_research_entries_for_confidence(
    confidence: PaperResearchConfidence,
    plan: PaperResearchPlan | None = None,
) -> tuple[PaperResearchEntry, ...]:
    """Return paper research entries by confidence band."""

    source_plan = plan or build_paper_research()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    decomposer: ResearchDecomposerPlan,
) -> tuple[PaperResearchEntry, ...]:
    return (
        _entry(
            kind="literature_scope_planning",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="literature_scope",
            decomposition_entry_ids=decomposer.entry_ids,
            decomposer=decomposer,
            literature_scope_score=90,
            citation_strategy_score=80,
            governance_alignment_score=88,
            mutation_risk_score=50,
            governance_weight=110,
        ),
        _entry(
            kind="citation_strategy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="citation_strategy",
            decomposition_entry_ids=(
                "research_decomposer::evidence_thread_decomposition",
                "research_decomposer::source_validation_decomposition",
            ),
            decomposer=decomposer,
            literature_scope_score=82,
            citation_strategy_score=76,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=100,
        ),
        _entry(
            kind="paper_source_validation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_validation",
            decomposition_entry_ids=(
                "research_decomposer::source_validation_decomposition",
                "research_decomposer::decomposition_governance_gate",
            ),
            decomposer=decomposer,
            literature_scope_score=76,
            citation_strategy_score=74,
            governance_alignment_score=86,
            mutation_risk_score=42,
            governance_weight=95,
        ),
        _entry(
            kind="paper_evidence_extraction_plan",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_extraction",
            decomposition_entry_ids=(
                "research_decomposer::distillation_sequence_decomposition",
                "research_decomposer::evidence_thread_decomposition",
            ),
            decomposer=decomposer,
            literature_scope_score=68,
            citation_strategy_score=66,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="paper_research_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            decomposition_entry_ids=decomposer.entry_ids,
            decomposer=decomposer,
            literature_scope_score=44,
            citation_strategy_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _entry(
    *,
    kind: PaperResearchKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: PaperResearchAxis,
    decomposition_entry_ids: tuple[str, ...],
    decomposer: ResearchDecomposerPlan,
    literature_scope_score: int,
    citation_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> PaperResearchEntry:
    score = _paper_research_score(
        literature_scope_score=literature_scope_score,
        citation_strategy_score=citation_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return PaperResearchEntry(
        entry_id=f"paper_research::{kind}",
        paper_research_kind=kind,
        status=_paper_research_status(score),
        confidence=_paper_research_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        paper_axis=axis,
        decomposition_entry_ids=decomposition_entry_ids,
        decomposition_entry_count=len(decomposition_entry_ids),
        source_count=decomposer.source_count,
        domain_count=decomposer.domain_count,
        paper_research_summary=_paper_research_summary(kind),
        literature_scope_score=literature_scope_score,
        citation_strategy_score=citation_strategy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        paper_research_score=score,
        hitl_required_before_paper_research=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, decomposition_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"decomposition_entry_count:{len(decomposition_entry_ids)}",
            f"source_count:{decomposer.source_count}",
            f"domain_count:{decomposer.domain_count}",
            f"paper_axis:{axis}",
            f"status:{_paper_research_status(score)}",
            f"confidence:{_paper_research_confidence(score)}",
            "hitl_required_before_paper_research:true",
        ),
    )


def _paper_research_score(
    *,
    literature_scope_score: int,
    citation_strategy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            literature_scope_score * 3
            + citation_strategy_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _paper_research_status(score: int) -> PaperResearchStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _paper_research_confidence(score: int) -> PaperResearchConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_paper_research_score(entries: tuple[PaperResearchEntry, ...]) -> int:
    return round(sum(entry.paper_research_score for entry in entries) / len(entries))


def _overall_paper_research_posture(
    entries: tuple[PaperResearchEntry, ...],
) -> PaperResearchPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[PaperResearchEntry, ...],
    status: PaperResearchStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[PaperResearchEntry, ...],
    *confidences: PaperResearchConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[PaperResearchEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_paper_research_entries:{len(entries)}",
        "confirm_paper_scope_before_execution",
        "confirm_no_external_paper_search",
        "confirm_no_paper_download",
        "request_hitl_before_paper_research",
    )


def _entry_actions(kind: PaperResearchKind) -> tuple[str, ...]:
    actions: dict[PaperResearchKind, tuple[str, ...]] = {
        "literature_scope_planning": (
            "review_literature_scope",
            "confirm_allowed_paper_sources",
            "confirm_offline_metadata_only",
        ),
        "citation_strategy_review": (
            "review_citation_strategy",
            "confirm_citation_quality_criteria",
            "confirm_no_citation_extraction",
        ),
        "paper_source_validation_review": (
            "review_paper_validation_plan",
            "confirm_provenance_requirements",
            "confirm_no_live_source_validation",
        ),
        "paper_evidence_extraction_plan": (
            "review_evidence_extraction_outline",
            "confirm_no_pdf_parse",
            "confirm_no_generated_output_mutation",
        ),
        "paper_research_governance_gate": (
            "review_paper_research_hitl_gate",
            "confirm_no_paper_download",
            "confirm_no_storage_write",
        ),
    }
    return actions[kind]


def _paper_research_summary(kind: PaperResearchKind) -> str:
    summaries: dict[PaperResearchKind, str] = {
        "literature_scope_planning": (
            "Frames literature scope and paper-source criteria without "
            "searching external paper indexes."
        ),
        "citation_strategy_review": (
            "Models citation strategy review without extracting citations or "
            "fetching paper records."
        ),
        "paper_source_validation_review": (
            "Describes paper source validation posture without live validation "
            "or credibility scoring."
        ),
        "paper_evidence_extraction_plan": (
            "Plans paper evidence extraction without downloading papers, "
            "parsing PDFs, or writing records."
        ),
        "paper_research_governance_gate": (
            "Models the HITL gate required before paper research execution or "
            "external paper access."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: PaperResearchKind,
    axis: PaperResearchAxis,
) -> tuple[str, ...]:
    return (
        "paper_research",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: PaperResearchKind,
    decomposition_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"paper_research_kind:{kind}",
        f"decomposition_entry_count:{len(decomposition_entry_ids)}",
        "research_decomposer_metadata_only",
        "no_paper_research_execution_performed",
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
