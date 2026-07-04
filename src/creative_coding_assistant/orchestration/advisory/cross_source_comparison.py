"""V6.4 advisory cross-source comparison metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.paper_research import (
    PaperResearchPlan,
    build_paper_research,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.web_research import (
    WebResearchPlan,
    build_web_research,
)

CrossSourceComparisonKind = Literal[
    "paper_web_alignment_review",
    "source_claim_mapping_review",
    "provenance_comparison_review",
    "contradiction_readiness_review",
    "comparison_governance_gate",
]
CrossSourceComparisonStatus = Literal["candidate", "review_required", "guarded"]
CrossSourceComparisonConfidence = Literal["low", "medium", "high", "guarded"]
CrossSourceComparisonPosture = Literal["candidate", "review_required", "guarded"]
CrossSourceComparisonAxis = Literal[
    "paper_web_alignment",
    "claim_mapping",
    "provenance_comparison",
    "contradiction_readiness",
    "governance_gate",
]

CROSS_SOURCE_COMPARISON_ENTRY_SERIALIZATION_VERSION = "cross_source_comparison_entry.v1"
CROSS_SOURCE_COMPARISON_PLAN_SERIALIZATION_VERSION = "cross_source_comparison_plan.v1"
PAPER_RESEARCH_PLAN_SERIALIZATION_VERSION = "paper_research_plan.v1"
WEB_RESEARCH_PLAN_SERIALIZATION_VERSION = "web_research_plan.v1"

CROSS_SOURCE_COMPARISON_AUTHORITY_BOUNDARY = (
    "V6.4 Cross-source Comparison exposes paper/web alignment, source-claim "
    "mapping, provenance comparison, contradiction-readiness, and governance "
    "posture as inspectable advisory metadata only; it does not execute "
    "cross-source comparison, compare live claims, detect contradictions, "
    "score source credibility, score research confidence, fetch external "
    "sources, browse the web, download papers, enrich the KB, write storage, "
    "mutate source registries, mutate retrieval configuration, execute "
    "retrieval, mutate ranking, provision providers, infer API keys, route "
    "providers or models, execute providers, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Cross-source Comparison",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "cross_source_comparison_execution",
    "live_claim_comparison",
    "contradiction_detection_execution",
    "source_credibility_scoring",
    "research_confidence_scoring",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
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


class CrossSourceComparisonEntry(BaseModel):
    """One advisory cross-source comparison entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    comparison_kind: CrossSourceComparisonKind
    status: CrossSourceComparisonStatus
    confidence: CrossSourceComparisonConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    comparison_axis: CrossSourceComparisonAxis
    paper_research_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    web_research_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    paper_research_entry_count: int = Field(ge=1, le=5)
    web_research_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    comparison_summary: str = Field(min_length=1, max_length=360)
    alignment_score: int = Field(ge=0, le=100)
    provenance_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    comparison_score: int = Field(ge=0, le=1_000)
    hitl_required_before_comparison: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    cross_source_comparison_capability_implemented: Literal[True] = True
    cross_source_comparison_metadata_implemented: Literal[True] = True
    paper_research_metadata_used: Literal[True] = True
    web_research_metadata_used: Literal[True] = True
    cross_source_comparison_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
    serialization_version: Literal["cross_source_comparison_entry.v1"] = (
        CROSS_SOURCE_COMPARISON_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"cross_source_comparison::{self.comparison_kind}":
            raise ValueError("entry_id must match comparison_kind")
        if self.paper_research_entry_count != len(self.paper_research_entry_ids):
            raise ValueError("paper_research_entry_count must match paper ids")
        if self.web_research_entry_count != len(self.web_research_entry_ids):
            raise ValueError("web_research_entry_count must match web ids")
        if self.comparison_score != _comparison_score(
            alignment_score=self.alignment_score,
            provenance_score=self.provenance_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("comparison_score must combine source scores")
        if self.status != _comparison_status(self.comparison_score):
            raise ValueError("status must match comparison_score")
        if self.confidence != _comparison_confidence(self.comparison_score):
            raise ValueError("confidence must match comparison_score")
        if not self.hitl_required_before_comparison:
            raise ValueError("cross-source comparison requires HITL posture")
        return self


class CrossSourceComparisonPlan(BaseModel):
    """Bounded V6.4 advisory cross-source comparison plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cross_source_comparison"] = "cross_source_comparison"
    serialization_version: Literal["cross_source_comparison_plan.v1"] = (
        CROSS_SOURCE_COMPARISON_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CROSS_SOURCE_COMPARISON_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    paper_research_role: Literal["paper_research"] = "paper_research"
    web_research_role: Literal["web_research"] = "web_research"
    paper_research_serialization_version: Literal["paper_research_plan.v1"]
    web_research_serialization_version: Literal["web_research_plan.v1"]
    paper_research_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    web_research_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    paper_research_entry_count: int = Field(ge=5, le=5)
    web_research_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[CrossSourceComparisonEntry, ...] = Field(
        min_length=5,
        max_length=5,
    )
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
    executed_comparison_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    compared_live_claim_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    detected_contradiction_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    scored_source_credibility_ids: tuple[str, ...] = Field(
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
    highest_comparison_score: int = Field(ge=0, le=1_000)
    overall_comparison_score: int = Field(ge=0, le=1_000)
    overall_comparison_posture: CrossSourceComparisonPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    cross_source_comparison_capability_implemented: Literal[True] = True
    cross_source_comparison_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    paper_research_metadata_used: Literal[True] = True
    web_research_metadata_used: Literal[True] = True
    cross_source_comparison_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
            if entry.hitl_required_before_comparison
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_comparison_ids:
            raise ValueError("executed_comparison_ids must remain empty")
        if self.compared_live_claim_ids:
            raise ValueError("compared_live_claim_ids must remain empty")
        if self.detected_contradiction_ids:
            raise ValueError("detected_contradiction_ids must remain empty")
        if self.scored_source_credibility_ids:
            raise ValueError("scored_source_credibility_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.paper_research_entry_count != len(self.paper_research_entry_ids):
            raise ValueError("paper_research_entry_count must match paper ids")
        if self.web_research_entry_count != len(self.web_research_entry_ids):
            raise ValueError("web_research_entry_count must match web ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 6 roadmap")
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
        if self.highest_comparison_score != max(
            entry.comparison_score for entry in self.entries
        ):
            raise ValueError("highest_comparison_score must match entries")
        if self.overall_comparison_score != _overall_comparison_score(self.entries):
            raise ValueError("overall_comparison_score must match entries")
        if self.overall_comparison_posture != _overall_comparison_posture(self.entries):
            raise ValueError("overall_comparison_posture must match entries")
        paper_ids = set(self.paper_research_entry_ids)
        web_ids = set(self.web_research_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.paper_research_entry_ids).issubset(paper_ids):
                raise ValueError("entry paper ids must be declared")
            if not set(entry.web_research_entry_ids).issubset(web_ids):
                raise ValueError("entry web ids must be declared")
        return self


def build_cross_source_comparison(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    paper_research: PaperResearchPlan | None = None,
    web_research: WebResearchPlan | None = None,
) -> CrossSourceComparisonPlan:
    """Build V6.4 Task 6 comparison metadata without comparing sources."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    paper = paper_research or build_paper_research(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    web = web_research or build_web_research(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        paper=paper,
        web=web,
    )
    return CrossSourceComparisonPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=paper.checked_at,
        paper_research_serialization_version=paper.serialization_version,
        web_research_serialization_version=web.serialization_version,
        paper_research_entry_ids=paper.entry_ids,
        web_research_entry_ids=web.entry_ids,
        paper_research_entry_count=len(paper.entry_ids),
        web_research_entry_count=len(web.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=max(paper.source_count, web.source_count),
        domain_count=max(paper.domain_count, web.domain_count),
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
            entry.entry_id for entry in entries if entry.hitl_required_before_comparison
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
            1 for entry in entries if entry.hitl_required_before_comparison
        ),
        highest_comparison_score=max(entry.comparison_score for entry in entries),
        overall_comparison_score=_overall_comparison_score(entries),
        overall_comparison_posture=_overall_comparison_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def cross_source_comparison_entry_by_id(
    entry_id: str,
    plan: CrossSourceComparisonPlan | None = None,
) -> CrossSourceComparisonEntry | None:
    """Return one cross-source comparison entry without comparing sources."""

    source_plan = plan or build_cross_source_comparison()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def cross_source_comparison_entries_for_status(
    status: CrossSourceComparisonStatus,
    plan: CrossSourceComparisonPlan | None = None,
) -> tuple[CrossSourceComparisonEntry, ...]:
    """Return cross-source comparison entries by advisory status."""

    source_plan = plan or build_cross_source_comparison()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def cross_source_comparison_entries_for_confidence(
    confidence: CrossSourceComparisonConfidence,
    plan: CrossSourceComparisonPlan | None = None,
) -> tuple[CrossSourceComparisonEntry, ...]:
    """Return cross-source comparison entries by confidence band."""

    source_plan = plan or build_cross_source_comparison()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    paper: PaperResearchPlan,
    web: WebResearchPlan,
) -> tuple[CrossSourceComparisonEntry, ...]:
    return (
        _entry(
            kind="paper_web_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="paper_web_alignment",
            paper_entry_ids=paper.entry_ids,
            web_entry_ids=web.entry_ids,
            paper=paper,
            web=web,
            alignment_score=90,
            provenance_score=80,
            governance_alignment_score=88,
            mutation_risk_score=50,
            governance_weight=110,
        ),
        _entry(
            kind="source_claim_mapping_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="claim_mapping",
            paper_entry_ids=(
                "paper_research::citation_strategy_review",
                "paper_research::paper_source_validation_review",
            ),
            web_entry_ids=(
                "web_research::web_source_strategy_review",
                "web_research::web_validation_strategy_review",
            ),
            paper=paper,
            web=web,
            alignment_score=82,
            provenance_score=76,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=100,
        ),
        _entry(
            kind="provenance_comparison_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_comparison",
            paper_entry_ids=(
                "paper_research::paper_source_validation_review",
                "paper_research::paper_research_governance_gate",
            ),
            web_entry_ids=(
                "web_research::web_validation_strategy_review",
                "web_research::web_research_governance_gate",
            ),
            paper=paper,
            web=web,
            alignment_score=76,
            provenance_score=74,
            governance_alignment_score=86,
            mutation_risk_score=42,
            governance_weight=95,
        ),
        _entry(
            kind="contradiction_readiness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="contradiction_readiness",
            paper_entry_ids=(
                "paper_research::paper_evidence_extraction_plan",
                "paper_research::citation_strategy_review",
            ),
            web_entry_ids=(
                "web_research::web_capture_policy_review",
                "web_research::web_source_strategy_review",
            ),
            paper=paper,
            web=web,
            alignment_score=68,
            provenance_score=66,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="comparison_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            paper_entry_ids=paper.entry_ids,
            web_entry_ids=web.entry_ids,
            paper=paper,
            web=web,
            alignment_score=44,
            provenance_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _entry(
    *,
    kind: CrossSourceComparisonKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: CrossSourceComparisonAxis,
    paper_entry_ids: tuple[str, ...],
    web_entry_ids: tuple[str, ...],
    paper: PaperResearchPlan,
    web: WebResearchPlan,
    alignment_score: int,
    provenance_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> CrossSourceComparisonEntry:
    score = _comparison_score(
        alignment_score=alignment_score,
        provenance_score=provenance_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    source_count = max(paper.source_count, web.source_count)
    domain_count = max(paper.domain_count, web.domain_count)
    return CrossSourceComparisonEntry(
        entry_id=f"cross_source_comparison::{kind}",
        comparison_kind=kind,
        status=_comparison_status(score),
        confidence=_comparison_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        comparison_axis=axis,
        paper_research_entry_ids=paper_entry_ids,
        web_research_entry_ids=web_entry_ids,
        paper_research_entry_count=len(paper_entry_ids),
        web_research_entry_count=len(web_entry_ids),
        source_count=source_count,
        domain_count=domain_count,
        comparison_summary=_comparison_summary(kind),
        alignment_score=alignment_score,
        provenance_score=provenance_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        comparison_score=score,
        hitl_required_before_comparison=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(
            kind,
            paper_entry_ids,
            web_entry_ids,
        ),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"paper_research_entry_count:{len(paper_entry_ids)}",
            f"web_research_entry_count:{len(web_entry_ids)}",
            f"source_count:{source_count}",
            f"domain_count:{domain_count}",
            f"comparison_axis:{axis}",
            f"status:{_comparison_status(score)}",
            f"confidence:{_comparison_confidence(score)}",
            "hitl_required_before_comparison:true",
        ),
    )


def _comparison_score(
    *,
    alignment_score: int,
    provenance_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            alignment_score * 3
            + provenance_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _comparison_status(score: int) -> CrossSourceComparisonStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _comparison_confidence(score: int) -> CrossSourceComparisonConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_comparison_score(
    entries: tuple[CrossSourceComparisonEntry, ...],
) -> int:
    return round(sum(entry.comparison_score for entry in entries) / len(entries))


def _overall_comparison_posture(
    entries: tuple[CrossSourceComparisonEntry, ...],
) -> CrossSourceComparisonPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[CrossSourceComparisonEntry, ...],
    status: CrossSourceComparisonStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[CrossSourceComparisonEntry, ...],
    *confidences: CrossSourceComparisonConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(
    entries: tuple[CrossSourceComparisonEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_cross_source_comparison_entries:{len(entries)}",
        "confirm_comparison_scope_before_execution",
        "confirm_no_live_claim_comparison",
        "confirm_no_contradiction_detection",
        "request_hitl_before_cross_source_comparison",
    )


def _entry_actions(kind: CrossSourceComparisonKind) -> tuple[str, ...]:
    actions: dict[CrossSourceComparisonKind, tuple[str, ...]] = {
        "paper_web_alignment_review": (
            "review_paper_web_alignment_plan",
            "confirm_source_roles",
            "confirm_no_live_comparison",
        ),
        "source_claim_mapping_review": (
            "review_claim_mapping_plan",
            "confirm_claim_grouping_policy",
            "confirm_no_contradiction_detection",
        ),
        "provenance_comparison_review": (
            "review_provenance_comparison_plan",
            "confirm_provenance_fields",
            "confirm_no_source_scoring",
        ),
        "contradiction_readiness_review": (
            "review_contradiction_readiness",
            "confirm_escalation_policy",
            "confirm_no_contradiction_execution",
        ),
        "comparison_governance_gate": (
            "review_comparison_hitl_gate",
            "confirm_no_comparison_execution",
            "confirm_no_storage_write",
        ),
    }
    return actions[kind]


def _comparison_summary(kind: CrossSourceComparisonKind) -> str:
    summaries: dict[CrossSourceComparisonKind, str] = {
        "paper_web_alignment_review": (
            "Frames paper and web source alignment without comparing live "
            "claims or fetching new sources."
        ),
        "source_claim_mapping_review": (
            "Models source-claim mapping readiness without executing claim "
            "comparison or contradiction detection."
        ),
        "provenance_comparison_review": (
            "Describes provenance comparison posture without scoring source "
            "credibility or writing records."
        ),
        "contradiction_readiness_review": (
            "Plans contradiction-readiness review without detecting or "
            "resolving contradictions."
        ),
        "comparison_governance_gate": (
            "Models the HITL gate required before cross-source comparison or "
            "confidence scoring."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: CrossSourceComparisonKind,
    axis: CrossSourceComparisonAxis,
) -> tuple[str, ...]:
    return (
        "cross_source_comparison",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: CrossSourceComparisonKind,
    paper_entry_ids: tuple[str, ...],
    web_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"comparison_kind:{kind}",
        f"paper_entry_count:{len(paper_entry_ids)}",
        f"web_entry_count:{len(web_entry_ids)}",
        "no_cross_source_comparison_performed",
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
