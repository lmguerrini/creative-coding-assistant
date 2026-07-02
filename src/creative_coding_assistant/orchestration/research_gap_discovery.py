"""V6.4 advisory research gap discovery metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_confidence_engine import (
    RESEARCH_CONFIDENCE_PLAN_SERIALIZATION_VERSION,
    ResearchConfidencePlan,
    build_research_confidence_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchGapKind = Literal[
    "coverage_gap_review",
    "evidence_gap_review",
    "methodology_gap_review",
    "source_diversity_gap_review",
    "gap_governance_gate",
]
ResearchGapStatus = Literal["candidate", "review_required", "guarded"]
ResearchGapConfidence = Literal["low", "medium", "high", "guarded"]
ResearchGapPosture = Literal["candidate", "review_required", "guarded"]
ResearchGapAxis = Literal[
    "coverage_gap",
    "evidence_gap",
    "methodology_gap",
    "source_diversity_gap",
    "governance_gate",
]

RESEARCH_GAP_ENTRY_SERIALIZATION_VERSION = "research_gap_entry.v1"
RESEARCH_GAP_PLAN_SERIALIZATION_VERSION = "research_gap_plan.v1"

RESEARCH_GAP_AUTHORITY_BOUNDARY = (
    "V6.4 Research Gap Discovery exposes coverage-gap posture, evidence-gap "
    "readiness, methodology-gap review, source-diversity gap posture, and gap "
    "governance readiness as inspectable advisory metadata only; it does not "
    "execute research gap discovery, analyze live gaps, write gap records, "
    "create research tasks, generate recommendations, mutate research plans, "
    "execute research confidence scoring, write confidence records, execute "
    "contradiction detection, compare live claims, execute source credibility "
    "scoring, fetch external sources, browse the web, download papers, mutate "
    "source registries, execute retrieval, mutate retrieval configuration, "
    "mutate vector indexes, enrich the KB, write storage, provision "
    "providers, infer API keys, route providers or models, execute providers, "
    "control workflows, mutate workflow graphs, modify generated output, or "
    "apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Gap Discovery",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_gap_discovery_execution",
    "live_gap_analysis",
    "gap_record_write",
    "research_task_creation",
    "research_recommendation_generation",
    "research_plan_mutation",
    "research_confidence_scoring_execution",
    "confidence_record_write",
    "contradiction_detection_execution",
    "live_claim_comparison",
    "source_credibility_scoring_execution",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "source_registry_mutation",
    "retrieval_execution",
    "retrieval_configuration_mutation",
    "vector_index_mutation",
    "kb_enrichment_execution",
    "kb_storage_write",
    "persistent_storage_write",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "provider_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ResearchGapEntry(BaseModel):
    """One advisory research gap discovery entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    gap_kind: ResearchGapKind
    status: ResearchGapStatus
    confidence: ResearchGapConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    gap_axis: ResearchGapAxis
    confidence_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    confidence_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    gap_summary: str = Field(min_length=1, max_length=360)
    coverage_gap_score: int = Field(ge=0, le=100)
    evidence_gap_score: int = Field(ge=0, le=100)
    methodology_gap_score: int = Field(ge=0, le=100)
    source_diversity_gap_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    gap_discovery_score: int = Field(ge=0, le=1_000)
    hitl_required_before_gap_discovery: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_gap_discovery_capability_implemented: Literal[True] = True
    research_gap_discovery_metadata_implemented: Literal[True] = True
    research_confidence_metadata_used: Literal[True] = True
    research_gap_discovery_execution_implemented: Literal[False] = False
    live_gap_analysis_implemented: Literal[False] = False
    gap_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["research_gap_entry.v1"] = (
        RESEARCH_GAP_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_gap_discovery::{self.gap_kind}":
            raise ValueError("entry_id must match gap_kind")
        if self.confidence_entry_count != len(self.confidence_entry_ids):
            raise ValueError("confidence_entry_count must match confidence ids")
        if self.gap_discovery_score != _gap_discovery_score(
            coverage_gap_score=self.coverage_gap_score,
            evidence_gap_score=self.evidence_gap_score,
            methodology_gap_score=self.methodology_gap_score,
            source_diversity_gap_score=self.source_diversity_gap_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("gap_discovery_score must combine source scores")
        if self.status != _gap_status(self.gap_discovery_score):
            raise ValueError("status must match gap_discovery_score")
        if self.confidence != _gap_confidence(self.gap_discovery_score):
            raise ValueError("confidence must match gap_discovery_score")
        if not self.hitl_required_before_gap_discovery:
            raise ValueError("research gap discovery requires HITL posture")
        return self


class ResearchGapDiscoveryPlan(BaseModel):
    """Bounded V6.4 advisory research gap discovery plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_gap_discovery"] = "research_gap_discovery"
    serialization_version: Literal["research_gap_plan.v1"] = (
        RESEARCH_GAP_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_GAP_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_confidence_role: Literal["research_confidence_engine"] = (
        "research_confidence_engine"
    )
    research_confidence_serialization_version: Literal[
        "research_confidence_plan.v1"
    ]
    confidence_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    confidence_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchGapEntry, ...] = Field(min_length=5, max_length=5)
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
    discovered_research_gap_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    analyzed_live_gap_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_gap_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_research_task_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_research_plan_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_gap_discovery_score: int = Field(ge=0, le=1_000)
    overall_gap_discovery_score: int = Field(ge=0, le=1_000)
    overall_gap_discovery_posture: ResearchGapPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_gap_discovery_capability_implemented: Literal[True] = True
    research_gap_discovery_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_confidence_metadata_used: Literal[True] = True
    research_gap_discovery_execution_implemented: Literal[False] = False
    live_gap_analysis_implemented: Literal[False] = False
    gap_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    research_recommendation_generation_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
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
            if entry.hitl_required_before_gap_discovery
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.discovered_research_gap_ids:
            raise ValueError("discovered_research_gap_ids must remain empty")
        if self.analyzed_live_gap_ids:
            raise ValueError("analyzed_live_gap_ids must remain empty")
        if self.written_gap_record_ids:
            raise ValueError("written_gap_record_ids must remain empty")
        if self.created_research_task_ids:
            raise ValueError("created_research_task_ids must remain empty")
        if self.generated_recommendation_ids:
            raise ValueError("generated_recommendation_ids must remain empty")
        if self.mutated_research_plan_ids:
            raise ValueError("mutated_research_plan_ids must remain empty")
        if self.confidence_entry_count != len(self.confidence_entry_ids):
            raise ValueError("confidence_entry_count must match confidence ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 15 roadmap")
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
        if self.highest_gap_discovery_score != max(
            entry.gap_discovery_score for entry in self.entries
        ):
            raise ValueError("highest_gap_discovery_score must match entries")
        if self.overall_gap_discovery_score != _overall_gap_score(self.entries):
            raise ValueError("overall_gap_discovery_score must match entries")
        if self.overall_gap_discovery_posture != _overall_gap_posture(self.entries):
            raise ValueError("overall_gap_discovery_posture must match entries")
        confidence_ids = set(self.confidence_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.confidence_entry_ids).issubset(confidence_ids):
                raise ValueError("entry confidence ids must be declared")
        return self


def build_research_gap_discovery(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    confidence: ResearchConfidencePlan | None = None,
) -> ResearchGapDiscoveryPlan:
    """Build V6.4 Task 15 gap metadata without discovering gaps."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    confidence_plan = confidence or build_research_confidence_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        confidence=confidence_plan,
    )
    return ResearchGapDiscoveryPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=confidence_plan.checked_at,
        research_confidence_serialization_version=(
            RESEARCH_CONFIDENCE_PLAN_SERIALIZATION_VERSION
        ),
        confidence_entry_ids=confidence_plan.entry_ids,
        confidence_entry_count=len(confidence_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=confidence_plan.source_count,
        domain_count=confidence_plan.domain_count,
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
            if entry.hitl_required_before_gap_discovery
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
            1 for entry in entries if entry.hitl_required_before_gap_discovery
        ),
        highest_gap_discovery_score=max(
            entry.gap_discovery_score for entry in entries
        ),
        overall_gap_discovery_score=_overall_gap_score(entries),
        overall_gap_discovery_posture=_overall_gap_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_gap_entry_by_id(
    entry_id: str,
    plan: ResearchGapDiscoveryPlan | None = None,
) -> ResearchGapEntry | None:
    """Return one research gap entry without discovering gaps."""

    source_plan = plan or build_research_gap_discovery()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_gap_entries_for_status(
    status: ResearchGapStatus,
    plan: ResearchGapDiscoveryPlan | None = None,
) -> tuple[ResearchGapEntry, ...]:
    """Return research gap entries by advisory status."""

    source_plan = plan or build_research_gap_discovery()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_gap_entries_for_confidence(
    confidence: ResearchGapConfidence,
    plan: ResearchGapDiscoveryPlan | None = None,
) -> tuple[ResearchGapEntry, ...]:
    """Return research gap entries by confidence band."""

    source_plan = plan or build_research_gap_discovery()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    confidence: ResearchConfidencePlan,
) -> tuple[ResearchGapEntry, ...]:
    return (
        _entry(
            kind="coverage_gap_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="coverage_gap",
            confidence_entry_ids=confidence.entry_ids,
            confidence=confidence,
            coverage_gap_score=88,
            evidence_gap_score=82,
            methodology_gap_score=76,
            source_diversity_gap_score=74,
            governance_alignment_score=84,
            mutation_risk_score=22,
            governance_weight=25,
        ),
        _entry(
            kind="evidence_gap_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_gap",
            confidence_entry_ids=(
                "research_confidence_engine::evidence_strength_review",
                "research_confidence_engine::source_reliability_review",
            ),
            confidence=confidence,
            coverage_gap_score=78,
            evidence_gap_score=90,
            methodology_gap_score=74,
            source_diversity_gap_score=70,
            governance_alignment_score=82,
            mutation_risk_score=20,
            governance_weight=30,
        ),
        _entry(
            kind="methodology_gap_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="methodology_gap",
            confidence_entry_ids=(
                "research_confidence_engine::coverage_completeness_review",
                "research_confidence_engine::contradiction_risk_review",
            ),
            confidence=confidence,
            coverage_gap_score=76,
            evidence_gap_score=74,
            methodology_gap_score=88,
            source_diversity_gap_score=72,
            governance_alignment_score=80,
            mutation_risk_score=18,
            governance_weight=35,
        ),
        _entry(
            kind="source_diversity_gap_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_diversity_gap",
            confidence_entry_ids=(
                "research_confidence_engine::source_reliability_review",
                "research_confidence_engine::coverage_completeness_review",
            ),
            confidence=confidence,
            coverage_gap_score=70,
            evidence_gap_score=72,
            methodology_gap_score=68,
            source_diversity_gap_score=80,
            governance_alignment_score=76,
            mutation_risk_score=16,
            governance_weight=25,
        ),
        _entry(
            kind="gap_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            confidence_entry_ids=confidence.entry_ids,
            confidence=confidence,
            coverage_gap_score=38,
            evidence_gap_score=40,
            methodology_gap_score=42,
            source_diversity_gap_score=44,
            governance_alignment_score=90,
            mutation_risk_score=10,
            governance_weight=35,
        ),
    )


def _entry(
    *,
    kind: ResearchGapKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchGapAxis,
    confidence_entry_ids: tuple[str, ...],
    confidence: ResearchConfidencePlan,
    coverage_gap_score: int,
    evidence_gap_score: int,
    methodology_gap_score: int,
    source_diversity_gap_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchGapEntry:
    score = _gap_discovery_score(
        coverage_gap_score=coverage_gap_score,
        evidence_gap_score=evidence_gap_score,
        methodology_gap_score=methodology_gap_score,
        source_diversity_gap_score=source_diversity_gap_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchGapEntry(
        entry_id=f"research_gap_discovery::{kind}",
        gap_kind=kind,
        status=_gap_status(score),
        confidence=_gap_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        gap_axis=axis,
        confidence_entry_ids=confidence_entry_ids,
        confidence_entry_count=len(confidence_entry_ids),
        source_count=confidence.source_count,
        domain_count=confidence.domain_count,
        gap_summary=_gap_summary(kind),
        coverage_gap_score=coverage_gap_score,
        evidence_gap_score=evidence_gap_score,
        methodology_gap_score=methodology_gap_score,
        source_diversity_gap_score=source_diversity_gap_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        gap_discovery_score=score,
        hitl_required_before_gap_discovery=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, confidence_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"confidence_entry_count:{len(confidence_entry_ids)}",
            f"source_count:{confidence.source_count}",
            f"domain_count:{confidence.domain_count}",
            f"gap_axis:{axis}",
            f"status:{_gap_status(score)}",
            f"confidence:{_gap_confidence(score)}",
            "hitl_required_before_gap_discovery:true",
        ),
    )


def _gap_discovery_score(
    *,
    coverage_gap_score: int,
    evidence_gap_score: int,
    methodology_gap_score: int,
    source_diversity_gap_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            coverage_gap_score * 3
            + evidence_gap_score * 2
            + methodology_gap_score * 2
            + source_diversity_gap_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _gap_status(score: int) -> ResearchGapStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _gap_confidence(score: int) -> ResearchGapConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_gap_score(entries: tuple[ResearchGapEntry, ...]) -> int:
    return round(sum(entry.gap_discovery_score for entry in entries) / len(entries))


def _overall_gap_posture(
    entries: tuple[ResearchGapEntry, ...],
) -> ResearchGapPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchGapEntry, ...],
    status: ResearchGapStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchGapEntry, ...],
    *confidences: ResearchGapConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(entries: tuple[ResearchGapEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_gap_entries:{len(entries)}",
        "confirm_gap_scope_before_discovery",
        "confirm_no_research_task_creation",
        "confirm_no_recommendation_generation",
        "request_hitl_before_gap_discovery",
    )


def _entry_actions(kind: ResearchGapKind) -> tuple[str, ...]:
    actions: dict[ResearchGapKind, tuple[str, ...]] = {
        "coverage_gap_review": (
            "review_coverage_gap_boundary",
            "confirm_coverage_inputs_are_advisory",
            "confirm_no_gap_discovery_execution",
        ),
        "evidence_gap_review": (
            "review_evidence_gap_boundary",
            "confirm_evidence_gap_metadata",
            "confirm_no_external_source_fetch",
        ),
        "methodology_gap_review": (
            "review_methodology_gap_boundary",
            "confirm_no_research_plan_mutation",
            "confirm_no_task_creation",
        ),
        "source_diversity_gap_review": (
            "review_source_diversity_gap_boundary",
            "confirm_no_source_registry_mutation",
            "confirm_no_retrieval_execution",
        ),
        "gap_governance_gate": (
            "review_gap_discovery_hitl_gate",
            "confirm_no_recommendation_generation",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _gap_summary(kind: ResearchGapKind) -> str:
    summaries: dict[ResearchGapKind, str] = {
        "coverage_gap_review": (
            "Frames coverage-gap readiness without discovering gaps or "
            "creating research tasks."
        ),
        "evidence_gap_review": (
            "Models evidence-gap posture without fetching external sources or "
            "executing retrieval."
        ),
        "methodology_gap_review": (
            "Describes methodology-gap readiness without mutating research "
            "plans or workflows."
        ),
        "source_diversity_gap_review": (
            "Models source-diversity gap posture without source registry or "
            "retrieval configuration mutation."
        ),
        "gap_governance_gate": (
            "Models the HITL gate required before gap discovery, task "
            "creation, recommendation generation, or record writes."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchGapKind,
    axis: ResearchGapAxis,
) -> tuple[str, ...]:
    return (
        "research_gap_discovery",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchGapKind,
    confidence_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"gap_kind:{kind}",
        f"confidence_entry_count:{len(confidence_entry_ids)}",
        "research_confidence_metadata_used:true",
        "no_research_gap_discovery_performed",
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
