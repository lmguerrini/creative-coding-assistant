"""V6.4 advisory research recommendation engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_gap_discovery import (
    RESEARCH_GAP_PLAN_SERIALIZATION_VERSION,
    ResearchGapDiscoveryPlan,
    build_research_gap_discovery,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchRecommendationKind = Literal[
    "gap_prioritization_review",
    "source_followup_review",
    "confidence_improvement_review",
    "execution_readiness_review",
    "recommendation_governance_gate",
]
ResearchRecommendationStatus = Literal["candidate", "review_required", "guarded"]
ResearchRecommendationConfidence = Literal["low", "medium", "high", "guarded"]
ResearchRecommendationPosture = Literal["candidate", "review_required", "guarded"]
ResearchRecommendationAxis = Literal[
    "gap_prioritization",
    "source_followup",
    "confidence_improvement",
    "execution_readiness",
    "governance_gate",
]

RESEARCH_RECOMMENDATION_ENTRY_SERIALIZATION_VERSION = (
    "research_recommendation_entry.v1"
)
RESEARCH_RECOMMENDATION_PLAN_SERIALIZATION_VERSION = (
    "research_recommendation_plan.v1"
)

RESEARCH_RECOMMENDATION_AUTHORITY_BOUNDARY = (
    "V6.4 Research Recommendation Engine exposes gap-prioritization posture, "
    "source-follow-up readiness, confidence-improvement posture, execution "
    "readiness, and recommendation governance as inspectable advisory "
    "metadata only; it does not generate research recommendations, generate "
    "live recommendations, execute recommendations, write recommendation "
    "records, create research tasks, mutate research plans, execute research "
    "gap discovery, analyze live gaps, execute research confidence scoring, "
    "write confidence records, execute contradiction detection, execute "
    "source credibility scoring, fetch external sources, browse the web, "
    "download papers, mutate source registries, execute retrieval, mutate "
    "retrieval configuration, mutate vector indexes, enrich the KB, write "
    "storage, provision providers, infer API keys, route providers or models, "
    "execute providers, control workflows, mutate workflow graphs, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Recommendation Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_recommendation_generation",
    "live_recommendation_generation",
    "recommendation_execution",
    "recommendation_record_write",
    "research_task_creation",
    "research_plan_mutation",
    "research_gap_discovery_execution",
    "live_gap_analysis",
    "research_confidence_scoring_execution",
    "confidence_record_write",
    "contradiction_detection_execution",
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


class ResearchRecommendationEntry(BaseModel):
    """One advisory research recommendation entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=190)
    recommendation_kind: ResearchRecommendationKind
    status: ResearchRecommendationStatus
    confidence: ResearchRecommendationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    recommendation_axis: ResearchRecommendationAxis
    gap_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    gap_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    recommendation_summary: str = Field(min_length=1, max_length=380)
    gap_priority_score: int = Field(ge=0, le=100)
    source_followup_score: int = Field(ge=0, le=100)
    confidence_improvement_score: int = Field(ge=0, le=100)
    execution_readiness_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    recommendation_score: int = Field(ge=0, le=1_000)
    hitl_required_before_recommendation_generation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_recommendation_engine_capability_implemented: Literal[True] = True
    research_recommendation_engine_metadata_implemented: Literal[True] = True
    research_gap_metadata_used: Literal[True] = True
    research_recommendation_generation_implemented: Literal[False] = False
    live_recommendation_generation_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    recommendation_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    live_gap_analysis_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
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
    serialization_version: Literal["research_recommendation_entry.v1"] = (
        RESEARCH_RECOMMENDATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if (
            self.entry_id
            != f"research_recommendation_engine::{self.recommendation_kind}"
        ):
            raise ValueError("entry_id must match recommendation_kind")
        if self.gap_entry_count != len(self.gap_entry_ids):
            raise ValueError("gap_entry_count must match gap ids")
        if self.recommendation_score != _recommendation_score(
            gap_priority_score=self.gap_priority_score,
            source_followup_score=self.source_followup_score,
            confidence_improvement_score=self.confidence_improvement_score,
            execution_readiness_score=self.execution_readiness_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("recommendation_score must combine source scores")
        if self.status != _recommendation_status(self.recommendation_score):
            raise ValueError("status must match recommendation_score")
        if self.confidence != _recommendation_confidence(
            self.recommendation_score
        ):
            raise ValueError("confidence must match recommendation_score")
        if not self.hitl_required_before_recommendation_generation:
            raise ValueError("research recommendation generation requires HITL")
        return self


class ResearchRecommendationPlan(BaseModel):
    """Bounded V6.4 advisory research recommendation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_recommendation_engine"] = (
        "research_recommendation_engine"
    )
    serialization_version: Literal["research_recommendation_plan.v1"] = (
        RESEARCH_RECOMMENDATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_RECOMMENDATION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_gap_role: Literal["research_gap_discovery"] = "research_gap_discovery"
    research_gap_serialization_version: Literal["research_gap_plan.v1"]
    gap_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gap_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchRecommendationEntry, ...] = Field(
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
    generated_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_live_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_recommendation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_recommendation_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_research_task_ids: tuple[str, ...] = Field(
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
    highest_recommendation_score: int = Field(ge=0, le=1_000)
    overall_recommendation_score: int = Field(ge=0, le=1_000)
    overall_recommendation_posture: ResearchRecommendationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    research_recommendation_engine_capability_implemented: Literal[True] = True
    research_recommendation_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_gap_metadata_used: Literal[True] = True
    research_recommendation_generation_implemented: Literal[False] = False
    live_recommendation_generation_implemented: Literal[False] = False
    recommendation_execution_implemented: Literal[False] = False
    recommendation_record_write_implemented: Literal[False] = False
    research_task_creation_implemented: Literal[False] = False
    research_plan_mutation_implemented: Literal[False] = False
    research_gap_discovery_execution_implemented: Literal[False] = False
    live_gap_analysis_implemented: Literal[False] = False
    research_confidence_scoring_execution_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
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
            if entry.hitl_required_before_recommendation_generation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.generated_recommendation_ids:
            raise ValueError("generated_recommendation_ids must remain empty")
        if self.generated_live_recommendation_ids:
            raise ValueError("generated_live_recommendation_ids must remain empty")
        if self.executed_recommendation_ids:
            raise ValueError("executed_recommendation_ids must remain empty")
        if self.written_recommendation_record_ids:
            raise ValueError("written_recommendation_record_ids must remain empty")
        if self.created_research_task_ids:
            raise ValueError("created_research_task_ids must remain empty")
        if self.mutated_research_plan_ids:
            raise ValueError("mutated_research_plan_ids must remain empty")
        if self.gap_entry_count != len(self.gap_entry_ids):
            raise ValueError("gap_entry_count must match gap ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 16 roadmap")
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
        if self.highest_recommendation_score != max(
            entry.recommendation_score for entry in self.entries
        ):
            raise ValueError("highest_recommendation_score must match entries")
        if self.overall_recommendation_score != _overall_recommendation_score(
            self.entries
        ):
            raise ValueError("overall_recommendation_score must match entries")
        if self.overall_recommendation_posture != _overall_recommendation_posture(
            self.entries
        ):
            raise ValueError("overall_recommendation_posture must match entries")
        gap_ids = set(self.gap_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.gap_entry_ids).issubset(gap_ids):
                raise ValueError("entry gap ids must be declared")
        return self


def build_research_recommendation_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    gaps: ResearchGapDiscoveryPlan | None = None,
) -> ResearchRecommendationPlan:
    """Build V6.4 Task 16 recommendation metadata without generating output."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    gap_plan = gaps or build_research_gap_discovery(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        gaps=gap_plan,
    )
    return ResearchRecommendationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=gap_plan.checked_at,
        research_gap_serialization_version=RESEARCH_GAP_PLAN_SERIALIZATION_VERSION,
        gap_entry_ids=gap_plan.entry_ids,
        gap_entry_count=len(gap_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=gap_plan.source_count,
        domain_count=gap_plan.domain_count,
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
            if entry.hitl_required_before_recommendation_generation
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
            1
            for entry in entries
            if entry.hitl_required_before_recommendation_generation
        ),
        highest_recommendation_score=max(
            entry.recommendation_score for entry in entries
        ),
        overall_recommendation_score=_overall_recommendation_score(entries),
        overall_recommendation_posture=_overall_recommendation_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_recommendation_entry_by_id(
    entry_id: str,
    plan: ResearchRecommendationPlan | None = None,
) -> ResearchRecommendationEntry | None:
    """Return one research recommendation entry without generating output."""

    source_plan = plan or build_research_recommendation_engine()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_recommendation_entries_for_status(
    status: ResearchRecommendationStatus,
    plan: ResearchRecommendationPlan | None = None,
) -> tuple[ResearchRecommendationEntry, ...]:
    """Return recommendation entries by advisory status."""

    source_plan = plan or build_research_recommendation_engine()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_recommendation_entries_for_confidence(
    confidence: ResearchRecommendationConfidence,
    plan: ResearchRecommendationPlan | None = None,
) -> tuple[ResearchRecommendationEntry, ...]:
    """Return recommendation entries by confidence band."""

    source_plan = plan or build_research_recommendation_engine()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    gaps: ResearchGapDiscoveryPlan,
) -> tuple[ResearchRecommendationEntry, ...]:
    return (
        _entry(
            kind="gap_prioritization_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="gap_prioritization",
            gap_entry_ids=gaps.entry_ids,
            gaps=gaps,
            gap_priority_score=88,
            source_followup_score=80,
            confidence_improvement_score=78,
            execution_readiness_score=74,
            governance_alignment_score=84,
            mutation_risk_score=22,
            governance_weight=25,
        ),
        _entry(
            kind="source_followup_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_followup",
            gap_entry_ids=(
                "research_gap_discovery::coverage_gap_review",
                "research_gap_discovery::evidence_gap_review",
            ),
            gaps=gaps,
            gap_priority_score=76,
            source_followup_score=90,
            confidence_improvement_score=72,
            execution_readiness_score=70,
            governance_alignment_score=82,
            mutation_risk_score=20,
            governance_weight=30,
        ),
        _entry(
            kind="confidence_improvement_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="confidence_improvement",
            gap_entry_ids=(
                "research_gap_discovery::methodology_gap_review",
                "research_gap_discovery::gap_governance_gate",
            ),
            gaps=gaps,
            gap_priority_score=74,
            source_followup_score=72,
            confidence_improvement_score=88,
            execution_readiness_score=76,
            governance_alignment_score=80,
            mutation_risk_score=18,
            governance_weight=35,
        ),
        _entry(
            kind="execution_readiness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="execution_readiness",
            gap_entry_ids=(
                "research_gap_discovery::source_diversity_gap_review",
                "research_gap_discovery::coverage_gap_review",
            ),
            gaps=gaps,
            gap_priority_score=70,
            source_followup_score=68,
            confidence_improvement_score=72,
            execution_readiness_score=82,
            governance_alignment_score=76,
            mutation_risk_score=16,
            governance_weight=25,
        ),
        _entry(
            kind="recommendation_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            gap_entry_ids=gaps.entry_ids,
            gaps=gaps,
            gap_priority_score=38,
            source_followup_score=40,
            confidence_improvement_score=42,
            execution_readiness_score=44,
            governance_alignment_score=90,
            mutation_risk_score=10,
            governance_weight=35,
        ),
    )


def _entry(
    *,
    kind: ResearchRecommendationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchRecommendationAxis,
    gap_entry_ids: tuple[str, ...],
    gaps: ResearchGapDiscoveryPlan,
    gap_priority_score: int,
    source_followup_score: int,
    confidence_improvement_score: int,
    execution_readiness_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchRecommendationEntry:
    score = _recommendation_score(
        gap_priority_score=gap_priority_score,
        source_followup_score=source_followup_score,
        confidence_improvement_score=confidence_improvement_score,
        execution_readiness_score=execution_readiness_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchRecommendationEntry(
        entry_id=f"research_recommendation_engine::{kind}",
        recommendation_kind=kind,
        status=_recommendation_status(score),
        confidence=_recommendation_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        recommendation_axis=axis,
        gap_entry_ids=gap_entry_ids,
        gap_entry_count=len(gap_entry_ids),
        source_count=gaps.source_count,
        domain_count=gaps.domain_count,
        recommendation_summary=_recommendation_summary(kind),
        gap_priority_score=gap_priority_score,
        source_followup_score=source_followup_score,
        confidence_improvement_score=confidence_improvement_score,
        execution_readiness_score=execution_readiness_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        recommendation_score=score,
        hitl_required_before_recommendation_generation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, gap_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"gap_entry_count:{len(gap_entry_ids)}",
            f"source_count:{gaps.source_count}",
            f"domain_count:{gaps.domain_count}",
            f"recommendation_axis:{axis}",
            f"status:{_recommendation_status(score)}",
            f"confidence:{_recommendation_confidence(score)}",
            "hitl_required_before_recommendation_generation:true",
        ),
    )


def _recommendation_score(
    *,
    gap_priority_score: int,
    source_followup_score: int,
    confidence_improvement_score: int,
    execution_readiness_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            gap_priority_score * 3
            + source_followup_score * 2
            + confidence_improvement_score * 2
            + execution_readiness_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _recommendation_status(score: int) -> ResearchRecommendationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _recommendation_confidence(score: int) -> ResearchRecommendationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_recommendation_score(
    entries: tuple[ResearchRecommendationEntry, ...],
) -> int:
    return round(sum(entry.recommendation_score for entry in entries) / len(entries))


def _overall_recommendation_posture(
    entries: tuple[ResearchRecommendationEntry, ...],
) -> ResearchRecommendationPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchRecommendationEntry, ...],
    status: ResearchRecommendationStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchRecommendationEntry, ...],
    *confidences: ResearchRecommendationConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[ResearchRecommendationEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_research_recommendation_entries:{len(entries)}",
        "confirm_recommendation_scope_before_generation",
        "confirm_no_research_task_creation",
        "confirm_no_research_plan_mutation",
        "request_hitl_before_recommendation_generation",
    )


def _entry_actions(kind: ResearchRecommendationKind) -> tuple[str, ...]:
    actions: dict[ResearchRecommendationKind, tuple[str, ...]] = {
        "gap_prioritization_review": (
            "review_gap_prioritization_boundary",
            "confirm_gap_priority_inputs_are_advisory",
            "confirm_no_recommendation_generation",
        ),
        "source_followup_review": (
            "review_source_followup_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_source_registry_mutation",
        ),
        "confidence_improvement_review": (
            "review_confidence_improvement_boundary",
            "confirm_no_confidence_scoring",
            "confirm_no_confidence_record_write",
        ),
        "execution_readiness_review": (
            "review_execution_readiness_boundary",
            "confirm_no_recommendation_execution",
            "confirm_no_workflow_control",
        ),
        "recommendation_governance_gate": (
            "review_recommendation_hitl_gate",
            "confirm_no_task_creation",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _recommendation_summary(kind: ResearchRecommendationKind) -> str:
    summaries: dict[ResearchRecommendationKind, str] = {
        "gap_prioritization_review": (
            "Frames gap-prioritization readiness without generating research "
            "recommendations or creating tasks."
        ),
        "source_followup_review": (
            "Models source-follow-up posture without fetching sources or "
            "mutating source registries."
        ),
        "confidence_improvement_review": (
            "Describes confidence-improvement readiness without confidence "
            "scoring execution or confidence record writes."
        ),
        "execution_readiness_review": (
            "Models execution readiness without executing recommendations, "
            "controlling workflows, or mutating workflow graphs."
        ),
        "recommendation_governance_gate": (
            "Models the HITL gate required before recommendation generation, "
            "task creation, research plan mutation, or record writes."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchRecommendationKind,
    axis: ResearchRecommendationAxis,
) -> tuple[str, ...]:
    return (
        "research_recommendation_engine",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchRecommendationKind,
    gap_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"recommendation_kind:{kind}",
        f"gap_entry_count:{len(gap_entry_ids)}",
        "research_gap_metadata_used:true",
        "no_research_recommendations_generated",
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
