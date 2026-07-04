"""V6.4 advisory knowledge distillation metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.cross_source_comparison import (
    CROSS_SOURCE_COMPARISON_PLAN_SERIALIZATION_VERSION,
    CrossSourceComparisonPlan,
    build_cross_source_comparison,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeDistillationKind = Literal[
    "source_claim_distillation_readiness",
    "evidence_abstraction_review",
    "provenance_preservation_review",
    "research_summary_boundary_review",
    "distillation_governance_gate",
]
KnowledgeDistillationStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeDistillationConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeDistillationPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeDistillationAxis = Literal[
    "claim_distillation",
    "evidence_abstraction",
    "provenance_preservation",
    "summary_boundary",
    "governance_gate",
]

KNOWLEDGE_DISTILLATION_ENTRY_SERIALIZATION_VERSION = "knowledge_distillation_entry.v1"
KNOWLEDGE_DISTILLATION_PLAN_SERIALIZATION_VERSION = "knowledge_distillation_plan.v1"

KNOWLEDGE_DISTILLATION_AUTHORITY_BOUNDARY = (
    "V6.4 Knowledge Distillation exposes claim distillation readiness, "
    "evidence abstraction posture, provenance preservation, summary "
    "boundaries, and governance readiness as inspectable advisory metadata "
    "only; it does not execute knowledge distillation, generate distilled "
    "outputs, synthesize claims, summarize evidence, write provenance "
    "records, generate research reports, enrich the KB, write storage, "
    "validate sources live, score source credibility, detect contradictions, "
    "score research confidence, fetch external sources, browse the web, "
    "download papers, mutate source registries, mutate retrieval "
    "configuration, execute retrieval, mutate ranking, provision providers, "
    "infer API keys, route providers or models, execute providers, control "
    "workflows, mutate workflow graphs, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Distillation",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_distillation_execution",
    "distilled_output_generation",
    "generated_output_modification",
    "claim_synthesis_execution",
    "evidence_summarization_execution",
    "provenance_record_write",
    "research_report_generation",
    "kb_enrichment_execution",
    "kb_storage_write",
    "source_validation_execution",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "research_confidence_scoring",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
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
    "runtime_evolution_application",
)


class KnowledgeDistillationEntry(BaseModel):
    """One advisory knowledge distillation entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    distillation_kind: KnowledgeDistillationKind
    status: KnowledgeDistillationStatus
    confidence: KnowledgeDistillationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    distillation_axis: KnowledgeDistillationAxis
    comparison_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    comparison_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    distillation_summary: str = Field(min_length=1, max_length=360)
    source_alignment_score: int = Field(ge=0, le=100)
    provenance_preservation_score: int = Field(ge=0, le=100)
    abstraction_quality_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    distillation_score: int = Field(ge=0, le=1_000)
    hitl_required_before_distillation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_distillation_capability_implemented: Literal[True] = True
    knowledge_distillation_metadata_implemented: Literal[True] = True
    cross_source_comparison_metadata_used: Literal[True] = True
    knowledge_distillation_execution_implemented: Literal[False] = False
    distilled_output_generation_implemented: Literal[False] = False
    claim_synthesis_execution_implemented: Literal[False] = False
    evidence_summarization_execution_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_distillation_entry.v1"] = (
        KNOWLEDGE_DISTILLATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"knowledge_distillation::{self.distillation_kind}":
            raise ValueError("entry_id must match distillation_kind")
        if self.comparison_entry_count != len(self.comparison_entry_ids):
            raise ValueError("comparison_entry_count must match comparison ids")
        if self.distillation_score != _distillation_score(
            source_alignment_score=self.source_alignment_score,
            provenance_preservation_score=self.provenance_preservation_score,
            abstraction_quality_score=self.abstraction_quality_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("distillation_score must combine source scores")
        if self.status != _distillation_status(self.distillation_score):
            raise ValueError("status must match distillation_score")
        if self.confidence != _distillation_confidence(self.distillation_score):
            raise ValueError("confidence must match distillation_score")
        if not self.hitl_required_before_distillation:
            raise ValueError("knowledge distillation requires HITL posture")
        return self


class KnowledgeDistillationPlan(BaseModel):
    """Bounded V6.4 advisory knowledge distillation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_distillation"] = "knowledge_distillation"
    serialization_version: Literal["knowledge_distillation_plan.v1"] = (
        KNOWLEDGE_DISTILLATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_DISTILLATION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    cross_source_comparison_role: Literal["cross_source_comparison"] = (
        "cross_source_comparison"
    )
    cross_source_comparison_serialization_version: Literal[
        "cross_source_comparison_plan.v1"
    ]
    comparison_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    comparison_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[KnowledgeDistillationEntry, ...] = Field(
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
    executed_distillation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_distilled_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    synthesized_claim_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    summarized_evidence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_provenance_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_distillation_score: int = Field(ge=0, le=1_000)
    overall_distillation_score: int = Field(ge=0, le=1_000)
    overall_distillation_posture: KnowledgeDistillationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_distillation_capability_implemented: Literal[True] = True
    knowledge_distillation_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    cross_source_comparison_metadata_used: Literal[True] = True
    knowledge_distillation_execution_implemented: Literal[False] = False
    distilled_output_generation_implemented: Literal[False] = False
    claim_synthesis_execution_implemented: Literal[False] = False
    evidence_summarization_execution_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
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
            if entry.hitl_required_before_distillation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_distillation_ids:
            raise ValueError("executed_distillation_ids must remain empty")
        if self.generated_distilled_output_ids:
            raise ValueError("generated_distilled_output_ids must remain empty")
        if self.synthesized_claim_ids:
            raise ValueError("synthesized_claim_ids must remain empty")
        if self.summarized_evidence_ids:
            raise ValueError("summarized_evidence_ids must remain empty")
        if self.written_provenance_record_ids:
            raise ValueError("written_provenance_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.comparison_entry_count != len(self.comparison_entry_ids):
            raise ValueError("comparison_entry_count must match comparison ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 7 roadmap")
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
        if self.highest_distillation_score != max(
            entry.distillation_score for entry in self.entries
        ):
            raise ValueError("highest_distillation_score must match entries")
        if self.overall_distillation_score != _overall_distillation_score(self.entries):
            raise ValueError("overall_distillation_score must match entries")
        if self.overall_distillation_posture != _overall_distillation_posture(
            self.entries
        ):
            raise ValueError("overall_distillation_posture must match entries")
        comparison_ids = set(self.comparison_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.comparison_entry_ids).issubset(comparison_ids):
                raise ValueError("entry comparison ids must be declared")
        return self


def build_knowledge_distillation(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    comparison: CrossSourceComparisonPlan | None = None,
) -> KnowledgeDistillationPlan:
    """Build V6.4 Task 7 distillation metadata without distilling knowledge."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    comparison_plan = comparison or build_cross_source_comparison(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        comparison=comparison_plan,
    )
    return KnowledgeDistillationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=comparison_plan.checked_at,
        cross_source_comparison_serialization_version=(
            CROSS_SOURCE_COMPARISON_PLAN_SERIALIZATION_VERSION
        ),
        comparison_entry_ids=comparison_plan.entry_ids,
        comparison_entry_count=len(comparison_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=comparison_plan.source_count,
        domain_count=comparison_plan.domain_count,
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
            if entry.hitl_required_before_distillation
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
            1 for entry in entries if entry.hitl_required_before_distillation
        ),
        highest_distillation_score=max(entry.distillation_score for entry in entries),
        overall_distillation_score=_overall_distillation_score(entries),
        overall_distillation_posture=_overall_distillation_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def knowledge_distillation_entry_by_id(
    entry_id: str,
    plan: KnowledgeDistillationPlan | None = None,
) -> KnowledgeDistillationEntry | None:
    """Return one knowledge distillation entry without distilling knowledge."""

    source_plan = plan or build_knowledge_distillation()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def knowledge_distillation_entries_for_status(
    status: KnowledgeDistillationStatus,
    plan: KnowledgeDistillationPlan | None = None,
) -> tuple[KnowledgeDistillationEntry, ...]:
    """Return knowledge distillation entries by advisory status."""

    source_plan = plan or build_knowledge_distillation()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def knowledge_distillation_entries_for_confidence(
    confidence: KnowledgeDistillationConfidence,
    plan: KnowledgeDistillationPlan | None = None,
) -> tuple[KnowledgeDistillationEntry, ...]:
    """Return knowledge distillation entries by confidence band."""

    source_plan = plan or build_knowledge_distillation()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    comparison: CrossSourceComparisonPlan,
) -> tuple[KnowledgeDistillationEntry, ...]:
    return (
        _entry(
            kind="source_claim_distillation_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="claim_distillation",
            comparison_entry_ids=comparison.entry_ids,
            comparison=comparison,
            source_alignment_score=88,
            provenance_preservation_score=88,
            abstraction_quality_score=86,
            governance_alignment_score=90,
            mutation_risk_score=50,
            governance_weight=120,
        ),
        _entry(
            kind="evidence_abstraction_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_abstraction",
            comparison_entry_ids=(
                "cross_source_comparison::source_claim_mapping_review",
                "cross_source_comparison::provenance_comparison_review",
            ),
            comparison=comparison,
            source_alignment_score=82,
            provenance_preservation_score=84,
            abstraction_quality_score=80,
            governance_alignment_score=86,
            mutation_risk_score=44,
            governance_weight=105,
        ),
        _entry(
            kind="provenance_preservation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_preservation",
            comparison_entry_ids=(
                "cross_source_comparison::provenance_comparison_review",
                "cross_source_comparison::comparison_governance_gate",
            ),
            comparison=comparison,
            source_alignment_score=76,
            provenance_preservation_score=90,
            abstraction_quality_score=74,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="research_summary_boundary_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="summary_boundary",
            comparison_entry_ids=(
                "cross_source_comparison::contradiction_readiness_review",
                "cross_source_comparison::paper_web_alignment_review",
            ),
            comparison=comparison,
            source_alignment_score=68,
            provenance_preservation_score=72,
            abstraction_quality_score=70,
            governance_alignment_score=82,
            mutation_risk_score=34,
            governance_weight=85,
        ),
        _entry(
            kind="distillation_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            comparison_entry_ids=comparison.entry_ids,
            comparison=comparison,
            source_alignment_score=30,
            provenance_preservation_score=60,
            abstraction_quality_score=40,
            governance_alignment_score=90,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: KnowledgeDistillationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeDistillationAxis,
    comparison_entry_ids: tuple[str, ...],
    comparison: CrossSourceComparisonPlan,
    source_alignment_score: int,
    provenance_preservation_score: int,
    abstraction_quality_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeDistillationEntry:
    score = _distillation_score(
        source_alignment_score=source_alignment_score,
        provenance_preservation_score=provenance_preservation_score,
        abstraction_quality_score=abstraction_quality_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeDistillationEntry(
        entry_id=f"knowledge_distillation::{kind}",
        distillation_kind=kind,
        status=_distillation_status(score),
        confidence=_distillation_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        distillation_axis=axis,
        comparison_entry_ids=comparison_entry_ids,
        comparison_entry_count=len(comparison_entry_ids),
        source_count=comparison.source_count,
        domain_count=comparison.domain_count,
        distillation_summary=_distillation_summary(kind),
        source_alignment_score=source_alignment_score,
        provenance_preservation_score=provenance_preservation_score,
        abstraction_quality_score=abstraction_quality_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        distillation_score=score,
        hitl_required_before_distillation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, comparison_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"comparison_entry_count:{len(comparison_entry_ids)}",
            f"source_count:{comparison.source_count}",
            f"domain_count:{comparison.domain_count}",
            f"distillation_axis:{axis}",
            f"status:{_distillation_status(score)}",
            f"confidence:{_distillation_confidence(score)}",
            "hitl_required_before_distillation:true",
        ),
    )


def _distillation_score(
    *,
    source_alignment_score: int,
    provenance_preservation_score: int,
    abstraction_quality_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_alignment_score * 2
            + provenance_preservation_score * 3
            + abstraction_quality_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _distillation_status(score: int) -> KnowledgeDistillationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _distillation_confidence(score: int) -> KnowledgeDistillationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_distillation_score(
    entries: tuple[KnowledgeDistillationEntry, ...],
) -> int:
    return round(sum(entry.distillation_score for entry in entries) / len(entries))


def _overall_distillation_posture(
    entries: tuple[KnowledgeDistillationEntry, ...],
) -> KnowledgeDistillationPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[KnowledgeDistillationEntry, ...],
    status: KnowledgeDistillationStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[KnowledgeDistillationEntry, ...],
    *confidences: KnowledgeDistillationConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(
    entries: tuple[KnowledgeDistillationEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_knowledge_distillation_entries:{len(entries)}",
        "confirm_distillation_scope_before_execution",
        "confirm_no_distilled_output_generation",
        "confirm_no_provenance_record_write",
        "request_hitl_before_knowledge_distillation",
    )


def _entry_actions(kind: KnowledgeDistillationKind) -> tuple[str, ...]:
    actions: dict[KnowledgeDistillationKind, tuple[str, ...]] = {
        "source_claim_distillation_readiness": (
            "review_claim_distillation_readiness",
            "confirm_claim_grouping_boundary",
            "confirm_no_claim_synthesis",
        ),
        "evidence_abstraction_review": (
            "review_evidence_abstraction_boundary",
            "confirm_source_evidence_traceability",
            "confirm_no_evidence_summarization",
        ),
        "provenance_preservation_review": (
            "review_provenance_preservation_fields",
            "confirm_source_links_remain_advisory",
            "confirm_no_provenance_write",
        ),
        "research_summary_boundary_review": (
            "review_summary_boundary",
            "confirm_no_report_generation",
            "confirm_no_generated_output_mutation",
        ),
        "distillation_governance_gate": (
            "review_distillation_hitl_gate",
            "confirm_no_distillation_execution",
            "confirm_no_kb_write",
        ),
    }
    return actions[kind]


def _distillation_summary(kind: KnowledgeDistillationKind) -> str:
    summaries: dict[KnowledgeDistillationKind, str] = {
        "source_claim_distillation_readiness": (
            "Frames source-claim distillation readiness without synthesizing "
            "claims or producing distilled output."
        ),
        "evidence_abstraction_review": (
            "Models evidence abstraction posture without summarizing evidence "
            "or mutating generated output."
        ),
        "provenance_preservation_review": (
            "Describes provenance preservation requirements without writing "
            "provenance records or KB storage."
        ),
        "research_summary_boundary_review": (
            "Defines summary boundaries without generating research reports "
            "or modifying generated content."
        ),
        "distillation_governance_gate": (
            "Models the HITL gate required before knowledge distillation, "
            "report generation, or KB enrichment."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: KnowledgeDistillationKind,
    axis: KnowledgeDistillationAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_distillation",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: KnowledgeDistillationKind,
    comparison_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"distillation_kind:{kind}",
        f"comparison_entry_count:{len(comparison_entry_ids)}",
        "cross_source_comparison_metadata_used:true",
        "no_knowledge_distillation_performed",
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
