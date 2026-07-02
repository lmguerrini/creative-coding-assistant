"""V6.4 advisory automatic KB enrichment metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_distillation import (
    KNOWLEDGE_DISTILLATION_PLAN_SERIALIZATION_VERSION,
    KnowledgeDistillationPlan,
    build_knowledge_distillation,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

AutomaticKbEnrichmentKind = Literal[
    "enrichment_candidate_review",
    "distilled_knowledge_mapping_review",
    "provenance_attachment_review",
    "kb_write_policy_review",
    "enrichment_governance_gate",
]
AutomaticKbEnrichmentStatus = Literal["candidate", "review_required", "guarded"]
AutomaticKbEnrichmentConfidence = Literal["low", "medium", "high", "guarded"]
AutomaticKbEnrichmentPosture = Literal["candidate", "review_required", "guarded"]
AutomaticKbEnrichmentAxis = Literal[
    "candidate_selection",
    "knowledge_mapping",
    "provenance_attachment",
    "write_policy",
    "governance_gate",
]

AUTOMATIC_KB_ENRICHMENT_ENTRY_SERIALIZATION_VERSION = (
    "automatic_kb_enrichment_entry.v1"
)
AUTOMATIC_KB_ENRICHMENT_PLAN_SERIALIZATION_VERSION = (
    "automatic_kb_enrichment_plan.v1"
)

AUTOMATIC_KB_ENRICHMENT_AUTHORITY_BOUNDARY = (
    "V6.4 Automatic KB Enrichment exposes enrichment candidate selection, "
    "distilled knowledge mapping, provenance attachment, KB write-policy "
    "readiness, and governance posture as inspectable advisory metadata only; "
    "it does not execute automatic KB enrichment, create KB records, update "
    "KB records, delete KB records, write KB storage, write provenance "
    "records, mutate source registries, mutate retrieval configuration, "
    "execute retrieval, mutate vector indexes, upsert vectors, refresh "
    "embeddings, execute knowledge distillation, generate research reports, "
    "fetch external sources, browse the web, download papers, validate "
    "sources live, score source credibility, detect contradictions, score "
    "research confidence, provision providers, infer API keys, route "
    "providers or models, execute providers, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Automatic KB Enrichment",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "automatic_kb_enrichment_execution",
    "kb_enrichment_execution",
    "kb_record_creation",
    "kb_record_update",
    "kb_record_deletion",
    "kb_storage_write",
    "provenance_record_write",
    "source_registry_mutation",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "vector_index_mutation",
    "vector_upsert",
    "embedding_refresh_execution",
    "knowledge_distillation_execution",
    "research_report_generation",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "source_validation_execution",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "research_confidence_scoring",
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


class AutomaticKbEnrichmentEntry(BaseModel):
    """One advisory automatic KB enrichment entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    enrichment_kind: AutomaticKbEnrichmentKind
    status: AutomaticKbEnrichmentStatus
    confidence: AutomaticKbEnrichmentConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    enrichment_axis: AutomaticKbEnrichmentAxis
    distillation_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    distillation_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    enrichment_summary: str = Field(min_length=1, max_length=360)
    candidate_quality_score: int = Field(ge=0, le=100)
    provenance_readiness_score: int = Field(ge=0, le=100)
    storage_safety_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    enrichment_score: int = Field(ge=0, le=1_000)
    hitl_required_before_enrichment: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    automatic_kb_enrichment_capability_implemented: Literal[True] = True
    automatic_kb_enrichment_metadata_implemented: Literal[True] = True
    knowledge_distillation_metadata_used: Literal[True] = True
    automatic_kb_enrichment_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_record_creation_implemented: Literal[False] = False
    kb_record_update_implemented: Literal[False] = False
    kb_record_deletion_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
    serialization_version: Literal["automatic_kb_enrichment_entry.v1"] = (
        AUTOMATIC_KB_ENRICHMENT_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"automatic_kb_enrichment::{self.enrichment_kind}":
            raise ValueError("entry_id must match enrichment_kind")
        if self.distillation_entry_count != len(self.distillation_entry_ids):
            raise ValueError("distillation_entry_count must match distillation ids")
        if self.enrichment_score != _enrichment_score(
            candidate_quality_score=self.candidate_quality_score,
            provenance_readiness_score=self.provenance_readiness_score,
            storage_safety_score=self.storage_safety_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("enrichment_score must combine source scores")
        if self.status != _enrichment_status(self.enrichment_score):
            raise ValueError("status must match enrichment_score")
        if self.confidence != _enrichment_confidence(self.enrichment_score):
            raise ValueError("confidence must match enrichment_score")
        if not self.hitl_required_before_enrichment:
            raise ValueError("automatic KB enrichment requires HITL posture")
        return self


class AutomaticKbEnrichmentPlan(BaseModel):
    """Bounded V6.4 advisory automatic KB enrichment plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["automatic_kb_enrichment"] = "automatic_kb_enrichment"
    serialization_version: Literal["automatic_kb_enrichment_plan.v1"] = (
        AUTOMATIC_KB_ENRICHMENT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AUTOMATIC_KB_ENRICHMENT_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_distillation_role: Literal["knowledge_distillation"] = (
        "knowledge_distillation"
    )
    knowledge_distillation_serialization_version: Literal[
        "knowledge_distillation_plan.v1"
    ]
    distillation_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    distillation_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[AutomaticKbEnrichmentEntry, ...] = Field(
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
    executed_enrichment_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    updated_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    deleted_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_config_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_vector_index_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_enrichment_score: int = Field(ge=0, le=1_000)
    overall_enrichment_score: int = Field(ge=0, le=1_000)
    overall_enrichment_posture: AutomaticKbEnrichmentPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    automatic_kb_enrichment_capability_implemented: Literal[True] = True
    automatic_kb_enrichment_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_distillation_metadata_used: Literal[True] = True
    automatic_kb_enrichment_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_record_creation_implemented: Literal[False] = False
    kb_record_update_implemented: Literal[False] = False
    kb_record_deletion_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    knowledge_distillation_execution_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
            if entry.hitl_required_before_enrichment
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_enrichment_ids:
            raise ValueError("executed_enrichment_ids must remain empty")
        if self.created_kb_record_ids:
            raise ValueError("created_kb_record_ids must remain empty")
        if self.updated_kb_record_ids:
            raise ValueError("updated_kb_record_ids must remain empty")
        if self.deleted_kb_record_ids:
            raise ValueError("deleted_kb_record_ids must remain empty")
        if self.written_kb_storage_record_ids:
            raise ValueError("written_kb_storage_record_ids must remain empty")
        if self.mutated_retrieval_config_ids:
            raise ValueError("mutated_retrieval_config_ids must remain empty")
        if self.mutated_vector_index_ids:
            raise ValueError("mutated_vector_index_ids must remain empty")
        if self.distillation_entry_count != len(self.distillation_entry_ids):
            raise ValueError("distillation_entry_count must match distillation ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 8 roadmap")
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
        if self.highest_enrichment_score != max(
            entry.enrichment_score for entry in self.entries
        ):
            raise ValueError("highest_enrichment_score must match entries")
        if self.overall_enrichment_score != _overall_enrichment_score(self.entries):
            raise ValueError("overall_enrichment_score must match entries")
        if self.overall_enrichment_posture != _overall_enrichment_posture(
            self.entries
        ):
            raise ValueError("overall_enrichment_posture must match entries")
        distillation_ids = set(self.distillation_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.distillation_entry_ids).issubset(distillation_ids):
                raise ValueError("entry distillation ids must be declared")
        return self


def build_automatic_kb_enrichment(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    distillation: KnowledgeDistillationPlan | None = None,
) -> AutomaticKbEnrichmentPlan:
    """Build V6.4 Task 8 enrichment metadata without enriching the KB."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    distillation_plan = distillation or build_knowledge_distillation(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        distillation=distillation_plan,
    )
    return AutomaticKbEnrichmentPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=distillation_plan.checked_at,
        knowledge_distillation_serialization_version=(
            KNOWLEDGE_DISTILLATION_PLAN_SERIALIZATION_VERSION
        ),
        distillation_entry_ids=distillation_plan.entry_ids,
        distillation_entry_count=len(distillation_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=distillation_plan.source_count,
        domain_count=distillation_plan.domain_count,
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
            if entry.hitl_required_before_enrichment
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
            1 for entry in entries if entry.hitl_required_before_enrichment
        ),
        highest_enrichment_score=max(entry.enrichment_score for entry in entries),
        overall_enrichment_score=_overall_enrichment_score(entries),
        overall_enrichment_posture=_overall_enrichment_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def automatic_kb_enrichment_entry_by_id(
    entry_id: str,
    plan: AutomaticKbEnrichmentPlan | None = None,
) -> AutomaticKbEnrichmentEntry | None:
    """Return one automatic KB enrichment entry without enriching the KB."""

    source_plan = plan or build_automatic_kb_enrichment()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def automatic_kb_enrichment_entries_for_status(
    status: AutomaticKbEnrichmentStatus,
    plan: AutomaticKbEnrichmentPlan | None = None,
) -> tuple[AutomaticKbEnrichmentEntry, ...]:
    """Return automatic KB enrichment entries by advisory status."""

    source_plan = plan or build_automatic_kb_enrichment()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def automatic_kb_enrichment_entries_for_confidence(
    confidence: AutomaticKbEnrichmentConfidence,
    plan: AutomaticKbEnrichmentPlan | None = None,
) -> tuple[AutomaticKbEnrichmentEntry, ...]:
    """Return automatic KB enrichment entries by confidence band."""

    source_plan = plan or build_automatic_kb_enrichment()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    distillation: KnowledgeDistillationPlan,
) -> tuple[AutomaticKbEnrichmentEntry, ...]:
    return (
        _entry(
            kind="enrichment_candidate_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="candidate_selection",
            distillation_entry_ids=distillation.entry_ids,
            distillation=distillation,
            candidate_quality_score=86,
            provenance_readiness_score=88,
            storage_safety_score=84,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="distilled_knowledge_mapping_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="knowledge_mapping",
            distillation_entry_ids=(
                "knowledge_distillation::source_claim_distillation_readiness",
                "knowledge_distillation::evidence_abstraction_review",
            ),
            distillation=distillation,
            candidate_quality_score=82,
            provenance_readiness_score=84,
            storage_safety_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="provenance_attachment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_attachment",
            distillation_entry_ids=(
                "knowledge_distillation::provenance_preservation_review",
                "knowledge_distillation::distillation_governance_gate",
            ),
            distillation=distillation,
            candidate_quality_score=76,
            provenance_readiness_score=90,
            storage_safety_score=74,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="kb_write_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="write_policy",
            distillation_entry_ids=(
                "knowledge_distillation::research_summary_boundary_review",
                "knowledge_distillation::provenance_preservation_review",
            ),
            distillation=distillation,
            candidate_quality_score=66,
            provenance_readiness_score=72,
            storage_safety_score=78,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="enrichment_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            distillation_entry_ids=distillation.entry_ids,
            distillation=distillation,
            candidate_quality_score=30,
            provenance_readiness_score=60,
            storage_safety_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: AutomaticKbEnrichmentKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: AutomaticKbEnrichmentAxis,
    distillation_entry_ids: tuple[str, ...],
    distillation: KnowledgeDistillationPlan,
    candidate_quality_score: int,
    provenance_readiness_score: int,
    storage_safety_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> AutomaticKbEnrichmentEntry:
    score = _enrichment_score(
        candidate_quality_score=candidate_quality_score,
        provenance_readiness_score=provenance_readiness_score,
        storage_safety_score=storage_safety_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return AutomaticKbEnrichmentEntry(
        entry_id=f"automatic_kb_enrichment::{kind}",
        enrichment_kind=kind,
        status=_enrichment_status(score),
        confidence=_enrichment_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        enrichment_axis=axis,
        distillation_entry_ids=distillation_entry_ids,
        distillation_entry_count=len(distillation_entry_ids),
        source_count=distillation.source_count,
        domain_count=distillation.domain_count,
        enrichment_summary=_enrichment_summary(kind),
        candidate_quality_score=candidate_quality_score,
        provenance_readiness_score=provenance_readiness_score,
        storage_safety_score=storage_safety_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        enrichment_score=score,
        hitl_required_before_enrichment=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, distillation_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"distillation_entry_count:{len(distillation_entry_ids)}",
            f"source_count:{distillation.source_count}",
            f"domain_count:{distillation.domain_count}",
            f"enrichment_axis:{axis}",
            f"status:{_enrichment_status(score)}",
            f"confidence:{_enrichment_confidence(score)}",
            "hitl_required_before_enrichment:true",
        ),
    )


def _enrichment_score(
    *,
    candidate_quality_score: int,
    provenance_readiness_score: int,
    storage_safety_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            candidate_quality_score * 2
            + provenance_readiness_score * 3
            + storage_safety_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _enrichment_status(score: int) -> AutomaticKbEnrichmentStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _enrichment_confidence(score: int) -> AutomaticKbEnrichmentConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_enrichment_score(
    entries: tuple[AutomaticKbEnrichmentEntry, ...],
) -> int:
    return round(sum(entry.enrichment_score for entry in entries) / len(entries))


def _overall_enrichment_posture(
    entries: tuple[AutomaticKbEnrichmentEntry, ...],
) -> AutomaticKbEnrichmentPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[AutomaticKbEnrichmentEntry, ...],
    status: AutomaticKbEnrichmentStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[AutomaticKbEnrichmentEntry, ...],
    *confidences: AutomaticKbEnrichmentConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[AutomaticKbEnrichmentEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_automatic_kb_enrichment_entries:{len(entries)}",
        "confirm_enrichment_scope_before_execution",
        "confirm_no_kb_record_mutation",
        "confirm_no_vector_or_retrieval_mutation",
        "request_hitl_before_automatic_kb_enrichment",
    )


def _entry_actions(kind: AutomaticKbEnrichmentKind) -> tuple[str, ...]:
    actions: dict[AutomaticKbEnrichmentKind, tuple[str, ...]] = {
        "enrichment_candidate_review": (
            "review_enrichment_candidate_set",
            "confirm_candidate_traceability",
            "confirm_no_kb_record_creation",
        ),
        "distilled_knowledge_mapping_review": (
            "review_distilled_knowledge_mapping",
            "confirm_distillation_source_links",
            "confirm_no_kb_record_update",
        ),
        "provenance_attachment_review": (
            "review_provenance_attachment_boundary",
            "confirm_provenance_remains_advisory",
            "confirm_no_provenance_write",
        ),
        "kb_write_policy_review": (
            "review_kb_write_policy",
            "confirm_no_storage_write",
            "confirm_no_retrieval_mutation",
        ),
        "enrichment_governance_gate": (
            "review_enrichment_hitl_gate",
            "confirm_no_enrichment_execution",
            "confirm_no_vector_index_mutation",
        ),
    }
    return actions[kind]


def _enrichment_summary(kind: AutomaticKbEnrichmentKind) -> str:
    summaries: dict[AutomaticKbEnrichmentKind, str] = {
        "enrichment_candidate_review": (
            "Frames KB enrichment candidate readiness without creating, "
            "updating, or deleting KB records."
        ),
        "distilled_knowledge_mapping_review": (
            "Models distilled knowledge mapping without writing KB storage "
            "or changing retrieval configuration."
        ),
        "provenance_attachment_review": (
            "Describes provenance attachment posture without writing "
            "provenance records or mutating source registries."
        ),
        "kb_write_policy_review": (
            "Defines KB write-policy constraints without executing storage, "
            "retrieval, embedding, or vector operations."
        ),
        "enrichment_governance_gate": (
            "Models the HITL gate required before automatic KB enrichment or "
            "any persistent knowledge-store mutation."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: AutomaticKbEnrichmentKind,
    axis: AutomaticKbEnrichmentAxis,
) -> tuple[str, ...]:
    return (
        "automatic_kb_enrichment",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: AutomaticKbEnrichmentKind,
    distillation_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"enrichment_kind:{kind}",
        f"distillation_entry_count:{len(distillation_entry_ids)}",
        "knowledge_distillation_metadata_used:true",
        "no_automatic_kb_enrichment_performed",
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
