"""V6.4 advisory source validation engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_memory import (
    RESEARCH_MEMORY_PLAN_SERIALIZATION_VERSION,
    ResearchMemoryPlan,
    build_research_memory,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

SourceValidationKind = Literal[
    "source_presence_validation_review",
    "provenance_completeness_review",
    "source_freshness_validation_review",
    "source_access_policy_review",
    "validation_governance_gate",
]
SourceValidationStatus = Literal["candidate", "review_required", "guarded"]
SourceValidationConfidence = Literal["low", "medium", "high", "guarded"]
SourceValidationPosture = Literal["candidate", "review_required", "guarded"]
SourceValidationAxis = Literal[
    "source_presence",
    "provenance_completeness",
    "freshness_policy",
    "access_policy",
    "governance_gate",
]

SOURCE_VALIDATION_ENTRY_SERIALIZATION_VERSION = "source_validation_entry.v1"
SOURCE_VALIDATION_PLAN_SERIALIZATION_VERSION = "source_validation_plan.v1"

SOURCE_VALIDATION_AUTHORITY_BOUNDARY = (
    "V6.4 Source Validation Engine exposes source presence, provenance "
    "completeness, freshness policy, access policy, and validation governance "
    "readiness as inspectable advisory metadata only; it does not execute "
    "source validation, validate sources live, run source health checks, fetch "
    "external sources, browse the web, download papers, mutate source "
    "registries, write validation records, score source credibility, detect "
    "contradictions, score research confidence, execute retrieval, mutate "
    "retrieval configuration, mutate vector indexes, enrich the KB, write "
    "storage, provision providers, infer API keys, route providers or models, "
    "execute providers, control workflows, mutate workflow graphs, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Source Validation Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "source_validation_execution",
    "live_source_validation",
    "source_health_check_execution",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "source_registry_mutation",
    "validation_record_write",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "research_confidence_scoring",
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


class SourceValidationEntry(BaseModel):
    """One advisory source validation entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    validation_kind: SourceValidationKind
    status: SourceValidationStatus
    confidence: SourceValidationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    validation_axis: SourceValidationAxis
    memory_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    memory_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    validation_summary: str = Field(min_length=1, max_length=360)
    source_presence_score: int = Field(ge=0, le=100)
    provenance_completeness_score: int = Field(ge=0, le=100)
    freshness_policy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    validation_score: int = Field(ge=0, le=1_000)
    hitl_required_before_source_validation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    source_validation_engine_capability_implemented: Literal[True] = True
    source_validation_engine_metadata_implemented: Literal[True] = True
    research_memory_metadata_used: Literal[True] = True
    source_validation_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    validation_record_write_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
    serialization_version: Literal["source_validation_entry.v1"] = (
        SOURCE_VALIDATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"source_validation_engine::{self.validation_kind}":
            raise ValueError("entry_id must match validation_kind")
        if self.memory_entry_count != len(self.memory_entry_ids):
            raise ValueError("memory_entry_count must match memory ids")
        if self.validation_score != _validation_score(
            source_presence_score=self.source_presence_score,
            provenance_completeness_score=self.provenance_completeness_score,
            freshness_policy_score=self.freshness_policy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("validation_score must combine source scores")
        if self.status != _validation_status(self.validation_score):
            raise ValueError("status must match validation_score")
        if self.confidence != _validation_confidence(self.validation_score):
            raise ValueError("confidence must match validation_score")
        if not self.hitl_required_before_source_validation:
            raise ValueError("source validation requires HITL posture")
        return self


class SourceValidationPlan(BaseModel):
    """Bounded V6.4 advisory source validation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["source_validation_engine"] = "source_validation_engine"
    serialization_version: Literal["source_validation_plan.v1"] = (
        SOURCE_VALIDATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SOURCE_VALIDATION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_memory_role: Literal["research_memory"] = "research_memory"
    research_memory_serialization_version: Literal["research_memory_plan.v1"]
    memory_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    memory_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[SourceValidationEntry, ...] = Field(min_length=5, max_length=5)
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
    executed_validation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    live_checked_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_external_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_source_registry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_validation_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_validation_score: int = Field(ge=0, le=1_000)
    overall_validation_score: int = Field(ge=0, le=1_000)
    overall_validation_posture: SourceValidationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    source_validation_engine_capability_implemented: Literal[True] = True
    source_validation_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_memory_metadata_used: Literal[True] = True
    source_validation_execution_implemented: Literal[False] = False
    live_source_validation_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    validation_record_write_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
            if entry.hitl_required_before_source_validation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.executed_validation_ids:
            raise ValueError("executed_validation_ids must remain empty")
        if self.live_checked_source_ids:
            raise ValueError("live_checked_source_ids must remain empty")
        if self.fetched_external_source_ids:
            raise ValueError("fetched_external_source_ids must remain empty")
        if self.mutated_source_registry_ids:
            raise ValueError("mutated_source_registry_ids must remain empty")
        if self.written_validation_record_ids:
            raise ValueError("written_validation_record_ids must remain empty")
        if self.memory_entry_count != len(self.memory_entry_ids):
            raise ValueError("memory_entry_count must match memory ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 11 roadmap")
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
        if self.highest_validation_score != max(
            entry.validation_score for entry in self.entries
        ):
            raise ValueError("highest_validation_score must match entries")
        if self.overall_validation_score != _overall_validation_score(self.entries):
            raise ValueError("overall_validation_score must match entries")
        if self.overall_validation_posture != _overall_validation_posture(self.entries):
            raise ValueError("overall_validation_posture must match entries")
        memory_ids = set(self.memory_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.memory_entry_ids).issubset(memory_ids):
                raise ValueError("entry memory ids must be declared")
        return self


def build_source_validation_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    memory: ResearchMemoryPlan | None = None,
) -> SourceValidationPlan:
    """Build V6.4 Task 11 validation metadata without validating sources."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    memory_plan = memory or build_research_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        memory=memory_plan,
    )
    return SourceValidationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=memory_plan.checked_at,
        research_memory_serialization_version=(
            RESEARCH_MEMORY_PLAN_SERIALIZATION_VERSION
        ),
        memory_entry_ids=memory_plan.entry_ids,
        memory_entry_count=len(memory_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=memory_plan.source_count,
        domain_count=memory_plan.domain_count,
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
            if entry.hitl_required_before_source_validation
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
            1 for entry in entries if entry.hitl_required_before_source_validation
        ),
        highest_validation_score=max(entry.validation_score for entry in entries),
        overall_validation_score=_overall_validation_score(entries),
        overall_validation_posture=_overall_validation_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def source_validation_entry_by_id(
    entry_id: str,
    plan: SourceValidationPlan | None = None,
) -> SourceValidationEntry | None:
    """Return one source validation entry without validating sources."""

    source_plan = plan or build_source_validation_engine()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def source_validation_entries_for_status(
    status: SourceValidationStatus,
    plan: SourceValidationPlan | None = None,
) -> tuple[SourceValidationEntry, ...]:
    """Return source validation entries by advisory status."""

    source_plan = plan or build_source_validation_engine()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def source_validation_entries_for_confidence(
    confidence: SourceValidationConfidence,
    plan: SourceValidationPlan | None = None,
) -> tuple[SourceValidationEntry, ...]:
    """Return source validation entries by confidence band."""

    source_plan = plan or build_source_validation_engine()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    memory: ResearchMemoryPlan,
) -> tuple[SourceValidationEntry, ...]:
    return (
        _entry(
            kind="source_presence_validation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_presence",
            memory_entry_ids=memory.entry_ids,
            memory=memory,
            source_presence_score=88,
            provenance_completeness_score=86,
            freshness_policy_score=82,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="provenance_completeness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_completeness",
            memory_entry_ids=(
                "research_memory::memory_scope_review",
                "research_memory::report_memory_linkage_review",
            ),
            memory=memory,
            source_presence_score=84,
            provenance_completeness_score=82,
            freshness_policy_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="source_freshness_validation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="freshness_policy",
            memory_entry_ids=(
                "research_memory::research_session_recall_review",
                "research_memory::memory_retention_policy_review",
            ),
            memory=memory,
            source_presence_score=78,
            provenance_completeness_score=90,
            freshness_policy_score=76,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="source_access_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="access_policy",
            memory_entry_ids=(
                "research_memory::memory_retention_policy_review",
                "research_memory::memory_governance_gate",
            ),
            memory=memory,
            source_presence_score=70,
            provenance_completeness_score=72,
            freshness_policy_score=82,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="validation_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            memory_entry_ids=memory.entry_ids,
            memory=memory,
            source_presence_score=30,
            provenance_completeness_score=60,
            freshness_policy_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: SourceValidationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: SourceValidationAxis,
    memory_entry_ids: tuple[str, ...],
    memory: ResearchMemoryPlan,
    source_presence_score: int,
    provenance_completeness_score: int,
    freshness_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> SourceValidationEntry:
    score = _validation_score(
        source_presence_score=source_presence_score,
        provenance_completeness_score=provenance_completeness_score,
        freshness_policy_score=freshness_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return SourceValidationEntry(
        entry_id=f"source_validation_engine::{kind}",
        validation_kind=kind,
        status=_validation_status(score),
        confidence=_validation_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        validation_axis=axis,
        memory_entry_ids=memory_entry_ids,
        memory_entry_count=len(memory_entry_ids),
        source_count=memory.source_count,
        domain_count=memory.domain_count,
        validation_summary=_validation_summary(kind),
        source_presence_score=source_presence_score,
        provenance_completeness_score=provenance_completeness_score,
        freshness_policy_score=freshness_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        validation_score=score,
        hitl_required_before_source_validation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, memory_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"memory_entry_count:{len(memory_entry_ids)}",
            f"source_count:{memory.source_count}",
            f"domain_count:{memory.domain_count}",
            f"validation_axis:{axis}",
            f"status:{_validation_status(score)}",
            f"confidence:{_validation_confidence(score)}",
            "hitl_required_before_source_validation:true",
        ),
    )


def _validation_score(
    *,
    source_presence_score: int,
    provenance_completeness_score: int,
    freshness_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_presence_score * 2
            + provenance_completeness_score * 3
            + freshness_policy_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _validation_status(score: int) -> SourceValidationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _validation_confidence(score: int) -> SourceValidationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_validation_score(entries: tuple[SourceValidationEntry, ...]) -> int:
    return round(sum(entry.validation_score for entry in entries) / len(entries))


def _overall_validation_posture(
    entries: tuple[SourceValidationEntry, ...],
) -> SourceValidationPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[SourceValidationEntry, ...],
    status: SourceValidationStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[SourceValidationEntry, ...],
    *confidences: SourceValidationConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[SourceValidationEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_source_validation_entries:{len(entries)}",
        "confirm_validation_scope_before_execution",
        "confirm_no_live_source_validation",
        "confirm_no_source_fetch_or_registry_mutation",
        "request_hitl_before_source_validation",
    )


def _entry_actions(kind: SourceValidationKind) -> tuple[str, ...]:
    actions: dict[SourceValidationKind, tuple[str, ...]] = {
        "source_presence_validation_review": (
            "review_source_presence_policy",
            "confirm_source_inventory_boundary",
            "confirm_no_live_source_check",
        ),
        "provenance_completeness_review": (
            "review_provenance_completeness",
            "confirm_provenance_requirements",
            "confirm_no_validation_record_write",
        ),
        "source_freshness_validation_review": (
            "review_freshness_policy",
            "confirm_no_external_fetch",
            "confirm_no_source_health_check",
        ),
        "source_access_policy_review": (
            "review_source_access_policy",
            "confirm_no_web_or_paper_download",
            "confirm_no_registry_mutation",
        ),
        "validation_governance_gate": (
            "review_validation_hitl_gate",
            "confirm_no_source_validation_execution",
            "confirm_no_source_credibility_scoring",
        ),
    }
    return actions[kind]


def _validation_summary(kind: SourceValidationKind) -> str:
    summaries: dict[SourceValidationKind, str] = {
        "source_presence_validation_review": (
            "Frames source presence validation readiness without live checks "
            "or external source access."
        ),
        "provenance_completeness_review": (
            "Models provenance completeness requirements without writing "
            "validation records."
        ),
        "source_freshness_validation_review": (
            "Defines freshness policy boundaries without fetching sources or "
            "running source health checks."
        ),
        "source_access_policy_review": (
            "Describes source access constraints without browsing, downloads, "
            "or source registry mutation."
        ),
        "validation_governance_gate": (
            "Models the HITL gate required before source validation, "
            "credibility scoring, or confidence scoring."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: SourceValidationKind,
    axis: SourceValidationAxis,
) -> tuple[str, ...]:
    return (
        "source_validation_engine",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: SourceValidationKind,
    memory_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"validation_kind:{kind}",
        f"memory_entry_count:{len(memory_entry_ids)}",
        "research_memory_metadata_used:true",
        "no_source_validation_performed",
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
