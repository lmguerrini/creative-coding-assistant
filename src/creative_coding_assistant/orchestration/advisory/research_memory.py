"""V6.4 advisory research memory metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.research_reports import (
    RESEARCH_REPORT_PLAN_SERIALIZATION_VERSION,
    ResearchReportPlan,
    build_research_reports,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchMemoryKind = Literal[
    "memory_scope_review",
    "research_session_recall_review",
    "report_memory_linkage_review",
    "memory_retention_policy_review",
    "memory_governance_gate",
]
ResearchMemoryStatus = Literal["candidate", "review_required", "guarded"]
ResearchMemoryConfidence = Literal["low", "medium", "high", "guarded"]
ResearchMemoryPosture = Literal["candidate", "review_required", "guarded"]
ResearchMemoryAxis = Literal[
    "memory_scope",
    "session_recall",
    "report_linkage",
    "retention_policy",
    "governance_gate",
]

RESEARCH_MEMORY_ENTRY_SERIALIZATION_VERSION = "research_memory_entry.v1"
RESEARCH_MEMORY_PLAN_SERIALIZATION_VERSION = "research_memory_plan.v1"

RESEARCH_MEMORY_AUTHORITY_BOUNDARY = (
    "V6.4 Research Memory exposes memory scope, session recall posture, "
    "research report linkage, retention policy, and governance readiness as "
    "inspectable advisory metadata only; it does not create research memory "
    "records, update research memory records, delete research memory records, "
    "execute research memory retrieval, write memory storage, mutate memory "
    "indexes, generate research reports, write report storage, enrich the KB, "
    "write KB storage, write provenance records, execute retrieval, mutate "
    "retrieval configuration, mutate vector indexes, fetch external sources, "
    "browse the web, download papers, validate sources live, score source "
    "credibility, detect contradictions, score research confidence, provision "
    "providers, infer API keys, route providers or models, execute providers, "
    "control workflows, mutate workflow graphs, modify generated output, or "
    "apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Memory",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_memory_record_creation",
    "research_memory_record_update",
    "research_memory_record_deletion",
    "research_memory_retrieval_execution",
    "research_memory_storage_write",
    "memory_index_mutation",
    "research_report_generation",
    "report_storage_write",
    "kb_enrichment_execution",
    "kb_storage_write",
    "provenance_record_write",
    "retrieval_execution",
    "retrieval_configuration_mutation",
    "vector_index_mutation",
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


class ResearchMemoryEntry(BaseModel):
    """One advisory research memory entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    memory_kind: ResearchMemoryKind
    status: ResearchMemoryStatus
    confidence: ResearchMemoryConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    memory_axis: ResearchMemoryAxis
    report_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    report_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    memory_summary: str = Field(min_length=1, max_length=360)
    memory_scope_score: int = Field(ge=0, le=100)
    provenance_linkage_score: int = Field(ge=0, le=100)
    retention_policy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    memory_score: int = Field(ge=0, le=1_000)
    hitl_required_before_memory_mutation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_memory_capability_implemented: Literal[True] = True
    research_memory_metadata_implemented: Literal[True] = True
    research_reports_metadata_used: Literal[True] = True
    research_memory_record_creation_implemented: Literal[False] = False
    research_memory_record_update_implemented: Literal[False] = False
    research_memory_record_deletion_implemented: Literal[False] = False
    research_memory_retrieval_execution_implemented: Literal[False] = False
    research_memory_storage_write_implemented: Literal[False] = False
    memory_index_mutation_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    report_storage_write_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["research_memory_entry.v1"] = (
        RESEARCH_MEMORY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_memory::{self.memory_kind}":
            raise ValueError("entry_id must match memory_kind")
        if self.report_entry_count != len(self.report_entry_ids):
            raise ValueError("report_entry_count must match report ids")
        if self.memory_score != _memory_score(
            memory_scope_score=self.memory_scope_score,
            provenance_linkage_score=self.provenance_linkage_score,
            retention_policy_score=self.retention_policy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("memory_score must combine source scores")
        if self.status != _memory_status(self.memory_score):
            raise ValueError("status must match memory_score")
        if self.confidence != _memory_confidence(self.memory_score):
            raise ValueError("confidence must match memory_score")
        if not self.hitl_required_before_memory_mutation:
            raise ValueError("research memory mutation requires HITL posture")
        return self


class ResearchMemoryPlan(BaseModel):
    """Bounded V6.4 advisory research memory plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_memory"] = "research_memory"
    serialization_version: Literal["research_memory_plan.v1"] = (
        RESEARCH_MEMORY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_MEMORY_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    research_reports_role: Literal["research_reports"] = "research_reports"
    research_reports_serialization_version: Literal["research_report_plan.v1"]
    report_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    report_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchMemoryEntry, ...] = Field(min_length=5, max_length=5)
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
    created_memory_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    updated_memory_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    deleted_memory_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    retrieved_memory_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_memory_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_memory_index_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_memory_score: int = Field(ge=0, le=1_000)
    overall_memory_score: int = Field(ge=0, le=1_000)
    overall_memory_posture: ResearchMemoryPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_memory_capability_implemented: Literal[True] = True
    research_memory_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    research_reports_metadata_used: Literal[True] = True
    research_memory_record_creation_implemented: Literal[False] = False
    research_memory_record_update_implemented: Literal[False] = False
    research_memory_record_deletion_implemented: Literal[False] = False
    research_memory_retrieval_execution_implemented: Literal[False] = False
    research_memory_storage_write_implemented: Literal[False] = False
    memory_index_mutation_implemented: Literal[False] = False
    research_report_generation_implemented: Literal[False] = False
    report_storage_write_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
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
            if entry.hitl_required_before_memory_mutation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.created_memory_record_ids:
            raise ValueError("created_memory_record_ids must remain empty")
        if self.updated_memory_record_ids:
            raise ValueError("updated_memory_record_ids must remain empty")
        if self.deleted_memory_record_ids:
            raise ValueError("deleted_memory_record_ids must remain empty")
        if self.retrieved_memory_record_ids:
            raise ValueError("retrieved_memory_record_ids must remain empty")
        if self.written_memory_storage_record_ids:
            raise ValueError("written_memory_storage_record_ids must remain empty")
        if self.mutated_memory_index_ids:
            raise ValueError("mutated_memory_index_ids must remain empty")
        if self.report_entry_count != len(self.report_entry_ids):
            raise ValueError("report_entry_count must match report ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 10 roadmap")
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
        if self.highest_memory_score != max(
            entry.memory_score for entry in self.entries
        ):
            raise ValueError("highest_memory_score must match entries")
        if self.overall_memory_score != _overall_memory_score(self.entries):
            raise ValueError("overall_memory_score must match entries")
        if self.overall_memory_posture != _overall_memory_posture(self.entries):
            raise ValueError("overall_memory_posture must match entries")
        report_ids = set(self.report_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.report_entry_ids).issubset(report_ids):
                raise ValueError("entry report ids must be declared")
        return self


def build_research_memory(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    reports: ResearchReportPlan | None = None,
) -> ResearchMemoryPlan:
    """Build V6.4 Task 10 memory metadata without mutating memory."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    report_plan = reports or build_research_reports(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        reports=report_plan,
    )
    return ResearchMemoryPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=report_plan.checked_at,
        research_reports_serialization_version=(
            RESEARCH_REPORT_PLAN_SERIALIZATION_VERSION
        ),
        report_entry_ids=report_plan.entry_ids,
        report_entry_count=len(report_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=report_plan.source_count,
        domain_count=report_plan.domain_count,
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
            if entry.hitl_required_before_memory_mutation
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
            1 for entry in entries if entry.hitl_required_before_memory_mutation
        ),
        highest_memory_score=max(entry.memory_score for entry in entries),
        overall_memory_score=_overall_memory_score(entries),
        overall_memory_posture=_overall_memory_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_memory_entry_by_id(
    entry_id: str,
    plan: ResearchMemoryPlan | None = None,
) -> ResearchMemoryEntry | None:
    """Return one research memory entry without mutating memory."""

    source_plan = plan or build_research_memory()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_memory_entries_for_status(
    status: ResearchMemoryStatus,
    plan: ResearchMemoryPlan | None = None,
) -> tuple[ResearchMemoryEntry, ...]:
    """Return research memory entries by advisory status."""

    source_plan = plan or build_research_memory()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_memory_entries_for_confidence(
    confidence: ResearchMemoryConfidence,
    plan: ResearchMemoryPlan | None = None,
) -> tuple[ResearchMemoryEntry, ...]:
    """Return research memory entries by confidence band."""

    source_plan = plan or build_research_memory()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    reports: ResearchReportPlan,
) -> tuple[ResearchMemoryEntry, ...]:
    return (
        _entry(
            kind="memory_scope_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="memory_scope",
            report_entry_ids=reports.entry_ids,
            reports=reports,
            memory_scope_score=88,
            provenance_linkage_score=86,
            retention_policy_score=82,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="research_session_recall_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="session_recall",
            report_entry_ids=(
                "research_reports::report_scope_review",
                "research_reports::evidence_section_mapping_review",
            ),
            reports=reports,
            memory_scope_score=84,
            provenance_linkage_score=82,
            retention_policy_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="report_memory_linkage_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="report_linkage",
            report_entry_ids=(
                "research_reports::source_provenance_appendix_review",
                "research_reports::report_governance_gate",
            ),
            reports=reports,
            memory_scope_score=78,
            provenance_linkage_score=90,
            retention_policy_score=76,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="memory_retention_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="retention_policy",
            report_entry_ids=(
                "research_reports::confidence_disclosure_review",
                "research_reports::source_provenance_appendix_review",
            ),
            reports=reports,
            memory_scope_score=70,
            provenance_linkage_score=72,
            retention_policy_score=82,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="memory_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            report_entry_ids=reports.entry_ids,
            reports=reports,
            memory_scope_score=30,
            provenance_linkage_score=60,
            retention_policy_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: ResearchMemoryKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchMemoryAxis,
    report_entry_ids: tuple[str, ...],
    reports: ResearchReportPlan,
    memory_scope_score: int,
    provenance_linkage_score: int,
    retention_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchMemoryEntry:
    score = _memory_score(
        memory_scope_score=memory_scope_score,
        provenance_linkage_score=provenance_linkage_score,
        retention_policy_score=retention_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchMemoryEntry(
        entry_id=f"research_memory::{kind}",
        memory_kind=kind,
        status=_memory_status(score),
        confidence=_memory_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        memory_axis=axis,
        report_entry_ids=report_entry_ids,
        report_entry_count=len(report_entry_ids),
        source_count=reports.source_count,
        domain_count=reports.domain_count,
        memory_summary=_memory_summary(kind),
        memory_scope_score=memory_scope_score,
        provenance_linkage_score=provenance_linkage_score,
        retention_policy_score=retention_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        memory_score=score,
        hitl_required_before_memory_mutation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, report_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"report_entry_count:{len(report_entry_ids)}",
            f"source_count:{reports.source_count}",
            f"domain_count:{reports.domain_count}",
            f"memory_axis:{axis}",
            f"status:{_memory_status(score)}",
            f"confidence:{_memory_confidence(score)}",
            "hitl_required_before_memory_mutation:true",
        ),
    )


def _memory_score(
    *,
    memory_scope_score: int,
    provenance_linkage_score: int,
    retention_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            memory_scope_score * 2
            + provenance_linkage_score * 3
            + retention_policy_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _memory_status(score: int) -> ResearchMemoryStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _memory_confidence(score: int) -> ResearchMemoryConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_memory_score(entries: tuple[ResearchMemoryEntry, ...]) -> int:
    return round(sum(entry.memory_score for entry in entries) / len(entries))


def _overall_memory_posture(
    entries: tuple[ResearchMemoryEntry, ...],
) -> ResearchMemoryPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchMemoryEntry, ...],
    status: ResearchMemoryStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchMemoryEntry, ...],
    *confidences: ResearchMemoryConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[ResearchMemoryEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_memory_entries:{len(entries)}",
        "confirm_memory_scope_before_mutation",
        "confirm_no_memory_record_write",
        "confirm_no_memory_retrieval_execution",
        "request_hitl_before_research_memory_mutation",
    )


def _entry_actions(kind: ResearchMemoryKind) -> tuple[str, ...]:
    actions: dict[ResearchMemoryKind, tuple[str, ...]] = {
        "memory_scope_review": (
            "review_memory_scope",
            "confirm_research_memory_boundary",
            "confirm_no_memory_record_creation",
        ),
        "research_session_recall_review": (
            "review_session_recall_boundary",
            "confirm_no_memory_retrieval",
            "confirm_no_memory_storage_write",
        ),
        "report_memory_linkage_review": (
            "review_report_memory_linkage",
            "confirm_report_traceability",
            "confirm_no_memory_index_mutation",
        ),
        "memory_retention_policy_review": (
            "review_memory_retention_policy",
            "confirm_retention_requires_hitl",
            "confirm_no_record_update_or_delete",
        ),
        "memory_governance_gate": (
            "review_memory_hitl_gate",
            "confirm_no_memory_mutation",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _memory_summary(kind: ResearchMemoryKind) -> str:
    summaries: dict[ResearchMemoryKind, str] = {
        "memory_scope_review": (
            "Frames research memory scope without creating, updating, or "
            "deleting memory records."
        ),
        "research_session_recall_review": (
            "Models session recall posture without executing memory retrieval "
            "or writing storage."
        ),
        "report_memory_linkage_review": (
            "Describes report-to-memory linkage without mutating memory "
            "indexes or report storage."
        ),
        "memory_retention_policy_review": (
            "Defines retention policy boundaries without applying record "
            "updates, deletes, or lifecycle changes."
        ),
        "memory_governance_gate": (
            "Models the HITL gate required before research memory mutation, "
            "retrieval execution, or persistence."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchMemoryKind,
    axis: ResearchMemoryAxis,
) -> tuple[str, ...]:
    return (
        "research_memory",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchMemoryKind,
    report_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"memory_kind:{kind}",
        f"report_entry_count:{len(report_entry_ids)}",
        "research_reports_metadata_used:true",
        "no_research_memory_mutation_performed",
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
