"""V6.4 advisory research report metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.automatic_kb_enrichment import (
    AUTOMATIC_KB_ENRICHMENT_PLAN_SERIALIZATION_VERSION,
    AutomaticKbEnrichmentPlan,
    build_automatic_kb_enrichment,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchReportKind = Literal[
    "report_scope_review",
    "evidence_section_mapping_review",
    "source_provenance_appendix_review",
    "confidence_disclosure_review",
    "report_governance_gate",
]
ResearchReportStatus = Literal["candidate", "review_required", "guarded"]
ResearchReportConfidence = Literal["low", "medium", "high", "guarded"]
ResearchReportPosture = Literal["candidate", "review_required", "guarded"]
ResearchReportAxis = Literal[
    "report_scope",
    "evidence_mapping",
    "provenance_appendix",
    "confidence_disclosure",
    "governance_gate",
]

RESEARCH_REPORT_ENTRY_SERIALIZATION_VERSION = "research_report_entry.v1"
RESEARCH_REPORT_PLAN_SERIALIZATION_VERSION = "research_report_plan.v1"

RESEARCH_REPORTS_AUTHORITY_BOUNDARY = (
    "V6.4 Research Reports exposes report scope, evidence section mapping, "
    "source provenance appendices, confidence disclosure, and report "
    "governance readiness as inspectable advisory metadata only; it does not "
    "generate research reports, generate report output, modify generated "
    "output, export report files, write report storage, execute automatic KB "
    "enrichment, write KB storage, write provenance records, fetch external "
    "sources, browse the web, download papers, validate sources live, score "
    "source credibility, detect contradictions, score research confidence, "
    "execute retrieval, mutate retrieval configuration, mutate vector "
    "indexes, provision providers, infer API keys, route providers or "
    "models, execute providers, control workflows, mutate workflow graphs, "
    "or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Reports",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_report_generation",
    "report_output_generation",
    "generated_output_modification",
    "file_export_generation",
    "report_storage_write",
    "automatic_kb_enrichment_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "provenance_record_write",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "source_validation_execution",
    "source_credibility_scoring",
    "contradiction_detection_execution",
    "research_confidence_scoring",
    "retrieval_execution",
    "retrieval_configuration_mutation",
    "vector_index_mutation",
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


class ResearchReportEntry(BaseModel):
    """One advisory research report entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    report_kind: ResearchReportKind
    status: ResearchReportStatus
    confidence: ResearchReportConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    report_axis: ResearchReportAxis
    enrichment_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    enrichment_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    report_summary: str = Field(min_length=1, max_length=360)
    scope_readiness_score: int = Field(ge=0, le=100)
    provenance_disclosure_score: int = Field(ge=0, le=100)
    confidence_disclosure_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    report_score: int = Field(ge=0, le=1_000)
    hitl_required_before_report_generation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    research_reports_capability_implemented: Literal[True] = True
    research_reports_metadata_implemented: Literal[True] = True
    automatic_kb_enrichment_metadata_used: Literal[True] = True
    research_report_generation_implemented: Literal[False] = False
    report_output_generation_implemented: Literal[False] = False
    file_export_generation_implemented: Literal[False] = False
    report_storage_write_implemented: Literal[False] = False
    automatic_kb_enrichment_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["research_report_entry.v1"] = (
        RESEARCH_REPORT_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_reports::{self.report_kind}":
            raise ValueError("entry_id must match report_kind")
        if self.enrichment_entry_count != len(self.enrichment_entry_ids):
            raise ValueError("enrichment_entry_count must match enrichment ids")
        if self.report_score != _report_score(
            scope_readiness_score=self.scope_readiness_score,
            provenance_disclosure_score=self.provenance_disclosure_score,
            confidence_disclosure_score=self.confidence_disclosure_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("report_score must combine source scores")
        if self.status != _report_status(self.report_score):
            raise ValueError("status must match report_score")
        if self.confidence != _report_confidence(self.report_score):
            raise ValueError("confidence must match report_score")
        if not self.hitl_required_before_report_generation:
            raise ValueError("research report generation requires HITL posture")
        return self


class ResearchReportPlan(BaseModel):
    """Bounded V6.4 advisory research report plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_reports"] = "research_reports"
    serialization_version: Literal["research_report_plan.v1"] = (
        RESEARCH_REPORT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_REPORTS_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    automatic_kb_enrichment_role: Literal["automatic_kb_enrichment"] = (
        "automatic_kb_enrichment"
    )
    automatic_kb_enrichment_serialization_version: Literal[
        "automatic_kb_enrichment_plan.v1"
    ]
    enrichment_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    enrichment_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchReportEntry, ...] = Field(min_length=5, max_length=5)
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
    generated_report_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_report_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    exported_report_file_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_report_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_generated_output_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_report_score: int = Field(ge=0, le=1_000)
    overall_report_score: int = Field(ge=0, le=1_000)
    overall_report_posture: ResearchReportPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=30,
    )
    research_reports_capability_implemented: Literal[True] = True
    research_reports_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    automatic_kb_enrichment_metadata_used: Literal[True] = True
    research_report_generation_implemented: Literal[False] = False
    report_output_generation_implemented: Literal[False] = False
    file_export_generation_implemented: Literal[False] = False
    report_storage_write_implemented: Literal[False] = False
    automatic_kb_enrichment_execution_implemented: Literal[False] = False
    kb_enrichment_execution_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    source_credibility_scoring_implemented: Literal[False] = False
    contradiction_detection_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    vector_index_mutation_implemented: Literal[False] = False
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
            if entry.hitl_required_before_report_generation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.generated_report_ids:
            raise ValueError("generated_report_ids must remain empty")
        if self.generated_report_output_ids:
            raise ValueError("generated_report_output_ids must remain empty")
        if self.exported_report_file_ids:
            raise ValueError("exported_report_file_ids must remain empty")
        if self.written_report_storage_record_ids:
            raise ValueError("written_report_storage_record_ids must remain empty")
        if self.mutated_generated_output_ids:
            raise ValueError("mutated_generated_output_ids must remain empty")
        if self.enrichment_entry_count != len(self.enrichment_entry_ids):
            raise ValueError("enrichment_entry_count must match enrichment ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 9 roadmap")
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
        if self.highest_report_score != max(
            entry.report_score for entry in self.entries
        ):
            raise ValueError("highest_report_score must match entries")
        if self.overall_report_score != _overall_report_score(self.entries):
            raise ValueError("overall_report_score must match entries")
        if self.overall_report_posture != _overall_report_posture(self.entries):
            raise ValueError("overall_report_posture must match entries")
        enrichment_ids = set(self.enrichment_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.enrichment_entry_ids).issubset(enrichment_ids):
                raise ValueError("entry enrichment ids must be declared")
        return self


def build_research_reports(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    enrichment: AutomaticKbEnrichmentPlan | None = None,
) -> ResearchReportPlan:
    """Build V6.4 Task 9 report metadata without generating reports."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    enrichment_plan = enrichment or build_automatic_kb_enrichment(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        enrichment=enrichment_plan,
    )
    return ResearchReportPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=enrichment_plan.checked_at,
        automatic_kb_enrichment_serialization_version=(
            AUTOMATIC_KB_ENRICHMENT_PLAN_SERIALIZATION_VERSION
        ),
        enrichment_entry_ids=enrichment_plan.entry_ids,
        enrichment_entry_count=len(enrichment_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=enrichment_plan.source_count,
        domain_count=enrichment_plan.domain_count,
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
            if entry.hitl_required_before_report_generation
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
            1 for entry in entries if entry.hitl_required_before_report_generation
        ),
        highest_report_score=max(entry.report_score for entry in entries),
        overall_report_score=_overall_report_score(entries),
        overall_report_posture=_overall_report_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_report_entry_by_id(
    entry_id: str,
    plan: ResearchReportPlan | None = None,
) -> ResearchReportEntry | None:
    """Return one research report entry without generating reports."""

    source_plan = plan or build_research_reports()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_report_entries_for_status(
    status: ResearchReportStatus,
    plan: ResearchReportPlan | None = None,
) -> tuple[ResearchReportEntry, ...]:
    """Return research report entries by advisory status."""

    source_plan = plan or build_research_reports()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_report_entries_for_confidence(
    confidence: ResearchReportConfidence,
    plan: ResearchReportPlan | None = None,
) -> tuple[ResearchReportEntry, ...]:
    """Return research report entries by confidence band."""

    source_plan = plan or build_research_reports()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    enrichment: AutomaticKbEnrichmentPlan,
) -> tuple[ResearchReportEntry, ...]:
    return (
        _entry(
            kind="report_scope_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="report_scope",
            enrichment_entry_ids=enrichment.entry_ids,
            enrichment=enrichment,
            scope_readiness_score=88,
            provenance_disclosure_score=86,
            confidence_disclosure_score=82,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="evidence_section_mapping_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_mapping",
            enrichment_entry_ids=(
                "automatic_kb_enrichment::enrichment_candidate_review",
                "automatic_kb_enrichment::distilled_knowledge_mapping_review",
            ),
            enrichment=enrichment,
            scope_readiness_score=84,
            provenance_disclosure_score=82,
            confidence_disclosure_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="source_provenance_appendix_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_appendix",
            enrichment_entry_ids=(
                "automatic_kb_enrichment::provenance_attachment_review",
                "automatic_kb_enrichment::enrichment_governance_gate",
            ),
            enrichment=enrichment,
            scope_readiness_score=78,
            provenance_disclosure_score=90,
            confidence_disclosure_score=76,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="confidence_disclosure_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="confidence_disclosure",
            enrichment_entry_ids=(
                "automatic_kb_enrichment::kb_write_policy_review",
                "automatic_kb_enrichment::provenance_attachment_review",
            ),
            enrichment=enrichment,
            scope_readiness_score=70,
            provenance_disclosure_score=72,
            confidence_disclosure_score=82,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="report_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            enrichment_entry_ids=enrichment.entry_ids,
            enrichment=enrichment,
            scope_readiness_score=30,
            provenance_disclosure_score=60,
            confidence_disclosure_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: ResearchReportKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchReportAxis,
    enrichment_entry_ids: tuple[str, ...],
    enrichment: AutomaticKbEnrichmentPlan,
    scope_readiness_score: int,
    provenance_disclosure_score: int,
    confidence_disclosure_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchReportEntry:
    score = _report_score(
        scope_readiness_score=scope_readiness_score,
        provenance_disclosure_score=provenance_disclosure_score,
        confidence_disclosure_score=confidence_disclosure_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchReportEntry(
        entry_id=f"research_reports::{kind}",
        report_kind=kind,
        status=_report_status(score),
        confidence=_report_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        report_axis=axis,
        enrichment_entry_ids=enrichment_entry_ids,
        enrichment_entry_count=len(enrichment_entry_ids),
        source_count=enrichment.source_count,
        domain_count=enrichment.domain_count,
        report_summary=_report_summary(kind),
        scope_readiness_score=scope_readiness_score,
        provenance_disclosure_score=provenance_disclosure_score,
        confidence_disclosure_score=confidence_disclosure_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        report_score=score,
        hitl_required_before_report_generation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, enrichment_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"enrichment_entry_count:{len(enrichment_entry_ids)}",
            f"source_count:{enrichment.source_count}",
            f"domain_count:{enrichment.domain_count}",
            f"report_axis:{axis}",
            f"status:{_report_status(score)}",
            f"confidence:{_report_confidence(score)}",
            "hitl_required_before_report_generation:true",
        ),
    )


def _report_score(
    *,
    scope_readiness_score: int,
    provenance_disclosure_score: int,
    confidence_disclosure_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            scope_readiness_score * 2
            + provenance_disclosure_score * 3
            + confidence_disclosure_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _report_status(score: int) -> ResearchReportStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _report_confidence(score: int) -> ResearchReportConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_report_score(entries: tuple[ResearchReportEntry, ...]) -> int:
    return round(sum(entry.report_score for entry in entries) / len(entries))


def _overall_report_posture(
    entries: tuple[ResearchReportEntry, ...],
) -> ResearchReportPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchReportEntry, ...],
    status: ResearchReportStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchReportEntry, ...],
    *confidences: ResearchReportConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[ResearchReportEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_research_report_entries:{len(entries)}",
        "confirm_report_scope_before_generation",
        "confirm_no_report_output_generation",
        "confirm_no_file_export_or_storage_write",
        "request_hitl_before_research_report_generation",
    )


def _entry_actions(kind: ResearchReportKind) -> tuple[str, ...]:
    actions: dict[ResearchReportKind, tuple[str, ...]] = {
        "report_scope_review": (
            "review_report_scope",
            "confirm_audience_and_boundary",
            "confirm_no_report_generation",
        ),
        "evidence_section_mapping_review": (
            "review_evidence_section_mapping",
            "confirm_source_traceability",
            "confirm_no_evidence_rendering",
        ),
        "source_provenance_appendix_review": (
            "review_provenance_appendix",
            "confirm_source_provenance_fields",
            "confirm_no_provenance_write",
        ),
        "confidence_disclosure_review": (
            "review_confidence_disclosure",
            "confirm_uncertainty_language",
            "confirm_no_confidence_scoring",
        ),
        "report_governance_gate": (
            "review_report_hitl_gate",
            "confirm_no_output_mutation",
            "confirm_no_report_storage_write",
        ),
    }
    return actions[kind]


def _report_summary(kind: ResearchReportKind) -> str:
    summaries: dict[ResearchReportKind, str] = {
        "report_scope_review": (
            "Frames research report scope and audience without generating "
            "report output or modifying content."
        ),
        "evidence_section_mapping_review": (
            "Models evidence section mapping without rendering evidence "
            "summaries or exporting files."
        ),
        "source_provenance_appendix_review": (
            "Describes provenance appendix requirements without writing "
            "provenance records or report storage."
        ),
        "confidence_disclosure_review": (
            "Defines confidence disclosure posture without scoring research "
            "confidence or detecting contradictions."
        ),
        "report_governance_gate": (
            "Models the HITL gate required before research report generation, "
            "export, or storage."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchReportKind,
    axis: ResearchReportAxis,
) -> tuple[str, ...]:
    return (
        "research_reports",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchReportKind,
    enrichment_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"report_kind:{kind}",
        f"enrichment_entry_count:{len(enrichment_entry_ids)}",
        "automatic_kb_enrichment_metadata_used:true",
        "no_research_report_generated",
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
