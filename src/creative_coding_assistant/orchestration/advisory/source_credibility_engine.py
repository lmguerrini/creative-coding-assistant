"""V6.4 advisory source credibility engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.source_validation_engine import (
    SOURCE_VALIDATION_PLAN_SERIALIZATION_VERSION,
    SourceValidationPlan,
    build_source_validation_engine,
)

SourceCredibilityKind = Literal[
    "authority_signal_review",
    "provenance_reliability_review",
    "validation_coverage_review",
    "bias_risk_review",
    "credibility_governance_gate",
]
SourceCredibilityStatus = Literal["candidate", "review_required", "guarded"]
SourceCredibilityConfidence = Literal["low", "medium", "high", "guarded"]
SourceCredibilityPosture = Literal["candidate", "review_required", "guarded"]
SourceCredibilityAxis = Literal[
    "authority_signal",
    "provenance_reliability",
    "validation_coverage",
    "bias_risk",
    "governance_gate",
]

SOURCE_CREDIBILITY_ENTRY_SERIALIZATION_VERSION = "source_credibility_entry.v1"
SOURCE_CREDIBILITY_PLAN_SERIALIZATION_VERSION = "source_credibility_plan.v1"

SOURCE_CREDIBILITY_AUTHORITY_BOUNDARY = (
    "V6.4 Source Credibility Engine exposes authority-signal posture, "
    "provenance reliability, validation coverage, bias-risk review, and "
    "credibility governance readiness as inspectable advisory metadata only; "
    "it does not execute source credibility scoring, score sources live, "
    "mutate credibility rankings, write credibility records, execute source "
    "validation, fetch external sources, browse the web, download papers, "
    "mutate source registries, detect contradictions, score research "
    "confidence, execute retrieval, mutate retrieval configuration, mutate "
    "vector indexes, enrich the KB, write storage, provision providers, infer "
    "API keys, route providers or models, execute providers, control "
    "workflows, mutate workflow graphs, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Source Credibility Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "source_credibility_scoring_execution",
    "live_source_credibility_scoring",
    "credibility_ranking_mutation",
    "credibility_record_write",
    "source_validation_execution",
    "external_source_fetch",
    "web_browsing",
    "paper_download",
    "source_registry_mutation",
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


class SourceCredibilityEntry(BaseModel):
    """One advisory source credibility entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    credibility_kind: SourceCredibilityKind
    status: SourceCredibilityStatus
    confidence: SourceCredibilityConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    credibility_axis: SourceCredibilityAxis
    validation_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    validation_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    credibility_summary: str = Field(min_length=1, max_length=360)
    authority_signal_score: int = Field(ge=0, le=100)
    provenance_reliability_score: int = Field(ge=0, le=100)
    validation_coverage_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    credibility_score: int = Field(ge=0, le=1_000)
    hitl_required_before_credibility_scoring: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    source_credibility_engine_capability_implemented: Literal[True] = True
    source_credibility_engine_metadata_implemented: Literal[True] = True
    source_validation_metadata_used: Literal[True] = True
    source_credibility_scoring_execution_implemented: Literal[False] = False
    live_source_credibility_scoring_implemented: Literal[False] = False
    credibility_ranking_mutation_implemented: Literal[False] = False
    credibility_record_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["source_credibility_entry.v1"] = (
        SOURCE_CREDIBILITY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"source_credibility_engine::{self.credibility_kind}":
            raise ValueError("entry_id must match credibility_kind")
        if self.validation_entry_count != len(self.validation_entry_ids):
            raise ValueError("validation_entry_count must match validation ids")
        if self.credibility_score != _credibility_score(
            authority_signal_score=self.authority_signal_score,
            provenance_reliability_score=self.provenance_reliability_score,
            validation_coverage_score=self.validation_coverage_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("credibility_score must combine source scores")
        if self.status != _credibility_status(self.credibility_score):
            raise ValueError("status must match credibility_score")
        if self.confidence != _credibility_confidence(self.credibility_score):
            raise ValueError("confidence must match credibility_score")
        if not self.hitl_required_before_credibility_scoring:
            raise ValueError("source credibility scoring requires HITL posture")
        return self


class SourceCredibilityPlan(BaseModel):
    """Bounded V6.4 advisory source credibility plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["source_credibility_engine"] = "source_credibility_engine"
    serialization_version: Literal["source_credibility_plan.v1"] = (
        SOURCE_CREDIBILITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SOURCE_CREDIBILITY_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_validation_role: Literal["source_validation_engine"] = (
        "source_validation_engine"
    )
    source_validation_serialization_version: Literal["source_validation_plan.v1"]
    validation_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    validation_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[SourceCredibilityEntry, ...] = Field(min_length=5, max_length=5)
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
    scored_source_credibility_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    live_scored_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_credibility_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_credibility_ranking_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_external_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_credibility_score: int = Field(ge=0, le=1_000)
    overall_credibility_score: int = Field(ge=0, le=1_000)
    overall_credibility_posture: SourceCredibilityPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    source_credibility_engine_capability_implemented: Literal[True] = True
    source_credibility_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    source_validation_metadata_used: Literal[True] = True
    source_credibility_scoring_execution_implemented: Literal[False] = False
    live_source_credibility_scoring_implemented: Literal[False] = False
    credibility_ranking_mutation_implemented: Literal[False] = False
    credibility_record_write_implemented: Literal[False] = False
    source_validation_execution_implemented: Literal[False] = False
    external_source_fetch_implemented: Literal[False] = False
    web_browsing_implemented: Literal[False] = False
    paper_download_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
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
            if entry.hitl_required_before_credibility_scoring
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.scored_source_credibility_ids:
            raise ValueError("scored_source_credibility_ids must remain empty")
        if self.live_scored_source_ids:
            raise ValueError("live_scored_source_ids must remain empty")
        if self.written_credibility_record_ids:
            raise ValueError("written_credibility_record_ids must remain empty")
        if self.mutated_credibility_ranking_ids:
            raise ValueError("mutated_credibility_ranking_ids must remain empty")
        if self.fetched_external_source_ids:
            raise ValueError("fetched_external_source_ids must remain empty")
        if self.validation_entry_count != len(self.validation_entry_ids):
            raise ValueError("validation_entry_count must match validation ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 12 roadmap")
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
        if self.highest_credibility_score != max(
            entry.credibility_score for entry in self.entries
        ):
            raise ValueError("highest_credibility_score must match entries")
        if self.overall_credibility_score != _overall_credibility_score(self.entries):
            raise ValueError("overall_credibility_score must match entries")
        if self.overall_credibility_posture != _overall_credibility_posture(
            self.entries
        ):
            raise ValueError("overall_credibility_posture must match entries")
        validation_ids = set(self.validation_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.validation_entry_ids).issubset(validation_ids):
                raise ValueError("entry validation ids must be declared")
        return self


def build_source_credibility_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    validation: SourceValidationPlan | None = None,
) -> SourceCredibilityPlan:
    """Build V6.4 Task 12 credibility metadata without scoring sources."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    validation_plan = validation or build_source_validation_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        validation=validation_plan,
    )
    return SourceCredibilityPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=validation_plan.checked_at,
        source_validation_serialization_version=(
            SOURCE_VALIDATION_PLAN_SERIALIZATION_VERSION
        ),
        validation_entry_ids=validation_plan.entry_ids,
        validation_entry_count=len(validation_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=validation_plan.source_count,
        domain_count=validation_plan.domain_count,
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
            if entry.hitl_required_before_credibility_scoring
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
            1 for entry in entries if entry.hitl_required_before_credibility_scoring
        ),
        highest_credibility_score=max(entry.credibility_score for entry in entries),
        overall_credibility_score=_overall_credibility_score(entries),
        overall_credibility_posture=_overall_credibility_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def source_credibility_entry_by_id(
    entry_id: str,
    plan: SourceCredibilityPlan | None = None,
) -> SourceCredibilityEntry | None:
    """Return one source credibility entry without scoring sources."""

    source_plan = plan or build_source_credibility_engine()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def source_credibility_entries_for_status(
    status: SourceCredibilityStatus,
    plan: SourceCredibilityPlan | None = None,
) -> tuple[SourceCredibilityEntry, ...]:
    """Return source credibility entries by advisory status."""

    source_plan = plan or build_source_credibility_engine()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def source_credibility_entries_for_confidence(
    confidence: SourceCredibilityConfidence,
    plan: SourceCredibilityPlan | None = None,
) -> tuple[SourceCredibilityEntry, ...]:
    """Return source credibility entries by confidence band."""

    source_plan = plan or build_source_credibility_engine()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    validation: SourceValidationPlan,
) -> tuple[SourceCredibilityEntry, ...]:
    return (
        _entry(
            kind="authority_signal_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="authority_signal",
            validation_entry_ids=validation.entry_ids,
            validation=validation,
            authority_signal_score=88,
            provenance_reliability_score=86,
            validation_coverage_score=82,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="provenance_reliability_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="provenance_reliability",
            validation_entry_ids=(
                "source_validation_engine::provenance_completeness_review",
                "source_validation_engine::source_presence_validation_review",
            ),
            validation=validation,
            authority_signal_score=84,
            provenance_reliability_score=82,
            validation_coverage_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="validation_coverage_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="validation_coverage",
            validation_entry_ids=(
                "source_validation_engine::source_freshness_validation_review",
                "source_validation_engine::validation_governance_gate",
            ),
            validation=validation,
            authority_signal_score=78,
            provenance_reliability_score=90,
            validation_coverage_score=76,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="bias_risk_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="bias_risk",
            validation_entry_ids=(
                "source_validation_engine::source_access_policy_review",
                "source_validation_engine::provenance_completeness_review",
            ),
            validation=validation,
            authority_signal_score=70,
            provenance_reliability_score=72,
            validation_coverage_score=82,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="credibility_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            validation_entry_ids=validation.entry_ids,
            validation=validation,
            authority_signal_score=30,
            provenance_reliability_score=60,
            validation_coverage_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: SourceCredibilityKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: SourceCredibilityAxis,
    validation_entry_ids: tuple[str, ...],
    validation: SourceValidationPlan,
    authority_signal_score: int,
    provenance_reliability_score: int,
    validation_coverage_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> SourceCredibilityEntry:
    score = _credibility_score(
        authority_signal_score=authority_signal_score,
        provenance_reliability_score=provenance_reliability_score,
        validation_coverage_score=validation_coverage_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return SourceCredibilityEntry(
        entry_id=f"source_credibility_engine::{kind}",
        credibility_kind=kind,
        status=_credibility_status(score),
        confidence=_credibility_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        credibility_axis=axis,
        validation_entry_ids=validation_entry_ids,
        validation_entry_count=len(validation_entry_ids),
        source_count=validation.source_count,
        domain_count=validation.domain_count,
        credibility_summary=_credibility_summary(kind),
        authority_signal_score=authority_signal_score,
        provenance_reliability_score=provenance_reliability_score,
        validation_coverage_score=validation_coverage_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        credibility_score=score,
        hitl_required_before_credibility_scoring=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, validation_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"validation_entry_count:{len(validation_entry_ids)}",
            f"source_count:{validation.source_count}",
            f"domain_count:{validation.domain_count}",
            f"credibility_axis:{axis}",
            f"status:{_credibility_status(score)}",
            f"confidence:{_credibility_confidence(score)}",
            "hitl_required_before_credibility_scoring:true",
        ),
    )


def _credibility_score(
    *,
    authority_signal_score: int,
    provenance_reliability_score: int,
    validation_coverage_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            authority_signal_score * 2
            + provenance_reliability_score * 3
            + validation_coverage_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _credibility_status(score: int) -> SourceCredibilityStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _credibility_confidence(score: int) -> SourceCredibilityConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_credibility_score(entries: tuple[SourceCredibilityEntry, ...]) -> int:
    return round(sum(entry.credibility_score for entry in entries) / len(entries))


def _overall_credibility_posture(
    entries: tuple[SourceCredibilityEntry, ...],
) -> SourceCredibilityPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[SourceCredibilityEntry, ...],
    status: SourceCredibilityStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[SourceCredibilityEntry, ...],
    *confidences: SourceCredibilityConfidence,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(entries: tuple[SourceCredibilityEntry, ...]) -> tuple[str, ...]:
    return (
        f"review_source_credibility_entries:{len(entries)}",
        "confirm_credibility_scope_before_scoring",
        "confirm_no_live_source_credibility_scoring",
        "confirm_no_credibility_record_write",
        "request_hitl_before_source_credibility_scoring",
    )


def _entry_actions(kind: SourceCredibilityKind) -> tuple[str, ...]:
    actions: dict[SourceCredibilityKind, tuple[str, ...]] = {
        "authority_signal_review": (
            "review_authority_signal_policy",
            "confirm_authority_metadata_boundary",
            "confirm_no_live_credibility_scoring",
        ),
        "provenance_reliability_review": (
            "review_provenance_reliability",
            "confirm_validation_traceability",
            "confirm_no_credibility_record_write",
        ),
        "validation_coverage_review": (
            "review_validation_coverage",
            "confirm_no_source_validation_execution",
            "confirm_no_source_fetch",
        ),
        "bias_risk_review": (
            "review_bias_risk_policy",
            "confirm_no_ranking_mutation",
            "confirm_no_contradiction_detection",
        ),
        "credibility_governance_gate": (
            "review_credibility_hitl_gate",
            "confirm_no_credibility_scoring_execution",
            "confirm_no_confidence_scoring",
        ),
    }
    return actions[kind]


def _credibility_summary(kind: SourceCredibilityKind) -> str:
    summaries: dict[SourceCredibilityKind, str] = {
        "authority_signal_review": (
            "Frames authority-signal credibility posture without scoring "
            "sources live or fetching source data."
        ),
        "provenance_reliability_review": (
            "Models provenance reliability requirements without writing "
            "credibility records."
        ),
        "validation_coverage_review": (
            "Describes validation coverage posture without executing source "
            "validation or external checks."
        ),
        "bias_risk_review": (
            "Defines bias-risk review boundaries without mutating rankings or "
            "detecting contradictions."
        ),
        "credibility_governance_gate": (
            "Models the HITL gate required before source credibility scoring "
            "or research confidence scoring."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: SourceCredibilityKind,
    axis: SourceCredibilityAxis,
) -> tuple[str, ...]:
    return (
        "source_credibility_engine",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: SourceCredibilityKind,
    validation_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"credibility_kind:{kind}",
        f"validation_entry_count:{len(validation_entry_ids)}",
        "source_validation_metadata_used:true",
        "no_source_credibility_scoring_performed",
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
