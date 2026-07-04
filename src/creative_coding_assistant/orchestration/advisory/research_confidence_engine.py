"""V6.4 advisory research confidence engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.contradiction_detection import (
    CONTRADICTION_DETECTION_PLAN_SERIALIZATION_VERSION,
    ContradictionDetectionPlan,
    build_contradiction_detection,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

ResearchConfidenceKind = Literal[
    "evidence_strength_review",
    "source_reliability_review",
    "contradiction_risk_review",
    "coverage_completeness_review",
    "confidence_governance_gate",
]
ResearchConfidenceStatus = Literal["candidate", "review_required", "guarded"]
ResearchConfidenceBand = Literal["low", "medium", "high", "guarded"]
ResearchConfidencePosture = Literal["candidate", "review_required", "guarded"]
ResearchConfidenceAxis = Literal[
    "evidence_strength",
    "source_reliability",
    "contradiction_risk",
    "coverage_completeness",
    "governance_gate",
]

RESEARCH_CONFIDENCE_ENTRY_SERIALIZATION_VERSION = "research_confidence_entry.v1"
RESEARCH_CONFIDENCE_PLAN_SERIALIZATION_VERSION = "research_confidence_plan.v1"

RESEARCH_CONFIDENCE_AUTHORITY_BOUNDARY = (
    "V6.4 Research Confidence Engine exposes evidence strength, source "
    "reliability, contradiction-risk readiness, coverage completeness, and "
    "confidence governance posture as inspectable advisory metadata only; it "
    "does not execute research confidence scoring, calculate live confidence "
    "scores, mutate confidence labels, write confidence records, emit "
    "confidence escalations, execute contradiction detection, compare live "
    "claims, resolve contradictions, write contradiction records, execute "
    "source credibility scoring, fetch external sources, browse the web, "
    "download papers, mutate source registries, execute retrieval, mutate "
    "retrieval configuration, mutate vector indexes, enrich the KB, write "
    "storage, provision providers, infer API keys, route providers or models, "
    "execute providers, control workflows, mutate workflow graphs, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Research Confidence Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "research_confidence_scoring_execution",
    "live_research_confidence_scoring",
    "confidence_score_calculation",
    "confidence_label_mutation",
    "confidence_record_write",
    "confidence_escalation_emission",
    "contradiction_detection_execution",
    "live_claim_comparison",
    "contradiction_resolution_execution",
    "contradiction_record_write",
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


class ResearchConfidenceEntry(BaseModel):
    """One advisory research confidence entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    confidence_kind: ResearchConfidenceKind
    status: ResearchConfidenceStatus
    confidence: ResearchConfidenceBand
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    confidence_axis: ResearchConfidenceAxis
    contradiction_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    contradiction_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    confidence_summary: str = Field(min_length=1, max_length=360)
    evidence_strength_score: int = Field(ge=0, le=100)
    source_reliability_score: int = Field(ge=0, le=100)
    contradiction_risk_score: int = Field(ge=0, le=100)
    coverage_completeness_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    research_confidence_score: int = Field(ge=0, le=1_000)
    hitl_required_before_confidence_scoring: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_confidence_engine_capability_implemented: Literal[True] = True
    research_confidence_engine_metadata_implemented: Literal[True] = True
    contradiction_detection_metadata_used: Literal[True] = True
    research_confidence_scoring_execution_implemented: Literal[False] = False
    live_research_confidence_scoring_implemented: Literal[False] = False
    confidence_score_calculation_implemented: Literal[False] = False
    confidence_label_mutation_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    confidence_escalation_emission_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_resolution_execution_implemented: Literal[False] = False
    contradiction_record_write_implemented: Literal[False] = False
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
    serialization_version: Literal["research_confidence_entry.v1"] = (
        RESEARCH_CONFIDENCE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"research_confidence_engine::{self.confidence_kind}":
            raise ValueError("entry_id must match confidence_kind")
        if self.contradiction_entry_count != len(self.contradiction_entry_ids):
            raise ValueError("contradiction_entry_count must match contradiction ids")
        if self.research_confidence_score != _research_confidence_score(
            evidence_strength_score=self.evidence_strength_score,
            source_reliability_score=self.source_reliability_score,
            contradiction_risk_score=self.contradiction_risk_score,
            coverage_completeness_score=self.coverage_completeness_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("research_confidence_score must combine source scores")
        if self.status != _research_confidence_status(self.research_confidence_score):
            raise ValueError("status must match research_confidence_score")
        if self.confidence != _research_confidence_band(self.research_confidence_score):
            raise ValueError("confidence must match research_confidence_score")
        if not self.hitl_required_before_confidence_scoring:
            raise ValueError("research confidence scoring requires HITL posture")
        return self


class ResearchConfidencePlan(BaseModel):
    """Bounded V6.4 advisory research confidence plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["research_confidence_engine"] = "research_confidence_engine"
    serialization_version: Literal["research_confidence_plan.v1"] = (
        RESEARCH_CONFIDENCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RESEARCH_CONFIDENCE_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    contradiction_detection_role: Literal["contradiction_detection"] = (
        "contradiction_detection"
    )
    contradiction_detection_serialization_version: Literal[
        "contradiction_detection_plan.v1"
    ]
    contradiction_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    contradiction_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ResearchConfidenceEntry, ...] = Field(
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
    scored_research_confidence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    calculated_confidence_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_confidence_label_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_confidence_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_confidence_escalation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_research_confidence_score: int = Field(ge=0, le=1_000)
    overall_research_confidence_score: int = Field(ge=0, le=1_000)
    overall_research_confidence_posture: ResearchConfidencePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    research_confidence_engine_capability_implemented: Literal[True] = True
    research_confidence_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    contradiction_detection_metadata_used: Literal[True] = True
    research_confidence_scoring_execution_implemented: Literal[False] = False
    live_research_confidence_scoring_implemented: Literal[False] = False
    confidence_score_calculation_implemented: Literal[False] = False
    confidence_label_mutation_implemented: Literal[False] = False
    confidence_record_write_implemented: Literal[False] = False
    confidence_escalation_emission_implemented: Literal[False] = False
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_resolution_execution_implemented: Literal[False] = False
    contradiction_record_write_implemented: Literal[False] = False
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
            if entry.hitl_required_before_confidence_scoring
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.scored_research_confidence_ids:
            raise ValueError("scored_research_confidence_ids must remain empty")
        if self.calculated_confidence_score_ids:
            raise ValueError("calculated_confidence_score_ids must remain empty")
        if self.mutated_confidence_label_ids:
            raise ValueError("mutated_confidence_label_ids must remain empty")
        if self.written_confidence_record_ids:
            raise ValueError("written_confidence_record_ids must remain empty")
        if self.emitted_confidence_escalation_ids:
            raise ValueError("emitted_confidence_escalation_ids must remain empty")
        if self.contradiction_entry_count != len(self.contradiction_entry_ids):
            raise ValueError("contradiction_entry_count must match contradiction ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 14 roadmap")
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
        if self.highest_research_confidence_score != max(
            entry.research_confidence_score for entry in self.entries
        ):
            raise ValueError("highest_research_confidence_score must match entries")
        if self.overall_research_confidence_score != _overall_confidence_score(
            self.entries
        ):
            raise ValueError("overall_research_confidence_score must match entries")
        if self.overall_research_confidence_posture != _overall_confidence_posture(
            self.entries
        ):
            raise ValueError("overall_research_confidence_posture must match entries")
        contradiction_ids = set(self.contradiction_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.contradiction_entry_ids).issubset(contradiction_ids):
                raise ValueError("entry contradiction ids must be declared")
        return self


def build_research_confidence_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    contradiction: ContradictionDetectionPlan | None = None,
) -> ResearchConfidencePlan:
    """Build V6.4 Task 14 confidence metadata without confidence scoring."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    contradiction_plan = contradiction or build_contradiction_detection(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        contradiction=contradiction_plan,
    )
    return ResearchConfidencePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=contradiction_plan.checked_at,
        contradiction_detection_serialization_version=(
            CONTRADICTION_DETECTION_PLAN_SERIALIZATION_VERSION
        ),
        contradiction_entry_ids=contradiction_plan.entry_ids,
        contradiction_entry_count=len(contradiction_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=contradiction_plan.source_count,
        domain_count=contradiction_plan.domain_count,
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
            if entry.hitl_required_before_confidence_scoring
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
            1 for entry in entries if entry.hitl_required_before_confidence_scoring
        ),
        highest_research_confidence_score=max(
            entry.research_confidence_score for entry in entries
        ),
        overall_research_confidence_score=_overall_confidence_score(entries),
        overall_research_confidence_posture=_overall_confidence_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def research_confidence_entry_by_id(
    entry_id: str,
    plan: ResearchConfidencePlan | None = None,
) -> ResearchConfidenceEntry | None:
    """Return one research confidence entry without confidence scoring."""

    source_plan = plan or build_research_confidence_engine()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def research_confidence_entries_for_status(
    status: ResearchConfidenceStatus,
    plan: ResearchConfidencePlan | None = None,
) -> tuple[ResearchConfidenceEntry, ...]:
    """Return research confidence entries by advisory status."""

    source_plan = plan or build_research_confidence_engine()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def research_confidence_entries_for_confidence(
    confidence: ResearchConfidenceBand,
    plan: ResearchConfidencePlan | None = None,
) -> tuple[ResearchConfidenceEntry, ...]:
    """Return research confidence entries by confidence band."""

    source_plan = plan or build_research_confidence_engine()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    contradiction: ContradictionDetectionPlan,
) -> tuple[ResearchConfidenceEntry, ...]:
    return (
        _entry(
            kind="evidence_strength_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_strength",
            contradiction_entry_ids=contradiction.entry_ids,
            contradiction=contradiction,
            evidence_strength_score=85,
            source_reliability_score=83,
            contradiction_risk_score=75,
            coverage_completeness_score=82,
            governance_alignment_score=84,
            mutation_risk_score=25,
            governance_weight=30,
        ),
        _entry(
            kind="source_reliability_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_reliability",
            contradiction_entry_ids=(
                "contradiction_detection::claim_alignment_review",
                "contradiction_detection::evidence_conflict_review",
            ),
            contradiction=contradiction,
            evidence_strength_score=82,
            source_reliability_score=88,
            contradiction_risk_score=70,
            coverage_completeness_score=78,
            governance_alignment_score=80,
            mutation_risk_score=20,
            governance_weight=25,
        ),
        _entry(
            kind="contradiction_risk_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="contradiction_risk",
            contradiction_entry_ids=(
                "contradiction_detection::source_disagreement_review",
                "contradiction_detection::contradiction_escalation_review",
            ),
            contradiction=contradiction,
            evidence_strength_score=78,
            source_reliability_score=74,
            contradiction_risk_score=88,
            coverage_completeness_score=76,
            governance_alignment_score=82,
            mutation_risk_score=18,
            governance_weight=35,
        ),
        _entry(
            kind="coverage_completeness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="coverage_completeness",
            contradiction_entry_ids=(
                "contradiction_detection::evidence_conflict_review",
                "contradiction_detection::claim_alignment_review",
            ),
            contradiction=contradiction,
            evidence_strength_score=72,
            source_reliability_score=70,
            contradiction_risk_score=68,
            coverage_completeness_score=76,
            governance_alignment_score=78,
            mutation_risk_score=16,
            governance_weight=25,
        ),
        _entry(
            kind="confidence_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            contradiction_entry_ids=contradiction.entry_ids,
            contradiction=contradiction,
            evidence_strength_score=40,
            source_reliability_score=38,
            contradiction_risk_score=44,
            coverage_completeness_score=42,
            governance_alignment_score=90,
            mutation_risk_score=10,
            governance_weight=35,
        ),
    )


def _entry(
    *,
    kind: ResearchConfidenceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ResearchConfidenceAxis,
    contradiction_entry_ids: tuple[str, ...],
    contradiction: ContradictionDetectionPlan,
    evidence_strength_score: int,
    source_reliability_score: int,
    contradiction_risk_score: int,
    coverage_completeness_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ResearchConfidenceEntry:
    score = _research_confidence_score(
        evidence_strength_score=evidence_strength_score,
        source_reliability_score=source_reliability_score,
        contradiction_risk_score=contradiction_risk_score,
        coverage_completeness_score=coverage_completeness_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ResearchConfidenceEntry(
        entry_id=f"research_confidence_engine::{kind}",
        confidence_kind=kind,
        status=_research_confidence_status(score),
        confidence=_research_confidence_band(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        confidence_axis=axis,
        contradiction_entry_ids=contradiction_entry_ids,
        contradiction_entry_count=len(contradiction_entry_ids),
        source_count=contradiction.source_count,
        domain_count=contradiction.domain_count,
        confidence_summary=_confidence_summary(kind),
        evidence_strength_score=evidence_strength_score,
        source_reliability_score=source_reliability_score,
        contradiction_risk_score=contradiction_risk_score,
        coverage_completeness_score=coverage_completeness_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        research_confidence_score=score,
        hitl_required_before_confidence_scoring=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, contradiction_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"contradiction_entry_count:{len(contradiction_entry_ids)}",
            f"source_count:{contradiction.source_count}",
            f"domain_count:{contradiction.domain_count}",
            f"confidence_axis:{axis}",
            f"status:{_research_confidence_status(score)}",
            f"confidence:{_research_confidence_band(score)}",
            "hitl_required_before_confidence_scoring:true",
        ),
    )


def _research_confidence_score(
    *,
    evidence_strength_score: int,
    source_reliability_score: int,
    contradiction_risk_score: int,
    coverage_completeness_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            evidence_strength_score * 2
            + source_reliability_score * 2
            + contradiction_risk_score * 3
            + coverage_completeness_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _research_confidence_status(score: int) -> ResearchConfidenceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _research_confidence_band(score: int) -> ResearchConfidenceBand:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_confidence_score(
    entries: tuple[ResearchConfidenceEntry, ...],
) -> int:
    return round(
        sum(entry.research_confidence_score for entry in entries) / len(entries)
    )


def _overall_confidence_posture(
    entries: tuple[ResearchConfidenceEntry, ...],
) -> ResearchConfidencePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ResearchConfidenceEntry, ...],
    status: ResearchConfidenceStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ResearchConfidenceEntry, ...],
    *confidences: ResearchConfidenceBand,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.confidence in confidences)


def _plan_actions(
    entries: tuple[ResearchConfidenceEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_research_confidence_entries:{len(entries)}",
        "confirm_confidence_scope_before_scoring",
        "confirm_no_live_confidence_calculation",
        "confirm_no_confidence_record_write",
        "request_hitl_before_confidence_scoring",
    )


def _entry_actions(kind: ResearchConfidenceKind) -> tuple[str, ...]:
    actions: dict[ResearchConfidenceKind, tuple[str, ...]] = {
        "evidence_strength_review": (
            "review_evidence_strength_boundary",
            "confirm_evidence_inputs_are_advisory",
            "confirm_no_confidence_scoring",
        ),
        "source_reliability_review": (
            "review_source_reliability_boundary",
            "confirm_credibility_metadata_boundary",
            "confirm_no_source_scoring_execution",
        ),
        "contradiction_risk_review": (
            "review_contradiction_risk_boundary",
            "confirm_no_live_claim_comparison",
            "confirm_no_contradiction_resolution",
        ),
        "coverage_completeness_review": (
            "review_coverage_completeness_boundary",
            "confirm_no_external_source_fetch",
            "confirm_no_retrieval_execution",
        ),
        "confidence_governance_gate": (
            "review_confidence_hitl_gate",
            "confirm_no_confidence_record_write",
            "confirm_no_runtime_evolution",
        ),
    }
    return actions[kind]


def _confidence_summary(kind: ResearchConfidenceKind) -> str:
    summaries: dict[ResearchConfidenceKind, str] = {
        "evidence_strength_review": (
            "Frames evidence-strength readiness without calculating research "
            "confidence or writing confidence records."
        ),
        "source_reliability_review": (
            "Models source-reliability confidence posture without live source "
            "credibility scoring."
        ),
        "contradiction_risk_review": (
            "Describes contradiction-risk confidence readiness without live "
            "claim comparison or contradiction resolution."
        ),
        "coverage_completeness_review": (
            "Models coverage completeness posture without fetching external "
            "sources or executing retrieval."
        ),
        "confidence_governance_gate": (
            "Models the HITL gate required before research confidence scoring, "
            "label mutation, or confidence record writes."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ResearchConfidenceKind,
    axis: ResearchConfidenceAxis,
) -> tuple[str, ...]:
    return (
        "research_confidence_engine",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ResearchConfidenceKind,
    contradiction_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"confidence_kind:{kind}",
        f"contradiction_entry_count:{len(contradiction_entry_ids)}",
        "contradiction_detection_metadata_used:true",
        "no_research_confidence_scoring_performed",
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
