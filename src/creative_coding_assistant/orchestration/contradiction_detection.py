"""V6.4 advisory contradiction detection metadata."""

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
from creative_coding_assistant.orchestration.source_credibility_engine import (
    SOURCE_CREDIBILITY_PLAN_SERIALIZATION_VERSION,
    SourceCredibilityPlan,
    build_source_credibility_engine,
)

ContradictionDetectionKind = Literal[
    "claim_alignment_review",
    "evidence_conflict_review",
    "source_disagreement_review",
    "contradiction_escalation_review",
    "contradiction_governance_gate",
]
ContradictionDetectionStatus = Literal["candidate", "review_required", "guarded"]
ContradictionDetectionConfidence = Literal["low", "medium", "high", "guarded"]
ContradictionDetectionPosture = Literal["candidate", "review_required", "guarded"]
ContradictionDetectionAxis = Literal[
    "claim_alignment",
    "evidence_conflict",
    "source_disagreement",
    "escalation_policy",
    "governance_gate",
]

CONTRADICTION_DETECTION_ENTRY_SERIALIZATION_VERSION = (
    "contradiction_detection_entry.v1"
)
CONTRADICTION_DETECTION_PLAN_SERIALIZATION_VERSION = (
    "contradiction_detection_plan.v1"
)

CONTRADICTION_DETECTION_AUTHORITY_BOUNDARY = (
    "V6.4 Contradiction Detection exposes claim alignment, evidence-conflict "
    "readiness, source disagreement posture, escalation policy, and governance "
    "readiness as inspectable advisory metadata only; it does not execute "
    "contradiction detection, compare live claims, resolve contradictions, "
    "emit contradiction escalations, write contradiction records, execute "
    "source credibility scoring, score research confidence, fetch external "
    "sources, browse the web, download papers, mutate source registries, "
    "execute retrieval, mutate retrieval configuration, mutate vector indexes, "
    "enrich the KB, write storage, provision providers, infer API keys, route "
    "providers or models, execute providers, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Contradiction Detection",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "contradiction_detection_execution",
    "live_claim_comparison",
    "contradiction_resolution_execution",
    "contradiction_escalation_emission",
    "contradiction_record_write",
    "source_credibility_scoring_execution",
    "research_confidence_scoring",
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


class ContradictionDetectionEntry(BaseModel):
    """One advisory contradiction detection entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_id: str = Field(min_length=1, max_length=180)
    contradiction_kind: ContradictionDetectionKind
    status: ContradictionDetectionStatus
    confidence: ContradictionDetectionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    contradiction_axis: ContradictionDetectionAxis
    credibility_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    credibility_entry_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    contradiction_summary: str = Field(min_length=1, max_length=360)
    claim_alignment_score: int = Field(ge=0, le=100)
    evidence_conflict_score: int = Field(ge=0, le=100)
    source_disagreement_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    contradiction_score: int = Field(ge=0, le=1_000)
    hitl_required_before_contradiction_detection: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    contradiction_detection_capability_implemented: Literal[True] = True
    contradiction_detection_metadata_implemented: Literal[True] = True
    source_credibility_metadata_used: Literal[True] = True
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_resolution_execution_implemented: Literal[False] = False
    contradiction_escalation_emission_implemented: Literal[False] = False
    contradiction_record_write_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
    serialization_version: Literal["contradiction_detection_entry.v1"] = (
        CONTRADICTION_DETECTION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.entry_id != f"contradiction_detection::{self.contradiction_kind}":
            raise ValueError("entry_id must match contradiction_kind")
        if self.credibility_entry_count != len(self.credibility_entry_ids):
            raise ValueError("credibility_entry_count must match credibility ids")
        if self.contradiction_score != _contradiction_score(
            claim_alignment_score=self.claim_alignment_score,
            evidence_conflict_score=self.evidence_conflict_score,
            source_disagreement_score=self.source_disagreement_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("contradiction_score must combine source scores")
        if self.status != _contradiction_status(self.contradiction_score):
            raise ValueError("status must match contradiction_score")
        if self.confidence != _contradiction_confidence(self.contradiction_score):
            raise ValueError("confidence must match contradiction_score")
        if not self.hitl_required_before_contradiction_detection:
            raise ValueError("contradiction detection requires HITL posture")
        return self


class ContradictionDetectionPlan(BaseModel):
    """Bounded V6.4 advisory contradiction detection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["contradiction_detection"] = "contradiction_detection"
    serialization_version: Literal["contradiction_detection_plan.v1"] = (
        CONTRADICTION_DETECTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CONTRADICTION_DETECTION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_credibility_role: Literal["source_credibility_engine"] = (
        "source_credibility_engine"
    )
    source_credibility_serialization_version: Literal["source_credibility_plan.v1"]
    credibility_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    credibility_entry_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[ContradictionDetectionEntry, ...] = Field(
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
    detected_contradiction_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    compared_live_claim_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    resolved_contradiction_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_contradiction_escalation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_contradiction_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_contradiction_score: int = Field(ge=0, le=1_000)
    overall_contradiction_score: int = Field(ge=0, le=1_000)
    overall_contradiction_posture: ContradictionDetectionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    contradiction_detection_capability_implemented: Literal[True] = True
    contradiction_detection_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    source_credibility_metadata_used: Literal[True] = True
    contradiction_detection_execution_implemented: Literal[False] = False
    live_claim_comparison_implemented: Literal[False] = False
    contradiction_resolution_execution_implemented: Literal[False] = False
    contradiction_escalation_emission_implemented: Literal[False] = False
    contradiction_record_write_implemented: Literal[False] = False
    source_credibility_scoring_execution_implemented: Literal[False] = False
    research_confidence_scoring_implemented: Literal[False] = False
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
            if entry.hitl_required_before_contradiction_detection
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.detected_contradiction_ids:
            raise ValueError("detected_contradiction_ids must remain empty")
        if self.compared_live_claim_ids:
            raise ValueError("compared_live_claim_ids must remain empty")
        if self.resolved_contradiction_ids:
            raise ValueError("resolved_contradiction_ids must remain empty")
        if self.emitted_contradiction_escalation_ids:
            raise ValueError(
                "emitted_contradiction_escalation_ids must remain empty"
            )
        if self.written_contradiction_record_ids:
            raise ValueError("written_contradiction_record_ids must remain empty")
        if self.credibility_entry_count != len(self.credibility_entry_ids):
            raise ValueError("credibility_entry_count must match credibility ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.4 Task 13 roadmap")
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
        if self.highest_contradiction_score != max(
            entry.contradiction_score for entry in self.entries
        ):
            raise ValueError("highest_contradiction_score must match entries")
        if self.overall_contradiction_score != _overall_contradiction_score(
            self.entries
        ):
            raise ValueError("overall_contradiction_score must match entries")
        if self.overall_contradiction_posture != _overall_contradiction_posture(
            self.entries
        ):
            raise ValueError("overall_contradiction_posture must match entries")
        credibility_ids = set(self.credibility_entry_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if entry.source_count != self.source_count:
                raise ValueError("entry source_count must match plan")
            if entry.domain_count != self.domain_count:
                raise ValueError("entry domain_count must match plan")
            if not set(entry.credibility_entry_ids).issubset(credibility_ids):
                raise ValueError("entry credibility ids must be declared")
        return self


def build_contradiction_detection(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    credibility: SourceCredibilityPlan | None = None,
) -> ContradictionDetectionPlan:
    """Build V6.4 Task 13 contradiction metadata without detecting conflicts."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    credibility_plan = credibility or build_source_credibility_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        credibility=credibility_plan,
    )
    return ContradictionDetectionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=credibility_plan.checked_at,
        source_credibility_serialization_version=(
            SOURCE_CREDIBILITY_PLAN_SERIALIZATION_VERSION
        ),
        credibility_entry_ids=credibility_plan.entry_ids,
        credibility_entry_count=len(credibility_plan.entry_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=credibility_plan.source_count,
        domain_count=credibility_plan.domain_count,
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
            if entry.hitl_required_before_contradiction_detection
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
            if entry.hitl_required_before_contradiction_detection
        ),
        highest_contradiction_score=max(
            entry.contradiction_score for entry in entries
        ),
        overall_contradiction_score=_overall_contradiction_score(entries),
        overall_contradiction_posture=_overall_contradiction_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def contradiction_detection_entry_by_id(
    entry_id: str,
    plan: ContradictionDetectionPlan | None = None,
) -> ContradictionDetectionEntry | None:
    """Return one contradiction detection entry without detecting conflicts."""

    source_plan = plan or build_contradiction_detection()
    for entry in source_plan.entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def contradiction_detection_entries_for_status(
    status: ContradictionDetectionStatus,
    plan: ContradictionDetectionPlan | None = None,
) -> tuple[ContradictionDetectionEntry, ...]:
    """Return contradiction detection entries by advisory status."""

    source_plan = plan or build_contradiction_detection()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def contradiction_detection_entries_for_confidence(
    confidence: ContradictionDetectionConfidence,
    plan: ContradictionDetectionPlan | None = None,
) -> tuple[ContradictionDetectionEntry, ...]:
    """Return contradiction detection entries by confidence band."""

    source_plan = plan or build_contradiction_detection()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    credibility: SourceCredibilityPlan,
) -> tuple[ContradictionDetectionEntry, ...]:
    return (
        _entry(
            kind="claim_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="claim_alignment",
            credibility_entry_ids=credibility.entry_ids,
            credibility=credibility,
            claim_alignment_score=88,
            evidence_conflict_score=86,
            source_disagreement_score=82,
            governance_alignment_score=90,
            mutation_risk_score=52,
            governance_weight=120,
        ),
        _entry(
            kind="evidence_conflict_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evidence_conflict",
            credibility_entry_ids=(
                "source_credibility_engine::provenance_reliability_review",
                "source_credibility_engine::validation_coverage_review",
            ),
            credibility=credibility,
            claim_alignment_score=84,
            evidence_conflict_score=82,
            source_disagreement_score=80,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=105,
        ),
        _entry(
            kind="source_disagreement_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_disagreement",
            credibility_entry_ids=(
                "source_credibility_engine::authority_signal_review",
                "source_credibility_engine::bias_risk_review",
            ),
            credibility=credibility,
            claim_alignment_score=78,
            evidence_conflict_score=90,
            source_disagreement_score=76,
            governance_alignment_score=88,
            mutation_risk_score=40,
            governance_weight=95,
        ),
        _entry(
            kind="contradiction_escalation_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="escalation_policy",
            credibility_entry_ids=(
                "source_credibility_engine::credibility_governance_gate",
                "source_credibility_engine::bias_risk_review",
            ),
            credibility=credibility,
            claim_alignment_score=70,
            evidence_conflict_score=72,
            source_disagreement_score=82,
            governance_alignment_score=84,
            mutation_risk_score=30,
            governance_weight=80,
        ),
        _entry(
            kind="contradiction_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            credibility_entry_ids=credibility.entry_ids,
            credibility=credibility,
            claim_alignment_score=30,
            evidence_conflict_score=60,
            source_disagreement_score=40,
            governance_alignment_score=92,
            mutation_risk_score=12,
            governance_weight=50,
        ),
    )


def _entry(
    *,
    kind: ContradictionDetectionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: ContradictionDetectionAxis,
    credibility_entry_ids: tuple[str, ...],
    credibility: SourceCredibilityPlan,
    claim_alignment_score: int,
    evidence_conflict_score: int,
    source_disagreement_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> ContradictionDetectionEntry:
    score = _contradiction_score(
        claim_alignment_score=claim_alignment_score,
        evidence_conflict_score=evidence_conflict_score,
        source_disagreement_score=source_disagreement_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return ContradictionDetectionEntry(
        entry_id=f"contradiction_detection::{kind}",
        contradiction_kind=kind,
        status=_contradiction_status(score),
        confidence=_contradiction_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        contradiction_axis=axis,
        credibility_entry_ids=credibility_entry_ids,
        credibility_entry_count=len(credibility_entry_ids),
        source_count=credibility.source_count,
        domain_count=credibility.domain_count,
        contradiction_summary=_contradiction_summary(kind),
        claim_alignment_score=claim_alignment_score,
        evidence_conflict_score=evidence_conflict_score,
        source_disagreement_score=source_disagreement_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        contradiction_score=score,
        hitl_required_before_contradiction_detection=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, credibility_entry_ids),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"credibility_entry_count:{len(credibility_entry_ids)}",
            f"source_count:{credibility.source_count}",
            f"domain_count:{credibility.domain_count}",
            f"contradiction_axis:{axis}",
            f"status:{_contradiction_status(score)}",
            f"confidence:{_contradiction_confidence(score)}",
            "hitl_required_before_contradiction_detection:true",
        ),
    )


def _contradiction_score(
    *,
    claim_alignment_score: int,
    evidence_conflict_score: int,
    source_disagreement_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            claim_alignment_score * 2
            + evidence_conflict_score * 3
            + source_disagreement_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _contradiction_status(score: int) -> ContradictionDetectionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _contradiction_confidence(score: int) -> ContradictionDetectionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 600:
        return "medium"
    return "low"


def _overall_contradiction_score(
    entries: tuple[ContradictionDetectionEntry, ...],
) -> int:
    return round(sum(entry.contradiction_score for entry in entries) / len(entries))


def _overall_contradiction_posture(
    entries: tuple[ContradictionDetectionEntry, ...],
) -> ContradictionDetectionPosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[ContradictionDetectionEntry, ...],
    status: ContradictionDetectionStatus,
) -> tuple[str, ...]:
    return tuple(entry.entry_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[ContradictionDetectionEntry, ...],
    *confidences: ContradictionDetectionConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.entry_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[ContradictionDetectionEntry, ...],
) -> tuple[str, ...]:
    return (
        f"review_contradiction_detection_entries:{len(entries)}",
        "confirm_contradiction_scope_before_execution",
        "confirm_no_live_claim_comparison",
        "confirm_no_contradiction_record_write",
        "request_hitl_before_contradiction_detection",
    )


def _entry_actions(kind: ContradictionDetectionKind) -> tuple[str, ...]:
    actions: dict[ContradictionDetectionKind, tuple[str, ...]] = {
        "claim_alignment_review": (
            "review_claim_alignment_boundary",
            "confirm_claim_groups_are_advisory",
            "confirm_no_live_claim_comparison",
        ),
        "evidence_conflict_review": (
            "review_evidence_conflict_policy",
            "confirm_evidence_traces",
            "confirm_no_contradiction_detection",
        ),
        "source_disagreement_review": (
            "review_source_disagreement_policy",
            "confirm_credibility_metadata_boundary",
            "confirm_no_source_scoring_execution",
        ),
        "contradiction_escalation_review": (
            "review_contradiction_escalation_policy",
            "confirm_no_escalation_emission",
            "confirm_no_resolution_execution",
        ),
        "contradiction_governance_gate": (
            "review_contradiction_hitl_gate",
            "confirm_no_contradiction_record_write",
            "confirm_no_confidence_scoring",
        ),
    }
    return actions[kind]


def _contradiction_summary(kind: ContradictionDetectionKind) -> str:
    summaries: dict[ContradictionDetectionKind, str] = {
        "claim_alignment_review": (
            "Frames claim alignment readiness without comparing live claims or "
            "detecting contradictions."
        ),
        "evidence_conflict_review": (
            "Models evidence-conflict review posture without contradiction "
            "detection execution or record writes."
        ),
        "source_disagreement_review": (
            "Describes source disagreement readiness without live credibility "
            "scoring or source validation execution."
        ),
        "contradiction_escalation_review": (
            "Defines escalation policy boundaries without emitting "
            "escalations or resolving contradictions."
        ),
        "contradiction_governance_gate": (
            "Models the HITL gate required before contradiction detection, "
            "resolution, or confidence scoring."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: ContradictionDetectionKind,
    axis: ContradictionDetectionAxis,
) -> tuple[str, ...]:
    return (
        "contradiction_detection",
        kind,
        f"axis:{axis}",
        "advisory_metadata",
        "hitl_required",
    )


def _explainability_notes(
    kind: ContradictionDetectionKind,
    credibility_entry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"contradiction_kind:{kind}",
        f"credibility_entry_count:{len(credibility_entry_ids)}",
        "source_credibility_metadata_used:true",
        "no_contradiction_detection_performed",
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
