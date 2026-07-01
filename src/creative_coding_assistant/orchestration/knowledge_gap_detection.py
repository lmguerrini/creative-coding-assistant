"""V6.3 advisory knowledge gap detection metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_quality_scoring import (
    KnowledgeQualityScoringPlan,
    build_knowledge_quality_scoring,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeGapDetectionKind = Literal[
    "knowledge_gap_inventory_review",
    "knowledge_gap_coverage_review",
    "knowledge_gap_priority_review",
    "knowledge_gap_detection_readiness",
    "knowledge_gap_governance_gate",
]
KnowledgeGapDetectionStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeGapDetectionConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeGapDetectionPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeGapDetectionAxis = Literal[
    "inventory_review",
    "coverage_review",
    "priority_review",
    "detection_readiness",
    "governance_gate",
]

KNOWLEDGE_GAP_ENTRY_SERIALIZATION_VERSION = "knowledge_gap_entry.v1"
KNOWLEDGE_GAP_PLAN_SERIALIZATION_VERSION = "knowledge_gap_plan.v1"
KNOWLEDGE_GAP_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Gap Detection exposes knowledge quality, gap inventory, "
    "coverage, priority, detection readiness, and governance posture as "
    "inspectable advisory metadata only; it does not execute gap detection, "
    "scan for gaps, assign gap priority, remediate gaps, backfill sources, add "
    "sources, enrich the KB, compute or persist quality scores, execute "
    "retrieval queries, mutate retrieval configuration, mutate ranking, request "
    "embeddings, refresh embeddings, index vectors, upsert vectors, write KB "
    "storage, fetch documentation, update source records, provision providers, "
    "infer API keys, route providers or models, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Gap Detection",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_gap_detection_execution",
    "gap_scan_execution",
    "gap_priority_assignment",
    "gap_remediation_execution",
    "gap_backfill_execution",
    "source_addition",
    "kb_enrichment_execution",
    "quality_score_computation",
    "quality_score_persistence",
    "retrieval_query_execution",
    "retrieval_configuration_mutation",
    "ranking_mutation",
    "embedding_request_execution",
    "embedding_refresh_execution",
    "vector_indexing",
    "vector_upsert",
    "kb_storage_write",
    "documentation_fetch_execution",
    "source_record_update",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeGapSignal(BaseModel):
    """One advisory knowledge gap detection signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeGapDetectionKind
    status: KnowledgeGapDetectionStatus
    confidence: KnowledgeGapDetectionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    gap_axis: KnowledgeGapDetectionAxis
    knowledge_quality_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    knowledge_quality_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    gap_signal_summary: str = Field(min_length=1, max_length=360)
    gap_signal_score: int = Field(ge=0, le=100)
    coverage_signal_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    gap_score: int = Field(ge=0, le=1_000)
    hitl_required_before_gap_detection: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_gap_detection_capability_implemented: Literal[True] = True
    knowledge_gap_detection_metadata_implemented: Literal[True] = True
    knowledge_quality_metadata_used: Literal[True] = True
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_priority_assignment_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    gap_backfill_execution_implemented: Literal[False] = False
    source_addition_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    quality_score_persistence_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["knowledge_gap_entry.v1"] = (
        KNOWLEDGE_GAP_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_gap_detection::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_quality_signal_count != len(
            self.knowledge_quality_signal_ids
        ):
            raise ValueError("knowledge_quality_signal_count must match signals")
        if self.gap_score != _gap_score(
            gap_signal_score=self.gap_signal_score,
            coverage_signal_score=self.coverage_signal_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("gap_score must combine source scores")
        if self.status != _gap_status(self.gap_score):
            raise ValueError("status must match gap_score")
        if self.confidence != _gap_confidence(self.gap_score):
            raise ValueError("confidence must match gap_score")
        if not self.hitl_required_before_gap_detection:
            raise ValueError("knowledge gap detection requires HITL posture")
        return self


class KnowledgeGapDetectionPlan(BaseModel):
    """Bounded V6.3 advisory knowledge gap detection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_gap_detection"] = "knowledge_gap_detection"
    serialization_version: Literal["knowledge_gap_plan.v1"] = (
        KNOWLEDGE_GAP_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_GAP_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_quality_role: Literal["knowledge_quality_scoring"] = (
        "knowledge_quality_scoring"
    )
    knowledge_quality_serialization_version: Literal["knowledge_quality_plan.v1"]
    knowledge_quality_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    knowledge_quality_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeGapSignal, ...] = Field(min_length=5, max_length=5)
    signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    signal_count: int = Field(ge=5, le=5)
    candidate_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    planned_gap_detection_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    detected_gap_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    prioritized_gap_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    remediated_gap_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    backfilled_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    enriched_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_gap_score: int = Field(ge=0, le=1_000)
    overall_gap_score: int = Field(ge=0, le=1_000)
    overall_gap_posture: KnowledgeGapDetectionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_gap_detection_capability_implemented: Literal[True] = True
    knowledge_gap_detection_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_quality_metadata_used: Literal[True] = True
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_priority_assignment_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    gap_backfill_execution_implemented: Literal[False] = False
    source_addition_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    quality_score_persistence_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
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
    def _plan_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.candidate_signal_ids != _signal_ids_for_status(
            self.signals,
            "candidate",
        ):
            raise ValueError("candidate_signal_ids must match signals")
        if self.review_required_signal_ids != _signal_ids_for_status(
            self.signals,
            "review_required",
        ):
            raise ValueError("review_required_signal_ids must match signals")
        if self.guarded_signal_ids != _signal_ids_for_status(self.signals, "guarded"):
            raise ValueError("guarded_signal_ids must match signals")
        if self.high_confidence_signal_ids != _signal_ids_for_confidence(
            self.signals,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_signal_ids must match signals")
        if self.hitl_required_signal_ids != tuple(
            signal.signal_id
            for signal in self.signals
            if signal.hitl_required_before_gap_detection
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_gap_detection_ids:
            raise ValueError("planned_gap_detection_ids must remain empty")
        if self.detected_gap_ids:
            raise ValueError("detected_gap_ids must remain empty")
        if self.prioritized_gap_ids:
            raise ValueError("prioritized_gap_ids must remain empty")
        if self.remediated_gap_ids:
            raise ValueError("remediated_gap_ids must remain empty")
        if self.backfilled_source_ids:
            raise ValueError("backfilled_source_ids must remain empty")
        if self.enriched_kb_record_ids:
            raise ValueError("enriched_kb_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_quality_signal_count != len(
            self.knowledge_quality_signal_ids
        ):
            raise ValueError("knowledge_quality_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 9 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.candidate_signal_count != len(self.candidate_signal_ids):
            raise ValueError("candidate_signal_count must match signals")
        if self.review_required_signal_count != len(
            self.review_required_signal_ids
        ):
            raise ValueError("review_required_signal_count must match signals")
        if self.guarded_signal_count != len(self.guarded_signal_ids):
            raise ValueError("guarded_signal_count must match signals")
        if self.high_confidence_signal_count != len(self.high_confidence_signal_ids):
            raise ValueError("high_confidence_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.highest_gap_score != max(signal.gap_score for signal in self.signals):
            raise ValueError("highest_gap_score must match signals")
        if self.overall_gap_score != _overall_gap_score(self.signals):
            raise ValueError("overall_gap_score must match signals")
        if self.overall_gap_posture != _overall_gap_posture(self.signals):
            raise ValueError("overall_gap_posture must match signals")
        declared_quality_signals = set(self.knowledge_quality_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_quality_signal_ids).issubset(
                declared_quality_signals
            ):
                raise ValueError("signal knowledge_quality_signal_ids must be known")
        return self


def build_knowledge_gap_detection(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_quality: KnowledgeQualityScoringPlan | None = None,
) -> KnowledgeGapDetectionPlan:
    """Build V6.3 Task 9 knowledge gap detection metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    quality_plan = knowledge_quality or build_knowledge_quality_scoring(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        quality_plan=quality_plan,
    )
    return KnowledgeGapDetectionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=quality_plan.checked_at,
        knowledge_quality_serialization_version=quality_plan.serialization_version,
        knowledge_quality_signal_ids=quality_plan.signal_ids,
        knowledge_quality_signal_count=len(quality_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=quality_plan.source_count,
        domain_count=quality_plan.domain_count,
        execution_mode_ids=execution_modes.execution_mode_ids,
        signals=signals,
        signal_ids=tuple(signal.signal_id for signal in signals),
        signal_count=len(signals),
        candidate_signal_ids=_signal_ids_for_status(signals, "candidate"),
        review_required_signal_ids=_signal_ids_for_status(
            signals,
            "review_required",
        ),
        guarded_signal_ids=_signal_ids_for_status(signals, "guarded"),
        high_confidence_signal_ids=_signal_ids_for_confidence(
            signals,
            "high",
            "guarded",
        ),
        hitl_required_signal_ids=tuple(
            signal.signal_id
            for signal in signals
            if signal.hitl_required_before_gap_detection
        ),
        candidate_signal_count=len(_signal_ids_for_status(signals, "candidate")),
        review_required_signal_count=len(
            _signal_ids_for_status(signals, "review_required")
        ),
        guarded_signal_count=len(_signal_ids_for_status(signals, "guarded")),
        high_confidence_signal_count=len(
            _signal_ids_for_confidence(signals, "high", "guarded")
        ),
        hitl_required_signal_count=sum(
            1 for signal in signals if signal.hitl_required_before_gap_detection
        ),
        highest_gap_score=max(signal.gap_score for signal in signals),
        overall_gap_score=_overall_gap_score(signals),
        overall_gap_posture=_overall_gap_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_gap_signal_by_id(
    signal_id: str,
    plan: KnowledgeGapDetectionPlan | None = None,
) -> KnowledgeGapSignal | None:
    """Return one knowledge gap signal without executing gap detection."""

    source_plan = plan or build_knowledge_gap_detection()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_gap_signals_for_status(
    status: KnowledgeGapDetectionStatus,
    plan: KnowledgeGapDetectionPlan | None = None,
) -> tuple[KnowledgeGapSignal, ...]:
    """Return knowledge gap signals by advisory status."""

    source_plan = plan or build_knowledge_gap_detection()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_gap_signals_for_confidence(
    confidence: KnowledgeGapDetectionConfidence,
    plan: KnowledgeGapDetectionPlan | None = None,
) -> tuple[KnowledgeGapSignal, ...]:
    """Return knowledge gap signals by confidence band."""

    source_plan = plan or build_knowledge_gap_detection()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    quality_plan: KnowledgeQualityScoringPlan,
) -> tuple[KnowledgeGapSignal, ...]:
    return (
        _signal(
            kind="knowledge_gap_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            quality_signal_ids=quality_plan.signal_ids,
            quality_plan=quality_plan,
            gap_signal_score=88,
            coverage_signal_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_gap_coverage_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="coverage_review",
            quality_signal_ids=(
                "knowledge_quality_scoring::knowledge_quality_inventory_review",
                "knowledge_quality_scoring::knowledge_quality_completeness_review",
            ),
            quality_plan=quality_plan,
            gap_signal_score=78,
            coverage_signal_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_gap_priority_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="priority_review",
            quality_signal_ids=(
                "knowledge_quality_scoring::knowledge_quality_accuracy_review",
                "knowledge_quality_scoring::knowledge_quality_scoring_readiness",
            ),
            quality_plan=quality_plan,
            gap_signal_score=70,
            coverage_signal_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_gap_detection_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="detection_readiness",
            quality_signal_ids=(
                "knowledge_quality_scoring::knowledge_quality_scoring_readiness",
                "knowledge_quality_scoring::knowledge_quality_governance_gate",
            ),
            quality_plan=quality_plan,
            gap_signal_score=62,
            coverage_signal_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_gap_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            quality_signal_ids=quality_plan.signal_ids,
            quality_plan=quality_plan,
            gap_signal_score=44,
            coverage_signal_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeGapDetectionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeGapDetectionAxis,
    quality_signal_ids: tuple[str, ...],
    quality_plan: KnowledgeQualityScoringPlan,
    gap_signal_score: int,
    coverage_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeGapSignal:
    score = _gap_score(
        gap_signal_score=gap_signal_score,
        coverage_signal_score=coverage_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeGapSignal(
        signal_id=f"knowledge_gap_detection::{kind}",
        signal_kind=kind,
        status=_gap_status(score),
        confidence=_gap_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        gap_axis=axis,
        knowledge_quality_signal_ids=quality_signal_ids,
        knowledge_quality_signal_count=len(quality_signal_ids),
        source_count=quality_plan.source_count,
        domain_count=quality_plan.domain_count,
        gap_signal_summary=_signal_summary(kind),
        gap_signal_score=gap_signal_score,
        coverage_signal_score=coverage_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        gap_score=score,
        hitl_required_before_gap_detection=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, quality_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_quality_signal_count:{len(quality_signal_ids)}",
            f"source_count:{quality_plan.source_count}",
            f"domain_count:{quality_plan.domain_count}",
            f"gap_axis:{axis}",
            f"gap_signal_score:{gap_signal_score}",
            f"coverage_signal_score:{coverage_signal_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_gap_detection:true",
        ),
    )


def _gap_score(
    *,
    gap_signal_score: int,
    coverage_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            gap_signal_score * 3
            + coverage_signal_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _gap_status(score: int) -> KnowledgeGapDetectionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _gap_confidence(score: int) -> KnowledgeGapDetectionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_gap_score(signals: tuple[KnowledgeGapSignal, ...]) -> int:
    base = sum(signal.gap_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_gap_posture(
    signals: tuple[KnowledgeGapSignal, ...],
) -> KnowledgeGapDetectionPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeGapSignal, ...],
    status: KnowledgeGapDetectionStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeGapSignal, ...],
    *confidences: KnowledgeGapDetectionConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeGapSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_gap_detection_metadata",
        "verify_knowledge_gap_roadmap_traceability",
        "review_quality_signals_before_any_gap_action",
        "require_hitl_before_gap_scan_priority_remediation_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeGapDetectionKind,
    axis: KnowledgeGapDetectionAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_gap_detection",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeGapDetectionKind,
    quality_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_quality_signal_count:{len(quality_signal_ids)}",
        "composes_knowledge_quality_metadata",
        "keeps_knowledge_gap_detection_execution_disabled",
        "requires_human_review_before_gap_scan_or_remediation",
    )


def _signal_actions(kind: KnowledgeGapDetectionKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_gap_signal_metadata",
        "verify_knowledge_quality_traceability",
        "keep_knowledge_gap_detection_execution_disabled",
        "require_hitl_before_knowledge_gap_action",
    )
    if kind == "knowledge_gap_coverage_review":
        return base_actions + ("review_gap_coverage_metadata",)
    if kind == "knowledge_gap_priority_review":
        return base_actions + ("review_gap_priority_metadata",)
    if kind == "knowledge_gap_detection_readiness":
        return base_actions + ("review_gap_detection_readiness_metadata",)
    if kind == "knowledge_gap_governance_gate":
        return base_actions + ("confirm_manual_gap_governance_gate",)
    return base_actions + ("review_knowledge_gap_inventory_metadata",)


def _signal_summary(kind: KnowledgeGapDetectionKind) -> str:
    summaries: dict[KnowledgeGapDetectionKind, str] = {
        "knowledge_gap_inventory_review": (
            "Advisory knowledge gap inventory posture over quality metadata "
            "without executing gap scans."
        ),
        "knowledge_gap_coverage_review": (
            "Advisory posture for reviewing coverage gap metadata before any "
            "source addition or KB enrichment."
        ),
        "knowledge_gap_priority_review": (
            "Advisory posture for reviewing gap priority metadata while "
            "priority assignment and remediation remain disabled."
        ),
        "knowledge_gap_detection_readiness": (
            "Advisory posture for reviewing gap detection readiness without "
            "backfills, retrieval mutation, or storage writes."
        ),
        "knowledge_gap_governance_gate": (
            "Governed manual gate that keeps knowledge gap detection disabled "
            "until HITL approval."
        ),
    }
    return summaries[kind]


def _resolve_route(route: RouteName | str) -> RouteName:
    return route if isinstance(route, RouteName) else RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    normalized = str(task_type).strip()
    if normalized not in get_args(TaskRoutingType):
        raise ValueError(f"Unknown task routing type: {task_type}")
    return cast(TaskRoutingType, normalized)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in allowed_modes:
        raise ValueError(f"Unknown execution mode: {execution_mode_id}")
    return cast(ExecutionModeId, normalized)
