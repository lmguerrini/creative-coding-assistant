"""V6.3 advisory knowledge drift detection metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_conflict_resolver import (
    KnowledgeConflictResolverPlan,
    build_knowledge_conflict_resolver,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeDriftDetectionKind = Literal[
    "knowledge_drift_inventory_review",
    "knowledge_drift_temporal_signal_review",
    "knowledge_drift_conflict_alignment_review",
    "knowledge_drift_detection_readiness",
    "knowledge_drift_governance_gate",
]
KnowledgeDriftDetectionStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeDriftDetectionConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeDriftDetectionPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeDriftDetectionAxis = Literal[
    "inventory_review",
    "temporal_signal_review",
    "conflict_alignment_review",
    "detection_readiness",
    "governance_gate",
]

KNOWLEDGE_DRIFT_ENTRY_SERIALIZATION_VERSION = "knowledge_drift_entry.v1"
KNOWLEDGE_DRIFT_PLAN_SERIALIZATION_VERSION = "knowledge_drift_plan.v1"
KNOWLEDGE_DRIFT_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Drift Detection exposes knowledge conflict, drift "
    "inventory, temporal signal, conflict alignment, detection readiness, and "
    "governance posture as inspectable advisory metadata only; it does not "
    "execute drift detection, detect drift, scan timelines, compare snapshots, "
    "compute drift scores, write drift records, execute conflict resolution, "
    "detect conflicts, resolve conflicts, arbitrate sources, write conflict "
    "records, mutate source precedence, scan for gaps, remediate gaps, enrich "
    "the KB, write KB storage, compute quality scores, execute retrieval "
    "queries, mutate retrieval configuration, mutate ranking, request "
    "embeddings, refresh embeddings, index vectors, upsert vectors, fetch "
    "documentation, update source records, provision providers, infer API "
    "keys, route providers or models, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, modify generated output, or "
    "apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Drift Detection",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_drift_detection_execution",
    "drift_detection_execution",
    "drift_scan_execution",
    "timeline_scan_execution",
    "snapshot_comparison_execution",
    "drift_score_computation",
    "drift_record_write",
    "knowledge_conflict_resolver_execution",
    "conflict_detection_execution",
    "conflict_resolution_execution",
    "conflict_arbitration_execution",
    "conflict_record_write",
    "source_precedence_mutation",
    "gap_scan_execution",
    "gap_remediation_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "quality_score_computation",
    "retrieval_query_execution",
    "retrieval_configuration_mutation",
    "ranking_mutation",
    "embedding_request_execution",
    "embedding_refresh_execution",
    "vector_indexing",
    "vector_upsert",
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
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeDriftSignal(BaseModel):
    """One advisory knowledge drift detection signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeDriftDetectionKind
    status: KnowledgeDriftDetectionStatus
    confidence: KnowledgeDriftDetectionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    drift_axis: KnowledgeDriftDetectionAxis
    knowledge_conflict_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    knowledge_conflict_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    drift_signal_summary: str = Field(min_length=1, max_length=360)
    drift_signal_score: int = Field(ge=0, le=100)
    temporal_signal_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    drift_score: int = Field(ge=0, le=1_000)
    hitl_required_before_drift_detection: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    knowledge_drift_detection_capability_implemented: Literal[True] = True
    knowledge_drift_detection_metadata_implemented: Literal[True] = True
    knowledge_conflict_metadata_used: Literal[True] = True
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    drift_scan_execution_implemented: Literal[False] = False
    timeline_scan_execution_implemented: Literal[False] = False
    snapshot_comparison_execution_implemented: Literal[False] = False
    drift_score_computation_implemented: Literal[False] = False
    drift_record_write_implemented: Literal[False] = False
    knowledge_conflict_resolver_execution_implemented: Literal[False] = False
    conflict_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    conflict_record_write_implemented: Literal[False] = False
    source_precedence_mutation_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_drift_entry.v1"] = (
        KNOWLEDGE_DRIFT_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_drift_detection::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_conflict_signal_count != len(
            self.knowledge_conflict_signal_ids
        ):
            raise ValueError("knowledge_conflict_signal_count must match signals")
        if self.drift_score != _drift_score(
            drift_signal_score=self.drift_signal_score,
            temporal_signal_score=self.temporal_signal_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("drift_score must combine source scores")
        if self.status != _drift_status(self.drift_score):
            raise ValueError("status must match drift_score")
        if self.confidence != _drift_confidence(self.drift_score):
            raise ValueError("confidence must match drift_score")
        if not self.hitl_required_before_drift_detection:
            raise ValueError("knowledge drift detection requires HITL posture")
        return self


class KnowledgeDriftDetectionPlan(BaseModel):
    """Bounded V6.3 advisory knowledge drift detection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_drift_detection"] = "knowledge_drift_detection"
    serialization_version: Literal["knowledge_drift_plan.v1"] = (
        KNOWLEDGE_DRIFT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_DRIFT_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_conflict_role: Literal["knowledge_conflict_resolver"] = (
        "knowledge_conflict_resolver"
    )
    knowledge_conflict_serialization_version: Literal["knowledge_conflict_plan.v1"]
    knowledge_conflict_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    knowledge_conflict_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeDriftSignal, ...] = Field(min_length=5, max_length=5)
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
    planned_drift_detection_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    detected_drift_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    scanned_timeline_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    compared_snapshot_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_drift_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_drift_record_ids: tuple[str, ...] = Field(
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
    highest_drift_score: int = Field(ge=0, le=1_000)
    overall_drift_score: int = Field(ge=0, le=1_000)
    overall_drift_posture: KnowledgeDriftDetectionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    knowledge_drift_detection_capability_implemented: Literal[True] = True
    knowledge_drift_detection_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_conflict_metadata_used: Literal[True] = True
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    drift_scan_execution_implemented: Literal[False] = False
    timeline_scan_execution_implemented: Literal[False] = False
    snapshot_comparison_execution_implemented: Literal[False] = False
    drift_score_computation_implemented: Literal[False] = False
    drift_record_write_implemented: Literal[False] = False
    knowledge_conflict_resolver_execution_implemented: Literal[False] = False
    conflict_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    conflict_record_write_implemented: Literal[False] = False
    source_precedence_mutation_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
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
            if signal.hitl_required_before_drift_detection
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_drift_detection_ids:
            raise ValueError("planned_drift_detection_ids must remain empty")
        if self.detected_drift_ids:
            raise ValueError("detected_drift_ids must remain empty")
        if self.scanned_timeline_ids:
            raise ValueError("scanned_timeline_ids must remain empty")
        if self.compared_snapshot_ids:
            raise ValueError("compared_snapshot_ids must remain empty")
        if self.computed_drift_score_ids:
            raise ValueError("computed_drift_score_ids must remain empty")
        if self.written_drift_record_ids:
            raise ValueError("written_drift_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_conflict_signal_count != len(
            self.knowledge_conflict_signal_ids
        ):
            raise ValueError("knowledge_conflict_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 11 roadmap")
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
        if self.highest_drift_score != max(
            signal.drift_score for signal in self.signals
        ):
            raise ValueError("highest_drift_score must match signals")
        if self.overall_drift_score != _overall_drift_score(self.signals):
            raise ValueError("overall_drift_score must match signals")
        if self.overall_drift_posture != _overall_drift_posture(self.signals):
            raise ValueError("overall_drift_posture must match signals")
        declared_conflict_signals = set(self.knowledge_conflict_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_conflict_signal_ids).issubset(
                declared_conflict_signals
            ):
                raise ValueError("signal knowledge_conflict_signal_ids must be known")
        return self


def build_knowledge_drift_detection(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_conflict: KnowledgeConflictResolverPlan | None = None,
) -> KnowledgeDriftDetectionPlan:
    """Build V6.3 Task 11 knowledge drift detection metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    conflict_plan = knowledge_conflict or build_knowledge_conflict_resolver(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        conflict_plan=conflict_plan,
    )
    return KnowledgeDriftDetectionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=conflict_plan.checked_at,
        knowledge_conflict_serialization_version=conflict_plan.serialization_version,
        knowledge_conflict_signal_ids=conflict_plan.signal_ids,
        knowledge_conflict_signal_count=len(conflict_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=conflict_plan.source_count,
        domain_count=conflict_plan.domain_count,
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
            if signal.hitl_required_before_drift_detection
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
            1 for signal in signals if signal.hitl_required_before_drift_detection
        ),
        highest_drift_score=max(signal.drift_score for signal in signals),
        overall_drift_score=_overall_drift_score(signals),
        overall_drift_posture=_overall_drift_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_drift_signal_by_id(
    signal_id: str,
    plan: KnowledgeDriftDetectionPlan | None = None,
) -> KnowledgeDriftSignal | None:
    """Return one knowledge drift signal without detecting drift."""

    source_plan = plan or build_knowledge_drift_detection()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_drift_signals_for_status(
    status: KnowledgeDriftDetectionStatus,
    plan: KnowledgeDriftDetectionPlan | None = None,
) -> tuple[KnowledgeDriftSignal, ...]:
    """Return knowledge drift signals by advisory status."""

    source_plan = plan or build_knowledge_drift_detection()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_drift_signals_for_confidence(
    confidence: KnowledgeDriftDetectionConfidence,
    plan: KnowledgeDriftDetectionPlan | None = None,
) -> tuple[KnowledgeDriftSignal, ...]:
    """Return knowledge drift signals by confidence band."""

    source_plan = plan or build_knowledge_drift_detection()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    conflict_plan: KnowledgeConflictResolverPlan,
) -> tuple[KnowledgeDriftSignal, ...]:
    return (
        _signal(
            kind="knowledge_drift_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            conflict_signal_ids=conflict_plan.signal_ids,
            conflict_plan=conflict_plan,
            drift_signal_score=88,
            temporal_signal_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_drift_temporal_signal_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="temporal_signal_review",
            conflict_signal_ids=(
                "knowledge_conflict_resolver::knowledge_conflict_inventory_review",
                "knowledge_conflict_resolver::"
                "knowledge_conflict_source_disagreement_review",
            ),
            conflict_plan=conflict_plan,
            drift_signal_score=78,
            temporal_signal_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_drift_conflict_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="conflict_alignment_review",
            conflict_signal_ids=(
                "knowledge_conflict_resolver::"
                "knowledge_conflict_resolution_policy_review",
                "knowledge_conflict_resolver::knowledge_conflict_governance_gate",
            ),
            conflict_plan=conflict_plan,
            drift_signal_score=70,
            temporal_signal_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_drift_detection_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="detection_readiness",
            conflict_signal_ids=(
                "knowledge_conflict_resolver::knowledge_conflict_resolver_readiness",
                "knowledge_conflict_resolver::knowledge_conflict_governance_gate",
            ),
            conflict_plan=conflict_plan,
            drift_signal_score=62,
            temporal_signal_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_drift_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            conflict_signal_ids=conflict_plan.signal_ids,
            conflict_plan=conflict_plan,
            drift_signal_score=44,
            temporal_signal_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeDriftDetectionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeDriftDetectionAxis,
    conflict_signal_ids: tuple[str, ...],
    conflict_plan: KnowledgeConflictResolverPlan,
    drift_signal_score: int,
    temporal_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeDriftSignal:
    score = _drift_score(
        drift_signal_score=drift_signal_score,
        temporal_signal_score=temporal_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeDriftSignal(
        signal_id=f"knowledge_drift_detection::{kind}",
        signal_kind=kind,
        status=_drift_status(score),
        confidence=_drift_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        drift_axis=axis,
        knowledge_conflict_signal_ids=conflict_signal_ids,
        knowledge_conflict_signal_count=len(conflict_signal_ids),
        source_count=conflict_plan.source_count,
        domain_count=conflict_plan.domain_count,
        drift_signal_summary=_signal_summary(kind),
        drift_signal_score=drift_signal_score,
        temporal_signal_score=temporal_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        drift_score=score,
        hitl_required_before_drift_detection=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, conflict_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_conflict_signal_count:{len(conflict_signal_ids)}",
            f"source_count:{conflict_plan.source_count}",
            f"domain_count:{conflict_plan.domain_count}",
            f"drift_axis:{axis}",
            f"drift_signal_score:{drift_signal_score}",
            f"temporal_signal_score:{temporal_signal_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_drift_detection:true",
        ),
    )


def _drift_score(
    *,
    drift_signal_score: int,
    temporal_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            drift_signal_score * 3
            + temporal_signal_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _drift_status(score: int) -> KnowledgeDriftDetectionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _drift_confidence(score: int) -> KnowledgeDriftDetectionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_drift_score(signals: tuple[KnowledgeDriftSignal, ...]) -> int:
    base = sum(signal.drift_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_drift_posture(
    signals: tuple[KnowledgeDriftSignal, ...],
) -> KnowledgeDriftDetectionPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeDriftSignal, ...],
    status: KnowledgeDriftDetectionStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeDriftSignal, ...],
    *confidences: KnowledgeDriftDetectionConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeDriftSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_drift_detection_metadata",
        "verify_knowledge_drift_roadmap_traceability",
        "review_conflict_signals_before_any_drift_action",
        "require_hitl_before_drift_detection_timeline_scan_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeDriftDetectionKind,
    axis: KnowledgeDriftDetectionAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_drift_detection",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeDriftDetectionKind,
    conflict_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_conflict_signal_count:{len(conflict_signal_ids)}",
        "composes_knowledge_conflict_metadata",
        "keeps_drift_detection_execution_disabled",
        "requires_human_review_before_drift_scan_snapshot_compare_or_write",
    )


def _signal_actions(kind: KnowledgeDriftDetectionKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_drift_signal_metadata",
        "verify_knowledge_conflict_traceability",
        "keep_knowledge_drift_detection_disabled",
        "require_hitl_before_knowledge_drift_action",
    )
    if kind == "knowledge_drift_temporal_signal_review":
        return base_actions + ("review_temporal_drift_signal_metadata",)
    if kind == "knowledge_drift_conflict_alignment_review":
        return base_actions + ("review_conflict_alignment_metadata",)
    if kind == "knowledge_drift_detection_readiness":
        return base_actions + ("review_drift_detection_readiness_metadata",)
    if kind == "knowledge_drift_governance_gate":
        return base_actions + ("confirm_manual_drift_governance_gate",)
    return base_actions + ("review_knowledge_drift_inventory_metadata",)


def _signal_summary(kind: KnowledgeDriftDetectionKind) -> str:
    summaries: dict[KnowledgeDriftDetectionKind, str] = {
        "knowledge_drift_inventory_review": (
            "Advisory knowledge drift inventory posture over conflict metadata "
            "without executing drift detection."
        ),
        "knowledge_drift_temporal_signal_review": (
            "Advisory posture for reviewing temporal drift signal metadata "
            "before timeline scanning or snapshot comparison."
        ),
        "knowledge_drift_conflict_alignment_review": (
            "Advisory posture for aligning drift metadata with conflict "
            "signals while conflict resolution remains disabled."
        ),
        "knowledge_drift_detection_readiness": (
            "Advisory posture for reviewing drift detection readiness without "
            "drift scans, snapshot comparison, KB writes, or retrieval mutation."
        ),
        "knowledge_drift_governance_gate": (
            "Governed manual gate that keeps knowledge drift detection "
            "disabled until HITL approval."
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
