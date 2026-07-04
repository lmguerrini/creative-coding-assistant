"""V6.3 advisory source reliability engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_drift_detection import (
    KnowledgeDriftDetectionPlan,
    build_knowledge_drift_detection,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

SourceReliabilityKind = Literal[
    "source_reliability_inventory_review",
    "source_reliability_health_signal_review",
    "source_reliability_drift_alignment_review",
    "source_reliability_engine_readiness",
    "source_reliability_governance_gate",
]
SourceReliabilityStatus = Literal["candidate", "review_required", "guarded"]
SourceReliabilityConfidence = Literal["low", "medium", "high", "guarded"]
SourceReliabilityPosture = Literal["candidate", "review_required", "guarded"]
SourceReliabilityAxis = Literal[
    "inventory_review",
    "health_signal_review",
    "drift_alignment_review",
    "engine_readiness",
    "governance_gate",
]

SOURCE_RELIABILITY_ENTRY_SERIALIZATION_VERSION = "source_reliability_entry.v1"
SOURCE_RELIABILITY_PLAN_SERIALIZATION_VERSION = "source_reliability_plan.v1"
SOURCE_RELIABILITY_AUTHORITY_BOUNDARY = (
    "V6.3 Source Reliability Engine exposes knowledge drift, source "
    "reliability inventory, source health signal, drift alignment, engine "
    "readiness, and governance posture as inspectable advisory metadata only; "
    "it does not execute source reliability engine actions, score source "
    "reliability, check source health, compute source trust scores, mutate "
    "source rank, mutate source registries, fetch sources, update source "
    "records, scan source freshness, execute drift detection, detect drift, "
    "scan timelines, compare snapshots, compute drift scores, write drift "
    "records, execute conflict resolution, detect conflicts, resolve "
    "conflicts, arbitrate sources, write conflict records, mutate source "
    "precedence, scan for gaps, remediate gaps, enrich the KB, write KB "
    "storage, compute quality scores, execute retrieval queries, mutate "
    "retrieval configuration, mutate ranking, request embeddings, refresh "
    "embeddings, index vectors, upsert vectors, fetch documentation, "
    "provision providers, infer API keys, route providers or models, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Source Reliability Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "source_reliability_engine_execution",
    "source_reliability_scoring_execution",
    "source_health_check_execution",
    "source_trust_score_computation",
    "source_rank_mutation",
    "source_registry_mutation",
    "source_fetch_execution",
    "source_record_update",
    "freshness_scan_execution",
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


class SourceReliabilitySignal(BaseModel):
    """One advisory source reliability signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: SourceReliabilityKind
    status: SourceReliabilityStatus
    confidence: SourceReliabilityConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    reliability_axis: SourceReliabilityAxis
    knowledge_drift_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    knowledge_drift_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    reliability_signal_summary: str = Field(min_length=1, max_length=360)
    reliability_signal_score: int = Field(ge=0, le=100)
    health_signal_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    reliability_score: int = Field(ge=0, le=1_000)
    hitl_required_before_source_reliability: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    source_reliability_engine_capability_implemented: Literal[True] = True
    source_reliability_engine_metadata_implemented: Literal[True] = True
    knowledge_drift_metadata_used: Literal[True] = True
    source_reliability_engine_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    source_trust_score_computation_implemented: Literal[False] = False
    source_rank_mutation_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
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
    serialization_version: Literal["source_reliability_entry.v1"] = (
        SOURCE_RELIABILITY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"source_reliability_engine::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_drift_signal_count != len(self.knowledge_drift_signal_ids):
            raise ValueError("knowledge_drift_signal_count must match signals")
        if self.reliability_score != _reliability_score(
            reliability_signal_score=self.reliability_signal_score,
            health_signal_score=self.health_signal_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("reliability_score must combine source scores")
        if self.status != _reliability_status(self.reliability_score):
            raise ValueError("status must match reliability_score")
        if self.confidence != _reliability_confidence(self.reliability_score):
            raise ValueError("confidence must match reliability_score")
        if not self.hitl_required_before_source_reliability:
            raise ValueError("source reliability actions require HITL posture")
        return self


class SourceReliabilityEnginePlan(BaseModel):
    """Bounded V6.3 advisory source reliability engine plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["source_reliability_engine"] = "source_reliability_engine"
    serialization_version: Literal["source_reliability_plan.v1"] = (
        SOURCE_RELIABILITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SOURCE_RELIABILITY_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_drift_role: Literal["knowledge_drift_detection"] = (
        "knowledge_drift_detection"
    )
    knowledge_drift_serialization_version: Literal["knowledge_drift_plan.v1"]
    knowledge_drift_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    knowledge_drift_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[SourceReliabilitySignal, ...] = Field(min_length=5, max_length=5)
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
    planned_source_reliability_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_source_reliability_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    checked_source_health_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_source_trust_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_source_rank_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_source_registry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    updated_source_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    scanned_freshness_ids: tuple[str, ...] = Field(
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
    highest_reliability_score: int = Field(ge=0, le=1_000)
    overall_reliability_score: int = Field(ge=0, le=1_000)
    overall_reliability_posture: SourceReliabilityPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    source_reliability_engine_capability_implemented: Literal[True] = True
    source_reliability_engine_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_drift_metadata_used: Literal[True] = True
    source_reliability_engine_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    source_trust_score_computation_implemented: Literal[False] = False
    source_rank_mutation_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
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
            if signal.hitl_required_before_source_reliability
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_source_reliability_ids:
            raise ValueError("planned_source_reliability_ids must remain empty")
        if self.computed_source_reliability_ids:
            raise ValueError("computed_source_reliability_ids must remain empty")
        if self.checked_source_health_ids:
            raise ValueError("checked_source_health_ids must remain empty")
        if self.computed_source_trust_score_ids:
            raise ValueError("computed_source_trust_score_ids must remain empty")
        if self.mutated_source_rank_ids:
            raise ValueError("mutated_source_rank_ids must remain empty")
        if self.mutated_source_registry_ids:
            raise ValueError("mutated_source_registry_ids must remain empty")
        if self.fetched_source_ids:
            raise ValueError("fetched_source_ids must remain empty")
        if self.updated_source_record_ids:
            raise ValueError("updated_source_record_ids must remain empty")
        if self.scanned_freshness_ids:
            raise ValueError("scanned_freshness_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_drift_signal_count != len(self.knowledge_drift_signal_ids):
            raise ValueError("knowledge_drift_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 12 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.candidate_signal_count != len(self.candidate_signal_ids):
            raise ValueError("candidate_signal_count must match signals")
        if self.review_required_signal_count != len(self.review_required_signal_ids):
            raise ValueError("review_required_signal_count must match signals")
        if self.guarded_signal_count != len(self.guarded_signal_ids):
            raise ValueError("guarded_signal_count must match signals")
        if self.high_confidence_signal_count != len(self.high_confidence_signal_ids):
            raise ValueError("high_confidence_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.highest_reliability_score != max(
            signal.reliability_score for signal in self.signals
        ):
            raise ValueError("highest_reliability_score must match signals")
        if self.overall_reliability_score != _overall_reliability_score(self.signals):
            raise ValueError("overall_reliability_score must match signals")
        if self.overall_reliability_posture != _overall_reliability_posture(
            self.signals
        ):
            raise ValueError("overall_reliability_posture must match signals")
        declared_drift_signals = set(self.knowledge_drift_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_drift_signal_ids).issubset(
                declared_drift_signals
            ):
                raise ValueError("signal knowledge_drift_signal_ids must be known")
        return self


def build_source_reliability_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_drift: KnowledgeDriftDetectionPlan | None = None,
) -> SourceReliabilityEnginePlan:
    """Build V6.3 Task 12 source reliability engine metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    drift_plan = knowledge_drift or build_knowledge_drift_detection(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        drift_plan=drift_plan,
    )
    return SourceReliabilityEnginePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=drift_plan.checked_at,
        knowledge_drift_serialization_version=drift_plan.serialization_version,
        knowledge_drift_signal_ids=drift_plan.signal_ids,
        knowledge_drift_signal_count=len(drift_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=drift_plan.source_count,
        domain_count=drift_plan.domain_count,
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
            if signal.hitl_required_before_source_reliability
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
            1 for signal in signals if signal.hitl_required_before_source_reliability
        ),
        highest_reliability_score=max(signal.reliability_score for signal in signals),
        overall_reliability_score=_overall_reliability_score(signals),
        overall_reliability_posture=_overall_reliability_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def source_reliability_signal_by_id(
    signal_id: str,
    plan: SourceReliabilityEnginePlan | None = None,
) -> SourceReliabilitySignal | None:
    """Return one source reliability signal without scoring sources."""

    source_plan = plan or build_source_reliability_engine()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def source_reliability_signals_for_status(
    status: SourceReliabilityStatus,
    plan: SourceReliabilityEnginePlan | None = None,
) -> tuple[SourceReliabilitySignal, ...]:
    """Return source reliability signals by advisory status."""

    source_plan = plan or build_source_reliability_engine()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def source_reliability_signals_for_confidence(
    confidence: SourceReliabilityConfidence,
    plan: SourceReliabilityEnginePlan | None = None,
) -> tuple[SourceReliabilitySignal, ...]:
    """Return source reliability signals by confidence band."""

    source_plan = plan or build_source_reliability_engine()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    drift_plan: KnowledgeDriftDetectionPlan,
) -> tuple[SourceReliabilitySignal, ...]:
    return (
        _signal(
            kind="source_reliability_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            drift_signal_ids=drift_plan.signal_ids,
            drift_plan=drift_plan,
            reliability_signal_score=88,
            health_signal_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="source_reliability_health_signal_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="health_signal_review",
            drift_signal_ids=(
                "knowledge_drift_detection::knowledge_drift_inventory_review",
                "knowledge_drift_detection::knowledge_drift_temporal_signal_review",
            ),
            drift_plan=drift_plan,
            reliability_signal_score=78,
            health_signal_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="source_reliability_drift_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="drift_alignment_review",
            drift_signal_ids=(
                "knowledge_drift_detection::knowledge_drift_conflict_alignment_review",
                "knowledge_drift_detection::knowledge_drift_governance_gate",
            ),
            drift_plan=drift_plan,
            reliability_signal_score=70,
            health_signal_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="source_reliability_engine_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="engine_readiness",
            drift_signal_ids=(
                "knowledge_drift_detection::knowledge_drift_detection_readiness",
                "knowledge_drift_detection::knowledge_drift_governance_gate",
            ),
            drift_plan=drift_plan,
            reliability_signal_score=62,
            health_signal_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="source_reliability_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            drift_signal_ids=drift_plan.signal_ids,
            drift_plan=drift_plan,
            reliability_signal_score=44,
            health_signal_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: SourceReliabilityKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: SourceReliabilityAxis,
    drift_signal_ids: tuple[str, ...],
    drift_plan: KnowledgeDriftDetectionPlan,
    reliability_signal_score: int,
    health_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> SourceReliabilitySignal:
    score = _reliability_score(
        reliability_signal_score=reliability_signal_score,
        health_signal_score=health_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return SourceReliabilitySignal(
        signal_id=f"source_reliability_engine::{kind}",
        signal_kind=kind,
        status=_reliability_status(score),
        confidence=_reliability_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        reliability_axis=axis,
        knowledge_drift_signal_ids=drift_signal_ids,
        knowledge_drift_signal_count=len(drift_signal_ids),
        source_count=drift_plan.source_count,
        domain_count=drift_plan.domain_count,
        reliability_signal_summary=_signal_summary(kind),
        reliability_signal_score=reliability_signal_score,
        health_signal_score=health_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        reliability_score=score,
        hitl_required_before_source_reliability=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, drift_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_drift_signal_count:{len(drift_signal_ids)}",
            f"source_count:{drift_plan.source_count}",
            f"domain_count:{drift_plan.domain_count}",
            f"reliability_axis:{axis}",
            f"reliability_signal_score:{reliability_signal_score}",
            f"health_signal_score:{health_signal_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_source_reliability:true",
        ),
    )


def _reliability_score(
    *,
    reliability_signal_score: int,
    health_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            reliability_signal_score * 3
            + health_signal_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _reliability_status(score: int) -> SourceReliabilityStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _reliability_confidence(score: int) -> SourceReliabilityConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_reliability_score(signals: tuple[SourceReliabilitySignal, ...]) -> int:
    base = sum(signal.reliability_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_reliability_posture(
    signals: tuple[SourceReliabilitySignal, ...],
) -> SourceReliabilityPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[SourceReliabilitySignal, ...],
    status: SourceReliabilityStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[SourceReliabilitySignal, ...],
    *confidences: SourceReliabilityConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[SourceReliabilitySignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_source_reliability_engine_metadata",
        "verify_source_reliability_roadmap_traceability",
        "review_drift_signals_before_any_source_reliability_action",
        "require_hitl_before_source_scoring_health_check_fetch_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: SourceReliabilityKind,
    axis: SourceReliabilityAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "source_reliability_engine",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: SourceReliabilityKind,
    drift_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_drift_signal_count:{len(drift_signal_ids)}",
        "composes_knowledge_drift_metadata",
        "keeps_source_reliability_execution_disabled",
        "requires_human_review_before_source_scoring_fetch_update_or_write",
    )


def _signal_actions(kind: SourceReliabilityKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_source_reliability_signal_metadata",
        "verify_knowledge_drift_traceability",
        "keep_source_reliability_engine_disabled",
        "require_hitl_before_source_reliability_action",
    )
    if kind == "source_reliability_health_signal_review":
        return base_actions + ("review_source_health_signal_metadata",)
    if kind == "source_reliability_drift_alignment_review":
        return base_actions + ("review_source_drift_alignment_metadata",)
    if kind == "source_reliability_engine_readiness":
        return base_actions + ("review_source_reliability_readiness_metadata",)
    if kind == "source_reliability_governance_gate":
        return base_actions + ("confirm_manual_source_reliability_governance_gate",)
    return base_actions + ("review_source_reliability_inventory_metadata",)


def _signal_summary(kind: SourceReliabilityKind) -> str:
    summaries: dict[SourceReliabilityKind, str] = {
        "source_reliability_inventory_review": (
            "Advisory source reliability inventory posture over drift metadata "
            "without executing source scoring or source checks."
        ),
        "source_reliability_health_signal_review": (
            "Advisory posture for reviewing source health signal metadata "
            "before source reliability scoring or source fetches."
        ),
        "source_reliability_drift_alignment_review": (
            "Advisory posture for aligning source reliability metadata with "
            "knowledge drift signals while mutation remains disabled."
        ),
        "source_reliability_engine_readiness": (
            "Advisory posture for reviewing source reliability readiness "
            "without health checks, source updates, KB writes, or retrieval "
            "mutation."
        ),
        "source_reliability_governance_gate": (
            "Governed manual gate that keeps source reliability engine actions "
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
