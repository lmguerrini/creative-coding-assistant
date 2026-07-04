"""V6.3 advisory knowledge freshness tracking metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_rollback import (
    KnowledgeRollbackPlan,
    build_knowledge_rollback,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeFreshnessKind = Literal[
    "knowledge_freshness_inventory_review",
    "knowledge_freshness_source_age_review",
    "knowledge_freshness_staleness_readiness",
    "knowledge_freshness_rollback_alignment_review",
    "knowledge_freshness_governance_gate",
]
KnowledgeFreshnessStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeFreshnessConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeFreshnessPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeFreshnessAxis = Literal[
    "inventory_review",
    "source_age",
    "staleness_readiness",
    "rollback_alignment",
    "governance_gate",
]

KNOWLEDGE_FRESHNESS_ENTRY_SERIALIZATION_VERSION = "knowledge_freshness_entry.v1"
KNOWLEDGE_FRESHNESS_PLAN_SERIALIZATION_VERSION = "knowledge_freshness_plan.v1"
KNOWLEDGE_FRESHNESS_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Freshness Tracking exposes knowledge rollback, freshness "
    "inventory, source age, staleness readiness, rollback alignment, and "
    "governance posture as inspectable advisory metadata only; it does not "
    "execute freshness tracking, run freshness scans, compute freshness "
    "scores, write freshness records, update source timestamps, mutate "
    "staleness state, fetch sources, restore KB state, execute rollback, "
    "apply rollback plans, execute snapshot operations, write snapshot "
    "records or storage, execute knowledge versioning, mutate version graphs, "
    "write version records, mutate provenance graphs, write provenance "
    "records, execute lifecycle management, mutate lifecycle or retention "
    "policy, execute knowledge consolidation, write canonical records, write "
    "KB storage, update source records, execute retrieval queries, mutate "
    "retrieval configuration, mutate ranking, request embeddings, refresh "
    "embeddings, index vectors, upsert vectors, fetch documentation, "
    "provision providers, infer API keys, route providers or models, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "execute workflows, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Freshness Tracking",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_freshness_tracking_execution",
    "freshness_scan_execution",
    "freshness_score_computation",
    "freshness_record_write",
    "source_timestamp_update",
    "staleness_state_mutation",
    "source_fetch_execution",
    "kb_state_restore",
    "knowledge_rollback_execution",
    "rollback_plan_application",
    "rollback_state_mutation",
    "rollback_record_write",
    "snapshot_restore_execution",
    "knowledge_snapshot_engine_execution",
    "snapshot_creation",
    "snapshot_record_write",
    "snapshot_storage_write",
    "knowledge_versioning_execution",
    "version_graph_mutation",
    "version_record_write",
    "provenance_graph_mutation",
    "provenance_record_write",
    "knowledge_lifecycle_management_execution",
    "lifecycle_policy_mutation",
    "retention_policy_mutation",
    "knowledge_consolidation_execution",
    "knowledge_merge_execution",
    "knowledge_deduplication_execution",
    "canonical_record_write",
    "kb_storage_write",
    "source_record_update",
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


class KnowledgeFreshnessSignal(BaseModel):
    """One advisory knowledge freshness tracking signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeFreshnessKind
    status: KnowledgeFreshnessStatus
    confidence: KnowledgeFreshnessConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    freshness_axis: KnowledgeFreshnessAxis
    knowledge_rollback_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_rollback_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    freshness_signal_summary: str = Field(min_length=1, max_length=360)
    freshness_signal_score: int = Field(ge=0, le=100)
    rollback_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    freshness_score: int = Field(ge=0, le=1_000)
    hitl_required_before_freshness_tracking: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_freshness_tracking_capability_implemented: Literal[True] = True
    knowledge_freshness_metadata_implemented: Literal[True] = True
    knowledge_rollback_metadata_used: Literal[True] = True
    knowledge_freshness_tracking_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    freshness_score_computation_implemented: Literal[False] = False
    freshness_record_write_implemented: Literal[False] = False
    source_timestamp_update_implemented: Literal[False] = False
    staleness_state_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    kb_state_restore_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    rollback_state_mutation_implemented: Literal[False] = False
    rollback_record_write_implemented: Literal[False] = False
    snapshot_restore_execution_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_record_write_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_freshness_entry.v1"] = (
        KNOWLEDGE_FRESHNESS_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_freshness_tracking::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_rollback_signal_count != len(
            self.knowledge_rollback_signal_ids
        ):
            raise ValueError("knowledge_rollback_signal_count must match signals")
        if self.freshness_score != _freshness_score(
            freshness_signal_score=self.freshness_signal_score,
            rollback_alignment_score=self.rollback_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("freshness_score must combine source scores")
        if self.status != _freshness_status(self.freshness_score):
            raise ValueError("status must match freshness_score")
        if self.confidence != _freshness_confidence(self.freshness_score):
            raise ValueError("confidence must match freshness_score")
        if not self.hitl_required_before_freshness_tracking:
            raise ValueError("knowledge freshness tracking requires HITL posture")
        return self


class KnowledgeFreshnessTrackingPlan(BaseModel):
    """Bounded V6.3 advisory knowledge freshness tracking plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_freshness_tracking"] = "knowledge_freshness_tracking"
    serialization_version: Literal["knowledge_freshness_plan.v1"] = (
        KNOWLEDGE_FRESHNESS_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_FRESHNESS_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_rollback_role: Literal["knowledge_rollback"] = "knowledge_rollback"
    knowledge_rollback_serialization_version: Literal["knowledge_rollback_plan.v1"]
    knowledge_rollback_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_rollback_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeFreshnessSignal, ...] = Field(
        min_length=5,
        max_length=5,
    )
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
    planned_freshness_tracking_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_freshness_scan_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_freshness_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_freshness_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    updated_source_timestamp_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_staleness_state_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_freshness_score: int = Field(ge=0, le=1_000)
    overall_freshness_score: int = Field(ge=0, le=1_000)
    overall_freshness_posture: KnowledgeFreshnessPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_freshness_tracking_capability_implemented: Literal[True] = True
    knowledge_freshness_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_rollback_metadata_used: Literal[True] = True
    knowledge_freshness_tracking_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    freshness_score_computation_implemented: Literal[False] = False
    freshness_record_write_implemented: Literal[False] = False
    source_timestamp_update_implemented: Literal[False] = False
    staleness_state_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    kb_state_restore_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    rollback_state_mutation_implemented: Literal[False] = False
    rollback_record_write_implemented: Literal[False] = False
    snapshot_restore_execution_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_record_write_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
            if signal.hitl_required_before_freshness_tracking
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_freshness_tracking_ids:
            raise ValueError("planned_freshness_tracking_ids must remain empty")
        if self.executed_freshness_scan_ids:
            raise ValueError("executed_freshness_scan_ids must remain empty")
        if self.computed_freshness_score_ids:
            raise ValueError("computed_freshness_score_ids must remain empty")
        if self.written_freshness_record_ids:
            raise ValueError("written_freshness_record_ids must remain empty")
        if self.updated_source_timestamp_ids:
            raise ValueError("updated_source_timestamp_ids must remain empty")
        if self.mutated_staleness_state_ids:
            raise ValueError("mutated_staleness_state_ids must remain empty")
        if self.fetched_source_ids:
            raise ValueError("fetched_source_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_rollback_signal_count != len(
            self.knowledge_rollback_signal_ids
        ):
            raise ValueError("knowledge_rollback_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 19 roadmap")
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
        if self.highest_freshness_score != max(
            signal.freshness_score for signal in self.signals
        ):
            raise ValueError("highest_freshness_score must match signals")
        if self.overall_freshness_score != _overall_freshness_score(self.signals):
            raise ValueError("overall_freshness_score must match signals")
        if self.overall_freshness_posture != _overall_freshness_posture(self.signals):
            raise ValueError("overall_freshness_posture must match signals")
        declared_rollback_signals = set(self.knowledge_rollback_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_rollback_signal_ids).issubset(
                declared_rollback_signals
            ):
                raise ValueError("signal knowledge_rollback_signal_ids must be known")
        return self


def build_knowledge_freshness_tracking(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_rollback: KnowledgeRollbackPlan | None = None,
) -> KnowledgeFreshnessTrackingPlan:
    """Build V6.3 Task 19 knowledge freshness metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    rollback_plan = knowledge_rollback or build_knowledge_rollback(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        rollback_plan=rollback_plan,
    )
    return KnowledgeFreshnessTrackingPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=rollback_plan.checked_at,
        knowledge_rollback_serialization_version=rollback_plan.serialization_version,
        knowledge_rollback_signal_ids=rollback_plan.signal_ids,
        knowledge_rollback_signal_count=len(rollback_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=rollback_plan.source_count,
        domain_count=rollback_plan.domain_count,
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
            if signal.hitl_required_before_freshness_tracking
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
            1 for signal in signals if signal.hitl_required_before_freshness_tracking
        ),
        highest_freshness_score=max(signal.freshness_score for signal in signals),
        overall_freshness_score=_overall_freshness_score(signals),
        overall_freshness_posture=_overall_freshness_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_freshness_signal_by_id(
    signal_id: str,
    plan: KnowledgeFreshnessTrackingPlan | None = None,
) -> KnowledgeFreshnessSignal | None:
    """Return one knowledge freshness signal without freshness mutation."""

    source_plan = plan or build_knowledge_freshness_tracking()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_freshness_signals_for_status(
    status: KnowledgeFreshnessStatus,
    plan: KnowledgeFreshnessTrackingPlan | None = None,
) -> tuple[KnowledgeFreshnessSignal, ...]:
    """Return knowledge freshness signals by advisory status."""

    source_plan = plan or build_knowledge_freshness_tracking()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_freshness_signals_for_confidence(
    confidence: KnowledgeFreshnessConfidence,
    plan: KnowledgeFreshnessTrackingPlan | None = None,
) -> tuple[KnowledgeFreshnessSignal, ...]:
    """Return knowledge freshness signals by confidence band."""

    source_plan = plan or build_knowledge_freshness_tracking()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    rollback_plan: KnowledgeRollbackPlan,
) -> tuple[KnowledgeFreshnessSignal, ...]:
    return (
        _signal(
            kind="knowledge_freshness_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            rollback_signal_ids=rollback_plan.signal_ids,
            rollback_plan=rollback_plan,
            freshness_signal_score=88,
            rollback_alignment_score=86,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_freshness_source_age_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_age",
            rollback_signal_ids=(
                "knowledge_rollback::knowledge_rollback_inventory_review",
                "knowledge_rollback::knowledge_rollback_snapshot_alignment_review",
            ),
            rollback_plan=rollback_plan,
            freshness_signal_score=80,
            rollback_alignment_score=78,
            governance_alignment_score=76,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_freshness_staleness_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="staleness_readiness",
            rollback_signal_ids=(
                "knowledge_rollback::knowledge_rollback_plan_readiness",
                "knowledge_rollback::knowledge_rollback_governance_gate",
            ),
            rollback_plan=rollback_plan,
            freshness_signal_score=76,
            rollback_alignment_score=74,
            governance_alignment_score=78,
            mutation_risk_score=42,
            governance_weight=105,
        ),
        _signal(
            kind="knowledge_freshness_rollback_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="rollback_alignment",
            rollback_signal_ids=(
                "knowledge_rollback::knowledge_rollback_safety_readiness",
                "knowledge_rollback::knowledge_rollback_governance_gate",
            ),
            rollback_plan=rollback_plan,
            freshness_signal_score=66,
            rollback_alignment_score=68,
            governance_alignment_score=72,
            mutation_risk_score=36,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_freshness_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            rollback_signal_ids=rollback_plan.signal_ids,
            rollback_plan=rollback_plan,
            freshness_signal_score=44,
            rollback_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeFreshnessKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeFreshnessAxis,
    rollback_signal_ids: tuple[str, ...],
    rollback_plan: KnowledgeRollbackPlan,
    freshness_signal_score: int,
    rollback_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeFreshnessSignal:
    score = _freshness_score(
        freshness_signal_score=freshness_signal_score,
        rollback_alignment_score=rollback_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeFreshnessSignal(
        signal_id=f"knowledge_freshness_tracking::{kind}",
        signal_kind=kind,
        status=_freshness_status(score),
        confidence=_freshness_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        freshness_axis=axis,
        knowledge_rollback_signal_ids=rollback_signal_ids,
        knowledge_rollback_signal_count=len(rollback_signal_ids),
        source_count=rollback_plan.source_count,
        domain_count=rollback_plan.domain_count,
        freshness_signal_summary=_signal_summary(kind),
        freshness_signal_score=freshness_signal_score,
        rollback_alignment_score=rollback_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        freshness_score=score,
        hitl_required_before_freshness_tracking=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, rollback_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_rollback_signal_count:{len(rollback_signal_ids)}",
            f"source_count:{rollback_plan.source_count}",
            f"domain_count:{rollback_plan.domain_count}",
            f"freshness_axis:{axis}",
            f"freshness_signal_score:{freshness_signal_score}",
            f"rollback_alignment_score:{rollback_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_freshness_tracking:true",
        ),
    )


def _freshness_score(
    *,
    freshness_signal_score: int,
    rollback_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            freshness_signal_score * 3
            + rollback_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _freshness_status(score: int) -> KnowledgeFreshnessStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _freshness_confidence(score: int) -> KnowledgeFreshnessConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_freshness_score(signals: tuple[KnowledgeFreshnessSignal, ...]) -> int:
    base = sum(signal.freshness_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_freshness_posture(
    signals: tuple[KnowledgeFreshnessSignal, ...],
) -> KnowledgeFreshnessPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeFreshnessSignal, ...],
    status: KnowledgeFreshnessStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeFreshnessSignal, ...],
    *confidences: KnowledgeFreshnessConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeFreshnessSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_freshness_metadata",
        "verify_knowledge_freshness_roadmap_traceability",
        "review_rollback_signals_before_any_freshness_action",
        "require_hitl_before_freshness_scan_score_timestamp_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeFreshnessKind,
    axis: KnowledgeFreshnessAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_freshness_tracking",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeFreshnessKind,
    rollback_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_rollback_signal_count:{len(rollback_signal_ids)}",
        "composes_knowledge_rollback_metadata",
        "keeps_knowledge_freshness_tracking_execution_disabled",
        "requires_human_review_before_freshness_scan_score_timestamp_or_write",
    )


def _signal_actions(kind: KnowledgeFreshnessKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_freshness_signal_metadata",
        "verify_knowledge_rollback_traceability",
        "keep_knowledge_freshness_tracking_disabled",
        "require_hitl_before_knowledge_freshness_action",
    )
    if kind == "knowledge_freshness_source_age_review":
        return base_actions + ("review_source_age_metadata",)
    if kind == "knowledge_freshness_staleness_readiness":
        return base_actions + ("review_staleness_readiness_metadata",)
    if kind == "knowledge_freshness_rollback_alignment_review":
        return base_actions + ("review_freshness_rollback_alignment_metadata",)
    if kind == "knowledge_freshness_governance_gate":
        return base_actions + ("confirm_manual_freshness_governance_gate",)
    return base_actions + ("review_knowledge_freshness_inventory_metadata",)


def _signal_summary(kind: KnowledgeFreshnessKind) -> str:
    summaries: dict[KnowledgeFreshnessKind, str] = {
        "knowledge_freshness_inventory_review": (
            "Advisory knowledge freshness inventory posture over rollback "
            "metadata without executing freshness scans."
        ),
        "knowledge_freshness_source_age_review": (
            "Advisory posture for reviewing source age metadata before "
            "timestamp updates, source fetches, or freshness record writes."
        ),
        "knowledge_freshness_staleness_readiness": (
            "Advisory posture for staleness readiness while freshness score "
            "computation and staleness state mutation remain disabled."
        ),
        "knowledge_freshness_rollback_alignment_review": (
            "Advisory posture for aligning freshness metadata with rollback "
            "posture while restore and KB mutation remain disabled."
        ),
        "knowledge_freshness_governance_gate": (
            "Governed manual gate that keeps freshness tracking disabled until "
            "HITL approval."
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
