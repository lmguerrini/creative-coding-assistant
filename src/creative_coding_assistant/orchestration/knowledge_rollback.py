"""V6.3 advisory knowledge rollback metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_snapshot_engine import (
    KnowledgeSnapshotEnginePlan,
    build_knowledge_snapshot_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeRollbackKind = Literal[
    "knowledge_rollback_inventory_review",
    "knowledge_rollback_snapshot_alignment_review",
    "knowledge_rollback_plan_readiness",
    "knowledge_rollback_safety_readiness",
    "knowledge_rollback_governance_gate",
]
KnowledgeRollbackStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeRollbackConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeRollbackPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeRollbackAxis = Literal[
    "inventory_review",
    "snapshot_alignment",
    "plan_readiness",
    "safety_readiness",
    "governance_gate",
]

KNOWLEDGE_ROLLBACK_ENTRY_SERIALIZATION_VERSION = "knowledge_rollback_entry.v1"
KNOWLEDGE_ROLLBACK_PLAN_SERIALIZATION_VERSION = "knowledge_rollback_plan.v1"
KNOWLEDGE_ROLLBACK_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Rollback exposes knowledge snapshot, rollback inventory, "
    "snapshot alignment, rollback plan readiness, safety readiness, and "
    "governance posture as inspectable advisory metadata only; it does not "
    "execute rollback, apply rollback plans, mutate rollback state, write "
    "rollback records, restore snapshots, mutate snapshot selection, restore "
    "KB state, execute snapshot operations, create or capture snapshots, write "
    "snapshot records, storage, indexes, or manifests, mutate snapshot "
    "retention, execute knowledge versioning, mutate version graphs, write "
    "version records, assign version ids, write version history, mutate "
    "provenance graphs, write provenance records, execute lifecycle "
    "management, mutate lifecycle or retention policy, write lifecycle "
    "records, execute knowledge consolidation, merge or deduplicate "
    "knowledge, write canonical records, write KB storage, update source "
    "records, execute retrieval queries, mutate retrieval configuration, "
    "mutate ranking, request embeddings, refresh embeddings, index vectors, "
    "upsert vectors, fetch documentation, provision providers, infer API "
    "keys, route providers or models, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, execute workflows, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Rollback",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_rollback_execution",
    "rollback_plan_application",
    "rollback_state_mutation",
    "rollback_record_write",
    "snapshot_restore_execution",
    "snapshot_selection_mutation",
    "kb_state_restore",
    "knowledge_snapshot_engine_execution",
    "snapshot_creation",
    "snapshot_capture_execution",
    "snapshot_record_write",
    "snapshot_storage_write",
    "snapshot_index_write",
    "snapshot_manifest_write",
    "snapshot_retention_mutation",
    "knowledge_versioning_execution",
    "version_graph_mutation",
    "version_record_write",
    "version_id_assignment",
    "version_history_write",
    "provenance_graph_mutation",
    "provenance_record_write",
    "knowledge_lifecycle_management_execution",
    "lifecycle_policy_mutation",
    "retention_policy_mutation",
    "lifecycle_record_write",
    "knowledge_consolidation_execution",
    "knowledge_merge_execution",
    "knowledge_deduplication_execution",
    "canonical_record_write",
    "consolidation_record_write",
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


class KnowledgeRollbackSignal(BaseModel):
    """One advisory knowledge rollback signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeRollbackKind
    status: KnowledgeRollbackStatus
    confidence: KnowledgeRollbackConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    rollback_axis: KnowledgeRollbackAxis
    knowledge_snapshot_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_snapshot_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    rollback_signal_summary: str = Field(min_length=1, max_length=360)
    rollback_signal_score: int = Field(ge=0, le=100)
    snapshot_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    rollback_score: int = Field(ge=0, le=1_000)
    hitl_required_before_rollback: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_rollback_capability_implemented: Literal[True] = True
    knowledge_rollback_metadata_implemented: Literal[True] = True
    knowledge_snapshot_metadata_used: Literal[True] = True
    knowledge_rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    rollback_state_mutation_implemented: Literal[False] = False
    rollback_record_write_implemented: Literal[False] = False
    snapshot_restore_execution_implemented: Literal[False] = False
    snapshot_selection_mutation_implemented: Literal[False] = False
    kb_state_restore_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_capture_execution_implemented: Literal[False] = False
    snapshot_record_write_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
    snapshot_index_write_implemented: Literal[False] = False
    snapshot_manifest_write_implemented: Literal[False] = False
    snapshot_retention_mutation_implemented: Literal[False] = False
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    version_id_assignment_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    lifecycle_record_write_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_rollback_entry.v1"] = (
        KNOWLEDGE_ROLLBACK_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_rollback::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_snapshot_signal_count != len(
            self.knowledge_snapshot_signal_ids
        ):
            raise ValueError("knowledge_snapshot_signal_count must match signals")
        if self.rollback_score != _rollback_score(
            rollback_signal_score=self.rollback_signal_score,
            snapshot_alignment_score=self.snapshot_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("rollback_score must combine source scores")
        if self.status != _rollback_status(self.rollback_score):
            raise ValueError("status must match rollback_score")
        if self.confidence != _rollback_confidence(self.rollback_score):
            raise ValueError("confidence must match rollback_score")
        if not self.hitl_required_before_rollback:
            raise ValueError("knowledge rollback requires HITL posture")
        return self


class KnowledgeRollbackPlan(BaseModel):
    """Bounded V6.3 advisory knowledge rollback plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_rollback"] = "knowledge_rollback"
    serialization_version: Literal["knowledge_rollback_plan.v1"] = (
        KNOWLEDGE_ROLLBACK_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_ROLLBACK_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_snapshot_role: Literal["knowledge_snapshot_engine"] = (
        "knowledge_snapshot_engine"
    )
    knowledge_snapshot_serialization_version: Literal["knowledge_snapshot_plan.v1"]
    knowledge_snapshot_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_snapshot_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeRollbackSignal, ...] = Field(
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
    planned_rollback_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_rollback_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_rollback_state_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_rollback_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    restored_snapshot_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    selected_snapshot_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    restored_kb_state_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_rollback_score: int = Field(ge=0, le=1_000)
    overall_rollback_score: int = Field(ge=0, le=1_000)
    overall_rollback_posture: KnowledgeRollbackPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_rollback_capability_implemented: Literal[True] = True
    knowledge_rollback_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_snapshot_metadata_used: Literal[True] = True
    knowledge_rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    rollback_state_mutation_implemented: Literal[False] = False
    rollback_record_write_implemented: Literal[False] = False
    snapshot_restore_execution_implemented: Literal[False] = False
    snapshot_selection_mutation_implemented: Literal[False] = False
    kb_state_restore_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_capture_execution_implemented: Literal[False] = False
    snapshot_record_write_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
    snapshot_index_write_implemented: Literal[False] = False
    snapshot_manifest_write_implemented: Literal[False] = False
    snapshot_retention_mutation_implemented: Literal[False] = False
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    version_id_assignment_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    lifecycle_record_write_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
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
            if signal.hitl_required_before_rollback
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_rollback_ids:
            raise ValueError("planned_rollback_ids must remain empty")
        if self.applied_rollback_ids:
            raise ValueError("applied_rollback_ids must remain empty")
        if self.mutated_rollback_state_ids:
            raise ValueError("mutated_rollback_state_ids must remain empty")
        if self.written_rollback_record_ids:
            raise ValueError("written_rollback_record_ids must remain empty")
        if self.restored_snapshot_ids:
            raise ValueError("restored_snapshot_ids must remain empty")
        if self.selected_snapshot_ids:
            raise ValueError("selected_snapshot_ids must remain empty")
        if self.restored_kb_state_ids:
            raise ValueError("restored_kb_state_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_snapshot_signal_count != len(
            self.knowledge_snapshot_signal_ids
        ):
            raise ValueError("knowledge_snapshot_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 18 roadmap")
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
        if self.highest_rollback_score != max(
            signal.rollback_score for signal in self.signals
        ):
            raise ValueError("highest_rollback_score must match signals")
        if self.overall_rollback_score != _overall_rollback_score(self.signals):
            raise ValueError("overall_rollback_score must match signals")
        if self.overall_rollback_posture != _overall_rollback_posture(self.signals):
            raise ValueError("overall_rollback_posture must match signals")
        declared_snapshot_signals = set(self.knowledge_snapshot_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_snapshot_signal_ids).issubset(
                declared_snapshot_signals
            ):
                raise ValueError("signal knowledge_snapshot_signal_ids must be known")
        return self


def build_knowledge_rollback(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_snapshot: KnowledgeSnapshotEnginePlan | None = None,
) -> KnowledgeRollbackPlan:
    """Build V6.3 Task 18 knowledge rollback metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    snapshot_plan = knowledge_snapshot or build_knowledge_snapshot_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        snapshot_plan=snapshot_plan,
    )
    return KnowledgeRollbackPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=snapshot_plan.checked_at,
        knowledge_snapshot_serialization_version=snapshot_plan.serialization_version,
        knowledge_snapshot_signal_ids=snapshot_plan.signal_ids,
        knowledge_snapshot_signal_count=len(snapshot_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=snapshot_plan.source_count,
        domain_count=snapshot_plan.domain_count,
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
            if signal.hitl_required_before_rollback
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
            1 for signal in signals if signal.hitl_required_before_rollback
        ),
        highest_rollback_score=max(signal.rollback_score for signal in signals),
        overall_rollback_score=_overall_rollback_score(signals),
        overall_rollback_posture=_overall_rollback_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_rollback_signal_by_id(
    signal_id: str,
    plan: KnowledgeRollbackPlan | None = None,
) -> KnowledgeRollbackSignal | None:
    """Return one knowledge rollback signal without rollback mutation."""

    source_plan = plan or build_knowledge_rollback()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_rollback_signals_for_status(
    status: KnowledgeRollbackStatus,
    plan: KnowledgeRollbackPlan | None = None,
) -> tuple[KnowledgeRollbackSignal, ...]:
    """Return knowledge rollback signals by advisory status."""

    source_plan = plan or build_knowledge_rollback()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_rollback_signals_for_confidence(
    confidence: KnowledgeRollbackConfidence,
    plan: KnowledgeRollbackPlan | None = None,
) -> tuple[KnowledgeRollbackSignal, ...]:
    """Return knowledge rollback signals by confidence band."""

    source_plan = plan or build_knowledge_rollback()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    snapshot_plan: KnowledgeSnapshotEnginePlan,
) -> tuple[KnowledgeRollbackSignal, ...]:
    return (
        _signal(
            kind="knowledge_rollback_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            snapshot_signal_ids=snapshot_plan.signal_ids,
            snapshot_plan=snapshot_plan,
            rollback_signal_score=88,
            snapshot_alignment_score=86,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_rollback_snapshot_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="snapshot_alignment",
            snapshot_signal_ids=(
                "knowledge_snapshot_engine::knowledge_snapshot_inventory_review",
                "knowledge_snapshot_engine::"
                "knowledge_snapshot_version_alignment_review",
            ),
            snapshot_plan=snapshot_plan,
            rollback_signal_score=80,
            snapshot_alignment_score=78,
            governance_alignment_score=76,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_rollback_plan_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="plan_readiness",
            snapshot_signal_ids=(
                "knowledge_snapshot_engine::knowledge_snapshot_capture_readiness",
                "knowledge_snapshot_engine::knowledge_snapshot_governance_gate",
            ),
            snapshot_plan=snapshot_plan,
            rollback_signal_score=76,
            snapshot_alignment_score=74,
            governance_alignment_score=78,
            mutation_risk_score=42,
            governance_weight=105,
        ),
        _signal(
            kind="knowledge_rollback_safety_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="safety_readiness",
            snapshot_signal_ids=(
                "knowledge_snapshot_engine::knowledge_snapshot_storage_readiness",
                "knowledge_snapshot_engine::knowledge_snapshot_governance_gate",
            ),
            snapshot_plan=snapshot_plan,
            rollback_signal_score=66,
            snapshot_alignment_score=68,
            governance_alignment_score=72,
            mutation_risk_score=36,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_rollback_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            snapshot_signal_ids=snapshot_plan.signal_ids,
            snapshot_plan=snapshot_plan,
            rollback_signal_score=44,
            snapshot_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeRollbackKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeRollbackAxis,
    snapshot_signal_ids: tuple[str, ...],
    snapshot_plan: KnowledgeSnapshotEnginePlan,
    rollback_signal_score: int,
    snapshot_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeRollbackSignal:
    score = _rollback_score(
        rollback_signal_score=rollback_signal_score,
        snapshot_alignment_score=snapshot_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeRollbackSignal(
        signal_id=f"knowledge_rollback::{kind}",
        signal_kind=kind,
        status=_rollback_status(score),
        confidence=_rollback_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        rollback_axis=axis,
        knowledge_snapshot_signal_ids=snapshot_signal_ids,
        knowledge_snapshot_signal_count=len(snapshot_signal_ids),
        source_count=snapshot_plan.source_count,
        domain_count=snapshot_plan.domain_count,
        rollback_signal_summary=_signal_summary(kind),
        rollback_signal_score=rollback_signal_score,
        snapshot_alignment_score=snapshot_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        rollback_score=score,
        hitl_required_before_rollback=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, snapshot_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_snapshot_signal_count:{len(snapshot_signal_ids)}",
            f"source_count:{snapshot_plan.source_count}",
            f"domain_count:{snapshot_plan.domain_count}",
            f"rollback_axis:{axis}",
            f"rollback_signal_score:{rollback_signal_score}",
            f"snapshot_alignment_score:{snapshot_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_rollback:true",
        ),
    )


def _rollback_score(
    *,
    rollback_signal_score: int,
    snapshot_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            rollback_signal_score * 3
            + snapshot_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _rollback_status(score: int) -> KnowledgeRollbackStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _rollback_confidence(score: int) -> KnowledgeRollbackConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_rollback_score(signals: tuple[KnowledgeRollbackSignal, ...]) -> int:
    base = sum(signal.rollback_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_rollback_posture(
    signals: tuple[KnowledgeRollbackSignal, ...],
) -> KnowledgeRollbackPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeRollbackSignal, ...],
    status: KnowledgeRollbackStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeRollbackSignal, ...],
    *confidences: KnowledgeRollbackConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeRollbackSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_rollback_metadata",
        "verify_knowledge_rollback_roadmap_traceability",
        "review_snapshot_signals_before_any_rollback_action",
        "require_hitl_before_rollback_restore_state_or_kb_mutation",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeRollbackKind,
    axis: KnowledgeRollbackAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_rollback",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeRollbackKind,
    snapshot_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_snapshot_signal_count:{len(snapshot_signal_ids)}",
        "composes_knowledge_snapshot_metadata",
        "keeps_knowledge_rollback_execution_disabled",
        "requires_human_review_before_rollback_restore_state_or_kb_mutation",
    )


def _signal_actions(kind: KnowledgeRollbackKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_rollback_signal_metadata",
        "verify_knowledge_snapshot_traceability",
        "keep_knowledge_rollback_disabled",
        "require_hitl_before_knowledge_rollback_action",
    )
    if kind == "knowledge_rollback_snapshot_alignment_review":
        return base_actions + ("review_rollback_snapshot_alignment_metadata",)
    if kind == "knowledge_rollback_plan_readiness":
        return base_actions + ("review_rollback_plan_readiness_metadata",)
    if kind == "knowledge_rollback_safety_readiness":
        return base_actions + ("review_rollback_safety_readiness_metadata",)
    if kind == "knowledge_rollback_governance_gate":
        return base_actions + ("confirm_manual_rollback_governance_gate",)
    return base_actions + ("review_knowledge_rollback_inventory_metadata",)


def _signal_summary(kind: KnowledgeRollbackKind) -> str:
    summaries: dict[KnowledgeRollbackKind, str] = {
        "knowledge_rollback_inventory_review": (
            "Advisory knowledge rollback inventory posture over snapshot "
            "metadata without executing rollback."
        ),
        "knowledge_rollback_snapshot_alignment_review": (
            "Advisory posture for reviewing rollback alignment with snapshot "
            "metadata before restore, state mutation, KB restore, or writes."
        ),
        "knowledge_rollback_plan_readiness": (
            "Advisory posture for rollback plan readiness while rollback plan "
            "application and state mutation remain disabled."
        ),
        "knowledge_rollback_safety_readiness": (
            "Advisory posture for rollback safety readiness while snapshot "
            "restore and KB state restore remain disabled."
        ),
        "knowledge_rollback_governance_gate": (
            "Governed manual gate that keeps knowledge rollback disabled "
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
