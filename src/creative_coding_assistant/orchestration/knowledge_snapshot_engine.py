"""V6.3 advisory knowledge snapshot engine metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_versioning import (
    KnowledgeVersioningPlan,
    build_knowledge_versioning,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeSnapshotKind = Literal[
    "knowledge_snapshot_inventory_review",
    "knowledge_snapshot_version_alignment_review",
    "knowledge_snapshot_capture_readiness",
    "knowledge_snapshot_storage_readiness",
    "knowledge_snapshot_governance_gate",
]
KnowledgeSnapshotStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeSnapshotConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeSnapshotPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeSnapshotAxis = Literal[
    "inventory_review",
    "version_alignment",
    "capture_readiness",
    "storage_readiness",
    "governance_gate",
]

KNOWLEDGE_SNAPSHOT_ENTRY_SERIALIZATION_VERSION = "knowledge_snapshot_entry.v1"
KNOWLEDGE_SNAPSHOT_PLAN_SERIALIZATION_VERSION = "knowledge_snapshot_plan.v1"
KNOWLEDGE_SNAPSHOT_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Snapshot Engine exposes knowledge versioning, snapshot "
    "inventory, version alignment, capture readiness, storage readiness, and "
    "governance posture as inspectable advisory metadata only; it does not "
    "execute snapshot operations, create snapshots, capture snapshots, write "
    "snapshot records, write snapshot storage, write snapshot indexes, write "
    "snapshot manifests, mutate snapshot retention, execute knowledge "
    "versioning, mutate version graphs, write version records, assign version "
    "ids, reconstruct version lineage, write version history, execute "
    "rollback, apply rollback plans, mutate provenance graphs, write "
    "provenance records, reconstruct lineage, relink sources, execute "
    "lifecycle management, mutate lifecycle or retention policy, write "
    "lifecycle records, execute knowledge consolidation, merge or "
    "deduplicate knowledge, write canonical records, write KB storage, update "
    "source records, execute retrieval queries, mutate retrieval "
    "configuration, mutate ranking, request embeddings, refresh embeddings, "
    "index vectors, upsert vectors, fetch documentation, provision providers, "
    "infer API keys, route providers or models, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, execute workflows, "
    "modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Snapshot Engine",)

_BLOCKED_RUNTIME_BEHAVIORS = (
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
    "version_lineage_reconstruction",
    "version_history_write",
    "rollback_execution",
    "rollback_plan_application",
    "provenance_graph_mutation",
    "provenance_record_write",
    "lineage_reconstruction_execution",
    "source_relinking_execution",
    "knowledge_lifecycle_management_execution",
    "lifecycle_stage_transition",
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


class KnowledgeSnapshotSignal(BaseModel):
    """One advisory knowledge snapshot signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeSnapshotKind
    status: KnowledgeSnapshotStatus
    confidence: KnowledgeSnapshotConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    snapshot_axis: KnowledgeSnapshotAxis
    knowledge_versioning_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_versioning_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    snapshot_signal_summary: str = Field(min_length=1, max_length=360)
    snapshot_signal_score: int = Field(ge=0, le=100)
    version_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    snapshot_score: int = Field(ge=0, le=1_000)
    hitl_required_before_snapshot: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_snapshot_engine_capability_implemented: Literal[True] = True
    knowledge_snapshot_metadata_implemented: Literal[True] = True
    knowledge_versioning_metadata_used: Literal[True] = True
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
    version_lineage_reconstruction_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    lineage_reconstruction_execution_implemented: Literal[False] = False
    source_relinking_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_stage_transition_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_snapshot_entry.v1"] = (
        KNOWLEDGE_SNAPSHOT_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_snapshot_engine::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_versioning_signal_count != len(
            self.knowledge_versioning_signal_ids
        ):
            raise ValueError("knowledge_versioning_signal_count must match signals")
        if self.snapshot_score != _snapshot_score(
            snapshot_signal_score=self.snapshot_signal_score,
            version_alignment_score=self.version_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("snapshot_score must combine source scores")
        if self.status != _snapshot_status(self.snapshot_score):
            raise ValueError("status must match snapshot_score")
        if self.confidence != _snapshot_confidence(self.snapshot_score):
            raise ValueError("confidence must match snapshot_score")
        if not self.hitl_required_before_snapshot:
            raise ValueError("knowledge snapshot requires HITL posture")
        return self


class KnowledgeSnapshotEnginePlan(BaseModel):
    """Bounded V6.3 advisory knowledge snapshot engine plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_snapshot_engine"] = "knowledge_snapshot_engine"
    serialization_version: Literal["knowledge_snapshot_plan.v1"] = (
        KNOWLEDGE_SNAPSHOT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_SNAPSHOT_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_versioning_role: Literal["knowledge_versioning"] = "knowledge_versioning"
    knowledge_versioning_serialization_version: Literal["knowledge_versioning_plan.v1"]
    knowledge_versioning_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_versioning_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeSnapshotSignal, ...] = Field(
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
    planned_snapshot_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    created_snapshot_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    captured_snapshot_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_snapshot_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_snapshot_storage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_snapshot_index_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_snapshot_manifest_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_snapshot_retention_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_snapshot_score: int = Field(ge=0, le=1_000)
    overall_snapshot_score: int = Field(ge=0, le=1_000)
    overall_snapshot_posture: KnowledgeSnapshotPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_snapshot_engine_capability_implemented: Literal[True] = True
    knowledge_snapshot_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_versioning_metadata_used: Literal[True] = True
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
    version_lineage_reconstruction_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    rollback_execution_implemented: Literal[False] = False
    rollback_plan_application_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    provenance_record_write_implemented: Literal[False] = False
    lineage_reconstruction_execution_implemented: Literal[False] = False
    source_relinking_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_stage_transition_implemented: Literal[False] = False
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
            if signal.hitl_required_before_snapshot
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_snapshot_ids:
            raise ValueError("planned_snapshot_ids must remain empty")
        if self.created_snapshot_ids:
            raise ValueError("created_snapshot_ids must remain empty")
        if self.captured_snapshot_ids:
            raise ValueError("captured_snapshot_ids must remain empty")
        if self.written_snapshot_record_ids:
            raise ValueError("written_snapshot_record_ids must remain empty")
        if self.written_snapshot_storage_ids:
            raise ValueError("written_snapshot_storage_ids must remain empty")
        if self.written_snapshot_index_ids:
            raise ValueError("written_snapshot_index_ids must remain empty")
        if self.written_snapshot_manifest_ids:
            raise ValueError("written_snapshot_manifest_ids must remain empty")
        if self.mutated_snapshot_retention_ids:
            raise ValueError("mutated_snapshot_retention_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_versioning_signal_count != len(
            self.knowledge_versioning_signal_ids
        ):
            raise ValueError("knowledge_versioning_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 17 roadmap")
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
        if self.highest_snapshot_score != max(
            signal.snapshot_score for signal in self.signals
        ):
            raise ValueError("highest_snapshot_score must match signals")
        if self.overall_snapshot_score != _overall_snapshot_score(self.signals):
            raise ValueError("overall_snapshot_score must match signals")
        if self.overall_snapshot_posture != _overall_snapshot_posture(self.signals):
            raise ValueError("overall_snapshot_posture must match signals")
        declared_versioning_signals = set(self.knowledge_versioning_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_versioning_signal_ids).issubset(
                declared_versioning_signals
            ):
                raise ValueError("signal knowledge_versioning_signal_ids must be known")
        return self


def build_knowledge_snapshot_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_versioning: KnowledgeVersioningPlan | None = None,
) -> KnowledgeSnapshotEnginePlan:
    """Build V6.3 Task 17 knowledge snapshot engine metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    versioning_plan = knowledge_versioning or build_knowledge_versioning(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        versioning_plan=versioning_plan,
    )
    return KnowledgeSnapshotEnginePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=versioning_plan.checked_at,
        knowledge_versioning_serialization_version=(
            versioning_plan.serialization_version
        ),
        knowledge_versioning_signal_ids=versioning_plan.signal_ids,
        knowledge_versioning_signal_count=len(versioning_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=versioning_plan.source_count,
        domain_count=versioning_plan.domain_count,
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
            if signal.hitl_required_before_snapshot
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
            1 for signal in signals if signal.hitl_required_before_snapshot
        ),
        highest_snapshot_score=max(signal.snapshot_score for signal in signals),
        overall_snapshot_score=_overall_snapshot_score(signals),
        overall_snapshot_posture=_overall_snapshot_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_snapshot_signal_by_id(
    signal_id: str,
    plan: KnowledgeSnapshotEnginePlan | None = None,
) -> KnowledgeSnapshotSignal | None:
    """Return one knowledge snapshot signal without snapshot mutation."""

    source_plan = plan or build_knowledge_snapshot_engine()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_snapshot_signals_for_status(
    status: KnowledgeSnapshotStatus,
    plan: KnowledgeSnapshotEnginePlan | None = None,
) -> tuple[KnowledgeSnapshotSignal, ...]:
    """Return knowledge snapshot signals by advisory status."""

    source_plan = plan or build_knowledge_snapshot_engine()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_snapshot_signals_for_confidence(
    confidence: KnowledgeSnapshotConfidence,
    plan: KnowledgeSnapshotEnginePlan | None = None,
) -> tuple[KnowledgeSnapshotSignal, ...]:
    """Return knowledge snapshot signals by confidence band."""

    source_plan = plan or build_knowledge_snapshot_engine()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    versioning_plan: KnowledgeVersioningPlan,
) -> tuple[KnowledgeSnapshotSignal, ...]:
    return (
        _signal(
            kind="knowledge_snapshot_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            versioning_signal_ids=versioning_plan.signal_ids,
            versioning_plan=versioning_plan,
            snapshot_signal_score=88,
            version_alignment_score=86,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_snapshot_version_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="version_alignment",
            versioning_signal_ids=(
                "knowledge_versioning::knowledge_version_inventory_review",
                "knowledge_versioning::knowledge_version_lineage_alignment_review",
            ),
            versioning_plan=versioning_plan,
            snapshot_signal_score=80,
            version_alignment_score=78,
            governance_alignment_score=76,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_snapshot_capture_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="capture_readiness",
            versioning_signal_ids=(
                "knowledge_versioning::knowledge_version_snapshot_readiness",
                "knowledge_versioning::knowledge_version_governance_gate",
            ),
            versioning_plan=versioning_plan,
            snapshot_signal_score=76,
            version_alignment_score=74,
            governance_alignment_score=78,
            mutation_risk_score=42,
            governance_weight=105,
        ),
        _signal(
            kind="knowledge_snapshot_storage_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="storage_readiness",
            versioning_signal_ids=(
                "knowledge_versioning::knowledge_version_snapshot_readiness",
                "knowledge_versioning::knowledge_version_rollback_readiness",
            ),
            versioning_plan=versioning_plan,
            snapshot_signal_score=66,
            version_alignment_score=68,
            governance_alignment_score=72,
            mutation_risk_score=36,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_snapshot_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            versioning_signal_ids=versioning_plan.signal_ids,
            versioning_plan=versioning_plan,
            snapshot_signal_score=44,
            version_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeSnapshotKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeSnapshotAxis,
    versioning_signal_ids: tuple[str, ...],
    versioning_plan: KnowledgeVersioningPlan,
    snapshot_signal_score: int,
    version_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeSnapshotSignal:
    score = _snapshot_score(
        snapshot_signal_score=snapshot_signal_score,
        version_alignment_score=version_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeSnapshotSignal(
        signal_id=f"knowledge_snapshot_engine::{kind}",
        signal_kind=kind,
        status=_snapshot_status(score),
        confidence=_snapshot_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        snapshot_axis=axis,
        knowledge_versioning_signal_ids=versioning_signal_ids,
        knowledge_versioning_signal_count=len(versioning_signal_ids),
        source_count=versioning_plan.source_count,
        domain_count=versioning_plan.domain_count,
        snapshot_signal_summary=_signal_summary(kind),
        snapshot_signal_score=snapshot_signal_score,
        version_alignment_score=version_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        snapshot_score=score,
        hitl_required_before_snapshot=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, versioning_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_versioning_signal_count:{len(versioning_signal_ids)}",
            f"source_count:{versioning_plan.source_count}",
            f"domain_count:{versioning_plan.domain_count}",
            f"snapshot_axis:{axis}",
            f"snapshot_signal_score:{snapshot_signal_score}",
            f"version_alignment_score:{version_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_snapshot:true",
        ),
    )


def _snapshot_score(
    *,
    snapshot_signal_score: int,
    version_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            snapshot_signal_score * 3
            + version_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _snapshot_status(score: int) -> KnowledgeSnapshotStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _snapshot_confidence(score: int) -> KnowledgeSnapshotConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_snapshot_score(signals: tuple[KnowledgeSnapshotSignal, ...]) -> int:
    base = sum(signal.snapshot_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_snapshot_posture(
    signals: tuple[KnowledgeSnapshotSignal, ...],
) -> KnowledgeSnapshotPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeSnapshotSignal, ...],
    status: KnowledgeSnapshotStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeSnapshotSignal, ...],
    *confidences: KnowledgeSnapshotConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeSnapshotSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_snapshot_metadata",
        "verify_knowledge_snapshot_roadmap_traceability",
        "review_versioning_signals_before_any_snapshot_action",
        "require_hitl_before_snapshot_capture_storage_index_or_manifest_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeSnapshotKind,
    axis: KnowledgeSnapshotAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_snapshot_engine",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeSnapshotKind,
    versioning_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_versioning_signal_count:{len(versioning_signal_ids)}",
        "composes_knowledge_versioning_metadata",
        "keeps_knowledge_snapshot_engine_execution_disabled",
        "requires_human_review_before_snapshot_capture_storage_or_write",
    )


def _signal_actions(kind: KnowledgeSnapshotKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_snapshot_signal_metadata",
        "verify_knowledge_versioning_traceability",
        "keep_knowledge_snapshot_engine_disabled",
        "require_hitl_before_knowledge_snapshot_action",
    )
    if kind == "knowledge_snapshot_version_alignment_review":
        return base_actions + ("review_snapshot_version_alignment_metadata",)
    if kind == "knowledge_snapshot_capture_readiness":
        return base_actions + ("review_snapshot_capture_readiness_metadata",)
    if kind == "knowledge_snapshot_storage_readiness":
        return base_actions + ("review_snapshot_storage_readiness_metadata",)
    if kind == "knowledge_snapshot_governance_gate":
        return base_actions + ("confirm_manual_snapshot_governance_gate",)
    return base_actions + ("review_knowledge_snapshot_inventory_metadata",)


def _signal_summary(kind: KnowledgeSnapshotKind) -> str:
    summaries: dict[KnowledgeSnapshotKind, str] = {
        "knowledge_snapshot_inventory_review": (
            "Advisory knowledge snapshot inventory posture over versioning "
            "metadata without executing snapshot operations."
        ),
        "knowledge_snapshot_version_alignment_review": (
            "Advisory posture for reviewing snapshot alignment with version "
            "metadata before capture, record, storage, index, or manifest writes."
        ),
        "knowledge_snapshot_capture_readiness": (
            "Advisory posture for snapshot capture readiness while snapshot "
            "creation and capture execution remain disabled."
        ),
        "knowledge_snapshot_storage_readiness": (
            "Advisory posture for snapshot storage readiness while snapshot "
            "storage, index, manifest, and KB writes remain disabled."
        ),
        "knowledge_snapshot_governance_gate": (
            "Governed manual gate that keeps knowledge snapshot operations "
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
