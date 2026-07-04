"""V6.3 advisory knowledge versioning metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_provenance_evolution import (
    KnowledgeProvenanceEvolutionPlan,
    build_knowledge_provenance_evolution,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeVersioningKind = Literal[
    "knowledge_version_inventory_review",
    "knowledge_version_lineage_alignment_review",
    "knowledge_version_snapshot_readiness",
    "knowledge_version_rollback_readiness",
    "knowledge_version_governance_gate",
]
KnowledgeVersioningStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeVersioningConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeVersioningPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeVersioningAxis = Literal[
    "version_inventory",
    "lineage_alignment",
    "snapshot_readiness",
    "rollback_readiness",
    "governance_gate",
]

KNOWLEDGE_VERSIONING_ENTRY_SERIALIZATION_VERSION = "knowledge_versioning_entry.v1"
KNOWLEDGE_VERSIONING_PLAN_SERIALIZATION_VERSION = "knowledge_versioning_plan.v1"
KNOWLEDGE_VERSIONING_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Versioning exposes knowledge provenance, version "
    "inventory, lineage alignment, snapshot readiness, rollback readiness, "
    "and governance posture as inspectable advisory metadata only; it does "
    "not execute knowledge versioning, mutate version graphs, write version "
    "records, assign version ids, reconstruct version lineage, write version "
    "history, create snapshots, write snapshot storage, execute rollback, "
    "apply rollback plans, mutate provenance graphs, write provenance "
    "records, reconstruct lineage, relink sources, execute lifecycle "
    "management, transition lifecycle stages, mutate lifecycle or retention "
    "policy, archive, deprecate, delete, write lifecycle records, execute "
    "knowledge consolidation, merge or deduplicate knowledge, write canonical "
    "records, write KB storage, update source records, execute retrieval "
    "queries, mutate retrieval configuration, mutate ranking, request "
    "embeddings, refresh embeddings, index vectors, upsert vectors, fetch "
    "documentation, provision providers, infer API keys, route providers or "
    "models, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, execute workflows, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Versioning",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_versioning_execution",
    "version_graph_mutation",
    "version_record_write",
    "version_id_assignment",
    "version_lineage_reconstruction",
    "version_history_write",
    "snapshot_creation",
    "snapshot_storage_write",
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
    "archival_execution",
    "deprecation_execution",
    "deletion_execution",
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


class KnowledgeVersioningSignal(BaseModel):
    """One advisory knowledge versioning signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeVersioningKind
    status: KnowledgeVersioningStatus
    confidence: KnowledgeVersioningConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    versioning_axis: KnowledgeVersioningAxis
    knowledge_provenance_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_provenance_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    version_signal_summary: str = Field(min_length=1, max_length=360)
    version_signal_score: int = Field(ge=0, le=100)
    provenance_alignment_score: int = Field(ge=0, le=100)
    lifecycle_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    versioning_score: int = Field(ge=0, le=1_000)
    hitl_required_before_versioning: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=56,
    )
    knowledge_versioning_capability_implemented: Literal[True] = True
    knowledge_versioning_metadata_implemented: Literal[True] = True
    knowledge_provenance_metadata_used: Literal[True] = True
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    version_id_assignment_implemented: Literal[False] = False
    version_lineage_reconstruction_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
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
    archival_execution_implemented: Literal[False] = False
    deprecation_execution_implemented: Literal[False] = False
    deletion_execution_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_versioning_entry.v1"] = (
        KNOWLEDGE_VERSIONING_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_versioning::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_provenance_signal_count != len(
            self.knowledge_provenance_signal_ids
        ):
            raise ValueError("knowledge_provenance_signal_count must match signals")
        if self.versioning_score != _versioning_score(
            version_signal_score=self.version_signal_score,
            provenance_alignment_score=self.provenance_alignment_score,
            lifecycle_alignment_score=self.lifecycle_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("versioning_score must combine source scores")
        if self.status != _versioning_status(self.versioning_score):
            raise ValueError("status must match versioning_score")
        if self.confidence != _versioning_confidence(self.versioning_score):
            raise ValueError("confidence must match versioning_score")
        if not self.hitl_required_before_versioning:
            raise ValueError("knowledge versioning requires HITL posture")
        return self


class KnowledgeVersioningPlan(BaseModel):
    """Bounded V6.3 advisory knowledge versioning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_versioning"] = "knowledge_versioning"
    serialization_version: Literal["knowledge_versioning_plan.v1"] = (
        KNOWLEDGE_VERSIONING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_VERSIONING_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_provenance_role: Literal["knowledge_provenance_evolution"] = (
        "knowledge_provenance_evolution"
    )
    knowledge_provenance_serialization_version: Literal["knowledge_provenance_plan.v1"]
    knowledge_provenance_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_provenance_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeVersioningSignal, ...] = Field(
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
    planned_version_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_version_graph_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_version_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    assigned_version_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_version_history_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    created_snapshot_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_rollback_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_versioning_score: int = Field(ge=0, le=1_000)
    overall_versioning_score: int = Field(ge=0, le=1_000)
    overall_versioning_posture: KnowledgeVersioningPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=56,
    )
    knowledge_versioning_capability_implemented: Literal[True] = True
    knowledge_versioning_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_provenance_metadata_used: Literal[True] = True
    knowledge_versioning_execution_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    version_record_write_implemented: Literal[False] = False
    version_id_assignment_implemented: Literal[False] = False
    version_lineage_reconstruction_implemented: Literal[False] = False
    version_history_write_implemented: Literal[False] = False
    snapshot_creation_implemented: Literal[False] = False
    snapshot_storage_write_implemented: Literal[False] = False
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
    archival_execution_implemented: Literal[False] = False
    deprecation_execution_implemented: Literal[False] = False
    deletion_execution_implemented: Literal[False] = False
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
            if signal.hitl_required_before_versioning
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_version_ids:
            raise ValueError("planned_version_ids must remain empty")
        if self.mutated_version_graph_ids:
            raise ValueError("mutated_version_graph_ids must remain empty")
        if self.written_version_record_ids:
            raise ValueError("written_version_record_ids must remain empty")
        if self.assigned_version_ids:
            raise ValueError("assigned_version_ids must remain empty")
        if self.written_version_history_ids:
            raise ValueError("written_version_history_ids must remain empty")
        if self.created_snapshot_ids:
            raise ValueError("created_snapshot_ids must remain empty")
        if self.applied_rollback_ids:
            raise ValueError("applied_rollback_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_provenance_signal_count != len(
            self.knowledge_provenance_signal_ids
        ):
            raise ValueError("knowledge_provenance_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 16 roadmap")
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
        if self.highest_versioning_score != max(
            signal.versioning_score for signal in self.signals
        ):
            raise ValueError("highest_versioning_score must match signals")
        if self.overall_versioning_score != _overall_versioning_score(self.signals):
            raise ValueError("overall_versioning_score must match signals")
        if self.overall_versioning_posture != _overall_versioning_posture(self.signals):
            raise ValueError("overall_versioning_posture must match signals")
        declared_provenance_signals = set(self.knowledge_provenance_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_provenance_signal_ids).issubset(
                declared_provenance_signals
            ):
                raise ValueError("signal knowledge_provenance_signal_ids must be known")
        return self


def build_knowledge_versioning(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_provenance: KnowledgeProvenanceEvolutionPlan | None = None,
) -> KnowledgeVersioningPlan:
    """Build V6.3 Task 16 knowledge versioning metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    provenance_plan = knowledge_provenance or build_knowledge_provenance_evolution(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        provenance_plan=provenance_plan,
    )
    return KnowledgeVersioningPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=provenance_plan.checked_at,
        knowledge_provenance_serialization_version=(
            provenance_plan.serialization_version
        ),
        knowledge_provenance_signal_ids=provenance_plan.signal_ids,
        knowledge_provenance_signal_count=len(provenance_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=provenance_plan.source_count,
        domain_count=provenance_plan.domain_count,
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
            if signal.hitl_required_before_versioning
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
            1 for signal in signals if signal.hitl_required_before_versioning
        ),
        highest_versioning_score=max(signal.versioning_score for signal in signals),
        overall_versioning_score=_overall_versioning_score(signals),
        overall_versioning_posture=_overall_versioning_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_versioning_signal_by_id(
    signal_id: str,
    plan: KnowledgeVersioningPlan | None = None,
) -> KnowledgeVersioningSignal | None:
    """Return one knowledge versioning signal without version mutation."""

    source_plan = plan or build_knowledge_versioning()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_versioning_signals_for_status(
    status: KnowledgeVersioningStatus,
    plan: KnowledgeVersioningPlan | None = None,
) -> tuple[KnowledgeVersioningSignal, ...]:
    """Return knowledge versioning signals by advisory status."""

    source_plan = plan or build_knowledge_versioning()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_versioning_signals_for_confidence(
    confidence: KnowledgeVersioningConfidence,
    plan: KnowledgeVersioningPlan | None = None,
) -> tuple[KnowledgeVersioningSignal, ...]:
    """Return knowledge versioning signals by confidence band."""

    source_plan = plan or build_knowledge_versioning()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    provenance_plan: KnowledgeProvenanceEvolutionPlan,
) -> tuple[KnowledgeVersioningSignal, ...]:
    return (
        _signal(
            kind="knowledge_version_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="version_inventory",
            provenance_signal_ids=provenance_plan.signal_ids,
            provenance_plan=provenance_plan,
            version_signal_score=88,
            provenance_alignment_score=86,
            lifecycle_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_version_lineage_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="lineage_alignment",
            provenance_signal_ids=(
                "knowledge_provenance_evolution::knowledge_provenance_inventory_review",
                "knowledge_provenance_evolution::knowledge_provenance_lineage_review",
            ),
            provenance_plan=provenance_plan,
            version_signal_score=80,
            provenance_alignment_score=78,
            lifecycle_alignment_score=76,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_version_snapshot_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="snapshot_readiness",
            provenance_signal_ids=(
                "knowledge_provenance_evolution::"
                "knowledge_provenance_lifecycle_alignment_review",
                "knowledge_provenance_evolution::knowledge_provenance_governance_gate",
            ),
            provenance_plan=provenance_plan,
            version_signal_score=76,
            provenance_alignment_score=74,
            lifecycle_alignment_score=78,
            mutation_risk_score=42,
            governance_weight=105,
        ),
        _signal(
            kind="knowledge_version_rollback_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="rollback_readiness",
            provenance_signal_ids=(
                "knowledge_provenance_evolution::"
                "knowledge_provenance_evolution_readiness",
                "knowledge_provenance_evolution::knowledge_provenance_governance_gate",
            ),
            provenance_plan=provenance_plan,
            version_signal_score=66,
            provenance_alignment_score=68,
            lifecycle_alignment_score=72,
            mutation_risk_score=36,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_version_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            provenance_signal_ids=provenance_plan.signal_ids,
            provenance_plan=provenance_plan,
            version_signal_score=44,
            provenance_alignment_score=46,
            lifecycle_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeVersioningKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeVersioningAxis,
    provenance_signal_ids: tuple[str, ...],
    provenance_plan: KnowledgeProvenanceEvolutionPlan,
    version_signal_score: int,
    provenance_alignment_score: int,
    lifecycle_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeVersioningSignal:
    score = _versioning_score(
        version_signal_score=version_signal_score,
        provenance_alignment_score=provenance_alignment_score,
        lifecycle_alignment_score=lifecycle_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeVersioningSignal(
        signal_id=f"knowledge_versioning::{kind}",
        signal_kind=kind,
        status=_versioning_status(score),
        confidence=_versioning_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        versioning_axis=axis,
        knowledge_provenance_signal_ids=provenance_signal_ids,
        knowledge_provenance_signal_count=len(provenance_signal_ids),
        source_count=provenance_plan.source_count,
        domain_count=provenance_plan.domain_count,
        version_signal_summary=_signal_summary(kind),
        version_signal_score=version_signal_score,
        provenance_alignment_score=provenance_alignment_score,
        lifecycle_alignment_score=lifecycle_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        versioning_score=score,
        hitl_required_before_versioning=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, provenance_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_provenance_signal_count:{len(provenance_signal_ids)}",
            f"source_count:{provenance_plan.source_count}",
            f"domain_count:{provenance_plan.domain_count}",
            f"versioning_axis:{axis}",
            f"version_signal_score:{version_signal_score}",
            f"provenance_alignment_score:{provenance_alignment_score}",
            f"lifecycle_alignment_score:{lifecycle_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_versioning:true",
        ),
    )


def _versioning_score(
    *,
    version_signal_score: int,
    provenance_alignment_score: int,
    lifecycle_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            version_signal_score * 3
            + provenance_alignment_score * 3
            + lifecycle_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _versioning_status(score: int) -> KnowledgeVersioningStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _versioning_confidence(score: int) -> KnowledgeVersioningConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_versioning_score(signals: tuple[KnowledgeVersioningSignal, ...]) -> int:
    base = sum(signal.versioning_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_versioning_posture(
    signals: tuple[KnowledgeVersioningSignal, ...],
) -> KnowledgeVersioningPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeVersioningSignal, ...],
    status: KnowledgeVersioningStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeVersioningSignal, ...],
    *confidences: KnowledgeVersioningConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeVersioningSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_versioning_metadata",
        "verify_knowledge_versioning_roadmap_traceability",
        "review_provenance_signals_before_any_version_action",
        "require_hitl_before_version_graph_snapshot_rollback_or_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeVersioningKind,
    axis: KnowledgeVersioningAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_versioning",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeVersioningKind,
    provenance_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_provenance_signal_count:{len(provenance_signal_ids)}",
        "composes_knowledge_provenance_metadata",
        "keeps_knowledge_versioning_execution_disabled",
        "requires_human_review_before_version_graph_snapshot_rollback_or_write",
    )


def _signal_actions(kind: KnowledgeVersioningKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_versioning_signal_metadata",
        "verify_knowledge_provenance_traceability",
        "keep_knowledge_versioning_disabled",
        "require_hitl_before_knowledge_versioning_action",
    )
    if kind == "knowledge_version_lineage_alignment_review":
        return base_actions + ("review_version_lineage_alignment_metadata",)
    if kind == "knowledge_version_snapshot_readiness":
        return base_actions + ("review_snapshot_readiness_metadata",)
    if kind == "knowledge_version_rollback_readiness":
        return base_actions + ("review_rollback_readiness_metadata",)
    if kind == "knowledge_version_governance_gate":
        return base_actions + ("confirm_manual_versioning_governance_gate",)
    return base_actions + ("review_knowledge_version_inventory_metadata",)


def _signal_summary(kind: KnowledgeVersioningKind) -> str:
    summaries: dict[KnowledgeVersioningKind, str] = {
        "knowledge_version_inventory_review": (
            "Advisory knowledge version inventory posture over provenance "
            "metadata without executing knowledge versioning."
        ),
        "knowledge_version_lineage_alignment_review": (
            "Advisory posture for reviewing version lineage alignment before "
            "version graph mutation, id assignment, history writes, or records."
        ),
        "knowledge_version_snapshot_readiness": (
            "Advisory posture for snapshot readiness while snapshot creation "
            "and snapshot storage writes remain disabled."
        ),
        "knowledge_version_rollback_readiness": (
            "Advisory posture for rollback readiness while rollback execution "
            "and rollback plan application remain disabled."
        ),
        "knowledge_version_governance_gate": (
            "Governed manual gate that keeps knowledge versioning disabled "
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
