"""V6.3 advisory knowledge lifecycle management metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_consolidation import (
    KnowledgeConsolidationPlan,
    build_knowledge_consolidation,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeLifecycleKind = Literal[
    "knowledge_lifecycle_inventory_review",
    "knowledge_lifecycle_stage_policy_review",
    "knowledge_lifecycle_consolidation_alignment_review",
    "knowledge_lifecycle_management_readiness",
    "knowledge_lifecycle_governance_gate",
]
KnowledgeLifecycleStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeLifecycleConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeLifecyclePosture = Literal["candidate", "review_required", "guarded"]
KnowledgeLifecycleAxis = Literal[
    "inventory_review",
    "stage_policy_review",
    "consolidation_alignment_review",
    "lifecycle_readiness",
    "governance_gate",
]

KNOWLEDGE_LIFECYCLE_ENTRY_SERIALIZATION_VERSION = "knowledge_lifecycle_entry.v1"
KNOWLEDGE_LIFECYCLE_PLAN_SERIALIZATION_VERSION = "knowledge_lifecycle_plan.v1"
KNOWLEDGE_LIFECYCLE_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Lifecycle Management exposes knowledge consolidation, "
    "lifecycle inventory, stage policy, consolidation alignment, readiness, "
    "and governance posture as inspectable advisory metadata only; it does "
    "not execute lifecycle management, transition lifecycle stages, mutate "
    "lifecycle policy, mutate retention policy, archive knowledge, deprecate "
    "knowledge, delete knowledge, write lifecycle records, execute knowledge "
    "consolidation, generate consolidation candidates, merge knowledge, "
    "deduplicate knowledge, write canonical records, write consolidation "
    "records, write KB storage, update source records, fetch sources, execute "
    "drift detection, resolve conflicts, remediate gaps, enrich the KB, "
    "execute retrieval queries, mutate retrieval configuration, mutate "
    "ranking, request embeddings, refresh embeddings, index vectors, upsert "
    "vectors, fetch documentation, provision providers, infer API keys, route "
    "providers or models, execute providers, invoke agents, control "
    "workflows, mutate workflow graphs, modify generated output, or apply "
    "Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Lifecycle Management",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_lifecycle_management_execution",
    "lifecycle_stage_transition",
    "lifecycle_policy_mutation",
    "retention_policy_mutation",
    "archival_execution",
    "deprecation_execution",
    "deletion_execution",
    "lifecycle_record_write",
    "knowledge_consolidation_execution",
    "consolidation_candidate_generation",
    "knowledge_merge_execution",
    "knowledge_deduplication_execution",
    "canonical_record_write",
    "consolidation_record_write",
    "kb_storage_write",
    "source_record_update",
    "source_fetch_execution",
    "knowledge_drift_detection_execution",
    "drift_detection_execution",
    "conflict_resolution_execution",
    "gap_remediation_execution",
    "kb_enrichment_execution",
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


class KnowledgeLifecycleSignal(BaseModel):
    """One advisory knowledge lifecycle management signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeLifecycleKind
    status: KnowledgeLifecycleStatus
    confidence: KnowledgeLifecycleConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    lifecycle_axis: KnowledgeLifecycleAxis
    knowledge_consolidation_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_consolidation_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    lifecycle_signal_summary: str = Field(min_length=1, max_length=360)
    lifecycle_signal_score: int = Field(ge=0, le=100)
    policy_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    lifecycle_score: int = Field(ge=0, le=1_000)
    hitl_required_before_lifecycle_management: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    knowledge_lifecycle_management_capability_implemented: Literal[True] = True
    knowledge_lifecycle_management_metadata_implemented: Literal[True] = True
    knowledge_consolidation_metadata_used: Literal[True] = True
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_stage_transition_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    archival_execution_implemented: Literal[False] = False
    deprecation_execution_implemented: Literal[False] = False
    deletion_execution_implemented: Literal[False] = False
    lifecycle_record_write_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    consolidation_candidate_generation_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_lifecycle_entry.v1"] = (
        KNOWLEDGE_LIFECYCLE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_lifecycle_management::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_consolidation_signal_count != len(
            self.knowledge_consolidation_signal_ids
        ):
            raise ValueError(
                "knowledge_consolidation_signal_count must match signals"
            )
        if self.lifecycle_score != _lifecycle_score(
            lifecycle_signal_score=self.lifecycle_signal_score,
            policy_alignment_score=self.policy_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("lifecycle_score must combine source scores")
        if self.status != _lifecycle_status(self.lifecycle_score):
            raise ValueError("status must match lifecycle_score")
        if self.confidence != _lifecycle_confidence(self.lifecycle_score):
            raise ValueError("confidence must match lifecycle_score")
        if not self.hitl_required_before_lifecycle_management:
            raise ValueError("knowledge lifecycle management requires HITL posture")
        return self


class KnowledgeLifecycleManagementPlan(BaseModel):
    """Bounded V6.3 advisory knowledge lifecycle management plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_lifecycle_management"] = (
        "knowledge_lifecycle_management"
    )
    serialization_version: Literal["knowledge_lifecycle_plan.v1"] = (
        KNOWLEDGE_LIFECYCLE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_LIFECYCLE_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_consolidation_role: Literal["knowledge_consolidation"] = (
        "knowledge_consolidation"
    )
    knowledge_consolidation_serialization_version: Literal[
        "knowledge_consolidation_plan.v1"
    ]
    knowledge_consolidation_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_consolidation_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeLifecycleSignal, ...] = Field(
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
    planned_lifecycle_management_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    transitioned_lifecycle_stage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_lifecycle_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retention_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    archived_knowledge_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    deprecated_knowledge_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    deleted_knowledge_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_lifecycle_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_lifecycle_score: int = Field(ge=0, le=1_000)
    overall_lifecycle_score: int = Field(ge=0, le=1_000)
    overall_lifecycle_posture: KnowledgeLifecyclePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    knowledge_lifecycle_management_capability_implemented: Literal[True] = True
    knowledge_lifecycle_management_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_consolidation_metadata_used: Literal[True] = True
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    lifecycle_stage_transition_implemented: Literal[False] = False
    lifecycle_policy_mutation_implemented: Literal[False] = False
    retention_policy_mutation_implemented: Literal[False] = False
    archival_execution_implemented: Literal[False] = False
    deprecation_execution_implemented: Literal[False] = False
    deletion_execution_implemented: Literal[False] = False
    lifecycle_record_write_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    consolidation_candidate_generation_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
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
            if signal.hitl_required_before_lifecycle_management
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_lifecycle_management_ids:
            raise ValueError("planned_lifecycle_management_ids must remain empty")
        if self.transitioned_lifecycle_stage_ids:
            raise ValueError("transitioned_lifecycle_stage_ids must remain empty")
        if self.mutated_lifecycle_policy_ids:
            raise ValueError("mutated_lifecycle_policy_ids must remain empty")
        if self.mutated_retention_policy_ids:
            raise ValueError("mutated_retention_policy_ids must remain empty")
        if self.archived_knowledge_record_ids:
            raise ValueError("archived_knowledge_record_ids must remain empty")
        if self.deprecated_knowledge_record_ids:
            raise ValueError("deprecated_knowledge_record_ids must remain empty")
        if self.deleted_knowledge_record_ids:
            raise ValueError("deleted_knowledge_record_ids must remain empty")
        if self.written_lifecycle_record_ids:
            raise ValueError("written_lifecycle_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_consolidation_signal_count != len(
            self.knowledge_consolidation_signal_ids
        ):
            raise ValueError("knowledge_consolidation_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 14 roadmap")
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
        if self.highest_lifecycle_score != max(
            signal.lifecycle_score for signal in self.signals
        ):
            raise ValueError("highest_lifecycle_score must match signals")
        if self.overall_lifecycle_score != _overall_lifecycle_score(self.signals):
            raise ValueError("overall_lifecycle_score must match signals")
        if self.overall_lifecycle_posture != _overall_lifecycle_posture(
            self.signals
        ):
            raise ValueError("overall_lifecycle_posture must match signals")
        declared_consolidation_signals = set(self.knowledge_consolidation_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_consolidation_signal_ids).issubset(
                declared_consolidation_signals
            ):
                raise ValueError(
                    "signal knowledge_consolidation_signal_ids must be known"
                )
        return self


def build_knowledge_lifecycle_management(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_consolidation: KnowledgeConsolidationPlan | None = None,
) -> KnowledgeLifecycleManagementPlan:
    """Build V6.3 Task 14 knowledge lifecycle management metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    consolidation_plan = knowledge_consolidation or build_knowledge_consolidation(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        consolidation_plan=consolidation_plan,
    )
    return KnowledgeLifecycleManagementPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=consolidation_plan.checked_at,
        knowledge_consolidation_serialization_version=(
            consolidation_plan.serialization_version
        ),
        knowledge_consolidation_signal_ids=consolidation_plan.signal_ids,
        knowledge_consolidation_signal_count=len(consolidation_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=consolidation_plan.source_count,
        domain_count=consolidation_plan.domain_count,
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
            if signal.hitl_required_before_lifecycle_management
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
            1
            for signal in signals
            if signal.hitl_required_before_lifecycle_management
        ),
        highest_lifecycle_score=max(signal.lifecycle_score for signal in signals),
        overall_lifecycle_score=_overall_lifecycle_score(signals),
        overall_lifecycle_posture=_overall_lifecycle_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_lifecycle_signal_by_id(
    signal_id: str,
    plan: KnowledgeLifecycleManagementPlan | None = None,
) -> KnowledgeLifecycleSignal | None:
    """Return one knowledge lifecycle signal without lifecycle mutation."""

    source_plan = plan or build_knowledge_lifecycle_management()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_lifecycle_signals_for_status(
    status: KnowledgeLifecycleStatus,
    plan: KnowledgeLifecycleManagementPlan | None = None,
) -> tuple[KnowledgeLifecycleSignal, ...]:
    """Return knowledge lifecycle signals by advisory status."""

    source_plan = plan or build_knowledge_lifecycle_management()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_lifecycle_signals_for_confidence(
    confidence: KnowledgeLifecycleConfidence,
    plan: KnowledgeLifecycleManagementPlan | None = None,
) -> tuple[KnowledgeLifecycleSignal, ...]:
    """Return knowledge lifecycle signals by confidence band."""

    source_plan = plan or build_knowledge_lifecycle_management()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    consolidation_plan: KnowledgeConsolidationPlan,
) -> tuple[KnowledgeLifecycleSignal, ...]:
    return (
        _signal(
            kind="knowledge_lifecycle_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            consolidation_signal_ids=consolidation_plan.signal_ids,
            consolidation_plan=consolidation_plan,
            lifecycle_signal_score=88,
            policy_alignment_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_lifecycle_stage_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="stage_policy_review",
            consolidation_signal_ids=(
                "knowledge_consolidation::knowledge_consolidation_inventory_review",
                "knowledge_consolidation::knowledge_consolidation_candidate_review",
            ),
            consolidation_plan=consolidation_plan,
            lifecycle_signal_score=78,
            policy_alignment_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_lifecycle_consolidation_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="consolidation_alignment_review",
            consolidation_signal_ids=(
                "knowledge_consolidation::"
                "knowledge_consolidation_source_alignment_review",
                "knowledge_consolidation::knowledge_consolidation_governance_gate",
            ),
            consolidation_plan=consolidation_plan,
            lifecycle_signal_score=70,
            policy_alignment_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_lifecycle_management_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="lifecycle_readiness",
            consolidation_signal_ids=(
                "knowledge_consolidation::knowledge_consolidation_readiness",
                "knowledge_consolidation::knowledge_consolidation_governance_gate",
            ),
            consolidation_plan=consolidation_plan,
            lifecycle_signal_score=62,
            policy_alignment_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_lifecycle_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            consolidation_signal_ids=consolidation_plan.signal_ids,
            consolidation_plan=consolidation_plan,
            lifecycle_signal_score=44,
            policy_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeLifecycleKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeLifecycleAxis,
    consolidation_signal_ids: tuple[str, ...],
    consolidation_plan: KnowledgeConsolidationPlan,
    lifecycle_signal_score: int,
    policy_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeLifecycleSignal:
    score = _lifecycle_score(
        lifecycle_signal_score=lifecycle_signal_score,
        policy_alignment_score=policy_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeLifecycleSignal(
        signal_id=f"knowledge_lifecycle_management::{kind}",
        signal_kind=kind,
        status=_lifecycle_status(score),
        confidence=_lifecycle_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        lifecycle_axis=axis,
        knowledge_consolidation_signal_ids=consolidation_signal_ids,
        knowledge_consolidation_signal_count=len(consolidation_signal_ids),
        source_count=consolidation_plan.source_count,
        domain_count=consolidation_plan.domain_count,
        lifecycle_signal_summary=_signal_summary(kind),
        lifecycle_signal_score=lifecycle_signal_score,
        policy_alignment_score=policy_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        lifecycle_score=score,
        hitl_required_before_lifecycle_management=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, consolidation_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_consolidation_signal_count:{len(consolidation_signal_ids)}",
            f"source_count:{consolidation_plan.source_count}",
            f"domain_count:{consolidation_plan.domain_count}",
            f"lifecycle_axis:{axis}",
            f"lifecycle_signal_score:{lifecycle_signal_score}",
            f"policy_alignment_score:{policy_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_lifecycle_management:true",
        ),
    )


def _lifecycle_score(
    *,
    lifecycle_signal_score: int,
    policy_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            lifecycle_signal_score * 3
            + policy_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _lifecycle_status(score: int) -> KnowledgeLifecycleStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _lifecycle_confidence(score: int) -> KnowledgeLifecycleConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_lifecycle_score(signals: tuple[KnowledgeLifecycleSignal, ...]) -> int:
    base = sum(signal.lifecycle_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_lifecycle_posture(
    signals: tuple[KnowledgeLifecycleSignal, ...],
) -> KnowledgeLifecyclePosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeLifecycleSignal, ...],
    status: KnowledgeLifecycleStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeLifecycleSignal, ...],
    *confidences: KnowledgeLifecycleConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeLifecycleSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_lifecycle_management_metadata",
        "verify_knowledge_lifecycle_roadmap_traceability",
        "review_consolidation_signals_before_any_lifecycle_action",
        "require_hitl_before_lifecycle_transition_policy_archive_or_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeLifecycleKind,
    axis: KnowledgeLifecycleAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_lifecycle_management",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeLifecycleKind,
    consolidation_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_consolidation_signal_count:{len(consolidation_signal_ids)}",
        "composes_knowledge_consolidation_metadata",
        "keeps_knowledge_lifecycle_management_execution_disabled",
        "requires_human_review_before_lifecycle_transition_policy_or_write",
    )


def _signal_actions(kind: KnowledgeLifecycleKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_lifecycle_signal_metadata",
        "verify_knowledge_consolidation_traceability",
        "keep_knowledge_lifecycle_management_disabled",
        "require_hitl_before_knowledge_lifecycle_action",
    )
    if kind == "knowledge_lifecycle_stage_policy_review":
        return base_actions + ("review_lifecycle_stage_policy_metadata",)
    if kind == "knowledge_lifecycle_consolidation_alignment_review":
        return base_actions + ("review_lifecycle_consolidation_alignment_metadata",)
    if kind == "knowledge_lifecycle_management_readiness":
        return base_actions + ("review_lifecycle_management_readiness_metadata",)
    if kind == "knowledge_lifecycle_governance_gate":
        return base_actions + ("confirm_manual_lifecycle_governance_gate",)
    return base_actions + ("review_knowledge_lifecycle_inventory_metadata",)


def _signal_summary(kind: KnowledgeLifecycleKind) -> str:
    summaries: dict[KnowledgeLifecycleKind, str] = {
        "knowledge_lifecycle_inventory_review": (
            "Advisory knowledge lifecycle inventory posture over consolidation "
            "metadata without executing lifecycle management."
        ),
        "knowledge_lifecycle_stage_policy_review": (
            "Advisory posture for reviewing lifecycle stage policy metadata "
            "before transitions, policy mutation, archive, deprecate, or delete."
        ),
        "knowledge_lifecycle_consolidation_alignment_review": (
            "Advisory posture for aligning lifecycle metadata with "
            "consolidation signals while source and KB writes remain disabled."
        ),
        "knowledge_lifecycle_management_readiness": (
            "Advisory posture for lifecycle readiness while transitions, "
            "retention mutation, archive, deprecate, delete, and writes remain "
            "disabled."
        ),
        "knowledge_lifecycle_governance_gate": (
            "Governed manual gate that keeps knowledge lifecycle management "
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
