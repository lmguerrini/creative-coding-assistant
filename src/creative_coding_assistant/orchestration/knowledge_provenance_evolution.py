"""V6.3 advisory knowledge provenance evolution metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_lifecycle_management import (
    KnowledgeLifecycleManagementPlan,
    build_knowledge_lifecycle_management,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeProvenanceKind = Literal[
    "knowledge_provenance_inventory_review",
    "knowledge_provenance_lineage_review",
    "knowledge_provenance_lifecycle_alignment_review",
    "knowledge_provenance_evolution_readiness",
    "knowledge_provenance_governance_gate",
]
KnowledgeProvenanceStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeProvenanceConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeProvenancePosture = Literal["candidate", "review_required", "guarded"]
KnowledgeProvenanceAxis = Literal[
    "inventory_review",
    "lineage_review",
    "lifecycle_alignment_review",
    "evolution_readiness",
    "governance_gate",
]

KNOWLEDGE_PROVENANCE_ENTRY_SERIALIZATION_VERSION = "knowledge_provenance_entry.v1"
KNOWLEDGE_PROVENANCE_PLAN_SERIALIZATION_VERSION = "knowledge_provenance_plan.v1"
KNOWLEDGE_PROVENANCE_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Provenance Evolution exposes knowledge lifecycle, "
    "provenance inventory, lineage, lifecycle alignment, readiness, and "
    "governance posture as inspectable advisory metadata only; it does not "
    "execute provenance evolution, mutate provenance graphs, write provenance "
    "records, reconstruct lineage, relink sources, execute lifecycle "
    "management, transition lifecycle stages, mutate lifecycle or retention "
    "policy, archive, deprecate, delete, write lifecycle records, execute "
    "knowledge consolidation, merge or deduplicate knowledge, write canonical "
    "records, write KB storage, update source records, execute retrieval "
    "queries, mutate retrieval configuration, mutate ranking, request "
    "embeddings, refresh embeddings, index vectors, upsert vectors, fetch "
    "documentation, provision providers, infer API keys, route providers or "
    "models, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Provenance Evolution",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_provenance_evolution_execution",
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


class KnowledgeProvenanceSignal(BaseModel):
    """One advisory knowledge provenance evolution signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeProvenanceKind
    status: KnowledgeProvenanceStatus
    confidence: KnowledgeProvenanceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    provenance_axis: KnowledgeProvenanceAxis
    knowledge_lifecycle_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_lifecycle_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    provenance_signal_summary: str = Field(min_length=1, max_length=360)
    provenance_signal_score: int = Field(ge=0, le=100)
    lineage_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    provenance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_provenance_evolution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    knowledge_provenance_evolution_capability_implemented: Literal[True] = True
    knowledge_provenance_evolution_metadata_implemented: Literal[True] = True
    knowledge_lifecycle_metadata_used: Literal[True] = True
    knowledge_provenance_evolution_execution_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_provenance_entry.v1"] = (
        KNOWLEDGE_PROVENANCE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_provenance_evolution::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_lifecycle_signal_count != len(
            self.knowledge_lifecycle_signal_ids
        ):
            raise ValueError("knowledge_lifecycle_signal_count must match signals")
        if self.provenance_score != _provenance_score(
            provenance_signal_score=self.provenance_signal_score,
            lineage_alignment_score=self.lineage_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("provenance_score must combine source scores")
        if self.status != _provenance_status(self.provenance_score):
            raise ValueError("status must match provenance_score")
        if self.confidence != _provenance_confidence(self.provenance_score):
            raise ValueError("confidence must match provenance_score")
        if not self.hitl_required_before_provenance_evolution:
            raise ValueError("knowledge provenance evolution requires HITL posture")
        return self


class KnowledgeProvenanceEvolutionPlan(BaseModel):
    """Bounded V6.3 advisory knowledge provenance evolution plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_provenance_evolution"] = (
        "knowledge_provenance_evolution"
    )
    serialization_version: Literal["knowledge_provenance_plan.v1"] = (
        KNOWLEDGE_PROVENANCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_PROVENANCE_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_lifecycle_role: Literal["knowledge_lifecycle_management"] = (
        "knowledge_lifecycle_management"
    )
    knowledge_lifecycle_serialization_version: Literal["knowledge_lifecycle_plan.v1"]
    knowledge_lifecycle_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_lifecycle_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeProvenanceSignal, ...] = Field(
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
    planned_provenance_evolution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_provenance_graph_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_provenance_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reconstructed_lineage_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    relinked_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_provenance_score: int = Field(ge=0, le=1_000)
    overall_provenance_score: int = Field(ge=0, le=1_000)
    overall_provenance_posture: KnowledgeProvenancePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=48,
    )
    knowledge_provenance_evolution_capability_implemented: Literal[True] = True
    knowledge_provenance_evolution_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_lifecycle_metadata_used: Literal[True] = True
    knowledge_provenance_evolution_execution_implemented: Literal[False] = False
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
            if signal.hitl_required_before_provenance_evolution
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_provenance_evolution_ids:
            raise ValueError("planned_provenance_evolution_ids must remain empty")
        if self.mutated_provenance_graph_ids:
            raise ValueError("mutated_provenance_graph_ids must remain empty")
        if self.written_provenance_record_ids:
            raise ValueError("written_provenance_record_ids must remain empty")
        if self.reconstructed_lineage_ids:
            raise ValueError("reconstructed_lineage_ids must remain empty")
        if self.relinked_source_ids:
            raise ValueError("relinked_source_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_lifecycle_signal_count != len(
            self.knowledge_lifecycle_signal_ids
        ):
            raise ValueError("knowledge_lifecycle_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 15 roadmap")
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
        if self.highest_provenance_score != max(
            signal.provenance_score for signal in self.signals
        ):
            raise ValueError("highest_provenance_score must match signals")
        if self.overall_provenance_score != _overall_provenance_score(self.signals):
            raise ValueError("overall_provenance_score must match signals")
        if self.overall_provenance_posture != _overall_provenance_posture(
            self.signals
        ):
            raise ValueError("overall_provenance_posture must match signals")
        declared_lifecycle_signals = set(self.knowledge_lifecycle_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_lifecycle_signal_ids).issubset(
                declared_lifecycle_signals
            ):
                raise ValueError(
                    "signal knowledge_lifecycle_signal_ids must be known"
                )
        return self


def build_knowledge_provenance_evolution(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_lifecycle: KnowledgeLifecycleManagementPlan | None = None,
) -> KnowledgeProvenanceEvolutionPlan:
    """Build V6.3 Task 15 knowledge provenance evolution metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    lifecycle_plan = knowledge_lifecycle or build_knowledge_lifecycle_management(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        lifecycle_plan=lifecycle_plan,
    )
    return KnowledgeProvenanceEvolutionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=lifecycle_plan.checked_at,
        knowledge_lifecycle_serialization_version=(
            lifecycle_plan.serialization_version
        ),
        knowledge_lifecycle_signal_ids=lifecycle_plan.signal_ids,
        knowledge_lifecycle_signal_count=len(lifecycle_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=lifecycle_plan.source_count,
        domain_count=lifecycle_plan.domain_count,
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
            if signal.hitl_required_before_provenance_evolution
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
            if signal.hitl_required_before_provenance_evolution
        ),
        highest_provenance_score=max(signal.provenance_score for signal in signals),
        overall_provenance_score=_overall_provenance_score(signals),
        overall_provenance_posture=_overall_provenance_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_provenance_signal_by_id(
    signal_id: str,
    plan: KnowledgeProvenanceEvolutionPlan | None = None,
) -> KnowledgeProvenanceSignal | None:
    """Return one knowledge provenance signal without provenance mutation."""

    source_plan = plan or build_knowledge_provenance_evolution()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_provenance_signals_for_status(
    status: KnowledgeProvenanceStatus,
    plan: KnowledgeProvenanceEvolutionPlan | None = None,
) -> tuple[KnowledgeProvenanceSignal, ...]:
    """Return knowledge provenance signals by advisory status."""

    source_plan = plan or build_knowledge_provenance_evolution()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_provenance_signals_for_confidence(
    confidence: KnowledgeProvenanceConfidence,
    plan: KnowledgeProvenanceEvolutionPlan | None = None,
) -> tuple[KnowledgeProvenanceSignal, ...]:
    """Return knowledge provenance signals by confidence band."""

    source_plan = plan or build_knowledge_provenance_evolution()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    lifecycle_plan: KnowledgeLifecycleManagementPlan,
) -> tuple[KnowledgeProvenanceSignal, ...]:
    return (
        _signal(
            kind="knowledge_provenance_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            lifecycle_signal_ids=lifecycle_plan.signal_ids,
            lifecycle_plan=lifecycle_plan,
            provenance_signal_score=88,
            lineage_alignment_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_provenance_lineage_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="lineage_review",
            lifecycle_signal_ids=(
                "knowledge_lifecycle_management::"
                "knowledge_lifecycle_inventory_review",
                "knowledge_lifecycle_management::"
                "knowledge_lifecycle_stage_policy_review",
            ),
            lifecycle_plan=lifecycle_plan,
            provenance_signal_score=78,
            lineage_alignment_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_provenance_lifecycle_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="lifecycle_alignment_review",
            lifecycle_signal_ids=(
                "knowledge_lifecycle_management::"
                "knowledge_lifecycle_consolidation_alignment_review",
                "knowledge_lifecycle_management::knowledge_lifecycle_governance_gate",
            ),
            lifecycle_plan=lifecycle_plan,
            provenance_signal_score=70,
            lineage_alignment_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_provenance_evolution_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evolution_readiness",
            lifecycle_signal_ids=(
                "knowledge_lifecycle_management::"
                "knowledge_lifecycle_management_readiness",
                "knowledge_lifecycle_management::knowledge_lifecycle_governance_gate",
            ),
            lifecycle_plan=lifecycle_plan,
            provenance_signal_score=62,
            lineage_alignment_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_provenance_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            lifecycle_signal_ids=lifecycle_plan.signal_ids,
            lifecycle_plan=lifecycle_plan,
            provenance_signal_score=44,
            lineage_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeProvenanceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeProvenanceAxis,
    lifecycle_signal_ids: tuple[str, ...],
    lifecycle_plan: KnowledgeLifecycleManagementPlan,
    provenance_signal_score: int,
    lineage_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeProvenanceSignal:
    score = _provenance_score(
        provenance_signal_score=provenance_signal_score,
        lineage_alignment_score=lineage_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeProvenanceSignal(
        signal_id=f"knowledge_provenance_evolution::{kind}",
        signal_kind=kind,
        status=_provenance_status(score),
        confidence=_provenance_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        provenance_axis=axis,
        knowledge_lifecycle_signal_ids=lifecycle_signal_ids,
        knowledge_lifecycle_signal_count=len(lifecycle_signal_ids),
        source_count=lifecycle_plan.source_count,
        domain_count=lifecycle_plan.domain_count,
        provenance_signal_summary=_signal_summary(kind),
        provenance_signal_score=provenance_signal_score,
        lineage_alignment_score=lineage_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        provenance_score=score,
        hitl_required_before_provenance_evolution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, lifecycle_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_lifecycle_signal_count:{len(lifecycle_signal_ids)}",
            f"source_count:{lifecycle_plan.source_count}",
            f"domain_count:{lifecycle_plan.domain_count}",
            f"provenance_axis:{axis}",
            f"provenance_signal_score:{provenance_signal_score}",
            f"lineage_alignment_score:{lineage_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_provenance_evolution:true",
        ),
    )


def _provenance_score(
    *,
    provenance_signal_score: int,
    lineage_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            provenance_signal_score * 3
            + lineage_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _provenance_status(score: int) -> KnowledgeProvenanceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _provenance_confidence(score: int) -> KnowledgeProvenanceConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_provenance_score(signals: tuple[KnowledgeProvenanceSignal, ...]) -> int:
    base = sum(signal.provenance_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_provenance_posture(
    signals: tuple[KnowledgeProvenanceSignal, ...],
) -> KnowledgeProvenancePosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeProvenanceSignal, ...],
    status: KnowledgeProvenanceStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeProvenanceSignal, ...],
    *confidences: KnowledgeProvenanceConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeProvenanceSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_provenance_evolution_metadata",
        "verify_knowledge_provenance_roadmap_traceability",
        "review_lifecycle_signals_before_any_provenance_action",
        "require_hitl_before_provenance_graph_lineage_relink_or_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeProvenanceKind,
    axis: KnowledgeProvenanceAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_provenance_evolution",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeProvenanceKind,
    lifecycle_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_lifecycle_signal_count:{len(lifecycle_signal_ids)}",
        "composes_knowledge_lifecycle_metadata",
        "keeps_knowledge_provenance_evolution_execution_disabled",
        "requires_human_review_before_provenance_graph_lineage_relink_or_write",
    )


def _signal_actions(kind: KnowledgeProvenanceKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_provenance_signal_metadata",
        "verify_knowledge_lifecycle_traceability",
        "keep_knowledge_provenance_evolution_disabled",
        "require_hitl_before_knowledge_provenance_action",
    )
    if kind == "knowledge_provenance_lineage_review":
        return base_actions + ("review_provenance_lineage_metadata",)
    if kind == "knowledge_provenance_lifecycle_alignment_review":
        return base_actions + ("review_provenance_lifecycle_alignment_metadata",)
    if kind == "knowledge_provenance_evolution_readiness":
        return base_actions + ("review_provenance_evolution_readiness_metadata",)
    if kind == "knowledge_provenance_governance_gate":
        return base_actions + ("confirm_manual_provenance_governance_gate",)
    return base_actions + ("review_knowledge_provenance_inventory_metadata",)


def _signal_summary(kind: KnowledgeProvenanceKind) -> str:
    summaries: dict[KnowledgeProvenanceKind, str] = {
        "knowledge_provenance_inventory_review": (
            "Advisory knowledge provenance inventory posture over lifecycle "
            "metadata without executing provenance evolution."
        ),
        "knowledge_provenance_lineage_review": (
            "Advisory posture for reviewing lineage metadata before "
            "reconstruction, source relinking, graph mutation, or writes."
        ),
        "knowledge_provenance_lifecycle_alignment_review": (
            "Advisory posture for aligning provenance metadata with lifecycle "
            "signals while lifecycle and storage mutation remain disabled."
        ),
        "knowledge_provenance_evolution_readiness": (
            "Advisory posture for provenance readiness while graph mutation, "
            "lineage reconstruction, source relinking, and writes remain "
            "disabled."
        ),
        "knowledge_provenance_governance_gate": (
            "Governed manual gate that keeps knowledge provenance evolution "
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
