"""V6.3 advisory knowledge consolidation metadata."""

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
from creative_coding_assistant.orchestration.source_reliability_engine import (
    SourceReliabilityEnginePlan,
    build_source_reliability_engine,
)

KnowledgeConsolidationKind = Literal[
    "knowledge_consolidation_inventory_review",
    "knowledge_consolidation_candidate_review",
    "knowledge_consolidation_source_alignment_review",
    "knowledge_consolidation_readiness",
    "knowledge_consolidation_governance_gate",
]
KnowledgeConsolidationStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeConsolidationConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeConsolidationPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeConsolidationAxis = Literal[
    "inventory_review",
    "candidate_review",
    "source_alignment_review",
    "consolidation_readiness",
    "governance_gate",
]

KNOWLEDGE_CONSOLIDATION_ENTRY_SERIALIZATION_VERSION = (
    "knowledge_consolidation_entry.v1"
)
KNOWLEDGE_CONSOLIDATION_PLAN_SERIALIZATION_VERSION = (
    "knowledge_consolidation_plan.v1"
)
KNOWLEDGE_CONSOLIDATION_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Consolidation exposes source reliability, consolidation "
    "inventory, candidate, source alignment, readiness, and governance posture "
    "as inspectable advisory metadata only; it does not execute knowledge "
    "consolidation, generate consolidation candidates, merge knowledge, "
    "deduplicate knowledge, write canonical records, write consolidation "
    "records, write KB storage, update source records, execute source "
    "reliability scoring, check source health, compute source trust scores, "
    "mutate source rank, mutate source registries, fetch sources, scan source "
    "freshness, execute drift detection, scan timelines, compare snapshots, "
    "write drift records, execute conflict resolution, arbitrate sources, scan "
    "for gaps, remediate gaps, enrich the KB, compute quality scores, execute "
    "retrieval queries, mutate retrieval configuration, mutate ranking, "
    "request embeddings, refresh embeddings, index vectors, upsert vectors, "
    "fetch documentation, provision providers, infer API keys, route providers "
    "or models, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Consolidation",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_consolidation_execution",
    "consolidation_candidate_generation",
    "knowledge_merge_execution",
    "knowledge_deduplication_execution",
    "canonical_record_write",
    "consolidation_record_write",
    "kb_storage_write",
    "source_record_update",
    "source_reliability_engine_execution",
    "source_reliability_scoring_execution",
    "source_health_check_execution",
    "source_trust_score_computation",
    "source_rank_mutation",
    "source_registry_mutation",
    "source_fetch_execution",
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


class KnowledgeConsolidationSignal(BaseModel):
    """One advisory knowledge consolidation signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeConsolidationKind
    status: KnowledgeConsolidationStatus
    confidence: KnowledgeConsolidationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    consolidation_axis: KnowledgeConsolidationAxis
    source_reliability_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    source_reliability_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    consolidation_signal_summary: str = Field(min_length=1, max_length=360)
    consolidation_signal_score: int = Field(ge=0, le=100)
    source_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    consolidation_score: int = Field(ge=0, le=1_000)
    hitl_required_before_consolidation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=56,
    )
    knowledge_consolidation_capability_implemented: Literal[True] = True
    knowledge_consolidation_metadata_implemented: Literal[True] = True
    source_reliability_metadata_used: Literal[True] = True
    knowledge_consolidation_execution_implemented: Literal[False] = False
    consolidation_candidate_generation_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    source_reliability_engine_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    source_trust_score_computation_implemented: Literal[False] = False
    source_rank_mutation_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    drift_scan_execution_implemented: Literal[False] = False
    timeline_scan_execution_implemented: Literal[False] = False
    snapshot_comparison_execution_implemented: Literal[False] = False
    drift_record_write_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_consolidation_entry.v1"] = (
        KNOWLEDGE_CONSOLIDATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_consolidation::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.source_reliability_signal_count != len(
            self.source_reliability_signal_ids
        ):
            raise ValueError("source_reliability_signal_count must match signals")
        if self.consolidation_score != _consolidation_score(
            consolidation_signal_score=self.consolidation_signal_score,
            source_alignment_score=self.source_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("consolidation_score must combine source scores")
        if self.status != _consolidation_status(self.consolidation_score):
            raise ValueError("status must match consolidation_score")
        if self.confidence != _consolidation_confidence(self.consolidation_score):
            raise ValueError("confidence must match consolidation_score")
        if not self.hitl_required_before_consolidation:
            raise ValueError("knowledge consolidation requires HITL posture")
        return self


class KnowledgeConsolidationPlan(BaseModel):
    """Bounded V6.3 advisory knowledge consolidation plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_consolidation"] = "knowledge_consolidation"
    serialization_version: Literal["knowledge_consolidation_plan.v1"] = (
        KNOWLEDGE_CONSOLIDATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_CONSOLIDATION_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_reliability_role: Literal["source_reliability_engine"] = (
        "source_reliability_engine"
    )
    source_reliability_serialization_version: Literal["source_reliability_plan.v1"]
    source_reliability_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_reliability_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeConsolidationSignal, ...] = Field(
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
    planned_consolidation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generated_consolidation_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    merged_knowledge_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    deduplicated_knowledge_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_canonical_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_consolidation_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    updated_source_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_consolidation_score: int = Field(ge=0, le=1_000)
    overall_consolidation_score: int = Field(ge=0, le=1_000)
    overall_consolidation_posture: KnowledgeConsolidationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=56,
    )
    knowledge_consolidation_capability_implemented: Literal[True] = True
    knowledge_consolidation_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    source_reliability_metadata_used: Literal[True] = True
    knowledge_consolidation_execution_implemented: Literal[False] = False
    consolidation_candidate_generation_implemented: Literal[False] = False
    knowledge_merge_execution_implemented: Literal[False] = False
    knowledge_deduplication_execution_implemented: Literal[False] = False
    canonical_record_write_implemented: Literal[False] = False
    consolidation_record_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    source_reliability_engine_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    source_health_check_execution_implemented: Literal[False] = False
    source_trust_score_computation_implemented: Literal[False] = False
    source_rank_mutation_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    drift_detection_execution_implemented: Literal[False] = False
    drift_scan_execution_implemented: Literal[False] = False
    timeline_scan_execution_implemented: Literal[False] = False
    snapshot_comparison_execution_implemented: Literal[False] = False
    drift_record_write_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
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
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
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
            if signal.hitl_required_before_consolidation
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_consolidation_ids:
            raise ValueError("planned_consolidation_ids must remain empty")
        if self.generated_consolidation_candidate_ids:
            raise ValueError(
                "generated_consolidation_candidate_ids must remain empty"
            )
        if self.merged_knowledge_record_ids:
            raise ValueError("merged_knowledge_record_ids must remain empty")
        if self.deduplicated_knowledge_record_ids:
            raise ValueError("deduplicated_knowledge_record_ids must remain empty")
        if self.written_canonical_record_ids:
            raise ValueError("written_canonical_record_ids must remain empty")
        if self.written_consolidation_record_ids:
            raise ValueError("written_consolidation_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.updated_source_record_ids:
            raise ValueError("updated_source_record_ids must remain empty")
        if self.source_reliability_signal_count != len(
            self.source_reliability_signal_ids
        ):
            raise ValueError("source_reliability_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 13 roadmap")
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
        if self.highest_consolidation_score != max(
            signal.consolidation_score for signal in self.signals
        ):
            raise ValueError("highest_consolidation_score must match signals")
        if self.overall_consolidation_score != _overall_consolidation_score(
            self.signals
        ):
            raise ValueError("overall_consolidation_score must match signals")
        if self.overall_consolidation_posture != _overall_consolidation_posture(
            self.signals
        ):
            raise ValueError("overall_consolidation_posture must match signals")
        declared_reliability_signals = set(self.source_reliability_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.source_reliability_signal_ids).issubset(
                declared_reliability_signals
            ):
                raise ValueError(
                    "signal source_reliability_signal_ids must be known"
                )
        return self


def build_knowledge_consolidation(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    source_reliability: SourceReliabilityEnginePlan | None = None,
) -> KnowledgeConsolidationPlan:
    """Build V6.3 Task 13 knowledge consolidation metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    reliability_plan = source_reliability or build_source_reliability_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        reliability_plan=reliability_plan,
    )
    return KnowledgeConsolidationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=reliability_plan.checked_at,
        source_reliability_serialization_version=(
            reliability_plan.serialization_version
        ),
        source_reliability_signal_ids=reliability_plan.signal_ids,
        source_reliability_signal_count=len(reliability_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=reliability_plan.source_count,
        domain_count=reliability_plan.domain_count,
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
            if signal.hitl_required_before_consolidation
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
            1 for signal in signals if signal.hitl_required_before_consolidation
        ),
        highest_consolidation_score=max(
            signal.consolidation_score for signal in signals
        ),
        overall_consolidation_score=_overall_consolidation_score(signals),
        overall_consolidation_posture=_overall_consolidation_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_consolidation_signal_by_id(
    signal_id: str,
    plan: KnowledgeConsolidationPlan | None = None,
) -> KnowledgeConsolidationSignal | None:
    """Return one knowledge consolidation signal without consolidation."""

    source_plan = plan or build_knowledge_consolidation()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_consolidation_signals_for_status(
    status: KnowledgeConsolidationStatus,
    plan: KnowledgeConsolidationPlan | None = None,
) -> tuple[KnowledgeConsolidationSignal, ...]:
    """Return knowledge consolidation signals by advisory status."""

    source_plan = plan or build_knowledge_consolidation()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_consolidation_signals_for_confidence(
    confidence: KnowledgeConsolidationConfidence,
    plan: KnowledgeConsolidationPlan | None = None,
) -> tuple[KnowledgeConsolidationSignal, ...]:
    """Return knowledge consolidation signals by confidence band."""

    source_plan = plan or build_knowledge_consolidation()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    reliability_plan: SourceReliabilityEnginePlan,
) -> tuple[KnowledgeConsolidationSignal, ...]:
    return (
        _signal(
            kind="knowledge_consolidation_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            reliability_signal_ids=reliability_plan.signal_ids,
            reliability_plan=reliability_plan,
            consolidation_signal_score=88,
            source_alignment_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_consolidation_candidate_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="candidate_review",
            reliability_signal_ids=(
                "source_reliability_engine::source_reliability_inventory_review",
                "source_reliability_engine::"
                "source_reliability_health_signal_review",
            ),
            reliability_plan=reliability_plan,
            consolidation_signal_score=78,
            source_alignment_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_consolidation_source_alignment_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_alignment_review",
            reliability_signal_ids=(
                "source_reliability_engine::"
                "source_reliability_drift_alignment_review",
                "source_reliability_engine::source_reliability_governance_gate",
            ),
            reliability_plan=reliability_plan,
            consolidation_signal_score=70,
            source_alignment_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_consolidation_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="consolidation_readiness",
            reliability_signal_ids=(
                "source_reliability_engine::source_reliability_engine_readiness",
                "source_reliability_engine::source_reliability_governance_gate",
            ),
            reliability_plan=reliability_plan,
            consolidation_signal_score=62,
            source_alignment_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_consolidation_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            reliability_signal_ids=reliability_plan.signal_ids,
            reliability_plan=reliability_plan,
            consolidation_signal_score=44,
            source_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeConsolidationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeConsolidationAxis,
    reliability_signal_ids: tuple[str, ...],
    reliability_plan: SourceReliabilityEnginePlan,
    consolidation_signal_score: int,
    source_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeConsolidationSignal:
    score = _consolidation_score(
        consolidation_signal_score=consolidation_signal_score,
        source_alignment_score=source_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeConsolidationSignal(
        signal_id=f"knowledge_consolidation::{kind}",
        signal_kind=kind,
        status=_consolidation_status(score),
        confidence=_consolidation_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        consolidation_axis=axis,
        source_reliability_signal_ids=reliability_signal_ids,
        source_reliability_signal_count=len(reliability_signal_ids),
        source_count=reliability_plan.source_count,
        domain_count=reliability_plan.domain_count,
        consolidation_signal_summary=_signal_summary(kind),
        consolidation_signal_score=consolidation_signal_score,
        source_alignment_score=source_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        consolidation_score=score,
        hitl_required_before_consolidation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, reliability_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"source_reliability_signal_count:{len(reliability_signal_ids)}",
            f"source_count:{reliability_plan.source_count}",
            f"domain_count:{reliability_plan.domain_count}",
            f"consolidation_axis:{axis}",
            f"consolidation_signal_score:{consolidation_signal_score}",
            f"source_alignment_score:{source_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_consolidation:true",
        ),
    )


def _consolidation_score(
    *,
    consolidation_signal_score: int,
    source_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            consolidation_signal_score * 3
            + source_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _consolidation_status(score: int) -> KnowledgeConsolidationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _consolidation_confidence(score: int) -> KnowledgeConsolidationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_consolidation_score(
    signals: tuple[KnowledgeConsolidationSignal, ...],
) -> int:
    base = sum(signal.consolidation_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_consolidation_posture(
    signals: tuple[KnowledgeConsolidationSignal, ...],
) -> KnowledgeConsolidationPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeConsolidationSignal, ...],
    status: KnowledgeConsolidationStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeConsolidationSignal, ...],
    *confidences: KnowledgeConsolidationConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(
    signals: tuple[KnowledgeConsolidationSignal, ...],
) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_consolidation_metadata",
        "verify_knowledge_consolidation_roadmap_traceability",
        "review_source_reliability_signals_before_any_consolidation_action",
        "require_hitl_before_consolidation_merge_dedup_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeConsolidationKind,
    axis: KnowledgeConsolidationAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_consolidation",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeConsolidationKind,
    reliability_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"source_reliability_signal_count:{len(reliability_signal_ids)}",
        "composes_source_reliability_metadata",
        "keeps_knowledge_consolidation_execution_disabled",
        "requires_human_review_before_consolidation_merge_dedup_or_write",
    )


def _signal_actions(kind: KnowledgeConsolidationKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_consolidation_signal_metadata",
        "verify_source_reliability_traceability",
        "keep_knowledge_consolidation_disabled",
        "require_hitl_before_knowledge_consolidation_action",
    )
    if kind == "knowledge_consolidation_candidate_review":
        return base_actions + ("review_consolidation_candidate_metadata",)
    if kind == "knowledge_consolidation_source_alignment_review":
        return base_actions + ("review_source_alignment_metadata",)
    if kind == "knowledge_consolidation_readiness":
        return base_actions + ("review_consolidation_readiness_metadata",)
    if kind == "knowledge_consolidation_governance_gate":
        return base_actions + ("confirm_manual_consolidation_governance_gate",)
    return base_actions + ("review_knowledge_consolidation_inventory_metadata",)


def _signal_summary(kind: KnowledgeConsolidationKind) -> str:
    summaries: dict[KnowledgeConsolidationKind, str] = {
        "knowledge_consolidation_inventory_review": (
            "Advisory knowledge consolidation inventory posture over source "
            "reliability metadata without executing consolidation."
        ),
        "knowledge_consolidation_candidate_review": (
            "Advisory posture for reviewing consolidation candidate metadata "
            "before merge, deduplication, canonical writes, or KB writes."
        ),
        "knowledge_consolidation_source_alignment_review": (
            "Advisory posture for reviewing source alignment metadata before "
            "any source update or knowledge merge."
        ),
        "knowledge_consolidation_readiness": (
            "Advisory posture for reviewing consolidation readiness while "
            "merge, deduplication, storage writes, and retrieval mutation "
            "remain disabled."
        ),
        "knowledge_consolidation_governance_gate": (
            "Governed manual gate that keeps knowledge consolidation disabled "
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
