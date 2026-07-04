"""V6.3 advisory knowledge trust score metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_freshness_tracking import (
    KnowledgeFreshnessTrackingPlan,
    build_knowledge_freshness_tracking,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeTrustKind = Literal[
    "knowledge_trust_inventory_review",
    "knowledge_trust_source_reliability_review",
    "knowledge_trust_freshness_alignment",
    "knowledge_trust_score_readiness",
    "knowledge_trust_governance_gate",
]
KnowledgeTrustStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeTrustConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeTrustPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeTrustAxis = Literal[
    "inventory_review",
    "source_reliability",
    "freshness_alignment",
    "trust_score_readiness",
    "governance_gate",
]

KNOWLEDGE_TRUST_ENTRY_SERIALIZATION_VERSION = "knowledge_trust_entry.v1"
KNOWLEDGE_TRUST_PLAN_SERIALIZATION_VERSION = "knowledge_trust_plan.v1"
KNOWLEDGE_TRUST_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Trust Score exposes knowledge freshness, source "
    "reliability, trust inventory, trust score readiness, freshness "
    "alignment, and governance posture as inspectable advisory metadata "
    "only; it does not execute trust scoring, compute runtime trust scores, "
    "write trust records, enforce trust thresholds, mutate source trust, "
    "mutate source reliability scores, execute freshness scans, compute "
    "freshness scores, write freshness records, update source timestamps, "
    "mutate staleness state, fetch sources, write KB storage, update source "
    "records, execute retrieval queries, mutate retrieval configuration, "
    "mutate ranking, request embeddings, refresh embeddings, index vectors, "
    "upsert vectors, fetch documentation, provision providers, infer API "
    "keys, route providers or models, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, execute workflows, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Trust Score",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_trust_score_execution",
    "trust_score_computation",
    "trust_score_record_write",
    "trust_threshold_enforcement",
    "source_trust_mutation",
    "source_reliability_score_mutation",
    "knowledge_freshness_tracking_execution",
    "freshness_scan_execution",
    "freshness_score_computation",
    "freshness_record_write",
    "source_timestamp_update",
    "staleness_state_mutation",
    "source_fetch_execution",
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


class KnowledgeTrustSignal(BaseModel):
    """One advisory knowledge trust score signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeTrustKind
    status: KnowledgeTrustStatus
    confidence: KnowledgeTrustConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    trust_axis: KnowledgeTrustAxis
    knowledge_freshness_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    knowledge_freshness_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    trust_signal_summary: str = Field(min_length=1, max_length=360)
    trust_signal_score: int = Field(ge=0, le=100)
    freshness_alignment_score: int = Field(ge=0, le=100)
    reliability_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    trust_score: int = Field(ge=0, le=1_000)
    hitl_required_before_trust_action: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_trust_score_capability_implemented: Literal[True] = True
    knowledge_trust_metadata_implemented: Literal[True] = True
    knowledge_freshness_metadata_used: Literal[True] = True
    knowledge_trust_score_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    trust_score_record_write_implemented: Literal[False] = False
    trust_threshold_enforcement_implemented: Literal[False] = False
    source_trust_mutation_implemented: Literal[False] = False
    source_reliability_score_mutation_implemented: Literal[False] = False
    knowledge_freshness_tracking_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    freshness_score_computation_implemented: Literal[False] = False
    freshness_record_write_implemented: Literal[False] = False
    source_timestamp_update_implemented: Literal[False] = False
    staleness_state_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_trust_entry.v1"] = (
        KNOWLEDGE_TRUST_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_trust_score::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_freshness_signal_count != len(
            self.knowledge_freshness_signal_ids
        ):
            raise ValueError("knowledge_freshness_signal_count must match signals")
        if self.trust_score != _trust_score(
            trust_signal_score=self.trust_signal_score,
            freshness_alignment_score=self.freshness_alignment_score,
            reliability_alignment_score=self.reliability_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("trust_score must combine source scores")
        if self.status != _trust_status(self.trust_score):
            raise ValueError("status must match trust_score")
        if self.confidence != _trust_confidence(self.trust_score):
            raise ValueError("confidence must match trust_score")
        if not self.hitl_required_before_trust_action:
            raise ValueError("knowledge trust score requires HITL posture")
        return self


class KnowledgeTrustScorePlan(BaseModel):
    """Bounded V6.3 advisory knowledge trust score plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_trust_score"] = "knowledge_trust_score"
    serialization_version: Literal["knowledge_trust_plan.v1"] = (
        KNOWLEDGE_TRUST_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_TRUST_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_freshness_role: Literal["knowledge_freshness_tracking"] = (
        "knowledge_freshness_tracking"
    )
    knowledge_freshness_serialization_version: Literal["knowledge_freshness_plan.v1"]
    knowledge_freshness_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    knowledge_freshness_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeTrustSignal, ...] = Field(
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
    planned_trust_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_trust_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_trust_score_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    enforced_trust_threshold_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_source_trust_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_source_reliability_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_freshness_scan_ids: tuple[str, ...] = Field(
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
    highest_trust_score: int = Field(ge=0, le=1_000)
    overall_trust_score: int = Field(ge=0, le=1_000)
    overall_trust_posture: KnowledgeTrustPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=60,
    )
    knowledge_trust_score_capability_implemented: Literal[True] = True
    knowledge_trust_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_freshness_metadata_used: Literal[True] = True
    knowledge_trust_score_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    trust_score_record_write_implemented: Literal[False] = False
    trust_threshold_enforcement_implemented: Literal[False] = False
    source_trust_mutation_implemented: Literal[False] = False
    source_reliability_score_mutation_implemented: Literal[False] = False
    knowledge_freshness_tracking_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    freshness_score_computation_implemented: Literal[False] = False
    freshness_record_write_implemented: Literal[False] = False
    source_timestamp_update_implemented: Literal[False] = False
    staleness_state_mutation_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
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
            if signal.hitl_required_before_trust_action
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_trust_score_ids:
            raise ValueError("planned_trust_score_ids must remain empty")
        if self.computed_trust_score_ids:
            raise ValueError("computed_trust_score_ids must remain empty")
        if self.written_trust_score_record_ids:
            raise ValueError("written_trust_score_record_ids must remain empty")
        if self.enforced_trust_threshold_ids:
            raise ValueError("enforced_trust_threshold_ids must remain empty")
        if self.mutated_source_trust_ids:
            raise ValueError("mutated_source_trust_ids must remain empty")
        if self.mutated_source_reliability_score_ids:
            raise ValueError("mutated_source_reliability_score_ids must remain empty")
        if self.executed_freshness_scan_ids:
            raise ValueError("executed_freshness_scan_ids must remain empty")
        if self.fetched_source_ids:
            raise ValueError("fetched_source_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_freshness_signal_count != len(
            self.knowledge_freshness_signal_ids
        ):
            raise ValueError("knowledge_freshness_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 20 roadmap")
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
        if self.highest_trust_score != max(
            signal.trust_score for signal in self.signals
        ):
            raise ValueError("highest_trust_score must match signals")
        if self.overall_trust_score != _overall_trust_score(self.signals):
            raise ValueError("overall_trust_score must match signals")
        if self.overall_trust_posture != _overall_trust_posture(self.signals):
            raise ValueError("overall_trust_posture must match signals")
        declared_freshness_signals = set(self.knowledge_freshness_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_freshness_signal_ids).issubset(
                declared_freshness_signals
            ):
                raise ValueError("signal knowledge_freshness_signal_ids must be known")
        return self


def build_knowledge_trust_score(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_freshness: KnowledgeFreshnessTrackingPlan | None = None,
) -> KnowledgeTrustScorePlan:
    """Build V6.3 Task 20 knowledge trust score metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    freshness_plan = knowledge_freshness or build_knowledge_freshness_tracking(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        freshness_plan=freshness_plan,
    )
    return KnowledgeTrustScorePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=freshness_plan.checked_at,
        knowledge_freshness_serialization_version=(
            freshness_plan.serialization_version
        ),
        knowledge_freshness_signal_ids=freshness_plan.signal_ids,
        knowledge_freshness_signal_count=len(freshness_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=freshness_plan.source_count,
        domain_count=freshness_plan.domain_count,
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
            if signal.hitl_required_before_trust_action
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
            1 for signal in signals if signal.hitl_required_before_trust_action
        ),
        highest_trust_score=max(signal.trust_score for signal in signals),
        overall_trust_score=_overall_trust_score(signals),
        overall_trust_posture=_overall_trust_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_trust_signal_by_id(
    signal_id: str,
    plan: KnowledgeTrustScorePlan | None = None,
) -> KnowledgeTrustSignal | None:
    """Return one knowledge trust signal without trust mutation."""

    source_plan = plan or build_knowledge_trust_score()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_trust_signals_for_status(
    status: KnowledgeTrustStatus,
    plan: KnowledgeTrustScorePlan | None = None,
) -> tuple[KnowledgeTrustSignal, ...]:
    """Return knowledge trust signals by advisory status."""

    source_plan = plan or build_knowledge_trust_score()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_trust_signals_for_confidence(
    confidence: KnowledgeTrustConfidence,
    plan: KnowledgeTrustScorePlan | None = None,
) -> tuple[KnowledgeTrustSignal, ...]:
    """Return knowledge trust signals by confidence band."""

    source_plan = plan or build_knowledge_trust_score()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    freshness_plan: KnowledgeFreshnessTrackingPlan,
) -> tuple[KnowledgeTrustSignal, ...]:
    return (
        _signal(
            kind="knowledge_trust_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            freshness_signal_ids=freshness_plan.signal_ids,
            freshness_plan=freshness_plan,
            trust_signal_score=88,
            freshness_alignment_score=86,
            reliability_alignment_score=84,
            governance_alignment_score=84,
            mutation_risk_score=46,
            governance_weight=130,
        ),
        _signal(
            kind="knowledge_trust_source_reliability_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_reliability",
            freshness_signal_ids=(
                "knowledge_freshness_tracking::knowledge_freshness_inventory_review",
                "knowledge_freshness_tracking::knowledge_freshness_source_age_review",
            ),
            freshness_plan=freshness_plan,
            trust_signal_score=78,
            freshness_alignment_score=76,
            reliability_alignment_score=76,
            governance_alignment_score=74,
            mutation_risk_score=42,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_trust_freshness_alignment",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="freshness_alignment",
            freshness_signal_ids=(
                "knowledge_freshness_tracking::knowledge_freshness_source_age_review",
                "knowledge_freshness_tracking::knowledge_freshness_staleness_readiness",
                "knowledge_freshness_tracking::"
                "knowledge_freshness_rollback_alignment_review",
            ),
            freshness_plan=freshness_plan,
            trust_signal_score=74,
            freshness_alignment_score=82,
            reliability_alignment_score=70,
            governance_alignment_score=76,
            mutation_risk_score=40,
            governance_weight=85,
        ),
        _signal(
            kind="knowledge_trust_score_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="trust_score_readiness",
            freshness_signal_ids=(
                "knowledge_freshness_tracking::knowledge_freshness_staleness_readiness",
                "knowledge_freshness_tracking::knowledge_freshness_governance_gate",
            ),
            freshness_plan=freshness_plan,
            trust_signal_score=64,
            freshness_alignment_score=68,
            reliability_alignment_score=66,
            governance_alignment_score=72,
            mutation_risk_score=34,
            governance_weight=70,
        ),
        _signal(
            kind="knowledge_trust_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            freshness_signal_ids=freshness_plan.signal_ids,
            freshness_plan=freshness_plan,
            trust_signal_score=42,
            freshness_alignment_score=48,
            reliability_alignment_score=50,
            governance_alignment_score=92,
            mutation_risk_score=14,
            governance_weight=55,
        ),
    )


def _signal(
    *,
    kind: KnowledgeTrustKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeTrustAxis,
    freshness_signal_ids: tuple[str, ...],
    freshness_plan: KnowledgeFreshnessTrackingPlan,
    trust_signal_score: int,
    freshness_alignment_score: int,
    reliability_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeTrustSignal:
    score = _trust_score(
        trust_signal_score=trust_signal_score,
        freshness_alignment_score=freshness_alignment_score,
        reliability_alignment_score=reliability_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeTrustSignal(
        signal_id=f"knowledge_trust_score::{kind}",
        signal_kind=kind,
        status=_trust_status(score),
        confidence=_trust_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        trust_axis=axis,
        knowledge_freshness_signal_ids=freshness_signal_ids,
        knowledge_freshness_signal_count=len(freshness_signal_ids),
        source_count=freshness_plan.source_count,
        domain_count=freshness_plan.domain_count,
        trust_signal_summary=_signal_summary(kind),
        trust_signal_score=trust_signal_score,
        freshness_alignment_score=freshness_alignment_score,
        reliability_alignment_score=reliability_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        trust_score=score,
        hitl_required_before_trust_action=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, freshness_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_freshness_signal_count:{len(freshness_signal_ids)}",
            f"source_count:{freshness_plan.source_count}",
            f"domain_count:{freshness_plan.domain_count}",
            f"trust_axis:{axis}",
            f"trust_signal_score:{trust_signal_score}",
            f"freshness_alignment_score:{freshness_alignment_score}",
            f"reliability_alignment_score:{reliability_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_trust_action:true",
        ),
    )


def _trust_score(
    *,
    trust_signal_score: int,
    freshness_alignment_score: int,
    reliability_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            trust_signal_score * 3
            + freshness_alignment_score * 2
            + reliability_alignment_score * 2
            + governance_alignment_score * 2
            + mutation_risk_score
            + governance_weight,
        ),
    )


def _trust_status(score: int) -> KnowledgeTrustStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _trust_confidence(score: int) -> KnowledgeTrustConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_trust_score(signals: tuple[KnowledgeTrustSignal, ...]) -> int:
    base = sum(signal.trust_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_trust_posture(
    signals: tuple[KnowledgeTrustSignal, ...],
) -> KnowledgeTrustPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeTrustSignal, ...],
    status: KnowledgeTrustStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeTrustSignal, ...],
    *confidences: KnowledgeTrustConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeTrustSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_trust_score_metadata",
        "verify_knowledge_trust_score_roadmap_traceability",
        "review_freshness_signals_before_any_trust_action",
        "require_hitl_before_trust_score_threshold_storage_or_ranking_action",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeTrustKind,
    axis: KnowledgeTrustAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_trust_score",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeTrustKind,
    freshness_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_freshness_signal_count:{len(freshness_signal_ids)}",
        "composes_knowledge_freshness_metadata",
        "keeps_knowledge_trust_score_execution_disabled",
        "requires_human_review_before_trust_score_threshold_storage_or_ranking",
    )


def _signal_actions(kind: KnowledgeTrustKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_trust_score_signal_metadata",
        "verify_knowledge_freshness_traceability",
        "keep_knowledge_trust_score_execution_disabled",
        "require_hitl_before_knowledge_trust_action",
    )
    if kind == "knowledge_trust_source_reliability_review":
        return base_actions + ("review_source_reliability_trust_metadata",)
    if kind == "knowledge_trust_freshness_alignment":
        return base_actions + ("review_trust_freshness_alignment_metadata",)
    if kind == "knowledge_trust_score_readiness":
        return base_actions + ("review_trust_score_readiness_metadata",)
    if kind == "knowledge_trust_governance_gate":
        return base_actions + ("confirm_manual_trust_score_governance_gate",)
    return base_actions + ("review_knowledge_trust_inventory_metadata",)


def _signal_summary(kind: KnowledgeTrustKind) -> str:
    summaries: dict[KnowledgeTrustKind, str] = {
        "knowledge_trust_inventory_review": (
            "Advisory knowledge trust inventory posture over freshness "
            "metadata without executing trust scoring."
        ),
        "knowledge_trust_source_reliability_review": (
            "Advisory posture for reviewing source reliability trust metadata "
            "before source trust mutation or threshold enforcement."
        ),
        "knowledge_trust_freshness_alignment": (
            "Advisory posture for aligning trust metadata with freshness "
            "signals while freshness scans and trust writes remain disabled."
        ),
        "knowledge_trust_score_readiness": (
            "Advisory posture for trust score readiness while runtime trust "
            "score computation and persistence remain disabled."
        ),
        "knowledge_trust_governance_gate": (
            "Governed manual gate that keeps trust scoring disabled until "
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
