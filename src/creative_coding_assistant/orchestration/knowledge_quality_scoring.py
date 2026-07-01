"""V6.3 advisory knowledge quality scoring metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_health_monitoring import (
    KnowledgeHealthMonitoringPlan,
    build_knowledge_health_monitoring,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeQualityScoringKind = Literal[
    "knowledge_quality_inventory_review",
    "knowledge_quality_accuracy_review",
    "knowledge_quality_completeness_review",
    "knowledge_quality_scoring_readiness",
    "knowledge_quality_governance_gate",
]
KnowledgeQualityScoringStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeQualityScoringConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeQualityScoringPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeQualityScoringAxis = Literal[
    "inventory_review",
    "accuracy_review",
    "completeness_review",
    "scoring_readiness",
    "governance_gate",
]

KNOWLEDGE_QUALITY_ENTRY_SERIALIZATION_VERSION = "knowledge_quality_entry.v1"
KNOWLEDGE_QUALITY_PLAN_SERIALIZATION_VERSION = "knowledge_quality_plan.v1"
KNOWLEDGE_QUALITY_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Quality Scoring exposes knowledge health, quality "
    "inventory, accuracy, completeness, scoring readiness, and governance "
    "posture as inspectable advisory metadata only; it does not execute "
    "quality scoring, compute quality scores, persist quality scores, collect "
    "quality metrics, mutate quality policies, emit quality alerts, collect "
    "health metrics, mutate health monitors, execute ranking optimization, "
    "mutate ranking, execute retrieval queries, mutate retrieval "
    "configuration, rerank retrieval results, request embeddings, refresh "
    "embeddings, index vectors, upsert vectors, write KB storage, fetch "
    "documentation, enrich the KB, update source records, provision providers, "
    "infer API keys, route providers or models, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Quality Scoring",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_quality_scoring_execution",
    "quality_score_computation",
    "quality_score_persistence",
    "quality_metric_collection_execution",
    "quality_policy_mutation",
    "quality_alert_emission",
    "health_metric_collection_execution",
    "health_monitor_mutation",
    "ranking_optimization_execution",
    "ranking_mutation",
    "retrieval_query_execution",
    "retrieval_configuration_mutation",
    "retrieval_reranking",
    "embedding_request_execution",
    "embedding_refresh_execution",
    "vector_indexing",
    "vector_upsert",
    "kb_storage_write",
    "documentation_fetch_execution",
    "kb_enrichment_execution",
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


class KnowledgeQualitySignal(BaseModel):
    """One advisory knowledge quality scoring signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeQualityScoringKind
    status: KnowledgeQualityScoringStatus
    confidence: KnowledgeQualityScoringConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    quality_axis: KnowledgeQualityScoringAxis
    knowledge_health_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    knowledge_health_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    quality_signal_summary: str = Field(min_length=1, max_length=360)
    quality_signal_score: int = Field(ge=0, le=100)
    health_alignment_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    quality_score: int = Field(ge=0, le=1_000)
    hitl_required_before_quality_scoring: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_quality_scoring_capability_implemented: Literal[True] = True
    knowledge_quality_scoring_metadata_implemented: Literal[True] = True
    knowledge_health_metadata_used: Literal[True] = True
    knowledge_quality_scoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    quality_score_persistence_implemented: Literal[False] = False
    quality_metric_collection_implemented: Literal[False] = False
    quality_policy_mutation_implemented: Literal[False] = False
    quality_alert_emission_implemented: Literal[False] = False
    health_metric_collection_implemented: Literal[False] = False
    health_monitor_mutation_implemented: Literal[False] = False
    ranking_optimization_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["knowledge_quality_entry.v1"] = (
        KNOWLEDGE_QUALITY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_quality_scoring::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_health_signal_count != len(self.knowledge_health_signal_ids):
            raise ValueError("knowledge_health_signal_count must match signals")
        if self.quality_score != _quality_score(
            quality_signal_score=self.quality_signal_score,
            health_alignment_score=self.health_alignment_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("quality_score must combine source scores")
        if self.status != _quality_status(self.quality_score):
            raise ValueError("status must match quality_score")
        if self.confidence != _quality_confidence(self.quality_score):
            raise ValueError("confidence must match quality_score")
        if not self.hitl_required_before_quality_scoring:
            raise ValueError("knowledge quality scoring requires HITL posture")
        return self


class KnowledgeQualityScoringPlan(BaseModel):
    """Bounded V6.3 advisory knowledge quality scoring plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_quality_scoring"] = "knowledge_quality_scoring"
    serialization_version: Literal["knowledge_quality_plan.v1"] = (
        KNOWLEDGE_QUALITY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_QUALITY_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_health_role: Literal["knowledge_health_monitoring"] = (
        "knowledge_health_monitoring"
    )
    knowledge_health_serialization_version: Literal["knowledge_health_plan.v1"]
    knowledge_health_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    knowledge_health_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeQualitySignal, ...] = Field(min_length=5, max_length=5)
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
    planned_quality_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    computed_quality_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_quality_score_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    collected_quality_metric_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_quality_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_quality_alert_ids: tuple[str, ...] = Field(
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
    highest_quality_score: int = Field(ge=0, le=1_000)
    overall_quality_score: int = Field(ge=0, le=1_000)
    overall_quality_posture: KnowledgeQualityScoringPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_quality_scoring_capability_implemented: Literal[True] = True
    knowledge_quality_scoring_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_health_metadata_used: Literal[True] = True
    knowledge_quality_scoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    quality_score_persistence_implemented: Literal[False] = False
    quality_metric_collection_implemented: Literal[False] = False
    quality_policy_mutation_implemented: Literal[False] = False
    quality_alert_emission_implemented: Literal[False] = False
    health_metric_collection_implemented: Literal[False] = False
    health_monitor_mutation_implemented: Literal[False] = False
    ranking_optimization_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
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
            if signal.hitl_required_before_quality_scoring
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_quality_score_ids:
            raise ValueError("planned_quality_score_ids must remain empty")
        if self.computed_quality_score_ids:
            raise ValueError("computed_quality_score_ids must remain empty")
        if self.persisted_quality_score_ids:
            raise ValueError("persisted_quality_score_ids must remain empty")
        if self.collected_quality_metric_ids:
            raise ValueError("collected_quality_metric_ids must remain empty")
        if self.mutated_quality_policy_ids:
            raise ValueError("mutated_quality_policy_ids must remain empty")
        if self.emitted_quality_alert_ids:
            raise ValueError("emitted_quality_alert_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_health_signal_count != len(self.knowledge_health_signal_ids):
            raise ValueError("knowledge_health_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 8 roadmap")
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
        if self.highest_quality_score != max(
            signal.quality_score for signal in self.signals
        ):
            raise ValueError("highest_quality_score must match signals")
        if self.overall_quality_score != _overall_quality_score(self.signals):
            raise ValueError("overall_quality_score must match signals")
        if self.overall_quality_posture != _overall_quality_posture(self.signals):
            raise ValueError("overall_quality_posture must match signals")
        declared_health_signals = set(self.knowledge_health_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_health_signal_ids).issubset(
                declared_health_signals
            ):
                raise ValueError("signal knowledge_health_signal_ids must be known")
        return self


def build_knowledge_quality_scoring(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_health: KnowledgeHealthMonitoringPlan | None = None,
) -> KnowledgeQualityScoringPlan:
    """Build V6.3 Task 8 knowledge quality scoring metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    health_plan = knowledge_health or build_knowledge_health_monitoring(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        health_plan=health_plan,
    )
    return KnowledgeQualityScoringPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=health_plan.checked_at,
        knowledge_health_serialization_version=health_plan.serialization_version,
        knowledge_health_signal_ids=health_plan.signal_ids,
        knowledge_health_signal_count=len(health_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=health_plan.source_count,
        domain_count=health_plan.domain_count,
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
            if signal.hitl_required_before_quality_scoring
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
            1 for signal in signals if signal.hitl_required_before_quality_scoring
        ),
        highest_quality_score=max(signal.quality_score for signal in signals),
        overall_quality_score=_overall_quality_score(signals),
        overall_quality_posture=_overall_quality_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_quality_signal_by_id(
    signal_id: str,
    plan: KnowledgeQualityScoringPlan | None = None,
) -> KnowledgeQualitySignal | None:
    """Return one knowledge quality signal without executing scoring."""

    source_plan = plan or build_knowledge_quality_scoring()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_quality_signals_for_status(
    status: KnowledgeQualityScoringStatus,
    plan: KnowledgeQualityScoringPlan | None = None,
) -> tuple[KnowledgeQualitySignal, ...]:
    """Return knowledge quality signals by advisory status."""

    source_plan = plan or build_knowledge_quality_scoring()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_quality_signals_for_confidence(
    confidence: KnowledgeQualityScoringConfidence,
    plan: KnowledgeQualityScoringPlan | None = None,
) -> tuple[KnowledgeQualitySignal, ...]:
    """Return knowledge quality signals by confidence band."""

    source_plan = plan or build_knowledge_quality_scoring()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    health_plan: KnowledgeHealthMonitoringPlan,
) -> tuple[KnowledgeQualitySignal, ...]:
    return (
        _signal(
            kind="knowledge_quality_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            health_signal_ids=health_plan.signal_ids,
            health_plan=health_plan,
            quality_signal_score=88,
            health_alignment_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_quality_accuracy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="accuracy_review",
            health_signal_ids=(
                "knowledge_health_monitoring::knowledge_health_inventory_review",
                "knowledge_health_monitoring::knowledge_health_reliability_review",
            ),
            health_plan=health_plan,
            quality_signal_score=78,
            health_alignment_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_quality_completeness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="completeness_review",
            health_signal_ids=(
                "knowledge_health_monitoring::knowledge_health_freshness_review",
                "knowledge_health_monitoring::knowledge_health_monitoring_readiness",
            ),
            health_plan=health_plan,
            quality_signal_score=70,
            health_alignment_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_quality_scoring_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="scoring_readiness",
            health_signal_ids=(
                "knowledge_health_monitoring::knowledge_health_monitoring_readiness",
                "knowledge_health_monitoring::knowledge_health_governance_gate",
            ),
            health_plan=health_plan,
            quality_signal_score=62,
            health_alignment_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_quality_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            health_signal_ids=health_plan.signal_ids,
            health_plan=health_plan,
            quality_signal_score=44,
            health_alignment_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeQualityScoringKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeQualityScoringAxis,
    health_signal_ids: tuple[str, ...],
    health_plan: KnowledgeHealthMonitoringPlan,
    quality_signal_score: int,
    health_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeQualitySignal:
    score = _quality_score(
        quality_signal_score=quality_signal_score,
        health_alignment_score=health_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeQualitySignal(
        signal_id=f"knowledge_quality_scoring::{kind}",
        signal_kind=kind,
        status=_quality_status(score),
        confidence=_quality_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        quality_axis=axis,
        knowledge_health_signal_ids=health_signal_ids,
        knowledge_health_signal_count=len(health_signal_ids),
        source_count=health_plan.source_count,
        domain_count=health_plan.domain_count,
        quality_signal_summary=_signal_summary(kind),
        quality_signal_score=quality_signal_score,
        health_alignment_score=health_alignment_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        quality_score=score,
        hitl_required_before_quality_scoring=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, health_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_health_signal_count:{len(health_signal_ids)}",
            f"source_count:{health_plan.source_count}",
            f"domain_count:{health_plan.domain_count}",
            f"quality_axis:{axis}",
            f"quality_signal_score:{quality_signal_score}",
            f"health_alignment_score:{health_alignment_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_quality_scoring:true",
        ),
    )


def _quality_score(
    *,
    quality_signal_score: int,
    health_alignment_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            quality_signal_score * 3
            + health_alignment_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _quality_status(score: int) -> KnowledgeQualityScoringStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _quality_confidence(score: int) -> KnowledgeQualityScoringConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_quality_score(signals: tuple[KnowledgeQualitySignal, ...]) -> int:
    base = sum(signal.quality_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_quality_posture(
    signals: tuple[KnowledgeQualitySignal, ...],
) -> KnowledgeQualityScoringPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeQualitySignal, ...],
    status: KnowledgeQualityScoringStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeQualitySignal, ...],
    *confidences: KnowledgeQualityScoringConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeQualitySignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_quality_scoring_metadata",
        "verify_knowledge_quality_roadmap_traceability",
        "review_knowledge_health_signals_before_any_quality_action",
        "require_hitl_before_quality_score_compute_persist_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeQualityScoringKind,
    axis: KnowledgeQualityScoringAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_quality_scoring",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeQualityScoringKind,
    health_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_health_signal_count:{len(health_signal_ids)}",
        "composes_knowledge_health_metadata",
        "keeps_knowledge_quality_scoring_execution_disabled",
        "requires_human_review_before_score_computation_or_persistence",
    )


def _signal_actions(kind: KnowledgeQualityScoringKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_quality_signal_metadata",
        "verify_knowledge_health_traceability",
        "keep_knowledge_quality_scoring_execution_disabled",
        "require_hitl_before_knowledge_quality_action",
    )
    if kind == "knowledge_quality_accuracy_review":
        return base_actions + ("review_quality_accuracy_metadata",)
    if kind == "knowledge_quality_completeness_review":
        return base_actions + ("review_quality_completeness_metadata",)
    if kind == "knowledge_quality_scoring_readiness":
        return base_actions + ("review_quality_scoring_readiness_metadata",)
    if kind == "knowledge_quality_governance_gate":
        return base_actions + ("confirm_manual_quality_governance_gate",)
    return base_actions + ("review_knowledge_quality_inventory_metadata",)


def _signal_summary(kind: KnowledgeQualityScoringKind) -> str:
    summaries: dict[KnowledgeQualityScoringKind, str] = {
        "knowledge_quality_inventory_review": (
            "Advisory knowledge quality inventory posture over health metadata "
            "without computing or persisting quality scores."
        ),
        "knowledge_quality_accuracy_review": (
            "Advisory posture for reviewing quality accuracy metadata before "
            "any scoring or metric collection is enabled."
        ),
        "knowledge_quality_completeness_review": (
            "Advisory posture for reviewing quality completeness metadata "
            "while keeping KB enrichment and storage writes disabled."
        ),
        "knowledge_quality_scoring_readiness": (
            "Advisory posture for reviewing scoring readiness without quality "
            "model execution, score persistence, or alerts."
        ),
        "knowledge_quality_governance_gate": (
            "Governed manual gate that keeps knowledge quality scoring "
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
