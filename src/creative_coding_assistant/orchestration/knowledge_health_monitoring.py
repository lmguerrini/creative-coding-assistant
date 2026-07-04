"""V6.3 advisory knowledge health monitoring metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.ranking_optimization import (
    RankingOptimizationPlan,
    build_ranking_optimization,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeHealthMonitoringKind = Literal[
    "knowledge_health_inventory_review",
    "knowledge_health_freshness_review",
    "knowledge_health_reliability_review",
    "knowledge_health_monitoring_readiness",
    "knowledge_health_governance_gate",
]
KnowledgeHealthMonitoringStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeHealthMonitoringConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeHealthMonitoringPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeHealthMonitoringAxis = Literal[
    "inventory_review",
    "freshness_review",
    "reliability_review",
    "monitoring_readiness",
    "governance_gate",
]

KNOWLEDGE_HEALTH_ENTRY_SERIALIZATION_VERSION = "knowledge_health_entry.v1"
KNOWLEDGE_HEALTH_PLAN_SERIALIZATION_VERSION = "knowledge_health_plan.v1"
KNOWLEDGE_HEALTH_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Health Monitoring exposes ranking optimization, knowledge "
    "inventory, freshness, reliability, monitoring readiness, and governance "
    "posture as inspectable advisory metadata only; it does not execute health "
    "monitoring, collect live health metrics, mutate health monitors, emit "
    "alerts, execute ranking optimization, mutate ranking, execute retrieval "
    "queries, mutate retrieval configuration, rerank retrieval results, "
    "request embeddings, refresh embeddings, index vectors, upsert vectors, "
    "write KB storage, fetch documentation, enrich the KB, update source "
    "records, provision providers, infer API keys, route providers or models, "
    "execute providers, invoke agents, control workflows, mutate workflow "
    "graphs, trigger retries or refinements, mutate prompts, modify generated "
    "output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Health Monitoring",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_health_monitoring_execution",
    "health_metric_collection_execution",
    "health_monitor_mutation",
    "health_alert_emission",
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
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeHealthSignal(BaseModel):
    """One advisory knowledge health monitoring signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeHealthMonitoringKind
    status: KnowledgeHealthMonitoringStatus
    confidence: KnowledgeHealthMonitoringConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    health_axis: KnowledgeHealthMonitoringAxis
    ranking_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=5,
    )
    ranking_optimization_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    health_signal_summary: str = Field(min_length=1, max_length=360)
    health_signal_score: int = Field(ge=0, le=100)
    reliability_signal_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    health_score: int = Field(ge=0, le=1_000)
    hitl_required_before_health_monitoring: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_health_monitoring_capability_implemented: Literal[True] = True
    knowledge_health_monitoring_metadata_implemented: Literal[True] = True
    ranking_optimization_metadata_used: Literal[True] = True
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    health_metric_collection_implemented: Literal[False] = False
    health_monitor_mutation_implemented: Literal[False] = False
    health_alert_emission_implemented: Literal[False] = False
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
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["knowledge_health_entry.v1"] = (
        KNOWLEDGE_HEALTH_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_health_monitoring::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.ranking_optimization_signal_count != len(
            self.ranking_optimization_signal_ids
        ):
            raise ValueError("ranking_optimization_signal_count must match signals")
        if self.health_score != _health_score(
            health_signal_score=self.health_signal_score,
            reliability_signal_score=self.reliability_signal_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("health_score must combine source scores")
        if self.status != _health_status(self.health_score):
            raise ValueError("status must match health_score")
        if self.confidence != _health_confidence(self.health_score):
            raise ValueError("confidence must match health_score")
        if not self.hitl_required_before_health_monitoring:
            raise ValueError("knowledge health monitoring requires HITL posture")
        return self


class KnowledgeHealthMonitoringPlan(BaseModel):
    """Bounded V6.3 advisory knowledge health monitoring plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_health_monitoring"] = "knowledge_health_monitoring"
    serialization_version: Literal["knowledge_health_plan.v1"] = (
        KNOWLEDGE_HEALTH_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_HEALTH_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    ranking_optimization_role: Literal["ranking_optimization"] = "ranking_optimization"
    ranking_optimization_serialization_version: Literal["ranking_optimization_plan.v1"]
    ranking_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    ranking_optimization_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeHealthSignal, ...] = Field(min_length=5, max_length=5)
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
    planned_health_monitor_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    collected_health_metric_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_health_alert_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_health_monitor_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_ranking_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_retrieval_query_ids: tuple[str, ...] = Field(
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
    highest_health_score: int = Field(ge=0, le=1_000)
    overall_health_score: int = Field(ge=0, le=1_000)
    overall_health_posture: KnowledgeHealthMonitoringPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_health_monitoring_capability_implemented: Literal[True] = True
    knowledge_health_monitoring_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    ranking_optimization_metadata_used: Literal[True] = True
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    health_metric_collection_implemented: Literal[False] = False
    health_monitor_mutation_implemented: Literal[False] = False
    health_alert_emission_implemented: Literal[False] = False
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
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
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
            if signal.hitl_required_before_health_monitoring
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_health_monitor_ids:
            raise ValueError("planned_health_monitor_ids must remain empty")
        if self.collected_health_metric_ids:
            raise ValueError("collected_health_metric_ids must remain empty")
        if self.emitted_health_alert_ids:
            raise ValueError("emitted_health_alert_ids must remain empty")
        if self.mutated_health_monitor_ids:
            raise ValueError("mutated_health_monitor_ids must remain empty")
        if self.mutated_ranking_profile_ids:
            raise ValueError("mutated_ranking_profile_ids must remain empty")
        if self.executed_retrieval_query_ids:
            raise ValueError("executed_retrieval_query_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.ranking_optimization_signal_count != len(
            self.ranking_optimization_signal_ids
        ):
            raise ValueError("ranking_optimization_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 7 roadmap")
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
        if self.highest_health_score != max(
            signal.health_score for signal in self.signals
        ):
            raise ValueError("highest_health_score must match signals")
        if self.overall_health_score != _overall_health_score(self.signals):
            raise ValueError("overall_health_score must match signals")
        if self.overall_health_posture != _overall_health_posture(self.signals):
            raise ValueError("overall_health_posture must match signals")
        declared_ranking_signals = set(self.ranking_optimization_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.ranking_optimization_signal_ids).issubset(
                declared_ranking_signals
            ):
                raise ValueError("signal ranking_optimization_signal_ids must be known")
        return self


def build_knowledge_health_monitoring(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    ranking_optimization: RankingOptimizationPlan | None = None,
) -> KnowledgeHealthMonitoringPlan:
    """Build V6.3 Task 7 knowledge health monitoring metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    ranking_plan = ranking_optimization or build_ranking_optimization(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        ranking_plan=ranking_plan,
    )
    return KnowledgeHealthMonitoringPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=ranking_plan.checked_at,
        ranking_optimization_serialization_version=ranking_plan.serialization_version,
        ranking_optimization_signal_ids=ranking_plan.signal_ids,
        ranking_optimization_signal_count=len(ranking_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=ranking_plan.source_count,
        domain_count=ranking_plan.domain_count,
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
            if signal.hitl_required_before_health_monitoring
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
            1 for signal in signals if signal.hitl_required_before_health_monitoring
        ),
        highest_health_score=max(signal.health_score for signal in signals),
        overall_health_score=_overall_health_score(signals),
        overall_health_posture=_overall_health_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_health_signal_by_id(
    signal_id: str,
    plan: KnowledgeHealthMonitoringPlan | None = None,
) -> KnowledgeHealthSignal | None:
    """Return one knowledge health signal without executing monitoring."""

    source_plan = plan or build_knowledge_health_monitoring()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_health_signals_for_status(
    status: KnowledgeHealthMonitoringStatus,
    plan: KnowledgeHealthMonitoringPlan | None = None,
) -> tuple[KnowledgeHealthSignal, ...]:
    """Return knowledge health signals by advisory status."""

    source_plan = plan or build_knowledge_health_monitoring()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_health_signals_for_confidence(
    confidence: KnowledgeHealthMonitoringConfidence,
    plan: KnowledgeHealthMonitoringPlan | None = None,
) -> tuple[KnowledgeHealthSignal, ...]:
    """Return knowledge health signals by confidence band."""

    source_plan = plan or build_knowledge_health_monitoring()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    ranking_plan: RankingOptimizationPlan,
) -> tuple[KnowledgeHealthSignal, ...]:
    return (
        _signal(
            kind="knowledge_health_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            ranking_signal_ids=ranking_plan.signal_ids,
            ranking_plan=ranking_plan,
            health_signal_score=88,
            reliability_signal_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_health_freshness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="freshness_review",
            ranking_signal_ids=(
                "ranking_optimization::ranking_inventory_review",
                "ranking_optimization::ranking_evaluation_readiness",
            ),
            ranking_plan=ranking_plan,
            health_signal_score=78,
            reliability_signal_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_health_reliability_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="reliability_review",
            ranking_signal_ids=(
                "ranking_optimization::ranking_signal_weight_review",
                "ranking_optimization::ranking_bias_review",
            ),
            ranking_plan=ranking_plan,
            health_signal_score=70,
            reliability_signal_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_health_monitoring_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="monitoring_readiness",
            ranking_signal_ids=(
                "ranking_optimization::ranking_evaluation_readiness",
                "ranking_optimization::ranking_governance_gate",
            ),
            ranking_plan=ranking_plan,
            health_signal_score=62,
            reliability_signal_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_health_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            ranking_signal_ids=ranking_plan.signal_ids,
            ranking_plan=ranking_plan,
            health_signal_score=44,
            reliability_signal_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeHealthMonitoringKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeHealthMonitoringAxis,
    ranking_signal_ids: tuple[str, ...],
    ranking_plan: RankingOptimizationPlan,
    health_signal_score: int,
    reliability_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeHealthSignal:
    score = _health_score(
        health_signal_score=health_signal_score,
        reliability_signal_score=reliability_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeHealthSignal(
        signal_id=f"knowledge_health_monitoring::{kind}",
        signal_kind=kind,
        status=_health_status(score),
        confidence=_health_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        health_axis=axis,
        ranking_optimization_signal_ids=ranking_signal_ids,
        ranking_optimization_signal_count=len(ranking_signal_ids),
        source_count=ranking_plan.source_count,
        domain_count=ranking_plan.domain_count,
        health_signal_summary=_signal_summary(kind),
        health_signal_score=health_signal_score,
        reliability_signal_score=reliability_signal_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        health_score=score,
        hitl_required_before_health_monitoring=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, ranking_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"ranking_optimization_signal_count:{len(ranking_signal_ids)}",
            f"source_count:{ranking_plan.source_count}",
            f"domain_count:{ranking_plan.domain_count}",
            f"health_axis:{axis}",
            f"health_signal_score:{health_signal_score}",
            f"reliability_signal_score:{reliability_signal_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_health_monitoring:true",
        ),
    )


def _health_score(
    *,
    health_signal_score: int,
    reliability_signal_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            health_signal_score * 3
            + reliability_signal_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _health_status(score: int) -> KnowledgeHealthMonitoringStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _health_confidence(score: int) -> KnowledgeHealthMonitoringConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_health_score(signals: tuple[KnowledgeHealthSignal, ...]) -> int:
    base = sum(signal.health_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_health_posture(
    signals: tuple[KnowledgeHealthSignal, ...],
) -> KnowledgeHealthMonitoringPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeHealthSignal, ...],
    status: KnowledgeHealthMonitoringStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeHealthSignal, ...],
    *confidences: KnowledgeHealthMonitoringConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeHealthSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_health_monitoring_metadata",
        "verify_knowledge_health_roadmap_traceability",
        "review_ranking_optimization_signals_before_any_health_action",
        "require_hitl_before_metric_collection_alerting_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeHealthMonitoringKind,
    axis: KnowledgeHealthMonitoringAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_health_monitoring",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeHealthMonitoringKind,
    ranking_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"ranking_optimization_signal_count:{len(ranking_signal_ids)}",
        "composes_ranking_optimization_metadata",
        "keeps_knowledge_health_monitoring_execution_disabled",
        "requires_human_review_before_metrics_alerts_or_kb_mutation",
    )


def _signal_actions(kind: KnowledgeHealthMonitoringKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_health_signal_metadata",
        "verify_ranking_optimization_traceability",
        "keep_knowledge_health_monitoring_execution_disabled",
        "require_hitl_before_knowledge_health_action",
    )
    if kind == "knowledge_health_freshness_review":
        return base_actions + ("review_health_freshness_metadata",)
    if kind == "knowledge_health_reliability_review":
        return base_actions + ("review_health_reliability_metadata",)
    if kind == "knowledge_health_monitoring_readiness":
        return base_actions + ("review_health_monitoring_readiness_metadata",)
    if kind == "knowledge_health_governance_gate":
        return base_actions + ("confirm_manual_health_governance_gate",)
    return base_actions + ("review_knowledge_health_inventory_metadata",)


def _signal_summary(kind: KnowledgeHealthMonitoringKind) -> str:
    summaries: dict[KnowledgeHealthMonitoringKind, str] = {
        "knowledge_health_inventory_review": (
            "Advisory knowledge health inventory posture over ranking "
            "optimization metadata without collecting live metrics."
        ),
        "knowledge_health_freshness_review": (
            "Advisory posture for reviewing knowledge freshness metadata "
            "before any monitor or alert emission is enabled."
        ),
        "knowledge_health_reliability_review": (
            "Advisory posture for reviewing reliability metadata while "
            "keeping ranking and retrieval mutation disabled."
        ),
        "knowledge_health_monitoring_readiness": (
            "Advisory posture for reviewing monitoring readiness without "
            "metric collection, alerting, or KB storage writes."
        ),
        "knowledge_health_governance_gate": (
            "Governed manual gate that keeps knowledge health monitoring "
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
