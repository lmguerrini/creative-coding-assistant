"""V6.3 advisory ranking optimization metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.retrieval_evolution import (
    RetrievalEvolutionPlan,
    build_retrieval_evolution,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

RankingOptimizationKind = Literal[
    "ranking_inventory_review",
    "ranking_signal_weight_review",
    "ranking_bias_review",
    "ranking_evaluation_readiness",
    "ranking_governance_gate",
]
RankingOptimizationStatus = Literal["candidate", "review_required", "guarded"]
RankingOptimizationConfidence = Literal["low", "medium", "high", "guarded"]
RankingOptimizationPosture = Literal["candidate", "review_required", "guarded"]
RankingOptimizationAxis = Literal[
    "inventory_review",
    "signal_weight_review",
    "bias_review",
    "evaluation_readiness",
    "governance_gate",
]

RANKING_OPTIMIZATION_ENTRY_SERIALIZATION_VERSION = "ranking_optimization_entry.v1"
RANKING_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = "ranking_optimization_plan.v1"
RANKING_OPTIMIZATION_AUTHORITY_BOUNDARY = (
    "V6.3 Ranking Optimization exposes retrieval evolution, ranking inventory, "
    "signal weight, bias, evaluation readiness, and governance posture as "
    "inspectable advisory metadata only; it does not execute ranking "
    "optimization, mutate ranking profiles, rerank retrieval results, execute "
    "retrieval queries, mutate retrieval filters, mutate retrieval "
    "configuration, mutate retrieval gateways, mutate context routing, mutate "
    "prompt context, request embeddings, refresh embeddings, index vectors, "
    "upsert vectors, write KB storage, fetch documentation, enrich the KB, "
    "update source records, provision providers, infer API keys, route "
    "providers or models, execute providers, invoke agents, control workflows, "
    "mutate workflow graphs, trigger retries or refinements, mutate prompts, "
    "modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Ranking Optimization",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "ranking_optimization_execution",
    "ranking_profile_mutation",
    "ranking_mutation",
    "retrieval_reranking",
    "retrieval_query_execution",
    "retrieval_filter_mutation",
    "retrieval_configuration_mutation",
    "retrieval_gateway_mutation",
    "context_routing_mutation",
    "prompt_context_mutation",
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


class RankingOptimizationSignal(BaseModel):
    """One advisory ranking optimization signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: RankingOptimizationKind
    status: RankingOptimizationStatus
    confidence: RankingOptimizationConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    optimization_axis: RankingOptimizationAxis
    retrieval_evolution_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    retrieval_evolution_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    optimization_signal_summary: str = Field(min_length=1, max_length=360)
    ranking_signal_score: int = Field(ge=0, le=100)
    ranking_policy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    optimization_score: int = Field(ge=0, le=1_000)
    hitl_required_before_ranking_optimization: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    ranking_optimization_capability_implemented: Literal[True] = True
    ranking_optimization_metadata_implemented: Literal[True] = True
    retrieval_evolution_metadata_used: Literal[True] = True
    ranking_optimization_execution_implemented: Literal[False] = False
    ranking_profile_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_gateway_mutation_implemented: Literal[False] = False
    context_routing_mutation_implemented: Literal[False] = False
    prompt_context_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["ranking_optimization_entry.v1"] = (
        RANKING_OPTIMIZATION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"ranking_optimization::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.retrieval_evolution_signal_count != len(
            self.retrieval_evolution_signal_ids
        ):
            raise ValueError("retrieval_evolution_signal_count must match signals")
        if self.optimization_score != _optimization_score(
            ranking_signal_score=self.ranking_signal_score,
            ranking_policy_score=self.ranking_policy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("optimization_score must combine source scores")
        if self.status != _optimization_status(self.optimization_score):
            raise ValueError("status must match optimization_score")
        if self.confidence != _optimization_confidence(self.optimization_score):
            raise ValueError("confidence must match optimization_score")
        if not self.hitl_required_before_ranking_optimization:
            raise ValueError("ranking optimization requires HITL posture")
        return self


class RankingOptimizationPlan(BaseModel):
    """Bounded V6.3 advisory ranking optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["ranking_optimization"] = "ranking_optimization"
    serialization_version: Literal["ranking_optimization_plan.v1"] = (
        RANKING_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RANKING_OPTIMIZATION_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    retrieval_evolution_role: Literal["retrieval_evolution"] = (
        "retrieval_evolution"
    )
    retrieval_evolution_serialization_version: Literal["retrieval_evolution_plan.v1"]
    retrieval_evolution_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    retrieval_evolution_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[RankingOptimizationSignal, ...] = Field(
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
    planned_ranking_optimization_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_ranking_profile_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reranked_retrieval_result_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_retrieval_query_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_config_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_context_route_ids: tuple[str, ...] = Field(
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
    highest_optimization_score: int = Field(ge=0, le=1_000)
    overall_optimization_score: int = Field(ge=0, le=1_000)
    overall_optimization_posture: RankingOptimizationPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    ranking_optimization_capability_implemented: Literal[True] = True
    ranking_optimization_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    retrieval_evolution_metadata_used: Literal[True] = True
    ranking_optimization_execution_implemented: Literal[False] = False
    ranking_profile_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_gateway_mutation_implemented: Literal[False] = False
    context_routing_mutation_implemented: Literal[False] = False
    prompt_context_mutation_implemented: Literal[False] = False
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
            if signal.hitl_required_before_ranking_optimization
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_ranking_optimization_ids:
            raise ValueError("planned_ranking_optimization_ids must remain empty")
        if self.mutated_ranking_profile_ids:
            raise ValueError("mutated_ranking_profile_ids must remain empty")
        if self.reranked_retrieval_result_ids:
            raise ValueError("reranked_retrieval_result_ids must remain empty")
        if self.executed_retrieval_query_ids:
            raise ValueError("executed_retrieval_query_ids must remain empty")
        if self.mutated_retrieval_config_ids:
            raise ValueError("mutated_retrieval_config_ids must remain empty")
        if self.mutated_context_route_ids:
            raise ValueError("mutated_context_route_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.retrieval_evolution_signal_count != len(
            self.retrieval_evolution_signal_ids
        ):
            raise ValueError("retrieval_evolution_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 6 roadmap")
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
        if self.highest_optimization_score != max(
            signal.optimization_score for signal in self.signals
        ):
            raise ValueError("highest_optimization_score must match signals")
        if self.overall_optimization_score != _overall_optimization_score(
            self.signals
        ):
            raise ValueError("overall_optimization_score must match signals")
        if self.overall_optimization_posture != _overall_optimization_posture(
            self.signals
        ):
            raise ValueError("overall_optimization_posture must match signals")
        declared_retrieval_signals = set(self.retrieval_evolution_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.retrieval_evolution_signal_ids).issubset(
                declared_retrieval_signals
            ):
                raise ValueError("signal retrieval_evolution_signal_ids must be known")
        return self


def build_ranking_optimization(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    retrieval_evolution: RetrievalEvolutionPlan | None = None,
) -> RankingOptimizationPlan:
    """Build V6.3 Task 6 ranking optimization metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    retrieval_plan = retrieval_evolution or build_retrieval_evolution(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        retrieval_plan=retrieval_plan,
    )
    return RankingOptimizationPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=retrieval_plan.checked_at,
        retrieval_evolution_serialization_version=retrieval_plan.serialization_version,
        retrieval_evolution_signal_ids=retrieval_plan.signal_ids,
        retrieval_evolution_signal_count=len(retrieval_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=retrieval_plan.source_count,
        domain_count=retrieval_plan.domain_count,
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
            if signal.hitl_required_before_ranking_optimization
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
            if signal.hitl_required_before_ranking_optimization
        ),
        highest_optimization_score=max(
            signal.optimization_score for signal in signals
        ),
        overall_optimization_score=_overall_optimization_score(signals),
        overall_optimization_posture=_overall_optimization_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def ranking_optimization_signal_by_id(
    signal_id: str,
    plan: RankingOptimizationPlan | None = None,
) -> RankingOptimizationSignal | None:
    """Return one ranking optimization signal without executing ranking work."""

    source_plan = plan or build_ranking_optimization()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def ranking_optimization_signals_for_status(
    status: RankingOptimizationStatus,
    plan: RankingOptimizationPlan | None = None,
) -> tuple[RankingOptimizationSignal, ...]:
    """Return ranking optimization signals by advisory status."""

    source_plan = plan or build_ranking_optimization()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def ranking_optimization_signals_for_confidence(
    confidence: RankingOptimizationConfidence,
    plan: RankingOptimizationPlan | None = None,
) -> tuple[RankingOptimizationSignal, ...]:
    """Return ranking optimization signals by confidence band."""

    source_plan = plan or build_ranking_optimization()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    retrieval_plan: RetrievalEvolutionPlan,
) -> tuple[RankingOptimizationSignal, ...]:
    return (
        _signal(
            kind="ranking_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            retrieval_signal_ids=retrieval_plan.signal_ids,
            retrieval_plan=retrieval_plan,
            ranking_signal_score=88,
            ranking_policy_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="ranking_signal_weight_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="signal_weight_review",
            retrieval_signal_ids=(
                "retrieval_evolution::retrieval_inventory_review",
                "retrieval_evolution::retrieval_query_contract_review",
            ),
            retrieval_plan=retrieval_plan,
            ranking_signal_score=78,
            ranking_policy_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="ranking_bias_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="bias_review",
            retrieval_signal_ids=(
                "retrieval_evolution::retrieval_filter_policy_review",
                "retrieval_evolution::retrieval_governance_gate",
            ),
            retrieval_plan=retrieval_plan,
            ranking_signal_score=70,
            ranking_policy_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="ranking_evaluation_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="evaluation_readiness",
            retrieval_signal_ids=(
                "retrieval_evolution::retrieval_context_integration_review",
                "retrieval_evolution::retrieval_query_contract_review",
            ),
            retrieval_plan=retrieval_plan,
            ranking_signal_score=62,
            ranking_policy_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="ranking_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            retrieval_signal_ids=retrieval_plan.signal_ids,
            retrieval_plan=retrieval_plan,
            ranking_signal_score=44,
            ranking_policy_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: RankingOptimizationKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: RankingOptimizationAxis,
    retrieval_signal_ids: tuple[str, ...],
    retrieval_plan: RetrievalEvolutionPlan,
    ranking_signal_score: int,
    ranking_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> RankingOptimizationSignal:
    score = _optimization_score(
        ranking_signal_score=ranking_signal_score,
        ranking_policy_score=ranking_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return RankingOptimizationSignal(
        signal_id=f"ranking_optimization::{kind}",
        signal_kind=kind,
        status=_optimization_status(score),
        confidence=_optimization_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        optimization_axis=axis,
        retrieval_evolution_signal_ids=retrieval_signal_ids,
        retrieval_evolution_signal_count=len(retrieval_signal_ids),
        source_count=retrieval_plan.source_count,
        domain_count=retrieval_plan.domain_count,
        optimization_signal_summary=_signal_summary(kind),
        ranking_signal_score=ranking_signal_score,
        ranking_policy_score=ranking_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        optimization_score=score,
        hitl_required_before_ranking_optimization=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, retrieval_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"retrieval_evolution_signal_count:{len(retrieval_signal_ids)}",
            f"source_count:{retrieval_plan.source_count}",
            f"domain_count:{retrieval_plan.domain_count}",
            f"optimization_axis:{axis}",
            f"ranking_signal_score:{ranking_signal_score}",
            f"ranking_policy_score:{ranking_policy_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_ranking_optimization:true",
        ),
    )


def _optimization_score(
    *,
    ranking_signal_score: int,
    ranking_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            ranking_signal_score * 3
            + ranking_policy_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _optimization_status(score: int) -> RankingOptimizationStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _optimization_confidence(score: int) -> RankingOptimizationConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_optimization_score(
    signals: tuple[RankingOptimizationSignal, ...],
) -> int:
    base = sum(signal.optimization_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_optimization_posture(
    signals: tuple[RankingOptimizationSignal, ...],
) -> RankingOptimizationPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[RankingOptimizationSignal, ...],
    status: RankingOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[RankingOptimizationSignal, ...],
    *confidences: RankingOptimizationConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[RankingOptimizationSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_ranking_optimization_metadata",
        "verify_ranking_optimization_roadmap_traceability",
        "review_retrieval_evolution_signals_before_any_ranking_action",
        "require_hitl_before_ranking_profile_rerank_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: RankingOptimizationKind,
    axis: RankingOptimizationAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "ranking_optimization",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: RankingOptimizationKind,
    retrieval_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"retrieval_evolution_signal_count:{len(retrieval_signal_ids)}",
        "composes_retrieval_evolution_metadata",
        "keeps_ranking_optimization_execution_disabled",
        "requires_human_review_before_ranking_or_retrieval_mutation",
    )


def _signal_actions(kind: RankingOptimizationKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_ranking_optimization_signal_metadata",
        "verify_retrieval_evolution_traceability",
        "keep_ranking_optimization_execution_disabled",
        "require_hitl_before_ranking_optimization_action",
    )
    if kind == "ranking_signal_weight_review":
        return base_actions + ("review_ranking_signal_weight_metadata",)
    if kind == "ranking_bias_review":
        return base_actions + ("review_ranking_bias_metadata",)
    if kind == "ranking_evaluation_readiness":
        return base_actions + ("review_ranking_evaluation_readiness_metadata",)
    if kind == "ranking_governance_gate":
        return base_actions + ("confirm_manual_ranking_governance_gate",)
    return base_actions + ("review_ranking_inventory_metadata",)


def _signal_summary(kind: RankingOptimizationKind) -> str:
    summaries: dict[RankingOptimizationKind, str] = {
        "ranking_inventory_review": (
            "Advisory ranking inventory posture over retrieval evolution "
            "metadata without optimizing ranking profiles."
        ),
        "ranking_signal_weight_review": (
            "Advisory posture for reviewing ranking signal weight metadata "
            "before any ranker or scoring mutation."
        ),
        "ranking_bias_review": (
            "Advisory posture for reviewing ranking bias metadata without "
            "reranking retrieval results."
        ),
        "ranking_evaluation_readiness": (
            "Advisory posture for reviewing ranking evaluation readiness "
            "while keeping retrieval and context mutation disabled."
        ),
        "ranking_governance_gate": (
            "Governed manual gate that keeps ranking optimization execution "
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
