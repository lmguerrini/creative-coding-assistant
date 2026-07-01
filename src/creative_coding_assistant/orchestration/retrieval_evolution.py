"""V6.3 advisory retrieval evolution metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.embedding_refresh import (
    EmbeddingRefreshPlan,
    build_embedding_refresh,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

RetrievalEvolutionKind = Literal[
    "retrieval_inventory_review",
    "retrieval_filter_policy_review",
    "retrieval_query_contract_review",
    "retrieval_context_integration_review",
    "retrieval_governance_gate",
]
RetrievalEvolutionStatus = Literal["candidate", "review_required", "guarded"]
RetrievalEvolutionConfidence = Literal["low", "medium", "high", "guarded"]
RetrievalEvolutionPosture = Literal["candidate", "review_required", "guarded"]
RetrievalEvolutionAxis = Literal[
    "inventory_review",
    "filter_policy_review",
    "query_contract_review",
    "context_integration_review",
    "governance_gate",
]

RETRIEVAL_EVOLUTION_ENTRY_SERIALIZATION_VERSION = "retrieval_evolution_entry.v1"
RETRIEVAL_EVOLUTION_PLAN_SERIALIZATION_VERSION = "retrieval_evolution_plan.v1"
RETRIEVAL_EVOLUTION_AUTHORITY_BOUNDARY = (
    "V6.3 Retrieval Evolution exposes embedding refresh, retrieval inventory, "
    "filter policy, query contract, context integration, and governance "
    "posture as inspectable advisory metadata only; it does not execute "
    "retrieval queries, mutate retrieval filters, mutate retrieval "
    "configuration, mutate retrieval gateways, rerank retrieval results, "
    "change ranking, mutate context routing, mutate prompt context, request "
    "embeddings, refresh embeddings, index vectors, upsert vectors, write KB "
    "storage, fetch documentation, enrich the KB, update source records, "
    "provision providers, infer API keys, route providers or models, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "trigger retries or refinements, mutate prompts, modify generated output, "
    "or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Retrieval Evolution",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "retrieval_evolution_execution",
    "retrieval_query_execution",
    "retrieval_filter_mutation",
    "retrieval_configuration_mutation",
    "retrieval_gateway_mutation",
    "retrieval_reranking",
    "ranking_mutation",
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


class RetrievalEvolutionSignal(BaseModel):
    """One advisory retrieval evolution signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: RetrievalEvolutionKind
    status: RetrievalEvolutionStatus
    confidence: RetrievalEvolutionConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    evolution_axis: RetrievalEvolutionAxis
    embedding_refresh_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    embedding_refresh_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    evolution_signal_summary: str = Field(min_length=1, max_length=360)
    retrieval_inventory_score: int = Field(ge=0, le=100)
    query_contract_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    evolution_score: int = Field(ge=0, le=1_000)
    hitl_required_before_retrieval_evolution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    retrieval_evolution_capability_implemented: Literal[True] = True
    retrieval_evolution_metadata_implemented: Literal[True] = True
    embedding_refresh_metadata_used: Literal[True] = True
    retrieval_evolution_execution_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_gateway_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["retrieval_evolution_entry.v1"] = (
        RETRIEVAL_EVOLUTION_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"retrieval_evolution::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.embedding_refresh_signal_count != len(
            self.embedding_refresh_signal_ids
        ):
            raise ValueError("embedding_refresh_signal_count must match signals")
        if self.evolution_score != _evolution_score(
            retrieval_inventory_score=self.retrieval_inventory_score,
            query_contract_score=self.query_contract_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("evolution_score must combine source scores")
        if self.status != _evolution_status(self.evolution_score):
            raise ValueError("status must match evolution_score")
        if self.confidence != _evolution_confidence(self.evolution_score):
            raise ValueError("confidence must match evolution_score")
        if not self.hitl_required_before_retrieval_evolution:
            raise ValueError("retrieval evolution requires HITL posture")
        return self


class RetrievalEvolutionPlan(BaseModel):
    """Bounded V6.3 advisory retrieval evolution plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["retrieval_evolution"] = "retrieval_evolution"
    serialization_version: Literal["retrieval_evolution_plan.v1"] = (
        RETRIEVAL_EVOLUTION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=RETRIEVAL_EVOLUTION_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    embedding_refresh_role: Literal["embedding_refresh"] = "embedding_refresh"
    embedding_refresh_serialization_version: Literal["embedding_refresh_plan.v1"]
    embedding_refresh_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    embedding_refresh_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[RetrievalEvolutionSignal, ...] = Field(
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
    planned_retrieval_evolution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_retrieval_query_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_filter_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_config_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reranked_retrieval_result_ids: tuple[str, ...] = Field(
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
    highest_evolution_score: int = Field(ge=0, le=1_000)
    overall_evolution_score: int = Field(ge=0, le=1_000)
    overall_evolution_posture: RetrievalEvolutionPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    retrieval_evolution_capability_implemented: Literal[True] = True
    retrieval_evolution_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    embedding_refresh_metadata_used: Literal[True] = True
    retrieval_evolution_execution_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_filter_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_gateway_mutation_implemented: Literal[False] = False
    retrieval_reranking_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
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
            if signal.hitl_required_before_retrieval_evolution
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_retrieval_evolution_ids:
            raise ValueError("planned_retrieval_evolution_ids must remain empty")
        if self.executed_retrieval_query_ids:
            raise ValueError("executed_retrieval_query_ids must remain empty")
        if self.mutated_retrieval_filter_ids:
            raise ValueError("mutated_retrieval_filter_ids must remain empty")
        if self.mutated_retrieval_config_ids:
            raise ValueError("mutated_retrieval_config_ids must remain empty")
        if self.reranked_retrieval_result_ids:
            raise ValueError("reranked_retrieval_result_ids must remain empty")
        if self.mutated_context_route_ids:
            raise ValueError("mutated_context_route_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.embedding_refresh_signal_count != len(
            self.embedding_refresh_signal_ids
        ):
            raise ValueError("embedding_refresh_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 5 roadmap")
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
        if self.highest_evolution_score != max(
            signal.evolution_score for signal in self.signals
        ):
            raise ValueError("highest_evolution_score must match signals")
        if self.overall_evolution_score != _overall_evolution_score(self.signals):
            raise ValueError("overall_evolution_score must match signals")
        if self.overall_evolution_posture != _overall_evolution_posture(
            self.signals
        ):
            raise ValueError("overall_evolution_posture must match signals")
        declared_embedding_signals = set(self.embedding_refresh_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.embedding_refresh_signal_ids).issubset(
                declared_embedding_signals
            ):
                raise ValueError("signal embedding_refresh_signal_ids must be known")
        return self


def build_retrieval_evolution(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    embedding_refresh: EmbeddingRefreshPlan | None = None,
) -> RetrievalEvolutionPlan:
    """Build V6.3 Task 5 retrieval evolution metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    embedding_plan = embedding_refresh or build_embedding_refresh(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        embedding_plan=embedding_plan,
    )
    return RetrievalEvolutionPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=embedding_plan.checked_at,
        embedding_refresh_serialization_version=embedding_plan.serialization_version,
        embedding_refresh_signal_ids=embedding_plan.signal_ids,
        embedding_refresh_signal_count=len(embedding_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=embedding_plan.source_count,
        domain_count=embedding_plan.domain_count,
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
            if signal.hitl_required_before_retrieval_evolution
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
            if signal.hitl_required_before_retrieval_evolution
        ),
        highest_evolution_score=max(signal.evolution_score for signal in signals),
        overall_evolution_score=_overall_evolution_score(signals),
        overall_evolution_posture=_overall_evolution_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def retrieval_evolution_signal_by_id(
    signal_id: str,
    plan: RetrievalEvolutionPlan | None = None,
) -> RetrievalEvolutionSignal | None:
    """Return one retrieval evolution signal without executing retrieval."""

    source_plan = plan or build_retrieval_evolution()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def retrieval_evolution_signals_for_status(
    status: RetrievalEvolutionStatus,
    plan: RetrievalEvolutionPlan | None = None,
) -> tuple[RetrievalEvolutionSignal, ...]:
    """Return retrieval evolution signals by advisory status."""

    source_plan = plan or build_retrieval_evolution()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def retrieval_evolution_signals_for_confidence(
    confidence: RetrievalEvolutionConfidence,
    plan: RetrievalEvolutionPlan | None = None,
) -> tuple[RetrievalEvolutionSignal, ...]:
    """Return retrieval evolution signals by confidence band."""

    source_plan = plan or build_retrieval_evolution()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    embedding_plan: EmbeddingRefreshPlan,
) -> tuple[RetrievalEvolutionSignal, ...]:
    return (
        _signal(
            kind="retrieval_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            embedding_signal_ids=embedding_plan.signal_ids,
            embedding_plan=embedding_plan,
            retrieval_inventory_score=88,
            query_contract_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="retrieval_filter_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="filter_policy_review",
            embedding_signal_ids=(
                "embedding_refresh::embedding_inventory_review",
                "embedding_refresh::embedding_governance_gate",
            ),
            embedding_plan=embedding_plan,
            retrieval_inventory_score=78,
            query_contract_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="retrieval_query_contract_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="query_contract_review",
            embedding_signal_ids=(
                "embedding_refresh::embedding_staleness_review",
                "embedding_refresh::embedding_model_policy_review",
            ),
            embedding_plan=embedding_plan,
            retrieval_inventory_score=70,
            query_contract_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="retrieval_context_integration_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="context_integration_review",
            embedding_signal_ids=(
                "embedding_refresh::embedding_refresh_readiness",
                "embedding_refresh::embedding_staleness_review",
            ),
            embedding_plan=embedding_plan,
            retrieval_inventory_score=62,
            query_contract_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="retrieval_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            embedding_signal_ids=embedding_plan.signal_ids,
            embedding_plan=embedding_plan,
            retrieval_inventory_score=44,
            query_contract_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: RetrievalEvolutionKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: RetrievalEvolutionAxis,
    embedding_signal_ids: tuple[str, ...],
    embedding_plan: EmbeddingRefreshPlan,
    retrieval_inventory_score: int,
    query_contract_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> RetrievalEvolutionSignal:
    score = _evolution_score(
        retrieval_inventory_score=retrieval_inventory_score,
        query_contract_score=query_contract_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return RetrievalEvolutionSignal(
        signal_id=f"retrieval_evolution::{kind}",
        signal_kind=kind,
        status=_evolution_status(score),
        confidence=_evolution_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        evolution_axis=axis,
        embedding_refresh_signal_ids=embedding_signal_ids,
        embedding_refresh_signal_count=len(embedding_signal_ids),
        source_count=embedding_plan.source_count,
        domain_count=embedding_plan.domain_count,
        evolution_signal_summary=_signal_summary(kind),
        retrieval_inventory_score=retrieval_inventory_score,
        query_contract_score=query_contract_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        evolution_score=score,
        hitl_required_before_retrieval_evolution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, embedding_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"embedding_refresh_signal_count:{len(embedding_signal_ids)}",
            f"source_count:{embedding_plan.source_count}",
            f"domain_count:{embedding_plan.domain_count}",
            f"evolution_axis:{axis}",
            f"retrieval_inventory_score:{retrieval_inventory_score}",
            f"query_contract_score:{query_contract_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_retrieval_evolution:true",
        ),
    )


def _evolution_score(
    *,
    retrieval_inventory_score: int,
    query_contract_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            retrieval_inventory_score * 3
            + query_contract_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _evolution_status(score: int) -> RetrievalEvolutionStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _evolution_confidence(score: int) -> RetrievalEvolutionConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_evolution_score(signals: tuple[RetrievalEvolutionSignal, ...]) -> int:
    base = sum(signal.evolution_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_evolution_posture(
    signals: tuple[RetrievalEvolutionSignal, ...],
) -> RetrievalEvolutionPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[RetrievalEvolutionSignal, ...],
    status: RetrievalEvolutionStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[RetrievalEvolutionSignal, ...],
    *confidences: RetrievalEvolutionConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[RetrievalEvolutionSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_retrieval_evolution_metadata",
        "verify_retrieval_evolution_roadmap_traceability",
        "review_embedding_refresh_signals_before_any_retrieval_action",
        "require_hitl_before_retrieval_query_filter_rerank_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: RetrievalEvolutionKind,
    axis: RetrievalEvolutionAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "retrieval_evolution",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: RetrievalEvolutionKind,
    embedding_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"embedding_refresh_signal_count:{len(embedding_signal_ids)}",
        "composes_embedding_refresh_metadata",
        "keeps_retrieval_evolution_execution_disabled",
        "requires_human_review_before_retrieval_or_context_mutation",
    )


def _signal_actions(kind: RetrievalEvolutionKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_retrieval_evolution_signal_metadata",
        "verify_embedding_refresh_traceability",
        "keep_retrieval_evolution_execution_disabled",
        "require_hitl_before_retrieval_evolution_action",
    )
    if kind == "retrieval_filter_policy_review":
        return base_actions + ("review_retrieval_filter_policy_metadata",)
    if kind == "retrieval_query_contract_review":
        return base_actions + ("review_retrieval_query_contract_metadata",)
    if kind == "retrieval_context_integration_review":
        return base_actions + ("review_retrieval_context_integration_metadata",)
    if kind == "retrieval_governance_gate":
        return base_actions + ("confirm_manual_retrieval_governance_gate",)
    return base_actions + ("review_retrieval_inventory_metadata",)


def _signal_summary(kind: RetrievalEvolutionKind) -> str:
    summaries: dict[RetrievalEvolutionKind, str] = {
        "retrieval_inventory_review": (
            "Advisory retrieval inventory posture over embedding refresh "
            "metadata without executing retrieval queries."
        ),
        "retrieval_filter_policy_review": (
            "Advisory posture for reviewing retrieval filter policy metadata "
            "before any filter or configuration mutation."
        ),
        "retrieval_query_contract_review": (
            "Advisory posture for reviewing retrieval query contracts without "
            "query execution, rewriting, or provider routing."
        ),
        "retrieval_context_integration_review": (
            "Advisory posture for reviewing context integration while keeping "
            "context routing and prompt context mutation disabled."
        ),
        "retrieval_governance_gate": (
            "Governed manual gate that keeps retrieval evolution execution "
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
