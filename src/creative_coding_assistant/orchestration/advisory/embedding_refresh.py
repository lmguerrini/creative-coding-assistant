"""V6.3 advisory embedding refresh metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.documentation_intelligence import (
    DocumentationIntelligencePlan,
    build_documentation_intelligence,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

EmbeddingRefreshKind = Literal[
    "embedding_inventory_review",
    "embedding_staleness_review",
    "embedding_model_policy_review",
    "embedding_refresh_readiness",
    "embedding_governance_gate",
]
EmbeddingRefreshStatus = Literal["candidate", "review_required", "guarded"]
EmbeddingRefreshConfidence = Literal["low", "medium", "high", "guarded"]
EmbeddingRefreshPosture = Literal["candidate", "review_required", "guarded"]
EmbeddingRefreshAxis = Literal[
    "inventory_review",
    "staleness_review",
    "model_policy_review",
    "refresh_readiness",
    "governance_gate",
]

EMBEDDING_REFRESH_ENTRY_SERIALIZATION_VERSION = "embedding_refresh_entry.v1"
EMBEDDING_REFRESH_PLAN_SERIALIZATION_VERSION = "embedding_refresh_plan.v1"
EMBEDDING_REFRESH_AUTHORITY_BOUNDARY = (
    "V6.3 Embedding Refresh exposes documentation intelligence, inventory, "
    "staleness, model policy, readiness, and governance posture as "
    "inspectable advisory metadata only; it does not request embeddings, "
    "refresh embeddings, select embedding models, route embedding providers, "
    "index vectors, upsert vectors, delete vectors, write embedding caches, "
    "write KB storage, mutate retrieval configuration, execute retrieval, "
    "change ranking, fetch documentation, enrich the KB, update source "
    "records, provision providers, infer API keys, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Embedding Refresh",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "embedding_refresh_execution",
    "embedding_request_execution",
    "embedding_model_selection",
    "embedding_provider_routing",
    "vector_indexing",
    "vector_upsert",
    "vector_deletion",
    "embedding_cache_write",
    "kb_storage_write",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "ranking_mutation",
    "documentation_fetch_execution",
    "kb_enrichment_execution",
    "source_record_update",
    "provider_provisioning",
    "api_key_inference",
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


class EmbeddingRefreshSignal(BaseModel):
    """One advisory embedding refresh signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: EmbeddingRefreshKind
    status: EmbeddingRefreshStatus
    confidence: EmbeddingRefreshConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    refresh_axis: EmbeddingRefreshAxis
    documentation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    documentation_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    refresh_signal_summary: str = Field(min_length=1, max_length=360)
    embedding_inventory_score: int = Field(ge=0, le=100)
    staleness_review_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    refresh_score: int = Field(ge=0, le=1_000)
    hitl_required_before_embedding_refresh: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    embedding_refresh_capability_implemented: Literal[True] = True
    embedding_refresh_metadata_implemented: Literal[True] = True
    documentation_intelligence_metadata_used: Literal[True] = True
    embedding_refresh_execution_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_model_selection_implemented: Literal[False] = False
    embedding_provider_routing_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    vector_deletion_implemented: Literal[False] = False
    embedding_cache_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
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
    serialization_version: Literal["embedding_refresh_entry.v1"] = (
        EMBEDDING_REFRESH_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"embedding_refresh::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.documentation_signal_count != len(self.documentation_signal_ids):
            raise ValueError("documentation_signal_count must match signals")
        if self.refresh_score != _refresh_score(
            embedding_inventory_score=self.embedding_inventory_score,
            staleness_review_score=self.staleness_review_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("refresh_score must combine source scores")
        if self.status != _refresh_status(self.refresh_score):
            raise ValueError("status must match refresh_score")
        if self.confidence != _refresh_confidence(self.refresh_score):
            raise ValueError("confidence must match refresh_score")
        if not self.hitl_required_before_embedding_refresh:
            raise ValueError("embedding refresh requires HITL posture")
        return self


class EmbeddingRefreshPlan(BaseModel):
    """Bounded V6.3 advisory embedding refresh plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["embedding_refresh"] = "embedding_refresh"
    serialization_version: Literal["embedding_refresh_plan.v1"] = (
        EMBEDDING_REFRESH_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EMBEDDING_REFRESH_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    documentation_intelligence_role: Literal["documentation_intelligence"] = (
        "documentation_intelligence"
    )
    documentation_intelligence_serialization_version: Literal[
        "documentation_intelligence_plan.v1"
    ]
    documentation_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    documentation_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[EmbeddingRefreshSignal, ...] = Field(
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
    planned_embedding_refresh_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    requested_embedding_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    refreshed_embedding_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    indexed_vector_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    upserted_vector_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_source_ids: tuple[str, ...] = Field(
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
    highest_refresh_score: int = Field(ge=0, le=1_000)
    overall_refresh_score: int = Field(ge=0, le=1_000)
    overall_refresh_posture: EmbeddingRefreshPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    embedding_refresh_capability_implemented: Literal[True] = True
    embedding_refresh_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    documentation_intelligence_metadata_used: Literal[True] = True
    embedding_refresh_execution_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_model_selection_implemented: Literal[False] = False
    embedding_provider_routing_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    vector_deletion_implemented: Literal[False] = False
    embedding_cache_write_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    api_key_inference_implemented: Literal[False] = False
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
            if signal.hitl_required_before_embedding_refresh
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_embedding_refresh_ids:
            raise ValueError("planned_embedding_refresh_ids must remain empty")
        if self.requested_embedding_ids:
            raise ValueError("requested_embedding_ids must remain empty")
        if self.refreshed_embedding_ids:
            raise ValueError("refreshed_embedding_ids must remain empty")
        if self.indexed_vector_ids:
            raise ValueError("indexed_vector_ids must remain empty")
        if self.upserted_vector_ids:
            raise ValueError("upserted_vector_ids must remain empty")
        if self.mutated_retrieval_source_ids:
            raise ValueError("mutated_retrieval_source_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.documentation_signal_count != len(self.documentation_signal_ids):
            raise ValueError("documentation_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 4 roadmap")
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
        if self.highest_refresh_score != max(
            signal.refresh_score for signal in self.signals
        ):
            raise ValueError("highest_refresh_score must match signals")
        if self.overall_refresh_score != _overall_refresh_score(self.signals):
            raise ValueError("overall_refresh_score must match signals")
        if self.overall_refresh_posture != _overall_refresh_posture(self.signals):
            raise ValueError("overall_refresh_posture must match signals")
        declared_documentation_signals = set(self.documentation_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.documentation_signal_ids).issubset(
                declared_documentation_signals
            ):
                raise ValueError("signal documentation_signal_ids must be known")
        return self


def build_embedding_refresh(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    documentation_intelligence: DocumentationIntelligencePlan | None = None,
) -> EmbeddingRefreshPlan:
    """Build V6.3 Task 4 embedding refresh metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    documentation_plan = documentation_intelligence or build_documentation_intelligence(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        documentation_plan=documentation_plan,
    )
    return EmbeddingRefreshPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=documentation_plan.checked_at,
        documentation_intelligence_serialization_version=(
            documentation_plan.serialization_version
        ),
        documentation_signal_ids=documentation_plan.signal_ids,
        documentation_signal_count=len(documentation_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=documentation_plan.source_count,
        domain_count=documentation_plan.domain_count,
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
            if signal.hitl_required_before_embedding_refresh
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
            1 for signal in signals if signal.hitl_required_before_embedding_refresh
        ),
        highest_refresh_score=max(signal.refresh_score for signal in signals),
        overall_refresh_score=_overall_refresh_score(signals),
        overall_refresh_posture=_overall_refresh_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def embedding_refresh_signal_by_id(
    signal_id: str,
    plan: EmbeddingRefreshPlan | None = None,
) -> EmbeddingRefreshSignal | None:
    """Return one embedding refresh signal without executing refresh work."""

    source_plan = plan or build_embedding_refresh()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def embedding_refresh_signals_for_status(
    status: EmbeddingRefreshStatus,
    plan: EmbeddingRefreshPlan | None = None,
) -> tuple[EmbeddingRefreshSignal, ...]:
    """Return embedding refresh signals by advisory status."""

    source_plan = plan or build_embedding_refresh()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def embedding_refresh_signals_for_confidence(
    confidence: EmbeddingRefreshConfidence,
    plan: EmbeddingRefreshPlan | None = None,
) -> tuple[EmbeddingRefreshSignal, ...]:
    """Return embedding refresh signals by confidence band."""

    source_plan = plan or build_embedding_refresh()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    documentation_plan: DocumentationIntelligencePlan,
) -> tuple[EmbeddingRefreshSignal, ...]:
    return (
        _signal(
            kind="embedding_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            documentation_signal_ids=documentation_plan.signal_ids,
            documentation_plan=documentation_plan,
            embedding_inventory_score=88,
            staleness_review_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="embedding_staleness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="staleness_review",
            documentation_signal_ids=(
                "documentation_intelligence::documentation_freshness_review",
                "documentation_intelligence::documentation_source_map",
            ),
            documentation_plan=documentation_plan,
            embedding_inventory_score=78,
            staleness_review_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="embedding_model_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="model_policy_review",
            documentation_signal_ids=(
                "documentation_intelligence::documentation_structure_review",
                "documentation_intelligence::documentation_governance_gate",
            ),
            documentation_plan=documentation_plan,
            embedding_inventory_score=70,
            staleness_review_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="embedding_refresh_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="refresh_readiness",
            documentation_signal_ids=(
                "documentation_intelligence::documentation_gap_review",
                "documentation_intelligence::documentation_freshness_review",
            ),
            documentation_plan=documentation_plan,
            embedding_inventory_score=62,
            staleness_review_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="embedding_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            documentation_signal_ids=documentation_plan.signal_ids,
            documentation_plan=documentation_plan,
            embedding_inventory_score=44,
            staleness_review_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: EmbeddingRefreshKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: EmbeddingRefreshAxis,
    documentation_signal_ids: tuple[str, ...],
    documentation_plan: DocumentationIntelligencePlan,
    embedding_inventory_score: int,
    staleness_review_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> EmbeddingRefreshSignal:
    score = _refresh_score(
        embedding_inventory_score=embedding_inventory_score,
        staleness_review_score=staleness_review_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return EmbeddingRefreshSignal(
        signal_id=f"embedding_refresh::{kind}",
        signal_kind=kind,
        status=_refresh_status(score),
        confidence=_refresh_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        refresh_axis=axis,
        documentation_signal_ids=documentation_signal_ids,
        documentation_signal_count=len(documentation_signal_ids),
        source_count=documentation_plan.source_count,
        domain_count=documentation_plan.domain_count,
        refresh_signal_summary=_signal_summary(kind),
        embedding_inventory_score=embedding_inventory_score,
        staleness_review_score=staleness_review_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        refresh_score=score,
        hitl_required_before_embedding_refresh=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, documentation_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"documentation_signal_count:{len(documentation_signal_ids)}",
            f"source_count:{documentation_plan.source_count}",
            f"domain_count:{documentation_plan.domain_count}",
            f"refresh_axis:{axis}",
            f"embedding_inventory_score:{embedding_inventory_score}",
            f"staleness_review_score:{staleness_review_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_embedding_refresh:true",
        ),
    )


def _refresh_score(
    *,
    embedding_inventory_score: int,
    staleness_review_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            embedding_inventory_score * 3
            + staleness_review_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _refresh_status(score: int) -> EmbeddingRefreshStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _refresh_confidence(score: int) -> EmbeddingRefreshConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_refresh_score(signals: tuple[EmbeddingRefreshSignal, ...]) -> int:
    base = sum(signal.refresh_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_refresh_posture(
    signals: tuple[EmbeddingRefreshSignal, ...],
) -> EmbeddingRefreshPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[EmbeddingRefreshSignal, ...],
    status: EmbeddingRefreshStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[EmbeddingRefreshSignal, ...],
    *confidences: EmbeddingRefreshConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[EmbeddingRefreshSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_embedding_refresh_metadata",
        "verify_embedding_refresh_roadmap_traceability",
        "review_documentation_signals_before_any_embedding_action",
        "require_hitl_before_embedding_request_indexing_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: EmbeddingRefreshKind,
    axis: EmbeddingRefreshAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "embedding_refresh",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: EmbeddingRefreshKind,
    documentation_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"documentation_signal_count:{len(documentation_signal_ids)}",
        "composes_documentation_intelligence_metadata",
        "keeps_embedding_refresh_execution_disabled",
        "requires_human_review_before_embedding_or_vector_mutation",
    )


def _signal_actions(kind: EmbeddingRefreshKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_embedding_refresh_signal_metadata",
        "verify_documentation_intelligence_traceability",
        "keep_embedding_refresh_execution_disabled",
        "require_hitl_before_embedding_refresh_action",
    )
    if kind == "embedding_staleness_review":
        return base_actions + ("review_embedding_staleness_metadata",)
    if kind == "embedding_model_policy_review":
        return base_actions + ("review_embedding_model_policy_metadata",)
    if kind == "embedding_refresh_readiness":
        return base_actions + ("review_embedding_refresh_readiness_metadata",)
    if kind == "embedding_governance_gate":
        return base_actions + ("confirm_manual_embedding_governance_gate",)
    return base_actions + ("review_embedding_inventory_metadata",)


def _signal_summary(kind: EmbeddingRefreshKind) -> str:
    summaries: dict[EmbeddingRefreshKind, str] = {
        "embedding_inventory_review": (
            "Advisory embedding inventory posture over documentation "
            "intelligence metadata without requesting embeddings."
        ),
        "embedding_staleness_review": (
            "Advisory posture for reviewing embedding staleness metadata "
            "before any refresh job is enabled."
        ),
        "embedding_model_policy_review": (
            "Advisory posture for reviewing embedding model policy metadata "
            "without selecting models or routing providers."
        ),
        "embedding_refresh_readiness": (
            "Advisory posture for reviewing refresh readiness while keeping "
            "vector indexing and KB writes disabled."
        ),
        "embedding_governance_gate": (
            "Governed manual gate that keeps embedding refresh execution "
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
