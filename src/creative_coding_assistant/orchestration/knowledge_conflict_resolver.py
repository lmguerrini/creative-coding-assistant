"""V6.3 advisory knowledge conflict resolver metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.knowledge_gap_detection import (
    KnowledgeGapDetectionPlan,
    build_knowledge_gap_detection,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeConflictResolverKind = Literal[
    "knowledge_conflict_inventory_review",
    "knowledge_conflict_source_disagreement_review",
    "knowledge_conflict_resolution_policy_review",
    "knowledge_conflict_resolver_readiness",
    "knowledge_conflict_governance_gate",
]
KnowledgeConflictResolverStatus = Literal["candidate", "review_required", "guarded"]
KnowledgeConflictResolverConfidence = Literal["low", "medium", "high", "guarded"]
KnowledgeConflictResolverPosture = Literal["candidate", "review_required", "guarded"]
KnowledgeConflictResolverAxis = Literal[
    "inventory_review",
    "source_disagreement_review",
    "resolution_policy_review",
    "resolver_readiness",
    "governance_gate",
]

KNOWLEDGE_CONFLICT_ENTRY_SERIALIZATION_VERSION = "knowledge_conflict_entry.v1"
KNOWLEDGE_CONFLICT_PLAN_SERIALIZATION_VERSION = "knowledge_conflict_plan.v1"
KNOWLEDGE_CONFLICT_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Conflict Resolver exposes knowledge gap, conflict "
    "inventory, source disagreement, resolution policy, resolver readiness, "
    "and governance posture as inspectable advisory metadata only; it does not "
    "execute conflict resolution, detect conflicts, resolve conflicts, "
    "arbitrate sources, write conflict records, mutate source precedence, scan "
    "for gaps, remediate gaps, enrich the KB, write KB storage, compute quality "
    "scores, execute retrieval queries, mutate retrieval configuration, mutate "
    "ranking, request embeddings, refresh embeddings, index vectors, upsert "
    "vectors, fetch documentation, update source records, provision providers, "
    "infer API keys, route providers or models, execute providers, invoke "
    "agents, control workflows, mutate workflow graphs, trigger retries or "
    "refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Knowledge Conflict Resolver",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "knowledge_conflict_resolver_execution",
    "conflict_detection_execution",
    "conflict_resolution_execution",
    "conflict_arbitration_execution",
    "conflict_record_write",
    "source_precedence_mutation",
    "gap_scan_execution",
    "gap_remediation_execution",
    "kb_enrichment_execution",
    "kb_storage_write",
    "quality_score_computation",
    "retrieval_query_execution",
    "retrieval_configuration_mutation",
    "ranking_mutation",
    "embedding_request_execution",
    "embedding_refresh_execution",
    "vector_indexing",
    "vector_upsert",
    "documentation_fetch_execution",
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


class KnowledgeConflictSignal(BaseModel):
    """One advisory knowledge conflict resolver signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: KnowledgeConflictResolverKind
    status: KnowledgeConflictResolverStatus
    confidence: KnowledgeConflictResolverConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    conflict_axis: KnowledgeConflictResolverAxis
    knowledge_gap_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    knowledge_gap_signal_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    conflict_signal_summary: str = Field(min_length=1, max_length=360)
    conflict_signal_score: int = Field(ge=0, le=100)
    resolution_policy_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    conflict_score: int = Field(ge=0, le=1_000)
    hitl_required_before_conflict_resolution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_conflict_resolver_capability_implemented: Literal[True] = True
    knowledge_conflict_resolver_metadata_implemented: Literal[True] = True
    knowledge_gap_metadata_used: Literal[True] = True
    knowledge_conflict_resolver_execution_implemented: Literal[False] = False
    conflict_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    conflict_record_write_implemented: Literal[False] = False
    source_precedence_mutation_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
    serialization_version: Literal["knowledge_conflict_entry.v1"] = (
        KNOWLEDGE_CONFLICT_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"knowledge_conflict_resolver::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.knowledge_gap_signal_count != len(self.knowledge_gap_signal_ids):
            raise ValueError("knowledge_gap_signal_count must match signals")
        if self.conflict_score != _conflict_score(
            conflict_signal_score=self.conflict_signal_score,
            resolution_policy_score=self.resolution_policy_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("conflict_score must combine source scores")
        if self.status != _conflict_status(self.conflict_score):
            raise ValueError("status must match conflict_score")
        if self.confidence != _conflict_confidence(self.conflict_score):
            raise ValueError("confidence must match conflict_score")
        if not self.hitl_required_before_conflict_resolution:
            raise ValueError("knowledge conflict resolution requires HITL posture")
        return self


class KnowledgeConflictResolverPlan(BaseModel):
    """Bounded V6.3 advisory knowledge conflict resolver plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_conflict_resolver"] = "knowledge_conflict_resolver"
    serialization_version: Literal["knowledge_conflict_plan.v1"] = (
        KNOWLEDGE_CONFLICT_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_CONFLICT_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    knowledge_gap_role: Literal["knowledge_gap_detection"] = (
        "knowledge_gap_detection"
    )
    knowledge_gap_serialization_version: Literal["knowledge_gap_plan.v1"]
    knowledge_gap_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    knowledge_gap_signal_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[KnowledgeConflictSignal, ...] = Field(min_length=5, max_length=5)
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
    planned_conflict_resolution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    detected_conflict_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    resolved_conflict_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    arbitrated_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_source_precedence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_conflict_record_ids: tuple[str, ...] = Field(
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
    highest_conflict_score: int = Field(ge=0, le=1_000)
    overall_conflict_score: int = Field(ge=0, le=1_000)
    overall_conflict_posture: KnowledgeConflictResolverPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    knowledge_conflict_resolver_capability_implemented: Literal[True] = True
    knowledge_conflict_resolver_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    knowledge_gap_metadata_used: Literal[True] = True
    knowledge_conflict_resolver_execution_implemented: Literal[False] = False
    conflict_detection_execution_implemented: Literal[False] = False
    conflict_resolution_execution_implemented: Literal[False] = False
    conflict_arbitration_execution_implemented: Literal[False] = False
    conflict_record_write_implemented: Literal[False] = False
    source_precedence_mutation_implemented: Literal[False] = False
    gap_scan_execution_implemented: Literal[False] = False
    gap_remediation_execution_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    retrieval_query_execution_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_indexing_implemented: Literal[False] = False
    vector_upsert_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
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
            if signal.hitl_required_before_conflict_resolution
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.planned_conflict_resolution_ids:
            raise ValueError("planned_conflict_resolution_ids must remain empty")
        if self.detected_conflict_ids:
            raise ValueError("detected_conflict_ids must remain empty")
        if self.resolved_conflict_ids:
            raise ValueError("resolved_conflict_ids must remain empty")
        if self.arbitrated_source_ids:
            raise ValueError("arbitrated_source_ids must remain empty")
        if self.mutated_source_precedence_ids:
            raise ValueError("mutated_source_precedence_ids must remain empty")
        if self.written_conflict_record_ids:
            raise ValueError("written_conflict_record_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.knowledge_gap_signal_count != len(self.knowledge_gap_signal_ids):
            raise ValueError("knowledge_gap_signal_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 10 roadmap")
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
        if self.highest_conflict_score != max(
            signal.conflict_score for signal in self.signals
        ):
            raise ValueError("highest_conflict_score must match signals")
        if self.overall_conflict_score != _overall_conflict_score(self.signals):
            raise ValueError("overall_conflict_score must match signals")
        if self.overall_conflict_posture != _overall_conflict_posture(self.signals):
            raise ValueError("overall_conflict_posture must match signals")
        declared_gap_signals = set(self.knowledge_gap_signal_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if signal.source_count != self.source_count:
                raise ValueError("signal source_count must match plan")
            if signal.domain_count != self.domain_count:
                raise ValueError("signal domain_count must match plan")
            if not set(signal.knowledge_gap_signal_ids).issubset(
                declared_gap_signals
            ):
                raise ValueError("signal knowledge_gap_signal_ids must be known")
        return self


def build_knowledge_conflict_resolver(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_gap: KnowledgeGapDetectionPlan | None = None,
) -> KnowledgeConflictResolverPlan:
    """Build V6.3 Task 10 knowledge conflict resolver metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    gap_plan = knowledge_gap or build_knowledge_gap_detection(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        gap_plan=gap_plan,
    )
    return KnowledgeConflictResolverPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=gap_plan.checked_at,
        knowledge_gap_serialization_version=gap_plan.serialization_version,
        knowledge_gap_signal_ids=gap_plan.signal_ids,
        knowledge_gap_signal_count=len(gap_plan.signal_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=gap_plan.source_count,
        domain_count=gap_plan.domain_count,
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
            if signal.hitl_required_before_conflict_resolution
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
            if signal.hitl_required_before_conflict_resolution
        ),
        highest_conflict_score=max(signal.conflict_score for signal in signals),
        overall_conflict_score=_overall_conflict_score(signals),
        overall_conflict_posture=_overall_conflict_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def knowledge_conflict_signal_by_id(
    signal_id: str,
    plan: KnowledgeConflictResolverPlan | None = None,
) -> KnowledgeConflictSignal | None:
    """Return one knowledge conflict signal without resolving conflicts."""

    source_plan = plan or build_knowledge_conflict_resolver()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def knowledge_conflict_signals_for_status(
    status: KnowledgeConflictResolverStatus,
    plan: KnowledgeConflictResolverPlan | None = None,
) -> tuple[KnowledgeConflictSignal, ...]:
    """Return knowledge conflict signals by advisory status."""

    source_plan = plan or build_knowledge_conflict_resolver()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def knowledge_conflict_signals_for_confidence(
    confidence: KnowledgeConflictResolverConfidence,
    plan: KnowledgeConflictResolverPlan | None = None,
) -> tuple[KnowledgeConflictSignal, ...]:
    """Return knowledge conflict signals by confidence band."""

    source_plan = plan or build_knowledge_conflict_resolver()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    gap_plan: KnowledgeGapDetectionPlan,
) -> tuple[KnowledgeConflictSignal, ...]:
    return (
        _signal(
            kind="knowledge_conflict_inventory_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="inventory_review",
            gap_signal_ids=gap_plan.signal_ids,
            gap_plan=gap_plan,
            conflict_signal_score=88,
            resolution_policy_score=84,
            governance_alignment_score=86,
            mutation_risk_score=46,
            governance_weight=125,
        ),
        _signal(
            kind="knowledge_conflict_source_disagreement_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_disagreement_review",
            gap_signal_ids=(
                "knowledge_gap_detection::knowledge_gap_inventory_review",
                "knowledge_gap_detection::knowledge_gap_coverage_review",
            ),
            gap_plan=gap_plan,
            conflict_signal_score=78,
            resolution_policy_score=76,
            governance_alignment_score=82,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="knowledge_conflict_resolution_policy_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="resolution_policy_review",
            gap_signal_ids=(
                "knowledge_gap_detection::knowledge_gap_priority_review",
                "knowledge_gap_detection::knowledge_gap_governance_gate",
            ),
            gap_plan=gap_plan,
            conflict_signal_score=70,
            resolution_policy_score=72,
            governance_alignment_score=84,
            mutation_risk_score=38,
            governance_weight=100,
        ),
        _signal(
            kind="knowledge_conflict_resolver_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="resolver_readiness",
            gap_signal_ids=(
                "knowledge_gap_detection::knowledge_gap_detection_readiness",
                "knowledge_gap_detection::knowledge_gap_governance_gate",
            ),
            gap_plan=gap_plan,
            conflict_signal_score=62,
            resolution_policy_score=64,
            governance_alignment_score=78,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="knowledge_conflict_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            gap_signal_ids=gap_plan.signal_ids,
            gap_plan=gap_plan,
            conflict_signal_score=44,
            resolution_policy_score=46,
            governance_alignment_score=92,
            mutation_risk_score=16,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: KnowledgeConflictResolverKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeConflictResolverAxis,
    gap_signal_ids: tuple[str, ...],
    gap_plan: KnowledgeGapDetectionPlan,
    conflict_signal_score: int,
    resolution_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> KnowledgeConflictSignal:
    score = _conflict_score(
        conflict_signal_score=conflict_signal_score,
        resolution_policy_score=resolution_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeConflictSignal(
        signal_id=f"knowledge_conflict_resolver::{kind}",
        signal_kind=kind,
        status=_conflict_status(score),
        confidence=_conflict_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        conflict_axis=axis,
        knowledge_gap_signal_ids=gap_signal_ids,
        knowledge_gap_signal_count=len(gap_signal_ids),
        source_count=gap_plan.source_count,
        domain_count=gap_plan.domain_count,
        conflict_signal_summary=_signal_summary(kind),
        conflict_signal_score=conflict_signal_score,
        resolution_policy_score=resolution_policy_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        conflict_score=score,
        hitl_required_before_conflict_resolution=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, gap_signal_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"knowledge_gap_signal_count:{len(gap_signal_ids)}",
            f"source_count:{gap_plan.source_count}",
            f"domain_count:{gap_plan.domain_count}",
            f"conflict_axis:{axis}",
            f"conflict_signal_score:{conflict_signal_score}",
            f"resolution_policy_score:{resolution_policy_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_conflict_resolution:true",
        ),
    )


def _conflict_score(
    *,
    conflict_signal_score: int,
    resolution_policy_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            conflict_signal_score * 3
            + resolution_policy_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _conflict_status(score: int) -> KnowledgeConflictResolverStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _conflict_confidence(score: int) -> KnowledgeConflictResolverConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_conflict_score(signals: tuple[KnowledgeConflictSignal, ...]) -> int:
    base = sum(signal.conflict_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_conflict_posture(
    signals: tuple[KnowledgeConflictSignal, ...],
) -> KnowledgeConflictResolverPosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[KnowledgeConflictSignal, ...],
    status: KnowledgeConflictResolverStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[KnowledgeConflictSignal, ...],
    *confidences: KnowledgeConflictResolverConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(signals: tuple[KnowledgeConflictSignal, ...]) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_knowledge_conflict_resolver_metadata",
        "verify_knowledge_conflict_roadmap_traceability",
        "review_gap_signals_before_any_conflict_action",
        "require_hitl_before_conflict_detection_resolution_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeConflictResolverKind,
    axis: KnowledgeConflictResolverAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_conflict_resolver",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeConflictResolverKind,
    gap_signal_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"knowledge_gap_signal_count:{len(gap_signal_ids)}",
        "composes_knowledge_gap_metadata",
        "keeps_conflict_resolution_execution_disabled",
        "requires_human_review_before_conflict_arbitration_or_write",
    )


def _signal_actions(kind: KnowledgeConflictResolverKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_conflict_signal_metadata",
        "verify_knowledge_gap_traceability",
        "keep_knowledge_conflict_resolution_disabled",
        "require_hitl_before_knowledge_conflict_action",
    )
    if kind == "knowledge_conflict_source_disagreement_review":
        return base_actions + ("review_source_disagreement_metadata",)
    if kind == "knowledge_conflict_resolution_policy_review":
        return base_actions + ("review_conflict_resolution_policy_metadata",)
    if kind == "knowledge_conflict_resolver_readiness":
        return base_actions + ("review_conflict_resolver_readiness_metadata",)
    if kind == "knowledge_conflict_governance_gate":
        return base_actions + ("confirm_manual_conflict_governance_gate",)
    return base_actions + ("review_knowledge_conflict_inventory_metadata",)


def _signal_summary(kind: KnowledgeConflictResolverKind) -> str:
    summaries: dict[KnowledgeConflictResolverKind, str] = {
        "knowledge_conflict_inventory_review": (
            "Advisory knowledge conflict inventory posture over gap metadata "
            "without executing conflict detection."
        ),
        "knowledge_conflict_source_disagreement_review": (
            "Advisory posture for reviewing source disagreement metadata "
            "before arbitration or source precedence changes."
        ),
        "knowledge_conflict_resolution_policy_review": (
            "Advisory posture for reviewing conflict resolution policy "
            "metadata while conflict writes remain disabled."
        ),
        "knowledge_conflict_resolver_readiness": (
            "Advisory posture for reviewing resolver readiness without "
            "resolution, KB writes, or retrieval mutation."
        ),
        "knowledge_conflict_governance_gate": (
            "Governed manual gate that keeps knowledge conflict resolution "
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
