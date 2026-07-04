"""V6.3 advisory documentation intelligence metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.automatic_kb_updates import (
    AutomaticKBUpdatePlan,
    build_automatic_kb_updates,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

DocumentationIntelligenceKind = Literal[
    "documentation_source_map",
    "documentation_structure_review",
    "documentation_freshness_review",
    "documentation_gap_review",
    "documentation_governance_gate",
]
DocumentationIntelligenceStatus = Literal["candidate", "review_required", "guarded"]
DocumentationIntelligenceConfidence = Literal["low", "medium", "high", "guarded"]
DocumentationIntelligencePosture = Literal["candidate", "review_required", "guarded"]
DocumentationIntelligenceAxis = Literal[
    "source_mapping",
    "structure_review",
    "freshness_review",
    "gap_review",
    "governance_gate",
]

DOCUMENTATION_INTELLIGENCE_ENTRY_SERIALIZATION_VERSION = (
    "documentation_intelligence_entry.v1"
)
DOCUMENTATION_INTELLIGENCE_PLAN_SERIALIZATION_VERSION = (
    "documentation_intelligence_plan.v1"
)
DOCUMENTATION_INTELLIGENCE_AUTHORITY_BOUNDARY = (
    "V6.3 Documentation Intelligence exposes approved documentation source, "
    "structure, freshness, gap, and governance posture as inspectable advisory "
    "metadata only; it does not fetch documentation, crawl websites, parse "
    "documents, summarize documentation, rewrite documentation, generate "
    "documentation, classify live content, enrich the KB, update source "
    "records, refresh embeddings, mutate retrieval configuration, execute "
    "retrieval, change ranking, write storage, provision providers, infer API "
    "keys, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, trigger retries or refinements, mutate prompts, modify "
    "generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = ("Documentation Intelligence",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "documentation_fetch_execution",
    "documentation_crawl_execution",
    "documentation_parse_execution",
    "documentation_summarization_execution",
    "documentation_rewrite_execution",
    "documentation_generation_execution",
    "live_content_classification",
    "kb_enrichment_execution",
    "source_record_update",
    "embedding_refresh_execution",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "ranking_mutation",
    "kb_storage_write",
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


class DocumentationIntelligenceSignal(BaseModel):
    """One advisory documentation intelligence signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: DocumentationIntelligenceKind
    status: DocumentationIntelligenceStatus
    confidence: DocumentationIntelligenceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    intelligence_axis: DocumentationIntelligenceAxis
    source_update_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_update_candidate_count: int = Field(ge=1, le=5)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    documentation_signal_summary: str = Field(min_length=1, max_length=360)
    documentation_mapping_score: int = Field(ge=0, le=100)
    source_traceability_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    intelligence_score: int = Field(ge=0, le=1_000)
    hitl_required_before_documentation_action: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    documentation_intelligence_implemented: Literal[True] = True
    documentation_intelligence_metadata_implemented: Literal[True] = True
    automatic_kb_update_metadata_used: Literal[True] = True
    documentation_action_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    documentation_parse_execution_implemented: Literal[False] = False
    documentation_summarization_implemented: Literal[False] = False
    documentation_rewrite_implemented: Literal[False] = False
    documentation_generation_implemented: Literal[False] = False
    live_content_classification_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    embedding_refresh_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
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
    serialization_version: Literal["documentation_intelligence_entry.v1"] = (
        DOCUMENTATION_INTELLIGENCE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"documentation_intelligence::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if self.source_update_candidate_count != len(self.source_update_candidate_ids):
            raise ValueError("source_update_candidate_count must match candidates")
        if self.intelligence_score != _intelligence_score(
            documentation_mapping_score=self.documentation_mapping_score,
            source_traceability_score=self.source_traceability_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("intelligence_score must combine source scores")
        if self.status != _intelligence_status(self.intelligence_score):
            raise ValueError("status must match intelligence_score")
        if self.confidence != _intelligence_confidence(self.intelligence_score):
            raise ValueError("confidence must match intelligence_score")
        if not self.hitl_required_before_documentation_action:
            raise ValueError("documentation actions require HITL posture")
        return self


class DocumentationIntelligencePlan(BaseModel):
    """Bounded V6.3 advisory documentation intelligence plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["documentation_intelligence"] = "documentation_intelligence"
    serialization_version: Literal["documentation_intelligence_plan.v1"] = (
        DOCUMENTATION_INTELLIGENCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=DOCUMENTATION_INTELLIGENCE_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_update_role: Literal["automatic_kb_updates"] = "automatic_kb_updates"
    source_update_serialization_version: Literal["automatic_kb_update_plan.v1"]
    source_update_candidate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_update_candidate_count: int = Field(ge=5, le=5)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    signals: tuple[DocumentationIntelligenceSignal, ...] = Field(
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
    executed_documentation_action_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_documentation_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    parsed_documentation_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    enriched_kb_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_retrieval_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_signal_count: int = Field(ge=0, le=5)
    review_required_signal_count: int = Field(ge=0, le=5)
    guarded_signal_count: int = Field(ge=0, le=5)
    high_confidence_signal_count: int = Field(ge=0, le=5)
    hitl_required_signal_count: int = Field(ge=0, le=5)
    highest_intelligence_score: int = Field(ge=0, le=1_000)
    overall_intelligence_score: int = Field(ge=0, le=1_000)
    overall_intelligence_posture: DocumentationIntelligencePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    documentation_intelligence_implemented: Literal[True] = True
    documentation_intelligence_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    automatic_kb_update_metadata_used: Literal[True] = True
    documentation_action_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    documentation_parse_execution_implemented: Literal[False] = False
    documentation_summarization_implemented: Literal[False] = False
    documentation_rewrite_implemented: Literal[False] = False
    documentation_generation_implemented: Literal[False] = False
    kb_enrichment_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    embedding_refresh_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
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
        if self.guarded_signal_ids != _signal_ids_for_status(
            self.signals,
            "guarded",
        ):
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
            if signal.hitl_required_before_documentation_action
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.executed_documentation_action_ids:
            raise ValueError("executed_documentation_action_ids must remain empty")
        if self.fetched_documentation_source_ids:
            raise ValueError("fetched_documentation_source_ids must remain empty")
        if self.parsed_documentation_source_ids:
            raise ValueError("parsed_documentation_source_ids must remain empty")
        if self.enriched_kb_record_ids:
            raise ValueError("enriched_kb_record_ids must remain empty")
        if self.mutated_retrieval_source_ids:
            raise ValueError("mutated_retrieval_source_ids must remain empty")
        if self.source_update_candidate_count != len(self.source_update_candidate_ids):
            raise ValueError("source_update_candidate_count must match source")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 3 roadmap")
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
        if self.highest_intelligence_score != max(
            signal.intelligence_score for signal in self.signals
        ):
            raise ValueError("highest_intelligence_score must match signals")
        if self.overall_intelligence_score != _overall_intelligence_score(self.signals):
            raise ValueError("overall_intelligence_score must match signals")
        if self.overall_intelligence_posture != _overall_intelligence_posture(
            self.signals
        ):
            raise ValueError("overall_intelligence_posture must match signals")
        declared_source_candidates = set(self.source_update_candidate_ids)
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
            if not set(signal.source_update_candidate_ids).issubset(
                declared_source_candidates
            ):
                raise ValueError("signal source_update_candidate_ids must be known")
        return self


def build_documentation_intelligence(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    automatic_kb_updates: AutomaticKBUpdatePlan | None = None,
) -> DocumentationIntelligencePlan:
    """Build V6.3 Task 3 documentation intelligence metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    update_plan = automatic_kb_updates or build_automatic_kb_updates(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    signals = _signals(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        update_plan=update_plan,
    )
    return DocumentationIntelligencePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=update_plan.checked_at,
        source_update_serialization_version=update_plan.serialization_version,
        source_update_candidate_ids=update_plan.candidate_ids,
        source_update_candidate_count=len(update_plan.candidate_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_count=update_plan.source_count,
        domain_count=update_plan.domain_count,
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
            if signal.hitl_required_before_documentation_action
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
            1 for signal in signals if signal.hitl_required_before_documentation_action
        ),
        highest_intelligence_score=max(signal.intelligence_score for signal in signals),
        overall_intelligence_score=_overall_intelligence_score(signals),
        overall_intelligence_posture=_overall_intelligence_posture(signals),
        advisory_actions=_plan_actions(signals),
    )


def documentation_intelligence_signal_by_id(
    signal_id: str,
    plan: DocumentationIntelligencePlan | None = None,
) -> DocumentationIntelligenceSignal | None:
    """Return one documentation intelligence signal without executing actions."""

    source_plan = plan or build_documentation_intelligence()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def documentation_intelligence_signals_for_status(
    status: DocumentationIntelligenceStatus,
    plan: DocumentationIntelligencePlan | None = None,
) -> tuple[DocumentationIntelligenceSignal, ...]:
    """Return documentation intelligence signals by advisory status."""

    source_plan = plan or build_documentation_intelligence()
    return tuple(signal for signal in source_plan.signals if signal.status == status)


def documentation_intelligence_signals_for_confidence(
    confidence: DocumentationIntelligenceConfidence,
    plan: DocumentationIntelligencePlan | None = None,
) -> tuple[DocumentationIntelligenceSignal, ...]:
    """Return documentation intelligence signals by confidence band."""

    source_plan = plan or build_documentation_intelligence()
    return tuple(
        signal for signal in source_plan.signals if signal.confidence == confidence
    )


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    update_plan: AutomaticKBUpdatePlan,
) -> tuple[DocumentationIntelligenceSignal, ...]:
    return (
        _signal(
            kind="documentation_source_map",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="source_mapping",
            source_candidate_ids=update_plan.candidate_ids,
            update_plan=update_plan,
            documentation_mapping_score=88,
            source_traceability_score=86,
            governance_alignment_score=84,
            mutation_risk_score=48,
            governance_weight=130,
        ),
        _signal(
            kind="documentation_structure_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="structure_review",
            source_candidate_ids=(
                "automatic_kb_updates::approved_source_registry_monitor",
                "automatic_kb_updates::domain_coverage_review",
            ),
            update_plan=update_plan,
            documentation_mapping_score=78,
            source_traceability_score=76,
            governance_alignment_score=80,
            mutation_risk_score=44,
            governance_weight=110,
        ),
        _signal(
            kind="documentation_freshness_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="freshness_review",
            source_candidate_ids=(
                "automatic_kb_updates::freshness_policy_monitor",
                "automatic_kb_updates::sync_metadata_review",
            ),
            update_plan=update_plan,
            documentation_mapping_score=72,
            source_traceability_score=74,
            governance_alignment_score=78,
            mutation_risk_score=42,
            governance_weight=105,
        ),
        _signal(
            kind="documentation_gap_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="gap_review",
            source_candidate_ids=(
                "automatic_kb_updates::domain_coverage_review",
                "automatic_kb_updates::manual_execution_gate",
            ),
            update_plan=update_plan,
            documentation_mapping_score=62,
            source_traceability_score=66,
            governance_alignment_score=74,
            mutation_risk_score=34,
            governance_weight=90,
        ),
        _signal(
            kind="documentation_governance_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="governance_gate",
            source_candidate_ids=update_plan.candidate_ids,
            update_plan=update_plan,
            documentation_mapping_score=46,
            source_traceability_score=48,
            governance_alignment_score=90,
            mutation_risk_score=18,
            governance_weight=70,
        ),
    )


def _signal(
    *,
    kind: DocumentationIntelligenceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: DocumentationIntelligenceAxis,
    source_candidate_ids: tuple[str, ...],
    update_plan: AutomaticKBUpdatePlan,
    documentation_mapping_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> DocumentationIntelligenceSignal:
    score = _intelligence_score(
        documentation_mapping_score=documentation_mapping_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    return DocumentationIntelligenceSignal(
        signal_id=f"documentation_intelligence::{kind}",
        signal_kind=kind,
        status=_intelligence_status(score),
        confidence=_intelligence_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        intelligence_axis=axis,
        source_update_candidate_ids=source_candidate_ids,
        source_update_candidate_count=len(source_candidate_ids),
        source_count=update_plan.source_count,
        domain_count=update_plan.domain_count,
        documentation_signal_summary=_signal_summary(kind),
        documentation_mapping_score=documentation_mapping_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        intelligence_score=score,
        hitl_required_before_documentation_action=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, source_candidate_ids),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"source_update_candidate_count:{len(source_candidate_ids)}",
            f"source_count:{update_plan.source_count}",
            f"domain_count:{update_plan.domain_count}",
            f"intelligence_axis:{axis}",
            f"documentation_mapping_score:{documentation_mapping_score}",
            f"source_traceability_score:{source_traceability_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"mutation_risk_score:{mutation_risk_score}",
            "hitl_required_before_documentation_action:true",
        ),
    )


def _intelligence_score(
    *,
    documentation_mapping_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            documentation_mapping_score * 3
            + source_traceability_score * 3
            + governance_alignment_score * 2
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _intelligence_status(score: int) -> DocumentationIntelligenceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _intelligence_confidence(score: int) -> DocumentationIntelligenceConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_intelligence_score(
    signals: tuple[DocumentationIntelligenceSignal, ...],
) -> int:
    base = sum(signal.intelligence_score for signal in signals) // len(signals)
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    review_count = len(_signal_ids_for_status(signals, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_intelligence_posture(
    signals: tuple[DocumentationIntelligenceSignal, ...],
) -> DocumentationIntelligencePosture:
    if any(signal.status == "guarded" for signal in signals):
        return "guarded"
    if any(signal.status == "review_required" for signal in signals):
        return "review_required"
    return "candidate"


def _signal_ids_for_status(
    signals: tuple[DocumentationIntelligenceSignal, ...],
    status: DocumentationIntelligenceStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _signal_ids_for_confidence(
    signals: tuple[DocumentationIntelligenceSignal, ...],
    *confidences: DocumentationIntelligenceConfidence,
) -> tuple[str, ...]:
    return tuple(
        signal.signal_id for signal in signals if signal.confidence in confidences
    )


def _plan_actions(
    signals: tuple[DocumentationIntelligenceSignal, ...],
) -> tuple[str, ...]:
    guarded_count = len(_signal_ids_for_status(signals, "guarded"))
    return (
        "inspect_documentation_intelligence_metadata",
        "verify_documentation_roadmap_traceability",
        "review_source_mapping_before_any_documentation_action",
        "require_hitl_before_fetch_parse_enrich_or_storage_write",
        f"guarded_signal_count:{guarded_count}",
    )


def _context_tags(
    kind: DocumentationIntelligenceKind,
    axis: DocumentationIntelligenceAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "documentation_intelligence",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: DocumentationIntelligenceKind,
    source_candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"signal:{kind}",
        f"source_update_candidate_count:{len(source_candidate_ids)}",
        "composes_automatic_kb_update_metadata",
        "keeps_documentation_actions_disabled",
        "requires_human_review_before_documentation_mutation",
    )


def _signal_actions(kind: DocumentationIntelligenceKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_documentation_signal_metadata",
        "verify_source_update_candidate_traceability",
        "keep_documentation_actions_disabled",
        "require_hitl_before_documentation_intelligence_action",
    )
    if kind == "documentation_structure_review":
        return base_actions + ("review_documentation_structure_metadata",)
    if kind == "documentation_freshness_review":
        return base_actions + ("review_documentation_freshness_metadata",)
    if kind == "documentation_gap_review":
        return base_actions + ("review_documentation_gap_metadata",)
    if kind == "documentation_governance_gate":
        return base_actions + ("confirm_manual_documentation_governance_gate",)
    return base_actions + ("review_documentation_source_map",)


def _signal_summary(kind: DocumentationIntelligenceKind) -> str:
    summaries: dict[DocumentationIntelligenceKind, str] = {
        "documentation_source_map": (
            "Advisory documentation source map over approved KB update metadata "
            "without crawling or parsing documentation."
        ),
        "documentation_structure_review": (
            "Advisory posture for reviewing documentation structure metadata "
            "before any parser or classifier is enabled."
        ),
        "documentation_freshness_review": (
            "Advisory posture for reviewing documentation freshness signals "
            "without executing sync, summaries, or KB enrichment."
        ),
        "documentation_gap_review": (
            "Advisory posture for identifying documentation coverage review "
            "needs while keeping gap remediation disabled."
        ),
        "documentation_governance_gate": (
            "Governed manual gate that keeps documentation intelligence actions "
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
