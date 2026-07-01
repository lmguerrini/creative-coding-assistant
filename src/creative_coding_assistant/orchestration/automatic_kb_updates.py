"""V6.3 advisory automatic KB update metadata."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.rag.source_health import (
    OfficialSourceHealthSnapshot,
    OfficialSourceSyncMetadata,
    evaluate_official_source_health,
)
from creative_coding_assistant.rag.sources import (
    OfficialSource,
    approved_official_sources,
    official_source_domains,
)

AutomaticKBUpdateKind = Literal[
    "approved_source_registry_monitor",
    "freshness_policy_monitor",
    "sync_metadata_review",
    "domain_coverage_review",
    "manual_execution_gate",
]
AutomaticKBUpdateStatus = Literal["candidate", "review_required", "guarded"]
AutomaticKBUpdateConfidence = Literal["low", "medium", "high", "guarded"]
AutomaticKBUpdatePosture = Literal["candidate", "review_required", "guarded"]
AutomaticKBUpdateAxis = Literal[
    "source_registry",
    "freshness_policy",
    "sync_metadata",
    "domain_coverage",
    "execution_gate",
]

AUTOMATIC_KB_UPDATE_ENTRY_SERIALIZATION_VERSION = "automatic_kb_update_entry.v1"
AUTOMATIC_KB_UPDATE_PLAN_SERIALIZATION_VERSION = "automatic_kb_update_plan.v1"
OFFICIAL_SOURCE_REGISTRY_SERIALIZATION_VERSION = "official_source_registry.v1"
OFFICIAL_SOURCE_HEALTH_SERIALIZATION_VERSION = "official_source_health_snapshot.v1"
OFFICIAL_SOURCE_SYNC_REQUEST_SERIALIZATION_VERSION = "official_source_sync_request.v1"
_DEFAULT_CHECKED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

AUTOMATIC_KB_UPDATE_AUTHORITY_BOUNDARY = (
    "V6.3 Automatic KB Updates exposes approved-source and source-health "
    "metadata as inspectable advisory update posture only; it does not fetch "
    "sources, normalize documents, chunk documents, request embeddings, refresh "
    "embeddings, index records, upsert vector records, write KB storage, mutate "
    "the source registry, mutate retrieval configuration, execute retrieval, "
    "change ranking, provision providers, infer API keys, execute providers, "
    "invoke agents, control workflows, mutate workflow graphs, trigger retries "
    "or refinements, mutate prompts, modify generated output, or apply Runtime "
    "Evolution."
)

_ROADMAP_ITEMS = ("Automatic KB Updates",)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "automatic_kb_update_execution",
    "source_fetch_execution",
    "source_normalization_execution",
    "source_chunking_execution",
    "embedding_request_execution",
    "embedding_refresh_execution",
    "vector_record_indexing",
    "vector_record_upsert",
    "kb_storage_write",
    "source_registry_mutation",
    "retrieval_configuration_mutation",
    "retrieval_execution",
    "ranking_mutation",
    "provider_provisioning",
    "api_key_inference",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
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


class AutomaticKBUpdateCandidate(BaseModel):
    """One advisory automatic KB update candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    update_id: str = Field(min_length=1, max_length=180)
    update_kind: AutomaticKBUpdateKind
    status: AutomaticKBUpdateStatus
    confidence: AutomaticKBUpdateConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    update_axis: AutomaticKBUpdateAxis
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=60)
    source_count: int = Field(ge=1, le=60)
    domain_count: int = Field(ge=1, le=60)
    source_type_count: int = Field(ge=1, le=4)
    unknown_health_count: int = Field(ge=0, le=60)
    stale_health_count: int = Field(ge=0, le=60)
    refresh_recommended_count: int = Field(ge=0, le=60)
    update_summary: str = Field(min_length=1, max_length=360)
    source_registry_score: int = Field(ge=0, le=100)
    health_metadata_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    mutation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    update_score: int = Field(ge=0, le=1_000)
    hitl_required_before_update_execution: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    automatic_kb_updates_implemented: Literal[True] = True
    automatic_kb_update_metadata_implemented: Literal[True] = True
    official_source_registry_used: Literal[True] = True
    source_health_metadata_used: Literal[True] = True
    sync_request_metadata_used: Literal[True] = True
    automatic_update_execution_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    source_normalization_execution_implemented: Literal[False] = False
    source_chunking_execution_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_record_indexing_implemented: Literal[False] = False
    vector_record_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
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
    serialization_version: Literal["automatic_kb_update_entry.v1"] = (
        AUTOMATIC_KB_UPDATE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.update_id != f"automatic_kb_updates::{self.update_kind}":
            raise ValueError("update_id must match update_kind")
        if self.source_count != len(self.source_ids):
            raise ValueError("source_count must match source_ids")
        if self.update_score != _update_score(
            source_registry_score=self.source_registry_score,
            health_metadata_score=self.health_metadata_score,
            governance_alignment_score=self.governance_alignment_score,
            mutation_risk_score=self.mutation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("update_score must combine source scores")
        if self.status != _update_status(self.update_score):
            raise ValueError("status must match update_score")
        if self.confidence != _update_confidence(self.update_score):
            raise ValueError("confidence must match update_score")
        if not self.hitl_required_before_update_execution:
            raise ValueError("automatic update execution requires HITL posture")
        if self.unknown_health_count > self.source_count:
            raise ValueError("unknown_health_count cannot exceed source_count")
        if self.stale_health_count > self.source_count:
            raise ValueError("stale_health_count cannot exceed source_count")
        if self.refresh_recommended_count > self.source_count:
            raise ValueError("refresh_recommended_count cannot exceed source_count")
        return self


class AutomaticKBUpdatePlan(BaseModel):
    """Bounded V6.3 advisory automatic KB update plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["automatic_kb_updates"] = "automatic_kb_updates"
    serialization_version: Literal["automatic_kb_update_plan.v1"] = (
        AUTOMATIC_KB_UPDATE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AUTOMATIC_KB_UPDATE_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    checked_at: datetime
    source_registry_role: Literal["approved_official_sources"] = (
        "approved_official_sources"
    )
    source_registry_serialization_version: Literal["official_source_registry.v1"] = (
        OFFICIAL_SOURCE_REGISTRY_SERIALIZATION_VERSION
    )
    source_health_serialization_version: Literal[
        "official_source_health_snapshot.v1"
    ] = OFFICIAL_SOURCE_HEALTH_SERIALIZATION_VERSION
    sync_request_serialization_version: Literal["official_source_sync_request.v1"] = (
        OFFICIAL_SOURCE_SYNC_REQUEST_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=80)
    domain_count: int = Field(ge=1, le=60)
    source_type_count: int = Field(ge=1, le=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    candidates: tuple[AutomaticKBUpdateCandidate, ...] = Field(
        min_length=5,
        max_length=5,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_entry_count: int = Field(ge=5, le=5)
    candidate_status_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    planned_update_execution_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    fetched_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    indexed_source_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_source_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_storage_record_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidate_count: int = Field(ge=5, le=5)
    candidate_status_count: int = Field(ge=0, le=5)
    review_required_candidate_count: int = Field(ge=0, le=5)
    guarded_candidate_count: int = Field(ge=0, le=5)
    high_confidence_candidate_count: int = Field(ge=0, le=5)
    hitl_required_candidate_count: int = Field(ge=0, le=5)
    highest_update_score: int = Field(ge=0, le=1_000)
    overall_update_score: int = Field(ge=0, le=1_000)
    overall_update_posture: AutomaticKBUpdatePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=32,
    )
    automatic_kb_updates_implemented: Literal[True] = True
    automatic_kb_update_metadata_implemented: Literal[True] = True
    roadmap_item_covered: Literal[True] = True
    official_source_registry_used: Literal[True] = True
    source_health_metadata_used: Literal[True] = True
    sync_request_metadata_used: Literal[True] = True
    automatic_update_execution_implemented: Literal[False] = False
    source_fetch_execution_implemented: Literal[False] = False
    source_normalization_execution_implemented: Literal[False] = False
    source_chunking_execution_implemented: Literal[False] = False
    embedding_request_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    vector_record_indexing_implemented: Literal[False] = False
    vector_record_upsert_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
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
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.update_id for candidate in self.candidates
        )
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_status_ids != _candidate_ids_for_status(
            self.candidates,
            "candidate",
        ):
            raise ValueError("candidate_status_ids must match candidates")
        if self.review_required_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "review_required",
        ):
            raise ValueError("review_required_candidate_ids must match candidates")
        if self.guarded_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "guarded",
        ):
            raise ValueError("guarded_candidate_ids must match candidates")
        if self.high_confidence_candidate_ids != _candidate_ids_for_confidence(
            self.candidates,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != tuple(
            candidate.update_id
            for candidate in self.candidates
            if candidate.hitl_required_before_update_execution
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.planned_update_execution_ids:
            raise ValueError("planned_update_execution_ids must remain empty")
        if self.fetched_source_ids:
            raise ValueError("fetched_source_ids must remain empty")
        if self.indexed_source_ids:
            raise ValueError("indexed_source_ids must remain empty")
        if self.mutated_retrieval_source_ids:
            raise ValueError("mutated_retrieval_source_ids must remain empty")
        if self.written_storage_record_ids:
            raise ValueError("written_storage_record_ids must remain empty")
        if self.source_count != len(self.source_ids):
            raise ValueError("source_count must match source_ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 Task 2 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.candidate_entry_count != len(self.candidates):
            raise ValueError("candidate_entry_count must match candidates")
        if self.candidate_status_count != len(self.candidate_status_ids):
            raise ValueError("candidate_status_count must match candidates")
        if self.review_required_candidate_count != len(
            self.review_required_candidate_ids
        ):
            raise ValueError("review_required_candidate_count must match candidates")
        if self.guarded_candidate_count != len(self.guarded_candidate_ids):
            raise ValueError("guarded_candidate_count must match candidates")
        if self.high_confidence_candidate_count != len(
            self.high_confidence_candidate_ids
        ):
            raise ValueError("high_confidence_candidate_count must match candidates")
        if self.hitl_required_candidate_count != len(
            self.hitl_required_candidate_ids
        ):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.highest_update_score != max(
            candidate.update_score for candidate in self.candidates
        ):
            raise ValueError("highest_update_score must match candidates")
        if self.overall_update_score != _overall_update_score(self.candidates):
            raise ValueError("overall_update_score must match candidates")
        if self.overall_update_posture != _overall_update_posture(self.candidates):
            raise ValueError("overall_update_posture must match candidates")
        declared_source_ids = set(self.source_ids)
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan")
            if not set(candidate.source_ids).issubset(declared_source_ids):
                raise ValueError("candidate source_ids must be declared")
        return self


def build_automatic_kb_updates(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    sources: tuple[OfficialSource, ...] | None = None,
    sync_metadata_by_source_id: Mapping[str, OfficialSourceSyncMetadata] | None = None,
    checked_at: datetime = _DEFAULT_CHECKED_AT,
) -> AutomaticKBUpdatePlan:
    """Build V6.3 Task 2 automatic KB update metadata without executing updates."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    source_registry = tuple(sources or approved_official_sources())
    sync_metadata = dict(sync_metadata_by_source_id or {})
    snapshots = tuple(
        evaluate_official_source_health(
            source,
            sync_metadata=sync_metadata.get(source.source_id),
            checked_at=checked_at,
        )
        for source in source_registry
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=source_registry,
        snapshots=snapshots,
    )
    source_ids = tuple(source.source_id for source in source_registry)
    return AutomaticKBUpdatePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        checked_at=checked_at,
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        source_ids=source_ids,
        source_count=len(source_ids),
        domain_count=len({source.domain for source in source_registry}),
        source_type_count=len({source.source_type for source in source_registry}),
        execution_mode_ids=execution_modes.execution_mode_ids,
        candidates=entries,
        candidate_ids=tuple(candidate.update_id for candidate in entries),
        candidate_entry_count=len(entries),
        candidate_status_ids=_candidate_ids_for_status(entries, "candidate"),
        review_required_candidate_ids=_candidate_ids_for_status(
            entries,
            "review_required",
        ),
        guarded_candidate_ids=_candidate_ids_for_status(entries, "guarded"),
        high_confidence_candidate_ids=_candidate_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_candidate_ids=tuple(
            candidate.update_id
            for candidate in entries
            if candidate.hitl_required_before_update_execution
        ),
        candidate_count=len(entries),
        candidate_status_count=len(_candidate_ids_for_status(entries, "candidate")),
        review_required_candidate_count=len(
            _candidate_ids_for_status(entries, "review_required")
        ),
        guarded_candidate_count=len(_candidate_ids_for_status(entries, "guarded")),
        high_confidence_candidate_count=len(
            _candidate_ids_for_confidence(entries, "high", "guarded")
        ),
        hitl_required_candidate_count=sum(
            1
            for candidate in entries
            if candidate.hitl_required_before_update_execution
        ),
        highest_update_score=max(candidate.update_score for candidate in entries),
        overall_update_score=_overall_update_score(entries),
        overall_update_posture=_overall_update_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def automatic_kb_update_candidate_by_id(
    candidate_id: str,
    plan: AutomaticKBUpdatePlan | None = None,
) -> AutomaticKBUpdateCandidate | None:
    """Return one automatic KB update candidate without executing updates."""

    source_plan = plan or build_automatic_kb_updates()
    for candidate in source_plan.candidates:
        if candidate.update_id == candidate_id:
            return candidate
    return None


def automatic_kb_update_candidates_for_status(
    status: AutomaticKBUpdateStatus,
    plan: AutomaticKBUpdatePlan | None = None,
) -> tuple[AutomaticKBUpdateCandidate, ...]:
    """Return automatic KB update candidates by advisory status."""

    source_plan = plan or build_automatic_kb_updates()
    return tuple(
        candidate for candidate in source_plan.candidates if candidate.status == status
    )


def automatic_kb_update_candidates_for_confidence(
    confidence: AutomaticKBUpdateConfidence,
    plan: AutomaticKBUpdatePlan | None = None,
) -> tuple[AutomaticKBUpdateCandidate, ...]:
    """Return automatic KB update candidates by confidence band."""

    source_plan = plan or build_automatic_kb_updates()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.confidence == confidence
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    sources: tuple[OfficialSource, ...],
    snapshots: tuple[OfficialSourceHealthSnapshot, ...],
) -> tuple[AutomaticKBUpdateCandidate, ...]:
    return (
        _entry(
            kind="approved_source_registry_monitor",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            update_axis="source_registry",
            sources=sources,
            snapshots=snapshots,
            source_registry_score=90,
            health_metadata_score=70,
            governance_alignment_score=86,
            mutation_risk_score=50,
            governance_weight=120,
        ),
        _entry(
            kind="freshness_policy_monitor",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            update_axis="freshness_policy",
            sources=sources,
            snapshots=snapshots,
            source_registry_score=78,
            health_metadata_score=72,
            governance_alignment_score=82,
            mutation_risk_score=48,
            governance_weight=100,
        ),
        _entry(
            kind="sync_metadata_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            update_axis="sync_metadata",
            sources=tuple(
                sorted(sources, key=lambda source: (source.priority, source.source_id))[
                    :12
                ]
            ),
            snapshots=snapshots,
            source_registry_score=68,
            health_metadata_score=64,
            governance_alignment_score=78,
            mutation_risk_score=44,
            governance_weight=95,
        ),
        _entry(
            kind="domain_coverage_review",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            update_axis="domain_coverage",
            sources=_first_source_per_domain(sources),
            snapshots=snapshots,
            source_registry_score=74,
            health_metadata_score=58,
            governance_alignment_score=80,
            mutation_risk_score=36,
            governance_weight=85,
        ),
        _entry(
            kind="manual_execution_gate",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            update_axis="execution_gate",
            sources=sources,
            snapshots=snapshots,
            source_registry_score=44,
            health_metadata_score=44,
            governance_alignment_score=90,
            mutation_risk_score=16,
            governance_weight=60,
        ),
    )


def _entry(
    *,
    kind: AutomaticKBUpdateKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    update_axis: AutomaticKBUpdateAxis,
    sources: tuple[OfficialSource, ...],
    snapshots: tuple[OfficialSourceHealthSnapshot, ...],
    source_registry_score: int,
    health_metadata_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> AutomaticKBUpdateCandidate:
    selected_source_ids = tuple(source.source_id for source in sources)
    selected_snapshots = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.source.source_id in selected_source_ids
    )
    score = _update_score(
        source_registry_score=source_registry_score,
        health_metadata_score=health_metadata_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
    )
    status = _update_status(score)
    confidence = _update_confidence(score)
    unknown_count = sum(
        1 for snapshot in selected_snapshots if snapshot.health_status == "unknown"
    )
    stale_count = sum(
        1 for snapshot in selected_snapshots if snapshot.health_status == "stale"
    )
    refresh_count = sum(
        1 for snapshot in selected_snapshots if snapshot.refresh_recommended
    )
    return AutomaticKBUpdateCandidate(
        update_id=f"automatic_kb_updates::{kind}",
        update_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        update_axis=update_axis,
        source_ids=selected_source_ids,
        source_count=len(selected_source_ids),
        domain_count=len({source.domain for source in sources}),
        source_type_count=len({source.source_type for source in sources}),
        unknown_health_count=unknown_count,
        stale_health_count=stale_count,
        refresh_recommended_count=refresh_count,
        update_summary=_update_summary(kind),
        source_registry_score=source_registry_score,
        health_metadata_score=health_metadata_score,
        governance_alignment_score=governance_alignment_score,
        mutation_risk_score=mutation_risk_score,
        governance_weight=governance_weight,
        update_score=score,
        hitl_required_before_update_execution=True,
        context_tags=_context_tags(kind, update_axis),
        explainability_notes=_explainability_notes(kind, selected_source_ids),
        advisory_actions=_candidate_actions(kind),
        evidence=(
            f"source_count:{len(selected_source_ids)}",
            f"domain_count:{len({source.domain for source in sources})}",
            f"source_type_count:{len({source.source_type for source in sources})}",
            f"unknown_health_count:{unknown_count}",
            f"stale_health_count:{stale_count}",
            f"refresh_recommended_count:{refresh_count}",
            f"update_axis:{update_axis}",
            "hitl_required_before_update_execution:true",
        ),
    )


def _first_source_per_domain(
    sources: tuple[OfficialSource, ...],
) -> tuple[OfficialSource, ...]:
    source_by_domain = {
        domain: tuple(
            sorted(
                (
                    source
                    for source in sources
                    if source.domain == domain
                ),
                key=lambda source: (source.priority, source.source_id),
            )
        )
        for domain in official_source_domains()
    }
    return tuple(
        domain_sources[0]
        for domain_sources in source_by_domain.values()
        if domain_sources
    )


def _update_score(
    *,
    source_registry_score: int,
    health_metadata_score: int,
    governance_alignment_score: int,
    mutation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_registry_score * 3
            + health_metadata_score * 2
            + governance_alignment_score * 3
            + mutation_risk_score * 2
            + governance_weight,
        ),
    )


def _update_status(score: int) -> AutomaticKBUpdateStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _update_confidence(score: int) -> AutomaticKBUpdateConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_update_score(
    candidates: tuple[AutomaticKBUpdateCandidate, ...],
) -> int:
    base = sum(candidate.update_score for candidate in candidates) // len(candidates)
    guarded_count = len(_candidate_ids_for_status(candidates, "guarded"))
    review_count = len(_candidate_ids_for_status(candidates, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_update_posture(
    candidates: tuple[AutomaticKBUpdateCandidate, ...],
) -> AutomaticKBUpdatePosture:
    if any(candidate.status == "guarded" for candidate in candidates):
        return "guarded"
    if any(candidate.status == "review_required" for candidate in candidates):
        return "review_required"
    return "candidate"


def _candidate_ids_for_status(
    candidates: tuple[AutomaticKBUpdateCandidate, ...],
    status: AutomaticKBUpdateStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.update_id for candidate in candidates if candidate.status == status
    )


def _candidate_ids_for_confidence(
    candidates: tuple[AutomaticKBUpdateCandidate, ...],
    *confidences: AutomaticKBUpdateConfidence,
) -> tuple[str, ...]:
    return tuple(
        candidate.update_id
        for candidate in candidates
        if candidate.confidence in confidences
    )


def _plan_actions(
    candidates: tuple[AutomaticKBUpdateCandidate, ...],
) -> tuple[str, ...]:
    guarded_count = len(_candidate_ids_for_status(candidates, "guarded"))
    return (
        "inspect_automatic_kb_update_metadata",
        "verify_approved_source_registry_traceability",
        "review_source_health_unknowns_before_any_update_execution",
        "require_hitl_before_fetch_index_or_storage_write",
        f"guarded_candidate_count:{guarded_count}",
    )


def _context_tags(
    kind: AutomaticKBUpdateKind,
    axis: AutomaticKBUpdateAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "automatic_kb_updates",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: AutomaticKBUpdateKind,
    source_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"candidate:{kind}",
        f"source_count:{len(source_ids)}",
        "uses_approved_source_registry_metadata",
        "uses_source_health_metadata_without_sync_execution",
        "requires_human_review_before_runtime_mutation",
    )


def _candidate_actions(kind: AutomaticKBUpdateKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_candidate_metadata",
        "verify_source_ids_against_approved_registry",
        "keep_update_execution_disabled",
        "require_hitl_before_automatic_kb_update",
    )
    if kind == "freshness_policy_monitor":
        return base_actions + ("review_freshness_policy_thresholds",)
    if kind == "sync_metadata_review":
        return base_actions + ("review_sync_metadata_before_fetch",)
    if kind == "domain_coverage_review":
        return base_actions + ("review_domain_source_coverage",)
    if kind == "manual_execution_gate":
        return base_actions + ("confirm_manual_execution_gate_before_mutation",)
    return base_actions + ("review_approved_source_registry",)


def _update_summary(kind: AutomaticKBUpdateKind) -> str:
    summaries: dict[AutomaticKBUpdateKind, str] = {
        "approved_source_registry_monitor": (
            "Advisory posture for reviewing approved official source coverage "
            "before any KB update is executed."
        ),
        "freshness_policy_monitor": (
            "Advisory posture for identifying source freshness review needs "
            "without triggering sync or retrieval mutation."
        ),
        "sync_metadata_review": (
            "Advisory posture for inspecting sync metadata readiness before "
            "fetch, normalize, embed, or index work."
        ),
        "domain_coverage_review": (
            "Advisory posture for checking one approved source per registered "
            "creative coding domain."
        ),
        "manual_execution_gate": (
            "Governed manual gate that keeps automatic KB update execution "
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
