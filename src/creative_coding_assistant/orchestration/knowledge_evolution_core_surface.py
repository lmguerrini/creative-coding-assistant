"""V6.3 advisory knowledge evolution core surface metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.automatic_kb_updates import (
    build_automatic_kb_updates,
)
from creative_coding_assistant.orchestration.documentation_intelligence import (
    build_documentation_intelligence,
)
from creative_coding_assistant.orchestration.embedding_refresh import (
    build_embedding_refresh,
)
from creative_coding_assistant.orchestration.knowledge_conflict_resolver import (
    build_knowledge_conflict_resolver,
)
from creative_coding_assistant.orchestration.knowledge_consolidation import (
    build_knowledge_consolidation,
)
from creative_coding_assistant.orchestration.knowledge_drift_detection import (
    build_knowledge_drift_detection,
)
from creative_coding_assistant.orchestration.knowledge_freshness_tracking import (
    build_knowledge_freshness_tracking,
)
from creative_coding_assistant.orchestration.knowledge_gap_detection import (
    build_knowledge_gap_detection,
)
from creative_coding_assistant.orchestration.knowledge_health_monitoring import (
    build_knowledge_health_monitoring,
)
from creative_coding_assistant.orchestration.knowledge_lifecycle_management import (
    build_knowledge_lifecycle_management,
)
from creative_coding_assistant.orchestration.knowledge_provenance_evolution import (
    build_knowledge_provenance_evolution,
)
from creative_coding_assistant.orchestration.knowledge_quality_scoring import (
    build_knowledge_quality_scoring,
)
from creative_coding_assistant.orchestration.knowledge_rollback import (
    build_knowledge_rollback,
)
from creative_coding_assistant.orchestration.knowledge_snapshot_engine import (
    build_knowledge_snapshot_engine,
)
from creative_coding_assistant.orchestration.knowledge_trust_score import (
    build_knowledge_trust_score,
)
from creative_coding_assistant.orchestration.knowledge_versioning import (
    build_knowledge_versioning,
)
from creative_coding_assistant.orchestration.ranking_optimization import (
    build_ranking_optimization,
)
from creative_coding_assistant.orchestration.retrieval_evolution import (
    build_retrieval_evolution,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.source_reliability_engine import (
    build_source_reliability_engine,
)

KnowledgeEvolutionCoreSurfaceKind = Literal[
    "knowledge_update_surface",
    "retrieval_quality_surface",
    "knowledge_integrity_surface",
    "knowledge_governance_surface",
    "recovery_trust_surface",
]
KnowledgeEvolutionCoreSurfaceStatus = Literal[
    "candidate",
    "review_required",
    "guarded",
]
KnowledgeEvolutionCoreSurfaceConfidence = Literal[
    "low",
    "medium",
    "high",
    "guarded",
]
KnowledgeEvolutionCoreSurfacePosture = Literal[
    "candidate",
    "review_required",
    "guarded",
]
KnowledgeEvolutionCoreSurfaceAxis = Literal[
    "knowledge_updates",
    "retrieval_quality",
    "knowledge_integrity",
    "knowledge_governance",
    "recovery_trust",
]

KNOWLEDGE_EVOLUTION_CORE_ENTRY_SERIALIZATION_VERSION = (
    "knowledge_evolution_core_surface_entry.v1"
)
KNOWLEDGE_EVOLUTION_CORE_PLAN_SERIALIZATION_VERSION = (
    "knowledge_evolution_core_surface_plan.v1"
)
KNOWLEDGE_EVOLUTION_CORE_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Evolution Core Surface exposes the validated V6.3 "
    "knowledge evolution roadmap surfaces as inspectable advisory metadata "
    "only; it does not activate core surfaces, execute automatic KB updates, "
    "fetch documentation, refresh embeddings, execute retrieval, mutate "
    "ranking, run health monitoring, compute quality or trust scores, execute "
    "gap detection, resolve conflicts, detect drift, score source reliability, "
    "consolidate knowledge, manage lifecycle state, mutate provenance graphs, "
    "mutate version graphs, execute snapshots, execute rollback, run freshness "
    "scans, write KB storage, update source records, change provider or model "
    "routing, execute providers, invoke agents, control workflows, mutate "
    "workflow graphs, modify generated output, or apply Runtime Evolution."
)

_SOURCE_PLAN_ROLES = (
    "automatic_kb_updates",
    "documentation_intelligence",
    "embedding_refresh",
    "retrieval_evolution",
    "ranking_optimization",
    "knowledge_health_monitoring",
    "knowledge_quality_scoring",
    "knowledge_gap_detection",
    "knowledge_conflict_resolver",
    "knowledge_drift_detection",
    "source_reliability_engine",
    "knowledge_consolidation",
    "knowledge_lifecycle_management",
    "knowledge_provenance_evolution",
    "knowledge_versioning",
    "knowledge_snapshot_engine",
    "knowledge_rollback",
    "knowledge_freshness_tracking",
    "knowledge_trust_score",
)

_ROADMAP_ITEMS = (
    "Automatic KB Updates",
    "Documentation Intelligence",
    "Embedding Refresh",
    "Retrieval Evolution",
    "Ranking Optimization",
    "Knowledge Health Monitoring",
    "Knowledge Quality Scoring",
    "Knowledge Gap Detection",
    "Knowledge Conflict Resolver",
    "Knowledge Drift Detection",
    "Source Reliability Engine",
    "Knowledge Consolidation",
    "Knowledge Lifecycle Management",
    "Knowledge Provenance Evolution",
    "Knowledge Versioning",
    "Knowledge Snapshot Engine",
    "Knowledge Rollback",
    "Knowledge Freshness Tracking",
    "Knowledge Trust Score",
)

_SURFACE_ROLE_GROUPS: dict[KnowledgeEvolutionCoreSurfaceKind, tuple[str, ...]] = {
    "knowledge_update_surface": (
        "automatic_kb_updates",
        "documentation_intelligence",
        "embedding_refresh",
    ),
    "retrieval_quality_surface": (
        "retrieval_evolution",
        "ranking_optimization",
        "knowledge_health_monitoring",
        "knowledge_quality_scoring",
    ),
    "knowledge_integrity_surface": (
        "knowledge_gap_detection",
        "knowledge_conflict_resolver",
        "knowledge_drift_detection",
        "source_reliability_engine",
    ),
    "knowledge_governance_surface": (
        "knowledge_consolidation",
        "knowledge_lifecycle_management",
        "knowledge_provenance_evolution",
        "knowledge_versioning",
    ),
    "recovery_trust_surface": (
        "knowledge_snapshot_engine",
        "knowledge_rollback",
        "knowledge_freshness_tracking",
        "knowledge_trust_score",
    ),
}

_BLOCKED_RUNTIME_BEHAVIORS = (
    "core_surface_activation",
    "automatic_kb_update_execution",
    "documentation_fetch_execution",
    "embedding_refresh_execution",
    "retrieval_execution",
    "ranking_mutation",
    "knowledge_health_monitoring_execution",
    "quality_score_computation",
    "knowledge_gap_detection_execution",
    "knowledge_conflict_resolution_execution",
    "knowledge_drift_detection_execution",
    "source_reliability_scoring_execution",
    "knowledge_consolidation_execution",
    "knowledge_lifecycle_management_execution",
    "provenance_graph_mutation",
    "version_graph_mutation",
    "knowledge_snapshot_engine_execution",
    "knowledge_rollback_execution",
    "freshness_scan_execution",
    "trust_score_computation",
    "kb_storage_write",
    "source_record_update",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeEvolutionCoreSurfaceEntry(BaseModel):
    """One advisory entry in the V6.3 knowledge evolution core surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    core_surface_id: str = Field(min_length=1, max_length=180)
    surface_kind: KnowledgeEvolutionCoreSurfaceKind
    status: KnowledgeEvolutionCoreSurfaceStatus
    confidence: KnowledgeEvolutionCoreSurfaceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    surface_axis: KnowledgeEvolutionCoreSurfaceAxis
    roadmap_items: tuple[str, ...] = Field(min_length=3, max_length=4)
    roadmap_item_count: int = Field(ge=3, le=4)
    source_plan_roles: tuple[str, ...] = Field(min_length=3, max_length=4)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=3,
        max_length=4,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=15, max_length=20)
    source_item_count: int = Field(ge=15, le=20)
    surface_summary: str = Field(min_length=1, max_length=360)
    surface_coverage_score: int = Field(ge=0, le=100)
    source_traceability_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    activation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    core_surface_score: int = Field(ge=0, le=1_000)
    hitl_required_before_core_surface_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    automatic_kb_update_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    knowledge_conflict_resolution_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["knowledge_evolution_core_surface_entry.v1"] = (
        KNOWLEDGE_EVOLUTION_CORE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        expected_id = f"knowledge_evolution_core::{self.surface_kind}"
        if self.core_surface_id != expected_id:
            raise ValueError("core_surface_id must match surface_kind")
        if self.roadmap_item_count != len(self.roadmap_items):
            raise ValueError("roadmap_item_count must match roadmap_items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.core_surface_score != _core_surface_score(
            surface_coverage_score=self.surface_coverage_score,
            source_traceability_score=self.source_traceability_score,
            governance_alignment_score=self.governance_alignment_score,
            activation_risk_score=self.activation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("core_surface_score must combine source scores")
        if self.status != _core_surface_status(self.core_surface_score):
            raise ValueError("status must match core_surface_score")
        if self.confidence != _core_surface_confidence(self.core_surface_score):
            raise ValueError("confidence must match core_surface_score")
        if not self.hitl_required_before_core_surface_activation:
            raise ValueError("core surface activation requires HITL posture")
        return self


class KnowledgeEvolutionCoreSurfacePlan(BaseModel):
    """Bounded V6.3 advisory knowledge evolution core surface plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_evolution_core_surface"] = (
        "knowledge_evolution_core_surface"
    )
    serialization_version: Literal["knowledge_evolution_core_surface_plan.v1"] = (
        KNOWLEDGE_EVOLUTION_CORE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_EVOLUTION_CORE_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_plan_roles: tuple[str, ...] = Field(min_length=19, max_length=19)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=19,
        max_length=19,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=95, max_length=95)
    source_item_count: int = Field(ge=95, le=95)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    covered_roadmap_item_count: int = Field(ge=19, le=19)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...] = Field(
        min_length=5,
        max_length=5,
    )
    entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_entry_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    activated_core_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_update_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    entry_count: int = Field(ge=5, le=5)
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_posture: KnowledgeEvolutionCoreSurfacePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=40,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    automatic_kb_update_execution_implemented: Literal[False] = False
    documentation_fetch_execution_implemented: Literal[False] = False
    embedding_refresh_execution_implemented: Literal[False] = False
    retrieval_execution_implemented: Literal[False] = False
    ranking_mutation_implemented: Literal[False] = False
    knowledge_health_monitoring_execution_implemented: Literal[False] = False
    quality_score_computation_implemented: Literal[False] = False
    knowledge_gap_detection_execution_implemented: Literal[False] = False
    knowledge_conflict_resolution_execution_implemented: Literal[False] = False
    knowledge_drift_detection_execution_implemented: Literal[False] = False
    source_reliability_scoring_execution_implemented: Literal[False] = False
    knowledge_consolidation_execution_implemented: Literal[False] = False
    knowledge_lifecycle_management_execution_implemented: Literal[False] = False
    provenance_graph_mutation_implemented: Literal[False] = False
    version_graph_mutation_implemented: Literal[False] = False
    knowledge_snapshot_engine_execution_implemented: Literal[False] = False
    knowledge_rollback_execution_implemented: Literal[False] = False
    freshness_scan_execution_implemented: Literal[False] = False
    trust_score_computation_implemented: Literal[False] = False
    kb_storage_write_implemented: Literal[False] = False
    source_record_update_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.core_surface_id for entry in self.entries)
        if len(set(derived_entry_ids)) != len(derived_entry_ids):
            raise ValueError("entry_ids must be unique")
        if self.entry_ids != derived_entry_ids:
            raise ValueError("entry_ids must match entries")
        if self.candidate_entry_ids != _entry_ids_for_status(
            self.entries,
            "candidate",
        ):
            raise ValueError("candidate_entry_ids must match entries")
        if self.review_required_entry_ids != _entry_ids_for_status(
            self.entries,
            "review_required",
        ):
            raise ValueError("review_required_entry_ids must match entries")
        if self.guarded_entry_ids != _entry_ids_for_status(self.entries, "guarded"):
            raise ValueError("guarded_entry_ids must match entries")
        if self.high_confidence_entry_ids != _entry_ids_for_confidence(
            self.entries,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_entry_ids must match entries")
        if self.hitl_required_entry_ids != tuple(
            entry.core_surface_id
            for entry in self.entries
            if entry.hitl_required_before_core_surface_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.activated_core_surface_ids:
            raise ValueError("activated_core_surface_ids must remain empty")
        if self.executed_update_ids:
            raise ValueError("executed_update_ids must remain empty")
        if self.mutated_retrieval_ids:
            raise ValueError("mutated_retrieval_ids must remain empty")
        if self.written_kb_record_ids:
            raise ValueError("written_kb_record_ids must remain empty")
        if self.mutated_output_ids:
            raise ValueError("mutated_output_ids must remain empty")
        if self.source_plan_roles != _SOURCE_PLAN_ROLES:
            raise ValueError("source_plan_roles must match V6.3 source roles")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match entries")
        if self.candidate_entry_count != len(self.candidate_entry_ids):
            raise ValueError("candidate_entry_count must match entries")
        if self.review_required_entry_count != len(self.review_required_entry_ids):
            raise ValueError("review_required_entry_count must match entries")
        if self.guarded_entry_count != len(self.guarded_entry_ids):
            raise ValueError("guarded_entry_count must match entries")
        if self.high_confidence_entry_count != len(self.high_confidence_entry_ids):
            raise ValueError("high_confidence_entry_count must match entries")
        if self.hitl_required_entry_count != len(self.hitl_required_entry_ids):
            raise ValueError("hitl_required_entry_count must match entries")
        if self.highest_core_surface_score != max(
            entry.core_surface_score for entry in self.entries
        ):
            raise ValueError("highest_core_surface_score must match entries")
        if self.overall_core_surface_score != _overall_core_surface_score(self.entries):
            raise ValueError("overall_core_surface_score must match entries")
        if self.overall_core_surface_posture != _overall_core_surface_posture(
            self.entries
        ):
            raise ValueError("overall_core_surface_posture must match entries")
        flattened_roadmap_items = tuple(
            item for entry in self.entries for item in entry.roadmap_items
        )
        if flattened_roadmap_items != self.covered_roadmap_items:
            raise ValueError("entry roadmap_items must cover roadmap items")
        declared_source_items = set(self.source_item_ids)
        declared_source_roles = set(self.source_plan_roles)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if not set(entry.source_item_ids).issubset(declared_source_items):
                raise ValueError("entry source_item_ids must be known")
            if not set(entry.source_plan_roles).issubset(declared_source_roles):
                raise ValueError("entry source_plan_roles must be known")
        return self


def build_knowledge_evolution_core_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    source_plans: tuple[BaseModel, ...] | None = None,
) -> KnowledgeEvolutionCoreSurfacePlan:
    """Build V6.3 Task 21 knowledge evolution core surface metadata only."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    plans = source_plans or _build_source_plans(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    plan_by_role = {str(plan.role): plan for plan in plans}
    source_item_ids = _source_item_ids_for_plans(plans)
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        plan_by_role=plan_by_role,
    )
    return KnowledgeEvolutionCoreSurfacePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_plan_roles=tuple(str(plan.role) for plan in plans),
        source_plan_serialization_versions=tuple(
            str(plan.serialization_version) for plan in plans
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        covered_roadmap_items=tuple(
            item
            for plan in plans
            for item in tuple(plan.model_dump()["covered_roadmap_items"])
        ),
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.core_surface_id for entry in entries),
        candidate_entry_ids=_entry_ids_for_status(entries, "candidate"),
        review_required_entry_ids=_entry_ids_for_status(entries, "review_required"),
        guarded_entry_ids=_entry_ids_for_status(entries, "guarded"),
        high_confidence_entry_ids=_entry_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_entry_ids=tuple(
            entry.core_surface_id
            for entry in entries
            if entry.hitl_required_before_core_surface_activation
        ),
        entry_count=len(entries),
        candidate_entry_count=len(_entry_ids_for_status(entries, "candidate")),
        review_required_entry_count=len(
            _entry_ids_for_status(entries, "review_required")
        ),
        guarded_entry_count=len(_entry_ids_for_status(entries, "guarded")),
        high_confidence_entry_count=len(
            _entry_ids_for_confidence(entries, "high", "guarded")
        ),
        hitl_required_entry_count=sum(
            1 for entry in entries if entry.hitl_required_before_core_surface_activation
        ),
        highest_core_surface_score=max(entry.core_surface_score for entry in entries),
        overall_core_surface_score=_overall_core_surface_score(entries),
        overall_core_surface_posture=_overall_core_surface_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def knowledge_evolution_core_surface_entry_by_id(
    core_surface_id: str,
    plan: KnowledgeEvolutionCoreSurfacePlan | None = None,
) -> KnowledgeEvolutionCoreSurfaceEntry | None:
    """Return one knowledge evolution core surface entry without activation."""

    source_plan = plan or build_knowledge_evolution_core_surface()
    for entry in source_plan.entries:
        if entry.core_surface_id == core_surface_id:
            return entry
    return None


def knowledge_evolution_core_surface_entries_for_status(
    status: KnowledgeEvolutionCoreSurfaceStatus,
    plan: KnowledgeEvolutionCoreSurfacePlan | None = None,
) -> tuple[KnowledgeEvolutionCoreSurfaceEntry, ...]:
    """Return knowledge evolution core entries by advisory status."""

    source_plan = plan or build_knowledge_evolution_core_surface()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def knowledge_evolution_core_surface_entries_for_confidence(
    confidence: KnowledgeEvolutionCoreSurfaceConfidence,
    plan: KnowledgeEvolutionCoreSurfacePlan | None = None,
) -> tuple[KnowledgeEvolutionCoreSurfaceEntry, ...]:
    """Return knowledge evolution core entries by confidence band."""

    source_plan = plan or build_knowledge_evolution_core_surface()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


def _build_source_plans(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> tuple[BaseModel, ...]:
    kwargs = {
        "route": route_name,
        "task_type": task_type,
        "execution_mode_id": execution_mode_id,
    }
    return (
        build_automatic_kb_updates(**kwargs),
        build_documentation_intelligence(**kwargs),
        build_embedding_refresh(**kwargs),
        build_retrieval_evolution(**kwargs),
        build_ranking_optimization(**kwargs),
        build_knowledge_health_monitoring(**kwargs),
        build_knowledge_quality_scoring(**kwargs),
        build_knowledge_gap_detection(**kwargs),
        build_knowledge_conflict_resolver(**kwargs),
        build_knowledge_drift_detection(**kwargs),
        build_source_reliability_engine(**kwargs),
        build_knowledge_consolidation(**kwargs),
        build_knowledge_lifecycle_management(**kwargs),
        build_knowledge_provenance_evolution(**kwargs),
        build_knowledge_versioning(**kwargs),
        build_knowledge_snapshot_engine(**kwargs),
        build_knowledge_rollback(**kwargs),
        build_knowledge_freshness_tracking(**kwargs),
        build_knowledge_trust_score(**kwargs),
    )


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    plan_by_role: dict[str, BaseModel],
) -> tuple[KnowledgeEvolutionCoreSurfaceEntry, ...]:
    return (
        _entry(
            kind="knowledge_update_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="knowledge_updates",
            plan_by_role=plan_by_role,
            surface_coverage_score=90,
            source_traceability_score=90,
            governance_alignment_score=84,
            activation_risk_score=40,
            governance_weight=130,
        ),
        _entry(
            kind="retrieval_quality_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="retrieval_quality",
            plan_by_role=plan_by_role,
            surface_coverage_score=84,
            source_traceability_score=86,
            governance_alignment_score=80,
            activation_risk_score=38,
            governance_weight=110,
        ),
        _entry(
            kind="knowledge_integrity_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="knowledge_integrity",
            plan_by_role=plan_by_role,
            surface_coverage_score=88,
            source_traceability_score=88,
            governance_alignment_score=86,
            activation_risk_score=42,
            governance_weight=125,
        ),
        _entry(
            kind="knowledge_governance_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="knowledge_governance",
            plan_by_role=plan_by_role,
            surface_coverage_score=66,
            source_traceability_score=70,
            governance_alignment_score=78,
            activation_risk_score=28,
            governance_weight=75,
        ),
        _entry(
            kind="recovery_trust_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            axis="recovery_trust",
            plan_by_role=plan_by_role,
            surface_coverage_score=54,
            source_traceability_score=58,
            governance_alignment_score=90,
            activation_risk_score=18,
            governance_weight=65,
        ),
    )


def _entry(
    *,
    kind: KnowledgeEvolutionCoreSurfaceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    axis: KnowledgeEvolutionCoreSurfaceAxis,
    plan_by_role: dict[str, BaseModel],
    surface_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> KnowledgeEvolutionCoreSurfaceEntry:
    source_roles = _SURFACE_ROLE_GROUPS[kind]
    source_plans = tuple(plan_by_role[role] for role in source_roles)
    source_item_ids = _source_item_ids_for_plans(source_plans)
    roadmap_items = tuple(
        item
        for plan in source_plans
        for item in tuple(plan.model_dump()["covered_roadmap_items"])
    )
    score = _core_surface_score(
        surface_coverage_score=surface_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
    )
    return KnowledgeEvolutionCoreSurfaceEntry(
        core_surface_id=f"knowledge_evolution_core::{kind}",
        surface_kind=kind,
        status=_core_surface_status(score),
        confidence=_core_surface_confidence(score),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        surface_axis=axis,
        roadmap_items=roadmap_items,
        roadmap_item_count=len(roadmap_items),
        source_plan_roles=source_roles,
        source_serialization_versions=tuple(
            str(plan.serialization_version) for plan in source_plans
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        surface_summary=_surface_summary(kind),
        surface_coverage_score=surface_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
        core_surface_score=score,
        hitl_required_before_core_surface_activation=True,
        context_tags=_context_tags(kind, axis),
        explainability_notes=_explainability_notes(kind, source_roles),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"roadmap_item_count:{len(roadmap_items)}",
            f"source_plan_count:{len(source_plans)}",
            f"source_item_count:{len(source_item_ids)}",
            f"surface_axis:{axis}",
            f"surface_coverage_score:{surface_coverage_score}",
            f"source_traceability_score:{source_traceability_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"activation_risk_score:{activation_risk_score}",
            "hitl_required_before_core_surface_activation:true",
        ),
    )


def _source_item_ids_for_plans(plans: tuple[BaseModel, ...]) -> tuple[str, ...]:
    return tuple(item_id for plan in plans for item_id in _source_item_ids(plan))


def _source_item_ids(plan: BaseModel) -> tuple[str, ...]:
    payload = plan.model_dump()
    if "signal_ids" in payload:
        return tuple(str(item_id) for item_id in payload["signal_ids"])
    return tuple(str(item_id) for item_id in payload["candidate_ids"])


def _core_surface_score(
    *,
    surface_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            surface_coverage_score * 3
            + source_traceability_score * 3
            + governance_alignment_score * 2
            + activation_risk_score * 2
            + governance_weight,
        ),
    )


def _core_surface_status(score: int) -> KnowledgeEvolutionCoreSurfaceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _core_surface_confidence(score: int) -> KnowledgeEvolutionCoreSurfaceConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_core_surface_score(
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...],
) -> int:
    base = sum(entry.core_surface_score for entry in entries) // len(entries)
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    review_count = len(_entry_ids_for_status(entries, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_core_surface_posture(
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...],
) -> KnowledgeEvolutionCoreSurfacePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...],
    status: KnowledgeEvolutionCoreSurfaceStatus,
) -> tuple[str, ...]:
    return tuple(entry.core_surface_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...],
    *confidences: KnowledgeEvolutionCoreSurfaceConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.core_surface_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[KnowledgeEvolutionCoreSurfaceEntry, ...],
) -> tuple[str, ...]:
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    return (
        "inspect_knowledge_evolution_core_surface_metadata",
        "verify_all_v6_3_roadmap_items_remain_individually_traceable",
        "review_source_surface_metadata_before_any_core_activation",
        "require_hitl_before_core_activation_storage_retrieval_or_runtime_mutation",
        f"guarded_entry_count:{guarded_count}",
    )


def _context_tags(
    kind: KnowledgeEvolutionCoreSurfaceKind,
    axis: KnowledgeEvolutionCoreSurfaceAxis,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "knowledge_evolution_core_surface",
        f"axis:{axis}",
        f"kind:{kind}",
        "metadata_only",
    )


def _explainability_notes(
    kind: KnowledgeEvolutionCoreSurfaceKind,
    source_roles: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"surface:{kind}",
        f"source_plan_count:{len(source_roles)}",
        "composes_v6_3_roadmap_metadata",
        "keeps_core_surface_activation_disabled",
        "requires_human_review_before_activation_storage_retrieval_or_mutation",
    )


def _entry_actions(kind: KnowledgeEvolutionCoreSurfaceKind) -> tuple[str, ...]:
    base_actions = (
        "inspect_knowledge_evolution_core_surface_entry_metadata",
        "verify_entry_roadmap_traceability",
        "keep_knowledge_evolution_core_surface_activation_disabled",
        "require_hitl_before_core_surface_action",
    )
    if kind == "retrieval_quality_surface":
        return base_actions + ("review_retrieval_quality_metadata",)
    if kind == "knowledge_integrity_surface":
        return base_actions + ("review_knowledge_integrity_metadata",)
    if kind == "knowledge_governance_surface":
        return base_actions + ("review_knowledge_governance_metadata",)
    if kind == "recovery_trust_surface":
        return base_actions + ("review_recovery_and_trust_metadata",)
    return base_actions + ("review_knowledge_update_metadata",)


def _surface_summary(kind: KnowledgeEvolutionCoreSurfaceKind) -> str:
    summaries: dict[KnowledgeEvolutionCoreSurfaceKind, str] = {
        "knowledge_update_surface": (
            "Advisory core surface grouping automatic KB update, "
            "documentation intelligence, and embedding refresh metadata."
        ),
        "retrieval_quality_surface": (
            "Advisory core surface grouping retrieval evolution, ranking, "
            "health monitoring, and quality scoring metadata."
        ),
        "knowledge_integrity_surface": (
            "Advisory core surface grouping gap, conflict, drift, and source "
            "reliability metadata."
        ),
        "knowledge_governance_surface": (
            "Advisory core surface grouping consolidation, lifecycle, "
            "provenance, and versioning metadata."
        ),
        "recovery_trust_surface": (
            "Advisory core surface grouping snapshot, rollback, freshness, and "
            "trust score metadata."
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
