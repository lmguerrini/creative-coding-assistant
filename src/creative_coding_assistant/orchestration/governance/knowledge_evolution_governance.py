"""V6.3 advisory knowledge evolution governance and safety metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hitl_budget_gate import (
    HitlBudgetGatePlan,
    evaluate_hitl_budget_gate,
)
from creative_coding_assistant.orchestration.knowledge_evolution_secondary_surface import (
    KnowledgeEvolutionSecondarySurfacePlan,
    build_knowledge_evolution_secondary_surface,
)
from creative_coding_assistant.orchestration.learning_governance import (
    LearningGovernancePlan,
    build_learning_governance,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_explainability import (
    RoutingExplainabilityPlan,
    explain_routing_decision,
)
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

KnowledgeEvolutionGovernanceBoundaryKind = Literal[
    "knowledge_update_governance",
    "retrieval_quality_governance",
    "knowledge_integrity_safety_governance",
    "provenance_lifecycle_hitl_governance",
    "recovery_trust_no_automation_governance",
]
KnowledgeEvolutionGovernanceStatus = Literal[
    "blocked",
    "review_required",
    "guarded",
]
KnowledgeEvolutionGovernancePriority = Literal[
    "standard",
    "elevated",
    "critical",
    "guarded",
]
KnowledgeEvolutionGovernancePosture = Literal[
    "blocked",
    "review_required",
    "guarded",
]
KnowledgeEvolutionGovernanceArea = Literal[
    "knowledge_updates",
    "retrieval_quality",
    "knowledge_integrity_safety",
    "provenance_lifecycle_hitl",
    "recovery_trust_no_automation",
]

KNOWLEDGE_EVOLUTION_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION = (
    "knowledge_evolution_governance_boundary.v1"
)
KNOWLEDGE_EVOLUTION_GOVERNANCE_PLAN_SERIALIZATION_VERSION = (
    "knowledge_evolution_governance_plan.v1"
)
KNOWLEDGE_EVOLUTION_GOVERNANCE_AUTHORITY_BOUNDARY = (
    "V6.3 Knowledge Evolution Governance and Safety describes governance, "
    "safety, HITL, explainability, and no-automation boundaries as inspectable "
    "metadata composed from the V6.3 secondary surface, V6.1 learning "
    "governance, V5.2 HITL budget gate metadata, and V5.2 routing "
    "explainability metadata; it does not enforce governance policies, enforce "
    "safety policies, emit HITL requests, request human input, activate "
    "automation, activate secondary surfaces, apply adaptive learning, apply "
    "execution policy, execute knowledge operations, execute automatic KB "
    "updates, fetch documentation, refresh embeddings, execute retrieval, "
    "mutate ranking, compute quality or trust scores, detect gaps, resolve "
    "conflicts, detect drift, score source reliability, consolidate knowledge, "
    "manage lifecycle state, mutate provenance graphs, mutate version graphs, "
    "execute snapshots, execute rollback, run freshness scans, write KB "
    "storage, update source records, apply routing decisions, change provider "
    "or model routing, execute providers, invoke agents, control workflows, "
    "mutate workflow graphs, modify generated output, or apply Runtime "
    "Evolution."
)

_SOURCE_PLAN_ROLES = (
    "knowledge_evolution_secondary_surface",
    "learning_governance",
    "hitl_budget_gate",
    "routing_explainability",
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

_BOUNDARY_ROADMAP_GROUPS: dict[
    KnowledgeEvolutionGovernanceBoundaryKind,
    tuple[str, ...],
] = {
    "knowledge_update_governance": (
        "Automatic KB Updates",
        "Documentation Intelligence",
        "Embedding Refresh",
    ),
    "retrieval_quality_governance": (
        "Retrieval Evolution",
        "Ranking Optimization",
        "Knowledge Health Monitoring",
        "Knowledge Quality Scoring",
    ),
    "knowledge_integrity_safety_governance": (
        "Knowledge Gap Detection",
        "Knowledge Conflict Resolver",
        "Knowledge Drift Detection",
        "Source Reliability Engine",
    ),
    "provenance_lifecycle_hitl_governance": (
        "Knowledge Consolidation",
        "Knowledge Lifecycle Management",
        "Knowledge Provenance Evolution",
        "Knowledge Versioning",
    ),
    "recovery_trust_no_automation_governance": (
        "Knowledge Snapshot Engine",
        "Knowledge Rollback",
        "Knowledge Freshness Tracking",
        "Knowledge Trust Score",
    ),
}

_BLOCKED_RUNTIME_BEHAVIORS = (
    "governance_policy_enforcement",
    "safety_policy_enforcement",
    "hitl_request_emission",
    "human_input_request",
    "automation_activation",
    "secondary_surface_activation",
    "adaptive_learning_application",
    "execution_policy_application",
    "knowledge_operation_execution",
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
    "routing_application",
    "provider_or_model_routing",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "generated_output_modification",
    "runtime_evolution_application",
)


class KnowledgeEvolutionGovernanceBoundary(BaseModel):
    """One advisory governance boundary for V6.3 knowledge evolution."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    boundary_id: str = Field(min_length=1, max_length=180)
    boundary_kind: KnowledgeEvolutionGovernanceBoundaryKind
    status: KnowledgeEvolutionGovernanceStatus
    priority: KnowledgeEvolutionGovernancePriority
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    governed_area: KnowledgeEvolutionGovernanceArea
    roadmap_items: tuple[str, ...] = Field(min_length=3, max_length=4)
    roadmap_item_count: int = Field(ge=3, le=4)
    source_plan_roles: tuple[str, ...] = Field(min_length=2, max_length=4)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=2,
        max_length=4,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    source_item_count: int = Field(ge=1, le=24)
    hitl_requirement_count: int = Field(ge=0, le=12)
    explainability_signal_count: int = Field(ge=0, le=12)
    no_automation_weight: int = Field(ge=0, le=240)
    safety_weight: int = Field(ge=0, le=240)
    governance_score: int = Field(ge=0, le=1_000)
    hitl_required_before_governance_application: bool
    governed_surface_summary: str = Field(min_length=1, max_length=360)
    review_requirement: str = Field(min_length=1, max_length=420)
    explainability_requirement: str = Field(min_length=1, max_length=420)
    no_automation_boundary: str = Field(min_length=1, max_length=420)
    safety_boundary: str = Field(min_length=1, max_length=420)
    governance_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    knowledge_evolution_governance_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    v5_v6_governance_sources_used: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    adaptive_learning_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
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
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["knowledge_evolution_governance_boundary.v1"] = (
        KNOWLEDGE_EVOLUTION_GOVERNANCE_BOUNDARY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _boundary_matches_contract(self) -> Self:
        expected_id = f"knowledge_evolution_governance::{self.boundary_kind}"
        if self.boundary_id != expected_id:
            raise ValueError("boundary_id must match boundary_kind")
        if self.roadmap_item_count != len(self.roadmap_items):
            raise ValueError("roadmap_item_count must match roadmap_items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.governance_score != _governance_score(
            source_item_count=self.source_item_count,
            hitl_requirement_count=self.hitl_requirement_count,
            explainability_signal_count=self.explainability_signal_count,
            no_automation_weight=self.no_automation_weight,
            safety_weight=self.safety_weight,
        ):
            raise ValueError("governance_score must combine source counts")
        if self.status != _governance_status(self.governance_score):
            raise ValueError("status must match governance_score")
        if self.priority != _governance_priority(self.governance_score, self.status):
            raise ValueError("priority must match governance_score")
        if not self.hitl_required_before_governance_application:
            raise ValueError("governance application requires HITL posture")
        return self


class KnowledgeEvolutionGovernancePlan(BaseModel):
    """Bounded V6.3 advisory knowledge evolution governance plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["knowledge_evolution_governance_safety"] = (
        "knowledge_evolution_governance_safety"
    )
    serialization_version: Literal["knowledge_evolution_governance_plan.v1"] = (
        KNOWLEDGE_EVOLUTION_GOVERNANCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=KNOWLEDGE_EVOLUTION_GOVERNANCE_AUTHORITY_BOUNDARY,
        max_length=3200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_plan_roles: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=19, max_length=19)
    source_item_count: int = Field(ge=19, le=19)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    covered_roadmap_item_count: int = Field(ge=19, le=19)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    boundaries: tuple[KnowledgeEvolutionGovernanceBoundary, ...] = Field(
        min_length=5,
        max_length=5,
    )
    boundary_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    guarded_boundary_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    hitl_required_boundary_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_governance_boundary_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    enforced_safety_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    emitted_hitl_request_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    requested_human_input_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    activated_automation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    activated_secondary_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_learning_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_execution_policy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_knowledge_operation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    boundary_count: int = Field(ge=5, le=5)
    guarded_boundary_count: int = Field(ge=0, le=5)
    hitl_required_boundary_count: int = Field(ge=0, le=5)
    highest_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_score: int = Field(ge=0, le=1_000)
    overall_governance_posture: KnowledgeEvolutionGovernancePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=44,
    )
    knowledge_evolution_governance_implemented: Literal[True] = True
    governance_boundary_metadata_implemented: Literal[True] = True
    hitl_boundary_metadata_implemented: Literal[True] = True
    explainability_boundary_metadata_implemented: Literal[True] = True
    no_automation_boundary_metadata_implemented: Literal[True] = True
    all_roadmap_items_traceable: Literal[True] = True
    v5_v6_governance_sources_used: Literal[True] = True
    governance_policy_enforcement_implemented: Literal[False] = False
    safety_policy_enforcement_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    automation_activation_implemented: Literal[False] = False
    secondary_surface_activation_implemented: Literal[False] = False
    adaptive_learning_application_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
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
    routing_application_implemented: Literal[False] = False
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
    def _plan_matches_boundaries(self) -> Self:
        derived_boundary_ids = tuple(
            boundary.boundary_id for boundary in self.boundaries
        )
        if len(set(derived_boundary_ids)) != len(derived_boundary_ids):
            raise ValueError("boundary_ids must be unique")
        if self.boundary_ids != derived_boundary_ids:
            raise ValueError("boundary_ids must match boundaries")
        if self.guarded_boundary_ids != _boundary_ids_for_status(
            self.boundaries,
            "guarded",
        ):
            raise ValueError("guarded_boundary_ids must match boundaries")
        if self.hitl_required_boundary_ids != tuple(
            boundary.boundary_id
            for boundary in self.boundaries
            if boundary.hitl_required_before_governance_application
        ):
            raise ValueError("hitl_required_boundary_ids must match boundaries")
        empty_collections = (
            (self.applied_governance_boundary_ids, "applied_governance_boundary_ids"),
            (self.enforced_safety_policy_ids, "enforced_safety_policy_ids"),
            (self.emitted_hitl_request_ids, "emitted_hitl_request_ids"),
            (self.requested_human_input_ids, "requested_human_input_ids"),
            (self.activated_automation_ids, "activated_automation_ids"),
            (self.activated_secondary_surface_ids, "activated_secondary_surface_ids"),
            (self.applied_learning_ids, "applied_learning_ids"),
            (self.applied_execution_policy_ids, "applied_execution_policy_ids"),
            (self.executed_knowledge_operation_ids, "executed_knowledge_operation_ids"),
            (self.written_kb_record_ids, "written_kb_record_ids"),
            (self.mutated_output_ids, "mutated_output_ids"),
        )
        for blocked_ids, label in empty_collections:
            if blocked_ids:
                raise ValueError(f"{label} must remain empty")
        if self.source_plan_roles != _SOURCE_PLAN_ROLES:
            raise ValueError("source_plan_roles must match governance sources")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.3 roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.boundary_count != len(self.boundaries):
            raise ValueError("boundary_count must match boundaries")
        if self.guarded_boundary_count != len(self.guarded_boundary_ids):
            raise ValueError("guarded_boundary_count must match boundaries")
        if self.hitl_required_boundary_count != len(self.hitl_required_boundary_ids):
            raise ValueError("hitl_required_boundary_count must match boundaries")
        if self.highest_governance_score != max(
            boundary.governance_score for boundary in self.boundaries
        ):
            raise ValueError("highest_governance_score must match boundaries")
        if self.overall_governance_score != _overall_governance_score(self.boundaries):
            raise ValueError("overall_governance_score must match boundaries")
        if self.overall_governance_posture != _overall_governance_posture(
            self.boundaries
        ):
            raise ValueError("overall_governance_posture must match boundaries")
        flattened_roadmap_items = tuple(
            item for boundary in self.boundaries for item in boundary.roadmap_items
        )
        if flattened_roadmap_items != self.covered_roadmap_items:
            raise ValueError("boundary roadmap_items must cover roadmap items")
        plan_roles = set(self.source_plan_roles)
        plan_items = set(self.source_item_ids)
        for boundary in self.boundaries:
            if boundary.route_name != self.route_name:
                raise ValueError("boundary route_name must match plan")
            if not set(boundary.source_plan_roles).issubset(plan_roles):
                raise ValueError("boundary source_plan_roles must be declared")
            if not set(boundary.source_item_ids).issubset(plan_items):
                raise ValueError("boundary source_item_ids must be declared")
        return self


def build_knowledge_evolution_governance(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    secondary_surface: KnowledgeEvolutionSecondarySurfacePlan | None = None,
    learning_governance: LearningGovernancePlan | None = None,
    hitl_budget_gate: HitlBudgetGatePlan | None = None,
    routing_explainability: RoutingExplainabilityPlan | None = None,
) -> KnowledgeEvolutionGovernancePlan:
    """Build V6.3 knowledge evolution governance without enforcing policies."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    secondary_plan = secondary_surface or build_knowledge_evolution_secondary_surface(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    learning_plan = learning_governance or build_learning_governance(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    hitl_plan = hitl_budget_gate or evaluate_hitl_budget_gate(route=route_name)
    explainability_plan = routing_explainability or explain_routing_decision(
        route=route_name
    )
    sources = (
        _Source(
            role="knowledge_evolution_secondary_surface",
            serialization_version=secondary_plan.serialization_version,
            item_ids=secondary_plan.entry_ids,
        ),
        _Source(
            role="learning_governance",
            serialization_version=learning_plan.serialization_version,
            item_ids=learning_plan.policy_ids,
        ),
        _Source(
            role="hitl_budget_gate",
            serialization_version=hitl_plan.serialization_version,
            item_ids=hitl_plan.gate_ids,
        ),
        _Source(
            role="routing_explainability",
            serialization_version=explainability_plan.serialization_version,
            item_ids=explainability_plan.explanation_ids,
        ),
    )
    boundaries = _boundaries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=sources,
        hitl_required_count=(
            learning_plan.hitl_required_policy_count + hitl_plan.required_count
        ),
        explanation_count=explainability_plan.explanation_count,
    )
    source_item_ids = _source_item_ids(sources)
    return KnowledgeEvolutionGovernancePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_plan_roles=tuple(source.role for source in sources),
        source_plan_serialization_versions=tuple(
            source.serialization_version for source in sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        covered_roadmap_items=secondary_plan.covered_roadmap_items,
        covered_roadmap_item_count=len(secondary_plan.covered_roadmap_items),
        execution_mode_ids=execution_modes.execution_mode_ids,
        boundaries=boundaries,
        boundary_ids=tuple(boundary.boundary_id for boundary in boundaries),
        guarded_boundary_ids=_boundary_ids_for_status(boundaries, "guarded"),
        hitl_required_boundary_ids=tuple(
            boundary.boundary_id
            for boundary in boundaries
            if boundary.hitl_required_before_governance_application
        ),
        boundary_count=len(boundaries),
        guarded_boundary_count=len(_boundary_ids_for_status(boundaries, "guarded")),
        hitl_required_boundary_count=sum(
            1
            for boundary in boundaries
            if boundary.hitl_required_before_governance_application
        ),
        highest_governance_score=max(
            boundary.governance_score for boundary in boundaries
        ),
        overall_governance_score=_overall_governance_score(boundaries),
        overall_governance_posture=_overall_governance_posture(boundaries),
        advisory_actions=_plan_actions(boundaries),
    )


def knowledge_evolution_governance_boundary_by_id(
    boundary_id: str,
    plan: KnowledgeEvolutionGovernancePlan | None = None,
) -> KnowledgeEvolutionGovernanceBoundary | None:
    """Return one governance boundary without enforcing it."""

    source_plan = plan or build_knowledge_evolution_governance()
    for boundary in source_plan.boundaries:
        if boundary.boundary_id == boundary_id:
            return boundary
    return None


def knowledge_evolution_governance_boundaries_for_status(
    status: KnowledgeEvolutionGovernanceStatus,
    plan: KnowledgeEvolutionGovernancePlan | None = None,
) -> tuple[KnowledgeEvolutionGovernanceBoundary, ...]:
    """Return governance boundaries by advisory status."""

    source_plan = plan or build_knowledge_evolution_governance()
    return tuple(
        boundary for boundary in source_plan.boundaries if boundary.status == status
    )


class _Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    serialization_version: str
    item_ids: tuple[str, ...]


def _boundaries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    sources: tuple[_Source, ...],
    hitl_required_count: int,
    explanation_count: int,
) -> tuple[KnowledgeEvolutionGovernanceBoundary, ...]:
    return (
        _boundary(
            kind="knowledge_update_governance",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            governed_area="knowledge_updates",
            source_roles=(
                "knowledge_evolution_secondary_surface",
                "learning_governance",
            ),
            sources=sources,
            hitl_requirement_count=hitl_required_count,
            explainability_signal_count=0,
            no_automation_weight=210,
            safety_weight=200,
        ),
        _boundary(
            kind="retrieval_quality_governance",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            governed_area="retrieval_quality",
            source_roles=(
                "knowledge_evolution_secondary_surface",
                "learning_governance",
                "hitl_budget_gate",
            ),
            sources=sources,
            hitl_requirement_count=hitl_required_count + 1,
            explainability_signal_count=0,
            no_automation_weight=190,
            safety_weight=185,
        ),
        _boundary(
            kind="knowledge_integrity_safety_governance",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            governed_area="knowledge_integrity_safety",
            source_roles=("learning_governance", "hitl_budget_gate"),
            sources=sources,
            hitl_requirement_count=hitl_required_count + 2,
            explainability_signal_count=0,
            no_automation_weight=220,
            safety_weight=205,
        ),
        _boundary(
            kind="provenance_lifecycle_hitl_governance",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            governed_area="provenance_lifecycle_hitl",
            source_roles=(
                "knowledge_evolution_secondary_surface",
                "learning_governance",
                "routing_explainability",
            ),
            sources=sources,
            hitl_requirement_count=hitl_required_count,
            explainability_signal_count=explanation_count,
            no_automation_weight=180,
            safety_weight=210,
        ),
        _boundary(
            kind="recovery_trust_no_automation_governance",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            governed_area="recovery_trust_no_automation",
            source_roles=tuple(source.role for source in sources),
            sources=sources,
            hitl_requirement_count=hitl_required_count + 2,
            explainability_signal_count=explanation_count,
            no_automation_weight=240,
            safety_weight=240,
        ),
    )


def _boundary(
    *,
    kind: KnowledgeEvolutionGovernanceBoundaryKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    governed_area: KnowledgeEvolutionGovernanceArea,
    source_roles: tuple[str, ...],
    sources: tuple[_Source, ...],
    hitl_requirement_count: int,
    explainability_signal_count: int,
    no_automation_weight: int,
    safety_weight: int,
) -> KnowledgeEvolutionGovernanceBoundary:
    selected_sources = _sources_for_roles(sources, source_roles)
    source_item_ids = _source_item_ids(selected_sources)
    score = _governance_score(
        source_item_count=len(source_item_ids),
        hitl_requirement_count=hitl_requirement_count,
        explainability_signal_count=explainability_signal_count,
        no_automation_weight=no_automation_weight,
        safety_weight=safety_weight,
    )
    status = _governance_status(score)
    return KnowledgeEvolutionGovernanceBoundary(
        boundary_id=f"knowledge_evolution_governance::{kind}",
        boundary_kind=kind,
        status=status,
        priority=_governance_priority(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        governed_area=governed_area,
        roadmap_items=_BOUNDARY_ROADMAP_GROUPS[kind],
        roadmap_item_count=len(_BOUNDARY_ROADMAP_GROUPS[kind]),
        source_plan_roles=tuple(source.role for source in selected_sources),
        source_serialization_versions=tuple(
            source.serialization_version for source in selected_sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        hitl_requirement_count=hitl_requirement_count,
        explainability_signal_count=explainability_signal_count,
        no_automation_weight=no_automation_weight,
        safety_weight=safety_weight,
        governance_score=score,
        hitl_required_before_governance_application=True,
        governed_surface_summary=_surface_summary(kind),
        review_requirement=_review_requirement(kind),
        explainability_requirement=_explainability_requirement(kind),
        no_automation_boundary=_no_automation_boundary(kind),
        safety_boundary=_safety_boundary(kind),
        governance_tags=_governance_tags(kind, governed_area),
        advisory_actions=_boundary_actions(kind),
        evidence=(
            f"roadmap_item_count:{len(_BOUNDARY_ROADMAP_GROUPS[kind])}",
            f"source_plan_count:{len(selected_sources)}",
            f"source_item_count:{len(source_item_ids)}",
            f"hitl_requirement_count:{hitl_requirement_count}",
            f"explainability_signal_count:{explainability_signal_count}",
            f"no_automation_weight:{no_automation_weight}",
            f"safety_weight:{safety_weight}",
            f"governed_area:{governed_area}",
            "hitl_required_before_governance_application:true",
        ),
    )


def _sources_for_roles(
    sources: tuple[_Source, ...],
    roles: tuple[str, ...],
) -> tuple[_Source, ...]:
    source_by_role = {source.role: source for source in sources}
    return tuple(source_by_role[role] for role in roles)


def _source_item_ids(sources: tuple[_Source, ...]) -> tuple[str, ...]:
    return tuple(item_id for source in sources for item_id in source.item_ids)


def _governance_score(
    *,
    source_item_count: int,
    hitl_requirement_count: int,
    explainability_signal_count: int,
    no_automation_weight: int,
    safety_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            source_item_count * 18
            + hitl_requirement_count * 65
            + explainability_signal_count * 25
            + no_automation_weight
            + safety_weight,
        ),
    )


def _governance_status(score: int) -> KnowledgeEvolutionGovernanceStatus:
    if score >= 760:
        return "guarded"
    if score >= 520:
        return "review_required"
    return "blocked"


def _governance_priority(
    score: int,
    status: KnowledgeEvolutionGovernanceStatus,
) -> KnowledgeEvolutionGovernancePriority:
    if status == "guarded":
        return "guarded"
    if score >= 640:
        return "critical"
    if score >= 520:
        return "elevated"
    return "standard"


def _overall_governance_score(
    boundaries: tuple[KnowledgeEvolutionGovernanceBoundary, ...],
) -> int:
    base = sum(boundary.governance_score for boundary in boundaries) // len(boundaries)
    guarded_count = len(_boundary_ids_for_status(boundaries, "guarded"))
    return min(1_000, base + guarded_count * 12)


def _overall_governance_posture(
    boundaries: tuple[KnowledgeEvolutionGovernanceBoundary, ...],
) -> KnowledgeEvolutionGovernancePosture:
    if any(boundary.status == "guarded" for boundary in boundaries):
        return "guarded"
    if any(boundary.status == "review_required" for boundary in boundaries):
        return "review_required"
    return "blocked"


def _boundary_ids_for_status(
    boundaries: tuple[KnowledgeEvolutionGovernanceBoundary, ...],
    status: KnowledgeEvolutionGovernanceStatus,
) -> tuple[str, ...]:
    return tuple(
        boundary.boundary_id for boundary in boundaries if boundary.status == status
    )


def _plan_actions(
    boundaries: tuple[KnowledgeEvolutionGovernanceBoundary, ...],
) -> tuple[str, ...]:
    guarded_count = len(_boundary_ids_for_status(boundaries, "guarded"))
    return (
        "inspect_knowledge_evolution_governance_boundaries",
        "verify_hitl_explainability_safety_and_no_automation_boundaries",
        "verify_all_v6_3_roadmap_items_remain_individually_traceable",
        "require_hitl_before_governance_safety_storage_or_runtime_application",
        f"review_guarded_governance_boundary_count:{guarded_count}",
    )


def _surface_summary(kind: KnowledgeEvolutionGovernanceBoundaryKind) -> str:
    summaries: dict[KnowledgeEvolutionGovernanceBoundaryKind, str] = {
        "knowledge_update_governance": (
            "Summarizes KB update, documentation, and embedding governance boundaries."
        ),
        "retrieval_quality_governance": (
            "Summarizes retrieval, ranking, health, and quality safety boundaries."
        ),
        "knowledge_integrity_safety_governance": (
            "Summarizes gap, conflict, drift, and source reliability safety boundaries."
        ),
        "provenance_lifecycle_hitl_governance": (
            "Summarizes consolidation, lifecycle, provenance, and versioning "
            "HITL boundaries."
        ),
        "recovery_trust_no_automation_governance": (
            "Summarizes snapshot, rollback, freshness, and trust "
            "no-automation boundaries."
        ),
    }
    return summaries[kind]


def _review_requirement(kind: KnowledgeEvolutionGovernanceBoundaryKind) -> str:
    return f"Manual review is required before applying {kind} to runtime behavior."


def _explainability_requirement(
    kind: KnowledgeEvolutionGovernanceBoundaryKind,
) -> str:
    return (
        f"{kind} must remain explainable through source metadata before any "
        "HITL-approved runtime application."
    )


def _no_automation_boundary(kind: KnowledgeEvolutionGovernanceBoundaryKind) -> str:
    return (
        f"{kind} cannot activate automation, routing changes, storage writes, "
        "knowledge mutation, or output mutation without explicit HITL approval."
    )


def _safety_boundary(kind: KnowledgeEvolutionGovernanceBoundaryKind) -> str:
    return (
        f"{kind} is safety metadata only and does not enforce policy, block "
        "execution, mutate retrieval, or mutate knowledge state."
    )


def _governance_tags(
    kind: KnowledgeEvolutionGovernanceBoundaryKind,
    governed_area: KnowledgeEvolutionGovernanceArea,
) -> tuple[str, ...]:
    return (
        "knowledge_evolution",
        "governance_safety",
        governed_area,
        kind.removesuffix("_governance"),
        "metadata_only",
    )


def _boundary_actions(
    kind: KnowledgeEvolutionGovernanceBoundaryKind,
) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_governance_sources_before_application",
        "preserve_hitl_explainability_safety_and_no_automation_boundaries",
        "preserve_no_knowledge_runtime_mutation_boundary",
    )


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
