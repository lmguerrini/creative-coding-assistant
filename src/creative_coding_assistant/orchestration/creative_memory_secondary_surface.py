"""V6.2 advisory creative memory secondary surface metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_policy_engine import (
    ControlledAdaptiveExecutionPlan,
    evaluate_adaptive_execution_policy,
)
from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.creative_memory_core_surface import (
    CreativeMemoryCoreSurfacePlan,
    build_creative_memory_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CreativeMemorySecondarySurfaceKind = Literal[
    "preference_learning_surface",
    "user_modeling_surface",
    "memory_operations_surface",
    "memory_governance_surface",
    "taste_evolution_surface",
]
CreativeMemorySecondarySurfaceStatus = Literal[
    "candidate",
    "review_required",
    "guarded",
]
CreativeMemorySecondarySurfaceConfidence = Literal["low", "medium", "high", "guarded"]
CreativeMemorySecondarySurfacePosture = Literal[
    "candidate",
    "review_required",
    "guarded",
]
CreativeMemorySecondarySurfaceAxis = Literal[
    "preference_learning",
    "user_modeling",
    "memory_operations",
    "memory_governance",
    "taste_evolution",
]

CREATIVE_MEMORY_SECONDARY_ENTRY_SERIALIZATION_VERSION = (
    "creative_memory_secondary_surface_entry.v1"
)
CREATIVE_MEMORY_SECONDARY_PLAN_SERIALIZATION_VERSION = (
    "creative_memory_secondary_surface_plan.v1"
)
CREATIVE_MEMORY_SECONDARY_AUTHORITY_BOUNDARY = (
    "V6.2 Creative Memory Secondary Surface exposes supporting creative memory "
    "roadmap items as inspectable advisory metadata composed from the V6.2 core "
    "memory surface, V6.1 adaptive learning metadata, and V5 controlled "
    "execution policy metadata; it does not activate secondary surfaces, learn "
    "preferences, write preference storage, create or update user models, apply "
    "user models, consolidate memory, execute memory retrieval, execute "
    "retrieval planning, resolve memory conflicts, generate memory "
    "explainability, enforce memory safety policies, apply creative taste "
    "models, apply creative preference evolution, apply personalization, "
    "change provider or model routing, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, trigger retries or refinements, "
    "mutate prompts, modify generated output, or apply Runtime Evolution."
)

_ROADMAP_ITEMS = (
    "Preference Learning",
    "User Modeling",
    "Memory Consolidation",
    "Memory Retrieval Intelligence",
    "Memory Retrieval Planner",
    "Memory Conflict Resolution",
    "Memory Explainability",
    "Memory Safety Policies",
    "Creative Taste Model",
    "Creative Preference Evolution",
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "secondary_surface_activation",
    "preference_learning_execution",
    "automatic_preference_learning",
    "preference_storage_write",
    "user_model_creation",
    "user_model_update",
    "user_model_application",
    "memory_consolidation_execution",
    "memory_retrieval_execution",
    "retrieval_planner_execution",
    "memory_conflict_resolution_execution",
    "memory_explainability_generation",
    "memory_safety_policy_enforcement",
    "creative_taste_model_application",
    "creative_preference_evolution_application",
    "creative_dna_application",
    "personalization_application",
    "memory_storage_write",
    "memory_record_creation",
    "memory_record_update",
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


class CreativeMemorySecondarySurfaceEntry(BaseModel):
    """One advisory entry in the V6.2 creative memory secondary surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    secondary_surface_id: str = Field(min_length=1, max_length=180)
    surface_kind: CreativeMemorySecondarySurfaceKind
    status: CreativeMemorySecondarySurfaceStatus
    confidence: CreativeMemorySecondarySurfaceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    surface_axis: CreativeMemorySecondarySurfaceAxis
    roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=4)
    roadmap_item_count: int = Field(ge=1, le=4)
    source_plan_roles: tuple[str, ...] = Field(min_length=1, max_length=3)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=1,
        max_length=3,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=15)
    source_item_count: int = Field(ge=1, le=15)
    surface_summary: str = Field(min_length=1, max_length=360)
    roadmap_coverage_score: int = Field(ge=0, le=100)
    source_traceability_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    v5_v6_composition_score: int = Field(ge=0, le=100)
    activation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    secondary_surface_score: int = Field(ge=0, le=1_000)
    hitl_required_before_secondary_surface_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=36,
    )
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    v5_policy_foundation_used: Literal[True] = True
    v6_memory_foundation_used: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    preference_learning_execution_implemented: Literal[False] = False
    preference_storage_write_implemented: Literal[False] = False
    user_model_creation_implemented: Literal[False] = False
    user_model_update_implemented: Literal[False] = False
    user_model_application_implemented: Literal[False] = False
    memory_consolidation_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    retrieval_planner_execution_implemented: Literal[False] = False
    memory_conflict_resolution_execution_implemented: Literal[False] = False
    memory_explainability_generation_implemented: Literal[False] = False
    memory_safety_policy_enforcement_implemented: Literal[False] = False
    creative_taste_model_application_implemented: Literal[False] = False
    preference_evolution_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
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
    serialization_version: Literal["creative_memory_secondary_surface_entry.v1"] = (
        CREATIVE_MEMORY_SECONDARY_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        expected_id = f"creative_memory_secondary::{self.surface_kind}"
        if self.secondary_surface_id != expected_id:
            raise ValueError("secondary_surface_id must match surface_kind")
        if self.roadmap_item_count != len(self.roadmap_items):
            raise ValueError("roadmap_item_count must match roadmap_items")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.secondary_surface_score != _secondary_surface_score(
            roadmap_coverage_score=self.roadmap_coverage_score,
            source_traceability_score=self.source_traceability_score,
            governance_alignment_score=self.governance_alignment_score,
            v5_v6_composition_score=self.v5_v6_composition_score,
            activation_risk_score=self.activation_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("secondary_surface_score must combine source scores")
        if self.status != _secondary_surface_status(self.secondary_surface_score):
            raise ValueError("status must match secondary_surface_score")
        if self.confidence != _secondary_surface_confidence(
            self.secondary_surface_score
        ):
            raise ValueError("confidence must match secondary_surface_score")
        if not self.hitl_required_before_secondary_surface_activation:
            raise ValueError("secondary surface activation requires HITL posture")
        return self


class CreativeMemorySecondarySurfacePlan(BaseModel):
    """Bounded V6.2 advisory creative memory secondary surface plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_memory_secondary_surface"] = (
        "creative_memory_secondary_surface"
    )
    serialization_version: Literal["creative_memory_secondary_surface_plan.v1"] = (
        CREATIVE_MEMORY_SECONDARY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_MEMORY_SECONDARY_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_plan_roles: tuple[str, ...] = Field(min_length=3, max_length=3)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=3,
        max_length=3,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=15, max_length=15)
    source_item_count: int = Field(ge=15, le=15)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=10, max_length=10)
    covered_roadmap_item_count: int = Field(ge=10, le=10)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...] = Field(
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
    activated_secondary_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    learned_preference_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    created_user_model_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    consolidated_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    executed_retrieval_plan_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    resolved_conflict_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    enforced_policy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    applied_taste_model_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_output_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    entry_count: int = Field(ge=5, le=5)
    candidate_entry_count: int = Field(ge=0, le=5)
    review_required_entry_count: int = Field(ge=0, le=5)
    guarded_entry_count: int = Field(ge=0, le=5)
    high_confidence_entry_count: int = Field(ge=0, le=5)
    hitl_required_entry_count: int = Field(ge=0, le=5)
    highest_secondary_surface_score: int = Field(ge=0, le=1_000)
    overall_secondary_surface_score: int = Field(ge=0, le=1_000)
    overall_secondary_surface_posture: CreativeMemorySecondarySurfacePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=36,
    )
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    secondary_roadmap_items_covered: Literal[True] = True
    v5_policy_foundation_used: Literal[True] = True
    v6_memory_foundation_used: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    preference_learning_execution_implemented: Literal[False] = False
    preference_storage_write_implemented: Literal[False] = False
    user_model_creation_implemented: Literal[False] = False
    user_model_update_implemented: Literal[False] = False
    user_model_application_implemented: Literal[False] = False
    memory_consolidation_execution_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    retrieval_planner_execution_implemented: Literal[False] = False
    memory_conflict_resolution_execution_implemented: Literal[False] = False
    memory_explainability_generation_implemented: Literal[False] = False
    memory_safety_policy_enforcement_implemented: Literal[False] = False
    creative_taste_model_application_implemented: Literal[False] = False
    preference_evolution_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
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

    @model_validator(mode="after")
    def _plan_matches_entries(self) -> Self:
        derived_entry_ids = tuple(entry.secondary_surface_id for entry in self.entries)
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
            entry.secondary_surface_id
            for entry in self.entries
            if entry.hitl_required_before_secondary_surface_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        for blocked_ids, label in (
            (self.activated_secondary_surface_ids, "activated_secondary_surface_ids"),
            (self.learned_preference_ids, "learned_preference_ids"),
            (self.created_user_model_ids, "created_user_model_ids"),
            (self.consolidated_memory_ids, "consolidated_memory_ids"),
            (self.executed_retrieval_plan_ids, "executed_retrieval_plan_ids"),
            (self.resolved_conflict_ids, "resolved_conflict_ids"),
            (self.enforced_policy_ids, "enforced_policy_ids"),
            (self.applied_taste_model_ids, "applied_taste_model_ids"),
            (self.mutated_output_ids, "mutated_output_ids"),
        ):
            if blocked_ids:
                raise ValueError(f"{label} must remain empty")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError(
                "covered_roadmap_items must match V6.2 secondary roadmap"
            )
        flattened_items = tuple(
            item for entry in self.entries for item in entry.roadmap_items
        )
        if flattened_items != self.covered_roadmap_items:
            raise ValueError("entry roadmap_items must cover secondary roadmap")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap items")
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
        if self.highest_secondary_surface_score != max(
            entry.secondary_surface_score for entry in self.entries
        ):
            raise ValueError("highest_secondary_surface_score must match entries")
        if self.overall_secondary_surface_score != _overall_secondary_surface_score(
            self.entries
        ):
            raise ValueError("overall_secondary_surface_score must match entries")
        if self.overall_secondary_surface_posture != _overall_secondary_surface_posture(
            self.entries
        ):
            raise ValueError("overall_secondary_surface_posture must match entries")
        plan_roles = set(self.source_plan_roles)
        plan_items = set(self.source_item_ids)
        for entry in self.entries:
            if entry.route_name != self.route_name:
                raise ValueError("entry route_name must match plan")
            if not set(entry.source_plan_roles).issubset(plan_roles):
                raise ValueError("entry source_plan_roles must be declared")
            if not set(entry.source_item_ids).issubset(plan_items):
                raise ValueError("entry source_item_ids must be declared")
        return self


def build_creative_memory_secondary_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    core_surface: CreativeMemoryCoreSurfacePlan | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
    adaptive_execution_policy: ControlledAdaptiveExecutionPlan | None = None,
) -> CreativeMemorySecondarySurfacePlan:
    """Build the V6.2 secondary surface without activating memory behavior."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    core_plan = core_surface or build_creative_memory_core_surface(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    learning_plan = adaptive_learning or evaluate_adaptive_learning_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    policy_plan = adaptive_execution_policy or evaluate_adaptive_execution_policy(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    sources = (
        _Source(
            role="creative_memory_core_surface",
            serialization_version=core_plan.serialization_version,
            item_ids=core_plan.entry_ids,
        ),
        _Source(
            role="adaptive_learning_engine",
            serialization_version=learning_plan.serialization_version,
            item_ids=learning_plan.signal_ids,
        ),
        _Source(
            role="adaptive_execution_policy_engine",
            serialization_version=policy_plan.serialization_version,
            item_ids=policy_plan.option_ids,
        ),
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=sources,
    )
    source_item_ids = _source_item_ids(sources)
    return CreativeMemorySecondarySurfacePlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_plan_roles=tuple(source.role for source in sources),
        source_plan_serialization_versions=tuple(
            source.serialization_version for source in sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        covered_roadmap_items=_ROADMAP_ITEMS,
        covered_roadmap_item_count=len(_ROADMAP_ITEMS),
        execution_mode_ids=execution_modes.execution_mode_ids,
        entries=entries,
        entry_ids=tuple(entry.secondary_surface_id for entry in entries),
        candidate_entry_ids=_entry_ids_for_status(entries, "candidate"),
        review_required_entry_ids=_entry_ids_for_status(entries, "review_required"),
        guarded_entry_ids=_entry_ids_for_status(entries, "guarded"),
        high_confidence_entry_ids=_entry_ids_for_confidence(
            entries,
            "high",
            "guarded",
        ),
        hitl_required_entry_ids=tuple(
            entry.secondary_surface_id
            for entry in entries
            if entry.hitl_required_before_secondary_surface_activation
        ),
        activated_secondary_surface_ids=(),
        learned_preference_ids=(),
        created_user_model_ids=(),
        consolidated_memory_ids=(),
        executed_retrieval_plan_ids=(),
        resolved_conflict_ids=(),
        enforced_policy_ids=(),
        applied_taste_model_ids=(),
        mutated_output_ids=(),
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
            1
            for entry in entries
            if entry.hitl_required_before_secondary_surface_activation
        ),
        highest_secondary_surface_score=max(
            entry.secondary_surface_score for entry in entries
        ),
        overall_secondary_surface_score=_overall_secondary_surface_score(entries),
        overall_secondary_surface_posture=_overall_secondary_surface_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def creative_memory_secondary_surface_entry_by_id(
    entry_id: str,
    plan: CreativeMemorySecondarySurfacePlan | None = None,
) -> CreativeMemorySecondarySurfaceEntry | None:
    """Return one secondary surface entry without activating the surface."""

    source_plan = plan or build_creative_memory_secondary_surface()
    for entry in source_plan.entries:
        if entry.secondary_surface_id == entry_id:
            return entry
    return None


def creative_memory_secondary_surface_entries_for_status(
    status: CreativeMemorySecondarySurfaceStatus,
    plan: CreativeMemorySecondarySurfacePlan | None = None,
) -> tuple[CreativeMemorySecondarySurfaceEntry, ...]:
    """Return secondary surface entries by advisory status."""

    source_plan = plan or build_creative_memory_secondary_surface()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def creative_memory_secondary_surface_entries_for_confidence(
    confidence: CreativeMemorySecondarySurfaceConfidence,
    plan: CreativeMemorySecondarySurfacePlan | None = None,
) -> tuple[CreativeMemorySecondarySurfaceEntry, ...]:
    """Return secondary surface entries by confidence band."""

    source_plan = plan or build_creative_memory_secondary_surface()
    return tuple(
        entry for entry in source_plan.entries if entry.confidence == confidence
    )


class _Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    serialization_version: str
    item_ids: tuple[str, ...]


def _entries(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    sources: tuple[_Source, ...],
) -> tuple[CreativeMemorySecondarySurfaceEntry, ...]:
    return (
        _entry(
            kind="preference_learning_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="preference_learning",
            roadmap_items=("Preference Learning",),
            source_roles=("creative_memory_core_surface", "adaptive_learning_engine"),
            sources=sources,
            roadmap_coverage_score=82,
            source_traceability_score=78,
            governance_alignment_score=82,
            v5_v6_composition_score=88,
            activation_risk_score=45,
            governance_weight=150,
        ),
        _entry(
            kind="user_modeling_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="user_modeling",
            roadmap_items=("User Modeling",),
            source_roles=(
                "creative_memory_core_surface",
                "adaptive_learning_engine",
                "adaptive_execution_policy_engine",
            ),
            sources=sources,
            roadmap_coverage_score=78,
            source_traceability_score=76,
            governance_alignment_score=80,
            v5_v6_composition_score=84,
            activation_risk_score=42,
            governance_weight=140,
        ),
        _entry(
            kind="memory_operations_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="memory_operations",
            roadmap_items=(
                "Memory Consolidation",
                "Memory Retrieval Intelligence",
                "Memory Retrieval Planner",
                "Memory Conflict Resolution",
            ),
            source_roles=(
                "creative_memory_core_surface",
                "adaptive_execution_policy_engine",
            ),
            sources=sources,
            roadmap_coverage_score=72,
            source_traceability_score=77,
            governance_alignment_score=75,
            v5_v6_composition_score=77,
            activation_risk_score=38,
            governance_weight=120,
        ),
        _entry(
            kind="memory_governance_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="memory_governance",
            roadmap_items=("Memory Explainability", "Memory Safety Policies"),
            source_roles=tuple(source.role for source in sources),
            sources=sources,
            roadmap_coverage_score=68,
            source_traceability_score=70,
            governance_alignment_score=85,
            v5_v6_composition_score=75,
            activation_risk_score=32,
            governance_weight=115,
        ),
        _entry(
            kind="taste_evolution_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="taste_evolution",
            roadmap_items=("Creative Taste Model", "Creative Preference Evolution"),
            source_roles=("creative_memory_core_surface", "adaptive_learning_engine"),
            sources=sources,
            roadmap_coverage_score=54,
            source_traceability_score=55,
            governance_alignment_score=58,
            v5_v6_composition_score=63,
            activation_risk_score=24,
            governance_weight=85,
        ),
    )


def _entry(
    *,
    kind: CreativeMemorySecondarySurfaceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    surface_axis: CreativeMemorySecondarySurfaceAxis,
    roadmap_items: tuple[str, ...],
    source_roles: tuple[str, ...],
    sources: tuple[_Source, ...],
    roadmap_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    v5_v6_composition_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> CreativeMemorySecondarySurfaceEntry:
    selected_sources = _sources_for_roles(sources, source_roles)
    source_item_ids = _source_item_ids(selected_sources)
    score = _secondary_surface_score(
        roadmap_coverage_score=roadmap_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        v5_v6_composition_score=v5_v6_composition_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
    )
    status = _secondary_surface_status(score)
    confidence = _secondary_surface_confidence(score)
    return CreativeMemorySecondarySurfaceEntry(
        secondary_surface_id=f"creative_memory_secondary::{kind}",
        surface_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        surface_axis=surface_axis,
        roadmap_items=roadmap_items,
        roadmap_item_count=len(roadmap_items),
        source_plan_roles=tuple(source.role for source in selected_sources),
        source_serialization_versions=tuple(
            source.serialization_version for source in selected_sources
        ),
        source_item_ids=source_item_ids,
        source_item_count=len(source_item_ids),
        surface_summary=_surface_summary(kind),
        roadmap_coverage_score=roadmap_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        v5_v6_composition_score=v5_v6_composition_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
        secondary_surface_score=score,
        hitl_required_before_secondary_surface_activation=True,
        context_tags=_context_tags(kind, surface_axis),
        explainability_notes=_explainability_notes(kind, selected_sources),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"source_plan_count:{len(selected_sources)}",
            f"source_item_count:{len(source_item_ids)}",
            f"roadmap_item_count:{len(roadmap_items)}",
            f"surface_axis:{surface_axis}",
            f"roadmap_coverage_score:{roadmap_coverage_score}",
            f"source_traceability_score:{source_traceability_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"v5_v6_composition_score:{v5_v6_composition_score}",
            f"activation_risk_score:{activation_risk_score}",
            "hitl_required_before_secondary_surface_activation:true",
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


def _secondary_surface_score(
    *,
    roadmap_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    v5_v6_composition_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            roadmap_coverage_score * 2
            + source_traceability_score * 2
            + governance_alignment_score * 2
            + v5_v6_composition_score * 2
            + activation_risk_score
            + governance_weight,
        ),
    )


def _secondary_surface_status(score: int) -> CreativeMemorySecondarySurfaceStatus:
    if score >= 840:
        return "guarded"
    if score >= 620:
        return "review_required"
    return "candidate"


def _secondary_surface_confidence(
    score: int,
) -> CreativeMemorySecondarySurfaceConfidence:
    if score >= 840:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_secondary_surface_score(
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...],
) -> int:
    base = sum(entry.secondary_surface_score for entry in entries) // len(entries)
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    review_count = len(_entry_ids_for_status(entries, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_secondary_surface_posture(
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...],
) -> CreativeMemorySecondarySurfacePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...],
    status: CreativeMemorySecondarySurfaceStatus,
) -> tuple[str, ...]:
    return tuple(
        entry.secondary_surface_id for entry in entries if entry.status == status
    )


def _entry_ids_for_confidence(
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...],
    *confidences: CreativeMemorySecondarySurfaceConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.secondary_surface_id
        for entry in entries
        if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[CreativeMemorySecondarySurfaceEntry, ...],
) -> tuple[str, ...]:
    guarded_entry_count = len(_entry_ids_for_status(entries, "guarded"))
    return (
        "inspect_creative_memory_secondary_surface",
        "verify_secondary_roadmap_metadata_coverage",
        "require_hitl_before_secondary_surface_activation",
        f"review_guarded_secondary_surface_count:{guarded_entry_count}",
    )


def _surface_summary(kind: CreativeMemorySecondarySurfaceKind) -> str:
    summaries = {
        "preference_learning_surface": (
            "Summarizes advisory preference learning posture over core memory "
            "and learning metadata."
        ),
        "user_modeling_surface": (
            "Summarizes advisory user modeling posture over memory, learning, "
            "and policy metadata."
        ),
        "memory_operations_surface": (
            "Summarizes advisory consolidation, retrieval, retrieval planning, "
            "and conflict posture."
        ),
        "memory_governance_surface": (
            "Summarizes advisory explainability and safety policy posture."
        ),
        "taste_evolution_surface": (
            "Summarizes advisory creative taste and preference evolution posture."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: CreativeMemorySecondarySurfaceKind,
    surface_axis: CreativeMemorySecondarySurfaceAxis,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "secondary_surface",
        surface_axis,
        kind.removesuffix("_surface"),
    )


def _explainability_notes(
    kind: CreativeMemorySecondarySurfaceKind,
    sources: tuple[_Source, ...],
) -> tuple[str, ...]:
    return (
        f"secondary_surface_kind:{kind}",
        "source_roles:" + ",".join(source.role for source in sources),
        "source_versions:"
        + ",".join(source.serialization_version for source in sources),
        "score_inputs:roadmap_coverage,source_traceability,governance_alignment,v5_v6_composition,activation_risk,governance",
        "activation_boundary:HITL_required_before_secondary_surface_activation",
    )


def _entry_actions(kind: CreativeMemorySecondarySurfaceKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_v5_v6_sources_before_secondary_surface_activation",
        "preserve_no_memory_runtime_mutation_boundary",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    value = str(task_type).strip()
    allowed = get_args(TaskRoutingType)
    if value not in allowed:
        raise ValueError("task_type must be a supported routing task type")
    return cast(TaskRoutingType, value)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in allowed_modes:
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
