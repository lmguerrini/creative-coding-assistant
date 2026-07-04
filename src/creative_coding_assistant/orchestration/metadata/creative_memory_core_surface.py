"""V6.2 advisory creative memory core surface metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.artifact_history import (
    ArtifactHistoryPlan,
    build_artifact_history,
)
from creative_coding_assistant.orchestration.creative_dna import (
    CreativeDNAPlan,
    build_creative_dna,
)
from creative_coding_assistant.orchestration.creative_lineage import (
    CreativeLineagePlan,
    build_creative_lineage,
)
from creative_coding_assistant.orchestration.creative_ontology import (
    CreativeOntologyPlan,
    build_creative_ontology,
)
from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
)
from creative_coding_assistant.orchestration.personalization_engine import (
    PersonalizationEnginePlan,
    build_personalization_engine,
)
from creative_coding_assistant.orchestration.project_memory import (
    ProjectMemoryPlan,
    build_project_memory,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.session_memory_evolution import (
    SessionMemoryEvolutionPlan,
    build_session_memory_evolution,
)
from creative_coding_assistant.orchestration.style_profiles import (
    StyleProfilePlan,
    build_style_profiles,
)
from creative_coding_assistant.orchestration.user_preferences import (
    UserPreferencesPlan,
    build_user_preferences,
)

CreativeMemoryCoreSurfaceKind = Literal[
    "memory_foundation_surface",
    "personalization_surface",
    "session_artifact_surface",
    "lineage_ontology_surface",
    "core_boundary_surface",
]
CreativeMemoryCoreSurfaceStatus = Literal["candidate", "review_required", "guarded"]
CreativeMemoryCoreSurfaceConfidence = Literal["low", "medium", "high", "guarded"]
CreativeMemoryCoreSurfacePosture = Literal[
    "candidate",
    "review_required",
    "guarded",
]
CreativeMemoryCoreSurfaceAxis = Literal[
    "memory_foundation",
    "personalization",
    "session_artifact_history",
    "lineage_ontology",
    "governance_boundary",
]

CREATIVE_MEMORY_CORE_ENTRY_SERIALIZATION_VERSION = (
    "creative_memory_core_surface_entry.v1"
)
CREATIVE_MEMORY_CORE_PLAN_SERIALIZATION_VERSION = "creative_memory_core_surface_plan.v1"
CREATIVE_MEMORY_CORE_AUTHORITY_BOUNDARY = (
    "V6.2 Creative Memory Core Surface exposes the validated creative memory "
    "metadata surfaces as inspectable advisory metadata only; it does not "
    "activate memory surfaces, write memory storage, execute memory retrieval, "
    "create or update records, persist preferences, learn preferences, apply "
    "personalization, apply Creative DNA, persist artifact history, infer "
    "creative lineage, infer ontology relationships, mutate taxonomies, "
    "materialize semantic graphs, change provider or model routing, execute "
    "providers, invoke agents, control workflows, mutate workflow graphs, "
    "trigger retries or refinements, mutate prompts, modify generated output, "
    "or apply Runtime Evolution."
)

_ROADMAP_ITEMS = (
    "Long-term Creative Memory",
    "User Preferences",
    "Style Profiles",
    "Project Memory",
    "Creative DNA",
    "Personalization Engine",
    "Session Memory Evolution",
    "Artifact History",
    "Creative Lineage",
    "Creative Ontology",
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "core_surface_activation",
    "memory_storage_write",
    "memory_retrieval_execution",
    "memory_record_creation",
    "memory_record_update",
    "preference_storage_write",
    "automatic_preference_learning",
    "automatic_preference_mutation",
    "automatic_personalization_application",
    "creative_dna_application",
    "artifact_history_persistence",
    "creative_lineage_inference",
    "ontology_relationship_inference",
    "taxonomy_mutation",
    "semantic_graph_materialization",
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


class CreativeMemoryCoreSurfaceEntry(BaseModel):
    """One advisory entry in the V6.2 creative memory core surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    core_surface_id: str = Field(min_length=1, max_length=180)
    surface_kind: CreativeMemoryCoreSurfaceKind
    status: CreativeMemoryCoreSurfaceStatus
    confidence: CreativeMemoryCoreSurfaceConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    surface_axis: CreativeMemoryCoreSurfaceAxis
    source_plan_roles: tuple[str, ...] = Field(min_length=1, max_length=10)
    source_serialization_versions: tuple[str, ...] = Field(
        min_length=1,
        max_length=10,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=60)
    source_item_count: int = Field(ge=1, le=60)
    surface_summary: str = Field(min_length=1, max_length=360)
    surface_coverage_score: int = Field(ge=0, le=100)
    source_traceability_score: int = Field(ge=0, le=100)
    governance_alignment_score: int = Field(ge=0, le=100)
    activation_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    core_surface_score: int = Field(ge=0, le=1_000)
    hitl_required_before_surface_activation: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_record_creation_implemented: Literal[False] = False
    memory_record_update_implemented: Literal[False] = False
    preference_storage_write_implemented: Literal[False] = False
    automatic_preference_learning_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    artifact_history_persistence_implemented: Literal[False] = False
    creative_lineage_inference_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    taxonomy_mutation_implemented: Literal[False] = False
    semantic_graph_materialization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
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
    serialization_version: Literal["creative_memory_core_surface_entry.v1"] = (
        CREATIVE_MEMORY_CORE_ENTRY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_contract(self) -> Self:
        if self.core_surface_id != f"creative_memory_core::{self.surface_kind}":
            raise ValueError("core_surface_id must match surface_kind")
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
        if not self.hitl_required_before_surface_activation:
            raise ValueError("core surface activation requires HITL posture")
        return self


class CreativeMemoryCoreSurfacePlan(BaseModel):
    """Bounded V6.2 advisory creative memory core surface plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_memory_core_surface"] = "creative_memory_core_surface"
    serialization_version: Literal["creative_memory_core_surface_plan.v1"] = (
        CREATIVE_MEMORY_CORE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_MEMORY_CORE_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_plan_roles: tuple[str, ...] = Field(min_length=10, max_length=10)
    source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=10,
        max_length=10,
    )
    source_item_ids: tuple[str, ...] = Field(min_length=50, max_length=50)
    source_item_count: int = Field(ge=50, le=50)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=10, max_length=10)
    covered_roadmap_item_count: int = Field(ge=10, le=10)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...] = Field(
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
    persisted_memory_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    applied_personalization_ids: tuple[str, ...] = Field(
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
    highest_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_score: int = Field(ge=0, le=1_000)
    overall_core_surface_posture: CreativeMemoryCoreSurfacePosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    roadmap_core_items_covered: Literal[True] = True
    all_sources_metadata_only: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_record_creation_implemented: Literal[False] = False
    memory_record_update_implemented: Literal[False] = False
    preference_storage_write_implemented: Literal[False] = False
    automatic_preference_learning_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    artifact_history_persistence_implemented: Literal[False] = False
    creative_lineage_inference_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    taxonomy_mutation_implemented: Literal[False] = False
    semantic_graph_materialization_implemented: Literal[False] = False
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
            if entry.hitl_required_before_surface_activation
        ):
            raise ValueError("hitl_required_entry_ids must match entries")
        if self.activated_core_surface_ids:
            raise ValueError("activated_core_surface_ids must remain empty")
        if self.persisted_memory_surface_ids:
            raise ValueError("persisted_memory_surface_ids must remain empty")
        if self.applied_personalization_ids:
            raise ValueError("applied_personalization_ids must remain empty")
        if self.mutated_output_ids:
            raise ValueError("mutated_output_ids must remain empty")
        if self.source_item_count != len(self.source_item_ids):
            raise ValueError("source_item_count must match source_item_ids")
        if self.covered_roadmap_items != _ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V6.2 core roadmap")
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


def build_creative_memory_core_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
    user_preferences: UserPreferencesPlan | None = None,
    style_profiles: StyleProfilePlan | None = None,
    project_memory: ProjectMemoryPlan | None = None,
    creative_dna: CreativeDNAPlan | None = None,
    personalization_engine: PersonalizationEnginePlan | None = None,
    session_memory_evolution: SessionMemoryEvolutionPlan | None = None,
    artifact_history: ArtifactHistoryPlan | None = None,
    creative_lineage: CreativeLineagePlan | None = None,
    creative_ontology: CreativeOntologyPlan | None = None,
) -> CreativeMemoryCoreSurfacePlan:
    """Build the V6.2 core surface without activating memory behavior."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    memory_plan = long_term_memory or build_long_term_creative_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    preference_plan = user_preferences or build_user_preferences(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    style_plan = style_profiles or build_style_profiles(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        user_preferences=preference_plan,
    )
    project_plan = project_memory or build_project_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        style_profiles=style_plan,
    )
    dna_plan = creative_dna or build_creative_dna(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
        user_preferences=preference_plan,
        style_profiles=style_plan,
        project_memory=project_plan,
    )
    personalization_plan = personalization_engine or build_personalization_engine(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_dna=dna_plan,
        user_preferences=preference_plan,
        style_profiles=style_plan,
        project_memory=project_plan,
    )
    session_plan = session_memory_evolution or build_session_memory_evolution(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        personalization_engine=personalization_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
        project_memory=project_plan,
    )
    history_plan = artifact_history or build_artifact_history(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        session_memory_evolution=session_plan,
        project_memory=project_plan,
        long_term_memory=memory_plan,
    )
    lineage_plan = creative_lineage or build_creative_lineage(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        artifact_history=history_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
    )
    ontology_plan = creative_ontology or build_creative_ontology(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_lineage=lineage_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
    )
    sources = (
        _Source(
            role="long_term_creative_memory",
            serialization_version=memory_plan.serialization_version,
            item_ids=memory_plan.record_ids,
        ),
        _Source(
            role="user_preferences",
            serialization_version=preference_plan.serialization_version,
            item_ids=preference_plan.preference_ids,
        ),
        _Source(
            role="style_profiles",
            serialization_version=style_plan.serialization_version,
            item_ids=style_plan.profile_ids,
        ),
        _Source(
            role="project_memory",
            serialization_version=project_plan.serialization_version,
            item_ids=project_plan.signal_ids,
        ),
        _Source(
            role="creative_dna",
            serialization_version=dna_plan.serialization_version,
            item_ids=dna_plan.signature_ids,
        ),
        _Source(
            role="personalization_engine",
            serialization_version=personalization_plan.serialization_version,
            item_ids=personalization_plan.recommendation_ids,
        ),
        _Source(
            role="session_memory_evolution",
            serialization_version=session_plan.serialization_version,
            item_ids=session_plan.signal_ids,
        ),
        _Source(
            role="artifact_history",
            serialization_version=history_plan.serialization_version,
            item_ids=history_plan.record_ids,
        ),
        _Source(
            role="creative_lineage",
            serialization_version=lineage_plan.serialization_version,
            item_ids=lineage_plan.record_ids,
        ),
        _Source(
            role="creative_ontology",
            serialization_version=ontology_plan.serialization_version,
            item_ids=ontology_plan.concept_ids,
        ),
    )
    entries = _entries(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        sources=sources,
    )
    source_item_ids = _source_item_ids(sources)
    return CreativeMemoryCoreSurfacePlan(
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
            if entry.hitl_required_before_surface_activation
        ),
        activated_core_surface_ids=(),
        persisted_memory_surface_ids=(),
        applied_personalization_ids=(),
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
            1 for entry in entries if entry.hitl_required_before_surface_activation
        ),
        highest_core_surface_score=max(entry.core_surface_score for entry in entries),
        overall_core_surface_score=_overall_core_surface_score(entries),
        overall_core_surface_posture=_overall_core_surface_posture(entries),
        advisory_actions=_plan_actions(entries),
    )


def creative_memory_core_surface_entry_by_id(
    entry_id: str,
    plan: CreativeMemoryCoreSurfacePlan | None = None,
) -> CreativeMemoryCoreSurfaceEntry | None:
    """Return one core surface entry without activating the surface."""

    source_plan = plan or build_creative_memory_core_surface()
    for entry in source_plan.entries:
        if entry.core_surface_id == entry_id:
            return entry
    return None


def creative_memory_core_surface_entries_for_status(
    status: CreativeMemoryCoreSurfaceStatus,
    plan: CreativeMemoryCoreSurfacePlan | None = None,
) -> tuple[CreativeMemoryCoreSurfaceEntry, ...]:
    """Return core surface entries by advisory status."""

    source_plan = plan or build_creative_memory_core_surface()
    return tuple(entry for entry in source_plan.entries if entry.status == status)


def creative_memory_core_surface_entries_for_confidence(
    confidence: CreativeMemoryCoreSurfaceConfidence,
    plan: CreativeMemoryCoreSurfacePlan | None = None,
) -> tuple[CreativeMemoryCoreSurfaceEntry, ...]:
    """Return core surface entries by confidence band."""

    source_plan = plan or build_creative_memory_core_surface()
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
) -> tuple[CreativeMemoryCoreSurfaceEntry, ...]:
    return (
        _entry(
            kind="memory_foundation_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="memory_foundation",
            source_roles=(
                "long_term_creative_memory",
                "user_preferences",
                "style_profiles",
                "project_memory",
            ),
            sources=sources,
            surface_coverage_score=86,
            source_traceability_score=84,
            governance_alignment_score=82,
            activation_risk_score=46,
            governance_weight=150,
        ),
        _entry(
            kind="personalization_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="personalization",
            source_roles=("creative_dna", "personalization_engine"),
            sources=sources,
            surface_coverage_score=82,
            source_traceability_score=78,
            governance_alignment_score=80,
            activation_risk_score=44,
            governance_weight=140,
        ),
        _entry(
            kind="session_artifact_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="session_artifact_history",
            source_roles=("session_memory_evolution", "artifact_history"),
            sources=sources,
            surface_coverage_score=74,
            source_traceability_score=76,
            governance_alignment_score=72,
            activation_risk_score=42,
            governance_weight=120,
        ),
        _entry(
            kind="lineage_ontology_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="lineage_ontology",
            source_roles=("creative_lineage", "creative_ontology"),
            sources=sources,
            surface_coverage_score=64,
            source_traceability_score=60,
            governance_alignment_score=66,
            activation_risk_score=32,
            governance_weight=110,
        ),
        _entry(
            kind="core_boundary_surface",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            surface_axis="governance_boundary",
            source_roles=tuple(source.role for source in sources),
            sources=sources,
            surface_coverage_score=50,
            source_traceability_score=52,
            governance_alignment_score=54,
            activation_risk_score=18,
            governance_weight=85,
        ),
    )


def _entry(
    *,
    kind: CreativeMemoryCoreSurfaceKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    surface_axis: CreativeMemoryCoreSurfaceAxis,
    source_roles: tuple[str, ...],
    sources: tuple[_Source, ...],
    surface_coverage_score: int,
    source_traceability_score: int,
    governance_alignment_score: int,
    activation_risk_score: int,
    governance_weight: int,
) -> CreativeMemoryCoreSurfaceEntry:
    selected_sources = _sources_for_roles(sources, source_roles)
    source_item_ids = _source_item_ids(selected_sources)
    score = _core_surface_score(
        surface_coverage_score=surface_coverage_score,
        source_traceability_score=source_traceability_score,
        governance_alignment_score=governance_alignment_score,
        activation_risk_score=activation_risk_score,
        governance_weight=governance_weight,
    )
    status = _core_surface_status(score)
    confidence = _core_surface_confidence(score)
    return CreativeMemoryCoreSurfaceEntry(
        core_surface_id=f"creative_memory_core::{kind}",
        surface_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        surface_axis=surface_axis,
        source_plan_roles=tuple(source.role for source in selected_sources),
        source_serialization_versions=tuple(
            source.serialization_version for source in selected_sources
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
        hitl_required_before_surface_activation=True,
        context_tags=_context_tags(kind, surface_axis),
        explainability_notes=_explainability_notes(kind, selected_sources),
        advisory_actions=_entry_actions(kind),
        evidence=(
            f"source_plan_count:{len(selected_sources)}",
            f"source_item_count:{len(source_item_ids)}",
            f"surface_axis:{surface_axis}",
            f"surface_coverage_score:{surface_coverage_score}",
            f"source_traceability_score:{source_traceability_score}",
            f"governance_alignment_score:{governance_alignment_score}",
            f"activation_risk_score:{activation_risk_score}",
            "hitl_required_before_surface_activation:true",
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


def _core_surface_status(score: int) -> CreativeMemoryCoreSurfaceStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _core_surface_confidence(score: int) -> CreativeMemoryCoreSurfaceConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_core_surface_score(
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...],
) -> int:
    base = sum(entry.core_surface_score for entry in entries) // len(entries)
    guarded_count = len(_entry_ids_for_status(entries, "guarded"))
    review_count = len(_entry_ids_for_status(entries, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_core_surface_posture(
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...],
) -> CreativeMemoryCoreSurfacePosture:
    if any(entry.status == "guarded" for entry in entries):
        return "guarded"
    if any(entry.status == "review_required" for entry in entries):
        return "review_required"
    return "candidate"


def _entry_ids_for_status(
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...],
    status: CreativeMemoryCoreSurfaceStatus,
) -> tuple[str, ...]:
    return tuple(entry.core_surface_id for entry in entries if entry.status == status)


def _entry_ids_for_confidence(
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...],
    *confidences: CreativeMemoryCoreSurfaceConfidence,
) -> tuple[str, ...]:
    return tuple(
        entry.core_surface_id for entry in entries if entry.confidence in confidences
    )


def _plan_actions(
    entries: tuple[CreativeMemoryCoreSurfaceEntry, ...],
) -> tuple[str, ...]:
    guarded_entry_count = len(_entry_ids_for_status(entries, "guarded"))
    return (
        "inspect_creative_memory_core_surface",
        "verify_core_roadmap_metadata_coverage",
        "require_hitl_before_core_surface_activation",
        f"review_guarded_core_surface_count:{guarded_entry_count}",
    )


def _surface_summary(kind: CreativeMemoryCoreSurfaceKind) -> str:
    summaries = {
        "memory_foundation_surface": (
            "Summarizes advisory long-term, preference, style, and project memory."
        ),
        "personalization_surface": (
            "Summarizes advisory Creative DNA and personalization metadata."
        ),
        "session_artifact_surface": (
            "Summarizes advisory session evolution and artifact history metadata."
        ),
        "lineage_ontology_surface": (
            "Summarizes advisory creative lineage and ontology metadata."
        ),
        "core_boundary_surface": (
            "Summarizes governance coverage across all core V6.2 sources."
        ),
    }
    return summaries[kind]


def _context_tags(
    kind: CreativeMemoryCoreSurfaceKind,
    surface_axis: CreativeMemoryCoreSurfaceAxis,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "core_surface",
        surface_axis,
        kind.removesuffix("_surface"),
    )


def _explainability_notes(
    kind: CreativeMemoryCoreSurfaceKind,
    sources: tuple[_Source, ...],
) -> tuple[str, ...]:
    return (
        f"core_surface_kind:{kind}",
        "source_roles:" + ",".join(source.role for source in sources),
        "source_versions:"
        + ",".join(source.serialization_version for source in sources),
        "score_inputs:surface_coverage,source_traceability,governance_alignment,activation_risk,governance",
        "activation_boundary:HITL_required_before_core_surface_activation",
    )


def _entry_actions(kind: CreativeMemoryCoreSurfaceKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_core_surface_activation",
        "preserve_no_memory_runtime_mutation_boundary",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    return route if isinstance(route, RouteName) else RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    normalized = str(task_type).strip()
    if normalized not in get_args(TaskRoutingType):
        raise ValueError("task_type must be a known routing task type")
    return cast(TaskRoutingType, normalized)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_execution_mode_ids: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in allowed_execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
