"""V6.6 Cognitive OS secondary surface metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.cognitive_os_core_surface import (
    CognitiveOSCoreSurfacePlan,
    build_cognitive_os_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_OS_SECONDARY_SURFACE_SERIALIZATION_VERSION = (
    "cognitive_os_secondary_surface.v1"
)
COGNITIVE_OS_SECONDARY_SURFACE_TASK_ITEM = "Secondary Surface Implementation"
COGNITIVE_OS_FOUNDATION_SYSTEMS = (
    "V5 Decision Engine",
    *COGNITIVE_OS_CAPABILITIES,
)
COGNITIVE_OS_SECONDARY_REPORT_SECTIONS = (
    "cognitive_os_sequence",
    "foundation_composition",
    "roadmap_traceability",
    "dependency_traceability",
    "governance_hitl_explainability",
    "safety_and_runtime_boundaries",
)
COGNITIVE_OS_SECONDARY_SURFACE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive OS Secondary Surface composes supporting capability "
    "surface metadata from the Task 26 core surface across V5 Decision "
    "Engine context and the V6 Learning, Memory, Knowledge, Research, Self "
    "Evolution, and Cognitive Core sequence. It exposes report-view, "
    "foundation composition, ownership, dependency, governance, "
    "explainability, safety, HITL, and future HoloMind extensibility "
    "metadata only; it does not generate report artifacts, activate "
    "secondary surfaces, persist surface state, execute workflows, apply "
    "routing, emit HITL requests, apply HITL decisions, mutate prompts, "
    "memory, retrieval, storage, provider selection, generated output, "
    "runtime state, or apply Runtime Evolution."
)
COGNITIVE_OS_SECONDARY_SURFACE_KINDS = (
    "learning_secondary_surface",
    "memory_secondary_surface",
    "knowledge_secondary_surface",
    "research_secondary_surface",
    "self_evolution_secondary_surface",
    "cognitive_core_secondary_surface",
)

SecondarySurfaceKind = Literal[
    "learning_secondary_surface",
    "memory_secondary_surface",
    "knowledge_secondary_surface",
    "research_secondary_surface",
    "self_evolution_secondary_surface",
    "cognitive_core_secondary_surface",
]


class CognitiveOSSecondarySurfaceEntry(BaseModel):
    """One read-only supporting Cognitive OS surface entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    secondary_surface_id: str = Field(min_length=1, max_length=190)
    source_core_surface_id: str = Field(min_length=1, max_length=190)
    consolidation_unit_id: str = Field(min_length=1, max_length=190)
    execution_node_id: str = Field(min_length=1, max_length=190)
    hitl_id: str = Field(min_length=1, max_length=190)
    safety_id: str = Field(min_length=1, max_length=190)
    explanation_id: str = Field(min_length=1, max_length=190)
    route_decision_id: str = Field(min_length=1, max_length=190)
    plan_id: str = Field(min_length=1, max_length=190)
    schedule_id: str = Field(min_length=1, max_length=190)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    secondary_surface_kind: SecondarySurfaceKind
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    surface_sequence_position: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    source_core_surface_status: CognitiveOSPosture
    secondary_surface_status: CognitiveOSPosture
    source_core_surface_score: int = Field(ge=0, le=100)
    secondary_surface_score: int = Field(ge=0, le=100)
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    context_tags: tuple[str, ...] = Field(min_length=6, max_length=10)
    source_trace_ids: tuple[str, ...] = Field(min_length=14, max_length=18)
    hitl_required_before_secondary_surface_activation: Literal[True] = True
    secondary_surface_activation_authorized: Literal[False] = False
    runtime_activation_authorized: Literal[False] = False
    surface_summary: str = Field(min_length=1, max_length=820)
    foundation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    hitl_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    report_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    generated_report_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    foundation_composition_metadata_implemented: Literal[True] = True
    advisory_report_view_metadata_implemented: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_sources_and_boundary(self) -> Self:
        expected_secondary_id = f"cognitive_os_secondary::{self.capability_id}"
        if self.secondary_surface_id != expected_secondary_id:
            raise ValueError("secondary_surface_id must match capability_id")
        expected_core_id = f"cognitive_os_core::{self.capability_id}"
        if self.source_core_surface_id != expected_core_id:
            raise ValueError("source_core_surface_id must match capability_id")
        expected_unit_id = f"core_os::{self.capability_id}"
        if self.consolidation_unit_id != expected_unit_id:
            raise ValueError("consolidation_unit_id must match capability_id")
        expected_execution_id = f"unified_execution::{self.capability_id}"
        if self.execution_node_id != expected_execution_id:
            raise ValueError("execution_node_id must match capability_id")
        expected_hitl_id = f"cognitive_hitl::{self.capability_id}"
        if self.hitl_id != expected_hitl_id:
            raise ValueError("hitl_id must match capability_id")
        expected_safety_id = f"cognitive_safety::{self.capability_id}"
        if self.safety_id != expected_safety_id:
            raise ValueError("safety_id must match capability_id")
        expected_explanation_id = f"cognitive_explanation::{self.capability_id}"
        if self.explanation_id != expected_explanation_id:
            raise ValueError("explanation_id must match capability_id")
        expected_route_id = f"cognitive_router::{self.capability_id}"
        if self.route_decision_id != expected_route_id:
            raise ValueError("route_decision_id must match capability_id")
        expected_plan_id = f"cognitive_planner::{self.capability_id}"
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id must match capability_id")
        expected_schedule_id = f"cognitive_scheduler::{self.capability_id}"
        if self.schedule_id != expected_schedule_id:
            raise ValueError("schedule_id must match capability_id")
        if self.secondary_surface_status != self.source_core_surface_status:
            raise ValueError("secondary_surface_status must mirror core surface")
        if self.secondary_surface_score != max(self.source_core_surface_score - 2, 0):
            raise ValueError("secondary_surface_score must derive from core score")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.report_sections != COGNITIVE_OS_SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match secondary surface contract")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.generated_report_artifact_ids,
                self.written_storage_record_ids,
                self.mutated_secondary_surface_ids,
            )
        ):
            raise ValueError("secondary surface entry mutation ids must be empty")
        return self


class CognitiveOSSecondarySurfacePlan(BaseModel):
    """Read-only supporting surface over the V6.6 core surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_os_secondary_surface"] = "cognitive_os_secondary_surface"
    serialization_version: Literal["cognitive_os_secondary_surface.v1"] = (
        COGNITIVE_OS_SECONDARY_SURFACE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_OS_SECONDARY_SURFACE_AUTHORITY_BOUNDARY,
        max_length=3000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    source_core_surface_role: Literal["cognitive_os_core_surface"]
    source_core_surface_serialization_version: Literal["cognitive_os_core_surface.v1"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    foundation_systems: tuple[str, ...] = Field(min_length=7, max_length=7)
    foundation_system_count: int = Field(ge=7, le=7)
    source_core_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_core_surface_count: int = Field(ge=6, le=6)
    source_core_surface_scores: tuple[int, ...] = Field(min_length=6, max_length=6)
    source_core_surface_score_count: int = Field(ge=6, le=6)
    source_consolidation_unit_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_consolidation_unit_count: int = Field(ge=6, le=6)
    source_execution_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_execution_node_count: int = Field(ge=6, le=6)
    source_hitl_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_hitl_count: int = Field(ge=6, le=6)
    source_safety_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_safety_count: int = Field(ge=6, le=6)
    source_explanation_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_explanation_count: int = Field(ge=6, le=6)
    source_route_decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_route_decision_count: int = Field(ge=6, le=6)
    source_plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_plan_count: int = Field(ge=6, le=6)
    source_schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_schedule_count: int = Field(ge=6, le=6)
    report_sections: tuple[str, ...] = Field(min_length=6, max_length=6)
    secondary_surface_entries: tuple[CognitiveOSSecondarySurfaceEntry, ...] = Field(
        min_length=6,
        max_length=6,
    )
    secondary_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    secondary_surface_count: int = Field(ge=6, le=6)
    guarded_secondary_surface_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    guarded_secondary_surface_count: int = Field(ge=6, le=6)
    hitl_required_secondary_surface_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    hitl_required_secondary_surface_count: int = Field(ge=6, le=6)
    top_secondary_surface_id: str = Field(min_length=1, max_length=190)
    highest_secondary_surface_score: int = Field(ge=0, le=100)
    overall_secondary_surface_score: int = Field(ge=0, le=100)
    overall_secondary_surface_posture: CognitiveOSPosture
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_task_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_task_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    secondary_surface_implemented: Literal[True] = True
    secondary_surface_metadata_implemented: Literal[True] = True
    secondary_surface_lookup_helpers_implemented: Literal[True] = True
    foundation_composition_metadata_implemented: Literal[True] = True
    advisory_report_view_metadata_implemented: Literal[True] = True
    roadmap_traceability_implemented: Literal[True] = True
    dependency_traceability_implemented: Literal[True] = True
    governance_contract_implemented: Literal[True] = True
    explainability_contract_implemented: Literal[True] = True
    safety_contract_implemented: Literal[True] = True
    hitl_contract_implemented: Literal[True] = True
    future_holomind_extensibility_prepared: Literal[True] = True
    secondary_surface_activation_implemented: Literal[False] = False
    runtime_activation_implemented: Literal[False] = False
    report_artifact_generation_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    generated_report_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    persisted_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_secondary_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _surface_matches_entries_and_boundary(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.foundation_systems != COGNITIVE_OS_FOUNDATION_SYSTEMS:
            raise ValueError("foundation_systems must match V5/V6 foundations")
        if self.foundation_system_count != len(self.foundation_systems):
            raise ValueError("foundation_system_count must match foundations")
        if self.source_core_surface_ids != tuple(
            entry.source_core_surface_id for entry in self.secondary_surface_entries
        ):
            raise ValueError("source_core_surface_ids must match entries")
        if self.source_core_surface_count != len(self.source_core_surface_ids):
            raise ValueError("source_core_surface_count must match ids")
        if self.source_core_surface_scores != tuple(
            entry.source_core_surface_score for entry in self.secondary_surface_entries
        ):
            raise ValueError("source_core_surface_scores must match entries")
        if self.source_core_surface_score_count != len(self.source_core_surface_scores):
            raise ValueError("source_core_surface_score_count must match scores")
        count_fields = (
            (self.source_consolidation_unit_count, self.source_consolidation_unit_ids),
            (self.source_execution_node_count, self.source_execution_node_ids),
            (self.source_hitl_count, self.source_hitl_ids),
            (self.source_safety_count, self.source_safety_ids),
            (self.source_explanation_count, self.source_explanation_ids),
            (self.source_route_decision_count, self.source_route_decision_ids),
            (self.source_plan_count, self.source_plan_ids),
            (self.source_schedule_count, self.source_schedule_ids),
        )
        if any(count != len(ids) for count, ids in count_fields):
            raise ValueError("source counts must match source ids")
        if self.report_sections != COGNITIVE_OS_SECONDARY_REPORT_SECTIONS:
            raise ValueError("report_sections must match secondary surface contract")
        if self.secondary_surface_ids != tuple(
            entry.secondary_surface_id for entry in self.secondary_surface_entries
        ):
            raise ValueError("secondary_surface_ids must match entries")
        if len(set(self.secondary_surface_ids)) != len(self.secondary_surface_ids):
            raise ValueError("secondary_surface_ids must be unique")
        if self.secondary_surface_count != len(self.secondary_surface_entries):
            raise ValueError("secondary_surface_count must match entries")
        if self.guarded_secondary_surface_ids != tuple(
            entry.secondary_surface_id
            for entry in self.secondary_surface_entries
            if entry.secondary_surface_status == "guarded"
        ):
            raise ValueError("guarded_secondary_surface_ids must match entries")
        if self.hitl_required_secondary_surface_ids != tuple(
            entry.secondary_surface_id
            for entry in self.secondary_surface_entries
            if entry.hitl_required_before_secondary_surface_activation
        ):
            raise ValueError(
                "hitl_required_secondary_surface_ids must match entries",
            )
        if self.guarded_secondary_surface_count != len(
            self.guarded_secondary_surface_ids
        ):
            raise ValueError("guarded_secondary_surface_count must match ids")
        if self.hitl_required_secondary_surface_count != len(
            self.hitl_required_secondary_surface_ids
        ):
            raise ValueError("hitl_required_secondary_surface_count must match ids")
        if self.top_secondary_surface_id not in self.secondary_surface_ids:
            raise ValueError("top_secondary_surface_id must match entries")
        if self.highest_secondary_surface_score != max(
            entry.secondary_surface_score for entry in self.secondary_surface_entries
        ):
            raise ValueError("highest_secondary_surface_score must match entries")
        expected_average = round(
            sum(
                entry.secondary_surface_score
                for entry in self.secondary_surface_entries
            )
            / len(self.secondary_surface_entries)
        )
        if self.overall_secondary_surface_score != expected_average:
            raise ValueError("overall_secondary_surface_score must match entries")
        if self.overall_secondary_surface_posture != "guarded":
            raise ValueError("overall_secondary_surface_posture must remain guarded")

        declared_capabilities = set(self.capability_ids)
        declared_core_surfaces = set(self.source_core_surface_ids)
        declared_units = set(self.source_consolidation_unit_ids)
        declared_nodes = set(self.source_execution_node_ids)
        declared_hitl = set(self.source_hitl_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_agents = set(self.linked_agent_ids)
        for entry in self.secondary_surface_entries:
            if entry.capability_id not in declared_capabilities:
                raise ValueError("entry capability_id must be declared")
            if entry.source_core_surface_id not in declared_core_surfaces:
                raise ValueError("entry core surface id must be declared")
            if entry.consolidation_unit_id not in declared_units:
                raise ValueError("entry consolidation_unit_id must be declared")
            if entry.execution_node_id not in declared_nodes:
                raise ValueError("entry execution_node_id must be declared")
            if entry.hitl_id not in declared_hitl:
                raise ValueError("entry hitl_id must be declared")
            if entry.safety_id not in declared_safety:
                raise ValueError("entry safety_id must be declared")
            if entry.explanation_id not in declared_explanations:
                raise ValueError("entry explanation_id must be declared")
            if entry.route_decision_id not in declared_routes:
                raise ValueError("entry route_decision_id must be declared")
            if entry.plan_id not in declared_plans:
                raise ValueError("entry plan_id must be declared")
            if entry.schedule_id not in declared_schedules:
                raise ValueError("entry schedule_id must be declared")
            if not set(entry.linked_agent_ids).issubset(declared_agents):
                raise ValueError("entry linked_agent_ids must be declared")
        if self.covered_task_items != (COGNITIVE_OS_SECONDARY_SURFACE_TASK_ITEM,):
            raise ValueError("covered_task_items must be Task 27 only")
        if self.covered_task_item_count != len(self.covered_task_items):
            raise ValueError("covered_task_item_count must match tasks")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.generated_report_artifact_ids,
                self.written_storage_record_ids,
                self.activated_secondary_surface_ids,
                self.persisted_secondary_surface_ids,
                self.emitted_hitl_request_ids,
                self.applied_hitl_decision_ids,
                self.mutated_secondary_surface_ids,
            )
        ):
            raise ValueError(
                "secondary surface artifacts, activation, persistence, HITL, "
                "and mutation ids must be empty",
            )
        if not all(entry.advisory_only for entry in self.secondary_surface_entries):
            raise ValueError("all secondary surface entries must be advisory")
        return self


def build_cognitive_os_secondary_surface(
    cognitive_os_core_surface: CognitiveOSCoreSurfacePlan | None = None,
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> CognitiveOSSecondarySurfacePlan:
    """Build read-only Cognitive OS secondary surface metadata."""

    core_surface = cognitive_os_core_surface or build_cognitive_os_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    entries = _secondary_surface_entries(core_surface)
    secondary_surface_ids = tuple(entry.secondary_surface_id for entry in entries)
    guarded_ids = tuple(
        entry.secondary_surface_id
        for entry in entries
        if entry.secondary_surface_status == "guarded"
    )
    hitl_required_ids = tuple(
        entry.secondary_surface_id
        for entry in entries
        if entry.hitl_required_before_secondary_surface_activation
    )
    top_entry = max(entries, key=lambda entry: entry.secondary_surface_score)
    return CognitiveOSSecondarySurfacePlan(
        route_name=core_surface.route_name,
        task_type=core_surface.task_type,
        execution_mode_ids=core_surface.execution_mode_ids,
        source_core_surface_role=core_surface.role,
        source_core_surface_serialization_version=(core_surface.serialization_version),
        layer_order=core_surface.layer_order,
        capabilities=core_surface.capabilities,
        capability_ids=core_surface.capability_ids,
        capability_count=core_surface.capability_count,
        foundation_systems=COGNITIVE_OS_FOUNDATION_SYSTEMS,
        foundation_system_count=len(COGNITIVE_OS_FOUNDATION_SYSTEMS),
        source_core_surface_ids=tuple(
            entry.core_surface_id for entry in core_surface.core_surface_entries
        ),
        source_core_surface_count=core_surface.core_surface_count,
        source_core_surface_scores=tuple(
            entry.surface_readiness_score for entry in core_surface.core_surface_entries
        ),
        source_core_surface_score_count=core_surface.core_surface_count,
        source_consolidation_unit_ids=core_surface.source_consolidation_unit_ids,
        source_consolidation_unit_count=(core_surface.source_consolidation_unit_count),
        source_execution_node_ids=core_surface.source_execution_node_ids,
        source_execution_node_count=core_surface.source_execution_node_count,
        source_hitl_ids=core_surface.source_hitl_ids,
        source_hitl_count=core_surface.source_hitl_count,
        source_safety_ids=core_surface.source_safety_ids,
        source_safety_count=core_surface.source_safety_count,
        source_explanation_ids=core_surface.source_explanation_ids,
        source_explanation_count=core_surface.source_explanation_count,
        source_route_decision_ids=tuple(
            entry.route_decision_id for entry in core_surface.core_surface_entries
        ),
        source_route_decision_count=core_surface.core_surface_count,
        source_plan_ids=tuple(
            entry.plan_id for entry in core_surface.core_surface_entries
        ),
        source_plan_count=core_surface.core_surface_count,
        source_schedule_ids=tuple(
            entry.schedule_id for entry in core_surface.core_surface_entries
        ),
        source_schedule_count=core_surface.core_surface_count,
        report_sections=COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
        secondary_surface_entries=entries,
        secondary_surface_ids=secondary_surface_ids,
        secondary_surface_count=len(entries),
        guarded_secondary_surface_ids=guarded_ids,
        guarded_secondary_surface_count=len(guarded_ids),
        hitl_required_secondary_surface_ids=hitl_required_ids,
        hitl_required_secondary_surface_count=len(hitl_required_ids),
        top_secondary_surface_id=top_entry.secondary_surface_id,
        highest_secondary_surface_score=top_entry.secondary_surface_score,
        overall_secondary_surface_score=round(
            sum(entry.secondary_surface_score for entry in entries) / len(entries)
        ),
        overall_secondary_surface_posture="guarded",
        linked_agent_ids=core_surface.linked_agent_ids,
        covered_task_items=(COGNITIVE_OS_SECONDARY_SURFACE_TASK_ITEM,),
        covered_task_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    )


def cognitive_os_secondary_surface_entry_by_id(
    secondary_surface_id: str,
    surface: CognitiveOSSecondarySurfacePlan | None = None,
) -> CognitiveOSSecondarySurfaceEntry | None:
    """Return one secondary surface entry without activating it."""

    source_surface = surface or build_cognitive_os_secondary_surface()
    for entry in source_surface.secondary_surface_entries:
        if entry.secondary_surface_id == secondary_surface_id:
            return entry
    return None


def cognitive_os_secondary_surface_entries_for_layer(
    cognitive_layer: CognitiveOSLayer,
    surface: CognitiveOSSecondarySurfacePlan | None = None,
) -> tuple[CognitiveOSSecondarySurfaceEntry, ...]:
    """Return secondary surface entries for one Cognitive OS layer."""

    source_surface = surface or build_cognitive_os_secondary_surface()
    return tuple(
        entry
        for entry in source_surface.secondary_surface_entries
        if entry.cognitive_layer == cognitive_layer
    )


def cognitive_os_secondary_surface_entries_for_agent(
    agent_id: str,
    surface: CognitiveOSSecondarySurfacePlan | None = None,
) -> tuple[CognitiveOSSecondarySurfaceEntry, ...]:
    """Return secondary surface entries linked to one agent."""

    source_surface = surface or build_cognitive_os_secondary_surface()
    return tuple(
        entry
        for entry in source_surface.secondary_surface_entries
        if agent_id in entry.linked_agent_ids
    )


def cognitive_os_secondary_surface_entries_for_status(
    status: CognitiveOSPosture,
    surface: CognitiveOSSecondarySurfacePlan | None = None,
) -> tuple[CognitiveOSSecondarySurfaceEntry, ...]:
    """Return secondary surface entries by guarded review status."""

    source_surface = surface or build_cognitive_os_secondary_surface()
    return tuple(
        entry
        for entry in source_surface.secondary_surface_entries
        if entry.secondary_surface_status == status
    )


def _secondary_surface_entries(
    core_surface: CognitiveOSCoreSurfacePlan,
) -> tuple[CognitiveOSSecondarySurfaceEntry, ...]:
    return tuple(
        CognitiveOSSecondarySurfaceEntry(
            secondary_surface_id=f"cognitive_os_secondary::{entry.capability_id}",
            source_core_surface_id=entry.core_surface_id,
            consolidation_unit_id=entry.consolidation_unit_id,
            execution_node_id=entry.execution_node_id,
            hitl_id=entry.hitl_id,
            safety_id=entry.safety_id,
            explanation_id=entry.explanation_id,
            route_decision_id=entry.route_decision_id,
            plan_id=entry.plan_id,
            schedule_id=entry.schedule_id,
            capability_id=entry.capability_id,
            capability_name=entry.capability_name,
            cognitive_layer=entry.cognitive_layer,
            secondary_surface_kind=COGNITIVE_OS_SECONDARY_SURFACE_KINDS[index],
            linked_agent_ids=entry.linked_agent_ids,
            surface_sequence_position=entry.surface_sequence_position,
            dependency_depth=entry.dependency_depth,
            source_core_surface_status=entry.surface_status,
            secondary_surface_status=entry.surface_status,
            source_core_surface_score=entry.surface_readiness_score,
            secondary_surface_score=max(entry.surface_readiness_score - 2, 0),
            foundation_systems=COGNITIVE_OS_FOUNDATION_SYSTEMS,
            report_sections=COGNITIVE_OS_SECONDARY_REPORT_SECTIONS,
            context_tags=(
                "cognitive_os_secondary_surface",
                "secondary_surface_metadata",
                "foundation_composition",
                "report_view_metadata",
                "dependency_awareness",
                "hitl_required",
            ),
            source_trace_ids=(
                entry.core_surface_id,
                *entry.source_trace_ids,
            ),
            surface_summary=(
                f"Read-only supporting surface for {entry.capability_name}; "
                "composes V5/V6 foundation context and Task 26 core metadata "
                "without report generation, activation, persistence, or "
                "runtime mutation."
            ),
            foundation_contracts=(
                "secondary surface references V5 Decision Engine context",
                f"V6 cognitive layer:{entry.cognitive_layer}",
                f"source core surface:{entry.core_surface_id}",
                "upstream capability ownership remains unchanged",
            ),
            dependency_contracts=(
                "secondary surface follows core surface dependency chain",
                f"core surface:{entry.core_surface_id}",
                f"execution node:{entry.execution_node_id}",
            ),
            governance_contracts=(
                "secondary surface does not activate runtime behavior",
                "secondary surface does not persist report state",
                "HITL required before any secondary surface activation",
            ),
            explanation_contracts=(
                "secondary surface cites the full Cognitive OS source chain",
                "secondary surface explains foundation composition",
                "secondary surface preserves capability and agent ownership",
            ),
            safety_contracts=(
                "secondary surface preserves safety boundary metadata",
                "secondary surface preserves mutation blocking metadata",
                "secondary surface preserves provider execution boundary",
            ),
            hitl_contracts=(
                "secondary surface preserves HITL review requirement",
                "secondary surface preserves decision ownership boundary",
                "secondary surface preserves request emission boundary",
            ),
            report_contracts=(
                "secondary report view is metadata only",
                "secondary report view does not generate artifacts",
                "secondary report view does not write storage",
            ),
            cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for index, entry in enumerate(core_surface.core_surface_entries)
    )
