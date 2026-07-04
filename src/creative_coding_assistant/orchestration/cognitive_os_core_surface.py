"""V6.6 Cognitive OS core surface metadata."""

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
from creative_coding_assistant.orchestration.core_os_consolidation import (
    CoreOSConsolidationPlan,
    build_core_os_consolidation,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_OS_CORE_SURFACE_SERIALIZATION_VERSION = "cognitive_os_core_surface.v1"
COGNITIVE_OS_CORE_SURFACE_TASK_ITEM = "Core Surface Implementation"
COGNITIVE_OS_CORE_SURFACE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive OS Core Surface projects Core OS consolidation units into "
    "read-only capability surface metadata. It exposes surface readiness, "
    "ownership, dependency, governance, explainability, safety, and HITL "
    "contracts only; it does not activate core surfaces, persist surface "
    "state, execute workflow nodes, apply routing, emit HITL requests, apply "
    "HITL decisions, mutate prompts, memory, retrieval, storage, provider "
    "selection, generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_OS_CORE_SURFACE_KINDS = (
    "learning_core_surface",
    "memory_core_surface",
    "knowledge_core_surface",
    "research_core_surface",
    "self_evolution_core_surface",
    "cognitive_core_surface",
)

CoreSurfaceKind = Literal[
    "learning_core_surface",
    "memory_core_surface",
    "knowledge_core_surface",
    "research_core_surface",
    "self_evolution_core_surface",
    "cognitive_core_surface",
]


class CognitiveOSCoreSurfaceEntry(BaseModel):
    """One read-only Cognitive OS core surface entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    core_surface_id: str = Field(min_length=1, max_length=190)
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
    surface_kind: CoreSurfaceKind
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    surface_sequence_position: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    surface_status: CognitiveOSPosture
    surface_readiness_score: int = Field(ge=0, le=100)
    context_tags: tuple[str, ...] = Field(min_length=6, max_length=10)
    source_trace_ids: tuple[str, ...] = Field(min_length=13, max_length=17)
    hitl_required_before_core_surface_activation: Literal[True] = True
    core_surface_activation_authorized: Literal[False] = False
    runtime_activation_authorized: Literal[False] = False
    surface_summary: str = Field(min_length=1, max_length=760)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    hitl_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_sources_and_boundary(self) -> Self:
        expected_surface_id = f"cognitive_os_core::{self.capability_id}"
        if self.core_surface_id != expected_surface_id:
            raise ValueError("core_surface_id must match capability_id")
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
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveOSCoreSurfacePlan(BaseModel):
    """Read-only first core capability surface for V6.6."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_os_core_surface"] = "cognitive_os_core_surface"
    serialization_version: Literal["cognitive_os_core_surface.v1"] = (
        COGNITIVE_OS_CORE_SURFACE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_OS_CORE_SURFACE_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    source_consolidation_role: Literal["core_os_consolidation"]
    source_consolidation_serialization_version: Literal["core_os_consolidation.v1"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
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
    core_surface_entries: tuple[CognitiveOSCoreSurfaceEntry, ...] = Field(
        min_length=6,
        max_length=6,
    )
    core_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    core_surface_count: int = Field(ge=6, le=6)
    guarded_core_surface_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    guarded_core_surface_count: int = Field(ge=6, le=6)
    hitl_required_core_surface_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    hitl_required_core_surface_count: int = Field(ge=6, le=6)
    highest_core_surface_score: int = Field(ge=0, le=100)
    overall_core_surface_score: int = Field(ge=0, le=100)
    overall_core_surface_posture: CognitiveOSPosture
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_task_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_task_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    core_surface_implemented: Literal[True] = True
    core_surface_metadata_implemented: Literal[True] = True
    core_surface_lookup_helpers_implemented: Literal[True] = True
    core_surface_activation_implemented: Literal[False] = False
    runtime_activation_implemented: Literal[False] = False
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
    activated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    persisted_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_core_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _surface_matches_entries_and_boundary(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_consolidation_unit_count != len(
            self.source_consolidation_unit_ids
        ):
            raise ValueError("source_consolidation_unit_count must match units")
        if self.source_execution_node_count != len(self.source_execution_node_ids):
            raise ValueError("source_execution_node_count must match nodes")
        if self.source_hitl_count != len(self.source_hitl_ids):
            raise ValueError("source_hitl_count must match HITL ids")
        if self.source_safety_count != len(self.source_safety_ids):
            raise ValueError("source_safety_count must match safety ids")
        if self.source_explanation_count != len(self.source_explanation_ids):
            raise ValueError("source_explanation_count must match explanation ids")
        if self.core_surface_ids != tuple(
            entry.core_surface_id for entry in self.core_surface_entries
        ):
            raise ValueError("core_surface_ids must match entries")
        if self.core_surface_count != len(self.core_surface_entries):
            raise ValueError("core_surface_count must match entries")
        if len(set(self.core_surface_ids)) != len(self.core_surface_ids):
            raise ValueError("core_surface_ids must be unique")
        if self.guarded_core_surface_ids != tuple(
            entry.core_surface_id
            for entry in self.core_surface_entries
            if entry.surface_status == "guarded"
        ):
            raise ValueError("guarded_core_surface_ids must match entries")
        if self.hitl_required_core_surface_ids != tuple(
            entry.core_surface_id
            for entry in self.core_surface_entries
            if entry.hitl_required_before_core_surface_activation
        ):
            raise ValueError("hitl_required_core_surface_ids must match entries")
        if self.guarded_core_surface_count != len(self.guarded_core_surface_ids):
            raise ValueError("guarded_core_surface_count must match ids")
        if self.hitl_required_core_surface_count != len(
            self.hitl_required_core_surface_ids
        ):
            raise ValueError("hitl_required_core_surface_count must match ids")
        if self.highest_core_surface_score != max(
            entry.surface_readiness_score for entry in self.core_surface_entries
        ):
            raise ValueError("highest_core_surface_score must match entries")
        expected_average = round(
            sum(entry.surface_readiness_score for entry in self.core_surface_entries)
            / len(self.core_surface_entries)
        )
        if self.overall_core_surface_score != expected_average:
            raise ValueError("overall_core_surface_score must match entries")
        if self.overall_core_surface_posture != "guarded":
            raise ValueError("overall_core_surface_posture must remain guarded")

        declared_capabilities = set(self.capability_ids)
        declared_units = set(self.source_consolidation_unit_ids)
        declared_nodes = set(self.source_execution_node_ids)
        declared_hitl = set(self.source_hitl_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_agents = set(self.linked_agent_ids)
        for entry in self.core_surface_entries:
            if entry.capability_id not in declared_capabilities:
                raise ValueError("entry capability_id must be declared")
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
            if not set(entry.linked_agent_ids).issubset(declared_agents):
                raise ValueError("entry linked_agent_ids must be declared")
        if self.covered_task_items != (COGNITIVE_OS_CORE_SURFACE_TASK_ITEM,):
            raise ValueError("covered_task_items must be Task 26 only")
        if self.covered_task_item_count != len(self.covered_task_items):
            raise ValueError("covered_task_item_count must match tasks")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.activated_core_surface_ids,
                self.persisted_core_surface_ids,
                self.emitted_hitl_request_ids,
                self.applied_hitl_decision_ids,
                self.mutated_core_surface_ids,
            )
        ):
            raise ValueError(
                "core surface activation, persistence, HITL, and mutation ids "
                "must be empty",
            )
        if not all(entry.advisory_only for entry in self.core_surface_entries):
            raise ValueError("all Cognitive OS core surface entries must be advisory")
        return self


def build_cognitive_os_core_surface(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    core_os_consolidation: CoreOSConsolidationPlan | None = None,
) -> CognitiveOSCoreSurfacePlan:
    """Build read-only Cognitive OS core surface metadata."""

    consolidation = core_os_consolidation or build_core_os_consolidation(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    entries = _core_surface_entries(consolidation)
    core_surface_ids = tuple(entry.core_surface_id for entry in entries)
    guarded_ids = tuple(
        entry.core_surface_id for entry in entries if entry.surface_status == "guarded"
    )
    hitl_required_ids = tuple(
        entry.core_surface_id
        for entry in entries
        if entry.hitl_required_before_core_surface_activation
    )
    return CognitiveOSCoreSurfacePlan(
        route_name=consolidation.route_name,
        task_type=consolidation.task_type,
        execution_mode_ids=consolidation.execution_mode_ids,
        source_consolidation_role=consolidation.role,
        source_consolidation_serialization_version=(
            consolidation.serialization_version
        ),
        layer_order=consolidation.layer_order,
        capabilities=consolidation.capabilities,
        capability_ids=consolidation.capability_ids,
        capability_count=consolidation.capability_count,
        source_consolidation_unit_ids=consolidation.consolidation_unit_ids,
        source_consolidation_unit_count=consolidation.consolidation_unit_count,
        source_execution_node_ids=consolidation.source_execution_node_ids,
        source_execution_node_count=consolidation.source_execution_node_count,
        source_hitl_ids=consolidation.source_hitl_ids,
        source_hitl_count=consolidation.source_hitl_count,
        source_safety_ids=consolidation.source_safety_ids,
        source_safety_count=consolidation.source_safety_count,
        source_explanation_ids=consolidation.source_explanation_ids,
        source_explanation_count=consolidation.source_explanation_count,
        core_surface_entries=entries,
        core_surface_ids=core_surface_ids,
        core_surface_count=len(entries),
        guarded_core_surface_ids=guarded_ids,
        guarded_core_surface_count=len(guarded_ids),
        hitl_required_core_surface_ids=hitl_required_ids,
        hitl_required_core_surface_count=len(hitl_required_ids),
        highest_core_surface_score=max(
            entry.surface_readiness_score for entry in entries
        ),
        overall_core_surface_score=round(
            sum(entry.surface_readiness_score for entry in entries) / len(entries)
        ),
        overall_core_surface_posture="guarded",
        linked_agent_ids=consolidation.linked_agent_ids,
        covered_task_items=(COGNITIVE_OS_CORE_SURFACE_TASK_ITEM,),
        covered_task_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    )


def cognitive_os_core_surface_entry_by_id(
    core_surface_id: str,
    surface: CognitiveOSCoreSurfacePlan | None = None,
) -> CognitiveOSCoreSurfaceEntry | None:
    """Return one core surface entry without activating it."""

    source_surface = surface or build_cognitive_os_core_surface()
    for entry in source_surface.core_surface_entries:
        if entry.core_surface_id == core_surface_id:
            return entry
    return None


def cognitive_os_core_surface_entries_for_layer(
    cognitive_layer: CognitiveOSLayer,
    surface: CognitiveOSCoreSurfacePlan | None = None,
) -> tuple[CognitiveOSCoreSurfaceEntry, ...]:
    """Return core surface entries for one Cognitive OS layer."""

    source_surface = surface or build_cognitive_os_core_surface()
    return tuple(
        entry
        for entry in source_surface.core_surface_entries
        if entry.cognitive_layer == cognitive_layer
    )


def cognitive_os_core_surface_entries_for_agent(
    agent_id: str,
    surface: CognitiveOSCoreSurfacePlan | None = None,
) -> tuple[CognitiveOSCoreSurfaceEntry, ...]:
    """Return core surface entries linked to one agent."""

    source_surface = surface or build_cognitive_os_core_surface()
    return tuple(
        entry
        for entry in source_surface.core_surface_entries
        if agent_id in entry.linked_agent_ids
    )


def cognitive_os_core_surface_entries_for_status(
    status: CognitiveOSPosture,
    surface: CognitiveOSCoreSurfacePlan | None = None,
) -> tuple[CognitiveOSCoreSurfaceEntry, ...]:
    """Return core surface entries by guarded review status."""

    source_surface = surface or build_cognitive_os_core_surface()
    return tuple(
        entry
        for entry in source_surface.core_surface_entries
        if entry.surface_status == status
    )


def _core_surface_entries(
    consolidation: CoreOSConsolidationPlan,
) -> tuple[CognitiveOSCoreSurfaceEntry, ...]:
    return tuple(
        CognitiveOSCoreSurfaceEntry(
            core_surface_id=f"cognitive_os_core::{unit.capability_id}",
            consolidation_unit_id=unit.consolidation_unit_id,
            execution_node_id=unit.execution_node_id,
            hitl_id=unit.hitl_id,
            safety_id=unit.safety_id,
            explanation_id=unit.explanation_id,
            route_decision_id=unit.route_decision_id,
            plan_id=unit.plan_id,
            schedule_id=unit.schedule_id,
            capability_id=unit.capability_id,
            capability_name=unit.capability_name,
            cognitive_layer=unit.cognitive_layer,
            surface_kind=COGNITIVE_OS_CORE_SURFACE_KINDS[index],
            linked_agent_ids=unit.linked_agent_ids,
            surface_sequence_position=unit.os_sequence_position,
            dependency_depth=unit.dependency_depth,
            surface_status=unit.consolidation_posture,
            surface_readiness_score=94 - index,
            context_tags=(
                "cognitive_os_core_surface",
                "core_surface_metadata",
                "roadmap_traceability",
                "dependency_awareness",
                "governance_boundary",
                "hitl_required",
            ),
            source_trace_ids=(
                unit.consolidation_unit_id,
                *unit.source_trace_ids,
            ),
            surface_summary=(
                f"Read-only Cognitive OS core surface for "
                f"{unit.capability_name}; exposes consolidated metadata "
                "without activation, persistence, execution, or routing."
            ),
            dependency_contracts=(
                "core surface follows Core OS consolidation unit",
                f"core OS unit:{unit.consolidation_unit_id}",
                f"unified execution:{unit.execution_node_id}",
            ),
            governance_contracts=(
                "core surface does not activate runtime behavior",
                "core surface does not persist surface state",
                "HITL required before any core surface activation",
            ),
            explanation_contracts=(
                "core surface cites the full Cognitive OS source chain",
                "core surface preserves capability and agent ownership",
                "core surface explains why activation is not authorized",
            ),
            safety_contracts=(
                "core surface preserves safety boundary metadata",
                "core surface preserves workflow blocking boundary",
                "core surface preserves mutation boundary metadata",
            ),
            hitl_contracts=(
                "core surface preserves HITL review requirement",
                "core surface preserves decision ownership boundary",
                "core surface preserves request emission boundary",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for index, unit in enumerate(consolidation.consolidation_units)
    )
