"""V6.6 Core OS Consolidation metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    COGNITIVE_OS_ROADMAP_ITEMS,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_execution_graph import (
    UnifiedExecutionGraphPlan,
    build_unified_execution_graph,
)

CORE_OS_CONSOLIDATION_SERIALIZATION_VERSION = "core_os_consolidation.v1"
CORE_OS_CONSOLIDATION_ROADMAP_ITEM = "Core OS Consolidation"
CORE_OS_CONSOLIDATION_AUTHORITY_BOUNDARY = (
    "V6.6 Core OS Consolidation projects the unified execution graph into a "
    "read-only Cognitive Operating System contract surface across Learning, "
    "Memory, Knowledge, Research, Self Evolution, and Cognitive Core. It "
    "consolidates roadmap traceability, ownership, dependency awareness, "
    "governance, explainability, safety, HITL, and future HoloMind "
    "extensibility metadata only; it does not activate the OS, execute graph "
    "nodes, traverse edges, apply routing, emit HITL requests, apply HITL "
    "decisions, mutate workflows, prompts, memory, retrieval, storage, "
    "provider selection, generated output, runtime state, or apply Runtime "
    "Evolution."
)
CORE_OS_CONSOLIDATION_CONTRACTS = (
    "cognitive OS sequence consolidation",
    "roadmap traceability consolidation",
    "ownership boundary consolidation",
    "dependency traceability consolidation",
    "governance and HITL consolidation",
    "explainability and safety consolidation",
    "future HoloMind extensibility consolidation",
)


class CoreOSConsolidationUnit(BaseModel):
    """One read-only consolidated Cognitive OS unit."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    consolidation_unit_id: str = Field(min_length=1, max_length=190)
    execution_node_id: str = Field(min_length=1, max_length=190)
    hitl_id: str = Field(min_length=1, max_length=190)
    safety_id: str = Field(min_length=1, max_length=190)
    explanation_id: str = Field(min_length=1, max_length=190)
    blackboard_entry_id: str = Field(min_length=1, max_length=190)
    route_decision_id: str = Field(min_length=1, max_length=190)
    plan_id: str = Field(min_length=1, max_length=190)
    schedule_id: str = Field(min_length=1, max_length=190)
    emergence_id: str = Field(min_length=1, max_length=190)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    os_sequence_position: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    upstream_consolidation_unit_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    downstream_consolidation_unit_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    execution_posture: CognitiveOSPosture
    consolidation_posture: CognitiveOSPosture
    core_os_status: Literal["consolidated_metadata_only"] = (
        "consolidated_metadata_only"
    )
    runtime_activation_authorized: Literal[False] = False
    execution_authorized: Literal[False] = False
    source_trace_ids: tuple[str, ...] = Field(min_length=12, max_length=16)
    consolidation_summary: str = Field(min_length=1, max_length=820)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    hitl_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    consolidation_contracts: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _unit_matches_sources_and_boundary(self) -> Self:
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
        expected_blackboard_id = f"cognitive_blackboard::{self.capability_id}"
        if self.blackboard_entry_id != expected_blackboard_id:
            raise ValueError("blackboard_entry_id must match capability_id")
        expected_route_id = f"cognitive_router::{self.capability_id}"
        if self.route_decision_id != expected_route_id:
            raise ValueError("route_decision_id must match capability_id")
        expected_plan_id = f"cognitive_planner::{self.capability_id}"
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id must match capability_id")
        expected_schedule_id = f"cognitive_scheduler::{self.capability_id}"
        if self.schedule_id != expected_schedule_id:
            raise ValueError("schedule_id must match capability_id")
        expected_emergence_id = f"emergent_creativity::{self.capability_id}"
        if self.emergence_id != expected_emergence_id:
            raise ValueError("emergence_id must match capability_id")
        if self.consolidation_contracts != CORE_OS_CONSOLIDATION_CONTRACTS:
            raise ValueError("consolidation_contracts must match Core OS")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CoreOSConsolidationPlan(BaseModel):
    """Read-only consolidation plan for the V6.6 Cognitive OS."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["core_os_consolidation"] = "core_os_consolidation"
    serialization_version: Literal["core_os_consolidation.v1"] = (
        CORE_OS_CONSOLIDATION_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CORE_OS_CONSOLIDATION_AUTHORITY_BOUNDARY,
        max_length=2800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    unified_execution_graph_role: Literal["unified_execution_graph"]
    unified_execution_graph_serialization_version: Literal[
        "unified_execution_graph.v1"
    ]
    cognitive_hitl_layer_role: Literal["cognitive_hitl_layer"]
    cognitive_safety_layer_role: Literal["cognitive_safety_layer"]
    cognitive_explanation_engine_role: Literal["cognitive_explanation_engine"]
    cognitive_blackboard_role: Literal["cognitive_blackboard"]
    cognitive_router_role: Literal["cognitive_router"]
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_execution_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_execution_node_count: int = Field(ge=6, le=6)
    source_execution_edge_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_execution_edge_count: int = Field(ge=5, le=5)
    source_hitl_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_hitl_count: int = Field(ge=6, le=6)
    source_safety_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_safety_count: int = Field(ge=6, le=6)
    source_explanation_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_explanation_count: int = Field(ge=6, le=6)
    source_blackboard_entry_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_blackboard_entry_count: int = Field(ge=6, le=6)
    source_route_decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_route_decision_count: int = Field(ge=6, le=6)
    source_plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_plan_count: int = Field(ge=6, le=6)
    source_schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_schedule_count: int = Field(ge=6, le=6)
    source_emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_emergence_count: int = Field(ge=6, le=6)
    consolidation_units: tuple[CoreOSConsolidationUnit, ...] = Field(
        min_length=6,
        max_length=6,
    )
    consolidation_unit_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    consolidation_unit_count: int = Field(ge=6, le=6)
    core_os_entry_unit_id: str = Field(min_length=1, max_length=190)
    core_os_terminal_unit_id: str = Field(min_length=1, max_length=190)
    blocked_pending_hitl_unit_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    blocked_pending_hitl_unit_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    consolidated_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    consolidated_roadmap_item_count: int = Field(ge=24, le=24)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    core_os_consolidation_implemented: Literal[True] = True
    unified_execution_graph_integrated: Literal[True] = True
    cognitive_os_sequence_consolidated: Literal[True] = True
    roadmap_traceability_consolidated: Literal[True] = True
    dependency_traceability_consolidated: Literal[True] = True
    governance_contract_consolidated: Literal[True] = True
    explainability_contract_consolidated: Literal[True] = True
    safety_contract_consolidated: Literal[True] = True
    hitl_contract_consolidated: Literal[True] = True
    future_holomind_extensibility_prepared: Literal[True] = True
    core_os_runtime_activation_implemented: Literal[False] = False
    execution_application_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    autonomous_workflow_planning_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    scheduler_application_implemented: Literal[False] = False
    plan_execution_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    safety_enforcement_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    activated_core_os_unit_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    traversed_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_core_os_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _consolidation_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_execution_node_count != len(self.source_execution_node_ids):
            raise ValueError("source_execution_node_count must match nodes")
        if self.source_execution_edge_count != len(self.source_execution_edge_ids):
            raise ValueError("source_execution_edge_count must match edges")
        if self.source_hitl_count != len(self.source_hitl_ids):
            raise ValueError("source_hitl_count must match HITL ids")
        if self.source_safety_count != len(self.source_safety_ids):
            raise ValueError("source_safety_count must match safety ids")
        if self.source_explanation_count != len(self.source_explanation_ids):
            raise ValueError("source_explanation_count must match explanation ids")
        if self.source_blackboard_entry_count != len(
            self.source_blackboard_entry_ids
        ):
            raise ValueError("source_blackboard_entry_count must match entries")
        if self.source_route_decision_count != len(self.source_route_decision_ids):
            raise ValueError("source_route_decision_count must match route ids")
        if self.source_plan_count != len(self.source_plan_ids):
            raise ValueError("source_plan_count must match plan ids")
        if self.source_schedule_count != len(self.source_schedule_ids):
            raise ValueError("source_schedule_count must match schedule ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
        if self.consolidation_unit_ids != tuple(
            unit.consolidation_unit_id for unit in self.consolidation_units
        ):
            raise ValueError("consolidation_unit_ids must match units")
        if self.consolidation_unit_count != len(self.consolidation_units):
            raise ValueError("consolidation_unit_count must match units")
        if len(set(self.consolidation_unit_ids)) != len(
            self.consolidation_unit_ids
        ):
            raise ValueError("consolidation_unit_ids must be unique")
        if self.core_os_entry_unit_id != self.consolidation_unit_ids[0]:
            raise ValueError("core_os_entry_unit_id must be first unit")
        if self.core_os_terminal_unit_id != self.consolidation_unit_ids[-1]:
            raise ValueError("core_os_terminal_unit_id must be last unit")
        if self.blocked_pending_hitl_unit_ids != self.consolidation_unit_ids:
            raise ValueError("blocked_pending_hitl_unit_ids must match units")
        if self.blocked_pending_hitl_unit_count != len(
            self.blocked_pending_hitl_unit_ids
        ):
            raise ValueError("blocked_pending_hitl_unit_count must match units")

        declared_capabilities = set(self.capability_ids)
        declared_execution_nodes = set(self.source_execution_node_ids)
        declared_hitl = set(self.source_hitl_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_blackboard = set(self.source_blackboard_entry_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        declared_units = set(self.consolidation_unit_ids)
        for unit in self.consolidation_units:
            if unit.capability_id not in declared_capabilities:
                raise ValueError("unit capability_id must be declared")
            if unit.execution_node_id not in declared_execution_nodes:
                raise ValueError("unit execution_node_id must be declared")
            if unit.hitl_id not in declared_hitl:
                raise ValueError("unit hitl_id must be declared")
            if unit.safety_id not in declared_safety:
                raise ValueError("unit safety_id must be declared")
            if unit.explanation_id not in declared_explanations:
                raise ValueError("unit explanation_id must be declared")
            if unit.blackboard_entry_id not in declared_blackboard:
                raise ValueError("unit blackboard_entry_id must be declared")
            if unit.route_decision_id not in declared_routes:
                raise ValueError("unit route_decision_id must be declared")
            if unit.plan_id not in declared_plans:
                raise ValueError("unit plan_id must be declared")
            if unit.schedule_id not in declared_schedules:
                raise ValueError("unit schedule_id must be declared")
            if unit.emergence_id not in declared_emergence:
                raise ValueError("unit emergence_id must be declared")
            if not set(unit.linked_agent_ids).issubset(declared_agents):
                raise ValueError("unit linked_agent_ids must be declared")
            if not set(unit.upstream_consolidation_unit_ids).issubset(
                declared_units
            ):
                raise ValueError("unit upstream ids must be declared")
            if not set(unit.downstream_consolidation_unit_ids).issubset(
                declared_units
            ):
                raise ValueError("unit downstream ids must be declared")
        if self.covered_roadmap_items != (
            CORE_OS_CONSOLIDATION_ROADMAP_ITEM,
        ):
            raise ValueError("covered_roadmap_items must be Task 25 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.consolidated_roadmap_items != COGNITIVE_OS_ROADMAP_ITEMS:
            raise ValueError("consolidated_roadmap_items must match V6.6 roadmap")
        if self.consolidated_roadmap_item_count != len(
            self.consolidated_roadmap_items
        ):
            raise ValueError("consolidated_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.activated_core_os_unit_ids,
                self.executed_node_ids,
                self.traversed_edge_ids,
                self.applied_route_decision_ids,
                self.emitted_hitl_request_ids,
                self.applied_hitl_decision_ids,
                self.mutated_core_os_ids,
            )
        ):
            raise ValueError(
                "Core OS activation, execution, traversal, routing, HITL, "
                "and mutation ids must be empty",
            )
        if not all(unit.advisory_only for unit in self.consolidation_units):
            raise ValueError("all Core OS consolidation units must be advisory only")
        return self


def build_core_os_consolidation(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    unified_execution_graph: UnifiedExecutionGraphPlan | None = None,
) -> CoreOSConsolidationPlan:
    """Build the read-only V6.6 Core OS consolidation surface."""

    execution_graph = (
        unified_execution_graph
        or build_unified_execution_graph(
            route=route,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        )
    )
    units = _core_os_consolidation_units(execution_graph)
    unit_ids = tuple(unit.consolidation_unit_id for unit in units)
    return CoreOSConsolidationPlan(
        route_name=execution_graph.route_name,
        task_type=execution_graph.task_type,
        execution_mode_ids=execution_graph.execution_mode_ids,
        unified_execution_graph_role=execution_graph.role,
        unified_execution_graph_serialization_version=(
            execution_graph.serialization_version
        ),
        cognitive_hitl_layer_role=execution_graph.cognitive_hitl_layer_role,
        cognitive_safety_layer_role=execution_graph.cognitive_safety_layer_role,
        cognitive_explanation_engine_role=(
            execution_graph.cognitive_explanation_engine_role
        ),
        cognitive_blackboard_role=execution_graph.cognitive_blackboard_role,
        cognitive_router_role=execution_graph.cognitive_router_role,
        cognitive_planner_role=execution_graph.cognitive_planner_role,
        cognitive_scheduler_role=execution_graph.cognitive_scheduler_role,
        layer_order=execution_graph.layer_order,
        capabilities=execution_graph.capabilities,
        capability_ids=execution_graph.capability_ids,
        capability_count=execution_graph.capability_count,
        source_execution_node_ids=execution_graph.execution_node_ids,
        source_execution_node_count=execution_graph.execution_node_count,
        source_execution_edge_ids=execution_graph.execution_edge_ids,
        source_execution_edge_count=execution_graph.execution_edge_count,
        source_hitl_ids=execution_graph.source_hitl_ids,
        source_hitl_count=execution_graph.source_hitl_count,
        source_safety_ids=execution_graph.source_safety_ids,
        source_safety_count=execution_graph.source_safety_count,
        source_explanation_ids=execution_graph.source_explanation_ids,
        source_explanation_count=execution_graph.source_explanation_count,
        source_blackboard_entry_ids=(
            execution_graph.source_blackboard_entry_ids
        ),
        source_blackboard_entry_count=(
            execution_graph.source_blackboard_entry_count
        ),
        source_route_decision_ids=execution_graph.source_route_decision_ids,
        source_route_decision_count=execution_graph.source_route_decision_count,
        source_plan_ids=execution_graph.source_plan_ids,
        source_plan_count=execution_graph.source_plan_count,
        source_schedule_ids=execution_graph.source_schedule_ids,
        source_schedule_count=execution_graph.source_schedule_count,
        source_emergence_ids=execution_graph.source_emergence_ids,
        source_emergence_count=execution_graph.source_emergence_count,
        consolidation_units=units,
        consolidation_unit_ids=unit_ids,
        consolidation_unit_count=len(units),
        core_os_entry_unit_id=unit_ids[0],
        core_os_terminal_unit_id=unit_ids[-1],
        blocked_pending_hitl_unit_ids=unit_ids,
        blocked_pending_hitl_unit_count=len(unit_ids),
        linked_agent_ids=execution_graph.linked_agent_ids,
        covered_roadmap_items=(CORE_OS_CONSOLIDATION_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        consolidated_roadmap_items=COGNITIVE_OS_ROADMAP_ITEMS,
        consolidated_roadmap_item_count=len(COGNITIVE_OS_ROADMAP_ITEMS),
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=execution_graph.graph_posture,
    )


def core_os_consolidation_unit_by_id(
    consolidation_unit_id: str,
    consolidation: CoreOSConsolidationPlan | None = None,
) -> CoreOSConsolidationUnit | None:
    """Return one Core OS consolidation unit without activating it."""

    source_consolidation = consolidation or build_core_os_consolidation()
    for unit in source_consolidation.consolidation_units:
        if unit.consolidation_unit_id == consolidation_unit_id:
            return unit
    return None


def core_os_consolidation_units_for_layer(
    cognitive_layer: CognitiveOSLayer,
    consolidation: CoreOSConsolidationPlan | None = None,
) -> tuple[CoreOSConsolidationUnit, ...]:
    """Return consolidation units for one Cognitive OS layer."""

    source_consolidation = consolidation or build_core_os_consolidation()
    return tuple(
        unit
        for unit in source_consolidation.consolidation_units
        if unit.cognitive_layer == cognitive_layer
    )


def core_os_consolidation_units_for_agent(
    agent_id: str,
    consolidation: CoreOSConsolidationPlan | None = None,
) -> tuple[CoreOSConsolidationUnit, ...]:
    """Return consolidation units linked to one agent."""

    source_consolidation = consolidation or build_core_os_consolidation()
    return tuple(
        unit
        for unit in source_consolidation.consolidation_units
        if agent_id in unit.linked_agent_ids
    )


def core_os_consolidation_units_for_posture(
    posture: CognitiveOSPosture,
    consolidation: CoreOSConsolidationPlan | None = None,
) -> tuple[CoreOSConsolidationUnit, ...]:
    """Return consolidation units by posture without activation."""

    source_consolidation = consolidation or build_core_os_consolidation()
    return tuple(
        unit
        for unit in source_consolidation.consolidation_units
        if unit.consolidation_posture == posture
    )


def _core_os_consolidation_units(
    execution_graph: UnifiedExecutionGraphPlan,
) -> tuple[CoreOSConsolidationUnit, ...]:
    unit_ids = tuple(
        f"core_os::{node.capability_id}" for node in execution_graph.execution_nodes
    )
    return tuple(
        CoreOSConsolidationUnit(
            consolidation_unit_id=unit_ids[index],
            execution_node_id=node.execution_node_id,
            hitl_id=node.hitl_id,
            safety_id=node.safety_id,
            explanation_id=node.explanation_id,
            blackboard_entry_id=node.blackboard_entry_id,
            route_decision_id=node.route_decision_id,
            plan_id=node.plan_id,
            schedule_id=node.schedule_id,
            emergence_id=node.emergence_id,
            capability_id=node.capability_id,
            capability_name=node.capability_name,
            cognitive_layer=node.cognitive_layer,
            linked_agent_ids=node.linked_agent_ids,
            os_sequence_position=node.execution_order,
            dependency_depth=node.dependency_depth,
            upstream_consolidation_unit_ids=(
                (unit_ids[index - 1],) if index > 0 else ()
            ),
            downstream_consolidation_unit_ids=(
                (unit_ids[index + 1],) if index + 1 < len(unit_ids) else ()
            ),
            execution_posture=node.execution_posture,
            consolidation_posture=node.execution_posture,
            source_trace_ids=(node.execution_node_id, *node.source_trace_ids),
            consolidation_summary=(
                f"Read-only Core OS consolidation unit for "
                f"{node.capability_name}; preserves execution, HITL, safety, "
                "explanation, blackboard, routing, planning, scheduling, and "
                "emergence traceability without activation."
            ),
            dependency_contracts=(
                "Core OS consolidation follows unified execution node",
                f"unified execution:{node.execution_node_id}",
                f"cognitive HITL:{node.hitl_id}",
            ),
            governance_contracts=(
                "Core OS consolidation does not activate runtime behavior",
                "Core OS consolidation does not execute workflow nodes",
                "HITL required before any Core OS behavioral application",
            ),
            explanation_contracts=(
                "Core OS consolidation cites the full cognitive source chain",
                "Core OS consolidation preserves capability and agent ownership",
                "Core OS consolidation explains why activation is not authorized",
            ),
            safety_contracts=(
                "Core OS consolidation preserves safety boundary metadata",
                "Core OS consolidation preserves workflow blocking boundary",
                "Core OS consolidation preserves mutation boundary metadata",
            ),
            hitl_contracts=(
                "Core OS consolidation preserves HITL review requirement",
                "Core OS consolidation preserves decision ownership boundary",
                "Core OS consolidation preserves request emission boundary",
            ),
            consolidation_contracts=CORE_OS_CONSOLIDATION_CONTRACTS,
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for index, node in enumerate(execution_graph.execution_nodes)
    )
