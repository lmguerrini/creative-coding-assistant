"""V6.6 Unified Execution Graph metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_hitl_layer import (
    CognitiveHITLLayerPlan,
    build_cognitive_hitl_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

UNIFIED_EXECUTION_GRAPH_SERIALIZATION_VERSION = "unified_execution_graph.v1"
UNIFIED_EXECUTION_GRAPH_ROADMAP_ITEM = "Unified Execution Graph"
UNIFIED_EXECUTION_GRAPH_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Execution Graph projects cognitive HITL checkpoints into "
    "a read-only execution topology across Learning, Memory, Knowledge, "
    "Research, Self Evolution, and Cognitive Core. It exposes execution "
    "readiness metadata, source traceability, governance, explainability, "
    "safety, and HITL boundaries only; it does not execute nodes, traverse "
    "edges, apply plans, apply routing, emit HITL requests, apply HITL "
    "decisions, mutate workflows, prompts, memory, retrieval, storage, "
    "provider selection, generated output, runtime state, or apply Runtime "
    "Evolution."
)
UNIFIED_EXECUTION_RELATIONSHIPS = (
    "learning_execution_contextualizes_memory",
    "memory_execution_contextualizes_knowledge",
    "knowledge_execution_contextualizes_research",
    "research_execution_contextualizes_self_evolution",
    "self_evolution_execution_contextualizes_cognitive_core",
)


class UnifiedExecutionNode(BaseModel):
    """One read-only execution node in the V6.6 Cognitive OS."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    execution_order: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    upstream_execution_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    downstream_execution_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    hitl_posture: CognitiveOSPosture
    execution_posture: CognitiveOSPosture
    execution_authorized: Literal[False] = False
    execution_state: Literal["blocked_pending_hitl"] = "blocked_pending_hitl"
    source_trace_ids: tuple[str, ...] = Field(min_length=11, max_length=15)
    execution_summary: str = Field(min_length=1, max_length=760)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    hitl_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    execution_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _node_matches_sources_and_boundary(self) -> Self:
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
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class UnifiedExecutionEdge(BaseModel):
    """One read-only execution edge in the V6.6 Cognitive OS."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    execution_edge_id: str = Field(min_length=1, max_length=420)
    from_execution_node_id: str = Field(min_length=1, max_length=190)
    to_execution_node_id: str = Field(min_length=1, max_length=190)
    relationship: str = Field(min_length=1, max_length=120)
    dependency_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    safety_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    hitl_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    execution_transition_authorized: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _edge_id_matches_nodes(self) -> Self:
        expected = f"{self.from_execution_node_id}->{self.to_execution_node_id}"
        if self.execution_edge_id != expected:
            raise ValueError("execution_edge_id must match endpoint nodes")
        if self.relationship not in UNIFIED_EXECUTION_RELATIONSHIPS:
            raise ValueError("relationship must match V6.6 execution order")
        return self


class UnifiedExecutionGraphPlan(BaseModel):
    """Read-only execution graph over the Cognitive OS chain."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_execution_graph"] = "unified_execution_graph"
    serialization_version: Literal["unified_execution_graph.v1"] = (
        UNIFIED_EXECUTION_GRAPH_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_EXECUTION_GRAPH_AUTHORITY_BOUNDARY,
        max_length=2600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_hitl_layer_role: Literal["cognitive_hitl_layer"]
    cognitive_hitl_layer_serialization_version: Literal["cognitive_hitl_layer.v1"]
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
    execution_nodes: tuple[UnifiedExecutionNode, ...] = Field(
        min_length=6,
        max_length=6,
    )
    execution_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    execution_node_count: int = Field(ge=6, le=6)
    execution_edges: tuple[UnifiedExecutionEdge, ...] = Field(
        min_length=5,
        max_length=5,
    )
    execution_edge_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    execution_edge_count: int = Field(ge=5, le=5)
    execution_entry_node_id: str = Field(min_length=1, max_length=190)
    execution_terminal_node_id: str = Field(min_length=1, max_length=190)
    blocked_pending_hitl_node_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    blocked_pending_hitl_node_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_execution_graph_implemented: Literal[True] = True
    cognitive_hitl_layer_integrated: Literal[True] = True
    execution_node_contract_implemented: Literal[True] = True
    execution_edge_contract_implemented: Literal[True] = True
    execution_dependency_traceability_implemented: Literal[True] = True
    execution_governance_contract_implemented: Literal[True] = True
    execution_explainability_contract_implemented: Literal[True] = True
    execution_safety_contract_implemented: Literal[True] = True
    execution_hitl_contract_implemented: Literal[True] = True
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
    executed_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    traversed_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_execution_graph_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _execution_graph_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_hitl_count != len(self.source_hitl_ids):
            raise ValueError("source_hitl_count must match HITL ids")
        if self.source_safety_count != len(self.source_safety_ids):
            raise ValueError("source_safety_count must match safety ids")
        if self.source_explanation_count != len(self.source_explanation_ids):
            raise ValueError("source_explanation_count must match explanation ids")
        if self.source_blackboard_entry_count != len(self.source_blackboard_entry_ids):
            raise ValueError("source_blackboard_entry_count must match entries")
        if self.source_route_decision_count != len(self.source_route_decision_ids):
            raise ValueError("source_route_decision_count must match route ids")
        if self.source_plan_count != len(self.source_plan_ids):
            raise ValueError("source_plan_count must match plan ids")
        if self.source_schedule_count != len(self.source_schedule_ids):
            raise ValueError("source_schedule_count must match schedule ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
        if self.execution_node_ids != tuple(
            node.execution_node_id for node in self.execution_nodes
        ):
            raise ValueError("execution_node_ids must match nodes")
        if self.execution_node_count != len(self.execution_nodes):
            raise ValueError("execution_node_count must match nodes")
        if len(set(self.execution_node_ids)) != len(self.execution_node_ids):
            raise ValueError("execution_node_ids must be unique")
        if self.execution_edge_ids != tuple(
            edge.execution_edge_id for edge in self.execution_edges
        ):
            raise ValueError("execution_edge_ids must match edges")
        if self.execution_edge_count != len(self.execution_edges):
            raise ValueError("execution_edge_count must match edges")
        if len(set(self.execution_edge_ids)) != len(self.execution_edge_ids):
            raise ValueError("execution_edge_ids must be unique")
        if self.execution_entry_node_id != self.execution_node_ids[0]:
            raise ValueError("execution_entry_node_id must be first node")
        if self.execution_terminal_node_id != self.execution_node_ids[-1]:
            raise ValueError("execution_terminal_node_id must be last node")
        if self.blocked_pending_hitl_node_ids != self.execution_node_ids:
            raise ValueError("blocked_pending_hitl_node_ids must match nodes")
        if self.blocked_pending_hitl_node_count != len(
            self.blocked_pending_hitl_node_ids
        ):
            raise ValueError("blocked_pending_hitl_node_count must match nodes")

        declared_capabilities = set(self.capability_ids)
        declared_hitl = set(self.source_hitl_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_blackboard = set(self.source_blackboard_entry_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        declared_nodes = set(self.execution_node_ids)
        for node in self.execution_nodes:
            if node.capability_id not in declared_capabilities:
                raise ValueError("node capability_id must be declared")
            if node.hitl_id not in declared_hitl:
                raise ValueError("node hitl_id must be declared")
            if node.safety_id not in declared_safety:
                raise ValueError("node safety_id must be declared")
            if node.explanation_id not in declared_explanations:
                raise ValueError("node explanation_id must be declared")
            if node.blackboard_entry_id not in declared_blackboard:
                raise ValueError("node blackboard_entry_id must be declared")
            if node.route_decision_id not in declared_routes:
                raise ValueError("node route_decision_id must be declared")
            if node.plan_id not in declared_plans:
                raise ValueError("node plan_id must be declared")
            if node.schedule_id not in declared_schedules:
                raise ValueError("node schedule_id must be declared")
            if node.emergence_id not in declared_emergence:
                raise ValueError("node emergence_id must be declared")
            if not set(node.linked_agent_ids).issubset(declared_agents):
                raise ValueError("node linked_agent_ids must be declared")
            if not set(node.upstream_execution_node_ids).issubset(declared_nodes):
                raise ValueError("node upstream ids must be declared")
            if not set(node.downstream_execution_node_ids).issubset(declared_nodes):
                raise ValueError("node downstream ids must be declared")
        for edge in self.execution_edges:
            if edge.from_execution_node_id not in declared_nodes:
                raise ValueError("edge source node must be declared")
            if edge.to_execution_node_id not in declared_nodes:
                raise ValueError("edge target node must be declared")
        if tuple(edge.relationship for edge in self.execution_edges) != (
            UNIFIED_EXECUTION_RELATIONSHIPS
        ):
            raise ValueError("execution edge relationships must match OS order")
        if self.covered_roadmap_items != (UNIFIED_EXECUTION_GRAPH_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 24 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.executed_node_ids,
                self.traversed_edge_ids,
                self.applied_route_decision_ids,
                self.emitted_hitl_request_ids,
                self.applied_hitl_decision_ids,
                self.mutated_execution_graph_ids,
            )
        ):
            raise ValueError(
                "execution, traversal, routing, HITL, and mutation ids must be empty",
            )
        if not all(node.advisory_only for node in self.execution_nodes):
            raise ValueError("all unified execution nodes must be advisory only")
        if not all(edge.advisory_only for edge in self.execution_edges):
            raise ValueError("all unified execution edges must be advisory only")
        return self


def build_unified_execution_graph(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_hitl_layer: CognitiveHITLLayerPlan | None = None,
) -> UnifiedExecutionGraphPlan:
    """Build the read-only V6.6 unified execution graph."""

    hitl_layer = cognitive_hitl_layer or build_cognitive_hitl_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    nodes = _unified_execution_nodes(hitl_layer)
    edges = _unified_execution_edges(nodes)
    return UnifiedExecutionGraphPlan(
        route_name=hitl_layer.route_name,
        task_type=hitl_layer.task_type,
        execution_mode_ids=hitl_layer.execution_mode_ids,
        cognitive_hitl_layer_role=hitl_layer.role,
        cognitive_hitl_layer_serialization_version=(hitl_layer.serialization_version),
        cognitive_safety_layer_role=hitl_layer.cognitive_safety_layer_role,
        cognitive_explanation_engine_role=(
            hitl_layer.cognitive_explanation_engine_role
        ),
        cognitive_blackboard_role=hitl_layer.cognitive_blackboard_role,
        cognitive_router_role=hitl_layer.cognitive_router_role,
        cognitive_planner_role=hitl_layer.cognitive_planner_role,
        cognitive_scheduler_role=hitl_layer.cognitive_scheduler_role,
        layer_order=hitl_layer.layer_order,
        capabilities=hitl_layer.capabilities,
        capability_ids=hitl_layer.capability_ids,
        capability_count=hitl_layer.capability_count,
        source_hitl_ids=hitl_layer.hitl_ids,
        source_hitl_count=hitl_layer.hitl_count,
        source_safety_ids=hitl_layer.source_safety_ids,
        source_safety_count=hitl_layer.source_safety_count,
        source_explanation_ids=hitl_layer.source_explanation_ids,
        source_explanation_count=hitl_layer.source_explanation_count,
        source_blackboard_entry_ids=hitl_layer.source_blackboard_entry_ids,
        source_blackboard_entry_count=hitl_layer.source_blackboard_entry_count,
        source_route_decision_ids=hitl_layer.source_route_decision_ids,
        source_route_decision_count=hitl_layer.source_route_decision_count,
        source_plan_ids=hitl_layer.source_plan_ids,
        source_plan_count=hitl_layer.source_plan_count,
        source_schedule_ids=hitl_layer.source_schedule_ids,
        source_schedule_count=hitl_layer.source_schedule_count,
        source_emergence_ids=hitl_layer.source_emergence_ids,
        source_emergence_count=hitl_layer.source_emergence_count,
        execution_nodes=nodes,
        execution_node_ids=tuple(node.execution_node_id for node in nodes),
        execution_node_count=len(nodes),
        execution_edges=edges,
        execution_edge_ids=tuple(edge.execution_edge_id for edge in edges),
        execution_edge_count=len(edges),
        execution_entry_node_id=nodes[0].execution_node_id,
        execution_terminal_node_id=nodes[-1].execution_node_id,
        blocked_pending_hitl_node_ids=tuple(node.execution_node_id for node in nodes),
        blocked_pending_hitl_node_count=len(nodes),
        linked_agent_ids=hitl_layer.linked_agent_ids,
        covered_roadmap_items=(UNIFIED_EXECUTION_GRAPH_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=hitl_layer.graph_posture,
    )


def unified_execution_node_by_id(
    execution_node_id: str,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> UnifiedExecutionNode | None:
    """Return one execution node without executing it."""

    source_graph = graph or build_unified_execution_graph()
    for node in source_graph.execution_nodes:
        if node.execution_node_id == execution_node_id:
            return node
    return None


def unified_execution_nodes_for_layer(
    cognitive_layer: CognitiveOSLayer,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> tuple[UnifiedExecutionNode, ...]:
    """Return execution nodes for one Cognitive OS layer."""

    source_graph = graph or build_unified_execution_graph()
    return tuple(
        node
        for node in source_graph.execution_nodes
        if node.cognitive_layer == cognitive_layer
    )


def unified_execution_nodes_for_agent(
    agent_id: str,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> tuple[UnifiedExecutionNode, ...]:
    """Return execution nodes linked to one agent."""

    source_graph = graph or build_unified_execution_graph()
    return tuple(
        node
        for node in source_graph.execution_nodes
        if agent_id in node.linked_agent_ids
    )


def unified_execution_nodes_for_posture(
    posture: CognitiveOSPosture,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> tuple[UnifiedExecutionNode, ...]:
    """Return execution nodes by posture without executing them."""

    source_graph = graph or build_unified_execution_graph()
    return tuple(
        node
        for node in source_graph.execution_nodes
        if node.execution_posture == posture
    )


def unified_execution_edges_from_node(
    execution_node_id: str,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> tuple[UnifiedExecutionEdge, ...]:
    """Return outgoing execution edges without traversing them."""

    source_graph = graph or build_unified_execution_graph()
    return tuple(
        edge
        for edge in source_graph.execution_edges
        if edge.from_execution_node_id == execution_node_id
    )


def unified_execution_edges_to_node(
    execution_node_id: str,
    graph: UnifiedExecutionGraphPlan | None = None,
) -> tuple[UnifiedExecutionEdge, ...]:
    """Return incoming execution edges without traversing them."""

    source_graph = graph or build_unified_execution_graph()
    return tuple(
        edge
        for edge in source_graph.execution_edges
        if edge.to_execution_node_id == execution_node_id
    )


def _unified_execution_nodes(
    hitl_layer: CognitiveHITLLayerPlan,
) -> tuple[UnifiedExecutionNode, ...]:
    node_ids = tuple(
        f"unified_execution::{checkpoint.capability_id}"
        for checkpoint in hitl_layer.hitl_checkpoints
    )
    return tuple(
        UnifiedExecutionNode(
            execution_node_id=node_ids[index],
            hitl_id=checkpoint.hitl_id,
            safety_id=checkpoint.safety_id,
            explanation_id=checkpoint.explanation_id,
            blackboard_entry_id=checkpoint.blackboard_entry_id,
            route_decision_id=checkpoint.route_decision_id,
            plan_id=checkpoint.plan_id,
            schedule_id=checkpoint.schedule_id,
            emergence_id=checkpoint.emergence_id,
            capability_id=checkpoint.capability_id,
            capability_name=checkpoint.capability_name,
            cognitive_layer=checkpoint.cognitive_layer,
            linked_agent_ids=checkpoint.linked_agent_ids,
            execution_order=index + 1,
            dependency_depth=checkpoint.dependency_depth,
            upstream_execution_node_ids=((node_ids[index - 1],) if index > 0 else ()),
            downstream_execution_node_ids=(
                (node_ids[index + 1],) if index + 1 < len(node_ids) else ()
            ),
            hitl_posture=checkpoint.hitl_posture,
            execution_posture=checkpoint.hitl_posture,
            source_trace_ids=(checkpoint.hitl_id, *checkpoint.source_trace_ids),
            execution_summary=(
                f"Read-only unified execution node for "
                f"{checkpoint.capability_name}; preserves HITL, safety, "
                "explanation, blackboard, routing, planning, scheduling, and "
                "emergence traceability without executing or traversing."
            ),
            dependency_contracts=(
                "unified execution follows cognitive HITL checkpoint",
                f"cognitive HITL:{checkpoint.hitl_id}",
                f"cognitive safety:{checkpoint.safety_id}",
            ),
            governance_contracts=(
                "unified execution graph does not execute nodes",
                "unified execution graph does not apply routing",
                "HITL required before any execution behavior",
            ),
            explanation_contracts=(
                "unified execution cites the full cognitive source chain",
                "unified execution preserves capability and agent ownership",
                "unified execution explains why traversal is not authorized",
            ),
            safety_contracts=(
                "unified execution preserves safety boundary metadata",
                "unified execution preserves workflow blocking boundary",
                "unified execution preserves mutation boundary metadata",
            ),
            hitl_contracts=(
                "unified execution preserves HITL review requirement",
                "unified execution preserves decision ownership boundary",
                "unified execution preserves request emission boundary",
            ),
            execution_contracts=(
                "execution node is metadata-only",
                "execution traversal is not authorized",
                "execution state remains blocked pending HITL",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for index, checkpoint in enumerate(hitl_layer.hitl_checkpoints)
    )


def _unified_execution_edges(
    nodes: tuple[UnifiedExecutionNode, ...],
) -> tuple[UnifiedExecutionEdge, ...]:
    return tuple(
        UnifiedExecutionEdge(
            execution_edge_id=(
                f"{from_node.execution_node_id}->{to_node.execution_node_id}"
            ),
            from_execution_node_id=from_node.execution_node_id,
            to_execution_node_id=to_node.execution_node_id,
            relationship=UNIFIED_EXECUTION_RELATIONSHIPS[index],
            dependency_trace=(from_node.plan_id, to_node.plan_id),
            governance_trace=(from_node.hitl_id, to_node.hitl_id),
            explanation_trace=(from_node.explanation_id, to_node.explanation_id),
            safety_trace=(from_node.safety_id, to_node.safety_id),
            hitl_trace=(from_node.hitl_id, to_node.hitl_id),
        )
        for index, (from_node, to_node) in enumerate(
            zip(nodes, nodes[1:], strict=False)
        )
    )
