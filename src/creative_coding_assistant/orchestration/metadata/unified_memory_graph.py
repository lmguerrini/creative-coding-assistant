"""V6.6 Unified Memory Graph metadata over the cognitive graph backbone."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_cognitive_graph import (
    UnifiedCognitiveGraphPlan,
    build_unified_cognitive_graph,
)

UNIFIED_MEMORY_GRAPH_SERIALIZATION_VERSION = "unified_memory_graph.v1"
UNIFIED_MEMORY_GRAPH_ROADMAP_ITEM = "Unified Memory Graph"
UNIFIED_MEMORY_GRAPH_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Memory Graph composes memory-related metadata across the "
    "Unified Cognitive Graph. It reads source-owned learning, creative "
    "memory, knowledge, research, self-evolution, and cognitive-core graph "
    "signals, makes their memory dependencies inspectable, and does not "
    "persist memories, mutate memory records, mutate retrieval, write "
    "storage, apply preferences, execute providers, modify generated output, "
    "emit HITL requests, apply HITL decisions, or apply Runtime Evolution."
)


class UnifiedMemoryGraphNode(BaseModel):
    """One memory context node linked to a cognitive graph node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    memory_node_id: str = Field(min_length=1, max_length=140)
    cognitive_node_id: str = Field(min_length=1, max_length=120)
    capability: CognitiveOSCapability
    layer: CognitiveOSLayer
    source_snapshot_id: str = Field(min_length=1, max_length=120)
    source_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    source_signal_count: int = Field(ge=1, le=140)
    memory_role: str = Field(min_length=1, max_length=180)
    ownership_boundary: str = Field(min_length=1, max_length=420)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _node_matches_signal_count(self) -> Self:
        if self.source_signal_count != len(self.source_signal_ids):
            raise ValueError("source_signal_count must match source_signal_ids")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class UnifiedMemoryGraphEdge(BaseModel):
    """One read-only memory dependency edge."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    from_memory_node_id: str = Field(min_length=1, max_length=140)
    to_memory_node_id: str = Field(min_length=1, max_length=140)
    cognitive_edge_id: str = Field(min_length=1, max_length=180)
    relationship: str = Field(min_length=1, max_length=180)
    dependency_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _edge_id_matches_nodes(self) -> Self:
        expected = f"{self.from_memory_node_id}->{self.to_memory_node_id}"
        if self.edge_id != expected:
            raise ValueError("edge_id must match memory node ids")
        return self


class UnifiedMemoryGraphPlan(BaseModel):
    """Unified memory graph that extends, rather than forks, the cognitive graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_memory_graph"] = "unified_memory_graph"
    serialization_version: Literal["unified_memory_graph.v1"] = (
        UNIFIED_MEMORY_GRAPH_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_MEMORY_GRAPH_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    backbone_graph_role: Literal["unified_cognitive_graph"]
    backbone_graph_serialization_version: str = Field(min_length=1, max_length=120)
    backbone_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    backbone_edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    memory_nodes: tuple[UnifiedMemoryGraphNode, ...] = Field(
        min_length=6,
        max_length=6,
    )
    memory_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    memory_edges: tuple[UnifiedMemoryGraphEdge, ...] = Field(
        min_length=9,
        max_length=9,
    )
    memory_edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    creative_memory_node_id: Literal["memory::v6_2_memory_node"]
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_signal_ids: tuple[str, ...] = Field(min_length=131, max_length=131)
    source_signal_id_count: int = Field(ge=131, le=131)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_memory_graph_implemented: Literal[True] = True
    unified_cognitive_graph_integrated: Literal[True] = True
    memory_ownership_boundary_check_implemented: Literal[True] = True
    memory_dependency_traceability_implemented: Literal[True] = True
    memory_explainability_contract_implemented: Literal[True] = True
    memory_hitl_governance_contract_implemented: Literal[True] = True
    memory_persistence_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    persisted_memory_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_memory_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_retrieval_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_storage_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _memory_graph_matches_backbone(self) -> Self:
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.covered_roadmap_items != (UNIFIED_MEMORY_GRAPH_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 3 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.memory_node_ids != tuple(
            node.memory_node_id for node in self.memory_nodes
        ):
            raise ValueError("memory_node_ids must match memory_nodes")
        if len(set(self.memory_node_ids)) != len(self.memory_node_ids):
            raise ValueError("memory_node_ids must be unique")
        if self.memory_edge_ids != tuple(edge.edge_id for edge in self.memory_edges):
            raise ValueError("memory_edge_ids must match memory_edges")
        if len(set(self.memory_edge_ids)) != len(self.memory_edge_ids):
            raise ValueError("memory_edge_ids must be unique")
        if tuple(node.cognitive_node_id for node in self.memory_nodes) != (
            self.backbone_node_ids
        ):
            raise ValueError("memory nodes must follow backbone nodes")
        if tuple(edge.cognitive_edge_id for edge in self.memory_edges) != (
            self.backbone_edge_ids
        ):
            raise ValueError("memory edges must follow backbone edges")
        source_signal_ids = tuple(
            signal_id
            for node in self.memory_nodes
            for signal_id in node.source_signal_ids
        )
        if self.source_signal_ids != source_signal_ids:
            raise ValueError("source_signal_ids must match memory nodes")
        if self.source_signal_id_count != len(self.source_signal_ids):
            raise ValueError("source_signal_id_count must match source signals")
        declared_memory_nodes = set(self.memory_node_ids)
        for edge in self.memory_edges:
            if edge.from_memory_node_id not in declared_memory_nodes:
                raise ValueError("edge from_memory_node_id must be declared")
            if edge.to_memory_node_id not in declared_memory_nodes:
                raise ValueError("edge to_memory_node_id must be declared")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.persisted_memory_record_ids,
                self.mutated_memory_record_ids,
                self.mutated_retrieval_record_ids,
                self.written_storage_record_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError("memory mutation, storage, and HITL ids must be empty")
        if not all(node.advisory_only for node in self.memory_nodes):
            raise ValueError("all memory nodes must be advisory only")
        if not all(edge.advisory_only for edge in self.memory_edges):
            raise ValueError("all memory edges must be advisory only")
        return self


def build_unified_memory_graph(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_graph: UnifiedCognitiveGraphPlan | None = None,
) -> UnifiedMemoryGraphPlan:
    """Build a read-only memory graph over the Unified Cognitive Graph."""

    graph = cognitive_graph or build_unified_cognitive_graph(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    memory_nodes = _memory_nodes(graph)
    memory_edges = _memory_edges(graph)
    return UnifiedMemoryGraphPlan(
        route_name=graph.route_name,
        task_type=graph.task_type,
        execution_mode_ids=graph.execution_mode_ids,
        backbone_graph_role=graph.role,
        backbone_graph_serialization_version=graph.serialization_version,
        backbone_node_ids=graph.node_ids,
        backbone_edge_ids=graph.edge_ids,
        capabilities=graph.capabilities,
        memory_nodes=memory_nodes,
        memory_node_ids=tuple(node.memory_node_id for node in memory_nodes),
        memory_edges=memory_edges,
        memory_edge_ids=tuple(edge.edge_id for edge in memory_edges),
        creative_memory_node_id="memory::v6_2_memory_node",
        covered_roadmap_items=(UNIFIED_MEMORY_GRAPH_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        source_signal_ids=tuple(
            signal_id for node in memory_nodes for signal_id in node.source_signal_ids
        ),
        source_signal_id_count=sum(node.source_signal_count for node in memory_nodes),
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def unified_memory_graph_node_by_id(
    memory_node_id: str,
    graph: UnifiedMemoryGraphPlan | None = None,
) -> UnifiedMemoryGraphNode | None:
    """Return one memory graph node without activating persistence."""

    source_graph = graph or build_unified_memory_graph()
    for node in source_graph.memory_nodes:
        if node.memory_node_id == memory_node_id:
            return node
    return None


def unified_memory_graph_nodes_for_layer(
    layer: CognitiveOSLayer,
    graph: UnifiedMemoryGraphPlan | None = None,
) -> tuple[UnifiedMemoryGraphNode, ...]:
    """Return memory graph nodes for one cognitive layer."""

    source_graph = graph or build_unified_memory_graph()
    return tuple(node for node in source_graph.memory_nodes if node.layer == layer)


def unified_memory_graph_edge_by_id(
    edge_id: str,
    graph: UnifiedMemoryGraphPlan | None = None,
) -> UnifiedMemoryGraphEdge | None:
    """Return one memory graph edge without activating persistence."""

    source_graph = graph or build_unified_memory_graph()
    for edge in source_graph.memory_edges:
        if edge.edge_id == edge_id:
            return edge
    return None


def _memory_nodes(
    graph: UnifiedCognitiveGraphPlan,
) -> tuple[UnifiedMemoryGraphNode, ...]:
    memory_roles = {
        "learning": "learning memory context",
        "memory": "creative memory ownership context",
        "knowledge": "knowledge memory provenance context",
        "research": "research memory evidence context",
        "self_evolution": "evolution memory proposal context",
        "cognitive_core": "cognitive graph memory context",
    }
    return tuple(
        UnifiedMemoryGraphNode(
            memory_node_id=f"memory::{node.node_id}",
            cognitive_node_id=node.node_id,
            capability=node.capability,
            layer=node.layer,
            source_snapshot_id=node.source_snapshot_id,
            source_signal_ids=node.source_signal_ids,
            source_signal_count=node.source_signal_count,
            memory_role=memory_roles[node.layer],
            ownership_boundary=node.ownership_boundary,
            dependency_contracts=(
                "memory graph follows unified cognitive graph",
                node.dependency_contracts[0],
            ),
            governance_contracts=(
                "memory graph is advisory metadata only",
                node.governance_contracts[0],
            ),
            explanation_contracts=(
                "memory graph explains source-owned continuity",
                node.explanation_contracts[0],
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for node in graph.nodes
    )


def _memory_edges(
    graph: UnifiedCognitiveGraphPlan,
) -> tuple[UnifiedMemoryGraphEdge, ...]:
    return tuple(
        UnifiedMemoryGraphEdge(
            edge_id=f"memory::{edge.from_node_id}->memory::{edge.to_node_id}",
            from_memory_node_id=f"memory::{edge.from_node_id}",
            to_memory_node_id=f"memory::{edge.to_node_id}",
            cognitive_edge_id=edge.edge_id,
            relationship=f"memory_{edge.relationship}",
            dependency_trace=(
                "memory dependency mirrors cognitive graph edge",
                edge.dependency_trace[0],
            ),
            governance_trace=(
                "memory dependency does not authorize mutation",
                edge.governance_trace[0],
            ),
            explanation_trace=(
                "memory dependency cites cognitive graph explanation",
                edge.explanation_trace[0],
            ),
        )
        for edge in graph.edges
    )
