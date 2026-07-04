"""V6.6 Unified Knowledge Graph metadata over the cognitive OS backbone."""

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
from creative_coding_assistant.orchestration.knowledge_evolution_core_surface import (
    KnowledgeEvolutionCoreSurfacePlan,
    build_knowledge_evolution_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_memory_graph import (
    UnifiedMemoryGraphNode,
    UnifiedMemoryGraphPlan,
    build_unified_memory_graph,
)

UNIFIED_KNOWLEDGE_GRAPH_SERIALIZATION_VERSION = "unified_knowledge_graph.v1"
UNIFIED_KNOWLEDGE_GRAPH_ROADMAP_ITEM = "Unified Knowledge Graph"
UNIFIED_KNOWLEDGE_GRAPH_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Knowledge Graph composes knowledge-related metadata across "
    "the Unified Memory Graph and Unified Cognitive Graph. It exposes V6.3 "
    "Knowledge Evolution source roles, source items, roadmap coverage, "
    "dependency traces, governance traces, and explanation traces as "
    "inspectable advisory metadata only, and does not activate knowledge "
    "surfaces, execute retrieval, compute knowledge scores, write KB storage, "
    "update source records, mutate knowledge graphs, execute providers, "
    "modify generated output, emit HITL requests, apply HITL decisions, or "
    "apply Runtime Evolution."
)


class UnifiedKnowledgeGraphNode(BaseModel):
    """One knowledge context node linked to memory and cognitive graph nodes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    knowledge_node_id: str = Field(min_length=1, max_length=150)
    memory_node_id: str = Field(min_length=1, max_length=140)
    cognitive_node_id: str = Field(min_length=1, max_length=120)
    capability: CognitiveOSCapability
    layer: CognitiveOSLayer
    source_snapshot_id: str = Field(min_length=1, max_length=120)
    source_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    source_signal_count: int = Field(ge=1, le=140)
    knowledge_role: str = Field(min_length=1, max_length=180)
    knowledge_source_roles: tuple[str, ...] = Field(min_length=1, max_length=19)
    knowledge_source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    knowledge_source_item_count: int = Field(ge=1, le=140)
    ownership_boundary: str = Field(min_length=1, max_length=420)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _node_matches_counts_and_boundaries(self) -> Self:
        if self.source_signal_count != len(self.source_signal_ids):
            raise ValueError("source_signal_count must match source_signal_ids")
        if self.knowledge_source_item_count != len(self.knowledge_source_item_ids):
            raise ValueError(
                "knowledge_source_item_count must match knowledge_source_item_ids",
            )
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class UnifiedKnowledgeGraphEdge(BaseModel):
    """One read-only knowledge dependency edge."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=190)
    from_knowledge_node_id: str = Field(min_length=1, max_length=150)
    to_knowledge_node_id: str = Field(min_length=1, max_length=150)
    memory_edge_id: str = Field(min_length=1, max_length=180)
    cognitive_edge_id: str = Field(min_length=1, max_length=180)
    relationship: str = Field(min_length=1, max_length=190)
    dependency_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _edge_id_matches_nodes(self) -> Self:
        expected = f"{self.from_knowledge_node_id}->{self.to_knowledge_node_id}"
        if self.edge_id != expected:
            raise ValueError("edge_id must match knowledge node ids")
        return self


class UnifiedKnowledgeGraphPlan(BaseModel):
    """Unified knowledge graph that extends the memory and cognitive graphs."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_knowledge_graph"] = "unified_knowledge_graph"
    serialization_version: Literal["unified_knowledge_graph.v1"] = (
        UNIFIED_KNOWLEDGE_GRAPH_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_KNOWLEDGE_GRAPH_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    backbone_graph_role: Literal["unified_cognitive_graph"]
    backbone_graph_serialization_version: str = Field(min_length=1, max_length=120)
    memory_graph_role: Literal["unified_memory_graph"]
    memory_graph_serialization_version: str = Field(min_length=1, max_length=120)
    backbone_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    backbone_edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    memory_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    memory_edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    knowledge_nodes: tuple[UnifiedKnowledgeGraphNode, ...] = Field(
        min_length=6,
        max_length=6,
    )
    knowledge_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    knowledge_edges: tuple[UnifiedKnowledgeGraphEdge, ...] = Field(
        min_length=9,
        max_length=9,
    )
    knowledge_edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    v6_3_knowledge_node_id: Literal["knowledge::v6_3_knowledge_node"]
    v6_3_knowledge_source_role: Literal["knowledge_evolution_core_surface"]
    v6_3_knowledge_serialization_version: Literal[
        "knowledge_evolution_core_surface_plan.v1"
    ]
    v6_3_source_plan_roles: tuple[str, ...] = Field(min_length=19, max_length=19)
    v6_3_source_plan_serialization_versions: tuple[str, ...] = Field(
        min_length=19,
        max_length=19,
    )
    v6_3_entry_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    v6_3_entry_count: int = Field(ge=5, le=5)
    v6_3_source_item_ids: tuple[str, ...] = Field(min_length=95, max_length=95)
    v6_3_source_item_count: int = Field(ge=95, le=95)
    v6_3_roadmap_items: tuple[str, ...] = Field(min_length=19, max_length=19)
    v6_3_roadmap_item_count: int = Field(ge=19, le=19)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    source_signal_ids: tuple[str, ...] = Field(min_length=131, max_length=131)
    source_signal_id_count: int = Field(ge=131, le=131)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_knowledge_graph_implemented: Literal[True] = True
    unified_cognitive_graph_integrated: Literal[True] = True
    unified_memory_graph_integrated: Literal[True] = True
    v6_3_knowledge_surface_integrated: Literal[True] = True
    knowledge_ownership_boundary_check_implemented: Literal[True] = True
    knowledge_dependency_traceability_implemented: Literal[True] = True
    knowledge_explainability_contract_implemented: Literal[True] = True
    knowledge_hitl_governance_contract_implemented: Literal[True] = True
    knowledge_surface_activation_implemented: Literal[False] = False
    knowledge_retrieval_execution_implemented: Literal[False] = False
    knowledge_scoring_execution_implemented: Literal[False] = False
    knowledge_storage_write_implemented: Literal[False] = False
    knowledge_source_record_update_implemented: Literal[False] = False
    knowledge_graph_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    activated_knowledge_surface_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_retrieval_ids: tuple[str, ...] = Field(default_factory=tuple)
    computed_knowledge_score_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_kb_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    updated_source_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_knowledge_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _knowledge_graph_matches_sources(self) -> Self:
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.covered_roadmap_items != (UNIFIED_KNOWLEDGE_GRAPH_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 4 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.knowledge_node_ids != tuple(
            node.knowledge_node_id for node in self.knowledge_nodes
        ):
            raise ValueError("knowledge_node_ids must match knowledge_nodes")
        if len(set(self.knowledge_node_ids)) != len(self.knowledge_node_ids):
            raise ValueError("knowledge_node_ids must be unique")
        if self.knowledge_edge_ids != tuple(
            edge.edge_id for edge in self.knowledge_edges
        ):
            raise ValueError("knowledge_edge_ids must match knowledge_edges")
        if len(set(self.knowledge_edge_ids)) != len(self.knowledge_edge_ids):
            raise ValueError("knowledge_edge_ids must be unique")
        if tuple(node.memory_node_id for node in self.knowledge_nodes) != (
            self.memory_node_ids
        ):
            raise ValueError("knowledge nodes must follow memory graph nodes")
        if tuple(node.cognitive_node_id for node in self.knowledge_nodes) != (
            self.backbone_node_ids
        ):
            raise ValueError("knowledge nodes must follow cognitive graph nodes")
        if tuple(edge.memory_edge_id for edge in self.knowledge_edges) != (
            self.memory_edge_ids
        ):
            raise ValueError("knowledge edges must follow memory graph edges")
        if tuple(edge.cognitive_edge_id for edge in self.knowledge_edges) != (
            self.backbone_edge_ids
        ):
            raise ValueError("knowledge edges must follow cognitive graph edges")
        source_signal_ids = tuple(
            signal_id
            for node in self.knowledge_nodes
            for signal_id in node.source_signal_ids
        )
        if self.source_signal_ids != source_signal_ids:
            raise ValueError("source_signal_ids must match knowledge nodes")
        if self.source_signal_id_count != len(self.source_signal_ids):
            raise ValueError("source_signal_id_count must match source signals")
        knowledge_node = self._knowledge_source_node()
        if knowledge_node.source_signal_ids != self.v6_3_entry_ids:
            raise ValueError("v6_3_entry_ids must match knowledge source node")
        if self.v6_3_entry_count != len(self.v6_3_entry_ids):
            raise ValueError("v6_3_entry_count must match entries")
        if knowledge_node.knowledge_source_roles != self.v6_3_source_plan_roles:
            raise ValueError("v6_3_source_plan_roles must match knowledge node")
        if knowledge_node.knowledge_source_item_ids != self.v6_3_source_item_ids:
            raise ValueError("v6_3_source_item_ids must match knowledge node")
        if self.v6_3_source_item_count != len(self.v6_3_source_item_ids):
            raise ValueError("v6_3_source_item_count must match source items")
        if self.v6_3_roadmap_item_count != len(self.v6_3_roadmap_items):
            raise ValueError("v6_3_roadmap_item_count must match roadmap items")
        declared_knowledge_nodes = set(self.knowledge_node_ids)
        for edge in self.knowledge_edges:
            if edge.from_knowledge_node_id not in declared_knowledge_nodes:
                raise ValueError("edge from_knowledge_node_id must be declared")
            if edge.to_knowledge_node_id not in declared_knowledge_nodes:
                raise ValueError("edge to_knowledge_node_id must be declared")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.activated_knowledge_surface_ids,
                self.executed_retrieval_ids,
                self.computed_knowledge_score_ids,
                self.written_kb_record_ids,
                self.updated_source_record_ids,
                self.mutated_knowledge_node_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "knowledge mutation, storage, retrieval, and HITL ids must be empty",
            )
        if not all(node.advisory_only for node in self.knowledge_nodes):
            raise ValueError("all knowledge nodes must be advisory only")
        if not all(edge.advisory_only for edge in self.knowledge_edges):
            raise ValueError("all knowledge edges must be advisory only")
        return self

    def _knowledge_source_node(self) -> UnifiedKnowledgeGraphNode:
        for node in self.knowledge_nodes:
            if node.knowledge_node_id == self.v6_3_knowledge_node_id:
                return node
        raise ValueError("v6_3_knowledge_node_id must be declared")


def build_unified_knowledge_graph(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    memory_graph: UnifiedMemoryGraphPlan | None = None,
    knowledge_core: KnowledgeEvolutionCoreSurfacePlan | None = None,
) -> UnifiedKnowledgeGraphPlan:
    """Build a read-only knowledge graph over the V6.6 graph backbone."""

    graph = memory_graph or build_unified_memory_graph(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    knowledge_plan = knowledge_core or build_knowledge_evolution_core_surface(
        route=graph.route_name,
        task_type=graph.task_type,
        execution_mode_id=execution_mode_id,
    )
    knowledge_nodes = _knowledge_nodes(graph, knowledge_plan)
    knowledge_edges = _knowledge_edges(graph)
    return UnifiedKnowledgeGraphPlan(
        route_name=graph.route_name,
        task_type=graph.task_type,
        execution_mode_ids=graph.execution_mode_ids,
        backbone_graph_role=graph.backbone_graph_role,
        backbone_graph_serialization_version=graph.backbone_graph_serialization_version,
        memory_graph_role=graph.role,
        memory_graph_serialization_version=graph.serialization_version,
        backbone_node_ids=graph.backbone_node_ids,
        backbone_edge_ids=graph.backbone_edge_ids,
        memory_node_ids=graph.memory_node_ids,
        memory_edge_ids=graph.memory_edge_ids,
        capabilities=graph.capabilities,
        knowledge_nodes=knowledge_nodes,
        knowledge_node_ids=tuple(node.knowledge_node_id for node in knowledge_nodes),
        knowledge_edges=knowledge_edges,
        knowledge_edge_ids=tuple(edge.edge_id for edge in knowledge_edges),
        v6_3_knowledge_node_id="knowledge::v6_3_knowledge_node",
        v6_3_knowledge_source_role=knowledge_plan.role,
        v6_3_knowledge_serialization_version=knowledge_plan.serialization_version,
        v6_3_source_plan_roles=knowledge_plan.source_plan_roles,
        v6_3_source_plan_serialization_versions=(
            knowledge_plan.source_plan_serialization_versions
        ),
        v6_3_entry_ids=knowledge_plan.entry_ids,
        v6_3_entry_count=knowledge_plan.entry_count,
        v6_3_source_item_ids=knowledge_plan.source_item_ids,
        v6_3_source_item_count=knowledge_plan.source_item_count,
        v6_3_roadmap_items=knowledge_plan.covered_roadmap_items,
        v6_3_roadmap_item_count=knowledge_plan.covered_roadmap_item_count,
        covered_roadmap_items=(UNIFIED_KNOWLEDGE_GRAPH_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        source_signal_ids=tuple(
            signal_id
            for node in knowledge_nodes
            for signal_id in node.source_signal_ids
        ),
        source_signal_id_count=sum(
            node.source_signal_count for node in knowledge_nodes
        ),
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def unified_knowledge_graph_node_by_id(
    knowledge_node_id: str,
    graph: UnifiedKnowledgeGraphPlan | None = None,
) -> UnifiedKnowledgeGraphNode | None:
    """Return one knowledge graph node without activating knowledge surfaces."""

    source_graph = graph or build_unified_knowledge_graph()
    for node in source_graph.knowledge_nodes:
        if node.knowledge_node_id == knowledge_node_id:
            return node
    return None


def unified_knowledge_graph_nodes_for_layer(
    layer: CognitiveOSLayer,
    graph: UnifiedKnowledgeGraphPlan | None = None,
) -> tuple[UnifiedKnowledgeGraphNode, ...]:
    """Return knowledge graph nodes for one cognitive layer."""

    source_graph = graph or build_unified_knowledge_graph()
    return tuple(node for node in source_graph.knowledge_nodes if node.layer == layer)


def unified_knowledge_graph_edge_by_id(
    edge_id: str,
    graph: UnifiedKnowledgeGraphPlan | None = None,
) -> UnifiedKnowledgeGraphEdge | None:
    """Return one knowledge graph edge without activating the graph."""

    source_graph = graph or build_unified_knowledge_graph()
    for edge in source_graph.knowledge_edges:
        if edge.edge_id == edge_id:
            return edge
    return None


def _knowledge_nodes(
    graph: UnifiedMemoryGraphPlan,
    knowledge_plan: KnowledgeEvolutionCoreSurfacePlan,
) -> tuple[UnifiedKnowledgeGraphNode, ...]:
    knowledge_roles = {
        "learning": "learning-derived knowledge context",
        "memory": "memory-grounded knowledge context",
        "knowledge": "V6.3 knowledge evolution source context",
        "research": "research evidence knowledge context",
        "self_evolution": "evolution proposal knowledge context",
        "cognitive_core": "cognitive-core knowledge integration context",
    }
    return tuple(
        UnifiedKnowledgeGraphNode(
            knowledge_node_id=f"knowledge::{node.cognitive_node_id}",
            memory_node_id=node.memory_node_id,
            cognitive_node_id=node.cognitive_node_id,
            capability=node.capability,
            layer=node.layer,
            source_snapshot_id=node.source_snapshot_id,
            source_signal_ids=node.source_signal_ids,
            source_signal_count=node.source_signal_count,
            knowledge_role=knowledge_roles[node.layer],
            knowledge_source_roles=_knowledge_source_roles(node, knowledge_plan),
            knowledge_source_item_ids=_knowledge_source_item_ids(
                node,
                knowledge_plan,
            ),
            knowledge_source_item_count=len(
                _knowledge_source_item_ids(node, knowledge_plan),
            ),
            ownership_boundary=node.ownership_boundary,
            dependency_contracts=(
                "knowledge graph follows unified memory graph",
                node.dependency_contracts[0],
            ),
            governance_contracts=(
                "knowledge graph does not authorize KB writes",
                node.governance_contracts[0],
            ),
            explanation_contracts=(
                "knowledge graph explains source provenance and freshness",
                node.explanation_contracts[0],
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for node in graph.memory_nodes
    )


def _knowledge_edges(
    graph: UnifiedMemoryGraphPlan,
) -> tuple[UnifiedKnowledgeGraphEdge, ...]:
    return tuple(
        UnifiedKnowledgeGraphEdge(
            edge_id=(
                f"{_knowledge_node_id(edge.from_memory_node_id)}"
                f"->{_knowledge_node_id(edge.to_memory_node_id)}"
            ),
            from_knowledge_node_id=_knowledge_node_id(edge.from_memory_node_id),
            to_knowledge_node_id=_knowledge_node_id(edge.to_memory_node_id),
            memory_edge_id=edge.edge_id,
            cognitive_edge_id=edge.cognitive_edge_id,
            relationship=f"knowledge_{edge.relationship}",
            dependency_trace=(
                "knowledge dependency mirrors unified memory graph edge",
                edge.dependency_trace[0],
            ),
            governance_trace=(
                "knowledge dependency does not authorize KB writes",
                edge.governance_trace[0],
            ),
            explanation_trace=(
                "knowledge dependency cites memory and cognitive graph traces",
                edge.explanation_trace[0],
            ),
        )
        for edge in graph.memory_edges
    )


def _knowledge_source_roles(
    node: UnifiedMemoryGraphNode,
    knowledge_plan: KnowledgeEvolutionCoreSurfacePlan,
) -> tuple[str, ...]:
    if node.layer == "knowledge":
        return knowledge_plan.source_plan_roles
    return (node.source_snapshot_id,)


def _knowledge_source_item_ids(
    node: UnifiedMemoryGraphNode,
    knowledge_plan: KnowledgeEvolutionCoreSurfacePlan,
) -> tuple[str, ...]:
    if node.layer == "knowledge":
        return knowledge_plan.source_item_ids
    return node.source_signal_ids


def _knowledge_node_id(memory_node_id: str) -> str:
    return f"knowledge::{memory_node_id.removeprefix('memory::')}"
