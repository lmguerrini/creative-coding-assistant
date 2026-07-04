"""V6.6 Unified Cognitive Graph backbone metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    evaluate_adaptive_learning_engine,
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
from creative_coding_assistant.orchestration.creative_memory_core_surface import (
    build_creative_memory_core_surface,
)
from creative_coding_assistant.orchestration.knowledge_evolution_core_surface import (
    build_knowledge_evolution_core_surface,
)
from creative_coding_assistant.orchestration.research_core_surface import (
    build_research_core_surface,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    build_self_evolution_core_surface,
)

UNIFIED_COGNITIVE_GRAPH_SERIALIZATION_VERSION = "unified_cognitive_graph.v1"
UNIFIED_COGNITIVE_GRAPH_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Cognitive Graph exposes the V6 architecture as one "
    "inspectable cognitive system. It reads deterministic metadata from "
    "V6.1 Adaptive Learning, V6.2 Creative Memory, V6.3 Knowledge Evolution, "
    "V6.4 Autonomous Research, and V6.5 Self Evolution, links those sources "
    "to the V6.6 Cognitive Core through explicit ownership, dependency, "
    "governance, explainability, safety, and HITL contracts, and does not "
    "mutate prompts, workflows, routing, memory, retrieval, storage, "
    "providers, generated output, HITL decisions, or Runtime Evolution."
)

UNIFIED_COGNITIVE_GRAPH_ROADMAP_ITEM = "Unified Cognitive Graph"

_GRAPH_EDGE_DEFINITIONS = (
    (
        "v6_1_learning_node",
        "v6_2_memory_node",
        "learning_patterns_contextualize_memory",
    ),
    (
        "v6_2_memory_node",
        "v6_3_knowledge_node",
        "memory_grounding_informs_knowledge",
    ),
    (
        "v6_3_knowledge_node",
        "v6_4_research_node",
        "knowledge_gaps_drive_research",
    ),
    (
        "v6_4_research_node",
        "v6_5_self_evolution_node",
        "research_evidence_informs_evolution",
    ),
    (
        "v6_1_learning_node",
        "v6_6_cognitive_core_node",
        "learning_signals_feed_cognitive_core",
    ),
    (
        "v6_2_memory_node",
        "v6_6_cognitive_core_node",
        "memory_signals_feed_cognitive_core",
    ),
    (
        "v6_3_knowledge_node",
        "v6_6_cognitive_core_node",
        "knowledge_signals_feed_cognitive_core",
    ),
    (
        "v6_4_research_node",
        "v6_6_cognitive_core_node",
        "research_signals_feed_cognitive_core",
    ),
    (
        "v6_5_self_evolution_node",
        "v6_6_cognitive_core_node",
        "evolution_governance_feeds_cognitive_core",
    ),
)


class CognitiveOSSourceSnapshot(BaseModel):
    """One source-owned V6 metadata snapshot read by the graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=120)
    capability: CognitiveOSCapability
    layer: CognitiveOSLayer
    source_role: str = Field(min_length=1, max_length=120)
    serialization_version: str = Field(min_length=1, max_length=120)
    signal_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    signal_count: int = Field(ge=1, le=140)
    ownership_boundary: str = Field(min_length=1, max_length=420)
    dependency_role: str = Field(min_length=1, max_length=240)
    governance_role: str = Field(min_length=1, max_length=240)
    explanation_role: str = Field(min_length=1, max_length=240)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _snapshot_matches_signals(self) -> Self:
        if self.signal_count != len(self.signal_ids):
            raise ValueError("signal_count must match signal_ids")
        return self


class CognitiveGraphNode(BaseModel):
    """One node in the Unified Cognitive Graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=120)
    capability: CognitiveOSCapability
    layer: CognitiveOSLayer
    source_snapshot_id: str = Field(min_length=1, max_length=120)
    source_role: str = Field(min_length=1, max_length=120)
    source_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=140)
    source_signal_count: int = Field(ge=1, le=140)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=24)
    upstream_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    downstream_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    ownership_boundary: str = Field(min_length=1, max_length=420)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _node_matches_signals_and_boundaries(self) -> Self:
        if self.source_signal_count != len(self.source_signal_ids):
            raise ValueError("source_signal_count must match source_signal_ids")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveGraphEdge(BaseModel):
    """One dependency-aware edge in the Unified Cognitive Graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    from_node_id: str = Field(min_length=1, max_length=120)
    to_node_id: str = Field(min_length=1, max_length=120)
    relationship: str = Field(min_length=1, max_length=160)
    dependency_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_trace: tuple[str, ...] = Field(min_length=2, max_length=8)
    ownership_boundary: str = Field(min_length=1, max_length=420)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _edge_id_matches_nodes(self) -> Self:
        expected = f"{self.from_node_id}->{self.to_node_id}"
        if self.edge_id != expected:
            raise ValueError("edge_id must match from_node_id and to_node_id")
        return self


class UnifiedCognitiveGraphPlan(BaseModel):
    """The V6.6 backbone graph over all prior V6 cognitive subsystems."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_cognitive_graph"] = "unified_cognitive_graph"
    serialization_version: Literal["unified_cognitive_graph.v1"] = (
        UNIFIED_COGNITIVE_GRAPH_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_COGNITIVE_GRAPH_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    source_snapshots: tuple[CognitiveOSSourceSnapshot, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_snapshot_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    nodes: tuple[CognitiveGraphNode, ...] = Field(min_length=6, max_length=6)
    node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    edges: tuple[CognitiveGraphEdge, ...] = Field(min_length=9, max_length=9)
    edge_ids: tuple[str, ...] = Field(min_length=9, max_length=9)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    upstream_signal_ids: tuple[str, ...] = Field(min_length=131, max_length=131)
    upstream_signal_id_count: int = Field(ge=131, le=131)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_cognitive_graph_implemented: Literal[True] = True
    cognitive_os_backbone_implemented: Literal[True] = True
    cross_capability_dependency_awareness_implemented: Literal[True] = True
    cross_capability_governance_audit_implemented: Literal[True] = True
    capability_ownership_boundary_check_implemented: Literal[True] = True
    unified_graph_consistency_implemented: Literal[True] = True
    registry_consistency_prepared: Literal[True] = True
    cognitive_explainability_contract_implemented: Literal[True] = True
    cognitive_hitl_governance_contract_implemented: Literal[True] = True
    runtime_evolution_implemented: Literal[False] = False
    workflow_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    applied_graph_mutation_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_node_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_edge_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _graph_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive system order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.covered_roadmap_items != (UNIFIED_COGNITIVE_GRAPH_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 2 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.source_snapshot_ids != tuple(
            snapshot.source_id for snapshot in self.source_snapshots
        ):
            raise ValueError("source_snapshot_ids must match source_snapshots")
        if len(set(self.source_snapshot_ids)) != len(self.source_snapshot_ids):
            raise ValueError("source_snapshot_ids must be unique")
        snapshot_layers = tuple(snapshot.layer for snapshot in self.source_snapshots)
        if snapshot_layers != self.layer_order:
            raise ValueError("source_snapshots must follow layer_order")
        snapshot_capabilities = tuple(
            snapshot.capability for snapshot in self.source_snapshots
        )
        if snapshot_capabilities != self.capabilities:
            raise ValueError("source_snapshots must follow capabilities")
        if self.node_ids != tuple(node.node_id for node in self.nodes):
            raise ValueError("node_ids must match nodes")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if self.edge_ids != tuple(edge.edge_id for edge in self.edges):
            raise ValueError("edge_ids must match edges")
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("edge_ids must be unique")
        source_signal_ids = tuple(
            signal_id
            for snapshot in self.source_snapshots
            for signal_id in snapshot.signal_ids
        )
        if self.upstream_signal_ids != source_signal_ids:
            raise ValueError("upstream_signal_ids must match source snapshots")
        if self.upstream_signal_id_count != len(self.upstream_signal_ids):
            raise ValueError("upstream_signal_id_count must match source signals")
        source_by_id = {
            snapshot.source_id: snapshot for snapshot in self.source_snapshots
        }
        for node in self.nodes:
            source = source_by_id.get(node.source_snapshot_id)
            if source is None:
                raise ValueError("node source_snapshot_id must be declared")
            if node.capability != source.capability:
                raise ValueError("node capability must match source snapshot")
            if node.layer != source.layer:
                raise ValueError("node layer must match source snapshot")
            if node.source_signal_ids != source.signal_ids:
                raise ValueError("node source_signal_ids must match snapshot")
        declared_nodes = set(self.node_ids)
        for edge in self.edges:
            if edge.from_node_id not in declared_nodes:
                raise ValueError("edge from_node_id must be declared")
            if edge.to_node_id not in declared_nodes:
                raise ValueError("edge to_node_id must be declared")
        graph_edge_ids = tuple(
            f"{source}->{target}" for source, target, _ in _GRAPH_EDGE_DEFINITIONS
        )
        if self.edge_ids != graph_edge_ids:
            raise ValueError("edge_ids must match V6.6 graph backbone")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.applied_graph_mutation_ids,
                self.mutated_node_ids,
                self.mutated_edge_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError("graph mutation and HITL emission ids must be empty")
        if not all(node.advisory_only for node in self.nodes):
            raise ValueError("all nodes must be advisory only")
        if not all(edge.advisory_only for edge in self.edges):
            raise ValueError("all edges must be advisory only")
        if not all(snapshot.advisory_only for snapshot in self.source_snapshots):
            raise ValueError("all source snapshots must be advisory only")
        return self


def build_unified_cognitive_graph(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
) -> UnifiedCognitiveGraphPlan:
    """Build the V6.6 cognitive backbone without mutating any subsystem."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    source_snapshots = _source_snapshots(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    nodes = _graph_nodes(source_snapshots)
    edges = _graph_edges()
    return UnifiedCognitiveGraphPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_ids=execution_modes.execution_mode_ids,
        layer_order=COGNITIVE_OS_LAYER_ORDER,
        capabilities=COGNITIVE_OS_CAPABILITIES,
        source_snapshots=source_snapshots,
        source_snapshot_ids=tuple(snapshot.source_id for snapshot in source_snapshots),
        nodes=nodes,
        node_ids=tuple(node.node_id for node in nodes),
        edges=edges,
        edge_ids=tuple(edge.edge_id for edge in edges),
        covered_roadmap_items=(UNIFIED_COGNITIVE_GRAPH_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        upstream_signal_ids=tuple(
            signal_id
            for snapshot in source_snapshots
            for signal_id in snapshot.signal_ids
        ),
        upstream_signal_id_count=sum(
            snapshot.signal_count for snapshot in source_snapshots
        ),
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def unified_cognitive_graph_node_by_id(
    node_id: str,
    graph: UnifiedCognitiveGraphPlan | None = None,
) -> CognitiveGraphNode | None:
    """Return one graph node without activating the graph."""

    source_graph = graph or build_unified_cognitive_graph()
    for node in source_graph.nodes:
        if node.node_id == node_id:
            return node
    return None


def unified_cognitive_graph_nodes_for_layer(
    layer: CognitiveOSLayer,
    graph: UnifiedCognitiveGraphPlan | None = None,
) -> tuple[CognitiveGraphNode, ...]:
    """Return graph nodes for one cognitive layer."""

    source_graph = graph or build_unified_cognitive_graph()
    return tuple(node for node in source_graph.nodes if node.layer == layer)


def unified_cognitive_graph_edge_by_id(
    edge_id: str,
    graph: UnifiedCognitiveGraphPlan | None = None,
) -> CognitiveGraphEdge | None:
    """Return one graph edge without activating the graph."""

    source_graph = graph or build_unified_cognitive_graph()
    for edge in source_graph.edges:
        if edge.edge_id == edge_id:
            return edge
    return None


def _source_snapshots(
    *,
    route: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
) -> tuple[CognitiveOSSourceSnapshot, ...]:
    adaptive = evaluate_adaptive_learning_engine(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    memory = build_creative_memory_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    knowledge = build_knowledge_evolution_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    research = build_research_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    evolution = build_self_evolution_core_surface(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    return (
        CognitiveOSSourceSnapshot(
            source_id="v6_1_adaptive_learning",
            capability="V6.1 Adaptive Learning",
            layer="learning",
            source_role=str(adaptive.role),
            serialization_version=str(adaptive.serialization_version),
            signal_ids=tuple(str(signal_id) for signal_id in adaptive.signal_ids),
            signal_count=len(adaptive.signal_ids),
            ownership_boundary="V6.1 owns learning signals; V6.6 only links them.",
            dependency_role="Learning patterns seed cognitive dependency context.",
            governance_role="Learning guardrails remain HITL-governed upstream.",
            explanation_role="Learning signals explain observed strategy history.",
        ),
        CognitiveOSSourceSnapshot(
            source_id="v6_2_creative_memory",
            capability="V6.2 Creative Memory",
            layer="memory",
            source_role=str(memory.role),
            serialization_version=str(memory.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in memory.entry_ids),
            signal_count=len(memory.entry_ids),
            ownership_boundary="V6.2 owns creative memory; V6.6 only links it.",
            dependency_role="Memory grounds cognitive context and identity.",
            governance_role="Memory safety and retrieval policies remain upstream.",
            explanation_role="Memory entries explain user and project continuity.",
        ),
        CognitiveOSSourceSnapshot(
            source_id="v6_3_knowledge_evolution",
            capability="V6.3 Knowledge Evolution",
            layer="knowledge",
            source_role=str(knowledge.role),
            serialization_version=str(knowledge.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in knowledge.entry_ids),
            signal_count=len(knowledge.entry_ids),
            ownership_boundary="V6.3 owns knowledge evolution; V6.6 only links it.",
            dependency_role="Knowledge health frames cognitive evidence quality.",
            governance_role="Knowledge lifecycle and rollback remain upstream.",
            explanation_role="Knowledge metadata explains source and freshness.",
        ),
        CognitiveOSSourceSnapshot(
            source_id="v6_4_autonomous_research",
            capability="V6.4 Autonomous Research",
            layer="research",
            source_role=str(research.role),
            serialization_version=str(research.serialization_version),
            signal_ids=tuple(str(entry_id) for entry_id in research.entry_ids),
            signal_count=len(research.entry_ids),
            ownership_boundary="V6.4 owns research metadata; V6.6 only links it.",
            dependency_role="Research evidence informs cognitive uncertainty.",
            governance_role="Research execution and HITL policy remain upstream.",
            explanation_role="Research surfaces explain evidence provenance.",
        ),
        CognitiveOSSourceSnapshot(
            source_id="v6_5_self_evolution",
            capability="V6.5 Self Evolution",
            layer="self_evolution",
            source_role=str(evolution.role),
            serialization_version=str(evolution.serialization_version),
            signal_ids=tuple(
                str(proposal_id) for proposal_id in evolution.proposal_ids
            ),
            signal_count=len(evolution.proposal_ids),
            ownership_boundary="V6.5 owns evolution proposals; V6.6 only links them.",
            dependency_role="Evolution proposals expose improvement dependencies.",
            governance_role="Proposal application remains HITL-governed upstream.",
            explanation_role="Evolution metadata explains improvement rationale.",
        ),
        CognitiveOSSourceSnapshot(
            source_id="v6_6_cognitive_core",
            capability="V6.6 Cognitive Core",
            layer="cognitive_core",
            source_role="unified_cognitive_graph",
            serialization_version=UNIFIED_COGNITIVE_GRAPH_SERIALIZATION_VERSION,
            signal_ids=(UNIFIED_COGNITIVE_GRAPH_ROADMAP_ITEM,),
            signal_count=1,
            ownership_boundary="V6.6 owns only cognitive OS metadata composition.",
            dependency_role="Cognitive core composes upstream signals into one graph.",
            governance_role="Cognitive governance remains inspectable and HITL-ready.",
            explanation_role="Cognitive graph explains cross-system dependencies.",
        ),
    )


def _graph_nodes(
    source_snapshots: tuple[CognitiveOSSourceSnapshot, ...],
) -> tuple[CognitiveGraphNode, ...]:
    downstream_by_node: dict[str, tuple[str, ...]] = {
        "v6_1_learning_node": (
            "v6_2_memory_node",
            "v6_6_cognitive_core_node",
        ),
        "v6_2_memory_node": (
            "v6_3_knowledge_node",
            "v6_6_cognitive_core_node",
        ),
        "v6_3_knowledge_node": (
            "v6_4_research_node",
            "v6_6_cognitive_core_node",
        ),
        "v6_4_research_node": (
            "v6_5_self_evolution_node",
            "v6_6_cognitive_core_node",
        ),
        "v6_5_self_evolution_node": ("v6_6_cognitive_core_node",),
        "v6_6_cognitive_core_node": (),
    }
    upstream_by_node: dict[str, tuple[str, ...]] = {
        "v6_1_learning_node": (),
        "v6_2_memory_node": ("v6_1_learning_node",),
        "v6_3_knowledge_node": ("v6_2_memory_node",),
        "v6_4_research_node": ("v6_3_knowledge_node",),
        "v6_5_self_evolution_node": ("v6_4_research_node",),
        "v6_6_cognitive_core_node": (
            "v6_1_learning_node",
            "v6_2_memory_node",
            "v6_3_knowledge_node",
            "v6_4_research_node",
            "v6_5_self_evolution_node",
        ),
    }
    node_ids = (
        "v6_1_learning_node",
        "v6_2_memory_node",
        "v6_3_knowledge_node",
        "v6_4_research_node",
        "v6_5_self_evolution_node",
        "v6_6_cognitive_core_node",
    )
    roadmap_items = {
        "v6_6_cognitive_core_node": (UNIFIED_COGNITIVE_GRAPH_ROADMAP_ITEM,),
    }
    return tuple(
        CognitiveGraphNode(
            node_id=node_id,
            capability=snapshot.capability,
            layer=snapshot.layer,
            source_snapshot_id=snapshot.source_id,
            source_role=snapshot.source_role,
            source_signal_ids=snapshot.signal_ids,
            source_signal_count=snapshot.signal_count,
            covered_roadmap_items=roadmap_items.get(node_id, (snapshot.capability,)),
            upstream_node_ids=upstream_by_node[node_id],
            downstream_node_ids=downstream_by_node[node_id],
            ownership_boundary=snapshot.ownership_boundary,
            dependency_contracts=(
                snapshot.dependency_role,
                "dependency edges are metadata only",
            ),
            governance_contracts=(
                snapshot.governance_role,
                "HITL required before behavioral application",
            ),
            explanation_contracts=(
                snapshot.explanation_role,
                "edge traces must explain upstream evidence",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for node_id, snapshot in zip(node_ids, source_snapshots, strict=True)
    )


def _graph_edges() -> tuple[CognitiveGraphEdge, ...]:
    return tuple(
        CognitiveGraphEdge(
            edge_id=f"{from_node_id}->{to_node_id}",
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship=relationship,
            dependency_trace=(
                f"{relationship}: dependency metadata",
                "source ownership remains unchanged",
            ),
            governance_trace=(
                f"{relationship}: governance metadata",
                "HITL required before runtime application",
            ),
            explanation_trace=(
                f"{relationship}: explanation metadata",
                "downstream node cites upstream signal ids",
            ),
            ownership_boundary=(
                "Edge links metadata only; it does not transfer ownership or "
                "authorize mutation."
            ),
        )
        for from_node_id, to_node_id, relationship in _GRAPH_EDGE_DEFINITIONS
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    try:
        return RouteName(str(route).strip())
    except ValueError as exc:
        raise ValueError("route must be a known RouteName") from exc


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    normalized = str(task_type).strip()
    allowed = set(get_args(TaskRoutingType))
    if normalized not in allowed:
        raise ValueError("task_type must be a known TaskRoutingType")
    return cast(TaskRoutingType, normalized)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_modes: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in set(allowed_modes):
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
