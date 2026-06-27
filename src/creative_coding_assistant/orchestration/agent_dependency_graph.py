"""Passive V4.2 agent dependency graph metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEWS,
)

DependencyNodeType = Literal[
    "orchestration_stage",
    "shared_context_view",
    "agent",
]
DependencyEdgeType = Literal[
    "stage_precedes",
    "stage_scopes_context_view",
    "context_view_required_by_agent",
    "agent_feeds_downstream_stage",
]

AGENT_DEPENDENCY_NODE_SERIALIZATION_VERSION = "agent_dependency_node.v1"
AGENT_DEPENDENCY_EDGE_SERIALIZATION_VERSION = "agent_dependency_edge.v1"
AGENT_DEPENDENCY_GRAPH_SERIALIZATION_VERSION = "agent_dependency_graph.v1"
AGENT_DEPENDENCY_GRAPH_AUTHORITY_BOUNDARY = (
    "Agent dependency graph metadata describes static upstream dependencies, "
    "downstream consumers, required inputs, context view relationships, and "
    "blocked cyclic patterns only; it does not execute graph scheduling, "
    "change workflow node order, perform runtime orchestration, invoke agents, "
    "route providers or models, trigger retries, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "graph_scheduling_execution",
    "workflow_node_order_change",
    "runtime_orchestration",
    "agent_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "context_materialization",
    "generated_output_modification",
)

_BLOCKED_CYCLIC_PATTERNS = (
    "agent_self_dependency",
    "downstream_stage_back_edge",
    "context_view_to_stage_feedback",
    "final_synthesis_to_upstream_agent",
)

_STAGE_ORDER = (
    "foundational_context",
    "domain_context",
    "execution_context",
    "quality_review",
    "refinement_context",
    "final_synthesis",
)

_AGENT_STAGE: dict[str, str] = {
    "planner_agent": "foundational_context",
    "research_agent": "foundational_context",
    "style_agent": "domain_context",
    "art_direction_agent": "domain_context",
    "narrative_symbolic_agent": "domain_context",
    "runtime_agent": "execution_context",
    "artifact_agent": "execution_context",
    "aesthetic_critic_agent": "quality_review",
    "creative_curator_agent": "quality_review",
    "critic_agent": "quality_review",
    "refiner_agent": "refinement_context",
    "final_synthesizer_agent": "final_synthesis",
}


class AgentDependencyNode(BaseModel):
    """Static dependency graph node metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=160)
    node_type: DependencyNodeType
    stage_id: str = Field(min_length=1, max_length=80)
    agent_id: str | None = Field(default=None, max_length=80)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    upstream_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    downstream_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    source_registries: tuple[str, ...] = Field(min_length=1, max_length=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    scheduling_implemented: Literal[False] = False
    runtime_orchestration_implemented: Literal[False] = False
    workflow_node_order_changed: Literal[False] = False
    serialization_version: Literal["agent_dependency_node.v1"] = (
        AGENT_DEPENDENCY_NODE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDependencyEdge(BaseModel):
    """Static dependency graph edge metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    edge_type: DependencyEdgeType
    from_node_id: str = Field(min_length=1, max_length=160)
    to_node_id: str = Field(min_length=1, max_length=160)
    required_input: str = Field(min_length=1, max_length=120)
    dependency_boundary: str = Field(min_length=1, max_length=420)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    scheduling_implemented: Literal[False] = False
    runtime_orchestration_implemented: Literal[False] = False
    serialization_version: Literal["agent_dependency_edge.v1"] = (
        AGENT_DEPENDENCY_EDGE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDependencyGraphRegistry(BaseModel):
    """Stable passive dependency graph for V4.2 orchestration metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_dependency_graph_registry"] = (
        "agent_dependency_graph_registry"
    )
    serialization_version: Literal["agent_dependency_graph.v1"] = (
        AGENT_DEPENDENCY_GRAPH_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_DEPENDENCY_GRAPH_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    nodes: tuple[AgentDependencyNode, ...] = Field(min_length=30, max_length=30)
    edges: tuple[AgentDependencyEdge, ...] = Field(min_length=1, max_length=80)
    node_ids: tuple[str, ...] = Field(min_length=30, max_length=30)
    edge_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    stage_order: tuple[str, ...] = Field(min_length=6, max_length=6)
    topological_node_order: tuple[str, ...] = Field(min_length=30, max_length=30)
    node_count: int = Field(ge=30, le=30)
    edge_count: int = Field(ge=1, le=80)
    source_registries: tuple[str, ...] = Field(min_length=3, max_length=3)
    blocked_cyclic_patterns: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    scheduling_implemented: Literal[False] = False
    runtime_orchestration_implemented: Literal[False] = False
    workflow_node_order_changed: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_is_static_and_acyclic(self) -> Self:
        derived_node_ids = tuple(node.node_id for node in self.nodes)
        derived_edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(set(derived_node_ids)) != len(derived_node_ids):
            raise ValueError("node_ids must be unique")
        if len(set(derived_edge_ids)) != len(derived_edge_ids):
            raise ValueError("edge_ids must be unique")
        if self.node_ids != derived_node_ids:
            raise ValueError("node_ids must match nodes")
        if self.edge_ids != derived_edge_ids:
            raise ValueError("edge_ids must match edges")
        if self.topological_node_order != derived_node_ids:
            raise ValueError("topological_node_order must match node order")
        if self.node_count != len(self.nodes):
            raise ValueError("node_count must match nodes")
        if self.edge_count != len(self.edges):
            raise ValueError("edge_count must match edges")

        node_set = set(self.node_ids)
        order_index = {
            node_id: index for index, node_id in enumerate(self.topological_node_order)
        }
        for edge in self.edges:
            if edge.from_node_id not in node_set or edge.to_node_id not in node_set:
                raise ValueError("edges must reference known nodes")
            if edge.from_node_id == edge.to_node_id:
                raise ValueError("dependency graph must not contain self edges")
            if order_index[edge.from_node_id] >= order_index[edge.to_node_id]:
                raise ValueError("dependency graph must be acyclic")
        return self


def agent_dependency_graph_registry() -> AgentDependencyGraphRegistry:
    """Return passive V4.2 agent dependency graph metadata."""

    return AGENT_DEPENDENCY_GRAPH_REGISTRY


def agent_dependency_node_by_id(
    node_id: str,
    registry: AgentDependencyGraphRegistry | None = None,
) -> AgentDependencyNode | None:
    """Return one dependency node without executing graph scheduling."""

    source_registry = registry or AGENT_DEPENDENCY_GRAPH_REGISTRY
    for node in source_registry.nodes:
        if node.node_id == node_id:
            return node
    return None


def agent_dependency_downstream_nodes(
    node_id: str,
    registry: AgentDependencyGraphRegistry | None = None,
) -> tuple[AgentDependencyNode, ...]:
    """Return downstream node metadata without orchestrating execution."""

    source_registry = registry or AGENT_DEPENDENCY_GRAPH_REGISTRY
    node = agent_dependency_node_by_id(node_id, source_registry)
    if node is None:
        return ()
    node_by_id = {candidate.node_id: candidate for candidate in source_registry.nodes}
    return tuple(node_by_id[downstream] for downstream in node.downstream_node_ids)


def agent_dependency_upstream_nodes(
    node_id: str,
    registry: AgentDependencyGraphRegistry | None = None,
) -> tuple[AgentDependencyNode, ...]:
    """Return upstream node metadata without materializing dependencies."""

    source_registry = registry or AGENT_DEPENDENCY_GRAPH_REGISTRY
    node = agent_dependency_node_by_id(node_id, source_registry)
    if node is None:
        return ()
    node_by_id = {candidate.node_id: candidate for candidate in source_registry.nodes}
    return tuple(node_by_id[upstream] for upstream in node.upstream_node_ids)


def _stage_node_id(stage_id: str) -> str:
    return f"stage::{stage_id}"


def _view_node_id(agent_id: str) -> str:
    return f"context_view::{agent_id}"


def _agent_node_id(agent_id: str) -> str:
    return f"agent::{agent_id}"


def _edge(
    edge_type: DependencyEdgeType,
    from_node_id: str,
    to_node_id: str,
    required_input: str,
) -> AgentDependencyEdge:
    return AgentDependencyEdge(
        edge_id=f"{edge_type}::{from_node_id}->{to_node_id}",
        edge_type=edge_type,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        required_input=required_input,
        dependency_boundary=(
            "This edge is a passive dependency declaration only; it does not "
            "schedule graph execution, invoke agents, change workflow node "
            "order, or materialize runtime orchestration."
        ),
    )


def _edge_specs() -> tuple[AgentDependencyEdge, ...]:
    edges: list[AgentDependencyEdge] = []
    for index, stage_id in enumerate(_STAGE_ORDER[:-1]):
        edges.append(
            _edge(
                "stage_precedes",
                _stage_node_id(stage_id),
                _stage_node_id(_STAGE_ORDER[index + 1]),
                "stage_order_metadata",
            )
        )
    for stage_id in _STAGE_ORDER:
        stage_agents = tuple(
            agent_id
            for agent_id in (contract.agent_id for contract in AGENT_CONTRACTS)
            if _AGENT_STAGE[agent_id] == stage_id
        )
        for agent_id in stage_agents:
            edges.append(
                _edge(
                    "stage_scopes_context_view",
                    _stage_node_id(stage_id),
                    _view_node_id(agent_id),
                    "stage_context_scope_metadata",
                )
            )
            edges.append(
                _edge(
                    "context_view_required_by_agent",
                    _view_node_id(agent_id),
                    _agent_node_id(agent_id),
                    "shared_context_view_metadata",
                )
            )
            stage_index = _STAGE_ORDER.index(stage_id)
            if stage_index < len(_STAGE_ORDER) - 1:
                edges.append(
                    _edge(
                        "agent_feeds_downstream_stage",
                        _agent_node_id(agent_id),
                        _stage_node_id(_STAGE_ORDER[stage_index + 1]),
                        "agent_handoff_metadata",
                    )
                )
    return tuple(edges)


def _node_order() -> tuple[str, ...]:
    ordered: list[str] = []
    for stage_id in _STAGE_ORDER:
        ordered.append(_stage_node_id(stage_id))
        stage_agents = tuple(
            contract.agent_id
            for contract in AGENT_CONTRACTS
            if _AGENT_STAGE[contract.agent_id] == stage_id
        )
        ordered.extend(_view_node_id(agent_id) for agent_id in stage_agents)
        ordered.extend(_agent_node_id(agent_id) for agent_id in stage_agents)
    return tuple(ordered)


_EDGES = _edge_specs()
_NODE_ORDER = _node_order()


def _node(node_id: str) -> AgentDependencyNode:
    upstream = tuple(edge.from_node_id for edge in _EDGES if edge.to_node_id == node_id)
    downstream = tuple(edge.to_node_id for edge in _EDGES if edge.from_node_id == node_id)
    if node_id.startswith("stage::"):
        stage_id = node_id.removeprefix("stage::")
        return AgentDependencyNode(
            node_id=node_id,
            node_type="orchestration_stage",
            stage_id=stage_id,
            required_inputs=("stage_order_metadata",),
            upstream_node_ids=upstream,
            downstream_node_ids=downstream,
            source_registries=("agent_contract_registry",),
        )
    if node_id.startswith("context_view::"):
        agent_id = node_id.removeprefix("context_view::")
        view = next(view for view in SHARED_CONTEXT_VIEWS if view.agent_id == agent_id)
        return AgentDependencyNode(
            node_id=node_id,
            node_type="shared_context_view",
            stage_id=_AGENT_STAGE[agent_id],
            agent_id=agent_id,
            required_inputs=view.visible_metadata_keys,
            upstream_node_ids=upstream,
            downstream_node_ids=downstream,
            source_registries=("shared_context_view_registry",),
        )
    agent_id = node_id.removeprefix("agent::")
    contract = next(contract for contract in AGENT_CONTRACTS if contract.agent_id == agent_id)
    return AgentDependencyNode(
        node_id=node_id,
        node_type="agent",
        stage_id=_AGENT_STAGE[agent_id],
        agent_id=agent_id,
        required_inputs=contract.required_inputs + ("shared_context_view_metadata",),
        upstream_node_ids=upstream,
        downstream_node_ids=downstream,
        source_registries=(
            "agent_contract_registry",
            "shared_context_view_registry",
        ),
    )


AGENT_DEPENDENCY_NODES = tuple(_node(node_id) for node_id in _NODE_ORDER)
AGENT_DEPENDENCY_GRAPH_REGISTRY = AgentDependencyGraphRegistry(
    nodes=AGENT_DEPENDENCY_NODES,
    edges=_EDGES,
    node_ids=tuple(node.node_id for node in AGENT_DEPENDENCY_NODES),
    edge_ids=tuple(edge.edge_id for edge in _EDGES),
    stage_order=_STAGE_ORDER,
    topological_node_order=tuple(node.node_id for node in AGENT_DEPENDENCY_NODES),
    node_count=len(AGENT_DEPENDENCY_NODES),
    edge_count=len(_EDGES),
    source_registries=(
        "agent_contract_registry",
        "shared_context_view_registry",
        "blackboard_memory_registry",
    ),
    blocked_cyclic_patterns=_BLOCKED_CYCLIC_PATTERNS,
)
