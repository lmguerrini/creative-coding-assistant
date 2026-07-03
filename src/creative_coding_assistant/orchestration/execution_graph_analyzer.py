"""V5.1 execution graph analyzer for assistant workflow topology."""

from __future__ import annotations

from typing import Literal, Self

from langgraph.graph import END, START
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    assistant_workflow_conditional_edge_specs,
)

ExecutionGraphEdgeKind = Literal[
    "entry",
    "linear",
    "conditional",
    "short_circuit",
    "retry",
    "failure",
    "terminal",
]

EXECUTION_GRAPH_NODE_SERIALIZATION_VERSION = "execution_graph_node.v1"
EXECUTION_GRAPH_EDGE_SERIALIZATION_VERSION = "execution_graph_edge.v1"
EXECUTION_GRAPH_ANALYSIS_SERIALIZATION_VERSION = "execution_graph_analysis.v1"
EXECUTION_GRAPH_ANALYZER_AUTHORITY_BOUNDARY = (
    "Execution graph analysis inspects the assistant workflow topology, "
    "transition surfaces, retry paths, failure edges, and terminal paths only; "
    "it does not compile or execute the LangGraph graph, invoke node handlers, "
    "change workflow node order, trigger retries, alter provider or model "
    "routing, materialize context, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "langgraph_compilation",
    "workflow_execution",
    "node_handler_invocation",
    "workflow_node_order_change",
    "retry_triggering",
    "provider_or_model_routing",
    "context_materialization",
    "generated_output_modification",
)
_NODE_ANALYSIS_FLAGS = (
    "static_topology_observation",
    "runtime_boundary_preserved",
    "node_handler_not_invoked",
)


class ExecutionGraphNode(BaseModel):
    """Typed node metadata produced by the execution graph analyzer."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=120)
    order_index: int = Field(ge=0)
    workflow_step: str = Field(min_length=1, max_length=120)
    can_enter_failure_path: bool
    analysis_flags: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    node_handler_invocation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_graph_node.v1"] = (
        EXECUTION_GRAPH_NODE_SERIALIZATION_VERSION
    )
    analysis_only: Literal[True] = True


class ExecutionGraphEdge(BaseModel):
    """Typed edge metadata produced by the execution graph analyzer."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    edge_kind: ExecutionGraphEdgeKind
    source_node_id: str = Field(min_length=1, max_length=120)
    target_node_id: str = Field(min_length=1, max_length=120)
    selector_name: str = Field(min_length=1, max_length=120)
    optimization_signals: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    node_handler_invocation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_graph_edge.v1"] = (
        EXECUTION_GRAPH_EDGE_SERIALIZATION_VERSION
    )
    analysis_only: Literal[True] = True


class ExecutionGraphAnalysis(BaseModel):
    """Bounded V5.1 analysis of assistant workflow graph topology."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_graph_analyzer"] = "execution_graph_analyzer"
    serialization_version: Literal["execution_graph_analysis.v1"] = (
        EXECUTION_GRAPH_ANALYSIS_SERIALIZATION_VERSION
    )
    source_graph: Literal["assistant_workflow_graph"] = "assistant_workflow_graph"
    authority_boundary: str = Field(
        default=EXECUTION_GRAPH_ANALYZER_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    node_order: tuple[str, ...] = Field(min_length=1, max_length=40)
    nodes: tuple[ExecutionGraphNode, ...] = Field(min_length=1, max_length=40)
    edges: tuple[ExecutionGraphEdge, ...] = Field(min_length=1, max_length=120)
    start_node_id: str = Field(min_length=1, max_length=120)
    terminal_node_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    critical_path_node_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    branch_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    retry_entry_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    failure_entry_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=40,
    )
    recursion_limit: int = Field(ge=1)
    node_count: int = Field(ge=1, le=40)
    edge_count: int = Field(ge=1, le=120)
    branch_count: int = Field(ge=0, le=40)
    failure_edge_count: int = Field(ge=0, le=40)
    bounded_retry_cycle_detected: bool
    failure_path_reachable: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    analysis_only: Literal[True] = True

    @model_validator(mode="after")
    def _analysis_matches_topology(self) -> Self:
        derived_node_ids = tuple(node.node_id for node in self.nodes)
        derived_edge_ids = tuple(edge.edge_id for edge in self.edges)
        if self.node_order != derived_node_ids:
            raise ValueError("node_order must match nodes")
        if len(set(derived_node_ids)) != len(derived_node_ids):
            raise ValueError("node ids must be unique")
        if len(set(derived_edge_ids)) != len(derived_edge_ids):
            raise ValueError("edge ids must be unique")
        if self.node_count != len(self.nodes):
            raise ValueError("node_count must match nodes")
        if self.edge_count != len(self.edges):
            raise ValueError("edge_count must match edges")

        known_nodes = set(self.node_order)
        allowed_sources = known_nodes | {str(START)}
        allowed_targets = known_nodes | {str(END)}
        node_index = {node_id: index for index, node_id in enumerate(self.node_order)}
        start_edges = tuple(
            edge for edge in self.edges if edge.source_node_id == str(START)
        )
        if len(start_edges) != 1 or start_edges[0].target_node_id != self.start_node_id:
            raise ValueError("analysis must include one start edge to start_node_id")

        terminal_targets = tuple(
            edge.source_node_id
            for edge in self.edges
            if edge.target_node_id == str(END)
        )
        if self.terminal_node_ids != terminal_targets:
            raise ValueError("terminal_node_ids must match edges to END")

        for edge in self.edges:
            if edge.source_node_id not in allowed_sources:
                raise ValueError("edges must reference known source nodes")
            if edge.target_node_id not in allowed_targets:
                raise ValueError("edges must reference known target nodes")
            if edge.source_node_id == str(END):
                raise ValueError("END cannot be an edge source")
            if edge.target_node_id == str(START):
                raise ValueError("START cannot be an edge target")
            if (
                edge.source_node_id in node_index
                and edge.target_node_id in node_index
                and node_index[edge.source_node_id] >= node_index[edge.target_node_id]
                and edge.edge_kind != "retry"
            ):
                raise ValueError("back edges must be retry edges")

        if self.branch_count != len(self.branch_node_ids):
            raise ValueError("branch_count must match branch_node_ids")
        if self.failure_edge_count != len(self.failure_entry_node_ids):
            raise ValueError("failure_edge_count must match failure_entry_node_ids")
        if self.failure_path_reachable and not self.failure_entry_node_ids:
            raise ValueError("failure_path_reachable requires failure edges")
        return self


def analyze_assistant_execution_graph() -> ExecutionGraphAnalysis:
    """Return a bounded V5.1 workflow topology analysis without execution."""

    return ASSISTANT_EXECUTION_GRAPH_ANALYSIS


def execution_graph_node_by_id(
    node_id: str,
    analysis: ExecutionGraphAnalysis | None = None,
) -> ExecutionGraphNode | None:
    """Return one analyzed node without invoking workflow handlers."""

    source_analysis = analysis or ASSISTANT_EXECUTION_GRAPH_ANALYSIS
    for node in source_analysis.nodes:
        if node.node_id == node_id:
            return node
    return None


def execution_graph_edges_from(
    node_id: str,
    analysis: ExecutionGraphAnalysis | None = None,
) -> tuple[ExecutionGraphEdge, ...]:
    """Return analyzed outgoing edges without running graph transitions."""

    source_analysis = analysis or ASSISTANT_EXECUTION_GRAPH_ANALYSIS
    return tuple(
        edge for edge in source_analysis.edges if edge.source_node_id == node_id
    )


def execution_graph_edges_to(
    node_id: str,
    analysis: ExecutionGraphAnalysis | None = None,
) -> tuple[ExecutionGraphEdge, ...]:
    """Return analyzed incoming edges without materializing workflow state."""

    source_analysis = analysis or ASSISTANT_EXECUTION_GRAPH_ANALYSIS
    return tuple(
        edge for edge in source_analysis.edges if edge.target_node_id == node_id
    )


def _nodes() -> tuple[ExecutionGraphNode, ...]:
    return tuple(
        ExecutionGraphNode(
            node_id=node_id,
            order_index=index,
            workflow_step=node_id,
            can_enter_failure_path=node_id != "failure",
            analysis_flags=_NODE_ANALYSIS_FLAGS,
        )
        for index, node_id in enumerate(ASSISTANT_WORKFLOW_NODE_ORDER)
    )


def _edges() -> tuple[ExecutionGraphEdge, ...]:
    edges = [
        ExecutionGraphEdge(
            edge_id=f"{START}->{ASSISTANT_WORKFLOW_NODE_ORDER[0]}",
            edge_kind="entry",
            source_node_id=str(START),
            target_node_id=ASSISTANT_WORKFLOW_NODE_ORDER[0],
            selector_name="langgraph_start",
            optimization_signals=("entry_path",),
        )
    ]
    for spec in assistant_workflow_conditional_edge_specs():
        for target in spec.targets.values():
            source = spec.source
            target_node = str(target)
            edge_kind = _edge_kind(source, target_node)
            edges.append(
                ExecutionGraphEdge(
                    edge_id=f"{source}->{target_node}",
                    edge_kind=edge_kind,
                    source_node_id=source,
                    target_node_id=target_node,
                    selector_name=_selector_name(source),
                    optimization_signals=_optimization_signals(
                        source,
                        target_node,
                        edge_kind,
                    ),
                )
            )
    edges.append(
        ExecutionGraphEdge(
            edge_id=f"failure->{END}",
            edge_kind="terminal",
            source_node_id="failure",
            target_node_id=str(END),
            selector_name="failure_terminal_edge",
            optimization_signals=("terminal_failure_path",),
        )
    )
    return tuple(edges)


def _edge_kind(source: str, target: str) -> ExecutionGraphEdgeKind:
    if target == str(END):
        return "terminal"
    if target == "failure":
        return "failure"
    if source == "prompt_input" and target == "finalization":
        return "short_circuit"
    if source == "review" and target == "refinement":
        return "retry"
    if source == "refinement" and target == "generation":
        return "retry"
    if _is_linear_successor(source, target):
        return "linear"
    return "conditional"


def _selector_name(source: str) -> str:
    if source == "prompt_input":
        return "prompt_input_clarification_selector"
    if source == "review":
        return "review_retry_selector"
    if source == "finalization":
        return "finalization_terminal_selector"
    return "pending_failure_selector"


def _optimization_signals(
    source: str,
    target: str,
    edge_kind: ExecutionGraphEdgeKind,
) -> tuple[str, ...]:
    if edge_kind == "entry":
        return ("entry_path",)
    if edge_kind == "terminal":
        return ("terminal_path",)
    if edge_kind == "failure":
        return ("failure_recovery_path",)
    if edge_kind == "short_circuit":
        return ("clarification_short_circuit", "generation_avoidance_path")
    if source == "review" and target == "refinement":
        return ("bounded_retry_decision", "quality_repair_path")
    if source == "refinement" and target == "generation":
        return ("bounded_retry_reentry", "generation_reuse_path")
    if edge_kind == "conditional":
        return ("conditional_branch_path",)
    return ("linear_execution_path",)


def _is_linear_successor(source: str, target: str) -> bool:
    try:
        source_index = ASSISTANT_WORKFLOW_NODE_ORDER.index(source)
    except ValueError:
        return False
    target_index = source_index + 1
    return (
        target_index < len(ASSISTANT_WORKFLOW_NODE_ORDER)
        and ASSISTANT_WORKFLOW_NODE_ORDER[target_index] == target
    )


def _critical_path_node_ids() -> tuple[str, ...]:
    return tuple(
        node_id
        for node_id in ASSISTANT_WORKFLOW_NODE_ORDER
        if node_id not in {"refinement", "failure"}
    )


def _branch_node_ids(edges: tuple[ExecutionGraphEdge, ...]) -> tuple[str, ...]:
    branch_kinds = {"conditional", "short_circuit", "retry", "failure"}
    ordered: list[str] = []
    for edge in edges:
        if edge.source_node_id == str(START) or edge.source_node_id == "failure":
            continue
        if edge.edge_kind in branch_kinds and edge.source_node_id not in ordered:
            ordered.append(edge.source_node_id)
    return tuple(ordered)


def _retry_entry_node_ids(edges: tuple[ExecutionGraphEdge, ...]) -> tuple[str, ...]:
    return tuple(
        edge.source_node_id
        for edge in edges
        if edge.edge_kind == "retry" and edge.target_node_id != "refinement"
    )


def _failure_entry_node_ids(edges: tuple[ExecutionGraphEdge, ...]) -> tuple[str, ...]:
    return tuple(edge.source_node_id for edge in edges if edge.edge_kind == "failure")


_NODES = _nodes()
_EDGES = _edges()
_BRANCH_NODE_IDS = _branch_node_ids(_EDGES)
_FAILURE_ENTRY_NODE_IDS = _failure_entry_node_ids(_EDGES)
ASSISTANT_EXECUTION_GRAPH_ANALYSIS = ExecutionGraphAnalysis(
    node_order=tuple(node.node_id for node in _NODES),
    nodes=_NODES,
    edges=_EDGES,
    start_node_id=ASSISTANT_WORKFLOW_NODE_ORDER[0],
    terminal_node_ids=tuple(
        edge.source_node_id for edge in _EDGES if edge.target_node_id == str(END)
    ),
    critical_path_node_ids=_critical_path_node_ids(),
    branch_node_ids=_BRANCH_NODE_IDS,
    retry_entry_node_ids=_retry_entry_node_ids(_EDGES),
    failure_entry_node_ids=_FAILURE_ENTRY_NODE_IDS,
    recursion_limit=ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    node_count=len(_NODES),
    edge_count=len(_EDGES),
    branch_count=len(_BRANCH_NODE_IDS),
    failure_edge_count=len(_FAILURE_ENTRY_NODE_IDS),
    bounded_retry_cycle_detected=True,
    failure_path_reachable=bool(_FAILURE_ENTRY_NODE_IDS),
)
