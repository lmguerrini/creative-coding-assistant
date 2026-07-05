"""LangGraph builder for the assistant workflow runtime."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.runtime.nodes.handlers import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
    AssistantWorkflowRuntime,
)
from creative_coding_assistant.orchestration.runtime.nodes.registry import (
    registered_workflow_conditional_edge_specs,
    registered_workflow_node_specs,
)
from creative_coding_assistant.orchestration.workflow import (
    WorkflowStep,
    begin_assistant_workflow,
)


def build_initial_workflow_graph_state(
    request: AssistantRequest,
) -> AssistantWorkflowGraphState:
    return {"workflow_state": begin_assistant_workflow(request)}


def build_assistant_workflow_graph() -> Any:
    graph = new_assistant_workflow_state_graph()
    add_assistant_workflow_nodes(graph)
    add_assistant_workflow_edges(graph)
    return graph.compile()


def new_assistant_workflow_state_graph() -> Any:
    return StateGraph(
        AssistantWorkflowGraphState,
        context_schema=AssistantWorkflowGraphContext,
    )


def add_assistant_workflow_nodes(graph: Any) -> None:
    for spec in registered_workflow_node_specs():
        graph.add_node(spec.name, spec.handler)


def add_assistant_workflow_edges(graph: Any) -> None:
    graph.add_edge(START, ASSISTANT_WORKFLOW_NODE_ORDER[0])
    for spec in registered_workflow_conditional_edge_specs():
        graph.add_conditional_edges(spec.source, spec.selector, spec.targets)
    graph.add_edge(WorkflowStep.FAILURE.value, END)


def stream_assistant_workflow_events(
    *,
    graph: Any,
    request: AssistantRequest,
    runtime: AssistantWorkflowRuntime,
) -> Iterator[StreamEvent]:
    initial_state = build_initial_workflow_graph_state(request)
    for item in graph.stream(
        initial_state,
        config={"recursion_limit": ASSISTANT_WORKFLOW_RECURSION_LIMIT},
        context={"runtime": runtime},
        stream_mode="custom",
    ):
        if isinstance(item, StreamEvent):
            yield item
