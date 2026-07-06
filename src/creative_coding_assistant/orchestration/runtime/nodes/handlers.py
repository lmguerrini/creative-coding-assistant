"""Compatibility surface for assistant workflow runtime nodes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration.runtime.nodes.artifacts import (
    _artifact_critique_node,
    _artifact_extraction_node,
    _preview_preparation_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.constants import (
    _FINAL_EVENT_MODEL_PAYLOAD_KEYS,
    _WORKFLOW_RUNTIME_MODEL_PAYLOAD_SPECS,
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    assistant_workflow_final_payload_keys,
    assistant_workflow_model_payload_specs,
)
from creative_coding_assistant.orchestration.runtime.nodes.context import (
    _context_assembly_node,
    _prompt_input_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
    AssistantWorkflowRuntime,
    GenerationResultLike,
    _GraphNodeHandler,
    _GraphTransitionSelector,
    _WorkflowGraphConditionalEdgeSpec,
    _WorkflowGraphNodeSpec,
    _WorkflowModelPayloadSpec,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import (
    _default_transition_target,
    _emit,
    _emit_artifact_critique_completed,
    _emit_artifact_critique_started,
    _emit_artifact_recommendation,
    _emit_artifact_refinement_requested,
    _emit_artifact_scored,
    _emit_node_completed,
    _emit_node_failed,
    _emit_node_started,
    _emit_refinement_completed,
    _emit_refinement_requested,
    _emit_retry_completed,
    _emit_retry_started,
    _emit_review_outcome,
    _emit_streaming_step,
    _final_event_model_payloads,
    _model_json_payload,
    _node_attempt_count,
    _review_reason_text,
    _serialize_workflow_runtime,
    _step_label,
    _transition_payload,
    _workflow_runtime_model_payloads,
)
from creative_coding_assistant.orchestration.runtime.nodes.finalization import (
    _derive_consistency_validation_result,
    _derive_creative_confidence_result,
    _derive_creative_improvement_planner_result,
    _derive_creative_score_result,
    _derive_evaluation_report_result,
    _derive_reflection_loop_result,
    _derive_self_evaluation_result,
    _failure_node,
    _finalization_node,
    _reflection_planning_metadata,
)
from creative_coding_assistant.orchestration.runtime.nodes.generation import (
    _generation_node,
    _prompt_rendering_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.intake import _intake_node
from creative_coding_assistant.orchestration.runtime.nodes.memory import _memory_node
from creative_coding_assistant.orchestration.runtime.nodes.planning import (
    _derive_director_brief,
    _derive_reasoning_result,
    _director_node,
    _evaluation_planning_metadata,
    _planning_node,
    _reasoning_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.refinement import (
    _refinement_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.registry import (
    _linear_workflow_edge_spec,
    _linear_workflow_edge_specs_before_review,
    registered_workflow_conditional_edge_specs,
    registered_workflow_node_specs,
)
from creative_coding_assistant.orchestration.runtime.nodes.retrieval import (
    _retrieval_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.review import _review_node
from creative_coding_assistant.orchestration.runtime.nodes.review_logic import (
    _review_requests_retry,
    _review_transition,
)
from creative_coding_assistant.orchestration.runtime.nodes.routing import _routing_node
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _answer_for_review,
    _append_refinement_guidance,
    _complete_node,
    _failure_answer,
    _failure_info_from_generation_result,
    _format_clarification_answer,
    _handle_workflow_exception,
    _has_pending_failure,
    _pending_failure_info,
    _route_decision,
    _runtime,
    _skip_node,
    _start_graph_workflow_step,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.runtime.nodes.transitions import (
    next_node_after_finalization as _next_node_after_finalization,
)
from creative_coding_assistant.orchestration.runtime.nodes.transitions import (
    next_node_after_prompt_input as _next_node_after_prompt_input,
)
from creative_coding_assistant.orchestration.runtime.nodes.transitions import (
    next_node_after_review as _next_node_after_review,
)
from creative_coding_assistant.orchestration.runtime.nodes.transitions import (
    next_node_or_failure as _next_node_or_failure,
)
from creative_coding_assistant.orchestration.runtime.nodes.transitions import (
    next_node_selector as _next_node_selector,
)

__all__ = (
    "END",
    "START",
    "ASSISTANT_WORKFLOW_NODE_ORDER",
    "ASSISTANT_WORKFLOW_RECURSION_LIMIT",
    "AssistantWorkflowGraphContext",
    "AssistantWorkflowGraphState",
    "AssistantWorkflowRuntime",
    "GenerationResultLike",
    "Runtime",
    "StateGraph",
    "_FINAL_EVENT_MODEL_PAYLOAD_KEYS",
    "_GraphNodeHandler",
    "_GraphTransitionSelector",
    "_WORKFLOW_RUNTIME_MODEL_PAYLOAD_SPECS",
    "_WorkflowGraphConditionalEdgeSpec",
    "_WorkflowGraphNodeSpec",
    "_WorkflowModelPayloadSpec",
    "_add_assistant_workflow_edges",
    "_add_assistant_workflow_nodes",
    "_answer_for_review",
    "_append_refinement_guidance",
    "_artifact_critique_node",
    "_artifact_extraction_node",
    "_assistant_workflow_conditional_edge_specs",
    "_assistant_workflow_node_specs",
    "_complete_node",
    "_context_assembly_node",
    "_default_transition_target",
    "_derive_consistency_validation_result",
    "_derive_creative_confidence_result",
    "_derive_creative_improvement_planner_result",
    "_derive_creative_score_result",
    "_derive_director_brief",
    "_derive_evaluation_report_result",
    "_derive_reasoning_result",
    "_derive_reflection_loop_result",
    "_derive_self_evaluation_result",
    "_director_node",
    "_emit",
    "_emit_artifact_critique_completed",
    "_emit_artifact_critique_started",
    "_emit_artifact_recommendation",
    "_emit_artifact_refinement_requested",
    "_emit_artifact_scored",
    "_emit_node_completed",
    "_emit_node_failed",
    "_emit_node_started",
    "_emit_refinement_completed",
    "_emit_refinement_requested",
    "_emit_retry_completed",
    "_emit_retry_started",
    "_emit_review_outcome",
    "_emit_streaming_step",
    "_evaluation_planning_metadata",
    "_failure_answer",
    "_failure_info_from_generation_result",
    "_failure_node",
    "_final_event_model_payloads",
    "_finalization_node",
    "_format_clarification_answer",
    "_generation_node",
    "_handle_workflow_exception",
    "_has_pending_failure",
    "_intake_node",
    "_linear_workflow_edge_spec",
    "_linear_workflow_edge_specs_before_review",
    "_memory_node",
    "_model_json_payload",
    "_new_assistant_workflow_state_graph",
    "_next_node_after_finalization",
    "_next_node_after_prompt_input",
    "_next_node_after_review",
    "_next_node_or_failure",
    "_next_node_selector",
    "_node_attempt_count",
    "_pending_failure_info",
    "_planning_node",
    "_preview_preparation_node",
    "_prompt_input_node",
    "_prompt_rendering_node",
    "_reasoning_node",
    "_reflection_planning_metadata",
    "_refinement_node",
    "_retrieval_node",
    "_review_node",
    "_review_reason_text",
    "_review_requests_retry",
    "_review_transition",
    "_route_decision",
    "_routing_node",
    "_runtime",
    "_serialize_workflow_runtime",
    "_skip_node",
    "_start_graph_workflow_step",
    "_start_node",
    "_step_label",
    "_transition_payload",
    "_workflow_runtime_model_payloads",
    "_workflow_state",
    "assistant_workflow_conditional_edge_specs",
    "assistant_workflow_final_payload_keys",
    "assistant_workflow_model_payload_specs",
    "assistant_workflow_node_specs",
    "build_assistant_workflow_graph",
    "build_initial_workflow_graph_state",
    "stream_assistant_workflow_events",
)


def build_initial_workflow_graph_state(
    request: AssistantRequest,
) -> AssistantWorkflowGraphState:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        build_initial_workflow_graph_state as _build_initial_workflow_graph_state,
    )

    return _build_initial_workflow_graph_state(request)


def build_assistant_workflow_graph() -> Any:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        build_assistant_workflow_graph as _build_assistant_workflow_graph,
    )

    return _build_assistant_workflow_graph()


def _new_assistant_workflow_state_graph() -> Any:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        new_assistant_workflow_state_graph,
    )

    return new_assistant_workflow_state_graph()


def _add_assistant_workflow_nodes(graph: Any) -> None:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        add_assistant_workflow_nodes,
    )

    add_assistant_workflow_nodes(graph)


def _add_assistant_workflow_edges(graph: Any) -> None:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        add_assistant_workflow_edges,
    )

    add_assistant_workflow_edges(graph)


def _assistant_workflow_node_specs() -> tuple[_WorkflowGraphNodeSpec, ...]:
    return registered_workflow_node_specs()


def _assistant_workflow_conditional_edge_specs() -> tuple[
    _WorkflowGraphConditionalEdgeSpec, ...
]:
    return registered_workflow_conditional_edge_specs()


def assistant_workflow_node_specs() -> tuple[_WorkflowGraphNodeSpec, ...]:
    """Return the registered workflow nodes without compiling the graph."""

    return registered_workflow_node_specs()


def assistant_workflow_conditional_edge_specs() -> tuple[
    _WorkflowGraphConditionalEdgeSpec, ...
]:
    """Return workflow edge specs without executing transition selectors."""

    return registered_workflow_conditional_edge_specs()


def stream_assistant_workflow_events(
    *,
    graph: Any,
    request: AssistantRequest,
    runtime: AssistantWorkflowRuntime,
) -> Iterator[StreamEvent]:
    from creative_coding_assistant.orchestration.runtime.graph_builder import (
        stream_assistant_workflow_events as _stream_assistant_workflow_events,
    )

    yield from _stream_assistant_workflow_events(
        graph=graph,
        request=request,
        runtime=runtime,
    )
