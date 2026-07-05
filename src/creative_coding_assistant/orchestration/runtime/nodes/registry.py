"""Registration surface for assistant workflow runtime nodes."""

from __future__ import annotations

from langgraph.graph import END

from creative_coding_assistant.orchestration.runtime.nodes import handlers, transitions
from creative_coding_assistant.orchestration.workflow import WorkflowStep


def registered_workflow_node_specs() -> tuple[handlers._WorkflowGraphNodeSpec, ...]:
    """Return node handler registrations in graph execution order."""

    return (
        handlers._WorkflowGraphNodeSpec("intake", handlers._intake_node),
        handlers._WorkflowGraphNodeSpec("routing", handlers._routing_node),
        handlers._WorkflowGraphNodeSpec("memory", handlers._memory_node),
        handlers._WorkflowGraphNodeSpec("retrieval", handlers._retrieval_node),
        handlers._WorkflowGraphNodeSpec(
            "context_assembly",
            handlers._context_assembly_node,
        ),
        handlers._WorkflowGraphNodeSpec("prompt_input", handlers._prompt_input_node),
        handlers._WorkflowGraphNodeSpec("planning", handlers._planning_node),
        handlers._WorkflowGraphNodeSpec("director", handlers._director_node),
        handlers._WorkflowGraphNodeSpec("reasoning", handlers._reasoning_node),
        handlers._WorkflowGraphNodeSpec(
            "prompt_rendering",
            handlers._prompt_rendering_node,
        ),
        handlers._WorkflowGraphNodeSpec("generation", handlers._generation_node),
        handlers._WorkflowGraphNodeSpec(
            "artifact_extraction",
            handlers._artifact_extraction_node,
        ),
        handlers._WorkflowGraphNodeSpec(
            "preview_preparation",
            handlers._preview_preparation_node,
        ),
        handlers._WorkflowGraphNodeSpec(
            "artifact_critique",
            handlers._artifact_critique_node,
        ),
        handlers._WorkflowGraphNodeSpec("review", handlers._review_node),
        handlers._WorkflowGraphNodeSpec("refinement", handlers._refinement_node),
        handlers._WorkflowGraphNodeSpec("finalization", handlers._finalization_node),
        handlers._WorkflowGraphNodeSpec("failure", handlers._failure_node),
    )


def registered_workflow_conditional_edge_specs() -> tuple[
    handlers._WorkflowGraphConditionalEdgeSpec, ...
]:
    """Return registered conditional transitions for the workflow graph."""

    return (
        *_linear_workflow_edge_specs_before_review(),
        handlers._WorkflowGraphConditionalEdgeSpec(
            source=WorkflowStep.REVIEW.value,
            selector=transitions.next_node_after_review,
            targets={
                WorkflowStep.FINALIZATION.value: WorkflowStep.FINALIZATION.value,
                WorkflowStep.REFINEMENT.value: WorkflowStep.REFINEMENT.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        ),
        handlers._WorkflowGraphConditionalEdgeSpec(
            source=WorkflowStep.REFINEMENT.value,
            selector=transitions.next_node_selector(WorkflowStep.GENERATION.value),
            targets={
                WorkflowStep.GENERATION.value: WorkflowStep.GENERATION.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        ),
        handlers._WorkflowGraphConditionalEdgeSpec(
            source=WorkflowStep.FINALIZATION.value,
            selector=transitions.next_node_after_finalization,
            targets={
                "end": END,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        ),
    )


def registered_workflow_final_payload_keys() -> tuple[str, ...]:
    """Return final-event payload keys registered by the workflow runtime."""

    return handlers.assistant_workflow_final_payload_keys()


def registered_workflow_model_payload_specs() -> tuple[
    handlers._WorkflowModelPayloadSpec, ...
]:
    """Return workflow-state payload mappings registered by the runtime."""

    return handlers.assistant_workflow_model_payload_specs()


def _linear_workflow_edge_specs_before_review() -> tuple[
    handlers._WorkflowGraphConditionalEdgeSpec, ...
]:
    review_index = handlers.ASSISTANT_WORKFLOW_NODE_ORDER.index(
        WorkflowStep.REVIEW.value
    )
    linear_nodes = handlers.ASSISTANT_WORKFLOW_NODE_ORDER[: review_index + 1]
    return tuple(
        _linear_workflow_edge_spec(current_node, next_node)
        for current_node, next_node in zip(
            linear_nodes,
            linear_nodes[1:],
            strict=False,
        )
    )


def _linear_workflow_edge_spec(
    current_node: str,
    next_node: str,
) -> handlers._WorkflowGraphConditionalEdgeSpec:
    if current_node == WorkflowStep.PROMPT_INPUT.value:
        return handlers._WorkflowGraphConditionalEdgeSpec(
            source=current_node,
            selector=transitions.next_node_after_prompt_input,
            targets={
                WorkflowStep.PLANNING.value: WorkflowStep.PLANNING.value,
                WorkflowStep.FINALIZATION.value: WorkflowStep.FINALIZATION.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        )
    return handlers._WorkflowGraphConditionalEdgeSpec(
        source=current_node,
        selector=transitions.next_node_selector(next_node),
        targets={
            next_node: next_node,
            WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
        },
    )
