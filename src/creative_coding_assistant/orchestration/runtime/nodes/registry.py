"""Registration surface for assistant workflow runtime nodes."""

from __future__ import annotations

from langgraph.graph import END

from creative_coding_assistant.orchestration.runtime.nodes import transitions
from creative_coding_assistant.orchestration.runtime.nodes.artifacts import (
    _artifact_critique_node,
    _artifact_extraction_node,
    _preview_preparation_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.constants import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    assistant_workflow_final_payload_keys,
    assistant_workflow_model_payload_specs,
)
from creative_coding_assistant.orchestration.runtime.nodes.context import (
    _context_assembly_node,
    _prompt_input_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    _WorkflowGraphConditionalEdgeSpec,
    _WorkflowGraphNodeSpec,
    _WorkflowModelPayloadSpec,
)
from creative_coding_assistant.orchestration.runtime.nodes.director import _director_node
from creative_coding_assistant.orchestration.runtime.nodes.finalization import (
    _failure_node,
    _finalization_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.generation import (
    _generation_node,
    _prompt_rendering_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.intake import _intake_node
from creative_coding_assistant.orchestration.runtime.nodes.memory import _memory_node
from creative_coding_assistant.orchestration.runtime.nodes.planning_node import (
    _planning_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.reasoning import (
    _reasoning_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.refinement import (
    _refinement_node,
)
from creative_coding_assistant.orchestration.runtime.nodes.retrieval import _retrieval_node
from creative_coding_assistant.orchestration.runtime.nodes.review import _review_node
from creative_coding_assistant.orchestration.runtime.nodes.routing import _routing_node
from creative_coding_assistant.orchestration.workflow import WorkflowStep


def registered_workflow_node_specs() -> tuple[_WorkflowGraphNodeSpec, ...]:
    """Return node handler registrations in graph execution order."""

    return (
        _WorkflowGraphNodeSpec("intake", _intake_node),
        _WorkflowGraphNodeSpec("routing", _routing_node),
        _WorkflowGraphNodeSpec("memory", _memory_node),
        _WorkflowGraphNodeSpec("retrieval", _retrieval_node),
        _WorkflowGraphNodeSpec("context_assembly", _context_assembly_node),
        _WorkflowGraphNodeSpec("prompt_input", _prompt_input_node),
        _WorkflowGraphNodeSpec("planning", _planning_node),
        _WorkflowGraphNodeSpec("director", _director_node),
        _WorkflowGraphNodeSpec("reasoning", _reasoning_node),
        _WorkflowGraphNodeSpec("prompt_rendering", _prompt_rendering_node),
        _WorkflowGraphNodeSpec("generation", _generation_node),
        _WorkflowGraphNodeSpec("artifact_extraction", _artifact_extraction_node),
        _WorkflowGraphNodeSpec("preview_preparation", _preview_preparation_node),
        _WorkflowGraphNodeSpec("artifact_critique", _artifact_critique_node),
        _WorkflowGraphNodeSpec("review", _review_node),
        _WorkflowGraphNodeSpec("refinement", _refinement_node),
        _WorkflowGraphNodeSpec("finalization", _finalization_node),
        _WorkflowGraphNodeSpec("failure", _failure_node),
    )


def registered_workflow_conditional_edge_specs() -> tuple[
    _WorkflowGraphConditionalEdgeSpec, ...
]:
    """Return registered conditional transitions for the workflow graph."""

    return (
        *_linear_workflow_edge_specs_before_review(),
        _WorkflowGraphConditionalEdgeSpec(
            source=WorkflowStep.REVIEW.value,
            selector=transitions.next_node_after_review,
            targets={
                WorkflowStep.FINALIZATION.value: WorkflowStep.FINALIZATION.value,
                WorkflowStep.REFINEMENT.value: WorkflowStep.REFINEMENT.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        ),
        _WorkflowGraphConditionalEdgeSpec(
            source=WorkflowStep.REFINEMENT.value,
            selector=transitions.next_node_selector(WorkflowStep.GENERATION.value),
            targets={
                WorkflowStep.GENERATION.value: WorkflowStep.GENERATION.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        ),
        _WorkflowGraphConditionalEdgeSpec(
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

    return assistant_workflow_final_payload_keys()


def registered_workflow_model_payload_specs() -> tuple[_WorkflowModelPayloadSpec, ...]:
    """Return workflow-state payload mappings registered by the runtime."""

    return assistant_workflow_model_payload_specs()


def _linear_workflow_edge_specs_before_review() -> tuple[
    _WorkflowGraphConditionalEdgeSpec, ...
]:
    review_index = ASSISTANT_WORKFLOW_NODE_ORDER.index(WorkflowStep.REVIEW.value)
    linear_nodes = ASSISTANT_WORKFLOW_NODE_ORDER[: review_index + 1]
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
) -> _WorkflowGraphConditionalEdgeSpec:
    if current_node == WorkflowStep.PROMPT_INPUT.value:
        return _WorkflowGraphConditionalEdgeSpec(
            source=current_node,
            selector=transitions.next_node_after_prompt_input,
            targets={
                WorkflowStep.PLANNING.value: WorkflowStep.PLANNING.value,
                WorkflowStep.FINALIZATION.value: WorkflowStep.FINALIZATION.value,
                WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
            },
        )
    return _WorkflowGraphConditionalEdgeSpec(
        source=current_node,
        selector=transitions.next_node_selector(next_node),
        targets={
            next_node: next_node,
            WorkflowStep.FAILURE.value: WorkflowStep.FAILURE.value,
        },
    )
