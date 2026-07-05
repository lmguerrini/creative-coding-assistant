"""State transition selectors for the assistant workflow runtime graph."""

from __future__ import annotations

from creative_coding_assistant.orchestration.runtime.nodes import handlers


def next_node_after_review(state: handlers.AssistantWorkflowGraphState) -> str:
    return handlers._next_node_after_review(state)


def next_node_after_finalization(state: handlers.AssistantWorkflowGraphState) -> str:
    return handlers._next_node_after_finalization(state)


def next_node_after_prompt_input(state: handlers.AssistantWorkflowGraphState) -> str:
    return handlers._next_node_after_prompt_input(state)


def next_node_or_failure(
    state: handlers.AssistantWorkflowGraphState,
    next_node: str,
) -> str:
    return handlers._next_node_or_failure(state, next_node)


def next_node_selector(next_node: str) -> handlers._GraphTransitionSelector:
    return lambda state: next_node_or_failure(state, next_node)
