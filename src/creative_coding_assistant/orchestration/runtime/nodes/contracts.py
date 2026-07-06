"""Runtime node graph contracts."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.runtime import Runtime

from creative_coding_assistant.analytics import (
    LangSmithObservability,
    LangSmithRunMetadata,
)
from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
)


class GenerationResultLike(Protocol):
    answer: str


class AssistantWorkflowGraphState(TypedDict, total=False):
    workflow_state: AssistantWorkflowState
    route_payload: dict[str, object]
    generation_result: GenerationResultLike | None
    pending_failure: WorkflowFailureInfo | None
    failure_event_emitted: bool


class AssistantWorkflowGraphContext(TypedDict):
    runtime: AssistantWorkflowRuntime


_GraphNodeHandler = Callable[
    [AssistantWorkflowGraphState, Runtime[AssistantWorkflowGraphContext]],
    AssistantWorkflowGraphState,
]
_GraphTransitionSelector = Callable[[AssistantWorkflowGraphState], str]


@dataclass(frozen=True)
class _WorkflowGraphNodeSpec:
    name: str
    handler: _GraphNodeHandler


@dataclass(frozen=True)
class _WorkflowGraphConditionalEdgeSpec:
    source: str
    selector: _GraphTransitionSelector
    targets: dict[str, Any]


@dataclass(frozen=True)
class _WorkflowModelPayloadSpec:
    payload_key: str
    state_attribute: str
    availability_key: str | None = None


@dataclass(frozen=True)
class AssistantWorkflowRuntime:
    """Runtime services needed by graph nodes for one assistant turn."""

    event_builder: StreamEventBuilder
    observability: LangSmithObservability
    observability_run: LangSmithRunMetadata
    route_fn: Callable[[AssistantRequest], RouteDecision]
    stream_request_received: Callable[..., Iterator[object]]
    stream_route_selected: Callable[..., Iterator[object]]
    stream_memory_context: Callable[..., Iterator[object]]
    stream_retrieval_context: Callable[..., Iterator[object]]
    stream_assembled_context: Callable[..., Iterator[object]]
    stream_prompt_inputs: Callable[..., Iterator[object]]
    stream_rendered_prompt: Callable[..., Iterator[object]]
    stream_generation: Callable[..., Iterator[object]]
    build_shell_answer: Callable[[RouteDecision], str]
