"""Lightweight workflow state foundation for assistant orchestration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.context import AssembledContextResponse
from creative_coding_assistant.orchestration.memory import MemoryContextResponse
from creative_coding_assistant.orchestration.prompt_inputs import PromptInputResponse
from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
)
from creative_coding_assistant.orchestration.retrieval import RetrievalContextResponse
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.workflow_review import (
    WorkflowReviewResult,
)


class WorkflowStep(StrEnum):
    INTAKE = "intake"
    ROUTING = "routing"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    CONTEXT_ASSEMBLY = "context_assembly"
    PROMPT_INPUT = "prompt_input"
    PROMPT_RENDERING = "prompt_rendering"
    GENERATION = "generation"
    REVIEW = "review"
    REFINEMENT = "refinement"
    FINALIZATION = "finalization"
    FAILURE = "failure"


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


WORKFLOW_STEP_ORDER: tuple[WorkflowStep, ...] = (
    WorkflowStep.INTAKE,
    WorkflowStep.ROUTING,
    WorkflowStep.MEMORY,
    WorkflowStep.RETRIEVAL,
    WorkflowStep.CONTEXT_ASSEMBLY,
    WorkflowStep.PROMPT_INPUT,
    WorkflowStep.PROMPT_RENDERING,
    WorkflowStep.GENERATION,
    WorkflowStep.REVIEW,
    WorkflowStep.REFINEMENT,
    WorkflowStep.FINALIZATION,
)


class WorkflowEventMetadata(BaseModel):
    """Small workflow snapshot that can be attached to future stream events."""

    model_config = ConfigDict(frozen=True)

    current_step: WorkflowStep | None = None
    status: WorkflowStatus
    completed_steps: tuple[WorkflowStep, ...] = ()
    skipped_steps: tuple[WorkflowStep, ...] = ()


class WorkflowFailureInfo(BaseModel):
    """Typed metadata for terminal workflow failures."""

    model_config = ConfigDict(frozen=True)

    step: WorkflowStep
    code: str
    message: str


class AssistantWorkflowState(BaseModel):
    """Explicit state for one assistant workflow run.

    The state intentionally mirrors the existing deterministic pipeline while
    remaining small enough to move through graph runtime nodes.
    """

    model_config = ConfigDict(frozen=True)

    request: AssistantRequest
    status: WorkflowStatus = WorkflowStatus.RUNNING
    current_step: WorkflowStep | None = None
    completed_steps: tuple[WorkflowStep, ...] = ()
    skipped_steps: tuple[WorkflowStep, ...] = ()
    route_decision: RouteDecision | None = None
    memory_context: MemoryContextResponse | None = None
    retrieval_context: RetrievalContextResponse | None = None
    assembled_context: AssembledContextResponse | None = None
    prompt_input: PromptInputResponse | None = None
    rendered_prompt: RenderedPromptResponse | None = None
    review_result: WorkflowReviewResult | None = None
    refinement_count: int = 0
    failure_info: WorkflowFailureInfo | None = None
    final_answer: str | None = None
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
        }

    def event_metadata(self) -> WorkflowEventMetadata:
        return WorkflowEventMetadata(
            current_step=self.current_step,
            status=self.status,
            completed_steps=self.completed_steps,
            skipped_steps=self.skipped_steps,
        )


def begin_assistant_workflow(request: AssistantRequest) -> AssistantWorkflowState:
    return AssistantWorkflowState(request=request)


def start_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    if state.is_terminal:
        raise ValueError("Cannot start a workflow step after terminal state.")
    if state.current_step is not None:
        raise ValueError("Cannot start a workflow step while another step is active.")
    if step in state.completed_steps or step in state.skipped_steps:
        raise ValueError(f"Workflow step already resolved: {step.value}")
    return state.model_copy(update={"current_step": step})


def restart_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    if state.is_terminal:
        raise ValueError("Cannot restart a workflow step after terminal state.")
    if state.current_step is not None:
        raise ValueError("Cannot restart a workflow step while another step is active.")
    return state.model_copy(update={"current_step": step})


def complete_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
    **updates: object,
) -> AssistantWorkflowState:
    _validate_active_step(state, step)
    return state.model_copy(
        update={
            "current_step": None,
            "completed_steps": _append_unique(state.completed_steps, step),
            **updates,
        }
    )


def skip_workflow_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> AssistantWorkflowState:
    _validate_active_step(state, step)
    return state.model_copy(
        update={
            "current_step": None,
            "skipped_steps": _append_unique(state.skipped_steps, step),
        }
    )


def finish_workflow(
    state: AssistantWorkflowState,
    *,
    final_answer: str,
) -> AssistantWorkflowState:
    if state.current_step is not WorkflowStep.FINALIZATION:
        raise ValueError("Workflow must be in finalization before completion.")
    return complete_workflow_step(
        state,
        WorkflowStep.FINALIZATION,
        final_answer=final_answer,
        status=WorkflowStatus.COMPLETED,
    )


def fail_workflow(
    state: AssistantWorkflowState,
    *,
    error_message: str,
    failure_info: WorkflowFailureInfo | None = None,
    final_answer: str | None = None,
) -> AssistantWorkflowState:
    return state.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "current_step": None,
            "error_message": error_message,
            "failure_info": failure_info,
            "final_answer": final_answer,
        }
    )


def next_workflow_step(step: WorkflowStep) -> WorkflowStep | None:
    index = WORKFLOW_STEP_ORDER.index(step)
    next_index = index + 1
    if next_index >= len(WORKFLOW_STEP_ORDER):
        return None
    return WORKFLOW_STEP_ORDER[next_index]


def _validate_active_step(
    state: AssistantWorkflowState,
    step: WorkflowStep,
) -> None:
    if state.current_step is not step:
        current = state.current_step.value if state.current_step is not None else None
        raise ValueError(
            f"Workflow step mismatch: expected {step.value}, current {current}."
        )


def _append_unique(
    steps: tuple[WorkflowStep, ...],
    step: WorkflowStep,
) -> tuple[WorkflowStep, ...]:
    if step in steps:
        return steps
    return (*steps, step)
