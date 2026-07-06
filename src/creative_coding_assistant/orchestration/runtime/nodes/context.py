"""Context assembly and prompt-input node handlers."""

from __future__ import annotations

from langgraph.runtime import Runtime

from creative_coding_assistant.orchestration.runtime.nodes.contracts import (
    AssistantWorkflowGraphContext,
    AssistantWorkflowGraphState,
)
from creative_coding_assistant.orchestration.runtime.nodes.emissions import _emit, _emit_streaming_step
from creative_coding_assistant.orchestration.runtime.nodes.state import (
    _complete_node,
    _handle_workflow_exception,
    _route_decision,
    _runtime,
    _skip_node,
    _start_node,
    _workflow_state,
)
from creative_coding_assistant.orchestration.workflow import WorkflowStep


def _context_assembly_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.CONTEXT_ASSEMBLY,
    )
    try:
        assembled_context = _emit_streaming_step(
            runtime_context.stream_assembled_context(
                builder=runtime_context.event_builder,
                decision=_route_decision(workflow_state),
                memory_context=workflow_state.memory_context,
                retrieval_context=workflow_state.retrieval_context,
            ),
            workflow_state=workflow_state,
        )
        if assembled_context is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.CONTEXT_ASSEMBLY,
                    decision_reason="context_assembly_unavailable",
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.CONTEXT_ASSEMBLY,
                decision_reason="context_assembled",
                assembled_context=assembled_context,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.CONTEXT_ASSEMBLY,
            exc=exc,
        )

def _prompt_input_node(
    state: AssistantWorkflowGraphState,
    runtime: Runtime[AssistantWorkflowGraphContext],
) -> AssistantWorkflowGraphState:
    runtime_context = _runtime(runtime)
    workflow_state = _start_node(
        _workflow_state(state),
        runtime_context,
        WorkflowStep.PROMPT_INPUT,
    )
    try:
        prompt_input = _emit_streaming_step(
            runtime_context.stream_prompt_inputs(
                builder=runtime_context.event_builder,
                request=workflow_state.request,
                decision=_route_decision(workflow_state),
                assembled_context=workflow_state.assembled_context,
            ),
            workflow_state=workflow_state,
        )
        if prompt_input is None:
            return {
                "workflow_state": _skip_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_INPUT,
                    decision_reason="prompt_input_unavailable",
                )
            }
        if prompt_input.clarification is not None:
            clarification_state = workflow_state.model_copy(
                update={
                    "prompt_input": prompt_input,
                    "clarification": prompt_input.clarification,
                }
            )
            _emit(
                runtime_context.event_builder.prompt_input(
                    code="clarification_required",
                    message="Clarification required before generation.",
                    clarification=prompt_input.clarification.model_dump(mode="json"),
                ),
                workflow_state=clarification_state,
                step=WorkflowStep.PROMPT_INPUT,
            )
            return {
                "workflow_state": _complete_node(
                    workflow_state,
                    runtime_context,
                    WorkflowStep.PROMPT_INPUT,
                    transition_target=WorkflowStep.FINALIZATION.value,
                    decision_reason="clarification_required",
                    prompt_input=prompt_input,
                    clarification=prompt_input.clarification,
                )
            }
        return {
            "workflow_state": _complete_node(
                workflow_state,
                runtime_context,
                WorkflowStep.PROMPT_INPUT,
                decision_reason="prompt_input_prepared",
                prompt_input=prompt_input,
            )
        }
    except Exception as exc:
        return _handle_workflow_exception(
            workflow_state=workflow_state,
            runtime=runtime_context,
            step=WorkflowStep.PROMPT_INPUT,
            exc=exc,
        )
