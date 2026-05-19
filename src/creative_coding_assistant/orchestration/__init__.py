"""Assistant orchestration and explicit routing."""

from __future__ import annotations

from importlib import import_module

_CTX = "creative_coding_assistant.orchestration.context"
_EVENTS = "creative_coding_assistant.orchestration.events"
_GEN = "creative_coding_assistant.orchestration.generation"
_MEM = "creative_coding_assistant.orchestration.memory"
_PROMPT_INPUTS = "creative_coding_assistant.orchestration.prompt_inputs"
_PROMPT_TEMPLATES = "creative_coding_assistant.orchestration.prompt_templates"
_RETRIEVAL = "creative_coding_assistant.orchestration.retrieval"
_ROUTING = "creative_coding_assistant.orchestration.routing"
_SERVICE = "creative_coding_assistant.orchestration.service"
_WORKFLOW = "creative_coding_assistant.orchestration.workflow"
_WORKFLOW_GRAPH = "creative_coding_assistant.orchestration.workflow_graph"
_WORKFLOW_REVIEW = "creative_coding_assistant.orchestration.workflow_review"

_EXPORT_MAP = {
    "ASSISTANT_WORKFLOW_NODE_ORDER": _WORKFLOW_GRAPH,
    "AssistantWorkflowGraphState": _WORKFLOW_GRAPH,
    "AssistantWorkflowRuntime": _WORKFLOW_GRAPH,
    "AssistantWorkflowState": _WORKFLOW,
    "AssistantService": _SERVICE,
    "AssembledContextRequest": _CTX,
    "AssembledContextResponse": _CTX,
    "AssembledContextSummary": _CTX,
    "ChromaMemoryAdapter": _MEM,
    "ContextAssembler": _CTX,
    "ConversationSummaryContext": _MEM,
    "DEFAULT_RECENT_TURN_LIMIT": _MEM,
    "DEFAULT_RETRIEVAL_LIMIT": _RETRIEVAL,
    "DomainSelectionShape": _ROUTING,
    "KnowledgeBaseRetrievalAdapter": _RETRIEVAL,
    "LlmGenerationAdapter": _GEN,
    "MemoryContextRequest": _MEM,
    "MemoryContextResponse": _MEM,
    "MemoryContextSource": _MEM,
    "MemoryGateway": _MEM,
    "OrchestrationContextAssembler": _CTX,
    "ProjectMemoryContext": _MEM,
    "PromptConversationTurnInput": _PROMPT_INPUTS,
    "PromptInputBuilder": _PROMPT_INPUTS,
    "PromptInputRequest": _PROMPT_INPUTS,
    "PromptInputResponse": _PROMPT_INPUTS,
    "PromptKnowledgeChunkInput": _PROMPT_INPUTS,
    "PromptMemoryInput": _PROMPT_INPUTS,
    "PromptProjectMemoryInput": _PROMPT_INPUTS,
    "PromptRetrievalInput": _PROMPT_INPUTS,
    "PromptRenderer": _PROMPT_TEMPLATES,
    "PromptRunningSummaryInput": _PROMPT_INPUTS,
    "PromptUserInput": _PROMPT_INPUTS,
    "ProviderGenerationGateway": _GEN,
    "ProviderGenerationRequest": _GEN,
    "RecentConversationTurn": _MEM,
    "RenderedPromptRequest": _PROMPT_TEMPLATES,
    "RenderedPromptResponse": _PROMPT_TEMPLATES,
    "RenderedPromptRole": _PROMPT_TEMPLATES,
    "RenderedPromptSection": _PROMPT_TEMPLATES,
    "RenderedPromptSectionName": _PROMPT_TEMPLATES,
    "RetrievalContextFilter": _RETRIEVAL,
    "RetrievalContextRequest": _RETRIEVAL,
    "RetrievalContextResponse": _RETRIEVAL,
    "RetrievalContextSource": _RETRIEVAL,
    "RetrievalGateway": _RETRIEVAL,
    "RetrievedKnowledgeChunk": _RETRIEVAL,
    "RouteCapability": _ROUTING,
    "RouteDecision": _ROUTING,
    "RouteName": _ROUTING,
    "StreamEventBuilder": _EVENTS,
    "StructuredPromptInputBuilder": _PROMPT_INPUTS,
    "WORKFLOW_STEP_ORDER": _WORKFLOW,
    "MAX_WORKFLOW_REFINEMENT_COUNT": _WORKFLOW_REVIEW,
    "WorkflowEventMetadata": _WORKFLOW,
    "WorkflowReviewOutcome": _WORKFLOW_REVIEW,
    "WorkflowReviewResult": _WORKFLOW_REVIEW,
    "WorkflowStatus": _WORKFLOW,
    "WorkflowStep": _WORKFLOW,
    "JinjaPromptRenderer": _PROMPT_TEMPLATES,
    "begin_assistant_workflow": _WORKFLOW,
    "build_assistant_workflow_graph": _WORKFLOW_GRAPH,
    "build_initial_workflow_graph_state": _WORKFLOW_GRAPH,
    "build_assembled_context_request": _CTX,
    "build_memory_context_request": _MEM,
    "build_prompt_input_request": _PROMPT_INPUTS,
    "build_provider_generation_request": _GEN,
    "build_rendered_prompt_request": _PROMPT_TEMPLATES,
    "build_retrieval_context_request": _RETRIEVAL,
    "complete_workflow_step": _WORKFLOW,
    "fail_workflow": _WORKFLOW,
    "finish_workflow": _WORKFLOW,
    "next_workflow_step": _WORKFLOW,
    "restart_workflow_step": _WORKFLOW,
    "review_assistant_answer": _WORKFLOW_REVIEW,
    "route_request": _ROUTING,
    "skip_workflow_step": _WORKFLOW,
    "stream_assistant_workflow_events": _WORKFLOW_GRAPH,
    "start_workflow_step": _WORKFLOW,
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
