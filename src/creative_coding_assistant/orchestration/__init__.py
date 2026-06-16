"""Assistant orchestration and explicit routing."""

from __future__ import annotations

from importlib import import_module

_CTX = "creative_coding_assistant.orchestration.context"
_CREATIVE_TRANSLATION = (
    "creative_coding_assistant.orchestration.creative_translation"
)
_AUDIO_REACTIVE = "creative_coding_assistant.orchestration.audio_reactive"
_CREATIVE_QUALITY = "creative_coding_assistant.orchestration.creative_quality"
_SACRED_CONSISTENCY = (
    "creative_coding_assistant.orchestration.sacred_consistency"
)
_SACRED_GEOMETRY = "creative_coding_assistant.orchestration.sacred_geometry"
_SHADER_PRESETS = "creative_coding_assistant.orchestration.shader_presets"
_VISUAL_STYLES = "creative_coding_assistant.orchestration.visual_styles"
_ARTIFACTS = "creative_coding_assistant.orchestration.artifacts"
_ARTIFACT_CRITIQUE = "creative_coding_assistant.orchestration.artifact_critique"
_EVENTS = "creative_coding_assistant.orchestration.events"
_GEN = "creative_coding_assistant.orchestration.generation"
_MEM = "creative_coding_assistant.orchestration.memory"
_PROMPT_INPUTS = "creative_coding_assistant.orchestration.prompt_inputs"
_PROMPT_TEMPLATES = "creative_coding_assistant.orchestration.prompt_templates"
_QUALITY_CALIBRATION = (
    "creative_coding_assistant.orchestration.quality_calibration"
)
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
    "ArtifactCritiqueDimension": _ARTIFACTS,
    "ArtifactCritiqueSummary": _ARTIFACT_CRITIQUE,
    "AudioReactiveGuidance": _AUDIO_REACTIVE,
    "AudioReactiveIntensity": _AUDIO_REACTIVE,
    "AudioReactiveMapping": _AUDIO_REACTIVE,
    "AudioReactiveSource": _AUDIO_REACTIVE,
    "AudioReactiveVisualTarget": _AUDIO_REACTIVE,
    "AssistantService": _SERVICE,
    "AssembledContextRequest": _CTX,
    "AssembledContextResponse": _CTX,
    "AssembledContextSummary": _CTX,
    "ChromaMemoryAdapter": _MEM,
    "ContextAssembler": _CTX,
    "ConversationSummaryContext": _MEM,
    "CalibratedQualityEvaluation": _ARTIFACTS,
    "CalibratedQualitySignal": _ARTIFACTS,
    "CreativeOutputModality": _CREATIVE_TRANSLATION,
    "CreativeQualityEvaluation": _ARTIFACTS,
    "CreativeQualityObservation": _ARTIFACTS,
    "CreativeTranslation": _CREATIVE_TRANSLATION,
    "SacredConsistencyEvaluation": _ARTIFACTS,
    "SacredConsistencyObservation": _ARTIFACTS,
    "SacredGeometryGuidance": _SACRED_GEOMETRY,
    "ShaderPresetGuidance": _SHADER_PRESETS,
    "ShaderPresetId": _SHADER_PRESETS,
    "VisualStyleGuidance": _VISUAL_STYLES,
    "VisualStyleId": _VISUAL_STYLES,
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
    "PromptArtifactRefinementInput": _PROMPT_INPUTS,
    "PromptImageReferenceInput": _PROMPT_INPUTS,
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
    "WorkflowFailureInfo": _WORKFLOW,
    "WorkflowReviewOutcome": _WORKFLOW_REVIEW,
    "WorkflowReviewResult": _WORKFLOW_REVIEW,
    "WorkflowStatus": _WORKFLOW,
    "WorkflowStep": _WORKFLOW,
    "WorkflowArtifact": _ARTIFACTS,
    "WorkflowArtifactCritique": _ARTIFACTS,
    "JinjaPromptRenderer": _PROMPT_TEMPLATES,
    "begin_assistant_workflow": _WORKFLOW,
    "build_assistant_workflow_graph": _WORKFLOW_GRAPH,
    "calibrate_artifact_quality": _QUALITY_CALIBRATION,
    "build_initial_workflow_graph_state": _WORKFLOW_GRAPH,
    "build_assembled_context_request": _CTX,
    "build_memory_context_request": _MEM,
    "build_prompt_input_request": _PROMPT_INPUTS,
    "build_provider_generation_request": _GEN,
    "build_rendered_prompt_request": _PROMPT_TEMPLATES,
    "build_retrieval_context_request": _RETRIEVAL,
    "complete_workflow_step": _WORKFLOW,
    "critique_workflow_artifacts": _ARTIFACT_CRITIQUE,
    "creative_translation_prompt_lines": _CREATIVE_TRANSLATION,
    "audio_reactive_prompt_lines": _AUDIO_REACTIVE,
    "derive_audio_reactive_guidance": _AUDIO_REACTIVE,
    "derive_creative_translation": _CREATIVE_TRANSLATION,
    "evaluate_artifact_sacred_consistency": _SACRED_CONSISTENCY,
    "evaluate_artifact_creative_quality": _CREATIVE_QUALITY,
    "derive_sacred_geometry_guidance": _SACRED_GEOMETRY,
    "detect_sacred_geometry_concepts": _SACRED_GEOMETRY,
    "detect_shader_presets": _SHADER_PRESETS,
    "derive_shader_preset_guidance": _SHADER_PRESETS,
    "derive_visual_style_guidance": _VISUAL_STYLES,
    "detect_visual_styles": _VISUAL_STYLES,
    "extract_workflow_artifacts": _ARTIFACTS,
    "fail_workflow": _WORKFLOW,
    "finish_workflow": _WORKFLOW,
    "next_workflow_step": _WORKFLOW,
    "prepare_workflow_preview_results": _ARTIFACTS,
    "restart_workflow_step": _WORKFLOW,
    "review_assistant_answer": _WORKFLOW_REVIEW,
    "route_request": _ROUTING,
    "sacred_geometry_prompt_lines": _SACRED_GEOMETRY,
    "shader_preset_prompt_lines": _SHADER_PRESETS,
    "visual_style_prompt_lines": _VISUAL_STYLES,
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
