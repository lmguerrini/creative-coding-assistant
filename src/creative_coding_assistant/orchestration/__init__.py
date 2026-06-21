"""Assistant orchestration and explicit routing."""

from __future__ import annotations

from importlib import import_module

_CTX = "creative_coding_assistant.orchestration.context"
_CREATIVE_TRANSLATION = (
    "creative_coding_assistant.orchestration.creative_translation"
)
_CREATIVE_PLANNING = "creative_coding_assistant.orchestration.creative_planning"
_CREATIVE_CONSTRAINTS = (
    "creative_coding_assistant.orchestration.creative_constraints"
)
_CREATIVE_HIERARCHY = "creative_coding_assistant.orchestration.creative_hierarchy"
_CREATIVE_INTENT = "creative_coding_assistant.orchestration.creative_intent"
_CREATIVE_STRATEGY = "creative_coding_assistant.orchestration.creative_strategy"
_CREATIVE_TECHNIQUE = "creative_coding_assistant.orchestration.creative_technique"
_CREATIVE_TRADEOFFS = "creative_coding_assistant.orchestration.creative_tradeoffs"
_CREATIVE_REASONING = "creative_coding_assistant.orchestration.creative_reasoning"
_RUNTIME_CAPABILITIES = (
    "creative_coding_assistant.orchestration.runtime_capabilities"
)
_CREATIVE_DIRECTOR = "creative_coding_assistant.orchestration.creative_director"
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
_CLARIFICATION = "creative_coding_assistant.orchestration.clarification"
_EVENTS = "creative_coding_assistant.orchestration.events"
_GEN = "creative_coding_assistant.orchestration.generation"
_MEM = "creative_coding_assistant.orchestration.memory"
_PROMPT_INPUTS = "creative_coding_assistant.orchestration.prompt_inputs"
_PROMPT_TEMPLATES = "creative_coding_assistant.orchestration.prompt_templates"
_QUALITY_CALIBRATION = (
    "creative_coding_assistant.orchestration.quality_calibration"
)
_REFINEMENT_PASSES = "creative_coding_assistant.orchestration.refinement_passes"
_REFERENCE_FUSION = "creative_coding_assistant.orchestration.reference_fusion"
_RETRIEVAL = "creative_coding_assistant.orchestration.retrieval"
_ROUTING = "creative_coding_assistant.orchestration.routing"
_SERVICE = "creative_coding_assistant.orchestration.service"
_WORKFLOW = "creative_coding_assistant.orchestration.workflow"
_WORKFLOW_GRAPH = "creative_coding_assistant.orchestration.workflow_graph"
_WORKFLOW_REVIEW = "creative_coding_assistant.orchestration.workflow_review"

_EXPORT_MAP = {
    "ASSISTANT_WORKFLOW_NODE_ORDER": _WORKFLOW_GRAPH,
    "ASSISTANT_WORKFLOW_RECURSION_LIMIT": _WORKFLOW_GRAPH,
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
    "ClarificationQuestion": _CLARIFICATION,
    "ClarificationReason": _CLARIFICATION,
    "ClarificationRequest": _CLARIFICATION,
    "CreativeOutputModality": _CREATIVE_TRANSLATION,
    "CreativeExecutionPlan": _CREATIVE_PLANNING,
    "CreativeHierarchyPlan": _CREATIVE_HIERARCHY,
    "CreativeHierarchyPriority": _CREATIVE_HIERARCHY,
    "CreativeIntentDecomposition": _CREATIVE_INTENT,
    "CreativeIntentDimension": _CREATIVE_INTENT,
    "CreativeConstraint": _CREATIVE_CONSTRAINTS,
    "CreativeConstraintSolution": _CREATIVE_CONSTRAINTS,
    "CreativeConstraintTradeoff": _CREATIVE_CONSTRAINTS,
    "CreativeStrategyAlternative": _CREATIVE_STRATEGY,
    "CreativeStrategyProfile": _CREATIVE_STRATEGY,
    "CreativeTechniqueAlternative": _CREATIVE_TECHNIQUE,
    "CreativeTechniqueProfile": _CREATIVE_TECHNIQUE,
    "CreativeTradeoff": _CREATIVE_TRADEOFFS,
    "CreativeTradeoffProfile": _CREATIVE_TRADEOFFS,
    "CreativeReasoningEvidence": _CREATIVE_REASONING,
    "CreativeReasoningResult": _CREATIVE_REASONING,
    "CreativeReasoningStep": _CREATIVE_REASONING,
    "CreativeRejectedAlternative": _CREATIVE_REASONING,
    "RuntimeCapabilityCandidate": _RUNTIME_CAPABILITIES,
    "RuntimeCapabilityProfile": _RUNTIME_CAPABILITIES,
    "CreativeAssistantDirectorBrief": _CREATIVE_DIRECTOR,
    "CreativeQualityEvaluation": _ARTIFACTS,
    "CreativeQualityObservation": _ARTIFACTS,
    "CreativeTranslation": _CREATIVE_TRANSLATION,
    "DEFAULT_REFINEMENT_PASS_LIMIT": _REFINEMENT_PASSES,
    "MAX_REFINEMENT_PASS_LIMIT": _REFINEMENT_PASSES,
    "QUALITY_IMPROVEMENT_THRESHOLD": _REFINEMENT_PASSES,
    "RefinementPassDecision": _REFINEMENT_PASSES,
    "RefinementPassRecord": _ARTIFACTS,
    "RefinementPassStopReason": _ARTIFACTS,
    "ReferenceFusionGuidance": _REFERENCE_FUSION,
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
    "artifact_quality_score": _REFINEMENT_PASSES,
    "attach_refinement_history": _REFINEMENT_PASSES,
    "build_initial_workflow_graph_state": _WORKFLOW_GRAPH,
    "build_refinement_objective": _REFINEMENT_PASSES,
    "build_assembled_context_request": _CTX,
    "build_memory_context_request": _MEM,
    "build_prompt_input_request": _PROMPT_INPUTS,
    "build_provider_generation_request": _GEN,
    "build_rendered_prompt_request": _PROMPT_TEMPLATES,
    "build_retrieval_context_request": _RETRIEVAL,
    "complete_workflow_step": _WORKFLOW,
    "complete_latest_refinement_pass": _REFINEMENT_PASSES,
    "critique_workflow_artifacts": _ARTIFACT_CRITIQUE,
    "creative_translation_prompt_lines": _CREATIVE_TRANSLATION,
    "audio_reactive_prompt_lines": _AUDIO_REACTIVE,
    "derive_audio_reactive_guidance": _AUDIO_REACTIVE,
    "derive_creative_translation": _CREATIVE_TRANSLATION,
    "derive_creative_hierarchy_plan": _CREATIVE_HIERARCHY,
    "creative_hierarchy_plan_prompt_lines": _CREATIVE_HIERARCHY,
    "derive_creative_intent_decomposition": _CREATIVE_INTENT,
    "creative_intent_decomposition_prompt_lines": _CREATIVE_INTENT,
    "derive_creative_execution_plan": _CREATIVE_PLANNING,
    "creative_execution_plan_prompt_lines": _CREATIVE_PLANNING,
    "derive_creative_constraint_solution": _CREATIVE_CONSTRAINTS,
    "creative_constraint_solution_prompt_lines": _CREATIVE_CONSTRAINTS,
    "derive_creative_strategy_profile": _CREATIVE_STRATEGY,
    "creative_strategy_prompt_lines": _CREATIVE_STRATEGY,
    "derive_creative_technique_profile": _CREATIVE_TECHNIQUE,
    "creative_technique_prompt_lines": _CREATIVE_TECHNIQUE,
    "derive_creative_tradeoff_profile": _CREATIVE_TRADEOFFS,
    "creative_tradeoff_prompt_lines": _CREATIVE_TRADEOFFS,
    "derive_creative_reasoning_result": _CREATIVE_REASONING,
    "creative_reasoning_prompt_lines": _CREATIVE_REASONING,
    "derive_runtime_capability_profile": _RUNTIME_CAPABILITIES,
    "runtime_capability_prompt_lines": _RUNTIME_CAPABILITIES,
    "derive_creative_assistant_director_brief": _CREATIVE_DIRECTOR,
    "creative_assistant_director_prompt_lines": _CREATIVE_DIRECTOR,
    "derive_hitl_clarification": _CLARIFICATION,
    "evaluate_artifact_sacred_consistency": _SACRED_CONSISTENCY,
    "evaluate_artifact_creative_quality": _CREATIVE_QUALITY,
    "derive_sacred_geometry_guidance": _SACRED_GEOMETRY,
    "detect_sacred_geometry_concepts": _SACRED_GEOMETRY,
    "detect_shader_presets": _SHADER_PRESETS,
    "derive_shader_preset_guidance": _SHADER_PRESETS,
    "derive_visual_style_guidance": _VISUAL_STYLES,
    "derive_reference_fusion_guidance": _REFERENCE_FUSION,
    "detect_visual_styles": _VISUAL_STYLES,
    "extract_workflow_artifacts": _ARTIFACTS,
    "fail_workflow": _WORKFLOW,
    "finish_workflow": _WORKFLOW,
    "next_workflow_step": _WORKFLOW,
    "prepare_workflow_preview_results": _ARTIFACTS,
    "plan_next_refinement_pass": _REFINEMENT_PASSES,
    "refinement_opportunities": _REFINEMENT_PASSES,
    "reference_fusion_prompt_lines": _REFERENCE_FUSION,
    "restart_workflow_step": _WORKFLOW,
    "review_assistant_answer": _WORKFLOW_REVIEW,
    "route_request": _ROUTING,
    "sacred_geometry_prompt_lines": _SACRED_GEOMETRY,
    "shader_preset_prompt_lines": _SHADER_PRESETS,
    "visual_style_prompt_lines": _VISUAL_STYLES,
    "skip_workflow_step": _WORKFLOW,
    "start_refinement_pass_record": _REFINEMENT_PASSES,
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
