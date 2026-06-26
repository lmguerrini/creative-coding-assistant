import {
  workflowNodeOrder,
  type ArtifactCritique,
  type ArtifactCapabilityConfidenceSummary,
  type ArtifactCapabilityFit,
  type ArtifactCapabilityMatrixSummary,
  type ArtifactCapabilityProfileSummary,
  type ArtifactCriticRiskAssessment,
  type ArtifactCriticSummary,
  type ArtifactImplementationComplexity,
  type ArtifactImplementationPriority,
  type ArtifactImplementationReadiness,
  type ArtifactImplementationRisk,
  type ArtifactIntelligenceSynthesisSummary,
  type ArtifactIntelligenceEngineContractRegistrySummary,
  type ArtifactIntelligenceEngineContractSummary,
  type ArtifactEngineCacheability,
  type ArtifactEngineCategory,
  type ArtifactEngineParallelizationSupport,
  type ArtifactExportIntelligenceSummary,
  type ArtifactExportReadiness,
  type ArtifactMergePlannerSummary,
  type ArtifactMergeStrategy,
  type ArtifactRefinerSummary,
  type ArtifactDependencyEdgeSummary,
  type ArtifactDependencyGraphSummary,
  type ArtifactDependencyNodeSummary,
  type ArtifactFamily,
  type ArtifactPlanSummary,
  type ArtifactType,
  type MultiArtifactStrategyAction,
  type MultiArtifactStrategyArtifactSummary,
  type MultiArtifactStrategyCombinationMode,
  type MultiArtifactStrategyGroupSummary,
  type MultiArtifactStrategyPriority,
  type MultiArtifactStrategyPriorityEntrySummary,
  type MultiArtifactStrategyRole,
  type MultiArtifactStrategySequenceStepSummary,
  type MultiArtifactStrategySummary,
  type AudioVisualCueType,
  type AudioVisualFallbackSceneStrategySummary,
  type AudioVisualSceneCueSummary,
  type AudioVisualScenePattern,
  type AudioVisualScenePhaseSummary,
  type AudioVisualSceneProfileSummary,
  type AudioVisualSceneTransitionSummary,
  type CreativeConstraintAxis,
  type CreativeConstraintPrioritizationSummary,
  type CreativeConstraintPriorityCategory,
  type CreativeConstraintPriorityConflictSummary,
  type CreativeConstraintPriorityLevel,
  type CreativeConstraintPrioritySource,
  type CreativeConstraintPrioritySummary,
  type CreativeConstraintSeverity,
  type CreativeConstraintSolverSummary,
  type CreativeConstraintSummary,
  type CreativeConstraintTradeoffSummary,
  type CreativeCompositionPattern,
  type CreativeCompositionPlanSummary,
  type CreativeCriticRiskAssessment,
  type CreativeCriticSummary,
  type ClarificationSummary,
  type CrossModalityChannel,
  type CrossModalityCompositionProfileSummary,
  type CrossModalityFallbackStrategySummary,
  type CrossModalityMappingSummary,
  type CrossModalityPattern,
  type CrossModalityRoleSummary,
  type CrossModalityTemporalCueSummary,
  type DependencyNodeStatus,
  type DependencyNodeType,
  type DependencyRelationship,
  type DependencyStrength,
  type ProceduralComplexityLevel,
  type ProceduralFamily,
  type ProceduralStructureChoiceSummary,
  type ProceduralStructurePlanSummary,
  type CreativeAssistantDirectorSummary,
  type CreativeConfidenceComponentSource,
  type CreativeConfidenceComponentSummary,
  type CreativeConfidenceLevel,
  type CreativeConfidenceSummary,
  type CreativeExecutionPlanSummary,
  type CreativeAbstractionLevel,
  type CreativeIntentDecompositionSummary,
  type CreativeIntentDimensionName,
  type CreativeIntentDimensionSummary,
  type CreativeIntentExplicitness,
  type CreativeHierarchyDimension,
  type CreativeHierarchyPlanSummary,
  type CreativeHierarchyPrioritySummary,
  type CreativeHierarchySource,
  type CreativeHierarchyTier,
  type CreativeImprovementPlannerSummary,
  type CreativeImprovementPrioritySummary,
  type CreativeQualityDimension,
  type CreativeQualityPredictionSummary,
  type CreativeQualityPredictionLevel,
  type CreativeQualitySignalSummary,
  type CreativeReasoningEvidenceSource,
  type CreativeReasoningEvidenceSummary,
  type CreativeReasoningStage,
  type CreativeReasoningStepSummary,
  type CreativeReasoningSummary,
  type CreativeRejectedAlternativeSummary,
  type CreativeScoreBand,
  type CreativeScoreBreakdownSummary,
  type CreativeScoreComponentSummary,
  type CreativeScoreDimension,
  type CreativeScoreSignalSource,
  type CreativeScoreSummary,
  type ConsistencyValidationAmbiguityLevel,
  type ConsistencyValidationCheckStatus,
  type ConsistencyValidationCheckSummary,
  type ConsistencyValidationContradictionLevel,
  type ConsistencyValidationIntegrity,
  type ConsistencyValidationStatus,
  type ConsistencyValidationSummary,
  type EvaluationDependencySummary,
  type EvaluationEvidenceLinkSummary,
  type EvaluationProvenanceEntrySummary,
  type EvaluationReportDependencyStatus,
  type EvaluationReportSource,
  type EvaluationReportSummary,
  type EvaluationTraceEntrySummary,
  type SelfEvaluationAmbiguity,
  type SelfEvaluationCompleteness,
  type SelfEvaluationRisk,
  type SelfEvaluationSummary,
  type ImprovementImpact,
  type ImprovementPriorityLevel,
  type ImprovementRisk,
  type ImprovementSource,
  type CreativeStrategyAlternativeSummary,
  type CreativeStrategyId,
  type CreativeStrategySummary,
  type CreativeTechniqueAlternativeSummary,
  type CreativeTechniqueCompatibility,
  type CreativeTechniqueId,
  type CreativeTechniquePressure,
  type CreativeTechniqueSummary,
  type CreativeTradeoffAxis,
  type CreativeTradeoffExplorerSummary,
  type CreativeTradeoffPressure,
  type CreativeTradeoffSeverity,
  type CreativeTradeoffSummary,
  type CreativeTranslationSummary,
  type EmotionalCompositionMappingSummary,
  type EmotionalConsistencyProfileSummary,
  type EmotionalFallbackStrategySummary,
  type EmotionalIntensity,
  type EmotionalMotifMappingSummary,
  type EmotionalNarrativeMappingSummary,
  type EmotionalParameterMappingSummary,
  type EmotionalPhaseMappingSummary,
  type EmotionalStructureMappingSummary,
  type EmotionalTone,
  type EscalationRecommendation,
  type ExpectedExecutionReadiness,
  type ExpectedHumanReviewNeed,
  type ExpectedOutputReliability,
  type GenerativeArchitecture,
  type GenerativeEvolutionPhase,
  type GenerativeEvolutionRuleSummary,
  type GenerativeEvolutionTrigger,
  type GenerativeFallbackBlueprintSummary,
  type GenerativeHookType,
  type GenerativeModuleKind,
  type GenerativeModuleRelationshipSummary,
  type GenerativeModuleSummary,
  type GenerativeParameterRole,
  type GenerativeParameterSummary,
  type GenerativeParameterValueType,
  type GenerativeRelationshipType,
  type GenerativeStructureBlueprintSummary,
  type GenerativeStructureHookSummary,
  type RefinementPassRecord,
  type RuntimeCapabilityCandidateSummary,
  type RuntimeCapabilityComplexity,
  type RuntimeCapabilityFit,
  type RuntimeCapabilityId,
  type RuntimeCapabilityReasonerSummary,
  type RuntimePreviewSupport,
  type ReflectionLoopDepth,
  type ReflectionLoopEstimate,
  type ReflectionLoopHitlRecommendation,
  type ReflectionLoopPriority,
  type ReflectionLoopSummary,
  type ConfidenceTrend,
  type RuntimeCompatibilityAssessmentSummary,
  type RuntimeCompatibilityConfidenceSummary,
  type RuntimeCompatibilityLevel,
  type RuntimeCompatibilityProfileSummary,
  type RuntimeInteroperability,
  type RuntimePortability,
  type SemanticMotifCompositionMappingSummary,
  type SemanticMotifFallbackPlanSummary,
  type SemanticMotifHierarchyLevel,
  type SemanticMotifId,
  type SemanticMotifNarrativeMappingSummary,
  type SemanticMotifParameterMappingSummary,
  type SemanticMotifRole,
  type SemanticMotifStructureMappingSummary,
  type SemanticMotifSummary,
  type SemanticMotifSystemSummary,
  type SymbolicNarrativeArchetype,
  type SymbolicNarrativePhaseName,
  type SymbolicNarrativePhaseSummary,
  type SymbolicNarrativePlanSummary,
  type WorkflowNodeId
} from "./assistant-client";
import type { AssistantRequestImageAttachment } from "./multimodal-attachments";
import {
  createWorkstationError,
  parseSubsystemErrorPayload,
  type WorkstationError
} from "./workstation-errors";

export type AssistantStreamEventType =
  | "status"
  | "memory"
  | "retrieval"
  | "context"
  | "prompt_input"
  | "planning"
  | "prompt_rendered"
  | "generation_input"
  | "tool_start"
  | "tool_result"
  | "token_delta"
  | "node_started"
  | "node_completed"
  | "node_failed"
  | "review_passed"
  | "review_failed"
  | "refinement_requested"
  | "refinement_completed"
  | "retry_started"
  | "retry_completed"
  | "artifact_extracted"
  | "artifact_critique"
  | "preview_artifact"
  | "eval_update"
  | "final"
  | "error";

export type AssistantStreamEvent = {
  event_type: AssistantStreamEventType;
  sequence: number;
  payload: Record<string, unknown>;
};

export type AssistantStreamWorkflowPhase = "running" | "completed" | "failed";

export type AssistantStreamImageReferenceMetadata = {
  id: string;
  name: string;
  mime_type: string;
  size_bytes: number;
};

export type AssistantStreamWorkflowMetadata = {
  step: WorkflowNodeId | null;
  phase: AssistantStreamWorkflowPhase;
  status: string;
  current_step: WorkflowNodeId | null;
  completed_steps: WorkflowNodeId[];
  skipped_steps: WorkflowNodeId[];
  refinement_count: number;
  review_outcome: string | null;
  review_reasons: string[];
  artifact_count: number;
  artifact_critique_count: number;
  recommended_artifact_id: string | null;
  preview_artifact_count: number;
  image_reference_count: number;
  image_references: AssistantStreamImageReferenceMetadata[];
  clarification?: ClarificationSummary | null;
  clarification_required?: boolean;
  clarification_reason?: string | null;
  clarification_question_count?: number;
  creative_intent?: CreativeIntentDecompositionSummary | null;
  intent_decomposer_available?: boolean;
  creative_hierarchy?: CreativeHierarchyPlanSummary | null;
  hierarchy_planner_available?: boolean;
  creative_strategy?: CreativeStrategySummary | null;
  strategy_available?: boolean;
  creative_techniques?: CreativeTechniqueSummary | null;
  technique_selector_available?: boolean;
  creative_plan?: CreativeExecutionPlanSummary | null;
  planning_available?: boolean;
  creative_constraints?: CreativeConstraintSolverSummary | null;
  constraint_solver_available?: boolean;
  creative_constraint_priorities?: CreativeConstraintPrioritizationSummary | null;
  constraint_prioritizer_available?: boolean;
  runtime_capabilities?: RuntimeCapabilityReasonerSummary | null;
  runtime_capability_reasoner_available?: boolean;
  runtime_compatibility?: RuntimeCompatibilityProfileSummary | null;
  runtime_compatibility_available?: boolean;
  artifact_capability_matrix?: ArtifactCapabilityMatrixSummary | null;
  artifact_capability_matrix_available?: boolean;
  multi_artifact_strategy?: MultiArtifactStrategySummary | null;
  multi_artifact_strategy_available?: boolean;
  artifact_critic?: ArtifactCriticSummary | null;
  artifact_critic_available?: boolean;
  artifact_refiner?: ArtifactRefinerSummary | null;
  artifact_refiner_available?: boolean;
  artifact_intelligence_synthesis?: ArtifactIntelligenceSynthesisSummary | null;
  artifact_intelligence_synthesis_available?: boolean;
  artifact_merge_planner?: ArtifactMergePlannerSummary | null;
  artifact_merge_planner_available?: boolean;
  artifact_export_intelligence?: ArtifactExportIntelligenceSummary | null;
  artifact_export_intelligence_available?: boolean;
  artifact_engine_contracts?: ArtifactIntelligenceEngineContractRegistrySummary | null;
  artifact_engine_contracts_available?: boolean;
  creative_critic?: CreativeCriticSummary | null;
  creative_critic_available?: boolean;
  self_evaluation?: SelfEvaluationSummary | null;
  self_evaluation_available?: boolean;
  creative_improvement_planner?: CreativeImprovementPlannerSummary | null;
  creative_improvement_planner_available?: boolean;
  reflection_loop?: ReflectionLoopSummary | null;
  reflection_loop_available?: boolean;
  creative_confidence?: CreativeConfidenceSummary | null;
  creative_confidence_available?: boolean;
  creative_score?: CreativeScoreSummary | null;
  creative_score_available?: boolean;
  consistency_validation?: ConsistencyValidationSummary | null;
  consistency_validation_available?: boolean;
  evaluation_report?: EvaluationReportSummary | null;
  evaluation_report_available?: boolean;
  creative_tradeoffs?: CreativeTradeoffExplorerSummary | null;
  tradeoff_explorer_available?: boolean;
  creative_quality_prediction?: CreativeQualityPredictionSummary | null;
  quality_predictor_available?: boolean;
  symbolic_narrative?: SymbolicNarrativePlanSummary | null;
  symbolic_narrative_available?: boolean;
  creative_composition?: CreativeCompositionPlanSummary | null;
  creative_composition_available?: boolean;
  procedural_structure?: ProceduralStructurePlanSummary | null;
  procedural_structure_available?: boolean;
  generative_structure?: GenerativeStructureBlueprintSummary | null;
  generative_structure_available?: boolean;
  semantic_motif?: SemanticMotifSystemSummary | null;
  semantic_motif_available?: boolean;
  emotional_consistency?: EmotionalConsistencyProfileSummary | null;
  emotional_consistency_available?: boolean;
  cross_modality?: CrossModalityCompositionProfileSummary | null;
  cross_modality_available?: boolean;
  audio_visual_scene?: AudioVisualSceneProfileSummary | null;
  audio_visual_scene_available?: boolean;
  artifact_plan?: ArtifactPlanSummary | null;
  artifact_planner_available?: boolean;
  artifact_dependency_graph?: ArtifactDependencyGraphSummary | null;
  artifact_dependency_graph_available?: boolean;
  creative_director?: CreativeAssistantDirectorSummary | null;
  director_available?: boolean;
  creative_reasoning?: CreativeReasoningSummary | null;
  creative_reasoning_available?: boolean;
};

export type AssistantPreviewArtifactStatus =
  | "succeeded"
  | "failed"
  | "skipped";

export type AssistantPreviewArtifactUpdate = {
  status: AssistantPreviewArtifactStatus;
  artifactId: string | null;
  previewArtifactId: string | null;
  rendererId: string | null;
  target: string | null;
  summary: string | null;
  errorMessage: string | null;
  error: WorkstationError | null;
  emittedAt: string | null;
  completedAt: string | null;
};

export type AssistantArtifactRefinementRequest = {
  artifactId: string;
  title: string;
  language: string;
  content: string;
  instruction: string;
  domain?: string | null;
  runtime?: string | null;
  rendererId?: string | null;
  previewEligible?: boolean | null;
  qualityScore?: number | null;
  qualityRank?: number | null;
  qualityBefore?: number | null;
  passNumber?: number | null;
  maxPasses?: number | null;
  refinementObjective?: string | null;
  refinementPasses?: RefinementPassRecord[];
  critiqueRationale?: string | null;
  refinementGuidance?: string | null;
  creativeTranslation?: CreativeTranslationSummary | null;
  creativePlan?: CreativeExecutionPlanSummary | null;
  critique?: ArtifactCritique | null;
};

export type AssistantStreamRequest = {
  query: string;
  conversationId?: string;
  projectId?: string;
  domain?: string;
  domains?: string[];
  mode?: string;
  attachments?: AssistantRequestImageAttachment[];
  artifactRefinement?: AssistantArtifactRefinementRequest;
  clarificationResponse?: string;
};

export type AssistantStreamOptions = {
  endpoint?: string;
  fetchImpl?: typeof fetch;
};

const defaultStreamEndpoint =
  process.env.NEXT_PUBLIC_ASSISTANT_STREAM_URL ??
  "http://localhost:8000/api/assistant/stream";

const streamEventTypes = new Set<AssistantStreamEventType>([
  "status",
  "memory",
  "retrieval",
  "context",
  "prompt_input",
  "planning",
  "prompt_rendered",
  "generation_input",
  "tool_start",
  "tool_result",
  "token_delta",
  "node_started",
  "node_completed",
  "node_failed",
  "review_passed",
  "review_failed",
  "refinement_requested",
  "refinement_completed",
  "retry_started",
  "retry_completed",
  "artifact_extracted",
  "artifact_critique",
  "preview_artifact",
  "eval_update",
  "final",
  "error"
]);

const streamEventWorkflowNodes: Partial<
  Record<AssistantStreamEventType, Record<string, WorkflowNodeId>>
> = {
  status: {
    request_received: "intake",
    route_selected: "routing"
  },
  memory: {
    memory_requested: "memory",
    memory_completed: "memory"
  },
  retrieval: {
    retrieval_requested: "retrieval",
    retrieval_completed: "retrieval"
  },
  context: {
    context_assembled: "context_assembly"
  },
  prompt_input: {
    clarification_required: "prompt_input",
    prompt_inputs_prepared: "prompt_input"
  },
  planning: {
    creative_plan_prepared: "planning",
    creative_director_prepared: "director",
    creative_reasoning_prepared: "reasoning"
  },
  prompt_rendered: {
    prompt_rendered: "prompt_rendering"
  },
  generation_input: {
    generation_input_prepared: "generation"
  },
  node_started: {
    node_started: "intake"
  },
  node_completed: {
    node_completed: "intake"
  },
  node_failed: {
    node_failed: "failure"
  },
  review_passed: {
    review_passed: "review"
  },
  review_failed: {
    review_failed: "review"
  },
  refinement_requested: {
    refinement_requested: "review"
  },
  refinement_completed: {
    refinement_completed: "refinement"
  },
  retry_started: {
    retry_started: "review"
  },
  retry_completed: {
    retry_completed: "review"
  },
  artifact_extracted: {
    artifact_extracted: "artifact_extraction"
  },
  artifact_critique: {
    artifact_refinement_requested: "artifact_critique",
    artifact_scored: "artifact_critique",
    artifact_selected_recommended: "artifact_critique",
    critique_completed: "artifact_critique",
    critique_started: "artifact_critique"
  },
  preview_artifact: {
    preview_artifact_prepared: "preview_preparation"
  }
};

export class AssistantStreamError extends Error {
  readonly detail: WorkstationError;

  constructor(message: string, detail?: WorkstationError) {
    super(message);
    this.name = "AssistantStreamError";
    this.detail =
      detail ??
      createWorkstationError({
        type: "assistant_stream_error",
        category: "stream",
        subsystem: "assistant_stream",
        userMessage: message,
        recoverable: true,
        suggestedAction: "Retry the request from the composer.",
        retryLabel: "Send prompt again"
      });
  }
}

export async function* streamAssistantEvents(
  request: AssistantStreamRequest,
  options: AssistantStreamOptions = {}
): AsyncGenerator<AssistantStreamEvent> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(options.endpoint ?? defaultStreamEndpoint, {
    body: JSON.stringify(request),
    headers: {
      "Content-Type": "application/json",
      Accept: "application/x-ndjson"
    },
    method: "POST"
  });

  yield* decodeAssistantStream(response);
}

export async function* decodeAssistantStream(
  response: Response
): AsyncGenerator<AssistantStreamEvent> {
  if (!response.ok) {
    throw new AssistantStreamError(
      `Assistant stream request failed with ${response.status}.`,
      await buildFailedHttpResponseError(response)
    );
  }

  if (!response.body) {
    throw new AssistantStreamError(
      "Assistant stream response did not include a body.",
      createWorkstationError({
        type: "empty_stream_body",
        category: "stream",
        subsystem: "assistant_stream",
        userMessage: "The live response did not include any stream data.",
        debugMessage: "Response.body was null.",
        recoverable: true,
        suggestedAction: "Retry the request from the composer.",
        retryLabel: "Send prompt again"
      })
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const event = parseAssistantStreamLine(line);
        if (event) {
          yield event;
        }
      }
    }

    buffer += decoder.decode();
    const event = parseAssistantStreamLine(buffer);
    if (event) {
      yield event;
    }
  } finally {
    reader.releaseLock();
  }
}

export function parseAssistantStreamLine(
  line: string
): AssistantStreamEvent | null {
  const trimmedLine = line.trim();
  if (!trimmedLine) {
    return null;
  }

  const parsed: unknown = JSON.parse(trimmedLine);
  if (!isAssistantStreamEvent(parsed)) {
    throw new AssistantStreamError(
      "Assistant stream line had an invalid shape.",
      createWorkstationError({
        type: "invalid_stream_event",
        category: "stream",
        subsystem: "assistant_stream_parser",
        userMessage: "The live response included an invalid stream event.",
        debugMessage: trimmedLine,
        recoverable: true,
        suggestedAction: "Retry the request. If it repeats, reset the workspace session.",
        retryLabel: "Send prompt again",
        resetLabel: "Clear workspace session"
      })
    );
  }

  return parsed;
}

export function workflowNodeFromAssistantStreamEvent(
  event: AssistantStreamEvent
): WorkflowNodeId | undefined {
  const workflow = readWorkflowMetadata(event);
  if (workflow?.current_step) {
    return workflow.current_step;
  }
  if (workflow?.step) {
    return workflow.step;
  }

  if (event.event_type === "token_delta") {
    return "generation";
  }

  if (event.event_type === "preview_artifact") {
    return "preview_preparation";
  }

  if (event.event_type === "final") {
    return "finalization";
  }

  if (event.event_type === "error") {
    return "failure";
  }

  const node = parseWorkflowNodeId(event.payload.node);
  if (node) {
    return node;
  }

  const code = event.payload.code;
  if (typeof code !== "string") {
    return undefined;
  }

  return streamEventWorkflowNodes[event.event_type]?.[code];
}

export function readWorkflowMetadata(
  event: AssistantStreamEvent
): AssistantStreamWorkflowMetadata | null {
  const rawWorkflow = event.payload.workflow;
  if (!isRecord(rawWorkflow)) {
    return null;
  }

  const step = parseWorkflowNodeId(rawWorkflow.step);
  const currentStep = parseWorkflowNodeId(rawWorkflow.current_step);
  const phase = rawWorkflow.phase;
  const status = rawWorkflow.status;
  const completedSteps = parseWorkflowNodeIdList(rawWorkflow.completed_steps);
  const skippedSteps = parseWorkflowNodeIdList(rawWorkflow.skipped_steps);
  const reviewReasons = parseStringList(rawWorkflow.review_reasons);
  const artifactCount =
    typeof rawWorkflow.artifact_count === "number"
      ? rawWorkflow.artifact_count
      : 0;
  const artifactCritiqueCount =
    typeof rawWorkflow.artifact_critique_count === "number"
      ? rawWorkflow.artifact_critique_count
      : 0;
  const recommendedArtifactId =
    typeof rawWorkflow.recommended_artifact_id === "string"
      ? rawWorkflow.recommended_artifact_id
      : null;
  const previewArtifactCount =
    typeof rawWorkflow.preview_artifact_count === "number"
      ? rawWorkflow.preview_artifact_count
      : 0;
  const imageReferences = parseImageReferenceMetadataList(
    rawWorkflow.image_references
  );
  const reviewOutcome =
    typeof rawWorkflow.review_outcome === "string"
      ? rawWorkflow.review_outcome
      : null;
  const refinementCount =
    typeof rawWorkflow.refinement_count === "number"
      ? rawWorkflow.refinement_count
      : 0;
  const imageReferenceCount =
    typeof rawWorkflow.image_reference_count === "number"
      ? rawWorkflow.image_reference_count
      : imageReferences.length;
  const clarification = readClarificationSummary(rawWorkflow.clarification);
  const clarificationRequired =
    rawWorkflow.clarification_required === true || clarification !== null;
  const clarificationReason =
    typeof rawWorkflow.clarification_reason === "string"
      ? rawWorkflow.clarification_reason
      : clarification?.reason ?? null;
  const clarificationQuestionCount =
    typeof rawWorkflow.clarification_question_count === "number"
      ? rawWorkflow.clarification_question_count
      : clarification?.questions.length ?? 0;
  const creativeIntent = readCreativeIntentDecompositionSummary(
    rawWorkflow.creative_intent ?? rawWorkflow.creativeIntent
  );
  const intentDecomposerAvailable =
    rawWorkflow.intent_decomposer_available === true || creativeIntent !== null;
  const creativeHierarchy = readCreativeHierarchyPlanSummary(
    rawWorkflow.creative_hierarchy ?? rawWorkflow.creativeHierarchy
  );
  const hierarchyPlannerAvailable =
    rawWorkflow.hierarchy_planner_available === true ||
    creativeHierarchy !== null;
  const creativeStrategy = readCreativeStrategySummary(
    rawWorkflow.creative_strategy ?? rawWorkflow.creativeStrategy
  );
  const strategyAvailable =
    rawWorkflow.strategy_available === true || creativeStrategy !== null;
  const creativeTechniques = readCreativeTechniqueSummary(
    rawWorkflow.creative_techniques ?? rawWorkflow.creativeTechniques
  );
  const techniqueSelectorAvailable =
    rawWorkflow.technique_selector_available === true ||
    creativeTechniques !== null;
  const creativePlan = readCreativeExecutionPlanSummary(
    rawWorkflow.creative_plan ?? rawWorkflow.creativePlan
  );
  const planningAvailable =
    rawWorkflow.planning_available === true || creativePlan !== null;
  const creativeConstraints = readCreativeConstraintSolverSummary(
    rawWorkflow.creative_constraints ?? rawWorkflow.creativeConstraints
  );
  const constraintSolverAvailable =
    rawWorkflow.constraint_solver_available === true ||
    creativeConstraints !== null;
  const creativeConstraintPriorities =
    readCreativeConstraintPrioritizationSummary(
      rawWorkflow.creative_constraint_priorities ??
        rawWorkflow.creativeConstraintPriorities
    );
  const constraintPrioritizerAvailable =
    rawWorkflow.constraint_prioritizer_available === true ||
    creativeConstraintPriorities !== null;
  const runtimeCapabilities = readRuntimeCapabilityReasonerSummary(
    rawWorkflow.runtime_capabilities ?? rawWorkflow.runtimeCapabilities
  );
  const runtimeCapabilityReasonerAvailable =
    rawWorkflow.runtime_capability_reasoner_available === true ||
    runtimeCapabilities !== null;
  const runtimeCompatibility = readRuntimeCompatibilityProfileSummary(
    rawWorkflow.runtime_compatibility ?? rawWorkflow.runtimeCompatibility
  );
  const runtimeCompatibilityAvailable =
    rawWorkflow.runtime_compatibility_available === true ||
    runtimeCompatibility !== null;
  const artifactCapabilityMatrix = readArtifactCapabilityMatrixSummary(
    rawWorkflow.artifact_capability_matrix ??
      rawWorkflow.artifactCapabilityMatrix
  );
  const artifactCapabilityMatrixAvailable =
    rawWorkflow.artifact_capability_matrix_available === true ||
    artifactCapabilityMatrix !== null;
  const multiArtifactStrategy = readMultiArtifactStrategySummary(
    rawWorkflow.multi_artifact_strategy ?? rawWorkflow.multiArtifactStrategy
  );
  const multiArtifactStrategyAvailable =
    rawWorkflow.multi_artifact_strategy_available === true ||
    multiArtifactStrategy !== null;
  const artifactCritic = readArtifactCriticSummary(
    rawWorkflow.artifact_critic ?? rawWorkflow.artifactCritic
  );
  const artifactCriticAvailable =
    rawWorkflow.artifact_critic_available === true || artifactCritic !== null;
  const artifactRefiner = readArtifactRefinerSummary(
    rawWorkflow.artifact_refiner ?? rawWorkflow.artifactRefiner
  );
  const artifactRefinerAvailable =
    rawWorkflow.artifact_refiner_available === true || artifactRefiner !== null;
  const artifactIntelligenceSynthesis =
    readArtifactIntelligenceSynthesisSummary(
      rawWorkflow.artifact_intelligence_synthesis ??
        rawWorkflow.artifactIntelligenceSynthesis
    );
  const artifactIntelligenceSynthesisAvailable =
    rawWorkflow.artifact_intelligence_synthesis_available === true ||
    artifactIntelligenceSynthesis !== null;
  const artifactMergePlanner = readArtifactMergePlannerSummary(
    rawWorkflow.artifact_merge_planner ?? rawWorkflow.artifactMergePlanner
  );
  const artifactMergePlannerAvailable =
    rawWorkflow.artifact_merge_planner_available === true ||
    artifactMergePlanner !== null;
  const artifactExportIntelligence = readArtifactExportIntelligenceSummary(
    rawWorkflow.artifact_export_intelligence ??
      rawWorkflow.artifactExportIntelligence
  );
  const artifactExportIntelligenceAvailable =
    rawWorkflow.artifact_export_intelligence_available === true ||
    artifactExportIntelligence !== null;
  const artifactEngineContracts = readArtifactEngineContractRegistrySummary(
    rawWorkflow.artifact_engine_contracts ?? rawWorkflow.artifactEngineContracts
  );
  const artifactEngineContractsAvailable =
    rawWorkflow.artifact_engine_contracts_available === true ||
    artifactEngineContracts !== null;
  const creativeCritic = readCreativeCriticSummary(
    rawWorkflow.creative_critic ?? rawWorkflow.creativeCritic
  );
  const creativeCriticAvailable =
    rawWorkflow.creative_critic_available === true || creativeCritic !== null;
  const selfEvaluation = readSelfEvaluationSummary(
    rawWorkflow.self_evaluation ?? rawWorkflow.selfEvaluation
  );
  const selfEvaluationAvailable =
    rawWorkflow.self_evaluation_available === true || selfEvaluation !== null;
  const creativeImprovementPlanner = readCreativeImprovementPlannerSummary(
    rawWorkflow.creative_improvement_planner ??
      rawWorkflow.creativeImprovementPlanner
  );
  const creativeImprovementPlannerAvailable =
    rawWorkflow.creative_improvement_planner_available === true ||
    creativeImprovementPlanner !== null;
  const reflectionLoop = readReflectionLoopSummary(
    rawWorkflow.reflection_loop ?? rawWorkflow.reflectionLoop
  );
  const reflectionLoopAvailable =
    rawWorkflow.reflection_loop_available === true || reflectionLoop !== null;
  const creativeConfidence = readCreativeConfidenceSummary(
    rawWorkflow.creative_confidence ?? rawWorkflow.creativeConfidence
  );
  const creativeConfidenceAvailable =
    rawWorkflow.creative_confidence_available === true ||
    creativeConfidence !== null;
  const creativeScore = readCreativeScoreSummary(
    rawWorkflow.creative_score ?? rawWorkflow.creativeScore
  );
  const creativeScoreAvailable =
    rawWorkflow.creative_score_available === true || creativeScore !== null;
  const consistencyValidation = readConsistencyValidationSummary(
    rawWorkflow.consistency_validation ?? rawWorkflow.consistencyValidation
  );
  const consistencyValidationAvailable =
    rawWorkflow.consistency_validation_available === true ||
    consistencyValidation !== null;
  const evaluationReport = readEvaluationReportSummary(
    rawWorkflow.evaluation_report ?? rawWorkflow.evaluationReport
  );
  const evaluationReportAvailable =
    rawWorkflow.evaluation_report_available === true ||
    evaluationReport !== null;
  const creativeTradeoffs = readCreativeTradeoffExplorerSummary(
    rawWorkflow.creative_tradeoffs ?? rawWorkflow.creativeTradeoffs
  );
  const tradeoffExplorerAvailable =
    rawWorkflow.tradeoff_explorer_available === true ||
    creativeTradeoffs !== null;
  const creativeQualityPrediction = readCreativeQualityPredictionSummary(
    rawWorkflow.creative_quality_prediction ??
      rawWorkflow.creativeQualityPrediction
  );
  const qualityPredictorAvailable =
    rawWorkflow.quality_predictor_available === true ||
    creativeQualityPrediction !== null;
  const symbolicNarrative = readSymbolicNarrativePlanSummary(
    rawWorkflow.symbolic_narrative ?? rawWorkflow.symbolicNarrative
  );
  const symbolicNarrativeAvailable =
    rawWorkflow.symbolic_narrative_available === true ||
    symbolicNarrative !== null;
  const creativeComposition = readCreativeCompositionPlanSummary(
    rawWorkflow.creative_composition ?? rawWorkflow.creativeComposition
  );
  const creativeCompositionAvailable =
    rawWorkflow.creative_composition_available === true ||
    creativeComposition !== null;
  const proceduralStructure = readProceduralStructurePlanSummary(
    rawWorkflow.procedural_structure ?? rawWorkflow.proceduralStructure
  );
  const proceduralStructureAvailable =
    rawWorkflow.procedural_structure_available === true ||
    proceduralStructure !== null;
  const generativeStructure = readGenerativeStructureBlueprintSummary(
    rawWorkflow.generative_structure ?? rawWorkflow.generativeStructure
  );
  const generativeStructureAvailable =
    rawWorkflow.generative_structure_available === true ||
    generativeStructure !== null;
  const semanticMotif = readSemanticMotifSystemSummary(
    rawWorkflow.semantic_motif ?? rawWorkflow.semanticMotif
  );
  const semanticMotifAvailable =
    rawWorkflow.semantic_motif_available === true || semanticMotif !== null;
  const emotionalConsistency = readEmotionalConsistencyProfileSummary(
    rawWorkflow.emotional_consistency ?? rawWorkflow.emotionalConsistency
  );
  const emotionalConsistencyAvailable =
    rawWorkflow.emotional_consistency_available === true ||
    emotionalConsistency !== null;
  const crossModality = readCrossModalityCompositionProfileSummary(
    rawWorkflow.cross_modality ?? rawWorkflow.crossModality
  );
  const crossModalityAvailable =
    rawWorkflow.cross_modality_available === true || crossModality !== null;
  const audioVisualScene = readAudioVisualSceneProfileSummary(
    rawWorkflow.audio_visual_scene ?? rawWorkflow.audioVisualScene
  );
  const audioVisualSceneAvailable =
    rawWorkflow.audio_visual_scene_available === true ||
    audioVisualScene !== null;
  const artifactPlan = readArtifactPlanSummary(
    rawWorkflow.artifact_plan ?? rawWorkflow.artifactPlan
  );
  const artifactPlannerAvailable =
    rawWorkflow.artifact_planner_available === true || artifactPlan !== null;
  const artifactDependencyGraph = readArtifactDependencyGraphSummary(
    rawWorkflow.artifact_dependency_graph ?? rawWorkflow.artifactDependencyGraph
  );
  const artifactDependencyGraphAvailable =
    rawWorkflow.artifact_dependency_graph_available === true ||
    artifactDependencyGraph !== null;
  const creativeDirector = readCreativeAssistantDirectorSummary(
    rawWorkflow.creative_director ?? rawWorkflow.creativeDirector
  );
  const directorAvailable =
    rawWorkflow.director_available === true || creativeDirector !== null;
  const creativeReasoning = readCreativeReasoningSummary(
    rawWorkflow.creative_reasoning ?? rawWorkflow.creativeReasoning
  );
  const creativeReasoningAvailable =
    rawWorkflow.creative_reasoning_available === true ||
    creativeReasoning !== null;

  if (
    (phase !== "running" && phase !== "completed" && phase !== "failed") ||
    typeof status !== "string"
  ) {
    return null;
  }

  return {
    step,
    phase,
    status,
    current_step: currentStep,
    completed_steps: completedSteps,
    skipped_steps: skippedSteps,
    refinement_count: refinementCount,
    review_outcome: reviewOutcome,
    review_reasons: reviewReasons,
    artifact_count: artifactCount,
    artifact_critique_count: artifactCritiqueCount,
    recommended_artifact_id: recommendedArtifactId,
    preview_artifact_count: previewArtifactCount,
    image_reference_count: imageReferenceCount,
    image_references: imageReferences,
    ...(clarificationRequired
      ? {
          clarification,
          clarification_required: true,
          clarification_reason: clarificationReason,
          clarification_question_count: clarificationQuestionCount
        }
      : {}),
    ...(intentDecomposerAvailable
      ? {
          creative_intent: creativeIntent,
          intent_decomposer_available: true
        }
      : {}),
    ...(hierarchyPlannerAvailable
      ? {
          creative_hierarchy: creativeHierarchy,
          hierarchy_planner_available: true
        }
      : {}),
    ...(strategyAvailable
      ? {
          creative_strategy: creativeStrategy,
          strategy_available: true
        }
      : {}),
    ...(techniqueSelectorAvailable
      ? {
          creative_techniques: creativeTechniques,
          technique_selector_available: true
        }
      : {}),
    ...(planningAvailable
      ? {
          creative_plan: creativePlan,
          planning_available: true
        }
      : {}),
    ...(constraintSolverAvailable
      ? {
          creative_constraints: creativeConstraints,
          constraint_solver_available: true
        }
      : {}),
    ...(constraintPrioritizerAvailable
      ? {
          creative_constraint_priorities: creativeConstraintPriorities,
          constraint_prioritizer_available: true
        }
      : {}),
    ...(runtimeCapabilityReasonerAvailable
      ? {
          runtime_capabilities: runtimeCapabilities,
          runtime_capability_reasoner_available: true
        }
      : {}),
    ...(runtimeCompatibilityAvailable
      ? {
          runtime_compatibility: runtimeCompatibility,
          runtime_compatibility_available: true
        }
      : {}),
    ...(artifactCapabilityMatrixAvailable
      ? {
          artifact_capability_matrix: artifactCapabilityMatrix,
          artifact_capability_matrix_available: true
        }
      : {}),
    ...(multiArtifactStrategyAvailable
      ? {
          multi_artifact_strategy: multiArtifactStrategy,
          multi_artifact_strategy_available: true
        }
      : {}),
    ...(artifactCriticAvailable
      ? {
          artifact_critic: artifactCritic,
          artifact_critic_available: true
        }
      : {}),
    ...(artifactRefinerAvailable
      ? {
          artifact_refiner: artifactRefiner,
          artifact_refiner_available: true
        }
      : {}),
    ...(artifactIntelligenceSynthesisAvailable
      ? {
          artifact_intelligence_synthesis: artifactIntelligenceSynthesis,
          artifact_intelligence_synthesis_available: true
        }
      : {}),
    ...(artifactMergePlannerAvailable
      ? {
          artifact_merge_planner: artifactMergePlanner,
          artifact_merge_planner_available: true
        }
      : {}),
    ...(artifactExportIntelligenceAvailable
      ? {
          artifact_export_intelligence: artifactExportIntelligence,
          artifact_export_intelligence_available: true
        }
      : {}),
    ...(artifactEngineContractsAvailable
      ? {
          artifact_engine_contracts: artifactEngineContracts,
          artifact_engine_contracts_available: true
        }
      : {}),
    ...(creativeCriticAvailable
      ? {
          creative_critic: creativeCritic,
          creative_critic_available: true
        }
      : {}),
    ...(selfEvaluationAvailable
      ? {
          self_evaluation: selfEvaluation,
          self_evaluation_available: true
        }
      : {}),
    ...(creativeImprovementPlannerAvailable
      ? {
          creative_improvement_planner: creativeImprovementPlanner,
          creative_improvement_planner_available: true
        }
      : {}),
    ...(reflectionLoopAvailable
      ? {
          reflection_loop: reflectionLoop,
          reflection_loop_available: true
        }
      : {}),
    ...(creativeConfidenceAvailable
      ? {
          creative_confidence: creativeConfidence,
          creative_confidence_available: true
        }
      : {}),
    ...(creativeScoreAvailable
      ? {
          creative_score: creativeScore,
          creative_score_available: true
        }
      : {}),
    ...(consistencyValidationAvailable
      ? {
          consistency_validation: consistencyValidation,
          consistency_validation_available: true
        }
      : {}),
    ...(evaluationReportAvailable
      ? {
          evaluation_report: evaluationReport,
          evaluation_report_available: true
        }
      : {}),
    ...(tradeoffExplorerAvailable
      ? {
          creative_tradeoffs: creativeTradeoffs,
          tradeoff_explorer_available: true
        }
      : {}),
    ...(qualityPredictorAvailable
      ? {
          creative_quality_prediction: creativeQualityPrediction,
          quality_predictor_available: true
        }
      : {}),
    ...(symbolicNarrativeAvailable
      ? {
          symbolic_narrative: symbolicNarrative,
          symbolic_narrative_available: true
        }
      : {}),
    ...(creativeCompositionAvailable
      ? {
          creative_composition: creativeComposition,
          creative_composition_available: true
        }
      : {}),
    ...(proceduralStructureAvailable
      ? {
          procedural_structure: proceduralStructure,
          procedural_structure_available: true
        }
      : {}),
    ...(generativeStructureAvailable
      ? {
          generative_structure: generativeStructure,
          generative_structure_available: true
        }
      : {}),
    ...(semanticMotifAvailable
      ? {
          semantic_motif: semanticMotif,
          semantic_motif_available: true
        }
      : {}),
    ...(emotionalConsistencyAvailable
      ? {
          emotional_consistency: emotionalConsistency,
          emotional_consistency_available: true
        }
      : {}),
    ...(crossModalityAvailable
      ? {
          cross_modality: crossModality,
          cross_modality_available: true
        }
      : {}),
    ...(audioVisualSceneAvailable
      ? {
          audio_visual_scene: audioVisualScene,
          audio_visual_scene_available: true
        }
      : {}),
    ...(artifactPlannerAvailable
      ? {
          artifact_plan: artifactPlan,
          artifact_planner_available: true
        }
      : {}),
    ...(artifactDependencyGraphAvailable
      ? {
          artifact_dependency_graph: artifactDependencyGraph,
          artifact_dependency_graph_available: true
        }
      : {}),
    ...(directorAvailable
      ? {
          creative_director: creativeDirector,
          director_available: true
        }
      : {}),
    ...(creativeReasoningAvailable
      ? {
          creative_reasoning: creativeReasoning,
          creative_reasoning_available: true
        }
      : {})
  };
}

export function readCreativeStrategySummary(
  value: unknown
): CreativeStrategySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryStrategy = readStringUnion(
    value,
    "primary_strategy",
    "primaryStrategy",
    creativeStrategyIds
  );
  const confidence = readFiniteNumberField(value, "confidence");
  const rationale = readStringField(value, "rationale");
  const implementationBoundary =
    readStringField(value, "implementation_boundary") ??
    readStringField(value, "implementationBoundary");
  const creativeGoals = readStringListField(
    value,
    "creative_goals",
    "creativeGoals"
  );
  const strategyDirectives = readStringListField(
    value,
    "strategy_directives",
    "strategyDirectives"
  );

  if (
    role !== "creative_strategy_engine" ||
    !primaryStrategy ||
    confidence === null ||
    !rationale ||
    !implementationBoundary ||
    creativeGoals.length === 0 ||
    strategyDirectives.length === 0
  ) {
    return null;
  }

  return {
    role,
    primaryStrategy,
    confidence,
    rationale,
    creativeGoals,
    symbolicAlignment: readStringListField(
      value,
      "symbolic_alignment",
      "symbolicAlignment"
    ),
    alternativeStrategies: readCreativeStrategyAlternativeSummaryList(
      value.alternative_strategies ?? value.alternativeStrategies
    ),
    strategyDirectives,
    implementationBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readCreativeIntentDecompositionSummary(
  value: unknown
): CreativeIntentDecompositionSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const normalizedIntent =
    readStringField(value, "normalized_intent") ??
    readStringField(value, "normalizedIntent");
  const primaryExpression =
    readStringField(value, "primary_expression") ??
    readStringField(value, "primaryExpression");
  const abstractionLevel = readStringUnion(
    value,
    "abstraction_level",
    "abstractionLevel",
    creativeAbstractionLevels
  );
  const experientialGoal =
    readStringField(value, "experiential_goal") ??
    readStringField(value, "experientialGoal");
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const atomicDimensions = readCreativeIntentDimensionSummaryList(
    value.atomic_dimensions ?? value.atomicDimensions
  );
  const narrativeIntent = readCreativeIntentDimensionSummary(
    value.narrative_intent ?? value.narrativeIntent
  );
  const symbolicIntent = readCreativeIntentDimensionSummary(
    value.symbolic_intent ?? value.symbolicIntent
  );
  const emotionalIntent = readCreativeIntentDimensionSummary(
    value.emotional_intent ?? value.emotionalIntent
  );
  const geometricIntent = readCreativeIntentDimensionSummary(
    value.geometric_intent ?? value.geometricIntent
  );
  const motionIntent = readCreativeIntentDimensionSummary(
    value.motion_intent ?? value.motionIntent
  );
  const rhythmIntent = readCreativeIntentDimensionSummary(
    value.rhythm_intent ?? value.rhythmIntent
  );
  const lightColorIntent = readCreativeIntentDimensionSummary(
    value.light_color_intent ?? value.lightColorIntent
  );
  const audioIntent = readCreativeIntentDimensionSummary(
    value.audio_intent ?? value.audioIntent
  );
  const interactionIntent = readCreativeIntentDimensionSummary(
    value.interaction_intent ?? value.interactionIntent
  );
  const climaxTransformationIntent = readCreativeIntentDimensionSummary(
    value.climax_transformation_intent ?? value.climaxTransformationIntent
  );

  if (
    role !== "creative_intent_decomposer" ||
    !normalizedIntent ||
    !primaryExpression ||
    !abstractionLevel ||
    !experientialGoal ||
    !authorityBoundary ||
    atomicDimensions.length === 0 ||
    !narrativeIntent ||
    !symbolicIntent ||
    !emotionalIntent ||
    !geometricIntent ||
    !motionIntent ||
    !rhythmIntent ||
    !lightColorIntent ||
    !audioIntent ||
    !interactionIntent ||
    !climaxTransformationIntent
  ) {
    return null;
  }

  return {
    role,
    normalizedIntent,
    primaryExpression,
    narrativeIntent,
    symbolicIntent,
    emotionalIntent,
    geometricIntent,
    motionIntent,
    rhythmIntent,
    lightColorIntent,
    audioIntent,
    interactionIntent,
    climaxTransformationIntent,
    abstractionLevel,
    experientialGoal,
    unresolvedIntentGaps: readStringListField(
      value,
      "unresolved_intent_gaps",
      "unresolvedIntentGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    atomicDimensions,
    promptGuidance: readStringListField(
      value,
      "prompt_guidance",
      "promptGuidance"
    ),
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeIntentDimensionNames = [
  "narrative",
  "symbolic",
  "emotional",
  "geometric",
  "motion",
  "rhythm",
  "light_color",
  "audio",
  "interaction",
  "climax_transformation"
] as const satisfies readonly CreativeIntentDimensionName[];

const creativeIntentExplicitnessValues = [
  "explicit",
  "inferred",
  "absent",
  "ambiguous"
] as const satisfies readonly CreativeIntentExplicitness[];

const creativeAbstractionLevels = [
  "literal",
  "stylized",
  "symbolic",
  "abstract",
  "mixed",
  "unspecified"
] as const satisfies readonly CreativeAbstractionLevel[];

function readCreativeIntentDimensionSummaryList(
  value: unknown
): CreativeIntentDimensionSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const parsed = readCreativeIntentDimensionSummary(item);
    return parsed ? [parsed] : [];
  });
}

function readCreativeIntentDimensionSummary(
  value: unknown
): CreativeIntentDimensionSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const name = readStringUnion(
    value,
    "name",
    "name",
    creativeIntentDimensionNames
  );
  const explicitness = readStringUnion(
    value,
    "explicitness",
    "explicitness",
    creativeIntentExplicitnessValues
  );
  const summary = readStringField(value, "summary");
  const guidance = readStringListField(value, "guidance", "guidance");

  if (!name || !explicitness || !summary || guidance.length === 0) {
    return null;
  }

  return {
    name,
    explicitness,
    summary,
    signals: readStringListField(value, "signals", "signals"),
    guidance
  };
}

export function readCreativeHierarchyPlanSummary(
  value: unknown
): CreativeHierarchyPlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryCreativePriorities = readCreativeHierarchyPrioritySummaryList(
    value.primary_creative_priorities ?? value.primaryCreativePriorities
  );
  const hierarchyConfidence =
    readFiniteNumberField(value, "hierarchy_confidence") ??
    readFiniteNumberField(value, "hierarchyConfidence");
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    role !== "creative_hierarchy_planner" ||
    primaryCreativePriorities.length === 0 ||
    hierarchyConfidence === null ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    primaryCreativePriorities,
    secondaryCreativePriorities: readCreativeHierarchyPrioritySummaryList(
      value.secondary_creative_priorities ?? value.secondaryCreativePriorities
    ),
    nonNegotiableDimensions: readCreativeHierarchyDimensionList(
      value.non_negotiable_dimensions ?? value.nonNegotiableDimensions
    ),
    flexibleDimensions: readCreativeHierarchyDimensionList(
      value.flexible_dimensions ?? value.flexibleDimensions
    ),
    priorityRationale: readStringListField(
      value,
      "priority_rationale",
      "priorityRationale"
    ),
    priorityConflicts: readStringListField(
      value,
      "priority_conflicts",
      "priorityConflicts"
    ),
    hierarchyConfidence,
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeHierarchyDimensions = [
  "symbolism",
  "narrative",
  "emotion",
  "geometry",
  "motion",
  "rhythm",
  "light_color",
  "audio",
  "interaction",
  "visual_impact",
  "performance",
  "simplicity",
  "complexity",
  "runtime_safety",
  "experiential_depth"
] as const satisfies readonly CreativeHierarchyDimension[];

const creativeHierarchyTiers = [
  "primary",
  "secondary",
  "flexible"
] as const satisfies readonly CreativeHierarchyTier[];

const creativeHierarchySources = [
  "explicit",
  "implied",
  "coherence",
  "constraint"
] as const satisfies readonly CreativeHierarchySource[];

function readCreativeHierarchyPrioritySummaryList(
  value: unknown
): CreativeHierarchyPrioritySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const dimension = readStringUnion(
      item,
      "dimension",
      "dimension",
      creativeHierarchyDimensions
    );
    const tier = readStringUnion(item, "tier", "tier", creativeHierarchyTiers);
    const rank = readFiniteNumberField(item, "rank");
    const priorityScore =
      readFiniteNumberField(item, "priority_score") ??
      readFiniteNumberField(item, "priorityScore");
    const source = readStringUnion(
      item,
      "source",
      "source",
      creativeHierarchySources
    );
    const rationale = readStringField(item, "rationale");
    const sacrificeGuidance =
      readStringField(item, "sacrifice_guidance") ??
      readStringField(item, "sacrificeGuidance");

    if (
      !dimension ||
      !tier ||
      rank === null ||
      priorityScore === null ||
      !source ||
      !rationale ||
      !sacrificeGuidance
    ) {
      return [];
    }

    return [
      {
        dimension,
        tier,
        rank,
        priorityScore,
        source,
        rationale,
        evidence: readStringListField(item, "evidence", "evidence"),
        sacrificeGuidance
      }
    ];
  });
}

function readCreativeHierarchyDimensionList(
  value: unknown
): CreativeHierarchyDimension[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is CreativeHierarchyDimension =>
    creativeHierarchyDimensions.includes(item as CreativeHierarchyDimension)
  );
}

const creativeStrategyIds = [
  "recursive_emergence",
  "fractal_growth",
  "particle_cosmology",
  "cellular_evolution",
  "sacred_geometry",
  "field_dynamics",
  "minimal_generative_systems"
] as const satisfies readonly CreativeStrategyId[];

function readCreativeStrategyAlternativeSummaryList(
  value: unknown
): CreativeStrategyAlternativeSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const strategy = readStringUnion(
      item,
      "strategy",
      "strategy",
      creativeStrategyIds
    );
    const confidence = readFiniteNumberField(item, "confidence");
    const rationale = readStringField(item, "rationale");

    if (!strategy || confidence === null || !rationale) {
      return [];
    }

    return [
      {
        strategy,
        confidence,
        rationale
      }
    ];
  });
}

export function readCreativeTechniqueSummary(
  value: unknown
): CreativeTechniqueSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryTechnique = readStringUnion(
    value,
    "primary_technique",
    "primaryTechnique",
    creativeTechniqueIds
  );
  const confidence = readFiniteNumberField(value, "confidence");
  const rationale = readStringField(value, "rationale");
  const strategyAlignment = readStringUnion(
    value,
    "strategy_alignment",
    "strategyAlignment",
    creativeStrategyIds
  );
  const compatibility = readStringUnion(
    value,
    "compatibility",
    "compatibility",
    creativeTechniqueCompatibilities
  );
  const complexityPressure = readStringUnion(
    value,
    "complexity_pressure",
    "complexityPressure",
    creativeTechniquePressures
  );
  const performancePressure = readStringUnion(
    value,
    "performance_pressure",
    "performancePressure",
    creativeTechniquePressures
  );
  const artisticSuitability = readStringListField(
    value,
    "artistic_suitability",
    "artisticSuitability"
  );
  const implementationNotes = readStringListField(
    value,
    "implementation_notes",
    "implementationNotes"
  );
  const techniqueConstraints = readStringListField(
    value,
    "technique_constraints",
    "techniqueConstraints"
  );
  const selectionBoundary =
    readStringField(value, "selection_boundary") ??
    readStringField(value, "selectionBoundary");

  if (
    role !== "creative_technique_selector" ||
    !primaryTechnique ||
    confidence === null ||
    !rationale ||
    !compatibility ||
    !complexityPressure ||
    !performancePressure ||
    artisticSuitability.length === 0 ||
    implementationNotes.length === 0 ||
    techniqueConstraints.length === 0 ||
    !selectionBoundary
  ) {
    return null;
  }

  return {
    role,
    primaryTechnique,
    confidence,
    rationale,
    strategyAlignment,
    compatibility,
    complexityPressure,
    performancePressure,
    artisticSuitability,
    implementationNotes,
    alternativeTechniques: readCreativeTechniqueAlternativeSummaryList(
      value.alternative_techniques ?? value.alternativeTechniques
    ),
    techniqueConstraints,
    selectionBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeTechniqueIds = [
  "fractal_recursion",
  "particle_systems",
  "reaction_diffusion",
  "boids",
  "cellular_automata",
  "voronoi",
  "noise_fields",
  "recursive_geometry",
  "sdf",
  "signed_distance_composition",
  "feedback_systems",
  "audio_reactive_mappings"
] as const satisfies readonly CreativeTechniqueId[];

const creativeTechniqueCompatibilities = [
  "strong",
  "moderate",
  "weak"
] as const satisfies readonly CreativeTechniqueCompatibility[];

const creativeTechniquePressures = [
  "low",
  "medium",
  "high"
] as const satisfies readonly CreativeTechniquePressure[];

function readCreativeTechniqueAlternativeSummaryList(
  value: unknown
): CreativeTechniqueAlternativeSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const technique = readStringUnion(
      item,
      "technique",
      "technique",
      creativeTechniqueIds
    );
    const confidence = readFiniteNumberField(item, "confidence");
    const rationale = readStringField(item, "rationale");

    if (!technique || confidence === null || !rationale) {
      return [];
    }

    return [
      {
        technique,
        confidence,
        rationale
      }
    ];
  });
}

export function readRuntimeCapabilityReasonerSummary(
  value: unknown
): RuntimeCapabilityReasonerSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const outputGoal =
    readStringField(value, "output_goal") ??
    readStringField(value, "outputGoal");
  const likelyCandidates = readRuntimeCapabilityIdList(
    value.likely_candidates ?? value.likelyCandidates
  );
  const candidateRuntimes = readRuntimeCapabilityCandidateSummaryList(
    value.candidate_runtimes ?? value.candidateRuntimes
  );
  const hitlAdvisable =
    readBooleanField(value, "hitl_advisable") ??
    readBooleanField(value, "hitlAdvisable");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "runtime_capability_reasoner" ||
    !outputGoal ||
    likelyCandidates.length === 0 ||
    candidateRuntimes.length === 0 ||
    hitlAdvisable === null ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    outputGoal,
    likelyCandidates,
    candidateRuntimes,
    strategyContext:
      readStringField(value, "strategy_context") ??
      readStringField(value, "strategyContext"),
    techniqueContext:
      readStringField(value, "technique_context") ??
      readStringField(value, "techniqueContext"),
    constraintContext:
      readStringField(value, "constraint_context") ??
      readStringField(value, "constraintContext"),
    hitlAdvisable,
    hitlReason:
      readStringField(value, "hitl_reason") ??
      readStringField(value, "hitlReason"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readRuntimeCompatibilityProfileSummary(
  value: unknown
): RuntimeCompatibilityProfileSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const compatibleRuntimes = readRuntimeCapabilityIdList(
    value.compatible_runtimes ?? value.compatibleRuntimes
  );
  const unsupportedRuntimes = readRuntimeCapabilityIdList(
    value.unsupported_runtimes ?? value.unsupportedRuntimes
  );
  const preferredRuntimes = readRuntimeCapabilityIdList(
    value.preferred_runtimes ?? value.preferredRuntimes
  );
  const runtimeConfidence = readRuntimeCompatibilityConfidenceSummaryList(
    value.runtime_confidence ?? value.runtimeConfidence
  );
  const compatibilityAssessments =
    readRuntimeCompatibilityAssessmentSummaryList(
      value.compatibility_assessments ?? value.compatibilityAssessments
    );
  const expectedImplementationComplexity = readStringUnion(
    value,
    "expected_implementation_complexity",
    "expectedImplementationComplexity",
    runtimeCapabilityComplexities
  );
  const portability = readStringUnion(
    value,
    "portability",
    "portability",
    runtimePortabilities
  );
  const interoperability = readStringUnion(
    value,
    "interoperability",
    "interoperability",
    runtimeInteroperabilities
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "runtime_compatibility_engine" ||
    compatibilityAssessments.length === 0 ||
    !expectedImplementationComplexity ||
    !portability ||
    !interoperability ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    compatibleRuntimes,
    unsupportedRuntimes,
    preferredRuntimes,
    runtimeConfidence,
    compatibilityAssessments,
    runtimeRequirements: readStringListField(
      value,
      "runtime_requirements",
      "runtimeRequirements"
    ),
    runtimeLimitations: readStringListField(
      value,
      "runtime_limitations",
      "runtimeLimitations"
    ),
    dependencyCompatibility: readStringListField(
      value,
      "dependency_compatibility",
      "dependencyCompatibility"
    ),
    expectedImplementationComplexity,
    portability,
    interoperability,
    missingRuntimeInformation: readStringListField(
      value,
      "missing_runtime_information",
      "missingRuntimeInformation"
    ),
    implementationRisks: readStringListField(
      value,
      "implementation_risks",
      "implementationRisks"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readRuntimeCompatibilityConfidenceSummaryList(
  value: unknown
): RuntimeCompatibilityConfidenceSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }
    const runtime = readStringUnion(
      item,
      "runtime",
      "runtime",
      runtimeCapabilityIds
    );
    const label = readStringField(item, "label");
    const confidence = readFiniteNumberField(item, "confidence");
    if (!runtime || !label || confidence === null) {
      return [];
    }
    return [{ runtime, label, confidence }];
  });
}

function readRuntimeCompatibilityAssessmentSummaryList(
  value: unknown
): RuntimeCompatibilityAssessmentSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const assessment = readRuntimeCompatibilityAssessmentSummary(item);
    return assessment ? [assessment] : [];
  });
}

function readRuntimeCompatibilityAssessmentSummary(
  value: unknown
): RuntimeCompatibilityAssessmentSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const runtime = readStringUnion(
    value,
    "runtime",
    "runtime",
    runtimeCapabilityIds
  );
  const label = readStringField(value, "label");
  const compatibility = readStringUnion(
    value,
    "compatibility",
    "compatibility",
    runtimeCompatibilityLevels
  );
  const confidence = readFiniteNumberField(value, "confidence");
  const expectedImplementationComplexity = readStringUnion(
    value,
    "expected_implementation_complexity",
    "expectedImplementationComplexity",
    runtimeCapabilityComplexities
  );
  const portability = readStringUnion(
    value,
    "portability",
    "portability",
    runtimePortabilities
  );
  const interoperability = readStringUnion(
    value,
    "interoperability",
    "interoperability",
    runtimeInteroperabilities
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !runtime ||
    !label ||
    !compatibility ||
    confidence === null ||
    !expectedImplementationComplexity ||
    !portability ||
    !interoperability ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    runtime,
    label,
    compatibility,
    confidence,
    compatibilityReasons: readStringListField(
      value,
      "compatibility_reasons",
      "compatibilityReasons"
    ),
    runtimeRequirements: readStringListField(
      value,
      "runtime_requirements",
      "runtimeRequirements"
    ),
    runtimeLimitations: readStringListField(
      value,
      "runtime_limitations",
      "runtimeLimitations"
    ),
    dependencyCompatibility: readStringListField(
      value,
      "dependency_compatibility",
      "dependencyCompatibility"
    ),
    expectedImplementationComplexity,
    portability,
    interoperability,
    implementationRisks: readStringListField(
      value,
      "implementation_risks",
      "implementationRisks"
    ),
    promptGuidance,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactCapabilityMatrixSummary(
  value: unknown
): ArtifactCapabilityMatrixSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const capabilityProfiles = readArtifactCapabilityProfileSummaryList(
    value.capability_profiles ?? value.capabilityProfiles
  );
  const capabilityConfidence = readArtifactCapabilityConfidenceSummaryList(
    value.capability_confidence ?? value.capabilityConfidence
  );
  const artifactFit = readArtifactCapabilityFit(value, "artifact_fit", "artifactFit");
  const creativeFit = readArtifactCapabilityFit(value, "creative_fit", "creativeFit");
  const generativeFit = readArtifactCapabilityFit(
    value,
    "generative_fit",
    "generativeFit"
  );
  const interactionFit = readArtifactCapabilityFit(
    value,
    "interaction_fit",
    "interactionFit"
  );
  const audiovisualFit = readArtifactCapabilityFit(
    value,
    "audiovisual_fit",
    "audiovisualFit"
  );
  const exportFit = readArtifactCapabilityFit(value, "export_fit", "exportFit");
  const interoperabilityFit = readArtifactCapabilityFit(
    value,
    "interoperability_fit",
    "interoperabilityFit"
  );
  const portabilityFit = readArtifactCapabilityFit(
    value,
    "portability_fit",
    "portabilityFit"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_capability_matrix" ||
    capabilityProfiles.length === 0 ||
    !artifactFit ||
    !creativeFit ||
    !generativeFit ||
    !interactionFit ||
    !audiovisualFit ||
    !exportFit ||
    !interoperabilityFit ||
    !portabilityFit ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    capabilityProfiles,
    strongestTargets: readRuntimeCapabilityIdList(
      value.strongest_targets ?? value.strongestTargets
    ),
    weakestTargets: readRuntimeCapabilityIdList(
      value.weakest_targets ?? value.weakestTargets
    ),
    targetStrengths: readStringListField(
      value,
      "target_strengths",
      "targetStrengths"
    ),
    targetWeaknesses: readStringListField(
      value,
      "target_weaknesses",
      "targetWeaknesses"
    ),
    unsupportedOrRiskyCapabilities: readStringListField(
      value,
      "unsupported_or_risky_capabilities",
      "unsupportedOrRiskyCapabilities"
    ),
    capabilityConfidence,
    artifactFit,
    creativeFit,
    generativeFit,
    interactionFit,
    audiovisualFit,
    exportFit,
    interoperabilityFit,
    portabilityFit,
    missingCapabilityInformation: readStringListField(
      value,
      "missing_capability_information",
      "missingCapabilityInformation"
    ),
    capabilityRisks: readStringListField(
      value,
      "capability_risks",
      "capabilityRisks"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readMultiArtifactStrategySummary(
  value: unknown
): MultiArtifactStrategySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const artifactStrategySummary =
    readStringField(value, "artifact_strategy_summary") ??
    readStringField(value, "artifactStrategySummary");
  const primaryArtifact = readMultiArtifactStrategyArtifactSummary(
    value.primary_artifact ?? value.primaryArtifact
  );
  const supportingArtifacts = readMultiArtifactStrategyArtifactSummaryList(
    value.supporting_artifacts ?? value.supportingArtifacts
  );
  const artifactSequence = readMultiArtifactStrategySequenceStepSummaryList(
    value.artifact_sequence ?? value.artifactSequence
  );
  const artifactPriority = readMultiArtifactStrategyPriorityEntrySummaryList(
    value.artifact_priority ?? value.artifactPriority
  );
  const artifactGrouping = readMultiArtifactStrategyGroupSummaryList(
    value.artifact_grouping ?? value.artifactGrouping
  );
  const combinationMode = readStringUnion(
    value,
    "combination_mode",
    "combinationMode",
    multiArtifactStrategyCombinationModes
  );
  const artifactSeparationStrategy = readStringListField(
    value,
    "artifact_separation_strategy",
    "artifactSeparationStrategy"
  );
  const artifactCombinationStrategy = readStringListField(
    value,
    "artifact_combination_strategy",
    "artifactCombinationStrategy"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "multi_artifact_strategy" ||
    !artifactStrategySummary ||
    primaryArtifact === null ||
    artifactSequence.length === 0 ||
    artifactPriority.length === 0 ||
    artifactGrouping.length === 0 ||
    artifactSeparationStrategy.length === 0 ||
    artifactCombinationStrategy.length === 0 ||
    !combinationMode ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    artifactStrategySummary,
    primaryArtifact,
    supportingArtifacts,
    artifactSequence,
    artifactPriority,
    artifactGrouping,
    artifactSeparationStrategy,
    artifactCombinationStrategy,
    artifactDependencyOrder: readStringListField(
      value,
      "artifact_dependency_order",
      "artifactDependencyOrder"
    ),
    artifactHandoffPoints: readStringListField(
      value,
      "artifact_handoff_points",
      "artifactHandoffPoints"
    ),
    runtimeAwareArtifactStrategy: readStringListField(
      value,
      "runtime_aware_artifact_strategy",
      "runtimeAwareArtifactStrategy"
    ),
    capabilityAwareArtifactStrategy: readStringListField(
      value,
      "capability_aware_artifact_strategy",
      "capabilityAwareArtifactStrategy"
    ),
    combinationMode,
    riskAreas: readStringListField(value, "risk_areas", "riskAreas"),
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readMultiArtifactStrategyArtifactSummaryList(
  value: unknown
): MultiArtifactStrategyArtifactSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const artifact = readMultiArtifactStrategyArtifactSummary(item);
    return artifact ? [artifact] : [];
  });
}

function readMultiArtifactStrategyArtifactSummary(
  value: unknown
): MultiArtifactStrategyArtifactSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const artifactId =
    readStringField(value, "artifact_id") ?? readStringField(value, "artifactId");
  const title = readStringField(value, "title");
  const role = readStringUnion(
    value,
    "role",
    "role",
    multiArtifactStrategyRoles
  );
  const artifactType = readStringUnion(
    value,
    "artifact_type",
    "artifactType",
    artifactTypes
  );
  const artifactFamily = readStringUnion(
    value,
    "artifact_family",
    "artifactFamily",
    artifactFamilies
  );
  const priority = readStringUnion(
    value,
    "priority",
    "priority",
    multiArtifactStrategyPriorities
  );
  const purpose = readStringField(value, "purpose");

  if (
    !artifactId ||
    !title ||
    !role ||
    !artifactType ||
    !artifactFamily ||
    !priority ||
    !purpose
  ) {
    return null;
  }

  return {
    artifactId,
    title,
    role,
    artifactType,
    artifactFamily,
    priority,
    purpose,
    runtimeTargets: readRuntimeCapabilityIdList(
      value.runtime_targets ?? value.runtimeTargets
    ),
    capabilityTargets: readRuntimeCapabilityIdList(
      value.capability_targets ?? value.capabilityTargets
    ),
    dependsOn: readStringListField(value, "depends_on", "dependsOn"),
    handoffPoints: readStringListField(
      value,
      "handoff_points",
      "handoffPoints"
    ),
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readMultiArtifactStrategySequenceStepSummaryList(
  value: unknown
): MultiArtifactStrategySequenceStepSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const step = readMultiArtifactStrategySequenceStepSummary(item);
    return step ? [step] : [];
  });
}

function readMultiArtifactStrategySequenceStepSummary(
  value: unknown
): MultiArtifactStrategySequenceStepSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const stepId =
    readStringField(value, "step_id") ?? readStringField(value, "stepId");
  const order = readFiniteNumberField(value, "order");
  const artifactId =
    readStringField(value, "artifact_id") ?? readStringField(value, "artifactId");
  const action = readStringUnion(
    value,
    "action",
    "action",
    multiArtifactStrategyActions
  );
  const rationale = readStringField(value, "rationale");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !stepId ||
    order === null ||
    !artifactId ||
    !action ||
    !rationale ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    stepId,
    order,
    artifactId,
    action,
    rationale,
    dependsOn: readStringListField(value, "depends_on", "dependsOn"),
    promptGuidance
  };
}

function readMultiArtifactStrategyPriorityEntrySummaryList(
  value: unknown
): MultiArtifactStrategyPriorityEntrySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }
    const artifactId =
      readStringField(item, "artifact_id") ??
      readStringField(item, "artifactId");
    const priority = readStringUnion(
      item,
      "priority",
      "priority",
      multiArtifactStrategyPriorities
    );
    const rationale = readStringField(item, "rationale");
    if (!artifactId || !priority || !rationale) {
      return [];
    }
    return [{ artifactId, priority, rationale }];
  });
}

function readMultiArtifactStrategyGroupSummaryList(
  value: unknown
): MultiArtifactStrategyGroupSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }
    const groupId =
      readStringField(item, "group_id") ?? readStringField(item, "groupId");
    const label = readStringField(item, "label");
    const artifactIds = readStringListField(
      item,
      "artifact_ids",
      "artifactIds"
    );
    const groupingRationale =
      readStringField(item, "grouping_rationale") ??
      readStringField(item, "groupingRationale");
    const separationRationale =
      readStringField(item, "separation_rationale") ??
      readStringField(item, "separationRationale");
    if (
      !groupId ||
      !label ||
      artifactIds.length === 0 ||
      !groupingRationale ||
      !separationRationale
    ) {
      return [];
    }
    return [
      {
        groupId,
        label,
        artifactIds,
        groupingRationale,
        separationRationale
      }
    ];
  });
}

export function readArtifactCriticSummary(
  value: unknown
): ArtifactCriticSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const critiqueConfidence =
    readFiniteNumberField(value, "critique_confidence") ??
    readFiniteNumberField(value, "critiqueConfidence");
  const critiqueSummary =
    readStringField(value, "critique_summary") ??
    readStringField(value, "critiqueSummary");
  const riskAssessment = readStringUnion(
    value,
    "risk_assessment",
    "riskAssessment",
    artifactCriticRiskAssessments
  );
  const strengths = readStringListField(value, "strengths", "strengths");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_critic" ||
    critiqueConfidence === null ||
    !critiqueSummary ||
    !riskAssessment ||
    strengths.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    critiqueConfidence,
    critiqueSummary,
    strengths,
    weaknesses: readStringListField(value, "weaknesses", "weaknesses"),
    capabilityGaps: readStringListField(
      value,
      "capability_gaps",
      "capabilityGaps"
    ),
    dependencyConcerns: readStringListField(
      value,
      "dependency_concerns",
      "dependencyConcerns"
    ),
    runtimeConcerns: readStringListField(
      value,
      "runtime_concerns",
      "runtimeConcerns"
    ),
    scalabilityConcerns: readStringListField(
      value,
      "scalability_concerns",
      "scalabilityConcerns"
    ),
    maintainabilityConcerns: readStringListField(
      value,
      "maintainability_concerns",
      "maintainabilityConcerns"
    ),
    complexityConcerns: readStringListField(
      value,
      "complexity_concerns",
      "complexityConcerns"
    ),
    riskAssessment,
    unsupportedAssumptions: readStringListField(
      value,
      "unsupported_assumptions",
      "unsupportedAssumptions"
    ),
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    openQuestions: readStringListField(
      value,
      "open_questions",
      "openQuestions"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    improvementOpportunities: readStringListField(
      value,
      "improvement_opportunities",
      "improvementOpportunities"
    ),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readCreativeCriticSummary(
  value: unknown
): CreativeCriticSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const criticConfidence =
    readFiniteNumberField(value, "critic_confidence") ??
    readFiniteNumberField(value, "criticConfidence");
  const critiqueSummary =
    readStringField(value, "critique_summary") ??
    readStringField(value, "critiqueSummary");
  const riskAssessment = readStringUnion(
    value,
    "risk_assessment",
    "riskAssessment",
    creativeCriticRiskAssessments
  );
  const creativeStrengths = readStringListField(
    value,
    "creative_strengths",
    "creativeStrengths"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const conceptQuality =
    readFiniteNumberField(value, "concept_quality") ??
    readFiniteNumberField(value, "conceptQuality");
  const executionQuality =
    readFiniteNumberField(value, "execution_quality") ??
    readFiniteNumberField(value, "executionQuality");
  const artifactQuality =
    readFiniteNumberField(value, "artifact_quality") ??
    readFiniteNumberField(value, "artifactQuality");
  const coherenceQuality =
    readFiniteNumberField(value, "coherence_quality") ??
    readFiniteNumberField(value, "coherenceQuality");
  const runtimeFitQuality =
    readFiniteNumberField(value, "runtime_fit_quality") ??
    readFiniteNumberField(value, "runtimeFitQuality");
  const originalityQuality =
    readFiniteNumberField(value, "originality_quality") ??
    readFiniteNumberField(value, "originalityQuality");
  const clarityQuality =
    readFiniteNumberField(value, "clarity_quality") ??
    readFiniteNumberField(value, "clarityQuality");
  const feasibilityQuality =
    readFiniteNumberField(value, "feasibility_quality") ??
    readFiniteNumberField(value, "feasibilityQuality");

  if (
    role !== "creative_critic_engine" ||
    criticConfidence === null ||
    !critiqueSummary ||
    creativeStrengths.length === 0 ||
    conceptQuality === null ||
    executionQuality === null ||
    artifactQuality === null ||
    coherenceQuality === null ||
    runtimeFitQuality === null ||
    originalityQuality === null ||
    clarityQuality === null ||
    feasibilityQuality === null ||
    !riskAssessment ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    criticConfidence,
    critiqueSummary,
    creativeStrengths,
    creativeWeaknesses: readStringListField(
      value,
      "creative_weaknesses",
      "creativeWeaknesses"
    ),
    conceptQuality,
    executionQuality,
    artifactQuality,
    coherenceQuality,
    runtimeFitQuality,
    originalityQuality,
    clarityQuality,
    feasibilityQuality,
    riskAssessment,
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    unsupportedAssumptions: readStringListField(
      value,
      "unsupported_assumptions",
      "unsupportedAssumptions"
    ),
    improvementOpportunities: readStringListField(
      value,
      "improvement_opportunities",
      "improvementOpportunities"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const selfEvaluationCompletenessValues = [
  "complete",
  "mostly_complete",
  "partial",
  "blocked"
] as const satisfies readonly SelfEvaluationCompleteness[];

const selfEvaluationRiskValues = [
  "low",
  "medium",
  "high"
] as const satisfies readonly SelfEvaluationRisk[];

const selfEvaluationAmbiguityValues = [
  "low",
  "medium",
  "high"
] as const satisfies readonly SelfEvaluationAmbiguity[];

export function readSelfEvaluationSummary(
  value: unknown
): SelfEvaluationSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const selfEvaluationConfidence =
    readFiniteNumberField(value, "self_evaluation_confidence") ??
    readFiniteNumberField(value, "selfEvaluationConfidence");
  const evaluationSummary =
    readStringField(value, "evaluation_summary") ??
    readStringField(value, "evaluationSummary");
  const requestAlignment =
    readFiniteNumberField(value, "request_alignment") ??
    readFiniteNumberField(value, "requestAlignment");
  const intentAlignment =
    readFiniteNumberField(value, "intent_alignment") ??
    readFiniteNumberField(value, "intentAlignment");
  const constraintAlignment =
    readFiniteNumberField(value, "constraint_alignment") ??
    readFiniteNumberField(value, "constraintAlignment");
  const artifactAlignment =
    readFiniteNumberField(value, "artifact_alignment") ??
    readFiniteNumberField(value, "artifactAlignment");
  const runtimeAlignment =
    readFiniteNumberField(value, "runtime_alignment") ??
    readFiniteNumberField(value, "runtimeAlignment");
  const creativeCoherence =
    readFiniteNumberField(value, "creative_coherence") ??
    readFiniteNumberField(value, "creativeCoherence");
  const technicalCoherence =
    readFiniteNumberField(value, "technical_coherence") ??
    readFiniteNumberField(value, "technicalCoherence");
  const completenessAssessment =
    readStringUnion(
      value,
      "completeness_assessment",
      "completenessAssessment",
      selfEvaluationCompletenessValues
    );
  const ambiguityAssessment =
    readStringUnion(
      value,
      "ambiguity_assessment",
      "ambiguityAssessment",
      selfEvaluationAmbiguityValues
    );
  const hallucinationRisk =
    readStringUnion(
      value,
      "hallucination_risk",
      "hallucinationRisk",
      selfEvaluationRiskValues
    );
  const overreachRisk =
    readStringUnion(
      value,
      "overreach_risk",
      "overreachRisk",
      selfEvaluationRiskValues
    );
  const underdeliveryRisk =
    readStringUnion(
      value,
      "underdelivery_risk",
      "underdeliveryRisk",
      selfEvaluationRiskValues
    );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "self_evaluation_engine" ||
    selfEvaluationConfidence === null ||
    !evaluationSummary ||
    requestAlignment === null ||
    intentAlignment === null ||
    constraintAlignment === null ||
    artifactAlignment === null ||
    runtimeAlignment === null ||
    creativeCoherence === null ||
    technicalCoherence === null ||
    !completenessAssessment ||
    !ambiguityAssessment ||
    !hallucinationRisk ||
    !overreachRisk ||
    !underdeliveryRisk ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    selfEvaluationConfidence,
    evaluationSummary,
    requestAlignment,
    intentAlignment,
    constraintAlignment,
    artifactAlignment,
    runtimeAlignment,
    creativeCoherence,
    technicalCoherence,
    completenessAssessment,
    ambiguityAssessment,
    hallucinationRisk,
    overreachRisk,
    underdeliveryRisk,
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    unsupportedAssumptions: readStringListField(
      value,
      "unsupported_assumptions",
      "unsupportedAssumptions"
    ),
    qualityGaps: readStringListField(value, "quality_gaps", "qualityGaps"),
    improvementOpportunities: readStringListField(
      value,
      "improvement_opportunities",
      "improvementOpportunities"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const improvementPriorityLevels = [
  "critical",
  "high",
  "medium",
  "low"
] as const satisfies readonly ImprovementPriorityLevel[];

const improvementImpacts = [
  "high",
  "medium",
  "low"
] as const satisfies readonly ImprovementImpact[];

const improvementRisks = [
  "low",
  "medium",
  "high"
] as const satisfies readonly ImprovementRisk[];

const improvementSources = [
  "creative_critic",
  "self_evaluation",
  "artifact_context",
  "workflow_context"
] as const satisfies readonly ImprovementSource[];

export function readCreativeImprovementPlannerSummary(
  value: unknown
): CreativeImprovementPlannerSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const confidence = readFiniteNumberField(value, "confidence");
  const improvementSummary =
    readStringField(value, "improvement_summary") ??
    readStringField(value, "improvementSummary");
  const improvementPriorities = readCreativeImprovementPrioritySummaryList(
    value.improvement_priorities ?? value.improvementPriorities
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_improvement_planner" ||
    serializationVersion !== "v1" ||
    confidence === null ||
    !improvementSummary ||
    improvementPriorities.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    confidence,
    improvementSummary,
    improvementPriorities,
    highestImpactOpportunities: readStringListField(
      value,
      "highest_impact_opportunities",
      "highestImpactOpportunities"
    ),
    lowRiskImprovements: readStringListField(
      value,
      "low_risk_improvements",
      "lowRiskImprovements"
    ),
    experimentalImprovements: readStringListField(
      value,
      "experimental_improvements",
      "experimentalImprovements"
    ),
    tradeOffRecommendations: readStringListField(
      value,
      "trade_off_recommendations",
      "tradeOffRecommendations"
    ),
    improvementRationale: readStringListField(
      value,
      "improvement_rationale",
      "improvementRationale"
    ),
    evidence: readStringListField(value, "evidence", "evidence"),
    futureRefinementCandidates: readStringListField(
      value,
      "future_refinement_candidates",
      "futureRefinementCandidates"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary
  };
}

function readCreativeImprovementPrioritySummaryList(
  value: unknown
): CreativeImprovementPrioritySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const priorityId =
      readStringField(item, "priority_id") ?? readStringField(item, "priorityId");
    const title = readStringField(item, "title");
    const priority = readStringUnion(
      item,
      "priority",
      "priority",
      improvementPriorityLevels
    );
    const impact = readStringUnion(item, "impact", "impact", improvementImpacts);
    const risk = readStringUnion(item, "risk", "risk", improvementRisks);
    const source = readStringUnion(item, "source", "source", improvementSources);
    const rationale = readStringField(item, "rationale");

    if (
      !priorityId ||
      !title ||
      !priority ||
      !impact ||
      !risk ||
      !source ||
      !rationale
    ) {
      return [];
    }

    return [
      {
        priorityId,
        title,
        priority,
        impact,
        risk,
        source,
        rationale,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

const reflectionLoopPriorities = [
  "critical",
  "high",
  "medium",
  "low",
  "none"
] as const satisfies readonly ReflectionLoopPriority[];

const reflectionLoopDepths = [
  "none",
  "light",
  "moderate",
  "deep"
] as const satisfies readonly ReflectionLoopDepth[];

const reflectionLoopEstimates = [
  "none",
  "low",
  "medium",
  "high"
] as const satisfies readonly ReflectionLoopEstimate[];

const reflectionLoopHitlRecommendations = [
  "not_needed",
  "optional",
  "recommended",
  "required"
] as const satisfies readonly ReflectionLoopHitlRecommendation[];

export function readReflectionLoopSummary(
  value: unknown
): ReflectionLoopSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const reflectionConfidence =
    readFiniteNumberField(value, "reflection_confidence") ??
    readFiniteNumberField(value, "reflectionConfidence");
  const reflectionSummary =
    readStringField(value, "reflection_summary") ??
    readStringField(value, "reflectionSummary");
  const reflectionRequired =
    readBooleanField(value, "reflection_required") ??
    readBooleanField(value, "reflectionRequired");
  const reflectionPriority = readStringUnion(
    value,
    "reflection_priority",
    "reflectionPriority",
    reflectionLoopPriorities
  );
  const reflectionDepth = readStringUnion(
    value,
    "reflection_depth",
    "reflectionDepth",
    reflectionLoopDepths
  );
  const expectedQualityGain = readStringUnion(
    value,
    "expected_quality_gain",
    "expectedQualityGain",
    reflectionLoopEstimates
  );
  const expectedRiskReduction = readStringUnion(
    value,
    "expected_risk_reduction",
    "expectedRiskReduction",
    reflectionLoopEstimates
  );
  const expectedCost = readStringUnion(
    value,
    "expected_cost",
    "expectedCost",
    reflectionLoopEstimates
  );
  const expectedLatency = readStringUnion(
    value,
    "expected_latency",
    "expectedLatency",
    reflectionLoopEstimates
  );
  const confidenceAfterReflection =
    readFiniteNumberField(value, "confidence_after_reflection") ??
    readFiniteNumberField(value, "confidenceAfterReflection");
  const hitlRecommendation = readStringUnion(
    value,
    "hitl_recommendation",
    "hitlRecommendation",
    reflectionLoopHitlRecommendations
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "reflection_loop_engine" ||
    serializationVersion !== "v1" ||
    reflectionConfidence === null ||
    !reflectionSummary ||
    reflectionRequired === null ||
    !reflectionPriority ||
    !reflectionDepth ||
    !expectedQualityGain ||
    !expectedRiskReduction ||
    !expectedCost ||
    !expectedLatency ||
    confidenceAfterReflection === null ||
    !hitlRecommendation ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    reflectionConfidence,
    reflectionSummary,
    reflectionRequired,
    reflectionPriority,
    reflectionRationale: readStringListField(
      value,
      "reflection_rationale",
      "reflectionRationale"
    ),
    reflectionDepth,
    expectedQualityGain,
    expectedRiskReduction,
    expectedCost,
    expectedLatency,
    confidenceAfterReflection,
    unresolvedQuestions: readStringListField(
      value,
      "unresolved_questions",
      "unresolvedQuestions"
    ),
    refinementCandidates: readStringListField(
      value,
      "refinement_candidates",
      "refinementCandidates"
    ),
    stopConditions: readStringListField(
      value,
      "stop_conditions",
      "stopConditions"
    ),
    hitlRecommendation,
    promptGuidance,
    evidence: readStringListField(value, "evidence", "evidence"),
    authorityBoundary
  };
}

const creativeConfidenceLevels = [
  "very_high",
  "high",
  "medium",
  "low",
  "critical"
] as const satisfies readonly CreativeConfidenceLevel[];

const creativeConfidenceComponentSources = [
  "creative_critic",
  "self_evaluation",
  "creative_improvement_planner",
  "reflection_loop",
  "planning_metadata"
] as const satisfies readonly CreativeConfidenceComponentSource[];

const expectedOutputReliabilities = [
  "very_high",
  "high",
  "medium",
  "low",
  "blocked"
] as const satisfies readonly ExpectedOutputReliability[];

const expectedExecutionReadinesses = [
  "ready",
  "needs_caveats",
  "needs_hitl",
  "blocked"
] as const satisfies readonly ExpectedExecutionReadiness[];

const expectedHumanReviewNeeds = [
  "not_needed",
  "optional",
  "recommended",
  "required"
] as const satisfies readonly ExpectedHumanReviewNeed[];

const escalationRecommendations = [
  "none",
  "monitor",
  "hitl_review",
  "future_escalation"
] as const satisfies readonly EscalationRecommendation[];

const confidenceTrends = [
  "improving",
  "stable",
  "declining",
  "conflicting",
  "unknown"
] as const satisfies readonly ConfidenceTrend[];

function readCreativeConfidenceComponentSummaryList(
  value: unknown
): CreativeConfidenceComponentSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      creativeConfidenceComponentSources
    );
    const score = readFiniteNumberField(item, "score");
    const weight = readFiniteNumberField(item, "weight");
    const rationale = readStringField(item, "rationale");

    if (!source || score === null || weight === null || !rationale) {
      return [];
    }

    return [
      {
        source,
        score,
        weight,
        rationale,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readCreativeConfidenceSummary(
  value: unknown
): CreativeConfidenceSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const confidenceScore =
    readFiniteNumberField(value, "confidence_score") ??
    readFiniteNumberField(value, "confidenceScore");
  const confidenceLevel = readStringUnion(
    value,
    "confidence_level",
    "confidenceLevel",
    creativeConfidenceLevels
  );
  const confidenceSummary =
    readStringField(value, "confidence_summary") ??
    readStringField(value, "confidenceSummary");
  const confidenceComponents = readCreativeConfidenceComponentSummaryList(
    value.confidence_components ?? value.confidenceComponents
  );
  const expectedOutputReliability = readStringUnion(
    value,
    "expected_output_reliability",
    "expectedOutputReliability",
    expectedOutputReliabilities
  );
  const expectedExecutionReadiness = readStringUnion(
    value,
    "expected_execution_readiness",
    "expectedExecutionReadiness",
    expectedExecutionReadinesses
  );
  const expectedHumanReviewNeed = readStringUnion(
    value,
    "expected_human_review_need",
    "expectedHumanReviewNeed",
    expectedHumanReviewNeeds
  );
  const escalationRecommendation = readStringUnion(
    value,
    "escalation_recommendation",
    "escalationRecommendation",
    escalationRecommendations
  );
  const confidenceTrend = readStringUnion(
    value,
    "confidence_trend",
    "confidenceTrend",
    confidenceTrends
  );
  const hitlRecommendation = readStringUnion(
    value,
    "hitl_recommendation",
    "hitlRecommendation",
    expectedHumanReviewNeeds
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_confidence_engine" ||
    serializationVersion !== "v1" ||
    confidenceScore === null ||
    !confidenceLevel ||
    !confidenceSummary ||
    confidenceComponents.length === 0 ||
    !expectedOutputReliability ||
    !expectedExecutionReadiness ||
    !expectedHumanReviewNeed ||
    !escalationRecommendation ||
    !confidenceTrend ||
    !hitlRecommendation ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    confidenceScore,
    confidenceLevel,
    confidenceSummary,
    confidenceRationale: readStringListField(
      value,
      "confidence_rationale",
      "confidenceRationale"
    ),
    confidenceComponents,
    confidenceLimitations: readStringListField(
      value,
      "confidence_limitations",
      "confidenceLimitations"
    ),
    confidenceUncertainties: readStringListField(
      value,
      "confidence_uncertainties",
      "confidenceUncertainties"
    ),
    confidenceStrengths: readStringListField(
      value,
      "confidence_strengths",
      "confidenceStrengths"
    ),
    confidenceWeaknesses: readStringListField(
      value,
      "confidence_weaknesses",
      "confidenceWeaknesses"
    ),
    expectedOutputReliability,
    expectedExecutionReadiness,
    expectedHumanReviewNeed,
    escalationRecommendation,
    confidenceTrend,
    confidenceEvidence: readStringListField(
      value,
      "confidence_evidence",
      "confidenceEvidence"
    ),
    hitlRecommendation,
    promptGuidance,
    authorityBoundary
  };
}

const creativeScoreDimensions = [
  "creativity",
  "technical",
  "coherence",
  "feasibility",
  "artifact",
  "runtime"
] as const satisfies readonly CreativeScoreDimension[];

const creativeScoreBands = [
  "excellent",
  "strong",
  "solid",
  "weak",
  "critical"
] as const satisfies readonly CreativeScoreBand[];

const creativeScoreSignalSources = [
  "creative_critic",
  "self_evaluation",
  "creative_improvement_planner",
  "reflection_loop",
  "creative_confidence",
  "planning_metadata"
] as const satisfies readonly CreativeScoreSignalSource[];

const consistencyValidationStatuses = [
  "consistent",
  "needs_attention",
  "inconsistent",
  "insufficient_evidence"
] as const satisfies readonly ConsistencyValidationStatus[];

const consistencyValidationCheckStatuses = [
  "aligned",
  "watch",
  "conflict",
  "missing"
] as const satisfies readonly ConsistencyValidationCheckStatus[];

const consistencyValidationContradictionLevels = [
  "none",
  "low",
  "medium",
  "high"
] as const satisfies readonly ConsistencyValidationContradictionLevel[];

const consistencyValidationAmbiguityLevels = [
  "low",
  "medium",
  "high"
] as const satisfies readonly ConsistencyValidationAmbiguityLevel[];

const consistencyValidationIntegrityLevels = [
  "strong",
  "adequate",
  "fragile",
  "compromised"
] as const satisfies readonly ConsistencyValidationIntegrity[];

const evaluationReportSources = [
  "creative_critic",
  "self_evaluation",
  "creative_improvement_planner",
  "reflection_loop",
  "creative_confidence",
  "creative_score",
  "consistency_validation"
] as const satisfies readonly EvaluationReportSource[];

const evaluationReportDependencyStatuses = [
  "available",
  "missing"
] as const satisfies readonly EvaluationReportDependencyStatus[];

function readCreativeScoreBreakdownSummaryList(
  value: unknown
): CreativeScoreBreakdownSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const dimension = readStringUnion(
      item,
      "dimension",
      "dimension",
      creativeScoreDimensions
    );
    const score = readFiniteNumberField(item, "score");
    const weight = readFiniteNumberField(item, "weight");
    const rationale = readStringField(item, "rationale");

    if (!dimension || score === null || weight === null || !rationale) {
      return [];
    }

    return [
      {
        dimension,
        score,
        weight,
        rationale,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCreativeScoreComponentSummaryList(
  value: unknown
): CreativeScoreComponentSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      creativeScoreSignalSources
    );
    const score = readFiniteNumberField(item, "score");
    const weight = readFiniteNumberField(item, "weight");
    const weightedContribution =
      readFiniteNumberField(item, "weighted_contribution") ??
      readFiniteNumberField(item, "weightedContribution");
    const rationale = readStringField(item, "rationale");

    if (
      !source ||
      score === null ||
      weight === null ||
      weightedContribution === null ||
      !rationale
    ) {
      return [];
    }

    return [
      {
        source,
        score,
        weight,
        weightedContribution,
        rationale,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readCreativeScoreSummary(
  value: unknown
): CreativeScoreSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const overallCreativeScore =
    readFiniteNumberField(value, "overall_creative_score") ??
    readFiniteNumberField(value, "overallCreativeScore");
  const scoreBand = readStringUnion(
    value,
    "score_band",
    "scoreBand",
    creativeScoreBands
  );
  const scoreSummary =
    readStringField(value, "score_summary") ??
    readStringField(value, "scoreSummary");
  const scoreBreakdown = readCreativeScoreBreakdownSummaryList(
    value.score_breakdown ?? value.scoreBreakdown
  );
  const scoreComponents = readCreativeScoreComponentSummaryList(
    value.score_components ?? value.scoreComponents
  );
  const creativityScore =
    readFiniteNumberField(value, "creativity_score") ??
    readFiniteNumberField(value, "creativityScore");
  const technicalScore =
    readFiniteNumberField(value, "technical_score") ??
    readFiniteNumberField(value, "technicalScore");
  const coherenceScore =
    readFiniteNumberField(value, "coherence_score") ??
    readFiniteNumberField(value, "coherenceScore");
  const feasibilityScore =
    readFiniteNumberField(value, "feasibility_score") ??
    readFiniteNumberField(value, "feasibilityScore");
  const artifactScore =
    readFiniteNumberField(value, "artifact_score") ??
    readFiniteNumberField(value, "artifactScore");
  const runtimeScore =
    readFiniteNumberField(value, "runtime_score") ??
    readFiniteNumberField(value, "runtimeScore");
  const confidenceWeight =
    readFiniteNumberField(value, "confidence_weight") ??
    readFiniteNumberField(value, "confidenceWeight");
  const reflectionWeight =
    readFiniteNumberField(value, "reflection_weight") ??
    readFiniteNumberField(value, "reflectionWeight");
  const consistencyWeight =
    readFiniteNumberField(value, "consistency_weight") ??
    readFiniteNumberField(value, "consistencyWeight");
  const artifactWeight =
    readFiniteNumberField(value, "artifact_weight") ??
    readFiniteNumberField(value, "artifactWeight");
  const runtimeWeight =
    readFiniteNumberField(value, "runtime_weight") ??
    readFiniteNumberField(value, "runtimeWeight");
  const uncertaintyPenalty =
    readFiniteNumberField(value, "uncertainty_penalty") ??
    readFiniteNumberField(value, "uncertaintyPenalty");
  const riskPenalty =
    readFiniteNumberField(value, "risk_penalty") ??
    readFiniteNumberField(value, "riskPenalty");
  const scoreExplainability =
    readStringField(value, "score_explainability") ??
    readStringField(value, "scoreExplainability");
  const hitlRecommendation = readStringUnion(
    value,
    "hitl_recommendation",
    "hitlRecommendation",
    expectedHumanReviewNeeds
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_score_engine" ||
    serializationVersion !== "v1" ||
    overallCreativeScore === null ||
    !scoreBand ||
    !scoreSummary ||
    scoreBreakdown.length !== 6 ||
    scoreComponents.length === 0 ||
    creativityScore === null ||
    technicalScore === null ||
    coherenceScore === null ||
    feasibilityScore === null ||
    artifactScore === null ||
    runtimeScore === null ||
    confidenceWeight === null ||
    reflectionWeight === null ||
    consistencyWeight === null ||
    artifactWeight === null ||
    runtimeWeight === null ||
    uncertaintyPenalty === null ||
    riskPenalty === null ||
    !scoreExplainability ||
    !hitlRecommendation ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    overallCreativeScore,
    scoreBand,
    scoreSummary,
    scoreBreakdown,
    scoreComponents,
    creativityScore,
    technicalScore,
    coherenceScore,
    feasibilityScore,
    artifactScore,
    runtimeScore,
    confidenceWeight,
    reflectionWeight,
    consistencyWeight,
    artifactWeight,
    runtimeWeight,
    uncertaintyPenalty,
    riskPenalty,
    positiveContributions: readStringListField(
      value,
      "positive_contributions",
      "positiveContributions"
    ),
    negativeContributions: readStringListField(
      value,
      "negative_contributions",
      "negativeContributions"
    ),
    strengths: readStringListField(value, "strengths", "strengths"),
    weaknesses: readStringListField(value, "weaknesses", "weaknesses"),
    scoreRationale: readStringListField(
      value,
      "score_rationale",
      "scoreRationale"
    ),
    scoreCalibrationNotes: readStringListField(
      value,
      "score_calibration_notes",
      "scoreCalibrationNotes"
    ),
    scoreExplainability,
    scoreEvidence: readStringListField(
      value,
      "score_evidence",
      "scoreEvidence"
    ),
    hitlRecommendation,
    promptGuidance,
    authorityBoundary
  };
}

function readConsistencyValidationCheckSummary(
  value: unknown
): ConsistencyValidationCheckSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const status = readStringUnion(
    value,
    "status",
    "status",
    consistencyValidationCheckStatuses
  );
  const summary = readStringField(value, "summary");
  if (!status || !summary) {
    return null;
  }

  return {
    status,
    summary,
    evidence: readStringListField(value, "evidence", "evidence"),
    conflictSignals: readStringListField(
      value,
      "conflict_signals",
      "conflictSignals"
    )
  };
}

export function readConsistencyValidationSummary(
  value: unknown
): ConsistencyValidationSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const consistencyStatus = readStringUnion(
    value,
    "consistency_status",
    "consistencyStatus",
    consistencyValidationStatuses
  );
  const consistencySummary =
    readStringField(value, "consistency_summary") ??
    readStringField(value, "consistencySummary");
  const scoreConsistency = readConsistencyValidationCheckSummary(
    value.score_consistency ?? value.scoreConsistency
  );
  const confidenceConsistency = readConsistencyValidationCheckSummary(
    value.confidence_consistency ?? value.confidenceConsistency
  );
  const reflectionConsistency = readConsistencyValidationCheckSummary(
    value.reflection_consistency ?? value.reflectionConsistency
  );
  const criticConsistency = readConsistencyValidationCheckSummary(
    value.critic_consistency ?? value.criticConsistency
  );
  const plannerConsistency = readConsistencyValidationCheckSummary(
    value.planner_consistency ?? value.plannerConsistency
  );
  const reasoningConsistency = readConsistencyValidationCheckSummary(
    value.reasoning_consistency ?? value.reasoningConsistency
  );
  const contradictionLevel = readStringUnion(
    value,
    "contradiction_level",
    "contradictionLevel",
    consistencyValidationContradictionLevels
  );
  const ambiguityLevel = readStringUnion(
    value,
    "ambiguity_level",
    "ambiguityLevel",
    consistencyValidationAmbiguityLevels
  );
  const evaluationIntegrity = readStringUnion(
    value,
    "evaluation_integrity",
    "evaluationIntegrity",
    consistencyValidationIntegrityLevels
  );
  const hitlRecommendation = readStringUnion(
    value,
    "hitl_recommendation",
    "hitlRecommendation",
    expectedHumanReviewNeeds
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "consistency_validation_engine" ||
    serializationVersion !== "v1" ||
    !consistencyStatus ||
    !consistencySummary ||
    !scoreConsistency ||
    !confidenceConsistency ||
    !reflectionConsistency ||
    !criticConsistency ||
    !plannerConsistency ||
    !reasoningConsistency ||
    !contradictionLevel ||
    !ambiguityLevel ||
    !evaluationIntegrity ||
    !hitlRecommendation ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    consistencyStatus,
    consistencySummary,
    detectedConflicts: readStringListField(
      value,
      "detected_conflicts",
      "detectedConflicts"
    ),
    scoreConsistency,
    confidenceConsistency,
    reflectionConsistency,
    criticConsistency,
    plannerConsistency,
    reasoningConsistency,
    contradictionLevel,
    ambiguityLevel,
    unsupportedConclusions: readStringListField(
      value,
      "unsupported_conclusions",
      "unsupportedConclusions"
    ),
    evaluationIntegrity,
    evidence: readStringListField(value, "evidence", "evidence"),
    hitlRecommendation,
    promptGuidance,
    authorityBoundary
  };
}

function readEvaluationTraceEntrySummaryList(
  value: unknown
): EvaluationTraceEntrySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const step = readFiniteNumberField(item, "step");
    const source = readStringUnion(
      item,
      "source",
      "source",
      evaluationReportSources
    );
    const role = readStringField(item, "role");
    const contribution = readStringField(item, "contribution");
    if (step === null || !source || !role || !contribution) {
      return [];
    }

    return [
      {
        step,
        source,
        role,
        contribution,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEvaluationProvenanceEntrySummaryList(
  value: unknown
): EvaluationProvenanceEntrySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      evaluationReportSources
    );
    const role = readStringField(item, "role");
    const summary = readStringField(item, "summary");
    if (!source || !role || !summary) {
      return [];
    }

    return [
      {
        source,
        role,
        summary,
        confidence: readFiniteNumberField(item, "confidence"),
        hitlRecommendation: readStringUnion(
          item,
          "hitl_recommendation",
          "hitlRecommendation",
          expectedHumanReviewNeeds
        )
      }
    ];
  });
}

function readEvaluationDependencySummaryList(
  value: unknown
): EvaluationDependencySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      evaluationReportSources
    );
    const status = readStringUnion(
      item,
      "status",
      "status",
      evaluationReportDependencyStatuses
    );
    const required = readBooleanField(item, "required");
    const note = readStringField(item, "note");
    if (!source || !status || required === null || !note) {
      return [];
    }

    return [{ source, status, required, note }];
  });
}

function readEvaluationEvidenceLinkSummaryList(
  value: unknown
): EvaluationEvidenceLinkSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      evaluationReportSources
    );
    const claim = readStringField(item, "claim");
    if (!source || !claim) {
      return [];
    }

    return [
      {
        source,
        claim,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readEvaluationReportSummary(
  value: unknown
): EvaluationReportSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const executiveSummary =
    readStringField(value, "executive_summary") ??
    readStringField(value, "executiveSummary");
  const qualitySummary =
    readStringField(value, "quality_summary") ??
    readStringField(value, "qualitySummary");
  const confidenceSummary =
    readStringField(value, "confidence_summary") ??
    readStringField(value, "confidenceSummary");
  const consistencySummary =
    readStringField(value, "consistency_summary") ??
    readStringField(value, "consistencySummary");
  const improvementSummary =
    readStringField(value, "improvement_summary") ??
    readStringField(value, "improvementSummary");
  const scoreSummary =
    readStringField(value, "score_summary") ??
    readStringField(value, "scoreSummary");
  const hitlRecommendation = readStringUnion(
    value,
    "hitl_recommendation",
    "hitlRecommendation",
    expectedHumanReviewNeeds
  );
  const evaluationTrace = readEvaluationTraceEntrySummaryList(
    value.evaluation_trace ?? value.evaluationTrace
  );
  const evaluationProvenance = readEvaluationProvenanceEntrySummaryList(
    value.evaluation_provenance ?? value.evaluationProvenance
  );
  const evaluationDependencies = readEvaluationDependencySummaryList(
    value.evaluation_dependencies ?? value.evaluationDependencies
  );
  const evidenceChain = readEvaluationEvidenceLinkSummaryList(
    value.evidence_chain ?? value.evidenceChain
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "evaluation_reports" ||
    serializationVersion !== "v1" ||
    !executiveSummary ||
    !qualitySummary ||
    !confidenceSummary ||
    !consistencySummary ||
    !improvementSummary ||
    !scoreSummary ||
    !hitlRecommendation ||
    evaluationTrace.length === 0 ||
    evaluationProvenance.length === 0 ||
    evaluationDependencies.length === 0 ||
    evidenceChain.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    serializationVersion,
    executiveSummary,
    qualitySummary,
    confidenceSummary,
    consistencySummary,
    improvementSummary,
    scoreSummary,
    strengths: readStringListField(value, "strengths", "strengths"),
    weaknesses: readStringListField(value, "weaknesses", "weaknesses"),
    risks: readStringListField(value, "risks", "risks"),
    recommendations: readStringListField(
      value,
      "recommendations",
      "recommendations"
    ),
    hitlRecommendation,
    evaluationTrace,
    evaluationProvenance,
    evaluationExplainability: readStringListField(
      value,
      "evaluation_explainability",
      "evaluationExplainability"
    ),
    evaluationDependencies,
    evidenceChain,
    promptGuidance,
    authorityBoundary
  };
}

export function readArtifactRefinerSummary(
  value: unknown
): ArtifactRefinerSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const refinementConfidence =
    readFiniteNumberField(value, "refinement_confidence") ??
    readFiniteNumberField(value, "refinementConfidence");
  const refinementSummary =
    readStringField(value, "refinement_summary") ??
    readStringField(value, "refinementSummary");
  const recommendedImprovements = readStringListField(
    value,
    "recommended_improvements",
    "recommendedImprovements"
  );
  const priorityImprovements = readStringListField(
    value,
    "priority_improvements",
    "priorityImprovements"
  );
  const refinementCandidates = readStringListField(
    value,
    "refinement_candidates",
    "refinementCandidates"
  );
  const implementationSuggestions = readStringListField(
    value,
    "implementation_suggestions",
    "implementationSuggestions"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_refiner" ||
    refinementConfidence === null ||
    !refinementSummary ||
    recommendedImprovements.length === 0 ||
    priorityImprovements.length === 0 ||
    refinementCandidates.length === 0 ||
    implementationSuggestions.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    refinementConfidence,
    refinementSummary,
    recommendedImprovements,
    priorityImprovements,
    capabilityImprovements: readStringListField(
      value,
      "capability_improvements",
      "capabilityImprovements"
    ),
    dependencyImprovements: readStringListField(
      value,
      "dependency_improvements",
      "dependencyImprovements"
    ),
    runtimeImprovements: readStringListField(
      value,
      "runtime_improvements",
      "runtimeImprovements"
    ),
    scalabilityImprovements: readStringListField(
      value,
      "scalability_improvements",
      "scalabilityImprovements"
    ),
    maintainabilityImprovements: readStringListField(
      value,
      "maintainability_improvements",
      "maintainabilityImprovements"
    ),
    complexityReductions: readStringListField(
      value,
      "complexity_reductions",
      "complexityReductions"
    ),
    riskReductions: readStringListField(
      value,
      "risk_reductions",
      "riskReductions"
    ),
    refinementCandidates,
    implementationSuggestions,
    alternativeRefinementPaths: readStringListField(
      value,
      "alternative_refinement_paths",
      "alternativeRefinementPaths"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactIntelligenceSynthesisSummary(
  value: unknown
): ArtifactIntelligenceSynthesisSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const synthesisConfidence =
    readFiniteNumberField(value, "synthesis_confidence") ??
    readFiniteNumberField(value, "synthesisConfidence");
  const synthesisSummary =
    readStringField(value, "synthesis_summary") ??
    readStringField(value, "synthesisSummary");
  const recommendedArtifactPath =
    readStringField(value, "recommended_artifact_path") ??
    readStringField(value, "recommendedArtifactPath");
  const recommendedStrategySummary =
    readStringField(value, "recommended_strategy_summary") ??
    readStringField(value, "recommendedStrategySummary");
  const recommendedRuntimeDirection =
    readStringField(value, "recommended_runtime_direction") ??
    readStringField(value, "recommendedRuntimeDirection");
  const majorStrengths = readStringListField(
    value,
    "major_strengths",
    "majorStrengths"
  );
  const dependencyOverview =
    readStringField(value, "dependency_overview") ??
    readStringField(value, "dependencyOverview");
  const capabilityOverview =
    readStringField(value, "capability_overview") ??
    readStringField(value, "capabilityOverview");
  const refinementOverview =
    readStringField(value, "refinement_overview") ??
    readStringField(value, "refinementOverview");
  const critiqueOverview =
    readStringField(value, "critique_overview") ??
    readStringField(value, "critiqueOverview");
  const implementationReadiness = readStringUnion(
    value,
    "implementation_readiness",
    "implementationReadiness",
    artifactImplementationReadinesses
  );
  const implementationComplexity = readStringUnion(
    value,
    "implementation_complexity",
    "implementationComplexity",
    artifactImplementationComplexities
  );
  const implementationRisk = readStringUnion(
    value,
    "implementation_risk",
    "implementationRisk",
    artifactImplementationRisks
  );
  const implementationPriority = readStringUnion(
    value,
    "implementation_priority",
    "implementationPriority",
    artifactImplementationPriorities
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_intelligence_synthesis" ||
    synthesisConfidence === null ||
    !synthesisSummary ||
    !recommendedArtifactPath ||
    !recommendedStrategySummary ||
    !recommendedRuntimeDirection ||
    majorStrengths.length === 0 ||
    !dependencyOverview ||
    !capabilityOverview ||
    !refinementOverview ||
    !critiqueOverview ||
    !implementationReadiness ||
    !implementationComplexity ||
    !implementationRisk ||
    !implementationPriority ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    synthesisConfidence,
    synthesisSummary,
    recommendedArtifactPath,
    recommendedStrategySummary,
    recommendedRuntimeDirection,
    majorStrengths,
    majorWeaknesses: readStringListField(
      value,
      "major_weaknesses",
      "majorWeaknesses"
    ),
    majorRisks: readStringListField(value, "major_risks", "majorRisks"),
    dependencyOverview,
    capabilityOverview,
    refinementOverview,
    critiqueOverview,
    implementationReadiness,
    implementationComplexity,
    implementationRisk,
    implementationPriority,
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactMergePlannerSummary(
  value: unknown
): ArtifactMergePlannerSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const mergeConfidence =
    readFiniteNumberField(value, "merge_confidence") ??
    readFiniteNumberField(value, "mergeConfidence");
  const mergeSummary =
    readStringField(value, "merge_summary") ??
    readStringField(value, "mergeSummary");
  const mergeStrategy = readStringUnion(
    value,
    "merge_strategy",
    "mergeStrategy",
    artifactMergeStrategies
  );
  const compositionStrategy =
    readStringField(value, "composition_strategy") ??
    readStringField(value, "compositionStrategy");
  const artifactBoundaries = readStringListField(
    value,
    "artifact_boundaries",
    "artifactBoundaries"
  );
  const integrationOrder = readStringListField(
    value,
    "integration_order",
    "integrationOrder"
  );
  const recommendedMergePath =
    readStringField(value, "recommended_merge_path") ??
    readStringField(value, "recommendedMergePath");
  const rejectedMergePaths = readStringListField(
    value,
    "rejected_merge_paths",
    "rejectedMergePaths"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_merge_planner" ||
    mergeConfidence === null ||
    !mergeSummary ||
    !mergeStrategy ||
    !compositionStrategy ||
    artifactBoundaries.length === 0 ||
    integrationOrder.length === 0 ||
    !recommendedMergePath ||
    rejectedMergePaths.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    mergeConfidence,
    mergeSummary,
    mergeStrategy,
    compositionStrategy,
    artifactBoundaries,
    artifactJoinPoints: readStringListField(
      value,
      "artifact_join_points",
      "artifactJoinPoints"
    ),
    artifactSeparationPoints: readStringListField(
      value,
      "artifact_separation_points",
      "artifactSeparationPoints"
    ),
    integrationOrder,
    compositionRisks: readStringListField(
      value,
      "composition_risks",
      "compositionRisks"
    ),
    dependencyMergeRisks: readStringListField(
      value,
      "dependency_merge_risks",
      "dependencyMergeRisks"
    ),
    runtimeMergeRisks: readStringListField(
      value,
      "runtime_merge_risks",
      "runtimeMergeRisks"
    ),
    capabilityMergeRisks: readStringListField(
      value,
      "capability_merge_risks",
      "capabilityMergeRisks"
    ),
    recommendedMergePath,
    alternativeMergePaths: readStringListField(
      value,
      "alternative_merge_paths",
      "alternativeMergePaths"
    ),
    rejectedMergePaths,
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactExportIntelligenceSummary(
  value: unknown
): ArtifactExportIntelligenceSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const exportConfidence =
    readFiniteNumberField(value, "export_confidence") ??
    readFiniteNumberField(value, "exportConfidence");
  const exportSummary =
    readStringField(value, "export_summary") ??
    readStringField(value, "exportSummary");
  const exportTargets = readStringListField(
    value,
    "export_targets",
    "exportTargets"
  );
  const preferredExportTarget =
    readStringField(value, "preferred_export_target") ??
    readStringField(value, "preferredExportTarget");
  const exportReadiness = readStringUnion(
    value,
    "export_readiness",
    "exportReadiness",
    artifactExportReadinessValues
  );
  const exportFormatRecommendations = readStringListField(
    value,
    "export_format_recommendations",
    "exportFormatRecommendations"
  );
  const exportRequirements = readStringListField(
    value,
    "export_requirements",
    "exportRequirements"
  );
  const exportConstraints = readStringListField(
    value,
    "export_constraints",
    "exportConstraints"
  );
  const rejectedExportPaths = readStringListField(
    value,
    "rejected_export_paths",
    "rejectedExportPaths"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_export_intelligence" ||
    exportConfidence === null ||
    !exportSummary ||
    exportTargets.length === 0 ||
    !preferredExportTarget ||
    !exportReadiness ||
    exportFormatRecommendations.length === 0 ||
    exportRequirements.length === 0 ||
    exportConstraints.length === 0 ||
    rejectedExportPaths.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    exportConfidence,
    exportSummary,
    exportTargets,
    preferredExportTarget,
    exportFormatRecommendations,
    exportReadiness,
    exportRequirements,
    exportConstraints,
    exportRisks: readStringListField(value, "export_risks", "exportRisks"),
    runtimeExportNotes: readStringListField(
      value,
      "runtime_export_notes",
      "runtimeExportNotes"
    ),
    artifactPackageNotes: readStringListField(
      value,
      "artifact_package_notes",
      "artifactPackageNotes"
    ),
    portabilityNotes: readStringListField(
      value,
      "portability_notes",
      "portabilityNotes"
    ),
    interoperabilityNotes: readStringListField(
      value,
      "interoperability_notes",
      "interoperabilityNotes"
    ),
    documentationRequirements: readStringListField(
      value,
      "documentation_requirements",
      "documentationRequirements"
    ),
    downstreamToolHandoffs: readStringListField(
      value,
      "downstream_tool_handoffs",
      "downstreamToolHandoffs"
    ),
    rejectedExportPaths,
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactEngineContractRegistrySummary(
  value: unknown
): ArtifactIntelligenceEngineContractRegistrySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const engineCategory = readStringUnion(
    value,
    "engine_category",
    "engineCategory",
    artifactEngineCategories
  );
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const engineContracts = readArtifactEngineContractSummaryList(
    value.engine_contracts ?? value.engineContracts
  );
  const engineIds = readStringListField(value, "engine_ids", "engineIds");
  const contractCount =
    readFiniteNumberField(value, "contract_count") ??
    readFiniteNumberField(value, "contractCount");
  const futureAgentConsumers = readStringListField(
    value,
    "future_agent_consumers",
    "futureAgentConsumers"
  );

  if (
    role !== "artifact_intelligence_engine_contract_registry" ||
    engineCategory !== "artifact_intelligence" ||
    serializationVersion !== "artifact_engine_contract_registry.v1" ||
    !authorityBoundary ||
    engineContracts.length === 0 ||
    engineIds.length === 0 ||
    contractCount === null ||
    futureAgentConsumers.length === 0
  ) {
    return null;
  }

  return {
    role,
    engineCategory,
    serializationVersion,
    authorityBoundary,
    engineContracts,
    engineIds,
    contractCount,
    futureAgentConsumers
  };
}

function readArtifactEngineContractSummaryList(
  value: unknown
): ArtifactIntelligenceEngineContractSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const contract = readArtifactEngineContractSummary(item);
    return contract === null ? [] : [contract];
  });
}

function readArtifactEngineContractSummary(
  value: unknown
): ArtifactIntelligenceEngineContractSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const engineId =
    readStringField(value, "engine_id") ?? readStringField(value, "engineId");
  const engineName =
    readStringField(value, "engine_name") ?? readStringField(value, "engineName");
  const engineVersion =
    readStringField(value, "engine_version") ??
    readStringField(value, "engineVersion");
  const engineCategory = readStringUnion(
    value,
    "engine_category",
    "engineCategory",
    artifactEngineCategories
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const cacheability = readStringUnion(
    value,
    "cacheability",
    "cacheability",
    artifactEngineCacheabilityValues
  );
  const parallelizationSupport = readStringUnion(
    value,
    "parallelization_support",
    "parallelizationSupport",
    artifactEngineParallelizationSupportValues
  );
  const estimatedCostMetadata = readArtifactEngineCostMetadataSummary(
    value.estimated_cost_metadata ?? value.estimatedCostMetadata
  );
  const estimatedLatencyMetadata = readArtifactEngineLatencyMetadataSummary(
    value.estimated_latency_metadata ?? value.estimatedLatencyMetadata
  );
  const serializationVersion =
    readStringField(value, "serialization_version") ??
    readStringField(value, "serializationVersion");

  if (
    !engineId ||
    !engineName ||
    !engineVersion ||
    engineCategory !== "artifact_intelligence" ||
    !authorityBoundary ||
    !cacheability ||
    !parallelizationSupport ||
    estimatedCostMetadata === null ||
    estimatedLatencyMetadata === null ||
    serializationVersion !== "artifact_engine_contract.v1"
  ) {
    return null;
  }

  return {
    engineId,
    engineName,
    engineVersion,
    engineCategory,
    authorityBoundary,
    requiredInputs: readStringListField(
      value,
      "required_inputs",
      "requiredInputs"
    ),
    optionalInputs: readStringListField(
      value,
      "optional_inputs",
      "optionalInputs"
    ),
    producedMetadata: readStringListField(
      value,
      "produced_metadata",
      "producedMetadata"
    ),
    producedSignals: readStringListField(
      value,
      "produced_signals",
      "producedSignals"
    ),
    confidenceSignals: readStringListField(
      value,
      "confidence_signals",
      "confidenceSignals"
    ),
    ambiguitySignals: readStringListField(
      value,
      "ambiguity_signals",
      "ambiguitySignals"
    ),
    riskSignals: readStringListField(value, "risk_signals", "riskSignals"),
    escalationCandidates: readStringListField(
      value,
      "escalation_candidates",
      "escalationCandidates"
    ),
    downstreamDependencies: readStringListField(
      value,
      "downstream_dependencies",
      "downstreamDependencies"
    ),
    upstreamDependencies: readStringListField(
      value,
      "upstream_dependencies",
      "upstreamDependencies"
    ),
    cacheability,
    parallelizationSupport,
    estimatedCostMetadata,
    estimatedLatencyMetadata,
    serializationVersion,
    futureAgentHooks: readStringListField(
      value,
      "future_agent_hooks",
      "futureAgentHooks"
    ),
    futureExecutionHooks: readStringListField(
      value,
      "future_execution_hooks",
      "futureExecutionHooks"
    )
  };
}

function readArtifactEngineCostMetadataSummary(
  value: unknown
): ArtifactIntelligenceEngineContractSummary["estimatedCostMetadata"] | null {
  if (!isRecord(value)) {
    return null;
  }

  const relativeCost = readStringUnion(
    value,
    "relative_cost",
    "relativeCost",
    artifactEngineCostValues
  );
  const externalProviderCalls =
    readBooleanField(value, "external_provider_calls") ??
    readBooleanField(value, "externalProviderCalls");
  const costBasis =
    readStringField(value, "cost_basis") ?? readStringField(value, "costBasis");
  const cacheSensitivity =
    readStringField(value, "cache_sensitivity") ??
    readStringField(value, "cacheSensitivity");

  if (
    !relativeCost ||
    externalProviderCalls === null ||
    !costBasis ||
    !cacheSensitivity
  ) {
    return null;
  }

  return {
    relativeCost,
    externalProviderCalls,
    costBasis,
    cacheSensitivity
  };
}

function readArtifactEngineLatencyMetadataSummary(
  value: unknown
): ArtifactIntelligenceEngineContractSummary["estimatedLatencyMetadata"] | null {
  if (!isRecord(value)) {
    return null;
  }

  const relativeLatency = readStringUnion(
    value,
    "relative_latency",
    "relativeLatency",
    artifactEngineLatencyValues
  );
  const latencyBasis =
    readStringField(value, "latency_basis") ??
    readStringField(value, "latencyBasis");

  if (!relativeLatency || !latencyBasis) {
    return null;
  }

  return {
    relativeLatency,
    latencyBasis,
    blockingInputs: readStringListField(
      value,
      "blocking_inputs",
      "blockingInputs"
    )
  };
}

function readArtifactCapabilityConfidenceSummaryList(
  value: unknown
): ArtifactCapabilityConfidenceSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }
    const target = readStringUnion(
      item,
      "target",
      "target",
      runtimeCapabilityIds
    );
    const label = readStringField(item, "label");
    const confidence = readFiniteNumberField(item, "confidence");
    if (!target || !label || confidence === null) {
      return [];
    }
    return [{ target, label, confidence }];
  });
}

function readArtifactCapabilityProfileSummaryList(
  value: unknown
): ArtifactCapabilityProfileSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const profile = readArtifactCapabilityProfileSummary(item);
    return profile ? [profile] : [];
  });
}

function readArtifactCapabilityProfileSummary(
  value: unknown
): ArtifactCapabilityProfileSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const target = readStringUnion(
    value,
    "target",
    "target",
    runtimeCapabilityIds
  );
  const label = readStringField(value, "label");
  const capabilityConfidence =
    readFiniteNumberField(value, "capability_confidence") ??
    readFiniteNumberField(value, "capabilityConfidence");
  const artifactFit = readArtifactCapabilityFit(value, "artifact_fit", "artifactFit");
  const creativeFit = readArtifactCapabilityFit(value, "creative_fit", "creativeFit");
  const generativeFit = readArtifactCapabilityFit(
    value,
    "generative_fit",
    "generativeFit"
  );
  const interactionFit = readArtifactCapabilityFit(
    value,
    "interaction_fit",
    "interactionFit"
  );
  const audiovisualFit = readArtifactCapabilityFit(
    value,
    "audiovisual_fit",
    "audiovisualFit"
  );
  const exportFit = readArtifactCapabilityFit(value, "export_fit", "exportFit");
  const interoperabilityFit = readArtifactCapabilityFit(
    value,
    "interoperability_fit",
    "interoperabilityFit"
  );
  const portabilityFit = readArtifactCapabilityFit(
    value,
    "portability_fit",
    "portabilityFit"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !target ||
    !label ||
    capabilityConfidence === null ||
    !artifactFit ||
    !creativeFit ||
    !generativeFit ||
    !interactionFit ||
    !audiovisualFit ||
    !exportFit ||
    !interoperabilityFit ||
    !portabilityFit ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    target,
    label,
    capabilityConfidence,
    capabilityReasons: readStringListField(
      value,
      "capability_reasons",
      "capabilityReasons"
    ),
    strengths: readStringListField(value, "strengths", "strengths"),
    weaknesses: readStringListField(value, "weaknesses", "weaknesses"),
    unsupportedCapabilities: readStringListField(
      value,
      "unsupported_capabilities",
      "unsupportedCapabilities"
    ),
    riskyCapabilities: readStringListField(
      value,
      "risky_capabilities",
      "riskyCapabilities"
    ),
    artifactFit,
    creativeFit,
    generativeFit,
    interactionFit,
    audiovisualFit,
    exportFit,
    interoperabilityFit,
    portabilityFit,
    capabilityRisks: readStringListField(
      value,
      "capability_risks",
      "capabilityRisks"
    ),
    promptGuidance,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readArtifactCapabilityFit(
  value: Record<string, unknown>,
  snakeKey: string,
  camelKey: string
): ArtifactCapabilityFit | null {
  return readStringUnion(value, snakeKey, camelKey, artifactCapabilityFits);
}

const runtimeCapabilityIds = [
  "p5_js",
  "three_js",
  "react_three_fiber",
  "glsl",
  "hydra",
  "tone_js",
  "gsap",
  "svg",
  "canvas"
] as const satisfies readonly RuntimeCapabilityId[];

const runtimeCapabilityFits = [
  "strong",
  "moderate",
  "weak"
] as const satisfies readonly RuntimeCapabilityFit[];

const artifactCapabilityFits = [
  "strong",
  "moderate",
  "weak",
  "unsupported"
] as const satisfies readonly ArtifactCapabilityFit[];

const multiArtifactStrategyRoles = [
  "primary",
  "supporting",
  "optional"
] as const satisfies readonly MultiArtifactStrategyRole[];

const multiArtifactStrategyPriorities = [
  "critical",
  "high",
  "medium",
  "low"
] as const satisfies readonly MultiArtifactStrategyPriority[];

const multiArtifactStrategyActions = [
  "produce",
  "separate",
  "document",
  "handoff"
] as const satisfies readonly MultiArtifactStrategyAction[];

const multiArtifactStrategyCombinationModes = [
  "primary_with_supporting_sections",
  "separated_parallel_sections",
  "defer_combination"
] as const satisfies readonly MultiArtifactStrategyCombinationMode[];

const artifactCriticRiskAssessments = [
  "low",
  "medium",
  "high",
  "blocked"
] as const satisfies readonly ArtifactCriticRiskAssessment[];

const creativeCriticRiskAssessments = [
  "low",
  "medium",
  "high",
  "blocked"
] as const satisfies readonly CreativeCriticRiskAssessment[];

const artifactImplementationReadinesses = [
  "ready",
  "needs_caveats",
  "needs_hitl",
  "blocked"
] as const satisfies readonly ArtifactImplementationReadiness[];

const artifactImplementationComplexities = [
  "low",
  "medium",
  "high"
] as const satisfies readonly ArtifactImplementationComplexity[];

const artifactImplementationRisks = [
  "low",
  "medium",
  "high",
  "blocked"
] as const satisfies readonly ArtifactImplementationRisk[];

const artifactImplementationPriorities = [
  "critical",
  "high",
  "medium",
  "low"
] as const satisfies readonly ArtifactImplementationPriority[];

const artifactMergeStrategies = [
  "single_artifact_no_merge",
  "primary_with_supporting_sections",
  "separated_advisory_sections",
  "defer_merge_preserve_separation"
] as const satisfies readonly ArtifactMergeStrategy[];

const artifactExportReadinessValues = [
  "ready_with_caveats",
  "needs_handoff_metadata",
  "blocked_by_missing_metadata",
  "defer_export"
] as const satisfies readonly ArtifactExportReadiness[];

const artifactEngineCategories = [
  "artifact_intelligence"
] as const satisfies readonly ArtifactEngineCategory[];

const artifactEngineCacheabilityValues = [
  "deterministic_per_request",
  "deterministic_with_upstream_metadata"
] as const satisfies readonly ArtifactEngineCacheability[];

const artifactEngineParallelizationSupportValues = [
  "requires_ordered_upstream_metadata",
  "parallel_after_required_inputs"
] as const satisfies readonly ArtifactEngineParallelizationSupport[];

const artifactEngineCostValues = ["low", "medium"] as const;

const artifactEngineLatencyValues = ["low", "medium"] as const;

const runtimeCapabilityComplexities = [
  "low",
  "medium",
  "high"
] as const satisfies readonly RuntimeCapabilityComplexity[];

const runtimeCompatibilityLevels = [
  "compatible",
  "partially_compatible",
  "unsupported"
] as const satisfies readonly RuntimeCompatibilityLevel[];

const runtimePortabilities = [
  "high",
  "medium",
  "low"
] as const satisfies readonly RuntimePortability[];

const runtimeInteroperabilities = [
  "high",
  "medium",
  "low"
] as const satisfies readonly RuntimeInteroperability[];

const runtimePreviewSupports = [
  "backend_preview_supported",
  "workstation_preview_bounded",
  "code_only"
] as const satisfies readonly RuntimePreviewSupport[];

function readRuntimeCapabilityIdList(value: unknown): RuntimeCapabilityId[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is RuntimeCapabilityId =>
      typeof item === "string" &&
      runtimeCapabilityIds.includes(item as RuntimeCapabilityId)
  );
}

function readRuntimeCapabilityCandidateSummaryList(
  value: unknown
): RuntimeCapabilityCandidateSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const runtime = readStringUnion(
      item,
      "runtime",
      "runtime",
      runtimeCapabilityIds
    );
    const label = readStringField(item, "label");
    const suitability = readStringUnion(
      item,
      "suitability",
      "suitability",
      runtimeCapabilityFits
    );
    const confidence = readFiniteNumberField(item, "confidence");
    const strategyAlignment = readStringUnion(
      item,
      "strategy_alignment",
      "strategyAlignment",
      runtimeCapabilityFits
    );
    const techniqueCompatibility = readStringUnion(
      item,
      "technique_compatibility",
      "techniqueCompatibility",
      runtimeCapabilityFits
    );
    const outputGoalFit = readStringUnion(
      item,
      "output_goal_fit",
      "outputGoalFit",
      runtimeCapabilityFits
    );
    const implementationComplexity = readStringUnion(
      item,
      "implementation_complexity",
      "implementationComplexity",
      runtimeCapabilityComplexities
    );
    const performancePressure = readStringUnion(
      item,
      "performance_pressure",
      "performancePressure",
      ["low", "medium", "high"]
    );
    const previewSupport = readStringUnion(
      item,
      "preview_support",
      "previewSupport",
      runtimePreviewSupports
    );
    const strengths = readStringListField(item, "strengths", "strengths");
    const limitations = readStringListField(item, "limitations", "limitations");
    const risks = readStringListField(item, "risks", "risks");
    const promptGuidance = readStringListField(
      item,
      "prompt_guidance",
      "promptGuidance"
    );

    if (
      !runtime ||
      !label ||
      !suitability ||
      confidence === null ||
      !strategyAlignment ||
      !techniqueCompatibility ||
      !outputGoalFit ||
      !implementationComplexity ||
      !performancePressure ||
      !previewSupport ||
      strengths.length === 0 ||
      limitations.length === 0 ||
      risks.length === 0 ||
      promptGuidance.length === 0
    ) {
      return [];
    }

    return [
      {
        runtime,
        label,
        suitability,
        confidence,
        strategyAlignment,
        techniqueCompatibility,
        outputGoalFit,
        implementationComplexity,
        performancePressure,
        previewSupport,
        strengths,
        limitations,
        risks,
        promptGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readCreativeTradeoffExplorerSummary(
  value: unknown
): CreativeTradeoffExplorerSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const outputGoal =
    readStringField(value, "output_goal") ??
    readStringField(value, "outputGoal");
  const primaryTradeoffs = readCreativeTradeoffSummaryList(
    value.primary_tradeoffs ?? value.primaryTradeoffs
  );
  const creativeBenefits = readStringListField(
    value,
    "creative_benefits",
    "creativeBenefits"
  );
  const technicalCosts = readStringListField(
    value,
    "technical_costs",
    "technicalCosts"
  );
  const costSensitivity = readStringUnion(
    value,
    "cost_sensitivity",
    "costSensitivity",
    creativeTradeoffPressures
  );
  const hitlAdvisable =
    readBooleanField(value, "hitl_advisable") ??
    readBooleanField(value, "hitlAdvisable");
  const directorDiscussionPoints = readStringListField(
    value,
    "director_discussion_points",
    "directorDiscussionPoints"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_tradeoff_explorer" ||
    !outputGoal ||
    primaryTradeoffs.length === 0 ||
    creativeBenefits.length === 0 ||
    technicalCosts.length === 0 ||
    !costSensitivity ||
    hitlAdvisable === null ||
    directorDiscussionPoints.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    outputGoal,
    primaryTradeoffs,
    creativeBenefits,
    technicalCosts,
    runtimeRisks: readStringListField(
      value,
      "runtime_risks",
      "runtimeRisks"
    ),
    performanceConcerns: readStringListField(
      value,
      "performance_concerns",
      "performanceConcerns"
    ),
    complexityRisks: readStringListField(
      value,
      "complexity_risks",
      "complexityRisks"
    ),
    fidelityRisks: readStringListField(
      value,
      "fidelity_risks",
      "fidelityRisks"
    ),
    costSensitivity,
    safetyConcerns: readStringListField(
      value,
      "safety_concerns",
      "safetyConcerns"
    ),
    maintainabilityConcerns: readStringListField(
      value,
      "maintainability_concerns",
      "maintainabilityConcerns"
    ),
    hitlAdvisable,
    hitlReason:
      readStringField(value, "hitl_reason") ??
      readStringField(value, "hitlReason"),
    directorDiscussionPoints,
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeTradeoffAxes = [
  "creative_expressiveness",
  "concept_fidelity",
  "implementation_complexity",
  "performance",
  "runtime_support",
  "previewability",
  "cost_sensitivity",
  "safety",
  "maintainability",
  "hitl"
] as const satisfies readonly CreativeTradeoffAxis[];

const creativeTradeoffSeverities = [
  "info",
  "watch",
  "risk",
  "blocking"
] as const satisfies readonly CreativeTradeoffSeverity[];

const creativeTradeoffPressures = [
  "low",
  "medium",
  "high"
] as const satisfies readonly CreativeTradeoffPressure[];

function readCreativeTradeoffSummaryList(
  value: unknown
): CreativeTradeoffSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const sourceAxis = readStringUnion(
      item,
      "source_axis",
      "sourceAxis",
      creativeTradeoffAxes
    );
    const targetAxis = readStringUnion(
      item,
      "target_axis",
      "targetAxis",
      creativeTradeoffAxes
    );
    const severity = readStringUnion(
      item,
      "severity",
      "severity",
      creativeTradeoffSeverities
    );
    const summary = readStringField(item, "summary");
    const creativeBenefit =
      readStringField(item, "creative_benefit") ??
      readStringField(item, "creativeBenefit");
    const technicalCost =
      readStringField(item, "technical_cost") ??
      readStringField(item, "technicalCost");
    const runtimeImplication =
      readStringField(item, "runtime_implication") ??
      readStringField(item, "runtimeImplication");
    const mitigation = readStringField(item, "mitigation");
    const directorDiscussionPoint =
      readStringField(item, "director_discussion_point") ??
      readStringField(item, "directorDiscussionPoint");
    const hitlRecommended =
      readBooleanField(item, "hitl_recommended") ??
      readBooleanField(item, "hitlRecommended");

    if (
      !sourceAxis ||
      !targetAxis ||
      !severity ||
      !summary ||
      !creativeBenefit ||
      !technicalCost ||
      !runtimeImplication ||
      !mitigation ||
      !directorDiscussionPoint ||
      hitlRecommended === null
    ) {
      return [];
    }

    return [
      {
        sourceAxis,
        targetAxis,
        severity,
        summary,
        creativeBenefit,
        technicalCost,
        runtimeImplication,
        mitigation,
        directorDiscussionPoint,
        hitlRecommended,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readCreativeQualityPredictionSummary(
  value: unknown
): CreativeQualityPredictionSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const predictedQualityLevel = readStringUnion(
    value,
    "predicted_quality_level",
    "predictedQualityLevel",
    creativeQualityLevels
  );
  const confidence = readFiniteNumberField(value, "confidence");
  const readinessScore =
    readFiniteNumberField(value, "readiness_score") ??
    readFiniteNumberField(value, "readinessScore");
  const strongestQualitySignals = readCreativeQualitySignalSummaryList(
    value.strongest_quality_signals ?? value.strongestQualitySignals
  );
  const weakestQualitySignals = readCreativeQualitySignalSummaryList(
    value.weakest_quality_signals ?? value.weakestQualitySignals
  );
  const likelyFailureModes = readStringListField(
    value,
    "likely_failure_modes",
    "likelyFailureModes"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_quality_predictor" ||
    !predictedQualityLevel ||
    confidence === null ||
    readinessScore === null ||
    strongestQualitySignals.length === 0 ||
    weakestQualitySignals.length === 0 ||
    likelyFailureModes.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    predictedQualityLevel,
    confidence,
    readinessScore,
    strongestQualitySignals,
    weakestQualitySignals,
    qualityRisks: readStringListField(
      value,
      "quality_risks",
      "qualityRisks"
    ),
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    likelyFailureModes,
    suggestedImprovements: readStringListField(
      value,
      "suggested_improvements",
      "suggestedImprovements"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeQualityLevels = [
  "strong",
  "promising",
  "ambiguous",
  "risky",
  "blocked"
] as const satisfies readonly CreativeQualityPredictionLevel[];

const creativeQualityDimensions = [
  "intent_clarity",
  "symbolic_coherence",
  "narrative_coherence",
  "emotional_coherence",
  "geometric_formal_clarity",
  "technique_suitability",
  "runtime_suitability",
  "tradeoff_balance",
  "constraint_alignment",
  "implementation_feasibility",
  "previewability",
  "performance_risk",
  "originality_potential",
  "aesthetic_coherence_potential"
] as const satisfies readonly CreativeQualityDimension[];

function readCreativeQualitySignalSummaryList(
  value: unknown
): CreativeQualitySignalSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const dimension = readStringUnion(
      item,
      "dimension",
      "dimension",
      creativeQualityDimensions
    );
    const score = readFiniteNumberField(item, "score");
    const summary = readStringField(item, "summary");

    if (!dimension || score === null || !summary) {
      return [];
    }

    return [
      {
        dimension,
        score,
        summary,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readSymbolicNarrativePlanSummary(
  value: unknown
): SymbolicNarrativePlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const narrativeArchetype = readStringUnion(
    value,
    "narrative_archetype",
    "narrativeArchetype",
    symbolicNarrativeArchetypes
  );
  const symbolicArc =
    readStringField(value, "symbolic_arc") ??
    readStringField(value, "symbolicArc");
  const openingPhase = readSymbolicNarrativePhaseSummary(
    value.opening_phase ?? value.openingPhase
  );
  const developmentPhase = readSymbolicNarrativePhaseSummary(
    value.development_phase ?? value.developmentPhase
  );
  const thresholdPhase = readSymbolicNarrativePhaseSummary(
    value.threshold_phase ?? value.thresholdPhase
  );
  const climaxPhase = readSymbolicNarrativePhaseSummary(
    value.climax_phase ?? value.climaxPhase
  );
  const resolutionPhase = readSymbolicNarrativePhaseSummary(
    value.resolution_phase ?? value.resolutionPhase
  );
  const symbolicTransitions = readStringListField(
    value,
    "symbolic_transitions",
    "symbolicTransitions"
  );
  const emotionalProgression = readStringListField(
    value,
    "emotional_progression",
    "emotionalProgression"
  );
  const visualProgression = readStringListField(
    value,
    "visual_progression",
    "visualProgression"
  );
  const motionProgression = readStringListField(
    value,
    "motion_progression",
    "motionProgression"
  );
  const experientialGoal =
    readStringField(value, "experiential_goal") ??
    readStringField(value, "experientialGoal");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "symbolic_narrative_planner" ||
    !narrativeArchetype ||
    !symbolicArc ||
    !openingPhase ||
    !developmentPhase ||
    !thresholdPhase ||
    !climaxPhase ||
    !resolutionPhase ||
    symbolicTransitions.length === 0 ||
    emotionalProgression.length === 0 ||
    visualProgression.length === 0 ||
    motionProgression.length === 0 ||
    !experientialGoal ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    narrativeArchetype,
    symbolicArc,
    openingPhase,
    developmentPhase,
    thresholdPhase,
    climaxPhase,
    resolutionPhase,
    symbolicTransitions,
    emotionalProgression,
    visualProgression,
    motionProgression,
    audioProgression: readStringListField(
      value,
      "audio_progression",
      "audioProgression"
    ),
    experientialGoal,
    unresolvedNarrativeGaps: readStringListField(
      value,
      "unresolved_narrative_gaps",
      "unresolvedNarrativeGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readSymbolicNarrativePhaseSummary(
  value: unknown
): SymbolicNarrativePhaseSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const phase = readStringUnion(
    value,
    "phase",
    "phase",
    symbolicNarrativePhaseNames
  );
  const title = readStringField(value, "title");
  const symbolicFunction =
    readStringField(value, "symbolic_function") ??
    readStringField(value, "symbolicFunction");
  const emotionalState =
    readStringField(value, "emotional_state") ??
    readStringField(value, "emotionalState");
  const visualState =
    readStringField(value, "visual_state") ??
    readStringField(value, "visualState");
  const motionState =
    readStringField(value, "motion_state") ??
    readStringField(value, "motionState");

  if (
    !phase ||
    !title ||
    !symbolicFunction ||
    !emotionalState ||
    !visualState ||
    !motionState
  ) {
    return null;
  }

  return {
    phase,
    title,
    symbolicFunction,
    emotionalState,
    visualState,
    motionState,
    audioState:
      readStringField(value, "audio_state") ??
      readStringField(value, "audioState"),
    guidance: readStringListField(value, "guidance", "guidance"),
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const symbolicNarrativeArchetypes = [
  "descent_and_return",
  "death_and_rebirth",
  "emergence_from_chaos",
  "initiation",
  "ascent",
  "dissolution_and_reintegration",
  "expansion_from_seed_to_cosmos",
  "fragmentation_and_recomposition",
  "threshold_crossing",
  "spiral_transformation",
  "mirror_reflection_journey",
  "dark_to_light_transformation",
  "symbolic_vignette"
] as const satisfies readonly SymbolicNarrativeArchetype[];

const symbolicNarrativePhaseNames = [
  "opening",
  "development",
  "threshold",
  "climax",
  "resolution"
] as const satisfies readonly SymbolicNarrativePhaseName[];

export function readCreativeCompositionPlanSummary(
  value: unknown
): CreativeCompositionPlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const compositionPattern = readStringUnion(
    value,
    "composition_pattern",
    "compositionPattern",
    creativeCompositionPatterns
  );
  const primaryFocalPoint =
    readStringField(value, "primary_focal_point") ??
    readStringField(value, "primaryFocalPoint");
  const spatialOrganization =
    readStringField(value, "spatial_organization") ??
    readStringField(value, "spatialOrganization");
  const foregroundBackgroundRelationship =
    readStringField(value, "foreground_background_relationship") ??
    readStringField(value, "foregroundBackgroundRelationship");
  const densityPlan =
    readStringField(value, "density_plan") ??
    readStringField(value, "densityPlan");
  const rhythmPlan =
    readStringField(value, "rhythm_plan") ??
    readStringField(value, "rhythmPlan");
  const balancePlan =
    readStringField(value, "balance_plan") ??
    readStringField(value, "balancePlan");
  const symmetryAsymmetryGuidance =
    readStringField(value, "symmetry_asymmetry_guidance") ??
    readStringField(value, "symmetryAsymmetryGuidance");
  const depthLayeringGuidance =
    readStringField(value, "depth_layering_guidance") ??
    readStringField(value, "depthLayeringGuidance");
  const secondaryFocalElements = readStringListField(
    value,
    "secondary_focal_elements",
    "secondaryFocalElements"
  );
  const visualHierarchy = readStringListField(
    value,
    "visual_hierarchy",
    "visualHierarchy"
  );
  const transitionGuidance = readStringListField(
    value,
    "transition_guidance",
    "transitionGuidance"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_composition_planner" ||
    !compositionPattern ||
    !primaryFocalPoint ||
    secondaryFocalElements.length === 0 ||
    !spatialOrganization ||
    !foregroundBackgroundRelationship ||
    visualHierarchy.length === 0 ||
    !densityPlan ||
    !rhythmPlan ||
    !balancePlan ||
    !symmetryAsymmetryGuidance ||
    !depthLayeringGuidance ||
    transitionGuidance.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    compositionPattern,
    primaryFocalPoint,
    secondaryFocalElements,
    spatialOrganization,
    foregroundBackgroundRelationship,
    visualHierarchy,
    densityPlan,
    rhythmPlan,
    balancePlan,
    symmetryAsymmetryGuidance,
    depthLayeringGuidance,
    transitionGuidance,
    cameraViewpointGuidance:
      readStringField(value, "camera_viewpoint_guidance") ??
      readStringField(value, "cameraViewpointGuidance"),
    audiovisualCompositionNotes: readStringListField(
      value,
      "audiovisual_composition_notes",
      "audiovisualCompositionNotes"
    ),
    compositionRisks: readStringListField(
      value,
      "composition_risks",
      "compositionRisks"
    ),
    unresolvedCompositionGaps: readStringListField(
      value,
      "unresolved_composition_gaps",
      "unresolvedCompositionGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeCompositionPatterns = [
  "central_emergence",
  "radial_expansion",
  "spiral_composition",
  "layered_depth",
  "field_composition",
  "threshold_composition",
  "descent_ascent_composition",
  "fragmented_recomposition",
  "mirrored_composition",
  "orbiting_focal_structure",
  "distributed_constellation",
  "minimal_void_and_form_composition"
] as const satisfies readonly CreativeCompositionPattern[];

export function readProceduralStructurePlanSummary(
  value: unknown
): ProceduralStructurePlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const recommendedFamilies = readProceduralFamilyList(
    value.recommended_families ?? value.recommendedFamilies
  );
  const primaryStructure = readProceduralStructureChoiceSummary(
    value.primary_structure ?? value.primaryStructure
  );
  const secondaryStructures = readProceduralStructureChoiceSummaryList(
    value.secondary_structures ?? value.secondaryStructures
  );
  const fallbackStructureOptions = readProceduralStructureChoiceSummaryList(
    value.fallback_structure_options ?? value.fallbackStructureOptions
  );
  const combinationStrategy =
    readStringField(value, "combination_strategy") ??
    readStringField(value, "combinationStrategy");
  const spatialStructurePlan =
    readStringField(value, "spatial_structure_plan") ??
    readStringField(value, "spatialStructurePlan");
  const temporalStructurePlan =
    readStringField(value, "temporal_structure_plan") ??
    readStringField(value, "temporalStructurePlan");
  const complexityLevel = readStringUnion(
    value,
    "complexity_level",
    "complexityLevel",
    proceduralComplexityLevels
  );
  const runtimeSuitabilityNotes = readStringListField(
    value,
    "runtime_suitability_notes",
    "runtimeSuitabilityNotes"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "procedural_structure_planner" ||
    recommendedFamilies.length === 0 ||
    !primaryStructure ||
    secondaryStructures.length === 0 ||
    !combinationStrategy ||
    !spatialStructurePlan ||
    !temporalStructurePlan ||
    !complexityLevel ||
    runtimeSuitabilityNotes.length === 0 ||
    fallbackStructureOptions.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    recommendedFamilies,
    primaryStructure,
    secondaryStructures,
    combinationStrategy,
    spatialStructurePlan,
    temporalStructurePlan,
    interactionStructurePlan:
      readStringField(value, "interaction_structure_plan") ??
      readStringField(value, "interactionStructurePlan"),
    audiovisualStructurePlan:
      readStringField(value, "audiovisual_structure_plan") ??
      readStringField(value, "audiovisualStructurePlan"),
    complexityLevel,
    runtimeSuitabilityNotes,
    performanceRisks: readStringListField(
      value,
      "performance_risks",
      "performanceRisks"
    ),
    implementationRisks: readStringListField(
      value,
      "implementation_risks",
      "implementationRisks"
    ),
    fallbackStructureOptions,
    unresolvedProceduralGaps: readStringListField(
      value,
      "unresolved_procedural_gaps",
      "unresolvedProceduralGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const proceduralFamilies = [
  "fractals",
  "recursive_geometry",
  "l_systems",
  "particle_systems",
  "boids",
  "cellular_automata",
  "reaction_diffusion",
  "voronoi_systems",
  "noise_fields",
  "flow_fields",
  "signed_distance_fields",
  "polar_radial_systems",
  "grid_systems",
  "graph_network_systems",
  "swarm_systems",
  "wave_systems",
  "harmonic_oscillators",
  "modular_tiling",
  "sacred_geometry_pattern_systems"
] as const satisfies readonly ProceduralFamily[];

const proceduralComplexityLevels = [
  "low",
  "medium",
  "high"
] as const satisfies readonly ProceduralComplexityLevel[];

function readProceduralFamilyList(value: unknown): ProceduralFamily[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is ProceduralFamily =>
      typeof item === "string" &&
      proceduralFamilies.includes(item as ProceduralFamily)
  );
}

function readProceduralStructureChoiceSummaryList(
  value: unknown
): ProceduralStructureChoiceSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const parsed = readProceduralStructureChoiceSummary(item);
    return parsed ? [parsed] : [];
  });
}

function readProceduralStructureChoiceSummary(
  value: unknown
): ProceduralStructureChoiceSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const family = readStringUnion(
    value,
    "family",
    "family",
    proceduralFamilies
  );
  const label = readStringField(value, "label");
  const rationale = readStringField(value, "rationale");

  if (!family || !label || !rationale) {
    return null;
  }

  return {
    family,
    label,
    rationale,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readGenerativeStructureBlueprintSummary(
  value: unknown
): GenerativeStructureBlueprintSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const blueprintName =
    readStringField(value, "blueprint_name") ??
    readStringField(value, "blueprintName");
  const generativeArchitecture = readStringUnion(
    value,
    "generative_architecture",
    "generativeArchitecture",
    generativeArchitectures
  );
  const proceduralModules = readGenerativeModuleSummaryList(
    value.procedural_modules ?? value.proceduralModules
  );
  const moduleRelationships = readGenerativeModuleRelationshipSummaryList(
    value.module_relationships ?? value.moduleRelationships
  );
  const parameterSchema = readGenerativeParameterSummaryList(
    value.parameter_schema ?? value.parameterSchema
  );
  const controlParameters = readStringListField(
    value,
    "control_parameters",
    "controlParameters"
  );
  const evolutionRules = readGenerativeEvolutionRuleSummaryList(
    value.evolution_rules ?? value.evolutionRules
  );
  const spatialEvolution =
    readStringField(value, "spatial_evolution") ??
    readStringField(value, "spatialEvolution");
  const temporalEvolution =
    readStringField(value, "temporal_evolution") ??
    readStringField(value, "temporalEvolution");
  const runtimeImplementationGuidance = readStringListField(
    value,
    "runtime_implementation_guidance",
    "runtimeImplementationGuidance"
  );
  const performanceSafeguards = readStringListField(
    value,
    "performance_safeguards",
    "performanceSafeguards"
  );
  const fallbackBlueprint = readGenerativeFallbackBlueprintSummary(
    value.fallback_blueprint ?? value.fallbackBlueprint
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "generative_structure_engine" ||
    !blueprintName ||
    !generativeArchitecture ||
    proceduralModules.length === 0 ||
    moduleRelationships.length === 0 ||
    parameterSchema.length === 0 ||
    controlParameters.length === 0 ||
    evolutionRules.length === 0 ||
    !spatialEvolution ||
    !temporalEvolution ||
    runtimeImplementationGuidance.length === 0 ||
    performanceSafeguards.length === 0 ||
    !fallbackBlueprint ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    blueprintName,
    generativeArchitecture,
    proceduralModules,
    moduleRelationships,
    parameterSchema,
    controlParameters,
    evolutionRules,
    spatialEvolution,
    temporalEvolution,
    interactionHooks: readGenerativeStructureHookSummaryList(
      value.interaction_hooks ?? value.interactionHooks
    ),
    audiovisualHooks: readGenerativeStructureHookSummaryList(
      value.audiovisual_hooks ?? value.audiovisualHooks
    ),
    runtimeImplementationGuidance,
    performanceSafeguards,
    fallbackBlueprint,
    unresolvedImplementationGaps: readStringListField(
      value,
      "unresolved_implementation_gaps",
      "unresolvedImplementationGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const generativeArchitectures = [
  "recursive_modular_blueprint",
  "agent_field_blueprint",
  "grid_state_blueprint",
  "radial_pattern_blueprint",
  "network_relation_blueprint",
  "wave_modulation_blueprint",
  "minimal_parameter_blueprint"
] as const satisfies readonly GenerativeArchitecture[];

const generativeModuleKinds = [
  "seed_system",
  "recursive_module",
  "particle_emitter",
  "force_field",
  "attractor_field",
  "noise_modulation_layer",
  "symmetry_transform",
  "tiling_layer",
  "graph_network_layer",
  "cellular_grid_layer",
  "wave_oscillator",
  "geometry_reassembly_layer",
  "color_modulation_layer",
  "audio_reactive_modulation_layer",
  "camera_motion_path_hook"
] as const satisfies readonly GenerativeModuleKind[];

const generativeRelationshipTypes = [
  "feeds",
  "modulates",
  "constrains",
  "emits",
  "attracts",
  "mirrors",
  "samples",
  "reassembles",
  "times",
  "fallback_for"
] as const satisfies readonly GenerativeRelationshipType[];

const generativeParameterValueTypes = [
  "integer",
  "float",
  "boolean",
  "vector",
  "color",
  "enum"
] as const satisfies readonly GenerativeParameterValueType[];

const generativeParameterRoles = [
  "control",
  "derived",
  "constraint"
] as const satisfies readonly GenerativeParameterRole[];

const generativeEvolutionPhases = [
  "seed",
  "growth",
  "fragmentation",
  "threshold",
  "reassembly",
  "stabilization",
  "loop"
] as const satisfies readonly GenerativeEvolutionPhase[];

const generativeEvolutionTriggers = [
  "time",
  "interaction",
  "audio",
  "parameter",
  "narrative_phase"
] as const satisfies readonly GenerativeEvolutionTrigger[];

const generativeHookTypes = [
  "interaction",
  "audiovisual"
] as const satisfies readonly GenerativeHookType[];

function readGenerativeModuleSummaryList(
  value: unknown
): GenerativeModuleSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const parsed = readGenerativeModuleSummary(item);
    return parsed ? [parsed] : [];
  });
}

function readGenerativeModuleSummary(
  value: unknown
): GenerativeModuleSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const moduleId =
    readStringField(value, "module_id") ?? readStringField(value, "moduleId");
  const kind = readStringUnion(value, "kind", "kind", generativeModuleKinds);
  const label = readStringField(value, "label");
  const sourceFamily = readStringUnion(
    value,
    "source_family",
    "sourceFamily",
    proceduralFamilies
  );
  const purpose = readStringField(value, "purpose");
  const outputs = readStringListField(value, "outputs", "outputs");
  const evolutionRole =
    readStringField(value, "evolution_role") ??
    readStringField(value, "evolutionRole");
  const implementationNotes = readStringListField(
    value,
    "implementation_notes",
    "implementationNotes"
  );

  if (
    !moduleId ||
    !kind ||
    !label ||
    !purpose ||
    outputs.length === 0 ||
    !evolutionRole ||
    implementationNotes.length === 0
  ) {
    return null;
  }

  return {
    moduleId,
    kind,
    label,
    sourceFamily,
    purpose,
    inputs: readStringListField(value, "inputs", "inputs"),
    outputs,
    parameters: readStringListField(value, "parameters", "parameters"),
    evolutionRole,
    implementationNotes,
    safeguards: readStringListField(value, "safeguards", "safeguards"),
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readGenerativeModuleRelationshipSummaryList(
  value: unknown
): GenerativeModuleRelationshipSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const sourceModuleId =
      readStringField(item, "source_module_id") ??
      readStringField(item, "sourceModuleId");
    const targetModuleId =
      readStringField(item, "target_module_id") ??
      readStringField(item, "targetModuleId");
    const relationshipType = readStringUnion(
      item,
      "relationship_type",
      "relationshipType",
      generativeRelationshipTypes
    );
    const description = readStringField(item, "description");

    if (!sourceModuleId || !targetModuleId || !relationshipType || !description) {
      return [];
    }

    return [
      {
        sourceModuleId,
        targetModuleId,
        relationshipType,
        description,
        parameters: readStringListField(item, "parameters", "parameters"),
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readGenerativeParameterSummaryList(
  value: unknown
): GenerativeParameterSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const name = readStringField(item, "name");
    const label = readStringField(item, "label");
    const valueType = readStringUnion(
      item,
      "value_type",
      "valueType",
      generativeParameterValueTypes
    );
    const role = readStringUnion(
      item,
      "role",
      "role",
      generativeParameterRoles
    );
    const defaultValue =
      readStringField(item, "default_value") ??
      readStringField(item, "defaultValue");
    const targetModules = readStringListField(
      item,
      "target_modules",
      "targetModules"
    );
    const rationale = readStringField(item, "rationale");

    if (
      !name ||
      !label ||
      !valueType ||
      !role ||
      !defaultValue ||
      targetModules.length === 0 ||
      !rationale
    ) {
      return [];
    }

    return [
      {
        name,
        label,
        valueType,
        role,
        defaultValue,
        bounds: readStringField(item, "bounds"),
        controlledBy:
          readStringField(item, "controlled_by") ??
          readStringField(item, "controlledBy"),
        targetModules,
        rationale
      }
    ];
  });
}

function readGenerativeEvolutionRuleSummaryList(
  value: unknown
): GenerativeEvolutionRuleSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const phase = readStringUnion(
      item,
      "phase",
      "phase",
      generativeEvolutionPhases
    );
    const trigger = readStringUnion(
      item,
      "trigger",
      "trigger",
      generativeEvolutionTriggers
    );
    const rule = readStringField(item, "rule");
    const affectedModules = readStringListField(
      item,
      "affected_modules",
      "affectedModules"
    );

    if (!phase || !trigger || !rule || affectedModules.length === 0) {
      return [];
    }

    return [
      {
        phase,
        trigger,
        rule,
        affectedModules,
        parameterChanges: readStringListField(
          item,
          "parameter_changes",
          "parameterChanges"
        ),
        safeguards: readStringListField(item, "safeguards", "safeguards")
      }
    ];
  });
}

function readGenerativeStructureHookSummaryList(
  value: unknown
): GenerativeStructureHookSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const hookId =
      readStringField(item, "hook_id") ?? readStringField(item, "hookId");
    const hookType = readStringUnion(
      item,
      "hook_type",
      "hookType",
      generativeHookTypes
    );
    const signal = readStringField(item, "signal");
    const targetModules = readStringListField(
      item,
      "target_modules",
      "targetModules"
    );
    const parameterMapping = readStringListField(
      item,
      "parameter_mapping",
      "parameterMapping"
    );
    const fallbackBehavior =
      readStringField(item, "fallback_behavior") ??
      readStringField(item, "fallbackBehavior");

    if (
      !hookId ||
      !hookType ||
      !signal ||
      targetModules.length === 0 ||
      parameterMapping.length === 0 ||
      !fallbackBehavior
    ) {
      return [];
    }

    return [
      {
        hookId,
        hookType,
        signal,
        targetModules,
        parameterMapping,
        fallbackBehavior
      }
    ];
  });
}

function readGenerativeFallbackBlueprintSummary(
  value: unknown
): GenerativeFallbackBlueprintSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const name = readStringField(value, "name");
  const architecture = readStringUnion(
    value,
    "architecture",
    "architecture",
    generativeArchitectures
  );
  const moduleKinds = readGenerativeModuleKindList(
    value.module_kinds ?? value.moduleKinds
  );
  const parameterReductions = readStringListField(
    value,
    "parameter_reductions",
    "parameterReductions"
  );
  const reason = readStringField(value, "reason");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !name ||
    !architecture ||
    moduleKinds.length === 0 ||
    parameterReductions.length === 0 ||
    !reason ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    name,
    architecture,
    moduleKinds,
    parameterReductions,
    reason,
    promptGuidance
  };
}

function readGenerativeModuleKindList(value: unknown): GenerativeModuleKind[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is GenerativeModuleKind =>
      typeof item === "string" &&
      generativeModuleKinds.includes(item as GenerativeModuleKind)
  );
}

export function readSemanticMotifSystemSummary(
  value: unknown
): SemanticMotifSystemSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const motifSystemName =
    readStringField(value, "motif_system_name") ??
    readStringField(value, "motifSystemName");
  const primaryMotifs = readSemanticMotifSummaryList(
    value.primary_motifs ?? value.primaryMotifs
  );
  const secondaryMotifs = readSemanticMotifSummaryList(
    value.secondary_motifs ?? value.secondaryMotifs
  );
  const motifHierarchy = readStringListField(
    value,
    "motif_hierarchy",
    "motifHierarchy"
  );
  const motifRecurrencePlan = readStringListField(
    value,
    "motif_recurrence_plan",
    "motifRecurrencePlan"
  );
  const motifTransformationPlan = readStringListField(
    value,
    "motif_transformation_plan",
    "motifTransformationPlan"
  );
  const motifToStructureMapping = readSemanticMotifStructureMappingSummaryList(
    value.motif_to_structure_mapping ?? value.motifToStructureMapping
  );
  const motifToCompositionMapping = readSemanticMotifCompositionMappingSummaryList(
    value.motif_to_composition_mapping ?? value.motifToCompositionMapping
  );
  const motifToNarrativeMapping = readSemanticMotifNarrativeMappingSummaryList(
    value.motif_to_narrative_mapping ?? value.motifToNarrativeMapping
  );
  const motifToParameterMapping = readSemanticMotifParameterMappingSummaryList(
    value.motif_to_parameter_mapping ?? value.motifToParameterMapping
  );
  const motifFallbackPlan = readSemanticMotifFallbackPlanSummary(
    value.motif_fallback_plan ?? value.motifFallbackPlan
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "semantic_motif_engine" ||
    !motifSystemName ||
    primaryMotifs.length === 0 ||
    secondaryMotifs.length === 0 ||
    motifHierarchy.length === 0 ||
    motifRecurrencePlan.length === 0 ||
    motifTransformationPlan.length === 0 ||
    motifToStructureMapping.length === 0 ||
    motifToCompositionMapping.length === 0 ||
    motifToNarrativeMapping.length === 0 ||
    motifToParameterMapping.length === 0 ||
    !motifFallbackPlan ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    motifSystemName,
    primaryMotifs,
    secondaryMotifs,
    motifHierarchy,
    motifRecurrencePlan,
    motifTransformationPlan,
    motifToStructureMapping,
    motifToCompositionMapping,
    motifToNarrativeMapping,
    motifToParameterMapping,
    coherenceRisks: readStringListField(
      value,
      "coherence_risks",
      "coherenceRisks"
    ),
    overuseRisks: readStringListField(value, "overuse_risks", "overuseRisks"),
    underuseRisks: readStringListField(value, "underuse_risks", "underuseRisks"),
    unsupportedSymbolicClaims: readStringListField(
      value,
      "unsupported_symbolic_claims",
      "unsupportedSymbolicClaims"
    ),
    motifFallbackPlan,
    unresolvedMotifGaps: readStringListField(
      value,
      "unresolved_motif_gaps",
      "unresolvedMotifGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const semanticMotifIds = [
  "seed",
  "spiral",
  "threshold",
  "mirror",
  "void",
  "center",
  "circumference",
  "axis",
  "descent",
  "ascent",
  "fragmentation",
  "reintegration",
  "wave",
  "lattice",
  "network",
  "pearl",
  "flame",
  "root",
  "tree",
  "vessel",
  "mandala",
  "grid",
  "swarm",
  "orbit",
  "pulse",
  "breath",
  "gate",
  "eye",
  "river",
  "constellation"
] as const satisfies readonly SemanticMotifId[];

const semanticMotifRoles = [
  "anchor",
  "threshold",
  "transformation",
  "connector",
  "counterpoint",
  "rhythm",
  "spatial_order",
  "material_signal",
  "fallback"
] as const satisfies readonly SemanticMotifRole[];

const semanticMotifHierarchyLevels = [
  "primary",
  "secondary",
  "supporting",
  "fallback"
] as const satisfies readonly SemanticMotifHierarchyLevel[];

function readSemanticMotifSummaryList(
  value: unknown
): SemanticMotifSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const motifId = readStringUnion(
      item,
      "motif_id",
      "motifId",
      semanticMotifIds
    );
    const label = readStringField(item, "label");
    const role = readStringUnion(item, "role", "role", semanticMotifRoles);
    const hierarchyLevel = readStringUnion(
      item,
      "hierarchy_level",
      "hierarchyLevel",
      semanticMotifHierarchyLevels
    );
    const rationale = readStringField(item, "rationale");
    const recurrenceGuidance = readStringListField(
      item,
      "recurrence_guidance",
      "recurrenceGuidance"
    );
    const transformationGuidance = readStringListField(
      item,
      "transformation_guidance",
      "transformationGuidance"
    );

    if (
      !motifId ||
      !label ||
      !role ||
      !hierarchyLevel ||
      !rationale ||
      recurrenceGuidance.length === 0 ||
      transformationGuidance.length === 0
    ) {
      return [];
    }

    return [
      {
        motifId,
        label,
        role,
        hierarchyLevel,
        rationale,
        recurrenceGuidance,
        transformationGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readSemanticMotifStructureMappingSummaryList(
  value: unknown
): SemanticMotifStructureMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const motifId = readStringUnion(
      item,
      "motif_id",
      "motifId",
      semanticMotifIds
    );
    const structuralBehavior =
      readStringField(item, "structural_behavior") ??
      readStringField(item, "structuralBehavior");

    if (!motifId || !structuralBehavior) {
      return [];
    }

    return [
      {
        motifId,
        proceduralFamilies: readProceduralFamilyList(
          item.procedural_families ?? item.proceduralFamilies
        ),
        generativeModuleIds: readStringListField(
          item,
          "generative_module_ids",
          "generativeModuleIds"
        ),
        generativeModuleKinds: readGenerativeModuleKindList(
          item.generative_module_kinds ?? item.generativeModuleKinds
        ),
        structuralBehavior,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readSemanticMotifCompositionMappingSummaryList(
  value: unknown
): SemanticMotifCompositionMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const motifId = readStringUnion(
      item,
      "motif_id",
      "motifId",
      semanticMotifIds
    );
    const compositionRole =
      readStringField(item, "composition_role") ??
      readStringField(item, "compositionRole");
    const spatialAnchor =
      readStringField(item, "spatial_anchor") ??
      readStringField(item, "spatialAnchor");
    const rhythmOrDensityGuidance =
      readStringField(item, "rhythm_or_density_guidance") ??
      readStringField(item, "rhythmOrDensityGuidance");

    if (!motifId || !compositionRole || !spatialAnchor || !rhythmOrDensityGuidance) {
      return [];
    }

    return [
      {
        motifId,
        compositionRole,
        spatialAnchor,
        rhythmOrDensityGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readSemanticMotifNarrativeMappingSummaryList(
  value: unknown
): SemanticMotifNarrativeMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const motifId = readStringUnion(
      item,
      "motif_id",
      "motifId",
      semanticMotifIds
    );
    const narrativeFunction =
      readStringField(item, "narrative_function") ??
      readStringField(item, "narrativeFunction");
    const phaseAlignment = readStringListField(
      item,
      "phase_alignment",
      "phaseAlignment"
    );

    if (!motifId || !narrativeFunction || phaseAlignment.length === 0) {
      return [];
    }

    return [
      {
        motifId,
        narrativeFunction,
        phaseAlignment,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readSemanticMotifParameterMappingSummaryList(
  value: unknown
): SemanticMotifParameterMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const motifId = readStringUnion(
      item,
      "motif_id",
      "motifId",
      semanticMotifIds
    );
    const parameterNames = readStringListField(
      item,
      "parameter_names",
      "parameterNames"
    );
    const parameterGuidance =
      readStringField(item, "parameter_guidance") ??
      readStringField(item, "parameterGuidance");

    if (!motifId || parameterNames.length === 0 || !parameterGuidance) {
      return [];
    }

    return [
      {
        motifId,
        parameterNames,
        parameterGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readSemanticMotifFallbackPlanSummary(
  value: unknown
): SemanticMotifFallbackPlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const fallbackPrimaryMotif = readStringUnion(
    value,
    "fallback_primary_motif",
    "fallbackPrimaryMotif",
    semanticMotifIds
  );
  const simplificationStrategy =
    readStringField(value, "simplification_strategy") ??
    readStringField(value, "simplificationStrategy");
  const preservedMeaning =
    readStringField(value, "preserved_meaning") ??
    readStringField(value, "preservedMeaning");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !fallbackPrimaryMotif ||
    !simplificationStrategy ||
    !preservedMeaning ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    fallbackPrimaryMotif,
    fallbackSecondaryMotifs: readSemanticMotifIdList(
      value.fallback_secondary_motifs ?? value.fallbackSecondaryMotifs
    ),
    simplificationStrategy,
    preservedMeaning,
    promptGuidance
  };
}

function readSemanticMotifIdList(value: unknown): SemanticMotifId[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is SemanticMotifId =>
      typeof item === "string" &&
      semanticMotifIds.includes(item as SemanticMotifId)
  );
}

export function readEmotionalConsistencyProfileSummary(
  value: unknown
): EmotionalConsistencyProfileSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryEmotionalTone = readStringUnion(
    value,
    "primary_emotional_tone",
    "primaryEmotionalTone",
    emotionalTones
  );
  const secondaryEmotionalTones = readEmotionalToneList(
    value.secondary_emotional_tones ?? value.secondaryEmotionalTones
  );
  const emotionalArc = readStringListField(
    value,
    "emotional_arc",
    "emotionalArc"
  );
  const emotionalPhaseMapping = readEmotionalPhaseMappingSummaryList(
    value.emotional_phase_mapping ?? value.emotionalPhaseMapping
  );
  const emotionalToNarrativeMapping =
    readEmotionalNarrativeMappingSummaryList(
      value.emotional_to_narrative_mapping ?? value.emotionalToNarrativeMapping
    );
  const emotionalToMotifMapping = readEmotionalMotifMappingSummaryList(
    value.emotional_to_motif_mapping ?? value.emotionalToMotifMapping
  );
  const emotionalToCompositionMapping =
    readEmotionalCompositionMappingSummaryList(
      value.emotional_to_composition_mapping ??
        value.emotionalToCompositionMapping
    );
  const emotionalToStructureMapping = readEmotionalStructureMappingSummaryList(
    value.emotional_to_structure_mapping ?? value.emotionalToStructureMapping
  );
  const emotionalToParameterMapping = readEmotionalParameterMappingSummaryList(
    value.emotional_to_parameter_mapping ?? value.emotionalToParameterMapping
  );
  const colorLightGuidance = readStringListField(
    value,
    "color_light_guidance",
    "colorLightGuidance"
  );
  const motionRhythmGuidance = readStringListField(
    value,
    "motion_rhythm_guidance",
    "motionRhythmGuidance"
  );
  const emotionalCoherenceScore =
    readFiniteNumberField(value, "emotional_coherence_score") ??
    readFiniteNumberField(value, "emotionalCoherenceScore");
  const fallbackEmotionalStrategy = readEmotionalFallbackStrategySummary(
    value.fallback_emotional_strategy ?? value.fallbackEmotionalStrategy
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "emotional_consistency_engine" ||
    !primaryEmotionalTone ||
    secondaryEmotionalTones.length === 0 ||
    emotionalArc.length === 0 ||
    emotionalPhaseMapping.length === 0 ||
    emotionalToNarrativeMapping.length === 0 ||
    emotionalToMotifMapping.length === 0 ||
    emotionalToCompositionMapping.length === 0 ||
    emotionalToStructureMapping.length === 0 ||
    emotionalToParameterMapping.length === 0 ||
    colorLightGuidance.length === 0 ||
    motionRhythmGuidance.length === 0 ||
    emotionalCoherenceScore === null ||
    emotionalCoherenceScore < 0 ||
    emotionalCoherenceScore > 100 ||
    !fallbackEmotionalStrategy ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    primaryEmotionalTone,
    secondaryEmotionalTones,
    emotionalArc,
    emotionalPhaseMapping,
    emotionalToNarrativeMapping,
    emotionalToMotifMapping,
    emotionalToCompositionMapping,
    emotionalToStructureMapping,
    emotionalToParameterMapping,
    colorLightGuidance,
    motionRhythmGuidance,
    audiovisualGuidance: readStringListField(
      value,
      "audiovisual_guidance",
      "audiovisualGuidance"
    ),
    emotionalCoherenceScore,
    emotionalTensions: readStringListField(
      value,
      "emotional_tensions",
      "emotionalTensions"
    ),
    mismatchRisks: readStringListField(value, "mismatch_risks", "mismatchRisks"),
    flatteningRisks: readStringListField(
      value,
      "flattening_risks",
      "flatteningRisks"
    ),
    overIntensityRisks: readStringListField(
      value,
      "over_intensity_risks",
      "overIntensityRisks"
    ),
    underIntensityRisks: readStringListField(
      value,
      "under_intensity_risks",
      "underIntensityRisks"
    ),
    fallbackEmotionalStrategy,
    unresolvedEmotionalGaps: readStringListField(
      value,
      "unresolved_emotional_gaps",
      "unresolvedEmotionalGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const emotionalTones = [
  "awe",
  "wonder",
  "mystery",
  "serenity",
  "tension",
  "rupture",
  "grief",
  "dissolution",
  "suspension",
  "emergence",
  "ecstasy",
  "clarity",
  "intimacy",
  "vastness",
  "ritual solemnity",
  "playful curiosity",
  "dread",
  "release",
  "transformation",
  "integration"
] as const satisfies readonly EmotionalTone[];

const emotionalIntensities = [
  "low",
  "medium",
  "high",
  "variable"
] as const satisfies readonly EmotionalIntensity[];

function readEmotionalToneList(value: unknown): EmotionalTone[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is EmotionalTone =>
      typeof item === "string" && emotionalTones.includes(item as EmotionalTone)
  );
}

function readEmotionalPhaseMappingSummaryList(
  value: unknown
): EmotionalPhaseMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const phase = readStringUnion(
      item,
      "phase",
      "phase",
      symbolicNarrativePhaseNames
    );
    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const intensity = readStringUnion(
      item,
      "intensity",
      "intensity",
      emotionalIntensities
    );
    const guidance = readStringField(item, "guidance");

    if (!phase || !tone || !intensity || !guidance) {
      return [];
    }

    return [
      {
        phase,
        tone,
        intensity,
        guidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalNarrativeMappingSummaryList(
  value: unknown
): EmotionalNarrativeMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const narrativePhase = readStringUnion(
      item,
      "narrative_phase",
      "narrativePhase",
      symbolicNarrativePhaseNames
    );
    const narrativeFunction =
      readStringField(item, "narrative_function") ??
      readStringField(item, "narrativeFunction");

    if (!tone || !narrativePhase || !narrativeFunction) {
      return [];
    }

    return [
      {
        tone,
        narrativePhase,
        narrativeFunction,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalMotifMappingSummaryList(
  value: unknown
): EmotionalMotifMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const rawMotifId = item.motif_id ?? item.motifId;
    const motifId =
      rawMotifId === null || rawMotifId === undefined
        ? null
        : readStringUnion(item, "motif_id", "motifId", semanticMotifIds);
    const emotionalFunction =
      readStringField(item, "emotional_function") ??
      readStringField(item, "emotionalFunction");

    if (
      !tone ||
      (rawMotifId !== null && rawMotifId !== undefined && motifId === null) ||
      !emotionalFunction
    ) {
      return [];
    }

    return [
      {
        tone,
        motifId,
        emotionalFunction,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalCompositionMappingSummaryList(
  value: unknown
): EmotionalCompositionMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const rawCompositionPattern =
      item.composition_pattern ?? item.compositionPattern;
    const compositionPattern =
      rawCompositionPattern === null || rawCompositionPattern === undefined
        ? null
        : readStringUnion(
            item,
            "composition_pattern",
            "compositionPattern",
            creativeCompositionPatterns
          );
    const compositionGuidance =
      readStringField(item, "composition_guidance") ??
      readStringField(item, "compositionGuidance");
    const spatialOrDensityGuidance =
      readStringField(item, "spatial_or_density_guidance") ??
      readStringField(item, "spatialOrDensityGuidance");

    if (
      !tone ||
      (rawCompositionPattern !== null &&
        rawCompositionPattern !== undefined &&
        compositionPattern === null) ||
      !compositionGuidance ||
      !spatialOrDensityGuidance
    ) {
      return [];
    }

    return [
      {
        tone,
        compositionPattern,
        compositionGuidance,
        spatialOrDensityGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalStructureMappingSummaryList(
  value: unknown
): EmotionalStructureMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const structuralGuidance =
      readStringField(item, "structural_guidance") ??
      readStringField(item, "structuralGuidance");

    if (!tone || !structuralGuidance) {
      return [];
    }

    return [
      {
        tone,
        proceduralFamilies: readProceduralFamilyList(
          item.procedural_families ?? item.proceduralFamilies
        ),
        generativeModuleKinds: readGenerativeModuleKindList(
          item.generative_module_kinds ?? item.generativeModuleKinds
        ),
        structuralGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalParameterMappingSummaryList(
  value: unknown
): EmotionalParameterMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const tone = readStringUnion(item, "tone", "tone", emotionalTones);
    const parameterNames = readStringListField(
      item,
      "parameter_names",
      "parameterNames"
    );
    const parameterGuidance =
      readStringField(item, "parameter_guidance") ??
      readStringField(item, "parameterGuidance");

    if (!tone || parameterNames.length === 0 || !parameterGuidance) {
      return [];
    }

    return [
      {
        tone,
        parameterNames,
        parameterGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readEmotionalFallbackStrategySummary(
  value: unknown
): EmotionalFallbackStrategySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const fallbackPrimaryTone = readStringUnion(
    value,
    "fallback_primary_tone",
    "fallbackPrimaryTone",
    emotionalTones
  );
  const simplificationStrategy =
    readStringField(value, "simplification_strategy") ??
    readStringField(value, "simplificationStrategy");
  const preservedFeeling =
    readStringField(value, "preserved_feeling") ??
    readStringField(value, "preservedFeeling");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !fallbackPrimaryTone ||
    !simplificationStrategy ||
    !preservedFeeling ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    fallbackPrimaryTone,
    fallbackSecondaryTones: readEmotionalToneList(
      value.fallback_secondary_tones ?? value.fallbackSecondaryTones
    ),
    simplificationStrategy,
    preservedFeeling,
    promptGuidance
  };
}

export function readCrossModalityCompositionProfileSummary(
  value: unknown
): CrossModalityCompositionProfileSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const modalityPattern = readStringUnion(
    value,
    "modality_pattern",
    "modalityPattern",
    crossModalityPatterns
  );
  const primaryModality = readStringUnion(
    value,
    "primary_modality",
    "primaryModality",
    crossModalityChannels
  );
  const supportingModalities = readCrossModalityChannelList(
    value.supporting_modalities ?? value.supportingModalities
  );
  const modalityHierarchy = readCrossModalityRoleSummaryList(
    value.modality_hierarchy ?? value.modalityHierarchy
  );
  const visualRole =
    readStringField(value, "visual_role") ?? readStringField(value, "visualRole");
  const motionRole =
    readStringField(value, "motion_role") ?? readStringField(value, "motionRole");
  const rhythmRole =
    readStringField(value, "rhythm_role") ?? readStringField(value, "rhythmRole");
  const structureRole =
    readStringField(value, "structure_role") ??
    readStringField(value, "structureRole");
  const motifRole =
    readStringField(value, "motif_role") ?? readStringField(value, "motifRole");
  const emotionRole =
    readStringField(value, "emotion_role") ??
    readStringField(value, "emotionRole");
  const modalitySynchronizationPlan = readStringListField(
    value,
    "modality_synchronization_plan",
    "modalitySynchronizationPlan"
  );
  const visualToAudioMapping = readCrossModalityMappingSummaryList(
    value.visual_to_audio_mapping ?? value.visualToAudioMapping
  );
  const audioToMotionMapping = readCrossModalityMappingSummaryList(
    value.audio_to_motion_mapping ?? value.audioToMotionMapping
  );
  const motionToStructureMapping = readCrossModalityMappingSummaryList(
    value.motion_to_structure_mapping ?? value.motionToStructureMapping
  );
  const motifToModalityMapping = readCrossModalityMappingSummaryList(
    value.motif_to_modality_mapping ?? value.motifToModalityMapping
  );
  const emotionalToModalityMapping = readCrossModalityMappingSummaryList(
    value.emotional_to_modality_mapping ?? value.emotionalToModalityMapping
  );
  const temporalCuePlan = readCrossModalityTemporalCueSummaryList(
    value.temporal_cue_plan ?? value.temporalCuePlan
  );
  const contrastBalancePlan = readStringListField(
    value,
    "contrast_balance_plan",
    "contrastBalancePlan"
  );
  const fallbackMultimodalStrategy = readCrossModalityFallbackStrategySummary(
    value.fallback_multimodal_strategy ?? value.fallbackMultimodalStrategy
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "cross_modality_composer" ||
    !modalityPattern ||
    !primaryModality ||
    supportingModalities.length === 0 ||
    modalityHierarchy.length === 0 ||
    !visualRole ||
    !motionRole ||
    !rhythmRole ||
    !structureRole ||
    !motifRole ||
    !emotionRole ||
    modalitySynchronizationPlan.length === 0 ||
    motionToStructureMapping.length === 0 ||
    motifToModalityMapping.length === 0 ||
    emotionalToModalityMapping.length === 0 ||
    temporalCuePlan.length === 0 ||
    contrastBalancePlan.length === 0 ||
    !fallbackMultimodalStrategy ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    modalityPattern,
    primaryModality,
    supportingModalities,
    modalityHierarchy,
    visualRole,
    motionRole,
    audioRole:
      readStringField(value, "audio_role") ?? readStringField(value, "audioRole"),
    rhythmRole,
    cameraViewpointRole:
      readStringField(value, "camera_viewpoint_role") ??
      readStringField(value, "cameraViewpointRole"),
    structureRole,
    motifRole,
    emotionRole,
    modalitySynchronizationPlan,
    visualToAudioMapping,
    audioToMotionMapping,
    motionToStructureMapping,
    motifToModalityMapping,
    emotionalToModalityMapping,
    temporalCuePlan,
    contrastBalancePlan,
    modalityConflicts: readStringListField(
      value,
      "modality_conflicts",
      "modalityConflicts"
    ),
    overloadRisks: readStringListField(value, "overload_risks", "overloadRisks"),
    underuseRisks: readStringListField(value, "underuse_risks", "underuseRisks"),
    fallbackMultimodalStrategy,
    unresolvedModalityGaps: readStringListField(
      value,
      "unresolved_modality_gaps",
      "unresolvedModalityGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const crossModalityChannels = [
  "visual_structure",
  "motion",
  "audio",
  "rhythm",
  "camera",
  "structure",
  "motif",
  "emotion",
  "interaction"
] as const satisfies readonly CrossModalityChannel[];

const crossModalityPatterns = [
  "visual_led_composition",
  "audio_reactive_composition",
  "motion_led_transformation",
  "rhythm_led_scene_evolution",
  "camera_led_immersion",
  "motif_led_symbolic_recurrence",
  "structure_led_procedural_evolution",
  "emotion_led_modulation",
  "balanced_audiovisual_composition",
  "minimal_visual_strong_sonic_cueing",
  "dense_visual_restrained_audio",
  "ritual_pulse_geometry_synchronization",
  "fragmentation_reassembly_visual_motion_layers"
] as const satisfies readonly CrossModalityPattern[];

const crossModalityRolePriorities = [
  "primary",
  "secondary",
  "supporting",
  "fallback"
] as const satisfies readonly CrossModalityRoleSummary["priority"][];

function readCrossModalityChannelList(value: unknown): CrossModalityChannel[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is CrossModalityChannel =>
      typeof item === "string" &&
      crossModalityChannels.includes(item as CrossModalityChannel)
  );
}

function readCrossModalityRoleSummaryList(
  value: unknown
): CrossModalityRoleSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const modality = readStringUnion(
      item,
      "modality",
      "modality",
      crossModalityChannels
    );
    const role = readStringField(item, "role");
    const priority = readStringUnion(
      item,
      "priority",
      "priority",
      crossModalityRolePriorities
    );

    if (!modality || !role || !priority) {
      return [];
    }

    return [
      {
        modality,
        role,
        priority,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCrossModalityMappingSummaryList(
  value: unknown
): CrossModalityMappingSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const sourceModality = readStringUnion(
      item,
      "source_modality",
      "sourceModality",
      crossModalityChannels
    );
    const targetModality = readStringUnion(
      item,
      "target_modality",
      "targetModality",
      crossModalityChannels
    );
    const mapping = readStringField(item, "mapping");
    const cues = readStringListField(item, "cues", "cues");
    const rawMotifId = item.motif_id ?? item.motifId;
    const motifId =
      rawMotifId === null || rawMotifId === undefined
        ? null
        : readStringUnion(item, "motif_id", "motifId", semanticMotifIds);
    const rawEmotionalTone = item.emotional_tone ?? item.emotionalTone;
    const emotionalTone =
      rawEmotionalTone === null || rawEmotionalTone === undefined
        ? null
        : readStringUnion(
            item,
            "emotional_tone",
            "emotionalTone",
            emotionalTones
          );

    if (
      !sourceModality ||
      !targetModality ||
      !mapping ||
      cues.length === 0 ||
      (rawMotifId !== null && rawMotifId !== undefined && motifId === null) ||
      (rawEmotionalTone !== null &&
        rawEmotionalTone !== undefined &&
        emotionalTone === null)
    ) {
      return [];
    }

    return [
      {
        sourceModality,
        targetModality,
        mapping,
        cues,
        motifId,
        emotionalTone,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCrossModalityTemporalCueSummaryList(
  value: unknown
): CrossModalityTemporalCueSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const phase = readStringUnion(
      item,
      "phase",
      "phase",
      symbolicNarrativePhaseNames
    );
    const cue = readStringField(item, "cue");
    const modalities = readCrossModalityChannelList(item.modalities);
    const timingGuidance =
      readStringField(item, "timing_guidance") ??
      readStringField(item, "timingGuidance");

    if (!phase || !cue || modalities.length === 0 || !timingGuidance) {
      return [];
    }

    return [
      {
        phase,
        cue,
        modalities,
        timingGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCrossModalityFallbackStrategySummary(
  value: unknown
): CrossModalityFallbackStrategySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const fallbackPattern = readStringUnion(
    value,
    "fallback_pattern",
    "fallbackPattern",
    crossModalityPatterns
  );
  const preservedModalities = readCrossModalityChannelList(
    value.preserved_modalities ?? value.preservedModalities
  );
  const simplificationStrategy =
    readStringField(value, "simplification_strategy") ??
    readStringField(value, "simplificationStrategy");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !fallbackPattern ||
    preservedModalities.length === 0 ||
    !simplificationStrategy ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    fallbackPattern,
    preservedModalities,
    reducedModalities: readCrossModalityChannelList(
      value.reduced_modalities ?? value.reducedModalities
    ),
    simplificationStrategy,
    promptGuidance
  };
}

export function readAudioVisualSceneProfileSummary(
  value: unknown
): AudioVisualSceneProfileSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const scenePattern = readStringUnion(
    value,
    "scene_pattern",
    "scenePattern",
    audioVisualScenePatterns
  );
  const sceneArc =
    readStringField(value, "scene_arc") ?? readStringField(value, "sceneArc");
  const scenePhases = readAudioVisualScenePhaseSummaryList(
    value.scene_phases ?? value.scenePhases
  );
  const openingScene = readAudioVisualScenePhaseSummary(
    value.opening_scene ?? value.openingScene
  );
  const developmentScene = readAudioVisualScenePhaseSummary(
    value.development_scene ?? value.developmentScene
  );
  const thresholdScene = readAudioVisualScenePhaseSummary(
    value.threshold_scene ?? value.thresholdScene
  );
  const climaxScene = readAudioVisualScenePhaseSummary(
    value.climax_scene ?? value.climaxScene
  );
  const resolutionScene = readAudioVisualScenePhaseSummary(
    value.resolution_scene ?? value.resolutionScene
  );
  const cuePlan = readAudioVisualSceneCueSummaryList(
    value.cue_plan ?? value.cuePlan
  );
  const transitionPlan = readAudioVisualSceneTransitionSummaryList(
    value.transition_plan ?? value.transitionPlan
  );
  const climaxStrategy =
    readStringField(value, "climax_strategy") ??
    readStringField(value, "climaxStrategy");
  const resolutionStrategy =
    readStringField(value, "resolution_strategy") ??
    readStringField(value, "resolutionStrategy");
  const visualTimingPlan = readStringListField(
    value,
    "visual_timing_plan",
    "visualTimingPlan"
  );
  const motionTimingPlan = readStringListField(
    value,
    "motion_timing_plan",
    "motionTimingPlan"
  );
  const rhythmTimingPlan = readStringListField(
    value,
    "rhythm_timing_plan",
    "rhythmTimingPlan"
  );
  const motifTimingPlan = readStringListField(
    value,
    "motif_timing_plan",
    "motifTimingPlan"
  );
  const emotionalTimingPlan = readStringListField(
    value,
    "emotional_timing_plan",
    "emotionalTimingPlan"
  );
  const proceduralTimingPlan = readStringListField(
    value,
    "procedural_timing_plan",
    "proceduralTimingPlan"
  );
  const synchronizationCheckpoints = readStringListField(
    value,
    "synchronization_checkpoints",
    "synchronizationCheckpoints"
  );
  const sceneContrastPlan = readStringListField(
    value,
    "scene_contrast_plan",
    "sceneContrastPlan"
  );
  const sceneContinuityPlan = readStringListField(
    value,
    "scene_continuity_plan",
    "sceneContinuityPlan"
  );
  const fallbackSceneStrategy = readAudioVisualFallbackSceneStrategySummary(
    value.fallback_scene_strategy ?? value.fallbackSceneStrategy
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "audio_visual_scene_system" ||
    !scenePattern ||
    !sceneArc ||
    scenePhases.length === 0 ||
    !openingScene ||
    !developmentScene ||
    !thresholdScene ||
    !climaxScene ||
    !resolutionScene ||
    cuePlan.length === 0 ||
    transitionPlan.length === 0 ||
    !climaxStrategy ||
    !resolutionStrategy ||
    visualTimingPlan.length === 0 ||
    motionTimingPlan.length === 0 ||
    rhythmTimingPlan.length === 0 ||
    motifTimingPlan.length === 0 ||
    emotionalTimingPlan.length === 0 ||
    proceduralTimingPlan.length === 0 ||
    synchronizationCheckpoints.length === 0 ||
    sceneContrastPlan.length === 0 ||
    sceneContinuityPlan.length === 0 ||
    !fallbackSceneStrategy ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    scenePattern,
    sceneArc,
    scenePhases,
    openingScene,
    developmentScene,
    thresholdScene,
    climaxScene,
    resolutionScene,
    cuePlan,
    transitionPlan,
    climaxStrategy,
    resolutionStrategy,
    visualTimingPlan,
    motionTimingPlan,
    audioTimingPlan: readStringListField(
      value,
      "audio_timing_plan",
      "audioTimingPlan"
    ),
    rhythmTimingPlan,
    cameraTimingPlan: readStringListField(
      value,
      "camera_timing_plan",
      "cameraTimingPlan"
    ),
    motifTimingPlan,
    emotionalTimingPlan,
    proceduralTimingPlan,
    synchronizationCheckpoints,
    sceneContrastPlan,
    sceneContinuityPlan,
    sceneRisks: readStringListField(value, "scene_risks", "sceneRisks"),
    pacingRisks: readStringListField(value, "pacing_risks", "pacingRisks"),
    overloadRisks: readStringListField(value, "overload_risks", "overloadRisks"),
    fallbackSceneStrategy,
    unresolvedSceneGaps: readStringListField(
      value,
      "unresolved_scene_gaps",
      "unresolvedSceneGaps"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const audioVisualScenePatterns = [
  "seed_to_expansion",
  "descent_to_return",
  "fragmentation_to_reintegration",
  "threshold_crossing",
  "spiral_ascent",
  "chaos_to_order",
  "void_to_emergence",
  "contraction_to_release",
  "ritual_opening_to_climax",
  "wave_build_and_collapse",
  "constellation_activation",
  "mirror_inversion",
  "pulse_escalation",
  "calm_expansion_after_rupture"
] as const satisfies readonly AudioVisualScenePattern[];

export function readArtifactPlanSummary(
  value: unknown
): ArtifactPlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryArtifactIntent =
    readStringField(value, "primary_artifact_intent") ??
    readStringField(value, "primaryArtifactIntent");
  const artifactType = readStringUnion(
    value,
    "artifact_type",
    "artifactType",
    artifactTypes
  );
  const artifactFamily = readStringUnion(
    value,
    "artifact_family",
    "artifactFamily",
    artifactFamilies
  );
  const requiredComponents = readStringListField(
    value,
    "required_components",
    "requiredComponents"
  );
  const expectedOutputStructure = readStringListField(
    value,
    "expected_output_structure",
    "expectedOutputStructure"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_planner" ||
    !primaryArtifactIntent ||
    !artifactType ||
    !artifactFamily ||
    requiredComponents.length === 0 ||
    expectedOutputStructure.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    primaryArtifactIntent,
    artifactType,
    artifactFamily,
    requiredComponents,
    runtimeRequirements: readStringListField(
      value,
      "runtime_requirements",
      "runtimeRequirements"
    ),
    creativeDependencies: readStringListField(
      value,
      "creative_dependencies",
      "creativeDependencies"
    ),
    generativeDependencies: readStringListField(
      value,
      "generative_dependencies",
      "generativeDependencies"
    ),
    expectedOutputStructure,
    implementationRisks: readStringListField(
      value,
      "implementation_risks",
      "implementationRisks"
    ),
    missingInformation: readStringListField(
      value,
      "missing_information",
      "missingInformation"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readArtifactDependencyGraphSummary(
  value: unknown
): ArtifactDependencyGraphSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const primaryArtifactNodeId =
    readStringField(value, "primary_artifact_node_id") ??
    readStringField(value, "primaryArtifactNodeId");
  const artifactNodes = readArtifactDependencyNodeSummaryList(
    value.artifact_nodes ?? value.artifactNodes
  );
  const dependencyEdges = readArtifactDependencyEdgeSummaryList(
    value.dependency_edges ?? value.dependencyEdges
  );
  const requiredUpstreamMetadata = readStringListField(
    value,
    "required_upstream_metadata",
    "requiredUpstreamMetadata"
  );
  const downstreamConsumers = readStringListField(
    value,
    "downstream_consumers",
    "downstreamConsumers"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "artifact_dependency_graph" ||
    !primaryArtifactNodeId ||
    artifactNodes.length === 0 ||
    requiredUpstreamMetadata.length === 0 ||
    downstreamConsumers.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    primaryArtifactNodeId,
    artifactNodes,
    dependencyEdges,
    requiredUpstreamMetadata,
    optionalUpstreamMetadata: readStringListField(
      value,
      "optional_upstream_metadata",
      "optionalUpstreamMetadata"
    ),
    blockingDependencies: readStringListField(
      value,
      "blocking_dependencies",
      "blockingDependencies"
    ),
    softDependencies: readStringListField(
      value,
      "soft_dependencies",
      "softDependencies"
    ),
    runtimeFacingDependencies: readStringListField(
      value,
      "runtime_facing_dependencies",
      "runtimeFacingDependencies"
    ),
    promptFacingDependencies: readStringListField(
      value,
      "prompt_facing_dependencies",
      "promptFacingDependencies"
    ),
    downstreamConsumers,
    missingDependencyRisks: readStringListField(
      value,
      "missing_dependency_risks",
      "missingDependencyRisks"
    ),
    dependencyConflicts: readStringListField(
      value,
      "dependency_conflicts",
      "dependencyConflicts"
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readArtifactDependencyNodeSummaryList(
  value: unknown
): ArtifactDependencyNodeSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const node = readArtifactDependencyNodeSummary(item);
    return node ? [node] : [];
  });
}

function readArtifactDependencyNodeSummary(
  value: unknown
): ArtifactDependencyNodeSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const nodeId =
    readStringField(value, "node_id") ?? readStringField(value, "nodeId");
  const label = readStringField(value, "label");
  const nodeType = readStringUnion(
    value,
    "node_type",
    "nodeType",
    dependencyNodeTypes
  );
  const status = readStringUnion(
    value,
    "status",
    "status",
    dependencyNodeStatuses
  );
  const summary = readStringField(value, "summary");

  if (!nodeId || !label || !nodeType || !status || !summary) {
    return null;
  }

  return {
    nodeId,
    label,
    nodeType,
    status,
    summary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readArtifactDependencyEdgeSummaryList(
  value: unknown
): ArtifactDependencyEdgeSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const edge = readArtifactDependencyEdgeSummary(item);
    return edge ? [edge] : [];
  });
}

function readArtifactDependencyEdgeSummary(
  value: unknown
): ArtifactDependencyEdgeSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const sourceNodeId =
    readStringField(value, "source_node_id") ??
    readStringField(value, "sourceNodeId");
  const targetNodeId =
    readStringField(value, "target_node_id") ??
    readStringField(value, "targetNodeId");
  const relationship = readStringUnion(
    value,
    "relationship",
    "relationship",
    dependencyRelationships
  );
  const strength = readStringUnion(
    value,
    "strength",
    "strength",
    dependencyStrengths
  );
  const rationale = readStringField(value, "rationale");

  if (!sourceNodeId || !targetNodeId || !relationship || !strength || !rationale) {
    return null;
  }

  return {
    sourceNodeId,
    targetNodeId,
    relationship,
    strength,
    rationale
  };
}

const artifactTypes = [
  "runnable_code",
  "design_spec",
  "explanation",
  "debug_patch",
  "review_report",
  "refinement_patch",
  "preview_request"
] as const satisfies readonly ArtifactType[];

const artifactFamilies = [
  "p5_sketch",
  "three_scene",
  "react_three_fiber_scene",
  "glsl_shader",
  "hydra_patch",
  "tone_sketch",
  "canvas_sketch",
  "audiovisual_scene",
  "generative_artifact",
  "multimodal_reference_artifact",
  "creative_coding_response"
] as const satisfies readonly ArtifactFamily[];

const dependencyNodeTypes = [
  "planned_artifact",
  "required_component",
  "runtime_requirement",
  "creative_metadata",
  "generative_metadata",
  "output_structure",
  "prompt_guidance",
  "downstream_consumer"
] as const satisfies readonly DependencyNodeType[];

const dependencyNodeStatuses = [
  "available",
  "inferred",
  "missing"
] as const satisfies readonly DependencyNodeStatus[];

const dependencyRelationships = [
  "requires",
  "informs",
  "blocks",
  "soft_informs",
  "feeds_prompt",
  "consumed_by"
] as const satisfies readonly DependencyRelationship[];

const dependencyStrengths = [
  "required",
  "optional",
  "blocking",
  "soft"
] as const satisfies readonly DependencyStrength[];

const audioVisualCueTypes = [
  "visual",
  "motion",
  "audio",
  "rhythm",
  "camera",
  "motif",
  "emotion",
  "procedural",
  "synchronization"
] as const satisfies readonly AudioVisualCueType[];

function readAudioVisualScenePhaseSummary(
  value: unknown
): AudioVisualScenePhaseSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const phase = readStringUnion(
    value,
    "phase",
    "phase",
    symbolicNarrativePhaseNames
  );
  const title = readStringField(value, "title");
  const sceneFunction =
    readStringField(value, "scene_function") ??
    readStringField(value, "sceneFunction");
  const visualState =
    readStringField(value, "visual_state") ??
    readStringField(value, "visualState");
  const motionState =
    readStringField(value, "motion_state") ??
    readStringField(value, "motionState");
  const rhythmState =
    readStringField(value, "rhythm_state") ??
    readStringField(value, "rhythmState");
  const motifState =
    readStringField(value, "motif_state") ??
    readStringField(value, "motifState");
  const emotionalState =
    readStringField(value, "emotional_state") ??
    readStringField(value, "emotionalState");
  const proceduralState =
    readStringField(value, "procedural_state") ??
    readStringField(value, "proceduralState");
  const cueIds = readStringListField(value, "cue_ids", "cueIds");
  const transitionOut =
    readStringField(value, "transition_out") ??
    readStringField(value, "transitionOut");

  if (
    !phase ||
    !title ||
    !sceneFunction ||
    !visualState ||
    !motionState ||
    !rhythmState ||
    !motifState ||
    !emotionalState ||
    !proceduralState ||
    cueIds.length === 0 ||
    !transitionOut
  ) {
    return null;
  }

  return {
    phase,
    title,
    sceneFunction,
    visualState,
    motionState,
    audioState:
      readStringField(value, "audio_state") ??
      readStringField(value, "audioState"),
    rhythmState,
    cameraState:
      readStringField(value, "camera_state") ??
      readStringField(value, "cameraState"),
    motifState,
    emotionalState,
    proceduralState,
    cueIds,
    transitionOut,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

function readAudioVisualScenePhaseSummaryList(
  value: unknown
): AudioVisualScenePhaseSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const phase = readAudioVisualScenePhaseSummary(item);
    return phase ? [phase] : [];
  });
}

function readAudioVisualSceneCueSummaryList(
  value: unknown
): AudioVisualSceneCueSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const cueId =
      readStringField(item, "cue_id") ?? readStringField(item, "cueId");
    const phase = readStringUnion(
      item,
      "phase",
      "phase",
      symbolicNarrativePhaseNames
    );
    const cueType = readStringUnion(
      item,
      "cue_type",
      "cueType",
      audioVisualCueTypes
    );
    const description = readStringField(item, "description");
    const timing = readStringField(item, "timing");
    const modalities = readCrossModalityChannelList(item.modalities);

    if (
      !cueId ||
      !phase ||
      !cueType ||
      !description ||
      !timing ||
      modalities.length === 0
    ) {
      return [];
    }

    return [
      {
        cueId,
        phase,
        cueType,
        description,
        timing,
        modalities,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readAudioVisualSceneTransitionSummaryList(
  value: unknown
): AudioVisualSceneTransitionSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const fromPhase = readStringUnion(
      item,
      "from_phase",
      "fromPhase",
      symbolicNarrativePhaseNames
    );
    const toPhase = readStringUnion(
      item,
      "to_phase",
      "toPhase",
      symbolicNarrativePhaseNames
    );
    const transition = readStringField(item, "transition");
    const visualMotionGuidance =
      readStringField(item, "visual_motion_guidance") ??
      readStringField(item, "visualMotionGuidance");
    const continuityGuidance =
      readStringField(item, "continuity_guidance") ??
      readStringField(item, "continuityGuidance");

    if (
      !fromPhase ||
      !toPhase ||
      !transition ||
      !visualMotionGuidance ||
      !continuityGuidance
    ) {
      return [];
    }

    return [
      {
        fromPhase,
        toPhase,
        transition,
        visualMotionGuidance,
        audioRhythmGuidance:
          readStringField(item, "audio_rhythm_guidance") ??
          readStringField(item, "audioRhythmGuidance"),
        continuityGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readAudioVisualFallbackSceneStrategySummary(
  value: unknown
): AudioVisualFallbackSceneStrategySummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const fallbackPattern = readStringUnion(
    value,
    "fallback_pattern",
    "fallbackPattern",
    audioVisualScenePatterns
  );
  const preservedPhases = readAudioVisualScenePhaseNameList(
    value.preserved_phases ?? value.preservedPhases
  );
  const simplificationStrategy =
    readStringField(value, "simplification_strategy") ??
    readStringField(value, "simplificationStrategy");
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    !fallbackPattern ||
    preservedPhases.length === 0 ||
    !simplificationStrategy ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    fallbackPattern,
    preservedPhases,
    reducedElements: readStringListField(
      value,
      "reduced_elements",
      "reducedElements"
    ),
    simplificationStrategy,
    promptGuidance
  };
}

function readAudioVisualScenePhaseNameList(
  value: unknown
): SymbolicNarrativePhaseName[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (item): item is SymbolicNarrativePhaseName =>
      typeof item === "string" &&
      symbolicNarrativePhaseNames.includes(item as SymbolicNarrativePhaseName)
  );
}

export function readCreativeReasoningSummary(
  value: unknown
): CreativeReasoningSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const recommendedCreativeDirection =
    readStringField(value, "recommended_creative_direction") ??
    readStringField(value, "recommendedCreativeDirection");
  const reasoningPath = readCreativeReasoningStepSummaryList(
    value.reasoning_path ?? value.reasoningPath
  );
  const evidenceChain = readCreativeReasoningEvidenceSummaryList(
    value.evidence_chain ?? value.evidenceChain
  );
  const strongestSupportingSignals = readStringListField(
    value,
    "strongest_supporting_signals",
    "strongestSupportingSignals"
  );
  const unresolvedDecisions = readStringListField(
    value,
    "unresolved_decisions",
    "unresolvedDecisions"
  );
  const implementationGuidance = readStringListField(
    value,
    "implementation_guidance",
    "implementationGuidance"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");

  if (
    role !== "creative_reasoning_engine" ||
    !recommendedCreativeDirection ||
    reasoningPath.length === 0 ||
    evidenceChain.length === 0 ||
    strongestSupportingSignals.length === 0 ||
    unresolvedDecisions.length === 0 ||
    implementationGuidance.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    recommendedCreativeDirection,
    reasoningPath,
    evidenceChain,
    strongestSupportingSignals,
    rejectedAlternatives: readCreativeRejectedAlternativeSummaryList(
      value.rejected_alternatives ?? value.rejectedAlternatives
    ),
    unresolvedDecisions,
    implementationGuidance,
    promptGuidance,
    hitlQuestions: readStringListField(
      value,
      "hitl_questions",
      "hitlQuestions"
    ),
    futureKnowledgeContext: readRecordField(
      value.future_knowledge_context ?? value.futureKnowledgeContext
    ),
    authorityBoundary
  };
}

const creativeReasoningStages = [
  "strategy",
  "technique",
  "runtime",
  "tradeoff",
  "recommendation"
] as const satisfies readonly CreativeReasoningStage[];

const creativeReasoningEvidenceSources = [
  "request",
  "translation",
  "creative_intent",
  "creative_hierarchy",
  "planning",
  "director",
  "constraint_solver",
  "constraint_prioritizer",
  "creative_strategy",
  "creative_technique",
  "runtime_capability",
  "tradeoff_explorer",
  "quality_predictor",
  "symbolic_narrative",
  "creative_composition",
  "procedural_structure",
  "generative_structure",
  "semantic_motif",
  "emotional_consistency",
  "cross_modality",
  "audio_visual_scene",
  "artifact_plan",
  "artifact_dependency_graph",
  "runtime_compatibility",
  "artifact_capability_matrix",
  "multi_artifact_strategy",
  "artifact_critic",
  "artifact_refiner",
  "artifact_intelligence_synthesis",
  "artifact_merge_planner",
  "artifact_export_intelligence",
  "creative_critic",
  "self_evaluation",
  "creative_improvement_planner",
  "reflection_loop",
  "creative_confidence",
  "creative_score",
  "consistency_validation",
  "evaluation_report",
  "future_knowledge"
] as const satisfies readonly CreativeReasoningEvidenceSource[];

function readCreativeReasoningStepSummaryList(
  value: unknown
): CreativeReasoningStepSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const stage = readStringUnion(
      item,
      "stage",
      "stage",
      creativeReasoningStages
    );
    const claim = readStringField(item, "claim");
    const because = readStringField(item, "because");
    const implications = readStringListField(
      item,
      "implications",
      "implications"
    );

    if (!stage || !claim || !because || implications.length === 0) {
      return [];
    }

    return [{ stage, claim, because, implications }];
  });
}

function readCreativeReasoningEvidenceSummaryList(
  value: unknown
): CreativeReasoningEvidenceSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const source = readStringUnion(
      item,
      "source",
      "source",
      creativeReasoningEvidenceSources
    );
    const signal = readStringField(item, "signal");
    const interpretation = readStringField(item, "interpretation");

    if (!source || !signal || !interpretation) {
      return [];
    }

    return [{ source, signal, interpretation }];
  });
}

function readCreativeRejectedAlternativeSummaryList(
  value: unknown
): CreativeRejectedAlternativeSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const alternative = readStringField(item, "alternative");
    const reason = readStringField(item, "reason");
    if (!alternative || !reason) {
      return [];
    }

    return [
      {
        alternative,
        reason,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

export function readCreativeConstraintSolverSummary(
  value: unknown
): CreativeConstraintSolverSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const intentSummary =
    readStringField(value, "intent_summary") ??
    readStringField(value, "intentSummary");
  const outputGoal =
    readStringField(value, "output_goal") ??
    readStringField(value, "outputGoal");
  const runtimeFit = readStringUnion(value, "runtime_fit", "runtimeFit", [
    "supported",
    "code_only",
    "undetermined"
  ]);
  const complexityPressure = readStringUnion(
    value,
    "complexity_pressure",
    "complexityPressure",
    ["low", "medium", "high"]
  );
  const safetyPressure = readStringUnion(
    value,
    "safety_pressure",
    "safetyPressure",
    ["low", "medium", "high"]
  );
  const performancePressure = readStringUnion(
    value,
    "performance_pressure",
    "performancePressure",
    ["low", "medium", "high"]
  );
  const costPressure = readStringUnion(
    value,
    "cost_pressure",
    "costPressure",
    ["low", "medium", "high"]
  );
  const hitlAdvisable =
    readBooleanField(value, "hitl_advisable") ??
    readBooleanField(value, "hitlAdvisable");
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const activeConstraints = readCreativeConstraintSummaryList(
    value.active_constraints ?? value.activeConstraints
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );

  if (
    role !== "creative_constraint_solver" ||
    !intentSummary ||
    !outputGoal ||
    !runtimeFit ||
    !complexityPressure ||
    !safetyPressure ||
    !performancePressure ||
    !costPressure ||
    hitlAdvisable === null ||
    !authorityBoundary ||
    activeConstraints.length === 0 ||
    promptGuidance.length === 0
  ) {
    return null;
  }

  return {
    role,
    intentSummary,
    outputGoal,
    modality: readStringField(value, "modality"),
    runtimeFit,
    recommendedRuntime:
      readStringField(value, "recommended_runtime") ??
      readStringField(value, "recommendedRuntime"),
    complexityPressure,
    safetyPressure,
    performancePressure,
    costPressure,
    hitlAdvisable,
    hitlReason:
      readStringField(value, "hitl_reason") ??
      readStringField(value, "hitlReason"),
    activeConstraints,
    tradeoffs: readCreativeConstraintTradeoffSummaryList(value.tradeoffs),
    conflicts: readStringListField(value, "conflicts", "conflicts"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeConstraintAxes = [
  "intent",
  "modality",
  "runtime",
  "safety",
  "performance",
  "complexity",
  "cost",
  "hitl",
  "output_goal"
] as const satisfies readonly CreativeConstraintAxis[];

const creativeConstraintSeverities = [
  "info",
  "watch",
  "risk",
  "blocking"
] as const satisfies readonly CreativeConstraintSeverity[];

function readCreativeConstraintSummaryList(
  value: unknown
): CreativeConstraintSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const axis = readStringUnion(item, "axis", "axis", creativeConstraintAxes);
    const severity = readStringUnion(
      item,
      "severity",
      "severity",
      creativeConstraintSeverities
    );
    const summary = readStringField(item, "summary");
    const recommendation = readStringField(item, "recommendation");

    if (!axis || !severity || !summary || !recommendation) {
      return [];
    }

    return [
      {
        axis,
        severity,
        summary,
        recommendation,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCreativeConstraintTradeoffSummaryList(
  value: unknown
): CreativeConstraintTradeoffSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const sourceAxis = readStringUnion(
      item,
      "source_axis",
      "sourceAxis",
      creativeConstraintAxes
    );
    const targetAxis = readStringUnion(
      item,
      "target_axis",
      "targetAxis",
      creativeConstraintAxes
    );
    const severity = readStringUnion(
      item,
      "severity",
      "severity",
      creativeConstraintSeverities
    );
    const summary = readStringField(item, "summary");
    const recommendation = readStringField(item, "recommendation");

    if (!sourceAxis || !targetAxis || !severity || !summary || !recommendation) {
      return [];
    }

    return [
      {
        sourceAxis,
        targetAxis,
        severity,
        summary,
        recommendation
      }
    ];
  });
}

export function readCreativeConstraintPrioritizationSummary(
  value: unknown
): CreativeConstraintPrioritizationSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const nonNegotiableConstraints = readCreativeConstraintPrioritySummaryList(
    value.non_negotiable_constraints ?? value.nonNegotiableConstraints
  );
  const highPriorityConstraints = readCreativeConstraintPrioritySummaryList(
    value.high_priority_constraints ?? value.highPriorityConstraints
  );
  const flexibleConstraints = readCreativeConstraintPrioritySummaryList(
    value.flexible_constraints ?? value.flexibleConstraints
  );
  const relaxableConstraints = readCreativeConstraintPrioritySummaryList(
    value.relaxable_constraints ?? value.relaxableConstraints
  );
  const sacrificialConstraints = readCreativeConstraintPrioritySummaryList(
    value.sacrificial_constraints ?? value.sacrificialConstraints
  );
  const priorityRationale = readStringListField(
    value,
    "priority_rationale",
    "priorityRationale"
  );
  const promptGuidance = readStringListField(
    value,
    "prompt_guidance",
    "promptGuidance"
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const priorityCount =
    nonNegotiableConstraints.length +
    highPriorityConstraints.length +
    flexibleConstraints.length +
    relaxableConstraints.length +
    sacrificialConstraints.length;

  if (
    role !== "creative_constraint_prioritizer" ||
    priorityCount === 0 ||
    priorityRationale.length === 0 ||
    promptGuidance.length === 0 ||
    !authorityBoundary
  ) {
    return null;
  }

  return {
    role,
    nonNegotiableConstraints,
    highPriorityConstraints,
    flexibleConstraints,
    relaxableConstraints,
    sacrificialConstraints,
    priorityRationale,
    negotiationNotes: readStringListField(
      value,
      "negotiation_notes",
      "negotiationNotes"
    ),
    conflictRelationships: readCreativeConstraintPriorityConflictSummaryList(
      value.conflict_relationships ?? value.conflictRelationships
    ),
    hitlQuestions: readStringListField(value, "hitl_questions", "hitlQuestions"),
    promptGuidance,
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

const creativeConstraintPriorityCategories = [
  "symbolic_fidelity",
  "narrative_fidelity",
  "emotional_fidelity",
  "geometric_fidelity",
  "visual_quality",
  "motion_quality",
  "audio_quality",
  "runtime_safety",
  "previewability",
  "performance",
  "implementation_simplicity",
  "cost_sensitivity",
  "interaction_complexity",
  "maintainability"
] as const satisfies readonly CreativeConstraintPriorityCategory[];

const creativeConstraintPriorityLevels = [
  "non_negotiable",
  "high_priority",
  "flexible",
  "relaxable",
  "sacrificial"
] as const satisfies readonly CreativeConstraintPriorityLevel[];

const creativeConstraintPrioritySources = [
  "explicit",
  "hierarchy",
  "solver",
  "runtime",
  "tradeoff",
  "coherence"
] as const satisfies readonly CreativeConstraintPrioritySource[];

function readCreativeConstraintPrioritySummaryList(
  value: unknown
): CreativeConstraintPrioritySummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const category = readStringUnion(
      item,
      "category",
      "category",
      creativeConstraintPriorityCategories
    );
    const priorityLevel = readStringUnion(
      item,
      "priority_level",
      "priorityLevel",
      creativeConstraintPriorityLevels
    );
    const rank = readFiniteNumberField(item, "rank");
    const priorityScore =
      readFiniteNumberField(item, "priority_score") ??
      readFiniteNumberField(item, "priorityScore");
    const source = readStringUnion(
      item,
      "source",
      "source",
      creativeConstraintPrioritySources
    );
    const rationale = readStringField(item, "rationale");
    const negotiationGuidance =
      readStringField(item, "negotiation_guidance") ??
      readStringField(item, "negotiationGuidance");

    if (
      !category ||
      !priorityLevel ||
      rank === null ||
      priorityScore === null ||
      !source ||
      !rationale ||
      !negotiationGuidance
    ) {
      return [];
    }

    return [
      {
        category,
        priorityLevel,
        rank,
        priorityScore,
        source,
        rationale,
        negotiationGuidance,
        evidence: readStringListField(item, "evidence", "evidence")
      }
    ];
  });
}

function readCreativeConstraintPriorityConflictSummaryList(
  value: unknown
): CreativeConstraintPriorityConflictSummary[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const protectedCategory = readStringUnion(
      item,
      "protected_category",
      "protectedCategory",
      creativeConstraintPriorityCategories
    );
    const competingCategory = readStringUnion(
      item,
      "competing_category",
      "competingCategory",
      creativeConstraintPriorityCategories
    );
    const severity = readStringUnion(
      item,
      "severity",
      "severity",
      creativeConstraintSeverities
    );
    const summary = readStringField(item, "summary");
    const negotiationNote =
      readStringField(item, "negotiation_note") ??
      readStringField(item, "negotiationNote");
    const hitlRecommended =
      readBooleanField(item, "hitl_recommended") ??
      readBooleanField(item, "hitlRecommended");

    if (
      !protectedCategory ||
      !competingCategory ||
      !severity ||
      !summary ||
      !negotiationNote ||
      hitlRecommended === null
    ) {
      return [];
    }

    return [
      {
        protectedCategory,
        competingCategory,
        severity,
        summary,
        negotiationNote,
        hitlRecommended
      }
    ];
  });
}

export function readCreativeExecutionPlanSummary(
  value: unknown
): CreativeExecutionPlanSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const outputModality = readStringUnion(value, "output_modality", "outputModality", [
    "visual",
    "audio",
    "audiovisual"
  ]);
  const generationStrategy =
    readStringField(value, "generation_strategy") ??
    readStringField(value, "generationStrategy");
  const expectedComplexity = readStringUnion(
    value,
    "expected_complexity",
    "expectedComplexity",
    ["low", "medium", "high"]
  );
  const exportReadiness = readStringUnion(
    value,
    "export_readiness",
    "exportReadiness",
    ["ready", "partial", "blocked"]
  );
  const candidateCount =
    readFiniteNumberField(value, "candidate_count") ??
    readFiniteNumberField(value, "candidateCount");
  const refinementBudget =
    readFiniteNumberField(value, "refinement_budget") ??
    readFiniteNumberField(value, "refinementBudget");
  const estimatedTokenCost =
    readFiniteNumberField(value, "estimated_token_cost") ??
    readFiniteNumberField(value, "estimatedTokenCost");
  const runtimeSupportSummary =
    readStringField(value, "runtime_support_summary") ??
    readStringField(value, "runtimeSupportSummary");
  const runtimeAvailable =
    readBooleanField(value, "runtime_available") ??
    readBooleanField(value, "runtimeAvailable");

  if (
    !outputModality ||
    !generationStrategy ||
    !expectedComplexity ||
    !exportReadiness ||
    candidateCount === null ||
    refinementBudget === null ||
    estimatedTokenCost === null ||
    !runtimeSupportSummary ||
    runtimeAvailable === null
  ) {
    return null;
  }

  return {
    outputModality,
    generationStrategy,
    recommendedRuntime:
      readStringField(value, "recommended_runtime") ??
      readStringField(value, "recommendedRuntime"),
    recommendedRendererId:
      readStringField(value, "recommended_renderer_id") ??
      readStringField(value, "recommendedRendererId"),
    recommendedPreviewTarget:
      readStringField(value, "recommended_preview_target") ??
      readStringField(value, "recommendedPreviewTarget"),
    recommendedShaderStyle:
      readStringField(value, "recommended_shader_style") ??
      readStringField(value, "recommendedShaderStyle"),
    candidateCount,
    refinementBudget,
    expectedComplexity,
    estimatedTokenCost,
    exportReadiness,
    runtimeAvailable,
    runtimeSupportSummary,
    planSteps: readStringListField(value, "plan_steps", "planSteps"),
    constraints: readStringListField(value, "constraints", "constraints"),
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readCreativeAssistantDirectorSummary(
  value: unknown
): CreativeAssistantDirectorSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const role = readStringField(value, "role");
  const creativeBrief =
    readStringField(value, "creative_brief") ??
    readStringField(value, "creativeBrief");
  const ambiguityLevel = readStringUnion(
    value,
    "ambiguity_level",
    "ambiguityLevel",
    ["low", "medium", "high"]
  );
  const retrievalPosture = readStringUnion(
    value,
    "retrieval_posture",
    "retrievalPosture",
    ["not_requested", "useful", "available"]
  );
  const authorityBoundary =
    readStringField(value, "authority_boundary") ??
    readStringField(value, "authorityBoundary");
  const hitlRequired =
    readBooleanField(value, "hitl_required") ??
    readBooleanField(value, "hitlRequired");
  const nextActions = readStringListField(
    value,
    "next_actions",
    "nextActions"
  );

  if (
    role !== "creative_assistant_director" ||
    !creativeBrief ||
    !ambiguityLevel ||
    !retrievalPosture ||
    !authorityBoundary ||
    hitlRequired === null ||
    nextActions.length === 0
  ) {
    return null;
  }

  return {
    role,
    creativeBrief,
    ambiguityLevel,
    ambiguitySignals: readStringListField(
      value,
      "ambiguity_signals",
      "ambiguitySignals"
    ),
    retrievalPosture,
    modalityDirection:
      readStringField(value, "modality_direction") ??
      readStringField(value, "modalityDirection"),
    runtimeDirection:
      readStringField(value, "runtime_direction") ??
      readStringField(value, "runtimeDirection"),
    planningFocus: readStringListField(
      value,
      "planning_focus",
      "planningFocus"
    ),
    critiqueFocus: readStringListField(value, "critique_focus", "critiqueFocus"),
    refinementFocus: readStringListField(
      value,
      "refinement_focus",
      "refinementFocus"
    ),
    nextActions,
    hitlRequired,
    hitlReason:
      readStringField(value, "hitl_reason") ??
      readStringField(value, "hitlReason"),
    authorityBoundary,
    evidence: readStringListField(value, "evidence", "evidence")
  };
}

export function readClarificationSummary(
  value: unknown
): ClarificationSummary | null {
  if (!isRecord(value)) {
    return null;
  }

  const reason = readStringField(value, "reason");
  const confidence = readFiniteNumberField(value, "confidence");
  const summary = readStringField(value, "summary");
  const originalQuery =
    readStringField(value, "original_query") ??
    readStringField(value, "originalQuery");
  const questions = parseClarificationQuestions(value.questions);

  if (
    !reason ||
    confidence === null ||
    !summary ||
    !originalQuery ||
    questions.length === 0
  ) {
    return null;
  }

  return {
    reason,
    confidence,
    summary,
    originalQuery,
    questions,
    suggestedOptions: readStringListField(
      value,
      "suggested_options",
      "suggestedOptions"
    ),
    defaultRecommendation:
      readStringField(value, "default_recommendation") ??
      readStringField(value, "defaultRecommendation"),
    signalSummary: readStringListField(
      value,
      "signal_summary",
      "signalSummary"
    )
  };
}

export function readEventTimestamp(event: AssistantStreamEvent): string | null {
  const emittedAt = event.payload.emitted_at;
  return typeof emittedAt === "string" ? emittedAt : null;
}

export function readPreviewArtifactUpdate(
  event: AssistantStreamEvent
): AssistantPreviewArtifactUpdate | null {
  if (event.event_type !== "preview_artifact") {
    return null;
  }

  const status = normalizePreviewArtifactStatus(event.payload.status);
  if (!status) {
    return null;
  }

  const rawResult = event.payload.result;
  const result = isRecord(rawResult) ? rawResult : null;
  const provenance = isRecord(result?.provenance) ? result.provenance : null;
  const request = isRecord(result?.request) ? result.request : null;
  const error = isRecord(result?.error) ? result.error : null;
  const structuredError = error
    ? buildPreviewArtifactError({
        artifactId:
          typeof event.payload.artifact_id === "string" ? event.payload.artifact_id : null,
        error,
        rendererId:
          typeof provenance?.renderer_id === "string" ? provenance.renderer_id : null
      })
    : null;

  return {
    status,
    artifactId: typeof event.payload.artifact_id === "string" ? event.payload.artifact_id : null,
    previewArtifactId:
      typeof result?.preview_artifact_id === "string"
        ? result.preview_artifact_id
        : null,
    rendererId:
      typeof provenance?.renderer_id === "string" ? provenance.renderer_id : null,
    target: typeof request?.target === "string" ? request.target : null,
    summary: typeof result?.summary === "string" ? result.summary : null,
    errorMessage: structuredError?.userMessage ?? (typeof error?.message === "string" ? error.message : null),
    error: structuredError,
    emittedAt: readEventTimestamp(event),
    completedAt:
      typeof result?.completed_at === "string" ? result.completed_at : null
  };
}

export function readStreamEventError(
  event: AssistantStreamEvent
): WorkstationError | null {
  if (event.event_type !== "error") {
    return null;
  }

  const parsed = parseSubsystemErrorPayload(event.payload);
  const code =
    parsed?.type ?? (typeof event.payload.code === "string" ? event.payload.code : "assistant_stream_failed");
  const rawMessage =
    parsed?.message ??
    (typeof event.payload.message === "string"
      ? event.payload.message
      : "The live response stopped before completion.");
  const recoverable = parsed?.recoverable ?? true;

  return createWorkstationError({
    type: code,
    category: "stream",
    subsystem: parsed?.subsystem ?? inferStreamSubsystem(code),
    userMessage: normalizeStreamUserMessage(code, rawMessage),
    debugMessage: parsed?.debugMessage,
    recoverable,
    suggestedAction:
      parsed?.suggestedAction ?? defaultStreamSuggestedAction(code, recoverable),
    retryLabel: parsed?.retryLabel ?? (recoverable ? "Send prompt again" : null),
    resetLabel:
      parsed?.resetLabel ?? (recoverable ? "Clear workspace session" : null)
  });
}

function isAssistantStreamEvent(value: unknown): value is AssistantStreamEvent {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.event_type === "string" &&
    streamEventTypes.has(value.event_type as AssistantStreamEventType) &&
    typeof value.sequence === "number" &&
    Number.isInteger(value.sequence) &&
    value.sequence >= 0 &&
    isRecord(value.payload)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizePreviewArtifactStatus(
  value: unknown
): AssistantPreviewArtifactStatus | null {
  switch (value) {
    case "succeeded":
    case "failed":
    case "skipped":
      return value;
    case "ready":
      return "succeeded";
    case "error":
      return "failed";
    case "unavailable":
      return "skipped";
    default:
      return null;
  }
}

async function buildFailedHttpResponseError(response: Response) {
  const payload = await readResponsePayload(response);
  const parsed = parseSubsystemErrorPayload(payload);
  const responseErrorCode =
    parsed?.type ?? readTopLevelErrorCode(payload) ?? `http_${response.status}`;
  const rawMessage =
    parsed?.message ??
    readTopLevelMessage(payload) ??
    `Assistant stream request failed with ${response.status}.`;
  const recoverable =
    parsed?.recoverable ??
    (response.status >= 500 ||
      response.status === 408 ||
      response.status === 429);

  return createWorkstationError({
    type: responseErrorCode,
    category: "stream",
    subsystem: parsed?.subsystem ?? "assistant_stream",
    userMessage: normalizeHttpStreamUserMessage(responseErrorCode, response.status, rawMessage),
    debugMessage: parsed?.debugMessage ?? buildHttpDebugMessage(response.status, payload),
    recoverable,
    suggestedAction:
      parsed?.suggestedAction ??
      defaultHttpStreamSuggestedAction(responseErrorCode, response.status, recoverable),
    retryLabel: parsed?.retryLabel ?? (recoverable ? "Send prompt again" : null),
    resetLabel:
      parsed?.resetLabel ??
      (response.status >= 500 ? "Clear workspace session" : null)
  });
}

async function readResponsePayload(response: Response) {
  try {
    return (await response.clone().json()) as unknown;
  } catch {
    try {
      return await response.clone().text();
    } catch {
      return null;
    }
  }
}

function buildPreviewArtifactError({
  artifactId,
  error,
  rendererId
}: {
  artifactId: string | null;
  error: Record<string, unknown>;
  rendererId: string | null;
}) {
  const parsed = parseSubsystemErrorPayload(error);
  const code = parsed?.type ?? "preview_runtime_failed";
  const recoverable = parsed?.recoverable ?? false;
  const artifactLabel = artifactId ?? "the active preview";

  return createWorkstationError({
    type: code,
    category: "preview_runtime",
    subsystem: parsed?.subsystem ?? rendererId ?? "preview_runtime",
    userMessage:
      parsed?.message ?? `Preview output failed for ${artifactLabel}.`,
    debugMessage: parsed?.debugMessage,
    recoverable,
    suggestedAction:
      parsed?.suggestedAction ??
      "Reload the preview state or reset the preview session before retrying.",
    retryLabel: parsed?.retryLabel ?? (recoverable ? "Reload preview state" : null),
    resetLabel: parsed?.resetLabel ?? "Reset preview session"
  });
}

function readTopLevelErrorCode(payload: unknown) {
  return isRecord(payload) && typeof payload.error === "string" ? payload.error : null;
}

function readTopLevelMessage(payload: unknown) {
  return isRecord(payload) && typeof payload.message === "string"
    ? payload.message
    : typeof payload === "string" && payload.trim()
      ? payload
      : null;
}

function inferStreamSubsystem(code: string) {
  if (code.startsWith("provider_")) {
    return "generation_provider";
  }

  if (code.startsWith("workflow_")) {
    return "assistant_workflow";
  }

  return "assistant_stream";
}

function normalizeStreamUserMessage(code: string, message: string) {
  if (code === "assistant_stream_failed") {
    return "The live response stopped before completion.";
  }

  if (code === "provider_unavailable") {
    return "The model provider is unavailable for this live response.";
  }

  return message;
}

function defaultStreamSuggestedAction(code: string, recoverable: boolean) {
  if (code === "provider_unavailable") {
    return "Retry the request after the provider recovers, or continue with the local fallback path.";
  }

  if (recoverable) {
    return "Retry the request from the composer.";
  }

  return "Reset the workspace session before trying again.";
}

function normalizeHttpStreamUserMessage(
  code: string,
  status: number,
  message: string
) {
  if (code === "invalid_request" || status === 400) {
    return "The live request was rejected before streaming started.";
  }

  if (status >= 500) {
    return "The backend could not open a live response stream.";
  }

  return message;
}

function defaultHttpStreamSuggestedAction(
  code: string,
  status: number,
  recoverable: boolean
) {
  if (code === "invalid_request" || status === 400) {
    return "Review the current prompt or session state, then try the request again.";
  }

  if (recoverable) {
    return "Retry the request from the composer.";
  }

  return "Reset the workspace session before opening another live response.";
}

function buildHttpDebugMessage(status: number, payload: unknown) {
  const topLevelMessage = readTopLevelMessage(payload);
  return topLevelMessage ? `HTTP ${status}: ${topLevelMessage}` : `HTTP ${status}`;
}

function parseWorkflowNodeId(value: unknown): WorkflowNodeId | null {
  if (typeof value !== "string") {
    return null;
  }

  return workflowNodeIds.has(value as WorkflowNodeId)
    ? (value as WorkflowNodeId)
    : null;
}

function parseWorkflowNodeIdList(value: unknown): WorkflowNodeId[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => parseWorkflowNodeId(item))
    .filter((item): item is WorkflowNodeId => item !== null);
}

function parseStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string");
}

function readStringField(
  value: Record<string, unknown>,
  key: string
): string | null {
  const item = value[key];
  return typeof item === "string" && item.trim() ? item : null;
}

function readBooleanField(
  value: Record<string, unknown>,
  key: string
): boolean | null {
  const item = value[key];
  return typeof item === "boolean" ? item : null;
}

function readRecordField(value: unknown): Record<string, unknown> {
  return isRecord(value) ? { ...value } : {};
}

function readStringUnion<const TValue extends string>(
  value: Record<string, unknown>,
  snakeKey: string,
  camelKey: string,
  allowedValues: readonly TValue[]
): TValue | null {
  const item = readStringField(value, snakeKey) ?? readStringField(value, camelKey);
  return item !== null && allowedValues.includes(item as TValue)
    ? (item as TValue)
    : null;
}

function readFiniteNumberField(
  value: Record<string, unknown>,
  key: string
): number | null {
  const item = value[key];
  return typeof item === "number" && Number.isFinite(item) ? item : null;
}

function readStringListField(
  value: Record<string, unknown>,
  snakeKey: string,
  camelKey: string
): string[] {
  const snakeValue = parseStringList(value[snakeKey]);
  if (snakeValue.length > 0) {
    return snakeValue;
  }
  return parseStringList(value[camelKey]);
}

function parseClarificationQuestions(
  value: unknown
): ClarificationSummary["questions"] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const id = readStringField(item, "id");
    const prompt = readStringField(item, "prompt");
    const kind = readStringField(item, "kind");
    const normalizedKind =
      kind === "short_answer" || kind === "single_choice"
        ? kind
        : "single_choice";

    if (!id || !prompt) {
      return [];
    }

    return [
      {
        id,
        prompt,
        kind: normalizedKind,
        suggestedOptions: readStringListField(
          item,
          "suggested_options",
          "suggestedOptions"
        ),
        defaultRecommendation:
          readStringField(item, "default_recommendation") ??
          readStringField(item, "defaultRecommendation")
      }
    ];
  });
}

function parseImageReferenceMetadataList(
  value: unknown
): AssistantStreamImageReferenceMetadata[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (
      !isRecord(item) ||
      typeof item.id !== "string" ||
      typeof item.name !== "string" ||
      typeof item.mime_type !== "string" ||
      typeof item.size_bytes !== "number"
    ) {
      return [];
    }

    return [
      {
        id: item.id,
        name: item.name,
        mime_type: item.mime_type,
        size_bytes: item.size_bytes
      }
    ];
  });
}

const workflowNodeIds = new Set<WorkflowNodeId>(workflowNodeOrder);
