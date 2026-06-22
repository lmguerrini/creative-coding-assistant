import {
  workflowNodeOrder,
  type ArtifactCritique,
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
  type ClarificationSummary,
  type CreativeAssistantDirectorSummary,
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
  type RefinementPassRecord,
  type RuntimeCapabilityCandidateSummary,
  type RuntimeCapabilityComplexity,
  type RuntimeCapabilityFit,
  type RuntimeCapabilityId,
  type RuntimeCapabilityReasonerSummary,
  type RuntimePreviewSupport,
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
  creative_tradeoffs?: CreativeTradeoffExplorerSummary | null;
  tradeoff_explorer_available?: boolean;
  creative_quality_prediction?: CreativeQualityPredictionSummary | null;
  quality_predictor_available?: boolean;
  symbolic_narrative?: SymbolicNarrativePlanSummary | null;
  symbolic_narrative_available?: boolean;
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

const runtimeCapabilityComplexities = [
  "low",
  "medium",
  "high"
] as const satisfies readonly RuntimeCapabilityComplexity[];

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
