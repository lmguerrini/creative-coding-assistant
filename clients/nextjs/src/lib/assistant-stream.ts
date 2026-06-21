import {
  workflowNodeOrder,
  type ArtifactCritique,
  type CreativeConstraintAxis,
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
  runtime_capabilities?: RuntimeCapabilityReasonerSummary | null;
  runtime_capability_reasoner_available?: boolean;
  creative_tradeoffs?: CreativeTradeoffExplorerSummary | null;
  tradeoff_explorer_available?: boolean;
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
  "creative_strategy",
  "creative_technique",
  "runtime_capability",
  "tradeoff_explorer",
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
