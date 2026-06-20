import {
  workflowNodeOrder,
  type ArtifactCritique,
  type ClarificationSummary,
  type CreativeAssistantDirectorSummary,
  type CreativeExecutionPlanSummary,
  type CreativeTranslationSummary,
  type RefinementPassRecord,
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
  creative_plan?: CreativeExecutionPlanSummary | null;
  planning_available?: boolean;
  creative_director?: CreativeAssistantDirectorSummary | null;
  director_available?: boolean;
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
    creative_director_prepared: "director"
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
  const creativePlan = readCreativeExecutionPlanSummary(
    rawWorkflow.creative_plan ?? rawWorkflow.creativePlan
  );
  const planningAvailable =
    rawWorkflow.planning_available === true || creativePlan !== null;
  const creativeDirector = readCreativeAssistantDirectorSummary(
    rawWorkflow.creative_director ?? rawWorkflow.creativeDirector
  );
  const directorAvailable =
    rawWorkflow.director_available === true || creativeDirector !== null;

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
    ...(planningAvailable
      ? {
          creative_plan: creativePlan,
          planning_available: true
        }
      : {}),
    ...(directorAvailable
      ? {
          creative_director: creativeDirector,
          director_available: true
        }
      : {})
  };
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
