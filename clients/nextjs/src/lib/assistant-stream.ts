import type {
  ArtifactCritique,
  CreativeTranslationSummary,
  RefinementPassRecord,
  WorkflowNodeId
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
    prompt_inputs_prepared: "prompt_input"
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
    image_references: imageReferences
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

const workflowNodeIds = new Set<WorkflowNodeId>([
  "intake",
  "routing",
  "memory",
  "retrieval",
  "context_assembly",
  "prompt_input",
  "prompt_rendering",
  "generation",
  "artifact_extraction",
  "preview_preparation",
  "artifact_critique",
  "review",
  "refinement",
  "finalization",
  "failure"
]);
