import type { WorkflowNodeId } from "./assistant-client";

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
  emittedAt: string | null;
  completedAt: string | null;
};

export type AssistantStreamRequest = {
  query: string;
  conversationId?: string;
  projectId?: string;
  domain?: string;
  domains?: string[];
  mode?: string;
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
  }
};

export class AssistantStreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssistantStreamError";
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
      `Assistant stream request failed with ${response.status}.`
    );
  }

  if (!response.body) {
    throw new AssistantStreamError("Assistant stream response did not include a body.");
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
    throw new AssistantStreamError("Assistant stream line had an invalid shape.");
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

  if (event.event_type === "token_delta") {
    return "generation";
  }

  if (event.event_type === "final") {
    return "finalization";
  }

  if (event.event_type === "error") {
    return "failure";
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
  const reviewOutcome =
    typeof rawWorkflow.review_outcome === "string"
      ? rawWorkflow.review_outcome
      : null;
  const refinementCount =
    typeof rawWorkflow.refinement_count === "number"
      ? rawWorkflow.refinement_count
      : 0;

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
    review_reasons: reviewReasons
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
    errorMessage: typeof error?.message === "string" ? error.message : null,
    emittedAt: readEventTimestamp(event),
    completedAt:
      typeof result?.completed_at === "string" ? result.completed_at : null
  };
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

const workflowNodeIds = new Set<WorkflowNodeId>([
  "intake",
  "routing",
  "memory",
  "retrieval",
  "context_assembly",
  "prompt_input",
  "prompt_rendering",
  "generation",
  "review",
  "refinement",
  "finalization",
  "failure"
]);
