import type { WorkflowNodeId } from "./assistant-client";
import {
  readEventTimestamp,
  readStreamEventError,
  readWorkflowMetadata,
  workflowNodeFromAssistantStreamEvent,
  type AssistantStreamEvent
} from "./assistant-stream";

export type WorkflowTimelineStatus =
  | "info"
  | "running"
  | "complete"
  | "skipped"
  | "warning"
  | "error";

export type WorkflowTimelineEvent = {
  id: string;
  sequence: number;
  eventType: AssistantStreamEvent["event_type"];
  label: string;
  detail: string;
  at: string;
  phase: string;
  nodeId: WorkflowNodeId | null;
  stageLabel: string;
  status: WorkflowTimelineStatus;
  durationMs: number | null;
  warning: string | null;
  error: string | null;
  transitionReason: string | null;
};

export type WorkflowTimelineModel = {
  state: "available" | "empty";
  events: WorkflowTimelineEvent[];
  summary: {
    eventCount: number;
    warningCount: number;
    errorCount: number;
    startedAt: string | null;
    completedAt: string | null;
    totalDurationMs: number | null;
  };
};

type WorkflowTimelineTraceEvent = {
  event: AssistantStreamEvent;
  receivedAt: string;
  receivedAtMs: number;
};

type OrderedTraceEvent = WorkflowTimelineTraceEvent & {
  originalIndex: number;
};

const excludedEventTypes = new Set<AssistantStreamEvent["event_type"]>([
  "token_delta",
  "tool_start",
  "tool_result",
  "eval_update"
]);

const eventLabels: Record<string, string> = {
  request_received: "Request received",
  route_selected: "Route selected",
  memory_requested: "Memory lookup started",
  memory_completed: "Memory lookup completed",
  retrieval_requested: "Retrieval started",
  retrieval_completed: "Retrieval completed",
  context_assembled: "Context assembled",
  prompt_inputs_prepared: "Prompt inputs prepared",
  creative_plan_prepared: "Creative plan prepared",
  creative_director_prepared: "Director guidance prepared",
  creative_reasoning_prepared: "Creative reasoning prepared",
  prompt_rendered: "Prompt assembled",
  generation_input_prepared: "Provider generation started",
  artifact_extracted: "Artifact extraction completed",
  preview_artifact_prepared: "Preview prepared",
  critique_started: "Artifact critique started",
  artifact_scored: "Artifact scored",
  artifact_selected_recommended: "Artifact candidate selected",
  artifact_refinement_requested: "Artifact refinement requested",
  critique_completed: "Artifact critique completed",
  review_passed: "Review passed",
  review_failed: "Review needs refinement",
  refinement_requested: "Refinement requested",
  refinement_completed: "Refinement completed",
  retry_started: "Retry started",
  retry_completed: "Retry completed"
};

export function buildWorkflowTimelineModel(
  traceEvents: WorkflowTimelineTraceEvent[]
): WorkflowTimelineModel {
  if (traceEvents.length === 0) {
    return emptyTimeline();
  }

  const orderedEvents = traceEvents
    .map((traceEvent, originalIndex) => ({ ...traceEvent, originalIndex }))
    .sort(
      (left, right) =>
        left.event.sequence - right.event.sequence ||
        left.receivedAtMs - right.receivedAtMs ||
        left.originalIndex - right.originalIndex
    );
  const openNodes = new Map<WorkflowNodeId, number>();
  const events: WorkflowTimelineEvent[] = [];

  for (const traceEvent of orderedEvents) {
    if (excludedEventTypes.has(traceEvent.event.event_type)) {
      continue;
    }

    const timelineEvent = buildTimelineEvent(traceEvent, openNodes);
    events.push(timelineEvent);
  }

  if (events.length === 0) {
    return emptyTimeline();
  }

  const warningCount = events.filter((event) => event.warning != null).length;
  const errorCount = events.filter((event) => event.error != null).length;
  const startedAt = events[0].at;
  const latestAt = events[events.length - 1].at;
  const completedAt =
    [...events].reverse().find(
      (event) =>
        event.eventType === "final" ||
        event.eventType === "error" ||
        event.phase === "failed"
    )?.at ?? null;
  const startedAtMs = parseTimestamp(startedAt);
  const latestAtMs = parseTimestamp(latestAt);

  return {
    state: "available",
    events,
    summary: {
      eventCount: events.length,
      warningCount,
      errorCount,
      startedAt,
      completedAt,
      totalDurationMs:
        startedAtMs != null && latestAtMs != null
          ? Math.max(latestAtMs - startedAtMs, 0)
          : null
    }
  };
}

function buildTimelineEvent(
  traceEvent: OrderedTraceEvent,
  openNodes: Map<WorkflowNodeId, number>
): WorkflowTimelineEvent {
  const event = traceEvent.event;
  const workflow = readWorkflowMetadata(event);
  const nodeId =
    workflow?.step ??
    workflow?.current_step ??
    workflowNodeFromAssistantStreamEvent(event) ??
    null;
  const at = readEventTimestamp(event) ?? traceEvent.receivedAt;
  const atMs = parseTimestamp(at) ?? traceEvent.receivedAtMs;
  const status = readTimelineStatus(event, workflow?.phase ?? null);
  const durationMs =
    readDuration(event) ??
    readLifecycleDuration(event, nodeId, atMs, openNodes);
  const error = readTimelineError(event);
  const warning = error ? null : readTimelineWarning(event);

  return {
    id: `${event.sequence}-${traceEvent.originalIndex}-${event.event_type}`,
    sequence: event.sequence,
    eventType: event.event_type,
    label: readTimelineLabel(event, nodeId),
    detail: readTimelineDetail(event),
    at,
    phase: workflow?.phase ?? readString(event.payload.phase) ?? inferPhase(event),
    nodeId,
    stageLabel:
      readString(event.payload.node_label) ??
      (nodeId ? formatCode(nodeId) : "Workflow runtime"),
    status: error ? "error" : warning ? "warning" : status,
    durationMs,
    warning,
    error,
    transitionReason: readTransitionReason(event)
  };
}

function readLifecycleDuration(
  event: AssistantStreamEvent,
  nodeId: WorkflowNodeId | null,
  atMs: number,
  openNodes: Map<WorkflowNodeId, number>
) {
  if (!nodeId) {
    return null;
  }

  if (event.event_type === "node_started") {
    openNodes.set(nodeId, atMs);
    return null;
  }

  if (
    event.event_type !== "node_completed" &&
    event.event_type !== "node_failed"
  ) {
    return null;
  }

  const startedAtMs = openNodes.get(nodeId);
  openNodes.delete(nodeId);
  return startedAtMs == null ? null : Math.max(atMs - startedAtMs, 0);
}

function readTimelineLabel(
  event: AssistantStreamEvent,
  nodeId: WorkflowNodeId | null
) {
  if (event.event_type === "node_started") {
    return `${stageLabel(event, nodeId)} started`;
  }
  if (event.event_type === "node_completed") {
    if (nodeId === "generation") {
      return "Provider generation completed";
    }
    return `${stageLabel(event, nodeId)} ${readResolution(event)}`;
  }
  if (event.event_type === "node_failed") {
    return `${stageLabel(event, nodeId)} failed`;
  }
  if (event.event_type === "final") {
    return "Final response";
  }
  if (event.event_type === "error") {
    return "Workflow error";
  }

  const code = readString(event.payload.code);
  return code
    ? eventLabels[code] ?? formatCode(code)
    : eventLabels[event.event_type] ?? formatCode(event.event_type);
}

function readTimelineDetail(event: AssistantStreamEvent) {
  const message = readString(event.payload.message);
  if (message) {
    return message;
  }
  if (event.event_type === "final") {
    return "The final response was emitted to the creative session.";
  }
  if (event.event_type === "node_started") {
    return `${stageLabel(event, workflowNodeFromAssistantStreamEvent(event) ?? null)} entered.`;
  }
  return readString(event.payload.code)
    ? formatCode(readString(event.payload.code) as string)
    : formatCode(event.event_type);
}

function readTimelineStatus(
  event: AssistantStreamEvent,
  phase: string | null
): WorkflowTimelineStatus {
  if (event.event_type === "node_failed" || event.event_type === "error") {
    return "error";
  }
  if (
    event.event_type === "review_failed" ||
    event.event_type === "refinement_requested" ||
    event.event_type === "retry_started" ||
    event.payload.code === "artifact_refinement_requested"
  ) {
    return "warning";
  }
  if (event.event_type === "node_started") {
    return "running";
  }
  if (
    event.event_type === "node_completed" &&
    readString(event.payload.resolution) === "skipped"
  ) {
    return "skipped";
  }
  if (event.event_type === "preview_artifact") {
    const previewStatus = readString(event.payload.status);
    if (previewStatus === "failed") {
      return "error";
    }
    if (previewStatus === "skipped") {
      return "skipped";
    }
    return "complete";
  }
  if (phase === "failed") {
    return "error";
  }
  if (phase === "completed" || event.event_type === "final") {
    return "complete";
  }
  const code = readString(event.payload.code);
  if (
    code &&
    ["_completed", "_prepared", "_assembled", "_rendered", "_extracted"].some(
      (suffix) => code.endsWith(suffix)
    )
  ) {
    return "complete";
  }
  if (
    event.event_type === "generation_input" ||
    event.payload.code === "retrieval_requested" ||
    event.payload.code === "memory_requested" ||
    event.payload.code === "critique_started"
  ) {
    return "running";
  }
  return "info";
}

function readTimelineWarning(event: AssistantStreamEvent) {
  const direct =
    readString(event.payload.warning) ??
    readString(event.payload.warning_message);
  if (direct) {
    return direct;
  }

  const warning = readMessageList(event.payload.warnings)[0];
  if (warning) {
    return warning;
  }

  if (
    event.event_type === "review_failed" ||
    event.event_type === "refinement_requested" ||
    event.event_type === "retry_started" ||
    event.payload.code === "artifact_refinement_requested"
  ) {
    return readString(event.payload.message) ?? "Workflow refinement requested.";
  }

  return null;
}

function readTimelineError(event: AssistantStreamEvent) {
  const streamError = readStreamEventError(event);
  if (streamError) {
    return streamError.userMessage;
  }

  if (event.event_type === "node_failed") {
    return (
      readString(event.payload.error_message) ??
      readString(event.payload.message) ??
      "Workflow node failed."
    );
  }

  if (event.event_type === "preview_artifact" && event.payload.status === "failed") {
    return (
      readString(event.payload.error_message) ??
      readString(readRecord(event.payload.result)?.error) ??
      "Preview preparation failed."
    );
  }

  return null;
}

function readTransitionReason(event: AssistantStreamEvent) {
  const edge = readRecord(event.payload.edge);
  return (
    readString(event.payload.decision_reason) ??
    readString(edge?.decision_reason) ??
    readString(event.payload.retry_reason) ??
    null
  );
}

function readDuration(event: AssistantStreamEvent) {
  const telemetry = readRecord(event.payload.telemetry);
  const execution = readRecord(telemetry?.execution);
  return (
    readNumber(event.payload.duration_ms) ??
    readNumber(event.payload.durationMs) ??
    readNumber(event.payload.elapsed_ms) ??
    readNumber(event.payload.runtime_ms) ??
    readNumber(execution?.request_duration_ms) ??
    null
  );
}

function readResolution(event: AssistantStreamEvent) {
  const resolution = readString(event.payload.resolution);
  return resolution ? resolution : "completed";
}

function stageLabel(
  event: AssistantStreamEvent,
  nodeId: WorkflowNodeId | null
) {
  return (
    readString(event.payload.node_label) ??
    (nodeId ? formatCode(nodeId) : "Workflow stage")
  );
}

function inferPhase(event: AssistantStreamEvent) {
  if (event.event_type === "error" || event.event_type === "node_failed") {
    return "failed";
  }
  if (event.event_type === "final" || event.event_type === "node_completed") {
    return "completed";
  }
  const code = readString(event.payload.code);
  if (
    code &&
    ["_completed", "_prepared", "_assembled", "_rendered", "_extracted"].some(
      (suffix) => code.endsWith(suffix)
    )
  ) {
    return "completed";
  }
  return "running";
}

function emptyTimeline(): WorkflowTimelineModel {
  return {
    state: "empty",
    events: [],
    summary: {
      eventCount: 0,
      warningCount: 0,
      errorCount: 0,
      startedAt: null,
      completedAt: null,
      totalDurationMs: null
    }
  };
}

function readMessageList(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) =>
      typeof item === "string"
        ? item
        : readString(readRecord(item)?.message)
    )
    .filter((item): item is string => item != null);
}

function parseTimestamp(value: string) {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}
