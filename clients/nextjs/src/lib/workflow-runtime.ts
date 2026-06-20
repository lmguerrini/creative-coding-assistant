import {
  workflowNodeOrder,
  type AssistantWorkspaceSnapshot,
  type WorkflowNodeId,
  type WorkflowStepState
} from "./assistant-client";
import {
  readEventTimestamp,
  readStreamEventError,
  readWorkflowMetadata,
  workflowNodeFromAssistantStreamEvent,
  type AssistantStreamEvent
} from "./assistant-stream";
import {
  createWorkstationError,
  type WorkstationError
} from "./workstation-errors";
import {
  buildWorkflowTimelineModel,
  type WorkflowTimelineModel
} from "./workflow-timeline";

export type WorkflowRuntimeTraceEvent = {
  event: AssistantStreamEvent;
  receivedAt: string;
  receivedAtMs: number;
};

export type WorkflowRuntimeVisualState =
  | "complete"
  | "active"
  | "queued"
  | "skipped"
  | "branch"
  | "failed";

export type WorkflowRuntimeStep = {
  nodeId: WorkflowNodeId;
  displayLabel: string;
  detail: string;
  state: WorkflowRuntimeVisualState;
  attemptCount: number;
  eventCount: number;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
  lastUpdatedAt: string | null;
  lastEventLabel: string | null;
  lastEventDetail: string | null;
};

export type WorkflowRuntimeTransition = {
  fromNodeId: WorkflowNodeId;
  toNodeId: WorkflowNodeId | "end";
  kind: "advance" | "retry" | "failure";
  label: string;
  reason: string | null;
  sequence: number;
  at: string;
};

export type WorkflowRuntimeEvent = {
  sequence: number;
  eventType: AssistantStreamEvent["event_type"];
  nodeId: WorkflowNodeId | null;
  phase: string | null;
  label: string;
  detail: string;
  at: string;
};

export type WorkflowRuntimeSummary = {
  status: string;
  currentNode: WorkflowNodeId;
  currentStep: string;
  reached: number;
  total: number;
  retryCount: number;
  transitionCount: number;
  traceEventCount: number;
  totalRuntimeMs: number | null;
  activeRuntimeMs: number | null;
};

export type WorkflowRuntimeModel = {
  steps: WorkflowRuntimeStep[];
  transitions: WorkflowRuntimeTransition[];
  events: WorkflowRuntimeEvent[];
  timeline: WorkflowTimelineModel;
  summary: WorkflowRuntimeSummary;
  error: WorkstationError | null;
};

type WorkflowRecord = {
  step: WorkflowRuntimeStep;
  openAttemptStartedAt: string | null;
  openAttemptStartedAtMs: number | null;
  totalDurationMs: number;
};

const workflowTerminalStates = new Set(["completed", "failed"]);

export function buildWorkflowRuntimeModel(
  workflow: AssistantWorkspaceSnapshot["workflow"],
  traceEvents: WorkflowRuntimeTraceEvent[]
): WorkflowRuntimeModel {
  if (traceEvents.length === 0) {
    return buildFallbackWorkflowRuntimeModel(workflow);
  }

  const records = new Map<WorkflowNodeId, WorkflowRecord>(
    workflow.steps.map((step) => [step.nodeId, createWorkflowRecord(step)])
  );
  const transitions: WorkflowRuntimeTransition[] = [];
  const events: WorkflowRuntimeEvent[] = [];
  let latestStatus = normalizeWorkflowStatus(workflow.status);
  let latestNode: WorkflowNodeId = workflow.currentNode;
  let lastObservedNode: WorkflowNodeId | null = null;
  let lastObservedAt = traceEvents[0]?.receivedAt ?? new Date().toISOString();
  let lastObservedAtMs = traceEvents[0]?.receivedAtMs ?? Date.now();
  let retryCount = 0;
  const explicitTransitionKeys = new Set<string>();

  for (const traceEvent of traceEvents) {
    const workflowMetadata = readWorkflowMetadata(traceEvent.event);
    const nodeId =
      workflowMetadata?.step ??
      workflowMetadata?.current_step ??
      workflowNodeFromAssistantStreamEvent(traceEvent.event) ??
      null;
    const at = readEventTimestamp(traceEvent.event) ?? traceEvent.receivedAt;
    const atMs = parseTimestamp(at) ?? traceEvent.receivedAtMs;
    const phase = workflowMetadata?.phase ?? inferWorkflowPhase(traceEvent.event);
    const label = formatRuntimeEventLabel(traceEvent.event);
    const detail = readRuntimeEventDetail(traceEvent.event);
    const explicitTransition = readExplicitWorkflowTransition(traceEvent.event);

    if (traceEvent.event.event_type === "retry_started") {
      retryCount = Math.max(
        retryCount,
        readRetryCount(traceEvent.event) ?? retryCount + 1
      );
    }

    if (workflowMetadata) {
      latestStatus = normalizeWorkflowStatus(workflowMetadata.status);
      latestNode = workflowMetadata.step ?? workflowMetadata.current_step ?? latestNode;
    } else if (nodeId) {
      latestNode = nodeId;
    }

    lastObservedAt = at;
    lastObservedAtMs = atMs;

    events.push({
      sequence: traceEvent.event.sequence,
      eventType: traceEvent.event.event_type,
      nodeId,
      phase,
      label,
      detail,
      at
    });

    if (explicitTransition) {
      const key = transitionKey(
        explicitTransition.fromNodeId,
        explicitTransition.toNodeId
      );
      explicitTransitionKeys.add(key);
      transitions.push({
        fromNodeId: explicitTransition.fromNodeId,
        toNodeId: explicitTransition.toNodeId,
        kind: explicitTransition.kind,
        label: formatTransitionLabel(
          workflowNodeLabel(records, explicitTransition.fromNodeId),
          workflowTargetLabel(records, explicitTransition.toNodeId),
          explicitTransition.kind
        ),
        reason: explicitTransition.reason,
        sequence: traceEvent.event.sequence,
        at
      });
    }

    if (!nodeId) {
      continue;
    }

    const record = records.get(nodeId);
    if (!record) {
      continue;
    }

    if (lastObservedNode === nodeId && isRetryMarkerEvent(traceEvent.event, nodeId)) {
      closeWorkflowAttempt(record, at, atMs);
      openWorkflowAttempt(record, at, atMs);
      retryCount += 1;
      transitions.push({
        fromNodeId: nodeId,
        toNodeId: nodeId,
        kind: "retry",
        label: `${record.step.displayLabel} retry`,
        reason: "generation_retry_detected",
        sequence: traceEvent.event.sequence,
        at
      });
    } else if (lastObservedNode !== nodeId) {
      if (lastObservedNode) {
        const previousRecord = records.get(lastObservedNode);
        if (previousRecord) {
          closeWorkflowAttempt(previousRecord, at, atMs);
        }

        const kind =
          nodeId === "failure" || phase === "failed"
            ? "failure"
            : record.step.attemptCount > 0
              ? "retry"
              : "advance";
        const inferredTransitionKey = transitionKey(lastObservedNode, nodeId);
        if (!explicitTransitionKeys.delete(inferredTransitionKey)) {
          if (kind === "retry") {
            retryCount += 1;
          }
          transitions.push({
            fromNodeId: lastObservedNode,
            toNodeId: nodeId,
            kind,
            label: formatTransitionLabel(
              workflowNodeLabel(records, lastObservedNode),
              record.step.displayLabel,
              kind
            ),
            reason: null,
            sequence: traceEvent.event.sequence,
            at
          });
        }
      }

      if (record.step.attemptCount === 0 || record.openAttemptStartedAtMs === null) {
        openWorkflowAttempt(record, at, atMs);
      }
      lastObservedNode = nodeId;
    } else if (record.step.attemptCount === 0) {
      openWorkflowAttempt(record, at, atMs);
    }

    record.step.eventCount += 1;
    record.step.lastUpdatedAt = at;
    record.step.lastEventLabel = label;
    record.step.lastEventDetail = detail;
  }

  const latestMetadata = readWorkflowMetadata(
    traceEvents[traceEvents.length - 1].event
  );
  const currentRecord = records.get(latestNode);

  if (currentRecord) {
    if (workflowTerminalStates.has(latestStatus)) {
      closeWorkflowAttempt(currentRecord, lastObservedAt, lastObservedAtMs);
    } else if (currentRecord.openAttemptStartedAtMs === null) {
      openWorkflowAttempt(currentRecord, lastObservedAt, lastObservedAtMs);
    }
  }

  if (latestMetadata) {
    for (const step of latestMetadata.completed_steps) {
      hydrateCompletedStep(records.get(step), lastObservedAt);
    }
    for (const step of latestMetadata.skipped_steps) {
      hydrateSkippedStep(records.get(step));
    }

    if (latestMetadata.refinement_count > 0) {
      hydrateRefinementStep(
        records.get("refinement"),
        latestMetadata.refinement_count,
        lastObservedAt
      );
    }

    if (latestMetadata.review_outcome) {
      hydrateCompletedStep(records.get("review"), lastObservedAt);
    }
  }

  const steps = workflow.steps.map((baseStep) => {
    const record = records.get(baseStep.nodeId);
    if (!record) {
      return createWorkflowRecord(baseStep).step;
    }

    const activeDurationMs =
      latestStatus === "running" &&
      baseStep.nodeId === latestNode &&
      record.openAttemptStartedAtMs !== null
        ? Math.max(lastObservedAtMs - record.openAttemptStartedAtMs, 0)
        : 0;
    const durationMs =
      record.totalDurationMs + (activeDurationMs > 0 ? activeDurationMs : 0);
    const state = deriveWorkflowVisualState({
      baseStep,
      currentNode: latestNode,
      latestMetadata,
      latestStatus,
      record
    });

    return {
      ...record.step,
      state,
      durationMs: durationMs > 0 ? durationMs : null,
      completedAt:
        record.step.completedAt ??
        (workflowTerminalStates.has(latestStatus) && baseStep.nodeId === latestNode
          ? lastObservedAt
          : null)
    };
  });

  const reached = steps.filter((step) =>
    ["complete", "active", "skipped", "failed"].includes(step.state)
  ).length;
  const total = steps.filter((step) => step.state !== "branch").length;
  const currentStep =
    steps.find((step) => step.nodeId === latestNode)?.displayLabel ?? workflow.currentStep;
  const totalRuntimeMs = Math.max(lastObservedAtMs - traceEvents[0].receivedAtMs, 0);
  const activeRuntimeMs =
    latestStatus === "running" && currentRecord?.openAttemptStartedAtMs != null
      ? Math.max(lastObservedAtMs - currentRecord.openAttemptStartedAtMs, 0)
      : null;

  return {
    steps,
    transitions,
    events,
    timeline: buildWorkflowTimelineModel(traceEvents),
    summary: {
      status: latestStatus,
      currentNode: latestNode,
      currentStep,
      reached,
      total,
      retryCount:
        latestMetadata?.refinement_count != null
          ? Math.max(latestMetadata.refinement_count, retryCount)
          : retryCount,
      transitionCount: transitions.length,
      traceEventCount: traceEvents.length,
      totalRuntimeMs,
      activeRuntimeMs
    },
    error: buildWorkflowRuntimeError({
      currentNode: latestNode,
      currentStep,
      latestStatus,
      traceEvents
    })
  };
}

function buildFallbackWorkflowRuntimeModel(
  workflow: AssistantWorkspaceSnapshot["workflow"]
): WorkflowRuntimeModel {
  const steps = workflow.steps.map((step) => ({
    ...step,
    attemptCount: ["complete", "active", "skipped"].includes(step.state) ? 1 : 0,
    eventCount: 0,
    startedAt: null,
    completedAt: null,
    durationMs: null,
    lastUpdatedAt: null,
    lastEventLabel: null,
    lastEventDetail: null
  }));
  const reached = steps.filter((step) =>
    ["complete", "active", "skipped"].includes(step.state)
  ).length;
  const total = steps.filter((step) => step.state !== "branch").length;

  return {
    steps,
    transitions: [],
    events: [],
    timeline: buildWorkflowTimelineModel([]),
    summary: {
      status: normalizeWorkflowStatus(workflow.status),
      currentNode: workflow.currentNode,
      currentStep: workflow.currentStep,
      reached,
      total,
      retryCount: 0,
      transitionCount: 0,
      traceEventCount: 0,
      totalRuntimeMs: null,
      activeRuntimeMs: null
    },
    error:
      normalizeWorkflowStatus(workflow.status) === "failed"
        ? createWorkstationError({
            type: "workflow_failed",
            category: "workflow_runtime",
            subsystem: workflow.currentNode,
            userMessage: `${workflow.currentStep} ended in a failed state.`,
            recoverable: true,
            suggestedAction:
              "Retry the request or clear the workspace session before trying again.",
            retryLabel: "Send prompt again",
            resetLabel: "Clear workspace session"
          })
        : null
  };
}

function createWorkflowRecord(step: WorkflowStepState): WorkflowRecord {
  return {
    step: {
      ...step,
      state: step.state,
      attemptCount: 0,
      eventCount: 0,
      startedAt: null,
      completedAt: null,
      durationMs: null,
      lastUpdatedAt: null,
      lastEventLabel: null,
      lastEventDetail: null
    },
    openAttemptStartedAt: null,
    openAttemptStartedAtMs: null,
    totalDurationMs: 0
  };
}

function openWorkflowAttempt(record: WorkflowRecord, at: string, atMs: number) {
  record.step.attemptCount += 1;
  record.step.startedAt ??= at;
  record.openAttemptStartedAt = at;
  record.openAttemptStartedAtMs = atMs;
}

function closeWorkflowAttempt(record: WorkflowRecord, at: string, atMs: number) {
  if (record.openAttemptStartedAtMs === null) {
    return;
  }

  record.totalDurationMs += Math.max(atMs - record.openAttemptStartedAtMs, 0);
  record.step.completedAt = at;
  record.openAttemptStartedAt = null;
  record.openAttemptStartedAtMs = null;
}

function hydrateCompletedStep(record: WorkflowRecord | undefined, at: string) {
  if (!record) {
    return;
  }

  if (record.step.attemptCount === 0) {
    record.step.attemptCount = 1;
  }
  record.step.completedAt ??= at;
}

function hydrateRefinementStep(
  record: WorkflowRecord | undefined,
  refinementCount: number,
  at: string
) {
  if (!record) {
    return;
  }

  record.step.attemptCount = Math.max(record.step.attemptCount, refinementCount);
  record.step.completedAt ??= at;
}

function hydrateSkippedStep(record: WorkflowRecord | undefined) {
  if (!record) {
    return;
  }

  record.step.attemptCount = 0;
}

function deriveWorkflowVisualState({
  baseStep,
  currentNode,
  latestMetadata,
  latestStatus,
  record
}: {
  baseStep: WorkflowStepState;
  currentNode: WorkflowNodeId;
  latestMetadata: ReturnType<typeof readWorkflowMetadata>;
  latestStatus: string;
  record: WorkflowRecord;
}): WorkflowRuntimeVisualState {
  if (baseStep.nodeId === "failure") {
    return latestStatus === "failed" && currentNode === "failure" ? "failed" : "branch";
  }

  if (latestStatus === "failed" && currentNode === baseStep.nodeId) {
    return "failed";
  }

  if (latestMetadata?.completed_steps.includes(baseStep.nodeId)) {
    return "complete";
  }

  if (latestMetadata?.skipped_steps.includes(baseStep.nodeId)) {
    return "skipped";
  }

  if (currentNode === baseStep.nodeId && latestStatus === "running") {
    return "active";
  }

  if (record.step.attemptCount > 0 && record.step.eventCount > 0) {
    return "complete";
  }

  if (!latestMetadata) {
    return baseStep.state;
  }

  return baseStep.state === "branch" ? "branch" : "queued";
}

function readRuntimeEventDetail(event: AssistantStreamEvent): string {
  const message = event.payload.message;
  if (typeof message === "string" && message) {
    return message;
  }

  const answer = event.payload.answer;
  if (typeof answer === "string" && answer) {
    return answer;
  }

  const text = event.payload.text;
  if (typeof text === "string" && text) {
    return text;
  }

  const code = event.payload.code;
  if (typeof code === "string" && code) {
    return formatRuntimeCode(code);
  }

  return formatRuntimeCode(event.event_type);
}

function formatRuntimeEventLabel(event: AssistantStreamEvent): string {
  const code = event.payload.code;
  if (typeof code === "string" && code) {
    return formatRuntimeCode(code);
  }

  return formatRuntimeCode(event.event_type);
}

function formatTransitionLabel(
  fromLabel: string,
  toLabel: string,
  kind: WorkflowRuntimeTransition["kind"]
) {
  if (kind === "retry" && fromLabel === toLabel) {
    return `${toLabel} retry`;
  }

  if (kind === "failure") {
    return `${fromLabel} failed`;
  }

  return `${fromLabel} -> ${toLabel}`;
}

function readExplicitWorkflowTransition(
  event: AssistantStreamEvent
): Pick<
  WorkflowRuntimeTransition,
  "fromNodeId" | "toNodeId" | "kind" | "reason"
> | null {
  if (
    event.event_type !== "node_completed" &&
    event.event_type !== "node_failed"
  ) {
    return null;
  }

  const edge = isRecord(event.payload.edge) ? event.payload.edge : null;
  const fromNodeId =
    readWorkflowNodeId(event.payload.transition_source) ??
    readWorkflowNodeId(edge?.source) ??
    readWorkflowNodeId(event.payload.node);
  const toNodeId =
    readWorkflowTransitionTarget(event.payload.transition_target) ??
    readWorkflowTransitionTarget(edge?.target);

  if (!fromNodeId || !toNodeId) {
    return null;
  }

  const reason =
    readPayloadString(event.payload.decision_reason) ??
    readPayloadString(edge?.decision_reason) ??
    null;
  const kind =
    event.event_type === "node_failed" || toNodeId === "failure"
      ? "failure"
      : fromNodeId === "refinement" ||
          toNodeId === "refinement" ||
          reason?.includes("retry")
        ? "retry"
        : "advance";

  return {
    fromNodeId,
    toNodeId,
    kind,
    reason
  };
}

function isRetryMarkerEvent(
  event: AssistantStreamEvent,
  nodeId: WorkflowNodeId
): boolean {
  if (isRecord(event.payload.workflow)) {
    return false;
  }

  return nodeId === "generation" && event.event_type === "generation_input";
}

const workflowNodeIds = new Set<string>(workflowNodeOrder);

function readWorkflowNodeId(value: unknown): WorkflowNodeId | null {
  return typeof value === "string" && workflowNodeIds.has(value)
    ? (value as WorkflowNodeId)
    : null;
}

function readWorkflowTransitionTarget(
  value: unknown
): WorkflowNodeId | "end" | null {
  if (value === "end") {
    return "end";
  }

  return readWorkflowNodeId(value);
}

function readRetryCount(event: AssistantStreamEvent): number | null {
  const retryCount = event.payload.retry_count;
  return typeof retryCount === "number" && Number.isFinite(retryCount)
    ? retryCount
    : null;
}

function readPayloadString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function transitionKey(
  fromNodeId: WorkflowNodeId,
  toNodeId: WorkflowNodeId | "end"
) {
  return `${fromNodeId}->${toNodeId}`;
}

function workflowNodeLabel(
  records: Map<WorkflowNodeId, WorkflowRecord>,
  nodeId: WorkflowNodeId
) {
  return records.get(nodeId)?.step.displayLabel ?? formatRuntimeCode(nodeId);
}

function workflowTargetLabel(
  records: Map<WorkflowNodeId, WorkflowRecord>,
  nodeId: WorkflowNodeId | "end"
) {
  return nodeId === "end" ? "End" : workflowNodeLabel(records, nodeId);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function inferWorkflowPhase(event: AssistantStreamEvent): string | null {
  if (event.event_type === "final") {
    return "completed";
  }

  if (event.event_type === "error") {
    return "failed";
  }

  return "running";
}

function parseTimestamp(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeWorkflowStatus(status: string) {
  return status.toLowerCase();
}

function buildWorkflowRuntimeError({
  currentNode,
  currentStep,
  latestStatus,
  traceEvents
}: {
  currentNode: WorkflowNodeId;
  currentStep: string;
  latestStatus: string;
  traceEvents: WorkflowRuntimeTraceEvent[];
}) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const streamError = readStreamEventError(traceEvents[index].event);
    if (!streamError) {
      continue;
    }

    return createWorkstationError({
      ...streamError,
      category: "workflow_runtime"
    });
  }

  if (latestStatus !== "failed") {
    return null;
  }

  return createWorkstationError({
    type: "workflow_failed",
    category: "workflow_runtime",
    subsystem: currentNode,
    userMessage: `${currentStep} ended in a failed state.`,
    recoverable: true,
    suggestedAction:
      "Retry the request or clear the workspace session before trying again.",
    retryLabel: "Send prompt again",
    resetLabel: "Clear workspace session"
  });
}

function formatRuntimeCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
