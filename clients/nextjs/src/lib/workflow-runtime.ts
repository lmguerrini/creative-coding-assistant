import type {
  AssistantWorkspaceSnapshot,
  WorkflowNodeId,
  WorkflowStepState
} from "./assistant-client";
import {
  readEventTimestamp,
  readWorkflowMetadata,
  workflowNodeFromAssistantStreamEvent,
  type AssistantStreamEvent
} from "./assistant-stream";

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
  toNodeId: WorkflowNodeId;
  kind: "advance" | "retry" | "failure";
  label: string;
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
  summary: WorkflowRuntimeSummary;
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
        if (kind === "retry") {
          retryCount += 1;
        }
        transitions.push({
          fromNodeId: lastObservedNode,
          toNodeId: nodeId,
          kind,
          label: formatTransitionLabel(
            records.get(lastObservedNode)?.step.displayLabel ?? lastObservedNode,
            record.step.displayLabel,
            kind
          ),
          sequence: traceEvent.event.sequence,
          at
        });
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
    }
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
    }
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
  if (kind === "retry") {
    return `${toLabel} retry`;
  }

  if (kind === "failure") {
    return `${fromLabel} failed`;
  }

  return `${fromLabel} -> ${toLabel}`;
}

function isRetryMarkerEvent(
  event: AssistantStreamEvent,
  nodeId: WorkflowNodeId
): boolean {
  return nodeId === "generation" && event.event_type === "generation_input";
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

function formatRuntimeCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
