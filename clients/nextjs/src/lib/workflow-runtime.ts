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
  type AssistantStreamEvent,
  type AssistantStreamProductOutcome
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

export type WorkflowRuntimeActivityState =
  | "planning"
  | "retrieving"
  | "generating"
  | "reviewing"
  | "refining"
  | "finalizing"
  | "completed"
  | "partial"
  | "failed";

/**
 * The single user-facing execution state. Every live surface must derive its
 * wording from this object instead of inferring a separate chat or telemetry
 * label from transport event types.
 */
export type WorkflowRuntimeActivity = {
  state: WorkflowRuntimeActivityState;
  label: string;
  detail: string;
  terminal: boolean;
};

export type WorkflowRuntimeSummary = {
  status: string;
  productOutcome: AssistantStreamProductOutcome;
  activity: WorkflowRuntimeActivity;
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

export function deriveWorkflowRuntimeActivity({
  currentNode,
  productOutcome,
  workflowStatus
}: {
  currentNode: WorkflowNodeId | null;
  productOutcome: AssistantStreamProductOutcome | null | undefined;
  workflowStatus: string;
}): WorkflowRuntimeActivity {
  switch (productOutcome?.product_outcome) {
    case "SUCCESS":
      return buildWorkflowRuntimeActivity(
        "completed",
        productOutcome.summary || "The requested output is ready."
      );
    case "PARTIAL":
      return buildWorkflowRuntimeActivity(
        "partial",
        productOutcome.summary || "A usable result is available with a limitation."
      );
    case "FAILURE":
      return buildWorkflowRuntimeActivity(
        "failed",
        productOutcome.summary || "The requested output could not be completed."
      );
    default:
      break;
  }

  if (normalizeWorkflowStatus(workflowStatus) === "failed" || currentNode === "failure") {
    return buildWorkflowRuntimeActivity(
      "failed",
      "The workflow stopped before the requested output was ready."
    );
  }

  if (
    normalizeWorkflowStatus(workflowStatus) === "completed" &&
    productOutcome?.product_outcome !== "IN_PROGRESS"
  ) {
    return buildWorkflowRuntimeActivity(
      "completed",
      "The requested output is ready."
    );
  }

  const state = workflowActivityStateForNode(currentNode);
  return buildWorkflowRuntimeActivity(state);
}

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
  let previewRuntimeError: WorkstationError | null = null;
  const explicitTransitionKeys = new Set<string>();

  for (const traceEvent of traceEvents) {
    previewRuntimeError ??= readPreviewRuntimeError(traceEvent.event);
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
    } else {
      if (nodeId) {
        latestNode = nodeId;
      } else if (traceEvent.event.event_type === "status" && events.length === 0) {
        // A minimal transport status is still the beginning of a real run. Do
        // not inherit an old workspace's generation node and tell chat/header
        // different stories before the first structured workflow event arrives.
        latestNode = "planning";
      }

      if (traceEvent.event.event_type === "final") {
        latestStatus = "completed";
      } else if (traceEvent.event.event_type === "error") {
        latestStatus = "failed";
      }
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
  const productOutcome = deriveProductOutcome({
    metadata: latestMetadata,
    previewRuntimeError,
    workflowStatus: latestStatus
  });
  const productOutcomeStatus = workflowStatusForProductOutcome(productOutcome);
  const productOutcomeStep =
    productOutcome.product_outcome === "IN_PROGRESS"
      ? currentStep
      : productOutcome.summary;
  const activity = deriveWorkflowRuntimeActivity({
    currentNode: latestNode,
    productOutcome,
    workflowStatus: latestStatus
  });
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
      status: productOutcomeStatus,
      productOutcome,
      activity,
      currentNode: latestNode,
      currentStep: productOutcomeStep,
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
    error: previewRuntimeError ?? buildProductOutcomeError(productOutcome) ?? buildWorkflowRuntimeError({
      currentNode: latestNode,
      currentStep,
      latestStatus,
      traceEvents
    })
  };
}

function readPreviewRuntimeError(event: AssistantStreamEvent): WorkstationError | null {
  if (event.event_type !== "status" || event.payload.code !== "preview_runtime_error") {
    return null;
  }

  const runtime = event.payload.preview_runtime;
  const runtimeError =
    runtime && typeof runtime === "object" && "error" in runtime
      ? (runtime as { error?: unknown }).error
      : null;
  const message =
    typeof runtimeError === "string"
      ? runtimeError
      : typeof event.payload.message === "string"
        ? event.payload.message
        : "The generated preview could not start.";

  return createWorkstationError({
    type: "preview_runtime_failed",
    category: "workflow_runtime",
    subsystem: "preview_runtime",
    userMessage: `Completed with preview error: ${message}`,
    recoverable: true,
    suggestedAction: "Open Code to review the generated artifact, then reload or regenerate the preview.",
    retryLabel: "Reload preview",
    resetLabel: "Clear workspace session"
  });
}

function deriveProductOutcome({
  metadata,
  previewRuntimeError,
  workflowStatus
}: {
  metadata: ReturnType<typeof readWorkflowMetadata>;
  previewRuntimeError: WorkstationError | null;
  workflowStatus: string;
}): AssistantStreamProductOutcome {
  const outcome = metadata?.product_outcome ?? fallbackProductOutcome(workflowStatus);
  if (!previewRuntimeError) {
    return outcome;
  }

  return {
    ...outcome,
    preview_status: "FAILED",
    runtime_health: "FAILED",
    product_outcome: "PARTIAL",
    summary: "A usable artifact was produced, but the live preview failed.",
    recovery_action:
      "Open Code to use the artifact, then reload or regenerate the preview."
  };
}

function fallbackProductOutcome(workflowStatus: string): AssistantStreamProductOutcome {
  const normalizedStatus = normalizeWorkflowStatus(workflowStatus);
  const productOutcome =
    normalizedStatus === "failed"
      ? "FAILURE"
      : normalizedStatus === "completed"
        ? "SUCCESS"
        : "IN_PROGRESS";
  const isTerminalFailure = productOutcome === "FAILURE";

  return {
    orchestration_status: normalizedStatus.toUpperCase(),
    provider_status: "UNKNOWN",
    generation_status: isTerminalFailure ? "FAILED" : "UNKNOWN",
    deliverable_status: "UNKNOWN",
    artifact_extraction_status: "UNKNOWN",
    artifact_runnability: "UNKNOWN",
    preview_status: "UNKNOWN",
    runtime_health: "UNKNOWN",
    product_outcome: productOutcome,
    summary:
      productOutcome === "FAILURE"
        ? "The workflow ended in failure."
        : productOutcome === "SUCCESS"
          ? "The workflow completed."
          : "Generation and product validation are in progress.",
    recovery_action:
      productOutcome === "FAILURE"
        ? "Review the failure details, then retry the request."
        : ""
  };
}

function workflowActivityStateForNode(
  currentNode: WorkflowNodeId | null
): Exclude<WorkflowRuntimeActivityState, "completed" | "partial" | "failed"> {
  switch (currentNode) {
    case "memory":
    case "retrieval":
    case "context_assembly":
      return "retrieving";
    case "generation":
      return "generating";
    case "artifact_extraction":
    case "preview_preparation":
    case "artifact_critique":
    case "review":
      return "reviewing";
    case "refinement":
      return "refining";
    case "finalization":
      return "finalizing";
    default:
      return "planning";
  }
}

function buildWorkflowRuntimeActivity(
  state: WorkflowRuntimeActivityState,
  terminalDetail?: string
): WorkflowRuntimeActivity {
  switch (state) {
    case "planning":
      return {
        state,
        label: "Planning",
        detail: "Planning the requested work.",
        terminal: false
      };
    case "retrieving":
      return {
        state,
        label: "Retrieving",
        detail: "Retrieving relevant context.",
        terminal: false
      };
    case "generating":
      return {
        state,
        label: "Generating",
        detail: "Generating the requested artifact.",
        terminal: false
      };
    case "reviewing":
      return {
        state,
        label: "Reviewing",
        detail: "Reviewing the generated output.",
        terminal: false
      };
    case "refining":
      return {
        state,
        label: "Refining",
        detail: "Refining the generated output.",
        terminal: false
      };
    case "finalizing":
      return {
        state,
        label: "Finalizing",
        detail: "Finalizing the product result.",
        terminal: false
      };
    case "completed":
      return {
        state,
        label: "Completed",
        detail: terminalDetail ?? "The requested output is ready.",
        terminal: true
      };
    case "partial":
      return {
        state,
        label: "Partial",
        detail:
          terminalDetail ?? "A usable result is available with a limitation.",
        terminal: true
      };
    case "failed":
      return {
        state,
        label: "Failed",
        detail:
          terminalDetail ?? "The requested output could not be completed.",
        terminal: true
      };
  }
}

function workflowStatusForProductOutcome(
  productOutcome: AssistantStreamProductOutcome
) {
  switch (productOutcome.product_outcome) {
    case "FAILURE":
      return "failed";
    case "PARTIAL":
      return "partial";
    case "SUCCESS":
      return "completed";
    default:
      return "running";
  }
}

function buildProductOutcomeError(
  productOutcome: AssistantStreamProductOutcome
): WorkstationError | null {
  if (productOutcome.product_outcome !== "PARTIAL") {
    return null;
  }

  return createWorkstationError({
    type: "product_outcome_partial",
    category: "workflow_runtime",
    subsystem: "product_outcome",
    userMessage: productOutcome.summary,
    recoverable: true,
    suggestedAction:
      productOutcome.recovery_action ||
      "Open Code to review the usable output, then retry the unavailable step.",
    retryLabel: "Reload preview",
    resetLabel: "Clear workspace session"
  });
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
  const persistedProductOutcome = workflow.productOutcome ?? null;
  const productOutcome = persistedProductOutcome ?? fallbackProductOutcome(workflow.status);
  const activity = deriveWorkflowRuntimeActivity({
    currentNode: workflow.currentNode,
    productOutcome,
    workflowStatus: workflow.status
  });

  return {
    steps,
    transitions: [],
    events: [],
    timeline: buildWorkflowTimelineModel([]),
    summary: {
      status: persistedProductOutcome
        ? workflowStatusForProductOutcome(productOutcome)
        : normalizeWorkflowStatus(workflow.status),
      productOutcome,
      activity,
      currentNode: workflow.currentNode,
      currentStep:
        !persistedProductOutcome || productOutcome.product_outcome === "IN_PROGRESS"
          ? workflow.currentStep
          : productOutcome.summary,
      reached,
      total,
      retryCount: 0,
      transitionCount: 0,
      traceEventCount: 0,
      totalRuntimeMs: null,
      activeRuntimeMs: null
    },
    error:
      buildProductOutcomeError(productOutcome) ??
      (normalizeWorkflowStatus(workflow.status) === "failed"
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
        : null)
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
