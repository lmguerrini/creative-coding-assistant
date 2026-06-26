import type {
  ArtifactSummary,
  AssistantWorkspaceSnapshot,
  InspectorTabName,
  WorkflowNodeId,
  WorkflowStepState
} from "./assistant-client";
import { readEventTimestamp, type AssistantStreamEvent } from "./assistant-stream";
import type { EvaluationSessionModel } from "./evaluation-session";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";
import type { WorkstationError } from "./workstation-errors";

export type WorkstationMetadataStatus =
  | "available"
  | "pending"
  | "missing"
  | "error";

export type WorkstationMetadataKey =
  | "session"
  | "assistant_run"
  | "selected_artifact"
  | "selected_workflow"
  | "selected_evaluation"
  | "preview"
  | "retrieval"
  | "creative_plan"
  | "multimodal";

export type WorkstationReadinessState =
  | "empty"
  | "ready"
  | "active"
  | "degraded";

export type WorkstationRunState =
  | "idle"
  | "streaming"
  | "completed"
  | "error";

export type WorkstationMetadataSummary = {
  key: WorkstationMetadataKey;
  label: string;
  status: WorkstationMetadataStatus;
  detail: string;
};

export type WorkstationSessionState = {
  userId: string;
  sessionId: string;
  projectId: string;
  title: string;
  updatedAt: string | null;
  metadata: WorkstationMetadataSummary;
};

export type WorkstationCurrentRunState = {
  state: WorkstationRunState;
  streamEventCount: number;
  traceEventCount: number;
  latestEventSequence: number | null;
  latestEventType: AssistantStreamEvent["event_type"] | null;
  startedAt: string | null;
  updatedAt: string | null;
  finalSeen: boolean;
  errorSeen: boolean;
};

export type WorkstationSelectionState = {
  activeArtifactId: string;
  activeArtifact: ArtifactSummary | null;
  previewArtifactId: string;
  previewArtifact: ArtifactSummary | null;
  activeInspectorTab: InspectorTabName;
  activeWorkflowNodeId: WorkflowNodeId;
  activeWorkflowStep: WorkflowStepState | null;
  selectedEvaluation: EvaluationSessionModel | null;
};

export type WorkstationPanelState = {
  activeTab: InspectorTabName;
  activeTabSummary: string;
  tabs: AssistantWorkspaceSnapshot["inspectorTabs"];
  inspectorCollapsed: boolean;
  previewOpen: boolean;
  previewFullscreen: boolean;
};

export type WorkstationReadinessSummary = {
  state: WorkstationReadinessState;
  label: string;
  detail: string;
  missingMetadata: WorkstationMetadataSummary[];
};

export type WorkstationStatusSummary = {
  label: string;
  detail: string;
  readinessLabel: string;
  artifactLabel: string;
  workflowLabel: string;
  evaluationLabel: string;
  metadataLabel: string;
};

export type WorkstationMetadataState = Record<
  WorkstationMetadataKey,
  WorkstationMetadataSummary
>;

export type WorkstationState = {
  session: WorkstationSessionState;
  currentRun: WorkstationCurrentRunState;
  selection: WorkstationSelectionState;
  panels: WorkstationPanelState;
  metadata: WorkstationMetadataState;
  readiness: WorkstationReadinessSummary;
  status: WorkstationStatusSummary;
};

export type BuildWorkstationStateInput = {
  snapshot: AssistantWorkspaceSnapshot;
  activeArtifactId?: string | null;
  previewArtifactId?: string | null;
  activeInspectorTab?: InspectorTabName | null;
  activeWorkflowNodeId?: WorkflowNodeId | null;
  selectedEvaluation?: EvaluationSessionModel | null;
  isStreaming?: boolean;
  streamError?: WorkstationError | null;
  traceEvents?: readonly WorkflowRuntimeTraceEvent[];
  inspectorCollapsed?: boolean;
  previewOpen?: boolean;
  previewFullscreen?: boolean;
};

const metadataLabels = {
  assistant_run: "Assistant run",
  creative_plan: "Creative plan",
  multimodal: "Image references",
  preview: "Preview",
  retrieval: "Retrieval",
  selected_artifact: "Selected artifact",
  selected_evaluation: "Selected evaluation",
  selected_workflow: "Selected workflow",
  session: "Session"
} satisfies Record<WorkstationMetadataKey, string>;

export function buildWorkstationState({
  activeArtifactId,
  activeInspectorTab,
  activeWorkflowNodeId,
  inspectorCollapsed = false,
  isStreaming = false,
  previewArtifactId,
  previewFullscreen = false,
  previewOpen,
  selectedEvaluation = null,
  snapshot,
  streamError = null,
  traceEvents = []
}: BuildWorkstationStateInput): WorkstationState {
  const run = buildCurrentRunState(traceEvents, isStreaming, streamError);
  const activeArtifact = resolveArtifact(snapshot.artifacts, activeArtifactId);
  const resolvedActiveArtifactId =
    activeArtifact?.id ?? activeArtifactId?.trim() ?? "";
  const previewArtifact = resolveArtifact(snapshot.artifacts, previewArtifactId);
  const resolvedPreviewArtifactId =
    previewArtifact?.id ?? previewArtifactId?.trim() ?? "";
  const activeTab =
    activeInspectorTab ??
    snapshot.inspectorTabs.find((tab) => tab.active)?.label ??
    "Overview";
  const activeWorkflowStep = resolveWorkflowStep(
    snapshot.workflow.steps,
    activeWorkflowNodeId ?? snapshot.workflow.currentNode
  );
  const resolvedWorkflowNodeId =
    activeWorkflowStep?.nodeId ??
    activeWorkflowNodeId ??
    snapshot.workflow.currentNode;
  const panels = buildPanelState({
    activeTab,
    inspectorCollapsed,
    previewFullscreen,
    previewOpen: previewOpen ?? snapshot.preview.active,
    snapshot
  });
  const metadata = buildMetadataState({
    activeArtifact,
    activeWorkflowStep,
    run,
    selectedEvaluation,
    snapshot,
    streamError
  });
  const readiness = buildReadinessSummary(metadata, run, streamError);
  const status = buildStatusSummary({
    activeArtifact,
    activeWorkflowStep,
    isStreaming,
    metadata,
    readiness,
    selectedEvaluation,
    snapshot,
    streamError
  });

  return {
    session: {
      userId: snapshot.session.userId,
      sessionId: snapshot.session.sessionId,
      projectId: snapshot.session.projectId,
      title: snapshot.session.title,
      updatedAt: snapshot.session.updatedAt ?? null,
      metadata: metadata.session
    },
    currentRun: run,
    selection: {
      activeArtifactId: resolvedActiveArtifactId,
      activeArtifact,
      previewArtifactId: resolvedPreviewArtifactId,
      previewArtifact,
      activeInspectorTab: activeTab,
      activeWorkflowNodeId: resolvedWorkflowNodeId,
      activeWorkflowStep,
      selectedEvaluation
    },
    panels,
    metadata,
    readiness,
    status
  };
}

function buildCurrentRunState(
  traceEvents: readonly WorkflowRuntimeTraceEvent[],
  isStreaming: boolean,
  streamError: WorkstationError | null
): WorkstationCurrentRunState {
  const latestTraceEvent =
    traceEvents.length > 0 ? traceEvents[traceEvents.length - 1] : null;
  const finalSeen = traceEvents.some(
    (traceEvent) => traceEvent.event.event_type === "final"
  );
  const errorSeen =
    Boolean(streamError) ||
    traceEvents.some((traceEvent) => traceEvent.event.event_type === "error");
  const startedAt =
    traceEvents.length > 0
      ? readTraceEventTimestamp(traceEvents[0])
      : null;
  const updatedAt = latestTraceEvent ? readTraceEventTimestamp(latestTraceEvent) : null;
  const state: WorkstationRunState = errorSeen
    ? "error"
    : isStreaming
      ? "streaming"
      : finalSeen
        ? "completed"
        : "idle";

  return {
    state,
    streamEventCount: traceEvents.length,
    traceEventCount: traceEvents.length,
    latestEventSequence: latestTraceEvent?.event.sequence ?? null,
    latestEventType: latestTraceEvent?.event.event_type ?? null,
    startedAt,
    updatedAt,
    finalSeen,
    errorSeen
  };
}

function buildPanelState({
  activeTab,
  inspectorCollapsed,
  previewFullscreen,
  previewOpen,
  snapshot
}: {
  activeTab: InspectorTabName;
  inspectorCollapsed: boolean;
  previewFullscreen: boolean;
  previewOpen: boolean;
  snapshot: AssistantWorkspaceSnapshot;
}): WorkstationPanelState {
  const tabs = snapshot.inspectorTabs.map((tab) => ({
    ...tab,
    active: tab.label === activeTab
  }));
  const activeTabSummary =
    tabs.find((tab) => tab.label === activeTab)?.summary ?? "";

  return {
    activeTab,
    activeTabSummary,
    tabs,
    inspectorCollapsed,
    previewOpen,
    previewFullscreen
  };
}

function buildMetadataState({
  activeArtifact,
  activeWorkflowStep,
  run,
  selectedEvaluation,
  snapshot,
  streamError
}: {
  activeArtifact: ArtifactSummary | null;
  activeWorkflowStep: WorkflowStepState | null;
  run: WorkstationCurrentRunState;
  selectedEvaluation: EvaluationSessionModel | null;
  snapshot: AssistantWorkspaceSnapshot;
  streamError: WorkstationError | null;
}): WorkstationMetadataState {
  return {
    session: metadataSummary(
      "session",
      hasRequiredText(
        snapshot.session.userId,
        snapshot.session.sessionId,
        snapshot.session.projectId
      )
        ? "available"
        : "missing",
      hasRequiredText(
        snapshot.session.userId,
        snapshot.session.sessionId,
        snapshot.session.projectId
      )
        ? `Session ${snapshot.session.sessionId} in ${snapshot.session.projectId}.`
        : "Session identifiers are not available."
    ),
    assistant_run: metadataSummary(
      "assistant_run",
      run.state === "error"
        ? "error"
        : run.state === "streaming"
          ? "pending"
          : run.state === "completed"
            ? "available"
            : "missing",
      streamError?.userMessage ??
        (run.latestEventType
          ? `${run.latestEventType} event ${run.latestEventSequence ?? ""}`.trim()
          : "No live assistant run has been captured yet.")
    ),
    selected_artifact: metadataSummary(
      "selected_artifact",
      activeArtifact ? "available" : "missing",
      activeArtifact
        ? `${activeArtifact.title} is selected.`
        : "No generated artifact is selected."
    ),
    selected_workflow: metadataSummary(
      "selected_workflow",
      activeWorkflowStep ? "available" : "missing",
      activeWorkflowStep
        ? `${activeWorkflowStep.displayLabel}: ${activeWorkflowStep.detail}`
        : "No workflow step metadata is selected."
    ),
    selected_evaluation: metadataSummary(
      "selected_evaluation",
      selectedEvaluationStatus(selectedEvaluation),
      selectedEvaluationDetail(selectedEvaluation)
    ),
    preview: metadataSummary(
      "preview",
      snapshot.preview.available ? "available" : "missing",
      snapshot.preview.available
        ? snapshot.preview.status
        : "No runnable preview artifact is available."
    ),
    retrieval: metadataSummary(
      "retrieval",
      snapshot.retrieval.sources.length > 0 || snapshot.retrieval.state === "available"
        ? "available"
        : snapshot.retrieval.state === "pending"
          ? "pending"
          : "missing",
      snapshot.retrieval.detail
    ),
    creative_plan: metadataSummary(
      "creative_plan",
      snapshot.creativePlan ? "available" : "missing",
      snapshot.creativePlan
        ? snapshot.creativePlan.generationStrategy
        : "No creative execution plan is attached to the workspace yet."
    ),
    multimodal: metadataSummary(
      "multimodal",
      snapshot.multimodal.imageAttachments.length > 0 ? "available" : "missing",
      snapshot.multimodal.status
    )
  };
}

function buildReadinessSummary(
  metadata: WorkstationMetadataState,
  run: WorkstationCurrentRunState,
  streamError: WorkstationError | null
): WorkstationReadinessSummary {
  const missingMetadata = Object.values(metadata).filter(
    (summary) => summary.status !== "available"
  );
  const state: WorkstationReadinessState =
    streamError || run.state === "error" || metadata.session.status !== "available"
      ? "degraded"
      : run.state === "streaming"
        ? "active"
        : metadata.selected_artifact.status === "missing" && run.state === "idle"
          ? "empty"
          : "ready";
  const label = {
    active: "Active run",
    degraded: "Needs attention",
    empty: "Ready for first prompt",
    ready: "Workspace ready"
  } satisfies Record<WorkstationReadinessState, string>;
  const detail =
    state === "degraded"
      ? streamError?.userMessage ?? "Required workstation metadata is unavailable."
      : state === "active"
        ? "Assistant stream metadata is being collected."
        : state === "empty"
          ? "Session is initialized and waiting for generated artifacts."
          : "Core workstation metadata is available for inspection.";

  return {
    state,
    label: label[state],
    detail,
    missingMetadata
  };
}

function buildStatusSummary({
  activeArtifact,
  activeWorkflowStep,
  isStreaming,
  metadata,
  readiness,
  selectedEvaluation,
  snapshot,
  streamError
}: {
  activeArtifact: ArtifactSummary | null;
  activeWorkflowStep: WorkflowStepState | null;
  isStreaming: boolean;
  metadata: WorkstationMetadataState;
  readiness: WorkstationReadinessSummary;
  selectedEvaluation: EvaluationSessionModel | null;
  snapshot: AssistantWorkspaceSnapshot;
  streamError: WorkstationError | null;
}): WorkstationStatusSummary {
  const availableCount = Object.values(metadata).filter(
    (summary) => summary.status === "available"
  ).length;
  const totalCount = Object.keys(metadata).length;

  return {
    label: isStreaming ? "Streaming" : streamError ? "Fallback" : snapshot.workflow.status,
    detail: snapshot.workflow.currentStep,
    readinessLabel: readiness.label,
    artifactLabel: activeArtifact?.title ?? "No artifact selected",
    workflowLabel: activeWorkflowStep?.displayLabel ?? snapshot.workflow.currentStep,
    evaluationLabel: selectedEvaluation
      ? selectedEvaluation.statusLabel
      : "No evaluation selected",
    metadataLabel: `${availableCount}/${totalCount} metadata surfaces available`
  };
}

function selectedEvaluationStatus(
  evaluation: EvaluationSessionModel | null
): WorkstationMetadataStatus {
  if (!evaluation) {
    return "missing";
  }
  if (evaluation.state === "available") {
    return "available";
  }
  if (evaluation.state === "pending") {
    return "pending";
  }
  return "missing";
}

function selectedEvaluationDetail(
  evaluation: EvaluationSessionModel | null
): string {
  if (!evaluation) {
    return "No evaluation metadata is selected for this session.";
  }

  return evaluation.runId
    ? `${evaluation.statusLabel}: ${evaluation.runId}`
    : evaluation.statusLabel;
}

function metadataSummary(
  key: WorkstationMetadataKey,
  status: WorkstationMetadataStatus,
  detail: string
): WorkstationMetadataSummary {
  return {
    key,
    label: metadataLabels[key],
    status,
    detail
  };
}

function resolveArtifact(
  artifacts: ArtifactSummary[],
  artifactId?: string | null
): ArtifactSummary | null {
  if (artifacts.length === 0) {
    return null;
  }

  const normalizedId = artifactId?.trim();
  if (!normalizedId) {
    return artifacts[0] ?? null;
  }

  return artifacts.find((artifact) => artifact.id === normalizedId) ?? artifacts[0];
}

function resolveWorkflowStep(
  steps: WorkflowStepState[],
  nodeId: WorkflowNodeId
): WorkflowStepState | null {
  return steps.find((step) => step.nodeId === nodeId) ?? null;
}

function readTraceEventTimestamp(traceEvent: WorkflowRuntimeTraceEvent): string {
  return readEventTimestamp(traceEvent.event) ?? traceEvent.receivedAt;
}

function hasRequiredText(...values: string[]) {
  return values.every((value) => value.trim().length > 0);
}
