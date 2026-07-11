import type { AssistantWorkspaceSnapshot } from "./assistant-client";
import { trimToSentence, uniqueStrings } from "./text-utils";
import type {
  WorkstationMetadataSummary,
  WorkstationReadinessState,
  WorkstationState
} from "./workstation-state";

export type SessionCompletionStatus =
  | "idle"
  | "running"
  | "completed"
  | "needs_attention";

export type SessionIntelligenceMetadata = {
  session_summary: string;
  active_request_summary: string;
  completion_status: SessionCompletionStatus;
  available_metadata_groups: string[];
  session_warnings: string[];
  recommended_next_user_actions: string[];
};

export type SessionIntelligenceMetadataInput = Partial<
  SessionIntelligenceMetadata
>;

export type SessionIntelligenceModel = {
  source: "derived" | "stream";
  readiness: WorkstationReadinessState;
  statusLabel: string;
  metadata: SessionIntelligenceMetadata;
  availableMetadataCount: number;
  warningCount: number;
  recommendedActionCount: number;
};

export type BuildSessionIntelligenceInput = {
  snapshot: AssistantWorkspaceSnapshot;
  workstationState: WorkstationState;
  streamedMetadata?: SessionIntelligenceMetadataInput | null;
};

const completionStatusLabels = {
  completed: "Success",
  idle: "Idle",
  needs_attention: "Needs attention",
  running: "Running"
} satisfies Record<SessionCompletionStatus, string>;

export function buildSessionIntelligenceModel({
  snapshot,
  streamedMetadata = null,
  workstationState
}: BuildSessionIntelligenceInput): SessionIntelligenceModel {
  const derivedMetadata = deriveSessionIntelligenceMetadata(
    snapshot,
    workstationState
  );
  const metadata = normalizeSessionIntelligenceMetadata(
    streamedMetadata,
    derivedMetadata
  );

  return {
    source: streamedMetadata ? "stream" : "derived",
    readiness: workstationState.readiness.state,
    statusLabel: completionStatusLabels[metadata.completion_status],
    metadata,
    availableMetadataCount: metadata.available_metadata_groups.length,
    warningCount: metadata.session_warnings.length,
    recommendedActionCount: metadata.recommended_next_user_actions.length
  };
}

export function readSessionIntelligenceMetadata(
  payload: Record<string, unknown>
): SessionIntelligenceMetadataInput | null {
  const source = readRecord(payload.session_intelligence) ?? payload;
  const metadata: SessionIntelligenceMetadataInput = {};

  const sessionSummary = readString(source.session_summary);
  if (sessionSummary) {
    metadata.session_summary = sessionSummary;
  }

  const activeRequestSummary = readString(source.active_request_summary);
  if (activeRequestSummary) {
    metadata.active_request_summary = activeRequestSummary;
  }

  const completionStatus = readCompletionStatus(source.completion_status);
  if (completionStatus) {
    metadata.completion_status = completionStatus;
  }

  const availableMetadataGroups = readStringList(
    source.available_metadata_groups
  );
  if (availableMetadataGroups) {
    metadata.available_metadata_groups = availableMetadataGroups;
  }

  const sessionWarnings = readStringList(source.session_warnings);
  if (sessionWarnings) {
    metadata.session_warnings = sessionWarnings;
  }

  const recommendedActions = readStringList(
    source.recommended_next_user_actions
  );
  if (recommendedActions) {
    metadata.recommended_next_user_actions = recommendedActions;
  }

  return Object.keys(metadata).length > 0 ? metadata : null;
}

function deriveSessionIntelligenceMetadata(
  snapshot: AssistantWorkspaceSnapshot,
  workstationState: WorkstationState
): SessionIntelligenceMetadata {
  const availableMetadata = Object.values(workstationState.metadata).filter(
    (metadata) => metadata.status === "available"
  );
  const completionStatus = deriveCompletionStatus(snapshot, workstationState);
  const sessionWarnings = deriveSessionWarnings(workstationState);

  return {
    session_summary: deriveSessionSummary({
      availableMetadata,
      snapshot,
      workstationState
    }),
    active_request_summary: deriveActiveRequestSummary(
      snapshot,
      workstationState
    ),
    completion_status: completionStatus,
    available_metadata_groups: availableMetadata.map((metadata) => metadata.label),
    session_warnings: sessionWarnings,
    recommended_next_user_actions: deriveRecommendedNextUserActions({
      completionStatus,
      sessionWarnings,
      snapshot,
      workstationState
    })
  };
}

function normalizeSessionIntelligenceMetadata(
  metadata: SessionIntelligenceMetadataInput | null,
  fallback: SessionIntelligenceMetadata
): SessionIntelligenceMetadata {
  if (!metadata) {
    return fallback;
  }

  return {
    session_summary: readString(metadata.session_summary) ?? fallback.session_summary,
    active_request_summary:
      readString(metadata.active_request_summary) ??
      fallback.active_request_summary,
    completion_status:
      readCompletionStatus(metadata.completion_status) ??
      fallback.completion_status,
    available_metadata_groups:
      normalizeStringList(metadata.available_metadata_groups) ??
      fallback.available_metadata_groups,
    session_warnings:
      normalizeStringList(metadata.session_warnings) ?? fallback.session_warnings,
    recommended_next_user_actions:
      normalizeStringList(metadata.recommended_next_user_actions) ??
      fallback.recommended_next_user_actions
  };
}

function deriveCompletionStatus(
  snapshot: AssistantWorkspaceSnapshot,
  workstationState: WorkstationState
): SessionCompletionStatus {
  if (
    workstationState.currentRun.state === "error" ||
    workstationState.readiness.state === "degraded"
  ) {
    return "needs_attention";
  }

  if (
    workstationState.currentRun.state === "streaming" ||
    snapshot.workflow.status.toLowerCase() === "running"
  ) {
    return "running";
  }

  if (workstationState.currentRun.state === "completed") {
    return "completed";
  }

  return "idle";
}

function deriveSessionSummary({
  availableMetadata,
  snapshot,
  workstationState
}: {
  availableMetadata: WorkstationMetadataSummary[];
  snapshot: AssistantWorkspaceSnapshot;
  workstationState: WorkstationState;
}): string {
  if (workstationState.readiness.state === "empty") {
    return `${snapshot.workspace.name} is ready for the first creative request.`;
  }

  if (workstationState.readiness.state === "degraded") {
    return `${snapshot.workspace.name} has session warnings that need review.`;
  }

  if (workstationState.currentRun.state === "streaming") {
    return `${snapshot.workspace.name} is collecting live run metadata.`;
  }

  const artifactCount = snapshot.artifacts.length;
  return `${snapshot.workspace.name} has ${artifactCount} artifact${
    artifactCount === 1 ? "" : "s"
  } and ${availableMetadata.length} available metadata group${
    availableMetadata.length === 1 ? "" : "s"
  }.`;
}

function deriveActiveRequestSummary(
  snapshot: AssistantWorkspaceSnapshot,
  workstationState: WorkstationState
): string {
  const latestUserMessage = [...snapshot.messages]
    .reverse()
    .find((message) => message.role === "user");

  if (workstationState.currentRun.latestEventType) {
    return `${workstationState.selection.activeWorkflowStep?.displayLabel ?? snapshot.workflow.currentStep} received ${workstationState.currentRun.latestEventType}.`;
  }

  if (latestUserMessage) {
    return `Last request: ${trimToSentence(latestUserMessage.content)}`;
  }

  return "No active request has been submitted yet.";
}

function deriveSessionWarnings(workstationState: WorkstationState): string[] {
  const warnings = Object.values(workstationState.metadata)
    .filter((metadata) => metadata.status === "error")
    .map((metadata) => `${metadata.label}: ${metadata.detail}`);

  if (
    workstationState.readiness.state === "degraded" &&
    workstationState.readiness.detail
  ) {
    warnings.unshift(workstationState.readiness.detail);
  }

  return uniqueStrings(warnings, { dropEmpty: true, trim: true });
}

function deriveRecommendedNextUserActions({
  completionStatus,
  sessionWarnings,
  snapshot,
  workstationState
}: {
  completionStatus: SessionCompletionStatus;
  sessionWarnings: string[];
  snapshot: AssistantWorkspaceSnapshot;
  workstationState: WorkstationState;
}): string[] {
  const actions: string[] = [];

  if (sessionWarnings.length > 0) {
    actions.push("Review session warnings before continuing.");
  }

  if (workstationState.readiness.state === "empty") {
    actions.push("Send a creative prompt to start the session.");
  } else if (completionStatus === "running") {
    actions.push("Wait for the active response to finish before sending another request.");
  } else if (snapshot.preview.available) {
    actions.push("Inspect the current preview before requesting another pass.");
  } else if (snapshot.artifacts.length > 0) {
    actions.push("Review the selected source in the Code inspector.");
  }

  if (
    completionStatus === "completed" &&
    workstationState.metadata.selected_evaluation.status === "missing"
  ) {
    actions.push("Run a session evaluation when quality evidence is needed.");
  }

  return uniqueStrings(actions, { dropEmpty: true, trim: true });
}

function readCompletionStatus(
  value: unknown
): SessionCompletionStatus | null {
  return typeof value === "string" && value in completionStatusLabels
    ? (value as SessionCompletionStatus)
    : null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function readStringList(value: unknown): string[] | null {
  return Array.isArray(value) ? normalizeStringList(value) : null;
}

function normalizeStringList(value: unknown): string[] | null {
  if (!Array.isArray(value)) {
    return null;
  }

  const values = uniqueStrings(
    value.filter((item): item is string => typeof item === "string"),
    { dropEmpty: true, trim: true }
  );
  return values.length > 0 ? values : null;
}
