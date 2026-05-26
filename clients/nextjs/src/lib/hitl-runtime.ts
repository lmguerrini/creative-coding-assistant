import type {
  AssistantWorkspaceSnapshot,
  WorkflowNodeId,
  WorkflowStepState
} from "./assistant-client";
import type { AssistantStreamEvent } from "./assistant-stream";

export type HitlActionId =
  | "workspace_clear"
  | "preview_runtime_restart"
  | "preview_runtime_clear"
  | "preview_runtime_reset"
  | "artifact_download"
  | "artifact_export"
  | "project_bundle_export";

export type HitlActionState =
  | "idle"
  | "pending_approval"
  | "approved"
  | "rejected"
  | "executing"
  | "completed"
  | "failed";

export type HitlActionKind = "destructive" | "runtime" | "transfer";

export type HitlApprovalRequest = {
  id: string;
  actionId: HitlActionId;
  state: HitlActionState;
  kind: HitlActionKind;
  nodeId: WorkflowNodeId;
  title: string;
  summary: string;
  detail: string;
  confirmLabel: string;
  cancelLabel: string;
  targetLabel: string;
  requestedAt: string;
  resolvedAt: string | null;
  failureReason: string | null;
};

export type HitlApprovalSummary = {
  activeRequest: HitlApprovalRequest | null;
  latestRequest: HitlApprovalRequest | null;
  pendingCount: number;
};

type HitlApprovalRequestInput = {
  actionId: HitlActionId;
  artifactTitle?: string | null;
  id: string;
  nodeId: WorkflowNodeId;
  requestedAt?: string;
  workspaceName: string;
};

const terminalStates = new Set<HitlActionState>(["rejected", "completed", "failed"]);
const blockingStates = new Set<HitlActionState>([
  "pending_approval",
  "approved",
  "executing"
]);

export function createHitlApprovalRequest({
  actionId,
  artifactTitle,
  id,
  nodeId,
  requestedAt = new Date().toISOString(),
  workspaceName
}: HitlApprovalRequestInput): HitlApprovalRequest {
  switch (actionId) {
    case "workspace_clear":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "destructive",
        nodeId,
        title: "Clear workspace session",
        summary: `Reset ${workspaceName} to the starter snapshot.`,
        detail:
          "This clears the current chat activity, workflow trace, preview runtime state, and selected artifact context while keeping layout and theme preferences in place.",
        confirmLabel: "Clear workspace",
        cancelLabel: "Keep session",
        targetLabel: workspaceName,
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "preview_runtime_restart":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "runtime",
        nodeId,
        title: "Restart preview runtime",
        summary: `Start a fresh preview runtime for ${artifactTitle ?? "the active preview"}.`,
        detail:
          "The preview shelf will reopen and rebuild the current runtime session from the active preview source.",
        confirmLabel: "Restart runtime",
        cancelLabel: "Keep runtime",
        targetLabel: artifactTitle ?? "Active preview",
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "preview_runtime_clear":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "runtime",
        nodeId,
        title: "Clear preview runtime",
        summary: `Clear the current preview state for ${artifactTitle ?? "the active preview"}.`,
        detail:
          "This drops the current preview runtime state and leaves the shelf ready for a manual reload or reset.",
        confirmLabel: "Clear runtime",
        cancelLabel: "Keep runtime",
        targetLabel: artifactTitle ?? "Active preview",
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "preview_runtime_reset":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "runtime",
        nodeId,
        title: "Reset preview runtime",
        summary: `Restore preview routing to ${artifactTitle ?? "the latest previewable artifact"}.`,
        detail:
          "This resets the preview session override and returns the shelf to the latest previewable artifact in the workspace.",
        confirmLabel: "Reset runtime",
        cancelLabel: "Keep runtime",
        targetLabel: artifactTitle ?? "Latest previewable artifact",
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "artifact_download":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "transfer",
        nodeId,
        title: "Download artifact",
        summary: `Download ${artifactTitle ?? "the selected artifact"} to a local file.`,
        detail:
          "This starts a browser download for the selected artifact document without changing the current workspace state.",
        confirmLabel: "Download file",
        cancelLabel: "Stay in workspace",
        targetLabel: artifactTitle ?? "Selected artifact",
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "artifact_export":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "transfer",
        nodeId,
        title: "Export artifact",
        summary: `Export ${artifactTitle ?? "the selected artifact"} from the workspace.`,
        detail:
          "This starts a local file export for the selected artifact document while keeping the rest of the session untouched.",
        confirmLabel: "Export file",
        cancelLabel: "Stay in workspace",
        targetLabel: artifactTitle ?? "Selected artifact",
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    case "project_bundle_export":
      return {
        id,
        actionId,
        state: "pending_approval",
        kind: "transfer",
        nodeId,
        title: "Export project bundle",
        summary: `Export ${workspaceName} as a local ZIP bundle.`,
        detail:
          "This packages generated artifacts, session metadata, workflow context, retrieval summaries, preview routing data, and multimodal image references into a single download.",
        confirmLabel: "Export bundle",
        cancelLabel: "Stay in workspace",
        targetLabel: artifactTitle ?? workspaceName,
        requestedAt,
        resolvedAt: null,
        failureReason: null
      };
    default:
      return assertNever(actionId);
  }
}

export function updateHitlApprovalRequest(
  request: HitlApprovalRequest,
  state: HitlActionState,
  at = new Date().toISOString(),
  failureReason: string | null = null
): HitlApprovalRequest {
  return {
    ...request,
    state,
    failureReason,
    resolvedAt: isHitlApprovalTerminalState(state) ? at : request.resolvedAt
  };
}

export function buildHitlApprovalStreamEvent({
  request,
  sequence,
  state,
  workflow
}: {
  request: HitlApprovalRequest;
  sequence: number;
  state: HitlActionState;
  workflow: AssistantWorkspaceSnapshot["workflow"];
}): AssistantStreamEvent {
  const emittedAt = request.resolvedAt ?? request.requestedAt;
  const code = buildHitlApprovalCode(request.actionId, state);

  return {
    event_type: isHitlApprovalBlockingState(state) ? "tool_start" : "tool_result",
    sequence,
    payload: {
      code,
      emitted_at: emittedAt,
      message: buildHitlApprovalMessage(request, state),
      approval: {
        id: request.id,
        action_id: request.actionId,
        state,
        target_label: request.targetLabel,
        kind: request.kind
      },
      workflow: {
        step: request.nodeId,
        phase: normalizeWorkflowPhase(workflow.status),
        status: workflow.status,
        current_step: workflow.currentNode,
        completed_steps: workflow.steps
          .filter((step) => step.state === "complete")
          .map((step) => step.nodeId),
        skipped_steps: workflow.steps
          .filter((step) => step.state === "skipped")
          .map((step) => step.nodeId),
        refinement_count: countRefinementAttempts(workflow.steps),
        review_outcome: readReviewOutcome(workflow.steps),
        review_reasons: []
      }
    }
  };
}

export function summarizeHitlApprovalRequests(
  requests: HitlApprovalRequest[]
): HitlApprovalSummary {
  let latestRequest: HitlApprovalRequest | null = null;
  let activeRequest: HitlApprovalRequest | null = null;
  let pendingCount = 0;

  for (const request of requests) {
    latestRequest = request;
    if (request.state === "pending_approval") {
      pendingCount += 1;
    }
    if (isHitlApprovalBlockingState(request.state)) {
      activeRequest = request;
    }
  }

  return {
    activeRequest,
    latestRequest,
    pendingCount
  };
}

export function getHitlApprovalStateLabel(state: HitlActionState) {
  switch (state) {
    case "idle":
      return "Idle";
    case "pending_approval":
      return "Pending approval";
    case "approved":
      return "Approved";
    case "rejected":
      return "Rejected";
    case "executing":
      return "Executing";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return assertNever(state);
  }
}

export function isHitlApprovalBlockingState(state: HitlActionState) {
  return blockingStates.has(state);
}

export function isHitlApprovalTerminalState(state: HitlActionState) {
  return terminalStates.has(state);
}

function buildHitlApprovalCode(actionId: HitlActionId, state: HitlActionState) {
  switch (state) {
    case "pending_approval":
      return `${actionId}_approval_requested`;
    case "approved":
      return `${actionId}_approval_approved`;
    case "rejected":
      return `${actionId}_approval_rejected`;
    case "executing":
      return `${actionId}_executing`;
    case "completed":
      return `${actionId}_completed`;
    case "failed":
      return `${actionId}_failed`;
    case "idle":
      return `${actionId}_idle`;
    default:
      return assertNever(state);
  }
}

function buildHitlApprovalMessage(
  request: HitlApprovalRequest,
  state: HitlActionState
) {
  switch (state) {
    case "pending_approval":
      return `Approval requested for ${request.title.toLowerCase()}.`;
    case "approved":
      return `Operator approved ${request.title.toLowerCase()}.`;
    case "rejected":
      return `Operator rejected ${request.title.toLowerCase()}.`;
    case "executing":
      return `Executing ${request.title.toLowerCase()}.`;
    case "completed":
      return `${request.title} completed.`;
    case "failed":
      return request.failureReason
        ? `${request.title} failed. ${request.failureReason}`
        : `${request.title} failed.`;
    case "idle":
      return request.summary;
    default:
      return assertNever(state);
  }
}

function countRefinementAttempts(steps: WorkspaceStepState[]) {
  const refinementStep = steps.find((step) => step.nodeId === "refinement");
  return refinementStep?.state === "complete" ? 1 : 0;
}

function readReviewOutcome(steps: WorkspaceStepState[]) {
  const reviewStep = steps.find((step) => step.nodeId === "review");
  return reviewStep?.state === "complete" ? "approved" : null;
}

function normalizeWorkflowPhase(
  status: AssistantWorkspaceSnapshot["workflow"]["status"]
) {
  const normalized = status.toLowerCase();
  if (normalized.includes("fail")) {
    return "failed";
  }

  return "running";
}

function assertNever(value: never): never {
  throw new Error(`Unexpected HITL runtime value: ${String(value)}`);
}

type WorkspaceStepState = Pick<WorkflowStepState, "nodeId" | "state">;
