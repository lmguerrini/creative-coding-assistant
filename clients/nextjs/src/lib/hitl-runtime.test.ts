import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildHitlApprovalStreamEvent,
  createHitlApprovalRequest,
  getHitlApprovalStateLabel,
  isHitlApprovalBlockingState,
  isHitlApprovalTerminalState,
  summarizeHitlApprovalRequests,
  updateHitlApprovalRequest
} from "./hitl-runtime";

describe("hitl-runtime", () => {
  it("builds action-specific approval requests", () => {
    const request = createHitlApprovalRequest({
      actionId: "preview_runtime_restart",
      artifactTitle: "signal-orbit.p5.ts",
      id: "approval-1",
      nodeId: "generation",
      requestedAt: "2026-05-23T09:00:00.000Z",
      workspaceName: "Session workspace"
    });

    expect(request.title).toBe("Restart preview runtime");
    expect(request.confirmLabel).toBe("Restart runtime");
    expect(request.targetLabel).toBe("signal-orbit.p5.ts");
    expect(request.state).toBe("pending_approval");

    const bundleRequest = createHitlApprovalRequest({
      actionId: "project_bundle_export",
      artifactTitle: "Session workspace",
      id: "approval-5",
      nodeId: "generation",
      requestedAt: "2026-05-23T09:01:00.000Z",
      workspaceName: "Session workspace"
    });

    expect(bundleRequest.title).toBe("Export project bundle");
    expect(bundleRequest.confirmLabel).toBe("Export bundle");
    expect(bundleRequest.summary).toContain("Session workspace");
  });

  it("tracks blocking and terminal states distinctly", () => {
    expect(isHitlApprovalBlockingState("pending_approval")).toBe(true);
    expect(isHitlApprovalBlockingState("executing")).toBe(true);
    expect(isHitlApprovalBlockingState("completed")).toBe(false);
    expect(isHitlApprovalTerminalState("rejected")).toBe(true);
    expect(isHitlApprovalTerminalState("failed")).toBe(true);
    expect(isHitlApprovalTerminalState("approved")).toBe(false);
    expect(getHitlApprovalStateLabel("pending_approval")).toBe("Pending approval");
  });

  it("creates synthetic runtime events for approval lifecycle changes", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const pendingRequest = createHitlApprovalRequest({
      actionId: "artifact_download",
      artifactTitle: "webgpu-particle-field.ts",
      id: "approval-2",
      nodeId: snapshot.workflow.currentNode,
      requestedAt: "2026-05-23T10:15:00.000Z",
      workspaceName: snapshot.workspace.name
    });
    const completedRequest = updateHitlApprovalRequest(
      pendingRequest,
      "completed",
      "2026-05-23T10:15:04.000Z"
    );
    const event = buildHitlApprovalStreamEvent({
      request: completedRequest,
      sequence: 1001,
      state: "completed",
      workflow: snapshot.workflow
    });

    expect(event.event_type).toBe("tool_result");
    expect(event.payload.code).toBe("artifact_download_completed");
    expect(event.payload.message).toBe("Download artifact completed.");
    expect(event.payload.workflow).toMatchObject({
      step: snapshot.workflow.currentNode,
      current_step: snapshot.workflow.currentNode
    });
  });

  it("summarizes the latest and active approval request", () => {
    const first = createHitlApprovalRequest({
      actionId: "workspace_clear",
      id: "approval-3",
      nodeId: "generation",
      requestedAt: "2026-05-23T11:00:00.000Z",
      workspaceName: "Session workspace"
    });
    const second = updateHitlApprovalRequest(
      createHitlApprovalRequest({
        actionId: "preview_runtime_clear",
        artifactTitle: "preview-request.json",
        id: "approval-4",
        nodeId: "generation",
        requestedAt: "2026-05-23T11:02:00.000Z",
        workspaceName: "Session workspace"
      }),
      "completed",
      "2026-05-23T11:02:05.000Z"
    );
    const summary = summarizeHitlApprovalRequests([first, second]);

    expect(summary.pendingCount).toBe(1);
    expect(summary.activeRequest?.id).toBe("approval-3");
    expect(summary.latestRequest?.id).toBe("approval-4");
  });
});
