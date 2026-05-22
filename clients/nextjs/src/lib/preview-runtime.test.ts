import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot, type ArtifactSummary } from "./assistant-client";
import type { PreviewRuntimeSessionOverride } from "./preview-controller";
import type { AssistantStreamEvent } from "./assistant-stream";
import { buildPreviewRuntimeSummary, isArtifactPreviewable } from "./preview-runtime";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("preview runtime", () => {
  it("treats preview-capable artifacts as previewable", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(isArtifactPreviewable(snapshot.artifacts[0])).toBe(true);
    expect(isArtifactPreviewable(snapshot.artifacts[1])).toBe(true);
    expect(isArtifactPreviewable(snapshot.artifacts[2])).toBe(false);
  });

  it("derives a generating preview state from the active workflow", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: false,
        previewArtifactId: "source-sketch",
        streamError: null,
        traceEvents: [],
        workflow: snapshot.workflow
      })
    ).toMatchObject({
      state: "generating",
      status: "Generating",
      artifactName: "webgpu-particle-field.ts",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "webgpu-particle-field.ts",
      outputArtifactName: ""
    });
  });

  it("hydrates deferred preview output from terminal preview events", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: false,
        previewArtifactId: "source-sketch",
        streamError: null,
        traceEvents: [
          previewTraceEvent({
            artifactId: "source-sketch",
            previewArtifactId: "preview-manifest",
            status: "skipped",
            summary:
              "Preview pipeline foundation only; renderer execution is deferred."
          })
        ],
        workflow: {
          ...snapshot.workflow,
          currentNode: "finalization",
          currentStep: "Finalization",
          status: "Completed"
        }
      })
    ).toMatchObject({
      state: "ready",
      status: "Deferred renderer",
      artifactName: "preview-request.json",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "webgpu-particle-field.ts",
      outputArtifactName: "preview-request.json",
      renderer: "preview.noop",
      target: "Browser sandbox"
    });
  });

  it("marks non-previewable artifact context as unavailable", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const notesArtifact = snapshot.artifacts.find(
      (artifact) => artifact.id === "session-notes"
    ) as ArtifactSummary;

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: false,
        previewArtifactId: notesArtifact.id,
        streamError: null,
        traceEvents: [],
        workflow: {
          ...snapshot.workflow,
          currentNode: "finalization",
          currentStep: "Finalization",
          status: "Completed"
        }
      })
    ).toMatchObject({
      state: "unavailable",
      status: "Unavailable",
      artifactName: "projection-notes.md"
    });
  });

  it("marks previewable context as unavailable when the workflow finishes without preview output", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: false,
        previewArtifactId: "source-sketch",
        streamError: null,
        traceEvents: [],
        workflow: {
          ...snapshot.workflow,
          currentNode: "finalization",
          currentStep: "Finalization",
          status: "Completed"
        }
      })
    ).toMatchObject({
      state: "unavailable",
      status: "Unavailable",
      artifactName: "webgpu-particle-field.ts",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "webgpu-particle-field.ts"
    });
  });

  it("surfaces preview failures from terminal preview events", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: snapshot.preview,
        isOpen: false,
        previewArtifactId: "source-sketch",
        streamError: null,
        traceEvents: [
          previewTraceEvent({
            artifactId: "source-sketch",
            message: "Renderer sandbox failed.",
            status: "failed",
            summary: "Preview renderer failed."
          })
        ],
        workflow: {
          ...snapshot.workflow,
          currentNode: "failure",
          currentStep: "Failure",
          status: "Failed"
        }
      })
    ).toMatchObject({
      state: "error",
      status: "Preview failed",
      artifactName: "webgpu-particle-field.ts"
    });
  });

  it("applies restart overrides without relying on stale preview output", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const sessionOverride: PreviewRuntimeSessionOverride = {
      artifactId: "source-sketch",
      mode: "restarting",
      requestedAt: "2026-05-22T11:00:00Z"
    };

    expect(
      buildPreviewRuntimeSummary({
        artifacts: snapshot.artifacts,
        basePreview: {
          ...snapshot.preview,
          outputArtifactName: "preview-request.json",
          state: "ready",
          status: "Ready when opened",
          summary: "Preview output is ready."
        },
        isOpen: true,
        previewArtifactId: "preview-manifest",
        sessionOverride,
        streamError: null,
        traceEvents: [
          previewTraceEvent({
            artifactId: "source-sketch",
            previewArtifactId: "preview-manifest",
            status: "skipped",
            summary:
              "Preview pipeline foundation only; renderer execution is deferred."
          })
        ],
        workflow: {
          ...snapshot.workflow,
          currentNode: "finalization",
          currentStep: "Finalization",
          status: "Completed"
        }
      })
    ).toMatchObject({
      state: "generating",
      status: "Restarting",
      outputArtifactName: "",
      trigger: "Preview restart preview-request.json"
    });
  });
});

function previewTraceEvent({
  artifactId,
  message = null,
  previewArtifactId = null,
  status,
  summary
}: {
  artifactId: string;
  message?: string | null;
  previewArtifactId?: string | null;
  status: "succeeded" | "failed" | "skipped";
  summary: string;
}): WorkflowRuntimeTraceEvent {
  const event: AssistantStreamEvent = {
    event_type: "preview_artifact",
    sequence: 4,
    payload: {
      artifact_id: artifactId,
      emitted_at: "2026-05-22T11:00:00Z",
      status,
      result: {
        completed_at: "2026-05-22T11:00:00Z",
        ...(message
          ? {
              error: {
                message
              }
            }
          : {}),
        ...(previewArtifactId ? { preview_artifact_id: previewArtifactId } : {}),
        provenance: {
          renderer_id: "preview.noop"
        },
        request: {
          target: "browser_sandbox"
        },
        summary
      }
    }
  };

  return {
    event,
    receivedAt: "2026-05-22T11:00:00Z",
    receivedAtMs: Date.parse("2026-05-22T11:00:00Z")
  };
}
