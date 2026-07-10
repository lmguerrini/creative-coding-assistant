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
      artifactName: "aurora-field.p5.js",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "aurora-field.p5.js",
      outputArtifactName: "",
      target: "Browser preview",
      targetId: "browser_sandbox"
    });
  });

  it("hydrates sandbox preview output from terminal preview events", () => {
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
            previewArtifactId: "source-sketch",
            rendererId: "surface.p5",
            status: "succeeded",
            summary: "p5.js runtime ready for sandbox execution."
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
      status: "Ready when opened",
      artifactName: "aurora-field.p5.js",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "aurora-field.p5.js",
      outputArtifactName: "aurora-field.p5.js",
      renderer: "surface.p5",
      target: "Browser preview",
      targetId: "browser_sandbox"
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
      artifactName: "projection-notes.md",
      targetId: ""
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
      artifactName: "aurora-field.p5.js",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "aurora-field.p5.js"
    });
  });

  it("does not restore a persisted preview claim for standalone Three.js HTML", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifact: ArtifactSummary = {
      ...snapshot.artifacts[0],
      content: "<!doctype html><html><script>new THREE.Scene()</script></html>",
      rendererId: "surface.three",
      runtime: "three",
      title: "generated-scene.three.ts"
    };

    const result = buildPreviewRuntimeSummary({
      artifacts: [artifact],
      basePreview: {
        ...snapshot.preview,
        artifactName: artifact.title,
        available: true,
        sourceArtifactId: artifact.id,
        sourceArtifactName: artifact.title,
        state: "ready",
        title: "Preview available"
      },
      isOpen: true,
      previewArtifactId: artifact.id,
      streamError: null,
      traceEvents: [],
      workflow: {
        ...snapshot.workflow,
        currentNode: "finalization",
        currentStep: "Finalization",
        status: "Completed"
      }
    });

    expect(isArtifactPreviewable(artifact)).toBe(false);
    expect(result).toMatchObject({
      active: false,
      available: false,
      state: "unavailable",
      title: "Preview unavailable"
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
      artifactName: "aurora-field.p5.js",
      error: {
        category: "preview_runtime",
        subsystem: "surface.p5",
        type: "preview_runtime_failed",
        resetLabel: "Reset preview session"
      }
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
            previewArtifactId: "source-sketch",
            rendererId: "surface.p5",
            status: "succeeded",
            summary: "p5.js runtime ready for sandbox execution."
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
  code = "preview_runtime_failed",
  message = null,
  previewArtifactId = null,
  rendererId = "surface.p5",
  retryable = false,
  status,
  summary
}: {
  artifactId: string;
  code?: string;
  message?: string | null;
  previewArtifactId?: string | null;
  rendererId?: string;
  retryable?: boolean;
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
                code,
                details: {
                  reset_label: "Reset preview session"
                },
                message,
                retryable
              }
            }
          : {}),
        ...(previewArtifactId ? { preview_artifact_id: previewArtifactId } : {}),
        provenance: {
          renderer_id: rendererId
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
