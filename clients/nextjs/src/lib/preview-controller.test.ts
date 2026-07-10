import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import { buildPreviewRendererRoute } from "./preview-renderers";
import {
  buildPreviewControllerModel,
  createPreviewSessionOverride
} from "./preview-controller";

describe("preview controller", () => {
  it("derives ready-state controls from the preview summary", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const preview = {
      ...snapshot.preview,
      active: true,
      outputArtifactName: "preview-request.json",
      state: "ready" as const,
      status: "Preview open"
    };
    const route = buildPreviewRendererRoute({
      artifacts: snapshot.artifacts,
      preview,
      previewArtifactId: "preview-manifest"
    });

    expect(
      buildPreviewControllerModel({
        isFullscreen: false,
        preview,
        route,
        sessionOverride: null
      })
    ).toMatchObject({
      canClear: true,
      canFullscreen: true,
      canReload: true,
      canRestart: true,
      isFullscreen: false,
      isSessionOverridden: false,
      sessionLabel: "Live",
      indicators: expect.arrayContaining([
        expect.objectContaining({ id: "artifact", value: "preview-request.json" }),
        expect.objectContaining({ id: "target", value: "JSON panel" }),
        expect.objectContaining({ id: "surface", value: "JSON panel surface" }),
        expect.objectContaining({ id: "support", value: "Foundation ready" })
      ])
    });
  });

  it("surfaces cleared session overrides as warning-state controls", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const route = buildPreviewRendererRoute({
      artifacts: snapshot.artifacts,
      preview: snapshot.preview,
      previewArtifactId: "source-sketch"
    });

    expect(
      buildPreviewControllerModel({
        isFullscreen: true,
        preview: snapshot.preview,
        route,
        sessionOverride: createPreviewSessionOverride("source-sketch", "cleared")
      })
    ).toMatchObject({
      canClear: false,
      canReload: true,
      isFullscreen: true,
      isSessionOverridden: true,
      sessionLabel: "Cleared"
    });
  });

  it("returns settled recovery sessions to the base preview controls", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const preview = {
      ...snapshot.preview,
      active: true,
      outputArtifactName: "preview-request.json",
      state: "ready" as const,
      status: "Preview open"
    };
    const route = buildPreviewRendererRoute({
      artifacts: snapshot.artifacts,
      preview,
      previewArtifactId: "preview-manifest"
    });

    expect(
      buildPreviewControllerModel({
        isFullscreen: false,
        preview,
        route,
        sessionOverride: {
          artifactId: "source-sketch",
          mode: "settled",
          requestedAt: "2026-07-10T13:00:00Z"
        }
      })
    ).toMatchObject({
      canReload: true,
      isSessionOverridden: false,
      sessionLabel: "Live"
    });
  });
});
