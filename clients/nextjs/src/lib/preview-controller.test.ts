import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildPreviewControllerModel,
  createPreviewSessionOverride
} from "./preview-controller";

describe("preview controller", () => {
  it("derives ready-state controls from the preview summary", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewControllerModel({
        isFullscreen: false,
        preview: {
          ...snapshot.preview,
          active: true,
          outputArtifactName: "preview-request.json",
          state: "ready",
          status: "Preview open"
        },
        sessionOverride: null
      })
    ).toMatchObject({
      canClear: true,
      canFullscreen: true,
      canReload: false,
      canReset: true,
      canRestart: true,
      isFullscreen: false,
      isSessionOverridden: false,
      sessionLabel: "Live"
    });
  });

  it("surfaces cleared session overrides as warning-state controls", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewControllerModel({
        isFullscreen: true,
        preview: snapshot.preview,
        sessionOverride: createPreviewSessionOverride("source-sketch", "cleared")
      })
    ).toMatchObject({
      canClear: false,
      canReload: true,
      canReset: true,
      isFullscreen: true,
      isSessionOverridden: true,
      sessionLabel: "Cleared"
    });
  });
});
