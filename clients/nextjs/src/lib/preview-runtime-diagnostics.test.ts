import { describe, expect, it } from "vitest";
import {
  buildPreviewRuntimeOverlayModel,
  createPreviewRuntimeMetricsTracker
} from "./preview-runtime-diagnostics";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import { buildPreviewRendererRoute } from "./preview-renderers";
import { buildPreviewRuntimeSource } from "./preview-runtime-adapters";

describe("preview runtime diagnostics", () => {
  it("tracks rolling fps and frame timing for running renderers", () => {
    let nowMs = 0;
    const tracker = createPreviewRuntimeMetricsTracker(
      {
        detail: "Rendering a controlled scene.",
        label: "Three.js runtime running",
        state: "running",
        error: null
      },
      {
        clock: () => nowMs,
        publishIntervalMs: 0
      }
    );

    tracker.recordFrame({ renderedAtMs: 0 });
    nowMs = 16;
    tracker.recordFrame({ renderedAtMs: 16 });
    nowMs = 32;
    tracker.recordFrame({ renderedAtMs: 32 });
    nowMs = 48;
    const snapshot = tracker.recordFrame({ renderedAtMs: 48 }) ?? tracker.snapshot();

    expect(snapshot.metricsAvailable).toBe(true);
    expect(snapshot.frameCount).toBe(4);
    expect(snapshot.fps).toBeCloseTo(62.5, 1);
    expect(snapshot.frameTimeMs).toBeCloseTo(16, 1);
    expect(snapshot.health).toBe("nominal");
  });

  it("classifies runtime-health thresholds and terminal errors deterministically", () => {
    const snapshotAtInterval = (intervalMs: number) => {
      const tracker = createPreviewRuntimeMetricsTracker(
        {
          detail: "Rendering a controlled scene.",
          label: "Three.js runtime running",
          state: "running",
          error: null
        },
        { publishIntervalMs: 0 }
      );

      tracker.recordFrame({ renderedAtMs: 0 });
      tracker.recordFrame({ renderedAtMs: intervalMs });
      return tracker.snapshot();
    };

    expect(snapshotAtInterval(21.9).health).toBe("nominal");
    expect(snapshotAtInterval(22).health).toBe("stressed");
    expect(snapshotAtInterval(34).health).toBe("degraded");

    const tracker = createPreviewRuntimeMetricsTracker(
      {
        detail: "Rendering a controlled scene.",
        label: "Three.js runtime running",
        state: "running",
        error: null
      },
      { publishIntervalMs: 0 }
    );
    tracker.recordFrame({ renderedAtMs: 0 });
    tracker.recordFrame({ renderedAtMs: 16 });
    tracker.publishStatus({
      detail: "WebGL became unavailable.",
      label: "Three.js runtime failed",
      state: "error",
      error: {
        category: "renderer",
        debugMessage: "WebGL became unavailable.",
        id: "renderer:three",
        recoverable: true,
        resetLabel: "Reset preview session",
        retryLabel: "Reload preview state",
        subsystem: "three_renderer",
        suggestedAction: "Reload the preview state.",
        type: "webgl_unavailable",
        userMessage: "The Three.js preview runtime is unavailable."
      }
    });

    expect(tracker.snapshot()).toMatchObject({
      health: "failed",
      runtimeState: "error"
    });
  });

  it("builds an operator-friendly overlay model when metrics are unavailable or failed", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const route = buildPreviewRendererRoute({
      artifacts: [
        {
          ...snapshot.artifacts[0],
          summary: "Fragment shader with gl_FragColor and uniforms.",
          title: "chromatic-field.frag",
          language: "GLSL"
        }
      ],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: "chromatic-field.frag",
        sourceArtifactName: "chromatic-field.frag"
      },
      previewArtifactId: "source-sketch"
    });
    const runtimeSource = buildPreviewRuntimeSource({
      code: {
        ...snapshot.code,
        title: "chromatic-field.frag",
        language: "GLSL",
        excerpt: ["void main() {", "  gl_FragColor = vec4(1.0);", "}"]
      },
      route
    });
    const tracker = createPreviewRuntimeMetricsTracker({
      detail: "Shader compile error.",
      diagnostics: [
        "ERROR: 0:12: 'texture2D' : no matching overloaded function found"
      ],
      label: "GLSL runtime failed",
      state: "error",
      error: snapshot.preview.error ?? {
        category: "renderer",
        debugMessage:
          "ERROR: 0:12: 'texture2D' : no matching overloaded function found",
        id: "renderer:glsl",
        recoverable: true,
        resetLabel: "Reset preview session",
        retryLabel: "Reload preview state",
        subsystem: "glsl_renderer",
        suggestedAction: "Reload the preview state.",
        type: "shader_program_failed",
        userMessage: "The bounded GLSL runtime could not compile the current shader."
      }
    });
    const overlay = buildPreviewRuntimeOverlayModel({
      kind: "glsl",
      route,
      runtimeSource,
      snapshot: tracker.snapshot(),
      status: {
        detail: "Shader compile error.",
        diagnostics: [
          "ERROR: 0:12: 'texture2D' : no matching overloaded function found"
        ],
        label: "GLSL runtime failed",
        state: "error",
        error: snapshot.preview.error ?? {
          category: "renderer",
          debugMessage:
            "ERROR: 0:12: 'texture2D' : no matching overloaded function found",
          id: "renderer:glsl",
          recoverable: true,
          resetLabel: "Reset preview session",
          retryLabel: "Reload preview state",
          subsystem: "glsl_renderer",
          suggestedAction: "Reload the preview state.",
          type: "shader_program_failed",
          userMessage: "The bounded GLSL runtime could not compile the current shader."
        }
      }
    });

    expect(overlay.healthLabel).toBe("Failed");
    expect(overlay.metrics).toContainEqual(
      expect.objectContaining({
        id: "fps",
        value: "N/A"
      })
    );
    expect(overlay.diagnostics[0]).toContain("texture2D");
  });

  it("reports Tone.js ready and stopped states without pretending audio is running", () => {
    const readyTracker = createPreviewRuntimeMetricsTracker({
      detail: "Audio is armed.",
      label: "Tone.js runtime ready",
      state: "ready",
      error: null
    });
    const stoppedTracker = createPreviewRuntimeMetricsTracker({
      detail: "Audio output is silent.",
      label: "Tone.js runtime stopped",
      state: "stopped",
      error: null
    });

    expect(readyTracker.snapshot()).toMatchObject({
      frameCount: 0,
      health: "warming",
      metricsAvailable: false,
      runtimeState: "ready"
    });
    expect(stoppedTracker.snapshot()).toMatchObject({
      frameCount: 0,
      health: "unavailable",
      metricsAvailable: false,
      runtimeState: "stopped"
    });
  });
});
