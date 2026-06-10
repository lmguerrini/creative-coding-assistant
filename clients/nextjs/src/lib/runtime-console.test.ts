import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildPreviewRuntimeSource,
  type PreviewRuntimeStatus
} from "./preview-runtime-adapters";
import { buildPreviewRendererRoute } from "./preview-renderers";
import {
  buildRuntimeConsoleModel,
  type RuntimeConsoleLiveSnapshot
} from "./runtime-console";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

describe("runtime console diagnostics", () => {
  it("projects healthy runtime metrics and chronological lifecycle events", () => {
    const fixture = createRuntimeFixture();
    const traceEvents = [
      createRuntimeEvent(1, "2026-06-09T10:00:00.000Z", {
        code: "preview_runtime_running",
        message: "Renderer started.",
        preview_runtime: {
          kind: "p5",
          runtime_id: "runtime-1",
          state: "running"
        }
      }),
      createRuntimeEvent(2, "2026-06-09T10:00:04.000Z", {
        code: "preview_runtime_reload_requested",
        message: "Reload requested.",
        preview_runtime: {
          kind: "p5",
          runtime_id: "runtime-1",
          state: "reloading"
        }
      }),
      createRuntimeEvent(3, "2026-06-09T10:00:05.000Z", {
        code: "preview_runtime_running",
        message: "Renderer restarted.",
        preview_runtime: {
          kind: "p5",
          runtime_id: "runtime-2",
          state: "running"
        }
      })
    ];
    const model = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: createLiveRuntime(fixture, {
        updatedAt: "2026-06-09T10:00:15.000Z"
      }),
      traceEvents
    });

    expect(model.health).toMatchObject({
      signal: "healthy",
      label: "Healthy"
    });
    expect(readMetric(model, "fps")).toBe("60 fps");
    expect(readMetric(model, "frameTime")).toBe("16.7 ms");
    expect(readMetric(model, "uptime")).toBe("10s");
    expect(readMetric(model, "reloadCount")).toBe("1");
    expect(readMetric(model, "executionDuration")).toBe("15s");
    expect(model.events.map((event) => event.kind)).toEqual([
      "start",
      "reload",
      "start"
    ]);
    expect(model.reloadHistory).toHaveLength(1);
  });

  it("maps warming and stressed renderers to a degraded signal with warnings", () => {
    const fixture = createRuntimeFixture();
    const liveRuntime = createLiveRuntime(fixture, {
      metrics: {
        diagnostics: ["Frame delivery is above the target budget."],
        health: "stressed"
      },
      status: {
        detail: "Rendering with unstable frame pacing.",
        diagnostics: ["Frame delivery is above the target budget."],
        error: null,
        label: "p5 runtime running",
        state: "running"
      }
    });
    const model = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime,
      traceEvents: []
    });

    expect(model.health).toMatchObject({
      signal: "degraded",
      label: "Degraded"
    });
    expect(model.health.explanation).toContain("Frame delivery");
    expect(model.warnings).toEqual([
      "Frame delivery is above the target budget."
    ]);
    expect(model.errors).toEqual([]);
  });

  it("reports controlled audio ready and stopped states as healthy", () => {
    const fixture = createRuntimeFixture();
    const readyModel = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: createLiveRuntime(fixture, {
        status: {
          detail: "Audio is armed and silent until started.",
          error: null,
          label: "Tone.js runtime ready",
          state: "ready"
        }
      }),
      traceEvents: []
    });
    const stoppedModel = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: createLiveRuntime(fixture, {
        status: {
          detail: "Audio transport stopped.",
          error: null,
          label: "Tone.js runtime stopped",
          state: "stopped"
        }
      }),
      traceEvents: [
        createRuntimeEvent(1, "2026-06-09T10:00:00.000Z", {
          code: "preview_runtime_stopped",
          message: "Audio transport stopped.",
          preview_runtime: {
            kind: "tone",
            runtime_id: "runtime-1",
            state: "stopped"
          }
        })
      ]
    });

    expect(readyModel.health).toMatchObject({
      signal: "healthy",
      label: "Ready"
    });
    expect(stoppedModel.health).toMatchObject({
      signal: "healthy",
      label: "Stopped safely"
    });
    expect(stoppedModel.events).toEqual([
      expect.objectContaining({
        kind: "stop",
        stateLabel: "Stopped"
      })
    ]);
  });

  it("maps renderer errors to a failed signal and preserves error history", () => {
    const fixture = createRuntimeFixture();
    const errorMessage = "WebGL context creation failed.";
    const model = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: createLiveRuntime(fixture, {
        metrics: {
          diagnostics: ["The browser rejected the WebGL context."],
          errorMessage,
          health: "failed",
          runtimeState: "error"
        },
        status: createRuntimeErrorStatus(errorMessage)
      }),
      traceEvents: [
        createRuntimeEvent(1, "2026-06-09T10:00:00.000Z", {
          code: "preview_runtime_error",
          message: errorMessage,
          preview_runtime: {
            error: errorMessage,
            kind: "p5",
            runtime_id: "runtime-1",
            state: "error"
          }
        })
      ]
    });

    expect(model.health).toMatchObject({
      signal: "failed",
      label: "Failed"
    });
    expect(model.errors).toContain(errorMessage);
    expect(model.events).toEqual([
      expect.objectContaining({
        kind: "error",
        label: "Error"
      })
    ]);
  });

  it("supports legacy code-only runtime traces without nested runtime metadata", () => {
    const fixture = createRuntimeFixture();
    const traceEvents = [
      createRuntimeEvent(1, "2026-06-09T10:00:00.000Z", {
        code: "preview_runtime_started",
        message: "Legacy renderer started."
      }),
      createRuntimeEvent(2, "2026-06-09T10:00:03.000Z", {
        code: "preview_runtime_reload_requested",
        message: "Legacy reload requested."
      }),
      createRuntimeEvent(3, "2026-06-09T10:00:05.000Z", {
        code: "preview_runtime_failed",
        error_message: "Legacy renderer failed.",
        message: "Legacy renderer failed."
      })
    ];
    const model = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: null,
      traceEvents
    });

    expect(model.hasRuntimeActivity).toBe(true);
    expect(model.health.signal).toBe("failed");
    expect(model.latestError).toBe("Legacy renderer failed.");
    expect(model.events.map((event) => event.kind)).toEqual([
      "start",
      "reload",
      "error"
    ]);
    expect(readMetric(model, "reloadCount")).toBe("1");
    expect(readMetric(model, "executionDuration")).toBe("5s");
    expect(model.context.artifactName).toBe(fixture.route.selectedArtifactName);
  });

  it("canonicalizes start, warning, reload, error, and stop events chronologically", () => {
    const fixture = createRuntimeFixture();
    const traceEvents = [
      createRuntimeEvent(1, "2026-06-09T10:00:00.000Z", {
        code: "preview_runtime_started",
        message: "Renderer started."
      }),
      createRuntimeEvent(2, "2026-06-09T10:00:01.000Z", {
        code: "preview_runtime_warning",
        message: "Frame budget exceeded."
      }),
      createRuntimeEvent(3, "2026-06-09T10:00:02.000Z", {
        code: "preview_runtime_reload_requested",
        message: "Reload requested."
      }),
      createRuntimeEvent(4, "2026-06-09T10:00:03.000Z", {
        code: "preview_runtime_error",
        error_message: "Renderer failed.",
        message: "Renderer failed."
      }),
      createRuntimeEvent(5, "2026-06-09T10:00:04.000Z", {
        code: "preview_runtime_stopped",
        message: "Renderer stopped."
      })
    ];
    const model = buildRuntimeConsoleModel({
      ...fixture,
      liveRuntime: null,
      traceEvents
    });

    expect(model.events.map((event) => event.kind)).toEqual([
      "start",
      "warning",
      "reload",
      "error",
      "stop"
    ]);
    expect(model.events.map((event) => event.at)).toEqual(
      traceEvents.map((event) => event.receivedAt)
    );
    expect(model.warnings).toContain("Frame budget exceeded.");
  });
});

function createRuntimeFixture() {
  const snapshot = getLocalWorkspaceSnapshot();
  const artifactTitle = "diagnostic-sketch.p5.ts";
  const artifacts = snapshot.artifacts.map((artifact, index) =>
    index === 0
      ? {
          ...artifact,
          summary: "p5.js diagnostic sketch with setup and draw.",
          title: artifactTitle
        }
      : artifact
  );
  const preview = {
    ...snapshot.preview,
    active: true,
    artifactName: artifactTitle,
    available: true,
    sourceArtifactName: artifactTitle,
    target: "Browser preview"
  };
  const route = buildPreviewRendererRoute({
    artifacts,
    preview,
    previewArtifactId: artifacts[0]?.id ?? ""
  });
  const runtimeSource = buildPreviewRuntimeSource({
    code: {
      ...snapshot.code,
      excerpt: [
        "function setup() { createCanvas(640, 360); }",
        "function draw() { background(8); }"
      ],
      language: "TypeScript + p5.js",
      title: artifactTitle
    },
    route
  });

  return {
    preview,
    route,
    runtimeSource
  };
}

function createLiveRuntime(
  fixture: ReturnType<typeof createRuntimeFixture>,
  overrides: {
    metrics?: Partial<RuntimeConsoleLiveSnapshot["metrics"]>;
    status?: PreviewRuntimeStatus;
    updatedAt?: string;
  } = {}
): RuntimeConsoleLiveSnapshot {
  const status: PreviewRuntimeStatus = overrides.status ?? {
    detail: "Rendering the diagnostic sketch.",
    error: null,
    label: "p5 runtime running",
    state: "running"
  };

  return {
    kind: "p5",
    metrics: {
      diagnostics: [],
      errorMessage: null,
      fps: 60,
      frameCount: 180,
      frameTimeMs: 16.7,
      health: "nominal",
      lastFrameAtMs: 3000,
      metricsAvailable: true,
      runtimeState: status.state,
      ...overrides.metrics
    },
    route: fixture.route,
    runtimeId: "runtime-1",
    source: fixture.runtimeSource,
    status,
    updatedAt: overrides.updatedAt ?? "2026-06-09T10:00:10.000Z"
  };
}

function createRuntimeErrorStatus(message: string): PreviewRuntimeStatus {
  return {
    detail: message,
    diagnostics: ["The renderer could not continue."],
    error: {
      category: "renderer",
      debugMessage: message,
      id: "renderer:test",
      recoverable: true,
      resetLabel: "Reset preview session",
      retryLabel: "Reload preview state",
      subsystem: "preview_runtime",
      suggestedAction: "Reload the preview runtime.",
      type: "runtime_failed",
      userMessage: message
    },
    label: "p5 runtime failed",
    state: "error"
  };
}

function createRuntimeEvent(
  sequence: number,
  at: string,
  payload: Record<string, unknown>
): WorkflowRuntimeTraceEvent {
  return {
    event: {
      event_type: "status",
      payload: {
        category: "preview_runtime",
        ...payload
      },
      sequence
    },
    receivedAt: at,
    receivedAtMs: Date.parse(at)
  };
}

function readMetric(
  model: ReturnType<typeof buildRuntimeConsoleModel>,
  id: ReturnType<typeof buildRuntimeConsoleModel>["metrics"][number]["id"]
) {
  return model.metrics.find((metric) => metric.id === id)?.value;
}
