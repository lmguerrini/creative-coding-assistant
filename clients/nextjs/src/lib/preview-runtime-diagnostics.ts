import type {
  PreviewExecutableRuntimeKind,
  PreviewRuntimeFrameSample,
  PreviewRuntimeLifecycleState,
  PreviewRuntimeSource,
  PreviewRuntimeStatus
} from "./preview-runtime-adapters";
import type { PreviewRendererRoute } from "./preview-renderers";

export type PreviewRuntimeHealth =
  | "nominal"
  | "warming"
  | "stressed"
  | "degraded"
  | "failed"
  | "unavailable";

export type PreviewRuntimeMetricsSnapshot = {
  runtimeState: PreviewRuntimeLifecycleState;
  health: PreviewRuntimeHealth;
  fps: number | null;
  frameTimeMs: number | null;
  frameCount: number;
  metricsAvailable: boolean;
  lastFrameAtMs: number | null;
  diagnostics: readonly string[];
  errorMessage: string | null;
};

export type PreviewRuntimeOverlayMetricTone =
  | "neutral"
  | "good"
  | "warning"
  | "danger";

export type PreviewRuntimeOverlayMetric = {
  id: "fps" | "frame" | "health" | "state";
  label: string;
  value: string;
  tone: PreviewRuntimeOverlayMetricTone;
};

export type PreviewRuntimeOverlayModel = {
  healthLabel: string;
  healthTone: PreviewRuntimeOverlayMetricTone;
  metrics: readonly PreviewRuntimeOverlayMetric[];
  diagnostics: readonly string[];
};

export type PreviewRuntimeMetricsTracker = {
  publishStatus: (status: PreviewRuntimeStatus) => PreviewRuntimeMetricsSnapshot;
  recordFrame: (
    sample: PreviewRuntimeFrameSample
  ) => PreviewRuntimeMetricsSnapshot | null;
  reset: (status: PreviewRuntimeStatus) => PreviewRuntimeMetricsSnapshot;
  snapshot: () => PreviewRuntimeMetricsSnapshot;
};

type PreviewRuntimeMetricsState = {
  status: PreviewRuntimeStatus;
  frameTimes: number[];
  frameCount: number;
  lastFrameAtMs: number | null;
  lastPublishedAtMs: number | null;
};

type CreatePreviewRuntimeMetricsTrackerOptions = {
  clock?: () => number;
  publishIntervalMs?: number;
  sampleWindowSize?: number;
};

const defaultSampleWindowSize = 18;
const defaultPublishIntervalMs = 220;

export function createPreviewRuntimeMetricsTracker(
  initialStatus: PreviewRuntimeStatus,
  options: CreatePreviewRuntimeMetricsTrackerOptions = {}
): PreviewRuntimeMetricsTracker {
  const sampleWindowSize = options.sampleWindowSize ?? defaultSampleWindowSize;
  const publishIntervalMs = options.publishIntervalMs ?? defaultPublishIntervalMs;
  const clock = options.clock ?? (() => Date.now());
  let state = createMetricsState(initialStatus);

  return {
    publishStatus(nextStatus) {
      state = {
        ...state,
        status: nextStatus
      };
      return buildMetricsSnapshot(state);
    },
    recordFrame(sample) {
      const renderedAtMs = sample.renderedAtMs;
      const nextFrameTimes =
        state.lastFrameAtMs != null && renderedAtMs > state.lastFrameAtMs
          ? appendFrameTime(
              state.frameTimes,
              renderedAtMs - state.lastFrameAtMs,
              sampleWindowSize
            )
          : state.frameTimes;
      const nextState: PreviewRuntimeMetricsState = {
        ...state,
        frameCount: state.frameCount + 1,
        frameTimes: nextFrameTimes,
        lastFrameAtMs: renderedAtMs
      };
      state = nextState;
      const nowMs = Number.isFinite(renderedAtMs) ? renderedAtMs : clock();

      if (
        state.lastPublishedAtMs != null &&
        nowMs - state.lastPublishedAtMs < publishIntervalMs
      ) {
        return null;
      }

      state = {
        ...state,
        lastPublishedAtMs: nowMs
      };
      return buildMetricsSnapshot(state);
    },
    reset(nextStatus) {
      state = createMetricsState(nextStatus);
      return buildMetricsSnapshot(state);
    },
    snapshot() {
      return buildMetricsSnapshot(state);
    }
  };
}

export function buildPreviewRuntimeOverlayModel({
  kind,
  route,
  runtimeSource,
  snapshot,
  status
}: {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeSource: PreviewRuntimeSource;
  snapshot: PreviewRuntimeMetricsSnapshot;
  status: PreviewRuntimeStatus;
}): PreviewRuntimeOverlayModel {
  const healthLabel = formatHealthLabel(snapshot.health);
  const healthTone = toneForHealth(snapshot.health);
  const diagnostics = buildOverlayDiagnostics({
    kind,
    route,
    runtimeSource,
    snapshot,
    status
  });

  return {
    healthLabel,
    healthTone,
    diagnostics,
    metrics: [
      {
        id: "fps",
        label: "FPS",
        value:
          snapshot.fps != null
            ? formatFps(snapshot.fps)
            : snapshot.runtimeState === "running"
              ? "Sampling"
              : "N/A",
        tone:
          snapshot.runtimeState === "running" && snapshot.fps == null
            ? "warning"
            : toneForHealth(snapshot.health)
      },
      {
        id: "frame",
        label: "Frame",
        value:
          snapshot.frameTimeMs != null
            ? `${snapshot.frameTimeMs.toFixed(1)} ms`
            : snapshot.runtimeState === "running"
              ? "Sampling"
              : "N/A",
        tone: toneForHealth(snapshot.health)
      },
      {
        id: "health",
        label: "Health",
        value: healthLabel,
        tone: healthTone
      },
      {
        id: "state",
        label: "State",
        value: formatRuntimeStateLabel(snapshot.runtimeState),
        tone: snapshot.runtimeState === "error" ? "danger" : "neutral"
      }
    ]
  };
}

function buildOverlayDiagnostics({
  kind,
  route,
  runtimeSource,
  snapshot,
  status
}: {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeSource: PreviewRuntimeSource;
  snapshot: PreviewRuntimeMetricsSnapshot;
  status: PreviewRuntimeStatus;
}) {
  const explicitDiagnostics =
    snapshot.diagnostics.length > 0
      ? snapshot.diagnostics
      : status.diagnostics ?? [];

  if (explicitDiagnostics.length > 0) {
    return explicitDiagnostics.slice(0, 2).map(compactDiagnostic);
  }

  if (snapshot.runtimeState === "starting") {
    return ["Collecting first frame samples."];
  }

  if (snapshot.runtimeState === "ready") {
    return ["Audio is armed and waiting for explicit playback."];
  }

  if (snapshot.runtimeState === "stopped") {
    return kind === "tone"
      ? ["Audio transport is stopped and output is silent."]
      : ["All scheduled motion completed and the stage is idle."];
  }

  if (snapshot.runtimeState === "running" && !snapshot.metricsAvailable) {
    return ["Waiting for the first completed frame."];
  }

  if (snapshot.runtimeState === "idle") {
    return ["Metrics become available when a live renderer starts."];
  }

  if (snapshot.metricsAvailable) {
    return [
      `${formatRuntimeKindLabel(kind)} / ${route.rendererLabel}`,
      `${runtimeSource.lineCount} lines / ${runtimeSource.fingerprint}`
    ];
  }

  return [`${formatRuntimeKindLabel(kind)} / ${route.rendererLabel}`];
}

function createMetricsState(status: PreviewRuntimeStatus): PreviewRuntimeMetricsState {
  return {
    status,
    frameTimes: [],
    frameCount: 0,
    lastFrameAtMs: null,
    lastPublishedAtMs: null
  };
}

function buildMetricsSnapshot(
  state: PreviewRuntimeMetricsState
): PreviewRuntimeMetricsSnapshot {
  const frameTimeMs = average(state.frameTimes);
  const fps = frameTimeMs != null && frameTimeMs > 0 ? 1000 / frameTimeMs : null;
  const diagnostics = (
    state.status.diagnostics ??
    (state.status.error?.debugMessage ? [state.status.error.debugMessage] : [])
  ).map(compactDiagnostic);

  return {
    runtimeState: state.status.state,
    health: deriveRuntimeHealth({
      error: state.status.error,
      fps,
      frameTimeMs,
      metricsAvailable: state.frameTimes.length > 0,
      runtimeState: state.status.state
    }),
    fps,
    frameTimeMs,
    frameCount: state.frameCount,
    metricsAvailable: state.frameTimes.length > 0,
    lastFrameAtMs: state.lastFrameAtMs,
    diagnostics,
    errorMessage: state.status.error?.userMessage ?? null
  };
}

function appendFrameTime(
  frameTimes: number[],
  nextFrameTimeMs: number,
  sampleWindowSize: number
) {
  const clampedFrameTimeMs = Math.max(0.1, nextFrameTimeMs);
  const nextFrameTimes = [...frameTimes, clampedFrameTimeMs];

  if (nextFrameTimes.length <= sampleWindowSize) {
    return nextFrameTimes;
  }

  return nextFrameTimes.slice(nextFrameTimes.length - sampleWindowSize);
}

function deriveRuntimeHealth({
  error,
  fps,
  frameTimeMs,
  metricsAvailable,
  runtimeState
}: {
  error: PreviewRuntimeStatus["error"];
  fps: number | null;
  frameTimeMs: number | null;
  metricsAvailable: boolean;
  runtimeState: PreviewRuntimeLifecycleState;
}): PreviewRuntimeHealth {
  if (runtimeState === "error" || error) {
    return "failed";
  }

  if (runtimeState !== "running") {
    return runtimeState === "starting" || runtimeState === "ready"
      ? "warming"
      : "unavailable";
  }

  if (!metricsAvailable || fps == null || frameTimeMs == null) {
    return "warming";
  }

  if (frameTimeMs >= 34 || fps < 28) {
    return "degraded";
  }

  if (frameTimeMs >= 22 || fps < 45) {
    return "stressed";
  }

  return "nominal";
}

function average(values: number[]) {
  if (values.length === 0) {
    return null;
  }

  return values.reduce((total, value) => total + value, 0) / values.length;
}

function formatFps(value: number) {
  return `${Math.round(value)} fps`;
}

function compactDiagnostic(value: string) {
  const normalized = value.replace(/\s+/g, " ").trim();

  if (normalized.length <= 120) {
    return normalized;
  }

  return `${normalized.slice(0, 117).trimEnd()}...`;
}

function toneForHealth(
  health: PreviewRuntimeHealth
): PreviewRuntimeOverlayMetricTone {
  switch (health) {
    case "nominal":
      return "good";
    case "warming":
      return "warning";
    case "stressed":
      return "warning";
    case "degraded":
      return "danger";
    case "failed":
      return "danger";
    case "unavailable":
    default:
      return "neutral";
  }
}

function formatHealthLabel(health: PreviewRuntimeHealth) {
  switch (health) {
    case "nominal":
      return "Nominal";
    case "warming":
      return "Warming";
    case "stressed":
      return "Stressed";
    case "degraded":
      return "Degraded";
    case "failed":
      return "Failed";
    case "unavailable":
    default:
      return "Unavailable";
  }
}

function formatRuntimeStateLabel(state: PreviewRuntimeLifecycleState) {
  switch (state) {
    case "idle":
      return "Idle";
    case "starting":
      return "Starting";
    case "ready":
      return "Ready";
    case "running":
      return "Running";
    case "stopped":
      return "Stopped";
    case "error":
      return "Error";
    default:
      return "Idle";
  }
}

function formatRuntimeKindLabel(kind: PreviewExecutableRuntimeKind) {
  switch (kind) {
    case "p5":
      return "p5 runtime";
    case "three":
      return "Three runtime";
    case "glsl":
      return "GLSL runtime";
    case "hydra":
      return "Hydra runtime";
    case "gsap":
      return "GSAP runtime";
    case "svg":
      return "SVG runtime";
    case "canvas":
      return "Canvas runtime";
    case "tone":
      return "Tone.js runtime";
    default:
      return kind;
  }
}
