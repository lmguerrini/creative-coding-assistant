import type { PreviewSummary } from "./assistant-client";
import type {
  PreviewExecutableRuntimeKind,
  PreviewRuntimeLifecycleState,
  PreviewRuntimeSource,
  PreviewRuntimeStatus
} from "./preview-runtime-adapters";
import type {
  PreviewRuntimeHealth,
  PreviewRuntimeMetricsSnapshot
} from "./preview-runtime-diagnostics";
import type { PreviewRendererRoute } from "./preview-renderers";
import type { WorkflowRuntimeTraceEvent } from "./workflow-runtime";

export type RuntimeConsoleTone =
  | "neutral"
  | "active"
  | "success"
  | "warning"
  | "danger";

export type RuntimeConsoleLiveSnapshot = {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeId: string;
  source: PreviewRuntimeSource;
  status: PreviewRuntimeStatus;
  metrics: PreviewRuntimeMetricsSnapshot;
  updatedAt: string;
};

export type RuntimeConsoleEvent = {
  id: string;
  label: string;
  detail: string;
  at: string;
  atLabel: string;
  runtimeId: string | null;
  tone: RuntimeConsoleTone;
  stateLabel: string;
  artifactName: string | null;
  runtimeTypeLabel: string | null;
};

export type RuntimeConsoleMetric = {
  id: "status" | "fps" | "frameTime" | "frames" | "health" | "lastFrame";
  label: string;
  value: string;
  tone: RuntimeConsoleTone;
};

export type RuntimeConsoleModel = {
  emptyTitle: string | null;
  emptyDetail: string | null;
  hasRuntimeActivity: boolean;
  summary: string;
  badge: string | null;
  hero: {
    eyebrow: string;
    title: string;
    detail: string;
    tone: RuntimeConsoleTone;
    sessionLabel: string;
  };
  metrics: RuntimeConsoleMetric[];
  diagnostics: readonly string[];
  latestError: string | null;
  events: RuntimeConsoleEvent[];
  context: {
    artifactName: string;
    sourceName: string;
    fingerprint: string;
    lineCountLabel: string;
    rendererLabel: string;
    runtimeTypeLabel: string;
    supportLabel: string;
    targetLabel: string;
  };
};

export function buildRuntimeConsoleModel({
  liveRuntime,
  preview,
  route,
  runtimeSource,
  traceEvents
}: {
  liveRuntime: RuntimeConsoleLiveSnapshot | null;
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  runtimeSource: PreviewRuntimeSource;
  traceEvents: WorkflowRuntimeTraceEvent[];
}): RuntimeConsoleModel {
  const events = buildRuntimeConsoleEvents(traceEvents);
  const latestEvent = events[0] ?? null;
  const latestTraceState = extractLatestRuntimeState(traceEvents);
  const latestTraceError = extractLatestRuntimeError(traceEvents);
  const diagnostics =
    liveRuntime?.metrics.diagnostics.length
      ? liveRuntime.metrics.diagnostics
      : liveRuntime?.status.diagnostics?.slice(0, 3) ??
        extractLatestRuntimeDiagnostics(traceEvents);
  const latestError =
    liveRuntime != null
      ? liveRuntime.status.error?.userMessage ?? liveRuntime.metrics.errorMessage
      : latestTraceState === "error"
        ? latestTraceError
        : null;
  const hasRuntimeActivity = liveRuntime !== null || events.length > 0;
  const runtimeTypeLabel =
    liveRuntime != null
      ? formatRuntimeKindLabel(liveRuntime.kind)
      : route.rendererLabel === "No preview surface" || route.rendererLabel === "Pending target"
        ? "Pending runtime"
        : route.rendererLabel;
  const sessionLabel =
    liveRuntime?.runtimeId ??
    latestEvent?.runtimeId ??
    "Awaiting runtime";

  if (!hasRuntimeActivity) {
    return {
      emptyTitle: !preview.available
        ? "No runtime activity yet"
        : route.supportState === "supported"
          ? "Runtime console armed"
          : "No live runtime for this route",
      emptyDetail: !preview.available
        ? "Generate or select a preview-capable artifact to start a live runtime."
        : route.supportState === "supported"
          ? "Open the preview shelf to start the live renderer. Status, FPS, reloads, and runtime issues will appear here."
          : route.supportReason,
      hasRuntimeActivity,
      summary: !preview.available
        ? "Appears after a live preview runtime starts"
        : route.supportState === "supported"
          ? "Ready to collect runtime diagnostics when the preview opens"
          : "No live runtime is available for the current preview route",
      badge: null,
      hero: {
        eyebrow: "Runtime console",
        title: !preview.available
          ? "Idle"
          : route.supportState === "supported"
            ? "Ready to start"
            : "Unavailable",
        detail: !preview.available
          ? "The inspector stays quiet until a live preview runtime exists."
          : route.supportState === "supported"
            ? "Preview execution has not started yet."
            : route.supportReason,
        tone: !preview.available
          ? "neutral"
          : route.supportState === "supported"
            ? "warning"
            : "neutral",
        sessionLabel
      },
      metrics: buildRuntimeConsoleMetrics({
        liveRuntime,
        latestEventState: null,
        latestHealth: null
      }),
      diagnostics,
      latestError,
      events,
      context: {
        artifactName: route.selectedArtifactName,
        sourceName: route.sourceArtifactName || runtimeSource.title,
        fingerprint: runtimeSource.fingerprint,
        lineCountLabel: formatLineCount(runtimeSource.lineCount),
        rendererLabel: route.rendererLabel,
        runtimeTypeLabel,
        supportLabel: route.supportLabel,
        targetLabel: route.targetLabel
      }
    };
  }

  const runtimeState =
    liveRuntime?.status.state ?? latestTraceState ?? "idle";
  const health = liveRuntime?.metrics.health ?? deriveHealthFromState(runtimeState, latestError);
  const title =
    latestError != null
      ? "Runtime issue detected"
      : runtimeState === "running"
        ? "Live runtime"
        : runtimeState === "starting"
          ? "Runtime starting"
          : runtimeState === "error"
            ? "Runtime stopped"
            : latestEvent?.label ?? "Runtime activity";
  const detail =
    liveRuntime?.status.detail ??
    latestEvent?.detail ??
    "Runtime lifecycle events are flowing from the active preview renderer.";

  return {
    emptyTitle: null,
    emptyDetail: null,
    hasRuntimeActivity,
    summary:
      latestError != null
        ? "Runtime issues need attention"
        : liveRuntime?.status.state === "running"
          ? "Live runtime status, FPS, and lifecycle events"
          : "Runtime lifecycle history for the active preview",
    badge: buildRuntimeConsoleBadge(liveRuntime, latestError),
    hero: {
      eyebrow: "Runtime console",
      title,
      detail,
      tone: toneForHealth(health),
      sessionLabel
    },
    metrics: buildRuntimeConsoleMetrics({
      liveRuntime,
      latestEventState: runtimeState,
      latestHealth: health
    }),
    diagnostics,
    latestError,
    events,
    context: {
      artifactName: liveRuntime?.route.selectedArtifactName ?? route.selectedArtifactName,
      sourceName: liveRuntime?.source.title ?? runtimeSource.title,
      fingerprint: liveRuntime?.source.fingerprint ?? runtimeSource.fingerprint,
      lineCountLabel: formatLineCount(liveRuntime?.source.lineCount ?? runtimeSource.lineCount),
      rendererLabel: liveRuntime?.route.rendererLabel ?? route.rendererLabel,
      runtimeTypeLabel:
        liveRuntime != null
          ? formatRuntimeKindLabel(liveRuntime.kind)
          : runtimeTypeLabel,
      supportLabel: liveRuntime != null ? "Live runtime" : route.supportLabel,
      targetLabel: liveRuntime?.route.targetLabel ?? route.targetLabel
    }
  };
}

function buildRuntimeConsoleEvents(
  traceEvents: WorkflowRuntimeTraceEvent[]
): RuntimeConsoleEvent[] {
  return traceEvents
    .filter((traceEvent) => isPreviewRuntimeTraceEvent(traceEvent))
    .slice(-8)
    .reverse()
    .map((traceEvent) => {
      const payload = traceEvent.event.payload;
      const previewRuntime = readRecord(payload.preview_runtime);
      const code = readText(payload.code) ?? traceEvent.event.event_type;
      const runtimeState = readText(previewRuntime?.state);
      const normalizedRuntimeState = runtimeState
        ? normalizeRuntimeState(runtimeState)
        : inferRuntimeStateFromCode(code);
      const errorMessage =
        readText(previewRuntime?.error) ?? readText(payload.error_message);
      const detail =
        readText(payload.message) ??
        errorMessage ??
        "Preview runtime event recorded.";

      return {
        id: `${traceEvent.event.sequence}-${code}`,
        label: humanizeRuntimeCode(code),
        detail,
        at: traceEvent.receivedAt,
        atLabel: formatRuntimeConsoleTime(traceEvent.receivedAt),
        runtimeId: readText(previewRuntime?.runtime_id) ?? null,
        tone: toneForRuntimeEvent(code, runtimeState, errorMessage),
        stateLabel: formatRuntimeStateLabel(normalizedRuntimeState),
        artifactName: readText(previewRuntime?.artifact) ?? null,
        runtimeTypeLabel: formatRuntimeKindLabelFromValue(readText(previewRuntime?.kind))
      };
    });
}

function buildRuntimeConsoleMetrics({
  liveRuntime,
  latestEventState,
  latestHealth
}: {
  liveRuntime: RuntimeConsoleLiveSnapshot | null;
  latestEventState: PreviewRuntimeLifecycleState | "reloading" | null;
  latestHealth: PreviewRuntimeHealth | null;
}): RuntimeConsoleMetric[] {
  const fps = liveRuntime?.metrics.fps ?? null;
  const frameTimeMs = liveRuntime?.metrics.frameTimeMs ?? null;
  const frameCount = liveRuntime?.metrics.frameCount ?? null;
  const lastFrameAtMs = liveRuntime?.metrics.lastFrameAtMs ?? null;
  const runtimeState = liveRuntime?.status.state ?? latestEventState;
  const health = liveRuntime?.metrics.health ?? latestHealth;

  return [
    {
      id: "status",
      label: "Status",
      value: formatRuntimeStateLabel(runtimeState ?? "idle"),
      tone:
        runtimeState === "error"
          ? "danger"
          : runtimeState === "running"
            ? "active"
            : runtimeState === "starting" || runtimeState === "reloading"
              ? "warning"
              : "neutral"
    },
    {
      id: "fps",
      label: "FPS",
      value: fps != null ? `${Math.round(fps)} fps` : "Waiting",
      tone: health ? toneForHealth(health) : "neutral"
    },
    {
      id: "frameTime",
      label: "Frame time",
      value: frameTimeMs != null ? `${frameTimeMs.toFixed(1)} ms` : "Waiting",
      tone: health ? toneForHealth(health) : "neutral"
    },
    {
      id: "frames",
      label: "Frames",
      value: frameCount != null ? formatCompactNumber(frameCount) : "0",
      tone: frameCount && frameCount > 0 ? "success" : "neutral"
    },
    {
      id: "health",
      label: "Health",
      value: health ? formatHealthLabel(health) : "Pending",
      tone: health ? toneForHealth(health) : "neutral"
    },
    {
      id: "lastFrame",
      label: "Last frame",
      value: lastFrameAtMs != null ? `${Math.round(lastFrameAtMs)} ms` : "Awaiting frame",
      tone: lastFrameAtMs != null ? "active" : "neutral"
    }
  ];
}

function buildRuntimeConsoleBadge(
  liveRuntime: RuntimeConsoleLiveSnapshot | null,
  latestError: string | null
) {
  if (latestError) {
    return "Error";
  }

  if (liveRuntime?.metrics.fps != null) {
    return `${Math.round(liveRuntime.metrics.fps)}fps`;
  }

  if (liveRuntime?.status.state === "running") {
    return "Live";
  }

  if (liveRuntime?.status.state === "starting") {
    return "Warm";
  }

  return "Trace";
}

function isPreviewRuntimeTraceEvent(traceEvent: WorkflowRuntimeTraceEvent) {
  return (
    readText(traceEvent.event.payload.category) === "preview_runtime" ||
    (readText(traceEvent.event.payload.code) ?? "").startsWith("preview_runtime_")
  );
}

function extractLatestRuntimeState(
  traceEvents: WorkflowRuntimeTraceEvent[]
): PreviewRuntimeLifecycleState | "reloading" | null {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const traceEvent = traceEvents[index];

    if (!traceEvent || !isPreviewRuntimeTraceEvent(traceEvent)) {
      continue;
    }

    const previewRuntime = readRecord(traceEvent.event.payload.preview_runtime);
    const runtimeState = readText(previewRuntime?.state);
    const code = readText(traceEvent.event.payload.code) ?? traceEvent.event.event_type;

    return runtimeState
      ? normalizeRuntimeState(runtimeState)
      : inferRuntimeStateFromCode(code);
  }

  return null;
}

function extractLatestRuntimeDiagnostics(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const previewRuntime = readRecord(traceEvents[index]?.event.payload.preview_runtime);
    const diagnostics = Array.isArray(previewRuntime?.diagnostics)
      ? previewRuntime.diagnostics.filter((value): value is string => typeof value === "string")
      : [];

    if (diagnostics.length > 0) {
      return diagnostics.slice(0, 3);
    }
  }

  return [];
}

function extractLatestRuntimeError(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const previewRuntime = readRecord(traceEvents[index]?.event.payload.preview_runtime);
    const errorMessage = readText(previewRuntime?.error);

    if (errorMessage) {
      return errorMessage;
    }
  }

  return null;
}

function deriveHealthFromState(
  state: PreviewRuntimeLifecycleState | "reloading",
  latestError: string | null
): PreviewRuntimeHealth {
  if (latestError || state === "error") {
    return "failed";
  }

  if (state === "running") {
    return "nominal";
  }

  if (state === "starting" || state === "reloading") {
    return "warming";
  }

  return "unavailable";
}

function toneForRuntimeEvent(
  code: string,
  runtimeState: string | undefined,
  errorMessage: string | undefined
): RuntimeConsoleTone {
  if (errorMessage || runtimeState === "error" || code.includes("error")) {
    return "danger";
  }

  if (code.includes("recovered")) {
    return "success";
  }

  if (code.includes("reload") || runtimeState === "starting") {
    return "warning";
  }

  if (runtimeState === "running" || code.includes("frame")) {
    return "active";
  }

  return "neutral";
}

function toneForHealth(health: PreviewRuntimeHealth): RuntimeConsoleTone {
  switch (health) {
    case "nominal":
      return "success";
    case "warming":
    case "stressed":
      return "warning";
    case "degraded":
    case "failed":
      return "danger";
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
    default:
      return "Unavailable";
  }
}

function formatRuntimeStateLabel(
  state: PreviewRuntimeLifecycleState | "reloading"
) {
  switch (state) {
    case "idle":
      return "Idle";
    case "starting":
      return "Starting";
    case "running":
      return "Running";
    case "error":
      return "Error";
    case "reloading":
      return "Reloading";
    default:
      return "Idle";
  }
}

function inferRuntimeStateFromCode(
  code: string
): PreviewRuntimeLifecycleState | "reloading" {
  if (code.includes("reload")) {
    return "reloading";
  }

  if (code.includes("error")) {
    return "error";
  }

  if (code.includes("running") || code.includes("frame") || code.includes("recovered")) {
    return "running";
  }

  if (code.includes("starting")) {
    return "starting";
  }

  return "idle";
}

function normalizeRuntimeState(
  state: string
): PreviewRuntimeLifecycleState | "reloading" {
  switch (state) {
    case "idle":
    case "starting":
    case "running":
    case "error":
    case "reloading":
      return state;
    default:
      return "idle";
  }
}

function formatRuntimeKindLabel(kind: PreviewExecutableRuntimeKind) {
  switch (kind) {
    case "p5":
      return "p5.js";
    case "three":
      return "Three.js";
    case "glsl":
      return "GLSL";
    default:
      return kind;
  }
}

function formatRuntimeKindLabelFromValue(kind: string | undefined) {
  switch (kind) {
    case "p5":
      return "p5.js";
    case "three":
      return "Three.js";
    case "glsl":
      return "GLSL";
    default:
      return kind ?? null;
  }
}

function humanizeRuntimeCode(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatRuntimeConsoleTime(timestamp: string) {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "Pending";
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function formatLineCount(lineCount: number) {
  return `${formatCompactNumber(lineCount)} lines`;
}

function formatCompactNumber(value: number) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
    notation: value >= 1000 ? "compact" : "standard"
  }).format(value);
}

function readText(value: unknown) {
  return typeof value === "string" ? value : undefined;
}

function readRecord(value: unknown) {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}
