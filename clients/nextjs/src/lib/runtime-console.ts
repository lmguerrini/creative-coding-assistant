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

export type RuntimeConsoleHealthSignal = "healthy" | "degraded" | "failed";

export type RuntimeConsoleEventKind =
  | "start"
  | "stop"
  | "reload"
  | "warning"
  | "error";

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
  kind: RuntimeConsoleEventKind;
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
  id:
    | "status"
    | "fps"
    | "frameTime"
    | "frames"
    | "health"
    | "uptime"
    | "reloadCount"
    | "executionDuration";
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
  health: {
    signal: RuntimeConsoleHealthSignal;
    label: string;
    explanation: string;
    tone: RuntimeConsoleTone;
  };
  warnings: readonly string[];
  errors: readonly string[];
  reloadHistory: RuntimeConsoleEvent[];
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

type RuntimeConsoleTiming = {
  uptimeMs: number | null;
  reloadCount: number;
  executionDurationMs: number | null;
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
  const latestEvent = events.at(-1) ?? null;
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
  const health = buildRuntimeConsoleHealth({
    diagnostics,
    health: liveRuntime?.metrics.health ?? null,
    latestError,
    runtimeState: liveRuntime?.status.state ?? latestTraceState
  });
  const warnings = buildRuntimeWarnings({
    diagnostics,
    events,
    health: liveRuntime?.metrics.health ?? null,
    latestError
  });
  const errors = uniqueDiagnosticText([
    ...events
      .filter((event) => event.kind === "error")
      .map((event) => event.detail),
    latestError
  ]);
  const reloadHistory = events.filter((event) => event.kind === "reload");
  const timing = buildRuntimeTiming({
    events,
    liveRuntime,
    runtimeState: liveRuntime?.status.state ?? latestTraceState
  });
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
        health,
        liveRuntime,
        latestEventState: null,
        timing
      }),
      diagnostics,
      latestError,
      health,
      warnings,
      errors,
      reloadHistory,
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
      tone: health.tone,
      sessionLabel
    },
    metrics: buildRuntimeConsoleMetrics({
      health,
      liveRuntime,
      latestEventState: runtimeState,
      timing
    }),
    diagnostics,
    latestError,
    health,
    warnings,
    errors,
    reloadHistory,
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
      const kind = classifyRuntimeEvent(code, runtimeState, errorMessage);

      if (!kind) {
        return null;
      }

      return {
        id: `${traceEvent.event.sequence}-${code}`,
        kind,
        label: formatRuntimeEventKindLabel(kind),
        detail,
        at: traceEvent.receivedAt,
        atLabel: formatRuntimeConsoleTime(traceEvent.receivedAt),
        runtimeId: readText(previewRuntime?.runtime_id) ?? null,
        tone: toneForRuntimeEvent(code, runtimeState, errorMessage),
        stateLabel: formatRuntimeStateLabel(normalizedRuntimeState),
        artifactName: readText(previewRuntime?.artifact) ?? null,
        runtimeTypeLabel: formatRuntimeKindLabelFromValue(readText(previewRuntime?.kind))
      };
    })
    .filter((event): event is RuntimeConsoleEvent => event !== null)
    .sort((left, right) => {
      const timestampDifference = Date.parse(left.at) - Date.parse(right.at);

      return Number.isNaN(timestampDifference) ? 0 : timestampDifference;
    });
}

function buildRuntimeConsoleMetrics({
  health,
  liveRuntime,
  latestEventState,
  timing
}: {
  health: RuntimeConsoleModel["health"];
  liveRuntime: RuntimeConsoleLiveSnapshot | null;
  latestEventState: PreviewRuntimeLifecycleState | "reloading" | null;
  timing: RuntimeConsoleTiming;
}): RuntimeConsoleMetric[] {
  const fps = liveRuntime?.metrics.fps ?? null;
  const frameTimeMs = liveRuntime?.metrics.frameTimeMs ?? null;
  const frameCount = liveRuntime?.metrics.frameCount ?? null;
  const runtimeState = liveRuntime?.status.state ?? latestEventState;

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
      tone: health.tone
    },
    {
      id: "frameTime",
      label: "Frame time",
      value: frameTimeMs != null ? `${frameTimeMs.toFixed(1)} ms` : "Waiting",
      tone: health.tone
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
      value: health.label,
      tone: health.tone
    },
    {
      id: "uptime",
      label: "Uptime",
      value: formatRuntimeDuration(timing.uptimeMs),
      tone: timing.uptimeMs != null ? "active" : "neutral"
    },
    {
      id: "reloadCount",
      label: "Reloads",
      value: String(timing.reloadCount),
      tone: timing.reloadCount > 0 ? "warning" : "neutral"
    },
    {
      id: "executionDuration",
      label: "Execution",
      value: formatRuntimeDuration(timing.executionDurationMs),
      tone: timing.executionDurationMs != null ? "active" : "neutral"
    }
  ];
}

function buildRuntimeConsoleHealth({
  diagnostics,
  health,
  latestError,
  runtimeState
}: {
  diagnostics: readonly string[];
  health: PreviewRuntimeHealth | null;
  latestError: string | null;
  runtimeState: PreviewRuntimeLifecycleState | "reloading" | null;
}): RuntimeConsoleModel["health"] {
  if (latestError || health === "failed" || runtimeState === "error") {
    return {
      signal: "failed",
      label: "Failed",
      explanation:
        latestError ??
        diagnostics[0] ??
        "The preview renderer stopped after reporting a runtime error.",
      tone: "danger"
    };
  }

  if (health === "nominal" && runtimeState === "running") {
    return {
      signal: "healthy",
      label: "Healthy",
      explanation:
        "The renderer is running and frame delivery is within the expected budget.",
      tone: "success"
    };
  }

  return {
    signal: "degraded",
    label: "Degraded",
    explanation:
      diagnostics[0] ??
      (runtimeState === "starting" || runtimeState === "reloading"
        ? "The renderer is warming up and has not established stable frame delivery yet."
        : runtimeState === "idle"
          ? "The renderer is inactive, so live health metrics are unavailable."
          : "The renderer is active, but stable frame metrics are not available yet."),
    tone: "warning"
  };
}

function buildRuntimeWarnings({
  diagnostics,
  events,
  health,
  latestError
}: {
  diagnostics: readonly string[];
  events: RuntimeConsoleEvent[];
  health: PreviewRuntimeHealth | null;
  latestError: string | null;
}) {
  const eventWarnings = events
    .filter((event) => event.kind === "warning")
    .map((event) => event.detail);
  const healthWarnings =
    latestError == null &&
    (health === "warming" || health === "stressed" || health === "degraded")
      ? diagnostics
      : [];

  return uniqueText([...eventWarnings, ...healthWarnings]);
}

function buildRuntimeTiming({
  events,
  liveRuntime,
  runtimeState
}: {
  events: RuntimeConsoleEvent[];
  liveRuntime: RuntimeConsoleLiveSnapshot | null;
  runtimeState: PreviewRuntimeLifecycleState | "reloading" | null;
}): RuntimeConsoleTiming {
  const timedEvents = events
    .map((event) => ({
      event,
      atMs: Date.parse(event.at)
    }))
    .filter((entry) => Number.isFinite(entry.atMs));
  const startEvents = timedEvents.filter((entry) => entry.event.kind === "start");
  const firstStartAtMs = startEvents[0]?.atMs ?? null;
  const latestStartAtMs = startEvents.at(-1)?.atMs ?? null;
  const latestEventAtMs = timedEvents.at(-1)?.atMs ?? null;
  const liveUpdatedAtMs = liveRuntime ? Date.parse(liveRuntime.updatedAt) : Number.NaN;
  const observedAtMs = Number.isFinite(liveUpdatedAtMs)
    ? Math.max(liveUpdatedAtMs, latestEventAtMs ?? liveUpdatedAtMs)
    : latestEventAtMs;
  const latestTerminalAtMs =
    timedEvents
      .filter(
        (entry) =>
          entry.event.kind === "stop" ||
          entry.event.kind === "reload" ||
          entry.event.kind === "error"
      )
      .at(-1)?.atMs ?? null;
  const isLive =
    runtimeState === "running" &&
    latestStartAtMs != null &&
    (latestTerminalAtMs == null || latestStartAtMs >= latestTerminalAtMs);

  return {
    uptimeMs:
      isLive && observedAtMs != null
        ? Math.max(observedAtMs - latestStartAtMs, 0)
        : null,
    reloadCount: events.filter((event) => event.kind === "reload").length,
    executionDurationMs:
      firstStartAtMs != null && observedAtMs != null
        ? Math.max(observedAtMs - firstStartAtMs, 0)
        : null
  };
}

function classifyRuntimeEvent(
  code: string,
  runtimeState: string | undefined,
  errorMessage: string | undefined
): RuntimeConsoleEventKind | null {
  const normalizedCode = code.toLowerCase();

  if (normalizedCode.includes("frame")) {
    return null;
  }

  if (
    errorMessage ||
    runtimeState === "error" ||
    normalizedCode.includes("error") ||
    normalizedCode.includes("failed")
  ) {
    return "error";
  }

  if (normalizedCode.includes("reload")) {
    return "reload";
  }

  if (normalizedCode.includes("warning") || normalizedCode.includes("warn")) {
    return "warning";
  }

  if (
    runtimeState === "idle" ||
    normalizedCode.includes("stopped") ||
    normalizedCode.endsWith("_stop") ||
    normalizedCode.endsWith("_idle")
  ) {
    return "stop";
  }

  if (
    runtimeState === "starting" ||
    runtimeState === "running" ||
    normalizedCode.includes("starting") ||
    normalizedCode.includes("started") ||
    normalizedCode.includes("running") ||
    normalizedCode.includes("recovered")
  ) {
    return "start";
  }

  return null;
}

function formatRuntimeEventKindLabel(kind: RuntimeConsoleEventKind) {
  return kind.replace(/\b\w/g, (character) => character.toUpperCase());
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
    const payload = traceEvents[index]?.event.payload;
    const previewRuntime = readRecord(payload?.preview_runtime);
    const diagnostics = Array.isArray(previewRuntime?.diagnostics)
      ? previewRuntime.diagnostics.filter((value): value is string => typeof value === "string")
      : Array.isArray(payload?.diagnostics)
        ? payload.diagnostics.filter((value): value is string => typeof value === "string")
        : [];

    if (diagnostics.length > 0) {
      return diagnostics.slice(0, 3);
    }
  }

  return [];
}

function extractLatestRuntimeError(traceEvents: WorkflowRuntimeTraceEvent[]) {
  for (let index = traceEvents.length - 1; index >= 0; index -= 1) {
    const payload = traceEvents[index]?.event.payload;
    const previewRuntime = readRecord(payload?.preview_runtime);
    const errorMessage =
      readText(previewRuntime?.error) ?? readText(payload?.error_message);

    if (errorMessage) {
      return errorMessage;
    }
  }

  return null;
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

  if (code.includes("error") || code.includes("failed")) {
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

function formatRuntimeDuration(durationMs: number | null) {
  if (durationMs == null) {
    return "Waiting";
  }

  if (durationMs < 1000) {
    return "<1s";
  }

  const totalSeconds = Math.round(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return minutes > 0 ? `${minutes}m ${seconds}s` : `${totalSeconds}s`;
}

function uniqueText(values: Array<string | null | undefined>) {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

function uniqueDiagnosticText(values: Array<string | null | undefined>) {
  return uniqueText(values).reduce<string[]>((diagnostics, value) => {
    const relatedIndex = diagnostics.findIndex(
      (diagnostic) =>
        diagnostic.toLowerCase().includes(value.toLowerCase()) ||
        value.toLowerCase().includes(diagnostic.toLowerCase())
    );

    if (relatedIndex === -1) {
      return [...diagnostics, value];
    }

    const relatedDiagnostic = diagnostics[relatedIndex];
    if (relatedDiagnostic && value.length < relatedDiagnostic.length) {
      return diagnostics.map((diagnostic, index) =>
        index === relatedIndex ? value : diagnostic
      );
    }

    return diagnostics;
  }, []);
}

function readText(value: unknown) {
  return typeof value === "string" ? value : undefined;
}

function readRecord(value: unknown) {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}
