import type { PreviewSummary } from "./assistant-client";

export type PreviewRuntimeSessionOverrideMode =
  | "restarting"
  | "reloading"
  | "cleared";

export type PreviewRuntimeSessionOverride = {
  artifactId: string;
  mode: PreviewRuntimeSessionOverrideMode;
  requestedAt: string;
};

export type PreviewRuntimeIndicatorTone =
  | "active"
  | "success"
  | "warning"
  | "danger"
  | "muted";

export type PreviewRuntimeIndicator = {
  id: "session" | "target" | "renderer" | "context";
  label: string;
  tone: PreviewRuntimeIndicatorTone;
  value: string;
};

export type PreviewControllerModel = {
  canClear: boolean;
  canFullscreen: boolean;
  canReload: boolean;
  canReset: boolean;
  canRestart: boolean;
  indicators: PreviewRuntimeIndicator[];
  isFullscreen: boolean;
  isSessionOverridden: boolean;
  sessionLabel: string;
};

export function createPreviewSessionOverride(
  artifactId: string,
  mode: PreviewRuntimeSessionOverrideMode
): PreviewRuntimeSessionOverride {
  return {
    artifactId,
    mode,
    requestedAt: new Date().toISOString()
  };
}

export function buildPreviewControllerModel({
  isFullscreen,
  preview,
  sessionOverride
}: {
  isFullscreen: boolean;
  preview: PreviewSummary;
  sessionOverride: PreviewRuntimeSessionOverride | null;
}): PreviewControllerModel {
  const sessionLabel = formatPreviewSessionLabel(preview, sessionOverride);
  const indicators: PreviewRuntimeIndicator[] = [
    {
      id: "session",
      label: "Session",
      tone: resolveSessionTone(preview, sessionOverride),
      value: sessionLabel
    },
    {
      id: "target",
      label: "Target",
      tone: preview.available ? "muted" : "warning",
      value: preview.target || "Pending"
    },
    {
      id: "renderer",
      label: "Renderer",
      tone: preview.renderer ? "muted" : "warning",
      value: preview.renderer || "Deferred"
    },
    {
      id: "context",
      label: "Context",
      tone: preview.available ? "muted" : "warning",
      value: preview.sourceArtifactName || preview.artifactName || "No artifact"
    }
  ];

  return {
    canClear: preview.available && sessionOverride?.mode !== "cleared",
    canFullscreen: preview.available,
    canReload:
      sessionOverride !== null ||
      preview.state === "error" ||
      preview.state === "unavailable",
    canReset:
      preview.available &&
      (sessionOverride !== null ||
        isFullscreen ||
        Boolean(preview.outputArtifactName) ||
        preview.state === "unavailable"),
    canRestart: preview.available,
    indicators,
    isFullscreen,
    isSessionOverridden: sessionOverride !== null,
    sessionLabel
  };
}

function formatPreviewSessionLabel(
  preview: PreviewSummary,
  sessionOverride: PreviewRuntimeSessionOverride | null
) {
  if (sessionOverride?.mode === "restarting") {
    return "Restarting";
  }

  if (sessionOverride?.mode === "reloading") {
    return "Reloading";
  }

  if (sessionOverride?.mode === "cleared") {
    return "Cleared";
  }

  switch (preview.state) {
    case "generating":
      return "Generating";
    case "ready":
      return preview.active ? "Live" : "Ready";
    case "error":
      return "Failed";
    case "unavailable":
      return "Unavailable";
    default:
      return "Ready";
  }
}

function resolveSessionTone(
  preview: PreviewSummary,
  sessionOverride: PreviewRuntimeSessionOverride | null
): PreviewRuntimeIndicatorTone {
  if (
    sessionOverride?.mode === "restarting" ||
    sessionOverride?.mode === "reloading"
  ) {
    return "active";
  }

  if (sessionOverride?.mode === "cleared") {
    return "warning";
  }

  switch (preview.state) {
    case "generating":
      return "active";
    case "ready":
      return "success";
    case "error":
      return "danger";
    case "unavailable":
      return "warning";
    default:
      return "muted";
  }
}
