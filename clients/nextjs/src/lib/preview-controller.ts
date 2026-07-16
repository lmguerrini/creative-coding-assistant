import type { PreviewSummary } from "./assistant-client";
import type { PreviewRendererRoute } from "./preview-renderers";
import type { PreviewRuntimeLifecycleState } from "./preview-runtime-adapters";

export type PreviewRuntimeSessionOverrideMode =
  | "restarting"
  | "reloading"
  | "cleared"
  | "settled";

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
  id: "session" | "artifact" | "target" | "surface" | "support";
  label: string;
  tone: PreviewRuntimeIndicatorTone;
  value: string;
};

export type PreviewControllerModel = {
  canClear: boolean;
  canFullscreen: boolean;
  canReload: boolean;
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
  route,
  sessionOverride
}: {
  isFullscreen: boolean;
  preview: PreviewSummary;
  route: PreviewRendererRoute;
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
      id: "artifact",
      label: "Artifact",
      tone: preview.available ? "muted" : "warning",
      value: route.selectedArtifactName || preview.artifactName || "No preview artifact"
    },
    {
      id: "target",
      label: "Target",
      tone: route.targetId ? "muted" : "warning",
      value: route.targetLabel || preview.target || "Pending target"
    },
    {
      id: "surface",
      label: "Surface",
      tone: route.tone,
      value: route.rendererLabel
    },
    {
      id: "support",
      label: "Support",
      tone: route.tone,
      value: route.supportLabel
    }
  ];

  return {
    canClear: preview.available && sessionOverride?.mode !== "cleared",
    canFullscreen: preview.available,
    canReload: preview.available && sessionOverride?.mode !== "reloading",
    canRestart: preview.available,
    indicators,
    isFullscreen,
    isSessionOverridden: hasActiveSessionOverride(sessionOverride),
    sessionLabel
  };
}

function hasActiveSessionOverride(
  sessionOverride: PreviewRuntimeSessionOverride | null
) {
  return sessionOverride !== null && sessionOverride.mode !== "settled";
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
      return preview.active ? "Success" : "Ready";
    case "error":
      return "Failure";
    case "unavailable":
      return "Unavailable";
    default:
      return "Ready";
  }
}

/**
 * Bounded readiness budget for the one automatic preview recovery.
 *
 * A newly opened, valid preview sometimes mounts its sandbox iframe yet never
 * advances past "starting" — the runtime stays black at Starting/Warming even
 * though clicking the existing Reload control immediately renders it. There is
 * no runtime-level warming/readiness timeout to reuse, so this is the smallest
 * safe deadline that still comfortably outlasts a healthy cold start. That cold
 * start is entirely local (no network): the sandbox iframe loads a static
 * document, completes its postMessage handshake, and every route emits its first
 * "running"/"ready" status synchronously after the user script runs — the
 * heaviest one-time cost being Three.js r176 WebGL/context setup, still well
 * under ~500 ms on normal hardware (the metrics publish cadence is 220 ms). This
 * keeps a ≥2x margin over that worst case while roughly halving the visible
 * black-preview delay: long enough not to race a runtime that is about to run,
 * short enough to mirror the operator's own "it's stuck, reload it" reflex.
 */
export const previewAutoRecoveryReadinessBudgetMs = 1500;

const settledPreviewRuntimeStates: ReadonlySet<PreviewRuntimeLifecycleState> =
  new Set(["ready", "running", "stopped", "error"]);

/**
 * A runtime that never advanced past "starting"/"idle" (or has reported nothing
 * yet) has not produced its first runnable frame. "ready" (for example Tone.js
 * armed and awaiting explicit playback), "running", "stopped" and "error" are
 * all explicit outcomes that must never be auto-reloaded.
 */
export function isPreviewRuntimeAwaitingFirstFrame(
  runtimeState: PreviewRuntimeLifecycleState | null
): boolean {
  if (runtimeState === null) {
    return true;
  }

  return !settledPreviewRuntimeStates.has(runtimeState);
}

/**
 * Decide whether the current preview session is eligible to arm its single
 * automatic recovery. This never decides to reload on its own — a bounded
 * readiness timer must still confirm the runtime is still awaiting its first
 * frame ({@link isPreviewRuntimeAwaitingFirstFrame}) before reusing the manual
 * Reload path.
 */
export function canArmPreviewAutoRecovery({
  consumedIdentity,
  isPreviewable,
  isOpen,
  recoveryIdentity,
  sessionOverrideMode
}: {
  consumedIdentity: string | null;
  isPreviewable: boolean;
  isOpen: boolean;
  recoveryIdentity: string;
  sessionOverrideMode: PreviewRuntimeSessionOverrideMode | null;
}): boolean {
  if (!isOpen || !isPreviewable) {
    return false;
  }

  // At most one automatic recovery per artifact/version.
  if (consumedIdentity === recoveryIdentity) {
    return false;
  }

  // Never stack a recovery onto an in-flight override. An override only settles
  // once the runtime actually reaches "running", so a manual or automatic
  // reload that stays stuck keeps mode !== "settled" and can never trigger a
  // second automatic reload — this is what makes a reload loop impossible.
  if (sessionOverrideMode !== null && sessionOverrideMode !== "settled") {
    return false;
  }

  return true;
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
