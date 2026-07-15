"use client";

import { useEffect, useRef, useState } from "react";
import type { PreviewSummary } from "@/lib/assistant-client";
import {
  canRunPreviewRuntime,
  getInitialPreviewRuntimeStatus,
  type PreviewExecutableRuntimeKind,
  type PreviewRuntimeFrameSample,
  type PreviewRuntimeSource,
  type PreviewRuntimeStatus
} from "@/lib/preview-runtime-adapters";
import {
  buildPreviewRuntimeOverlayModel,
  createPreviewRuntimeMetricsTracker,
  type PreviewRuntimeMetricsSnapshot
} from "@/lib/preview-runtime-diagnostics";
import type { PreviewRendererRoute } from "@/lib/preview-renderers";
import {
  createPreviewSandboxRuntimeId,
  mountPreviewSandboxRuntime,
  type PreviewSandboxKeyboardBoundaryEvent
} from "@/lib/preview-sandbox-runtime";
import { PreviewRuntimeErrorCard } from "./preview-runtime-error-card";

export type PreviewRuntimeTelemetryEvent = {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeId: string;
  source: PreviewRuntimeSource;
};

export type PreviewRuntimeDiagnosticsEvent = PreviewRuntimeTelemetryEvent & {
  metrics: PreviewRuntimeMetricsSnapshot;
  status: PreviewRuntimeStatus;
};

type PreviewRuntimeStageProps = {
  captureHostKeyboard?: boolean;
  kind: PreviewExecutableRuntimeKind;
  onKeyboardBoundary?:
    | ((event: PreviewSandboxKeyboardBoundaryEvent) => void)
    | undefined;
  onRuntimeDiagnostics?: (event: PreviewRuntimeDiagnosticsEvent) => void;
  onOpenCode?: (() => void) | undefined;
  onRuntimeFrame?: (
    event: PreviewRuntimeTelemetryEvent & { sample: PreviewRuntimeFrameSample }
  ) => void;
  onRuntimeStatus?: (
    event: PreviewRuntimeTelemetryEvent & { status: PreviewRuntimeStatus }
  ) => void;
  onReload?: (() => void) | undefined;
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  runtimeSessionKey: string;
  showDiagnostics?: boolean;
  source: PreviewRuntimeSource;
};

export function PreviewRuntimeStage({
  captureHostKeyboard = false,
  kind,
  onKeyboardBoundary,
  onOpenCode,
  onRuntimeDiagnostics,
  onRuntimeFrame,
  onRuntimeStatus,
  onReload,
  preview,
  route,
  runtimeSessionKey,
  showDiagnostics = true,
  source
}: PreviewRuntimeStageProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const onKeyboardBoundaryRef = useRef(onKeyboardBoundary);
  const onRuntimeDiagnosticsRef = useRef(onRuntimeDiagnostics);
  const onRuntimeFrameRef = useRef(onRuntimeFrame);
  const onRuntimeStatusRef = useRef(onRuntimeStatus);
  const previewRef = useRef(preview);
  const routeRef = useRef(route);
  const sourceRef = useRef(source);
  const [status, setStatus] = useState<PreviewRuntimeStatus>(() =>
    getInitialPreviewRuntimeStatus({ kind, preview })
  );
  const [metrics, setMetrics] = useState<PreviewRuntimeMetricsSnapshot>(() => {
    const initialStatus = getInitialPreviewRuntimeStatus({ kind, preview });
    return createPreviewRuntimeMetricsTracker(initialStatus, {
      publishIntervalMs: 0
    }).snapshot();
  });
  // "generating" and "ready" are both runnable states. Treating their
  // transition as a new lifecycle remounts the iframe just after it reports a
  // healthy running status (for example, when a restart settles).
  const runtimeExecutionState =
    preview.state === "error" || preview.state === "unavailable"
      ? preview.state
      : "runnable";
  const overlay = buildPreviewRuntimeOverlayModel({
    kind,
    route,
    runtimeSource: source,
    snapshot: metrics,
    status
  });
  const canMountRuntime = canRunPreviewRuntime({ preview, route });
  const visibleRuntimeSource =
    showDiagnostics || !route.surfaceTitle.trim()
      ? source
      : {
          ...source,
          title: route.surfaceTitle
        };
  const userStatusLabel =
    !showDiagnostics && status.error ? "Preview fallback available" : status.label;
  const userStatusDetail =
    !showDiagnostics && status.error
      ? "This artifact needs a compatible preview route. Use the reload control after selecting runnable code, or continue with artifact evidence."
      : status.detail;

  useEffect(() => {
    onKeyboardBoundaryRef.current = onKeyboardBoundary;
    onRuntimeDiagnosticsRef.current = onRuntimeDiagnostics;
    onRuntimeFrameRef.current = onRuntimeFrame;
    onRuntimeStatusRef.current = onRuntimeStatus;
    previewRef.current = preview;
    routeRef.current = route;
    sourceRef.current = source;
  }, [
    onKeyboardBoundary,
    onRuntimeDiagnostics,
    onRuntimeFrame,
    onRuntimeStatus,
    preview,
    route,
    source
  ]);

  useEffect(() => {
    const currentPreview = previewRef.current;
    const currentRoute = routeRef.current;
    const rawSource = sourceRef.current;
    const currentSource =
      showDiagnostics || !currentRoute.surfaceTitle.trim()
        ? rawSource
        : {
            ...rawSource,
            title: currentRoute.surfaceTitle
          };
    const initialStatus = getInitialPreviewRuntimeStatus({
      kind,
      preview: currentPreview
    });
    const visibleInitialStatus = sanitizePreviewRuntimeStatusForUser({
      displayTitle: currentSource.title,
      rawTitle: rawSource.title,
      showDiagnostics,
      status: initialStatus
    });
    const tracker = createPreviewRuntimeMetricsTracker(initialStatus);
    const runtimeId = createPreviewSandboxRuntimeId();
    let latestStatus = visibleInitialStatus;

    setStatus(visibleInitialStatus);
    setMetrics(tracker.snapshot());

    function publishRuntimeDiagnostics(
      nextStatus: PreviewRuntimeStatus,
      nextMetrics: PreviewRuntimeMetricsSnapshot
    ) {
      onRuntimeDiagnosticsRef.current?.({
        kind,
        metrics: nextMetrics,
        route: currentRoute,
        runtimeId,
        source: currentSource,
        status: nextStatus
      });
    }

    publishRuntimeDiagnostics(visibleInitialStatus, tracker.snapshot());

    function handleStatus(nextStatus: PreviewRuntimeStatus) {
      const visibleStatus = sanitizePreviewRuntimeStatusForUser({
        displayTitle: currentSource.title,
        rawTitle: rawSource.title,
        showDiagnostics,
        status: nextStatus
      });
      latestStatus = visibleStatus;
      const nextMetrics = tracker.publishStatus(visibleStatus);
      setStatus(visibleStatus);
      setMetrics(nextMetrics);
      publishRuntimeDiagnostics(visibleStatus, nextMetrics);
      onRuntimeStatusRef.current?.({
        kind,
        route: currentRoute,
        runtimeId,
        source: currentSource,
        status: visibleStatus
      });
    }

    function handleFrame(sample: PreviewRuntimeFrameSample) {
      const nextMetrics = tracker.recordFrame(sample);

      if (nextMetrics) {
        setMetrics(nextMetrics);
        publishRuntimeDiagnostics(latestStatus, nextMetrics);
      }

      onRuntimeFrameRef.current?.({
        kind,
        route: currentRoute,
        runtimeId,
        sample,
        source: currentSource
      });
    }

    if (!canRunPreviewRuntime({ preview: currentPreview, route: currentRoute })) {
      return undefined;
    }

    const iframe = iframeRef.current;
    if (!iframe) {
      handleStatus({
        detail: "The preview frame is not ready yet.",
        label: "Preview runtime waiting",
        state: "idle",
        error: null
      });
      return undefined;
    }
    const runtime = mountPreviewSandboxRuntime({
      captureHostKeyboard,
      iframe,
      kind,
      onKeyboardBoundary: (event) => onKeyboardBoundaryRef.current?.(event),
      onFrame: handleFrame,
      onStatus: handleStatus,
      runtimeId,
      showStatusOverlay: false,
      source: currentSource
    });

    return () => {
      runtime.dispose();
    };
  }, [
    captureHostKeyboard,
    kind,
    preview.active,
    preview.error?.type,
    runtimeExecutionState,
    route.rendererId,
    route.rendererLabel,
    route.supportState,
    route.surfaceTitle,
    route.surfaceKind,
    runtimeSessionKey,
    showDiagnostics,
    source.fingerprint,
    source.lineCount,
    source.source,
    source.title
  ]);

  return (
    <div
      aria-label={`${route.rendererLabel} live runtime`}
      className="previewSurfaceStage previewRuntimeStage"
      data-runtime-health={metrics.health}
      data-runtime-kind={kind}
      data-runtime-state={status.state}
      role="group"
    >
      <iframe
        aria-label={`${route.rendererLabel} preview runtime frame`}
        aria-hidden={canMountRuntime ? undefined : true}
        className="previewRuntimeFrame"
        ref={iframeRef}
        sandbox="allow-scripts"
        tabIndex={canMountRuntime ? 0 : -1}
        title={`${route.rendererLabel} preview runtime`}
      />
      {status.error ? null : (
        <div className="previewRuntimeOverlay">
          <div
            aria-atomic="true"
            aria-live="polite"
            className="previewRuntimeStatus"
          >
            <div className="previewRuntimeOverlayHeader">
              <small>{userStatusLabel}</small>
              <span
                className="previewRuntimeOverlayHealth"
                data-tone={overlay.healthTone}
              >
                {overlay.healthLabel}
              </span>
            </div>
            <span>{userStatusDetail}</span>
          </div>
          {showDiagnostics ? (
            <div
              aria-label="Renderer health overlay"
              className="previewRuntimeMetrics"
              role="list"
            >
              {overlay.metrics.map((metric) => (
                <div
                  className="previewRuntimeMetric"
                  data-tone={metric.tone}
                  key={metric.id}
                  role="listitem"
                >
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>
          ) : null}
          {showDiagnostics && overlay.diagnostics.length > 0 ? (
            <div className="previewRuntimeDiagnostics" aria-label="Runtime notes">
              {overlay.diagnostics.map((diagnostic) => (
                <span key={diagnostic}>{diagnostic}</span>
              ))}
            </div>
          ) : null}
        </div>
      )}
      {status.error ? (
        <div className="previewRuntimeErrorBoundary">
          {showDiagnostics ? (
            <PreviewRuntimeErrorCard
              error={status.error}
              onOpenCode={onOpenCode}
              onReload={onReload}
            />
          ) : (
            <div className="previewRuntimeUserFallback" role="alert">
              <strong>Preview fallback ready</strong>
              <p>
                The current artifact is not runnable in this preview route. Use
                a compatible artifact, reload preview, or continue with the
                documented artifact evidence.
              </p>
            </div>
          )}
          {!showDiagnostics && onReload ? (
            <button
              aria-label="Reload preview runtime"
              className="previewRuntimeActionButton"
              data-action="reload"
              onClick={onReload}
              type="button"
            >
              Try reload
            </button>
          ) : null}
        </div>
      ) : null}
      {showDiagnostics ? (
        <div className="previewRuntimeMeta" aria-label="Preview runtime source">
          <span>{visibleRuntimeSource.title}</span>
          <small>
            {visibleRuntimeSource.lineCount} lines / {visibleRuntimeSource.fingerprint}
          </small>
        </div>
      ) : null}
    </div>
  );
}

function sanitizePreviewRuntimeStatusForUser({
  displayTitle,
  rawTitle,
  showDiagnostics,
  status
}: {
  displayTitle: string;
  rawTitle: string;
  showDiagnostics: boolean;
  status: PreviewRuntimeStatus;
}): PreviewRuntimeStatus {
  if (showDiagnostics || !rawTitle.trim() || rawTitle === displayTitle) {
    return status;
  }

  const replaceRawTitle = (value: string | undefined) =>
    value ? value.split(rawTitle).join(displayTitle) : value;

  return {
    ...status,
    detail: replaceRawTitle(status.detail) ?? status.detail,
    diagnostics: status.diagnostics?.map(
      (diagnostic) => replaceRawTitle(diagnostic) ?? diagnostic
    ),
    error: status.error
      ? {
          ...status.error,
          debugMessage:
            replaceRawTitle(status.error.debugMessage ?? undefined) ??
            status.error.debugMessage,
          suggestedAction:
            replaceRawTitle(status.error.suggestedAction) ??
            status.error.suggestedAction,
          userMessage:
            replaceRawTitle(status.error.userMessage) ?? status.error.userMessage
        }
      : status.error
  };
}
