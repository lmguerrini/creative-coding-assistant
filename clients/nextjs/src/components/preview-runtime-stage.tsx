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
  mountPreviewSandboxRuntime
} from "@/lib/preview-sandbox-runtime";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

export type PreviewRuntimeTelemetryEvent = {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeId: string;
  source: PreviewRuntimeSource;
};

type PreviewRuntimeStageProps = {
  kind: PreviewExecutableRuntimeKind;
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
  source: PreviewRuntimeSource;
};

export function PreviewRuntimeStage({
  kind,
  onRuntimeFrame,
  onRuntimeStatus,
  onReload,
  preview,
  route,
  runtimeSessionKey,
  source
}: PreviewRuntimeStageProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
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
  const overlay = buildPreviewRuntimeOverlayModel({
    kind,
    route,
    runtimeSource: source,
    snapshot: metrics,
    status
  });

  useEffect(() => {
    onRuntimeFrameRef.current = onRuntimeFrame;
    onRuntimeStatusRef.current = onRuntimeStatus;
    previewRef.current = preview;
    routeRef.current = route;
    sourceRef.current = source;
  }, [onRuntimeFrame, onRuntimeStatus, preview, route, source]);

  useEffect(() => {
    const currentPreview = previewRef.current;
    const currentRoute = routeRef.current;
    const currentSource = sourceRef.current;
    const initialStatus = getInitialPreviewRuntimeStatus({
      kind,
      preview: currentPreview
    });
    const tracker = createPreviewRuntimeMetricsTracker(initialStatus);

    setStatus(initialStatus);
    setMetrics(tracker.snapshot());

    function handleStatus(nextStatus: PreviewRuntimeStatus) {
      setStatus(nextStatus);
      setMetrics(tracker.publishStatus(nextStatus));
      onRuntimeStatusRef.current?.({
        kind,
        route: currentRoute,
        runtimeId,
        source: currentSource,
        status: nextStatus
      });
    }

    function handleFrame(sample: PreviewRuntimeFrameSample) {
      const nextMetrics = tracker.recordFrame(sample);

      if (nextMetrics) {
        setMetrics(nextMetrics);
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
        detail: "The sandbox frame is not ready yet.",
        label: "Runtime sandbox waiting",
        state: "idle",
        error: null
      });
      return undefined;
    }

    const runtimeId = createPreviewSandboxRuntimeId();
    const runtime = mountPreviewSandboxRuntime({
      iframe,
      kind,
      onFrame: handleFrame,
      onStatus: handleStatus,
      runtimeId,
      source: currentSource
    });

    return () => {
      runtime.dispose();
    };
  }, [
    kind,
    preview.active,
    preview.error?.type,
    preview.state,
    route.rendererId,
    route.rendererLabel,
    route.supportState,
    route.surfaceKind,
    runtimeSessionKey,
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
        aria-label={`${route.rendererLabel} sandbox runtime frame`}
        className="previewRuntimeFrame"
        ref={iframeRef}
        sandbox="allow-scripts allow-same-origin"
        title={`${route.rendererLabel} sandbox runtime`}
      />
      <div className="previewRuntimeOverlay" aria-live="polite">
        <div className="previewRuntimeOverlayHeader">
          <small>{status.label}</small>
          <span
            className="previewRuntimeOverlayHealth"
            data-tone={overlay.healthTone}
          >
            {overlay.healthLabel}
          </span>
        </div>
        <span>{status.detail}</span>
        <div
          aria-label="Renderer diagnostics overlay"
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
        {overlay.diagnostics.length > 0 ? (
          <div className="previewRuntimeDiagnostics" aria-label="Runtime diagnostics">
            {overlay.diagnostics.map((diagnostic) => (
              <span key={diagnostic}>{diagnostic}</span>
            ))}
          </div>
        ) : null}
      </div>
      {status.error ? (
        <div className="previewRuntimeErrorBoundary" role="alert">
          <SubsystemErrorCallout
            className="previewRuntimeErrorCallout"
            error={status.error}
            role="status"
            title="Renderer runtime failed"
          />
          {onReload ? (
            <button
              aria-label="Reload preview runtime"
              className="previewRuntimeRecoveryButton"
              onClick={onReload}
              type="button"
            >
              Reload preview
            </button>
          ) : null}
        </div>
      ) : null}
      <div className="previewRuntimeMeta" aria-label="Preview runtime source">
        <span>{source.title}</span>
        <small>
          {source.lineCount} lines / {source.fingerprint}
        </small>
      </div>
    </div>
  );
}
