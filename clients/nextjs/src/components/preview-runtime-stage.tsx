"use client";

import { useEffect, useRef, useState } from "react";
import type { PreviewSummary } from "@/lib/assistant-client";
import {
  canRunPreviewRuntime,
  getInitialPreviewRuntimeStatus,
  mountPreviewRuntime,
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
import { SubsystemErrorCallout } from "./subsystem-error-callout";

type PreviewRuntimeStageProps = {
  kind: PreviewExecutableRuntimeKind;
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  source: PreviewRuntimeSource;
};

export function PreviewRuntimeStage({
  kind,
  preview,
  route,
  source
}: PreviewRuntimeStageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
    const initialStatus = getInitialPreviewRuntimeStatus({ kind, preview });
    const tracker = createPreviewRuntimeMetricsTracker(initialStatus);

    setStatus(initialStatus);
    setMetrics(tracker.snapshot());

    function handleStatus(nextStatus: PreviewRuntimeStatus) {
      setStatus(nextStatus);
      setMetrics(tracker.publishStatus(nextStatus));
    }

    function handleFrame(sample: PreviewRuntimeFrameSample) {
      const nextMetrics = tracker.recordFrame(sample);

      if (nextMetrics) {
        setMetrics(nextMetrics);
      }
    }

    if (!canRunPreviewRuntime({ preview, route })) {
      return undefined;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      handleStatus({
        detail: "The runtime canvas is not ready yet.",
        label: "Runtime waiting",
        state: "idle",
        error: null
      });
      return undefined;
    }

    handleStatus({
      detail:
        kind === "glsl"
          ? "Mounting a bounded WebGL fragment runtime."
          : kind === "three"
            ? "Mounting a controlled Three.js-style WebGL scene runtime."
            : "Mounting a constrained canvas sketch runtime.",
      label: "Runtime starting",
      state: "starting",
      error: null
    });

    const runtime = mountPreviewRuntime({
      canvas,
      kind,
      onFrame: handleFrame,
      onStatus: handleStatus,
      source
    });

    return () => {
      runtime.dispose();
    };
  }, [kind, preview, route, source]);

  return (
    <div
      aria-label={`${route.rendererLabel} live runtime`}
      className="previewSurfaceStage previewRuntimeStage"
      data-runtime-health={metrics.health}
      data-runtime-kind={kind}
      data-runtime-state={status.state}
      role="group"
    >
      <canvas
        aria-label={`${route.rendererLabel} live runtime canvas`}
        className="previewRuntimeCanvas"
        ref={canvasRef}
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
        <SubsystemErrorCallout
          className="previewRuntimeErrorCallout"
          error={status.error}
          title="Renderer runtime failed"
        />
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
