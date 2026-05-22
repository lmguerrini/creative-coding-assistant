"use client";

import { useEffect, useRef, useState } from "react";
import type { PreviewSummary } from "@/lib/assistant-client";
import {
  canRunPreviewRuntime,
  getInitialPreviewRuntimeStatus,
  mountPreviewRuntime,
  type PreviewExecutableRuntimeKind,
  type PreviewRuntimeSource,
  type PreviewRuntimeStatus
} from "@/lib/preview-runtime-adapters";
import type { PreviewRendererRoute } from "@/lib/preview-renderers";

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

  useEffect(() => {
    setStatus(getInitialPreviewRuntimeStatus({ kind, preview }));

    if (!canRunPreviewRuntime({ preview, route })) {
      return undefined;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      setStatus({
        detail: "The runtime canvas is not ready yet.",
        label: "Runtime waiting",
        state: "idle"
      });
      return undefined;
    }

    setStatus({
      detail:
        kind === "glsl"
          ? "Mounting a bounded WebGL fragment runtime."
          : "Mounting a constrained canvas sketch runtime.",
      label: "Runtime starting",
      state: "starting"
    });

    const runtime = mountPreviewRuntime({
      canvas,
      kind,
      onStatus: setStatus,
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
        <small>{status.label}</small>
        <span>{status.detail}</span>
      </div>
      <div className="previewRuntimeMeta" aria-label="Preview runtime source">
        <span>{source.title}</span>
        <small>
          {source.lineCount} lines / {source.fingerprint}
        </small>
      </div>
    </div>
  );
}
