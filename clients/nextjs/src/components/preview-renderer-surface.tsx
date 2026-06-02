"use client";

import type { PreviewSummary } from "@/lib/assistant-client";
import {
  getExecutablePreviewRuntimeKind,
  type PreviewRuntimeFrameSample,
  type PreviewRuntimeSource,
  type PreviewRuntimeStatus
} from "@/lib/preview-runtime-adapters";
import type {
  CreativePreviewRendererKind,
  PreviewRendererRoute,
  PreviewRendererSurfaceKind
} from "@/lib/preview-renderers";
import {
  PreviewRuntimeStage,
  type PreviewRuntimeDiagnosticsEvent,
  type PreviewRuntimeTelemetryEvent
} from "./preview-runtime-stage";

type PreviewRendererSurfaceProps = {
  chrome?: "default" | "immersive";
  onRuntimeDiagnostics?: PreviewRuntimeCallbackProps["onRuntimeDiagnostics"];
  onReload?: (() => void) | undefined;
  onRuntimeFrame?: PreviewRuntimeCallbackProps["onRuntimeFrame"];
  onRuntimeStatus?: PreviewRuntimeCallbackProps["onRuntimeStatus"];
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  runtimeSessionKey: string;
  runtimeSource: PreviewRuntimeSource;
};

type PreviewRuntimeCallbackProps = {
  onRuntimeDiagnostics?: (event: PreviewRuntimeDiagnosticsEvent) => void;
  onRuntimeFrame?: (
    event: PreviewRuntimeTelemetryEvent & {
      sample: PreviewRuntimeFrameSample;
    }
  ) => void;
  onRuntimeStatus?: (
    event: PreviewRuntimeTelemetryEvent & {
      status: PreviewRuntimeStatus;
    }
  ) => void;
};

const creativeSurfaceLayers: Record<CreativePreviewRendererKind, readonly string[]> = {
  p5: ["setup()", "draw()", "canvas"],
  three: ["scene", "camera", "lights", "renderer"],
  glsl: ["vertex", "fragment", "uniforms"]
};

const mediaSurfaceLayers: Record<
  Exclude<PreviewRendererSurfaceKind, CreativePreviewRendererKind | "unsupported">,
  readonly string[]
> = {
  json_panel: ["manifest", "route", "renderer", "session"],
  text_panel: ["summary", "notes", "constraints"],
  image_asset: ["asset", "frame", "inspect"],
  audio_asset: ["transport", "levels", "waveform"],
  video_asset: ["timeline", "frame", "transport"]
};

export function PreviewRendererSurface({
  chrome = "default",
  onRuntimeDiagnostics,
  onReload,
  onRuntimeFrame,
  onRuntimeStatus,
  preview,
  route,
  runtimeSessionKey,
  runtimeSource
}: PreviewRendererSurfaceProps) {
  return (
    <section
      aria-label="Preview renderer surface"
      className="previewSurface"
      data-chrome={chrome}
      data-runtime-state={preview.state}
      data-surface-kind={route.surfaceKind}
      data-support-state={route.supportState}
      data-tone={route.tone}
      role="group"
    >
      {chrome === "default" ? (
        <header className="previewSurfaceHeader">
          <div>
            <span className="eyebrow">{route.surfaceEyebrow}</span>
            <strong>{route.surfaceTitle}</strong>
            <p>{route.rendererDescription}</p>
          </div>
          <div className="previewSurfaceStatus">
            <small>{route.supportLabel}</small>
            <span>{route.rendererLabel}</span>
          </div>
        </header>
      ) : null}
      {renderPreviewSurfaceStage({
        chrome,
        onRuntimeDiagnostics,
        onReload,
        onRuntimeFrame,
        onRuntimeStatus,
        preview,
        route,
        runtimeSessionKey,
        runtimeSource
      })}
      {chrome === "default" ? (
        <>
          <div className="previewSurfaceNotes" aria-label="Preview renderer notes">
            {route.notes.map((note) => (
              <span key={note}>{note}</span>
            ))}
          </div>
          <footer className="previewSurfaceFooter">
            <span>{route.targetLabel}</span>
            <small>
              {preview.state === "generating"
                ? "Runtime metadata is still arriving."
                : route.supportReason}
            </small>
          </footer>
        </>
      ) : null}
    </section>
  );
}

function renderPreviewSurfaceStage({
  chrome,
  onRuntimeDiagnostics,
  onReload,
  onRuntimeFrame,
  onRuntimeStatus,
  preview,
  route,
  runtimeSessionKey,
  runtimeSource
}: PreviewRendererSurfaceProps) {
  if (route.surfaceKind === "unsupported") {
    return (
      <div
        aria-label="Unsupported preview surface"
        className="previewSurfaceStage previewSurfaceStageUnsupported"
      >
        <div className="previewSurfaceNotice">
          <strong>{route.rendererLabel}</strong>
          <p>{route.surfaceSummary}</p>
        </div>
        <div className="previewSurfaceCapabilityList" aria-label="Supported foundations">
          <span>p5.js</span>
          <span>Three.js</span>
          <span>GLSL</span>
        </div>
        <div className="previewSurfaceMetaGrid">
          <div>
            <span>Selected</span>
            <strong>{route.selectedArtifactName}</strong>
          </div>
          <div>
            <span>Target</span>
            <strong>{route.targetLabel}</strong>
          </div>
          <div>
            <span>Lifecycle</span>
            <strong>{formatPreviewStateLabel(preview.state)}</strong>
          </div>
        </div>
      </div>
    );
  }

  if (isCreativeSurface(route.surfaceKind)) {
    const runtimeKind = getExecutablePreviewRuntimeKind(route);

    if (runtimeKind) {
      return (
        <PreviewRuntimeStage
          kind={runtimeKind}
          onRuntimeDiagnostics={onRuntimeDiagnostics}
          onReload={onReload}
          onRuntimeFrame={onRuntimeFrame}
          onRuntimeStatus={onRuntimeStatus}
          preview={preview}
          route={route}
          runtimeSessionKey={runtimeSessionKey}
          source={runtimeSource}
        />
      );
    }

    return (
      <div
        aria-label={`${route.rendererLabel} preview surface`}
        className="previewSurfaceStage previewSurfaceStageCreative"
      >
        <div className="previewSurfaceBackdrop" />
        <div className="previewSurfaceChipRow">
          {creativeSurfaceLayers[route.surfaceKind].map((layer) => (
            <span key={layer}>{layer}</span>
          ))}
        </div>
        {chrome === "immersive" ? (
          <div className="previewSurfaceImmersiveHint" aria-hidden="true">
            Visual output focused mode
          </div>
        ) : (
          <div className="previewSurfaceGrid" aria-hidden="true">
            {Array.from({ length: 9 }, (_, index) => (
              <span
                data-active={index === 4 ? "true" : "false"}
                key={`${route.surfaceKind}-${index}`}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      aria-label={`${route.rendererLabel} preview surface`}
      className="previewSurfaceStage previewSurfaceStagePanel"
    >
      <div className="previewSurfacePanelFrame">
        <div className="previewSurfacePanelHeader">
          <span>{route.selectedArtifactName}</span>
          <small>{route.targetLabel}</small>
        </div>
        <div className="previewSurfacePanelBody">
          {mediaSurfaceLayers[route.surfaceKind].map((layer) => (
            <span key={layer}>{layer}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function isCreativeSurface(
  surfaceKind: PreviewRendererSurfaceKind
): surfaceKind is CreativePreviewRendererKind {
  return (
    surfaceKind === "p5" ||
    surfaceKind === "three" ||
    surfaceKind === "glsl"
  );
}

function formatPreviewStateLabel(state: PreviewSummary["state"]) {
  switch (state) {
    case "generating":
      return "Generating";
    case "ready":
      return "Ready";
    case "error":
      return "Failed";
    case "unavailable":
      return "Unavailable";
    default:
      return "Ready";
  }
}
