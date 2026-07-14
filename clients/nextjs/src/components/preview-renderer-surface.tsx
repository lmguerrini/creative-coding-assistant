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
import type { PreviewSandboxKeyboardBoundaryEvent } from "@/lib/preview-sandbox-runtime";
import {
  PreviewRuntimeStage,
  type PreviewRuntimeDiagnosticsEvent,
  type PreviewRuntimeTelemetryEvent
} from "./preview-runtime-stage";

type PreviewRendererSurfaceProps = {
  captureHostKeyboard?: boolean;
  chrome?: "comparison" | "default" | "immersive";
  onKeyboardBoundary?: PreviewRuntimeCallbackProps["onKeyboardBoundary"];
  onOpenCode?: (() => void) | undefined;
  onRuntimeDiagnostics?: PreviewRuntimeCallbackProps["onRuntimeDiagnostics"];
  onReload?: (() => void) | undefined;
  onRuntimeFrame?: PreviewRuntimeCallbackProps["onRuntimeFrame"];
  onRuntimeStatus?: PreviewRuntimeCallbackProps["onRuntimeStatus"];
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  runtimeSessionKey: string;
  runtimeSource: PreviewRuntimeSource;
  showDiagnostics?: boolean;
};

type PreviewRuntimeCallbackProps = {
  onKeyboardBoundary?: (event: PreviewSandboxKeyboardBoundaryEvent) => void;
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
  glsl: ["vertex", "fragment", "uniforms"],
  hydra: ["sources", "operators", "feedback", "outputs"],
  tone: ["voices", "envelopes", "sequences", "transport"],
  gsap: ["timeline", "tweens", "stagger", "transforms"],
  svg: ["paths", "gradients", "viewBox", "animate"],
  canvas: ["context", "paths", "fills", "frames"]
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
  captureHostKeyboard = false,
  chrome = "default",
  onKeyboardBoundary,
  onOpenCode,
  onRuntimeDiagnostics,
  onReload,
  onRuntimeFrame,
  onRuntimeStatus,
  preview,
  route,
  runtimeSessionKey,
  runtimeSource,
  showDiagnostics = true
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
            <h3>{route.surfaceTitle}</h3>
            <p>{route.rendererDescription}</p>
          </div>
          <div className="previewSurfaceStatus">
            <small>{route.supportLabel}</small>
            <span>{route.rendererLabel}</span>
          </div>
        </header>
      ) : null}
      {renderPreviewSurfaceStage({
        captureHostKeyboard,
        chrome,
        onKeyboardBoundary,
        onOpenCode,
        onRuntimeDiagnostics,
        onReload,
        onRuntimeFrame,
        onRuntimeStatus,
        preview,
        route,
        runtimeSessionKey,
        runtimeSource,
        showDiagnostics
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
  captureHostKeyboard = false,
  chrome,
  onKeyboardBoundary,
  onOpenCode,
  onRuntimeDiagnostics,
  onReload,
  onRuntimeFrame,
  onRuntimeStatus,
  preview,
  route,
  runtimeSessionKey,
  runtimeSource,
  showDiagnostics = true
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
        <ul className="previewSurfaceCapabilityList" aria-label="Supported foundations">
          <li>p5.js</li>
          <li>Three.js</li>
          <li>GLSL</li>
          <li>Hydra</li>
          <li>Tone.js</li>
          <li>GSAP</li>
          <li>SVG</li>
          <li>Canvas</li>
        </ul>
        <dl className="previewSurfaceMetaGrid">
          <div>
            <dt>Selected</dt>
            <dd>{route.selectedArtifactName}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>{route.targetLabel}</dd>
          </div>
          <div>
            <dt>Lifecycle</dt>
            <dd>{formatPreviewStateLabel(preview.state)}</dd>
          </div>
        </dl>
      </div>
    );
  }

  if (isCreativeSurface(route.surfaceKind)) {
    const runtimeKind = getExecutablePreviewRuntimeKind(route);

    if (runtimeKind) {
      return (
        <PreviewRuntimeStage
          captureHostKeyboard={captureHostKeyboard}
          kind={runtimeKind}
          onKeyboardBoundary={onKeyboardBoundary}
          onOpenCode={onOpenCode}
          onRuntimeDiagnostics={onRuntimeDiagnostics}
          onReload={onReload}
          onRuntimeFrame={onRuntimeFrame}
          onRuntimeStatus={onRuntimeStatus}
          preview={preview}
          route={route}
          runtimeSessionKey={runtimeSessionKey}
          showDiagnostics={showDiagnostics}
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
        <ul className="previewSurfaceChipRow" aria-label="Renderer layers">
          {creativeSurfaceLayers[route.surfaceKind].map((layer) => (
            <li key={layer}>{layer}</li>
          ))}
        </ul>
        {chrome === "immersive" ? (
          <div className="previewSurfaceImmersiveHint" aria-hidden="true">
            Visual output focused mode
          </div>
        ) : chrome === "default" ? (
          <div className="previewSurfaceGrid" aria-hidden="true">
            {Array.from({ length: 9 }, (_, index) => (
              <span
                data-active={index === 4 ? "true" : "false"}
                key={`${route.surfaceKind}-${index}`}
              />
            ))}
          </div>
        ) : null}
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
        <ul className="previewSurfacePanelBody" aria-label="Preview surface capabilities">
          {mediaSurfaceLayers[route.surfaceKind].map((layer) => (
            <li key={layer}>{layer}</li>
          ))}
        </ul>
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
    surfaceKind === "glsl" ||
    surfaceKind === "hydra" ||
    surfaceKind === "tone" ||
    surfaceKind === "gsap" ||
    surfaceKind === "svg" ||
    surfaceKind === "canvas"
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
