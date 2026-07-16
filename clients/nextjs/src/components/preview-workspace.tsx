"use client";

import {
  useEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type SyntheticEvent
} from "react";
import { createPortal } from "react-dom";
import {
  ChevronDown,
  Maximize2,
  Minimize2,
  Play,
  RefreshCw,
  RotateCcw,
  X
} from "lucide-react";
import type { AssistantWorkspaceSnapshot } from "@/lib/assistant-client";
import type { PreviewControllerModel } from "@/lib/preview-controller";
import type { PreviewRendererRoute } from "@/lib/preview-renderers";
import type { RuntimeConsoleLiveSnapshot } from "@/lib/runtime-console";
import type {
  PreviewExecutableRuntimeKind,
  PreviewRuntimeFrameSample,
  PreviewRuntimeSource,
  PreviewRuntimeStatus
} from "@/lib/preview-runtime-adapters";
import type { PreviewSandboxKeyboardBoundaryEvent } from "@/lib/preview-sandbox-runtime";
import { workspaceLayoutBounds } from "@/lib/workspace-persistence";
import { PreviewRendererSurface } from "./preview-renderer-surface";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

type PreviewRuntimeTelemetryBase = {
  kind: PreviewExecutableRuntimeKind;
  route: PreviewRendererRoute;
  runtimeId: string;
  source: PreviewRuntimeSource;
};

export type PreviewRuntimeStatusTelemetryEvent = PreviewRuntimeTelemetryBase & {
  status: PreviewRuntimeStatus;
};

export type PreviewRuntimeFrameTelemetryEvent = PreviewRuntimeTelemetryBase & {
  sample: PreviewRuntimeFrameSample;
};

export type PreviewWorkspaceProps = {
  controller: PreviewControllerModel;
  height: number;
  onClear: () => void;
  onFullscreenToggle: (isFullscreen: boolean) => void;
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  onReload: () => void;
  onRuntimeDiagnostics: (
    event: Omit<RuntimeConsoleLiveSnapshot, "updatedAt">
  ) => void;
  onResizeKeyDown: (event: ReactKeyboardEvent<HTMLElement>) => void;
  onResizeStart: (event: ReactMouseEvent<HTMLElement>) => void;
  onRestart: () => void;
  onRuntimeFrame: (event: PreviewRuntimeFrameTelemetryEvent) => void;
  onRuntimeStatus: (event: PreviewRuntimeStatusTelemetryEvent) => void;
  onToggle: (isOpen: boolean) => void;
  resizing: boolean;
  route: PreviewRendererRoute;
  runtimeSessionKey: string;
  runtimeSource: PreviewRuntimeSource;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
  userArtifactLabel: string;
};

export function PreviewWorkspace({
  controller,
  height,
  onClear,
  onFullscreenToggle,
  onOpenArtifacts,
  onOpenCode,
  onReload,
  onRuntimeDiagnostics,
  onResizeKeyDown,
  onResizeStart,
  onRestart,
  onRuntimeFrame,
  onRuntimeStatus,
  onToggle,
  resizing,
  route,
  runtimeSessionKey,
  runtimeSource,
  showDebugPanels,
  snapshot,
  userArtifactLabel
}: PreviewWorkspaceProps) {
  const enterFullscreenButtonRef = useRef<HTMLButtonElement>(null);
  const exitFullscreenButtonRef = useRef<HTMLButtonElement>(null);
  const fullscreenLayerRef = useRef<HTMLDivElement>(null);
  const onFullscreenToggleRef = useRef(onFullscreenToggle);
  const shouldRestoreFullscreenFocusRef = useRef(false);
  const canOpenUserPreview =
    showDebugPanels ||
    snapshot.preview.state === "ready" ||
    (controller.isRuntimeRefreshing && snapshot.preview.available);
  const isPreviewPanelOpen = snapshot.preview.active && canOpenUserPreview;

  useEffect(() => {
    onFullscreenToggleRef.current = onFullscreenToggle;
  }, [onFullscreenToggle]);

  useEffect(() => {
    if (!controller.isFullscreen) {
      if (shouldRestoreFullscreenFocusRef.current) {
        shouldRestoreFullscreenFocusRef.current = false;
        enterFullscreenButtonRef.current?.focus();
      }
      return undefined;
    }

    const fullscreenLayer = fullscreenLayerRef.current;
    if (!fullscreenLayer) {
      return undefined;
    }

    shouldRestoreFullscreenFocusRef.current = true;

    const previousOverflow = document.body.style.overflow;
    const backgroundRoot = document.querySelector<HTMLElement>("main.workstation");
    const previousBackgroundAriaHidden = backgroundRoot?.getAttribute("aria-hidden");
    document.body.style.overflow = "hidden";

    exitFullscreenButtonRef.current?.focus();
    backgroundRoot?.setAttribute("aria-hidden", "true");

    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        onFullscreenToggleRef.current(false);
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const focusableControls = getFullscreenFocusableControls(fullscreenLayer);
      if (focusableControls.length === 0) {
        event.preventDefault();
        return;
      }

      const firstControl = focusableControls[0];
      const lastControl = focusableControls[focusableControls.length - 1];
      const activeControl = document.activeElement;

      if (!focusableControls.includes(activeControl as HTMLElement)) {
        event.preventDefault();
        (event.shiftKey ? lastControl : firstControl).focus();
        return;
      }

      if (event.shiftKey && activeControl === firstControl) {
        event.preventDefault();
        lastControl.focus();
        return;
      }

      if (!event.shiftKey && activeControl === lastControl) {
        event.preventDefault();
        firstControl.focus();
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);

    return () => {
      window.removeEventListener("keydown", handleKeyDown, true);
      document.body.style.overflow = previousOverflow;
      if (previousBackgroundAriaHidden === null) {
        backgroundRoot?.removeAttribute("aria-hidden");
      } else if (previousBackgroundAriaHidden !== undefined) {
        backgroundRoot?.setAttribute("aria-hidden", previousBackgroundAriaHidden);
      }
    };
  }, [controller.isFullscreen]);

  function handleSummaryClick(event: ReactMouseEvent<HTMLElement>) {
    event.preventDefault();
    if (!canOpenUserPreview) {
      onToggle(false);
      return;
    }
    onToggle(!isPreviewPanelOpen);
  }

  function handleToggle(event: SyntheticEvent<HTMLDetailsElement>) {
    if (!canOpenUserPreview && event.currentTarget.open) {
      onToggle(false);
      return;
    }
    onToggle(event.currentTarget.open);
  }

  function handleKeyboardBoundary(event: PreviewSandboxKeyboardBoundaryEvent) {
    if (!controller.isFullscreen) {
      return;
    }

    if (event.key === "Escape") {
      onFullscreenToggleRef.current(false);
      return;
    }

    const fullscreenLayer = fullscreenLayerRef.current;
    if (!fullscreenLayer) {
      return;
    }

    const focusableControls = getFullscreenFocusableControls(fullscreenLayer);
    const runtimeIndex = focusableControls.findIndex((control) =>
      control.matches("iframe.previewRuntimeFrame")
    );
    if (runtimeIndex < 0) {
      return;
    }

    const target = event.shiftKey
      ? focusableControls[runtimeIndex - 1] ??
        focusableControls[focusableControls.length - 1]
      : focusableControls[runtimeIndex + 1] ?? focusableControls[0];
    target?.focus();
  }

  const layoutSize = resolvePreviewShelfLayoutSize(snapshot.preview);
  const panelHeight = resolvePreviewShelfPanelHeight(height, snapshot.preview);
  const canResizePreview =
    isPreviewPanelOpen && layoutSize === "visual" && !controller.isFullscreen;
  const panelStyle = controller.isFullscreen ? undefined : { height: panelHeight };

  if (!showDebugPanels && !canOpenUserPreview) {
    return (
      <section className="previewZone" aria-label="Preview workspace">
        <section
          aria-label="Preview fallback"
          className="previewShelf previewShelf--userFallback"
          data-runtime-state={snapshot.preview.state}
          data-user-mode="true"
        >
          <div className="previewUserFallbackCard">
            <div>
              <span>Preview</span>
              <strong>Preview unavailable</strong>
              <p>
                Choose a previewable artifact, or inspect Code and Saved while
                a runnable visual is prepared.
              </p>
            </div>
            <div className="previewUserFallbackActions">
              <button onClick={onOpenCode} type="button">
                Open Code
              </button>
              <button onClick={onOpenArtifacts} type="button">
                Open Saved
              </button>
            </div>
          </div>
        </section>
      </section>
    );
  }

  const previewShelf = (
    <details
      className="previewShelf"
      data-fullscreen={controller.isFullscreen ? "true" : "false"}
      data-layout-size={layoutSize}
      data-runtime-state={snapshot.preview.state}
      data-state={isPreviewPanelOpen ? "open" : "closed"}
      data-user-mode={showDebugPanels ? "false" : "true"}
      onToggle={handleToggle}
      open={isPreviewPanelOpen}
    >
      {!controller.isFullscreen ? (
        <summary aria-expanded={isPreviewPanelOpen} onClick={handleSummaryClick}>
          <span className="previewSummaryIcon" aria-hidden="true">
            <Play size={16} />
          </span>
          <div>
            <strong>{snapshot.preview.title}</strong>
            <span>
              {showDebugPanels ? snapshot.preview.artifactName : userArtifactLabel}
            </span>
          </div>
          <div className="previewSummaryMeta">
            <small data-state={snapshot.preview.state}>{snapshot.preview.status}</small>
            <span className="previewSummaryChevron" aria-hidden="true">
              <ChevronDown size={15} />
            </span>
          </div>
        </summary>
      ) : null}
      <div className="previewPanel" style={panelStyle}>
        {controller.isFullscreen ? (
          <div className="previewFullscreenClose" aria-label="Fullscreen preview controls">
            <button
              aria-label="Exit preview fullscreen"
              aria-pressed
              className="previewControlButton"
              data-action="exit"
              onClick={() => onFullscreenToggle(false)}
              ref={exitFullscreenButtonRef}
              title="Exit preview fullscreen"
              type="button"
            >
              <Minimize2 size={15} />
              <span className="previewControlLabel">Exit</span>
            </button>
          </div>
        ) : (
          <div className="previewToolbar">
            <div className="previewToolbarFocus" aria-label="Focused preview context">
              <span>{route.surfaceEyebrow}</span>
              <strong>{route.surfaceTitle}</strong>
              <small>
                {showDebugPanels
                  ? `${snapshot.preview.status} / ${route.rendererLabel}`
                  : snapshot.preview.status}
              </small>
            </div>
          </div>
        )}
        <div className="previewBody">
          {snapshot.preview.error ? (
            <SubsystemErrorCallout
              className="previewErrorCallout"
              error={snapshot.preview.error}
              title="Preview runtime failed"
            />
          ) : null}
          {!controller.isFullscreen ? (
            <div className="previewArtworkControls" aria-label="Preview controls">
              <button
                aria-label="Enter preview fullscreen"
                aria-pressed={false}
                className="previewControlButton"
                data-action="fullscreen"
                disabled={!controller.canFullscreen}
                onClick={() => onFullscreenToggle(true)}
                ref={enterFullscreenButtonRef}
                title="Enter preview fullscreen"
                type="button"
              >
                <Maximize2 size={15} />
                <span className="previewControlLabel">Fullscreen</span>
              </button>
              <button
                aria-label="Restart preview session"
                className="previewControlButton"
                data-action="restart"
                disabled={!controller.canRestart}
                onClick={onRestart}
                title="Restart preview session"
                type="button"
              >
                <RotateCcw size={15} />
                <span className="previewControlLabel">Restart</span>
              </button>
              <button
                aria-label="Clear preview state"
                className="previewControlButton"
                data-action="clear"
                disabled={!controller.canClear}
                onClick={onClear}
                title="Clear preview state"
                type="button"
              >
                <X size={15} />
                <span className="previewControlLabel">Clear</span>
              </button>
              <button
                aria-label="Reload preview state"
                className="previewControlButton"
                data-action="reload"
                disabled={!controller.canReload}
                onClick={onReload}
                title="Reload preview state"
                type="button"
              >
                <RefreshCw size={15} />
                <span className="previewControlLabel">Reload</span>
              </button>
            </div>
          ) : null}
          <PreviewRendererSurface
            captureHostKeyboard={controller.isFullscreen}
            chrome="immersive"
            onKeyboardBoundary={handleKeyboardBoundary}
            onOpenCode={onOpenCode}
            onReload={onReload}
            onRuntimeDiagnostics={onRuntimeDiagnostics}
            onRuntimeFrame={onRuntimeFrame}
            onRuntimeStatus={onRuntimeStatus}
            preview={snapshot.preview}
            route={route}
            runtimeSessionKey={runtimeSessionKey}
            runtimeSource={runtimeSource}
            showDiagnostics={showDebugPanels}
          />
        </div>
      </div>
      <div
        aria-label="Resize preview shelf"
        aria-disabled={!canResizePreview}
        aria-orientation="horizontal"
        aria-valuemax={
          layoutSize === "visual"
            ? workspaceLayoutBounds.maxPreviewHeight
            : workspaceLayoutBounds.compactPreviewHeight
        }
        aria-valuemin={workspaceLayoutBounds.minPreviewHeight}
        aria-valuenow={panelHeight}
        className="layoutResizeHandle previewResizeHandle"
        data-active={resizing}
        onKeyDown={canResizePreview ? onResizeKeyDown : undefined}
        onMouseDown={canResizePreview ? onResizeStart : undefined}
        role="separator"
        tabIndex={canResizePreview ? 0 : -1}
      >
        <span aria-hidden="true" />
      </div>
    </details>
  );

  return (
    <>
      <section className="previewZone" aria-label="Preview workspace">
        {controller.isFullscreen ? null : previewShelf}
      </section>
      {controller.isFullscreen && typeof document !== "undefined"
        ? createPortal(
            <div
              aria-label="Fullscreen artwork canvas"
              aria-modal="true"
              className="previewFullscreenLayer"
              ref={fullscreenLayerRef}
              role="dialog"
            >
              {previewShelf}
            </div>,
            document.body
          )
        : null}
    </>
  );
}

export function resolvePreviewShelfLayoutSize(
  preview: AssistantWorkspaceSnapshot["preview"]
) {
  return preview.active && preview.state === "ready" ? "visual" : "compact";
}

export function resolvePreviewShelfPanelHeight(
  height: number,
  preview: AssistantWorkspaceSnapshot["preview"]
) {
  if (resolvePreviewShelfLayoutSize(preview) === "visual") {
    return height;
  }

  return Math.min(height, workspaceLayoutBounds.compactPreviewHeight);
}

function getFullscreenFocusableControls(fullscreenLayer: HTMLElement) {
  return Array.from(
    fullscreenLayer.querySelectorAll<HTMLElement>(
      '[data-action="exit"], iframe.previewRuntimeFrame:not([aria-hidden="true"]), .previewRuntimeActionButton:not(:disabled)'
    )
  ).filter(
    (control) =>
      control.tabIndex >= 0 && isControlVisible(control, fullscreenLayer)
  );
}

function isControlVisible(control: HTMLElement, fullscreenLayer: HTMLElement) {
  let current: HTMLElement | null = control;

  while (current && current !== fullscreenLayer) {
    if (current.hidden || current.getAttribute("aria-hidden") === "true") {
      return false;
    }

    const style = window.getComputedStyle(current);
    if (style.display === "none" || style.visibility === "hidden") {
      return false;
    }

    current = current.parentElement;
  }

  return true;
}
