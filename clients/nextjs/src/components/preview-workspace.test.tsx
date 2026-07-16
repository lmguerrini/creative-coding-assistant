import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot
} from "@/lib/assistant-client";
import { buildPreviewControllerModel } from "@/lib/preview-controller";
import type { PreviewRuntimeSessionOverride } from "@/lib/preview-controller";
import { buildPreviewRendererRoute } from "@/lib/preview-renderers";
import { buildPreviewRuntimeSource } from "@/lib/preview-runtime-adapters";
import { PreviewWorkspace } from "./preview-workspace";

afterEach(() => {
  document.body.style.overflow = "";
});

describe("PreviewWorkspace", () => {
  it("keeps a concise live status in User Mode without diagnostics metadata", () => {
    render(
      <PreviewWorkspaceHarness
        showDebugPanels={false}
        snapshot={createReadyPreviewSnapshot()}
      />
    );
    const runtime = screen.getByRole("group", { name: "p5.js live runtime" });
    const frame = within(runtime).getByLabelText(
      "p5.js preview runtime frame"
    ) as HTMLIFrameElement;
    const liveStatus = runtime.querySelector('[aria-live="polite"]');

    expect(liveStatus).toHaveAttribute("aria-atomic", "true");
    dispatchRuntimeStatus(frame, {
      detail:
        "Rendering aurora-field.p5.js inside an isolated p5-compatible preview frame.",
      label: "p5 runtime running",
      state: "running"
    });

    expect(within(runtime).getByText("p5 runtime running")).toBeInTheDocument();
    expect(runtime).toHaveTextContent(
      "Rendering P5 sketch surface inside an isolated p5-compatible preview frame."
    );
    expect(runtime).not.toHaveTextContent("aurora-field.p5.js");
    expect(
      screen.queryByLabelText("Renderer health overlay")
    ).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Runtime notes")).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Preview runtime source")
    ).not.toBeInTheDocument();
  });

  it("keeps runtime recovery accessible inside the fullscreen focus boundary", () => {
    const onReload = vi.fn();
    const { container } = render(
      <PreviewWorkspaceHarness
        onReload={onReload}
        showDebugPanels
        snapshot={createReadyPreviewSnapshot()}
      />
    );
    const workstation = container.querySelector("main.workstation");
    const enterFullscreen = screen.getByRole("button", {
      name: "Enter preview fullscreen"
    });

    enterFullscreen.focus();
    fireEvent.click(enterFullscreen);

    const fullscreen = screen.getByRole("dialog", {
      name: "Fullscreen artwork canvas"
    });
    const exitFullscreen = within(fullscreen).getByRole("button", {
      name: "Exit preview fullscreen"
    });
    const runtime = within(fullscreen).getByRole("group", {
      name: "p5.js live runtime"
    });
    const frame = within(runtime).getByLabelText(
      "p5.js preview runtime frame"
    ) as HTMLIFrameElement;

    expect(fullscreen).toHaveAttribute("aria-modal", "true");
    expect(exitFullscreen).toHaveFocus();
    expect(workstation).toHaveAttribute("aria-hidden", "true");
    expect(workstation).not.toHaveAttribute("inert");
    expect(document.body.style.overflow).toBe("hidden");
    dispatchRuntimeStatus(frame, {
      detail: "The p5 preview runtime failed inside the fullscreen canvas.",
      error: {
        message: "The p5 preview runtime failed inside the fullscreen canvas.",
        type: "preview_runtime_failed"
      },
      label: "p5 runtime failed",
      state: "error"
    });
    const reloadRuntime = within(fullscreen).getByRole("button", {
      name: "Reload preview runtime"
    });
    const reviewCode = within(fullscreen).getByRole("button", {
      name: "Open generated code"
    });

    expect(runtime).toHaveAttribute("data-runtime-state", "error");
    expect(within(fullscreen).getByRole("alert")).toHaveTextContent(
      "Preview could not start"
    );
    expect(reviewCode).toBeVisible();
    expect(reloadRuntime).toBeVisible();

    frame.focus();
    dispatchRuntimeKeyboardBoundary(frame, { key: "Tab", shiftKey: false });
    expect(reviewCode).toHaveFocus();
    frame.focus();
    dispatchRuntimeKeyboardBoundary(frame, { key: "Tab", shiftKey: true });
    expect(exitFullscreen).toHaveFocus();

    fireEvent.click(reloadRuntime);
    expect(onReload).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).toBeInTheDocument();
    expect(within(fullscreen).getByRole("alert")).toHaveTextContent(
      "Preview could not start"
    );

    frame.focus();
    dispatchRuntimeKeyboardBoundary(frame, { key: "Escape", shiftKey: false });

    expect(
      screen.queryByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).not.toBeInTheDocument();
    expect(workstation).not.toHaveAttribute("aria-hidden");
    expect(workstation).not.toHaveAttribute("inert");
    expect(document.body.style.overflow).toBe("");
    expect(
      screen.getByRole("button", { name: "Enter preview fullscreen" })
    ).toHaveFocus();
  });

  it("keeps a refreshing User Mode runtime mounted so reload can settle", () => {
    const snapshot = createReadyPreviewSnapshot();
    const sessionOverride: PreviewRuntimeSessionOverride = {
      artifactId: snapshot.preview.sourceArtifactId,
      mode: "reloading",
      requestedAt: "2026-07-16T08:00:00.000Z"
    };

    render(
      <PreviewWorkspaceHarness
        sessionOverride={sessionOverride}
        showDebugPanels={false}
        snapshot={{
          ...snapshot,
          preview: {
            ...snapshot.preview,
            state: "generating",
            status: "Reloading"
          }
        }}
      />
    );

    expect(
      screen.getByRole("group", { name: "p5.js live runtime" })
    ).toBeInTheDocument();
    expect(screen.queryByText("Preview unavailable")).not.toBeInTheDocument();
  });

  it("ignores sandbox keyboard boundaries outside fullscreen", () => {
    const onFullscreenChange = vi.fn();
    render(
      <PreviewWorkspaceHarness
        onFullscreenChange={onFullscreenChange}
        showDebugPanels
        snapshot={createReadyPreviewSnapshot()}
      />
    );
    const frame = screen.getByLabelText(
      "p5.js preview runtime frame"
    ) as HTMLIFrameElement;

    frame.focus();
    dispatchRuntimeKeyboardBoundary(frame, { key: "Tab", shiftKey: false });
    dispatchRuntimeKeyboardBoundary(frame, { key: "Tab", shiftKey: true });
    dispatchRuntimeKeyboardBoundary(frame, { key: "Escape", shiftKey: false });

    expect(frame).toHaveFocus();
    expect(onFullscreenChange).not.toHaveBeenCalled();
    expect(
      screen.queryByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).not.toBeInTheDocument();
  });
});

function PreviewWorkspaceHarness({
  onFullscreenChange = () => undefined,
  onReload = () => undefined,
  sessionOverride = null,
  showDebugPanels,
  snapshot
}: {
  onFullscreenChange?: (isFullscreen: boolean) => void;
  onReload?: () => void;
  sessionOverride?: PreviewRuntimeSessionOverride | null;
  showDebugPanels: boolean;
  snapshot: AssistantWorkspaceSnapshot;
}) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const route = buildPreviewRendererRoute({
    artifacts: snapshot.artifacts,
    preview: snapshot.preview,
    previewArtifactId: snapshot.preview.sourceArtifactId
  });
  const controller = buildPreviewControllerModel({
    isFullscreen,
    preview: snapshot.preview,
    route,
    sessionOverride
  });
  const runtimeSource = buildPreviewRuntimeSource({
    code: snapshot.code,
    route
  });
  const handleFullscreenToggle = (nextFullscreen: boolean) => {
    onFullscreenChange(nextFullscreen);
    setIsFullscreen(nextFullscreen);
  };

  return (
    <main className="workstation">
      <PreviewWorkspace
        controller={controller}
        height={320}
        onClear={() => undefined}
        onFullscreenToggle={handleFullscreenToggle}
        onOpenArtifacts={() => undefined}
        onOpenCode={() => undefined}
        onReload={onReload}
        onResizeKeyDown={() => undefined}
        onResizeStart={() => undefined}
        onRestart={() => undefined}
        onRuntimeDiagnostics={() => undefined}
        onRuntimeFrame={() => undefined}
        onRuntimeStatus={() => undefined}
        onToggle={() => undefined}
        resizing={false}
        route={route}
        runtimeSessionKey="preview-workspace-test"
        runtimeSource={runtimeSource}
        showDebugPanels={showDebugPanels}
        snapshot={snapshot}
        userArtifactLabel="P5 Sketch"
      />
    </main>
  );
}

function createReadyPreviewSnapshot(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    preview: {
      ...snapshot.preview,
      active: true,
      available: true,
      collapsed: false,
      error: null,
      outputArtifactName: snapshot.preview.artifactName,
      state: "ready",
      status: "Preview open",
      title: "Preview available"
    }
  };
}

const sandboxTestHandshakeId =
  "preview-handshake-00000000000000000000000000000000";
const sandboxTestHandshakeRuntimeIds = new WeakMap<
  HTMLIFrameElement,
  string
>();

function ensureRuntimeHandshake(frame: HTMLIFrameElement) {
  const runtimeId = frame.dataset.runtimeId;
  const sandboxUrl = new URL(
    frame.getAttribute("src") ?? "",
    window.location.href
  );
  const challengeId = sandboxUrl.hash.slice(1);

  expect(runtimeId).toBeTruthy();
  expect(challengeId).toMatch(/^preview-challenge-[a-f0-9]{32}$/);
  expect(sandboxUrl.searchParams.get("challenge")).toBe(challengeId);
  if (sandboxTestHandshakeRuntimeIds.get(frame) === runtimeId) {
    return sandboxTestHandshakeId;
  }

  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          challengeId,
          handshakeId: sandboxTestHandshakeId,
          source: "cca-preview-runtime",
          type: "ready"
        },
        origin: "null",
        source: frame.contentWindow
      })
    );
  });
  sandboxTestHandshakeRuntimeIds.set(frame, runtimeId as string);
  return sandboxTestHandshakeId;
}

function dispatchRuntimeStatus(
  frame: HTMLIFrameElement,
  status: {
    detail: string;
    error?: { message: string; type: string };
    label: string;
    state: "running" | "error";
  }
) {
  const runtimeId = frame.dataset.runtimeId;

  expect(runtimeId).toBeTruthy();
  const handshakeId = ensureRuntimeHandshake(frame);
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          handshakeId,
          runtimeId,
          source: "cca-preview-runtime",
          status,
          type: "status"
        },
        origin: "null",
        source: frame.contentWindow
      })
    );
  });
}

function dispatchRuntimeKeyboardBoundary(
  frame: HTMLIFrameElement,
  event: { key: "Escape" | "Tab"; shiftKey: boolean }
) {
  const runtimeId = frame.dataset.runtimeId;

  expect(runtimeId).toBeTruthy();
  const handshakeId = ensureRuntimeHandshake(frame);
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          ...event,
          handshakeId,
          runtimeId,
          source: "cca-preview-runtime",
          type: "keyboard-boundary"
        },
        origin: "null",
        source: frame.contentWindow
      })
    );
  });
}
