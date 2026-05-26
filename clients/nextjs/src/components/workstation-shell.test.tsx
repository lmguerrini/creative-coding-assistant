import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkstationShell } from "./workstation-shell";
import {
  getLocalWorkspaceSnapshot,
  type AssistantWorkspaceSnapshot,
  type InspectorTabName
} from "@/lib/assistant-client";
import type { AssistantStreamEvent } from "@/lib/assistant-stream";
import {
  createWorkspaceSessionRecord,
  type WorkspacePersistenceClient,
  type WorkspacePersistenceLoadResult,
  type WorkspacePersistenceSaveResult
} from "@/lib/workspace-persistence";
import { createWorkstationError } from "@/lib/workstation-errors";

const originalClipboard = navigator.clipboard;
const originalCancelAnimationFrame = window.cancelAnimationFrame;
const originalRequestAnimationFrame = window.requestAnimationFrame;
const originalCreateObjectURL = URL.createObjectURL;
const originalRevokeObjectURL = URL.revokeObjectURL;

function snapshotWithActiveTab(
  activeTab: InspectorTabName
): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
      ...tab,
      active: tab.label === activeTab
    }))
  };
}

function snapshotWithP5Preview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "signal-orbit.p5.ts";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        title,
        summary: "Reactive p5 loop with createCanvas() and draw()."
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      sourceArtifactName: title,
      summary: "Runtime is generating the current sketch and preview context for the p5 surface.",
      target: "Browser sandbox"
    },
    code: {
      ...snapshot.code,
      title,
      language: "TypeScript + p5.js",
      excerpt: [
        "function setup() {",
        "  createCanvas(windowWidth, windowHeight);",
        "}",
        "function draw() {",
        "  background(8, 12, 18);",
        "  circle(width * 0.5, height * 0.5, 120);",
        "}"
      ]
    }
  };
}

function snapshotWithGlslPreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "chromatic-field.frag";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        language: "GLSL",
        summary: "Fragment shader with gl_FragColor and uniforms.",
        title
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      sourceArtifactName: title,
      summary: "Runtime is ready to mount a bounded GLSL fragment preview.",
      target: "Browser sandbox"
    },
    code: {
      ...snapshot.code,
      language: "GLSL",
      title,
      excerpt: [
        "void main() {",
        "  vec2 uv = gl_FragCoord.xy / u_resolution.xy;",
        "  float field = sin((uv.x + u_time) * 8.0);",
        "  gl_FragColor = vec4(uv.x, field * 0.5 + 0.5, uv.y, 1.0);",
        "}"
      ]
    }
  };
}

function snapshotWithThreePreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "projection-scene.three.ts";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        summary:
          "Three scene with WebGLRenderer, PerspectiveCamera, MeshStandardMaterial, lights, and camera motion.",
        title
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      sourceArtifactName: title,
      summary: "Runtime is ready to mount a bounded Three.js-style scene preview.",
      target: "Browser sandbox",
      targetId: "browser_sandbox"
    },
    code: {
      ...snapshot.code,
      excerpt: [
        "import * as THREE from 'three';",
        "const scene = new THREE.Scene();",
        "scene.background = new THREE.Color(0x05080b);",
        "const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);",
        "const renderer = new THREE.WebGLRenderer({ antialias: true });",
        "renderer.setClearColor(0x05080b);",
        "const geometry = new THREE.BoxGeometry(1.4, 1.4, 1.4);",
        "const material = new THREE.MeshStandardMaterial({ color: 0x4cd7c8, emissive: 0x7ca7ff });",
        "const mesh = new THREE.Mesh(geometry, material);",
        "mesh.rotation.y += 0.01;",
        "scene.add(mesh);"
      ],
      language: "TypeScript + Three.js",
      title
    }
  };
}

function snapshotWithEmptyRetrieval(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    retrieval: {
      ...snapshot.retrieval,
      state: "empty",
      status: "No matches",
      headline: "No retrieved context",
      detail: "No retrieval chunks were returned for this mock request.",
      query: "Find TouchDesigner references for this projection loop.",
      requestedDomains: ["touchdesigner"],
      warning: "No retrieved chunks for TouchDesigner.",
      sources: []
    }
  };
}

function installAnimationFrameMock() {
  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    value: vi.fn(() => 1)
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    value: vi.fn()
  });
}

function installCanvasContextMock(
  context: unknown,
  contextIds: string | string[] = "2d"
) {
  const supportedContextIds = new Set(
    Array.isArray(contextIds) ? contextIds : [contextIds]
  );

  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
    ((contextId: string) =>
      supportedContextIds.has(contextId) ? context : null) as typeof HTMLCanvasElement.prototype.getContext
  );
}

function createMockWebGlContext(): WebGLRenderingContext {
  const buffer = {};
  const program = {};
  const shader = {};

  return {
    ARRAY_BUFFER: 0x8892,
    COLOR_BUFFER_BIT: 0x4000,
    COMPILE_STATUS: 0x8b81,
    CULL_FACE: 0x0b44,
    DEPTH_BUFFER_BIT: 0x0100,
    DEPTH_TEST: 0x0b71,
    FLOAT: 0x1406,
    FRAGMENT_SHADER: 0x8b30,
    LINK_STATUS: 0x8b82,
    STATIC_DRAW: 0x88e4,
    TRIANGLES: 0x0004,
    VERTEX_SHADER: 0x8b31,
    attachShader: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    clear: vi.fn(),
    clearColor: vi.fn(),
    compileShader: vi.fn(),
    createBuffer: vi.fn(() => buffer),
    createProgram: vi.fn(() => program),
    createShader: vi.fn(() => shader),
    deleteBuffer: vi.fn(),
    deleteProgram: vi.fn(),
    deleteShader: vi.fn(),
    drawArrays: vi.fn(),
    drawingBufferHeight: 360,
    drawingBufferWidth: 640,
    enable: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    getAttribLocation: vi.fn((_, name: string) =>
      name === "a_normal" ? 1 : 0
    ),
    getProgramInfoLog: vi.fn(() => null),
    getProgramParameter: vi.fn(() => true),
    getShaderInfoLog: vi.fn(() => null),
    getShaderParameter: vi.fn(() => true),
    getUniformLocation: vi.fn(() => ({})),
    linkProgram: vi.fn(),
    shaderSource: vi.fn(),
    uniform1f: vi.fn(),
    uniform3f: vi.fn(),
    useProgram: vi.fn(),
    vertexAttribPointer: vi.fn(),
    viewport: vi.fn()
  } as unknown as WebGLRenderingContext;
}

async function* streamEvents(
  events: AssistantStreamEvent[]
): AsyncGenerator<AssistantStreamEvent> {
  for (const event of events) {
    yield event;
  }
}

async function* failingStream(): AsyncGenerator<AssistantStreamEvent> {
  throw new Error("offline");
}

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });

  return { promise, resolve };
}

function runtimeWorkflowEvent({
  answer,
  at,
  code,
  completedSteps = [],
  currentStep,
  eventType,
  message,
  phase = "running",
  refinementCount = 0,
  reviewOutcome = null,
  sequence,
  skippedSteps = [],
  status = "running",
  step,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  completedSteps?: string[];
  currentStep: string | null;
  eventType: AssistantStreamEvent["event_type"];
  message?: string;
  phase?: string;
  refinementCount?: number;
  reviewOutcome?: string | null;
  sequence: number;
  skippedSteps?: string[];
  status?: string;
  step: string | null;
  text?: string;
}): AssistantStreamEvent {
  return {
    event_type: eventType,
    sequence,
    payload: {
      ...(answer ? { answer } : {}),
      ...(code ? { code } : {}),
      ...(message ? { message } : {}),
      ...(text ? { text } : {}),
      emitted_at: at,
      workflow: {
        step,
        phase,
        status,
        current_step: currentStep,
        completed_steps: completedSteps,
        skipped_steps: skippedSteps,
        refinement_count: refinementCount,
        review_outcome: reviewOutcome,
        review_reasons: []
      }
    }
  };
}

function createNoopPersistenceClient(): WorkspacePersistenceClient {
  return {
    load: vi.fn(
      () => new Promise<WorkspacePersistenceLoadResult>(() => undefined)
    ),
    save: vi.fn(async () => ({ error: null, target: "local" as const }))
  };
}

function renderShell(
  snapshot: AssistantWorkspaceSnapshot = getLocalWorkspaceSnapshot(),
  props: Partial<ComponentProps<typeof WorkstationShell>> = {}
) {
  return render(
    <WorkstationShell
      snapshot={snapshot}
      persistenceClient={createNoopPersistenceClient()}
      {...props}
    />
  );
}

describe("WorkstationShell", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: originalClipboard
    });
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      value: originalRequestAnimationFrame
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      value: originalCancelAnimationFrame
    });
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: originalCreateObjectURL
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: originalRevokeObjectURL
    });
  });

  it("renders the three-zone creative workspace shell", () => {
    renderShell();

    expect(screen.getByText("Creative Coding Assistant")).toBeVisible();
    expect(screen.getByRole("region", { name: "Creative session" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Preview workspace" })).toBeVisible();
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toBeVisible();
    expect(screen.getByRole("tablist", { name: "Inspector tabs" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Focus mode" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Workspace density" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Command menu" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Theme" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
  });

  it("opens the top-right utility panels one at a time", () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Command menu" }));
    expect(screen.getByRole("dialog", { name: "Quick actions" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Theme" }));
    expect(screen.queryByRole("dialog", { name: "Quick actions" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Theme presets" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.queryByRole("dialog", { name: "Theme presets" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Workspace settings" })).toBeVisible();
  });

  it("collapses the inspector into a compact rail and expands it again", () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Collapse inspector" }));

    expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );
    expect(screen.queryByRole("tablist", { name: "Inspector tabs" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand inspector" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));

    expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "open"
    );
    expect(screen.getByRole("tablist", { name: "Inspector tabs" })).toBeVisible();
    expect(screen.getByRole("tabpanel", { name: "Overview inspector" })).toBeVisible();
  });

  it("supports focus mode and density toggles without changing the mock data flow", () => {
    const { container } = renderShell();
    const workstation = container.querySelector(".workstation");

    expect(workstation).toHaveAttribute("data-density", "cozy");

    fireEvent.click(screen.getByRole("button", { name: "Workspace density" }));
    expect(workstation).toHaveAttribute("data-density", "compact");

    fireEvent.click(screen.getByRole("button", { name: "Focus mode" }));

    expect(screen.getByRole("button", { name: "Focus mode" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(workstation).toHaveAttribute("data-focus-mode", "true");
    expect(screen.queryByRole("complementary", { name: "Right inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Preview workspace" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Focus mode" }));

    expect(workstation).toHaveAttribute("data-focus-mode", "false");
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Preview workspace" })).toBeVisible();
  });

  it("clears the workspace session only after operator approval", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockImplementation(() => true);
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");
    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    fireEvent.click(screen.getByRole("button", { name: "Command menu" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear workspace session" }));

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Clear workspace session"
    );
    expect(screen.getByRole("button", { name: "Keep session" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Keep session" }));
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent("Rejected");

    fireEvent.click(screen.getByRole("button", { name: "Command menu" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear workspace session" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear workspace" }));
      await Promise.resolve();
    });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(preview.querySelector("details")).not.toHaveAttribute("open");

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    await waitFor(() => {
      const events = screen.getByRole("group", { name: "Workflow event trace" });
      expect(within(events).getByText("Workspace Clear Completed")).toBeVisible();
    });
  });

  it("defaults to a single Overview inspector panel", () => {
    renderShell();

    for (const tab of ["Overview", "Code", "Workflow", "Artifacts", "Retrieval"]) {
      expect(screen.getByRole("tab", { name: tab })).toBeVisible();
    }

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Overview inspector" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Workflow summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Artifacts summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Preview summary" })).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Image references summary" })
    ).toBeVisible();
    expect(screen.getByRole("group", { name: "Retrieval summary" })).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 11 workflow nodes reached");
    expect(screen.queryByRole("tabpanel", { name: "Code inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Preview" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("switches inspector tabs without stacking panels", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "data-active",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Code inspector" })).toBeVisible();
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    expect(screen.getByRole("tab", { name: "Retrieval" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Retrieval inspector" })).toBeVisible();
  });

  it("renders retrieval source cards with provenance, quality, and chunk previews", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval status" })
    ).toBeVisible();
    expect(within(retrievalPanel).getByText("Official knowledge base")).toBeVisible();
    expect(within(retrievalPanel).getByText("WebGPU API")).toBeVisible();
    expect(
      within(retrievalPanel).getByText(
        "OpenGL Shading Language 4.60 Specification"
      )
    ).toBeVisible();
    expect(within(retrievalPanel).getAllByText("Best match").length).toBeGreaterThan(0);
    expect(within(retrievalPanel).getByText("High relevance")).toBeVisible();
    expect(within(retrievalPanel).getByText("Review soon")).toBeVisible();
    expect(
      within(retrievalPanel).getByText(
        /Stable WebGPU particle field for a projection wall/
      )
    ).toBeVisible();
  });

  it("renders retrieval empty states without stale source cards", () => {
    renderShell(snapshotWithEmptyRetrieval());

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval empty state" })
    ).toBeVisible();
    expect(within(retrievalPanel).getByText("No matches")).toBeVisible();
    expect(
      within(retrievalPanel).getByText("No retrieved chunks for TouchDesigner.")
    ).toBeVisible();
    expect(
      within(retrievalPanel).queryByRole("link", { name: "Open source reference" })
    ).not.toBeInTheDocument();
  });

  it("uses the command menu to open focused inspector views", () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Command menu" }));
    fireEvent.click(screen.getByRole("button", { name: /Code inspector/ }));

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tabpanel", { name: "Code inspector" })).toBeVisible();
    expect(screen.queryByRole("dialog", { name: "Quick actions" })).not.toBeInTheDocument();
  });

  it("streams backend events into the conversation and workflow state", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "status",
          sequence: 0,
          payload: { code: "request_received", message: "Request accepted." }
        },
        {
          event_type: "status",
          sequence: 1,
          payload: { code: "route_selected", message: "Route selected." }
        },
        {
          event_type: "token_delta",
          sequence: 2,
          payload: { text: "Streaming " }
        },
        {
          event_type: "token_delta",
          sequence: 3,
          payload: { text: "answer." }
        },
        {
          event_type: "preview_artifact",
          sequence: 4,
          payload: {
            artifact_id: "source-sketch",
            status: "skipped",
            result: {
              preview_artifact_id: "preview-manifest",
              summary:
                "Preview pipeline foundation only; renderer execution is deferred.",
              request: {
                target: "browser_sandbox"
              },
              provenance: {
                renderer_id: "preview.noop"
              }
            }
          }
        },
        {
          event_type: "final",
          sequence: 5,
          payload: { answer: "Final backend answer." }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.getByText("Type a prompt to begin")).toBeVisible();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.getByText("Ready to generate")).toBeVisible();

    fireEvent.click(sendButton);

    expect(promptInput).toHaveValue("");
    expect(await screen.findByText("Final backend answer.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "local-nextjs-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: "Make the low-frequency motion calmer."
      })
    );
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Finalization"
    );
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "11");

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(
      within(preview).getByText("preview-request.json", { selector: "summary span" })
    ).toBeVisible();
    expect(
      within(preview).getByText(
        "Preview pipeline foundation only; renderer execution is deferred."
      )
    ).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
  });

  it("uploads image references and sends them with the next backend request", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: { answer: "Image-aware response." }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    const uploadInput = screen.getByLabelText("Upload image reference");
    const imageFile = new File(["palette-bytes"], "palette.png", {
      type: "image/png"
    });

    await act(async () => {
      fireEvent.change(uploadInput, {
        target: { files: [imageFile] }
      });
      await Promise.resolve();
    });

    const imageShelf = await screen.findByRole("region", {
      name: "Image references"
    });
    expect(within(imageShelf).getByText("1 image reference")).toBeVisible();
    expect(within(imageShelf).getByText("palette.png")).toBeVisible();
    expect(
      within(screen.getByRole("group", { name: "Image references summary" })).getByText(
        "1"
      )
    ).toBeVisible();

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Use this palette reference." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Image-aware response.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        attachments: [
          expect.objectContaining({
            type: "image",
            id: expect.stringContaining("palette-png"),
            name: "palette.png",
            mimeType: "image/png",
            sizeBytes: imageFile.size,
            dataUrl: expect.stringMatching(/^data:image\/png;base64,/)
          })
        ],
        query: "Use this palette reference."
      })
    );

    fireEvent.click(
      within(imageShelf).getByRole("button", {
        name: "Remove image reference palette.png"
      })
    );

    expect(screen.queryByText("palette.png")).not.toBeInTheDocument();
  });

  it("shows a graceful image upload error for unsupported files", async () => {
    renderShell();

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Upload image reference"), {
        target: {
          files: [new File(["notes"], "notes.txt", { type: "text/plain" })]
        }
      });
      await Promise.resolve();
    });

    const imageShelf = await screen.findByRole("region", {
      name: "Image references"
    });

    expect(within(imageShelf).getAllByText("Image upload issue")).toHaveLength(2);
    expect(
      within(imageShelf).getAllByText(
        "Only PNG, JPEG, WebP, or GIF image references can be attached."
      )
    ).toHaveLength(2);
    expect(
      within(imageShelf).getByRole("button", { name: "Dismiss image upload issue" })
    ).toBeVisible();
  });

  it("hydrates the retrieval inspector from streamed retrieval events", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "retrieval",
          sequence: 0,
          payload: {
            code: "retrieval_requested",
            emitted_at: "2026-05-23T10:21:00Z",
            request: {
              query: "Create a p5.js sketch with low-frequency motion.",
              limit: 5,
              filters: {
                domains: ["p5_js"]
              }
            }
          }
        },
        {
          event_type: "retrieval",
          sequence: 1,
          payload: {
            code: "retrieval_completed",
            emitted_at: "2026-05-23T10:21:01Z",
            context: {
              source: "official_kb",
              request: {
                query: "Create a p5.js sketch with low-frequency motion.",
                limit: 5,
                filters: {
                  domains: ["p5_js"]
                }
              },
              chunks: [
                {
                  source_id: "p5_reference",
                  domain: "p5_js",
                  source_type: "api_reference",
                  publisher: "p5.js",
                  registry_title: "p5.js Reference",
                  document_title: "createCanvas",
                  source_url: "https://p5js.org/reference/p5/createCanvas/",
                  resolved_url: "https://p5js.org/reference/p5/createCanvas/",
                  chunk_index: 0,
                  excerpt:
                    "createCanvas sets the main drawing surface and should be called once in setup.",
                  score: 0.88
                }
              ]
            }
          }
        },
        {
          event_type: "final",
          sequence: 2,
          payload: {
            answer: "Use p5.js createCanvas in setup and keep the motion broad."
          }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Create a p5.js sketch with low-frequency motion." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText(/Use p5.js createCanvas/)).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(retrievalPanel).getByText(
        "Create a p5.js sketch with low-frequency motion."
      )
    ).toBeVisible();
    expect(within(retrievalPanel).getByText("createCanvas")).toBeVisible();
    expect(within(retrievalPanel).getAllByText("p5.js").length).toBeGreaterThan(0);
    expect(within(retrievalPanel).getByText("88% match")).toBeVisible();
  });

  it("shows structured retrieval failures in the retrieval inspector", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "retrieval",
          sequence: 0,
          payload: {
            code: "retrieval_completed",
            emitted_at: "2026-05-23T10:21:01Z",
            error: {
              type: "retrieval_gateway_failed",
              message: "Retrieval references are unavailable for this request.",
              recoverable: true,
              retry_label: "Retry retrieval",
              subsystem: "retrieval_gateway"
            },
            context: {
              source: "official_kb",
              request: {
                query: "Find TouchDesigner references for this projection loop.",
                limit: 5,
                filters: {
                  domains: ["touchdesigner"]
                }
              },
              chunks: []
            }
          }
        },
        {
          event_type: "final",
          sequence: 1,
          payload: {
            answer: "Continuing without retrieval references."
          }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Find TouchDesigner references for this projection loop." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText("Continuing without retrieval references.")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(retrievalPanel).getAllByText("Retrieval failed").length
    ).toBeGreaterThan(0);
    expect(
      within(retrievalPanel).getAllByText(
        "Retrieval references are unavailable for this request."
      ).length
    ).toBeGreaterThan(0);
    expect(within(retrievalPanel).getByText("Retry retrieval")).toBeVisible();
  });

  it("shows connecting and live generation states during a streamed response", async () => {
    const beforeTokens = createDeferred<void>();
    const beforeFinal = createDeferred<void>();
    const backendStream = vi.fn(async function* () {
      yield {
        event_type: "status",
        sequence: 0,
        payload: { code: "request_received", message: "Request accepted." }
      } satisfies AssistantStreamEvent;
      await beforeTokens.promise;
      yield {
        event_type: "token_delta",
        sequence: 1,
        payload: { text: "Live draft" }
      } satisfies AssistantStreamEvent;
      await beforeFinal.promise;
      yield {
        event_type: "final",
        sequence: 2,
        payload: { answer: "Live draft completed." }
      } satisfies AssistantStreamEvent;
    });

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a calmer draft." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Opening the live response...")).toBeVisible();
    expect(screen.getByText("Request accepted.")).toBeVisible();
    expect(screen.getByText("Opening live response")).toBeVisible();
    expect(screen.getByRole("log", { name: "Conversation" })).toHaveAttribute(
      "aria-busy",
      "true"
    );

    beforeTokens.resolve();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText("Live draft")).toBeVisible();
    expect(screen.getByText("Generating response")).toBeVisible();
    expect(screen.getAllByText("Live").length).toBeGreaterThan(0);

    beforeFinal.resolve();

    expect(await screen.findByText("Live draft completed.")).toBeVisible();
    expect(screen.getByRole("log", { name: "Conversation" })).toHaveAttribute(
      "aria-busy",
      "false"
    );
  });

  it("applies theme and settings preferences and persists them", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });

    expect(await screen.findByText("Session saved")).toBeVisible();
    vi.mocked(persistenceClient.save).mockClear();

    fireEvent.click(screen.getByRole("button", { name: "Theme" }));
    fireEvent.click(screen.getByRole("button", { name: "Use Matrix theme" }));

    expect(document.documentElement).toHaveAttribute("data-cca-theme", "matrix");

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Preview auto-open" }));
    fireEvent.click(screen.getByRole("button", { name: "Advanced traces" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          preferences: {
            theme: "matrix",
            autoOpenPreview: false,
            showDebugPanels: false
          }
        })
      );
    });
  });

  it("shows a persistence issue when saves fall back to the local session copy", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({
        error: createWorkstationError({
          type: "session_save_unavailable",
          category: "persistence",
          subsystem: "workspace_session_store",
          userMessage: "Remote session save timed out or could not be reached.",
          recoverable: true,
          suggestedAction:
            "Keep editing locally and retry the session save after the backend recovers.",
          retryLabel: "Retry save"
        }),
        target: "local" as const
      }))
    };

    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });

    expect(await screen.findByText("Saved locally")).toBeVisible();
    expect(screen.getByText("Session persistence issue")).toBeVisible();
    expect(
      screen.getByText("Remote session save timed out or could not be reached.")
    ).toBeVisible();
  });

  it("hides workflow traces and keeps preview closed when auto-open is disabled", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "status",
          sequence: 0,
          payload: { code: "request_received", message: "Request accepted." }
        },
        {
          event_type: "preview_artifact",
          sequence: 1,
          payload: {
            artifact_id: "source-sketch",
            status: "skipped",
            result: {
              preview_artifact_id: "preview-manifest",
              summary:
                "Preview pipeline foundation only; renderer execution is deferred.",
              request: {
                target: "browser_sandbox"
              },
              provenance: {
                renderer_id: "preview.noop"
              }
            }
          }
        },
        {
          event_type: "final",
          sequence: 2,
          payload: { answer: "Preview left closed." }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Advanced traces" }));
    fireEvent.click(screen.getByRole("button", { name: "Preview auto-open" }));

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));

    expect(
      screen.getByRole("group", { name: "Workflow traces hidden" })
    ).toBeVisible();
    expect(
      screen.queryByRole("group", { name: "Workflow transition trace" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Prepare a preview artifact without opening it." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Preview left closed.")).toBeVisible();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(within(preview).getByText("Deferred renderer")).toBeVisible();
    expect(
      within(preview).getByText("preview-request.json", { selector: "summary span" })
    ).toBeVisible();
    expect(preview.querySelector("details")).not.toHaveAttribute("open");
  });

  it("falls back to the local mock path when the backend stream is unavailable", async () => {
    vi.useFakeTimers();
    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: failingStream });

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.getByText("Type a prompt to begin")).toBeVisible();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.getByText("Ready to generate")).toBeVisible();

    fireEvent.click(sendButton);

    await act(async () => {
      await Promise.resolve();
    });

    expect(promptInput).toHaveValue("");
    expect(screen.getByText(/Backend stream unavailable/)).toBeVisible();
    const userMessage = screen
      .getByText("Make the low-frequency motion calmer.")
      .closest("article");
    const assistantMessage = screen
      .getByText(/Mock orchestration pass started/)
      .closest("article");

    expect(userMessage).toHaveAttribute("data-fresh", "true");
    expect(assistantMessage).toHaveAttribute("data-fresh", "true");
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Intake");
    expect(screen.getByText("Stream interrupted")).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "1");

    act(() => {
      vi.advanceTimersByTime(850);
    });

    expect(screen.getByLabelText("Current session")).toHaveTextContent("Routing");
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "2");
  });

  it("surfaces backend error events without losing the user message", async () => {
    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: () =>
          streamEvents([
            {
              event_type: "status",
              sequence: 0,
              payload: { code: "request_received", message: "Request accepted." }
            },
            {
              event_type: "error",
              sequence: 1,
              payload: {
                code: "provider_unavailable",
                message: "Provider unavailable."
              }
            }
          ])
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a reactive sketch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(screen.getByText("Generate a reactive sketch.")).toBeVisible();
    expect(
      await screen.findByText(
        "Backend stream error: The model provider is unavailable for this live response."
      )
    ).toBeVisible();
    expect(screen.getByText("Live stream interrupted")).toBeVisible();
    expect(screen.getByText("Stream interrupted")).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    expect(await screen.findByText("Runtime issue")).toBeVisible();
  });

  it("keeps preview available, on demand, and collapsible in the main column", () => {
    renderShell();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const details = preview.querySelector("details");
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(within(preview).getByText("Preview available")).toBeVisible();
    expect(
      within(preview).getByText("webgpu-particle-field.ts", { selector: "summary span" })
    ).toBeVisible();
    expect(
      within(preview).getByText("Generating", { selector: "summary small" })
    ).toBeVisible();
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveAttribute("data-state", "closed");
    expect(screen.queryByRole("tabpanel", { name: "Preview inspector" })).not.toBeInTheDocument();

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    expect(details).toHaveAttribute("open");
    expect(details).toHaveAttribute("data-state", "open");
    expect(summary).toHaveAttribute("aria-expanded", "true");
    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(surface).toBeVisible();
    expect(within(surface).getByText("Browser route without renderer match")).toBeVisible();
    expect(within(surface).getByText("Unsupported")).toBeVisible();
  });

  it("opens the preview shelf in fullscreen without losing the current context", () => {
    renderShell();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const details = preview.querySelector("details");
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);
    fireEvent.click(
      within(preview).getByRole("button", { name: "Enter preview fullscreen" })
    );

    expect(details).toHaveAttribute("data-fullscreen", "true");
    expect(
      within(preview).getByRole("button", { name: "Exit preview fullscreen" })
    ).toBeVisible();

    fireEvent.click(
      within(preview).getByRole("button", { name: "Exit preview fullscreen" })
    );

    expect(details).toHaveAttribute("data-fullscreen", "false");
    expect(
      within(preview).getByText("webgpu-particle-field.ts", { selector: "summary span" })
    ).toBeVisible();
  });

  it("routes destructive preview runtime actions through an operator checkpoint", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockImplementation(() => true);
    renderShell();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);
    fireEvent.click(
      within(preview).getByRole("button", { name: "Clear preview state" })
    );

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Clear preview runtime"
    );
    expect(
      within(preview).queryByText("Cleared", { selector: "summary small" })
    ).not.toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear runtime" }));
      await Promise.resolve();
    });

    expect(
      within(preview).getByText("Cleared", { selector: "summary small" })
    ).toBeVisible();
    expect(
      within(preview).getByText(
        "Preview state cleared for webgpu-particle-field.ts. Reload or reset the session to restore the latest runtime context."
      )
    ).toBeVisible();

    fireEvent.click(
      within(preview).getByRole("button", { name: "Reload preview state" })
    );

    expect(
      within(preview).queryByText(
        "Preview state cleared for webgpu-particle-field.ts. Reload or reset the session to restore the latest runtime context."
      )
    ).not.toBeInTheDocument();
    expect(
      within(preview).getByText("Generating", { selector: "summary small" })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const notesArtifact = screen.getByLabelText("projection-notes.md artifact");
    fireEvent.click(
      within(notesArtifact).getByRole("button", {
        name: "Open in Code projection-notes.md"
      })
    );

    expect(
      within(preview).getByText("projection-notes.md", { selector: "summary span" })
    ).toBeVisible();

    fireEvent.click(
      within(preview).getByRole("button", { name: "Reset preview session" })
    );

    await waitFor(() => {
      expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
        "Reset preview runtime"
      );
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Reset runtime" }));
      await Promise.resolve();
    });

    expect(
      within(preview).getByText("webgpu-particle-field.ts", { selector: "summary span" })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(within(events).getByText("Preview Runtime Clear Completed")).toBeVisible();
    expect(within(events).getByText("Preview Runtime Reset Approval Requested")).toBeVisible();
    expect(within(events).getByText("Preview Runtime Reset Completed")).toBeVisible();
  });

  it("updates the preview context when the active artifact is not previewable", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const notesArtifact = screen.getByLabelText("projection-notes.md artifact");
    fireEvent.click(
      within(notesArtifact).getByRole("button", {
        name: "Open in Code projection-notes.md"
      })
    );

    const preview = screen.getByRole("region", { name: "Preview workspace" });

    expect(
      within(preview).getByText("projection-notes.md", { selector: "summary span" })
    ).toBeVisible();
    expect(
      within(preview).getByText("Unavailable", { selector: "summary small" })
    ).toBeVisible();
    expect(preview.querySelector("details")).not.toHaveAttribute("open");
  });

  it("opens artifacts, highlights the active artifact, and targets preview actions", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactList = screen.getByRole("tabpanel", { name: "Artifacts inspector" });
    const sourceArtifact = within(artifactList).getByLabelText(
      "webgpu-particle-field.ts artifact"
    );
    fireEvent.click(
      within(sourceArtifact).getByRole("button", {
        name: "Open in Code webgpu-particle-field.ts"
      })
    );

    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(codePanel).toHaveAttribute(
      "data-opened-artifact",
      "webgpu-particle-field.ts"
    );

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const notesArtifact = screen.getByLabelText("projection-notes.md artifact");
    fireEvent.click(
      within(notesArtifact).getByRole("button", {
        name: "Open in Code projection-notes.md"
      })
    );

    expect(screen.getByRole("tabpanel", { name: "Code inspector" })).toHaveAttribute(
      "data-opened-artifact",
      "projection-notes.md"
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "projection-notes.md"
    );

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const selectedArtifact = screen.getByLabelText("projection-notes.md artifact");

    expect(selectedArtifact).toHaveAttribute("data-active", "true");
    expect(within(selectedArtifact).getByText("Selected")).toBeVisible();
    const previewArtifact = screen.getByLabelText("preview-request.json artifact");

    fireEvent.click(
      within(previewArtifact).getByRole("button", {
        name: "Open Preview preview-request.json"
      })
    );

    const preview = screen.getByRole("region", { name: "Preview workspace" });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(
      within(preview).getByText("preview-request.json", { selector: "summary span" })
    ).toBeVisible();
    expect(within(preview).getByText("Preview open")).toBeVisible();
    expect(
      within(
        within(preview).getByRole("group", { name: "Preview renderer surface" })
      ).getByText("JSON panel surface")
    ).toBeVisible();
    expect(
      within(
        within(preview).getByRole("group", { name: "Preview renderer surface" })
      ).getByText("Preview manifest panel")
    ).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
    expect(preview.querySelector("details")).toHaveAttribute("data-state", "open");
  });

  it("mounts supported p5 artifacts into a live renderer canvas", async () => {
    installAnimationFrameMock();
    installCanvasContextMock({});
    renderShell(snapshotWithP5Preview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(surface).getByText("P5 sketch surface")).toBeVisible();
    expect(within(surface).getByText("p5.js")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "p5.js live runtime" })
    ).toBeVisible();
    expect(
      within(surface).getByLabelText("p5.js live runtime canvas")
    ).toBeVisible();
    expect(await within(surface).findByText("p5 runtime running")).toBeVisible();
  });

  it("mounts supported Three.js artifacts into a controlled 3D runtime", async () => {
    installAnimationFrameMock();
    installCanvasContextMock(createMockWebGlContext(), "webgl");
    renderShell(snapshotWithThreePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(surface).getByText("Three scene surface")).toBeVisible();
    expect(within(surface).getByText("Three.js")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Three.js live runtime" })
    ).toBeVisible();
    expect(
      within(surface).getByLabelText("Three.js live runtime canvas")
    ).toBeVisible();
    expect(
      await within(surface).findByText("Three.js runtime running")
    ).toBeVisible();
  });

  it("shows a stable Three.js runtime error when WebGL cannot mount", async () => {
    installCanvasContextMock(null, "webgl");
    renderShell(snapshotWithThreePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(surface).getByText("Three scene surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Three.js live runtime" })
    ).toBeVisible();
    expect(
      await within(surface).findByText("Three.js runtime unavailable")
    ).toBeVisible();
    expect(within(surface).getByText("Renderer runtime failed")).toBeVisible();
  });

  it("shows a stable GLSL runtime error when WebGL cannot mount", async () => {
    installCanvasContextMock(null);
    renderShell(snapshotWithGlslPreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(surface).getByText("Shader surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "GLSL live runtime" })
    ).toBeVisible();
    expect(await within(surface).findByText("GLSL runtime unavailable")).toBeVisible();
    expect(within(surface).getByText("Renderer runtime failed")).toBeVisible();
  });

  it("uses the full inspector panel for code when Code is active", () => {
    renderShell(snapshotWithActiveTab("Code"));

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(codePanel).toBeVisible();
    expect(within(codePanel).getByText("TypeScript + WGSL")).toBeVisible();
    expect(within(codePanel).getByText("Source code")).toBeVisible();
    expect(within(codePanel).getByText("7 lines")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "webgpu-particle-field.ts content"
      })
    ).toHaveTextContent("renderer.present({ palette, projectionScale });");
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();
  });

  it("shows focused artifact metadata and actions in the artifacts inspector", () => {
    renderShell(snapshotWithActiveTab("Artifacts"));

    const details = screen.getByRole("group", { name: "Active artifact details" });

    expect(within(details).getByText("Selected artifact")).toBeVisible();
    expect(within(details).getByText("webgpu-particle-field.ts")).toBeVisible();
    expect(within(details).getByText("Source code")).toBeVisible();
    expect(within(details).getByText("TypeScript + WGSL")).toBeVisible();
    expect(
      within(details).getByRole("button", {
        name: "Open in Code webgpu-particle-field.ts"
      })
    ).toBeVisible();
    expect(
      within(details).getByRole("button", {
        name: "Download File webgpu-particle-field.ts"
      })
    ).toBeVisible();
  });

  it("requires approval before downloading an artifact and records the action in workflow traces", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockImplementation(() => true);
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:artifact")
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    });

    renderShell(snapshotWithActiveTab("Artifacts"));
    const details = screen.getByRole("group", { name: "Active artifact details" });

    fireEvent.click(
      within(details).getByRole("button", {
        name: "Download File webgpu-particle-field.ts"
      })
    );

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(anchorClick).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Download artifact"
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Download file" }));
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(anchorClick).toHaveBeenCalledTimes(1);
    });
    expect(
      screen.getByText("webgpu-particle-field.ts downloaded.")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(within(events).getByText("Artifact Download Approval Requested")).toBeVisible();
    expect(within(events).getByText("Artifact Download Completed")).toBeVisible();
  });

  it("shows artifact transfer failures in the artifacts and workflow surfaces", async () => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => {
        throw new Error("disk blocked");
      })
    });

    renderShell(snapshotWithActiveTab("Artifacts"));
    const details = screen.getByRole("group", { name: "Active artifact details" });

    fireEvent.click(
      within(details).getByRole("button", {
        name: "Download File webgpu-particle-field.ts"
      })
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Download file" }));
      await Promise.resolve();
    });

    expect(screen.getByText("Artifact transfer failed")).toBeVisible();
    expect(
      screen.getAllByText(
        "The workspace could not download webgpu-particle-field.ts."
      ).length
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    expect(await screen.findByText("Runtime issue")).toBeVisible();
  });

  it("shows copy feedback in the code inspector", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    });

    renderShell(snapshotWithActiveTab("Code"));

    fireEvent.click(
      screen.getByRole("button", { name: "Copy webgpu-particle-field.ts" })
    );

    await waitFor(() => {
      expect(
        screen.getByText("webgpu-particle-field.ts copied to clipboard.")
      ).toBeVisible();
    });
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("renderer.present({ palette, projectionScale });")
    );
    expect(screen.getByText("Copied")).toBeVisible();
  });

  it("shows an elegant workflow inspector with live graph states", () => {
    renderShell(snapshotWithActiveTab("Workflow"));

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow inspector" });
    expect(workflowPanel).toBeVisible();
    const graph = screen.getByRole("group", {
      name: "LangGraph workflow visualization"
    });

    expect(graph).toBeVisible();
    expect(screen.getByLabelText("Workflow execution summary")).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Workflow inspector progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 11 workflow nodes reached");
    expect(within(graph).getByText("Generation")).toBeVisible();
    expect(within(graph).getByText("Generation").closest("article")).toHaveAttribute(
      "aria-current",
      "step"
    );
    expect(within(graph).getByText("context_assembly")).toBeVisible();
    expect(within(graph).getByText("prompt_rendering")).toBeVisible();
    expect(within(graph).getByText("failure")).toBeVisible();
    expect(
      within(workflowPanel).getByText("No runtime transitions recorded yet.")
    ).toBeVisible();
    expect(screen.queryByText("Preview request")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("renders runtime transitions, retries, and event traces from streamed metadata", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:00Z",
          code: "request_received",
          currentStep: "intake",
          eventType: "status",
          message: "Request accepted.",
          sequence: 0,
          step: "intake"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:01Z",
          code: "route_selected",
          completedSteps: ["intake"],
          currentStep: "routing",
          eventType: "status",
          message: "Route selected.",
          sequence: 1,
          step: "routing"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:02Z",
          code: "generation_input_prepared",
          completedSteps: ["intake", "routing"],
          currentStep: "generation",
          eventType: "generation_input",
          message: "Provider generation input prepared.",
          sequence: 2,
          skippedSteps: [
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering"
          ],
          step: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:03Z",
          currentStep: "generation",
          eventType: "token_delta",
          sequence: 3,
          step: "generation",
          text: "draft"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:04Z",
          code: "generation_input_prepared",
          completedSteps: ["intake", "routing"],
          currentStep: "generation",
          eventType: "generation_input",
          message: "Provider generation input prepared.",
          refinementCount: 1,
          sequence: 4,
          skippedSteps: [
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering"
          ],
          step: "generation"
        }),
        runtimeWorkflowEvent({
          answer: "```ts\nconsole.log('refined');\n```",
          at: "2026-05-22T10:00:05Z",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "review",
            "refinement",
            "finalization"
          ],
          currentStep: null,
          eventType: "final",
          phase: "completed",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 5,
          skippedSteps: [
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering"
          ],
          status: "completed",
          step: "finalization"
        })
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Write code for a Three.js scene." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText(/refined/)).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));

    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow inspector" });
    const workflowGraph = within(workflowPanel).getByRole("group", {
      name: "LangGraph workflow visualization"
    });
    const transitions = within(workflowPanel).getByRole("group", {
      name: "Workflow transition trace"
    });
    const events = within(workflowPanel).getByRole("group", {
      name: "Workflow event trace"
    });
    const retries = within(workflowPanel).getByRole("group", {
      name: "Workflow retries"
    });

    expect(within(retries).getByText("1 retry loop")).toBeVisible();
    expect(within(transitions).getByText("Generation retry")).toBeVisible();
    expect(within(events).getAllByText("Generation Input Prepared")).toHaveLength(2);
    expect(
      within(workflowGraph).getByText("Refinement").closest("article")
    ).toHaveAttribute("data-state", "complete");
    expect(
      within(workflowGraph).getByText("Finalization").closest("article")
    ).toHaveAttribute("data-state", "complete");
  });

  it("resizes workspace regions and persists the layout preferences", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });

    expect(await screen.findByText("Session saved")).toBeVisible();
    vi.mocked(persistenceClient.save).mockClear();

    const inspectorHandle = screen.getByRole("separator", {
      name: "Resize inspector"
    });
    fireEvent.mouseDown(inspectorHandle, { clientX: 500 });
    fireEvent.mouseMove(window, { clientX: 440 });
    fireEvent.mouseUp(window);

    expect(inspectorHandle).toHaveAttribute("aria-valuenow", "480");

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");
    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const previewHandle = screen.getByRole("separator", {
      name: "Resize preview shelf"
    });
    fireEvent.mouseDown(previewHandle, { clientY: 200 });
    fireEvent.mouseMove(window, { clientY: 260 });
    fireEvent.mouseUp(window);

    expect(previewHandle).toHaveAttribute("aria-valuenow", "280");

    fireEvent.click(screen.getByRole("button", { name: "Workspace density" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          layout: expect.objectContaining({
            density: "compact",
            inspectorCollapsed: false,
            inspectorWidth: 480,
            previewHeight: 280
          }),
          previewOpen: true
        })
      );
    });
  });

  it("restores a persisted workspace session on mount", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const persistedRecord = {
      ...createWorkspaceSessionRecord({
        activeArtifactId: "session-notes",
        activeInspectorTab: "Artifacts",
        layout: {
          density: "compact",
          inspectorCollapsed: false,
          inspectorWidth: 460,
          previewHeight: 260
        },
        preferences: {
          autoOpenPreview: false,
          showDebugPanels: false,
          theme: "codex"
        },
        previewArtifactId: "preview-manifest",
        previewOpen: true,
        snapshot
      }),
      messages: [
        {
          role: "user",
          time: "12:00",
          content: "Persist this workspace."
        },
        {
          role: "assistant",
          time: "12:01",
          content: "Workspace restored."
        }
      ],
      title: "Restored projection session",
      workspace: {
        name: "Restored projection session",
        focus: "Restored audio field"
      }
    } satisfies ReturnType<typeof createWorkspaceSessionRecord>;
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({
        error: null,
        record: persistedRecord,
        source: "remote" as const
      })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(snapshot, { persistenceClient });

    expect(await screen.findByText("Workspace restored.")).toBeVisible();
    expect(screen.getByText("Restored projection session")).toBeVisible();
    expect(screen.getByRole("tab", { name: "Artifacts" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "projection-notes.md"
    );
    expect(screen.getByRole("region", { name: "Preview workspace" })).toHaveTextContent(
      "Preview open"
    );
    expect(screen.getByRole("separator", { name: "Resize inspector" })).toHaveAttribute(
      "aria-valuenow",
      "460"
    );
    expect(screen.getByRole("separator", { name: "Resize preview shelf" })).toHaveAttribute(
      "aria-valuenow",
      "260"
    );
    expect(document.documentElement).toHaveAttribute("data-cca-theme", "codex");
    expect(screen.getByText("Session restored")).toBeVisible();
    expect(persistenceClient.save).not.toHaveBeenCalled();
  });

  it("saves workspace state changes after persistence is ready", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });

    expect(await screen.findByText("Session saved")).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          activeInspectorTab: "Artifacts",
          sessionId: "local-nextjs-session",
          userId: "local-user"
        })
      );
    });
  });

  it("falls back when persistence load and save calls hang", async () => {
    vi.useFakeTimers();
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(
        () => new Promise<WorkspacePersistenceLoadResult>(() => undefined)
      ),
      save: vi.fn(
        () => new Promise<WorkspacePersistenceSaveResult>(() => undefined)
      )
    };

    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });

    expect(screen.getByText("Restoring session")).toBeVisible();

    await act(async () => {
      vi.advanceTimersByTime(1501);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText("Saving session")).toBeVisible();

    await act(async () => {
      vi.advanceTimersByTime(1500);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText("Saved locally")).toBeVisible();
  });
});
