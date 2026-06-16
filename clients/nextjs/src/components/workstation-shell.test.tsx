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
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type ArtifactSummary,
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
      target: "Browser preview"
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
      target: "Browser preview"
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
      target: "Browser preview",
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

function snapshotWithHydraPreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "feedback-lattice.hydra.js";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        domain: "hydra",
        language: "JavaScript + Hydra",
        rendererId: "surface.hydra",
        runtime: "hydra",
        summary: "Hydra patch with oscillators, modulation, feedback, and output routing.",
        title
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      renderer: "surface.hydra",
      sourceArtifactName: title,
      summary: "Runtime is ready to mount a bounded Hydra synth preview.",
      target: "Browser preview / Hydra",
      targetId: "browser_sandbox"
    },
    code: {
      ...snapshot.code,
      excerpt: [
        "osc(12, 0.08, 1.4)",
        "  .kaleid(5)",
        "  .modulate(noise(3, 0.2), 0.14)",
        "  .blend(shape(4, 0.36, 0.02), 0.24)",
        "  .out(o0);"
      ],
      language: "JavaScript + Hydra",
      title
    }
  };
}

function snapshotWithTonePreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "generative-pulse.tone.js";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        domain: "tone_js",
        language: "JavaScript + Tone.js",
        rendererId: "surface.tone",
        runtime: "tone",
        summary: "Tone.js synth sequence with envelope, delay, and transport.",
        title
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      renderer: "surface.tone",
      sourceArtifactName: title,
      summary: "Runtime is ready to mount a controlled Tone.js audio preview.",
      target: "Browser preview / Tone.js",
      targetId: "browser_sandbox"
    },
    code: {
      ...snapshot.code,
      excerpt: [
        "const synth = new Tone.Synth({",
        "  envelope: { attack: 0.03, decay: 0.2, sustain: 0.4, release: 0.7 }",
        "}).toDestination();",
        "const delay = new Tone.FeedbackDelay('8n', 0.25).toDestination();",
        "new Tone.Sequence(",
        "  (time, note) => synth.triggerAttackRelease(note, '8n', time),",
        "  ['C4', 'E4', 'G4', 'B4'],",
        "  '8n'",
        ").start(0);",
        "Tone.Transport.bpm.value = 104;",
        "Tone.Transport.start();"
      ],
      language: "JavaScript + Tone.js",
      title
    }
  };
}

function snapshotWithArtifactComparison(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const artifacts: ArtifactSummary[] = [
    {
      ...snapshot.artifacts[0],
      content: [
        "function setup() {",
        "  createCanvas(windowWidth, 320);",
        "}",
        "function draw() {",
        "  background(8, 12, 18);",
        "}"
      ].join("\n"),
      critique: artifactCritique({
        artifactId: "source-sketch",
        artifactTitle: "aurora-field.p5.js",
        overallScore: 0.88,
        rank: 2,
        rationale: "Stable p5 fallback with a direct preview route.",
        recommended: false
      }),
      domain: "p5_js",
      isDefault: true,
      language: "p5.js",
      qualityRank: 2,
      qualityScore: 0.88,
      rendererId: "surface.p5",
      runtime: "p5",
      title: "aurora-field.p5.js"
    },
    {
      actions: ["Open", "Preview", "Copy", "Download"],
      content: [
        "void main() {",
        "  vec2 uv = gl_FragCoord.xy / u_resolution.xy;",
        "  gl_FragColor = vec4(uv, 0.8, 1.0);",
        "}"
      ].join("\n"),
      critique: artifactCritique({
        artifactId: "shader-field",
        artifactTitle: "shader-field.frag",
        overallScore: 0.94,
        rank: 1,
        rationale:
          "Shader candidate has the strongest prompt alignment and preview readiness.",
        refinementGuidance: "Keep the palette restrained while refining motion.",
        recommended: true
      }),
      domain: "glsl",
      creativeTranslation: {
        colorMaterialDirection: ["cyan"],
        creativeIntent: "Create a minimal sacred shader field.",
        generationConstraints: [],
        geometricReferences: ["sacred geometry"],
        moodAtmosphere: ["minimal"],
        movementLanguage: ["pulse"],
        musicalReferences: [],
        outputModality: "visual",
        refinementTargets: [],
        runtimeRecommendations: ["GLSL"],
        sacredGeometry: {
          audioImplications: [],
          colorMaterialDirection: [],
          concepts: ["mandala"],
          generationConstraints: [],
          geometricStructure: [],
          movementBehavior: [],
          runtimeRecommendations: ["GLSL"],
          symmetryType: [],
          visualComposition: []
        },
        shaderPresets: {
          colorBehavior: [],
          lightMaterialBehavior: [],
          motionBehavior: [],
          performanceConstraints: [],
          presets: ["glow"],
          runtimeSuitability: ["GLSL"],
          shaderStructure: []
        },
        structureDirection: [],
        symbolicReferences: [],
        visualStyle: {
          compositionTendencies: [],
          contrastBehavior: [],
          motionTendencies: [],
          paletteBehavior: [],
          runtimeSuitability: ["GLSL"],
          spatialOrganization: [],
          styles: ["minimal", "sacred geometry"],
          textureTendencies: []
        }
      },
      id: "shader-field",
      isRecommended: true,
      language: "GLSL",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      qualityRank: 1,
      qualityScore: 0.94,
      rendererId: "surface.glsl",
      runtime: "glsl",
      status: "Generated",
      summary: "Recommended GLSL fragment shader with live browser preview support.",
      title: "shader-field.frag",
      type: "code"
    },
    {
      actions: ["Open", "Preview", "Copy", "Download"],
      content: "osc(10, 0.1, 1.2).modulate(shape(4)).out();",
      critique: artifactCritique({
        artifactId: "hydra-lattice",
        artifactTitle: "feedback-lattice.hydra.js",
        overallScore: 0.71,
        rank: 3,
        rationale: "Hydra version provides a compact feedback-oriented runtime.",
        refinementGuidance: "Tune modulation depth before increasing source density.",
        recommended: false
      }),
      domain: "hydra",
      id: "hydra-lattice",
      language: "JavaScript",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      qualityRank: 3,
      qualityScore: 0.71,
      rendererId: "surface.hydra",
      runtime: "hydra",
      status: "Generated",
      summary: "Hydra code is ready for the bounded live preview runtime.",
      title: "feedback-lattice.hydra.js",
      type: "code"
    },
    {
      actions: ["Open", "Preview", "Copy", "Download"],
      content: [
        "const synth = new Tone.Synth().toDestination();",
        "new Tone.Sequence(() => {}, ['C4', 'E4'], '8n').start(0);",
        "Tone.Transport.start();"
      ].join("\n"),
      domain: "tone_js",
      id: "tone-pulse",
      language: "JavaScript + Tone.js",
      previewEligible: true,
      previewTarget: "browser_sandbox",
      rendererId: "surface.tone",
      runtime: "tone",
      status: "Generated",
      summary: "Tone.js candidate remains silent until explicitly started.",
      title: "pulse.tone.js",
      type: "code"
    },
    {
      actions: ["Open", "Copy", "Download"],
      content: "const shader = `@fragment fn main() {}`;",
      domain: "webgpu_wgsl",
      id: "wgsl-fallback",
      language: "WGSL",
      previewEligible: false,
      previewTarget: "",
      rendererId: null,
      runtime: null,
      status: "Generated",
      summary: "WebGPU candidate remains available for code inspection.",
      title: "field.wgsl",
      type: "code"
    }
  ];

  return {
    ...snapshot,
    artifacts,
    inspectorTabs: snapshot.inspectorTabs.map((tab) => ({
      ...tab,
      active: tab.label === "Artifacts"
    })),
    preview: {
      ...snapshot.preview,
      artifactName: "aurora-field.p5.js",
      sourceArtifactId: "source-sketch",
      sourceArtifactName: "aurora-field.p5.js",
      summary:
        "Runtime context is ready for the selected artifact. Open the preview shelf to render it in the browser preview.",
      target: "Browser preview / p5.js",
      targetId: "browser_sandbox"
    }
  };
}

function artifactCritique(
  overrides: Partial<NonNullable<ArtifactSummary["critique"]>>
): NonNullable<ArtifactSummary["critique"]> {
  return {
    artifactId: "artifact",
    artifactTitle: "artifact.p5.js",
    codeQuality: {
      rationale: "Source is complete.",
      score: 0.9
    },
    creativeQuality: {
      rationale: "Strong visual candidate.",
      score: 0.9
    },
    domainAppropriateness: {
      rationale: "Domain matches.",
      score: 0.9
    },
    overallScore: 0.9,
    passed: true,
    previewReadiness: {
      rationale: "Preview is ready.",
      score: 0.9
    },
    promptAlignment: {
      rationale: "Matches the prompt.",
      score: 0.9
    },
    rank: 1,
    rationale: "Strong artifact candidate.",
    reasons: [],
    recommended: false,
    refinementGuidance: null,
    runtimeSuitability: {
      rationale: "Runtime is supported.",
      score: 0.9
    },
    sourceOrder: 1,
    ...overrides
  };
}

function snapshotWithReadyPreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    preview: {
      ...snapshot.preview,
      active: true,
      collapsed: false,
      outputArtifactName: "preview-request.json",
      state: "ready",
      status: "Preview open"
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
      detail: "No retrieval chunks were returned for this request.",
      query: "Find TouchDesigner references for this projection loop.",
      requestedDomains: ["touchdesigner"],
      warning: "No retrieved chunks for TouchDesigner.",
      sources: []
    }
  };
}

function snapshotWithIgnoredRetrievalSource(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot,
    retrieval: {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source, index) =>
        index === 1
          ? {
              ...source,
              selectedForContext: false,
              chunks: source.chunks.map((chunk) => ({
                ...chunk,
                usedInContext: false
              }))
            }
          : source
      )
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

function installAnimationFrameStepper() {
  const callbacks: FrameRequestCallback[] = [];

  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    value: vi.fn((callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    })
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    value: vi.fn()
  });

  return {
    flush(frameTimes: number[]) {
      for (const frameTime of frameTimes) {
        const callback = callbacks.shift();
        if (!callback) {
          break;
        }

        callback(frameTime);
      }
    }
  };
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

type SandboxStatusInput = {
  state: "idle" | "starting" | "ready" | "running" | "stopped" | "error";
  label: string;
  detail: string;
  diagnostics?: string[];
  error?: {
    message: string;
    debugMessage?: string;
    type?: string;
  } | null;
};

async function waitForSandboxRuntimeFrame(
  surface: HTMLElement,
  label: string
) {
  const frame = (await within(surface).findByLabelText(
    label
  )) as HTMLIFrameElement;

  await waitFor(() => {
    expect(frame.dataset.runtimeId).toMatch(/^preview-runtime-/);
  });

  return frame;
}

function dispatchSandboxRuntimeStatus(
  frame: HTMLIFrameElement,
  status: SandboxStatusInput
) {
  const runtimeId = frame.dataset.runtimeId;

  expect(runtimeId).toBeTruthy();
  dispatchSandboxRuntimeStatusByRuntimeId(runtimeId as string, status);
}

function dispatchSandboxRuntimeStatusByRuntimeId(
  runtimeId: string,
  status: SandboxStatusInput
) {
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          source: "cca-preview-runtime",
          runtimeId,
          type: "status",
          status
        }
      })
    );
  });
}

function dispatchSandboxRuntimeFrame(
  frame: HTMLIFrameElement,
  renderedAtMs: number
) {
  const runtimeId = frame.dataset.runtimeId;

  expect(runtimeId).toBeTruthy();
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          renderedAtMs,
          runtimeId,
          source: "cca-preview-runtime",
          type: "frame"
        }
      })
    );
  });
}

function createMockCanvas2DContext(): CanvasRenderingContext2D {
  return {
    arc: vi.fn(),
    beginPath: vi.fn(),
    clearRect: vi.fn(),
    fill: vi.fn(),
    fillRect: vi.fn(),
    lineTo: vi.fn(),
    moveTo: vi.fn(),
    setTransform: vi.fn(),
    stroke: vi.fn(),
    globalAlpha: 1,
    fillStyle: "#000",
    lineWidth: 1,
    strokeStyle: "#fff"
  } as unknown as CanvasRenderingContext2D;
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
  decisionReason,
  evaluation,
  eventType,
  message,
  observability,
  phase = "running",
  refinementCount = 0,
  reviewOutcome = null,
  retryCount,
  retryReason,
  sequence,
  skippedSteps = [],
  status = "running",
  step,
  telemetry,
  transitionSource,
  transitionTarget,
  text
}: {
  answer?: string;
  at: string;
  code?: string;
  completedSteps?: string[];
  currentStep: string | null;
  decisionReason?: string;
  evaluation?: Record<string, unknown>;
  eventType: AssistantStreamEvent["event_type"];
  message?: string;
  observability?: Record<string, unknown>;
  phase?: string;
  refinementCount?: number;
  reviewOutcome?: string | null;
  retryCount?: number;
  retryReason?: string;
  sequence: number;
  skippedSteps?: string[];
  status?: string;
  step: string | null;
  telemetry?: Record<string, unknown>;
  transitionSource?: string;
  transitionTarget?: string;
  text?: string;
}): AssistantStreamEvent {
  return {
    event_type: eventType,
    sequence,
    payload: {
      ...(answer ? { answer } : {}),
      ...(code ? { code } : {}),
      ...(evaluation ? { evaluation } : {}),
      ...(message ? { message } : {}),
      ...(step ? { node: step } : {}),
      ...(observability ? { observability } : {}),
      ...(retryCount != null ? { retry_count: retryCount } : {}),
      ...(retryReason ? { retry_reason: retryReason } : {}),
      ...(telemetry ? { telemetry } : {}),
      ...(transitionSource ? { transition_source: transitionSource } : {}),
      ...(transitionTarget ? { transition_target: transitionTarget } : {}),
      ...(decisionReason ? { decision_reason: decisionReason } : {}),
      ...(transitionSource && transitionTarget && decisionReason
        ? {
            edge: {
              source: transitionSource,
              target: transitionTarget,
              decision_reason: decisionReason
            }
          }
        : {}),
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

  it("renders a polished first-run workspace without demo or infrastructure noise", () => {
    renderShell(getInitialWorkspaceSnapshot());

    expect(
      screen.getByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    expect(screen.getByText("New creative session")).toBeVisible();
    expect(screen.getByText("p5.js sketches")).toBeVisible();
    expect(screen.getByText("Brief -> generate -> preview -> refine")).toBeVisible();
    expect(screen.queryByText(/aurora/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Session persistence issue")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Selected artifact refinement" })
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Ready to start"
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "Ready for first prompt"
    );
    expect(screen.getByRole("group", { name: "Workflow summary" })).toHaveAttribute(
      "data-state",
      "idle"
    );
    expect(screen.getByRole("group", { name: "Artifacts summary" })).toHaveTextContent(
      "0"
    );
    expect(screen.getByRole("group", { name: "Preview summary" })).toHaveAttribute(
      "data-state",
      "unavailable"
    );
    expect(screen.getByRole("group", { name: "Retrieval summary" })).toHaveAttribute(
      "data-state",
      "empty"
    );
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuetext", "0 of 14 workflow nodes reached");

    fireEvent.click(
      screen.getByRole("button", {
        name: "Create a p5.js particle field that feels like slow bioluminescent drift."
      })
    );
    expect(screen.getByLabelText("Assistant prompt")).toHaveValue(
      "Create a p5.js particle field that feels like slow bioluminescent drift."
    );
  });

  it("ignores a persisted seeded demo session on first run", async () => {
    const seededSnapshot = getLocalWorkspaceSnapshot();
    const persistedRecord = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: true,
      snapshot: seededSnapshot
    });
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({
        error: null,
        record: persistedRecord,
        source: "local" as const
      })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(getInitialWorkspaceSnapshot(), { persistenceClient });

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenCalled();
    });
    expect(
      screen.getByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    expect(screen.queryByText(/aurora/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Workflow summary" })).toHaveAttribute(
      "data-state",
      "idle"
    );
  });

  it("keeps the runtime console quiet on first run", () => {
    renderShell(getInitialWorkspaceSnapshot());

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    const runtimePanel = screen.getByRole("tabpanel", {
      name: "Runtime console inspector"
    });

    expect(runtimePanel).toBeVisible();
    expect(
      within(runtimePanel).getByRole("group", { name: "Runtime console empty state" })
    ).toHaveTextContent("No runtime activity yet");
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime event history" })
    ).not.toBeInTheDocument();
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime metrics" })
    ).not.toBeInTheDocument();
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

  it("renders all workspace theme presets and applies them", () => {
    renderShell();

    for (const [label, theme] of [
      ["Aqua", "aqua"],
      ["Codex", "codex"],
      ["Matrix", "matrix"],
      ["Terminal", "terminal"],
      ["Horizon", "horizon"],
      ["Zen", "zen"],
      ["Blueprint", "blueprint"]
    ] as const) {
      fireEvent.click(screen.getByRole("button", { name: "Theme" }));
      fireEvent.click(screen.getByRole("button", { name: `Use ${label} theme` }));
      expect(document.documentElement).toHaveAttribute("data-cca-theme", theme);
    }
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

  it("supports focus mode and density toggles without changing the data flow", () => {
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

    for (const tab of [
      "Overview",
      "Preview",
      "Runtime",
      "Code",
      "Workflow",
      "Telemetry",
      "Artifacts",
      "Retrieval"
    ]) {
      expect(screen.getByRole("tab", { name: tab })).toBeVisible();
    }

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Overview inspector" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Workflow summary" })).toHaveAttribute(
      "data-state",
      "running"
    );
    expect(screen.getByRole("group", { name: "Artifacts summary" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Preview summary" })).toHaveAttribute(
      "data-state",
      "generating"
    );
    expect(screen.getByRole("group", { name: "Telemetry summary" })).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Image references summary" })
    ).toHaveAttribute("data-state", "empty");
    expect(screen.getByRole("group", { name: "Retrieval summary" })).toHaveAttribute(
      "data-state",
      "available"
    );
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 14 workflow nodes reached");
    expect(screen.queryByRole("tabpanel", { name: "Code inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel", { name: "Preview inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("switches inspector tabs without stacking panels", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Preview" }));

    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const previewPanel = screen.getByRole("tabpanel", {
      name: "Preview inspector"
    });
    expect(previewPanel).toBeVisible();
    expect(
      within(previewPanel).getByRole("group", { name: "Preview runtime metadata" })
    ).toBeVisible();
    expect(
      within(previewPanel).getByRole("group", { name: "Preview source metadata" })
    ).toHaveTextContent("4ff94984");
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();

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
    expect(screen.queryByRole("tabpanel", { name: "Preview inspector" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    expect(screen.getByRole("tab", { name: "Runtime" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(
      screen.getByRole("tabpanel", { name: "Runtime console inspector" })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    expect(screen.getByRole("tab", { name: "Retrieval" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Retrieval inspector" })).toBeVisible();
  });

  it("renders a grouped retrieval source explorer with chunk drilldown", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval status" })
    ).toBeVisible();
    expect(within(retrievalPanel).getByText("Official knowledge base")).toBeVisible();
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval confidence" })
    ).toHaveTextContent("Medium confidence");
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval coverage" })
    ).toHaveTextContent("2/2 domains covered");
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval context used" })
    ).toHaveTextContent("3 chunks used");
    const qualityDeepDive = within(retrievalPanel).getByLabelText(
      "Retrieval quality deep dive"
    );
    expect(qualityDeepDive).toHaveAttribute("data-open", "true");
    expect(qualityDeepDive).toHaveTextContent("Medium retrieval quality");
    expect(qualityDeepDive).toHaveTextContent("Balanced across 2 domains");
    const explorer = within(retrievalPanel).getByRole("region", {
      name: "Retrieval source explorer"
    });
    expect(explorer).toHaveTextContent(
      "2 selected sources · No ignored sources reported"
    );
    expect(explorer).toHaveTextContent(
      "WebGPU API contributed most with 2/3 context chunks."
    );
    const healthToggle = within(explorer).getByLabelText(
      "Toggle knowledge base source health dashboard"
    );
    expect(healthToggle.closest("details")).not.toHaveAttribute("open");
    expect(healthToggle).toHaveTextContent("Stale");

    fireEvent.click(healthToggle);

    expect(healthToggle.closest("details")).toHaveAttribute("open");
    expect(
      within(explorer).getByRole("list", {
        name: "Knowledge base source health metrics"
      })
    ).toHaveTextContent("280 indexed chunks");
    expect(
      within(explorer).getByRole("button", {
        name: "Inspect source WebGPU API"
      })
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      within(explorer).getByRole("button", {
        name: "Inspect source OpenGL Shading Language 4.60 Specification"
      })
    ).toBeVisible();
    const webgpuDetail = within(explorer).getByRole("group", {
      name: "WebGPU API source details"
    });
    expect(webgpuDetail).toHaveTextContent("webgpu_mdn_api");
    expect(webgpuDetail).toHaveTextContent("Top contributor");
    expect(webgpuDetail).toHaveTextContent("2 retrieved chunks");
    expect(webgpuDetail).toHaveTextContent("Ranks #1–#2");
    expect(webgpuDetail).toHaveTextContent("2/3 context chunks");
    expect(webgpuDetail).toHaveTextContent("Used in context");
    expect(within(webgpuDetail).getByText("Rank #1")).toBeVisible();
    expect(
      within(webgpuDetail).getAllByText(/Domain match ·/).length
    ).toBeGreaterThan(0);
    expect(
      within(webgpuDetail).getAllByText("Why selected").length
    ).toBeGreaterThan(0);
    expect(within(webgpuDetail).getByText("Best match")).toBeVisible();
    expect(within(webgpuDetail).getByText("High relevance")).toBeVisible();
    expect(
      within(webgpuDetail).getByRole("region", {
        name: "WebGPU API source health"
      })
    ).toHaveTextContent("184 indexed chunks");

    fireEvent.click(
      within(explorer).getByRole("button", {
        name: "Inspect source OpenGL Shading Language 4.60 Specification"
      })
    );

    const glslDetail = within(explorer).getByRole("group", {
      name: "OpenGL Shading Language 4.60 Specification source details"
    });
    expect(glslDetail).toHaveTextContent("glsl_language_spec_460");
    expect(glslDetail).toHaveTextContent("Rank #3");
    expect(glslDetail).toHaveTextContent("Review soon");
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
    expect(
      within(retrievalPanel).queryByRole("region", {
        name: "Retrieval source explorer"
      })
    ).not.toBeInTheDocument();
  });

  it("labels ignored sources and unused chunks without losing global rank", () => {
    renderShell(snapshotWithIgnoredRetrievalSource());

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const explorer = screen.getByRole("region", {
      name: "Retrieval source explorer"
    });
    expect(explorer).toHaveTextContent("1 selected source · 1 ignored source");

    fireEvent.click(
      within(explorer).getByRole("button", {
        name: "Inspect source OpenGL Shading Language 4.60 Specification"
      })
    );

    const ignoredDetail = within(explorer).getByRole("group", {
      name: "OpenGL Shading Language 4.60 Specification source details"
    });
    expect(ignoredDetail).toHaveTextContent("Not selected");
    expect(ignoredDetail).toHaveTextContent("0/2 context chunks");
    expect(ignoredDetail).toHaveTextContent("Rank #3");
    expect(ignoredDetail).toHaveTextContent("Not used");
    expect(ignoredDetail).toHaveTextContent("Selection note");
    expect(ignoredDetail).toHaveTextContent(
      "Retrieved as a candidate source but not included in the final context"
    );
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
            status: "succeeded",
            result: {
              preview_artifact_id: "source-sketch",
              summary: "p5.js runtime ready for browser preview execution.",
              request: {
                target: "browser_sandbox"
              },
              provenance: {
                renderer_id: "surface.p5"
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
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "14");

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();
    expect(
      within(preview).queryByText("p5.js runtime ready for browser preview execution.")
    ).not.toBeInTheDocument();
    expect(preview.querySelector("details")).toHaveAttribute("open");

    fireEvent.click(screen.getByRole("tab", { name: "Preview" }));
    const previewPanel = screen.getByRole("tabpanel", {
      name: "Preview inspector"
    });
    const previewStatus = within(previewPanel).getByRole("group", {
      name: "Preview canvas status"
    });

    expect(
      within(previewStatus).getByText(
        "p5.js runtime ready for browser preview execution."
      )
    ).toBeVisible();
  });

  it("hydrates final stream output into the active artifact and routed preview", async () => {
    const generatedAnswer = [
      "Generated a controlled scene artifact.",
      "```ts",
      "import * as THREE from 'three';",
      "const scene = new THREE.Scene();",
      "const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);",
      "const renderer = new THREE.WebGLRenderer({ antialias: true });",
      "scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));",
      "```"
    ].join("\n");
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "generation_input",
          sequence: 0,
          payload: {
            code: "generation_input_prepared",
            message: "Generation input ready."
          }
        },
        {
          event_type: "final",
          sequence: 1,
          payload: { answer: generatedAnswer }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Create a small Three.js preview scene." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText(/Generated a controlled scene artifact/)
    ).toBeVisible();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(
      within(preview).getByText("generated-scene.three.ts", {
        selector: "summary span"
      })
    ).toBeVisible();
    expect(
      within(preview).getByText("Preview open", { selector: "summary small" })
    ).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(within(preview).getByText("Three scene surface")).toBeVisible();
    expect(within(preview).getByText("Preview open / Three.js")).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });
    expect(within(codePanel).getByText("generated-scene.three.ts")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "generated-scene.three.ts content"
      })
    ).toHaveTextContent("const scene = new THREE.Scene();");

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts inspector"
    });
    expect(
      within(artifactsPanel).getByRole("article", {
        name: "generated-scene.three.ts artifact"
      })
    ).toHaveAttribute("aria-current", "true");
  });

  it("keeps multiple generated artifacts selectable while preview follows previewable candidates", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "artifact_extracted",
          sequence: 0,
          payload: {
            artifacts: [
              {
                id: "palette-notes",
                title: "palette-notes.py",
                language: "Python",
                source_language: "python",
                content: "palette = ['#0bf', '#111']",
                preview_eligible: false,
                source_order: 1,
                is_default: false
              },
              {
                id: "orbit-sketch",
                title: "orbit-sketch.p5.js",
                language: "JavaScript + p5.js",
                source_language: "javascript",
                content:
                  "function setup() {\n  createCanvas(640, 360);\n}\nfunction draw() {\n  background(12);\n}",
                preview_eligible: true,
                preview_target: "browser_sandbox",
                runtime: "p5",
                renderer_id: "surface.p5",
                source_order: 2,
                is_default: true
              }
            ]
          }
        },
        {
          event_type: "final",
          sequence: 1,
          payload: {
            answer: "Generated two creative candidates.",
            artifacts: [
              {
                id: "palette-notes",
                title: "palette-notes.py",
                language: "Python",
                source_language: "python",
                content: "palette = ['#0bf', '#111']",
                preview_eligible: false,
                source_order: 1,
                is_default: false
              },
              {
                id: "orbit-sketch",
                title: "orbit-sketch.p5.js",
                language: "JavaScript + p5.js",
                source_language: "javascript",
                content:
                  "function setup() {\n  createCanvas(640, 360);\n}\nfunction draw() {\n  background(12);\n}",
                preview_eligible: true,
                preview_target: "browser_sandbox",
                runtime: "p5",
                renderer_id: "surface.p5",
                source_order: 2,
                is_default: true
              }
            ]
          }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate two candidate artifacts." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Generated two creative candidates.")).toBeVisible();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(
      within(preview).getByText("orbit-sketch.p5.js", {
        selector: "summary span"
      })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts inspector"
    });
    expect(
      within(artifactsPanel).getByRole("article", {
        name: "palette-notes.py artifact"
      })
    ).toBeVisible();
    expect(
      within(artifactsPanel).getByRole("article", {
        name: "orbit-sketch.p5.js artifact"
      })
    ).toHaveAttribute("aria-current", "true");

    fireEvent.click(
      within(artifactsPanel).getByRole("button", {
        name: "Open in Code palette-notes.py"
      })
    );
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });
    expect(within(codePanel).getByText("palette-notes.py")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "palette-notes.py content"
      })
    ).toHaveTextContent("palette = ['#0bf', '#111']");
    expect(
      within(preview).getByText("orbit-sketch.p5.js", {
        selector: "summary span"
      })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    fireEvent.click(
      within(
        screen.getByRole("tabpanel", {
          name: "Artifacts inspector"
        })
      ).getByRole("button", {
        name: "Open Preview orbit-sketch.p5.js"
      })
    );
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(
      within(preview).getByText("orbit-sketch.p5.js", {
        selector: "summary span"
      })
    ).toBeVisible();
  });

  it("disables the preview shelf when final stream output has no runnable artifact", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer:
              "Keep the projection motion slower and document the next creative pass."
          }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Summarize refinements without code." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText(
        "Keep the projection motion slower and document the next creative pass."
      )
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(
      within(screen.getByRole("group", { name: "Preview summary" })).getByText(
        "No target"
      )
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });
    expect(within(codePanel).getByText("assistant-response.md")).toBeVisible();
    expect(within(codePanel).getByText("Markdown export")).toBeVisible();
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
                  score: 0.88,
                  rank: 1,
                  original_score: 0.8,
                  score_adjustment: 0.08,
                  domain_match: true,
                  selection_reason:
                    "Selected after semantic ranking and route-specific generation relevance adjustment."
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
    const sourceDetail = within(retrievalPanel).getByRole("group", {
      name: "createCanvas source details"
    });
    expect(sourceDetail).toBeVisible();
    expect(within(retrievalPanel).getAllByText("p5.js").length).toBeGreaterThan(0);
    expect(within(sourceDetail).getByText("88% match")).toBeVisible();
    expect(within(sourceDetail).getAllByText("Rank #1").length).toBeGreaterThan(0);
    expect(within(sourceDetail).getByText("88% score")).toBeVisible();
    expect(within(sourceDetail).getByText("Route adjustment +8 pts")).toBeVisible();
    expect(
      within(sourceDetail).getByText(
        "Selected after semantic ranking and route-specific generation relevance adjustment."
      )
    ).toBeVisible();
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

    renderShell(snapshotWithReadyPreview(), { persistenceClient });

    expect(await screen.findByText("Session saved")).toBeVisible();
    vi.mocked(persistenceClient.save).mockClear();

    fireEvent.click(screen.getByRole("button", { name: "Theme" }));
    fireEvent.click(screen.getByRole("button", { name: "Use Blueprint theme" }));

    expect(document.documentElement).toHaveAttribute("data-cca-theme", "blueprint");

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Preview auto-open" }));
    fireEvent.click(screen.getByRole("button", { name: "Advanced traces" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          preferences: {
            theme: "blueprint",
            autoOpenPreview: false,
            showDebugPanels: false
          }
        })
      );
    });
  });

  it("keeps persistence fallback compact when saves stay local", async () => {
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
            "Keep editing locally; the workspace can save again when the connection is available.",
          retryLabel: "Retry save"
        }),
        target: "local" as const
      }))
    };

    renderShell(snapshotWithReadyPreview(), { persistenceClient });

    expect(await screen.findByText("Stored locally")).toBeVisible();
    expect(screen.queryByText("Session persistence issue")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Remote session save timed out or could not be reached.")
    ).not.toBeInTheDocument();
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
            status: "succeeded",
            result: {
              preview_artifact_id: "source-sketch",
              summary: "p5.js runtime ready for browser preview execution.",
              request: {
                target: "browser_sandbox"
              },
              provenance: {
                renderer_id: "surface.p5"
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
    expect(within(preview).getByText("Ready when opened")).toBeVisible();
    expect(
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();
    expect(preview.querySelector("details")).not.toHaveAttribute("open");
  });

  it("falls back to the local draft path when the live response is unavailable", async () => {
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
    expect(screen.getByText(/Live response unavailable/)).toBeVisible();
    const userMessage = screen
      .getByText("Make the low-frequency motion calmer.")
      .closest("article");
    const assistantMessage = screen
      .getByText(/Local draft started/)
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
        "Live response error: The model provider is unavailable for this live response."
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
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
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
    expect(details).toHaveAttribute("data-layout-size", "compact");
    expect(summary).toHaveAttribute("aria-expanded", "true");
    expect(preview.querySelector(".previewPanel")).toHaveStyle({
      height: "280px"
    });
    expect(
      screen.getByRole("separator", { name: "Resize preview shelf" })
    ).toHaveAttribute("aria-disabled", "true");
    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(surface).toBeVisible();
    expect(within(preview).getByText("P5 sketch surface")).toBeVisible();
    expect(within(preview).getByText("Generating / p5.js")).toBeVisible();
    expect(
      within(preview).queryByRole("list", { name: "Preview runtime status" })
    ).not.toBeInTheDocument();
    expect(within(preview).queryByText("Opened from")).not.toBeInTheDocument();
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
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
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
      within(preview).queryByText(
        "Preview state cleared for aurora-field.p5.js. Reload or reset the session to restore the latest runtime context."
      )
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Preview" }));
    const previewPanel = screen.getByRole("tabpanel", {
      name: "Preview inspector"
    });
    expect(
      within(previewPanel).getByRole("group", { name: "Preview canvas status" })
    ).toHaveTextContent(
      "Preview state cleared for aurora-field.p5.js. Reload or reset the session to restore the latest runtime context."
    );

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
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();
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
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(within(events).getByText("Preview Runtime Clear Completed")).toBeVisible();
    expect(within(events).getByText("Preview Runtime Reset Approval Requested")).toBeVisible();
    expect(within(events).getByText("Preview Runtime Reset Completed")).toBeVisible();
  });

  it("keeps preview context available when inspecting a non-previewable artifact", () => {
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
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();
    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(
      within(screen.getByRole("tabpanel", { name: "Code inspector" })).getByText(
        "projection-notes.md"
      )
    ).toBeVisible();
  });

  it("opens artifacts, highlights the active artifact, and targets preview actions", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactList = screen.getByRole("tabpanel", { name: "Artifacts inspector" });
    const sourceArtifact = within(artifactList).getByLabelText(
      "aurora-field.p5.js artifact"
    );
    fireEvent.click(
      within(sourceArtifact).getByRole("button", {
        name: "Open in Code aurora-field.p5.js"
      })
    );

    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(codePanel).toHaveAttribute(
      "data-opened-artifact",
      "aurora-field.p5.js"
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

    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(
      within(preview).getByText("preview-request.json", { selector: "summary span" })
    ).toBeVisible();
    expect(within(preview).getByText("Preview open")).toBeVisible();
    expect(within(preview).getByText("Preview open / JSON panel surface")).toBeVisible();
    expect(within(preview).getByText("Preview manifest panel")).toBeVisible();
    expect(preview.querySelector("details")).toHaveAttribute("open");
    expect(preview.querySelector("details")).toHaveAttribute("data-state", "open");
  });

  it("surfaces artifact critique ranking and rationale in the artifacts inspector", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const critiquedSnapshot: AssistantWorkspaceSnapshot = {
      ...snapshot,
      artifacts: snapshot.artifacts.map((artifact, index) =>
        index === 0
          ? {
              ...artifact,
              isRecommended: true,
              qualityRank: 1,
              qualityScore: 0.93,
              critique: {
                artifactId: artifact.id,
                artifactTitle: artifact.title,
                sourceOrder: 1,
                overallScore: 0.93,
                rank: 1,
                passed: true,
                recommended: true,
                promptAlignment: {
                  score: 0.9,
                  rationale: "Matches the prompt."
                },
                creativeQuality: {
                  score: 0.95,
                  rationale: "Strong visual candidate."
                },
                runtimeSuitability: {
                  score: 1,
                  rationale: "p5 runtime is supported."
                },
                codeQuality: {
                  score: 0.91,
                  rationale: "Source is complete."
                },
                previewReadiness: {
                  score: 1,
                  rationale: "Preview is ready."
                },
                domainAppropriateness: {
                  score: 0.86,
                  rationale: "Domain matches."
                },
                creativeEvaluation: {
                  overallScore: 0.87,
                  composition: {
                    score: 0.88,
                    level: "strong",
                    observation: "Clear focal hierarchy detected.",
                    evidence: ["marker: center"]
                  },
                  originality: {
                    score: 0.82,
                    level: "strong",
                    observation: "Generative variation is present.",
                    evidence: ["marker: noise"]
                  },
                  coherence: {
                    score: 0.91,
                    level: "strong",
                    observation: "Runtime structure is coherent.",
                    evidence: ["balanced blocks"]
                  },
                  aestheticConsistency: {
                    score: 0.86,
                    level: "strong",
                    observation: "Palette signals are consistent.",
                    evidence: ["marker: color"]
                  },
                  expressiveness: {
                    score: 0.88,
                    level: "strong",
                    observation: "Motion develops over time.",
                    evidence: ["marker: framecount"]
                  },
                  strengths: [
                    "Composition: Clear focal hierarchy detected."
                  ],
                  refinementOpportunities: [],
                  summary: "5 of 5 creative dimensions are strong."
                },
                sacredConsistency: {
                  overallScore: 0.84,
                  alignment: {
                    score: 0.86,
                    level: "aligned",
                    observation: "Matched mandala metadata cues.",
                    evidence: ["marker: mandala"]
                  },
                  motifConsistency: {
                    score: 0.83,
                    level: "aligned",
                    observation: "Detected radial geometry signals.",
                    evidence: ["marker: radial"]
                  },
                  modalityCoherence: {
                    score: 0.78,
                    level: "aligned",
                    observation: "Detected visual and motion signals.",
                    evidence: ["marker: canvas"]
                  },
                  claimSafety: {
                    score: 0.9,
                    level: "aligned",
                    observation:
                      "No unsupported symbolic authority markers were detected.",
                    evidence: ["bounded design-motif language"]
                  },
                  strengths: [
                    "Claim safety: No unsupported symbolic authority markers."
                  ],
                  refinementOpportunities: [],
                  summary: "Checked 2 symbolic/geometric metadata cues."
                },
                calibratedQuality: {
                  score: 0.86,
                  legacyScore: 0.93,
                  decisionBand: "strong_candidate",
                  confidence: "medium",
                  signals: [
                    {
                      key: "legacy_critique",
                      label: "Legacy critique",
                      score: 0.93,
                      weight: 0.34,
                      rationale:
                        "Existing weighted artifact critique score is preserved."
                    },
                    {
                      key: "runtime_preview",
                      label: "Runtime and preview",
                      score: 1,
                      weight: 0.18,
                      rationale:
                        "Runtime suitability and preview readiness are aligned."
                    }
                  ],
                  adjustments: [],
                  rationale:
                    "strong candidate at 0.86; legacy score 0.93. No conservative caps were required.",
                  summary:
                    "Calibrated decision-support score 0.86 from 2 available signal(s). This is bounded guidance, not an objective measure of artistic quality."
                },
                legacyRank: 1,
                reasons: [],
                rationale: "aurora-field.p5.js is the recommended candidate.",
                refinementGuidance: null
              }
            }
          : artifact
      )
    };

    renderShell(critiquedSnapshot);

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts inspector"
    });

    expect(
      within(artifactsPanel).getByRole("group", {
        name: "Active artifact details"
      })
    ).toHaveTextContent("Quality");
    expect(
      within(artifactsPanel).getByRole("region", {
        name: "Artifact quality summary"
      })
    ).toHaveTextContent("Recommended candidate");
    const qualitySummary = within(artifactsPanel).getByRole("region", {
      name: "Artifact quality summary"
    });
    expect(
      within(qualitySummary).getByText(
        "aurora-field.p5.js is the recommended candidate."
      )
    ).toBeVisible();
    expect(
      within(qualitySummary).getByRole("region", {
        name: "Creative quality critic"
      })
    ).toHaveTextContent("Aesthetic consistency");
    expect(
      within(qualitySummary).getByText(
        "5 of 5 creative dimensions are strong."
      )
    ).toBeVisible();
    expect(
      within(qualitySummary).getByRole("region", {
        name: "Calibrated quality summary"
      })
    ).toHaveTextContent("Strong candidate");
    expect(
      within(qualitySummary).getByRole("region", {
        name: "Calibrated quality summary"
      })
    ).toHaveTextContent("Legacy score");
    expect(
      within(qualitySummary).getByRole("region", {
        name: "Sacred consistency evaluator"
      })
    ).toHaveTextContent("Claim safety");
    expect(
      within(artifactsPanel).getByText(/Rank #1 \/ Quality 93%/)
    ).toBeVisible();
  });

  it("mounts supported p5 artifacts into a controlled live runtime", async () => {
    renderShell(snapshotWithP5Preview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("P5 sketch surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "p5.js live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "p5.js preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering signal-orbit.p5.ts inside an isolated p5-compatible preview frame.",
      label: "p5 runtime running",
      state: "running"
    });
    expect(await within(surface).findByText("p5 runtime running")).toBeVisible();
  });

  it("mounts supported Hydra artifacts into a controlled live runtime", async () => {
    renderShell(snapshotWithHydraPreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("Hydra synth surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Hydra live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "Hydra preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering feedback-lattice.hydra.js as a bounded Hydra-compatible synth.",
      label: "Hydra runtime running",
      state: "running"
    });
    expect(await within(surface).findByText("Hydra runtime running")).toBeVisible();
  });

  it("keeps Tone.js silent until explicit start and reports stop state", async () => {
    renderShell(snapshotWithTonePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });
    expect(within(preview).getByText("Tone.js audio surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Tone.js live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "Tone.js preview runtime frame"
    );
    expect(frame).toHaveAttribute("sandbox", "allow-scripts");

    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "generative-pulse.tone.js is armed. Audio remains silent until Start audio is selected.",
      diagnostics: ["Explicit operator interaction is required before audio starts."],
      label: "Tone.js runtime ready",
      state: "ready"
    });
    expect(await within(surface).findByText("Tone.js runtime ready")).toBeVisible();
    expect(within(surface).getByText("Ready")).toBeVisible();

    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Playing generative-pulse.tone.js through a bounded Web Audio transport.",
      label: "Tone.js runtime running",
      state: "running"
    });
    expect(await within(surface).findByText("Tone.js runtime running")).toBeVisible();

    dispatchSandboxRuntimeStatus(frame, {
      detail: "Audio transport is stopped and output is silent.",
      label: "Tone.js runtime stopped",
      state: "stopped"
    });
    expect(await within(surface).findByText("Tone.js runtime stopped")).toBeVisible();
    expect(within(surface).getByText("Stopped")).toBeVisible();
  });

  it("shows a compact diagnostics overlay for live preview runtimes", async () => {
    renderShell(snapshotWithP5Preview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "p5.js preview runtime frame"
    );

    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering signal-orbit.p5.ts inside an isolated p5-compatible preview frame.",
      label: "p5 runtime running",
      state: "running"
    });
    for (const frameTime of [
      0,
      16,
      32,
      48,
      64,
      80,
      96,
      112,
      128,
      144,
      160,
      176,
      192,
      208,
      224,
      240
    ]) {
      dispatchSandboxRuntimeFrame(frame, frameTime);
    }

    const overlay = within(surface).getByLabelText("Renderer health overlay");

    expect(within(overlay).getByText("FPS")).toBeVisible();
    expect(within(overlay).getByText("Frame")).toBeVisible();
    expect(within(overlay).getByText("Health")).toBeVisible();
    expect(within(overlay).getByText("State")).toBeVisible();
    expect(within(overlay).getByText("Nominal")).toBeVisible();
    expect(within(surface).getByText("63 fps")).toBeVisible();
    expect(within(surface).getByText("16.0 ms")).toBeVisible();
  });

  it("shows runtime diagnostics, metrics, and event history for a live preview runtime", async () => {
    renderShell(snapshotWithP5Preview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "p5.js preview runtime frame"
    );

    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering signal-orbit.p5.ts inside an isolated p5-compatible preview frame.",
      label: "p5 runtime running",
      state: "running"
    });
    for (const frameTime of [
      0,
      16,
      32,
      48,
      64,
      80,
      96,
      112,
      128,
      144,
      160,
      176,
      192,
      208,
      224,
      240
    ]) {
      dispatchSandboxRuntimeFrame(frame, frameTime);
    }

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    const runtimePanel = screen.getByRole("tabpanel", {
      name: "Runtime console inspector"
    });
    const metrics = within(runtimePanel).getByRole("group", {
      name: "Runtime metrics"
    });
    const context = within(runtimePanel).getByRole("group", {
      name: "Runtime context"
    });
    const health = within(runtimePanel).getByRole("group", {
      name: "Runtime health"
    });
    const eventHistory = within(runtimePanel).getByRole("group", {
      name: "Runtime event history"
    });

    expect(metrics).toHaveTextContent("Running");
    expect(metrics).toHaveTextContent("63 fps");
    expect(metrics).toHaveTextContent("16.0 ms");
    expect(metrics).toHaveTextContent("Healthy");
    expect(metrics).toHaveTextContent("Uptime");
    expect(metrics).toHaveTextContent("Reloads");
    expect(metrics).toHaveTextContent("Execution");
    expect(health).toHaveTextContent("Healthy");
    expect(health).toHaveTextContent("frame delivery is within the expected budget");
    expect(context).toHaveTextContent("Browser preview");
    expect(context).toHaveTextContent("p5.js");
    expect(eventHistory).toHaveTextContent("Start");
    expect(eventHistory).toHaveTextContent("p5 runtime running");
    expect(eventHistory).not.toHaveTextContent("First preview frame rendered");
  });

  it("mounts supported Three.js artifacts into a controlled 3D runtime", async () => {
    renderShell(snapshotWithThreePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("Three scene surface")).toBeVisible();
    expect(within(preview).getByText("Generating / Three.js")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Three.js live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "Three.js preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering projection-scene.three.ts inside an isolated Three.js-compatible preview frame.",
      label: "Three.js runtime running",
      state: "running"
    });
    expect(
      await within(surface).findByText("Three.js runtime running")
    ).toBeVisible();
  });

  it("shows a stable Three.js runtime error from the preview frame", async () => {
    renderShell(snapshotWithThreePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("Three scene surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "Three.js live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "Three.js preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail: "WebGL is unavailable in the preview frame.",
      error: {
        message: "WebGL is unavailable in the preview frame.",
        type: "webgl_unavailable"
      },
      label: "Three.js runtime failed",
      state: "error"
    });
    expect(
      await within(surface).findByText("Three.js runtime failed")
    ).toBeVisible();
    expect(within(surface).getByText("Renderer runtime failed")).toBeVisible();
    expect(screen.getByLabelText("Current session")).not.toHaveTextContent("Failure");

    const failedRuntimeId = frame.dataset.runtimeId;
    fireEvent.click(
      within(surface).getByRole("button", { name: "Reload preview runtime" })
    );

    await waitFor(() => {
      expect(frame.dataset.runtimeId).toMatch(/^preview-runtime-/);
      expect(frame.dataset.runtimeId).not.toBe(failedRuntimeId);
    });
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering projection-scene.three.ts inside an isolated Three.js-compatible preview frame.",
      label: "Three.js runtime running",
      state: "running"
    });

    expect(
      await within(surface).findByText("Three.js runtime running")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(within(events).getByText("Preview Runtime Error")).toBeVisible();
    expect(within(events).getByText("Preview Runtime Recovered")).toBeVisible();
  });

  it("shows runtime errors and reload requests inside the runtime console", async () => {
    renderShell(snapshotWithThreePreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "Three.js preview runtime frame"
    );

    dispatchSandboxRuntimeStatus(frame, {
      detail: "WebGL is unavailable in the preview frame.",
      error: {
        message: "WebGL is unavailable in the preview frame.",
        type: "webgl_unavailable"
      },
      label: "Three.js runtime failed",
      state: "error"
    });

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    const runtimePanel = screen.getByRole("tabpanel", {
      name: "Runtime console inspector"
    });
    const diagnostics = within(runtimePanel).getByRole("group", {
      name: "Runtime diagnostics"
    });
    const health = within(runtimePanel).getByRole("group", {
      name: "Runtime health"
    });
    const eventHistory = within(runtimePanel).getByRole("group", {
      name: "Runtime event history"
    });

    expect(health).toHaveTextContent("Failed");
    expect(diagnostics).toHaveTextContent("1 active");
    expect(diagnostics).toHaveTextContent("WebGL is unavailable in the preview frame.");
    expect(eventHistory).toHaveTextContent("Error");

    const failedRuntimeId = frame.dataset.runtimeId;
    fireEvent.click(
      within(surface).getByRole("button", { name: "Reload preview runtime" })
    );

    await waitFor(() => {
      expect(frame.dataset.runtimeId).toMatch(/^preview-runtime-/);
      expect(frame.dataset.runtimeId).not.toBe(failedRuntimeId);
    });
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Rendering projection-scene.three.ts inside an isolated Three.js-compatible preview frame.",
      label: "Three.js runtime running",
      state: "running"
    });

    const reloadHistory = within(runtimePanel).getByRole("group", {
      name: "Runtime reload history"
    });

    expect(reloadHistory).toHaveTextContent("1 reload");
    expect(reloadHistory).toHaveTextContent("Reload requested");
    expect(eventHistory).toHaveTextContent("Reload");
  });

  it("shows a stable GLSL runtime error from the preview frame", async () => {
    renderShell(snapshotWithGlslPreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("Shader surface")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "GLSL live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "GLSL preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail: "Shader did not compile.",
      error: {
        message: "Shader did not compile.",
        type: "shader_compile_failed"
      },
      label: "GLSL runtime failed",
      state: "error"
    });
    expect(await within(surface).findByText("GLSL runtime failed")).toBeVisible();
    expect(within(surface).getByText("Renderer runtime failed")).toBeVisible();
  });

  it.each([
    {
      frameLabel: "p5.js preview runtime frame",
      makeSnapshot: snapshotWithP5Preview,
      runningDetail:
        "Rendering signal-orbit.p5.ts inside an isolated p5-compatible preview frame.",
      runningLabel: "p5 runtime running",
      surfaceTitle: "P5 sketch surface"
    },
    {
      frameLabel: "GLSL preview runtime frame",
      makeSnapshot: snapshotWithGlslPreview,
      runningDetail:
        "Rendering chromatic-field.frag as an isolated WebGL fragment shader.",
      runningLabel: "GLSL runtime running",
      surfaceTitle: "Shader surface"
    },
    {
      frameLabel: "Three.js preview runtime frame",
      makeSnapshot: snapshotWithThreePreview,
      runningDetail:
        "Rendering projection-scene.three.ts inside an isolated Three.js-compatible preview frame.",
      runningLabel: "Three.js runtime running",
      surfaceTitle: "Three scene surface"
    },
    {
      frameLabel: "Hydra preview runtime frame",
      makeSnapshot: snapshotWithHydraPreview,
      runningDetail:
        "Rendering feedback-lattice.hydra.js as a bounded Hydra-compatible synth.",
      runningLabel: "Hydra runtime running",
      surfaceTitle: "Hydra synth surface"
    },
    {
      frameLabel: "Tone.js preview runtime frame",
      makeSnapshot: snapshotWithTonePreview,
      runningDetail:
        "Playing generative-pulse.tone.js through a bounded Web Audio transport.",
      runningLabel: "Tone.js runtime running",
      surfaceTitle: "Tone.js audio surface"
    }
  ])(
    "reloads $surfaceTitle artifacts and ignores stale runtime events",
    async ({ frameLabel, makeSnapshot, runningDetail, runningLabel, surfaceTitle }) => {
      renderShell(makeSnapshot());

      const preview = screen.getByRole("region", { name: "Preview workspace" });
      const summary = within(preview)
        .getByText("Preview available")
        .closest("summary");

      expect(summary).not.toBeNull();
      fireEvent.click(summary as HTMLElement);

      const surface = within(preview).getByRole("group", {
        name: "Preview renderer surface"
      });

      expect(within(preview).getByText(surfaceTitle)).toBeVisible();
      const frame = await waitForSandboxRuntimeFrame(surface, frameLabel);
      const staleRuntimeId = frame.dataset.runtimeId;

      expect(staleRuntimeId).toBeTruthy();
      fireEvent.click(
        within(preview).getByRole("button", { name: "Reload preview state" })
      );

      await waitFor(() => {
        expect(frame.dataset.runtimeId).toMatch(/^preview-runtime-/);
        expect(frame.dataset.runtimeId).not.toBe(staleRuntimeId);
      });
      expect(
        within(preview).getByText("Reloading", { selector: "summary small" })
      ).toBeVisible();

      dispatchSandboxRuntimeStatusByRuntimeId(staleRuntimeId as string, {
        detail: "Stale runtime failed.",
        error: {
          message: "Stale runtime failed.",
          type: "preview_sandbox_runtime_failed"
        },
        label: "Stale runtime failed",
        state: "error"
      });
      expect(within(surface).queryByText("Stale runtime failed")).not.toBeInTheDocument();

      dispatchSandboxRuntimeStatus(frame, {
        detail: runningDetail,
        label: runningLabel,
        state: "running"
      });
      expect(await within(surface).findByText(runningLabel)).toBeVisible();

      fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
      const events = screen.getByRole("group", { name: "Workflow event trace" });

      expect(
        within(events).getByText("Preview Runtime Reload Requested")
      ).toBeVisible();
    }
  );

  it("uses the full inspector panel for code when Code is active", () => {
    renderShell(snapshotWithActiveTab("Code"));

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(codePanel).toBeVisible();
    expect(within(codePanel).getByText("p5.js")).toBeVisible();
    expect(within(codePanel).getByText("Source code")).toBeVisible();
    expect(within(codePanel).getByText("15 lines")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "aurora-field.p5.js content"
      })
    ).toHaveTextContent("function draw()");
    expect(screen.queryByRole("tabpanel", { name: "Overview inspector" })).not.toBeInTheDocument();
  });

  it("renders artifact comparison with recommendation, critique, and runtime support", () => {
    renderShell(snapshotWithArtifactComparison());

    const comparison = screen.getByRole("region", {
      name: "Artifact comparison"
    });
    const recommended = within(comparison).getByRole("group", {
      name: "Recommended artifact comparison"
    });
    const shaderCandidate = within(comparison).getByLabelText(
      "shader-field.frag comparison candidate"
    );
    const hydraCandidate = within(comparison).getByLabelText(
      "feedback-lattice.hydra.js comparison candidate"
    );
    const toneCandidate = within(comparison).getByLabelText(
      "pulse.tone.js comparison candidate"
    );
    const codeOnlyCandidate = within(comparison).getByLabelText(
      "field.wgsl comparison candidate"
    );

    expect(comparison).toHaveTextContent("Multi-preview workspace");
    expect(comparison).toHaveTextContent("5 candidates");
    expect(recommended).toHaveTextContent("Recommended");
    expect(recommended).toHaveTextContent("shader-field.frag");
    expect(comparison).toHaveTextContent(
      "Shader candidate has the strongest prompt alignment and preview readiness."
    );
    expect(shaderCandidate).toHaveAttribute("data-recommended", "true");
    expect(within(shaderCandidate).getByText("Previewable")).toBeVisible();
    expect(within(shaderCandidate).getByText("Visual / GLSL")).toBeVisible();
    expect(within(shaderCandidate).getByText("#1")).toBeVisible();
    expect(within(shaderCandidate).getByText("94%")).toBeVisible();
    expect(within(shaderCandidate).getByText("minimal / sacred geometry")).toBeVisible();
    expect(within(shaderCandidate).getByText("glow")).toBeVisible();
    expect(within(shaderCandidate).getByText("mandala")).toBeVisible();
    expect(
      within(shaderCandidate).getByRole("group", { name: "GLSL live runtime" })
    ).toBeVisible();
    expect(hydraCandidate).toHaveAttribute("data-runtime-support", "previewable");
    expect(within(hydraCandidate).getByText("Previewable")).toBeVisible();
    expect(within(hydraCandidate).getByText("Visual / Hydra")).toBeVisible();
    expect(
      within(hydraCandidate).getByRole("group", { name: "Hydra live runtime" })
    ).toBeVisible();
    expect(
      within(hydraCandidate).getByRole("button", {
        name: "Preview feedback-lattice.hydra.js from comparison"
      })
    ).toBeVisible();
    expect(toneCandidate).toHaveAttribute("data-output-kind", "audio");
    expect(
      within(toneCandidate).getByText("Silent until explicit start")
    ).toBeVisible();
    expect(
      within(toneCandidate).getByRole("group", {
        name: "Tone.js live runtime"
      })
    ).toHaveAttribute("data-runtime-state", "starting");
    expect(within(toneCandidate).queryByText("Tone.js runtime running")).not.toBeInTheDocument();
    expect(codeOnlyCandidate).toHaveAttribute(
      "data-runtime-support",
      "unsupported"
    );
    expect(
      within(codeOnlyCandidate).getByLabelText(
        "field.wgsl safe preview fallback"
      )
    ).toHaveTextContent("Unsupported runtime");
    expect(
      within(codeOnlyCandidate).queryByRole("button", {
        name: "Preview field.wgsl from comparison"
      })
    ).not.toBeInTheDocument();
  });

  it("shows a selected-artifact refinement action with guided instructions", () => {
    renderShell(snapshotWithArtifactComparison());

    const details = screen.getByRole("group", { name: "Active artifact details" });
    const refinement = within(details).getByRole("region", {
      name: "Selected artifact refinement"
    });
    const submitButton = within(refinement).getByRole("button", {
      name: "Refine selected artifact"
    });

    expect(
      within(refinement).getAllByText("Refine selected artifact").length
    ).toBeGreaterThan(1);
    expect(
      within(refinement).getByText(
        "Target aurora-field.p5.js without regenerating every candidate."
      )
    ).toBeVisible();
    expect(submitButton).toBeDisabled();

    fireEvent.click(within(refinement).getByRole("button", { name: "Make this faster" }));

    expect(within(refinement).getByLabelText("Refinement instruction")).toHaveValue(
      "Make this faster"
    );
    expect(submitButton).toBeEnabled();
  });

  it("serializes local artifact parameter changes into refinement context", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer: "Parameter-guided refinement received."
          }
        }
      ])
    );

    renderShell(snapshotWithArtifactComparison(), {
      streamAssistantEvents: backendStream
    });

    const refinement = screen.getByRole("region", {
      name: "Selected artifact refinement"
    });

    fireEvent.change(
      within(refinement).getByLabelText("Movement complexity parameter"),
      {
        target: { value: "9" }
      }
    );
    fireEvent.click(
      within(refinement).getByRole("button", {
        name: "Refine with parameter changes"
      })
    );

    expect(
      await screen.findByText("Parameter-guided refinement received.")
    ).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        query: expect.stringContaining("Movement complexity: 9"),
        artifactRefinement: expect.objectContaining({
          artifactId: "source-sketch",
          instruction: expect.stringContaining("Movement complexity: 9")
        })
      })
    );
  });

  it("sends selected artifact context and hydrates the refined artifact as a new version", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer: "Refined artifact ready.",
            artifacts: [
              {
                id: "source-sketch",
                title: "aurora-field.p5.js",
                type: "code",
                language: "p5.js",
                content: [
                  "function setup() {",
                  "  createCanvas(windowWidth, 320);",
                  "}",
                  "function draw() {",
                  "  background(4, 8, 14);",
                  "  circle(width * 0.5, height * 0.5, 120);",
                  "}"
                ].join("\n"),
                domain: "p5_js",
                runtime: "p5",
                renderer_id: "surface.p5",
                preview_eligible: true,
                preview_target: "browser_sandbox",
                summary: "Refined p5 sketch with calmer organic motion."
              }
            ]
          }
        }
      ])
    );

    renderShell(snapshotWithArtifactComparison(), { streamAssistantEvents: backendStream });

    const refinement = screen.getByRole("region", {
      name: "Selected artifact refinement"
    });

    fireEvent.click(
      within(refinement).getByRole("button", { name: "Make this more organic" })
    );
    fireEvent.click(
      within(refinement).getByRole("button", {
        name: "Refine selected artifact"
      })
    );

    expect(await screen.findByText("Refined artifact ready.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "local-nextjs-session",
        domain: "p5_js",
        domains: ["p5_js"],
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: "Make this more organic",
        artifactRefinement: expect.objectContaining({
          artifactId: "source-sketch",
          title: "aurora-field.p5.js",
          language: "p5.js",
          content: expect.stringContaining("function draw()"),
          instruction: "Make this more organic",
          domain: "p5_js",
          runtime: "p5",
          rendererId: "surface.p5",
          previewEligible: true,
          qualityScore: 0.88,
          qualityRank: 2,
          qualityBefore: 0.88,
          passNumber: 1,
          maxPasses: 2,
          refinementObjective: expect.stringContaining(
            "Make this more organic"
          ),
          refinementPasses: [],
          critiqueRationale: "Stable p5 fallback with a direct preview route."
        })
      })
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "aurora-field.p5.refined.js"
    );

    const preview = screen.getByRole("region", { name: "Preview workspace" });

    expect(
      within(preview).getByText("aurora-field.p5.refined.js", {
        selector: "summary span"
      })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(codePanel).toHaveAttribute(
      "data-opened-artifact",
      "aurora-field.p5.refined.js"
    );
    expect(
      within(codePanel).getByRole("region", {
        name: "aurora-field.p5.refined.js content"
      })
    ).toHaveTextContent("background(4, 8, 14)");

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const refinedDetails = screen.getByRole("group", {
      name: "Active artifact details"
    });

    expect(refinedDetails).toHaveTextContent("Refined");
    expect(refinedDetails).toHaveTextContent("Refined from aurora-field.p5.js");
    expect(refinedDetails).toHaveTextContent("Pass 1");
    expect(refinedDetails).toHaveTextContent("No useful opportunities");
    expect(
      screen.getByLabelText("aurora-field.p5.js artifact")
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("aurora-field.p5.refined.js artifact")
    ).toHaveAttribute("data-active", "true");
  });

  it("selects artifacts from comparison and keeps code plus preview context synced", () => {
    renderShell(snapshotWithArtifactComparison());

    const comparison = screen.getByRole("region", {
      name: "Artifact comparison"
    });
    const shaderCandidate = within(comparison).getByLabelText(
      "shader-field.frag comparison candidate"
    );

    fireEvent.click(
      within(shaderCandidate).getByRole("button", {
        name: "Select shader-field.frag as preferred candidate"
      })
    );

    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "shader-field.frag"
    );
    expect(
      within(screen.getByRole("region", { name: "Preview workspace" })).getByText(
        "shader-field.frag",
        { selector: "summary span" }
      )
    ).toBeVisible();
    expect(
      within(
        screen.getByRole("region", {
          name: "Selected artifact refinement"
        })
      ).getByText(
        "Target shader-field.frag without regenerating every candidate."
      )
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));

    const codePanel = screen.getByRole("tabpanel", { name: "Code inspector" });

    expect(codePanel).toHaveAttribute("data-opened-artifact", "shader-field.frag");
    expect(
      within(codePanel).getByRole("region", {
        name: "shader-field.frag content"
      })
    ).toHaveTextContent("void main()");

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const refreshedComparison = screen.getByRole("region", {
      name: "Artifact comparison"
    });
    fireEvent.click(
      within(refreshedComparison).getByRole("button", {
        name: "Select field.wgsl as preferred candidate"
      })
    );

    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "field.wgsl"
    );
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    expect(
      screen.getByRole("tabpanel", { name: "Code inspector" })
    ).toHaveAttribute("data-opened-artifact", "field.wgsl");
  });

  it("shows focused artifact metadata and actions in the artifacts inspector", () => {
    renderShell(snapshotWithActiveTab("Artifacts"));

    const details = screen.getByRole("group", { name: "Active artifact details" });

    expect(within(details).getByText("Selected artifact")).toBeVisible();
    expect(within(details).getByText("aurora-field.p5.js")).toBeVisible();
    expect(within(details).getByText("Source code")).toBeVisible();
    expect(within(details).getByText("p5.js")).toBeVisible();
    expect(within(details).getByText("Previewable")).toBeVisible();
    expect(within(details).getByText("Runtime")).toBeVisible();
    expect(within(details).getByText("p5.js / surface.p5")).toBeVisible();
    expect(within(details).getByText("Domain")).toBeVisible();
    expect(
      within(details).getByRole("button", {
        name: "Open in Code aurora-field.p5.js"
      })
    ).toBeVisible();
    expect(
      within(details).getByRole("button", {
        name: "Download File aurora-field.p5.js"
      })
    ).toBeVisible();
    expect(
      within(details).getByRole("region", {
        name: "Creative translation summary"
      })
    ).toHaveAttribute("data-state", "legacy");
    expect(
      within(details).getByRole("region", {
        name: "Audio-reactive mapping summary"
      })
    ).toHaveAttribute("data-state", "legacy");
  });

  it("shows compact creative translation guidance for generated artifacts", () => {
    const snapshot = snapshotWithActiveTab("Artifacts");
    renderShell({
      ...snapshot,
      artifacts: snapshot.artifacts.map((artifact, index) =>
        index === 0
          ? {
              ...artifact,
              creativeTranslation: {
                outputModality: "audiovisual",
                creativeIntent:
                  "Create an audio-reactive mandala with a meditative pulse.",
                symbolicReferences: ["mandala"],
                geometricReferences: ["sacred geometry"],
                musicalReferences: ["rhythm"],
                moodAtmosphere: ["meditative"],
                movementLanguage: ["pulse"],
                colorMaterialDirection: ["cyan"],
                runtimeRecommendations: ["p5.js", "Tone.js"],
                structureDirection: [
                  "Coordinate visual changes with the requested musical structure."
                ],
                generationConstraints: [
                  "Require explicit user interaction before audio playback"
                ],
                refinementTargets: [
                  "Preserve atmosphere: meditative",
                  "Tune motion character: pulse"
                ],
                sacredGeometry: {
                  concepts: ["mandala", "radial symmetry"],
                  geometricStructure: [
                    "Build nested rings around a clear center."
                  ],
                  symmetryType: [
                    "Use radial symmetry with a limited segment count."
                  ],
                  movementBehavior: [
                    "Animate layers with slow counter-rotation."
                  ],
                  visualComposition: ["Keep a strong center."],
                  colorMaterialDirection: ["Use controlled contrast."],
                  runtimeRecommendations: ["p5.js", "GLSL", "Tone.js"],
                  audioImplications: [
                    "Map concentric layers to frequency bands."
                  ],
                  generationConstraints: [
                    "Do not add unsupported symbolic claims."
                  ]
                },
                shaderPresets: {
                  presets: ["glow", "kaleidoscopic symmetry"],
                  colorBehavior: ["Use a restrained luminous palette."],
                  lightMaterialBehavior: ["Use bounded emission layers."],
                  motionBehavior: ["Pulse alternating radial segments."],
                  shaderStructure: ["Separate an emission mask."],
                  runtimeSuitability: [
                    "Use the selected compatible runtime: p5.js."
                  ],
                  performanceConstraints: [
                    "Use a bounded number of glow layers.",
                    "Treat presets as stylized guidance."
                  ]
                },
                visualStyle: {
                  styles: ["minimal", "sacred geometry"],
                  paletteBehavior: ["Use one dominant tone."],
                  contrastBehavior: ["Use clear value hierarchy."],
                  compositionTendencies: [
                    "Use deliberate negative space."
                  ],
                  motionTendencies: ["Use slow readable transitions."],
                  textureTendencies: ["Keep surfaces clean."],
                  spatialOrganization: ["Favor a stable focal point."],
                  runtimeSuitability: [
                    "Use the selected compatible runtime: p5.js."
                  ]
                },
                audioReactive: {
                  activation: "explicit_user_gesture",
                  audioRuntime: "Tone.js",
                  mappings: [
                    {
                      behavior:
                        "Smooth short peaks so scale and light remain readable.",
                      evidence: ["user prompt", "shader presets"],
                      intensity: "subtle",
                      source: "amplitude",
                      targets: ["scale", "glow"]
                    },
                    {
                      behavior:
                        "Quantize structural changes to the requested pulse or BPM.",
                      evidence: ["user prompt"],
                      intensity: "subtle",
                      source: "rhythm",
                      targets: ["rotation", "pattern_phase"]
                    }
                  ],
                  summary:
                    "amplitude -> scale / glow; rhythm -> rotation / pattern phase",
                  visualRuntime: "p5.js"
                }
              }
            }
          : artifact
      )
    });

    const translation = screen.getByRole("region", {
      name: "Creative translation summary"
    });

    expect(translation).toHaveAttribute("data-state", "available");
    expect(within(translation).getByText("Audiovisual")).toBeVisible();
    expect(within(translation).getByText("mandala")).toBeVisible();
    expect(within(translation).getByText("sacred geometry")).toBeVisible();
    expect(within(translation).getByText("rhythm")).toBeVisible();
    expect(within(translation).getByText("p5.js / Tone.js")).toBeVisible();
    expect(within(translation).getByText(/Preserve atmosphere/)).toBeVisible();
    const sacredGeometry = within(translation).getByRole("region", {
      name: "Sacred geometry guidance"
    });
    expect(sacredGeometry).toBeVisible();
    expect(
      within(sacredGeometry).getByText("mandala / radial symmetry")
    ).toBeVisible();
    expect(
      within(sacredGeometry).getByText("p5.js / GLSL / Tone.js")
    ).toBeVisible();
    const shaderPresets = within(translation).getByRole("region", {
      name: "Shader preset guidance"
    });
    expect(shaderPresets).toBeVisible();
    expect(
      within(shaderPresets).getByText("glow / kaleidoscopic symmetry")
    ).toBeVisible();
    expect(
      within(shaderPresets).getByText(
        "Use the selected compatible runtime: p5.js."
      )
    ).toBeVisible();
    const visualStyle = within(translation).getByRole("region", {
      name: "Visual style guidance"
    });
    expect(visualStyle).toBeVisible();
    expect(
      within(visualStyle).getByText("minimal / sacred geometry")
    ).toBeVisible();
    expect(
      within(visualStyle).getByText(
        "Use the selected compatible runtime: p5.js."
      )
    ).toBeVisible();
    const mapping = screen.getByRole("region", {
      name: "Audio-reactive mapping summary"
    });
    expect(mapping).toHaveAttribute("data-state", "available");
    expect(within(mapping).getByText("2 bounded links")).toBeVisible();
    expect(within(mapping).getByText("Scale / Glow")).toBeVisible();
    expect(
      within(mapping).getByText(/Audio remains silent until explicit start/)
    ).toBeVisible();
  });

  it("keeps optional creative guidance absent for legacy artifact metadata", () => {
    renderShell(snapshotWithActiveTab("Artifacts"));

    const translation = screen.getByRole("region", {
      name: "Creative translation summary"
    });

    expect(translation).toHaveAttribute("data-state", "legacy");
    expect(
      within(translation).queryByRole("region", {
        name: "Sacred geometry guidance"
      })
    ).not.toBeInTheDocument();
    expect(
      within(translation).queryByRole("region", {
        name: "Shader preset guidance"
      })
    ).not.toBeInTheDocument();
    expect(
      within(translation).queryByRole("region", {
        name: "Visual style guidance"
      })
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("region", {
        name: "Audio-reactive mapping summary"
      })
    ).toHaveAttribute("data-state", "legacy");
  });

  it("labels unsupported artifacts as code-only in the artifacts inspector", () => {
    const snapshot = snapshotWithActiveTab("Artifacts");
    const codeOnlySnapshot: AssistantWorkspaceSnapshot = {
      ...snapshot,
      artifacts: [
        {
          ...snapshot.artifacts[0],
          id: "webgpu-notes",
          title: "feedback-field.webgpu.ts",
          language: "JavaScript",
          status: "Generated",
          summary: "WebGPU code remains inspectable without live preview support.",
          content: "const device = await adapter.requestDevice();",
          domain: "webgpu",
          previewEligible: false,
          previewTarget: "",
          rendererId: null,
          runtime: null,
          actions: ["Open", "Copy", "Download"]
        }
      ],
      preview: {
        ...snapshot.preview,
        available: false,
        state: "unavailable",
        targetId: ""
      }
    };

    renderShell(codeOnlySnapshot);

    const details = screen.getByRole("group", { name: "Active artifact details" });

    expect(within(details).getAllByText("Code-only").length).toBeGreaterThan(0);
    expect(within(details).getAllByText("Webgpu").length).toBeGreaterThan(0);
    expect(within(details).getByText("Runtime")).toBeVisible();
    expect(
      within(details).queryByRole("button", {
        name: "Preview feedback-field.webgpu.ts"
      })
    ).not.toBeInTheDocument();
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
        name: "Download File aurora-field.p5.js"
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
      screen.getByText("aurora-field.p5.js downloaded.")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(within(events).getByText("Artifact Download Approval Requested")).toBeVisible();
    expect(within(events).getByText("Artifact Download Completed")).toBeVisible();
  });

  it("exports the current workspace bundle through the existing approval flow", async () => {
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:bundle")
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    });

    renderShell(snapshotWithActiveTab("Artifacts"));
    const notesArtifact = screen.getByLabelText("projection-notes.md artifact");

    fireEvent.click(
      within(notesArtifact).getByRole("button", {
        name: "Export Bundle projection-notes.md"
      })
    );

    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Export project bundle"
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Export bundle" }));
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(anchorClick).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByText("Project bundle exported.")).toBeVisible();

    const anchor = anchorClick.mock.instances[0] as unknown as HTMLAnchorElement;
    expect(anchor.download).toBe("local-nextjs-workspace-bundle.zip");

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const events = screen.getByRole("group", { name: "Workflow event trace" });

    expect(
      within(events).getByText("Project Bundle Export Approval Requested")
    ).toBeVisible();
    expect(within(events).getByText("Project Bundle Export Completed")).toBeVisible();
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
        name: "Download File aurora-field.p5.js"
      })
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Download file" }));
      await Promise.resolve();
    });

    expect(screen.getByText("Artifact transfer failed")).toBeVisible();
    expect(
      screen.getAllByText(
        "The workspace could not download aurora-field.p5.js."
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
      screen.getByRole("button", { name: "Copy aurora-field.p5.js" })
    );

    await waitFor(() => {
      expect(
        screen.getByText("aurora-field.p5.js copied to clipboard.")
      ).toBeVisible();
    });
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("function draw()")
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
    const timeline = within(workflowPanel).getByRole("group", {
      name: "Workflow timeline explorer"
    });
    expect(timeline).toBeVisible();
    expect(within(timeline).getByText("No workflow timeline yet")).toBeVisible();
    expect(screen.getByLabelText("Workflow execution summary")).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Workflow inspector progress" })
    ).toHaveAttribute("aria-valuetext", "8 of 14 workflow nodes reached");
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
    const skippedSteps = [
      "memory",
      "retrieval",
      "context_assembly",
      "prompt_input",
      "prompt_rendering"
    ];
    const completedAfterPreview = [
      "intake",
      "routing",
      "generation",
      "artifact_extraction",
      "preview_preparation",
      "review",
      "refinement"
    ];
    const backendStream = vi.fn(() =>
      streamEvents([
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:00Z",
          currentStep: "intake",
          eventType: "node_started",
          sequence: 0,
          step: "intake"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:01Z",
          completedSteps: ["intake"],
          currentStep: null,
          decisionReason: "request_received",
          eventType: "node_completed",
          phase: "completed",
          sequence: 1,
          step: "intake",
          transitionSource: "intake",
          transitionTarget: "routing"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:02Z",
          completedSteps: ["intake"],
          currentStep: "routing",
          eventType: "node_started",
          sequence: 2,
          step: "routing"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:03Z",
          completedSteps: ["intake", "routing"],
          currentStep: null,
          decisionReason: "route_selected:generate",
          eventType: "node_completed",
          phase: "completed",
          sequence: 3,
          step: "routing",
          transitionSource: "routing",
          transitionTarget: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:04Z",
          currentStep: "generation",
          eventType: "node_started",
          sequence: 4,
          skippedSteps,
          step: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:05Z",
          code: "generation_input_prepared",
          completedSteps: ["intake", "routing"],
          currentStep: "generation",
          eventType: "generation_input",
          message: "Provider generation input prepared.",
          sequence: 5,
          skippedSteps,
          step: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:06Z",
          completedSteps: ["intake", "routing"],
          currentStep: "generation",
          eventType: "token_delta",
          sequence: 6,
          skippedSteps,
          step: "generation",
          text: "draft"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:07Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: null,
          decisionReason: "generation_completed",
          eventType: "node_completed",
          phase: "completed",
          sequence: 7,
          skippedSteps,
          step: "generation",
          transitionSource: "generation",
          transitionTarget: "artifact_extraction"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:08Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: "artifact_extraction",
          eventType: "node_started",
          sequence: 8,
          skippedSteps,
          step: "artifact_extraction"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:09Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: null,
          decisionReason: "no_generated_artifacts",
          eventType: "node_completed",
          phase: "completed",
          sequence: 9,
          skippedSteps: [...skippedSteps, "artifact_extraction"],
          step: "artifact_extraction",
          transitionSource: "artifact_extraction",
          transitionTarget: "preview_preparation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:10Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: "preview_preparation",
          eventType: "node_started",
          sequence: 10,
          skippedSteps: [...skippedSteps, "artifact_extraction"],
          step: "preview_preparation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:11Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: null,
          decisionReason: "no_artifacts_for_preview",
          eventType: "node_completed",
          phase: "completed",
          sequence: 11,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "preview_preparation",
          transitionSource: "preview_preparation",
          transitionTarget: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:12Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: "review",
          eventType: "node_started",
          sequence: 12,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:13Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: "review",
          eventType: "review_failed",
          message: "Review failed: missing_code_block",
          reviewOutcome: "needs_refinement",
          refinementCount: 1,
          sequence: 13,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:14Z",
          completedSteps: ["intake", "routing", "generation"],
          currentStep: "review",
          eventType: "retry_started",
          message: "Retry 1 started: missing_code_block.",
          refinementCount: 1,
          retryCount: 1,
          retryReason: "missing_code_block",
          sequence: 14,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:15Z",
          completedSteps: ["intake", "routing", "generation", "review"],
          currentStep: null,
          decisionReason: "review_failed_retry_available",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          sequence: 15,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "review",
          transitionSource: "review",
          transitionTarget: "refinement"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:16Z",
          completedSteps: ["intake", "routing", "generation", "review"],
          currentStep: "refinement",
          eventType: "node_started",
          refinementCount: 1,
          sequence: 16,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "refinement"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:17Z",
          completedSteps: ["intake", "routing", "generation", "review"],
          currentStep: "refinement",
          eventType: "refinement_completed",
          message: "Refinement guidance prepared for retry 1.",
          refinementCount: 1,
          retryCount: 1,
          retryReason: "missing_code_block",
          sequence: 17,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "refinement"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:18Z",
          completedSteps: ["intake", "routing", "generation", "review", "refinement"],
          currentStep: null,
          decisionReason: "refinement_completed",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          sequence: 18,
          skippedSteps: [
            ...skippedSteps,
            "artifact_extraction",
            "preview_preparation"
          ],
          step: "refinement",
          transitionSource: "refinement",
          transitionTarget: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:19Z",
          completedSteps: ["intake", "routing", "review", "refinement"],
          currentStep: "generation",
          eventType: "node_started",
          refinementCount: 1,
          sequence: 19,
          skippedSteps,
          step: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:20Z",
          code: "generation_input_prepared",
          completedSteps: ["intake", "routing", "review", "refinement"],
          currentStep: "generation",
          eventType: "generation_input",
          message: "Provider generation input prepared.",
          refinementCount: 1,
          sequence: 20,
          skippedSteps,
          step: "generation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:21Z",
          completedSteps: ["intake", "routing", "review", "refinement"],
          currentStep: "generation",
          eventType: "token_delta",
          refinementCount: 1,
          sequence: 21,
          skippedSteps,
          step: "generation",
          text: "refined"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:22Z",
          completedSteps: ["intake", "routing", "generation", "review", "refinement"],
          currentStep: null,
          decisionReason: "generation_completed",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          sequence: 22,
          skippedSteps,
          step: "generation",
          transitionSource: "generation",
          transitionTarget: "artifact_extraction"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:23Z",
          completedSteps: ["intake", "routing", "generation", "review", "refinement"],
          currentStep: "artifact_extraction",
          eventType: "node_started",
          refinementCount: 1,
          sequence: 23,
          skippedSteps,
          step: "artifact_extraction"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:24Z",
          code: "artifact_extracted",
          completedSteps: ["intake", "routing", "generation", "review", "refinement"],
          currentStep: "artifact_extraction",
          eventType: "artifact_extracted",
          message: "Extracted 1 generated artifact from the answer.",
          refinementCount: 1,
          sequence: 24,
          skippedSteps,
          step: "artifact_extraction"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:25Z",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "artifact_extraction",
            "review",
            "refinement"
          ],
          currentStep: null,
          decisionReason: "artifacts_extracted",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          sequence: 25,
          skippedSteps,
          step: "artifact_extraction",
          transitionSource: "artifact_extraction",
          transitionTarget: "preview_preparation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:26Z",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "artifact_extraction",
            "review",
            "refinement"
          ],
          currentStep: "preview_preparation",
          eventType: "node_started",
          refinementCount: 1,
          sequence: 26,
          skippedSteps,
          step: "preview_preparation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:27Z",
          code: "preview_artifact_prepared",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "artifact_extraction",
            "review",
            "refinement"
          ],
          currentStep: "preview_preparation",
          eventType: "preview_artifact",
          message: "Preview runtime metadata prepared.",
          refinementCount: 1,
          sequence: 27,
          skippedSteps,
          step: "preview_preparation"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:28Z",
          completedSteps: completedAfterPreview,
          currentStep: null,
          decisionReason: "preview_metadata_prepared",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          sequence: 28,
          skippedSteps,
          step: "preview_preparation",
          transitionSource: "preview_preparation",
          transitionTarget: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:29Z",
          completedSteps: completedAfterPreview,
          currentStep: "review",
          eventType: "node_started",
          refinementCount: 1,
          sequence: 29,
          skippedSteps,
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:30Z",
          completedSteps: completedAfterPreview,
          currentStep: "review",
          eventType: "review_passed",
          message: "Review passed.",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 30,
          skippedSteps,
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:31Z",
          completedSteps: completedAfterPreview,
          currentStep: "review",
          eventType: "retry_completed",
          message: "Retry 1 passed.",
          refinementCount: 1,
          retryCount: 1,
          retryReason: "missing_code_block",
          reviewOutcome: "pass",
          sequence: 31,
          skippedSteps,
          step: "review"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:32Z",
          completedSteps: completedAfterPreview,
          currentStep: null,
          decisionReason: "review_passed",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 32,
          skippedSteps,
          step: "review",
          transitionSource: "review",
          transitionTarget: "finalization"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:33Z",
          completedSteps: completedAfterPreview,
          currentStep: "finalization",
          eventType: "node_started",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 33,
          skippedSteps,
          step: "finalization"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-22T10:00:34Z",
          completedSteps: [...completedAfterPreview, "finalization"],
          currentStep: null,
          decisionReason: "final_answer_emitted",
          eventType: "node_completed",
          phase: "completed",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 34,
          skippedSteps,
          status: "completed",
          step: "finalization",
          transitionSource: "finalization",
          transitionTarget: "end"
        }),
        runtimeWorkflowEvent({
          answer: "```ts\nconsole.log('refined');\n```",
          at: "2026-05-22T10:00:35Z",
          completedSteps: [...completedAfterPreview, "finalization"],
          currentStep: null,
          eventType: "final",
          phase: "completed",
          refinementCount: 1,
          reviewOutcome: "pass",
          sequence: 35,
          skippedSteps,
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
    const timeline = within(workflowPanel).getByRole("group", {
      name: "Workflow timeline explorer"
    });

    expect(within(retries).getByText("1 retry loop")).toBeVisible();
    expect(
      within(timeline).getByRole("list", {
        name: "Chronological workflow events"
      })
    ).toBeVisible();
    expect(
      within(timeline).getAllByText("Provider generation completed")
    ).toHaveLength(2);
    expect(within(timeline).getByText("Review needs refinement")).toBeVisible();
    expect(within(timeline).getByText("Final response")).toBeVisible();
    expect(
      within(timeline).getAllByText("Review Failed Retry Available").length
    ).toBeGreaterThan(0);
    expect(within(transitions).getByText("Review -> Refinement")).toBeVisible();
    expect(
      within(transitions).getByText("Review Failed Retry Available")
    ).toBeVisible();
    expect(within(transitions).getByText("Refinement -> Generation")).toBeVisible();
    expect(
      within(transitions).getAllByText("Generation -> Artifact extraction")
    ).toHaveLength(2);
    expect(
      within(transitions).getAllByText(
        "Artifact extraction -> Preview preparation"
      )
    ).toHaveLength(2);
    expect(within(events).getByText("Retry Completed")).toBeVisible();
    expect(within(events).getByText("Review Passed")).toBeVisible();
    expect(within(events).getByText("Final")).toBeVisible();
    expect(
      within(workflowGraph).getByText("Artifact extraction").closest("article")
    ).toHaveAttribute("data-state", "complete");
    expect(
      within(workflowGraph).getByText("Preview preparation").closest("article")
    ).toHaveAttribute("data-state", "complete");
    expect(
      within(workflowGraph).getByText("Refinement").closest("article")
    ).toHaveAttribute("data-state", "complete");
    expect(
      within(workflowGraph).getByText("Finalization").closest("article")
    ).toHaveAttribute("data-state", "complete");
  });

  it("renders provider telemetry in the overview and workflow inspector", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        runtimeWorkflowEvent({
          at: "2026-05-24T10:00:00Z",
          code: "generation_input_prepared",
          completedSteps: ["intake", "routing"],
          currentStep: "generation",
          eventType: "generation_input",
          message: "Provider generation input prepared.",
          sequence: 0,
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
          at: "2026-05-24T10:00:01Z",
          currentStep: "generation",
          eventType: "token_delta",
          sequence: 1,
          step: "generation",
          telemetry: {
            provider: {
              name: "openai",
              model: "gpt-5-mini"
            }
          },
          text: "Telemetry "
        }),
        runtimeWorkflowEvent({
          answer: "Telemetry answer.",
          at: "2026-05-24T10:00:02Z",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "review",
            "finalization"
          ],
          currentStep: null,
          eventType: "final",
          phase: "completed",
          reviewOutcome: "pass",
          sequence: 2,
          skippedSteps: [
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering"
          ],
          status: "completed",
          step: "finalization",
          telemetry: {
            execution: {
              generation_mode: "streaming",
              request_duration_ms: 780,
              retry_count: 0,
              streaming: true,
              streaming_status: "completed"
            },
            provider: {
              name: "openai",
              model: "gpt-5-mini",
              response_id: "resp_123"
            },
            token_usage: {
              input_tokens: 1200,
              output_tokens: 300,
              total_tokens: 1500,
              reasoning_tokens: 12
            },
            pricing: {
              input_usd_per_million_tokens: 0.25,
              output_usd_per_million_tokens: 2
            }
          }
        })
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate with telemetry." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Telemetry answer.")).toBeVisible();

    const overviewTelemetry = screen.getByRole("group", {
      name: "Telemetry summary"
    });
    expect(within(overviewTelemetry).getByText("$0.0009")).toBeVisible();
    expect(within(overviewTelemetry).getByText(/1,500 tokens/)).toBeVisible();
    expect(
      within(overviewTelemetry).getByText("openai / gpt-5-mini")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));

    const workflowPanel = screen.getByRole("tabpanel", {
      name: "Workflow inspector"
    });
    const tokenUsage = within(workflowPanel).getByRole("group", {
      name: "Workflow token usage"
    });
    const costEstimate = within(workflowPanel).getByRole("group", {
      name: "Workflow cost estimate"
    });
    const lifecycle = within(workflowPanel).getByRole("group", {
      name: "Generation telemetry lifecycle"
    });

    expect(within(tokenUsage).getByText("1,500")).toBeVisible();
    expect(within(tokenUsage).getByText(/1,200 input/)).toBeVisible();
    expect(within(costEstimate).getByText("$0.0009")).toBeVisible();
    expect(
      within(costEstimate).getByText("Estimated from pricing metadata")
    ).toBeVisible();
    expect(within(lifecycle).getByText("openai / gpt-5-mini")).toBeVisible();
    expect(within(lifecycle).getByText("Generation input")).toBeVisible();
    expect(within(lifecycle).getAllByText("First token").length).toBeGreaterThan(0);
  });

  it("renders the advanced telemetry dashboard without fake metrics", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        runtimeWorkflowEvent({
          at: "2026-05-24T10:00:00Z",
          code: "request_received",
          currentStep: "intake",
          eventType: "status",
          message: "Request accepted.",
          observability: {
            provider: "langsmith",
            trace_kind: "assistant_workflow",
            trace_id: "trace-local-123456",
            requested: true,
            enabled: false,
            project_name: "creative-local",
            reason: "missing_api_key",
            status: "disabled",
            tags: ["assistant", "workflow"]
          },
          sequence: 0,
          step: "intake"
        }),
        runtimeWorkflowEvent({
          at: "2026-05-24T10:00:01Z",
          currentStep: "generation",
          eventType: "token_delta",
          sequence: 1,
          step: "generation",
          telemetry: {
            provider: {
              name: "openai",
              model: "gpt-5-mini"
            }
          },
          text: "Telemetry "
        }),
        runtimeWorkflowEvent({
          at: "2026-05-24T10:00:02Z",
          code: "ragas_eval_completed",
          currentStep: "generation",
          evaluation: {
            dataset_id: "dataset-live-1",
            dry_run: true,
            metric_failures: 0,
            evaluation_type: "RAGAs live",
            metric_scores: {
              answer_relevancy: 0.91,
              artifact_quality: 0.86,
              context_precision: 0.84,
              faithfulness: 0.76,
              runtime_quality: 0.68
            },
            overall_score: 0.82,
            provider_calls_allowed: false,
            result_rows: 1,
            run_id: "eval-run-1",
            status: "Evaluation complete"
          },
          eventType: "eval_update",
          message: "Evaluation manifest ready.",
          sequence: 2,
          step: "generation"
        }),
        runtimeWorkflowEvent({
          answer: "Telemetry answer.",
          at: "2026-05-24T10:00:03Z",
          completedSteps: [
            "intake",
            "routing",
            "generation",
            "review",
            "finalization"
          ],
          currentStep: null,
          eventType: "final",
          phase: "completed",
          reviewOutcome: "pass",
          sequence: 3,
          skippedSteps: [
            "memory",
            "retrieval",
            "context_assembly",
            "prompt_input",
            "prompt_rendering"
          ],
          status: "completed",
          step: "finalization",
          telemetry: {
            execution: {
              generation_mode: "streaming",
              request_duration_ms: 780,
              retry_count: 0,
              streaming: true,
              streaming_status: "completed"
            },
            provider: {
              name: "openai",
              model: "gpt-5-mini",
              response_id: "resp_123"
            },
            token_usage: {
              input_tokens: 120,
              output_tokens: 30,
              total_tokens: 150
            }
          }
        })
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate with telemetry dashboard." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Telemetry answer.")).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Telemetry" }));

    const dashboard = screen.getByRole("tabpanel", {
      name: "Telemetry dashboard"
    });
    expect(dashboard).toBeVisible();
    expect(
      within(dashboard).getByRole("group", { name: "Stream lifecycle" })
    ).toBeVisible();
    expect(
      within(dashboard).getByRole("group", {
        name: "Provider observability deep dive"
      })
    ).toBeVisible();
    const providerDeepDive = within(dashboard).getByRole("group", {
      name: "Provider observability deep dive"
    });
    expect(within(providerDeepDive).getByText("Streaming generation")).toBeVisible();
    expect(within(providerDeepDive).getByText("780 ms")).toBeVisible();
    expect(within(providerDeepDive).getByText("0 retries")).toBeVisible();
    expect(within(providerDeepDive).getByText("Stream completed")).toBeVisible();
    const creativeCost = within(dashboard).getByRole("group", {
      name: "Creative cost intelligence dashboard"
    });
    expect(within(creativeCost).getByText("Session cost unavailable")).toBeVisible();
    expect(within(creativeCost).getByText("120 tokens")).toBeVisible();
    expect(within(creativeCost).getByText("30 tokens")).toBeVisible();
    expect(within(creativeCost).getByText("1 completed run")).toBeVisible();
    expect(
      within(dashboard).getByRole("group", { name: "Runtime lifecycle" })
    ).toBeVisible();
    expect(
      within(dashboard).getByRole("group", {
        name: "Renderer and preview health"
      })
    ).toBeVisible();
    expect(
      within(dashboard).getByRole("group", { name: "Retrieval activity" })
    ).toBeVisible();

    const langsmith = within(dashboard).getByRole("group", {
      name: "LangSmith trace deep dive"
    });
    const evaluation = within(dashboard).getByRole("group", {
      name: "Evaluation session dashboard"
    });

    expect(within(langsmith).getByText("Local trace metadata")).toBeVisible();
    expect(within(langsmith).getByLabelText("Trace status Complete")).toBeVisible();
    expect(within(langsmith).getByText("missing_api_key")).toBeVisible();
    expect(
      within(langsmith).getByRole("region", {
        name: "LangSmith trace hierarchy"
      })
    ).toBeVisible();
    expect(within(langsmith).getAllByText("Intake")).toHaveLength(2);
    expect(within(evaluation).getByText("Evaluation complete")).toBeVisible();
    expect(within(evaluation).getByText("82%")).toBeVisible();
    expect(
      within(evaluation).getByLabelText("Evaluation status Pass")
    ).toBeVisible();
    expect(within(evaluation).getByText("RAGAs live")).toBeVisible();
    expect(within(evaluation).getByText("eval-run-1")).toBeVisible();
    expect(within(evaluation).getByText("dataset-live-1")).toBeVisible();
    expect(
      within(evaluation).getByRole("group", {
        name: "Retrieval quality evaluation signal"
      })
    ).toHaveTextContent("84%");
    expect(
      within(dashboard).getByRole("group", { name: "Telemetry event type counts" })
    ).toBeVisible();
  });

  it("resizes workspace regions and persists the layout preferences", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(snapshotWithReadyPreview(), { persistenceClient });

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
    expect(preview.querySelector("details")).toHaveAttribute(
      "data-layout-size",
      "visual"
    );
    expect(preview.querySelector(".previewPanel")).toHaveStyle({
      height: "320px"
    });

    const previewHandle = screen.getByRole("separator", {
      name: "Resize preview shelf"
    });
    expect(previewHandle).toHaveAttribute("aria-disabled", "false");
    fireEvent.mouseDown(previewHandle, { clientY: 200 });
    fireEvent.mouseMove(window, { clientY: 260 });
    fireEvent.mouseUp(window);

    expect(previewHandle).toHaveAttribute("aria-valuenow", "380");
    expect(preview.querySelector(".previewPanel")).toHaveStyle({
      height: "380px"
    });

    fireEvent.click(screen.getByRole("button", { name: "Workspace density" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          layout: expect.objectContaining({
            density: "compact",
            inspectorCollapsed: false,
            inspectorWidth: 480,
            previewHeight: 380
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

    expect(screen.getByText("Stored locally")).toBeVisible();
  });
});
