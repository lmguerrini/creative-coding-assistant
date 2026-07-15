import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react";
import { StrictMode, type ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseRagasExecutionEvidence,
  WorkstationShell
} from "./workstation-shell";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot,
  type ArtifactSummary,
  type AssistantWorkspaceSnapshot,
  type InspectorTabName
} from "@/lib/assistant-client";
import type {
  AssistantStreamEvent,
  AssistantStreamRequest
} from "@/lib/assistant-stream";
import {
  createWorkspaceSessionRecord,
  type WorkspacePersistenceClient,
  type WorkspacePersistenceLoadResult,
  type WorkspacePersistenceSaveResult
} from "@/lib/workspace-persistence";
import { createWorkstationError } from "@/lib/workstation-errors";
import {
  CURRENT_PRODUCT_RETRIEVAL_CASE_IDS,
  CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
  currentProductRetrievalScoreFromEvidence
} from "@/lib/evaluation-benchmark";

const originalClipboard = navigator.clipboard;
const originalCancelAnimationFrame = window.cancelAnimationFrame;
const originalRequestAnimationFrame = window.requestAnimationFrame;
const originalCreateObjectURL = URL.createObjectURL;
const originalRevokeObjectURL = URL.revokeObjectURL;
const validPngBytes = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
]);
const evaluationHash = (character: string) => `sha256:${character.repeat(64)}`;
const canonicalEvaluationMetricScores = {
  context_precision: .88,
  faithfulness: .88,
  answer_relevancy: .88,
  context_relevancy: .88,
  context_recall: .88
};

function canonicalEvaluationCaseResults() {
  return CURRENT_PRODUCT_RETRIEVAL_CASE_IDS.map((caseId, index) => ({
    caseId,
    sourceIds: [`official-source-${index}`],
    domains: [`domain-${index}`],
    metrics: { ...canonicalEvaluationMetricScores },
    metricErrors: {},
    promptFingerprint: evaluationHash(String((index + 1) % 10)),
    generationFingerprint: evaluationHash(String((index + 2) % 10))
  }));
}

function canonicalEvaluationResult(overrides: Record<string, unknown> = {}) {
  return {
    schemaVersion: "current-product-ragas-evidence.v1",
    benchmarkMode: "current_product",
    scoreOrigin: "current_product",
    scope: "full",
    runId: "canonical-evaluation-result",
    status: "completed",
    benchmarkVersion: "current-product-retrieval.v1",
    datasetId: "creative_coding_retrieval_benchmark",
    datasetVersion: "current-product-retrieval.v1",
    selectedCaseIds: [...CURRENT_PRODUCT_RETRIEVAL_CASE_IDS],
    datasetFingerprint: CURRENT_PRODUCT_RETRIEVAL_DATASET_FINGERPRINT,
    retrievalFingerprint: evaluationHash("a"),
    promptFingerprint: evaluationHash("b"),
    generationFingerprint: evaluationHash("c"),
    outputFingerprint: evaluationHash("d"),
    selectionFingerprint: evaluationHash("e"),
    kbFingerprint: evaluationHash("f"),
    evaluatedAt: "2026-07-14T08:00:00.000Z",
    timestamp: "2026-07-14T08:00:00.000Z",
    metrics: [
      "context_precision",
      "faithfulness",
      "answer_relevancy",
      "context_relevancy",
      "context_recall"
    ],
    metricScores: { ...canonicalEvaluationMetricScores },
    retrievalScore: .88,
    resultRows: 7,
    totalSamples: 7,
    eligibleSamples: 7,
    skippedSamples: 0,
    metricFailures: 0,
    provider: "OpenAI",
    model: "gpt-5-mini",
    generationModel: "gpt-5-mini",
    evaluator: "OpenAI / gpt-5-mini",
    evaluatorModel: "gpt-5-mini",
    embeddingModel: "text-embedding-3-small",
    ragasVersion: "0.3.9",
    metricContract: "ragas-current-product-reference.v2",
    durationMs: 1_200,
    detail: "Current-product evaluation completed.",
    privacyClass: "public_official_contexts_with_authored_references",
    caseResults: canonicalEvaluationCaseResults(),
    ...overrides
  };
}

const testCreativePlan = {
  outputModality: "visual" as const,
  generationStrategy: "Generate one p5 candidate with a browser-safe runtime.",
  recommendedRuntime: "p5",
  recommendedRendererId: "surface.p5",
  recommendedPreviewTarget: "browser_sandbox",
  recommendedShaderStyle: "glow",
  candidateCount: 1,
  refinementBudget: 1,
  expectedComplexity: "medium" as const,
  estimatedTokenCost: 2800,
  exportReadiness: "ready" as const,
  runtimeAvailable: true,
  runtimeSupportSummary: "p5.js browser preview is available.",
  planSteps: ["Target p5 output.", "Preserve translated creative intent."],
  constraints: ["Keep code browser-safe."],
  evidence: ["Route selected: generate."]
};

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

function snapshotWithRefinedGlslPreview(): AssistantWorkspaceSnapshot {
  const snapshot = snapshotWithGlslPreview();
  const source = [
    "void main() {",
    "  vec2 uv = gl_FragCoord.xy / u_resolution.xy;",
    "  float field = sin((uv.x + u_time) * 8.0);",
    "  gl_FragColor = vec4(uv.x, field * 0.5 + 0.5, uv.y, 1.0);",
    "}"
  ].join("\n");
  const sourceArtifact = {
    ...snapshot.artifacts[0],
    id: "chladni-light-field-source",
    title: "chladni-light-field-2.frag",
    content: source,
    domain: "glsl",
    previewEligible: true,
    rendererId: "surface.glsl",
    runtime: "glsl"
  };
  const refinedArtifact = {
    ...sourceArtifact,
    id: "improve-performance-refined",
    title: "improve-performance.refined.frag",
    status: "Refined",
    refinedFromArtifactId: sourceArtifact.id,
    refinedFromTitle: sourceArtifact.title,
    refinementInstruction: "Improve performance"
  };

  return {
    ...snapshot,
    artifacts: [refinedArtifact, sourceArtifact, ...snapshot.artifacts.slice(1)],
    preview: {
      ...snapshot.preview,
      artifactName: refinedArtifact.title,
      sourceArtifactId: refinedArtifact.id,
      sourceArtifactName: refinedArtifact.title
    },
    code: {
      ...snapshot.code,
      title: refinedArtifact.title,
      excerpt: source.split("\n")
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
        content: [
          "const synth = new Tone.Synth({",
          "  envelope: { attack: 0.03, decay: 0.2, sustain: 0.4, release: 0.7 }",
          "}).toDestination();",
          "const delay = new Tone.FeedbackDelay('8n', 0.25).toDestination();",
          "new Tone.Sequence((time, note) => synth.triggerAttackRelease(note, '8n', time), ['C4', 'E4', 'G4', 'B4'], '8n').start(0);",
          "Tone.Transport.bpm.value = 104;",
          "Tone.Transport.start();"
        ].join("\\n"),
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

function snapshotWithGsapPreview(): AssistantWorkspaceSnapshot {
  const snapshot = getLocalWorkspaceSnapshot();
  const title = "signal-bloom.gsap.ts";

  return {
    ...snapshot,
    artifacts: [
      {
        ...snapshot.artifacts[0],
        content: [
          "const tl = gsap.timeline({ repeat: -1, yoyo: true });",
          "tl.to('.particle', { x: 140, rotation: 90, opacity: 0.35, stagger: 0.08 });",
          "tl.to('.ring', { scale: 1.2, duration: 1.4 }, 0);"
        ].join("\n"),
        domain: "gsap",
        language: "TypeScript + GSAP",
        rendererId: "surface.gsap",
        runtime: "gsap",
        summary: "GSAP timeline with bounded DOM transforms, stagger, and yoyo motion.",
        title
      },
      ...snapshot.artifacts.slice(1)
    ],
    preview: {
      ...snapshot.preview,
      artifactName: title,
      renderer: "surface.gsap",
      sourceArtifactName: title,
      summary: "Runtime is ready to mount a bounded GSAP motion preview.",
      target: "Browser preview / GSAP",
      targetId: "browser_sandbox"
    },
    code: {
      ...snapshot.code,
      excerpt: [
        "const tl = gsap.timeline({ repeat: -1, yoyo: true });",
        "tl.to('.particle', { x: 140, rotation: 90, opacity: 0.35, stagger: 0.08 });",
        "tl.to('.ring', { scale: 1.2, duration: 1.4 }, 0);"
      ],
      language: "TypeScript + GSAP",
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
      headline: "No retrieved context for this run.",
      detail:
        "No retrieved context for this run. No matching retrieval chunks were returned for this request.",
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
    writable: true,
    value: vi.fn(() => 1)
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    writable: true,
    value: vi.fn()
  });
}

function installAnimationFrameStepper() {
  const callbacks: FrameRequestCallback[] = [];

  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    writable: true,
    value: vi.fn((callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    })
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    writable: true,
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

function createEmptyPersistenceClient(): WorkspacePersistenceClient {
  return {
    load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
    save: vi.fn(async () => ({ error: null, target: "local" as const }))
  };
}

function renderShell(
  snapshot: AssistantWorkspaceSnapshot = getLocalWorkspaceSnapshot(),
  props: Partial<ComponentProps<typeof WorkstationShell>> = {},
  options: { mode?: "developer" | "user" } = {}
) {
  const result = render(
    <WorkstationShell
      snapshot={snapshot}
      persistenceClient={createNoopPersistenceClient()}
      {...props}
    />
  );

  if (options.mode !== "user") {
    openDeveloperInspector();
  }

  return result;
}

function renderUserShell(
  snapshot: AssistantWorkspaceSnapshot = getLocalWorkspaceSnapshot(),
  props: Partial<ComponentProps<typeof WorkstationShell>> = {}
) {
  const result = renderShell(snapshot, props);
  const displayMode = getWorkspaceSettingsControl("Display mode");
  if (displayMode.textContent?.includes("Developer")) {
    fireEvent.click(displayMode);
  }
  closeWorkspaceSettingsPanel();
  return result;
}

function openDeveloperInspector() {
  const displayMode = getWorkspaceSettingsControl("Display mode");
  if (displayMode.textContent?.includes("User")) {
    fireEvent.click(displayMode);
  }
  closeWorkspaceSettingsPanel();

  const expandInspector = screen.queryByRole("button", {
    name: "Expand inspector"
  });
  if (expandInspector) {
    fireEvent.click(expandInspector);
  }
}

function openWorkspaceSettingsPanel() {
  const currentPanel = screen.queryByRole("dialog", {
    name: "Workspace settings"
  });
  if (currentPanel) {
    return currentPanel;
  }

  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  return screen.getByRole("dialog", { name: "Workspace settings" });
}

function closeWorkspaceSettingsPanel() {
  if (
    screen.queryByRole("dialog", {
      name: "Workspace settings"
    })
  ) {
    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  }
}

function getWorkspaceSettingsControl(name: string) {
  return within(openWorkspaceSettingsPanel()).getByRole("button", { name });
}

function openDashboardKnowledgeBase() {
  const dashboard = screen.queryByRole("region", { name: "Advanced Dashboard" });
  if (!dashboard) {
    fireEvent.click(
      screen.getByRole("button", { name: "Open Product Intelligence Dashboard" })
    );
  }

  const navigation = screen.getByRole("navigation", {
    name: "Dashboard categories"
  });
  fireEvent.click(within(navigation).getByRole("button", { name: "Knowledge Base" }));
}

function openDashboardEvaluation() {
  const dashboard = screen.queryByRole("region", { name: "Advanced Dashboard" });
  if (!dashboard) {
    fireEvent.click(
      screen.getByRole("button", { name: "Open Product Intelligence Dashboard" })
    );
  }

  const navigation = screen.getByRole("navigation", {
    name: "Dashboard categories"
  });
  fireEvent.click(within(navigation).getByRole("button", { name: "Evaluation" }));
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
      writable: true,
      value: originalRequestAnimationFrame
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      writable: true,
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
    const settingsPanel = openWorkspaceSettingsPanel();
    expect(
      within(settingsPanel).getByRole("button", {
        name: "Toggle Fullscreen Creative Session from quick actions"
      })
    ).toBeVisible();
    expect(within(settingsPanel).getByRole("button", { name: "Display mode" })).toBeVisible();
    expect(
      within(settingsPanel).getByRole("group", {
        name: "Workspace density options"
      })
    ).toBeVisible();
    expect(screen.queryByRole("button", { name: "Command menu" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Theme" })).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Enter Fullscreen Creative Session" })
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
    expect(screen.getByRole("combobox", { name: "Workflow" })).toHaveValue("auto");
    expect(screen.queryByText("Type a prompt to begin")).not.toBeInTheDocument();
  });

  it("starts new workspaces in User Mode while keeping Developer Mode available", async () => {
    renderShell(getInitialWorkspaceSnapshot(), {}, { mode: "user" });

    const displayMode = getWorkspaceSettingsControl("Display mode");
    expect(displayMode).toHaveTextContent("User");
    expect(
      screen.getByRole("complementary", { name: "Right inspector" })
    ).toHaveAttribute("data-state", "collapsed");
    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();

    fireEvent.click(displayMode);

    expect(displayMode).toHaveTextContent("Developer");
    expect(screen.getByRole("tab", { name: "Workflow" })).toBeVisible();

    closeWorkspaceSettingsPanel();
    fireEvent.click(screen.getByRole("button", { name: "New session" }));

    await waitFor(() =>
      expect(
        screen.getByRole("complementary", { name: "Right inspector" })
      ).toHaveAttribute("data-state", "collapsed")
    );
    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent("User");
    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
  });

  it("opens the full Product Intelligence Dashboard from the workspace", () => {
    renderShell(getInitialWorkspaceSnapshot());

    fireEvent.click(
      screen.getByRole("button", { name: "Open Product Intelligence Dashboard" })
    );

    expect(
      screen.getByRole("region", { name: "Advanced Dashboard" })
    ).toBeVisible();
    expect(screen.queryByRole("region", { name: "Creative workspace" })).not.toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Dashboard categories" }))
      .toHaveTextContent("Telemetry");
    expect(screen.getByRole("navigation", { name: "Dashboard categories" }))
      .toHaveTextContent("Evaluation");
  });

  it("parses the backend canonical metric order as promotable current-product evidence", () => {
    const evidence = parseRagasExecutionEvidence(
      canonicalEvaluationResult(),
      {
        scope: "full",
        caseIds: [],
        allowProviderCalls: true,
        approvedRagasDataset: "sanitized_public"
      }
    );

    expect(evidence.scoreOrigin).toBe("current_product");
    expect(evidence.metrics).toEqual([
      "context_precision",
      "faithfulness",
      "answer_relevancy",
      "context_relevancy",
      "context_recall"
    ]);
    expect(currentProductRetrievalScoreFromEvidence(evidence)).toBe(.88);
  });

  it.each([
    ["schema", { schemaVersion: "current-product-ragas-evidence.v2" }],
    ["scope", { scope: "cases" }],
    ["privacy", { privacyClass: "public_official_evidence" }],
    ["dataset version", { datasetVersion: undefined }],
    ["generation model", { generationModel: undefined }],
    ["evaluator", { evaluator: undefined }],
    ["timestamp", { timestamp: undefined }],
    ["selected case IDs", {
      selectedCaseIds: [...CURRENT_PRODUCT_RETRIEVAL_CASE_IDS, 42]
    }],
    ["per-case metric errors", {
      caseResults: canonicalEvaluationCaseResults().map((row, index) => index === 0
        ? { ...row, metricErrors: undefined }
        : row)
    }]
  ])("downgrades canonical-looking evidence with an invalid %s contract", (_label, overrides) => {
    const evidence = parseRagasExecutionEvidence(
      canonicalEvaluationResult(overrides),
      {
        scope: "full",
        caseIds: [],
        allowProviderCalls: true,
        approvedRagasDataset: "sanitized_public"
      }
    );

    expect(evidence.scoreOrigin).toBe("unscored");
    expect(currentProductRetrievalScoreFromEvidence(evidence)).toBeNull();
  });

  it("keeps explicit historical fixtures History-only and rejects unknown benchmark modes", () => {
    const request = {
      scope: "full" as const,
      caseIds: [],
      allowProviderCalls: true,
      approvedRagasDataset: "sanitized_public" as const
    };
    const historical = parseRagasExecutionEvidence(canonicalEvaluationResult({
      benchmarkMode: "historical_fixture",
      scoreOrigin: "historical_fixture",
      metrics: ["context_precision", "faithfulness", "answer_relevancy", "context_relevancy"],
      metricScores: {
        context_precision: .8,
        faithfulness: .8,
        answer_relevancy: .8,
        context_relevancy: .8
      }
    }), request);
    const unknown = parseRagasExecutionEvidence(canonicalEvaluationResult({
      benchmarkMode: "future_benchmark_mode"
    }), request);

    expect(historical.benchmarkMode).toBe("historical_fixture");
    expect(historical.scoreOrigin).toBe("historical_fixture");
    expect(currentProductRetrievalScoreFromEvidence(historical)).toBeNull();
    expect(unknown.benchmarkMode).toBe("not_selected");
    expect(unknown.scoreOrigin).toBe("unscored");
    expect(currentProductRetrievalScoreFromEvidence(unknown)).toBeNull();
  });

  it("submits and records a canonical current-product evaluation while the repository anchor is absent", async () => {
    const runId = "async-current-product-1";
    let getCount = 0;
    const evaluationFetch = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Queued current-product benchmark",
            completedCases: 0,
            totalCases: 7,
            remainingCases: 7,
            percent: 0,
            executionState: "provider_authorized",
            detail: "The run is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        getCount += 1;
        if (getCount === 1) {
          return Promise.resolve(new Response(JSON.stringify({
            runId,
            status: "running",
            progress: {
              phase: "ragas_scoring",
              lane: "RAG / Retrieval",
              currentCaseId: "runtime_selection_hydra_vs_p5",
              currentCaseLabel: "Runtime selection for fast live visuals",
              completedCases: 1,
              totalCases: 7,
              remainingCases: 6,
              percent: 14,
              executionState: "provider_authorized",
              detail: "Scoring current retrieved contexts."
            }
          })));
        }
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "completed",
          progress: {
            phase: "evaluation",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Evaluation complete",
            completedCases: 7,
            totalCases: 7,
            remainingCases: 0,
            percent: 85,
            executionState: "provider_completed",
            detail: "Current-product evidence published."
          },
          result: canonicalEvaluationResult({ runId })
        })));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /current-product public benchmark/i }));
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(getCount).toBe(2), { timeout: 3_000 });
    const postCall = evaluationFetch.mock.calls.find(([, init]) => init?.method === "POST");
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toEqual({
      benchmarkMode: "current_product",
      scope: "full",
      caseIds: [],
      allowProviderCalls: true,
      approvedDataset: "sanitized_public",
      dryRun: false
    });
    expect(evaluationFetch).toHaveBeenCalledWith(
      expect.stringContaining(`/api/evaluation/run?runId=${runId}`),
      expect.objectContaining({ cache: "no-store", method: "GET" })
    );
    await waitFor(() => {
      expect(screen.getByLabelText("Evaluation benchmark summary")).toHaveTextContent("88.00%");
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("85% complete");
      expect(progress).toHaveTextContent("Full evaluation");
      expect(progress).toHaveTextContent("7 completed / 0 remaining of 7");
      expect(progress).toHaveTextContent("evaluation · completed");
      expect(progress).toHaveTextContent("seven canonical RAG cases plus current local workspace snapshots");
      expect(progress).not.toHaveTextContent("35");
    });
    expect(screen.getByLabelText("Current Retrieval Quality provenance")).toHaveTextContent(runId);
    const results = screen.getByLabelText("Evaluation results");
    expect(results.querySelectorAll(".evaluationCaseTable > details")).toHaveLength(4);
    expect(results).toHaveTextContent("Current-product RAGAS benchmark");
    expect(results).toHaveTextContent("Current workspace creative artifact evidence");
    expect(results).toHaveTextContent("Current workspace workflow evidence");
    expect(results).toHaveTextContent("Current workspace reliability evidence");
    expect(results).not.toHaveTextContent("NOT RUN");
    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    const history = screen.getByLabelText("Evaluation history and trends");
    expect(history).toHaveTextContent(runId);
    expect(history).toHaveTextContent("CURRENT PRODUCT");
  });

  it("keeps polling after a transient refresh failure and reconnects to the terminal snapshot", async () => {
    const runId = "async-reconnect-current-product-1";
    let getCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "Legacy full benchmark",
            currentCaseId: null,
            currentCaseLabel: "Queued evaluation",
            completedCases: 0,
            totalCases: 35,
            remainingCases: 35,
            percent: 0,
            executionState: "local_preflight",
            detail: "The run is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        getCount += 1;
        if (getCount === 1) {
          return Promise.reject(new TypeError("temporary connection loss"));
        }
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "prepared",
          progress: {
            phase: "terminal",
            lane: "Legacy full benchmark",
            currentCaseId: null,
            currentCaseLabel: "Dry-run preparation complete",
            completedCases: 35,
            totalCases: 35,
            remainingCases: 0,
            percent: 100,
            executionState: "local_preflight",
            detail: "Current-product preflight completed."
          },
          result: {
            benchmarkMode: "current_product",
            scoreOrigin: "unscored",
            scope: "full",
            runId,
            status: "prepared",
            benchmarkVersion: "current-product-retrieval.v1",
            metrics: [],
            metricScores: {},
            resultRows: 0,
            totalSamples: 7,
            eligibleSamples: 0,
            skippedSamples: 7,
            metricFailures: 0,
            detail: "Dry-run evidence only."
          }
        })));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(getCount).toBe(1));
    await waitFor(() => {
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("reconnecting · running");
      expect(progress).toHaveTextContent("Reconnecting to the evaluation service");
      expect(progress).toHaveTextContent("0 completed / 7 remaining of 7");
      expect(progress).toHaveTextContent("Full evaluation");
      expect(progress).toHaveTextContent("the server run remains active");
      expect(progress).not.toHaveTextContent("35");
    });

    await waitFor(() => expect(getCount).toBe(2), { timeout: 3_000 });
    await waitFor(() => {
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("terminal · prepared");
      expect(progress).toHaveTextContent("7 completed / 0 remaining of 7");
      expect(progress).not.toHaveTextContent("35");
    });
    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    expect(screen.getByLabelText("Evaluation history and trends")).toHaveTextContent(runId);
  });

  it.each([404, 410])(
    "terminates a permanently missing evaluation poll after HTTP %i and leaves the run action available",
    async (statusCode) => {
    const runId = `async-gone-${statusCode}-current-product-1`;
    let getCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Queued evaluation",
            completedCases: 0,
            totalCases: 7,
            remainingCases: 7,
            percent: 0,
            executionState: "provider_authorized",
            detail: "The run is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        getCount += 1;
        return Promise.resolve(new Response("gone", { status: statusCode }));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => {
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("terminal · failed");
      expect(progress).toHaveTextContent("polling failed");
      expect(progress).toHaveTextContent(`Evaluation run ${runId} is no longer available (HTTP ${statusCode}).`);
      expect(progress).toHaveTextContent("Use Run Evaluation to retry.");
    });
    expect(getCount).toBe(1);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Run Evaluation" })).toBeEnabled();
    }, { timeout: 3_000 });
  });

  it("stops after bounded persistent status-service failures and preserves retry UX", async () => {
    const runId = "async-persistent-5xx-current-product-1";
    let getCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Queued evaluation",
            completedCases: 0,
            totalCases: 7,
            remainingCases: 7,
            percent: 0,
            executionState: "provider_authorized",
            detail: "The run is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        getCount += 1;
        return Promise.resolve(new Response("temporarily unavailable", { status: 503 }));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(getCount).toBe(3), { timeout: 4_000 });
    await waitFor(() => {
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("terminal · failed");
      expect(progress).toHaveTextContent("polling failed");
      expect(progress).toHaveTextContent("status remained unavailable after 3 consecutive refresh attempts");
      expect(progress).toHaveTextContent("Use Run Evaluation to retry.");
    });
    expect(getCount).toBe(3);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Run Evaluation" })).toBeEnabled();
    }, { timeout: 3_000 });
  });

  it("aborts an active evaluation poll on unmount without persisting evaluation history", async () => {
    const runId = "async-unmount-current-product-1";
    const pollState: { signal: AbortSignal | null } = { signal: null };
    const persistenceClient = createEmptyPersistenceClient();
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Queued evaluation",
            completedCases: 0,
            totalCases: 7,
            remainingCases: 7,
            percent: 0,
            executionState: "local_preflight",
            detail: "The run is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        pollState.signal = init.signal ?? null;
        return new Promise<Response>((_resolve, reject) => {
          pollState.signal?.addEventListener("abort", () => {
            const abortError = new Error("Evaluation poll aborted");
            abortError.name = "AbortError";
            reject(abortError);
          }, { once: true });
        });
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    const view = renderShell(getInitialWorkspaceSnapshot(), { persistenceClient });
    await waitFor(() => expect(persistenceClient.load).toHaveBeenCalled());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    vi.mocked(persistenceClient.save).mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(pollState.signal).not.toBeNull());
    view.unmount();
    expect(pollState.signal?.aborted).toBe(true);
    await act(async () => {
      await Promise.resolve();
    });
    expect(vi.mocked(persistenceClient.save).mock.calls.every(
      ([record]) => (record.preferences?.evaluationHistory.length ?? 0) === 0
    )).toBe(true);
  });

  it("completes a non-RAG case selection locally without calling the Evaluation API", async () => {
    const evaluationFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("unexpected", { status: 500 })
    );

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("button", { name: /^Selected cases/ }));

    expect(screen.getByText("3 cases selected")).toBeVisible();
    expect(screen.queryByRole("checkbox", { name: /current-product public benchmark/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => {
      const progress = screen.getByLabelText("Live evaluation progress");
      expect(progress).toHaveTextContent("100% complete");
      expect(progress).toHaveTextContent("local snapshot completed · completed");
      expect(progress).toHaveTextContent("No retrieval, generation, or evaluator provider calls were made");
    });
    expect(evaluationFetch.mock.calls.some(([input]) => String(input).includes("/api/evaluation/run"))).toBe(false);

    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    const history = screen.getByLabelText("Evaluation history and trends");
    expect(history).toHaveTextContent("LOCAL WORKSPACE RUN");
    expect(history).toHaveTextContent("0/3 selected contracts observed");
  });

  it("filters a mixed case selection and keeps a complete-looking one-case diagnostic unscored", async () => {
    const canonicalCaseId = CURRENT_PRODUCT_RETRIEVAL_CASE_IDS[0];
    const runId = "mixed-canonical-subset-1";
    const evaluationFetch = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (init?.method === "POST" && url.includes("/api/evaluation/run")) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "queued",
          progress: {
            phase: "queued",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Queued canonical retrieval subset",
            completedCases: 0,
            totalCases: 1,
            remainingCases: 1,
            percent: 0,
            executionState: "provider_authorized",
            detail: "The canonical subset is queued."
          }
        }), { status: 202 }));
      }
      if (init?.method === "GET" && url.includes(`/api/evaluation/run?runId=${runId}`)) {
        return Promise.resolve(new Response(JSON.stringify({
          runId,
          status: "completed",
          progress: {
            phase: "terminal",
            lane: "RAG / Retrieval",
            currentCaseId: null,
            currentCaseLabel: "Canonical subset complete",
            completedCases: 1,
            totalCases: 1,
            remainingCases: 0,
            percent: 100,
            executionState: "provider_completed",
            detail: "One canonical retrieval case completed."
          },
          result: {
            benchmarkMode: "current_product",
            scoreOrigin: "unscored",
            scope: "cases",
            runId,
            status: "completed",
            benchmarkVersion: "current-product-retrieval.v1",
            datasetFingerprint: "sha256:dataset-mixed-1234567890",
            retrievalFingerprint: "sha256:retrieval-mixed-1234567890",
            promptFingerprint: "sha256:prompt-mixed-1234567890",
            generationFingerprint: "sha256:generation-mixed-1234567890",
            evaluatedAt: "2026-07-14T08:30:00.000Z",
            metrics: ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevancy"],
            metricScores: {
              faithfulness: .9,
              answer_relevancy: .9,
              context_precision: .9,
              context_recall: .9,
              context_relevancy: .9
            },
            resultRows: 1,
            totalSamples: 1,
            eligibleSamples: 1,
            skippedSamples: 0,
            metricFailures: 0,
            provider: "OpenAI",
            model: "gpt-5-mini",
            generationModel: "gpt-5-mini",
            evaluator: "OpenAI / gpt-5-mini",
            embeddingModel: "text-embedding-3-small",
            ragasVersion: "0.3.9",
            metricContract: "five-metric-equal-weight.v1",
            detail: "Diagnostic canonical subset completed.",
            caseResults: []
          }
        })));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });

    renderShell(getInitialWorkspaceSnapshot());
    openDashboardEvaluation();
    fireEvent.click(screen.getByRole("button", { name: "Configure run" }));
    fireEvent.click(screen.getByRole("button", { name: /^Selected cases/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Runtime selection for fast live visuals/i }));
    expect(screen.getByText("4 cases selected")).toBeVisible();
    fireEvent.click(screen.getByRole("checkbox", { name: /current-product public benchmark/i }));
    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));

    await waitFor(() => expect(evaluationFetch).toHaveBeenCalledWith(
      expect.stringContaining(`/api/evaluation/run?runId=${runId}`),
      expect.objectContaining({ method: "GET" })
    ));
    const postCall = evaluationFetch.mock.calls.find(([, init]) => init?.method === "POST");
    const postBody = JSON.parse(String(postCall?.[1]?.body));
    expect(postBody.caseIds).toEqual([canonicalCaseId]);
    expect(postBody.caseIds).toHaveLength(1);

    fireEvent.click(screen.getByText("Comparable stored runs", { selector: "summary" }));
    await waitFor(() => {
      const history = screen.getByLabelText("Evaluation history and trends");
      expect(history).toHaveTextContent(runId);
      expect(history).toHaveTextContent("0/4 selected contracts observed");
      expect(history).toHaveTextContent("UNSCORED CURRENT-PRODUCT RUN");
    });
    const retrievalEvaluation = screen.getByLabelText("RAGAS retrieval evaluation");
    expect(retrievalEvaluation).toHaveTextContent("68.03%");
    expect(retrievalEvaluation).toHaveTextContent("current-product-public-retained");
    expect(retrievalEvaluation).not.toHaveTextContent("90.00%");
  });

  it("lets a user inspect a Demo scenario before loading and running its prompt", () => {
    renderShell(getInitialWorkspaceSnapshot());

    expect(screen.queryByRole("region", { name: "Demo Mode" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Demo Mode" })).toHaveAttribute(
      "aria-expanded",
      "false"
    );

    fireEvent.click(screen.getByRole("button", { name: "Demo Mode" }));

    const demoMode = screen.getByRole("region", { name: "Demo Mode" });
    const demoScenarioList = within(demoMode).getByRole("list", {
      name: "Demo Mode scenarios"
    });
    expect(demoMode).toBeVisible();
    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent(
      "Developer"
    );
    expect(screen.getByRole("button", { name: "Demo Mode" })).toHaveAttribute(
      "aria-expanded",
      "true"
    );
    expect(
      within(demoMode).getByRole("heading", {
        level: 1,
        name: "Creative scenarios"
      })
    ).toBeVisible();
    expect(
      within(demoMode).getByRole("region", { name: "Featured demo paths" })
    ).toBeVisible();
    expect(
      within(demoMode).getByRole("region", { name: "Demo scenario workspace" })
    ).toBeVisible();
    expect(within(demoScenarioList).getAllByRole("listitem")).toHaveLength(10);
    expect(
      within(demoScenarioList).getByRole("button", {
        name: /Recursive aurora garden/
      })
    ).toBeVisible();
    expect(within(demoMode).getByText("10 flows")).toBeVisible();

    fireEvent.click(
      within(demoScenarioList).getByRole("button", {
        name: /Failure recovery/
      })
    );
    expect(
      within(demoMode).getByRole("article", {
        name: "Selected demo scenario Failure recovery"
      })
    ).toBeVisible();

    fireEvent.click(
      within(demoScenarioList).getByRole("button", {
        name: /Recursive aurora garden/
      })
    );

    expect(screen.getByRole("region", { name: "Demo Mode" })).toBeVisible();
    expect(
      within(demoMode).getAllByText("Recursive aurora garden").length
    ).toBeGreaterThan(1);
    const demoWorkflow = within(demoMode).getByText("Demo workflow").closest("details");
    expect(demoWorkflow).not.toHaveAttribute("open");
    fireEvent.click(within(demoMode).getByText("Demo workflow"));
    expect(demoWorkflow).toHaveAttribute("open");
    fireEvent.click(within(demoMode).getByRole("button", { name: /Load prompt & run/ }));

    expect(screen.queryByRole("region", { name: "Demo Mode" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Demo Mode" })).toHaveAttribute(
      "aria-expanded",
      "false"
    );
    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent(
      "Developer"
    );
  });

  it("forces the multi-agent Demo preview open and recovers unsupported provider source", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer: "The multi-agent orbit study is ready.",
            artifacts: [
              {
                id: "multi-agent-orbit-demo",
                title: "multi-agent-orbit-study-2.p5.js",
                language: "JavaScript + p5.js",
                runtime: "p5",
                content: [
                  "function setup() { createCanvas(640, 360); }",
                  "function draw() { createGraphics(20, 20); }"
                ].join("\n"),
                preview_eligible: true
              }
            ]
          }
        }
      ])
    );

    renderShell(getInitialWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    const settingsPanel = screen.getByRole("dialog", { name: "Workspace settings" });
    fireEvent.click(
      within(settingsPanel).getByRole("button", { name: "Preview auto-open" })
    );
    closeWorkspaceSettingsPanel();

    fireEvent.click(screen.getByRole("button", { name: "Demo Mode" }));
    const demoMode = screen.getByRole("region", { name: "Demo Mode" });
    const demoScenarioList = within(demoMode).getByRole("list", {
      name: "Demo Mode scenarios"
    });
    fireEvent.click(
      within(demoScenarioList).getByRole("button", {
        name: /Multi-agent production plan/
      })
    );
    fireEvent.click(
      within(demoMode).getByRole("button", { name: /Load prompt & run/ })
    );

    await waitFor(() =>
      expect(backendStream).toHaveBeenCalledWith(
        expect.objectContaining({
          query: expect.stringContaining("multi-agent-orbit-study.p5.js"),
          workflowMode: "multi_agent"
        })
      )
    );
    const preview = await screen.findByRole("region", {
      name: "Preview workspace"
    });
    expect(preview.querySelector("details")).toHaveAttribute("open");
    expect(within(preview).getByText("Preview open", { selector: "summary small" })).toBeVisible();
    expect(within(preview).getByText("P5 sketch surface")).toBeVisible();
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "multi-agent-orbit-study.p5.js"
    );
  });

  it("requires an image reference before the reference-guided Demo can run", async () => {
    renderShell(getInitialWorkspaceSnapshot());

    fireEvent.click(screen.getByRole("button", { name: "Demo Mode" }));
    const demoMode = screen.getByRole("region", { name: "Demo Mode" });
    const demoScenarioList = within(demoMode).getByRole("list", {
      name: "Demo Mode scenarios"
    });
    fireEvent.click(
      within(demoScenarioList).getByRole("button", {
        name: /Reference-guided palette study/
      })
    );

    const loadButton = within(demoMode).getByRole("button", {
      name: "Attach image to run"
    });
    expect(loadButton).toBeDisabled();
    expect(loadButton).toHaveAttribute(
      "aria-describedby",
      "demo-image-required-notice"
    );
    expect(document.getElementById("demo-image-required-notice")).toHaveTextContent(
      "Image reference required"
    );
    expect(demoMode).toHaveTextContent(
      "Add one image reference through the composer before running this demo."
    );

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    fireEvent.change(screen.getByLabelText("Upload image attachment"), {
      target: {
        files: [new File([validPngBytes], "palette.png", { type: "image/png" })]
      }
    });

    await waitFor(() =>
      expect(
        within(demoMode).getByRole("button", { name: "Load prompt & run" })
      ).toBeEnabled()
    );
  });

  it("keeps the User Mode inspector collapsed and limited to essential tabs", async () => {
    renderUserShell(getInitialWorkspaceSnapshot());

    fireEvent.click(screen.getByRole("button", { name: "Demo Mode" }));

    const userDemoMode = screen.getByRole("region", { name: "Demo Mode" });
    expect(
      within(userDemoMode).queryByRole("region", { name: "Featured demo paths" })
    ).not.toBeInTheDocument();
    expect(within(userDemoMode).queryByText("Technical contract")).not.toBeInTheDocument();
    expect(within(userDemoMode).getByText("Demo workflow").closest("details"))
      .toHaveAttribute("open");

    await waitFor(() =>
      expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
        "data-state",
        "collapsed"
      )
    );
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Demo ready"
    );
    expect(screen.getByLabelText("Current session")).not.toHaveTextContent(
      /Provider|Usage|Telemetry|Finalization|Workflow/i
    );

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));

    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Preview",
      "Code",
      "Saved"
    ]);
    expect(screen.queryByRole("tab", { name: "Overview" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Runtime" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Telemetry" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Retrieval" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Artifacts" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Add Inspector tab" }));

    const picker = screen.getByRole("menu", {
      name: "Available Inspector tabs"
    });
    expect(within(picker).getByRole("menuitem", { name: "Settings" })).toBeVisible();
    expect(
      within(picker).queryByRole("menuitem", { name: "Product Bugs" })
    ).not.toBeInTheDocument();
  });

  it("shows a compact User Mode fallback when preview is not ready", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    renderUserShell({
      ...snapshot,
      preview: {
        ...snapshot.preview,
        active: false,
        state: "unavailable",
        status: "Waiting for runnable artifact"
      }
    });

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(within(preview).getByText("Preview unavailable")).toBeVisible();
    expect(within(preview).queryByText(/runtime/i)).not.toBeInTheDocument();

    fireEvent.click(within(preview).getByRole("button", { name: "Open Code" }));
    expect(screen.getByRole("tab", { name: "Code" })).toHaveAttribute(
      "aria-selected",
      "true"
    );

    fireEvent.click(within(preview).getByRole("button", { name: "Open Saved" }));
    expect(screen.getByRole("tab", { name: "Saved" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
  });

  it("uses human artifact names for Hydra outputs in User Mode", async () => {
    renderUserShell(snapshotWithHydraPreview());

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));
    fireEvent.click(screen.getByRole("tab", { name: "Saved" }));

    const savedPanel = screen.getByRole("tabpanel", {
      name: "Saved"
    });
    expect(within(savedPanel).getAllByText("Hydra Pattern").length).toBeGreaterThan(0);
    expect(
      within(savedPanel).getByRole("region", {
        name: "Selected artifact refinement"
      })
    ).toBeVisible();
    expect(
      within(savedPanel).getByRole("button", { name: "Apply refinement" })
    ).toBeVisible();
    expect(within(savedPanel).queryByText("feedback-lattice.hydra.js")).not.toBeInTheDocument();
  });

  it("disambiguates repeated User Mode saved output labels without filenames", () => {
    const snapshot = snapshotWithP5Preview();
    renderUserShell({
      ...snapshot,
      artifacts: [
        snapshot.artifacts[0],
        {
          ...snapshot.artifacts[0],
          id: "second-p5-sketch",
          title: "second-orbit-field.p5.js",
          summary: "Second browser-safe p5 sketch."
        },
        ...snapshot.artifacts.slice(1)
      ]
    });

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));
    fireEvent.click(screen.getByRole("tab", { name: "Saved" }));

    const savedPanel = screen.getByRole("tabpanel", {
      name: "Saved"
    });
    expect(within(savedPanel).getAllByText("P5 Sketch 1").length).toBeGreaterThan(0);
    expect(within(savedPanel).getByText("P5 Sketch 2")).toBeVisible();
    expect(within(savedPanel).queryByText("signal-orbit.p5.ts")).not.toBeInTheDocument();
    expect(within(savedPanel).queryByText("second-orbit-field.p5.js")).not.toBeInTheDocument();
  });

  it("renders a polished first-run workspace without demo or infrastructure noise", async () => {
    renderUserShell(getInitialWorkspaceSnapshot(), {
      persistenceClient: createEmptyPersistenceClient()
    });

    const emptyWorkspace = await screen.findByRole("group", {
      name: "Empty creative workspace"
    });
    expect(emptyWorkspace).toBeVisible();
    expect(screen.getByLabelText("Creative session overview")).not.toHaveAttribute(
      "role",
      "log"
    );
    expect(within(emptyWorkspace).getByText("New creative session")).toBeVisible();
    expect(screen.getByText("Describe the creative system you want to build.")).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Assistant prompt" })).toHaveAttribute(
      "placeholder",
      "Describe the visual, audio, or interactive experience you want to create."
    );
    expect(screen.getByText("Build browser-native visuals")).toBeVisible();
    expect(screen.getByText("Ground answers in official sources")).toBeVisible();
    expect(screen.getByText("Preview, refine, and save artifacts")).toBeVisible();
    expect(screen.getByText("Support creative-coding workflows")).toBeVisible();
    expect(screen.queryByText("Tone.js")).not.toBeInTheDocument();
    expect(screen.queryByText("Describe a visual system")).not.toBeInTheDocument();
    expect(screen.queryByText("Generate browser-safe code")).not.toBeInTheDocument();
    expect(screen.queryByText("Ways to work")).not.toBeInTheDocument();
    expect(screen.getByText("How it works")).toBeVisible();
    const promptSuggestions = screen.getByLabelText("Prompt suggestions");
    const starterCards = within(promptSuggestions).getAllByRole("button");
    expect(starterCards).toHaveLength(4);
    for (const starterCard of starterCards) {
      expect(starterCard).toHaveAttribute("data-has-icon", "false");
      expect(starterCard.querySelector(".dashboardActionCardIcon")).toBeNull();
    }
    const kineticStarter = screen.getByRole("button", {
      name: "Kinetic orbit sculpture"
    });
    expect(kineticStarter).toBeVisible();
    expect(kineticStarter).toHaveAccessibleDescription(
      /Three\.js browser preview.*luminous spatial study/i
    );
    expect(screen.getByRole("list", { name: "Creative session capabilities" })).toBeVisible();
    expect(screen.queryByRole("button", { name: /Hydra feedback/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Workflows")).not.toBeInTheDocument();
    expect(screen.queryByText(/aurora/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Session persistence issue")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Selected artifact refinement" })
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Ready");
    expect(screen.getByLabelText("Current session")).not.toHaveTextContent("Start a prompt");
    expect(screen.getByLabelText("Active artifact")).not.toBeVisible();
    expect(
      screen.queryByRole("heading", { name: "Start a creative coding session" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: "Workflow summary" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: "Session intelligence summary" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("How it works"));
    expect(
      within(screen.getByRole("list", { name: "Creative workflow" })).getAllByRole(
        "listitem"
      )
    ).toHaveLength(5);

    fireEvent.click(
      screen.getByRole("button", {
        name: "Physarum drift"
      })
    );
    expect(
      (screen.getByLabelText("Assistant prompt") as HTMLTextAreaElement).value
    ).toContain("physarum-drift.p5.js");
    await waitFor(() => {
      expect(screen.getByLabelText("Assistant prompt")).toHaveFocus();
    });
  });

  it("keeps the first-run Homepage hidden while persistence restoration is pending", () => {
    renderUserShell(getInitialWorkspaceSnapshot(), {
      persistenceClient: createNoopPersistenceClient()
    });

    expect(
      screen.queryByRole("group", { name: "Empty creative workspace" })
    ).not.toBeInTheDocument();
    expect(screen.getByRole("log", { name: "Conversation" })).toBeInTheDocument();
    expect(screen.getByText("Restoring session")).toBeVisible();
  });

  it("keeps the User Mode composer minimal while preserving prompt controls", () => {
    renderUserShell(getInitialWorkspaceSnapshot());

    const promptInput = screen.getByRole("textbox", {
      name: "Assistant prompt"
    });
    const composer = promptInput.closest("form");
    const attachButton = screen.getByRole("button", { name: "Add attachment" });
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(composer).toHaveAttribute("data-mode", "user");
    expect(attachButton).toBeVisible();
    expect(sendButton).toBeVisible();
    expect(sendButton).toBeDisabled();
    expect(screen.getByLabelText("Generation controls")).toBeVisible();

    expect(screen.getByText("AI Providers")).toBeVisible();
    fireEvent.click(screen.getByLabelText("Selected AI provider: OpenAI"));
    expect(screen.getByLabelText("Selected AI provider")).toHaveTextContent(
      "OpenAI"
    );
    expect(screen.queryByText("Type a prompt to begin")).not.toBeInTheDocument();

    fireEvent.change(promptInput, {
      target: { value: "Create a soft p5 particle sketch." }
    });

    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.queryByText("Ready to generate")).not.toBeInTheDocument();
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

    renderUserShell(getInitialWorkspaceSnapshot(), { persistenceClient });

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
    expect(
      screen.queryByRole("group", { name: "Workflow summary" })
    ).not.toBeInTheDocument();
  });

  it("keeps the runtime console quiet on first run", () => {
    renderShell(getInitialWorkspaceSnapshot());

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    const runtimePanel = screen.getByRole("tabpanel", {
      name: "Runtime"
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

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.getByRole("dialog", { name: "Workspace settings" })).toBeVisible();
    expect(screen.getByRole("group", { name: "Quick actions" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Theme" }));
    expect(screen.queryByRole("dialog", { name: "Workspace settings" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Theme presets" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.queryByRole("dialog", { name: "Theme presets" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Workspace settings" })).toBeVisible();
  });

  it("focuses utility panels, restores their triggers, and keeps Settings compact", async () => {
    renderShell();

    const settingsTrigger = screen.getByRole("button", { name: "Settings" });
    fireEvent.click(settingsTrigger);
    const settingsPanel = screen.getByRole("dialog", {
      name: "Workspace settings"
    });

    await waitFor(() => expect(settingsPanel).toHaveFocus());
    expect(
      within(settingsPanel).getByRole("button", {
        name: /Open Dashboard Settings/
      })
    ).toBeVisible();
    for (const duplicatedDetail of [
      "Theme and colour",
      "Comfortable reading",
      "Privacy contract",
      "Stored preference signals"
    ]) {
      expect(within(settingsPanel).queryByText(duplicatedDetail)).not.toBeInTheDocument();
    }

    fireEvent.keyDown(settingsPanel, { key: "Escape" });
    expect(
      screen.queryByRole("dialog", { name: "Workspace settings" })
    ).not.toBeInTheDocument();
    await waitFor(() => expect(settingsTrigger).toHaveFocus());

    const themeTrigger = screen.getByRole("button", { name: "Theme" });
    fireEvent.click(themeTrigger);
    const themePanel = screen.getByRole("dialog", { name: "Theme presets" });

    await waitFor(() => expect(themePanel).toHaveFocus());
    fireEvent.keyDown(themePanel, { key: "Escape" });
    expect(
      screen.queryByRole("dialog", { name: "Theme presets" })
    ).not.toBeInTheDocument();
    await waitFor(() => expect(themeTrigger).toHaveFocus());
  });

  it("renders available workspace theme presets and applies them", () => {
    renderShell();

    for (const [label, theme] of [
      ["Aqua", "aqua"],
      ["Deep Blue", "codex"],
      ["Dark", "codex_white"],
      ["Light", "light"],
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

    expect(document.documentElement).toHaveAttribute("data-cca-theme", "blueprint");
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
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(screen.getByRole("tabpanel", { name: "Overview" })).toBeVisible();
  });

  it("supports focus mode and density toggles without changing the data flow", () => {
    const { container } = renderShell();
    const workstation = container.querySelector(".workstation");

    expect(workstation).toHaveAttribute("data-density", "cozy");

    fireEvent.click(getWorkspaceSettingsControl("Compact"));
    expect(workstation).toHaveAttribute("data-density", "compact");

    const focusMode = getWorkspaceSettingsControl(
      "Toggle Fullscreen Creative Session from quick actions"
    );
    fireEvent.click(focusMode);

    expect(
      getWorkspaceSettingsControl(
        "Toggle Fullscreen Creative Session from quick actions"
      )
    ).toHaveTextContent("Exit Fullscreen Creative Session");
    expect(workstation).toHaveAttribute("data-focus-mode", "true");
    expect(workstation).toHaveAttribute("data-sidebar-state", "collapsed");
    expect(workstation).toHaveAttribute("data-inspector-state", "collapsed");
    expect(screen.queryByRole("complementary", { name: "Right inspector" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Preview workspace" })).not.toBeInTheDocument();

    fireEvent.click(
      getWorkspaceSettingsControl(
        "Toggle Fullscreen Creative Session from quick actions"
      )
    );

    expect(workstation).toHaveAttribute("data-focus-mode", "false");
    expect(workstation).toHaveAttribute("data-sidebar-state", "open");
    expect(workstation).toHaveAttribute("data-inspector-state", "open");
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Preview workspace" })).toBeVisible();
  });

  it("updates the complete workflow graph from the selected route", () => {
    renderShell();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    const selector = screen.getByRole("combobox", { name: "Workflow" });
    const graph = screen.getByLabelText("Full live workflow");

    fireEvent.change(selector, { target: { value: "single_agent" } });
    expect(selector).toHaveValue("single_agent");
    expect(within(graph).queryByText("Planning")).not.toBeInTheDocument();
    expect(within(graph).queryByText("Review")).not.toBeInTheDocument();
    expect(within(graph).getByText("Generation")).toBeVisible();

    fireEvent.change(selector, { target: { value: "multi_agent" } });
    expect(selector).toHaveValue("multi_agent");
    expect(within(graph).getByText("Planning")).toBeVisible();
    expect(within(graph).getByText("Review")).toBeVisible();
    expect(within(graph).getByText("Generation")).toBeVisible();
  });

  it("uses one persisted workflow default in the composer and Dashboard", async () => {
    const persistenceClient = createEmptyPersistenceClient();
    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });
    expect(await screen.findByText("Local session ready")).toBeVisible();

    fireEvent.change(screen.getByRole("combobox", { name: "Workflow" }), {
      target: { value: "multi_agent" }
    });

    await waitFor(() =>
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          preferences: expect.objectContaining({ workflowMode: "multi_agent" })
        })
      )
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Open Product Intelligence Dashboard" })
    );
    const navigation = screen.getByRole("navigation", {
      name: "Dashboard categories"
    });
    fireEvent.click(within(navigation).getByRole("button", { name: "Settings" }));

    expect(screen.getByRole("combobox", { name: "Default workflow" })).toHaveValue(
      "multi_agent"
    );
  });

  it("keeps the workspace-clear checkpoint visible and restores focus across outcomes", async () => {
    const persistenceClient = createEmptyPersistenceClient();
    renderShell(getLocalWorkspaceSnapshot(), { persistenceClient });
    await screen.findByText("Local session ready");

    const settingsTrigger = screen.getByRole("button", { name: "Settings" });
    const requestWorkspaceClear = () => {
      const settingsPanel = openWorkspaceSettingsPanel();
      const clearButton = within(settingsPanel).getByRole("button", {
        name: "Clear workspace session"
      });
      clearButton.focus();
      fireEvent.click(clearButton);
      return screen.getByLabelText("Operator checkpoint");
    };

    let checkpoint = requestWorkspaceClear();
    expect(checkpoint).toBeVisible();
    await waitFor(() => expect(checkpoint).toHaveFocus());

    fireEvent.click(
      within(checkpoint).getByRole("button", { name: "Keep session" })
    );
    await waitFor(() => {
      expect(checkpoint).toHaveAttribute("data-state", "rejected");
      expect(checkpoint).toHaveFocus();
    });

    fireEvent.click(
      within(checkpoint).getByRole("button", {
        name: "Dismiss operator checkpoint"
      })
    );
    expect(screen.queryByLabelText("Operator checkpoint")).not.toBeInTheDocument();
    await waitFor(() => expect(settingsTrigger).toHaveFocus());

    checkpoint = requestWorkspaceClear();
    await waitFor(() => expect(checkpoint).toHaveFocus());
    await act(async () => {
      fireEvent.click(
        within(checkpoint).getByRole("button", { name: "Clear workspace" })
      );
      await Promise.resolve();
    });

    expect(
      await screen.findByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    expect(screen.getByLabelText("Creative session overview")).toBeVisible();
    checkpoint = screen.getByLabelText("Operator checkpoint");
    await waitFor(() => {
      expect(checkpoint).toBeVisible();
      expect(checkpoint).toHaveAttribute("data-state", "completed");
      expect(checkpoint).toHaveFocus();
    });

    fireEvent.click(
      within(checkpoint).getByRole("button", {
        name: "Dismiss operator checkpoint"
      })
    );
    await waitFor(() => expect(settingsTrigger).toHaveFocus());
  });

  it("clears the workspace session only after operator approval", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockImplementation(() => true);
    renderShell(getLocalWorkspaceSnapshot(), {
      persistenceClient: createEmptyPersistenceClient()
    });
    expect(await screen.findByText("Local session ready")).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");
    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear workspace session" }));

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Clear workspace session"
    );
    expect(screen.getByRole("button", { name: "Keep session" })).toBeVisible();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear workspace" }));
      await Promise.resolve();
    });

    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent("User");
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );
    expect(
      await screen.findByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Preview workspace" })
    ).not.toBeInTheDocument();

    openDeveloperInspector();
    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    expect(
      within(workflowPanel).getByRole("group", {
        name: "LangGraph workflow visualization"
      })
    ).toBeVisible();
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
  });

  it("clears Demo Mode and composer state without breaking the workspace", async () => {
    renderShell(getLocalWorkspaceSnapshot(), {
      persistenceClient: createEmptyPersistenceClient()
    });
    expect(await screen.findByText("Local session ready")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Demo Mode" }));
    const demoMode = screen.getByRole("region", { name: "Demo Mode" });
    const demoScenarioList = within(demoMode).getByRole("list", {
      name: "Demo Mode scenarios"
    });
    fireEvent.click(
      within(demoScenarioList).getByRole("button", {
        name: /Recursive aurora garden/
      })
    );

    expect(screen.getByRole("region", { name: "Demo Mode" })).toBeVisible();
    expect(
      within(demoMode).getAllByText("Recursive aurora garden").length
    ).toBeGreaterThan(1);

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear workspace session" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear workspace" }));
      await Promise.resolve();
    });

    expect(screen.queryByRole("region", { name: "Demo Mode" })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue("");
    expect(screen.getByRole("button", { name: "Demo Mode" })).toBeVisible();
    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent("User");
    expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );
    expect(
      await screen.findByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    await waitFor(() => {
      expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
      expect(
        screen.queryByRole("region", { name: "Preview workspace" })
      ).not.toBeInTheDocument();
      expect(screen.getByLabelText("Active artifact")).not.toBeVisible();
    });
  });

  it("shows the compact Overview cockpit in Developer Mode", () => {
    renderShell();

    const expandInspector = screen.queryByRole("button", {
      name: "Expand inspector"
    });
    if (expandInspector) {
      fireEvent.click(expandInspector);
    }

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

    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const overviewPanel = screen.getByRole("tabpanel", {
      name: "Overview"
    });
    expect(overviewPanel).toBeVisible();
    expect(screen.getByRole("group", { name: "Workflow summary" })).toHaveAttribute(
      "data-state",
      "running"
    );
    const workflowSummary = within(overviewPanel).getByRole("group", {
      name: "Workflow summary"
    });
    expect(workflowSummary).toHaveTextContent("Auto");
    expect(screen.getByRole("group", { name: "Artifacts summary" })).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Product outcome summary" })
    ).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Image references summary" })
    ).toHaveAttribute("data-state", "empty");
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuetext", "11 of 17 workflow nodes reached");
    expect(
      within(overviewPanel).queryByRole("group", {
        name: "Session intelligence summary"
      })
    ).not.toBeInTheDocument();
    expect(
      within(overviewPanel).queryByRole("group", { name: "Preview summary" })
    ).not.toBeInTheDocument();
    expect(
      within(overviewPanel).queryByRole("group", { name: "Telemetry summary" })
    ).not.toBeInTheDocument();
    expect(
      within(overviewPanel).queryByRole("group", { name: "Retrieval summary" })
    ).not.toBeInTheDocument();
    expect(
      within(overviewPanel).queryByRole("group", {
        name: "Workstation dashboard"
      })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel", { name: "Code" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel", { name: "Preview" })).not.toBeInTheDocument();
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
      name: "Preview"
    });
    expect(previewPanel).toBeVisible();
    expect(
      within(previewPanel).getByRole("group", { name: "Preview runtime metadata" })
    ).toBeVisible();
    expect(
      within(previewPanel).getByRole("group", { name: "Preview controls metadata" })
    ).toBeVisible();
    expect(
      within(previewPanel).getByRole("group", { name: "Preview renderer notes" })
    ).toBeVisible();
    expect(
      within(previewPanel).queryByRole("group", { name: "Preview source metadata" })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel", { name: "Overview" })).not.toBeInTheDocument();

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
    expect(screen.getByRole("tabpanel", { name: "Code" })).toBeVisible();
    expect(screen.queryByRole("tabpanel", { name: "Preview" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));

    expect(screen.getByRole("tab", { name: "Runtime" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(
      screen.getByRole("tabpanel", { name: "Runtime" })
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    expect(screen.getByRole("tab", { name: "Retrieval" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getByRole("tabpanel", { name: "Retrieval" })).toBeVisible();
  });

  it("keeps retrieval run signals compact and leaves source drilldown to Dashboard", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval status" })
    ).toBeVisible();
    expect(within(retrievalPanel).getByText("Retrieved context available")).toBeVisible();
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval confidence" })
    ).toHaveTextContent("Medium confidence");
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval coverage" })
    ).toHaveTextContent("2/2 domains covered");
    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval context used" })
    ).toHaveTextContent("3 chunks used");
    expect(
      within(retrievalPanel).getByRole("group", { name: "Knowledge Base status" })
    ).toBeVisible();
    expect(
      within(retrievalPanel).queryByLabelText("Retrieval quality deep dive")
    ).not.toBeInTheDocument();
    expect(
      within(retrievalPanel).queryByRole("region", {
        name: "Retrieval source explorer"
      })
    ).not.toBeInTheDocument();
  });

  it("renders retrieval empty states without stale source cards", () => {
    renderShell(snapshotWithEmptyRetrieval());

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval status" })
    ).toHaveTextContent("No retrieved context for this run.");
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

  it("surfaces app-level KB status in User Mode without exposing retrieval internals", () => {
    renderUserShell(snapshotWithEmptyRetrieval());

    openDashboardKnowledgeBase();

    expect(screen.queryByRole("button", { name: "Knowledge Base status" })).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Advanced Dashboard" })).toHaveTextContent(
      "Technical knowledge"
    );
    expect(screen.getByRole("region", { name: "Advanced Dashboard" })).toHaveTextContent(
      "Creative Knowledge Base"
    );

    expect(screen.queryByRole("tab", { name: "Retrieval" })).not.toBeInTheDocument();
  });

  it("keeps persistent KB inventory distinct from observed retrieval coverage", () => {
    renderUserShell();

    openDashboardKnowledgeBase();

    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    expect(dashboard).toHaveTextContent("Technical knowledge");
    expect(dashboard).toHaveTextContent("Creative Knowledge Base");
    expect(dashboard).toHaveTextContent("Retrieval status");
    expect(dashboard).toHaveTextContent("3 chunks from 2 sources");
  });

  it("keeps ignored-source drilldown in the Dashboard reference view", () => {
    renderShell(snapshotWithIgnoredRetrievalSource());

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval"
    });
    expect(
      within(retrievalPanel).queryByRole("region", {
        name: "Retrieval source explorer"
      })
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Retrieval in Dashboard" })
    );
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    fireEvent.click(
      within(dashboard).getByText("Open current-run retrieval evidence")
    );

    const explorer = within(dashboard).getByRole("region", {
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

  it("uses Settings quick actions to open and focus Inspector views", async () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: /Code Open generated code/ }));

    const codeTab = screen.getByRole("tab", { name: "Code" });
    expect(codeTab).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tabpanel", { name: "Code" })).toBeVisible();
    expect(screen.queryByRole("dialog", { name: "Workspace settings" })).not.toBeInTheDocument();
    await waitFor(() => expect(codeTab).toHaveFocus());
  });

  it("streams backend events into the conversation and workflow state", async () => {
    const backendStream = vi.fn((_request: AssistantStreamRequest) =>
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
          event_type: "planning",
          sequence: 2,
          payload: {
            code: "creative_plan_prepared",
            message: "Creative execution plan prepared.",
            creative_plan: testCreativePlan,
            workflow: {
              step: "planning",
              phase: "running",
              status: "running",
              current_step: "planning",
              completed_steps: ["intake", "routing", "prompt_input"],
              skipped_steps: [],
              refinement_count: 0,
              review_outcome: null,
              review_reasons: [],
              artifact_count: 0,
              preview_artifact_count: 0,
              image_reference_count: 0,
              image_references: [],
              planning_available: true,
              creative_plan: testCreativePlan
            }
          }
        },
        {
          event_type: "token_delta",
          sequence: 3,
          payload: { text: "Streaming " }
        },
        {
          event_type: "token_delta",
          sequence: 4,
          payload: { text: "answer." }
        },
        {
          event_type: "preview_artifact",
          sequence: 5,
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
          sequence: 6,
          payload: {
            answer: "Final backend answer.",
            creative_plan: testCreativePlan,
            session_intelligence: {
              active_request_summary: "Backend active request summary.",
              available_metadata_groups: ["Session", "Workflow", "Preview"],
              completion_status: "completed",
              recommended_next_user_actions: ["Review the hydrated preview."],
              session_summary: "Backend session summary.",
              session_warnings: ["Backend warning."]
            }
          }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.queryByText("Type a prompt to begin")).not.toBeInTheDocument();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.queryByText("Ready to generate")).not.toBeInTheDocument();

    fireEvent.click(sendButton);

    expect(promptInput).toHaveValue("");
    expect(await screen.findByText("Final backend answer.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "local-nextjs-session",
        mode: "generate",
        projectId: "local-nextjs-workspace",
        query: "Make the low-frequency motion calmer."
      })
    );
    const submittedRequest = (
      backendStream.mock.calls as unknown as Array<[AssistantStreamRequest]>
    )[0]?.[0];
    expect(submittedRequest).toBeDefined();
    expect(submittedRequest).not.toHaveProperty("domain");
    expect(submittedRequest).not.toHaveProperty("domains");
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Success"
    );
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(
      screen.getByRole("progressbar", { name: "Overview workflow progress" })
    ).toHaveAttribute("aria-valuenow", "17");
    expect(
      screen.queryByRole("group", { name: "Session intelligence summary" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: "Planning summary" })
    ).not.toBeInTheDocument();

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
      name: "Preview"
    });
    const previewRuntime = within(previewPanel).getByRole("group", {
      name: "Preview runtime metadata"
    });

    expect(previewRuntime).toHaveTextContent("aurora-field.p5.js");
    expect(previewRuntime).toHaveTextContent("Runtime ready");
    expect(
      within(previewPanel).queryByRole("group", { name: "Preview canvas status" })
    ).not.toBeInTheDocument();
  });

  it("answers informational questions without creating a visual artifact", async () => {
    const answer = [
      "Creative coding uses programming as an expressive medium.",
      "It combines algorithms, interaction, motion, sound, and visual systems to make work whose behavior is part of the design.",
      "Unlike conventional application development, the primary goal is often exploration, aesthetic expression, or an interactive experience.",
      "",
      "```javascript",
      "function setup() { createCanvas(640, 360); }",
      "```"
    ].join("\n\n");
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "status",
          sequence: 0,
          payload: {
            code: "route_selected",
            route: {
              execution: {
                requested_mode: "auto",
                resolved_mode: "single_agent",
                rationale: "Auto selected Single Agent for a lightweight explanation.",
                agent_roles: ["generator"],
                researcher_required: false,
                researcher_reason: "A separate researcher is not needed.",
                max_refinement_loops: 0
              }
            },
            workflow: {
              current_step: "routing",
              phase: "running",
              status: "running"
            }
          }
        },
        {
          event_type: "artifact_extracted",
          sequence: 1,
          payload: {
            artifacts: [
              {
                id: "stale-explanation-artifact",
                title: "what-creative-coding.p5.js",
                content: "function setup() { createCanvas(640, 360); }",
                language: "JavaScript",
                runtime: "p5",
                preview_eligible: true
              }
            ]
          }
        },
        {
          event_type: "preview_artifact",
          sequence: 2,
          payload: {
            artifact_id: "stale-explanation-artifact",
            status: "succeeded"
          }
        },
        {
          event_type: "final",
          sequence: 3,
          payload: {
            answer,
            artifacts: [
              {
                id: "stale-explanation-artifact",
                title: "what-creative-coding.p5.js",
                content: "function setup() { createCanvas(640, 360); }",
                language: "JavaScript",
                runtime: "p5",
                preview_eligible: true
              }
            ],
            workflow: {
              current_step: "finalization",
              phase: "completed",
              status: "completed"
            }
          }
        }
      ])
    );

    renderShell(getInitialWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "what's creative coding?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const conversation = screen.getByRole("log", { name: "Conversation" });
    await waitFor(() =>
      expect(conversation).toHaveTextContent(
        "Creative coding uses programming as an expressive medium."
      )
    );
    expect(conversation).toHaveTextContent(
      "It combines algorithms, interaction, motion, sound"
    );
    expect(conversation).toHaveTextContent(
      "the primary goal is often exploration"
    );
    expect(
      within(conversation).getByLabelText("JavaScript code example")
    ).toHaveTextContent("function setup()");
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "explain",
        query: "what's creative coding?",
        workflowMode: "auto"
      })
    );
    expect(screen.getByText("Auto (Single Agent)")).toBeVisible();
    expect(screen.queryByText("assistant-response.md")).not.toBeInTheDocument();
    expect(screen.queryByText("what-creative-coding.p5.js")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.getByRole("tab", { name: "Preview" })).toHaveAttribute(
      "aria-selected",
      "false"
    );
    expect(
      screen.queryByLabelText("Workflow transitions")
    ).not.toBeInTheDocument();
  });

  it("uses the owning browser-profile identity for generation requests", async () => {
    const identity = {
      userId: "browser-user-profile-a",
      sessionId: "browser-session-profile-a",
      projectId: "browser-workspace-profile-a"
    };
    const persistenceClient: WorkspacePersistenceClient = {
      identity,
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "local" as const }))
    };
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: { answer: "Profile-owned response." }
        }
      ])
    );

    renderShell(getInitialWorkspaceSnapshot(), {
      persistenceClient,
      streamAssistantEvents: backendStream
    });
    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a profile-owned sketch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Profile-owned response.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: identity.sessionId,
        projectId: identity.projectId
      })
    );
  });

  it("surfaces clarification questions and sends selected answers", async () => {
    const clarification = {
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The request has creative intent but no explicit output modality.",
      original_query: "Make something evocative about rain.",
      suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      default_recommendation: "Visual sketch",
      signal_summary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          default_recommendation: "Visual sketch"
        }
      ]
    };
    const backendStream = vi
      .fn()
      .mockImplementationOnce(() =>
        streamEvents([
          {
            event_type: "prompt_input",
            sequence: 0,
            payload: {
              code: "clarification_required",
              message: "Clarification required before generation.",
              clarification,
              workflow: {
                step: "prompt_input",
                phase: "running",
                status: "running",
                current_step: "prompt_input",
                completed_steps: ["intake", "routing"],
                skipped_steps: [],
                refinement_count: 0,
                review_outcome: null,
                review_reasons: [],
                artifact_count: 0,
                preview_artifact_count: 0,
                image_reference_count: 0,
                image_references: [],
                clarification_required: true,
                clarification_reason: "ambiguous_modality",
                clarification_question_count: 1,
                clarification
              }
            }
          },
          {
            event_type: "final",
            sequence: 1,
            payload: {
              answer: "I need one quick clarification before generating.",
              clarification
            }
          }
        ])
      )
      .mockImplementationOnce(() =>
        streamEvents([
          {
            event_type: "token_delta",
            sequence: 0,
            payload: { text: "Generated after clarification." }
          },
          {
            event_type: "final",
            sequence: 1,
            payload: { answer: "Generated after clarification." }
          }
        ])
      );

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Make something evocative about rain." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const clarificationCard = await screen.findByRole("group", {
      name: "Clarification summary"
    });
    expect(clarificationCard).toHaveTextContent("Ambiguous Modality");
    expect(
      within(clarificationCard).getByText("What should the assistant generate first?")
    ).toBeVisible();
    expect(
      within(clarificationCard).getByRole("button", {
        name: "Option 1: Visual sketch (Recommended)"
      })
    ).toBeVisible();
    expect(within(clarificationCard).getByText("Recommended")).toBeVisible();

    fireEvent.click(
      within(clarificationCard).getByRole("button", {
        name: "Option 1: Visual sketch (Recommended)"
      })
    );

    expect(await screen.findByText("Generated after clarification.")).toBeVisible();
    expect(backendStream).toHaveBeenLastCalledWith(
      expect.objectContaining({
        clarificationResponse: "Visual sketch",
        query:
          "Make something evocative about rain.\n\nClarification answer: Visual sketch"
      })
    );
  });

  it("maps numeric clarification replies to the matching option", async () => {
    const clarification = {
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The request has creative intent but no explicit output modality.",
      original_query: "Make something evocative about rain.",
      suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      default_recommendation: "Visual sketch",
      signal_summary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          default_recommendation: "Visual sketch"
        }
      ]
    };
    const backendStream = vi
      .fn()
      .mockImplementationOnce(() =>
        streamEvents([
          {
            event_type: "prompt_input",
            sequence: 0,
            payload: {
              code: "clarification_required",
              message: "Clarification required before generation.",
              clarification,
              workflow: {
                step: "prompt_input",
                phase: "running",
                status: "running",
                current_step: "prompt_input",
                completed_steps: ["intake", "routing"],
                skipped_steps: [],
                refinement_count: 0,
                review_outcome: null,
                review_reasons: [],
                artifact_count: 0,
                preview_artifact_count: 0,
                image_reference_count: 0,
                image_references: [],
                clarification_required: true,
                clarification_reason: "ambiguous_modality",
                clarification_question_count: 1,
                clarification
              }
            }
          },
          {
            event_type: "final",
            sequence: 1,
            payload: {
              answer: "I need one quick clarification before generating.",
              clarification
            }
          }
        ])
      )
      .mockImplementationOnce(() =>
        streamEvents([
          {
            event_type: "token_delta",
            sequence: 0,
            payload: { text: "Generated after numeric clarification." }
          },
          {
            event_type: "final",
            sequence: 1,
            payload: { answer: "Generated after numeric clarification." }
          }
        ])
      );

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Make something evocative about rain." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    await screen.findByRole("group", { name: "Clarification summary" });
    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "1" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText("Generated after numeric clarification.")
    ).toBeVisible();
    expect(backendStream).toHaveBeenLastCalledWith(
      expect.objectContaining({
        clarificationResponse: "Visual sketch",
        query:
          "Make something evocative about rain.\n\nClarification answer: Visual sketch"
      })
    );
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

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

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
    const codePanel = screen.getByRole("tabpanel", { name: "Code" });
    expect(within(codePanel).getByText("generated-scene.three.ts")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "generated-scene.three.ts content"
      })
    ).toHaveTextContent("const scene = new THREE.Scene();");

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts"
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

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

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
      name: "Artifacts"
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
    const codePanel = screen.getByRole("tabpanel", { name: "Code" });
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
          name: "Artifacts"
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

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

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
      screen.getByRole("group", { name: "Product outcome summary" })
    ).toBeVisible();
    expect(
      screen.queryByRole("group", { name: "Preview summary" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    const codePanel = screen.getByRole("tabpanel", { name: "Code" });
    expect(within(codePanel).getByText("assistant-response.md")).toBeVisible();
    expect(within(codePanel).getByText("Markdown export")).toBeVisible();
  });

  it("persists a final partial outcome when final hydration has no replacement artifact", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "local" as const }))
    };
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "preview_artifact",
          sequence: 0,
          payload: {
            artifact_id: "source-sketch",
            status: "succeeded"
          }
        },
        {
          event_type: "final",
          sequence: 1,
          payload: {
            answer: "The generated artifact remains available in Code.",
            workflow: {
              current_step: "finalization",
              phase: "completed",
              status: "completed",
              product_outcome: {
                orchestration_status: "COMPLETED",
                provider_status: "COMPLETED",
                generation_status: "COMPLETED",
                deliverable_status: "USABLE",
                artifact_extraction_status: "EXTRACTED",
                artifact_runnability: "UNSUPPORTED",
                preview_status: "UNAVAILABLE",
                runtime_health: "NOT_AVAILABLE",
                product_outcome: "PARTIAL",
                summary: "A usable artifact was produced, but live preview is unavailable.",
                recovery_action: "Open Code to use the artifact."
              }
            }
          }
        }
      ])
    );

    renderUserShell(getLocalWorkspaceSnapshot(), {
      persistenceClient,
      streamAssistantEvents: backendStream
    });
    await waitFor(() => expect(persistenceClient.save).toHaveBeenCalled());
    vi.mocked(persistenceClient.save).mockClear();

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Create a browser-safe p5 study." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText("The generated artifact remains available in Code.")
    ).toBeVisible();
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Partial");
    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          workflow: expect.objectContaining({
            productOutcome: expect.objectContaining({
              product_outcome: "PARTIAL",
              preview_status: "UNAVAILABLE"
            })
          })
        })
      );
    });
  });

  it("sends genuine image pixels once with a request-scoped privacy boundary", async () => {
    const backendStream = vi.fn((_request: AssistantStreamRequest) =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: { answer: "Image-aware response." }
        }
      ])
    );

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    expect(screen.getByRole("menuitem", { name: /Audio input/ })).toHaveAttribute(
      "aria-disabled",
      "true"
    );
    const uploadInput = screen.getByLabelText("Upload image attachment");
    const imageFile = new File([validPngBytes], "palette.png", {
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
    expect(within(imageShelf).getByText(/does not perform pixel analysis/i)).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
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
    expect(
      screen.queryByRole("region", { name: "Image references" })
    ).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Continue without the reference." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    await waitFor(() => expect(backendStream).toHaveBeenCalledTimes(2));
    expect(backendStream.mock.calls[1]?.[0]).toEqual(
      expect.not.objectContaining({ attachments: expect.anything() })
    );
  });

  it("pauses Send while an image reference is being prepared", async () => {
    const pendingBuffer = createDeferred<ArrayBuffer>();
    const imageFile = new File([validPngBytes], "slow-palette.png", {
      type: "image/png"
    });
    Object.defineProperty(imageFile, "arrayBuffer", {
      configurable: true,
      value: vi.fn(() => pendingBuffer.promise)
    });

    renderShell(getLocalWorkspaceSnapshot());
    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Use the queued image reference." }
    });
    expect(screen.getByRole("button", { name: "Send prompt" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    fireEvent.change(screen.getByLabelText("Upload image attachment"), {
      target: { files: [imageFile] }
    });

    await waitFor(() =>
      expect(
        screen.getByRole("form", { name: "Creative request composer" })
      ).toHaveAttribute("data-upload-state", "processing")
    );
    expect(
      screen.getByText("Preparing image reference. Send is paused.")
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add attachment" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send prompt" })).toBeDisabled();

    await act(async () => {
      pendingBuffer.resolve(validPngBytes.slice().buffer);
      await pendingBuffer.promise;
    });

    expect(
      await screen.findByRole("region", { name: "Image references" })
    ).toHaveTextContent("slow-palette.png");
    expect(
      screen.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("data-upload-state", "idle");
    expect(screen.getByRole("button", { name: "Send prompt" })).toBeEnabled();
  });

  it("discards an in-flight image read when the session is cleared", async () => {
    const pendingBuffer = createDeferred<ArrayBuffer>();
    const imageFile = new File([validPngBytes], "stale-palette.png", {
      type: "image/png"
    });
    Object.defineProperty(imageFile, "arrayBuffer", {
      configurable: true,
      value: vi.fn(() => pendingBuffer.promise)
    });

    renderShell(getLocalWorkspaceSnapshot(), {
      persistenceClient: createEmptyPersistenceClient()
    });
    expect(await screen.findByText("Local session ready")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    fireEvent.change(screen.getByLabelText("Upload image attachment"), {
      target: { files: [imageFile] }
    });
    await waitFor(() =>
      expect(
        screen.getByRole("form", { name: "Creative request composer" })
      ).toHaveAttribute("data-upload-state", "processing")
    );

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear workspace session" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear workspace" }));
      await Promise.resolve();
    });

    expect(
      await screen.findByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();

    await act(async () => {
      pendingBuffer.resolve(validPngBytes.slice().buffer);
      await pendingBuffer.promise;
      await Promise.resolve();
    });

    expect(
      screen.queryByRole("region", { name: "Image references" })
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("data-upload-state", "idle");
    expect(screen.getByRole("button", { name: "Add attachment" })).toBeEnabled();
  });

  it("clears request-scoped images before an asynchronously restored session can submit", async () => {
    const userId = "image-boundary-user";
    const projectId = "image-boundary-project";
    const currentSessionId = "image-boundary-current";
    const nextSessionId = "image-boundary-next";
    const currentSnapshot = {
      ...getInitialWorkspaceSnapshot(),
      session: {
        ...getInitialWorkspaceSnapshot().session,
        projectId,
        sessionId: currentSessionId,
        userId
      }
    };
    const nextSnapshot = {
      ...getInitialWorkspaceSnapshot(),
      session: {
        ...getInitialWorkspaceSnapshot().session,
        projectId,
        sessionId: nextSessionId,
        userId
      },
      workspace: {
        ...getInitialWorkspaceSnapshot().workspace,
        name: "Second image session"
      }
    };
    const baseNextRecord = createWorkspaceSessionRecord({
      activeArtifactId: nextSnapshot.artifacts[0]?.id ?? "",
      activeInspectorTab: "Overview",
      previewArtifactId: nextSnapshot.artifacts[0]?.id ?? "",
      previewOpen: false,
      snapshot: nextSnapshot
    });
    const nextRecord = {
      ...baseNextRecord,
      title: "Second image session",
      workspace: {
        ...baseNextRecord.workspace,
        name: "Second image session"
      }
    };
    const sessionKey = `cca.workspace.${userId}.${nextSessionId}`;
    const sessionIndexKey = `cca.workspace.${userId}.session-index.v1`;
    window.localStorage.setItem(sessionKey, JSON.stringify(nextRecord));
    window.localStorage.setItem(
      sessionIndexKey,
      JSON.stringify([
        {
          artifactCount: nextRecord.artifacts.length,
          projectId,
          sessionId: nextSessionId,
          title: nextRecord.title,
          updatedAt: nextRecord.updatedAt
        }
      ])
    );

    const delayedRestore = createDeferred<Response>();
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (
        init?.method === "GET" &&
        url.includes("/api/workspace/session") &&
        url.includes(`sessionId=${nextSessionId}`)
      ) {
        return delayedRestore.promise;
      }
      if (init?.method === "POST" && url.includes("/api/workspace/session")) {
        return Promise.resolve(new Response("{}"));
      }
      return Promise.resolve(new Response("unavailable", { status: 503 }));
    });
    const backendStream = vi.fn((_request: AssistantStreamRequest) =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: { answer: "Session-safe response." }
        }
      ])
    );
    const persistenceClient: WorkspacePersistenceClient = {
      ...createEmptyPersistenceClient(),
      identity: currentSnapshot.session
    };

    try {
      renderShell(currentSnapshot, {
        persistenceClient,
        streamAssistantEvents: backendStream
      });
      await waitFor(() => expect(persistenceClient.load).toHaveBeenCalled());

      fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
      await act(async () => {
        fireEvent.change(screen.getByLabelText("Upload image attachment"), {
          target: {
            files: [
              new File([validPngBytes], "session-a-reference.png", {
                type: "image/png"
              })
            ]
          }
        });
        await Promise.resolve();
      });
      expect(
        await screen.findByRole("region", { name: "Image references" })
      ).toHaveTextContent("session-a-reference.png");

      fireEvent.change(screen.getByLabelText("Assistant prompt"), {
        target: { value: "Submit only inside the newly selected session." }
      });
      fireEvent.click(
        screen.getByRole("button", { name: /Second image session/ })
      );

      expect(
        screen.queryByRole("region", { name: "Image references" })
      ).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

      await waitFor(() => expect(backendStream).toHaveBeenCalledOnce());
      expect(backendStream).toHaveBeenCalledWith(
        expect.objectContaining({
          conversationId: nextSessionId,
          query: "Submit only inside the newly selected session."
        })
      );
      expect(backendStream.mock.calls[0]?.[0]).toEqual(
        expect.not.objectContaining({ attachments: expect.anything() })
      );

      delayedRestore.resolve(new Response("session not found", { status: 404 }));
      await delayedRestore.promise;
    } finally {
      window.localStorage.removeItem(sessionKey);
      window.localStorage.removeItem(sessionIndexKey);
    }
  });

  it("shows a graceful image upload error for unsupported files", async () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Upload image attachment"), {
        target: {
          files: [new File(["notes"], "notes.txt", { type: "text/plain" })]
        }
      });
      await Promise.resolve();
    });

    const imageShelf = await screen.findByRole("region", {
      name: "Image references"
    });

    expect(within(imageShelf).getByText("Reference not added")).toBeVisible();
    expect(within(imageShelf).getByText("Image upload issue")).toBeVisible();
    expect(
      within(imageShelf).getByText(
        "Only PNG, JPEG, WebP, or GIF image references can be attached."
      )
    ).toBeVisible();
    expect(
      within(imageShelf).getByRole("button", { name: "Dismiss image upload issue" })
    ).toBeVisible();
  });

  it("keeps a mixed valid and invalid image batch atomic", async () => {
    const backendStream = vi.fn((_request: AssistantStreamRequest) =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: { answer: "No partial image batch was sent." }
        }
      ])
    );
    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Upload image attachment"), {
        target: {
          files: [
            new File([validPngBytes], "valid-first.png", { type: "image/png" }),
            new File(["notes"], "invalid-second.txt", { type: "text/plain" })
          ]
        }
      });
      await Promise.resolve();
    });

    const imageShelf = await screen.findByRole("region", {
      name: "Image references"
    });
    expect(within(imageShelf).getByText("Reference not added")).toBeVisible();
    expect(within(imageShelf).getByText(/queued request is unchanged/i)).toBeVisible();
    expect(within(imageShelf).queryByText("valid-first.png")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Continue without a partial image batch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    await waitFor(() => expect(backendStream).toHaveBeenCalledOnce());
    expect(backendStream.mock.calls[0]?.[0]).toEqual(
      expect.not.objectContaining({ attachments: expect.anything() })
    );
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

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Create a p5.js sketch with low-frequency motion." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText(/Use p5.js createCanvas/)).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval"
    });

    expect(
      within(retrievalPanel).getByRole("group", { name: "Retrieval status" })
    ).toHaveTextContent("1 retrieved sources");
    expect(
      within(retrievalPanel).queryByRole("group", {
        name: "createCanvas source details"
      })
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Retrieval in Dashboard" })
    );
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    fireEvent.click(
      within(dashboard).getByText("Open current-run retrieval evidence")
    );
    const sourceDetail = within(dashboard).getByRole("group", {
      name: "createCanvas source details"
    });
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

    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Find TouchDesigner references for this projection loop." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText("Continuing without retrieval references.")
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Retrieval" }));

    const retrievalPanel = screen.getByRole("tabpanel", {
      name: "Retrieval"
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

  it("synchronizes the active generation state without ambiguous live labels", async () => {
    const beforeTokens = createDeferred<void>();
    const beforeFinal = createDeferred<void>();
    const backendStream = vi.fn(async function* () {
      yield {
        event_type: "status",
        sequence: 0,
        payload: {
          code: "request_received",
          workflow: {
            current_step: "planning",
            phase: "running",
            status: "running"
          }
        }
      } satisfies AssistantStreamEvent;
      await beforeTokens.promise;
      yield {
        event_type: "token_delta",
        sequence: 1,
        payload: {
          text: "Live draft",
          workflow: {
            current_step: "generation",
            phase: "running",
            status: "running"
          }
        }
      } satisfies AssistantStreamEvent;
      await beforeFinal.promise;
      yield {
        event_type: "final",
        sequence: 2,
        payload: {
          answer: "Live draft completed.",
          workflow: {
            current_step: "finalization",
            phase: "completed",
            status: "completed"
          }
        }
      } satisfies AssistantStreamEvent;
    });

    renderUserShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a calmer draft." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Planning the requested work...")).toBeVisible();
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Planning");
    expect(screen.queryByText("Thinking")).not.toBeInTheDocument();
    expect(screen.getByRole("log", { name: "Conversation" })).not.toHaveAttribute(
      "aria-busy"
    );
    expect(
      screen.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Assistant Planning");

    beforeTokens.resolve();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      screen.getByText(
        "Generating the requested artifact. Code and long-form output will appear in the Code panel, artifacts, and preview surfaces when the run completes."
      )
    ).toBeVisible();
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Generating");
    expect(screen.getAllByText("Generating").length).toBeGreaterThan(0);
    expect(screen.queryByText("Live")).not.toBeInTheDocument();

    beforeFinal.resolve();

    expect(await screen.findByText("Live draft completed.")).toBeVisible();
    expect(screen.getByRole("log", { name: "Conversation" })).not.toHaveAttribute(
      "aria-busy"
    );
    expect(
      screen.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("aria-busy", "false");
  });

  it("keeps generated code out of the conversation summary", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer: [
              "Here is a browser-ready p5 sketch.",
              "",
              "```js",
              "function setup() {",
              "  createCanvas(640, 360);",
              "}",
              "function draw() {",
              "  background(12);",
              "}",
              "```"
            ].join("\n")
          }
        }
      ])
    );

    renderUserShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a p5 sketch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const conversation = screen.getByRole("log", { name: "Conversation" });

    expect(
      await within(conversation).findByText(/Code and long-form output are available/)
    ).toBeVisible();
    expect(within(conversation).queryByText(/function draw/)).not.toBeInTheDocument();
  });

  it("presents informational work as answering instead of artifact generation", async () => {
    const beforeFinal = createDeferred<void>();
    const backendStream = vi.fn(async function* () {
      yield {
        event_type: "status",
        sequence: 0,
        payload: {
          code: "route_selected",
          workflow: {
            current_step: "generation",
            phase: "running",
            status: "running"
          }
        }
      } satisfies AssistantStreamEvent;
      await beforeFinal.promise;
      yield {
        event_type: "final",
        sequence: 1,
        payload: {
          answer: "Creative coding uses software as an expressive medium.",
          workflow: {
            current_step: "finalization",
            phase: "completed",
            status: "completed"
          }
        }
      } satisfies AssistantStreamEvent;
    });

    renderUserShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "what's creative coding?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const assistantMessages = await screen.findAllByRole("article", {
      name: /Assistant message/
    });
    const assistantMessage = assistantMessages.at(-1)!;
    await waitFor(() =>
      expect(within(assistantMessage).getByText("Answering")).toBeVisible()
    );
    expect(within(assistantMessage).getAllByText("Writing the answer.").length).toBeGreaterThan(0);
    expect(
      within(assistantMessage).queryByText(/Generating the requested artifact/i)
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Answering");
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Writing the answer."
    );

    beforeFinal.resolve();
    expect(
      await screen.findByText(
        "Creative coding uses software as an expressive medium."
      )
    ).toBeVisible();
  });

  it("keeps an unterminated generated code fence out of the conversation", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "final",
          sequence: 0,
          payload: {
            answer: [
              "```html",
              "<!doctype html>",
              "<script>function setup() { createCanvas(640, 360); }</script>"
            ].join("\n")
          }
        }
      ])
    );

    renderUserShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a browser sketch." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const conversation = screen.getByRole("log", { name: "Conversation" });

    expect(
      await within(conversation).findByText(/Code and long-form output are available/)
    ).toBeVisible();
    expect(within(conversation).queryByText(/```html|<!doctype|function setup/i)).not.toBeInTheDocument();
    expect(conversation).toHaveTextContent(
      "Your requested creative-coding output is ready."
    );
  });

  it("summarizes mixed generated code in User Mode while routing code to panels", async () => {
    const artifact = {
      id: "mixed-visual-artifact",
      title: "mixed-visual-artifact.frag",
      language: "GLSL",
      source_language: "glsl",
      content: [
        "void main() {",
        "  vec2 uv = gl_FragCoord.xy / u_resolution.xy;",
        "  gl_FragColor = vec4(uv, 0.8, 1.0);",
        "}"
      ].join("\n"),
      preview_eligible: true,
      preview_target: "browser_sandbox",
      runtime: "glsl",
      renderer_id: "surface.glsl",
      source_order: 1,
      is_default: true
    };
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "artifact_extracted",
          sequence: 0,
          payload: {
            artifacts: [artifact]
          }
        },
        {
          event_type: "final",
          sequence: 1,
          payload: {
            answer: [
              "Here is the generated visual shader and HTML wrapper.",
              "",
              "```html",
              "<!doctype html>",
              "<html>",
              "<script>",
              "function setup() { createCanvas(640, 360); }",
              "</script>",
              "</html>",
              "```",
              "```glsl",
              artifact.content,
              "```"
            ].join("\n"),
            artifacts: [artifact]
          }
        }
      ])
    );

    renderUserShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Generate a GLSL post-processing visual." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    const conversation = screen.getByRole("log", { name: "Conversation" });

    expect(
      await within(conversation).findByText(/Code and long-form output are available/)
    ).toBeVisible();
    expect(within(conversation).queryByText(/<!doctype/i)).not.toBeInTheDocument();
    expect(within(conversation).queryByText(/function setup/i)).not.toBeInTheDocument();
    expect(within(conversation).queryByText(/gl_FragColor/)).not.toBeInTheDocument();

    const expandInspector = screen.queryByRole("button", { name: "Expand inspector" });
    if (expandInspector) {
      fireEvent.click(expandInspector);
    }
    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    expect(
      within(screen.getByRole("tabpanel", { name: "Code" })).getByText(
        /gl_FragColor/
      )
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Saved" }));
    expect(
      within(
        screen.getByRole("tabpanel", { name: "Saved" })
      ).getAllByText("GLSL Shader").length
    ).toBeGreaterThan(0);
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
    fireEvent.click(screen.getByRole("button", { name: "Use Matrix theme" }));

    expect(document.documentElement).toHaveAttribute("data-cca-theme", "matrix");

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    const settingsPanel = screen.getByRole("dialog", { name: "Workspace settings" });
    fireEvent.click(within(settingsPanel).getByRole("button", { name: "Preview auto-open" }));
    fireEvent.click(within(settingsPanel).getByRole("button", { name: "Display mode" }));

    await waitFor(() => {
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          preferences: expect.objectContaining({
            theme: "matrix",
            autoOpenPreview: false,
            showDebugPanels: false
          })
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

  it("hides developer traces and keeps preview closed when auto-open is disabled", async () => {
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
    const settingsPanel = screen.getByRole("dialog", { name: "Workspace settings" });
    fireEvent.click(within(settingsPanel).getByRole("button", { name: "Display mode" }));
    fireEvent.click(within(settingsPanel).getByRole("button", { name: "Preview auto-open" }));

    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Telemetry" })).not.toBeInTheDocument();
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
    expect(within(preview).getAllByText("Ready when opened").length).toBeGreaterThan(0);
    expect(
      within(preview).queryByRole("list", { name: "Renderer health overlay" })
    ).not.toBeInTheDocument();
    expect(
      within(preview).queryByLabelText("Preview runtime source")
    ).not.toBeInTheDocument();
    expect(
      within(preview).getByText("P5 Sketch", { selector: "summary span" })
    ).toBeVisible();
    expect(within(preview).queryByText("aurora-field.p5.js")).not.toBeInTheDocument();
    expect(preview.querySelector("details")).not.toHaveAttribute("open");
  });

  it("does not open a stale preview event for a React Three Fiber artifact", async () => {
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const snapshot: AssistantWorkspaceSnapshot = {
      ...baseSnapshot,
      artifacts: [
        {
          ...baseSnapshot.artifacts[0],
          id: "react-three-fiber-study",
          title: "installation.r3f.tsx",
          language: "TypeScript + React Three Fiber",
          content: [
            'import { Canvas, useFrame } from "@react-three/fiber";',
            "function Orb() { useFrame(() => {}); return <mesh />; }",
            "export default function Study() { return <Canvas><Orb /></Canvas>; }"
          ].join("\n"),
          domain: "react_three_fiber",
          runtime: null,
          rendererId: null,
          previewEligible: false,
          previewTarget: "",
          actions: ["Open", "Copy", "Download"]
        }
      ],
      preview: {
        ...baseSnapshot.preview,
        available: false,
        active: false,
        collapsed: true,
        state: "unavailable",
        title: "Preview unavailable"
      }
    };
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
            artifact_id: "react-three-fiber-study",
            status: "succeeded",
            result: {
              preview_artifact_id: "react-three-fiber-study",
              details: {
                artifact: {
                  domain: "react_three_fiber",
                  preview_eligible: false
                }
              },
              request: { target: "browser_sandbox" },
              provenance: { renderer_id: "surface.three" }
            }
          }
        },
        {
          event_type: "final",
          sequence: 2,
          payload: { answer: "React Three Fiber source is ready to inspect." }
        }
      ])
    );

    renderUserShell(snapshot, { streamAssistantEvents: backendStream });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "Return React Three Fiber source." }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(
      await screen.findByText("React Three Fiber source is ready to inspect.")
    ).toBeVisible();
    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(within(preview).getByText("Preview unavailable")).toBeVisible();
    expect(
      within(preview).queryByRole("button", { name: "Enter preview fullscreen" })
    ).not.toBeInTheDocument();
    expect(
      within(preview).queryByRole("button", { name: "Restart preview session" })
    ).not.toBeInTheDocument();
  });

  it("falls back to the local draft path when the live response is unavailable", async () => {
    vi.useFakeTimers();
    renderShell(getLocalWorkspaceSnapshot(), { streamAssistantEvents: failingStream });

    const promptInput = screen.getByLabelText("Assistant prompt");
    const sendButton = screen.getByRole("button", { name: "Send prompt" });

    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute("data-ready", "false");
    expect(screen.queryByText("Type a prompt to begin")).not.toBeInTheDocument();

    fireEvent.change(promptInput, {
      target: { value: "Make the low-frequency motion calmer." }
    });
    expect(sendButton).toHaveAttribute("data-ready", "true");
    expect(screen.queryByText("Ready to generate")).not.toBeInTheDocument();

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

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    expect(await screen.findByText("Runtime issue")).toBeVisible();
  });

  it("retains failure classification after partial streamed output", async () => {
    renderShell(getLocalWorkspaceSnapshot(), {
      streamAssistantEvents: () =>
        streamEvents([
          {
            event_type: "status",
            sequence: 0,
            payload: { code: "request_received", message: "Request accepted." }
          },
          {
            event_type: "token_delta",
            sequence: 1,
            payload: { text: "Partial response that cannot complete." }
          },
          {
            event_type: "error",
            sequence: 2,
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

    const assistantMessage = (
      await screen.findByText(/Partial response that cannot complete\./)
    ).closest("article");
    const errorCallout = screen
      .getByText("Live stream interrupted")
      .closest("article");

    expect(assistantMessage).toHaveAttribute("data-stream-phase", "failed");
    expect(errorCallout).toHaveClass("chatErrorCallout");
    expect(
      within(errorCallout as HTMLElement).getByText(
        "The model provider is unavailable for this live response."
      )
    ).toBeVisible();
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
    fireEvent.click(screen.getByRole("tab", { name: "Preview" }));
    expect(screen.getByRole("tabpanel", { name: "Preview" })).toBeVisible();

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
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);
    expect(within(preview).getByText("Fullscreen")).toBeVisible();
    expect(within(preview).getByText("Restart")).toBeVisible();
    expect(within(preview).getByText("Clear")).toBeVisible();
    expect(within(preview).getByText("Reload")).toBeVisible();
    expect(
      within(preview).queryByRole("button", { name: "Collapse preview" })
    ).not.toBeInTheDocument();
    expect(
      within(preview).queryByRole("button", { name: "Reset preview session" })
    ).not.toBeInTheDocument();
    fireEvent.click(
      within(preview).getByRole("button", { name: "Enter preview fullscreen" })
    );

    const fullscreenLayer = screen.getByRole("dialog", {
      name: "Fullscreen artwork canvas"
    });
    const fullscreenDetails = fullscreenLayer.querySelector("details");

    expect(fullscreenLayer.parentElement).toBe(document.body);
    expect(fullscreenDetails).toHaveAttribute("data-fullscreen", "true");
    expect(preview.querySelector("details")).not.toBeInTheDocument();
    expect(
      within(fullscreenLayer).getByRole("button", {
        name: "Exit preview fullscreen"
      })
    ).toBeVisible();
    expect(
      within(fullscreenLayer).queryByLabelText("Focused preview context")
    ).not.toBeInTheDocument();
    expect(
      within(fullscreenLayer).queryByLabelText("Preview controls")
    ).not.toBeInTheDocument();
    expect(
      within(fullscreenLayer).queryByRole("button", { name: "Restart preview session" })
    ).not.toBeInTheDocument();
    expect(
      within(fullscreenLayer).queryByRole("button", { name: "Reload preview state" })
    ).not.toBeInTheDocument();

    fireEvent.click(
      within(fullscreenLayer).getByRole("button", {
        name: "Exit preview fullscreen"
      })
    );

    expect(
      screen.queryByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).not.toBeInTheDocument();
    expect(
      within(preview).getByText("aurora-field.p5.js", { selector: "summary span" })
    ).toBeVisible();
  });

  it("exits the portal fullscreen artwork layer with Escape", () => {
    renderShell();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);
    fireEvent.click(
      within(preview).getByRole("button", { name: "Enter preview fullscreen" })
    );

    expect(
      screen.getByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).toBeVisible();
    expect(document.body.style.overflow).toBe("hidden");

    fireEvent.keyDown(window, { key: "Escape" });

    expect(
      screen.queryByRole("dialog", { name: "Fullscreen artwork canvas" })
    ).not.toBeInTheDocument();
    expect(document.body.style.overflow).toBe("");
  });

  it("routes destructive preview runtime actions through an operator checkpoint", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockImplementation(() => true);
    renderShell();
    openDeveloperInspector();

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
      name: "Preview"
    });
    expect(
      within(previewPanel).getByRole("group", { name: "Preview runtime metadata" })
    ).toHaveTextContent("aurora-field.p5.js");
    expect(
      within(previewPanel).queryByRole("group", { name: "Preview canvas status" })
    ).not.toBeInTheDocument();

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
    const workflowPanel = screen.getByRole("tabpanel", {
      name: "Workflow"
    });
    expect(
      within(workflowPanel).getByRole("group", {
        name: "LangGraph workflow visualization"
      })
    ).toBeVisible();
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
  });

  it("restores a cleared preview through the non-destructive reload path", async () => {
    renderShell();
    openDeveloperInspector();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);
    fireEvent.click(
      within(preview).getByRole("button", { name: "Clear preview state" })
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Clear runtime" }));
      await Promise.resolve();
    });

    expect(
      within(preview).getByText("Cleared", { selector: "summary small" })
    ).toBeVisible();

    fireEvent.click(
      within(preview).getByRole("button", { name: "Reload preview state" })
    );

    expect(
      within(preview).getByText("Generating", { selector: "summary small" })
    ).toBeVisible();
    expect(
      within(preview).queryByText(
        "Preview state cleared for aurora-field.p5.js. Reload or reset the session to restore the latest runtime context."
      )
    ).not.toBeInTheDocument();
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
      within(screen.getByRole("tabpanel", { name: "Code" })).getByText(
        "projection-notes.md"
      )
    ).toBeVisible();
  });

  it("renames an artifact without losing its preview or code references", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Rename aurora-field.p5.js" })
    );
    fireEvent.change(
      screen.getByRole("textbox", {
        name: "New file name for aurora-field.p5.js"
      }),
      { target: { value: "Luminous Flow Field.ts" } }
    );
    fireEvent.click(screen.getByRole("button", { name: "Save name" }));

    expect(screen.getAllByText("luminous-flow-field.p5.js").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("tab", { name: "Code" }));
    expect(
      screen.getByRole("region", { name: "luminous-flow-field.p5.js content" })
    ).toBeVisible();
    fireEvent.click(screen.getByRole("tab", { name: "Preview" }));
    expect(
      screen.getAllByText("luminous-flow-field.p5.js", { selector: "dd" }).length
    ).toBeGreaterThan(0);
  });

  it("opens artifacts, highlights the active artifact, and targets preview actions", () => {
    renderShell();

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const artifactList = screen.getByRole("tabpanel", { name: "Artifacts" });
    const sourceArtifact = within(artifactList).getByLabelText(
      "aurora-field.p5.js artifact"
    );
    fireEvent.click(
      within(sourceArtifact).getByRole("button", {
        name: "Open in Code aurora-field.p5.js"
      })
    );

    const codePanel = screen.getByRole("tabpanel", { name: "Code" });

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

    expect(screen.getByRole("tabpanel", { name: "Code" })).toHaveAttribute(
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

  it("keeps artifact ranking concise in the inspector without mounting critic deep dives", () => {
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
      name: "Artifacts"
    });

    expect(
      within(artifactsPanel).getByRole("group", {
        name: "Active artifact details"
      })
    ).toHaveTextContent("Quality");
    expect(
      within(artifactsPanel).queryByRole("region", {
        name: "Artifact quality summary"
      })
    ).not.toBeInTheDocument();
    expect(
      within(artifactsPanel).getByText(/Rank #1 \/ Quality 93%/)
    ).toBeVisible();
    expect(artifactsPanel).toHaveTextContent(
      "aurora-field.p5.js is the recommended candidate."
    );
  });

  it("mounts supported p5 artifacts into a controlled live runtime", async () => {
    renderShell(snapshotWithP5Preview());
    openDeveloperInspector();

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
    expect(
      await within(surface).findByText(
        "Rendering signal-orbit.p5.ts inside an isolated p5-compatible preview frame."
      )
    ).toBeVisible();
  });

  it("uses human preview runtime status text in User Mode", async () => {
    const snapshot = snapshotWithReadyPreview();
    const title = "signal-orbit.p5.ts";
    renderUserShell({
      ...snapshot,
      artifacts: [
        {
          ...snapshot.artifacts[0],
          language: "TypeScript + p5.js",
          runtime: "p5",
          summary: "Reactive p5 loop with createCanvas() and draw().",
          title
        },
        ...snapshot.artifacts.slice(1)
      ],
      code: {
        ...snapshot.code,
        language: "TypeScript + p5.js",
        title
      },
      preview: {
        ...snapshot.preview,
        artifactName: title,
        sourceArtifactName: title
      }
    });

    const preview = screen.getByRole("region", { name: "Preview workspace" });
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

    expect(await within(surface).findByText("p5 runtime running")).toBeVisible();
    expect(
      await within(surface).findByText(
        "Rendering P5 sketch surface inside an isolated p5-compatible preview frame."
      )
    ).toBeVisible();
    expect(within(surface).queryByText("signal-orbit.p5.ts")).not.toBeInTheDocument();
  });

  it("mounts supported Hydra artifacts into a controlled live runtime", async () => {
    renderShell(snapshotWithHydraPreview());
    openDeveloperInspector();

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
    openDeveloperInspector();

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

  it("mounts supported GSAP artifacts into the bounded motion runtime", async () => {
    renderShell(snapshotWithGsapPreview());
    openDeveloperInspector();

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });

    expect(within(preview).getByText("GSAP motion stage")).toBeVisible();
    expect(
      within(surface).getByRole("group", { name: "GSAP live runtime" })
    ).toBeVisible();
    const frame = await waitForSandboxRuntimeFrame(
      surface,
      "GSAP preview runtime frame"
    );
    dispatchSandboxRuntimeStatus(frame, {
      detail:
        "Animating signal-bloom.gsap.ts inside a bounded GSAP motion stage.",
      diagnostics: ["16 sandbox nodes / 16 tweens", "stagger enabled / yoyo enabled"],
      label: "GSAP runtime running",
      state: "running"
    });
    expect(await within(surface).findByText("GSAP runtime running")).toBeVisible();
    expect(within(surface).getByText("stagger enabled / yoyo enabled")).toBeVisible();
  });

  it("shows a compact diagnostics overlay for live preview runtimes", async () => {
    renderShell(snapshotWithP5Preview());
    openDeveloperInspector();

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
    openDeveloperInspector();

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
      name: "Runtime"
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
    expect(
      within(runtimePanel).getByRole("group", { name: "Runtime diagnostics" })
    ).toBeVisible();
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime event history" })
    ).not.toBeInTheDocument();
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime reload history" })
    ).not.toBeInTheDocument();
  });

  it("mounts supported Three.js artifacts into a controlled 3D runtime", async () => {
    const snapshot = snapshotWithThreePreview();
    renderShell({
      ...snapshot,
      preview: {
        ...snapshot.preview,
        active: true,
        collapsed: false,
        state: "ready",
        status: "Ready"
      }
    });

    const preview = screen.getByRole("region", { name: "Preview workspace" });
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
      await within(surface).findByText("WebGL is unavailable")
    ).toBeVisible();
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
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    expect(
      within(workflowPanel).getByRole("group", {
        name: "LangGraph workflow visualization"
      })
    ).toBeVisible();
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
  });

  it("keeps preview runtime failures user-friendly in User Mode", async () => {
    const snapshot = snapshotWithThreePreview();
    renderUserShell({
      ...snapshot,
      preview: {
        ...snapshot.preview,
        active: true,
        collapsed: false,
        outputArtifactName: "projection-scene.preview.html",
        state: "ready",
        status: "Ready"
      }
    });

    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent(
      "User"
    );

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    expect(preview.querySelector("details")).toHaveAttribute("open");

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

    expect(
      await within(surface).findByText("Preview fallback ready")
    ).toBeVisible();
    expect(within(surface).queryByText("Renderer runtime failed")).toBeNull();
    expect(
      within(surface).queryByText("WebGL is unavailable in the preview frame.")
    ).toBeNull();
    expect(
      within(surface).getByRole("button", { name: "Reload preview runtime" })
    ).toBeVisible();
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "Partial"
    );
    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "A usable artifact was produced, but the live preview failed."
    );

  });

  it("shows runtime errors and reload metrics in the compact runtime console", async () => {
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
      name: "Runtime"
    });
    const diagnostics = within(runtimePanel).getByRole("group", {
      name: "Runtime diagnostics"
    });
    const health = within(runtimePanel).getByRole("group", {
      name: "Runtime health"
    });

    expect(health).toHaveTextContent("Failure");
    expect(diagnostics).toHaveTextContent("1 active");
    expect(diagnostics).toHaveTextContent(
      "WebGL is not available for this preview session."
    );
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime event history" })
    ).not.toBeInTheDocument();

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

    const reloadMetric = within(runtimePanel)
      .getByText("Reloads")
      .closest("[role='listitem']");
    expect(reloadMetric).toHaveTextContent("1");
    expect(
      within(runtimePanel).queryByRole("group", { name: "Runtime reload history" })
    ).not.toBeInTheDocument();
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
      detail: "ERROR: 0:123: '' : syntax error",
      error: {
        message: "ERROR: 0:123: '' : syntax error",
        type: "shader_compile_failed"
      },
      label: "GLSL runtime failed",
      state: "error"
    });
    expect(
      await within(surface).findByText("Shader needs a quick repair")
    ).toBeVisible();
    expect(within(surface).getByText("Shader line 123")).toBeVisible();
    expect(
      within(surface).getByText(
        "The generated shader contains a syntax error, so the preview could not start."
      )
    ).toBeVisible();
    expect(
      within(surface).getByRole("button", { name: "Open generated code" })
    ).toBeVisible();
    expect(within(surface).getByText("Technical details")).toBeVisible();
  });

  it("restores the source preview when a refined GLSL artifact fails to compile", async () => {
    renderShell(snapshotWithRefinedGlslPreview());

    const preview = screen.getByRole("region", { name: "Preview workspace" });
    const summary = within(preview).getByText("Preview available").closest("summary");

    expect(summary).not.toBeNull();
    fireEvent.click(summary as HTMLElement);

    const surface = within(preview).getByRole("group", {
      name: "Preview renderer surface"
    });
    const failedFrame = await waitForSandboxRuntimeFrame(
      surface,
      "GLSL preview runtime frame"
    );
    const failedRuntimeId = failedFrame.dataset.runtimeId;

    dispatchSandboxRuntimeStatus(failedFrame, {
      detail: "ERROR: 0:74: ':' syntax error",
      error: {
        message: "ERROR: 0:74: ':' syntax error",
        type: "shader_compile_failed"
      },
      label: "GLSL runtime failed",
      state: "error"
    });

    await waitFor(() => {
      expect(
        within(preview).getByText("chladni-light-field-2.frag", {
          selector: "summary span"
        })
      ).toBeVisible();
      const restoredFrame = within(surface).getByLabelText(
        "GLSL preview runtime frame"
      );
      expect(restoredFrame.dataset.runtimeId).toMatch(/^preview-runtime-/);
      expect(restoredFrame.dataset.runtimeId).not.toBe(failedRuntimeId);
    });

    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "improve-performance.refined.frag"
    );
    expect(within(surface).queryByText("Renderer runtime failed")).not.toBeInTheDocument();
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
    },
    {
      frameLabel: "GSAP preview runtime frame",
      makeSnapshot: snapshotWithGsapPreview,
      runningDetail:
        "Animating signal-bloom.gsap.ts inside a bounded GSAP motion stage.",
      runningLabel: "GSAP runtime running",
      surfaceTitle: "GSAP motion stage"
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

      const recoveryRuntimeId = frame.dataset.runtimeId;
      dispatchSandboxRuntimeStatus(frame, {
        detail: runningDetail,
        label: runningLabel,
        state: "running"
      });
      expect(await within(surface).findByText(runningLabel)).toBeVisible();
      expect(frame.dataset.runtimeId).toBe(recoveryRuntimeId);
      expect(
        within(preview).queryByText("Reloading", { selector: "summary small" })
      ).not.toBeInTheDocument();
      expect(
        within(preview).getByRole("button", { name: "Reload preview state" })
      ).toBeEnabled();

      fireEvent.click(screen.getByRole("tab", { name: "Runtime" }));
      const runtimePanel = screen.getByRole("tabpanel", {
        name: "Runtime"
      });
      const reloadMetric = within(runtimePanel)
        .getByText("Reloads")
        .closest("[role='listitem']");
      expect(reloadMetric).toHaveTextContent("1");
      expect(
        within(runtimePanel).queryByRole("group", { name: "Runtime event history" })
      ).not.toBeInTheDocument();
    }
  );

  it("settles an approved preview restart after the replacement runtime runs", async () => {
    renderShell();
    openDeveloperInspector();

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
    const initialRuntimeId = frame.dataset.runtimeId;

    fireEvent.click(
      within(preview).getByRole("button", { name: "Restart preview session" })
    );
    expect(screen.getByLabelText("Operator checkpoint")).toHaveTextContent(
      "Restart preview runtime"
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Restart runtime" }));
      await Promise.resolve();
    });
    await waitFor(() => {
      expect(frame.dataset.runtimeId).toMatch(/^preview-runtime-/);
      expect(frame.dataset.runtimeId).not.toBe(initialRuntimeId);
    });
    expect(
      within(preview).getByText("Restarting", { selector: "summary small" })
    ).toBeVisible();

    const restartRuntimeId = frame.dataset.runtimeId;
    dispatchSandboxRuntimeStatus(frame, {
      detail: "Rendering aurora-field.p5.js inside an isolated p5-compatible preview frame.",
      label: "p5 runtime running",
      state: "running"
    });

    expect(await within(surface).findByText("p5 runtime running")).toBeVisible();
    expect(frame.dataset.runtimeId).toBe(restartRuntimeId);
    expect(
      within(preview).queryByText("Restarting", { selector: "summary small" })
    ).not.toBeInTheDocument();
  });

  it("uses the full inspector panel for code when Code is active", () => {
    renderShell(snapshotWithActiveTab("Code"));

    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const codePanel = screen.getByRole("tabpanel", { name: "Code" });

    expect(codePanel).toBeVisible();
    expect(within(codePanel).getByText("p5.js")).toBeVisible();
    expect(within(codePanel).getByText("Source code")).toBeVisible();
    expect(within(codePanel).getByText("15 lines")).toBeVisible();
    expect(
      within(codePanel).getByRole("region", {
        name: "aurora-field.p5.js content"
      })
    ).toHaveTextContent("function draw()");
    expect(screen.queryByRole("tabpanel", { name: "Overview" })).not.toBeInTheDocument();
  });

  it("keeps the comparison workspace out of the cockpit and exposes deliverables in Dashboard", () => {
    renderShell(snapshotWithArtifactComparison());

    expect(
      screen.queryByRole("region", { name: "Artifact comparison" })
    ).not.toBeInTheDocument();
    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts"
    });
    expect(
      within(artifactsPanel).getByLabelText("shader-field.frag artifact")
    ).toHaveTextContent("Recommended");
    expect(
      within(artifactsPanel).getByLabelText("field.wgsl artifact")
    ).toHaveTextContent("Code-only");

    fireEvent.click(
      screen.getByRole("button", { name: "Open Artifacts in Dashboard" })
    );
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    expect(
      within(dashboard).getByRole("list", { name: "Saved deliverables" })
    ).toBeVisible();
    expect(within(dashboard).getByText("5 saved deliverables")).toBeVisible();
    expect(
      within(dashboard).getByLabelText("shader-field.frag visual preview")
    ).toBeVisible();
    expect(
      within(dashboard).getByLabelText("field.wgsl export boundary")
    ).toBeVisible();
  });

  it("keeps creative planning metadata out of compact inspector tabs", () => {
    const snapshot = snapshotWithActiveTab("Artifacts");
    const plannedSnapshot: AssistantWorkspaceSnapshot = {
      ...snapshot,
      creativePlan: testCreativePlan,
      artifacts: snapshot.artifacts.map((artifact, index) =>
        index === 0 ? { ...artifact, creativePlan: testCreativePlan } : artifact
      )
    };

    renderShell(plannedSnapshot);

    const artifactsPanel = screen.getByRole("tabpanel", {
      name: "Artifacts"
    });
    expect(
      within(artifactsPanel).queryByRole("region", {
        name: "Artifact planning summary"
      })
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(
      screen.queryByRole("group", { name: "Planning summary" })
    ).not.toBeInTheDocument();
  });

  it("shows a selected-artifact refinement action with guided instructions", () => {
    renderShell(snapshotWithArtifactComparison());

    const details = screen.getByRole("group", { name: "Active artifact details" });
    const refinement = within(details).getByRole("region", {
      name: "Selected artifact refinement"
    });
    const submitButton = within(refinement).getByRole("button", {
      name: "Apply refinement"
    });

    expect(within(refinement).getByText("Create a refined version")).toBeVisible();
    expect(
      within(refinement).getByText(
        "Apply one focused change to aurora-field.p5.js. The current version stays saved."
      )
    ).toBeVisible();
    expect(submitButton).toBeDisabled();

    fireEvent.click(within(refinement).getByRole("button", { name: "Make this faster" }));

    expect(within(refinement).getByLabelText("Refinement instruction")).toHaveValue(
      "Make this faster"
    );
    expect(submitButton).toBeEnabled();
  });

  it("resolves a clear chat follow-up against the active artifact without asking for modality", async () => {
    const backendStream = vi.fn(() =>
      streamEvents([
        {
          event_type: "artifact_extracted",
          sequence: 0,
          payload: {
            artifacts: [
              {
                id: "make-it-brighter-output",
                title: "make-it-brighter-visual.p5.js",
                type: "code",
                language: "p5.js",
                content: [
                  "function setup() { createCanvas(320, 180); }",
                  "function draw() { background(42); circle(160, 90, 80); }"
                ].join("\n"),
                domain: "p5_js",
                runtime: "p5",
                renderer_id: "surface.p5",
                preview_eligible: true,
                summary: "Brighter version of the active sketch."
              }
            ]
          }
        },
        {
          event_type: "preview_artifact",
          sequence: 1,
          payload: {
            artifact_id: "make-it-brighter-output",
            status: "succeeded"
          }
        },
        {
          event_type: "final",
          sequence: 2,
          payload: {
            answer: "Brightness refinement received."
          }
        }
      ])
    );

    renderShell(snapshotWithArtifactComparison(), {
      streamAssistantEvents: backendStream
    });

    fireEvent.change(screen.getByLabelText("Assistant prompt"), {
      target: { value: "make it brighter" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send prompt" }));

    expect(await screen.findByText("Brightness refinement received.")).toBeVisible();
    expect(backendStream).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "make it brighter",
        artifactRefinement: expect.objectContaining({
          artifactId: "source-sketch",
          title: "aurora-field.p5.js",
          instruction: "make it brighter",
          content: expect.stringContaining("function draw()")
        })
      })
    );
    expect(screen.getAllByLabelText(/You message at/).at(-1)).toHaveTextContent(
      "make it brighter"
    );
    expect(screen.queryByText(/Clarification:/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "aurora-field.p5.refined.js"
    );
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

    fireEvent.click(within(refinement).getByText("Advanced parameters"));

    fireEvent.change(
      within(refinement).getByLabelText("Movement complexity parameter"),
      {
        target: { value: "9" }
      }
    );
    fireEvent.click(
      within(refinement).getByRole("button", {
        name: "Apply instruction + parameters"
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
          event_type: "artifact_extracted",
          sequence: 0,
          payload: {
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
        },
        {
          event_type: "preview_artifact",
          sequence: 1,
          payload: {
            artifact_id: "source-sketch",
            status: "succeeded"
          }
        },
        {
          event_type: "final",
          sequence: 2,
          payload: {
            answer: "Refined artifact ready."
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
        name: "Apply refinement"
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
    const codePanel = screen.getByRole("tabpanel", { name: "Code" });

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

  it("keeps artifact actions synced without mounting the comparison workspace", () => {
    renderShell(snapshotWithArtifactComparison());

    expect(
      screen.queryByRole("region", { name: "Artifact comparison" })
    ).not.toBeInTheDocument();
    const shaderCandidate = screen.getByLabelText("shader-field.frag artifact");
    fireEvent.click(
      within(shaderCandidate).getByRole("button", {
        name: "Open Preview shader-field.frag"
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
    fireEvent.click(screen.getByRole("tab", { name: "Code" }));

    const codePanel = screen.getByRole("tabpanel", { name: "Code" });

    expect(codePanel).toHaveAttribute("data-opened-artifact", "shader-field.frag");
    expect(
      within(codePanel).getByRole("region", {
        name: "shader-field.frag content"
      })
    ).toHaveTextContent("void main()");

    fireEvent.click(screen.getByRole("tab", { name: "Artifacts" }));
    const codeOnlyArtifact = screen.getByLabelText("field.wgsl artifact");
    fireEvent.click(
      within(codeOnlyArtifact).getByRole("button", {
        name: "Open in Code field.wgsl"
      })
    );

    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "field.wgsl"
    );
    expect(
      screen.getByRole("tabpanel", { name: "Code" })
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
      within(details).queryByRole("region", {
        name: "Creative translation summary"
      })
    ).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("region", {
        name: "Audio-reactive mapping summary"
      })
    ).not.toBeInTheDocument();
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
                referenceFusion: {
                  composition: ["grid-based spatial layout"],
                  geometricStructure: ["rectilinear grid"],
                  lightingContrast: ["soft emissive glow"],
                  moodAtmosphere: ["ethereal atmosphere"],
                  motionImplications: ["slow drifting motion"],
                  paletteDirection: [
                    "warm palette bias",
                    "neon accent contrast"
                  ],
                  runtimeStyleImplications: [
                    "Shader refraction presets may suit the material direction."
                  ],
                  safetyConstraints: [
                    "Use references for aesthetic, palette, composition, and material guidance only."
                  ],
                  sourceCount: 2,
                  sourceNames: ["warm-neon-grid.png", "glass-drift.webp"],
                  summary:
                    "Fused 2 references into non-identifying guidance.",
                  textureMaterialCues: ["glasslike refraction cues"]
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

    expect(
      screen.queryByRole("region", { name: "Creative translation summary" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Audio-reactive mapping summary" })
    ).not.toBeInTheDocument();

    openDashboardKnowledgeBase();
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    expect(
      within(dashboard).getAllByText("Creative Knowledge Base").length
    ).toBeGreaterThan(0);
    expect(
      within(dashboard).getByText(
        "Create an audio-reactive mandala with a meditative pulse."
      )
    ).toBeInTheDocument();
  });

  it("keeps optional creative guidance absent for legacy artifact metadata", () => {
    renderShell(snapshotWithActiveTab("Artifacts"));

    expect(
      screen.queryByRole("region", { name: "Creative translation summary" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Geometry guidance" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Shader preset guidance" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Visual style guidance" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Reference fusion guidance" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Audio-reactive mapping summary" })
    ).not.toBeInTheDocument();
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

  it("confirms artifact deletion, preserves cancel focus, and exposes Undo", async () => {
    const snapshot = snapshotWithActiveTab("Artifacts");
    const artifact =
      snapshot.artifacts.find((candidate) => candidate.isDefault) ??
      snapshot.artifacts[0];
    renderShell(snapshot);

    const details = screen.getByRole("group", { name: "Active artifact details" });
    const deleteButton = within(details).getByRole("button", {
      name: "Delete artifact"
    });

    deleteButton.focus();
    fireEvent.click(deleteButton);
    let dialog = screen.getByRole("alertdialog", {
      name: `Delete ${artifact.title}?`
    });
    const cancelButton = within(dialog).getByRole("button", {
      name: "Keep artifact"
    });
    await waitFor(() => expect(cancelButton).toHaveFocus());

    fireEvent.click(cancelButton);
    await waitFor(() => {
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
      expect(deleteButton).toHaveFocus();
    });
    expect(screen.getByLabelText(`${artifact.title} artifact`)).toBeVisible();

    deleteButton.focus();
    fireEvent.click(deleteButton);
    dialog = screen.getByRole("alertdialog", {
      name: `Delete ${artifact.title}?`
    });
    await act(async () => {
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Delete artifact" })
      );
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
      expect(
        screen.queryByLabelText(`${artifact.title} artifact`)
      ).not.toBeInTheDocument();
    });
    const undoNotice = screen.getByText(`${artifact.title} was deleted.`).closest(
      "[role='status']"
    );
    expect(undoNotice).not.toBeNull();
    expect(within(undoNotice as HTMLElement).getByRole("button", { name: "Undo" }))
      .toBeVisible();
  });

  it("requires approval before downloading an artifact", async () => {
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
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
  });

  it("downloads a live export artifact and exports its current workspace bundle", async () => {
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

    const snapshot = snapshotWithActiveTab("Artifacts");
    const liveExportArtifact: ArtifactSummary = {
      id: "live-response-artifact",
      title: "assistant-response.md",
      type: "export",
      language: "Markdown",
      status: "Generated",
      summary: "Hydrated from the latest live generation output.",
      content: "# Generated response\n\nA live-generated export artifact.",
      actions: ["Open", "Copy", "Download", "Export"]
    };
    renderShell({
      ...snapshot,
      artifacts: [liveExportArtifact]
    });
    const liveArtifact = screen.getByLabelText("assistant-response.md artifact");

    fireEvent.click(
      within(liveArtifact).getByRole("button", {
        name: "Download File assistant-response.md"
      })
    );

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
    expect(screen.getByText("assistant-response.md downloaded.")).toBeVisible();
    const downloadedArtifact = anchorClick.mock
      .instances[0] as unknown as HTMLAnchorElement;
    expect(downloadedArtifact.download).toBe("assistant-response.md");

    fireEvent.click(
      within(liveArtifact).getByRole("button", {
        name: "Export Bundle assistant-response.md"
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
      expect(anchorClick).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByText("Project bundle exported.")).toBeVisible();

    const anchor = anchorClick.mock.instances[1] as unknown as HTMLAnchorElement;
    expect(anchor.download).toBe("local-nextjs-workspace-bundle.zip");

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
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

  it("shows a compact workflow cockpit with live graph states", () => {
    renderShell(snapshotWithActiveTab("Workflow"));

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    expect(workflowPanel).toBeVisible();
    const graph = screen.getByRole("group", {
      name: "LangGraph workflow visualization"
    });

    expect(graph).toBeVisible();
    expect(
      screen.getByRole("group", { name: "Workflow execution decision" })
    ).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Workflow inspector progress" })
    ).toHaveAttribute("aria-valuetext", "11 of 17 workflow nodes reached");
    expect(within(graph).getByText("Generation")).toBeVisible();
    expect(within(graph).getByText("Generation").closest("article")).toHaveAttribute(
      "aria-current",
      "step"
    );
    expect(within(graph).getByText("Context assembly")).toBeVisible();
    expect(within(graph).getByText("Prompt rendering")).toBeVisible();
    for (const deepSurface of [
      "Workflow timeline explorer",
      "Workflow explorer",
      "Provenance summary",
      "Creative timeline",
      "V3 inspector panels",
      "Workflow transition trace",
      "Workflow event trace"
    ]) {
      expect(
        within(workflowPanel).queryByRole("group", { name: deepSurface })
      ).not.toBeInTheDocument();
    }
    expect(screen.queryByText("Preview request")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
  });

  it("keeps streamed workflow state compact and moves transition history to Dashboard", async () => {
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

    expect(
      await screen.findByText(/Code and long-form output are available/)
    ).toBeVisible();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));

    const workflowPanel = screen.getByRole("tabpanel", { name: "Workflow" });
    const workflowGraph = within(workflowPanel).getByRole("group", {
      name: "LangGraph workflow visualization"
    });
    expect(
      screen.getByRole("group", { name: "Workflow execution decision" })
    ).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "Workflow inspector progress" })
    ).toBeVisible();
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow retries" })
    ).not.toBeInTheDocument();
    expect(
      within(workflowPanel).queryByRole("group", {
        name: "Workflow transition trace"
      })
    ).not.toBeInTheDocument();
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow event trace" })
    ).not.toBeInTheDocument();
    expect(
      within(workflowPanel).queryByRole("group", {
        name: "Workflow timeline explorer"
      })
    ).not.toBeInTheDocument();
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

    fireEvent.click(
      screen.getByRole("button", { name: "Open Workflow in Dashboard" })
    );
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    const workflowMap = within(dashboard).getByLabelText("Live workflow map");
    expect(workflowMap).toBeVisible();
    expect(workflowMap).toHaveTextContent("1");
    const recentTransitions = within(workflowMap)
      .getByText(/Recent transitions/)
      .closest("details");
    expect(recentTransitions).not.toBeNull();
    fireEvent.click(
      (recentTransitions as HTMLElement).querySelector("summary") as HTMLElement
    );
    expect(
      within(workflowMap).getByRole("list", { name: "Recent workflow transitions" })
    ).toBeVisible();
  });

  it("keeps provider telemetry in the Telemetry cockpit instead of duplicating it", async () => {
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

    expect(screen.getByLabelText("Current session")).toHaveTextContent(
      "1,500 tokens · $0.0009"
    );
    expect(screen.getByLabelText("Current session")).toHaveTextContent("Total");
    expect(
      screen.queryByRole("group", { name: "Telemetry summary" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Workflow" }));

    const workflowPanel = screen.getByRole("tabpanel", {
      name: "Workflow"
    });
    expect(
      within(workflowPanel).queryByRole("group", { name: "Workflow token usage" })
    ).not.toBeInTheDocument();
    expect(
      within(workflowPanel).queryByRole("group", {
        name: "Workflow cost estimate"
      })
    ).not.toBeInTheDocument();
    expect(
      within(workflowPanel).queryByRole("group", {
        name: "Generation telemetry lifecycle"
      })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Telemetry" }));
    const telemetryPanel = screen.getByRole("tabpanel", {
      name: "Telemetry"
    });
    expect(
      within(telemetryPanel).getByRole("group", { name: "Session usage" })
    ).toHaveTextContent("Latest request1,500 tokens · $0.0009");
    expect(
      within(telemetryPanel).getByRole("group", { name: "Session usage" })
    ).toHaveTextContent("Current session total1,500 tokens · $0.0009");
    expect(
      within(telemetryPanel).getByRole("group", { name: "Provider signal" })
    ).toHaveTextContent("openai / $0.0009");
    expect(
      within(telemetryPanel).getByLabelText("Telemetry signal summary")
    ).toBeVisible();
  });

  it("keeps telemetry evidence concise in Inspector and complete in Dashboard", async () => {
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

    const telemetryPanel = screen.getByRole("tabpanel", {
      name: "Telemetry"
    });
    expect(telemetryPanel).toBeVisible();
    expect(
      within(telemetryPanel).getByLabelText("Telemetry signal summary")
    ).toBeVisible();
    expect(
      within(telemetryPanel).getByRole("group", { name: "Provider signal" })
    ).toHaveTextContent("150 tokens");
    expect(
      within(telemetryPanel).getByRole("group", { name: "Evaluation signal" })
    ).toHaveTextContent("eval-run-1");
    expect(
      within(telemetryPanel).queryByText("Evidence detail")
    ).not.toBeInTheDocument();
    expect(
      within(telemetryPanel).queryByRole("group", {
        name: "Provider observability deep dive"
      })
    ).not.toBeInTheDocument();
    expect(
      within(telemetryPanel).queryByRole("group", {
        name: "Evaluation session dashboard"
      })
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Telemetry in Dashboard" })
    );
    const dashboard = screen.getByRole("region", { name: "Advanced Dashboard" });
    const observatory = within(dashboard).getByLabelText("Telemetry observatory");
    expect(observatory).toBeVisible();
    expect(
      within(observatory).getByRole("list", { name: "Run evidence checkpoints" })
    ).toBeVisible();
    expect(
      within(observatory).getByLabelText("Run measurement facts")
    ).toHaveTextContent("150 tokens");
    expect(
      within(observatory).getByLabelText("Run measurement facts")
    ).toHaveTextContent("Cost pending");
    const profileUsage = within(dashboard).getByLabelText("Browser profile usage");
    expect(
      within(profileUsage).getByRole("table", { name: "Session token and cost totals" })
    ).toHaveTextContent("Grand total");
    expect(
      within(profileUsage).getByRole("table", { name: "Session token and cost totals" })
    ).toHaveTextContent("150 tokens");
    const evidenceDisclosure = within(observatory)
      .getByText("Provider, observability, and evaluation evidence")
      .closest("details");
    expect(evidenceDisclosure).not.toBeNull();
    fireEvent.click(
      (evidenceDisclosure as HTMLElement).querySelector("summary") as HTMLElement
    );
    expect(
      within(observatory).getByLabelText("Supporting telemetry signals")
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

    fireEvent.click(getWorkspaceSettingsControl("Compact"));

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
    expect(screen.getAllByText("Restored projection session").length).toBeGreaterThan(0);
    expect(getWorkspaceSettingsControl("Display mode")).toHaveTextContent(
      "User"
    );
    await waitFor(() =>
      expect(screen.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
        "data-state",
        "collapsed"
      )
    );
    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Telemetry" })).not.toBeInTheDocument();
    await waitFor(() =>
      expect(persistenceClient.save).toHaveBeenCalledWith(
        expect.objectContaining({
          layout: expect.objectContaining({
            inspectorCollapsed: true
          }),
          preferences: expect.objectContaining({
            showDebugPanels: false
          })
        })
      )
    );
    expect(await screen.findByText("Session saved")).toBeVisible();
    vi.mocked(persistenceClient.save).mockClear();

    fireEvent.click(screen.getByRole("button", { name: "Expand inspector" }));

    expect(screen.getByRole("tab", { name: "Preview" })).toBeVisible();
    expect(screen.getByRole("tab", { name: "Code" })).toBeVisible();
    expect(screen.getByRole("tab", { name: "Saved" })).toHaveAttribute(
      "aria-selected",
      "true"
    );
    expect(screen.queryByRole("tab", { name: "Artifacts" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Retrieval" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Preview",
      "Code",
      "Saved"
    ]);
    expect(screen.queryByRole("tab", { name: "Workflow" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Telemetry" })).not.toBeInTheDocument();
    await waitFor(() =>
      expect(persistenceClient.save).toHaveBeenCalledWith(
        expect.objectContaining({
          layout: expect.objectContaining({
            inspectorCollapsed: false
          })
        })
      )
    );
    expect(screen.getByLabelText("Active artifact")).toHaveTextContent(
      "Generated Code"
    );
    expect(screen.queryByText("projection-notes.md")).not.toBeInTheDocument();
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
  });

  it("repairs a stale restored active artifact before a later save", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const persistedRecord = createWorkspaceSessionRecord({
      activeArtifactId: "missing-artifact",
      activeInspectorTab: "Code",
      previewArtifactId: "missing-preview-artifact",
      previewOpen: true,
      snapshot
    });
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({
        error: null,
        record: persistedRecord,
        source: "remote" as const
      })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    renderShell(snapshot, { persistenceClient }, { mode: "user" });

    await waitFor(() =>
      expect(document.querySelector(".workstation")).toHaveAttribute(
        "data-active-tab",
        "code"
      )
    );
    vi.mocked(persistenceClient.save).mockClear();

    fireEvent.click(getWorkspaceSettingsControl("Compact"));

    await waitFor(() =>
      expect(persistenceClient.save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          activeArtifactId: "source-sketch",
          previewArtifactId: "source-sketch"
        })
      )
    );
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

  it("updates persistence status after Strict Mode remounts the shell", async () => {
    const persistenceClient: WorkspacePersistenceClient = {
      load: vi.fn(async () => ({ error: null, record: null, source: "none" as const })),
      save: vi.fn(async () => ({ error: null, target: "remote" as const }))
    };

    render(
      <StrictMode>
        <WorkstationShell
          snapshot={getLocalWorkspaceSnapshot()}
          persistenceClient={persistenceClient}
        />
      </StrictMode>
    );

    expect(await screen.findByText("Session saved")).toBeVisible();
    expect(persistenceClient.save).toHaveBeenCalled();
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
