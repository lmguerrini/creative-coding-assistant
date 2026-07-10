import type {
  ArtifactSummary,
  PreviewSummary,
  PreviewTargetId
} from "./assistant-client";
import {
  getGsapRuntimeSupportIssue,
  hasGsapPreviewSignal
} from "./gsap-runtime";
import {
  derivePreviewTargetIdFromArtifact,
  formatPreviewTargetLabel,
  normalizePreviewTargetId
} from "./preview-targets";
import {
  getCanvasRuntimeSupportIssue,
  getSvgRuntimeSupportIssue,
  hasCanvasPreviewSignal,
  hasSvgPreviewSignal
} from "./svg-canvas-runtime";
import {
  getP5RuntimeSourceSupportIssue,
  getThreeRuntimeSourceSupportIssue
} from "./preview-source-classification";

export type PreviewRendererTone =
  | "active"
  | "success"
  | "warning"
  | "danger"
  | "muted";

export type CreativePreviewRendererKind =
  | "p5"
  | "three"
  | "glsl"
  | "hydra"
  | "tone"
  | "gsap"
  | "svg"
  | "canvas";

export type PreviewRendererSurfaceKind =
  | CreativePreviewRendererKind
  | "json_panel"
  | "text_panel"
  | "image_asset"
  | "audio_asset"
  | "video_asset"
  | "unsupported";

export type PreviewRendererSupportState =
  | "supported"
  | "unsupported"
  | "unavailable";

export type CreativePreviewRendererDefinition = {
  id: string;
  kind: CreativePreviewRendererKind;
  displayName: string;
  surfaceLabel: string;
  description: string;
  matchExtensions: readonly string[];
  matchTokens: readonly string[];
  notes: readonly string[];
};

export type PreviewRendererRoute = {
  targetId: PreviewTargetId | null;
  targetLabel: string;
  selectedArtifactId: string | null;
  selectedArtifactName: string;
  sourceArtifactId: string | null;
  sourceArtifactName: string;
  rendererId: string | null;
  rendererLabel: string;
  rendererDescription: string;
  supportState: PreviewRendererSupportState;
  supportLabel: string;
  supportReason: string;
  surfaceKind: PreviewRendererSurfaceKind;
  surfaceTitle: string;
  surfaceEyebrow: string;
  surfaceSummary: string;
  notes: readonly string[];
  tone: PreviewRendererTone;
};

export const creativePreviewRendererRegistry: readonly CreativePreviewRendererDefinition[] = [
  {
    id: "surface.p5",
    kind: "p5",
    displayName: "p5.js",
    surfaceLabel: "P5 sketch surface",
    description: "2D sketch loop surface for p5.js browser previews.",
    matchExtensions: [".p5.js", ".p5.ts"],
    matchTokens: ["p5", "createcanvas", "background(", "draw("],
    notes: [
      "Controlled p5-compatible browser runtime",
      "Runs generated sketch source inside an isolated preview frame",
      "Runtime status, frames, and errors stream back to the inspector"
    ]
  },
  {
    id: "surface.three",
    kind: "three",
    displayName: "Three.js",
    surfaceLabel: "Three scene surface",
    description: "Scene-oriented surface for Three.js browser previews.",
    matchExtensions: [".three.js", ".three.ts", ".r3f.tsx"],
    matchTokens: [
      "three",
      "webglrenderer",
      "scene",
      "perspectivecamera",
      "react-three"
    ],
    notes: [
      "Controlled Three.js-compatible browser runtime",
      "Runs generated scene source inside an isolated preview frame",
      "Reset and reload remount the 3D runtime document"
    ]
  },
  {
    id: "surface.glsl",
    kind: "glsl",
    displayName: "GLSL",
    surfaceLabel: "Shader surface",
    description: "Fullscreen shader surface for GLSL browser previews.",
    matchExtensions: [".glsl", ".frag", ".vert", ".fs", ".vs"],
    matchTokens: [
      "glsl",
      "fragment shader",
      "vertex shader",
      "gl_fragcolor",
      "void main"
    ],
    notes: [
      "Controlled WebGL fragment runtime",
      "Compiles fragment shaders inside an isolated preview frame",
      "Rejects unsupported shader features with a visible runtime error"
    ]
  },
  {
    id: "surface.hydra",
    kind: "hydra",
    displayName: "Hydra",
    surfaceLabel: "Hydra synth surface",
    description: "Feedback-oriented synth surface for Hydra browser previews.",
    matchExtensions: [".hydra.js", ".hydra.ts"],
    matchTokens: ["hydra", "osc(", "voronoi(", "modulate(", ".out("],
    notes: [
      "Controlled Hydra-compatible browser runtime",
      "Parses supported source chains into a bounded execution plan",
      "Feedback frames, status, and errors stay isolated in the preview sandbox"
    ]
  },
  {
    id: "surface.tone",
    kind: "tone",
    displayName: "Tone.js",
    surfaceLabel: "Tone.js audio surface",
    description: "Generative audio surface for controlled Tone.js-compatible previews.",
    matchExtensions: [".tone.js", ".tone.ts"],
    matchTokens: [
      "tone.js",
      "tone.synth",
      "tone.oscillator",
      "tone.sequence",
      "tone.loop",
      "tone.transport"
    ],
    notes: [
      "Controlled Tone.js-compatible Web Audio runtime",
      "Audio remains silent until the operator explicitly starts playback",
      "Stop, mute, lifecycle status, and runtime errors stay inside the preview sandbox"
    ]
  },
  {
    id: "surface.gsap",
    kind: "gsap",
    displayName: "GSAP",
    surfaceLabel: "GSAP motion stage",
    description: "DOM motion stage for bounded GSAP browser previews.",
    matchExtensions: [".gsap.js", ".gsap.ts"],
    matchTokens: ["gsap.", "gsap.timeline", "stagger:", "repeat:", "yoyo:"],
    notes: [
      "Controlled GSAP-compatible DOM motion runtime",
      "Targets only the bounded sandbox stage and its prebuilt nodes",
      "Rejects plugins, remote assets, and unrestricted DOM access"
    ]
  },
  {
    id: "surface.svg",
    kind: "svg",
    displayName: "SVG",
    surfaceLabel: "SVG vector stage",
    description: "Sanitized inline SVG stage for bounded browser previews.",
    matchExtensions: [".svg"],
    matchTokens: ["<svg", "viewbox=", "<path", "<circle", "<animate"],
    notes: [
      "Sanitized inline SVG runtime",
      "Allows self-contained vector markup and deterministic native SVG animation",
      "Rejects scriptable containers, event handlers, and remote assets"
    ]
  },
  {
    id: "surface.canvas",
    kind: "canvas",
    displayName: "Canvas",
    surfaceLabel: "Canvas 2D stage",
    description: "Bounded HTML5 Canvas 2D surface for deterministic browser previews.",
    matchExtensions: [".canvas.js", ".canvas.ts"],
    matchTokens: [
      "getcontext(",
      "fillrect(",
      "clearrect(",
      "requestanimationframe(",
      "canvasrenderingcontext2d"
    ],
    notes: [
      "Controlled Canvas 2D runtime",
      "Runs deterministic drawing code against the provided preview surface only",
      "Rejects DOM mutation, remote assets, and interactive input handlers"
    ]
  }
] as const;

const supportedPreviewDomains = new Set([
  "p5_js",
  "glsl",
  "hydra",
  "tone_js",
  "gsap",
  "svg",
  "svg_markup",
  "canvas",
  "canvas_2d",
  "three_js",
  "react_three_fiber"
]);

const unsupportedBrowserRuntimeExtensions = [
  ".wgsl",
  ".webgpu.js",
  ".webgpu.ts"
] as const;

const supportedBrowserFoundationSummary =
  "Current browser preview foundations cover p5.js, Three.js, GLSL, Hydra, Tone.js, GSAP, SVG, and Canvas only.";

export function buildPreviewRendererRoute({
  artifacts,
  preview,
  previewArtifactId
}: {
  artifacts: ArtifactSummary[];
  preview: PreviewSummary;
  previewArtifactId: string;
}): PreviewRendererRoute {
  const selectedArtifact =
    artifacts.find((artifact) => artifact.id === previewArtifactId) ??
    artifacts.find((artifact) => artifact.title === preview.artifactName) ??
    artifacts.find((artifact) => artifact.id === preview.sourceArtifactId) ??
    artifacts.find((artifact) => artifact.title === preview.sourceArtifactName) ??
    null;
  const sourceArtifact =
    artifacts.find((artifact) => artifact.id === preview.sourceArtifactId) ??
    artifacts.find((artifact) => artifact.title === preview.sourceArtifactName) ??
    selectedArtifact;
  const targetId =
    derivePreviewTargetIdFromArtifact(selectedArtifact) ??
    normalizePreviewTargetId(preview.targetId) ??
    derivePreviewTargetIdFromArtifact(sourceArtifact);
  const targetLabel =
    formatPreviewTargetLabel(targetId) ?? (preview.target || "Pending target");
  const selectedArtifactName =
    selectedArtifact?.title ?? preview.artifactName ?? "No preview artifact";
  const sourceArtifactName =
    sourceArtifact?.title ??
    preview.sourceArtifactName ??
    preview.artifactName ??
    selectedArtifactName;
  const contextPreviewable = isArtifactPreviewable(selectedArtifact);
  const tone = resolvePreviewRendererTone(preview, contextPreviewable, targetId);

  if (preview.state === "unavailable" && !contextPreviewable) {
    return {
      targetId,
      targetLabel,
      selectedArtifactId: selectedArtifact?.id ?? null,
      selectedArtifactName,
      sourceArtifactId: sourceArtifact?.id ?? null,
      sourceArtifactName,
      rendererId: null,
      rendererLabel: "No preview surface",
      rendererDescription: "The selected artifact is not wired to a live preview surface yet.",
      supportState: "unavailable",
      supportLabel: "Unavailable",
      supportReason: `${selectedArtifactName} is not currently linked to a preview surface in this workspace.`,
      surfaceKind: "unsupported",
      surfaceTitle: "Preview unavailable",
      surfaceEyebrow: "No routed surface",
      surfaceSummary:
        "This artifact remains visible in the workspace, but it does not have a preview surface contract yet.",
      notes: [
        "Select a preview-capable artifact to restore the live shelf",
        "Renderer routing stays disabled until a compatible preview target exists"
      ],
      tone
    };
  }

  if (!targetId) {
    return {
      targetId: null,
      targetLabel,
      selectedArtifactId: selectedArtifact?.id ?? null,
      selectedArtifactName,
      sourceArtifactId: sourceArtifact?.id ?? null,
      sourceArtifactName,
      rendererId: null,
      rendererLabel: "Pending target",
      rendererDescription: "A preview target has not been resolved yet.",
      supportState: "unavailable",
      supportLabel: "Pending",
      supportReason: "The preview runtime has not resolved a target for this artifact yet.",
      surfaceKind: "unsupported",
      surfaceTitle: "Preview route pending",
      surfaceEyebrow: "Awaiting target",
      surfaceSummary:
        "The runtime surface is waiting for preview target metadata before a renderer container can be selected.",
      notes: [
        "Target resolution happens before renderer selection",
        "Live runtime execution starts after a supported target is selected"
      ],
      tone
    };
  }

  if (targetId === "browser_sandbox") {
    const rendererArtifact =
      selectedArtifact?.type === "code" ? selectedArtifact : sourceArtifact ?? selectedArtifact;
    const matchedRenderer = matchCreativePreviewRenderer(rendererArtifact);
    // Live artifacts carry their extracted source and are rejected here before
    // they are advertised as previewable. Local/legacy snapshot artifacts may
    // intentionally omit duplicated content; their active Code summary is
    // validated again immediately before the sandbox mounts it.
    const p5SupportIssue =
      rendererArtifact?.content?.trim() && hasP5PreviewContract(rendererArtifact)
        ? getP5RuntimeSourceSupportIssue(rendererArtifact.content)
        : null;
    const threeSupportIssue =
      rendererArtifact?.content?.trim() && rendererArtifact.runtime === "three"
        ? getThreeRuntimeSourceSupportIssue(rendererArtifact.content)
        : null;
    const gsapSupportIssue =
      rendererArtifact && hasGsapPreviewSignal(rendererArtifact)
        ? getGsapRuntimeSupportIssue(rendererArtifact.content)
        : null;
    const svgSupportIssue =
      rendererArtifact && hasSvgPreviewSignal(rendererArtifact)
        ? getSvgRuntimeSupportIssue(rendererArtifact.content)
        : null;
    const canvasSupportIssue =
      rendererArtifact && hasCanvasPreviewSignal(rendererArtifact)
        ? getCanvasRuntimeSupportIssue(rendererArtifact.content)
        : null;
    const runtimeSupportIssue =
      p5SupportIssue ??
      threeSupportIssue ??
      gsapSupportIssue ??
      svgSupportIssue ??
      canvasSupportIssue;
    const unsupportedSurfaceSummary = p5SupportIssue
      ? `${sourceArtifactName} was identified as a p5 artifact, but the current source is not executable JavaScript for the p5 runtime.`
      : threeSupportIssue
        ? `${sourceArtifactName} was identified as a Three.js artifact, but its current source does not meet the controlled JavaScript runtime contract.`
      : gsapSupportIssue
        ? `${sourceArtifactName} was identified as a GSAP motion artifact, but the current source exceeds the bounded sandbox rules for live execution.`
        : svgSupportIssue
        ? `${sourceArtifactName} was identified as an SVG artifact, but the current source exceeds the bounded sandbox rules for live execution.`
        : canvasSupportIssue
          ? `${sourceArtifactName} was identified as a Canvas artifact, but the current source exceeds the bounded sandbox rules for live execution.`
          : `${sourceArtifactName} still resolves to the browser preview, but no safe live renderer foundation matches its current signals yet.`;
    const unsupportedNotes = p5SupportIssue
      ? [
          "The artifact remains inspectable as code",
          "Use JavaScript p5 source with setup() or draw() to restore live preview support",
          "HTML documents are not executed inside the p5 JavaScript runtime"
        ]
      : threeSupportIssue
        ? [
            "The artifact remains inspectable as code",
            "Use plain self-contained Three.js JavaScript to restore live preview support",
            "Standalone HTML and React Three Fiber artifacts remain export-only in this controlled runtime"
          ]
      : gsapSupportIssue
      ? [
          "The artifact remains inspectable as code",
          "Remove plugin, network, or unrestricted DOM patterns to restore live preview support",
          "Only bounded GSAP tweens and timelines can execute in the browser preview"
        ]
      : svgSupportIssue
        ? [
            "The artifact remains inspectable as code",
            "Remove scriptable containers, event handlers, or external asset patterns to restore live preview support",
            "Only sanitized inline SVG markup can execute in the browser preview"
          ]
        : canvasSupportIssue
          ? [
              "The artifact remains inspectable as code",
              "Remove DOM, image-asset, or interactive input patterns to restore live preview support",
              "Only bounded Canvas 2D drawing and deterministic timers can execute in the browser preview"
            ]
          : [
              "Runtime target metadata is still preserved",
              "The preview shelf stays stable without executing arbitrary code",
              "A dedicated WebGPU or browser runtime surface can be added later"
            ];

    if (matchedRenderer) {
      return {
        targetId,
        targetLabel,
        selectedArtifactId: selectedArtifact?.id ?? null,
        selectedArtifactName,
        sourceArtifactId: sourceArtifact?.id ?? null,
        sourceArtifactName,
        rendererId: matchedRenderer.id,
        rendererLabel: matchedRenderer.displayName,
        rendererDescription: matchedRenderer.description,
        supportState: "supported",
        supportLabel: "Runtime ready",
        supportReason: `${matchedRenderer.displayName} matches the current browser preview artifact signals.`,
        surfaceKind: matchedRenderer.kind,
        surfaceTitle: matchedRenderer.surfaceLabel,
        surfaceEyebrow: "Renderer match",
        surfaceSummary: `${matchedRenderer.displayName} is selected as the live surface for ${sourceArtifactName}. Supported runtimes execute in an isolated browser preview frame.`,
        notes: matchedRenderer.notes,
        tone
      };
    }

    return {
      targetId,
      targetLabel,
      selectedArtifactId: selectedArtifact?.id ?? null,
      selectedArtifactName,
      sourceArtifactId: sourceArtifact?.id ?? null,
      sourceArtifactName,
      rendererId: null,
      rendererLabel: "No matching live renderer",
      rendererDescription:
        "The browser preview target is available, but the artifact does not match the current creative renderer foundations.",
      supportState: "unsupported",
      supportLabel: "Unsupported",
      supportReason:
        runtimeSupportIssue ?? supportedBrowserFoundationSummary,
      surfaceKind: "unsupported",
      surfaceTitle: "Browser preview without renderer match",
      surfaceEyebrow: "Unsupported creative surface",
      surfaceSummary: unsupportedSurfaceSummary,
      notes: unsupportedNotes,
      tone
    };
  }

  return buildNonBrowserPreviewRendererRoute({
    selectedArtifact,
    selectedArtifactName,
    sourceArtifact,
    sourceArtifactName,
    targetId,
    targetLabel,
    tone
  });
}

export function matchCreativePreviewRenderer(
  artifact: ArtifactSummary | null
): CreativePreviewRendererDefinition | null {
  if (!artifact) {
    return null;
  }

  const haystack = [artifact.title, artifact.language, artifact.summary, artifact.content]
    .join(" ")
    .trim()
    .toLowerCase();
  const normalizedTitle = artifact.title.trim().toLowerCase();

  if (hasUnsupportedBrowserRuntimeSignal(artifact, normalizedTitle)) {
    return null;
  }

  const explicitRenderer = matchExplicitPreviewRenderer(artifact);

  if (explicitRenderer !== undefined) {
    return validateCreativePreviewRenderer(artifact, explicitRenderer);
  }

  const signaledRenderer = matchSignaledPreviewRenderer(artifact);

  if (signaledRenderer) {
    return validateCreativePreviewRenderer(artifact, signaledRenderer);
  }

  const extensionMatch = creativePreviewRendererRegistry.find((renderer) =>
    renderer.matchExtensions.some((extension) => normalizedTitle.endsWith(extension))
  );

  if (extensionMatch) {
    return validateCreativePreviewRenderer(artifact, extensionMatch);
  }

  return validateCreativePreviewRenderer(
    artifact,
    creativePreviewRendererRegistry.find((renderer) => {
      return renderer.matchTokens.some((token) => haystack.includes(token));
    }) ?? null
  );
}

function matchExplicitPreviewRenderer(
  artifact: ArtifactSummary
): CreativePreviewRendererDefinition | null | undefined {
  const rendererId = artifact.rendererId?.trim();
  if (rendererId) {
    return (
      creativePreviewRendererRegistry.find((renderer) => renderer.id === rendererId) ??
      null
    );
  }

  const runtime = artifact.runtime?.trim().toLowerCase();
  if (!runtime) {
    return undefined;
  }

  return (
    creativePreviewRendererRegistry.find((renderer) => renderer.kind === runtime) ??
    null
  );
}

function matchSignaledPreviewRenderer(artifact: ArtifactSummary) {
  if (hasGsapPreviewSignal(artifact)) {
    return creativePreviewRendererRegistry.find((renderer) => renderer.kind === "gsap");
  }

  if (hasSvgPreviewSignal(artifact)) {
    return creativePreviewRendererRegistry.find((renderer) => renderer.kind === "svg");
  }

  if (hasCanvasPreviewSignal(artifact)) {
    return creativePreviewRendererRegistry.find((renderer) => renderer.kind === "canvas");
  }

  return null;
}

function hasUnsupportedBrowserRuntimeSignal(
  artifact: ArtifactSummary,
  normalizedTitle: string
) {
  const domain = artifact.domain?.trim().toLowerCase();
  if (domain && !supportedPreviewDomains.has(domain)) {
    return true;
  }

  return unsupportedBrowserRuntimeExtensions.some((extension) =>
    normalizedTitle.endsWith(extension)
  );
}

function validateCreativePreviewRenderer(
  artifact: ArtifactSummary,
  renderer: CreativePreviewRendererDefinition | null | undefined
) {
  if (!renderer) {
    return renderer ?? null;
  }

  if (renderer.kind !== "gsap") {
    if (renderer.kind === "p5") {
      return artifact.content?.trim() && getP5RuntimeSourceSupportIssue(artifact.content)
        ? null
        : renderer;
    }

    if (renderer.kind === "three") {
      return artifact.content?.trim() && getThreeRuntimeSourceSupportIssue(artifact.content)
        ? null
        : renderer;
    }

    if (renderer.kind === "svg") {
      return getSvgRuntimeSupportIssue(artifact.content) ? null : renderer;
    }

    if (renderer.kind === "canvas") {
      return getCanvasRuntimeSupportIssue(artifact.content) ? null : renderer;
    }

    return renderer;
  }

  return getGsapRuntimeSupportIssue(artifact.content) ? null : renderer;
}

function hasP5PreviewContract({ rendererId, runtime, title }: ArtifactSummary) {
  const normalizedRenderer = rendererId?.trim().toLowerCase();
  const normalizedRuntime = runtime?.trim().toLowerCase();
  const normalizedTitle = title.trim().toLowerCase();

  return (
    normalizedRenderer === "surface.p5" ||
    normalizedRuntime === "p5" ||
    normalizedTitle.endsWith(".p5.js") ||
    normalizedTitle.endsWith(".p5.ts")
  );
}

function buildNonBrowserPreviewRendererRoute({
  selectedArtifact,
  selectedArtifactName,
  sourceArtifact,
  sourceArtifactName,
  targetId,
  targetLabel,
  tone
}: {
  selectedArtifact: ArtifactSummary | null;
  selectedArtifactName: string;
  sourceArtifact: ArtifactSummary | null;
  sourceArtifactName: string;
  targetId: Exclude<PreviewTargetId, "browser_sandbox">;
  targetLabel: string;
  tone: PreviewRendererTone;
}): PreviewRendererRoute {
  const sharedRoute = {
    targetId,
    targetLabel,
    selectedArtifactId: selectedArtifact?.id ?? null,
    selectedArtifactName,
    sourceArtifactId: sourceArtifact?.id ?? null,
    sourceArtifactName,
    supportState: "supported" as const,
    supportLabel: "Foundation ready",
    tone
  };

  switch (targetId) {
    case "json_panel":
      return {
        ...sharedRoute,
        rendererId: "surface.json_panel",
        rendererLabel: "JSON panel surface",
        rendererDescription: "Structured preview manifest panel.",
        supportReason: "Preview manifests route into a structured JSON panel surface.",
        surfaceKind: "json_panel",
        surfaceTitle: "Preview manifest panel",
        surfaceEyebrow: "Structured surface",
        surfaceSummary: `${selectedArtifactName} is routed to a JSON panel surface so renderer metadata can stay visible without executing a creative runtime yet.`,
        notes: [
          "Safe manifest viewer foundation",
          "Useful for preview metadata and renderer routing inspection",
          "Future runtime surfaces can consume the same manifest contract"
        ]
      };
    case "text_panel":
      return {
        ...sharedRoute,
        rendererId: "surface.text_panel",
        rendererLabel: "Text panel surface",
        rendererDescription: "Readable text preview panel.",
        supportReason: "Text previews can route to a lightweight panel surface.",
        surfaceKind: "text_panel",
        surfaceTitle: "Text preview panel",
        surfaceEyebrow: "Readable surface",
        surfaceSummary: `${selectedArtifactName} is prepared for a simple text panel surface instead of a browser runtime.`,
        notes: [
          "Lightweight text container foundation",
          "No external renderer runtime required",
          "Can later expand into richer diff or prose previews"
        ]
      };
    case "image_asset":
      return {
        ...sharedRoute,
        rendererId: "surface.image_asset",
        rendererLabel: "Image asset surface",
        rendererDescription: "Dedicated still-image container.",
        supportReason: "Static image previews can mount into a media-specific asset surface.",
        surfaceKind: "image_asset",
        surfaceTitle: "Image preview surface",
        surfaceEyebrow: "Media surface",
        surfaceSummary: `${selectedArtifactName} is routed to an image asset surface rather than a browser renderer.`,
        notes: [
          "Still image preview container",
          "Prepared for future image asset loading and inspection"
        ]
      };
    case "audio_asset":
      return {
        ...sharedRoute,
        rendererId: "surface.audio_asset",
        rendererLabel: "Audio asset surface",
        rendererDescription: "Dedicated audio playback container.",
        supportReason: "Audio previews use a media-specific asset surface.",
        surfaceKind: "audio_asset",
        surfaceTitle: "Audio preview surface",
        surfaceEyebrow: "Media surface",
        surfaceSummary: `${selectedArtifactName} is routed to an audio asset surface without relying on a browser sketch renderer.`,
        notes: [
          "Playback surface frame",
          "Prepared for future waveform, transport, and output controls"
        ]
      };
    case "video_asset":
      return {
        ...sharedRoute,
        rendererId: "surface.video_asset",
        rendererLabel: "Video asset surface",
        rendererDescription: "Dedicated video playback container.",
        supportReason: "Video previews use a media-specific asset surface.",
        surfaceKind: "video_asset",
        surfaceTitle: "Video preview surface",
        surfaceEyebrow: "Media surface",
        surfaceSummary: `${selectedArtifactName} is routed to a video asset surface instead of a creative browser runtime.`,
        notes: [
          "Playback surface frame",
          "Prepared for future video mounting and transport controls"
        ]
      };
  }
}

function isArtifactPreviewable(artifact: ArtifactSummary | null) {
  return artifact ? artifact.actions.includes("Preview") || artifact.type === "preview" : false;
}

function resolvePreviewRendererTone(
  preview: PreviewSummary,
  contextPreviewable: boolean,
  targetId: PreviewTargetId | null
): PreviewRendererTone {
  if (preview.state === "error") {
    return "danger";
  }

  if (preview.state === "generating") {
    return "active";
  }

  if (preview.state === "unavailable" || !contextPreviewable || !targetId) {
    return "warning";
  }

  return "success";
}
