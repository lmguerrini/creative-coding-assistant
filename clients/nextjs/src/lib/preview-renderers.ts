import type {
  ArtifactSummary,
  PreviewSummary,
  PreviewTargetId
} from "./assistant-client";
import {
  derivePreviewTargetIdFromArtifact,
  formatPreviewTargetLabel,
  normalizePreviewTargetId
} from "./preview-targets";

export type PreviewRendererTone =
  | "active"
  | "success"
  | "warning"
  | "danger"
  | "muted";

export type CreativePreviewRendererKind = "p5" | "three" | "glsl" | "hydra";

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
      "Constrained canvas sketch runtime",
      "Interprets simple p5 sketch signals without executing generated JavaScript",
      "Reset and reload remount the runtime surface"
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
      "Controlled WebGL scene runtime",
      "Parses scene hints without executing generated JavaScript",
      "Reset and reload remount the 3D stage"
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
      "Bounded WebGL fragment runtime",
      "Compiles fragment shaders without JavaScript evaluation",
      "Rejects unsupported shader features with a visible runtime error"
    ]
  },
  {
    id: "surface.hydra",
    kind: "hydra",
    displayName: "Hydra",
    surfaceLabel: "Hydra patch surface",
    description: "Node-driven Hydra surface for browser preview routing.",
    matchExtensions: [".hydra.js", ".hydra.ts"],
    matchTokens: ["hydra", "osc(", "shape(", "render(", "out("],
    notes: [
      "Patch graph placeholder",
      "Safe routing surface only on this branch",
      "Prepared for future live Hydra graph execution"
    ]
  }
] as const;

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
        "No creative runtime engine is executed on this branch"
      ],
      tone
    };
  }

  if (targetId === "browser_sandbox") {
    const rendererArtifact =
      selectedArtifact?.type === "code" ? selectedArtifact : sourceArtifact ?? selectedArtifact;
    const matchedRenderer = matchCreativePreviewRenderer(rendererArtifact);

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
        supportLabel: "Foundation ready",
        supportReason: `${matchedRenderer.displayName} matches the current browser-sandbox artifact signals.`,
        surfaceKind: matchedRenderer.kind,
        surfaceTitle: matchedRenderer.surfaceLabel,
        surfaceEyebrow: "Renderer match",
        surfaceSummary: `${matchedRenderer.displayName} is selected as the live surface for ${sourceArtifactName}. Supported runtimes mount in place while unsupported engines stay in placeholder mode.`,
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
        "The browser sandbox target is available, but the artifact does not match the current creative renderer foundations.",
      supportState: "unsupported",
      supportLabel: "Unsupported",
      supportReason:
        "Current browser-sandbox foundations cover p5.js, Three.js, GLSL, and Hydra only.",
      surfaceKind: "unsupported",
      surfaceTitle: "Browser route without renderer match",
      surfaceEyebrow: "Unsupported creative surface",
      surfaceSummary: `${sourceArtifactName} still resolves to the browser sandbox, but no safe live renderer foundation matches its current signals yet.`,
      notes: [
        "Runtime target metadata is still preserved",
        "The preview shelf stays stable without executing arbitrary code",
        "A dedicated WebGPU or browser runtime surface can be added later"
      ],
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

  return (
    creativePreviewRendererRegistry.find((renderer) => {
      const normalizedTitle = artifact.title.trim().toLowerCase();
      return (
        renderer.matchExtensions.some((extension) => normalizedTitle.endsWith(extension)) ||
        renderer.matchTokens.some((token) => haystack.includes(token))
      );
    }) ?? null
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
          "Still image placeholder container",
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
          "Playback surface placeholder",
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
          "Playback surface placeholder",
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
