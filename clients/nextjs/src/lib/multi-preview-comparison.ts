import type {
  ArtifactSummary,
  CodeSummary,
  PreviewSummary,
  PreviewTargetId
} from "./assistant-client";
import {
  buildArtifactComparisonModel,
  type ArtifactComparisonRow
} from "./artifact-comparison";
import {
  buildPreviewRendererRoute,
  type PreviewRendererRoute
} from "./preview-renderers";
import {
  buildPreviewRuntimeSource,
  getExecutablePreviewRuntimeKind,
  type PreviewRuntimeSource
} from "./preview-runtime-adapters";
import {
  derivePreviewTargetIdFromArtifact,
  formatPreviewTargetLabel
} from "./preview-targets";

export type MultiPreviewLayout = "empty" | "single" | "split" | "grid";

export type MultiPreviewOutputKind =
  | "visual"
  | "audio"
  | "audiovisual"
  | "code";

export type MultiPreviewCandidate = {
  artifact: ArtifactSummary;
  audioSafetyLabel: string;
  canRender: boolean;
  geometryLabels: string[];
  outputKind: MultiPreviewOutputKind;
  outputLabel: string;
  preview: PreviewSummary;
  route: PreviewRendererRoute;
  row: ArtifactComparisonRow;
  runtimeSessionKey: string;
  runtimeSource: PreviewRuntimeSource;
  shaderPresetLabels: string[];
  visualStyleLabels: string[];
};

export type MultiPreviewComparisonModel = {
  candidates: MultiPreviewCandidate[];
  layout: MultiPreviewLayout;
  recommendedReason: string;
  recommendedTitle: string | null;
};

export function buildMultiPreviewComparisonModel({
  activeArtifactId,
  artifacts,
  code,
  preview
}: {
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
  code: CodeSummary;
  preview: PreviewSummary;
}): MultiPreviewComparisonModel {
  const comparison = buildArtifactComparisonModel({
    activeArtifactId,
    artifacts
  });
  const candidates = comparison.rows.map((row) =>
    buildMultiPreviewCandidate({
      baseCode: code,
      basePreview: preview,
      artifacts,
      row
    })
  );

  return {
    candidates,
    layout: resolveMultiPreviewLayout(candidates.length),
    recommendedReason: comparison.recommendedReason,
    recommendedTitle: comparison.recommendedRow?.title ?? null
  };
}

export function resolveMultiPreviewLayout(
  candidateCount: number
): MultiPreviewLayout {
  if (candidateCount <= 0) {
    return "empty";
  }

  if (candidateCount === 1) {
    return "single";
  }

  if (candidateCount === 2) {
    return "split";
  }

  return "grid";
}

function buildMultiPreviewCandidate({
  artifacts,
  baseCode,
  basePreview,
  row
}: {
  artifacts: ArtifactSummary[];
  baseCode: CodeSummary;
  basePreview: PreviewSummary;
  row: ArtifactComparisonRow;
}): MultiPreviewCandidate {
  const artifact = row.artifact;
  const targetId = derivePreviewTargetIdFromArtifact(artifact);
  const outputKind = resolveOutputKind(artifact);
  const hasMatchingBasePreview = matchesBasePreview(artifact, basePreview);
  const preview = buildCandidatePreview({
    artifact,
    basePreview,
    hasMatchingBasePreview,
    targetId
  });
  const route = buildPreviewRendererRoute({
    artifacts,
    preview,
    previewArtifactId: artifact.id
  });
  const runtimeSource = buildPreviewRuntimeSource({
    code: buildCandidateCodeSummary(artifact, baseCode),
    route
  });
  const executableRuntime = getExecutablePreviewRuntimeKind(route);
  const canRender =
    row.runtimeSupport.state === "previewable" &&
    (executableRuntime === null || runtimeSource.lineCount > 0);
  const resolvedPreview = canRender
    ? preview
    : {
        ...preview,
        active: false,
        state: "unavailable" as const,
        status:
          row.runtimeSupport.state === "previewable"
            ? "Source unavailable"
            : row.runtimeSupport.label
      };
  const translation = artifact.creativeTranslation;

  return {
    artifact,
    audioSafetyLabel: resolveAudioSafetyLabel(outputKind),
    canRender,
    geometryLabels: formatPublicComparisonLabels(
      translation?.sacredGeometry?.concepts ?? []
    ),
    outputKind,
    outputLabel: formatOutputLabel(outputKind),
    preview: resolvedPreview,
    route,
    row,
    runtimeSessionKey: `comparison:${artifact.id}:${runtimeSource.fingerprint}`,
    runtimeSource,
    shaderPresetLabels: formatPublicComparisonLabels(
      translation?.shaderPresets?.presets ?? []
    ),
    visualStyleLabels: formatPublicComparisonLabels(
      translation?.visualStyle?.styles ?? []
    )
  };
}

function formatPublicComparisonLabels(values: readonly string[]) {
  return values.map((value) =>
    value
      .replace(/\bsacred geometry\b/gi, "geometry")
      .replace(/\bsacred\b/gi, "geometric")
      .replace(/\bsymbolic\b/gi, "conceptual")
  );
}

function buildCandidatePreview({
  artifact,
  basePreview,
  hasMatchingBasePreview,
  targetId
}: {
  artifact: ArtifactSummary;
  basePreview: PreviewSummary;
  hasMatchingBasePreview: boolean;
  targetId: PreviewTargetId | null;
}): PreviewSummary {
  const previewable =
    artifact.actions.includes("Preview") || artifact.type === "preview";
  const retainsError =
    previewable && hasMatchingBasePreview && basePreview.state === "error";

  return {
    ...basePreview,
    active: previewable,
    artifactName: artifact.title,
    available: previewable,
    collapsed: !previewable,
    error: retainsError ? basePreview.error : null,
    outputArtifactName: "",
    renderer: artifact.rendererId ?? (hasMatchingBasePreview ? basePreview.renderer : ""),
    sourceArtifactId: artifact.id,
    sourceArtifactName: artifact.title,
    state: retainsError ? "error" : previewable ? "ready" : "unavailable",
    status: retainsError
      ? "Preview failed"
      : previewable
        ? "Comparison ready"
        : "No live preview",
    summary: previewable
      ? `${artifact.title} is prepared for side-by-side comparison.`
      : `${artifact.title} remains available as a safe code-only comparison fallback.`,
    target:
      formatPreviewTargetLabel(targetId) ??
      (hasMatchingBasePreview ? basePreview.target : "No preview target"),
    targetId: targetId ?? "",
    title: previewable ? "Comparison preview" : "Preview unavailable",
    trigger: "Artifact comparison"
  };
}

function buildCandidateCodeSummary(
  artifact: ArtifactSummary,
  baseCode: CodeSummary
): CodeSummary {
  if (artifact.type !== "code") {
    return {
      excerpt: [],
      language: artifact.language,
      status: artifact.status,
      title: artifact.title
    };
  }

  const content =
    artifact.content ??
    (artifact.title === baseCode.title ? baseCode.excerpt.join("\n") : "");

  return {
    excerpt: content ? content.split("\n") : [],
    language: artifact.language,
    status: artifact.status,
    title: artifact.title
  };
}

function matchesBasePreview(
  artifact: ArtifactSummary,
  preview: PreviewSummary
) {
  return (
    artifact.id === preview.sourceArtifactId ||
    artifact.title === preview.sourceArtifactName ||
    artifact.title === preview.artifactName
  );
}

function resolveOutputKind(
  artifact: ArtifactSummary
): MultiPreviewOutputKind {
  const modality = artifact.creativeTranslation?.outputModality;
  if (modality) {
    return modality;
  }

  if (
    artifact.runtime === "tone" ||
    artifact.domain === "tone_js" ||
    derivePreviewTargetIdFromArtifact(artifact) === "audio_asset"
  ) {
    return "audio";
  }

  return artifact.actions.includes("Preview") || artifact.type === "preview"
    ? "visual"
    : "code";
}

function resolveAudioSafetyLabel(outputKind: MultiPreviewOutputKind) {
  switch (outputKind) {
    case "audio":
      return "Silent until explicit start";
    case "audiovisual":
      return "Audio remains opt-in";
    case "visual":
      return "No audio output";
    case "code":
      return "No runtime output";
  }
}

function formatOutputLabel(outputKind: MultiPreviewOutputKind) {
  switch (outputKind) {
    case "audio":
      return "Audio";
    case "audiovisual":
      return "Audiovisual";
    case "visual":
      return "Visual";
    case "code":
      return "Code-only";
  }
}
