import type { ArtifactSummary } from "./assistant-client";
import { matchCreativePreviewRenderer } from "./preview-renderers";
import {
  derivePreviewTargetIdFromArtifact,
  formatPreviewTargetLabel
} from "./preview-targets";
import { isArtifactPreviewable } from "./preview-runtime";

export type ArtifactRuntimeSupportState =
  | "previewable"
  | "code_only"
  | "unsupported";

export type ArtifactRuntimeSupport = {
  state: ArtifactRuntimeSupportState;
  label: string;
  detail: string;
  targetLabel: string;
};

export type ArtifactComparisonRow = {
  artifact: ArtifactSummary;
  artifactId: string;
  domainLabel: string;
  isActive: boolean;
  isDefault: boolean;
  isRecommended: boolean;
  languageLabel: string;
  previewLabel: string;
  rankLabel: string;
  rationale: string;
  refinementGuidance: string | null;
  runtimeLabel: string;
  runtimeSupport: ArtifactRuntimeSupport;
  scoreLabel: string;
  statusLabel: string;
  title: string;
  typeLabel: string;
};

export type ArtifactComparisonModel = {
  recommendedReason: string;
  recommendedRow: ArtifactComparisonRow | null;
  rows: ArtifactComparisonRow[];
};

const supportedPreviewDomains = new Set([
  "p5_js",
  "glsl",
  "hydra",
  "gsap",
  "svg",
  "svg_markup",
  "canvas",
  "canvas_2d",
  "tone_js",
  "three_js",
  "react_three_fiber"
]);

const supportedBrowserRuntimeKinds = new Set([
  "p5",
  "three",
  "glsl",
  "hydra",
  "gsap",
  "tone",
  "svg",
  "canvas"
]);

const unsupportedBrowserRuntimeExtensions = [
  ".wgsl",
  ".webgpu.js",
  ".webgpu.ts"
] as const;

export function buildArtifactComparisonModel({
  activeArtifactId,
  artifacts
}: {
  activeArtifactId: string;
  artifacts: ArtifactSummary[];
}): ArtifactComparisonModel {
  const rows = artifacts.map((artifact) =>
    buildArtifactComparisonRow(artifact, artifact.id === activeArtifactId)
  );
  const recommendedRow = selectRecommendedComparisonRow(rows);

  return {
    recommendedReason: recommendedRow
      ? buildRecommendedReason(recommendedRow)
      : "No artifacts are available to compare yet.",
    recommendedRow,
    rows
  };
}

export function classifyArtifactRuntimeSupport(
  artifact: ArtifactSummary
): ArtifactRuntimeSupport {
  const targetId = derivePreviewTargetIdFromArtifact(artifact);
  const targetLabel = formatPreviewTargetLabel(targetId) ?? "No preview target";
  const matchedRenderer = matchCreativePreviewRenderer(artifact);
  const hasPreviewContract = isArtifactPreviewable(artifact);

  if (isReactThreeFiberExport(artifact)) {
    return {
      detail:
        "This React Three Fiber component is available as a code export. The workstation does not bundle an internal R3F live runtime.",
      label: "React export",
      state: "code_only",
      targetLabel: "Code export"
    };
  }

  if (hasPreviewContract) {
    if (targetId === "browser_sandbox" && !matchedRenderer) {
      return {
        detail:
          "Browser preview metadata exists, but no supported live renderer matches this artifact.",
        label: "Unsupported runtime",
        state: "unsupported",
        targetLabel
      };
    }

    return {
      detail: matchedRenderer
        ? `${matchedRenderer.displayName} can run in the live browser preview.`
        : `${targetLabel} can display this artifact without a creative runtime.`,
      label: "Previewable",
      state: "previewable",
      targetLabel
    };
  }

  if (hasUnsupportedRuntimeSignal(artifact)) {
    return {
      detail:
        "This artifact stays inspectable as code, but the current workstation has no safe live runtime for its domain.",
      label: "Unsupported runtime",
      state: "unsupported",
      targetLabel
    };
  }

  return {
    detail: "This artifact is available for code inspection without a live preview route.",
    label: "Code-only",
    state: "code_only",
    targetLabel
  };
}

function buildArtifactComparisonRow(
  artifact: ArtifactSummary,
  isActive: boolean
): ArtifactComparisonRow {
  const runtimeSupport = classifyArtifactRuntimeSupport(artifact);
  const critique = artifact.critique;

  return {
    artifact,
    artifactId: artifact.id,
    domainLabel: formatArtifactDomainLabel(artifact.domain),
    isActive,
    isDefault: artifact.isDefault === true,
    isRecommended: artifact.isRecommended === true || critique?.recommended === true,
    languageLabel: artifact.language || "Unknown language",
    previewLabel: buildPreviewLabel(artifact, runtimeSupport),
    rankLabel:
      artifact.qualityRank ?? critique?.rank
        ? `#${artifact.qualityRank ?? critique?.rank}`
        : "Unranked",
    rationale: critique?.rationale ?? artifact.summary,
    refinementGuidance:
      critique?.refinementGuidance ?? artifact.refinementReason ?? null,
    runtimeLabel: buildRuntimeLabel(artifact),
    runtimeSupport,
    scoreLabel: formatNullableQualityScore(
      artifact.qualityScore ?? critique?.overallScore ?? null
    ),
    statusLabel: buildStatusLabel(artifact),
    title: artifact.title,
    typeLabel: getArtifactTypeLabel(artifact.type)
  };
}

function selectRecommendedComparisonRow(rows: ArtifactComparisonRow[]) {
  return (
    rows.find((row) => row.isRecommended) ??
    rows
      .filter((row) => row.rankLabel !== "Unranked")
      .sort((a, b) => readRankValue(a.rankLabel) - readRankValue(b.rankLabel))[0] ??
    rows
      .filter((row) => row.scoreLabel !== "Unscored")
      .sort((a, b) => readScoreValue(b.scoreLabel) - readScoreValue(a.scoreLabel))[0] ??
    rows.find((row) => row.isDefault) ??
    rows[0] ??
    null
  );
}

function buildRecommendedReason(row: ArtifactComparisonRow) {
  if (row.isRecommended && row.rationale) {
    return row.rationale;
  }

  if (row.rankLabel !== "Unranked") {
    return `${row.title} is the highest ranked candidate in the artifact critique.`;
  }

  if (row.scoreLabel !== "Unscored") {
    return `${row.title} has the strongest available quality score.`;
  }

  if (row.isDefault) {
    return `${row.title} is marked as the default workspace artifact.`;
  }

  return `${row.title} is the first available artifact candidate.`;
}

function buildRuntimeLabel(artifact: ArtifactSummary) {
  const matchedRenderer = matchCreativePreviewRenderer(artifact);

  if (artifact.runtime) {
    return formatArtifactRuntimeLabel(artifact.runtime);
  }

  if (matchedRenderer) {
    return matchedRenderer.displayName;
  }

  if (artifact.domain) {
    return formatArtifactDomainLabel(artifact.domain);
  }

  return artifact.type === "preview" ? "Preview manifest" : artifact.language;
}

function buildPreviewLabel(
  artifact: ArtifactSummary,
  runtimeSupport: ArtifactRuntimeSupport
) {
  const targetId = derivePreviewTargetIdFromArtifact(artifact);
  const matchedRenderer = matchCreativePreviewRenderer(artifact);

  if (runtimeSupport.state === "previewable") {
    if (targetId === "browser_sandbox" && matchedRenderer) {
      return `${runtimeSupport.targetLabel} / ${matchedRenderer.displayName}`;
    }

    return runtimeSupport.targetLabel;
  }

  if (runtimeSupport.state === "unsupported") {
    return "No supported live runtime";
  }

  return "No live preview route";
}

function isReactThreeFiberExport(artifact: ArtifactSummary) {
  const searchable = [artifact.title, artifact.domain, artifact.content, artifact.summary]
    .filter((value): value is string => Boolean(value))
    .join(" ")
    .toLowerCase();
  return (
    artifact.domain === "react_three_fiber" ||
    artifact.title.toLowerCase().endsWith(".r3f.tsx") ||
    searchable.includes("@react-three/fiber") ||
    searchable.includes("react three fiber")
  );
}

function buildStatusLabel(artifact: ArtifactSummary) {
  const labels = [artifact.status];

  if (artifact.isDefault) {
    labels.push("default");
  }

  if (artifact.isRecommended) {
    labels.push("recommended");
  }

  if (artifact.refinedFromTitle) {
    labels.push("refined");
  }

  return labels.join(" / ");
}

function hasUnsupportedRuntimeSignal(artifact: ArtifactSummary) {
  if (artifact.type !== "code") {
    return false;
  }

  const domain = artifact.domain?.trim().toLowerCase();
  if (domain && !supportedPreviewDomains.has(domain)) {
    return true;
  }

  const runtime = artifact.runtime?.trim().toLowerCase();
  if (runtime && !supportedBrowserRuntimeKinds.has(runtime)) {
    return true;
  }

  if (artifact.previewEligible === false) {
    return true;
  }

  return hasUnsupportedBrowserRuntimeExtension(artifact.title);
}

function hasUnsupportedBrowserRuntimeExtension(title: string) {
  const normalizedTitle = title.trim().toLowerCase();
  return unsupportedBrowserRuntimeExtensions.some((extension) =>
    normalizedTitle.endsWith(extension)
  );
}

function formatNullableQualityScore(score: number | null | undefined) {
  return score == null ? "Unscored" : `${Math.round(score * 100)}%`;
}

function readRankValue(rankLabel: string) {
  const value = Number(rankLabel.replace("#", ""));
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

function readScoreValue(scoreLabel: string) {
  const value = Number(scoreLabel.replace("%", ""));
  return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
}

function getArtifactTypeLabel(type: ArtifactSummary["type"]) {
  switch (type) {
    case "code":
      return "Source code";
    case "preview":
      return "Preview manifest";
    case "export":
      return "Markdown export";
    default:
      return type;
  }
}

function formatArtifactRuntimeLabel(runtime: string) {
  switch (runtime) {
    case "p5":
      return "p5.js";
    case "three":
      return "Three.js";
    case "glsl":
      return "GLSL";
    case "hydra":
      return "Hydra";
    case "gsap":
      return "GSAP";
    case "svg":
      return "SVG";
    case "canvas":
      return "Canvas";
    case "tone":
      return "Tone.js";
    default:
      return sentenceCase(runtime.replace(/[_-]+/g, " "));
  }
}

function formatArtifactDomainLabel(domain: string | null | undefined) {
  if (!domain) {
    return "No selected domain";
  }

  switch (domain) {
    case "p5_js":
      return "p5.js";
    case "three_js":
      return "Three.js";
    case "react_three_fiber":
      return "React Three Fiber";
    case "glsl":
      return "GLSL";
    case "hydra":
      return "Hydra";
    case "gsap":
      return "GSAP";
    case "svg":
      return "SVG";
    case "canvas":
    case "canvas_2d":
      return "Canvas";
    case "tone_js":
      return "Tone.js";
    default:
      return sentenceCase(domain.replace(/_/g, " "));
  }
}

function sentenceCase(value: string) {
  const normalized = value.trim();
  return normalized ? normalized[0].toUpperCase() + normalized.slice(1) : normalized;
}
