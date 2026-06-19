import type { ArtifactSummary, PreviewTargetId } from "./assistant-client";
import { hasSvgPreviewSignal } from "./svg-canvas-runtime";

const previewTargetLabels: Record<PreviewTargetId, string> = {
  audio_asset: "Audio asset",
  browser_sandbox: "Browser preview",
  image_asset: "Image asset",
  json_panel: "JSON panel",
  text_panel: "Text panel",
  video_asset: "Video asset"
};

export function normalizePreviewTargetId(
  value: string | null | undefined
): PreviewTargetId | null {
  switch (value) {
    case "audio_asset":
    case "browser_sandbox":
    case "image_asset":
    case "json_panel":
    case "text_panel":
    case "video_asset":
      return value;
    default:
      return null;
  }
}

export function formatPreviewTargetLabel(
  targetId: PreviewTargetId | null | undefined
): string | null {
  return targetId ? previewTargetLabels[targetId] : null;
}

export function derivePreviewTargetIdFromArtifact(
  artifact: ArtifactSummary | null
): PreviewTargetId | null {
  if (!artifact) {
    return null;
  }

  if (artifact.type === "preview") {
    return "json_panel";
  }

  if (!artifact.actions.includes("Preview")) {
    return null;
  }

  if (artifact.type === "export") {
    return "text_panel";
  }

  const normalizedTitle = artifact.title.trim().toLowerCase();

  if (
    artifact.type === "code" &&
    normalizedTitle.endsWith(".svg") &&
    hasSvgPreviewSignal(artifact)
  ) {
    return "browser_sandbox";
  }

  if (normalizedTitle.endsWith(".json")) {
    return "json_panel";
  }

  if (normalizedTitle.endsWith(".md") || normalizedTitle.endsWith(".txt")) {
    return "text_panel";
  }

  if (
    normalizedTitle.endsWith(".png") ||
    normalizedTitle.endsWith(".jpg") ||
    normalizedTitle.endsWith(".jpeg") ||
    normalizedTitle.endsWith(".gif") ||
    normalizedTitle.endsWith(".webp") ||
    normalizedTitle.endsWith(".svg")
  ) {
    return "image_asset";
  }

  if (
    normalizedTitle.endsWith(".mp3") ||
    normalizedTitle.endsWith(".wav") ||
    normalizedTitle.endsWith(".aiff") ||
    normalizedTitle.endsWith(".ogg")
  ) {
    return "audio_asset";
  }

  if (
    normalizedTitle.endsWith(".mp4") ||
    normalizedTitle.endsWith(".webm") ||
    normalizedTitle.endsWith(".mov")
  ) {
    return "video_asset";
  }

  return "browser_sandbox";
}
