import type { ArtifactSummary, AssistantWorkspaceSnapshot } from "./assistant-client";
import { isArtifactPreviewable } from "./preview-runtime";

export type RemovedArtifact = {
  artifact: ArtifactSummary;
  index: number;
};

export type ArtifactLifecycleResult = {
  activeArtifactId: string;
  previewArtifactId: string;
  removed: RemovedArtifact;
  snapshot: AssistantWorkspaceSnapshot;
};

export function removeWorkspaceArtifact({
  activeArtifactId,
  artifactId,
  previewArtifactId,
  snapshot
}: {
  activeArtifactId: string;
  artifactId: string;
  previewArtifactId: string;
  snapshot: AssistantWorkspaceSnapshot;
}): ArtifactLifecycleResult | null {
  const index = snapshot.artifacts.findIndex((artifact) => artifact.id === artifactId);
  if (index < 0) {
    return null;
  }
  const artifact = snapshot.artifacts[index];
  const artifacts = snapshot.artifacts.filter((item) => item.id !== artifactId);
  const nextArtifact = artifacts[index] ?? artifacts[index - 1] ?? null;
  const sourceWasRemoved = snapshot.preview.sourceArtifactId === artifactId;
  const preview = sourceWasRemoved
    ? {
        ...snapshot.preview,
        active: false,
        available: false,
        collapsed: true,
        sourceArtifactId: "",
        sourceArtifactName: "",
        outputArtifactName: "",
        artifactName: nextArtifact?.title ?? "",
        status: "Unavailable",
        state: "unavailable" as const,
        summary: "The preview source was deleted. Select another saved artifact to preview it."
      }
    : snapshot.preview;
  const resolvedActive =
    activeArtifactId === artifactId ? nextArtifact?.id ?? "" : activeArtifactId;
  const resolvedPreview =
    previewArtifactId === artifactId && !sourceWasRemoved && nextArtifact && isArtifactPreviewable(nextArtifact)
      ? nextArtifact.id
      : previewArtifactId === artifactId
        ? ""
        : previewArtifactId;

  return {
    activeArtifactId: resolvedActive,
    previewArtifactId: resolvedPreview,
    removed: { artifact, index },
    snapshot: { ...snapshot, artifacts, preview }
  };
}

export function restoreWorkspaceArtifact({
  removed,
  snapshot
}: {
  removed: RemovedArtifact;
  snapshot: AssistantWorkspaceSnapshot;
}): AssistantWorkspaceSnapshot {
  if (snapshot.artifacts.some((artifact) => artifact.id === removed.artifact.id)) {
    return snapshot;
  }
  const artifacts = [...snapshot.artifacts];
  artifacts.splice(Math.min(removed.index, artifacts.length), 0, removed.artifact);
  return { ...snapshot, artifacts };
}
