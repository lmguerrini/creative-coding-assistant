import { describe, expect, it } from "vitest";

import { getLocalWorkspaceSnapshot } from "./assistant-client";
import { removeWorkspaceArtifact, restoreWorkspaceArtifact } from "./artifact-lifecycle";

describe("artifact lifecycle", () => {
  it("removes only the selected artifact and clears a deleted preview source", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = removeWorkspaceArtifact({
      activeArtifactId: "source-sketch",
      artifactId: "source-sketch",
      previewArtifactId: "source-sketch",
      snapshot: {
        ...snapshot,
        preview: { ...snapshot.preview, sourceArtifactId: "source-sketch" }
      }
    });

    expect(result?.snapshot.artifacts.map((artifact) => artifact.id)).not.toContain(
      "source-sketch"
    );
    expect(result?.snapshot.preview.available).toBe(false);
    expect(result?.snapshot.preview.state).toBe("unavailable");
    expect(result?.activeArtifactId).not.toBe("source-sketch");
  });

  it("restores an undo record at the original artifact position", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = removeWorkspaceArtifact({
      activeArtifactId: "source-sketch",
      artifactId: "source-sketch",
      previewArtifactId: "preview-manifest",
      snapshot
    });

    const restored = restoreWorkspaceArtifact({
      removed: result!.removed,
      snapshot: result!.snapshot
    });
    expect(restored.artifacts.map((artifact) => artifact.id)).toEqual(
      snapshot.artifacts.map((artifact) => artifact.id)
    );
  });
});
