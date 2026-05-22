import { describe, expect, it } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type ArtifactSummary,
  type PreviewSummary
} from "./assistant-client";
import {
  buildPreviewRendererRoute,
  matchCreativePreviewRenderer
} from "./preview-renderers";

describe("preview renderers", () => {
  it("routes the default WebGPU sketch into an unsupported browser surface", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRendererRoute({
        artifacts: snapshot.artifacts,
        preview: snapshot.preview,
        previewArtifactId: "source-sketch"
      })
    ).toMatchObject({
      targetId: "browser_sandbox",
      targetLabel: "Browser sandbox",
      rendererLabel: "No matching live renderer",
      supportState: "unsupported",
      surfaceKind: "unsupported"
    });
  });

  it("routes preview manifests into the json panel surface", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    expect(
      buildPreviewRendererRoute({
        artifacts: snapshot.artifacts,
        preview: {
          ...snapshot.preview,
          active: true,
          artifactName: "preview-request.json",
          outputArtifactName: "preview-request.json",
          sourceArtifactId: "source-sketch",
          sourceArtifactName: "webgpu-particle-field.ts",
          state: "ready",
          status: "Preview open"
        },
        previewArtifactId: "preview-manifest"
      })
    ).toMatchObject({
      targetId: "json_panel",
      targetLabel: "JSON panel",
      rendererId: "surface.json_panel",
      rendererLabel: "JSON panel surface",
      supportState: "supported",
      surfaceKind: "json_panel"
    });
  });

  it.each([
    {
      id: "surface.p5",
      kind: "p5",
      artifact: creativeArtifact({
        summary: "Reactive p5 loop with createCanvas() and draw().",
        title: "signal-orbit.p5.ts"
      })
    },
    {
      id: "surface.three",
      kind: "three",
      artifact: creativeArtifact({
        summary: "Three scene with WebGLRenderer, lights, and camera motion.",
        title: "projection-scene.three.ts"
      })
    },
    {
      id: "surface.glsl",
      kind: "glsl",
      artifact: creativeArtifact({
        language: "GLSL",
        summary: "Fragment shader with gl_FragColor and uniforms.",
        title: "chromatic-field.frag"
      })
    },
    {
      id: "surface.hydra",
      kind: "hydra",
      artifact: creativeArtifact({
        summary: "Hydra patch built from osc(), shape(), and out().",
        title: "feedback-lattice.hydra.js"
      })
    }
  ])(
    "matches $kind renderer signals and routes them into a supported surface",
    ({ artifact, id, kind }) => {
      const snapshot = getLocalWorkspaceSnapshot();
      const preview = creativePreviewSummary(artifact, snapshot.preview);

      expect(matchCreativePreviewRenderer(artifact)).toMatchObject({
        id,
        kind
      });
      expect(
        buildPreviewRendererRoute({
          artifacts: [artifact],
          preview,
          previewArtifactId: artifact.id
        })
      ).toMatchObject({
        targetId: "browser_sandbox",
        rendererId: id,
        supportState: "supported",
        surfaceKind: kind
      });
    }
  );
});

function creativeArtifact(
  overrides: Partial<ArtifactSummary>
): ArtifactSummary {
  const snapshot = getLocalWorkspaceSnapshot();

  return {
    ...snapshot.artifacts[0],
    ...overrides,
    actions: ["Open", "Preview", "Copy", "Download"]
  };
}

function creativePreviewSummary(
  artifact: ArtifactSummary,
  fallback: PreviewSummary
): PreviewSummary {
  return {
    ...fallback,
    active: true,
    artifactName: artifact.title,
    sourceArtifactId: artifact.id,
    sourceArtifactName: artifact.title,
    state: "ready",
    status: "Ready when opened",
    target: "Browser sandbox",
    targetId: "browser_sandbox"
  };
}
