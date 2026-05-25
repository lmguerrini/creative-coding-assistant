import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildPreviewRuntimeSource,
  canRunPreviewRuntime,
  getExecutablePreviewRuntimeKind
} from "./preview-runtime-adapters";
import { buildPreviewRendererRoute } from "./preview-renderers";

describe("preview runtime adapters", () => {
  it("extracts stable runtime source for routed code artifacts", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const route = buildPreviewRendererRoute({
      artifacts: [
        {
          ...snapshot.artifacts[0],
          summary: "Reactive p5 loop with createCanvas() and draw().",
          title: snapshot.code.title
        }
      ],
      preview: {
        ...snapshot.preview,
        artifactName: snapshot.code.title,
        sourceArtifactName: snapshot.code.title
      },
      previewArtifactId: "source-sketch"
    });
    const source = buildPreviewRuntimeSource({
      code: snapshot.code,
      route
    });

    expect(source).toMatchObject({
      lineCount: snapshot.code.excerpt.length,
      source: snapshot.code.excerpt.join("\n"),
      title: snapshot.code.title
    });
    expect(source.fingerprint).toMatch(/^[a-f0-9]+$/);
  });

  it("marks p5, Three.js, and GLSL routes as executable runtimes", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const p5Artifact = {
      ...snapshot.artifacts[0],
      summary: "Reactive p5 loop with createCanvas() and draw().",
      title: "signal-orbit.p5.ts"
    };
    const threeArtifact = {
      ...snapshot.artifacts[0],
      summary: "Three scene with WebGLRenderer, lights, and camera motion.",
      title: "projection-scene.three.ts"
    };
    const glslArtifact = {
      ...snapshot.artifacts[0],
      language: "GLSL",
      summary: "Fragment shader with gl_FragColor and uniforms.",
      title: "chromatic-field.frag"
    };
    const p5Route = buildPreviewRendererRoute({
      artifacts: [p5Artifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: p5Artifact.title,
        sourceArtifactName: p5Artifact.title
      },
      previewArtifactId: p5Artifact.id
    });
    const threeRoute = buildPreviewRendererRoute({
      artifacts: [threeArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: threeArtifact.title,
        sourceArtifactName: threeArtifact.title
      },
      previewArtifactId: threeArtifact.id
    });
    const glslRoute = buildPreviewRendererRoute({
      artifacts: [glslArtifact],
      preview: {
        ...snapshot.preview,
        active: true,
        artifactName: glslArtifact.title,
        sourceArtifactName: glslArtifact.title
      },
      previewArtifactId: glslArtifact.id
    });

    expect(getExecutablePreviewRuntimeKind(p5Route)).toBe("p5");
    expect(getExecutablePreviewRuntimeKind(threeRoute)).toBe("three");
    expect(getExecutablePreviewRuntimeKind(glslRoute)).toBe("glsl");
    expect(
      canRunPreviewRuntime({
        preview: { ...snapshot.preview, active: true, state: "ready" },
        route: p5Route
      })
    ).toBe(true);
    expect(
      canRunPreviewRuntime({
        preview: { ...snapshot.preview, active: true, state: "unavailable" },
        route: p5Route
      })
    ).toBe(false);
  });
});
