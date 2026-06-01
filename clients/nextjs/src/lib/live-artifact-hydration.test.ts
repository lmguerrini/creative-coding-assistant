import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import { buildArtifactDocument } from "./artifact-inspector";
import {
  hydrateWorkspaceFromArtifactExtractedEvent,
  hydrateWorkspaceFromFinalEvent
} from "./live-artifact-hydration";
import { buildPreviewRendererRoute } from "./preview-renderers";
import type { AssistantStreamEvent } from "./assistant-stream";

describe("live artifact hydration", () => {
  it("hydrates final stream code into a previewable Three.js artifact", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: [
          "Here is the scene:",
          "```ts",
          "import * as THREE from 'three';",
          "const scene = new THREE.Scene();",
          "const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);",
          "const renderer = new THREE.WebGLRenderer({ antialias: true });",
          "scene.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial()));",
          "```"
        ].join("\n")
      })
    );

    expect(result.artifact).toMatchObject({
      id: "live-generated-artifact",
      title: "generated-scene.three.ts",
      type: "code",
      language: "TypeScript + Three.js",
      status: "Generated",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("live-generated-artifact");
    expect(result.previewAvailable).toBe(true);
    expect(result.snapshot.code.title).toBe("generated-scene.three.ts");
    expect(result.snapshot.code.excerpt).toContain(
      "const renderer = new THREE.WebGLRenderer({ antialias: true });"
    );
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      artifactName: "generated-scene.three.ts",
      outputArtifactName: "generated-scene.three.ts",
      state: "ready",
      status: "Ready when opened",
      sourceArtifactId: "live-generated-artifact",
      targetId: "browser_sandbox"
    });

    expect(
      buildPreviewRendererRoute({
        artifacts: result.snapshot.artifacts,
        preview: result.snapshot.preview,
        previewArtifactId: result.previewArtifactId
      })
    ).toMatchObject({
      rendererId: "surface.three",
      supportState: "supported",
      surfaceKind: "three"
    });
  });

  it("uses structured artifact payloads before parsing chat prose", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer: "Generated shader attached.",
        artifacts: [
          {
            id: "shader-output",
            filename: "aurora.frag",
            language: "glsl",
            content: "void main() {\n  gl_FragColor = vec4(1.0);\n}"
          }
        ]
      })
    );

    expect(result.artifact).toMatchObject({
      id: "shader-output",
      title: "aurora.frag",
      language: "GLSL",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.snapshot.preview).toMatchObject({
      artifactName: "aurora.frag",
      renderer: "surface.glsl",
      targetId: "browser_sandbox"
    });
  });

  it("hydrates graph-owned artifact extraction events before finalization", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromArtifactExtractedEvent(
      snapshot,
      {
        event_type: "artifact_extracted",
        payload: {
          artifacts: [
            {
              id: "graph-sketch",
              title: "graph-sketch.p5.js",
              language: "JavaScript + p5.js",
              source_language: "javascript",
              content:
                "function setup() {\n  createCanvas(640, 360);\n}\nfunction draw() {\n  background(12);\n}"
            }
          ]
        },
        sequence: 3
      }
    );

    expect(result.artifact).toMatchObject({
      id: "graph-sketch",
      title: "graph-sketch.p5.js",
      language: "JavaScript + p5.js",
      actions: ["Open", "Preview", "Copy", "Download"]
    });
    expect(result.previewArtifactId).toBe("graph-sketch");
    expect(result.snapshot.preview).toMatchObject({
      available: true,
      renderer: "surface.p5",
      sourceArtifactId: "graph-sketch",
      trigger: "Artifact extraction"
    });
  });

  it("creates a readable artifact and disables preview for non-runnable answers", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const result = hydrateWorkspaceFromFinalEvent(
      snapshot,
      finalEvent({
        answer:
          "Use a slower modulation curve and keep the palette contrast high for projection."
      })
    );
    const document = buildArtifactDocument(result.snapshot, result.artifact!);

    expect(result.artifact).toMatchObject({
      id: "live-response-artifact",
      title: "assistant-response.md",
      type: "export",
      language: "Markdown",
      actions: ["Open", "Copy", "Download"]
    });
    expect(document.content).toContain("Use a slower modulation curve");
    expect(result.previewArtifactId).toBe("");
    expect(result.previewAvailable).toBe(false);
    expect(result.snapshot.preview).toMatchObject({
      available: false,
      state: "unavailable",
      status: "Unavailable",
      title: "Preview unavailable",
      targetId: ""
    });
  });
});

function finalEvent(payload: Record<string, unknown>): AssistantStreamEvent {
  return {
    event_type: "final",
    payload,
    sequence: 7
  };
}
