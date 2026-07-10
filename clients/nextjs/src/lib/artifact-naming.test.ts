import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot, type ArtifactSummary } from "./assistant-client";
import {
  assignSemanticArtifactTitles,
  renameWorkspaceArtifact
} from "./artifact-naming";

describe("artifact naming", () => {
  it("derives a semantic p5 filename from the request while preserving its runtime suffix", () => {
    const [artifact] = assignSemanticArtifactTitles({
      artifacts: [p5Artifact("generated-sketch.p5.js")],
      existingTitles: [],
      prompt: "Create a p5.js flow-field particle system with soft trails."
    });

    expect(artifact?.title).toBe("flow-field-particle-system-soft.p5.js");
  });

  it("resolves generated-name collisions without changing the runtime suffix", () => {
    const [artifact] = assignSemanticArtifactTitles({
      artifacts: [p5Artifact("generated-sketch.p5.js")],
      existingTitles: ["flow-field-particle-system-soft.p5.js"],
      prompt: "Create a p5.js flow-field particle system with soft trails."
    });

    expect(artifact?.title).toBe("flow-field-particle-system-soft-2.p5.js");
  });

  it("renames a saved artifact and repairs code, preview, and refinement references", () => {
    const baseSnapshot = getLocalWorkspaceSnapshot();
    const source = p5Artifact("aurora-field.p5.js");
    const refined: ArtifactSummary = {
      ...p5Artifact("aurora-field-refined.p5.js"),
      id: "refined-field",
      refinedFromArtifactId: source.id,
      refinedFromTitle: source.title,
      refinementPasses: [
        {
          passNumber: 1,
          sourceArtifactId: source.id,
          sourceArtifactTitle: source.title,
          resultArtifactId: "refined-field",
          resultArtifactTitle: "aurora-field-refined.p5.js",
          refinementObjective: "Increase contrast.",
          stopReason: "quality_improved",
          summary: "Contrast improved."
        }
      ]
    };
    const snapshot = {
      ...baseSnapshot,
      artifacts: [source, refined],
      code: { ...baseSnapshot.code, title: source.title },
      preview: {
        ...baseSnapshot.preview,
        artifactName: source.title,
        outputArtifactName: source.title,
        sourceArtifactId: source.id,
        sourceArtifactName: source.title
      }
    };

    const result = renameWorkspaceArtifact({
      artifactId: source.id,
      requestedTitle: "Luminous Flow Field.ts",
      snapshot
    });

    expect(result?.title).toBe("luminous-flow-field.p5.js");
    expect(result?.snapshot.code.title).toBe("luminous-flow-field.p5.js");
    expect(result?.snapshot.preview).toMatchObject({
      artifactName: "luminous-flow-field.p5.js",
      outputArtifactName: "luminous-flow-field.p5.js",
      sourceArtifactName: "luminous-flow-field.p5.js"
    });
    expect(result?.snapshot.artifacts[1]).toMatchObject({
      refinedFromTitle: "luminous-flow-field.p5.js",
      refinementPasses: [
        expect.objectContaining({
          sourceArtifactTitle: "luminous-flow-field.p5.js"
        })
      ]
    });
  });
});

function p5Artifact(title: string): ArtifactSummary {
  return {
    actions: ["Open", "Preview", "Copy", "Download"],
    content: "function setup() { createCanvas(320, 180); }\nfunction draw() {}",
    id: "flow-field",
    language: "JavaScript + p5.js",
    rendererId: "surface.p5",
    runtime: "p5",
    status: "Generated",
    summary: "A runnable p5.js field.",
    title,
    type: "code"
  };
}
