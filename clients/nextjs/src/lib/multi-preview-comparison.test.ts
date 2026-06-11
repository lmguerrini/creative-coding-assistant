import { describe, expect, it } from "vitest";
import type { ArtifactSummary } from "./assistant-client";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  buildMultiPreviewComparisonModel,
  resolveMultiPreviewLayout
} from "./multi-preview-comparison";

describe("multi-preview comparison model", () => {
  it("builds candidate-specific runtime routes and compact creative metadata", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const artifacts: ArtifactSummary[] = [
      {
        ...snapshot.artifacts[0],
        content: "function draw() { background(8); }",
        creativeTranslation: {
          colorMaterialDirection: [],
          creativeIntent: "Create a minimal sacred field.",
          generationConstraints: [],
          geometricReferences: [],
          moodAtmosphere: ["minimal"],
          movementLanguage: [],
          musicalReferences: [],
          outputModality: "visual",
          refinementTargets: [],
          runtimeRecommendations: ["p5.js"],
          sacredGeometry: {
            audioImplications: [],
            colorMaterialDirection: [],
            concepts: ["mandala"],
            generationConstraints: [],
            geometricStructure: [],
            movementBehavior: [],
            runtimeRecommendations: ["p5.js"],
            symmetryType: [],
            visualComposition: []
          },
          shaderPresets: {
            colorBehavior: [],
            lightMaterialBehavior: [],
            motionBehavior: [],
            performanceConstraints: [],
            presets: ["glow"],
            runtimeSuitability: ["p5.js"],
            shaderStructure: []
          },
          structureDirection: [],
          symbolicReferences: [],
          visualStyle: {
            compositionTendencies: [],
            contrastBehavior: [],
            motionTendencies: [],
            paletteBehavior: [],
            runtimeSuitability: ["p5.js"],
            spatialOrganization: [],
            styles: ["minimal", "sacred geometry"],
            textureTendencies: []
          }
        }
      },
      {
        actions: ["Open", "Preview", "Copy"],
        content:
          "const synth = new Tone.Synth().toDestination(); Tone.Transport.start();",
        domain: "tone_js",
        id: "tone-candidate",
        language: "JavaScript + Tone.js",
        previewEligible: true,
        previewTarget: "browser_sandbox",
        rendererId: "surface.tone",
        runtime: "tone",
        status: "Generated",
        summary: "Controlled audio candidate.",
        title: "pulse.tone.js",
        type: "code"
      },
      {
        actions: ["Open", "Copy"],
        domain: "webgpu_wgsl",
        id: "code-only",
        language: "WGSL",
        previewEligible: false,
        status: "Generated",
        summary: "Unsupported browser runtime candidate.",
        title: "field.wgsl",
        type: "code"
      }
    ];

    const model = buildMultiPreviewComparisonModel({
      activeArtifactId: artifacts[0].id,
      artifacts,
      code: snapshot.code,
      preview: snapshot.preview
    });

    expect(model.layout).toBe("grid");
    expect(model.candidates[0]).toMatchObject({
      audioSafetyLabel: "No audio output",
      canRender: true,
      geometryLabels: ["mandala"],
      outputLabel: "Visual",
      shaderPresetLabels: ["glow"],
      visualStyleLabels: ["minimal", "sacred geometry"]
    });
    expect(model.candidates[0].route.surfaceKind).toBe("p5");
    expect(model.candidates[0].runtimeSource.source).toContain("draw()");
    expect(model.candidates[1]).toMatchObject({
      audioSafetyLabel: "Silent until explicit start",
      canRender: true,
      outputLabel: "Audio"
    });
    expect(model.candidates[1].route.surfaceKind).toBe("tone");
    expect(model.candidates[2]).toMatchObject({
      audioSafetyLabel: "No runtime output",
      canRender: false,
      outputLabel: "Code-only"
    });
    expect(model.candidates[2].preview.state).toBe("unavailable");
  });

  it("uses safe layouts for empty, legacy, and populated comparisons", () => {
    expect(resolveMultiPreviewLayout(0)).toBe("empty");
    expect(resolveMultiPreviewLayout(1)).toBe("single");
    expect(resolveMultiPreviewLayout(2)).toBe("split");
    expect(resolveMultiPreviewLayout(5)).toBe("grid");

    const snapshot = getLocalWorkspaceSnapshot();
    const model = buildMultiPreviewComparisonModel({
      activeArtifactId: "legacy",
      artifacts: [
        {
          actions: ["Open"],
          id: "legacy",
          language: "JavaScript",
          status: "Ready",
          summary: "Legacy artifact without comparison metadata.",
          title: "legacy.js",
          type: "code"
        }
      ],
      code: snapshot.code,
      preview: snapshot.preview
    });

    expect(model.layout).toBe("single");
    expect(model.candidates[0]).toMatchObject({
      canRender: false,
      geometryLabels: [],
      shaderPresetLabels: [],
      visualStyleLabels: []
    });
  });
});
