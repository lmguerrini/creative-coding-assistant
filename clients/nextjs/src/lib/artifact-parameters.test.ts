import { describe, expect, it } from "vitest";
import type { ArtifactSummary } from "./assistant-client";
import {
  buildArtifactRefinementInstruction,
  createArtifactParameterValues,
  deriveArtifactParameterModel,
  serializeArtifactParameterGuidance,
  updateArtifactParameterValue
} from "./artifact-parameters";

describe("artifact parameter model", () => {
  it("derives bounded visual controls from runtime and creative metadata", () => {
    const model = deriveArtifactParameterModel(visualArtifact());

    expect(model.status).toBe("available");
    expect(model.parameters.length).toBeLessThanOrEqual(8);
    expect(model.parameters).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "runtime",
          defaultValue: "GLSL",
          type: "readonly"
        }),
        expect.objectContaining({
          id: "accent_color",
          defaultValue: "#45d8c8",
          type: "color"
        }),
        expect.objectContaining({
          id: "movement_complexity",
          defaultValue: 3,
          max: 10,
          min: 1,
          type: "range"
        }),
        expect.objectContaining({
          id: "bloom_intensity",
          source: "shader_preset"
        }),
        expect.objectContaining({
          id: "symmetry",
          defaultValue: "radial",
          source: "sacred_geometry"
        })
      ])
    );
  });

  it("derives audio controls without starting or executing the artifact", () => {
    const artifact: ArtifactSummary = {
      actions: ["Open", "Preview"],
      content:
        "const synth = new Tone.Synth().toDestination(); Tone.Transport.start();",
      domain: "tone_js",
      id: "tone-field",
      language: "JavaScript + Tone.js",
      previewEligible: true,
      runtime: "tone",
      status: "Generated",
      summary: "Sparse ambient drone.",
      title: "field.tone.js",
      type: "code"
    };

    const model = deriveArtifactParameterModel(artifact);

    expect(model.parameters).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "rhythm_density",
          defaultValue: 4
        }),
        expect.objectContaining({
          id: "drone_intensity",
          defaultValue: 0.7
        })
      ])
    );
    expect(model.parameters.some((parameter) => parameter.id === "rotation_speed")).toBe(
      false
    );
    expect(artifact.content).toContain("Tone.Transport.start()");
  });

  it("enables audio reactivity when explicit mapping metadata exists", () => {
    const artifact = visualArtifact();
    artifact.creativeTranslation = {
      ...artifact.creativeTranslation!,
      audioReactive: {
        activation: "explicit_user_gesture",
        audioRuntime: "Tone.js",
        mappings: [
          {
            behavior: "Use a smoothed envelope to shape the visual field.",
            evidence: ["explicit audiovisual intent"],
            intensity: "balanced",
            source: "amplitude",
            targets: ["scale", "glow"]
          }
        ],
        summary: "Amplitude shapes scale and glow.",
        visualRuntime: "GLSL"
      },
      outputModality: "audiovisual"
    };

    const model = deriveArtifactParameterModel(artifact);

    expect(model.parameters).toContainEqual(
      expect.objectContaining({
        defaultValue: true,
        id: "audio_reactivity"
      })
    );
  });

  it("derives bounded visual controls from reference fusion metadata", () => {
    const artifact = visualArtifact();
    artifact.summary = "Reference-guided shader.";
    artifact.creativeTranslation = {
      ...artifact.creativeTranslation!,
      colorMaterialDirection: [],
      referenceFusion: {
        composition: ["grid-based spatial layout"],
        geometricStructure: ["rectilinear grid"],
        lightingContrast: ["soft emissive glow"],
        moodAtmosphere: ["ethereal atmosphere"],
        motionImplications: ["slow drifting motion"],
        paletteDirection: ["amber highlights"],
        runtimeStyleImplications: [
          "Shader refraction presets may suit the material direction."
        ],
        safetyConstraints: [
          "Use references for aesthetic, palette, composition, and material guidance only."
        ],
        sourceCount: 1,
        sourceNames: ["amber-grid-reference.png"],
        summary: "Fused amber-grid-reference.png into non-identifying guidance.",
        textureMaterialCues: ["glasslike refraction cues"]
      },
      visualStyle: null
    };

    const model = deriveArtifactParameterModel(artifact);

    expect(model.parameters).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          defaultValue: "#f0b85a",
          id: "accent_color",
          source: "reference_fusion"
        })
      ])
    );
  });

  it("uses only bounded known code hints for runtime-specific controls", () => {
    const model = deriveArtifactParameterModel({
      actions: ["Open"],
      content:
        "const fog = new SceneFog(); const arbitraryUserCode = dangerousValue;",
      id: "legacy-three",
      language: "JavaScript",
      status: "Generated",
      summary: "Legacy scene code.",
      title: "legacy-scene.js",
      type: "code"
    });

    expect(model.status).toBe("available");
    expect(model.parameters).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "fog_density",
          source: "code_hint"
        })
      ])
    );
    expect(
      model.parameters.some((parameter) => parameter.id === "dangerousValue")
    ).toBe(false);
  });

  it("normalizes values and serializes only changed refinement guidance", () => {
    const model = deriveArtifactParameterModel(visualArtifact());
    const defaults = createArtifactParameterValues(model);
    const withMovement = updateArtifactParameterValue(
      model,
      defaults,
      "movement_complexity",
      14
    );
    const withColor = updateArtifactParameterValue(
      model,
      withMovement,
      "accent_color",
      "#FF7668"
    );
    const guidance = serializeArtifactParameterGuidance(model, withColor);

    expect(withColor.movement_complexity).toBe(10);
    expect(withColor.accent_color).toBe("#ff7668");
    expect(guidance?.changes).toEqual([
      expect.objectContaining({
        id: "accent_color",
        value: "#ff7668"
      }),
      expect.objectContaining({
        id: "movement_complexity",
        value: 10
      })
    ]);
    expect(guidance?.instruction).toContain("Accent color: #ff7668");
    expect(guidance?.instruction).toContain("Movement complexity: 10");
    expect(guidance?.instruction).toContain(
      "do not assume the source or preview was already mutated"
    );
    expect(
      buildArtifactRefinementInstruction({
        guidance: guidance ?? null,
        instruction: "Preserve the calm composition."
      })
    ).toContain("Preserve the calm composition.");
  });

  it("returns a safe fallback for legacy artifacts without derivation signals", () => {
    const model = deriveArtifactParameterModel({
      actions: ["Open"],
      id: "legacy",
      language: "Text",
      status: "Ready",
      summary: "Legacy artifact.",
      title: "legacy.txt",
      type: "code"
    });

    expect(model).toMatchObject({
      parameters: [],
      status: "unsupported"
    });
    expect(
      serializeArtifactParameterGuidance(
        model,
        createArtifactParameterValues(model)
      )
    ).toBeNull();
  });
});

function visualArtifact(): ArtifactSummary {
  return {
    actions: ["Open", "Preview", "Copy"],
    content: "void main() { vec3 glow = vec3(0.0); }",
    creativeTranslation: {
      colorMaterialDirection: ["cyan light"],
      creativeIntent: "Create a calm sacred shader field.",
      generationConstraints: [],
      geometricReferences: ["mandala"],
      moodAtmosphere: ["calm", "minimal"],
      movementLanguage: ["slow pulse"],
      musicalReferences: [],
      outputModality: "visual",
      refinementTargets: [],
      runtimeRecommendations: ["GLSL"],
      sacredGeometry: {
        audioImplications: [],
        colorMaterialDirection: [],
        concepts: ["mandala"],
        generationConstraints: [],
        geometricStructure: [],
        movementBehavior: [],
        runtimeRecommendations: ["GLSL"],
        symmetryType: ["radial"],
        visualComposition: []
      },
      shaderPresets: {
        colorBehavior: [],
        lightMaterialBehavior: [],
        motionBehavior: [],
        performanceConstraints: [],
        presets: ["glow"],
        runtimeSuitability: ["GLSL"],
        shaderStructure: []
      },
      structureDirection: ["center the primary field"],
      symbolicReferences: [],
      visualStyle: {
        compositionTendencies: [],
        contrastBehavior: [],
        motionTendencies: [],
        paletteBehavior: ["cyan monochrome"],
        runtimeSuitability: ["GLSL"],
        spatialOrganization: [],
        styles: ["minimal"],
        textureTendencies: []
      }
    },
    domain: "glsl",
    id: "shader-field",
    language: "GLSL",
    previewEligible: true,
    runtime: "glsl",
    status: "Generated",
    summary: "Calm cyan shader.",
    title: "field.frag",
    type: "code"
  };
}
