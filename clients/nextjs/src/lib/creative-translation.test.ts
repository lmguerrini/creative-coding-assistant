import { describe, expect, it } from "vitest";
import { normalizeCreativeTranslation } from "./creative-translation";

describe("creative translation normalization", () => {
  it("normalizes streamed snake-case metadata", () => {
    expect(
      normalizeCreativeTranslation({
        output_modality: "audiovisual",
        creative_intent: "Build an audio-reactive mandala.",
        symbolic_references: ["mandala"],
        geometric_references: ["sacred geometry"],
        musical_references: ["rhythm"],
        mood_atmosphere: ["meditative"],
        movement_language: ["pulse"],
        color_material_direction: ["cyan"],
        runtime_recommendations: ["p5.js", "Tone.js"],
        structure_direction: ["Coordinate visual changes with rhythm."],
        generation_constraints: ["Require explicit interaction."],
        refinement_targets: ["Preserve atmosphere: meditative"],
        sacred_geometry: {
          concepts: ["mandala"],
          geometric_structure: ["Build nested rings around a clear center."],
          symmetry_type: ["Use radial symmetry."],
          movement_behavior: ["Pulse concentric layers."],
          visual_composition: ["Keep a strong center."],
          color_material_direction: ["Use controlled contrast."],
          runtime_recommendations: ["p5.js", "GLSL"],
          audio_implications: ["Map rings to frequency bands."],
          generation_constraints: ["Avoid unsupported symbolic claims."]
        }
      })
    ).toMatchObject({
      outputModality: "audiovisual",
      creativeIntent: "Build an audio-reactive mandala.",
      symbolicReferences: ["mandala"],
      runtimeRecommendations: ["p5.js", "Tone.js"],
      sacredGeometry: {
        concepts: ["mandala"],
        symmetryType: ["Use radial symmetry."],
        runtimeRecommendations: ["p5.js", "GLSL"]
      }
    });
  });

  it("normalizes persisted camel-case sacred geometry metadata", () => {
    expect(
      normalizeCreativeTranslation({
        creativeIntent: "Create a torus.",
        sacredGeometry: {
          concepts: ["torus"],
          geometricStructure: ["Use a toroidal grid."],
          symmetryType: ["Use rotational symmetry."],
          movementBehavior: [],
          visualComposition: [],
          colorMaterialDirection: [],
          runtimeRecommendations: ["Three.js"],
          audioImplications: [],
          generationConstraints: []
        }
      })
    ).toMatchObject({
      sacredGeometry: {
        concepts: ["torus"],
        geometricStructure: ["Use a toroidal grid."],
        runtimeRecommendations: ["Three.js"]
      }
    });
  });

  it("returns a clean legacy fallback for absent metadata", () => {
    expect(normalizeCreativeTranslation(undefined)).toBeNull();
    expect(normalizeCreativeTranslation({ output_modality: "visual" })).toBeNull();
    expect(
      normalizeCreativeTranslation({
        creative_intent: "Create a legacy spiral."
      })
    ).toMatchObject({ sacredGeometry: null });
  });
});
