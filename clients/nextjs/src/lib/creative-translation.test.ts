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
        refinement_targets: ["Preserve atmosphere: meditative"]
      })
    ).toMatchObject({
      outputModality: "audiovisual",
      creativeIntent: "Build an audio-reactive mandala.",
      symbolicReferences: ["mandala"],
      runtimeRecommendations: ["p5.js", "Tone.js"]
    });
  });

  it("returns a clean legacy fallback for absent metadata", () => {
    expect(normalizeCreativeTranslation(undefined)).toBeNull();
    expect(normalizeCreativeTranslation({ output_modality: "visual" })).toBeNull();
  });
});
