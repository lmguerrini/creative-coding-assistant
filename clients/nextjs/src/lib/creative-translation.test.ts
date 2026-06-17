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
        },
        shader_presets: {
          presets: ["glow", "kaleidoscopic symmetry"],
          color_behavior: ["Use a restrained luminous palette."],
          light_material_behavior: ["Use bounded emission layers."],
          motion_behavior: ["Pulse intensity slowly."],
          shader_structure: ["Separate an emission mask."],
          runtime_suitability: [
            "Use the selected compatible runtime: p5.js."
          ],
          performance_constraints: [
            "Use a bounded number of glow layers."
          ]
        },
        visual_style: {
          styles: ["minimal", "sacred geometry", "unsupported"],
          palette_behavior: ["Use one dominant tone."],
          contrast_behavior: ["Use clear value hierarchy."],
          composition_tendencies: ["Use deliberate negative space."],
          motion_tendencies: ["Use slow readable transitions."],
          texture_tendencies: ["Keep surfaces clean."],
          spatial_organization: ["Favor a stable focal point."],
          runtime_suitability: [
            "Use the selected compatible runtime: p5.js."
          ]
        },
        reference_fusion: {
          source_count: 2,
          source_names: ["warm-neon-grid.png", "glass-drift.webp"],
          palette_direction: ["warm palette bias", "neon accent contrast"],
          composition: ["grid-based spatial layout"],
          lighting_contrast: ["soft emissive glow"],
          texture_material_cues: ["glasslike refraction cues"],
          geometric_structure: ["rectilinear grid"],
          mood_atmosphere: ["ethereal atmosphere"],
          motion_implications: ["slow drifting motion"],
          runtime_style_implications: [
            "Shader refraction presets may suit the material direction."
          ],
          safety_constraints: [
            "Use references for aesthetic, palette, composition, and material guidance only."
          ],
          summary: "Fused 2 references into non-identifying guidance."
        },
        audio_reactive: {
          mappings: [
            {
              source: "amplitude",
              targets: ["scale", "glow", "unsupported"],
              intensity: "subtle",
              behavior: "Smooth short peaks.",
              evidence: ["user prompt", "shader presets"]
            },
            {
              source: "unsupported",
              targets: ["scale"],
              behavior: "Ignore unsupported sources."
            }
          ],
          audio_runtime: "Tone.js",
          visual_runtime: "p5.js",
          activation: "explicit_user_gesture",
          summary: "amplitude -> scale / glow"
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
      },
      shaderPresets: {
        presets: ["glow", "kaleidoscopic symmetry"],
        runtimeSuitability: [
          "Use the selected compatible runtime: p5.js."
        ]
      },
      visualStyle: {
        styles: ["minimal", "sacred geometry"],
        paletteBehavior: ["Use one dominant tone."],
        runtimeSuitability: [
          "Use the selected compatible runtime: p5.js."
        ]
      },
      referenceFusion: {
        sourceCount: 2,
        sourceNames: ["warm-neon-grid.png", "glass-drift.webp"],
        paletteDirection: ["warm palette bias", "neon accent contrast"],
        composition: ["grid-based spatial layout"],
        textureMaterialCues: ["glasslike refraction cues"],
        motionImplications: ["slow drifting motion"],
        summary: "Fused 2 references into non-identifying guidance."
      },
      audioReactive: {
        activation: "explicit_user_gesture",
        audioRuntime: "Tone.js",
        mappings: [
          {
            source: "amplitude",
            targets: ["scale", "glow"],
            intensity: "subtle",
            behavior: "Smooth short peaks.",
            evidence: ["user prompt", "shader presets"]
          }
        ],
        summary: "amplitude -> scale / glow",
        visualRuntime: "p5.js"
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

  it("normalizes persisted camel-case shader preset metadata", () => {
    expect(
      normalizeCreativeTranslation({
        creativeIntent: "Create a refractive crystal.",
        shaderPresets: {
          presets: ["refraction", "glass / crystal", "unsupported"],
          colorBehavior: ["Use restrained tinting."],
          lightMaterialBehavior: ["Use stylized transmission."],
          motionBehavior: ["Rotate slowly."],
          shaderStructure: ["Separate distortion and source sampling."],
          runtimeSuitability: [
            "Use the selected compatible runtime: GLSL."
          ],
          performanceConstraints: ["Limit texture reads."]
        }
      })
    ).toMatchObject({
      shaderPresets: {
        presets: ["refraction", "glass / crystal"],
        runtimeSuitability: [
          "Use the selected compatible runtime: GLSL."
        ]
      }
    });
  });

  it("normalizes persisted camel-case visual style metadata", () => {
    expect(
      normalizeCreativeTranslation({
        creativeIntent: "Create an architectural field.",
        visualStyle: {
          styles: ["architectural", "monochrome", "unknown"],
          paletteBehavior: ["Use material-led neutrals."],
          contrastBehavior: ["Clarify planes with light and shadow."],
          compositionTendencies: ["Use axes and modules."],
          motionTendencies: ["Use measured camera movement."],
          textureTendencies: ["Use bounded material cues."],
          spatialOrganization: ["Prioritize scale and depth."],
          runtimeSuitability: [
            "Use the selected compatible runtime: Three.js."
          ]
        }
      })
    ).toMatchObject({
      visualStyle: {
        styles: ["architectural", "monochrome"],
        compositionTendencies: ["Use axes and modules."],
        runtimeSuitability: [
          "Use the selected compatible runtime: Three.js."
        ]
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
    ).toMatchObject({
      sacredGeometry: null,
      shaderPresets: null,
      visualStyle: null,
      audioReactive: null,
      referenceFusion: null
    });
  });
});
